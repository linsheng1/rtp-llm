#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <future>
#include <map>
#include <tuple>
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {
// GPU pages + recording SDK. CPU copies below are assertions only; the
// implementation must pass native GPU IOVs through the Prefill transfer path.
class EosGpuCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool>> {
protected:
    autil::EnvGuard enable{"RTP_DTOP_FILE_CP2", "1"}, pace{"RTP_DTOP_PACE", "1"}, audit{"RTP_DTOP_PAYLOAD_AUDIT", "0"},
        pending{"RTP_DTOP_GPU_PENDING_JOBS", "1"};
    // Failure cases deliberately freeze a large fraction of these tiny pools.
    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    struct Scenario {
        int                                 cp, shards, count = 3;
        BackendEnvironment                  env;
        BlockIndicesType                    full, state;
        BatchKVCacheResource                resource;
        std::shared_ptr<MockClientWrapper>  client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<KVCMStorageBackend> backend;
        std::map<std::string, std::string>  specs;
        std::atomic<int>                    starts{0}, saves{0}, finishes{0}, aborts{0};
        bool                                block_start = false, fail_save = false, lose_commit = false;
        std::promise<void>                  entered, released;
        std::shared_future<void>            release = released.get_future().share();
        std::future<void>                   started = entered.get_future();
        std::unique_ptr<autil::EnvGuard>    target;
        explicit Scenario(int target_cp, bool sharded, bool bounded = true): cp(target_cp), shards(sharded ? cp : 1) {
            target                 = std::make_unique<autil::EnvGuard>("RTP_DTOP_FILE_TARGET_P_TP", std::to_string(cp));
            auto f                 = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto s                 = defaultCacheGroupPolicy(CacheGroupType::SWA);
            s.enable_prefix_reuse  = true;
            s.cp_mapping           = sharded ? CpBlockMappingMode::COMPACT_LAST_RANK : CpBlockMappingMode::NONE;
            s.cp_slice             = sharded ? CpBlockSliceMode::EQUAL_BYTES : CpBlockSliceMode::NONE;
            s.sliding_window_size  = bounded ? 256 * shards : 0;
            env.cache_config.dtype = DataType::TYPE_FP16;
            env.cache_config.layer_num          = 2;
            env.cache_config.seq_size_per_block = 128;
            env.cache_config.linear_step        = shards;
            env.cache_config.fromGroupedSpecs({test::makeMhaSpec("full", 128, DataType::TYPE_FP16, 1, 2),
                                               test::makeMhaSpec("state", 128 * shards, DataType::TYPE_FP16, 1, 2)},
                                              {{0}, {1}},
                                              {CacheGroupType::FULL, CacheGroupType::SWA},
                                              {"full", "state"},
                                              {f, s});
            env.cache_config.finalizeBlockNums(32, RuntimeConfig{});
            initializeEnvironmentPools(env, "eos_native_gpu", 32);
            full  = *env.pools_by_tag.at("full")->malloc((count + 1) * shards);
            state = *env.pools_by_tag.at("state")->malloc(count + 1);
            env.pools_by_tag.at("full")->incRef(full);
            env.pools_by_tag.at("state")->incRef(state);
            for (size_t i = 0; i < full.size(); ++i)
                fill(0, "full", full[i], i + 1);
            for (size_t i = 0; i < state.size(); ++i)
                fill(1, "state", state[i], 100 + i);
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            resource.setBatchBlocks(0, "full", full);
            resource.setBatchBlocks(0, "state", state);
            CacheKeysType keys;
            for (size_t i = 0; i < full.size(); ++i)
                keys.push_back(101 + i);
            resource.setBatchCacheKeys(0, keys);
            EXPECT_CALL(*client, initForPools(_, _, _, _))
                .WillOnce(Invoke([&](const auto&, auto, const auto& registrations, const auto& tags) {
                    EXPECT_EQ(registrations.size(), tags.size());
                    for (size_t i = 0; i < tags.size(); ++i)
                        specs[tags[i]] = registrations[i].location_spec_name;
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                        ++starts;
                        if (block_start) {
                            entered.set_value();
                            EXPECT_EQ(release.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                        }
                        kv_cache_manager::WriteLocation location;
                        location.write_session_id = "EOS";
                        location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
                        for (size_t key = 0; key < keys.size(); ++key) {
                            EXPECT_EQ(keys[key], 100 + (key + 1) * shards);
                            kv_cache_manager::Location one;
                            for (int peer = 0; peer < cp; ++peer) {
                                const auto suffix = std::to_string(key) + ":" + std::to_string(peer);
                                one.push_back({specs.at("full#cp" + std::to_string(peer)), "F:" + suffix});
                                if (keys.size() - key
                                    <= env.cache_config.topology().group("state").reuseBlockCount(keys.size()))
                                    one.push_back({specs.at("state#cp" + std::to_string(peer)), "S:" + suffix});
                            }
                            location.locations.push_back(one);
                        }
                        return std::make_pair(true, location);
                    }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                .WillRepeatedly(Invoke([&](const auto& tag, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    EXPECT_EQ(uris.size(), buffers.size());
                    kv_cache_manager::UriStrVec actual;
                    for (size_t i = 0; i < uris.size(); ++i) {
                        const int key = uris[i][2] - '0', peer = uris[i][4] - '0';
                        EXPECT_EQ(tag, (uris[i][0] == 'F' ? "full#cp" : "state#cp") + std::to_string(peer));
                        const int expected = uris[i][0] == 'F' ? key * shards + (shards > 1 ? peer : 0) + 1 : 100 + key;
                        for (const auto& iov : buffers[i].iovs) {
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                            std::vector<unsigned char> bytes(iov.size);
                            EXPECT_EQ(cudaMemcpy(bytes.data(), iov.base, iov.size, cudaMemcpyDeviceToHost),
                                      cudaSuccess);
                            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto b) { return b == expected; }));
                        }
                        actual.push_back("actual_" + uris[i]);
                    }
                    return std::make_pair(!fail_save, actual);
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                        if (locations.empty()) {
                            ++aborts;
                            return true;
                        }
                        ++finishes;
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), locations.size());
                        for (const auto& row : locations)
                            for (const auto& item : row)
                                EXPECT_EQ(item.uri.rfind("actual_", 0), 0u);
                        return !lose_commit;
                    }));
            KVCacheConfig k;
            k.kvcm_server_address   = "unused";
            k.kvcm_instance_group   = "linshuo_decode_remote_eos_gpu_test";
            k.kvcm_instance_id_salt = "eos_native_gpu";
            k.kvcm_model_sdk_config = "[{\"type\":\"pace\"}]";
            ParallelismConfig parallel;
            parallel.tp_size                            = 1;
            parallel.dp_size                            = cp;
            parallel.ep_size                            = cp;
            parallel.role_type                          = RoleType::DECODE;
            parallel.prefill_cp_config.prefill_cp_size  = cp;
            parallel.prefill_cp_config.kv_cache_sharded = sharded;
            RuntimeConfig rt;
            rt.model_name = "eos_gpu";
            backend       = std::make_unique<KVCMStorageBackend>(
                env.cache_config, k, rt, parallel, SpeculativeExecutionConfig{}, nullptr, false, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Scenario() {
            unblock();
            if (backend)
                backend->shutdown();
            dropRequestRefs();
        }
        void unblock() {
            try {
                released.set_value();
            } catch (...) {}
        }
        void fill(int layer, const std::string& tag, int block, int value) {
            for (const auto& part : environmentBuffers(env, layer, tag, block))
                EXPECT_EQ(cudaMemset(part.addr, value, part.size_bytes), cudaSuccess);
        }
        size_t end() const {
            return count * shards * 128;
        }
        bool refs = true;
        void dropRequestRefs() {
            if (refs) {
                env.pools_by_tag.at("full")->decRef(full);
                env.pools_by_tag.at("state")->decRef(state);
                refs = false;
            }
        }
    };
};
TEST_P(EosGpuCompletionTest, NoProgressSnapshotAndNativePagesSurviveRequestRelease) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    s.block_start = true;
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 0xee);
    for (size_t n = 128; n <= s.end(); n += 128)
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, n, false));
    EXPECT_EQ(s.starts, 0);
    EXPECT_EQ(s.saves, 0);
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 100 + i);
    // Change the final state after all progress callbacks. EOS must read this
    // GPU state, not any early host snapshot. Values remain same per-key here.
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    ASSERT_TRUE(s.resource.decodeCompletionSubmitted());
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[1]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[2]));
    EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(s.state[0]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    for (auto block : s.state)
        EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, PartialTailUsesEarlierImmutableCheckpointNotFinalState) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    // Active, incomplete final slot intentionally differs from complete state.
    s.fill(1, "state", s.state.back(), 0xee);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 128 * s.shards - 1, true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, MissingStateBacksOffWithoutRelabeling) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    auto     state = s.state;
    state[2]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, UnknownSdkCompletionFreezesPagesAndKeepsAdmissionCharged) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    s.fail_save = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 0);
    EXPECT_EQ(s.aborts, 1);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_GT(s.env.pools_by_tag.at("full")->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u);
    EXPECT_EQ(s.backend->quarantined_completion_admissions_.size(), 1u);
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.starts, 1);
    s.backend->shutdown();
    EXPECT_TRUE(s.backend->quarantined_completion_admissions_.empty());
    // Upstream keeps uncertain DMA pages FROZEN until pool destruction.
    for (int i = 0; i < s.count * s.shards; ++i) {
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isFrozen(s.full[i]));
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    }
}
TEST_P(EosGpuCompletionTest, DuplicateEosAndFullAdmissionDoNotDoublePinOrPublish) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    const auto held = s.env.pools_by_tag.at("full")->refCount(s.full[0]);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.env.pools_by_tag.at("full")->refCount(s.full[0]), held);
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 1);
    EXPECT_EQ(s.finishes, 1);
}
TEST_P(EosGpuCompletionTest, LostCommitAckDoesNotAbortOrQuarantineCompletedGpuTransfer) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()));
    s.lose_commit = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, UnboundedSwaNeverPretendsTwoTailPagesAreEnough) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), false);
    auto     state = s.state;
    state[0]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
}
INSTANTIATE_TEST_SUITE_P(CP2AndCP4ShardedAndReplicated,
                         EosGpuCompletionTest,
                         ::testing::Combine(::testing::Values(2, 4), ::testing::Bool()));
}  // namespace
}  // namespace rtp_llm
