#include <cuda_runtime_api.h>
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include <algorithm>
#include <atomic>
#include <future>
#include <map>
#include <tuple>
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {
TEST(DecodePublicationConfigTest, GenericSwitchOverridesLegacyAndNeedsNoCpSpecificFlag) {
    autil::EnvGuard legacy("RTP_DTOP_FILE_CP2", "1");
    {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
    }
    {
        autil::EnvGuard on("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        EXPECT_TRUE(remoteCacheDecodePublicationEnabled());
    }
}
TEST(NativeSsdGpuPackingTest, GathersAndScattersDifferentGpuIovsWithoutHostPayloadStaging) {
    using namespace kv_cache_manager;
    using M = kv_cache_manager::MemoryType;
    auto allocate = [](size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error("test GPU allocation failed");
        return std::shared_ptr<void>(p, [](void* v) { cudaFree(v); });
    };
    auto arena = allocate(kvcm::PackedSsdTransferClient::max_operation_bytes);
    auto input = allocate(4096), output = allocate(4096), remote = allocate(4096);
    std::vector<unsigned char> expected(1280);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = (i * 31 + 17) % 256;
    ASSERT_EQ(cudaMemcpy(input.get(), expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
    auto* src = static_cast<unsigned char*>(input.get());
    auto* dst = static_cast<unsigned char*>(output.get());
    auto* storage = static_cast<unsigned char*>(remote.get());
    auto destroyed = std::make_shared<int>(0);
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    const UriStrVec uris{"object0", "object1"};
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(2).WillRepeatedly(
        Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(u.size(), 1u);
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].type, M::GPU);
            size_t offset = u[0] == uris[0] ? 0 : 257;
            EXPECT_EQ(b[0].iovs[0].size, u[0] == uris[0] ? 257u : 1023u);
            EXPECT_EQ(cudaMemcpy(storage + offset, b[0].iovs[0].base, b[0].iovs[0].size,
                                 cudaMemcpyDeviceToDevice), cudaSuccess);
            return std::make_pair(ER_OK, u);
        }));
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly(
        Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].type, M::GPU);
            size_t offset = u[0] == uris[0] ? 0 : 257;
            EXPECT_EQ(cudaMemcpy(b[0].iovs[0].base, storage + offset, b[0].iovs[0].size,
                                 cudaMemcpyDeviceToDevice), cudaSuccess);
            return ER_OK;
        }));
    kvcm::PackedSsdTransferClient client(
        std::move(delegate),
        [](void* d, M dt, const void* s, M st, size_t n) {
            EXPECT_EQ(dt, M::GPU);
            EXPECT_EQ(st, M::GPU);
            return dt == M::GPU && st == M::GPU
                   && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
        }, std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena));
    BlockBuffers source(2), destination(2);
    source[0].iovs = {{M::GPU, src, 17}, {M::GPU, src + 17, 240}};
    source[1].iovs = {{M::GPU, src + 257, 513}, {M::GPU, src + 770, 510}};
    destination[0].iovs = {{M::GPU, dst + 2048, 19}, {M::GPU, dst + 2067, 238}};
    destination[1].iovs = {{M::GPU, dst, 13}, {M::GPU, dst + 13, 1010}};
    ASSERT_EQ(client.SaveKvCaches(uris, source).first, ER_OK);
    ASSERT_EQ(client.LoadKvCaches(uris, destination), ER_OK);
    std::vector<unsigned char> first(257), second(1023);
    ASSERT_EQ(cudaMemcpy(first.data(), dst + 2048, first.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(second.data(), dst, second.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::equal(first.begin(), first.end(), expected.begin()));
    EXPECT_TRUE(std::equal(second.begin(), second.end(), expected.begin() + 257));
}
TEST(NativeSsdGpuPackingTest, ConcurrentLargeLogicalBatchesReserveOnlyOneGpuObjectEach) {
    using namespace kv_cache_manager;
    using M = kv_cache_manager::MemoryType;
    constexpr size_t object_bytes = 16ULL * 1024 * 1024;
    void* base = nullptr;
    void* source = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    auto arena = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMalloc(&source, object_bytes), cudaSuccess);
    auto input = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMemset(source, 0x5a, object_bytes), cudaSuccess);
    auto context = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto destroyed = std::make_shared<int>(0);
    auto first = std::make_unique<MockTransferClient>(destroyed);
    auto second = std::make_unique<MockTransferClient>(destroyed);
    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto released = release.get_future().share();
    std::atomic<int> first_calls{0};
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(
        Invoke([&](const auto& u, const auto& b, auto) {
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].size, object_bytes);
            if (first_calls.fetch_add(1) == 0) {
                entered.set_value();
                released.wait();
            }
            return std::make_pair(ER_OK, u);
        }));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(
        Invoke([&](const auto& u, const auto& b, auto) {
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].type, M::GPU);
            EXPECT_EQ(context->active_reservations, 2u);
            return std::make_pair(ER_OK, u);
        }));
    auto copy = [](void* d, M dt, const void* s, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    };
    kvcm::PackedSsdTransferClient a(std::move(first), copy, context);
    kvcm::PackedSsdTransferClient b(std::move(second), copy, context);
    BlockBuffers buffers(4);
    for (auto& block : buffers)
        block.iovs = {{M::GPU, source, object_bytes}};
    UriStrVec uris{"a", "b", "c", "d"};
    auto first_result = std::async(std::launch::async, [&] { return a.SaveKvCaches(uris, buffers).first; });
    const auto ready = started.wait_for(std::chrono::seconds(10));
    // Release before any fatal assertion so the test cannot strand the worker.
    const auto second_result = ready == std::future_status::ready ? b.SaveKvCaches(uris, buffers).first : ER_INVALID_PARAMS;
    release.set_value();
    EXPECT_EQ(first_result.get(), ER_OK);
    ASSERT_EQ(ready, std::future_status::ready);
    EXPECT_EQ(second_result, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST(NativeSsdGpuPackingTest, UnknownCompletionRetainsItsRangeButOtherTagsContinue) {
    using M = kv_cache_manager::MemoryType;
    void* base = nullptr;
    void* source = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&source, 1024), cudaSuccess);
    auto arena = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    auto input = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    auto context = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto destroyed = std::make_shared<int>(0);
    auto first = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    auto second = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).WillOnce(::testing::Throw(std::runtime_error("uncertain DMA")));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).WillOnce(Invoke([](const auto& u, const auto&, auto) { return std::make_pair(kv_cache_manager::ER_OK, u); }));
    auto copy = [](void* d, M dt, const void* s, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    };
    kvcm::PackedSsdTransferClient a(std::move(first), copy, context);
    kvcm::PackedSsdTransferClient b(std::move(second), copy, context);
    kv_cache_manager::BlockBuffers buffers(1);
    buffers[0].iovs = {{M::GPU, source, 1024}};
    ASSERT_EQ(cudaMemset(source, 0x5a, 1024), cudaSuccess);
    EXPECT_THROW(a.SaveKvCaches({"first"}, buffers), std::runtime_error);
    EXPECT_EQ(context->active_reservations, 1u);
    ASSERT_EQ(cudaMemset(source, 0x22, 1024), cudaSuccess);
    EXPECT_EQ(b.SaveKvCaches({"second"}, buffers).first, kv_cache_manager::ER_OK);
    unsigned char retained = 0;
    ASSERT_EQ(cudaMemcpy(&retained, arena.get(), 1, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(retained, 0x5a);
}
// GPU pages + recording SDK. CPU copies below are assertions only; the
// implementation must pass native GPU IOVs through the Prefill transfer path.
class EosGpuCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool, int>> {
protected:
    autil::EnvGuard enable{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1"}, pace{"RTP_DTOP_PACE", "0"}, audit{"RTP_DTOP_PAYLOAD_AUDIT", "0"},
        pending{"RTP_DTOP_GPU_PENDING_JOBS", "1"}, obsolete_disable{"RTP_DTOP_DECODE_PUBLISH", "0"};
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
        explicit Scenario(int target_cp, bool sharded, bool bounded = true, bool native_ssd_gdr = false, int decode_dp = 1):
            cp(target_cp), shards(sharded ? cp : 1) {
            // The target CP comes only from ParallelismConfig, not a CP-specific environment flag.
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
            const int pool_blocks = std::max(32, (count + 1) * shards + 8);
            env.cache_config.finalizeBlockNums(pool_blocks, RuntimeConfig{});
            initializeEnvironmentPools(env, "eos_native_gpu", pool_blocks);
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
                        const int key = std::stoi(uris[i].substr(2)), peer = std::stoi(uris[i].substr(4));
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
            if (native_ssd_gdr) {
                k.kvcm_model_sdk_config = "[{\"type\":\"pace_ssd\"}]";
                k.kvcm_read_backend_type = 9;
            }
            ParallelismConfig parallel;
            parallel.tp_size                            = 1;
            parallel.dp_size                            = decode_dp;
            parallel.ep_size                            = decode_dp;
            parallel.role_type                          = RoleType::DECODE;
            parallel.prefill_cp_config.prefill_cp_size  = cp;
            parallel.prefill_cp_config.kv_cache_sharded = sharded;
            RuntimeConfig rt;
            rt.model_name = "eos_gpu";
            backend       = std::make_unique<KVCMStorageBackend>(
                env.cache_config, k, rt, parallel, SpeculativeExecutionConfig{}, nullptr, true, nullptr, client);
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
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
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
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    // Active, incomplete final slot intentionally differs from complete state.
    s.fill(1, "state", s.state.back(), 0xee);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 128 * s.shards - 1, true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, NativeSsdGdrKeepsGpuPagesThroughEosTransfer) {
    autil::EnvGuard native_ssd{"RTP_DTOP_PACE_SSD_NATIVE_GDR", "1"};
    autil::EnvGuard packed_ssd{"RTP_DTOP_PACE_SSD_PACKED_IO", "0"};
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, true, std::get<2>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, MissingStateBacksOffWithoutRelabeling) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[2]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, UnknownSdkCompletionFreezesPagesAndKeepsAdmissionCharged) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
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
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
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
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
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
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), false, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[0]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
}
INSTANTIATE_TEST_SUITE_P(GenericCPAndIndependentDP,
                         EosGpuCompletionTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8, 16), ::testing::Bool(), ::testing::Values(1, 3, 8)));
}  // namespace
}  // namespace rtp_llm
