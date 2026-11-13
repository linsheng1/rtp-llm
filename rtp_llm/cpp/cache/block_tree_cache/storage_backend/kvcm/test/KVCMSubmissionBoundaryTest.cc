#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
// Isolated white-box target: compile the CURRENT backend source, not a copied
// proof/settlement model. A test-only genrule includes its exact current bytes.
// Other targets still link the normal backend library.
// Preload its dependencies so the substitutions below affect only this backend
// and its test fixture, never the real BroadcastManager or CUDA declarations.
#include <cuda_runtime_api.h>
#include <condition_variable>
#include <thread>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cctype>
#include <exception>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <new>
#include <set>
#include <tuple>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <unistd.h>
#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PayloadDispatchProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PublicationSettlement.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopOriginMetadata.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/WorkerPayloadProof.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"

namespace rtp_llm {
class BoundaryBroadcastManager: public BroadcastManager {
public:
    using BroadcastManager::BroadcastManager;
    std::function<void(size_t)> before_rank;
    mutable std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> observed;

    template<class Request, class Response, class RpcCall>
    auto broadcast(const std::vector<Request>& requests, int timeout, const RpcCall& call,
                   std::optional<int64_t> id = {}, std::string direction = "unknown", bool fatal = true) const {
        size_t rank = 0;
        auto gate = [&](const auto& stub, const auto& context, const auto& request, auto* queue) {
            if (before_rank) before_rank(rank);
            ++rank;
            // Invoke the backend's ACTUAL preflight/marker lambda, including its
            // BroadcastRpcNotStarted throw. The gate never manufactures proof.
            return call(stub, context, request, queue);
        };
        auto result = BroadcastManager::broadcast<Request, Response>(requests, timeout, gate, id, direction, fatal);
        observed = result;
        return result;
    }
};
} // namespace rtp_llm

namespace {
thread_local std::function<void()> boundary_before_cuda_query;
cudaError_t boundaryCudaGetDevice(int* device) {
    if (boundary_before_cuda_query) boundary_before_cuda_query();
    return ::cudaGetDevice(device); // Keep the real CUDA call and device selection.
}
} // namespace

// Compile-time dependency gates exist only in this test translation unit. No
// production header, submission API, clock or runtime protocol gains a hook.
#define BroadcastManager BoundaryBroadcastManager
#define cudaGetDevice boundaryCudaGetDevice
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMSubmissionBoundaryImplementation.inc"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#undef cudaGetDevice
#undef BroadcastManager

namespace rtp_llm {
namespace {
using BoundaryClock = std::chrono::steady_clock;

class KVCMSubmissionBoundaryTest: public ::testing::Test {
protected:
    static constexpr auto kSetupWait = std::chrono::seconds(5);
    static constexpr auto kOperationTimeout = 2 * kSetupWait;
    static constexpr auto kSettlementWait = kOperationTimeout + kSetupWait;

    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    autil::EnvGuard async{"RTP_LLM_STREAM_ASYNC", "0"};
    autil::EnvGuard pending_bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    void SetUp() override { initLogger(); }

    struct Fixture {
        BackendEnvironment env = makeBackendEnvironment("submission_boundary");
        std::shared_ptr<MockClientWrapper> client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<MockQuiescentKVCMBackend> backend;
        std::atomic<int> starts{0}, saves{0}, closes{0};
        std::atomic<bool> bad_close{false};
        size_t refs = env.device_pool->refCount(env.block_id);

        Fixture(bool decode, std::shared_ptr<BoundaryBroadcastManager> broadcast = {}) {
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                    ++starts;
                    kv_cache_manager::WriteLocation location;
                    location.write_session_id = "boundary-session";
                    location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                    for (size_t key = 0; key < keys.size(); ++key) {
                        kv_cache_manager::Location row;
                        for (const auto& item : backend->impl_->group_policy_->spec_info_map())
                            row.push_back({item.first, "pace://boundary:" + std::to_string(item.second.tp_rank)});
                        location.locations.push_back(std::move(row));
                    }
                    return std::make_pair(true, location);
                }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    for (const auto& buffer : buffers)
                        for (const auto& iov : buffer.iovs)
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                    return std::make_pair(true, uris); // Completed source; destination still needs all peers.
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).WillRepeatedly(Invoke(
                [this](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                    ++closes;
                    const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&mask);
                    bad_close = !locations.empty() || !offset || *offset != 0;
                    return true; // ACK only an unused-session close.
                }));
            EXPECT_CALL(*client, abortWrite(_, _, _)).Times(0);
            KVCacheConfig config;
            config.kvcm_server_address = "unused-test-address";
            config.kvcm_instance_id_salt = decode ? "projected-boundary" : "broadcast-boundary";
            config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
            config.kvcm_asyncwrapper_thread_num = 1;
            config.kvcm_asyncwrapper_queue_size = 8;
            auto parallel = singleRankConfig();
            parallel.role_type = decode ? RoleType::DECODE : RoleType::PREFILL;
            parallel.tp_size = decode ? 1 : 2;
            parallel.prefill_cp_config.prefill_cp_size = 2;
            parallel.prefill_cp_config.kv_cache_sharded = false;
            RuntimeConfig runtime;
            runtime.model_name = "boundary_test";
            backend = std::make_unique<MockQuiescentKVCMBackend>(env.cache_config, config, runtime, parallel,
                SpeculativeExecutionConfig{}, std::move(broadcast), true, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Fixture() { if (backend) { backend->proveMockQuiescence(); backend->shutdown(); } }

        StorageRequest request(bool decode) {
            if (!decode) return makeStorageRequest(env);
            BatchKVCacheResource resource;
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            resource.setBatchBlocks(0, "default", {env.block_id});
            resource.setBatchCacheKeys(0, {101});
            return backend->impl_->planCompletedDecode(resource, env.cache_config.seq_size_per_block);
        }
        bool submit(StorageRequest request) {
            return backend->write(backend->prepareWrite(std::move(request)));
        }
        void expectSettled(bool retained, bool source_unknown) {
            {
                std::unique_lock<std::mutex> lock(backend->lifecycle_mutex_);
                ASSERT_TRUE(backend->lifecycle_cv_.wait_for(
                    lock, kSettlementWait, [this] { return backend->in_flight_ == 0; }));
            }
            EXPECT_EQ(starts.load(), 1);
            EXPECT_EQ(closes.load(), retained ? 0 : 1);
            EXPECT_FALSE(bad_close.load());
            const bool protected_owner = source_unknown && backend->protectUnknownCompletion();
            // Off uses original FROZEN references. On retains the live source
            // owner/pins until quiescence; freezing happens during shutdown.
            EXPECT_EQ(env.device_pool->isFrozen(env.block_id), source_unknown && !protected_owner);
            if (!source_unknown) EXPECT_EQ(env.device_pool->refCount(env.block_id), refs);
            else if (protected_owner) EXPECT_GT(env.device_pool->refCount(env.block_id), refs);
            std::lock_guard<std::mutex> lock(backend->quarantine_mutex_);
            EXPECT_EQ(static_cast<bool>(backend->protected_quarantine_), protected_owner);
            EXPECT_EQ(backend->quarantined_task_count_, source_unknown ? 1u : 0u);
        }
    };

    void broadcastCase(bool partial) {
        for (const char* enabled : {"0", "1"}) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled);
            ::rtp_llm::test::withDecodePublicationStartup([&] {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " partial=" << partial);
            auto first_state = std::make_shared<KVCMBroadcastState>();
            auto second_state = std::make_shared<KVCMBroadcastState>();
            auto arrived = std::make_shared<std::promise<void>>();
            auto arrival = arrived->get_future().share();
            first_state->on_request = [arrived] { arrived->set_value(); };
            KVCMBroadcastRpcServer first(0, first_state), second(1, second_state);
            ASSERT_TRUE(first.start()); ASSERT_TRUE(second.start());
            auto broadcast = std::make_shared<BoundaryBroadcastManager>(
                std::vector<std::string>{first.address(), second.address()});
            ASSERT_TRUE(broadcast->init()); // This is NOT the uninitialized-manager test.
            Fixture s(false, broadcast);
            ASSERT_TRUE(s.backend->initialized_);
            auto request = s.request(false);
            request.transfer_task.emplace(std::vector<TransferDescriptor>{}, kOperationTimeout);
            const auto deadline = request.transfer_task->deadline();
            auto gates = std::make_shared<std::atomic<int>>(0);
            broadcast->before_rank = [partial, deadline, arrival, gates](size_t rank) {
                ++*gates;
                if (rank != (partial ? 1u : 0u)) return;
                if (partial) EXPECT_EQ(arrival.wait_for(kSetupWait), std::future_status::ready);
                std::this_thread::sleep_until(deadline + std::chrono::milliseconds(1));
            };
            ASSERT_TRUE(s.submit(std::move(request)));
            ASSERT_NO_FATAL_FAILURE(s.expectSettled(partial, partial));
            ASSERT_NE(broadcast->observed, nullptr);
            EXPECT_TRUE(broadcast->observed->done());
            EXPECT_FALSE(broadcast->observed->success());
            if (partial) EXPECT_TRUE(broadcast->observed->rpcSucceeded(0));
            EXPECT_FALSE(broadcast->observed->rpcSucceeded(1));
            EXPECT_EQ(gates->load(), partial ? 2 : 1);
            EXPECT_EQ(snapshotRequests(first_state).size(), partial ? 1u : 0u);
            EXPECT_TRUE(snapshotRequests(second_state).empty());
            EXPECT_EQ(s.saves.load(), 0); // Real test RPC servers, no local SDK path.
            });
        }
    }

    void projectedCase(bool mixed) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        Fixture s(true);
        ASSERT_TRUE(s.backend->initialized_);
        auto request = s.request(true);
        ASSERT_FALSE(request.empty()); ASSERT_TRUE(request.decode_completion);
        auto deadline = std::make_shared<BoundaryClock::time_point>();
        auto checks = std::make_shared<std::atomic<int>>(0);
        auto entered = std::make_shared<std::promise<void>>();
        auto entered_future = entered->get_future();
        auto release = std::make_shared<std::promise<void>>();
        auto released = release->get_future().share();
        // Pin the sole REAL payload worker, not the business/completion lane.
        // Its next CUDA query is runProjectedWrite AFTER tryStart, BEFORE SDK.
        ASSERT_TRUE(s.backend->impl_->local_payload_pool_->submit(BlockTreeTaskClass::BACKGROUND,
            [entered, released, checks, deadline, mixed] {
                boundary_before_cuda_query = [checks, deadline, mixed] {
                    if (++*checks == (mixed ? 2 : 1))
                        std::this_thread::sleep_until(*deadline + std::chrono::milliseconds(1));
                };
                entered->set_value();
                EXPECT_EQ(released.wait_for(kSettlementWait), std::future_status::ready);
            }));
        block_tree_cache_detail::ScopeRollback unblock([release] { release->set_value(); });
        ASSERT_EQ(entered_future.wait_for(kSetupWait), std::future_status::ready);
        // Start the operation clock only after the worker gate is installed.
        // Promise release synchronizes this value with the worker's first query.
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, kOperationTimeout);
        *deadline = request.transfer_task->deadline();
        ASSERT_TRUE(s.submit(std::move(request)));
        // One worker + FIFO BACKGROUND barrier proves projected dispatch has
        // enqueued its first child while the payload worker is still held.
        auto barrier = std::make_shared<std::promise<void>>();
        auto queued = barrier->get_future();
        ASSERT_TRUE(s.backend->executor_->submit([barrier] { barrier->set_value(); }, BlockTreeTaskClass::BACKGROUND));
        ASSERT_EQ(queued.wait_for(kSetupWait), std::future_status::ready);
        EXPECT_EQ(s.backend->impl_->local_payload_pool_->queueSizes().background, 1u);
        unblock.run();
        // SDK success of peer0 proves source BUFFER_SAFE in the mixed case; it
        // must NOT be misused as permission to abort an incomplete destination.
        ASSERT_NO_FATAL_FAILURE(s.expectSettled(mixed, false));
        EXPECT_EQ(checks->load(), mixed ? 2 : 1);
        EXPECT_EQ(s.saves.load(), mixed ? 1 : 0);
        EXPECT_EQ(s.backend->impl_->completion_resources_->unresolved.load(), 0u);
        });
    }
};

TEST_F(KVCMSubmissionBoundaryTest, InitializedBroadcastWithZeroRankHandoffsClosesUnusedSession) {
    broadcastCase(false);
}
TEST_F(KVCMSubmissionBoundaryTest, PartialRankHandoffCannotBeDowngradedByNextRankExpiry) {
    broadcastCase(true);
}
TEST_F(KVCMSubmissionBoundaryTest, ProjectedAllPeersStopBeforeSdkClosesUnusedSession) {
    projectedCase(false);
}
TEST_F(KVCMSubmissionBoundaryTest, ProjectedOneSdkHandoffAndOnePreflightExpiryRetainsSession) {
    projectedCase(true);
}
} // namespace
} // namespace rtp_llm
