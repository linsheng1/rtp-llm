#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include <cuda_runtime_api.h>
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PayloadDispatchProof.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PublicationSettlement.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <future>
#include <map>
#include <tuple>
#include <typeinfo>
#include <stdexcept>
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {
template<typename C>
C expectedPreflightError(C legacy) {
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<C>::value) return C::ER_SDK_ERROR_BUFFER_SAFE;
    return legacy;
}
TEST(DecodePublicationProofTest, AbortRequiresProvenNoPayloadAndNoFinishAttempt) {
    using Proof = kvcm::PayloadDispatchProof;
    using Action = kvcm::FailedPublicationAction;
    Proof proof;
    EXPECT_EQ(proof.state(), Proof::State::NOT_HANDED_OFF);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::ABORT_UNCOMMITTED);
    proof.pending();
    EXPECT_EQ(proof.state(), Proof::State::HANDOFF_PENDING);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::RETAIN_SESSION);
    proof.notStarted();
    EXPECT_EQ(proof.state(), Proof::State::PROVEN_NOT_STARTED);
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), false), Action::ABORT_UNCOMMITTED);
    // A FinishWrite attempt alone still forbids abort or replay, even without payload.
    EXPECT_EQ(kvcm::failedPublicationAction(!proof.neverStarted(), true), Action::RETAIN_SESSION);
    Proof possible;
    possible.pending();
    possible.started();
    possible.notStarted();
    possible.pending();
    EXPECT_EQ(possible.state(), Proof::State::MAY_HAVE_STARTED);
    EXPECT_FALSE(possible.neverStarted());
    EXPECT_EQ(kvcm::failedPublicationAction(!possible.neverStarted(), false), Action::RETAIN_SESSION);
}

TEST(ClientFactoryFeatureBoundaryTest, DisabledAndUnsetDelegateConfigurationErrorsToTheSdk) {
    using namespace kv_cache_manager;
    // Invalid JSON cannot initiate network I/O. Compare against the SDK itself,
    // whose legacy error/exception contract the disabled feature must preserve.
    // Our SSD parser used to throw before the SDK was called, even when off.
    const std::string         config = "{";
    InitParams                params{};
    ClientMemoryRegistrations registrations;
    kvcm::ClientFactory       factory;
    const auto                observe = [](auto call) -> std::pair<bool, std::string> {
        try {
            return {static_cast<bool>(call()), {}};
        } catch (const std::exception& e) {
            return {false, std::string(typeid(e).name()) + ":" + e.what()};
        } catch (...) {
            return {false, "non-standard exception"};
        }
    };
    for (bool unset : {false, true}) {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) {
            ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        }
        EXPECT_EQ(observe([&] { return factory.createTransferClient(config, params); }),
                  observe([&] { return TransferClient::Create(config, params); }))
            << unset;
        EXPECT_EQ(observe([&] { return factory.createTransferClientWithMemory(config, params, registrations); }),
                  observe([&] { return TransferClient::Create(config, params, registrations); }))
            << unset;
        });
    }
}
TEST(ClientFactoryFeatureBoundaryTest, EnabledSsdStillRejectsMissingGpuRegistrationAndMixedBackends) {
    autil::EnvGuard                             enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    kvcm::ClientFactory                         factory;
    kv_cache_manager::InitParams                params{};
    kv_cache_manager::ClientMemoryRegistrations registrations;
    const std::string                           ssd = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace_ssd"}]}})";
    const std::string mixed = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace"},{"type":"pace_ssd"}]}})";
    EXPECT_THROW(factory.createTransferClient(ssd, params), std::invalid_argument);
    EXPECT_THROW(factory.createTransferClientWithMemory(ssd, params, registrations), std::invalid_argument);
    EXPECT_THROW(factory.createTransferClientWithMemory(mixed, params, registrations), std::invalid_argument);
    });
}
TEST(ClientFactoryFeatureBoundaryTest, EmptyDescriptorsCanReuseOnlySuccessfullyRegisteredGpuSubspans) {
    using Context      = kvcm::PackedSsdTransferClient::GpuPackingContext;
    const auto pointer = [](uintptr_t value) { return reinterpret_cast<void*>(value); };
    Context    context(std::shared_ptr<void>(pointer(0x10000), [](void*) {}));
    context.registered_gpu_spans = {{pointer(0x10000), 0x4000, 0}, {pointer(0x20000), 0x8000, 0}};
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 0));
    context.registered = true;
    EXPECT_TRUE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 0));
    EXPECT_TRUE(context.coversRegisteredGpuSpan({pointer(0x22000), 0x6000}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x22000), 0x6001}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x28000), 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x0ffff), 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 1));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({nullptr, 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(UINTPTR_MAX - 1), 8}, 0));
}
TEST(DecodePublicationConfigTest, SingleSwitchNeedsNoCpSpecificFlag) {
    {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
        });
    }
    {
        autil::EnvGuard on("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        EXPECT_TRUE(remoteCacheDecodePublicationEnabled());
        });
    }
}
TEST(DecodePublicationConfigTest, PendingBudgetsHaveBoundedDefaultsAndStrictOverrides) {
    EXPECT_EQ(DecodePublicationLimits{}.bytes, 512ULL * 1024 * 1024);
    EXPECT_EQ(DecodePublicationLimits{}.jobs, 4u);
    autil::EnvGuard mib("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "256");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "8");
    auto            limits = remoteCacheDecodePublicationLimits();
    ASSERT_TRUE(limits.has_value());
    EXPECT_EQ(limits->bytes, 256ULL * 1024 * 1024);
    EXPECT_EQ(limits->jobs, 8u);
    for (const char* invalid : {"", "0", "-1", "1.5", " 8", "8x", "18446744073709551616"}) {
        autil::EnvGuard bad("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", invalid);
        EXPECT_FALSE(remoteCacheDecodePublicationLimits().has_value()) << invalid;
    }
    autil::EnvGuard too_many("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "65");
    EXPECT_FALSE(remoteCacheDecodePublicationLimits().has_value());
}
TEST(DecodePublicationConfigTest, DefaultBudgetTracksPhysicalPoolWithoutRoundingSmallPoolsUp) {
    autil::EnvGuard mib("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "4");
    ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB"), 0);
    ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS"), 0);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(0).has_value());
    for (const auto& sample :
         std::vector<std::pair<size_t, size_t>>{{8, 1},
                                                {8192, 1024},
                                                {4ULL * 1024 * 1024 * 1024, 512ULL * 1024 * 1024},
                                                {64ULL * 1024 * 1024 * 1024, 4ULL * 1024 * 1024 * 1024}}) {
        const auto limits = remoteCacheDecodePublicationLimits(sample.first);
        ASSERT_TRUE(limits.has_value());
        EXPECT_EQ(limits->bytes, sample.second);
        EXPECT_EQ(limits->jobs, 4u);
    }
}


TEST(FeatureOffNativeTest, PrefillRemoteRoundTripPreservesLocalPrefixPayloadAndReferences) {
    for (const bool unset : {false, true}) {
        SCOPED_TRACE(unset);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        auto env = makeBackendEnvironment("feature_off_prefill_payload");
        ScopedReferencedBlocks pages(env.device_pool, 3);
        const auto ids = pages.get();
        const auto pool = env.device_pool;
        std::vector<size_t> refs;
        for (size_t i = 0; i < ids.size(); ++i) {
            refs.push_back(pool->refCount(ids[i]));
            for (const auto& part : environmentBuffers(env, 0, "default", ids[i]))
                ASSERT_EQ(cudaMemset(part.addr, 0x41 + i, part.size_bytes), cudaSuccess);
        }
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::PREFILL;
        auto backend = makeBackend(env, parallel, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, env));
        kv_cache_manager::Locations locations{{{"tp0_Fdefault", "off://102"}},
                                             {{"tp0_Fdefault", "off://103"}}};
        EXPECT_CALL(*client, match(_, _, _, std::vector<int64_t>({101, 102, 103}), _, _))
            .WillOnce(Return(std::make_pair(true, locations)));
        auto request = makeStorageRequest(env, {101, 102, 103}, 1, ids);
        auto hit = match(*backend.backend, request);
        ASSERT_TRUE(hit.success);
        ASSERT_EQ(hit.matched_blocks_num, 3u);
        ASSERT_NE(hit.match_meta, nullptr);
        EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec({"off://102", "off://103"}), _, _))
            .WillOnce(Invoke([](const auto&, const auto&, auto& buffers, const auto&) {
                if (buffers.size() != 2u) return false;
                for (size_t i = 0; i < buffers.size(); ++i)
                    for (const auto& iov : buffers[i].iovs)
                        if (cudaMemset(iov.base, 0x52 + i, iov.size) != cudaSuccess) return false;
                return true;
            }));
        ASSERT_TRUE(read(*backend.backend, request, std::move(hit.match_meta)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_EQ(pool->refCount(ids[i]), refs[i]);
            EXPECT_FALSE(pool->isFrozen(ids[i]));
            for (const auto& part : environmentBuffers(env, 0, "default", ids[i])) {
                std::vector<unsigned char> bytes(part.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), part.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                const auto expected = i == 0 ? 0x41 : 0x51 + i;
                EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == expected; }));
            }
        }
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "feature-off-acknowledged";
        allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations = locations;
        EXPECT_CALL(*client, getWriteLocation(_, _, std::vector<int64_t>({102, 103}), _, _, _, 0))
            .WillOnce(Return(std::make_pair(true, allocation)));
        EXPECT_CALL(*client, saveKvCachesForTag("default", kv_cache_manager::UriStrVec({"off://102", "off://103"}), _, _))
            .WillOnce(Invoke([](const auto&, const auto& uris, const auto& buffers, const auto&) {
                if (buffers.size() != 2u) return std::make_pair(false, uris);
                for (size_t i = 0; i < buffers.size(); ++i)
                    for (const auto& iov : buffers[i].iovs) {
                        std::vector<unsigned char> bytes(iov.size);
                        if (cudaMemcpy(bytes.data(), iov.base, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
                            return std::make_pair(false, uris);
                        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == 0x52 + i; }));
                    }
                return std::make_pair(true, uris);
            }));
        EXPECT_CALL(*client, finishWrite(_, _, "feature-off-acknowledged",
                    ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{2}}), locations))
            .WillOnce(Return(true));
        EXPECT_CALL(*client, abortWrite(_, _, _)).Times(0);
        auto task = backend.backend->prepareWrite(makeStorageRequest(env, {102, 103}, 0, {ids[1], ids[2]}));
        ASSERT_TRUE(static_cast<bool>(task));
        ASSERT_TRUE(backend.backend->write(std::move(task)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        backend.backend->shutdown();
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_EQ(pool->refCount(ids[i]), refs[i]);
            EXPECT_FALSE(pool->isFrozen(ids[i]));
        }
        });
    }
}

TEST(NativeSsdGpuPackingTest, GathersAndScattersDifferentGpuIovsWithoutHostPayloadStaging) {
    using namespace kv_cache_manager;
    using M       = kv_cache_manager::MemoryType;
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    auto allocate = [](size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error("test GPU allocation failed");
        return std::shared_ptr<void>(p, [](void* v) { cudaFree(v); });
    };
    auto                       arena = allocate(kvcm::PackedSsdTransferClient::max_operation_bytes);
    auto                       input = allocate(4096), output = allocate(4096), remote = allocate(4096);
    std::vector<unsigned char> expected(1280);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = (i * 31 + 17) % 256;
    ASSERT_EQ(cudaMemcpy(input.get(), expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
    auto*           src       = static_cast<unsigned char*>(input.get());
    auto*           dst       = static_cast<unsigned char*>(output.get());
    auto*           storage   = static_cast<unsigned char*>(remote.get());
    auto            destroyed = std::make_shared<int>(0);
    auto            delegate  = std::make_unique<MockTransferClient>(destroyed);
    const UriStrVec uris{"object0", "object1"};
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .WillOnce(Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(u, uris);
            EXPECT_EQ(b.size(), 2u);
            for (size_t i = 0; i < b.size(); ++i) {
                EXPECT_EQ(b[i].iovs.size(), 1u);
                EXPECT_EQ(b[i].iovs[0].type, M::GPU);
                const size_t offset = i == 0 ? 0 : 257;
                EXPECT_EQ(b[i].iovs[0].size, i == 0 ? 257u : 1023u);
                EXPECT_EQ(cudaMemcpy(storage + offset, b[i].iovs[0].base, b[i].iovs[0].size,
                                     cudaMemcpyDeviceToDevice), cudaSuccess);
            }
            const auto first = reinterpret_cast<uintptr_t>(b[0].iovs[0].base);
            const auto second = reinterpret_cast<uintptr_t>(b[1].iovs[0].base);
            EXPECT_GE(second, first + b[0].iovs[0].size);
            return std::make_pair(ER_OK, u);
        }));
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].type, M::GPU);
            size_t offset = u[0] == uris[0] ? 0 : 257;
            EXPECT_EQ(cudaMemcpy(b[0].iovs[0].base, storage + offset, b[0].iovs[0].size, cudaMemcpyDeviceToDevice),
                      cudaSuccess);
            return ER_OK;
        }));
    auto context = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena, device);
    context->registered_gpu_spans = {{input.get(), 4096, device}, {output.get(), 4096, device}};
    context->registered = true;
    std::vector<size_t> copy_sizes;
    kvcm::PackedSsdTransferClient client(
        std::move(delegate),
        [&](void* d, M dt, const void* s, M st, size_t n) {
            EXPECT_EQ(dt, M::GPU);
            EXPECT_EQ(st, M::GPU);
            copy_sizes.push_back(n);
            return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess
                   && cudaStreamSynchronize(nullptr) == cudaSuccess;
        },
        context);
    BlockBuffers source(2), destination(2);
    source[0].iovs      = {{M::GPU, src, 17}, {M::GPU, src + 17, 80}, {M::GPU, src + 97, 160}};
    source[1].iovs      = {{M::GPU, src + 257, 513}, {M::GPU, src + 770, 510}};
    destination[0].iovs = {{M::GPU, dst + 2048, 19}, {M::GPU, dst + 2067, 238}};
    destination[1].iovs = {{M::GPU, dst, 13}, {M::GPU, dst + 13, 1010}};
    ASSERT_EQ(client.SaveKvCaches(uris, source).first, ER_OK);
    ASSERT_EQ(client.LoadKvCaches(uris, destination), ER_OK);
    std::vector<unsigned char> first(257), second(1023);
    ASSERT_EQ(cudaMemcpy(first.data(), dst + 2048, first.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(second.data(), dst, second.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::equal(first.begin(), first.end(), expected.begin()));
    EXPECT_TRUE(std::equal(second.begin(), second.end(), expected.begin() + 257));
    EXPECT_EQ(copy_sizes, (std::vector<size_t>{257, 1023, 257, 1023}));
}
TEST(NativeSsdGpuPackingTest, CopyRunsRespectRegistrationGapsIgnoredOffsetsAliasingAndFailure) {
    using namespace kv_cache_manager;
    using Client = kvcm::PackedSsdTransferClient;
    using M = kv_cache_manager::MemoryType;
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    const auto allocate = [](size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) throw std::runtime_error("test GPU allocation failed");
        return std::shared_ptr<void>(p, [](void* v) { cudaFree(v); });
    };
    auto arena = allocate(Client::max_operation_bytes), data = allocate(1024);
    auto* bytes = static_cast<unsigned char*>(data.get());
    auto* staging = static_cast<unsigned char*>(arena.get());
    enum Scenario { GAP, SPLIT_REGISTRATION, DEVICE_MISMATCH, IGNORED_READ, STAGING_ALIAS, FAILED_RUN };
    for (auto scenario : {GAP, SPLIT_REGISTRATION, DEVICE_MISMATCH, IGNORED_READ, STAGING_ALIAS, FAILED_RUN}) {
        SCOPED_TRACE(scenario);
        auto context = std::make_shared<Client::GpuPackingContext>(arena, device);
        context->registered = true;
        context->registered_gpu_spans = {{data.get(), 1024, device}, {arena.get(), Client::max_operation_bytes, device}};
        if (scenario == SPLIT_REGISTRATION)
            context->registered_gpu_spans = {{bytes, 4, device}, {bytes + 4, 4, device}};
        if (scenario == DEVICE_MISMATCH)
            context->registered_gpu_spans = {{data.get(), 1024, device + 1}};
        auto destroyed = std::make_shared<int>(0);
        auto delegate = std::make_unique<MockTransferClient>(destroyed);
        const UriStrVec uris{"object"};
        if (scenario == IGNORED_READ) {
            EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Invoke([&](const auto& u, const auto& b, auto) {
                EXPECT_EQ(u, uris);
                EXPECT_EQ(b.size(), 1u);
                EXPECT_EQ(b[0].iovs.size(), 1u);
                EXPECT_EQ(b[0].iovs[0].size, 12u);
                EXPECT_EQ(cudaMemset(b[0].iovs[0].base, 0x2a, 12), cudaSuccess);
                EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
                return ER_OK;
            }));
        } else if (scenario == FAILED_RUN) {
            EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
        } else {
            EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(Invoke([&](const auto& u, const auto& b, auto) {
                EXPECT_EQ(u, uris);
                EXPECT_EQ(b.size(), 1u);
                EXPECT_EQ(b[0].iovs.size(), 1u);
                EXPECT_EQ(b[0].iovs[0].size, 8u);
                return std::make_pair(ER_OK, u);
            }));
        }
        std::vector<std::tuple<void*, const void*, size_t>> copies;
        Client client(std::move(delegate), [&](void* d, M dt, const void* s, M st, size_t n) {
            copies.emplace_back(d, s, n);
            EXPECT_EQ(dt, M::GPU);
            EXPECT_EQ(st, M::GPU);
            if (scenario == FAILED_RUN) return false; // No GPU work submitted by this fake failure.
            return cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess
                   && cudaStreamSynchronize(nullptr) == cudaSuccess;
        }, context);
        BlockBuffers buffers(1);
        buffers[0].iovs = {{M::GPU, bytes, 4}, {M::GPU, bytes + (scenario == GAP ? 8 : 4), 4}};
        if (scenario == STAGING_ALIAS)
            buffers[0].iovs = {{M::GPU, staging + 4, 4}, {M::GPU, staging + 8, 4}};
        if (scenario == IGNORED_READ) {
            buffers[0].iovs = {{M::GPU, bytes, 4}, {M::GPU, bytes + 4, 4, true}, {M::GPU, bytes + 8, 4}};
            ASSERT_EQ(cudaMemset(bytes, 0x5a, 12), cudaSuccess);
            ASSERT_EQ(client.LoadKvCaches(uris, buffers), ER_OK);
            ASSERT_EQ(copies.size(), 2u);
            EXPECT_EQ(std::get<1>(copies[1]), staging + 8);
            EXPECT_EQ(std::get<0>(copies[1]), bytes + 8);
            std::vector<unsigned char> actual(12);
            ASSERT_EQ(cudaMemcpy(actual.data(), bytes, 12, cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(actual, (std::vector<unsigned char>{42,42,42,42,90,90,90,90,42,42,42,42}));
        } else if (scenario == FAILED_RUN) {
            EXPECT_EQ(client.SaveKvCaches(uris, buffers).first, ER_CUDAMEMCPY_ERROR);
            ASSERT_EQ(copies.size(), 1u);
            EXPECT_EQ(std::get<2>(copies[0]), 8u);
            EXPECT_EQ(client.SaveKvCaches(uris, buffers).first, expectedPreflightError(ER_INVALID_LOCAL_BUFFERS));
            EXPECT_EQ(copies.size(), 1u); // The quarantined operation cannot be overwritten.
        } else {
            ASSERT_EQ(client.SaveKvCaches(uris, buffers).first, ER_OK);
            ASSERT_EQ(copies.size(), 2u);
            EXPECT_EQ(std::get<2>(copies[0]), 4u);
            EXPECT_EQ(std::get<2>(copies[1]), 4u);
        }
    }
}

TEST(NativeSsdGpuPackingTest, ConcurrentLargeLogicalBatchesReserveOnlyOneGpuObjectEach) {
    using namespace kv_cache_manager;
    using M                       = kv_cache_manager::MemoryType;
    constexpr size_t object_bytes = 16ULL * 1024 * 1024;
    void*            base         = nullptr;
    void*            source       = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    auto arena = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMalloc(&source, object_bytes), cudaSuccess);
    auto input = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMemset(source, 0x5a, object_bytes), cudaSuccess);
    auto               context   = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto               destroyed = std::make_shared<int>(0);
    auto               first     = std::make_unique<MockTransferClient>(destroyed);
    auto               second    = std::make_unique<MockTransferClient>(destroyed);
    std::promise<void> entered, release;
    auto               started  = entered.get_future();
    auto               released = release.get_future().share();
    std::atomic<int>   first_calls{0};
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(Invoke([&](const auto& u, const auto& b, auto) {
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].iovs[0].size, object_bytes);
        if (first_calls.fetch_add(1) == 0) {
            entered.set_value();
            released.wait();
        }
        return std::make_pair(ER_OK, u);
    }));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(Invoke([&](const auto& u, const auto& b, auto) {
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
    BlockBuffers                  buffers(4);
    for (auto& block : buffers)
        block.iovs = {{M::GPU, source, object_bytes}};
    UriStrVec  uris{"a", "b", "c", "d"};
    auto       first_result = std::async(std::launch::async, [&] { return a.SaveKvCaches(uris, buffers).first; });
    const auto ready        = started.wait_for(std::chrono::seconds(10));
    // Release before any fatal assertion so the test cannot strand the worker.
    const auto second_result =
        ready == std::future_status::ready ? b.SaveKvCaches(uris, buffers).first : ER_INVALID_PARAMS;
    release.set_value();
    EXPECT_EQ(first_result.get(), ER_OK);
    ASSERT_EQ(ready, std::future_status::ready);
    EXPECT_EQ(second_result, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST(NativeSsdGpuPackingTest, UnknownCompletionRetainsItsRangeButOtherTagsContinue) {
    using M      = kv_cache_manager::MemoryType;
    void* base   = nullptr;
    void* source = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&source, 1024), cudaSuccess);
    auto arena     = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    auto input     = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    auto context   = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto destroyed = std::make_shared<int>(0);
    auto first     = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    auto second    = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).WillOnce(::testing::Throw(std::runtime_error("uncertain DMA")));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).WillOnce(Invoke([](const auto& u, const auto&, auto) {
        return std::make_pair(kv_cache_manager::ER_OK, u);
    }));
    auto copy = [](void* d, M dt, const void* s, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    };
    kvcm::PackedSsdTransferClient  a(std::move(first), copy, context);
    kvcm::PackedSsdTransferClient  b(std::move(second), copy, context);
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
class NativeSsdGpuReadWindowTest: public ::testing::Test {
protected:
    using M       = kv_cache_manager::MemoryType;
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Client  = kvcm::PackedSsdTransferClient;
    std::shared_ptr<void>                      arena, output;
    std::shared_ptr<Client::GpuPackingContext> context;
    std::shared_ptr<int>                       destroyed = std::make_shared<int>(0);
    void                                       SetUp() override {
        void* a = nullptr;
        void* b = nullptr;
        ASSERT_EQ(cudaMalloc(&a, Client::max_operation_bytes), cudaSuccess);
        arena = std::shared_ptr<void>(a, [](void* p) { cudaFree(p); });
        ASSERT_EQ(cudaMalloc(&b, 2 * 1024 * 1024), cudaSuccess);
        output = std::shared_ptr<void>(b, [](void* p) { cudaFree(p); });
        ASSERT_EQ(cudaMemset(b, 0, 2 * 1024 * 1024), cudaSuccess);
        context = std::make_shared<Client::GpuPackingContext>(arena);
    }
    static bool copy(void* dst, M dt, const void* src, M st, size_t bytes) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess
               && cudaStreamSynchronize(nullptr) == cudaSuccess;
    }
    Buffers destinations(const std::vector<size_t>& sizes) {
        Buffers b(sizes.size());
        size_t  offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            b[i].iovs = {{M::GPU, static_cast<unsigned char*>(output.get()) + offset, sizes[i], false}};
            offset += sizes[i];
        }
        return b;
    }
    void registerPipelineSpans() {
        int device = -1;
        ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
        context = std::make_shared<Client::GpuPackingContext>(arena, device);
        context->registered_gpu_spans = {{output.get(), 2 * 1024 * 1024, device},
                                          {arena.get(), Client::max_operation_bytes, device}};
        context->registered = true;
    }
    size_t leasedBytes() {
        std::lock_guard<std::mutex> lock(context->mutex);
        size_t free = 0;
        for (const auto& range : context->free_ranges) free += range.second;
        return Client::max_operation_bytes - free;
    }
    template<typename C>
    void checkSafeSdkRetry() {
        if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<C>::value) {
            for (bool write : {false, true}) {
                auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
                if (write) {
                    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
                        .WillOnce(Return(std::make_pair(C::ER_SDK_ERROR_BUFFER_SAFE, Uris{})))
                        .WillOnce(Return(std::make_pair(C::ER_OK, Uris{})));
                } else {
                    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
                        .WillOnce(Return(C::ER_SDK_TIMEOUT_BUFFER_SAFE)).WillOnce(Return(C::ER_OK));
                }
                Client client(std::move(delegate), copy, context);
                auto buffers = destinations({17});
                if (write) {
                    EXPECT_EQ(client.SaveKvCaches({"safe0", "safe1"}, destinations({17, 33})).first,
                              C::ER_SDK_ERROR_BUFFER_SAFE);
                    EXPECT_EQ(context->active_reservations, 0u);
                    EXPECT_EQ(client.SaveKvCaches({"retry0", "retry1"}, destinations({17, 33})).first, C::ER_OK);
                } else {
                    EXPECT_EQ(client.LoadKvCaches({"safe"}, buffers), C::ER_SDK_TIMEOUT_BUFFER_SAFE);
                    EXPECT_EQ(context->active_reservations, 0u);
                    EXPECT_EQ(client.LoadKvCaches({"retry"}, buffers), C::ER_OK);
                }
                EXPECT_EQ(context->active_reservations, 0u);
            }
        } else {
            GTEST_SKIP() << "SDK has no explicit buffer-safe failure contract";
        }
    }
    template<typename C>
    void checkMixedPeerSafety() {
        if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<C>::value) {
            for (bool safe_first : {true, false}) {
                auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
                const Uris uris = safe_first ? Uris{"safe", "unknown"} : Uris{"unknown", "safe"};
                EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Invoke([&](const Uris& u, const Buffers& b, auto) -> C {
                    EXPECT_EQ(u, uris);
                    EXPECT_EQ(b.size(), 2u);
                    return C::ER_SDK_TIMEOUT; // The SDK reports one unknown outcome for the whole batch.
                }));
                Client client(std::move(delegate), copy, context, 4);
                auto buffers = destinations({17, 17});
                EXPECT_EQ(client.LoadKvCaches(uris, buffers), C::ER_SDK_TIMEOUT);
                EXPECT_EQ(context->active_reservations, 1u);
                EXPECT_EQ(client.LoadKvCaches(uris, buffers), expectedPreflightError(C::ER_INVALID_LOCAL_BUFFERS));
            }
        } else {
            GTEST_SKIP() << "SDK has no explicit buffer-safe failure contract";
        }
    }
};
// Deferred test copies execute only at Fence. A one-bank implementation that
// starts the next SDK load before draining them corrupts the asserted GPU bytes.
TEST_F(NativeSsdGpuReadWindowTest, ReadPipelineAlternatesBanksAndPreservesAliasedDestinationOrder) {
    for (bool alias : {false, true}) {
        ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
        std::vector<std::string> events;
        std::vector<std::tuple<void*, const void*, size_t>> pending;
        std::vector<void*> banks;
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(3)
            .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto trace) {
                const size_t window = banks.size();
                events.push_back("L" + std::to_string(window));
                EXPECT_EQ(pending.size(), window ? 1u : 0u);
                EXPECT_EQ(u, (Uris{std::to_string(window)}));
                EXPECT_TRUE(trace && trace->block_ids == Uris{std::to_string(100 + window)});
                if (b.size() != 1 || b[0].iovs.size() != 1) return Code::ER_INVALID_PARAMS;
                banks.push_back(b[0].iovs[0].base);
                EXPECT_EQ(cudaMemset(b[0].iovs[0].base, 11 + window, 17), cudaSuccess);
                EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
                EXPECT_EQ(leasedBytes(), 34u);
                return Code::ER_OK;
            }));
        Client client(std::move(delegate), [&](void* dst, M, const void* src, M, size_t bytes) {
            events.push_back("S" + std::to_string(banks.size() - 1));
            pending.emplace_back(dst, src, bytes); return true;
        }, context, 1, [&] {
            events.push_back("F");
            for (const auto& [dst, src, bytes] : pending)
                EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), cudaSuccess);
            pending.clear();
            return cudaStreamSynchronize(nullptr) == cudaSuccess;
        }, true);
        auto buffers = destinations({17, 17, 17});
        if (alias) for (auto& b : buffers) b.iovs[0].base = output.get();
        auto trace = std::make_shared<kv_cache_manager::TransferTraceInfo>(); trace->block_ids = {"100", "101", "102"};
        ASSERT_EQ(client.LoadKvCaches({"0", "1", "2"}, buffers, trace), Code::ER_OK);
        EXPECT_EQ(events, (std::vector<std::string>{"L0", "S0", "L1", "F", "S1", "L2", "F", "S2", "F"}));
        ASSERT_EQ(banks.size(), 3u);
        EXPECT_NE(banks[0], banks[1]); EXPECT_EQ(banks[0], banks[2]);
        std::vector<unsigned char> bytes(alias ? 17 : 51);
        ASSERT_EQ(cudaMemcpy(bytes.data(), output.get(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t j = 0; j < bytes.size(); ++j) EXPECT_EQ(bytes[j], alias ? 13 : 11 + j / 17);
        EXPECT_TRUE(pending.empty()); EXPECT_EQ(leasedBytes(), 0u);
    }
}
TEST_F(NativeSsdGpuReadWindowTest, WritePipelineGathersAheadButKeepsSdkSavesSerialAndBytesReady) {
    ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
    // Count-based windows: 64, 64, 1 objects; no oversized fixture allocation.
    constexpr size_t objects = 129, bytes_per_object = 17, bank_bytes = 64 * bytes_per_object;
    auto buffers = destinations(std::vector<size_t>(objects, bytes_per_object));
    Uris uris;
    auto trace = std::make_shared<kv_cache_manager::TransferTraceInfo>();
    for (size_t j = 0; j < objects; ++j) {
        uris.push_back(std::to_string(j)); trace->block_ids.push_back(std::to_string(j + 100));
        ASSERT_EQ(cudaMemset(buffers[j].iovs[0].base, 1 + j, bytes_per_object), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    std::vector<std::string> events;
    std::vector<std::tuple<void*, const void*, size_t>> pending;
    std::vector<void*> banks;
    size_t copied = 0;
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(3)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto t) {
            const size_t window = banks.size(), first = window * 64;
            events.push_back("W" + std::to_string(window));
            EXPECT_EQ(pending.size(), window == 0 ? 64u : window == 1 ? 1u : 0u);
            EXPECT_EQ(u, Uris(uris.begin() + first, uris.begin() + first + b.size()));
            EXPECT_TRUE(t && t->block_ids == Uris(trace->block_ids.begin() + first, trace->block_ids.begin() + first + b.size()));
            if (b.empty() || b[0].iovs.size() != 1) return std::make_pair(Code::ER_INVALID_PARAMS, Uris{});
            banks.push_back(b[0].iovs[0].base);
            for (size_t j = 0; j < b.size(); ++j) {
                std::vector<unsigned char> data(bytes_per_object);
                EXPECT_EQ(cudaMemcpy(data.data(), b[j].iovs[0].base, data.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(std::all_of(data.begin(), data.end(), [=](unsigned char v) { return v == 1 + first + j; }));
            }
            EXPECT_EQ(leasedBytes(), 2 * bank_bytes);
            Uris actual = u; for (auto& uri : actual) uri += "-saved";
            return std::make_pair(Code::ER_OK, actual);
        }));
    Client client(std::move(delegate), [&](void* dst, M, const void* src, M, size_t bytes) {
        if (copied % 64 == 0) events.push_back("G" + std::to_string(copied / 64));
        ++copied; pending.emplace_back(dst, src, bytes); return true;
    }, context, 1, [&] {
        events.push_back("F");
        for (const auto& [dst, src, bytes] : pending)
            EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), cudaSuccess);
        pending.clear(); return cudaStreamSynchronize(nullptr) == cudaSuccess;
    }, true);
    const auto result = client.SaveKvCaches(uris, buffers, trace);
    EXPECT_EQ(result.first, Code::ER_OK);
    Uris expected = uris; for (auto& uri : expected) uri += "-saved";
    EXPECT_EQ(result.second, expected);
    EXPECT_EQ(events, (std::vector<std::string>{"G0", "F", "G1", "W0", "F", "G2", "W1", "F", "W2"}));
    ASSERT_EQ(banks.size(), 3u); EXPECT_NE(banks[0], banks[1]); EXPECT_EQ(banks[0], banks[2]);
    EXPECT_TRUE(pending.empty()); EXPECT_EQ(leasedBytes(), 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, PipelineFailureMustDrainCopiesBeforeReturningSafeOrKeepBothBanks) {
    const auto safe = expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS);
    if (!kvcm::sdkCodeProvesBufferSafety(safe)) GTEST_SKIP() << "SDK lacks explicit buffer-safe error codes";
    for (bool write : {false, true}) for (int failure = 0; failure < 4; ++failure) {
        ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
        size_t copies = 0, fences = 0, calls = 0, pending = 0;
        const size_t objects = write ? 65 : 2, bank_bytes = write ? 64 * 17 : 17;
        Uris uris(objects, "object"); auto buffers = destinations(std::vector<size_t>(objects, 17));
        auto fail = [&]() -> Code {
            EXPECT_GT(pending, 0u);
            if (failure == 3) throw std::runtime_error("uncertain SDK call");
            return failure == 2 ? Code::ER_SDK_TIMEOUT : safe;
        };
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        if (write) EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(Invoke([&](auto, auto, auto) {
            ++calls; return std::make_pair(fail(), Uris{});
        }));
        else EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly(Invoke([&](auto, auto, auto) {
            return ++calls == 1 ? Code::ER_OK : fail();
        }));
        {
            Client client(std::move(delegate), [&](auto, auto, auto, auto, size_t) {
                ++copies; ++pending; return true; // No DMA in this fault-injection test.
            }, context, 1, [&] {
                ++fences; pending = 0;
                return failure != 1 || fences != (write ? 2u : 1u);
            }, true);
            if (failure == 3) {
                if (write) EXPECT_THROW(client.SaveKvCaches(uris, buffers), std::runtime_error);
                else EXPECT_THROW(client.LoadKvCaches(uris, buffers), std::runtime_error);
            } else {
                const auto result = write ? client.SaveKvCaches(uris, buffers).first : client.LoadKvCaches(uris, buffers);
                EXPECT_EQ(result, failure == 1 ? Code::ER_CUDAMEMCPY_ERROR : failure == 2 ? Code::ER_SDK_TIMEOUT : safe);
            }
            EXPECT_EQ(calls, write ? 1u : 2u); EXPECT_EQ(copies, write ? 65u : 1u);
            EXPECT_EQ(fences, write ? 2u : 1u); EXPECT_EQ(pending, 0u);
            EXPECT_EQ(leasedBytes(), failure ? 2 * bank_bytes : 0u);
            if (failure) {
                EXPECT_EQ(client.LoadKvCaches({"blocked"}, destinations({17})),
                          expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
            }
        }
        EXPECT_EQ(leasedBytes(), 0u); // Fake SDK has no DMA; its client is now destroyed.
    }
}
TEST_F(NativeSsdGpuReadWindowTest, PipelineDrainsItsPendingCopyWhenAnotherReaderBecomesUnknown) {
    ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
    struct State {
        std::promise<void> entered, release, done;
        std::shared_future<void> released = release.get_future().share(), finished = done.get_future().share();
        std::atomic<bool> signalled{false};
        std::atomic<size_t> fences{0};
        std::shared_ptr<void> output;
        void signal() { if (!signalled.exchange(true)) release.set_value(); }
    };
    auto state = std::make_shared<State>(); state->output = output;
    struct Guard { std::shared_ptr<State> state; ~Guard() { state->signal(); } } guard{state};
    auto entered = state->entered.get_future();
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(Uris{"b"}, _, _)).WillOnce(Invoke([state](auto, auto, auto) {
        state->entered.set_value(); state->released.wait(); return Code::ER_SDK_TIMEOUT;
    }));
    EXPECT_CALL(*delegate, LoadKvCaches(Uris{"a0"}, _, _)).WillOnce(Return(Code::ER_OK));
    auto client = std::make_shared<Client>(std::move(delegate), [state](auto, auto, auto, auto, size_t) {
        // a0's scatter is now pending. Arrange for b to quarantine before a1.
        state->signal(); return state->finished.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    }, context, 1, [state] { ++state->fences; return true; }, true);
    auto b_buffers = destinations({17});
    block_tree_cache_test::BoundedThread<Code> b([client, state, b_buffers] {
        const auto result = client->LoadKvCaches({"b"}, b_buffers);
        state->done.set_value(); return result;
    });
    const auto status = entered.wait_for(std::chrono::seconds(5));
    const auto result = client->LoadKvCaches({"a0", "a1"}, destinations({17, 17}));
    state->signal();
    ASSERT_EQ(b.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(b.get(), Code::ER_SDK_TIMEOUT);
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_EQ(result, expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(state->fences.load(), 1u); // Safe early return must drain a0, even though a1 never starts.
    EXPECT_EQ(leasedBytes(), 17u); // Only b's one-bank UNKNOWN, not a's double bank.
    client.reset(); EXPECT_EQ(leasedBytes(), 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, PipelineCopyFailuresAndFenceExceptionsKeepBothBanks) {
    for (bool write : {false, true}) for (int failure = 0; failure < 3; ++failure) {
        const bool fence_throws = failure == 2;
        ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
        const size_t count = write ? 65 : 2, bank_bytes = write ? 64 * 17 : 17;
        size_t copied = 0, fences = 0;
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        if (write) EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(fence_throws ? 1 : 0)
            .WillRepeatedly(Return(std::make_pair(Code::ER_OK, Uris{})));
        else EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(fence_throws ? 2 : 1)
            .WillRepeatedly(Return(Code::ER_OK));
        {
            Client client(std::move(delegate), [&](auto, auto, auto, auto, size_t) {
                if (++copied == (write ? 65u : 1u) && !fence_throws) {
                    if (!failure) return false;
                    throw std::runtime_error("copy enqueue");
                }
                return true; // Fake copy, so no DMA survives the test.
            }, context, 1, [&] {
                if (++fences == (write ? 2u : 1u) && fence_throws) throw std::runtime_error("copy fence");
                return true;
            }, true);
            auto buffers = destinations(std::vector<size_t>(count, 17));
            if (!failure) {
                const auto result = write ? client.SaveKvCaches(Uris(count, "one"), buffers).first :
                                            client.LoadKvCaches(Uris(count, "one"), buffers);
                EXPECT_EQ(result, Code::ER_CUDAMEMCPY_ERROR);
            } else if (write) {
                EXPECT_THROW(client.SaveKvCaches(Uris(count, "one"), buffers), std::runtime_error);
            } else {
                EXPECT_THROW(client.LoadKvCaches(Uris(count, "one"), buffers), std::runtime_error);
            }
            EXPECT_EQ(leasedBytes(), 2 * bank_bytes);
            EXPECT_EQ(fences, (write ? 1u : 0u) + (fence_throws ? 2u : 1u));
        }
        EXPECT_EQ(leasedBytes(), 0u);
    }
}
TEST_F(NativeSsdGpuReadWindowTest, PipelineFallsBackWithoutRejectingSerialGeometryOrArenaCapacity) {
    for (int reason = 0; reason < 6; ++reason) {
        ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
        auto buffers = destinations({17, 17});
        if (reason == 0) { // Only one bank fits; optional second-bank reservation must roll back.
            context->free_ranges = {{0, 17}};
        } else if (reason == 1) {
            for (auto& block : buffers) block.iovs[0].base = arena.get();
        } else if (reason == 2) context->registered = false;
        else if (reason == 3) context->registered_gpu_spans.clear();
        else if (reason == 4) buffers.resize(1);
        else context = std::make_shared<Client::GpuPackingContext>(arena); // Unknown device.
        size_t calls = 0, fences = 0;
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(buffers.size()).WillRepeatedly(Invoke([&](auto, auto, auto) {
            EXPECT_EQ(fences, calls); ++calls; return Code::ER_OK;
        }));
        Client client(std::move(delegate), [](auto, auto, auto, auto, size_t) { return true; }, context, 1,
                      [&] { ++fences; return true; }, true);
        EXPECT_EQ(client.LoadKvCaches(Uris(buffers.size(), "one"), buffers), Code::ER_OK);
        EXPECT_EQ(calls, buffers.size()); EXPECT_EQ(fences, calls);
        EXPECT_EQ(context->active_reservations, 0u);
        if (reason == 0) {
            std::lock_guard<std::mutex> lock(context->mutex);
            EXPECT_EQ(context->free_ranges, (std::vector<std::pair<size_t, size_t>>{{0, 17}}));
        }
    }
}

TEST_F(NativeSsdGpuReadWindowTest, PipelineProcessBudgetFallsBackToOneBankWithoutLeakingTheFailedCharge) {
    ASSERT_NO_FATAL_FAILURE(registerPipelineSpans());
    std::vector<std::shared_ptr<Client>> held;
    // Independent real arenas; eight no-DMA mock UNKNOWNs leave exactly 17 B
    // below the existing 512 MiB process cap. Never reset the global counter.
    for (size_t index = 0; index < 8; ++index) {
        void* memory = nullptr;
        ASSERT_EQ(cudaMalloc(&memory, Client::max_operation_bytes), cudaSuccess);
        auto storage = std::shared_ptr<void>(memory, [](void* p) { cudaFree(p); });
        auto holder_context = std::make_shared<Client::GpuPackingContext>(storage);
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Return(Code::ER_SDK_TIMEOUT));
        auto client = std::make_shared<Client>(std::move(delegate),
            [](auto, auto, auto, auto, size_t) { return true; }, holder_context);
        Buffers buffers(1);
        buffers[0].iovs = {{M::GPU, memory, Client::max_operation_bytes - (index == 7 ? 17 : 0), false}};
        ASSERT_EQ(client->LoadKvCaches({"held"}, buffers), Code::ER_SDK_TIMEOUT);
        held.push_back(std::move(client));
    }
    size_t calls = 0, fences = 0;
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(4).WillRepeatedly(Invoke([&](auto, auto, auto) {
        EXPECT_EQ(fences, calls == 3 ? 2u : calls);
        EXPECT_EQ(leasedBytes(), calls < 2 ? 17u : 34u);
        ++calls; return Code::ER_OK;
    }));
    Client client(std::move(delegate), [](auto, auto, auto, auto, size_t) { return true; }, context, 1,
                  [&] { ++fences; return true; }, true);
    EXPECT_EQ(client.LoadKvCaches({"a", "b"}, destinations({17, 17})), Code::ER_OK);
    EXPECT_EQ(calls, 2u); EXPECT_EQ(fences, 2u); EXPECT_EQ(leasedBytes(), 0u);
    held.clear(); // The mock SDK has no DMA and all calls have returned.
    // Normal admission must again work after all process charges are returned.
    EXPECT_EQ(client.LoadKvCaches({"a", "b"}, destinations({17, 17})), Code::ER_OK);
    EXPECT_EQ(calls, 4u); EXPECT_EQ(fences, 4u); EXPECT_EQ(leasedBytes(), 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, PipelineRequiresAFenceEvenBeforeAnyOperationStarts) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_THROW(Client(std::move(delegate), copy, context, 1, {}, true), std::invalid_argument);
    EXPECT_EQ(context->active_reservations, 0u);
}

TEST_F(NativeSsdGpuReadWindowTest, ExplicitSafeSdkFailureReleasesArenaAndPermitsRetry) {
    checkSafeSdkRetry<Code>();
}
TEST_F(NativeSsdGpuReadWindowTest, PreflightRejectionsDoNotTouchPayloadOrBecomeUnknown) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    int copies = 0;
    Client client(std::move(delegate), [&](auto, auto, auto, auto, size_t) { ++copies; return false; }, context);
    auto buffers = destinations({17});
    auto trace = std::make_shared<kv_cache_manager::TransferTraceInfo>();
    trace->block_ids = {"one", "extra"};
    EXPECT_EQ(client.LoadKvCaches({"one"}, buffers, trace), expectedPreflightError(Code::ER_INVALID_PARAMS));
    EXPECT_EQ(client.SaveKvCaches({"one"}, buffers, trace).first, expectedPreflightError(Code::ER_INVALID_PARAMS));
    std::vector<std::pair<size_t, size_t>> free_ranges;
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        free_ranges.swap(context->free_ranges); // Simulate all arena ranges held by another operation.
    }
    const auto code = client.LoadKvCaches({"one"}, buffers);
    EXPECT_EQ(code, expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value)
        EXPECT_EQ(kvcm::ClientWrapper::sdkResultCompletion(code), kvcm::ClientWrapper::TransferCompletion::BUFFER_SAFE);
    EXPECT_EQ(client.SaveKvCaches({"one"}, buffers).first,
              expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        free_ranges.swap(context->free_ranges);
    }
    EXPECT_EQ(copies, 0);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, UnknownBatchRetainsTheWholeWindowInEitherUriOrder) {
    checkMixedPeerSafety<Code>();
}
TEST_F(NativeSsdGpuReadWindowTest, BatchReadPreservesTraceBytesAndFencesBeforeReusingTheWindow) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    size_t calls = 0, fences = 0, copies = 0;
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto trace) {
            EXPECT_EQ(fences, calls);
            EXPECT_EQ(u.size(), 4u);
            EXPECT_EQ(b.size(), 4u);
            int actual = -1;
            EXPECT_EQ(cudaGetDevice(&actual), cudaSuccess);
            EXPECT_EQ(actual, device);
            if (u.size() != 4 || b.size() != 4 || !trace || trace->block_ids.size() != 4)
                return Code::ER_INVALID_PARAMS;
            for (size_t i = 0; i < 4; ++i) {
                const size_t id = calls * 4 + i;
                EXPECT_EQ(u[i], std::to_string(id));
                EXPECT_EQ(trace->block_ids[i], std::to_string(100 + id));
                EXPECT_EQ(b[i].iovs.size(), 1u);
                if (b[i].iovs.size() != 1) return Code::ER_INVALID_PARAMS;
                EXPECT_EQ(b[i].iovs[0].size, 36u * 1024);
                EXPECT_EQ(cudaMemset(b[i].iovs[0].base, 11 + id, b[i].iovs[0].size), cudaSuccess);
            }
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess); // Fake SDK success means data ready.
            ++calls;
            return Code::ER_OK;
        }));
    Client client(std::move(delegate), [&](void* d, M dt, const void* src, M st, size_t n) {
        ++copies;
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, src, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    }, context, 4, [&] { ++fences; return cudaStreamSynchronize(nullptr) == cudaSuccess; });
    Uris uris;
    auto trace = std::make_shared<kv_cache_manager::TransferTraceInfo>();
    for (int i = 0; i < 8; ++i) {
        uris.push_back(std::to_string(i));
        trace->block_ids.push_back(std::to_string(100 + i));
    }
    ASSERT_EQ(client.LoadKvCaches(uris, destinations(std::vector<size_t>(8, 36 * 1024)), trace), Code::ER_OK);
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(fences, 2u);
    EXPECT_EQ(copies, 8u);
    std::vector<unsigned char> bytes(8 * 36 * 1024);
    ASSERT_EQ(cudaMemcpy(bytes.data(), output.get(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_TRUE(std::all_of(bytes.begin() + i * 36 * 1024, bytes.begin() + (i + 1) * 36 * 1024,
                                [i](unsigned char v) { return v == 11 + i; }));
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, BatchExceptionKeepsConfirmedPrefixAndQuarantinesTheRemainingWindow) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    int calls = 0;
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([&](const Uris&, const Buffers& b, auto) -> Code {
            if (++calls == 2) throw std::runtime_error("uncertain batch DMA");
            for (const auto& block : b)
                EXPECT_EQ(cudaMemset(block.iovs[0].base, 0x55, block.iovs[0].size), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            return Code::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, 4);
    size_t confirmed = 0, unknown = 0, observations = 0;
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool uncertain) {
        EXPECT_TRUE(read); confirmed += bytes; unknown += uncertain; ++observations;
    });
    auto b = destinations(std::vector<size_t>(8, 4096));
    Uris uris{"0","1","2","3","4","5","6","7"};
    EXPECT_THROW(client.LoadKvCaches(uris, b), std::runtime_error);
    EXPECT_EQ(confirmed, 4u * 4096);
    EXPECT_EQ(unknown, 1u);
    EXPECT_EQ(observations, 2u);
    EXPECT_EQ(context->active_reservations, 1u);
    EXPECT_EQ(client.LoadKvCaches(uris, b), expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    std::vector<unsigned char> actual(8 * 4096);
    ASSERT_EQ(cudaMemcpy(actual.data(), output.get(), actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + 4 * 4096, [](auto v) { return v == 0x55; }));
    EXPECT_TRUE(std::all_of(actual.begin() + 4 * 4096, actual.end(), [](auto v) { return v == 0; }));
}
TEST_F(NativeSsdGpuReadWindowTest, PayloadBudgetSplitsBatchBeforeTheObjectLimit) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    size_t calls = 0;
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto) {
            EXPECT_EQ(u.size(), calls == 0 ? 2u : 1u);
            {
                std::lock_guard<std::mutex> lock(context->mutex);
                size_t free = 0;
                for (const auto& range : context->free_ranges) free += range.second;
                EXPECT_EQ(Client::max_operation_bytes - free, 200u * 1024);
            }
            for (size_t i = 0; i < b.size(); ++i) {
                const int id = std::stoi(u[i]);
                EXPECT_EQ(b[i].iovs.size(), 1u);
                EXPECT_EQ(cudaMemset(b[i].iovs[0].base, 11 + id, 100 * 1024), cudaSuccess);
            }
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            ++calls;
            return Code::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, 4);
    ASSERT_EQ(client.LoadKvCaches({"0","1","2"}, destinations({100*1024,100*1024,100*1024})), Code::ER_OK);
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(context->active_reservations, 0u);
    std::vector<unsigned char> bytes(300 * 1024);
    ASSERT_EQ(cudaMemcpy(bytes.data(), output.get(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int id = 0; id < 3; ++id)
        EXPECT_TRUE(std::all_of(bytes.begin() + id * 100 * 1024, bytes.begin() + (id + 1) * 100 * 1024,
                                [id](auto v) { return v == 11 + id; }));
}
TEST_F(NativeSsdGpuReadWindowTest, OversizedStateObjectIsReadAlone) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([](const Uris& u, const Buffers&, auto) {
            EXPECT_EQ(u.size(), 1u);
            return Code::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, 4);
    EXPECT_EQ(client.LoadKvCaches({"state","small"}, destinations({354816,4096})), Code::ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, CopyAndFenceFailuresDrainThenKeepTheLease) {
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode);
        auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
        EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
        size_t copies = 0, fences = 0;
        {
            Client client(std::move(delegate), [&](auto, auto, auto, auto, size_t) {
                ++copies;
                if (copies == 2 && mode == 1) throw std::runtime_error("copy-sentinel");
                return !(copies == 2 && mode == 0);
            }, context, 4, [&] {
                ++fences;
                if (mode == 3) throw std::runtime_error("fence-sentinel");
                return mode != 2;
            });
            if (mode == 1 || mode == 3)
                EXPECT_THROW(client.SaveKvCaches({"0","1"}, destinations({17,33})), std::runtime_error);
            else
                EXPECT_EQ(client.SaveKvCaches({"0","1"}, destinations({17,33})).first, Code::ER_CUDAMEMCPY_ERROR);
            EXPECT_EQ(copies, 2u);
            EXPECT_EQ(fences, mode == 3 ? 2u : 1u);
            EXPECT_EQ(context->active_reservations, 1u);
            EXPECT_EQ(client.SaveKvCaches({"retry"}, destinations({17})).first,
                      expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
        } // Fake delegate has no outstanding I/O; destruction returns this test lease.
        EXPECT_EQ(context->active_reservations, 0u);
    }
}
TEST_F(NativeSsdGpuReadWindowTest, ConcurrentUnknownReadsKeepBothLeasesAndRejectNewOperations) {
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        size_t entered{0};
        std::vector<uintptr_t> staging;
        std::promise<void> release;
        std::shared_future<void> released = release.get_future().share();
        std::shared_ptr<void> payload;
    };
    auto state = std::make_shared<State>();
    state->payload = output;
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([state](const Uris&, const Buffers& b, auto) {
            { std::lock_guard<std::mutex> lock(state->mu); ++state->entered;
              state->staging.push_back(reinterpret_cast<uintptr_t>(b[0].iovs[0].base)); }
            state->cv.notify_all();
            state->released.wait(); // No DMA is submitted by this fake.
            return Code::ER_SDK_TIMEOUT;
        }));
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    auto client = std::make_shared<Client>(std::move(delegate), copy, context, 4);
    auto first = destinations({4096}), second = destinations({4096});
    second[0].iovs[0].base = static_cast<unsigned char*>(output.get()) + 4096;
    block_tree_cache_test::BoundedThread<Code> a([client, state, first, device] {
        if (cudaSetDevice(device) != cudaSuccess) return Code::ER_CUDAMEMCPY_ERROR;
        return client->LoadKvCaches({"a"}, first);
    });
    block_tree_cache_test::BoundedThread<Code> b([client, state, second, device] {
        if (cudaSetDevice(device) != cudaSuccess) return Code::ER_CUDAMEMCPY_ERROR;
        return client->LoadKvCaches({"b"}, second);
    });
    bool overlapping = false;
    { std::unique_lock<std::mutex> lock(state->mu);
      overlapping = state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->entered == 2; }); }
    state->release.set_value(); // Release before assertions; detached failures own all captured state.
    ASSERT_EQ(a.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(b.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(a.get(), Code::ER_SDK_TIMEOUT);
    EXPECT_EQ(b.get(), Code::ER_SDK_TIMEOUT);
    ASSERT_TRUE(overlapping);
    ASSERT_EQ(state->staging.size(), 2u);
    EXPECT_TRUE(state->staging[0] + 4096 <= state->staging[1] || state->staging[1] + 4096 <= state->staging[0]);
    EXPECT_EQ(context->active_reservations, 2u);
    EXPECT_EQ(client->LoadKvCaches({"after"}, first), expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(context->active_reservations, 2u);
    client.reset(); // All fake SDK calls have joined; both test leases can now be reclaimed.
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, SaveWaitsForAConcurrentReadToLeaveTheWrapper) {
    struct State {
        std::promise<void> entered, release, save_started;
        std::shared_future<void> released = release.get_future().share();
        std::atomic<bool> read_released{false}, save_entered{false};
        std::shared_ptr<void> payload;
    };
    auto state = std::make_shared<State>();
    state->payload = output;
    auto ready = state->entered.get_future(), attempted = state->save_started.get_future();
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Invoke([state](auto, auto, auto) {
        state->entered.set_value(); state->released.wait(); state->read_released.store(true); return Code::ER_OK;
    }));
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(Invoke([state](const Uris& u, auto, auto) {
        EXPECT_TRUE(state->read_released.load()); state->save_entered.store(true); return std::make_pair(Code::ER_OK, u);
    }));
    auto client = std::make_shared<Client>(std::move(delegate),
        [](auto, auto, auto, auto, size_t) { return true; }, context, 4);
    auto buffers = destinations({17});
    block_tree_cache_test::BoundedThread<Code> read([client, state, buffers] { return client->LoadKvCaches({"read"}, buffers); });
    const auto entered_status = ready.wait_for(std::chrono::seconds(5));
    block_tree_cache_test::BoundedThread<Code> save([client, state, buffers] {
        state->save_started.set_value(); return client->SaveKvCaches({"save"}, buffers).first;
    });
    const auto attempted_status = attempted.wait_for(std::chrono::seconds(5));
    const auto held = save.waitFor(std::chrono::milliseconds(20));
    const bool entered_early = state->save_entered.load();
    state->release.set_value();
    ASSERT_EQ(read.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(save.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(read.get(), Code::ER_OK);
    EXPECT_EQ(save.get(), Code::ER_OK);
    EXPECT_EQ(entered_status, std::future_status::ready);
    EXPECT_EQ(attempted_status, std::future_status::ready);
    EXPECT_EQ(held, std::future_status::timeout);
    EXPECT_FALSE(entered_early);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, ArenaPressureRejectsOtherReadersSafelyWhileAnOperationOwnsItsLease) {
    struct State {
        std::promise<void> entered, release;
        std::shared_future<void> released = release.get_future().share();
        std::atomic<bool> signalled{false};
        std::shared_ptr<void> payload;
        void signal() { if (!signalled.exchange(true)) release.set_value(); }
    };
    auto state = std::make_shared<State>();
    constexpr size_t bytes = 40ULL * 1024 * 1024;
    void* memory = nullptr;
    ASSERT_EQ(cudaMalloc(&memory, bytes), cudaSuccess);
    state->payload = std::shared_ptr<void>(memory, [](void* p) { cudaFree(p); });
    struct Guard { std::shared_ptr<State> state; ~Guard() { state->signal(); } } guard{state};
    auto entered = state->entered.get_future();
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Invoke([state](auto, auto, auto) {
        state->entered.set_value(); state->released.wait(); return Code::ER_OK;
    }));
    auto client = std::make_shared<Client>(std::move(delegate),
        [](auto, auto, auto, auto, size_t) { return true; }, context, 4);
    Buffers buffers(1);
    buffers[0].iovs = {{M::GPU, memory, bytes, false}};
    block_tree_cache_test::BoundedThread<Code> held([client, state, buffers] {
        return client->LoadKvCaches({"held"}, buffers);
    });
    const auto entered_status = entered.wait_for(std::chrono::seconds(5));
    std::vector<std::unique_ptr<block_tree_cache_test::BoundedThread<Code>>> pressure;
    for (int i = 0; i < 3; ++i)
        pressure.push_back(std::make_unique<block_tree_cache_test::BoundedThread<Code>>([client, state, buffers] {
            return client->LoadKvCaches({"pressure"}, buffers);
        }));
    std::vector<std::future_status> before_release;
    for (auto& task : pressure) before_release.push_back(task->waitFor(std::chrono::seconds(5)));
    state->signal(); // Drain even if a regression incorrectly sent a pressure request to the SDK.
    ASSERT_EQ(held.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(held.get(), Code::ER_OK);
    EXPECT_EQ(entered_status, std::future_status::ready);
    for (size_t i = 0; i < pressure.size(); ++i) {
        ASSERT_EQ(pressure[i]->waitFor(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(before_release[i], std::future_status::ready);
        EXPECT_EQ(pressure[i]->get(), expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    }
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, AnotherUnknownBetweenWindowsStopsOnlyFutureIoAndRetainsItsOwnLease) {
    struct State {
        std::promise<void> entered_b, release_b, done_b;
        std::shared_future<void> released = release_b.get_future().share(), done = done_b.get_future().share();
        std::atomic<bool> signalled{false};
        std::shared_ptr<void> payload;
        void signal() { if (!signalled.exchange(true)) release_b.set_value(); }
    };
    auto state = std::make_shared<State>(); state->payload = output;
    struct Guard { std::shared_ptr<State> state; ~Guard() { state->signal(); } } guard{state};
    auto entered = state->entered_b.get_future();
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([state](const Uris& u, const Buffers&, auto) {
            if (u[0] == "b") {
                state->entered_b.set_value(); state->released.wait(); return Code::ER_SDK_TIMEOUT;
            }
            EXPECT_EQ(u, (Uris{"a0"})); // a1 must never reach the SDK.
            return Code::ER_OK;
        }));
    auto client = std::make_shared<Client>(std::move(delegate),
        [](auto, auto, auto, auto, size_t) { return true; }, context, 1, [state] {
            state->signal(); return state->done.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        });
    auto b_buffers = destinations({33}), a_buffers = destinations({17, 17});
    block_tree_cache_test::BoundedThread<Code> b([client, state, b_buffers] {
        const auto result = client->LoadKvCaches({"b"}, b_buffers);
        state->done_b.set_value(); return result;
    });
    const auto entered_status = entered.wait_for(std::chrono::seconds(5));
    block_tree_cache_test::BoundedThread<Code> a([client, state, a_buffers] {
        return client->LoadKvCaches({"a0", "a1"}, a_buffers);
    });
    const auto completed_a = a.waitFor(std::chrono::seconds(5));
    state->signal();
    ASSERT_EQ(a.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(b.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(entered_status, std::future_status::ready);
    EXPECT_EQ(completed_a, std::future_status::ready);
    EXPECT_EQ(a.get(), expectedPreflightError(Code::ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(b.get(), Code::ER_SDK_TIMEOUT);
    EXPECT_EQ(context->active_reservations, 1u); // Only b's unresolved operation remains, not a's quiescent window.
    client.reset(); // No fake SDK DMA exists; all test workers have joined.
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, SaveObserverKeepsConfirmedPrefixBeforeSdkFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    Uris uris;
    std::vector<size_t> sizes;
    size_t first_window_bytes = 0;
    for (size_t i = 0; i < 129; ++i) {
        uris.push_back(std::to_string(i));
        sizes.push_back(17 + i % 7);
        if (i < 64)
            first_window_bytes += sizes.back();
    }
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([](const Uris& batch, const Buffers& buffers, auto) {
            EXPECT_EQ(batch.size(), 64u);
            EXPECT_EQ(buffers.size(), batch.size());
            EXPECT_TRUE(batch.front() == "0" || batch.front() == "64");
            return std::make_pair(batch.front() == "0" ? ER_OK : ER_SDK_TIMEOUT, batch);
        }));
    Client client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches(uris, destinations(sizes)).first, ER_SDK_TIMEOUT);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], std::make_tuple(false, first_window_bytes, false));
    EXPECT_EQ(seen[1], std::make_tuple(false, size_t(0), true));
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, PreSdkGeometryAndGatherRejectionsDoNotReportUnknownIo) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    Client client(std::move(delegate), [](void*, M, const void*, M, size_t) { return false; }, context);
    int    observed = 0;
    client.setSdkTransferObserver([&](bool, size_t, bool) { ++observed; });
    EXPECT_EQ(client.SaveKvCaches({"invalid"}, destinations({0})).first, expectedPreflightError(ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(context->active_reservations, 0u);
    EXPECT_EQ(client.SaveKvCaches({"gather-fails"}, destinations({17})).first, ER_CUDAMEMCPY_ERROR);
    EXPECT_EQ(observed, 0);
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, SdkReadSuccessRemainsConfirmedAfterScatterFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK));
    Client client(std::move(delegate), [](void*, M, const void*, M, size_t) { return false; }, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.LoadKvCaches({"scatter-fails"}, destinations({17})), ER_CUDAMEMCPY_ERROR);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0], std::make_tuple(true, size_t(17), false));
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, SdkWriteSuccessRemainsConfirmedAfterActualUriShapeFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, Uris{"wrong", "count"})));
    Client                                      client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches({"one"}, destinations({17})).first, ER_INVALID_PARAMS);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0], std::make_tuple(false, size_t(17), false));
}
TEST_F(NativeSsdGpuReadWindowTest, ObserverExceptionsPreserveSdkSuccessAndFailureResults) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK)).WillOnce(Return(ER_SDK_TIMEOUT));
    Client                                      client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool unknown) {
        seen.emplace_back(read, bytes, unknown);
        throw std::logic_error("observer-failure");
    });
    EXPECT_EQ(client.LoadKvCaches({"success"}, destinations({17})), ER_OK);
    EXPECT_EQ(client.LoadKvCaches({"failed"}, destinations({17})), ER_SDK_TIMEOUT);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], std::make_tuple(true, size_t(17), false));
    EXPECT_EQ(seen[1], std::make_tuple(true, size_t(0), true));
    EXPECT_EQ(client.LoadKvCaches({"quarantined"}, destinations({17})), expectedPreflightError(ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(seen.size(), 2u);
}
TEST_F(NativeSsdGpuReadWindowTest, ObserverExceptionCannotReplaceTheOriginalSdkException) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(::testing::Throw(std::runtime_error("sdk-sentinel")));
    Client client(std::move(delegate), copy, context);
    int    observed = 0;
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool unknown) {
        ++observed;
        EXPECT_FALSE(read);
        EXPECT_EQ(bytes, 0u);
        EXPECT_TRUE(unknown);
        throw std::logic_error("observer-failure");
    });
    try {
        (void)client.SaveKvCaches({"throw"}, destinations({17}));
        FAIL() << "SDK exception must propagate";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "sdk-sentinel");
    }
    EXPECT_EQ(observed, 1);
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, WriteBatchPreservesPositionsTracesAndGpuPayloadAcrossWindows) {
    using namespace kv_cache_manager;
    Uris uris;
    std::vector<size_t> sizes;
    std::vector<unsigned char> expected;
    auto trace = std::make_shared<TransferTraceInfo>();
    for (size_t i = 0; i < 130; ++i) {
        uris.push_back("uri" + std::to_string(i));
        sizes.push_back(17 + i % 7);
        trace->block_ids.push_back(std::to_string(1000 + i));
        for (size_t j = 0; j < sizes.back(); ++j)
            expected.push_back(static_cast<unsigned char>((i * 31 + j * 17) % 256));
    }
    ASSERT_EQ(cudaMemcpy(output.get(), expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
    void* remote = nullptr;
    ASSERT_EQ(cudaMalloc(&remote, expected.size()), cudaSuccess);
    auto storage = std::shared_ptr<void>(remote, [](void* p) { cudaFree(p); });
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    size_t next = 0, remote_offset = 0, calls = 0, fences = 0;
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(3)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, const auto& slice) {
            EXPECT_EQ(fences, calls + 1); // The whole gather window must be ready before SDK submission.
            EXPECT_EQ(u.size(), std::min(size_t(64), uris.size() - next));
            EXPECT_EQ(b.size(), u.size());
            EXPECT_NE(slice, nullptr);
            EXPECT_NE(slice, trace);
            if (!slice || b.size() != u.size() || slice->block_ids.size() != u.size())
                return std::make_pair(ER_INVALID_PARAMS, Uris{});
            uintptr_t previous_end = 0;
            Uris actual;
            for (size_t i = 0; i < u.size(); ++i, ++next) {
                EXPECT_EQ(u[i], uris[next]);
                EXPECT_EQ(slice->block_ids[i], std::to_string(1000 + next));
                EXPECT_EQ(b[i].iovs.size(), 1u);
                if (b[i].iovs.size() != 1 || next >= sizes.size())
                    return std::make_pair(ER_INVALID_PARAMS, Uris{});
                const auto& packed = b[i].iovs[0];
                EXPECT_EQ(packed.type, M::GPU);
                EXPECT_EQ(packed.size, sizes[next]);
                if (packed.size != sizes[next])
                    return std::make_pair(ER_INVALID_PARAMS, Uris{});
                const auto address = reinterpret_cast<uintptr_t>(packed.base);
                if (i) {
                    EXPECT_GE(address, previous_end);
                }
                previous_end = address + packed.size;
                EXPECT_EQ(cudaMemcpy(static_cast<unsigned char*>(storage.get()) + remote_offset,
                                     packed.base, packed.size, cudaMemcpyDeviceToDevice), cudaSuccess);
                remote_offset += packed.size;
                actual.push_back("actual" + std::to_string(next));
            }
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess); // Fake SDK copy has completed.
            ++calls;
            return std::make_pair(ER_OK, actual);
        }));
    Client client(std::move(delegate), [](void* d, M dt, const void* src, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, src, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    }, context, 4, [&] { ++fences; return cudaStreamSynchronize(nullptr) == cudaSuccess; });
    const auto result = client.SaveKvCaches(uris, destinations(sizes), trace);
    ASSERT_EQ(result.first, ER_OK);
    ASSERT_EQ(result.second.size(), uris.size());
    for (size_t i = 0; i < uris.size(); ++i)
        EXPECT_EQ(result.second[i], "actual" + std::to_string(i));
    EXPECT_EQ(calls, 3u);
    EXPECT_EQ(fences, 3u);
    EXPECT_EQ(next, uris.size());
    EXPECT_EQ(context->active_reservations, 0u);
    std::vector<unsigned char> observed(expected.size());
    ASSERT_EQ(cudaMemcpy(observed.data(), storage.get(), observed.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(observed, expected);
}
TEST_F(NativeSsdGpuReadWindowTest, WritePayloadBudgetSplitsBeforeObjectLimit) {
    using namespace kv_cache_manager;
    constexpr size_t bytes = 700 * 1024;
    Buffers source(3);
    for (auto& block : source)
        block.iovs = {{M::GPU, output.get(), bytes, false}};
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    size_t calls = 0;
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto) {
            EXPECT_EQ(u.size(), calls++ == 0 ? 2u : 1u);
            EXPECT_EQ(u.size(), b.size());
            size_t total = 0;
            for (const auto& block : b)
                total += block.iovs.front().size;
            EXPECT_LE(total, Client::max_write_window_bytes);
            return std::make_pair(ER_OK, u);
        }));
    Client client(std::move(delegate), copy, context);
    EXPECT_EQ(client.SaveKvCaches({"a", "b", "c"}, source).first, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, WriteLeaseFailureSubmitsNothingAndDoesNotLeakProcessCharge) {
    using namespace kv_cache_manager;
    context->free_ranges = {{0, 16}};
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    Client failed(std::move(delegate), copy, context);
    Buffers large(1);
    large[0].iovs = {{M::GPU, arena.get(), Client::max_operation_bytes, false}};
    for (size_t i = 0; i < 9; ++i)
        EXPECT_EQ(failed.SaveKvCaches({"blocked"}, large).first, expectedPreflightError(ER_INVALID_LOCAL_BUFFERS));
    EXPECT_EQ(context->active_reservations, 0u);
    context->free_ranges = {{0, Client::max_operation_bytes}};
    auto retry = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*retry, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, Uris{})));
    Client accepted(std::move(retry), copy, context);
    EXPECT_EQ(accepted.SaveKvCaches({"retry"}, destinations({17})).first, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, BatchWriteBadUriShapeKeepsConfirmedBytesAndQuarantinesWindow) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .WillOnce(Return(std::make_pair(ER_OK, Uris{"only-one"})));
    Client client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches({"one", "two"}, destinations({17, 33})).first, ER_INVALID_PARAMS);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen.front(), std::make_tuple(false, size_t(50), false));
    EXPECT_EQ(context->active_reservations, 1u);
    EXPECT_EQ(client.SaveKvCaches({"retry"}, destinations({17})).first, expectedPreflightError(ER_INVALID_LOCAL_BUFFERS));
}
TEST_F(NativeSsdGpuReadWindowTest, GatherExceptionBeforeBatchSubmissionRetainsWholeWindow) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    size_t copies = 0;
    Client client(std::move(delegate), [&](void* d, M dt, const void* s, M st, size_t n) {
        if (++copies == 2)
            throw std::runtime_error("uncertain-copy");
        return copy(d, dt, s, st, n);
    }, context);
    size_t observed = 0;
    client.setSdkTransferObserver([&](bool, size_t, bool) { ++observed; });
    EXPECT_THROW(client.SaveKvCaches({"one", "two"}, destinations({17, 33})), std::runtime_error);
    EXPECT_EQ(observed, 0u);
    EXPECT_EQ(context->active_reservations, 1u);
}
// GPU pages + recording SDK. CPU copies below are assertions only; the
// implementation must pass native GPU IOVs through the Prefill transfer path.
class EosGpuCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool, int>> {
protected:
    autil::EnvGuard enable{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1"},
        pending{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "1"},
        // This tiny fixture intentionally holds most of its synthetic pool.
        // Production's 1/8 default is covered separately; allow these positive
        // lifetime/projection cases to reach transfer rather than admission.
        pending_bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    // Failure cases deliberately freeze a large fraction of these tiny pools.
    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    struct Scenario {
        int                                 cp, shards, count = 3;
        BackendEnvironment                  env;
        BlockIndicesType                    full, state;
        BatchKVCacheResource                resource;
        std::shared_ptr<MockClientWrapper>  client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<MockQuiescentKVCMBackend> backend;
        std::map<std::string, std::string>  specs;
        std::atomic<int>                    starts{0}, saves{0}, finishes{0}, aborts{0};
        std::atomic<size_t>                 requested_keys{0};
        bool                                block_start = false, fail_save = false, lose_commit = false;
        std::promise<void>                  entered, released;
        std::shared_future<void>            release = released.get_future().share();
        std::future<void>                   started = entered.get_future();
        explicit Scenario(
            int target_cp, bool sharded, bool bounded = true, bool native_ssd_gdr = false, int decode_dp = 1):
            cp(target_cp), shards(sharded ? cp : 1) {
            // The target CP comes only from ParallelismConfig, not a CP-specific environment flag.
            auto f                     = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto s                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
            s.enable_prefix_reuse      = true;
            s.cp_mapping               = sharded ? CpBlockMappingMode::COMPACT_LAST_RANK : CpBlockMappingMode::NONE;
            s.cp_slice                 = sharded ? CpBlockSliceMode::EQUAL_BYTES : CpBlockSliceMode::NONE;
            s.sliding_window_size      = bounded ? 256 * shards : 0;
            env.cache_config.dtype     = DataType::TYPE_FP16;
            env.cache_config.layer_num = 2;
            env.cache_config.seq_size_per_block = 128;
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
                        requested_keys = keys.size();
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
                                one.push_back({specs.at("full#cp" + std::to_string(peer)), "pace://F:" + suffix});
                                if (keys.size() - key
                                    <= env.cache_config.topology().group("state").reuseBlockCount(keys.size()))
                                    one.push_back({specs.at("state#cp" + std::to_string(peer)), "pace://S:" + suffix});
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
                        const auto payload = uris[i].substr(7);
                        const int key = std::stoi(payload.substr(2)), peer = std::stoi(payload.substr(4));
                        // The first registered PACE client owns every GPU span.
                        EXPECT_EQ(tag, "full#cp0");
                        const int expected = payload[0] == 'F' ? key * shards + (shards > 1 ? peer : 0) + 1 : 100 + key;
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
            k.kvcm_instance_group   = "decode_remote_eos_gpu_test";
            k.kvcm_instance_id_salt = "eos_native_gpu";
            k.kvcm_model_sdk_config = "[{\"type\":\"pace\"}]";
            if (native_ssd_gdr) {
                k.kvcm_model_sdk_config  = "[{\"type\":\"pace_ssd\"}]";
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
            backend       = std::make_unique<MockQuiescentKVCMBackend>(
                env.cache_config, k, rt, parallel, SpeculativeExecutionConfig{}, nullptr, true, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Scenario() {
            unblock();
            if (backend) {
                backend->proveMockQuiescence();
                backend->shutdown();
            }
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

TEST_P(EosGpuCompletionTest, DisabledAtInitializationDoesNotPublishPinOrChangeNativePages) {
    for (const bool unset : {false, true}) {
        SCOPED_TRACE(unset);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        if (unset) ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        std::map<std::string, std::vector<size_t>> refs;
        for (const auto& tag : {std::string("full"), std::string("state")}) {
            const auto& blocks = tag == "full" ? s.full : s.state;
            for (const auto block : blocks) refs[tag].push_back(s.env.pools_by_tag.at(tag)->refCount(block));
        }
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), false));
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.starts, 0);
        EXPECT_EQ(s.saves, 0);
        EXPECT_EQ(s.finishes, 0);
        EXPECT_EQ(s.aborts, 0);
        EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
        EXPECT_FALSE(s.resource.decode_endpoint);
        EXPECT_TRUE(s.resource.decode_round_endpoints.empty());
        for (const auto& tag : {std::string("full"), std::string("state")}) {
            const auto& blocks = tag == "full" ? s.full : s.state;
            const auto pool = s.env.pools_by_tag.at(tag);
            for (size_t i = 0; i < blocks.size(); ++i) {
                EXPECT_EQ(pool->refCount(blocks[i]), refs[tag][i]);
                EXPECT_FALSE(pool->isFrozen(blocks[i]));
                for (const auto& part : environmentBuffers(s.env, tag == "full" ? 0 : 1, tag, blocks[i])) {
                    std::vector<unsigned char> bytes(part.size_bytes);
                    ASSERT_EQ(cudaMemcpy(bytes.data(), part.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                    const auto expected = tag == "full" ? i + 1 : 100 + i;
                    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto value) { return value == expected; }));
                }
            }
        }
        s.dropRequestRefs();
        for (const auto block : s.full) EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
        for (const auto block : s.state) EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
        });
    }
}

TEST_P(EosGpuCompletionTest, RuntimeDisableSkipsEosBeforePinsAndAllowsLaterPublication) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    const auto pool = s.env.pools_by_tag.at("full");
    const auto refs = pool->refCount(s.full.front());
    s.backend->setRemoteCacheEnabled(false);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    EXPECT_EQ(s.starts, 0);
    EXPECT_EQ(s.saves, 0);
    EXPECT_EQ(s.finishes, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
    EXPECT_EQ(pool->refCount(s.full.front()), refs);

    s.backend->setRemoteCacheEnabled(true);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 1);
    EXPECT_GT(s.saves, 0);
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    EXPECT_EQ(pool->refCount(s.full.front()), refs);
    });
}

TEST_P(EosGpuCompletionTest, NoProgressSnapshotAndNativePagesSurviveRequestRelease) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
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
    });
}
TEST_P(EosGpuCompletionTest, PartialTailUsesEarlierImmutableCheckpointNotFinalState) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    // Active, incomplete final slot intentionally differs from complete state.
    s.fill(1, "state", s.state.back(), 0xee);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 128 * s.shards - 1, true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    });
}
TEST_P(EosGpuCompletionTest, NativeSsdGdrKeepsGpuPagesThroughEosTransfer) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
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
    });
}
TEST_P(EosGpuCompletionTest, MissingStateBacksOffWithoutRelabeling) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[2]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    });
}
TEST_P(EosGpuCompletionTest, FullPrefixHolesAndTruncationKeepLargestCompleteCheckpoint) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (int variant = 0; variant < 8; ++variant) {
        SCOPED_TRACE(variant);
        Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        auto full = s.full;
        size_t expected_keys = 0;
        switch (variant) {
        case 0: full[0] = NULL_BLOCK_IDX; break;
        case 1: full[s.shards] = NULL_BLOCK_IDX; expected_keys = 1; break;
        case 2: full[3 * s.shards - 1] = NULL_BLOCK_IDX; expected_keys = 2; break;
        case 3: full.resize(2 * s.shards); expected_keys = 2; break;
        case 4:
            // An odd/native base column inside a CP canonical block, not a
            // canonical boundary. CP1 is only the degenerate control.
            full[s.shards > 1 ? s.shards + 1 : 1] = NULL_BLOCK_IDX;
            expected_keys = 1;
            break;
        case 5: {
            full[3 * s.shards - 1] = NULL_BLOCK_IDX; // FULL permits two canonical keys.
            auto state = s.state;
            state[1] = NULL_BLOCK_IDX; // The unchanged terminal check must back off to one.
            s.resource.setBatchBlocks(0, "state", state);
            expected_keys = 1;
            break;
        }
        case 6: full[3 * s.shards] = NULL_BLOCK_IDX; expected_keys = 3; break; // Uncomputed tail ignored.
        case 7: full.clear(); break;
        }
        s.resource.setBatchBlocks(0, "full", full);
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.requested_keys.load(), expected_keys);
        EXPECT_EQ(s.starts.load(), expected_keys ? 1 : 0);
        EXPECT_EQ(s.finishes.load(), expected_keys ? 1 : 0);
        EXPECT_EQ(s.aborts.load(), 0);
        EXPECT_EQ(s.resource.decodeCompletionSubmitted(), expected_keys != 0);
        if (expected_keys) EXPECT_GT(s.saves.load(), 0);
        else EXPECT_EQ(s.saves.load(), 0);
        for (const auto& [tag, pool] : s.env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
    }
    });
}
TEST_P(EosGpuCompletionTest, KnownPrefillBoundarySkipsCheckpointWithoutNewDecodeKv) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (const bool checkpoint_backoff : {false, true}) {
        SCOPED_TRACE(checkpoint_backoff);
        Scenario     s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        const size_t unit    = 128 * s.shards;
        const size_t handoff = checkpoint_backoff ? s.end() - unit : s.end();
        ASSERT_TRUE(s.resource.decodeCompletionLifecycle().recordPrefillComputedEnd(handoff));
        if (checkpoint_backoff) {
            auto state = s.state;
            state[2]   = NULL_BLOCK_IDX;
            s.resource.setBatchBlocks(0, "state", state);
        }
        // D has computed a tail beyond P. In the second case the newest full
        // boundary lacks its state page, so only P's older checkpoint is valid.
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 17, true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.starts, 0);
        EXPECT_EQ(s.saves, 0);
        EXPECT_EQ(s.finishes, 0);
        EXPECT_EQ(s.aborts, 0);
        EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
        for (const auto& [tag, pool] : s.env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
        s.dropRequestRefs();
        for (const auto block : s.full) {
            EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
        }
        for (const auto block : s.state) {
            EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
        }
    }
    });
}

TEST_P(EosGpuCompletionTest, UnknownSdkCompletionRetainsPagesAndWaitsForExplicitMockQuiescence) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.fail_save = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 0);
    // Unknown DMA completion must retain the remote WRITING session as well as
    // local GPU pages. An immediate abort could recycle its destination.
    EXPECT_EQ(s.aborts, 0);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_TRUE(s.backend->protected_quarantine_);
    EXPECT_EQ(s.backend->quarantined_task_count_, 1u);
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.starts, 1);
    auto stopped = std::async(std::launch::async, [&] { s.backend->shutdown(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    s.backend->proveMockQuiescence();
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    stopped.get();
    EXPECT_FALSE(s.backend->protected_quarantine_);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    });
}
TEST_P(EosGpuCompletionTest, DuplicateEosAndFullAdmissionDoNotDoublePinOrPublish) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
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
    });
}
TEST_P(EosGpuCompletionTest, LostCommitAckDoesNotAbortOrQuarantineCompletedGpuTransfer) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.lose_commit = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    });
}
TEST_P(EosGpuCompletionTest, UnboundedSwaNeverPretendsTwoTailPagesAreEnough) {
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), false, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[0]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
    });
}
INSTANTIATE_TEST_SUITE_P(GenericCPAndIndependentDP,
                         EosGpuCompletionTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8, 16),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 3, 8)));
}  // namespace
}  // namespace rtp_llm
