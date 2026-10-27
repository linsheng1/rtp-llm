#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CoalescedPaceWriteClient.h"
#include "autil/legacy/json.h"
#include <cstring>
#include <map>
#if USING_CUDA
#include <cuda_runtime_api.h>
#endif
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"

namespace rtp_llm {
namespace kvcm {
namespace {
bool packedSsdRequested() {
    return autil::EnvUtil::getEnv("RTP_DTOP_PACE_SSD_PACKED_IO", false);
}
std::unique_ptr<kv_cache_manager::TransferClient> createPackedSsd(const std::string&                  config,
                                                                  const kv_cache_manager::InitParams& params) {
#if USING_CUDA
    // This candidate accepts only one explicit SSD backend; never convert a
    // normal DRAM/GDR request silently into CPU staging.
    using namespace autil::legacy;
    const auto fields   = AnyCast<json::JsonMap>(json::ParseJson(config));
    const auto sdk      = AnyCast<json::JsonMap>(fields.at("sdk_config"));
    const auto backends = AnyCast<json::JsonArray>(sdk.at("sdk_backend_configs"));
    if (backends.size() != 1 || AnyCast<std::string>(AnyCast<json::JsonMap>(backends.front()).at("type")) != "pace_ssd")
        throw std::invalid_argument("packed SSD requires exactly one pace_ssd SDK backend");
    auto cpu_params        = params;
    cpu_params.regist_span = nullptr;
    auto delegate          = kv_cache_manager::TransferClient::Create(config, cpu_params);
    if (!delegate)
        return {};
    return std::make_unique<PackedSsdTransferClient>(
        std::move(delegate),
        [](void* dst, kv_cache_manager::MemoryType dt, const void* src, kv_cache_manager::MemoryType st, size_t bytes) {
            using M = kv_cache_manager::MemoryType;
            if (dt == M::CPU && st == M::CPU) {
                std::memcpy(dst, src, bytes);
                return true;
            }
            if ((dt != M::CPU && dt != M::GPU) || (st != M::CPU && st != M::GPU))
                return false;
            const auto kind = dt == M::GPU ? (st == M::GPU ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) :
                                             cudaMemcpyDeviceToHost;
            return cudaMemcpy(dst, src, bytes, kind) == cudaSuccess;
        });
#else
    throw std::invalid_argument("packed SSD GPU staging requires CUDA");
#endif
}
bool coalescedPaceWriteRequested() {
    return autil::EnvUtil::getEnv("RTP_KVCM_COALESCE_PACE_WRITES", false);
}
std::unique_ptr<kv_cache_manager::TransferClient>
createNativeSsdGpu(const std::string& config,
                   const kv_cache_manager::InitParams& params,
                   const kv_cache_manager::ClientMemoryRegistrations& registrations) {
#if USING_CUDA
    // PACE SSD treats each IOV as a complete object write. A model page has
    // several layer IOVs, so gather into one registered GPU span per URI.
    // This is D2D packing, not the legacy CPU snapshot/packed wrapper.
    using namespace autil::legacy;
    const auto fields = AnyCast<json::JsonMap>(json::ParseJson(config));
    const auto sdk = AnyCast<json::JsonMap>(fields.at("sdk_config"));
    const auto backends = AnyCast<json::JsonArray>(sdk.at("sdk_backend_configs"));
    if (backends.size() != 1 || AnyCast<std::string>(AnyCast<json::JsonMap>(backends.front()).at("type")) != "pace_ssd")
        throw std::invalid_argument("native SSD GDR requires exactly one pace_ssd backend");
    // PACE seals GPU registration at the first Init. Reuse one pre-registered
    // span across this process/device's tag clients; lease disjoint ranges per operation.
    using Context = PackedSsdTransferClient::GpuPackingContext;
    static std::mutex registry_mutex;
    static std::map<int, std::weak_ptr<Context>> contexts;
    static std::atomic<size_t> resident_bytes{0};
    constexpr size_t bytes = PackedSsdTransferClient::max_operation_bytes;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return {};
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto context = contexts[device].lock();
    if (!context) {
        size_t previous = resident_bytes.load();
        do {
            if (previous > PackedSsdTransferClient::max_process_bytes - bytes)
                throw std::runtime_error("native SSD registered GPU packing capacity exhausted");
        } while (!resident_bytes.compare_exchange_weak(previous, previous + bytes));
        void* base = nullptr;
        if (cudaMalloc(&base, bytes) != cudaSuccess) {
            resident_bytes.fetch_sub(bytes);
            return {};
        }
        auto arena = std::shared_ptr<void>(base, [device](void* p) {
            int previous_device = device;
            cudaGetDevice(&previous_device);
            cudaSetDevice(device);
            cudaFree(p);
            cudaSetDevice(previous_device);
            resident_bytes.fetch_sub(bytes);
        });
        context = std::make_shared<Context>(std::move(arena));
        contexts[device] = context;
    }
    auto memory = registrations;
    if (!context->registered)
        memory.gpu.push_back({context->memory.get(), bytes, device});
    auto delegate = kv_cache_manager::TransferClient::Create(config, params, memory);
    if (!delegate)
        return {};
    context->registered = true;
    return std::make_unique<PackedSsdTransferClient>(
        std::move(delegate),
        [](void* dst, kv_cache_manager::MemoryType dt, const void* src, kv_cache_manager::MemoryType st, size_t size) {
            using M = kv_cache_manager::MemoryType;
            // D2D cudaMemcpy need not wait on the host. External RDMA is
            // not ordered by the CUDA default stream: publish/scatter must
            // finish before handing off or reusing the registered range.
            return dt == M::GPU && st == M::GPU
                   && cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess
                   && cudaStreamSynchronize(nullptr) == cudaSuccess;
        },
        std::move(context));
#else
    throw std::invalid_argument("native SSD GDR requires CUDA");
#endif
}
std::unique_ptr<kv_cache_manager::TransferClient>
coalescePaceWrite(const std::string& config, std::unique_ptr<kv_cache_manager::TransferClient> delegate) {
#if USING_CUDA
    if (packedSsdRequested())
        throw std::invalid_argument("coalesced PACE writes cannot wrap packed SSD");
    using namespace autil::legacy;
    const auto fields   = AnyCast<json::JsonMap>(json::ParseJson(config));
    const auto sdk      = AnyCast<json::JsonMap>(fields.at("sdk_config"));
    const auto backends = AnyCast<json::JsonArray>(sdk.at("sdk_backend_configs"));
    if (backends.size() != 1 || AnyCast<std::string>(AnyCast<json::JsonMap>(backends.front()).at("type")) != "pace")
        throw std::invalid_argument("coalesced writes require one explicit pace DRAM backend");
    if (!delegate)
        return {};
    CoalescedPaceWriteClient::CopyRows rows;
    if (autil::EnvUtil::getEnv("RTP_KVCM_COALESCE_2D_PACK", false)) {
        rows = [](void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
            return cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDeviceToHost) == cudaSuccess;
        };
    }
    return std::make_unique<CoalescedPaceWriteClient>(
        std::move(delegate),
        [](void* dst, kv_cache_manager::MemoryType dt, const void* src, kv_cache_manager::MemoryType st, size_t bytes) {
            using M = kv_cache_manager::MemoryType;
            if (dt == M::CPU && st == M::CPU) {
                std::memcpy(dst, src, bytes);
                return true;
            }
            if (dt != M::CPU || st != M::GPU)
                return false;
            return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        },
        std::move(rows));
#else
    throw std::invalid_argument("coalesced PACE GPU writes require CUDA");
#endif
}
}  // namespace

std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    if (packedSsdRequested())
        return createPackedSsd(config, init_params);
    auto client = kv_cache_manager::TransferClient::Create(config, init_params);
    return coalescedPaceWriteRequested() ? coalescePaceWrite(config, std::move(client)) : std::move(client);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClientWithMemory(
    const std::string&                                config,
    const kv_cache_manager::InitParams&                init_params,
    const kv_cache_manager::ClientMemoryRegistrations& memory_registrations) const {
    if (packedSsdRequested())
        throw std::invalid_argument("packed SSD forbids GDR registrations");
    if (coalescedPaceWriteRequested())
        throw std::invalid_argument("GDR registrations forbid CPU coalesced PACE writes");
    if (autil::EnvUtil::getEnv("RTP_DTOP_PACE_SSD_NATIVE_GDR", false))
        return createNativeSsdGpu(config, init_params, memory_registrations);
    return kv_cache_manager::TransferClient::Create(config, init_params, memory_registrations);
}

std::unique_ptr<Subscriber> ClientFactory::createSubscriber(bool enable_vipserver) const {
    if (enable_vipserver) {
        return std::make_unique<VIPServerSubscriber>();
    }
    return std::make_unique<DirectSubscriber>();
}

}  // namespace kvcm
}  // namespace rtp_llm
