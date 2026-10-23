#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "autil/EnvUtil.h"
#include "autil/legacy/json.h"
#include <cstring>
#if USING_CUDA
#include <cuda_runtime_api.h>
#endif
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"

namespace rtp_llm {
namespace kvcm {
namespace {
bool packedSsdRequested() { return autil::EnvUtil::getEnv("RTP_DTOP_PACE_SSD_PACKED_IO", false); }
std::unique_ptr<kv_cache_manager::TransferClient> createPackedSsd(
    const std::string& config, const kv_cache_manager::InitParams& params) {
#if USING_CUDA
    // This candidate accepts only one explicit SSD backend; never convert a
    // normal DRAM/GDR request silently into CPU staging.
    using namespace autil::legacy;
    const auto fields=AnyCast<json::JsonMap>(json::ParseJson(config));
    const auto sdk=AnyCast<json::JsonMap>(fields.at("sdk_config"));
    const auto backends=AnyCast<json::JsonArray>(sdk.at("sdk_backend_configs"));
    if (backends.size()!=1 || AnyCast<std::string>(AnyCast<json::JsonMap>(backends.front()).at("type"))!="pace_ssd")
        throw std::invalid_argument("packed SSD requires exactly one pace_ssd SDK backend");
    auto cpu_params=params;cpu_params.regist_span=nullptr;
    auto delegate=kv_cache_manager::TransferClient::Create(config,cpu_params);
    if (!delegate) return {};
    return std::make_unique<PackedSsdTransferClient>(std::move(delegate),
        [](void* dst,kv_cache_manager::MemoryType dt,const void* src,kv_cache_manager::MemoryType st,size_t bytes) {
            using M=kv_cache_manager::MemoryType;
            if (dt==M::CPU && st==M::CPU) { std::memcpy(dst,src,bytes); return true; }
            if ((dt!=M::CPU && dt!=M::GPU) || (st!=M::CPU && st!=M::GPU)) return false;
            const auto kind=dt==M::GPU ? (st==M::GPU?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice):cudaMemcpyDeviceToHost;
            return cudaMemcpy(dst,src,bytes,kind)==cudaSuccess;
        });
#else
    throw std::invalid_argument("packed SSD GPU staging requires CUDA");
#endif
}
} // namespace


std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    if (packedSsdRequested()) return createPackedSsd(config,init_params);
    return kv_cache_manager::TransferClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClientWithMemory(
    const std::string&                                config,
    const kv_cache_manager::InitParams&                init_params,
    const kv_cache_manager::ClientMemoryRegistrations& memory_registrations) const {
    if (packedSsdRequested()) throw std::invalid_argument("packed SSD forbids GDR registrations");
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
