#pragma once

#include <atomic>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "kvcm_client/transfer_client.h"

namespace rtp_llm::kvcm {

// Explicit SSD compatibility candidate. One complete SDK IOV per URI, followed
// by ordered scatter to the original destinations. The legacy constructor uses
// CPU staging. Native SSD GDR supplies a separately registered GPU arena and
// performs only D2D gather/scatter; it never allocates a host payload snapshot.
class PackedSsdTransferClient final: public kv_cache_manager::TransferClient {
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Trace   = std::shared_ptr<kv_cache_manager::TransferTraceInfo>;

public:
    struct GpuPackingContext {
        explicit GpuPackingContext(std::shared_ptr<void> storage): memory(std::move(storage)) {
            free_ranges.reserve(1026);
            free_ranges.emplace_back(0, 64ULL * 1024 * 1024);
        }
        std::shared_ptr<void> memory;
        std::mutex mutex;
        // GPU ranges, not whole tags, remain quarantined after uncertain DMA.
        // Fixed metadata capacity prevents allocation in the lease destructor.
        std::vector<std::pair<size_t, size_t>> free_ranges;
        size_t active_reservations{0};
        // Accessed only while ClientFactory's device registry mutex is held.
        bool registered{false};
    };
    // Copy must finish before returning: external RDMA cannot rely on CUDA
    // stream ordering to observe a gather or protect an in-flight scatter.
    using Copy =
        std::function<bool(void*, kv_cache_manager::MemoryType, const void*, kv_cache_manager::MemoryType, size_t)>;
    static constexpr size_t max_operation_bytes = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes   = 512ULL * 1024 * 1024;

    PackedSsdTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> inner,
                           Copy copy,
                           std::shared_ptr<GpuPackingContext> gpu_context = {}):
        inner_(std::move(inner)), copy_(std::move(copy)), gpu_context_(std::move(gpu_context)),
        staging_type_(gpu_context_ ? kv_cache_manager::MemoryType::GPU : kv_cache_manager::MemoryType::CPU) {
        if (!inner_ || !copy_ || (gpu_context_ && !gpu_context_->memory))
            throw std::invalid_argument("missing SSD delegate/copy operation");
    }
    ~PackedSsdTransferClient() override {
        // The SDK delegate drains before staging memory (including failures)
        // is released. ClientWrapper already serializes shutdown against I/O.
        inner_.reset();
        quarantine_.reset();
    }

    Code LoadKvCaches(const Uris& uris, const Buffers& destination, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return kv_cache_manager::ER_INVALID_PARAMS;
        auto operation = makeOperation(uris, destination, false);
        if (!operation)
            return kv_cache_manager::ER_INVALID_LOCAL_BUFFERS;
        auto scatter = [&](size_t block) {
            size_t offset = 0;
            for (const auto& iov : destination[block].iovs) {
                if (!iov.ignore
                    && !copy_(iov.base,
                              iov.type,
                              static_cast<unsigned char*>(operation->bytes[block].get()) + offset,
                              staging_type_,
                              iov.size)) {
                    return false;
                }
                offset += iov.size;
            }
            return true;
        };
        // A small Provider object cache cannot pin an entire client batch.
        // Release each completed URI before admitting the next one. This is
        // an explicit correctness path; its throughput needs separate testing.
        for (size_t block = 0; block < uris.size(); ++block) {
            Code result;
            try {
                result = inner_->LoadKvCaches(
                    {uris[block]}, {operation->buffers[block]}, blockTrace(trace, block, uris.size()));
            } catch (...) {
                quarantine(operation);
                throw;
            }
            if (result != kv_cache_manager::ER_OK) {
                quarantine(operation);
                return result;
            }
            // The single GPU range can be reused only after this URI has
            // completed and its result has reached the original GPU pages.
            if (gpu_context_ && !scatter(block)) {
                quarantine(operation);
                return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
            }
        }
        if (!gpu_context_) {
            for (size_t block = 0; block < destination.size(); ++block) {
                if (!scatter(block)) {
                    quarantine(operation);
                    return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                }
            }
        }
        return kv_cache_manager::ER_OK;
    }

    std::pair<Code, Uris> SaveKvCaches(const Uris& uris, const Buffers& source, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return {kv_cache_manager::ER_INVALID_PARAMS, {}};
        auto operation = makeOperation(uris, source, true);
        if (!operation)
            return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
        auto gather = [&](size_t block) {
            size_t offset = 0;
            for (const auto& iov : source[block].iovs) {
                if (!copy_(static_cast<unsigned char*>(operation->bytes[block].get()) + offset,
                           staging_type_,
                           iov.base,
                           iov.type,
                           iov.size)) {
                    return false;
                }
                offset += iov.size;
            }
            return true;
        };
        if (!gpu_context_) {
            for (size_t block = 0; block < source.size(); ++block) {
                if (!gather(block)) {
                    quarantine(operation);
                    return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                }
            }
        }
        auto actual = uris;
        for (size_t block = 0; block < uris.size(); ++block) {
            // The delegate consumes one URI at a time. Pack just that URI,
            // and never overwrite its range before confirmed completion.
            if (gpu_context_ && !gather(block)) {
                quarantine(operation);
                return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
            }
            try {
                auto result = inner_->SaveKvCaches(
                    {uris[block]}, {operation->buffers[block]}, blockTrace(trace, block, uris.size()));
                if (result.first != kv_cache_manager::ER_OK) {
                    quarantine(operation);
                    return {result.first, {}};
                }
                if (!result.second.empty()) {
                    if (result.second.size() != 1) {
                        quarantine(operation);
                        return {kv_cache_manager::ER_INVALID_PARAMS, {}};
                    }
                    actual[block] = std::move(result.second.front());
                }
            } catch (...) {
                quarantine(operation);
                throw;
            }
        }
        return {kv_cache_manager::ER_OK, std::move(actual)};
    }

protected:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ER_SERVICE_UNSUPPORTED;
    }

private:
    static Trace blockTrace(const Trace& trace, size_t block, size_t count) {
        if (!trace)
            return {};
        auto result = std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
        if (trace->block_ids.size() == count)
            result->block_ids = {trace->block_ids[block]};
        else
            return {};
        return result;
    }
    struct GpuRangeLease {
        std::shared_ptr<GpuPackingContext> context;
        size_t begin{0}, bytes{0};
        ~GpuRangeLease() {
            if (!context)
                return;
            std::lock_guard<std::mutex> guard(context->mutex);
            auto& ranges = context->free_ranges;
            ranges.emplace_back(begin, bytes); // capacity reserved before any I/O
            std::sort(ranges.begin(), ranges.end());
            size_t kept = 0;
            for (size_t index = 0; index < ranges.size(); ++index) {
                const auto range = ranges[index];
                if (kept && ranges[kept - 1].first + ranges[kept - 1].second == range.first)
                    ranges[kept - 1].second += range.second;
                else
                    ranges[kept++] = range;
            }
            ranges.resize(kept);
            --context->active_reservations;
        }
    };
    struct Operation {
        size_t                                  charged = 0;
        std::shared_ptr<GpuRangeLease>           gpu_lease;
        std::vector<std::shared_ptr<void>>      bytes;
        Buffers                                 buffers;
        ~Operation() {
            process_bytes_.fetch_sub(charged);
        }
    };
    void quarantine(const std::shared_ptr<Operation>& operation) {
        quarantine_ = operation;
        // Only this client's unresolved GPU range stays reserved. Independent
        // tags can continue using other ranges of the registered arena.
    }
    std::shared_ptr<Operation> makeOperation(const Uris& uris, const Buffers& original, bool write) {
        // Any uncertain completion fails this client closed until shutdown;
        // failed buffers cannot be overwritten by a later operation.
        if (quarantine_ || uris.empty() || uris.size() != original.size())
            return {};
        size_t              total = 0;
        std::vector<size_t> sizes;
        for (const auto& block : original) {
            size_t size = 0;
            for (const auto& iov : block.iovs) {
                if (!iov.size || (write && iov.ignore) || (!iov.ignore && !iov.base)
                    || (iov.type != kv_cache_manager::MemoryType::CPU && iov.type != kv_cache_manager::MemoryType::GPU)
                    || (gpu_context_ && !iov.ignore && iov.type != kv_cache_manager::MemoryType::GPU)
                    || iov.size > max_operation_bytes - size)
                    return {};
                size += iov.size;
            }
            if (!size || size > max_operation_bytes - total)
                return {};
            total += size;
            sizes.push_back(size);
        }
        // Bound the logical batch as before, but charge only simultaneously
        // live GPU packing memory. Reserving the whole batch can starve other
        // healthy cache tags even though their SDK calls are per URI.
        const size_t charge = gpu_context_ ? *std::max_element(sizes.begin(), sizes.end()) : total;
        auto   operation = std::make_shared<Operation>();
        size_t previous  = process_bytes_.load();
        do {
            if (previous > max_process_bytes - charge)
                return {};
        } while (!process_bytes_.compare_exchange_weak(previous, previous + charge));
        operation->charged = charge;
        operation->bytes.resize(sizes.size());
        operation->buffers.resize(sizes.size());
        size_t offset = 0;
        if (gpu_context_) {
            auto lease = std::make_shared<GpuRangeLease>();
            {
                std::lock_guard<std::mutex> guard(gpu_context_->mutex);
                if (gpu_context_->active_reservations >= 1024)
                    return {};
                auto& free = gpu_context_->free_ranges;
                auto it = std::find_if(free.begin(), free.end(), [charge](const auto& r) { return r.second >= charge; });
                if (it == free.end())
                    return {};
                offset = it->first;
                it->first += charge;
                it->second -= charge;
                if (it->second == 0)
                    free.erase(it);
                ++gpu_context_->active_reservations;
                lease->context = gpu_context_;
                lease->begin = offset;
                lease->bytes = charge;
            }
            operation->gpu_lease = std::move(lease);
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (gpu_context_) {
                operation->bytes[i] = std::shared_ptr<void>(
                    gpu_context_->memory, static_cast<unsigned char*>(gpu_context_->memory.get()) + offset);
            } else {
                operation->bytes[i] = std::shared_ptr<void>(
                    new unsigned char[sizes[i]], [](void* p) { delete[] static_cast<unsigned char*>(p); });
            }
            operation->buffers[i].iovs.push_back(
                {staging_type_, operation->bytes[i].get(), sizes[i], false});
            // GPU blocks alias this operation's single-URI range. CPU
            // compatibility buffers remain distinct allocations.
        }
        return operation;
    }
    inline static std::atomic<size_t>                 process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy                                              copy_;
    std::shared_ptr<GpuPackingContext>                 gpu_context_;
    kv_cache_manager::MemoryType                       staging_type_;
    std::mutex                                        mutex_;
    std::shared_ptr<Operation>                        quarantine_;
};
}  // namespace rtp_llm::kvcm
