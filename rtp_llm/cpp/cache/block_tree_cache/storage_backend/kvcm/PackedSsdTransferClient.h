#pragma once

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "kvcm_client/transfer_client.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkTransferObserver.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvcm {

// One complete GPU IOV per SSD URI, with ordered D2D gather/scatter into a
// separately registered arena. No CPU payload allocation or fallback exists.
class PackedSsdTransferClient final: public kv_cache_manager::TransferClient, public SdkTransferObservable {
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
        std::vector<kv_cache_manager::GpuMemorySpan> registered_gpu_spans;

        bool coversRegisteredGpuSpan(const kv_cache_manager::RegistSpan& span, int device) const noexcept {
            if (!registered || !span.base || span.size == 0)
                return false;
            const auto address = reinterpret_cast<uintptr_t>(span.base);
            for (const auto& gpu : registered_gpu_spans) {
                if (gpu.device_id != device || !gpu.base)
                    continue;
                const auto base = reinterpret_cast<uintptr_t>(gpu.base);
                // Subtraction avoids wrapping either end address.
                if (address >= base && address - base <= gpu.size
                    && span.size <= gpu.size - (address - base))
                    return true;
            }
            return false;
        }
    };
    // Copy must finish before returning: external RDMA cannot rely on CUDA
    // stream ordering to observe a gather or protect an in-flight scatter.
    using Copy =
        std::function<bool(void*, kv_cache_manager::MemoryType, const void*, kv_cache_manager::MemoryType, size_t)>;
    using ReadThreadInit = std::function<bool()>;
    static constexpr size_t max_operation_bytes = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes   = 512ULL * 1024 * 1024;
    static constexpr size_t max_read_window_bytes = 256ULL * 1024;
    static constexpr size_t max_read_window_objects = 4;

    PackedSsdTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> inner,
                           Copy copy,
                           std::shared_ptr<GpuPackingContext> gpu_context,
                           ReadThreadInit read_thread_init = {}):
        inner_(std::move(inner)), copy_(std::move(copy)), gpu_context_(std::move(gpu_context)),
        staging_type_(kv_cache_manager::MemoryType::GPU), read_thread_init_(std::move(read_thread_init)) {
        if (!inner_ || !copy_ || (!gpu_context_ || !gpu_context_->memory))
            throw std::invalid_argument("missing SSD delegate/copy operation");
    }
    ~PackedSsdTransferClient() override {
        // The SDK delegate drains before staging memory (including failures)
        // is released. ClientWrapper already serializes shutdown against I/O.
        inner_.reset();
        quarantine_.reset();
    }

    void setSdkTransferObserver(SdkTransferObserver observer) override {
        std::lock_guard<std::mutex> lock(mutex_);
        transfer_observer_ = std::move(observer);
    }

    Code LoadKvCaches(const Uris& uris, const Buffers& destination, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return kv_cache_manager::ER_INVALID_PARAMS;
        auto operation = makeOperation(uris, destination, false, read_thread_init_ ? max_read_window_objects : 1);
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
        // Bound simultaneous object leases by count and payload. Every object
        // in a window has a disjoint registered GPU span. Join all SDK calls
        // before scattering or reusing any window memory, including failures.
        for (const auto& range : operation->read_ranges) {
            try {
                auto result = kv_cache_manager::ER_OK;
                if (read_thread_init_ && range.second - range.first > 1) {
                    std::vector<std::future<Code>> reads;
                    reads.reserve(range.second - range.first);
                    for (size_t block = range.first; block < range.second; ++block) {
                        reads.emplace_back(std::async(std::launch::async,
                            [this, uri = uris[block], buffer = operation->buffers[block],
                             one_trace = blockTrace(trace, block, uris.size())]() mutable {
                                if (!read_thread_init_())
                                    return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                                return loadOne(uri, buffer, std::move(one_trace));
                            }));
                    }
                    std::exception_ptr failure;
                    for (auto& read : reads) {
                        try {
                            const auto code = read.get();
                            if (result == kv_cache_manager::ER_OK)
                                result = code;
                        } catch (...) {
                            if (!failure)
                                failure = std::current_exception();
                        }
                    }
                    if (failure)
                        std::rethrow_exception(failure);
                } else {
                    const auto block = range.first;
                    result = loadOne(uris[block], operation->buffers[block], blockTrace(trace, block, uris.size()));
                }
                if (result != kv_cache_manager::ER_OK) {
                    quarantine(operation);
                    return result;
                }
                for (size_t block = range.first; block < range.second; ++block) {
                    if (!scatter(block)) {
                        quarantine(operation);
                        return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                    }
                }
            } catch (...) {
                quarantine(operation);
                throw;
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
        auto actual = uris;
        for (size_t block = 0; block < uris.size(); ++block) {
            // The delegate consumes one URI at a time. Pack just that URI,
            // and never overwrite its range before confirmed completion.
            if (!gather(block)) {
                quarantine(operation);
                return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
            }
            try {
                auto result = saveOne(uris[block], operation->buffers[block], blockTrace(trace, block, uris.size()));
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
    void observe(bool is_read, size_t bytes, bool unknown) const noexcept {
        if (transfer_observer_) {
            try {
                transfer_observer_(is_read, bytes, unknown);
            } catch (...) {
                // Telemetry must not rewrite an SDK result or create another
                // completion-unknown event for a successful SDK operation.
                try {
                    RTP_LLM_LOG_ERROR("failed to report inner SSD SDK payload");
                } catch (...) {}
            }
        }
    }
    Code loadOne(const std::string& uri, const kv_cache_manager::BlockBuffer& buffer, Trace trace) {
        // Prepare descriptors before the attempted-I/O boundary. Allocation or
        // trace failures here have not called the SDK and report no unknown I/O.
        const Uris uris{uri};
        const Buffers buffers{buffer};
        Code code;
        try {
            code = inner_->LoadKvCaches(uris, buffers, std::move(trace));
        } catch (...) {
            observe(true, 0, true);
            throw;
        }
        observe(true, code == kv_cache_manager::ER_OK ? buffer.iovs.front().size : 0,
                code != kv_cache_manager::ER_OK);
        return code;
    }
    std::pair<Code, Uris> saveOne(const std::string& uri, const kv_cache_manager::BlockBuffer& buffer, Trace trace) {
        const Uris uris{uri};
        const Buffers buffers{buffer};
        std::pair<Code, Uris> result;
        try {
            result = inner_->SaveKvCaches(uris, buffers, std::move(trace));
        } catch (...) {
            observe(false, 0, true);
            throw;
        }
        observe(false, result.first == kv_cache_manager::ER_OK ? buffer.iovs.front().size : 0,
                result.first != kv_cache_manager::ER_OK);
        return result;
    }
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
        std::vector<std::pair<size_t, size_t>>   read_ranges;
        ~Operation() {
            process_bytes_.fetch_sub(charged);
        }
    };
    void quarantine(const std::shared_ptr<Operation>& operation) {
        quarantine_ = operation;
        // Only this client's unresolved GPU range stays reserved. Independent
        // tags can continue using other ranges of the registered arena.
    }
    std::shared_ptr<Operation> makeOperation(const Uris& uris, const Buffers& original, bool write,
                                           size_t read_window_objects = 1) {
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
                    || iov.type != kv_cache_manager::MemoryType::GPU
                    || iov.size > max_operation_bytes - size)
                    return {};
                size += iov.size;
            }
            if (!size || size > max_operation_bytes - total)
                return {};
            total += size;
            sizes.push_back(size);
        }
        auto   operation = std::make_shared<Operation>();
        std::vector<size_t> offsets(sizes.size(), 0);
        size_t charge = 0;
        for (size_t begin = 0; begin < sizes.size();) {
            size_t end = begin, bytes = 0;
            while (end < sizes.size() && end - begin < read_window_objects) {
                // A single large state/SWA object is allowed alone, bounded
                // by max_operation_bytes, and cannot bring a sibling with it.
                if (end > begin && sizes[end] > max_read_window_bytes - std::min(bytes, max_read_window_bytes))
                    break;
                offsets[end] = bytes;
                bytes += sizes[end++];
            }
            operation->read_ranges.emplace_back(begin, end);
            charge = std::max(charge, bytes);
            begin = end;
        }
        // Charge the largest simultaneously live GPU window, not the whole
        // logical request. Writes remain single-object and unchanged.
        size_t previous  = process_bytes_.load();
        do {
            if (previous > max_process_bytes - charge)
                return {};
        } while (!process_bytes_.compare_exchange_weak(previous, previous + charge));
        operation->charged = charge;
        operation->bytes.resize(sizes.size());
        operation->buffers.resize(sizes.size());
        size_t offset = 0;
        {
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
            operation->bytes[i] = std::shared_ptr<void>(
                gpu_context_->memory, static_cast<unsigned char*>(gpu_context_->memory.get()) + offset + offsets[i]);
            operation->buffers[i].iovs.push_back(
                {staging_type_, operation->bytes[i].get(), sizes[i], false});
            // Reuse only after the previous window completed and scattered.
        }
        return operation;
    }
    inline static std::atomic<size_t>                 process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy                                              copy_;
    std::shared_ptr<GpuPackingContext>                 gpu_context_;
    kv_cache_manager::MemoryType                       staging_type_;
    ReadThreadInit                                    read_thread_init_;
    std::mutex                                        mutex_;
    SdkTransferObserver                               transfer_observer_;
    std::shared_ptr<Operation>                        quarantine_;
};
}  // namespace rtp_llm::kvcm
