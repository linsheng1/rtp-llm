#pragma once

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <new>
#include <shared_mutex>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "kvcm_client/transfer_client.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkTransferObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvcm {

// One complete GPU IOV per SSD URI, with ordered D2D gather/scatter into a
// separately registered arena. No CPU payload allocation or fallback exists.
class PackedSsdTransferClient final: public kv_cache_manager::TransferClient, public SdkTransferObservable {
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Trace   = std::shared_ptr<kv_cache_manager::TransferTraceInfo>;

    struct InnerIo {
        Uris uris;
        Buffers buffers;
        Trace trace;
        size_t payload_bytes{0};
    };

public:
    struct GpuPackingContext {
        explicit GpuPackingContext(std::shared_ptr<void> storage, int device = -1):
            memory(std::move(storage)), device_id(device) {
            free_ranges.reserve(1026);
            free_ranges.emplace_back(0, 64ULL * 1024 * 1024);
        }
        std::shared_ptr<void> memory;
        const int             device_id;
        std::mutex            mutex;
        // GPU ranges, not whole tags, remain quarantined after uncertain DMA.
        // Fixed metadata capacity prevents allocation in the lease destructor.
        std::vector<std::pair<size_t, size_t>> free_ranges;
        size_t                                 active_reservations{0};
        // Written under ClientFactory's registry mutex before publishing the
        // first client; immutable afterwards, including during copy planning.
        bool                                         registered{false};
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
                if (address >= base && address - base <= gpu.size && span.size <= gpu.size - (address - base))
                    return true;
            }
            return false;
        }
    };
    // Without Fence, Copy finishes before returning. With Fence, copies stay
    // ordered on one caller-thread stream and Fence drains the entire window
    // before external RDMA, reuse, or return. Failed drains keep the operation.
    using Copy =
        std::function<bool(void*, kv_cache_manager::MemoryType, const void*, kv_cache_manager::MemoryType, size_t)>;
    using Fence = std::function<bool()>;
    static constexpr size_t max_operation_bytes     = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes       = 512ULL * 1024 * 1024;
    static constexpr size_t max_read_window_bytes   = 256ULL * 1024;
    static constexpr size_t max_read_window_objects = 4;
    static constexpr size_t max_write_window_bytes  = 2ULL * 1024 * 1024;
    static constexpr size_t max_write_window_objects = 64;

    PackedSsdTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> inner,
                            Copy                                              copy,
                            std::shared_ptr<GpuPackingContext>                gpu_context,
                            size_t                                            read_window_objects = 1,
                            Fence                                             fence = {}):
        inner_(std::move(inner)),
        copy_(std::move(copy)),
        gpu_context_(std::move(gpu_context)),
        staging_type_(kv_cache_manager::MemoryType::GPU),
        read_window_objects_(read_window_objects), fence_(std::move(fence)) {
        if (!inner_ || !copy_ || (!gpu_context_ || !gpu_context_->memory))
            throw std::invalid_argument("missing SSD delegate/copy operation");
        if (!read_window_objects_ || read_window_objects_ > max_read_window_objects)
            throw std::invalid_argument("invalid SSD read window size");
    }
    ~PackedSsdTransferClient() override {
        // The SDK delegate drains before staging memory (including failures)
        // is released. ClientWrapper already serializes shutdown against I/O.
        inner_.reset();
        quarantine_.reset();
    }

    void setSdkTransferObserver(SdkTransferObserver observer) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        transfer_observer_ = std::move(observer);
    }

    Code LoadKvCaches(const Uris& uris, const Buffers& destination, Trace trace = nullptr) override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return preflightFailure(kv_cache_manager::ER_INVALID_PARAMS);
        std::shared_ptr<Operation> operation;
        try {
            operation = makeOperation(uris, destination, false, read_window_objects_);
            if (!operation) return preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
            for (size_t i = 0; i < operation->read_ranges.size(); ++i)
                operation->calls[i]->trace = rangeTrace(trace, operation->read_ranges[i]);
        } catch (const std::bad_alloc&) {
            return preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
        } catch (const std::length_error&) {
            return preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
        }
        for (size_t window = 0; window < operation->read_ranges.size(); ++window) {
            bool copies_pending = false;
            try {
                if (hasQuarantine()) return preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
                const auto result = loadBatch(*operation->calls[window]);
                if (result != kv_cache_manager::ER_OK) {
                    if (!sdkCodeProvesBufferSafety(result)) quarantine(operation);
                    return result;
                }
                const auto& range = operation->read_ranges[window];
                for (size_t block = range.first; block < range.second; ++block) {
                    copies_pending = true;
                    if (!copyBlock(destination[block], operation->bytes[block].get(), false)) {
                        drainCopies();
                        copies_pending = false;
                        quarantine(operation);
                        return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                    }
                }
                const bool complete = !fence_ || fence_();
                copies_pending = false;
                if (!complete) {
                    quarantine(operation);
                    return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                }
            } catch (...) {
                if (copies_pending) drainCopies();
                quarantine(operation);
                throw;
            }
        }
        return kv_cache_manager::ER_OK;
    }

    std::pair<Code, Uris> SaveKvCaches(const Uris& uris, const Buffers& source, Trace trace = nullptr) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return {preflightFailure(kv_cache_manager::ER_INVALID_PARAMS), {}};
        std::shared_ptr<Operation> operation;
        Uris actual;
        try {
            operation = makeOperation(uris, source, true);
            if (!operation) return {preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS), {}};
            for (size_t i = 0; i < operation->write_ranges.size(); ++i)
                operation->write_calls[i]->trace = rangeTrace(trace, operation->write_ranges[i]);
            actual = uris;
        } catch (const std::bad_alloc&) {
            return {preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS), {}};
        } catch (const std::length_error&) {
            return {preflightFailure(kv_cache_manager::ER_INVALID_LOCAL_BUFFERS), {}};
        }
        auto gather = [&](size_t block) {
            return copyBlock(source[block], operation->bytes[block].get(), true);
        };
        for (size_t window = 0; window < operation->write_ranges.size(); ++window) {
            const auto& range = operation->write_ranges[window];
            // Each URI has a disjoint GPU subspan. Complete every gather in the
            // window before submitting its single multi-URI SDK operation.
            bool copies_pending = false;
            try {
                for (size_t block = range.first; block < range.second; ++block) {
                    copies_pending = true;
                    if (!gather(block)) {
                        drainCopies();
                        copies_pending = false;
                        quarantine(operation);
                        return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                    }
                }
                const bool complete = !fence_ || fence_();
                copies_pending = false;
                if (!complete) {
                    quarantine(operation);
                    return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                }
                auto result = saveBatch(*operation->write_calls[window]);
                if (result.first != kv_cache_manager::ER_OK) {
                    if (!sdkCodeProvesBufferSafety(result.first))
                        quarantine(operation);
                    return {result.first, {}};
                }
                if (!result.second.empty()) {
                    if (result.second.size() != range.second - range.first) {
                        quarantine(operation);
                        return {kv_cache_manager::ER_INVALID_PARAMS, {}};
                    }
                    for (size_t i = 0; i < result.second.size(); ++i)
                        actual[range.first + i] = std::move(result.second[i]);
                }
            } catch (...) {
                if (copies_pending) drainCopies();
                quarantine(operation);
                throw;
            }
            // Only successful completion permits the next window to reuse
            // these addresses. Unknown completion retains the entire lease.
        }
        return {kv_cache_manager::ER_OK, std::move(actual)};
    }

protected:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ER_SERVICE_UNSUPPORTED;
    }

private:
    // This operation has not touched caller/staging bytes or submitted SDK I/O.
    // Do not use this for a copy, fence, or delegate failure.
    template<typename C = Code>
    static C preflightFailure(C fallback) {
        if constexpr (detail::HasExplicitBufferSafeCodes<C>::value)
            return C::ER_SDK_ERROR_BUFFER_SAFE;
        return fallback;
    }

    bool canCoalesceCopy(void* user, const void* packed, size_t bytes) const {
        if (!gpu_context_->registered || gpu_context_->device_id < 0) return false;
        const auto source = reinterpret_cast<uintptr_t>(user);
        const auto target = reinterpret_cast<uintptr_t>(packed);
        const auto limit = std::numeric_limits<uintptr_t>::max();
        if (bytes > limit - source || bytes > limit - target) return false;
        // Even adjacent IOVs may alias staging. The original copy order then
        // matters, so leave them separate instead of issuing an overlapping copy.
        if (source < target + bytes && target < source + bytes) return false;
        return gpu_context_->coversRegisteredGpuSpan({user, bytes}, gpu_context_->device_id);
    }

    bool copyBlock(const kv_cache_manager::BlockBuffer& block, void* packed, bool write) {
        size_t offset = 0;
        for (size_t index = 0; index < block.iovs.size();) {
            const auto& first = block.iovs[index];
            // makeOperation rejects ignored write IOVs. Ignored read regions
            // consume packed bytes but must neither be copied nor bridge a run.
            if (first.ignore) {
                offset += first.size;
                ++index;
                continue;
            }
            size_t bytes = first.size, end = index + 1;
            auto* staging = static_cast<unsigned char*>(packed) + offset;
            const auto base = reinterpret_cast<uintptr_t>(first.base);
            while (end < block.iovs.size()) {
                const auto& next = block.iovs[end];
                if (next.ignore || next.type != first.type || bytes > std::numeric_limits<uintptr_t>::max() - base
                    || reinterpret_cast<uintptr_t>(next.base) != base + bytes
                    || next.size > std::numeric_limits<size_t>::max() - bytes
                    || !canCoalesceCopy(first.base, staging, bytes + next.size)) break;
                bytes += next.size;
                ++end;
            }
            if (!copy_(write ? staging : first.base, write ? staging_type_ : first.type,
                       write ? first.base : staging, write ? first.type : staging_type_, bytes)) return false;
            offset += bytes;
            index = end;
        }
        return true;
    }

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
    Code loadBatch(const InnerIo& call) {
        Code code;
        try {
            code = inner_->LoadKvCaches(call.uris, call.buffers, call.trace);
        } catch (...) {
            observe(true, 0, true);
            throw;
        }
        observe(true, code == kv_cache_manager::ER_OK ? call.payload_bytes : 0,
                code != kv_cache_manager::ER_OK);
        return code;
    }
    std::pair<Code, Uris> saveBatch(const InnerIo& call) {
        std::pair<Code, Uris> result;
        try {
            result = inner_->SaveKvCaches(call.uris, call.buffers, call.trace);
        } catch (...) {
            observe(false, 0, true);
            throw;
        }
        observe(false, result.first == kv_cache_manager::ER_OK ? call.payload_bytes : 0,
                result.first != kv_cache_manager::ER_OK);
        return result;
    }
    void drainCopies() noexcept {
        if (fence_) {
            try { (void)fence_(); } catch (...) {}
        }
    }
    static Trace rangeTrace(const Trace& trace, const std::pair<size_t, size_t>& range) {
        if (!trace) return {};
        auto result = std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
        result->block_ids.assign(trace->block_ids.begin() + range.first, trace->block_ids.begin() + range.second);
        return result;
    }
    struct GpuRangeLease {
        std::shared_ptr<GpuPackingContext> context;
        size_t                             begin{0}, bytes{0};
        ~GpuRangeLease() {
            if (!context)
                return;
            std::lock_guard<std::mutex> guard(context->mutex);
            auto&                       ranges = context->free_ranges;
            ranges.emplace_back(begin, bytes);  // capacity reserved before any I/O
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
        size_t                                 charged = 0;
        std::shared_ptr<GpuRangeLease>         gpu_lease;
        std::vector<std::shared_ptr<void>>     bytes;
        Buffers                                buffers;
        std::vector<std::shared_ptr<InnerIo>> calls;
        std::vector<std::pair<size_t, size_t>> read_ranges;
        std::vector<std::pair<size_t, size_t>> write_ranges;
        std::vector<std::shared_ptr<InnerIo>> write_calls;
        std::shared_ptr<Operation> next_quarantined;
        bool quarantined{false};
        ~Operation() {
            process_bytes_.fetch_sub(charged);
        }
    };
    void quarantine(const std::shared_ptr<Operation>& operation) {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        if (!operation->quarantined) {
            operation->quarantined = true;
            operation->next_quarantined = std::move(quarantine_);
            quarantine_ = operation;
        }
        // Only this client's unresolved GPU range stays reserved. Independent
        // tags can continue using other ranges of the registered arena.
    }
    bool hasQuarantine() {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        return static_cast<bool>(quarantine_);
    }
    std::shared_ptr<Operation>
    makeOperation(const Uris& uris, const Buffers& original, bool write, size_t read_window_objects = 1) {
        // Any uncertain completion fails this client closed until shutdown;
        // failed buffers cannot be overwritten by a later operation.
        if (hasQuarantine() || uris.empty() || uris.size() != original.size())
            return {};
        size_t              total = 0;
        std::vector<size_t> sizes;
        for (const auto& block : original) {
            size_t size = 0;
            for (const auto& iov : block.iovs) {
                if (!iov.size || (write && iov.ignore) || (!iov.ignore && !iov.base)
                    || iov.type != kv_cache_manager::MemoryType::GPU || iov.size > max_operation_bytes - size)
                    return {};
                size += iov.size;
            }
            if (!size || size > max_operation_bytes - total)
                return {};
            total += size;
            sizes.push_back(size);
        }
        auto                operation = std::make_shared<Operation>();
        std::vector<size_t> offsets(sizes.size(), 0);
        size_t              charge = 0;
        const size_t window_objects = write ? max_write_window_objects : read_window_objects;
        const size_t window_bytes = write ? max_write_window_bytes : max_read_window_bytes;
        auto& ranges = write ? operation->write_ranges : operation->read_ranges;
        for (size_t begin = 0; begin < sizes.size();) {
            size_t end = begin, bytes = 0;
            while (end < sizes.size() && end - begin < window_objects) {
                // A single large state/SWA object is allowed alone, bounded
                // by max_operation_bytes, and cannot bring a sibling with it.
                if (end > begin && sizes[end] > window_bytes - std::min(bytes, window_bytes))
                    break;
                offsets[end] = bytes;
                bytes += sizes[end++];
            }
            ranges.emplace_back(begin, end);
            charge = std::max(charge, bytes);
            begin  = end;
        }
        // Charge the largest simultaneously live GPU window, not the whole
        // logical request. A large object still reserves its own window.
        size_t previous = process_bytes_.load();
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
                auto  it =
                    std::find_if(free.begin(), free.end(), [charge](const auto& r) { return r.second >= charge; });
                if (it == free.end())
                    return {};
                offset = it->first;
                it->first += charge;
                it->second -= charge;
                if (it->second == 0)
                    free.erase(it);
                ++gpu_context_->active_reservations;
                lease->context = gpu_context_;
                lease->begin   = offset;
                lease->bytes   = charge;
            }
            operation->gpu_lease = std::move(lease);
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            operation->bytes[i] = std::shared_ptr<void>(
                gpu_context_->memory, static_cast<unsigned char*>(gpu_context_->memory.get()) + offset + offsets[i]);
            operation->buffers[i].iovs.push_back({staging_type_, operation->bytes[i].get(), sizes[i], false});
        }
        auto& calls = write ? operation->write_calls : operation->calls;
        calls.reserve(ranges.size());
        for (const auto& range : ranges) {
            auto call = std::make_shared<InnerIo>();
            call->uris.assign(uris.begin() + range.first, uris.begin() + range.second);
            call->buffers.assign(operation->buffers.begin() + range.first, operation->buffers.begin() + range.second);
            for (size_t i = range.first; i < range.second; ++i)
                call->payload_bytes += sizes[i];
            calls.push_back(std::move(call));
        }
        return operation;
    }
    inline static std::atomic<size_t>                 process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy                                              copy_;
    std::shared_ptr<GpuPackingContext>                gpu_context_;
    kv_cache_manager::MemoryType                      staging_type_;
    const size_t                                      read_window_objects_;
    Fence                                             fence_;
    std::shared_mutex                                 mutex_;
    std::mutex                                        quarantine_mutex_;
    SdkTransferObserver                               transfer_observer_;
    std::shared_ptr<Operation>                        quarantine_;
};
}  // namespace rtp_llm::kvcm
