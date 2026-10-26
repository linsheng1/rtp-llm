#pragma once

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include "kvcm_client/transfer_client.h"

namespace rtp_llm::kvcm {
// Explicit RDMA CPU-staging candidate. The wire order/URI/checkpoint identity
// is unchanged: concatenate one URI's IOVs before entering the SDK. Reads keep
// their original batched GPU path. This is not a GDR implementation.
class CoalescedPaceWriteClient final: public kv_cache_manager::TransferClient {
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Trace   = std::shared_ptr<kv_cache_manager::TransferTraceInfo>;

public:
    using Copy =
        std::function<bool(void*, kv_cache_manager::MemoryType, const void*, kv_cache_manager::MemoryType, size_t)>;
    using CopyRows = std::function<bool(void*, size_t, const void*, size_t, size_t, size_t)>;
    static constexpr size_t max_operation_bytes = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes   = 512ULL * 1024 * 1024;
    CoalescedPaceWriteClient(std::unique_ptr<kv_cache_manager::TransferClient> inner, Copy copy, CopyRows rows = {}):
        inner_(std::move(inner)), copy_(std::move(copy)), rows_(std::move(rows)) {
        if (!inner_ || !copy_)
            throw std::invalid_argument("missing coalesced PACE delegate/copy");
    }
    ~CoalescedPaceWriteClient() override {
        // Drain possible DMA before freeing a failed operation's backing.
        inner_.reset();
        quarantine_.reset();
    }
    Code LoadKvCaches(const Uris& uris, const Buffers& buffers, Trace trace = nullptr) override {
        // Preserve the SDK's concurrent read admission. Only uncertain writes
        // close this wrapper; normal reads do not acquire the packing mutex.
        if (failed_.load(std::memory_order_acquire))
            return kv_cache_manager::ER_INVALID_LOCAL_BUFFERS;
        return inner_->LoadKvCaches(uris, buffers, std::move(trace));
    }
    std::pair<Code, Uris> SaveKvCaches(const Uris& uris, const Buffers& buffers, Trace trace = nullptr) override {
        const auto entered = std::chrono::steady_clock::now();
        const auto wall_begin =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::lock_guard<std::mutex> guard(mutex_);
        const auto                  locked = std::chrono::steady_clock::now();
        if (quarantine_)
            return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
        // A bounded SDK range can retain its parent's trace metadata. It is
        // opaque to this adapter and is forwarded without rebasing or cloning.
        if (uris.empty() || uris.size() != buffers.size())
            return {kv_cache_manager::ER_INVALID_PARAMS, {}};
        size_t                           total = 0, iovs = 0, columns = 0;
        bool                             useful = false, gpu = false;
        std::vector<size_t>              sizes;
        std::vector<std::vector<size_t>> offsets;
        for (const auto& block : buffers) {
            size_t              n = 0;
            std::vector<size_t> row_offsets;
            if (block.iovs.empty())
                return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
            for (const auto& iov : block.iovs) {
                // Ignored-write semantics belong to the SDK; do not invent bytes.
                if (iov.ignore)
                    return inner_->SaveKvCaches(uris, buffers, std::move(trace));
                if (!iov.base || !iov.size
                    || (iov.type != kv_cache_manager::MemoryType::CPU && iov.type != kv_cache_manager::MemoryType::GPU)
                    || iov.size > max_operation_bytes - n)
                    return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
                row_offsets.push_back(n);
                n += iov.size;
                ++iovs;
                gpu |= iov.type == kv_cache_manager::MemoryType::GPU;
            }
            if (n > max_operation_bytes - total)
                return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
            total += n;
            sizes.push_back(n);
            offsets.push_back(std::move(row_offsets));
            useful |= block.iovs.size() > 1;
            columns = std::max(columns, block.iovs.size());
        }
        // Already-packed CPU buffers need no extra host copy.
        if (!useful && !(rows_ && gpu && buffers.size() > 1))
            return inner_->SaveKvCaches(uris, buffers, std::move(trace));
        const auto begin    = std::chrono::steady_clock::now();
        auto       op       = std::make_shared<Operation>();
        size_t     previous = process_bytes_.load();
        do {
            if (previous > max_process_bytes - total)
                return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
        } while (!process_bytes_.compare_exchange_weak(previous, previous + total));
        op->charged = total;
        op->bytes.resize(total);
        op->buffers.resize(sizes.size());
        size_t position = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            op->buffers[i].iovs.push_back(
                {kv_cache_manager::MemoryType::CPU, op->bytes.data() + position, sizes[i], false});
            position += sizes[i];
        }
        size_t copies = 0, row_copies = 0;
        // Combine only proven regular rows. Fragmented slots, mixed memory
        // types, unequal widths and irregular offsets fall back to exact IOVs.
        for (size_t column = 0; column < columns; ++column) {
            for (size_t block = 0; block < buffers.size();) {
                if (column >= buffers[block].iovs.size()) {
                    ++block;
                    continue;
                }
                const auto& v   = buffers[block].iovs[column];
                auto*       dst = static_cast<unsigned char*>(op->buffers[block].iovs[0].base) + offsets[block][column];
                size_t      height = 1, sp = 0, dp = 0;
                if (rows_ && v.type == kv_cache_manager::MemoryType::GPU && block + 1 < buffers.size()
                    && column < buffers[block + 1].iovs.size()) {
                    const auto& next = buffers[block + 1].iovs[column];
                    auto*       nd =
                        static_cast<unsigned char*>(op->buffers[block + 1].iovs[0].base) + offsets[block + 1][column];
                    auto a = reinterpret_cast<uintptr_t>(v.base), b = reinterpret_cast<uintptr_t>(next.base);
                    if (next.type == v.type && next.size == v.size && b > a && nd > dst) {
                        sp = b - a;
                        dp = nd - dst;
                        if (sp >= v.size && dp >= v.size && sp <= (1ULL << 31) - 1 && dp <= (1ULL << 31) - 1) {
                            height = 2;
                            while (block + height < buffers.size() && column < buffers[block + height].iovs.size()) {
                                const auto& n     = buffers[block + height].iovs[column];
                                const auto& prior = buffers[block + height - 1].iovs[column];
                                auto        na    = reinterpret_cast<uintptr_t>(n.base),
                                     pa           = reinterpret_cast<uintptr_t>(prior.base);
                                auto* d = static_cast<unsigned char*>(op->buffers[block + height].iovs[0].base)
                                          + offsets[block + height][column];
                                auto* pd = static_cast<unsigned char*>(op->buffers[block + height - 1].iovs[0].base)
                                           + offsets[block + height - 1][column];
                                if (n.type != v.type || n.size != v.size || na <= pa || na - pa != sp
                                    || d - pd != static_cast<std::ptrdiff_t>(dp))
                                    break;
                                ++height;
                            }
                        }
                    }
                }
                bool ok;
                if (height > 1) {
                    ok = rows_(dst, dp, v.base, sp, v.size, height);
                    ++row_copies;
                } else
                    ok = copy_(dst, kv_cache_manager::MemoryType::CPU, v.base, v.type, v.size);
                ++copies;
                if (!ok)
                    return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                block += height;
            }
        }
        const auto copied = std::chrono::steady_clock::now();
        try {
            auto result = inner_->SaveKvCaches(uris, op->buffers, std::move(trace));
            if (result.first != kv_cache_manager::ER_OK
                || (!result.second.empty() && result.second.size() != uris.size())) {
                quarantine_ = op;
                failed_.store(true, std::memory_order_release);
                if (result.first == kv_cache_manager::ER_OK)
                    return {kv_cache_manager::ER_INVALID_PARAMS, {}};
            }
            const auto done     = std::chrono::steady_clock::now();
            const auto wall_end = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
            std::fprintf(
                stderr,
                "KV_COALESCE_WRITE pid=%d begin_us=%lld end_us=%lld blocks=%zu original_iovs=%zu packed_iovs=%zu bytes=%zu copy_calls=%zu row_copy_calls=%zu lock_wait_us=%ld pack_us=%ld sdk_us=%ld ec=%d\n",
                getpid(),
                static_cast<long long>(wall_begin),
                static_cast<long long>(wall_end),
                uris.size(),
                iovs,
                uris.size(),
                total,
                copies,
                row_copies,
                static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(locked - entered).count()),
                static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(copied - begin).count()),
                static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(done - copied).count()),
                static_cast<int>(result.first));
            return result;
        } catch (...) {
            quarantine_ = op;
            failed_.store(true, std::memory_order_release);
            throw;
        }
    }

protected:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ER_SERVICE_UNSUPPORTED;
    }

private:
    struct Operation {
        size_t                     charged = 0;
        std::vector<unsigned char> bytes;
        Buffers                    buffers;
        ~Operation() {
            process_bytes_.fetch_sub(charged);
        }
    };
    inline static std::atomic<size_t>                 process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy                                              copy_;
    CopyRows                                          rows_;
    std::mutex                                        mutex_;
    std::atomic<bool>                                 failed_{false};
    std::shared_ptr<Operation>                        quarantine_;
};
}  // namespace rtp_llm::kvcm
