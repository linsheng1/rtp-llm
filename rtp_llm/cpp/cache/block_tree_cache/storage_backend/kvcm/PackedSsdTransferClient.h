#pragma once

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "kvcm_client/transfer_client.h"

namespace rtp_llm::kvcm {

// Explicit SSD compatibility candidate. One complete SDK IOV per URI, followed
// by ordered scatter to the original destinations. This uses CPU staging;
// the normal DRAM/GDR path must never select it implicitly.
class PackedSsdTransferClient final : public kv_cache_manager::TransferClient {
    using Code = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris = kv_cache_manager::UriStrVec;
    using Trace = std::shared_ptr<kv_cache_manager::TransferTraceInfo>;
public:
    using Copy = std::function<bool(void*, kv_cache_manager::MemoryType,
                                    const void*, kv_cache_manager::MemoryType, size_t)>;
    static constexpr size_t max_operation_bytes = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes = 512ULL * 1024 * 1024;

    PackedSsdTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> inner, Copy copy)
        : inner_(std::move(inner)), copy_(std::move(copy)) {
        if (!inner_ || !copy_) throw std::invalid_argument("missing SSD delegate/copy operation");
    }
    ~PackedSsdTransferClient() override {
        // The SDK delegate drains before staging memory (including failures)
        // is released. ClientWrapper already serializes shutdown against I/O.
        inner_.reset();
        quarantine_.reset();
    }

    Code LoadKvCaches(const Uris& uris, const Buffers& destination, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size()!=uris.size()) return kv_cache_manager::ER_INVALID_PARAMS;
        auto operation = makeOperation(uris, destination, false);
        if (!operation) return kv_cache_manager::ER_INVALID_LOCAL_BUFFERS;
        // A small Provider object cache cannot pin an entire client batch.
        // Release each completed URI before admitting the next one. This is
        // an explicit correctness path; its throughput needs separate testing.
        for (size_t block=0;block<uris.size();++block) {
            Code result;
            try { result = inner_->LoadKvCaches({uris[block]}, {operation->buffers[block]},
                                               blockTrace(trace,block,uris.size())); }
            catch (...) { quarantine_ = operation; throw; }
            if (result != kv_cache_manager::ER_OK) { quarantine_ = operation; return result; }
        }
        for (size_t block = 0; block < destination.size(); ++block) {
            size_t offset = 0;
            for (const auto& iov : destination[block].iovs) {
                if (!iov.ignore && !copy_(iov.base, iov.type, operation->bytes[block].data()+offset,
                                         kv_cache_manager::MemoryType::CPU, iov.size)) {
                    return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                }
                offset += iov.size;
            }
        }
        return kv_cache_manager::ER_OK;
    }

    std::pair<Code, Uris> SaveKvCaches(const Uris& uris, const Buffers& source, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size()!=uris.size()) return {kv_cache_manager::ER_INVALID_PARAMS,{}};
        auto operation = makeOperation(uris, source, true);
        if (!operation) return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
        for (size_t block = 0; block < source.size(); ++block) {
            size_t offset = 0;
            for (const auto& iov : source[block].iovs) {
                if (!copy_(operation->bytes[block].data()+offset, kv_cache_manager::MemoryType::CPU,
                           iov.base, iov.type, iov.size)) return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                offset += iov.size;
            }
        }
        auto actual=uris;
        for (size_t block=0;block<uris.size();++block) {
            try {
                auto result = inner_->SaveKvCaches({uris[block]}, {operation->buffers[block]},
                                                  blockTrace(trace,block,uris.size()));
                if (result.first != kv_cache_manager::ER_OK) {
                    quarantine_ = operation; return {result.first,{}};
                }
                if (!result.second.empty()) {
                    if (result.second.size()!=1) {quarantine_=operation;return {kv_cache_manager::ER_INVALID_PARAMS,{}};}
                    actual[block]=std::move(result.second.front());
                }
            } catch (...) { quarantine_ = operation; throw; }
        }
        return {kv_cache_manager::ER_OK,std::move(actual)};
    }

protected:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ER_SERVICE_UNSUPPORTED;
    }
private:
    static Trace blockTrace(const Trace& trace,size_t block,size_t count) {
        if (!trace) return {};
        auto result=std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
        if (trace->block_ids.size()==count) result->block_ids={trace->block_ids[block]};
        else return {};
        return result;
    }
    struct Operation {
        size_t charged = 0;
        std::vector<std::vector<unsigned char>> bytes;
        Buffers buffers;
        ~Operation() { process_bytes_.fetch_sub(charged); }
    };
    std::shared_ptr<Operation> makeOperation(const Uris& uris, const Buffers& original, bool write) {
        // Any uncertain completion fails this client closed until shutdown;
        // failed buffers cannot be overwritten by a later operation.
        if (quarantine_ || uris.empty() || uris.size()!=original.size()) return {};
        size_t total = 0;
        std::vector<size_t> sizes;
        for (const auto& block : original) {
            size_t size = 0;
            for (const auto& iov : block.iovs) {
                if (!iov.size || (write && iov.ignore) || (!iov.ignore && !iov.base)
                    || (iov.type!=kv_cache_manager::MemoryType::CPU && iov.type!=kv_cache_manager::MemoryType::GPU)
                    || iov.size>max_operation_bytes-size) return {};
                size += iov.size;
            }
            if (!size || size>max_operation_bytes-total) return {};
            total += size; sizes.push_back(size);
        }
        auto operation = std::make_shared<Operation>();
        size_t previous = process_bytes_.load();
        do { if (previous>max_process_bytes-total) return {}; }
        while (!process_bytes_.compare_exchange_weak(previous,previous+total));
        operation->charged=total;
        operation->bytes.resize(sizes.size()); operation->buffers.resize(sizes.size());
        for (size_t i=0;i<sizes.size();++i) {
            operation->bytes[i].resize(sizes[i]);
            operation->buffers[i].iovs.push_back({kv_cache_manager::MemoryType::CPU,
                                                 operation->bytes[i].data(),sizes[i],false});
        }
        return operation;
    }
    inline static std::atomic<size_t> process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy copy_;
    std::mutex mutex_;
    std::shared_ptr<Operation> quarantine_;
};
} // namespace rtp_llm::kvcm
