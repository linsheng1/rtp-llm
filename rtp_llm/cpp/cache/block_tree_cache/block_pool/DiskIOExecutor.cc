#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "autil/ThreadPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool DiskIOExecutorConfig::setQueueSizesFromBatchCapacity(size_t read_batches,
                                                          size_t write_batches,
                                                          size_t max_blocks_per_batch) {
    const size_t read_tasks = std::min(max_blocks_per_batch, read_thread_count);
    const size_t write_tasks = std::min(max_blocks_per_batch, write_thread_count);
    // autil::CircularQueue requires capacity < INT32_MAX.
    constexpr size_t max_queue_size = std::numeric_limits<int32_t>::max() - 1;
    if (read_batches == 0 || write_batches == 0 || read_tasks == 0 || write_tasks == 0
        || read_batches > max_queue_size / read_tasks || write_batches > max_queue_size / write_tasks) {
        return false;
    }
    read_queue_size  = read_batches * read_tasks;
    write_queue_size = write_batches * write_tasks;
    return true;
}

bool DiskIOExecutorConfig::valid() const {
    constexpr size_t max_int = std::numeric_limits<int32_t>::max();
    return read_thread_count > 0 && read_thread_count <= max_int && write_thread_count > 0
           && write_thread_count <= max_int && read_queue_size > 0 && read_queue_size < max_int && write_queue_size > 0
           && write_queue_size < max_int;
}

struct DiskIOExecutor::State {
    explicit State(DiskIOExecutorConfig config): config(config) {}

    void start() {
        std::lock_guard<std::mutex> lock(admission_mutex);
        if (stopped)
            throw std::logic_error("cannot restart a stopped disk executor");
        if (read_pool)
            return;
        if (!config.valid())
            throw std::invalid_argument("invalid disk executor configuration");
        auto reads  = std::make_unique<autil::ThreadPool>(config.read_thread_count, config.read_queue_size);
        auto writes = std::make_unique<autil::ThreadPool>(config.write_thread_count, config.write_queue_size);
        if (!reads->start("btc_disk_read") || !writes->start("btc_disk_write"))
            throw std::runtime_error("failed to start async disk executor");
        read_pool  = std::move(reads);
        write_pool = std::move(writes);
    }

    template<typename Request>
    struct Batch {
        Batch(std::vector<Request> requests, Callback callback):
            requests(std::move(requests)), callback(std::move(callback)) {}

        void run(DiskBlockIO& io) {
            DiskWriteTiming local_timing;
            auto* observed = timing ? &local_timing : nullptr;
            if (observed)
                observed->worker_start_us = DiskWriteTiming::nowUs();
            try {
                process(io, observed);
            } catch (const std::exception& e) {
                fail(DiskBlockIOStatus::IO_ERROR);
                RTP_LLM_LOG_ERROR("async disk worker threw: %s", e.what());
            } catch (...) {
                fail(DiskBlockIOStatus::IO_ERROR);
                RTP_LLM_LOG_ERROR("async disk worker threw an unknown exception");
            }
            if (observed) {
                observed->io_end_us = DiskWriteTiming::nowUs();
                std::lock_guard<std::mutex> lock(mutex);
                if (timing->worker_start_us == 0 || observed->worker_start_us < timing->worker_start_us)
                    timing->worker_start_us = observed->worker_start_us;
                timing->io_end_us = std::max(timing->io_end_us, observed->io_end_us);
                timing->pwrite_count += observed->pwrite_count;
                timing->pwrite_sum_us += observed->pwrite_sum_us;
                timing->pwrite_max_us = std::max(timing->pwrite_max_us, observed->pwrite_max_us);
            }
            // No worker touches caller-owned timing after giving up its participant.
            finish();
        }

        void process(DiskBlockIO& io, DiskWriteTiming* observed) {
            if constexpr (std::is_same_v<Request, DiskWrite>) {
                bool serial = false;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!write_prepared) {
                        // Prepare once on a worker, before anyone claims from this vector.
                        std::sort(requests.begin(), requests.end(), [](const DiskWrite& lhs, const DiskWrite& rhs) {
                            return lhs.offset < rhs.offset;
                        });
                        serial_writes = requests.size() <= 1;
                        for (size_t i = 0; i < requests.size(); ++i) {
                            const auto& request = requests[i];
                            serial_writes = serial_writes || !request.buffer || request.bytes == 0
                                            || request.bytes > std::numeric_limits<uint64_t>::max() - request.offset;
                            if (i != 0) {
                                const auto& previous = requests[i - 1];
                                serial_writes = serial_writes || previous.offset == request.offset
                                                || previous.bytes > request.offset - previous.offset;
                            }
                        }
                        if (timing)
                            timing->sort_end_us = DiskWriteTiming::nowUs();
                        write_prepared = true;
                    }
                    if (serial_writes) {
                        if (serial_claimed || status != DiskBlockIOStatus::OK)
                            return;
                        serial_claimed = true;
                        serial = true;
                    }
                }
                if (serial) {
                    // Preserve whole-batch ordering and backend validation on overlap/degenerate input.
                    const auto result = observed ? io.writeObserved(requests, observed) : io.write(requests);
                    if (result != DiskBlockIOStatus::OK)
                        fail(result);
                    return;
                }
                while (true) {
                    size_t index;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (status != DiskBlockIOStatus::OK || next == requests.size())
                            break;
                        index = next++;
                    }
                    const auto& request = requests[index];
                    const auto result = observed ?
                                            io.writeBlockObserved(request.offset, request.buffer, request.bytes, observed) :
                                            io.write(request.offset, request.buffer, request.bytes);
                    if (result != DiskBlockIOStatus::OK) {
                        fail(result);
                        break;
                    }
                }
            } else {
                (void)observed;  // Read ownership and work sharing retain their original path.
                while (true) {
                    size_t index;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (status != DiskBlockIOStatus::OK || next == requests.size()) {
                            break;
                        }
                        index = next++;
                    }
                    const auto& request = requests[index];
                    const auto  result  = io.read(request.offset, request.buffer, request.bytes);
                    if (result != DiskBlockIOStatus::OK) {
                        fail(result);
                        break;
                    }
                }
            }
        }

        void fail(DiskBlockIOStatus error) {
            std::lock_guard<std::mutex> lock(mutex);
            if (status == DiskBlockIOStatus::OK) {
                status = error;
            }
        }

        void finish() {
            DiskBlockIOStatus result;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--workers != 0) {
                    return;
                }
                result = status;
            }
            try {
                callback(result);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("async disk callback threw: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("async disk callback threw an unknown exception");
            }
        }

        DiskWriteTiming*     timing{nullptr};  // Writes only; never accessed after callback publication.
        std::vector<Request> requests;
        Callback             callback;
        std::mutex           mutex;
        size_t               next{0};
        size_t               workers{0};
        bool                 write_prepared{false};
        bool                 serial_writes{false};
        bool                 serial_claimed{false};
        DiskBlockIOStatus    status{DiskBlockIOStatus::OK};
    };

    template<typename Request>
    DiskBlockIOStatus submit(DiskBlockIO& io, std::vector<Request> requests, Callback callback,
                             DiskWriteTiming* timing = nullptr) {
        if (!callback) {
            RTP_LLM_LOG_WARNING("async disk submission rejected: empty callback");
            return DiskBlockIOStatus::IO_ERROR;
        }

        auto                        batch = std::make_shared<Batch<Request>>(std::move(requests), std::move(callback));
        size_t                      accepted = 0;
        std::lock_guard<std::mutex> admission_lock(admission_mutex);
        auto*                       pool = std::is_same_v<Request, DiskRead> ? read_pool.get() : write_pool.get();
        if (stopped || pool == nullptr) {
            RTP_LLM_LOG_WARNING("async disk submission rejected: executor is not running");
            return DiskBlockIOStatus::IO_ERROR;
        }
        // Register all participants before any worker can finish.
        std::lock_guard<std::mutex> batch_lock(batch->mutex);

        const size_t task_count = std::max(size_t{1}, std::min(batch->requests.size(), pool->getThreadNum()));
        batch->timing = timing;
        if (timing) {
            // Wall-clock fields describe this submission; syscall totals retain original accumulation.
            timing->worker_start_us = 0;
            timing->io_end_us = 0;
            timing->queued_us = DiskWriteTiming::nowUs();
        }
        for (size_t i = 0; i < task_count; ++i) {
            try {
                const auto error = pool->pushTask([batch, &io] { batch->run(io); }, false, false);
                if (error != autil::ThreadPool::ERROR_NONE) {
                    RTP_LLM_LOG_DEBUG("async disk enqueue rejected: error=%d, accepted=%zu", int(error), accepted);
                    break;
                }
            } catch (...) {
                // Earlier participants already own completion, including after allocation failure.
                RTP_LLM_LOG_ERROR("async disk enqueue threw, accepted=%zu", accepted);
                break;
            }
            ++batch->workers;
            ++accepted;
        }
        // Any accepted participant owns the whole batch, even after later enqueue failures.
        return accepted == 0 ? DiskBlockIOStatus::IO_ERROR : DiskBlockIOStatus::OK;
    }

    const DiskIOExecutorConfig         config;
    std::mutex                         admission_mutex;
    std::unique_ptr<autil::ThreadPool> read_pool;
    std::unique_ptr<autil::ThreadPool> write_pool;
    bool                               stopped{false};
};

DiskIOExecutor::DiskIOExecutor(DiskIOExecutorConfig config): state_(std::make_unique<State>(config)) {}

DiskIOExecutor::~DiskIOExecutor() {
    stop();
}

const DiskIOExecutorConfig& DiskIOExecutor::config() const {
    return state_->config;
}

void DiskIOExecutor::start() {
    state_->start();
}

DiskBlockIOStatus DiskIOExecutor::asyncRead(DiskBlockIO& io, std::vector<DiskRead> reads, Callback callback) {
    return state_->submit(io, std::move(reads), std::move(callback));
}

DiskBlockIOStatus DiskIOExecutor::asyncWrite(DiskBlockIO& io, std::vector<DiskWrite> writes, Callback callback,
                                            DiskWriteTiming* timing) {
    return state_->submit(io, std::move(writes), std::move(callback), timing);
}

void DiskIOExecutor::stop() {
    auto&                              state = *state_;
    std::unique_ptr<autil::ThreadPool> reads;
    std::unique_ptr<autil::ThreadPool> writes;
    {
        std::lock_guard<std::mutex> admission_lock(state.admission_mutex);
        state.stopped = true;
        reads         = std::move(state.read_pool);
        writes        = std::move(state.write_pool);
    }
    // Close admission in both directions before draining.
    if (reads)
        reads->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
    if (writes)
        writes->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
}

}  // namespace rtp_llm
