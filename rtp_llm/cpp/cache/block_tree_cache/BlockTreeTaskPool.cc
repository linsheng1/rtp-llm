#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

#include <cassert>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <limits>

#include "autil/LambdaWorkItem.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

BlockTreeTaskPool::BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name,
                                     size_t completion_threads, size_t completion_capacity):
    thread_count_(thread_count), queue_size_(queue_size), completion_threads_(completion_threads),
    completion_capacity_(completion_capacity), thread_name_(std::move(thread_name)) {
    if (completion_threads > std::numeric_limits<size_t>::max() - thread_count
        || (completion_threads && !completion_capacity)) throw std::invalid_argument("invalid completion pool configuration");
}

BlockTreeTaskPool::~BlockTreeTaskPool() {
    shutdown();
}

bool BlockTreeTaskPool::start() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (started_ || shutdown_ || thread_count_ + completion_threads_ == 0) {
        return false;
    }

    auto thread_pool =
        std::make_shared<autil::LockFreeThreadPool>(thread_count_ + completion_threads_, thread_count_ + completion_threads_, nullptr, thread_name_.c_str());
    if (!thread_pool->start()) {
        return false;
    }
    thread_pool_ = thread_pool;
    started_     = true;
    for (size_t index = 0; index < thread_count_ + completion_threads_; ++index) {
        auto*                               work_item = new autil::LambdaWorkItem([this, index] { workerLoop(index >= thread_count_); });
        const autil::ThreadPool::ERROR_TYPE error     = thread_pool->pushWorkItem(work_item, false);
        if (error != autil::ThreadPool::ERROR_NONE) {
            work_item->destroy();
            shutdown_ = true;
            queue_cv_.notify_all();
            lock.unlock();
            thread_pool->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
            thread_pool->join();
            return false;
        }
    }
    return true;
}

bool BlockTreeTaskPool::submit(BlockTreeTaskClass               task_class,
                               std::function<void()>            task,
                               std::optional<Clock::time_point> deadline,
                               std::function<void()>            on_timeout) {
    if (!task || deadline.has_value() != static_cast<bool>(on_timeout)) {
        return false;
    }

    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || admission_stopped_ || shutdown_) {
        return false;
    }
    if (deadline.has_value() && Clock::now() >= *deadline) {
        lock.unlock();
        try {
            on_timeout();
        } catch (...) {}
        return true;
    }
    if (queue_size_ != 0) {
        if (normalQueueSizeLocked() >= queue_size_) {
            return false;
        }
        if (task_class == BlockTreeTaskClass::BACKGROUND && background_queue_.size() >= backgroundLimit()) {
            return false;
        }
    }

    auto& queue = task_class == BlockTreeTaskClass::LOAD ? load_queue_ : background_queue_;
    queue.push_back(QueuedTask{std::move(task), std::move(on_timeout), std::move(deadline)});
    taskStarted();
    // Completion-only workers share this CV but cannot consume normal work.
    // Wake the business workers as well; notify_one could strand this task.
    queue_cv_.notify_all();
    return true;
}

bool BlockTreeTaskPool::submitCompletion(std::function<void()> task) {
    if (!task) {
        return false;
    }
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || shutdown_) {
        return false;
    }
    completion_queue_.push_back(std::move(task));
    taskStarted();
    queue_cv_.notify_one();
    return true;
}

void BlockTreeTaskPool::stopAdmission() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    admission_stopped_ = true;
}

size_t BlockTreeTaskPool::backgroundLimit() const {
    return queue_size_ > kLoadReservedSlots ? queue_size_ - kLoadReservedSlots : queue_size_;
}

BlockTreeQueueSizes BlockTreeTaskPool::queueSizes() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return {load_queue_.size(), background_queue_.size(), completion_queue_.size() + completion_load_ + completion_background_};
}

size_t BlockTreeTaskPool::normalQueueSizeLocked() const {
    return load_queue_.size() + background_queue_.size();
}

BlockTreeTaskPool::QueuedTask BlockTreeTaskPool::popNextNormalTaskLocked() {
    const bool take_background =
        !background_queue_.empty() && (load_queue_.empty() || consecutive_load_dispatches_ >= kMaxLoadBurst);
    if (take_background) {
        QueuedTask task = std::move(background_queue_.front());
        background_queue_.pop_front();
        consecutive_load_dispatches_ = 0;
        return task;
    }

    QueuedTask task = std::move(load_queue_.front());
    load_queue_.pop_front();
    consecutive_load_dispatches_ = std::min(consecutive_load_dispatches_ + 1, kMaxLoadBurst);
    return task;
}

std::shared_ptr<BlockTreeTaskPool::CompletionTicket>
BlockTreeTaskPool::reserveCompletion(std::function<void()> task, BlockTreeTaskClass task_class) {
    if (!task) return {};
    auto node = std::make_shared<CompletionNode>(std::move(task), task_class == BlockTreeTaskClass::LOAD);
    auto ticket = std::shared_ptr<CompletionTicket>(new CompletionTicket(shared_from_this(), node));
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    // Keep LOAD admission headroom as well as dispatch priority. For tiny
    // pools retain at least one BACKGROUND slot; capacity one cannot be split.
    const size_t limit = node->load ? completion_capacity_ :
        completion_capacity_ - std::min(kLoadReservedSlots, completion_capacity_ / 2);
    if (!started_ || shutdown_ || completion_admission_stopped_ || completion_outstanding_ >= limit) {
        node->phase.store(CompletionPhase::ABANDONED); ++completion_rejected_; return {};
    }
    node->phase.store(CompletionPhase::RESERVED);
    ++completion_outstanding_;
    return ticket;
}
bool BlockTreeTaskPool::postCompletion(const std::shared_ptr<CompletionNode>& node) noexcept {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (node->phase.load() != CompletionPhase::RESERVED) return false;
    node->phase.store(CompletionPhase::QUEUED);
    auto& head = node->load ? completed_load_ : completed_background_;
    auto*& tail = node->load ? completed_load_tail_ : completed_background_tail_;
    if (tail) tail->next = node; else head = node;
    tail = node.get();
    ++(node->load ? completion_load_ : completion_background_);
    taskStarted(); queue_cv_.notify_all();
    return true;
}
void BlockTreeTaskPool::abandonCompletion(const std::shared_ptr<CompletionNode>& node) noexcept {
    std::function<void()> retired;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (node->phase.load() != CompletionPhase::RESERVED) return;
        node->phase.store(CompletionPhase::ABANDONED);
        retired.swap(node->task);
    }
    retired = {}; // Release owners outside the queue lock, before returning capacity.
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    --completion_outstanding_; completion_idle_.notify_all();
}
void BlockTreeTaskPool::stopCompletionAdmission() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_); completion_admission_stopped_ = true;
}
void BlockTreeTaskPool::waitForCompletionIdle() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    completion_idle_.wait(lock, [this] { return completion_outstanding_ == 0; });
}
BlockTreeTaskPool::CompletionSnapshot BlockTreeTaskPool::completionSnapshot() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return {completion_capacity_, completion_outstanding_, completion_load_, completion_background_, completion_running_,
            completion_rejected_, started_ && !shutdown_ && !completion_admission_stopped_};
}
void BlockTreeTaskPool::workerLoop(bool completion_only) {
    for (;;) {
        std::function<void()> task;
        std::shared_ptr<CompletionNode> reserved;
        {
            std::unique_lock<std::mutex> lock(lifecycle_mutex_);
            const auto can_reserve = [this, completion_only] {
                return (completion_only || !completion_threads_) && (completed_load_ || completed_background_);
            };
            queue_cv_.wait(lock, [&] {
                return shutdown_ || can_reserve() || !completion_queue_.empty()
                    || (!completion_only && normalQueueSizeLocked());
            });
            if (can_reserve()) {
                const bool load = completed_load_ && (!completed_background_ || completion_burst_ < kMaxLoadBurst);
                auto& head = load ? completed_load_ : completed_background_;
                auto*& tail = load ? completed_load_tail_ : completed_background_tail_;
                reserved = std::move(head); head = std::move(reserved->next); if (!head) tail = nullptr;
                --(load ? completion_load_ : completion_background_);
                completion_burst_ = load ? std::min(completion_burst_ + 1, kMaxLoadBurst) : 0;
                reserved->phase.store(CompletionPhase::RUNNING); ++completion_running_;
                task.swap(reserved->task);
            } else if (!completion_queue_.empty()) {
                task = std::move(completion_queue_.front()); completion_queue_.pop_front();
            } else if (!completion_only && normalQueueSizeLocked()) {
                auto queued = popNextNormalTaskLocked();
                task = queued.deadline && Clock::now() >= *queued.deadline ? std::move(queued.on_timeout) : std::move(queued.run);
            } else if (shutdown_) return;
        }
        try { task(); } catch (...) {}
        task = {};
        taskFinished();
        if (reserved) {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            reserved->phase.store(CompletionPhase::DONE);
            --completion_running_; --completion_outstanding_; completion_idle_.notify_all();
        }
    }
}

void BlockTreeTaskPool::waitForIdle() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    bool                         wait_observer_invoked = false;
    wait_cv_.wait(lock, [this, &wait_observer_invoked] {
        const int pending_tasks = pending_tasks_.load();
        if (pending_tasks > 0 && !wait_observer_invoked) {
            wait_observer_invoked = true;
            const auto observer   = pending_task_wait_observer_for_test_;
            if (observer) {
                observer();
            }
        }
        return pending_tasks <= 0;
    });
}

void BlockTreeTaskPool::shutdown() {
    std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
    if (completion_capacity_) {
        stopAdmission();
        waitForIdle(); // Accepted queued business operations may still reserve.
        stopCompletionAdmission();
        waitForCompletionIdle();
        waitForIdle();
    }
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool;
    bool                                       was_started   = false;
    size_t                                     dropped_tasks = 0;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (shutdown_) {
            return;
        }
        admission_stopped_ = true;
        shutdown_          = true;
        dropped_tasks      = normalQueueSizeLocked() + completion_queue_.size();
        load_queue_.clear();
        background_queue_.clear();
        completion_queue_.clear();
        thread_pool = thread_pool_;
        was_started = started_;
        queue_cv_.notify_all();
    }

    if (dropped_tasks > 0) {
        const int dropped   = static_cast<int>(dropped_tasks);
        const int remaining = pending_tasks_.fetch_sub(dropped) - dropped;
        if (remaining <= 0) {
            std::lock_guard<std::mutex> lock(wait_mutex_);
            wait_cv_.notify_all();
        }
    }

    if (thread_pool != nullptr && was_started) {
        thread_pool->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        thread_pool->join();
    }
}

void BlockTreeTaskPool::taskStarted() {
    pending_tasks_.fetch_add(1);
}

void BlockTreeTaskPool::taskFinished() {
    const int remaining = pending_tasks_.fetch_sub(1) - 1;
    if (remaining <= 0) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

}  // namespace rtp_llm
