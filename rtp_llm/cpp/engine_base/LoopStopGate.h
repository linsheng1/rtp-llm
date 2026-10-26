#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rtp_llm {

// CPU control-plane fence. No CUDA operation, Python callback or distributed
// collective is added to a normal step. A rank may finish an already-entered
// step; shutdown RPCs advance lagging ranks to the largest observed boundary.
class LoopStopGate {
public:
    struct State {
        uint64_t step = 0;
        bool parked = false;
        bool supported = false;
    };

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = true;
        started_ = true;
    }

    bool enter() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !running_ || !fenced_ || step_ < target_; });
        if (!running_) {
            return false;
        }
        entered_ = true;
        return true;
    }

    void complete() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++step_;
        entered_ = false;
    }

    State fence(uint64_t target) {
        std::lock_guard<std::mutex> lock(mutex_);
        fenced_ = true;
        target_ = std::max(target_, target);
        condition_.notify_all();
        return stateLocked();
    }

    State state() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stateLocked();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        condition_.notify_all();
    }

private:
    State stateLocked() const {
        // Include an entered step in the required boundary even if its GPU
        // collective is blocked waiting for another rank to catch up.
        return {step_ + (entered_ ? 1 : 0), started_ && fenced_ && !entered_ && step_ >= target_, started_};
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    uint64_t step_ = 0;
    uint64_t target_ = 0;
    bool running_ = false;
    bool started_ = false;
    bool entered_ = false;
    bool fenced_ = false;
};

}  // namespace rtp_llm
