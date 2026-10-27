#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace rtp_llm {

struct RemoteRestoreTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point read_started_at;
    Clock::time_point gpu_ready_at;
};

// Stream-lock-owned metadata only. No payload, allocation, timer or device sync.
// A backend read is not actual use. Preserve its clock through adoption until
// the existing successful context forward completes, once per logical request.
class RemoteRestoreLifecycle {
public:
    using Clock = RemoteRestoreTiming::Clock;

    bool adopt(const RemoteRestoreTiming& timing) {
        reset();
        if (timing.gpu_ready_at < timing.read_started_at) {
            return false;
        }
        timing_ = timing;
        return true;
    }

    bool pendingForward() const {
        return timing_.has_value() && !forward_reported_;
    }

    std::optional<int64_t> checkpointReadyUs() const {
        if (!timing_) {
            return std::nullopt;
        }
        return elapsed(timing_->gpu_ready_at);
    }

    std::optional<int64_t> completeContextForward(Clock::time_point at = Clock::now()) {
        if (!pendingForward() || at < timing_->gpu_ready_at) {
            return std::nullopt;
        }
        forward_reported_ = true;
        return elapsed(at);
    }

    void reset() {
        timing_.reset();
        forward_reported_ = false;
    }

private:
    int64_t elapsed(Clock::time_point at) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(at - timing_->read_started_at).count();
    }
    std::optional<RemoteRestoreTiming> timing_;
    bool                               forward_reported_{false};
};

}  // namespace rtp_llm
