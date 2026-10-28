#pragma once

namespace rtp_llm::kvcm {

// Aggregate source-buffer safety independently of transfer success. Every
// worker must explicitly prove safety; legacy/missing responses are unknown.
class WorkerPayloadProof {
public:
    enum class Result { SUCCESS, BUFFER_SAFE_FAILURE, UNKNOWN };

    void observe(bool explicit_buffer_safe, bool success, bool failed) noexcept {
        all_safe_ = all_safe_ && explicit_buffer_safe && (success || failed);
        all_success_ = all_success_ && success;
    }

    Result result() const noexcept {
        return !all_safe_ ? Result::UNKNOWN :
               all_success_ ? Result::SUCCESS : Result::BUFFER_SAFE_FAILURE;
    }

private:
    bool all_safe_ = true;
    bool all_success_ = true;
};

}  // namespace rtp_llm::kvcm
