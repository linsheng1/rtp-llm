#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Only a local preflight which has not entered gRPC may give this proof.
class BroadcastRpcNotStarted: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline bool isGrpcConnectFailure(const grpc::Status& status) {
    if (status.error_code() != grpc::StatusCode::UNAVAILABLE) {
        return false;
    }
    auto message = status.error_message();
    std::transform(message.begin(), message.end(), message.begin(), [](unsigned char c) { return std::tolower(c); });
    return message.find("connect failed") != std::string::npos
           || message.find("failed to connect to all addresses") != std::string::npos
           || message.find("failed to connect to remote host") != std::string::npos;
}

template<typename RequestPB, typename ResponsePB>
class BroadcastResult: public std::enable_shared_from_this<BroadcastResult<RequestPB, ResponsePB>> {
public:
    using DoneCallback = std::function<void()>;

    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>    stub;
        std::shared_ptr<grpc::ClientContext> client_context;
        RequestPB                            request;
        ResponsePB                           response;
        grpc::Status                         status;
        std::string                          server_addr;
        int                                  timeout_ms{0};
    };

public:
    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts,
                             std::optional<int64_t>                                request_id = std::nullopt,
                             std::string                                           direction  = "unknown",
                             std::function<void(const std::string&)>               connect_failure_reporter = {}):
        worker_contexts_(worker_rpc_contexts),
        finished_(worker_rpc_contexts.size(), false),
        rpc_succeeded_(worker_rpc_contexts.size(), false),
        completion_times_us_(worker_rpc_contexts.size(), 0),
        completion_wall_times_us_(worker_rpc_contexts.size(), 0),
        direction_(std::move(direction)),
        rpc_tag_("request_id=" + (request_id ? std::to_string(*request_id) : "none") + " direction=" + direction_),
        connect_failure_reporter_(std::move(connect_failure_reporter)) {}
    ~BroadcastResult() = default;

public:
    // DSV4.1 payloads reuse this result's terminal flags, lock, wait and onDone.
    // No driver queue/thread or second parent result owns the same operation.
    using Clock = std::chrono::steady_clock;
    using Task = DoneCallback;
    enum class Proof { NOT_STARTED, BUFFER_SAFE, UNKNOWN };
    enum class Result { SUCCESS, FAILED, TIMEOUT, CANCELED, UNKNOWN };
    struct Outcome {
        Result result{Result::FAILED}; Proof proof{Proof::NOT_STARTED};
        static Outcome success() noexcept { return {Result::SUCCESS, Proof::BUFFER_SAFE}; }
        static Outcome notStarted(Result why = Result::FAILED) noexcept { return {why, Proof::NOT_STARTED}; }
        static Outcome unknown() noexcept { return {Result::UNKNOWN, Proof::UNKNOWN}; }
    };
    struct Snapshot { size_t children, remaining, active, peak, submitting; bool submissions_closed, finished, canceled; };
    using Launch = std::function<bool(size_t, const std::shared_ptr<BroadcastResult>&)>;
    static std::shared_ptr<BroadcastResult> createPayloadWindow(size_t count, size_t window, Clock::time_point deadline,
                                                              Launch launch, Task finalize, Task notify) {
        if (!window || !launch || !finalize || !notify) throw std::invalid_argument("invalid payload window");
        // All result/callback storage is prepared before any RPC/SDK handoff.
        std::vector<std::shared_ptr<WorkerRpcContext>> contexts(count);
        auto result = std::make_shared<BroadcastResult>(contexts);
        result->payload_enabled_ = true; result->payload_slots_.resize(count); result->payload_window_ = window; result->payload_deadline_ = deadline;
        result->payload_launch_ = std::move(launch); result->payload_finalize_ = std::move(finalize);
        result->onDone(std::move(notify));
        return result;
    }
    void start() noexcept {
        { std::lock_guard<std::mutex> lock(wait_done_mutex_); if (!payload_enabled_ || payload_started_) return; payload_started_ = true; }
        drivePayload();
    }
    void cancel() noexcept {
        { std::lock_guard<std::mutex> lock(wait_done_mutex_); payload_canceled_ = true; closePayloadTail(Result::CANCELED); }
        finishPayload();
    }
    bool tryStart(size_t rank) noexcept {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        if (rank >= payload_slots_.size() || finished_[rank] || payload_slots_[rank].claimed
            || payload_canceled_ || Clock::now() >= payload_deadline_) return false;
        payload_slots_[rank].claimed = true; return true;
    }
    bool markMayTouch(size_t rank) noexcept {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        if (rank >= payload_slots_.size() || finished_[rank] || !payload_slots_[rank].claimed) return false;
        payload_slots_[rank].may_touch = true; return true;
    }
    bool complete(size_t rank, Outcome outcome) noexcept {
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (rank >= payload_slots_.size() || finished_[rank] || !payload_slots_[rank].submitted) return false;
            if (outcome.proof == Proof::NOT_STARTED && payload_slots_[rank].may_touch) outcome = Outcome::unknown();
            payload_slots_[rank].outcome = outcome; finished_[rank] = true;
            ++finished_count_; --payload_active_;
            if (outcome.result != Result::SUCCESS) { payload_canceled_ = true; closePayloadTail(outcome.result == Result::TIMEOUT ? Result::TIMEOUT : Result::CANCELED); }
        }
        drivePayload(); finishPayload(); return true;
    }
    Outcome outcome(size_t rank) const {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        if (rank >= payload_slots_.size() || !finished_[rank]) throw std::logic_error("payload rank not terminal");
        return payload_slots_[rank].outcome;
    }
    Snapshot snapshot() const noexcept {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        return {payload_slots_.size(), payload_slots_.size() - static_cast<size_t>(finished_count_), payload_active_,
                payload_peak_, payload_submitting_, payload_closed_, already_done_.load(), payload_canceled_};
    }
    Clock::time_point deadline() const noexcept { return payload_deadline_; }
public:
    bool done() const {
        return already_done_.load(std::memory_order_acquire);
    }

    bool waitDone(int timeout_ms, bool fatal_on_error = true) {
        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        const auto completed = [this] { return already_done_.load(std::memory_order_acquire); };
        if (timeout_ms > 0) {
            if (!wait_done_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), completed)) {
                return false;
            }
        } else {
            wait_done_cv_.wait(lock, completed);
        }
        const std::string fatal_error = fatal_on_error && !fatal_error_.empty() ? failureSummaryNoLock() : std::string{};
        lock.unlock();
        if (!fatal_error.empty()) {
            RTP_LLM_FAIL("%s", fatal_error.c_str());
        }
        return true;
    }

    void waitDone() {
        (void)waitDone(/*timeout_ms=*/0);
    }

    bool success() const {
        return all_request_success_.load(std::memory_order_acquire);
    }

    bool rpcSucceeded(size_t rank) const {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        return rank < rpc_succeeded_.size() && rpc_succeeded_[rank];
    }

    bool deadlineExceeded() const {
        return deadline_exceeded_.load(std::memory_order_acquire);
    }

    std::string failureSummary() const {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        return failureSummaryNoLock();
    }

    void onDone(DoneCallback callback) {
        if (!callback) {
            return;
        }
        bool run_now = false;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load(std::memory_order_acquire)) {
                run_now = true;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback();
        }
    }

    std::vector<ResponsePB> responses() const {
        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        std::vector<ResponsePB>      responses;
        responses.reserve(worker_contexts_.size());
        for (const auto& worker_rpc_context : worker_contexts_) {
            responses.push_back(worker_rpc_context->response);
        }
        return responses;
    }

    // Coordinator-local monotonic timestamps; no cross-host clock subtraction.
    std::vector<int64_t> completionTimesUs() const {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        return completion_times_us_;
    }

    std::vector<int64_t> completionWallTimesUs() const {
        std::lock_guard<std::mutex> lock(wait_done_mutex_);
        return completion_wall_times_us_;
    }

    void finishRank(size_t rank, bool cq_event_ok) {
        std::vector<DoneCallback> callbacks;
        bool                      report_connect_failure = false;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load(std::memory_order_relaxed)) {
                return;
            }
            if (rank >= worker_contexts_.size()) {
                fatal_error_ = "broadcast rpc cq tag rank out of range: rank=" + std::to_string(rank) + " " + rpc_tag_;
                finishLocked(callbacks);
            } else if (finished_[rank]) {
                return;
            } else {
                finished_[rank]                 = true;
                completion_times_us_[rank]      = std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now().time_since_epoch())
                                                      .count();
                completion_wall_times_us_[rank] = std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::system_clock::now().time_since_epoch())
                                                      .count();
                ++finished_count_;
                const auto& ctx      = worker_contexts_[rank];
                rpc_succeeded_[rank] = cq_event_ok && ctx->status.ok();
                if (!rpc_succeeded_[rank]) {
                    const auto summary = rpc_tag_ + " target_rank=" + std::to_string(rank) + " addr=" + ctx->server_addr
                                         + " grpc_code=" + std::to_string(ctx->status.error_code())
                                         + " error=" + ctx->status.error_message();
                    if (rpc_failure_summary_.empty()) {
                        rpc_failure_summary_ = summary;
                    }
                    report_connect_failure = cq_event_ok && isGrpcConnectFailure(ctx->status);
                    if (report_connect_failure && connect_failure_summary_.empty()) {
                        connect_failure_summary_ = summary;
                    }
                }
                if (!cq_event_ok) {
                    // A deadline reported by any rank is the aggregate error.
                    // Do not let a later CQ failure replace its message while
                    // deadlineExceeded() continues to report true.
                    if (!deadline_exceeded_.load(std::memory_order_acquire) && fatal_error_.empty()) {
                        fatal_error_ = "broadcast rpc cq event failed, rank=" + std::to_string(rank)
                                       + " addr=" + ctx->server_addr + " " + rpc_tag_;
                    }
                } else if (ctx->status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    if (!deadline_exceeded_.exchange(true, std::memory_order_acq_rel)) {
                        fatal_error_ = "broadcast rpc timeout, timeout_ms=" + std::to_string(ctx->timeout_ms) + " rank="
                                       + std::to_string(rank) + " err=" + std::to_string(ctx->status.error_code()) + "("
                                       + ctx->status.error_message() + ") addr=" + ctx->server_addr + " " + rpc_tag_;
                    }
                } else if (!ctx->status.ok()) {
                    RTP_LLM_LOG_WARNING("broadcast rpc failed, rank=%zu err=%d(%s) addr=%s %s",
                                        rank,
                                        ctx->status.error_code(),
                                        ctx->status.error_message().c_str(),
                                        ctx->server_addr.c_str(),
                                        rpc_tag_.c_str());
                    grpc_status_failure_seen_ = true;
                }
                if (finished_count_ == static_cast<int>(worker_contexts_.size())) {
                    finishLocked(callbacks);
                }
            }
        }
        if (report_connect_failure && connect_failure_reporter_) {
            try {
                connect_failure_reporter_(direction_);
            } catch (...) {
                RTP_LLM_LOG_WARNING("connect failure reporter threw: %s", rpc_tag_.c_str());
            }
        }
        wait_done_cv_.notify_all();
        for (auto& callback : callbacks) {
            callback();
        }
    }

    // CQ is terminal even if allocating a diagnostic string failed. Preserve
    // the original callback obligation with a conservative aggregate failure.
    void finishRankWithoutSummary(size_t rank) noexcept {
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load() || payload_enabled_) return;
            grpc_status_failure_seen_ = true;
            if (rank < finished_.size() && !finished_[rank]) { finished_[rank] = true; ++finished_count_; }
            if (rank < rpc_succeeded_.size()) rpc_succeeded_[rank] = false;
            if (finished_count_ == static_cast<int>(finished_.size())) finishLocked(callbacks);
        }
        wait_done_cv_.notify_all();
        for (auto& callback : callbacks) { try { callback(); } catch (...) {} }
    }

private:
    friend class BroadcastManager;

    struct PayloadSlot { bool submitted{false}, claimed{false}, may_touch{false}; Outcome outcome; };
    void closePayloadTail(Result reason) noexcept {
        for (size_t rank = payload_next_; rank < payload_slots_.size(); ++rank) {
            if (!payload_slots_[rank].submitted && !finished_[rank]) {
                finished_[rank] = true; ++finished_count_; payload_slots_[rank].outcome = Outcome::notStarted(reason);
            }
        }
        payload_next_ = payload_slots_.size(); payload_closed_ = true;
    }
    void finishPayload() noexcept {
        Task finalize;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load() || payload_finalizing_ || !payload_started_ || !payload_closed_
                || payload_active_ || payload_submitting_ || payload_driving_) return;
            payload_finalizing_ = true; finalize = std::move(payload_finalize_);
        }
        // Normalize NOT_STARTED/UNKNOWN wire responses BEFORE done/waiters.
        // Production finalization catches errors and marks aggregate UNKNOWN.
        try { finalize(); } catch (...) { std::terminate(); }
        std::vector<DoneCallback> callbacks;
        Launch retired_launch;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            grpc_status_failure_seen_ = std::any_of(payload_slots_.begin(), payload_slots_.end(),
                [](const auto& slot) { return slot.outcome.result != Result::SUCCESS || slot.outcome.proof != Proof::BUFFER_SAFE; });
            retired_launch = std::move(payload_launch_);
            finishLocked(callbacks);
        }
        wait_done_cv_.notify_all();
        for (auto& callback : callbacks) { try { callback(); } catch (...) {} }
    }
    void drivePayload() noexcept {
        const auto self = this->shared_from_this();
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (!payload_started_ || already_done_.load() || payload_driving_ || payload_finalizing_) return;
            payload_driving_ = true;
        }
        for (;;) {
            size_t rank = 0;
            bool launch = false;
            {
                std::lock_guard<std::mutex> lock(wait_done_mutex_);
                if (Clock::now() >= payload_deadline_) { payload_canceled_ = true; closePayloadTail(Result::TIMEOUT); }
                if (!payload_closed_ && payload_next_ < payload_slots_.size() && payload_active_ < payload_window_) {
                    rank = payload_next_++; payload_slots_[rank].submitted = true;
                    ++payload_active_; ++payload_submitting_; payload_peak_ = std::max(payload_peak_, payload_active_);
                    launch = true;
                    if (payload_next_ == payload_slots_.size()) payload_closed_ = true;
                } else {
                    if (payload_next_ == payload_slots_.size()) payload_closed_ = true;
                    payload_driving_ = false;
                }
            }
            if (!launch) { finishPayload(); return; }
            bool accepted = false;
            try { accepted = payload_launch_(rank, self); } catch (...) {}
            if (!accepted) complete(rank, Outcome::notStarted());
            {
                std::lock_guard<std::mutex> lock(wait_done_mutex_);
                --payload_submitting_;
            }
        }
    }
    std::vector<PayloadSlot> payload_slots_;
    Launch payload_launch_;
    Task payload_finalize_;
    Clock::time_point payload_deadline_{};
    size_t payload_window_{0}, payload_next_{0}, payload_active_{0}, payload_peak_{0}, payload_submitting_{0};
    bool payload_enabled_{false}, payload_started_{false}, payload_closed_{false}, payload_canceled_{false}, payload_driving_{false}, payload_finalizing_{false};

    std::string failureSummaryNoLock() const {
        const auto& rpc_error = connect_failure_summary_.empty() ? rpc_failure_summary_ : connect_failure_summary_;
        return fatal_error_.empty() ? rpc_error : fatal_error_ + (rpc_error.empty() ? "" : "; " + rpc_error);
    }

    void finishLocked(std::vector<DoneCallback>& callbacks) {
        finished_count_ = static_cast<int>(worker_contexts_.size());
        all_request_success_.store(fatal_error_.empty() && !grpc_status_failure_seen_, std::memory_order_release);
        already_done_.store(true, std::memory_order_release);
        callbacks.swap(callbacks_);
    }

    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::vector<bool>                              finished_;
    std::vector<bool>                              rpc_succeeded_;
    std::vector<int64_t>                           completion_times_us_;
    std::vector<int64_t>                           completion_wall_times_us_;
    const std::string                              direction_;
    const std::string                              rpc_tag_;
    const std::function<void(const std::string&)>  connect_failure_reporter_;
    std::string                                    rpc_failure_summary_;
    std::string                                    connect_failure_summary_;
    int                                            finished_count_{0};
    std::atomic<bool>                              already_done_{false};
    std::atomic<bool>                              all_request_success_{false};
    std::atomic<bool>                              deadline_exceeded_{false};
    bool                                           grpc_status_failure_seen_{false};
    std::string                                    fatal_error_;
    mutable std::mutex                             wait_done_mutex_;
    std::condition_variable                        wait_done_cv_;
    std::vector<DoneCallback>                      callbacks_;
};

class BroadcastManager {
public:
    explicit BroadcastManager(const std::vector<std::string>&         worker_addrs,
                              std::function<void(const std::string&)> connect_failure_reporter = {}):
        worker_addrs_(worker_addrs), connect_failure_reporter_(std::move(connect_failure_reporter)) {}
    ~BroadcastManager() {
        if (completion_queue_) {
            completion_queue_->Shutdown();
        }
        if (completion_thread_.joinable()) {
            completion_thread_.join();
        }
        rpc_pool_.reset();
    }

public:
    bool init() {
        if (worker_addrs_.empty()) {
            RTP_LLM_LOG_WARNING("init failed, worker_addrs is empty");
            return false;
        }

        rpc_pool_ = std::make_shared<RPCPool>();
        completion_queue_ = std::make_unique<grpc::CompletionQueue>();
        completion_thread_ = std::thread([this] { pollCompletions(); });
        return true;
    }

    template<typename RequestPB, typename ResponsePB, typename RpcCall>
    std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>> broadcast(const std::vector<RequestPB>& requests,
                                                                      int                           timeout_ms,
                                                                      const RpcCall&                rpc_call,
                                                                      std::optional<int64_t> request_id = std::nullopt,
                                                                      std::string direction = "unknown",
                                                                      bool fatal_on_error = true) const {
        const auto rpc_tag =
            "request_id=" + (request_id ? std::to_string(*request_id) : "none") + " direction=" + direction;
        const auto worker_size = worker_addrs_.size();
        if (requests.size() != worker_size) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
            return nullptr;
        }
        if (!rpc_pool_ || !completion_queue_) {
            RTP_LLM_LOG_WARNING("broadcast failed, manager is not initialized");
            return nullptr;
        }

        using CtxT = typename BroadcastResult<RequestPB, ResponsePB>::WorkerRpcContext;
        using ResultT = BroadcastResult<RequestPB, ResponsePB>;
        std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
        const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

        for (int rank = 0; rank < worker_size; ++rank) {
            const auto& addr        = worker_addrs_[rank];
            auto        conn_status = rpc_pool_->getConnection(addr);
            if (!conn_status.ok()) {
                RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%d addr=%s error=%s %s",
                                    rank,
                                    addr.c_str(),
                                    conn_status.status().ToString().c_str(),
                                    rpc_tag.c_str());
                return nullptr;
            }

            contexts[rank]      = std::make_shared<CtxT>();
            auto& ctx           = contexts.at(rank);
            ctx->stub           = conn_status.value().stub;
            ctx->request        = requests.at(rank);
            ctx->server_addr    = addr;
            ctx->timeout_ms     = timeout_ms;
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        auto result = std::make_shared<ResultT>(contexts, request_id, std::move(direction), connect_failure_reporter_);
        std::vector<std::shared_ptr<CompletionTag<ResultT>>> tags;
        tags.reserve(worker_size);
        for (size_t rank = 0; rank < worker_size; ++rank) tags.push_back(std::make_shared<CompletionTag<ResultT>>(result, rank));
        for (int rank = 0; rank < worker_size; ++rank) {
            auto& ctx = result->worker_contexts_.at(rank);
            try {
                // Prepared ownership exists before gRPC may retain a raw tag/context.
                // Ambiguous throws keep this anchor until a real CQ completion.
                tags[rank]->anchor = tags[rank];
                auto reader = [&] {
                    try { return rpc_call(ctx->stub, ctx->client_context, ctx->request, completion_queue_.get()); }
                    catch (const BroadcastRpcNotStarted&) {
                        tags[rank]->anchor.reset(); // No Finish/raw tag handoff and no CQ can arrive.
                        throw;
                    }
                }();
                if (!reader) {
                    throw std::runtime_error("broadcast rpc call returned null reader");
                }
                reader->Finish(&ctx->response, &ctx->status, tags[rank].get());
            } catch (const std::exception& error) {
                if (fatal_on_error) {
                    RTP_LLM_FAIL("broadcast rpc dispatch failed, rank=%d err=%s %s", rank, error.what(), rpc_tag.c_str());
                }
                for (size_t pending = rank; pending < worker_size; ++pending) {
                    try { result->finishRank(pending, false); }
                    catch (...) { result->finishRankWithoutSummary(pending); }
                }
                break;
            } catch (...) {
                if (fatal_on_error) {
                    RTP_LLM_FAIL("broadcast rpc dispatch failed, rank=%d unknown error %s", rank, rpc_tag.c_str());
                }
                for (size_t pending = rank; pending < worker_size; ++pending) {
                    try { result->finishRank(pending, false); }
                    catch (...) { result->finishRankWithoutSummary(pending); }
                }
                break;
            }
        }

        return result;
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    struct CompletionTagBase {
        virtual ~CompletionTagBase() = default;
        virtual void complete(bool ok) = 0;
        virtual void releaseFromCq() noexcept { delete this; }
    };

    template<typename ResultT>
    struct CompletionTag final: CompletionTagBase {
        CompletionTag(std::shared_ptr<ResultT> result, size_t rank): result(std::move(result)), rank(rank) {}

        void complete(bool ok) override {
            try { result->finishRank(rank, ok); }
            catch (...) { result->finishRankWithoutSummary(rank); }
        }
        void releaseFromCq() noexcept override { auto keep_alive = std::move(anchor); }
        std::shared_ptr<CompletionTag> anchor;

        std::shared_ptr<ResultT> result;
        size_t                   rank;
    };

    void pollCompletions() {
        void* raw_tag = nullptr;
        bool  ok      = false;
        while (completion_queue_->Next(&raw_tag, &ok)) {
            std::unique_ptr<CompletionTagBase, void (*)(CompletionTagBase*)> tag(
                static_cast<CompletionTagBase*>(raw_tag), [](CompletionTagBase* value) { value->releaseFromCq(); });
            try {
                tag->complete(ok);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("broadcast completion callback failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("broadcast completion callback failed: unknown error");
            }
        }
    }

    std::vector<std::string>                      worker_addrs_;
    const std::function<void(const std::string&)> connect_failure_reporter_;
    std::shared_ptr<RPCPool>                      rpc_pool_;
    std::unique_ptr<grpc::CompletionQueue>        completion_queue_;
    std::thread                                   completion_thread_;
};

}  // namespace rtp_llm
