#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

#ifdef NDEBUG
#error "BoundedFanoutHostTest needs active checks"
#endif

namespace {
using Fanout = rtp_llm::BroadcastResult<::FunctionRequestPB, ::FunctionResponsePB>;
std::atomic<size_t> checks{0};
#define CHECK(expression) do { ++checks; if (!(expression)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #expression); std::abort(); } } while (false)
auto future() { return Fanout::Clock::now() + std::chrono::seconds(30); }


void inlineExpiryTrueDoesNotFinishBeforeSubmitReturns() {
    int calls = 0, done = 0, submit_depth = 0;
    auto parent = Fanout::createPayloadWindow(3, 2, future(), [&](size_t child, const auto& self) {
        ++submit_depth; ++calls;
        CHECK(self->complete(child, Fanout::Outcome::notStarted(Fanout::Result::TIMEOUT)));
        CHECK(done == 0);
        --submit_depth;
        return true; // Actual BlockTreeTaskPool expiry contract.
    }, [] {}, [&] { CHECK(submit_depth == 0); ++done; });
    parent->start();
    CHECK(calls == 1 && done == 1);
    CHECK(parent->snapshot().submissions_closed && parent->snapshot().remaining == 0);
    CHECK(parent->outcome(2).proof == Fanout::Proof::NOT_STARTED);
}

void inlineLargeChainHasBoundedStackAndOneNotification() {
    int launch_depth = 0, peak_depth = 0, done = 0;
    auto parent = Fanout::createPayloadWindow(10000, 2, future(), [&](size_t child, const auto& self) {
        peak_depth = std::max(peak_depth, ++launch_depth);
        CHECK(self->tryStart(child)); CHECK(self->markMayTouch(child));
        CHECK(self->complete(child, Fanout::Outcome::success()));
        CHECK(!self->complete(child, Fanout::Outcome::unknown()));
        --launch_depth; return true;
    }, [] {}, [&] { ++done; });
    parent->start(); parent->start();
    CHECK(done == 1 && peak_depth == 1);
    CHECK(parent->snapshot().finished && parent->snapshot().peak <= 2);
}

void windowTwoAndReverseCompletionPreserveChildIdentity() {
    int done = 0;
    std::vector<size_t> launched;
    std::vector<std::shared_ptr<Fanout>> tickets(4);
    auto parent = Fanout::createPayloadWindow(4, 2, future(), [&](size_t child, const auto& self) {
        launched.push_back(child); tickets[child] = self; return true;
    }, [] {}, [&] { ++done; });
    parent->start();
    CHECK(launched == (std::vector<size_t>{0, 1}));
    CHECK(parent->snapshot().active == 2 && done == 0);
    CHECK(tickets[1]->tryStart(1)); CHECK(tickets[1]->markMayTouch(1));
    tickets[1]->complete(1, Fanout::Outcome::success()); tickets[1].reset();
    CHECK(launched == (std::vector<size_t>{0, 1, 2}));
    for (const size_t child : {size_t{2}, size_t{0}, size_t{3}}) {
        CHECK(tickets[child]->tryStart(child)); CHECK(tickets[child]->markMayTouch(child));
        CHECK(tickets[child]->complete(child, Fanout::Outcome::success())); tickets[child].reset();
    }
    CHECK(done == 1 && parent->snapshot().peak == 2);
    for (size_t i = 0; i < 4; ++i) CHECK(parent->outcome(i).result == Fanout::Result::SUCCESS);
}

void unknownStopsTailAndCannotBecomeNotStartedAfterHandoff() {
    int done = 0, launched = 0;
    auto parent = Fanout::createPayloadWindow(5, 2, future(), [&](size_t child, const auto& self) {
        ++launched; CHECK(self->tryStart(child)); CHECK(self->markMayTouch(child)); return child != 0;
    }, [] {}, [&] { ++done; });
    parent->start();
    CHECK(launched == 1 && done == 1);
    CHECK(parent->outcome(0).proof == Fanout::Proof::UNKNOWN);
    CHECK(parent->outcome(4).proof == Fanout::Proof::NOT_STARTED);
    CHECK(!parent->complete(0, Fanout::Outcome::success()));
}

void cancellationWaitsAcceptedChildAndReleasesDriverAnchor() {
    int done = 0;
    std::shared_ptr<Fanout> accepted;
    auto parent = Fanout::createPayloadWindow(3, 1, future(), [&](size_t, const auto& self) { accepted = self; return true; }, [] {}, [&] { ++done; });
    std::weak_ptr<Fanout> weak = parent;
    parent->start(); parent.reset(); CHECK(!weak.expired());
    CHECK(accepted && !weak.expired());
    accepted->cancel(); CHECK(done == 0);
    CHECK(!accepted->tryStart(0));
    CHECK(accepted->complete(0, Fanout::Outcome::notStarted(Fanout::Result::CANCELED)));
    CHECK(done == 1); accepted.reset(); CHECK(weak.expired());
}

void queueRejectThrowAndEmptyAreTerminalWithoutLaunch() {
    for (int mode = 0; mode < 3; ++mode) {
        int done = 0, launched = 0;
        auto parent = Fanout::createPayloadWindow(mode == 2 ? 0 : 3, 2, future(),
            [&](size_t, const auto&) -> Fanout::LaunchResult {
                ++launched; if (mode == 1) throw std::bad_alloc(); return false;
            }, [] {}, [&] { ++done; });
        parent->start();
        CHECK(done == 1 && launched == (mode == 2 ? 0 : 1));
        CHECK(parent->snapshot().remaining == 0);
        if (mode != 2) CHECK(parent->outcome(0).proof == Fanout::Proof::NOT_STARTED);
    }
}

void absoluteExpiredDeadlineNeverStartsAndLateSafeProofStaysSafe() {
    int launches = 0, done = 0;
    auto deadline = Fanout::Clock::now() - std::chrono::milliseconds(1);
    auto expired = Fanout::createPayloadWindow(4, 2, deadline, [&](size_t, const auto&) { ++launches; return true; }, [] {}, [&] { ++done; });
    expired->start(); CHECK(launches == 0 && done == 1);
    CHECK(expired->deadline() == deadline);
    CHECK(expired->outcome(0).result == Fanout::Result::TIMEOUT);
    auto late = Fanout::createPayloadWindow(1, 2, future(), [](size_t child, const auto& self) { return self->tryStart(child) && self->markMayTouch(child); }, [] {}, [&] { ++done; });
    late->start(); late->cancel();
    CHECK(done == 1);
    late->complete(0, {Fanout::Result::TIMEOUT, Fanout::Proof::BUFFER_SAFE});
    CHECK(done == 2 && late->outcome(0).proof == Fanout::Proof::BUFFER_SAFE);
}

void parallelDuplicateCallbacksSettleEachChildOnce() {
    std::atomic<int> done{0};
    auto parent = Fanout::createPayloadWindow(2, 2, future(), [](size_t child, const auto& self) { return self->tryStart(child) && self->markMayTouch(child); }, [] {}, [&] { ++done; });
    parent->start();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 16; ++i) threads.emplace_back([&, i] { parent->complete(i % 2, Fanout::Outcome::success()); });
    for (auto& thread : threads) thread.join();
    CHECK(done == 1 && parent->snapshot().remaining == 0);
}

void workerOneDefersWithoutHoldingAnotherResourceAndPrebuiltChildrenRemainAlive() {
    int done = 0, worker = 0, launched = 0;
    std::shared_ptr<Fanout> accepted;
    auto parent = Fanout::createPayloadWindow(3, 2, future(), [&](size_t child, const auto& self) -> Fanout::LaunchResult {
        if (worker) return Fanout::LaunchResult::deferred();
        ++worker; ++launched; accepted = self;
        CHECK(self->tryStart(child)); CHECK(self->markMayTouch(child)); return true;
    }, [] {}, [&] { ++done; });
    parent->start();
    CHECK(launched == 1 && worker == 1 && done == 0);
    CHECK(parent->snapshot().active == 1);
    for (size_t child = 0; child < 3; ++child) {
        --worker; // Actual safe Operation.release before notifying the pump.
        accepted->complete(child, Fanout::Outcome::success());
    }
    accepted.reset(); CHECK(done == 1 && worker == 0 && launched == 3);
    CHECK(parent->snapshot().remaining == 0);
}

void unknownKeepsAcceptedSiblingAndCancelsEveryUnsubmittedSlot() {
    int done = 0, launches = 0;
    auto parent = Fanout::createPayloadWindow(6, 2, future(), [&](size_t child, const auto& self) {
        ++launches; return self->tryStart(child) && self->markMayTouch(child);
    }, [] {}, [&] { ++done; });
    parent->start(); CHECK(launches == 2);
    CHECK(parent->complete(0, Fanout::Outcome::unknown()));
    CHECK(done == 0 && parent->snapshot().remaining == 1);
    for (size_t i = 2; i < 6; ++i) CHECK(parent->outcome(i).proof == Fanout::Proof::NOT_STARTED);
    CHECK(parent->complete(1, Fanout::Outcome::success()));
    CHECK(done == 1 && parent->outcome(0).proof == Fanout::Proof::UNKNOWN);
    CHECK(parent->outcome(0).proof == Fanout::Proof::UNKNOWN && parent->outcome(1).proof == Fanout::Proof::BUFFER_SAFE);
}

void laterDriverRejectionWaitsExistingChildAndReportsControlFailure() {
    int done = 0; bool reject = false;
    auto parent = Fanout::createPayloadWindow(5, 2, future(), [&](size_t child, const auto& self) {
        return !reject && self->tryStart(child) && self->markMayTouch(child);
    }, [] {}, [&] { ++done; });
    parent->start(); reject = true;
    CHECK(parent->complete(0, Fanout::Outcome::success()));
    CHECK(done == 0 && parent->snapshot().canceled && parent->snapshot().remaining == 1);
    CHECK(parent->complete(1, Fanout::Outcome::success()));
    CHECK(done == 1);
    for (size_t i = 2; i < 5; ++i) CHECK(parent->outcome(i).proof == Fanout::Proof::NOT_STARTED);
}

void originalWaitAndCallbacksObserveFinalizedResponses() {
    std::promise<void> entered, release;
    auto entered_future = entered.get_future(); auto released = release.get_future();
    std::atomic<bool> normalized{false}; std::atomic<int> notified{0};
    auto parent = Fanout::createPayloadWindow(1, 1, future(), [](size_t, const auto&) { return true; },
        [&] { entered.set_value(); released.wait(); normalized = true; },
        [&] { CHECK(normalized); ++notified; });
    parent->onDone([&] { CHECK(normalized); ++notified; });
    parent->start();
    std::thread finish([&] { parent->complete(0, Fanout::Outcome::success()); });
    CHECK(entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(!parent->done()); CHECK(!parent->waitDone(1, false)); CHECK(notified == 0);
    release.set_value(); finish.join();
    CHECK(parent->waitDone(1, false)); CHECK(parent->success()); CHECK(notified == 2);
    parent->onDone([&] { CHECK(normalized); ++notified; }); CHECK(notified == 3);
}

void cancellationDuringDeferredLaunchCannotReopenTail() {
    int launches = 0, done = 0;
    auto parent = Fanout::createPayloadWindow(3, 1, future(), [&](size_t, const auto& self) -> Fanout::LaunchResult {
        ++launches; self->cancel(); return Fanout::LaunchResult::deferred();
    }, [] {}, [&] { ++done; });
    parent->start();
    CHECK(done == 1 && launches == 1 && parent->snapshot().remaining == 0);
    CHECK(parent->snapshot().submissions_closed && parent->snapshot().active == 0);
    for (size_t i = 0; i < 3; ++i) CHECK(parent->outcome(i).proof == Fanout::Proof::NOT_STARTED);
}

}  // namespace

int main() {
    inlineExpiryTrueDoesNotFinishBeforeSubmitReturns();
    inlineLargeChainHasBoundedStackAndOneNotification();
    windowTwoAndReverseCompletionPreserveChildIdentity();
    unknownStopsTailAndCannotBecomeNotStartedAfterHandoff();
    cancellationWaitsAcceptedChildAndReleasesDriverAnchor();
    queueRejectThrowAndEmptyAreTerminalWithoutLaunch();
    absoluteExpiredDeadlineNeverStartsAndLateSafeProofStaysSafe();
    parallelDuplicateCallbacksSettleEachChildOnce();
    workerOneDefersWithoutHoldingAnotherResourceAndPrebuiltChildrenRemainAlive();
    unknownKeepsAcceptedSiblingAndCancelsEveryUnsubmittedSlot();
    laterDriverRejectionWaitsExistingChildAndReportsControlFailure();
    originalWaitAndCallbacksObserveFinalizedResponses();
    cancellationDuringDeferredLaunchCannotReopenTail();
    std::printf("BoundedFanoutHostTest: 13 original BroadcastResult window scenarios, %zu checks PASS\n", checks.load());
}
