#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/MetadataCallDisposition.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#ifdef NDEBUG
#error "Session tests require active checks"
#endif
namespace { std::atomic<size_t> allocations{0}, checks{0}; }
void* operator new(size_t bytes) { ++allocations; if (auto* p = std::malloc(bytes ? bytes : 1)) return p; throw std::bad_alloc(); }
void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
namespace rtp_llm::kvcm {
// Only this test translation unit inspects private records; production has no
// token injection, reset, resolve-UNKNOWN or diagnostic-record API.
struct SessionTestPeer {
    static size_t bytes(size_t n) { return sizeof(Session::Pool) + n * sizeof(Session::Entry); }
    static auto record(Session& s) {
        std::lock_guard<std::mutex> lock(s.pool_->mutex);
        return *s.entry();
    }
    static Session stale(Session& s) { return Session(s.pool_, s.slot_, s.generation_ - 1); }
    static void exhaust(rtp_llm::MetadataSessionLimits limits) {
        auto& p = Session::pool(limits); p.entries[p.free_head].generation = UINT64_MAX;
    }
};
}
namespace {
using S = rtp_llm::kvcm::Session;
using Peer = rtp_llm::kvcm::SessionTestPeer;
using D = rtp_llm::kvcm::MetadataCallDisposition;
using Limits = rtp_llm::MetadataSessionLimits;
#define CHECK(e) do { ++checks; if (!(e)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #e); std::abort(); } } while (false)
Limits limits(size_t n) { return {n, Peer::bytes(n)}; }
void close(S& s) { CHECK(s.finishInvoked(false)); s.observeFinish(D::ACKNOWLEDGED, true); CHECK(!s); }
void limitsAndFrozenConfiguration() {
    bool bad = false; try { S::configure({0, 1}); } catch (const std::invalid_argument&) { bad = true; } CHECK(bad);
    S::configure(limits(2)); const auto u = S::usage(limits(2)); CHECK(u.capacity == 2 && u.storage_bytes == Peer::bytes(2));
    bad = false; try { S::configure(limits(3)); } catch (const std::invalid_argument&) { bad = true; } CHECK(bad);
}
void moveAbandonAndQuota() {
    const auto l = limits(2); auto a = S::reserve(l); CHECK(a && *a);
    { auto b = S::reserve(l); CHECK(b); S moved(std::move(*b)); CHECK(!*b && moved); CHECK(!S::reserve(l)); }
    CHECK(S::usage(l).occupied == 1); a.reset(); CHECK(S::usage(l).occupied == 0);
}
void noSendAndDuplicateStart() {
    const auto l = limits(1); auto a = S::reserve(l); a->startInvoked();
    CHECK(!a->observeStart(D::NOT_SENT, false, "")); CHECK(!*a && S::usage(l).occupied == 0);
    auto b = S::reserve(l); CHECK(b); b->startInvoked(); CHECK(b->observeStart(D::ACKNOWLEDGED, true, "original"));
    CHECK(b->observeStart(D::NOT_SENT, false, "replacement"));
    const auto record = Peer::record(*b); CHECK(std::string_view(record.id.data(), record.id_length) == "original");
    a->startInvoked(); CHECK(!a->finishInvoked(false)); CHECK(Peer::record(*b).phase == S::Phase::OPEN); close(*b);
}
void emptyAndErrorStartNeverAuthorizeFinish() {
    const auto l = limits(3);
    for (int mode = 0; mode < 3; ++mode) {
        auto a = S::reserve(l); CHECK(a); a->startInvoked();
        CHECK(!a->observeStart(mode == 0 ? D::UNKNOWN : D::ACKNOWLEDGED, mode == 2, mode == 1 ? "error-sid" : ""));
        CHECK(Peer::record(*a).phase == S::Phase::UNKNOWN); CHECK(!a->finishInvoked(false));
    }
    CHECK(S::usage(l).unknown == 3 && !S::reserve(l));
}
void exactAndOversizedId() {
    const auto l = limits(2); const std::string exact(1024, 'x'), oversized(1025, 'y');
    auto a = S::reserve(l); a->startInvoked(); CHECK(a->observeStart(D::ACKNOWLEDGED, true, exact));
    auto record = Peer::record(*a); CHECK(record.id_length == 1024 && std::string_view(record.id.data(), record.id_length) == exact); close(*a);
    auto b = S::reserve(l); b->startInvoked(); CHECK(!b->observeStart(D::ACKNOWLEDGED, true, oversized));
    record = Peer::record(*b); CHECK(record.phase == S::Phase::UNKNOWN && record.reason == S::Reason::ID_TOO_LONG);
    CHECK(record.id_length == 0 && record.id_original_length == 1025);
    CHECK(std::all_of(record.id.begin(), record.id.end(), [](char c) { return c == 0; })); CHECK(!b->finishInvoked(true));
}
void finishOnceAndLostUnusedAck() {
    const auto l = limits(2);
    { auto a = S::reserve(l); a->startInvoked(); CHECK(a->observeStart(D::ACKNOWLEDGED, true, "unused"));
      CHECK(a->finishInvoked(true)); CHECK(!a->finishInvoked(false)); a->observeFinish(D::UNKNOWN, false);
      CHECK(Peer::record(*a).reason == S::Reason::UNUSED_CLOSE_UNKNOWN); }
    { auto b = S::reserve(l); b->startInvoked(); CHECK(b->observeStart(D::ACKNOWLEDGED, true, "write"));
      b->payloadMayHaveStarted(); CHECK(b->finishInvoked(false)); b->observeFinish(D::UNKNOWN, false);
      CHECK(Peer::record(*b).reason == S::Reason::FINISH_UNKNOWN); }
    CHECK(S::usage(l).occupied == 2 && S::usage(l).unknown == 2);
}
void actualIdsDoNotCoalesceAndStaleCannotTouchNewSession() {
    const auto l = limits(2); auto a = S::reserve(l), b = S::reserve(l); CHECK(a && b);
    a->startInvoked(); b->startInvoked(); CHECK(a->observeStart(D::ACKNOWLEDGED, true, "same")); CHECK(b->observeStart(D::ACKNOWLEDGED, true, "same"));
    CHECK(S::usage(l).occupied == 2); close(*a); close(*b);
    auto c = S::reserve(l); auto stale = Peer::stale(*c); c->startInvoked(); CHECK(c->observeStart(D::ACKNOWLEDGED, true, "current"));
    stale.startInvoked(); stale.retain(S::Reason::PAYLOAD_UNPROVEN); CHECK(!stale.finishInvoked(false));
    CHECK(Peer::record(*c).phase == S::Phase::OPEN); close(*c);
}
void fixedBytesAndCounterExhaustion() {
    const Limits l{3, Peer::bytes(1)}; auto a = S::reserve(l); CHECK(a && !S::reserve(l));
    CHECK(S::usage(l).capacity == 1 && S::usage(l).rejected_bytes == 1); a.reset();
    Peer::exhaust(l); CHECK(!S::reserve(l)); CHECK(S::usage(l).rejected_counters == 1 && S::usage(l).occupied == 0);
}
void noAllocationAfterReservationAndNoSourceOwner() {
    const auto l = limits(2); S::configure(l); auto a = S::reserve(l); CHECK(a);
    auto source = std::make_shared<int>(1); const std::weak_ptr<int> weak = source;
    const auto before = allocations.load(); a->startInvoked(); CHECK(a->observeStart(D::ACKNOWLEDGED, true, "owned"));
    a->payloadMayHaveStarted(); a->retain(S::Reason::PAYLOAD_UNPROVEN); CHECK(!a->finishInvoked(false));
    CHECK(allocations.load() == before); a.reset(); source.reset(); CHECK(weak.expired());
    CHECK(S::usage(l).unknown == 1);
}
void parallelSessions() {
    const auto l = limits(16); S::configure(l); std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back([&] {
        for (int n = 0; n < 32; ++n) { auto a = S::reserve(l); CHECK(a); a->startInvoked(); CHECK(a->observeStart(D::ACKNOWLEDGED, true, "parallel")); close(*a); }
    });
    for (auto& t : threads) t.join(); CHECK(S::usage(l).occupied == 0);
}
}
int main() {
    // Real process-lifetime records are never reset. Each scenario has its own
    // child process, just like the existing backend process regression tests.
    const std::vector<void(*)()> cases{limitsAndFrozenConfiguration, moveAbandonAndQuota, noSendAndDuplicateStart,
        emptyAndErrorStartNeverAuthorizeFinish, exactAndOversizedId, finishOnceAndLostUnusedAck,
        actualIdsDoNotCoalesceAndStaleCannotTouchNewSession, fixedBytesAndCounterExhaustion,
        noAllocationAfterReservationAndNoSourceOwner, parallelSessions};
    for (auto run : cases) {
        const auto pid = fork(); CHECK(pid >= 0);
        if (!pid) { run(); _exit(0); }
        int status = 0; CHECK(waitpid(pid, &status, 0) == pid); CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    std::puts("10 original Session scenarios completed");
}
