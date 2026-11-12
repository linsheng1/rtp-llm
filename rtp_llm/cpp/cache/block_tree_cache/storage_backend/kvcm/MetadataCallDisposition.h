#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

namespace rtp_llm::kvcm {
// Source-buffer completion says nothing about whether metadata executed.
enum class MetadataCallDisposition { NOT_SENT, UNKNOWN, ACKNOWLEDGED };

// The original publication Session owns one attempt. Only its fixed value
// record survives an unknown ACK; it owns no source, client, RPC or backend.
class Session {
public:
    static constexpr size_t kIdBytes = 1024;
    enum class Phase : uint8_t { FREE, RESERVED, START_INVOKED, OPEN, PAYLOAD_POSSIBLE, FINISH_INVOKED, UNKNOWN };
    enum class Reason : uint8_t { NONE, ABANDONED, START_UNKNOWN, EMPTY_ID_CONTRACT, ID_TOO_LONG,
        FINISH_UNKNOWN, UNUSED_CLOSE_UNKNOWN, PAYLOAD_UNPROVEN, NON_BUSINESS_SETTLEMENT };
    struct Usage { size_t capacity, occupied, unknown, storage_bytes, slot_bytes;
        uint64_t rejected_slots, rejected_bytes, rejected_counters; };
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept: pool_(std::exchange(other.pool_, nullptr)), slot_(other.slot_), generation_(other.generation_) {}
    Session& operator=(Session&& other) noexcept {
        if (this != &other) { abandon(); pool_ = std::exchange(other.pool_, nullptr); slot_ = other.slot_; generation_ = other.generation_; }
        return *this;
    }
    ~Session() { abandon(); }
    explicit operator bool() const noexcept { return pool_ != nullptr; }
    static void configure(MetadataSessionLimits limits) { (void)pool(limits); }
    static Usage usage(MetadataSessionLimits limits) {
        auto& p = pool(limits); std::lock_guard<std::mutex> lock(p.mutex);
        return {p.capacity, p.occupied, p.unknown, p.storage_bytes, sizeof(Entry),
                p.rejected_slots, p.rejected_bytes, p.rejected_counters};
    }
    static std::optional<Session> reserve(MetadataSessionLimits limits) {
        auto& p = pool(limits); std::lock_guard<std::mutex> lock(p.mutex);
        if (p.exhausted) { saturate(p.rejected_counters); return {}; }
        if (p.free_head == p.capacity) {
            saturate(p.capacity < p.limits.sessions ? p.rejected_bytes : p.rejected_slots); return {};
        }
        // Configure freezes the quota, but read-only/idle backends need no ID array.
        // Allocate before a slot is admitted and before any StartWrite can execute.
        if (!p.entries) {
            auto entries = std::make_unique<Entry[]>(p.capacity);
            for (size_t i = 0; i < p.capacity; ++i) entries[i].next = i + 1;
            p.entries = std::move(entries);
            p.storage_bytes = sizeof(Pool) + p.capacity * sizeof(Entry);
        }
        auto& entry = p.entries[p.free_head];
        if (entry.generation == std::numeric_limits<uint64_t>::max()) {
            p.exhausted = true; saturate(p.rejected_counters); return {};
        }
        const auto index = p.free_head; p.free_head = entry.next;
        const auto generation = entry.generation + 1;
        entry = Entry{}; entry.generation = generation; entry.phase = Phase::RESERVED; ++p.occupied;
        return Session(&p, index, generation);
    }
    void startInvoked() noexcept { phase(Phase::RESERVED, Phase::START_INVOKED); }
    void payloadMayHaveStarted() noexcept { phase(Phase::OPEN, Phase::PAYLOAD_POSSIBLE); }
    bool observeStart(MetadataCallDisposition disposition, bool success, std::string_view id) noexcept {
        if (!pool_) return false;
        std::lock_guard<std::mutex> lock(pool_->mutex);
        auto* e = entry(); if (!e) return false;
        if (e->start_observed) return e->phase == Phase::OPEN;
        e->start_observed = true; e->id_original_length = id.size();
        if (id.size() > kIdBytes) { retain(*e, Reason::ID_TOO_LONG); return false; }
        std::copy(id.begin(), id.end(), e->id.begin()); e->id_length = id.size();
        if (disposition == MetadataCallDisposition::NOT_SENT && !success && id.empty()) { release(*e); return false; }
        if (disposition != MetadataCallDisposition::ACKNOWLEDGED || !success) { retain(*e, Reason::START_UNKNOWN); return false; }
        // No SDK empty-ID/all-present contract has been proved.
        if (id.empty()) { retain(*e, Reason::EMPTY_ID_CONTRACT); return false; }
        if (e->phase == Phase::UNKNOWN) return false;
        e->phase = Phase::OPEN; return true;
    }
    bool finishInvoked(bool unused_close) noexcept {
        if (!pool_) return false;
        std::lock_guard<std::mutex> lock(pool_->mutex); auto* e = entry();
        if (!e || (e->phase != Phase::OPEN && e->phase != Phase::PAYLOAD_POSSIBLE)) return false;
        e->unused_close = unused_close; e->phase = Phase::FINISH_INVOKED; return true;
    }
    void observeFinish(MetadataCallDisposition disposition, bool success) noexcept {
        if (!pool_) return;
        std::lock_guard<std::mutex> lock(pool_->mutex); auto* e = entry(); if (!e) return;
        if (e->phase == Phase::FINISH_INVOKED && disposition == MetadataCallDisposition::ACKNOWLEDGED && success) release(*e);
        else retain(*e, e->unused_close ? Reason::UNUSED_CLOSE_UNKNOWN : Reason::FINISH_UNKNOWN);
    }
    void retain(Reason reason) noexcept {
        if (!pool_) return;
        std::lock_guard<std::mutex> lock(pool_->mutex); if (auto* e = entry()) retain(*e, reason);
    }
private:
    friend struct SessionTestPeer;
    struct Entry {
        std::array<char, kIdBytes> id{};
        size_t id_length{0}, id_original_length{0}, next{0};
        uint64_t generation{0};
        Phase phase{Phase::FREE}; Reason reason{Reason::NONE};
        bool start_observed{false}, unused_close{false};
    };
    struct Pool {
        explicit Pool(MetadataSessionLimits value): limits(value) {
            if (!limits.sessions || limits.sessions > 65536 || limits.bytes > 128ULL * 1024 * 1024 || limits.bytes < sizeof(Pool))
                throw std::invalid_argument("invalid metadata session limits");
            capacity = std::min(limits.sessions, (limits.bytes - sizeof(Pool)) / sizeof(Entry));
            if (!capacity) throw std::invalid_argument("metadata bytes cannot hold a Session");
            storage_bytes = sizeof(Pool);
        }
        MetadataSessionLimits limits;
        std::mutex mutex;
        std::unique_ptr<Entry[]> entries;
        size_t capacity{0}, free_head{0}, occupied{0}, unknown{0}, storage_bytes{0};
        uint64_t rejected_slots{0}, rejected_bytes{0}, rejected_counters{0};
        bool exhausted{false};
    };
    static Pool& pool(MetadataSessionLimits limits) {
        // Value records only. Unknown metadata outlives a backend rebuild;
        // process exit is not mistaken for an ACK or remote cleanup proof.
        static Pool* const p = new Pool(limits);
        if (p->limits.sessions != limits.sessions || p->limits.bytes != limits.bytes)
            throw std::invalid_argument("metadata session process limits already frozen");
        return *p;
    }
    Session(Pool* value, size_t slot, uint64_t generation): pool_(value), slot_(slot), generation_(generation) {}
    Entry* entry() noexcept {
        if (!pool_ || slot_ >= pool_->capacity) return nullptr;
        auto& e = pool_->entries[slot_];
        return e.generation == generation_ && e.phase != Phase::FREE ? &e : nullptr;
    }
    static void saturate(uint64_t& value) noexcept { if (value != std::numeric_limits<uint64_t>::max()) ++value; }
    void retain(Entry& e, Reason reason) noexcept {
        if (e.phase != Phase::UNKNOWN) ++pool_->unknown;
        e.phase = Phase::UNKNOWN; e.reason = reason;
    }
    void release(Entry& e) noexcept {
        if (e.phase == Phase::UNKNOWN) --pool_->unknown;
        --pool_->occupied;
        const auto generation = e.generation;
        e = Entry{}; e.generation = generation; e.next = pool_->free_head;
        pool_->free_head = slot_; pool_ = nullptr;
    }
    void phase(Phase before, Phase after) noexcept {
        if (!pool_) return;
        std::lock_guard<std::mutex> lock(pool_->mutex); if (auto* e = entry()) if (e->phase == before) e->phase = after;
    }
    void abandon() noexcept {
        if (!pool_) return;
        std::lock_guard<std::mutex> lock(pool_->mutex);
        if (auto* e = entry()) {
            if (e->phase == Phase::RESERVED) release(*e);
            else if (e->phase != Phase::UNKNOWN) retain(*e, Reason::ABANDONED);
        }
        pool_ = nullptr;
    }
    Pool* pool_{nullptr}; size_t slot_{0}; uint64_t generation_{0};
};
} // namespace rtp_llm::kvcm
