#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/DecodeCacheOrigin.h"

namespace rtp_llm {

// Metadata only; serialized by the owning physical block pool's existing
// mutex. A physical ID is not an identity without its allocation incarnation.
// Construct only for feature-on pools; no global cache-key/address registry.
class PhysicalDecodeOrigins {
public:
    using Origin = std::shared_ptr<const DecodeCacheOrigin>;
    explicit PhysicalDecodeOrigins(size_t capacity): incarnations_(capacity, 0) {}
    PhysicalDecodeOrigins(const PhysicalDecodeOrigins&)            = delete;
    PhysicalDecodeOrigins& operator=(const PhysicalDecodeOrigins&) = delete;

    uint64_t beginAllocation(size_t block) {
        auto& incarnation = incarnations_.at(block);
        origins_.erase(block);
        // Exhausting the metadata sequence must not wrap and accept a stale
        // callback. Leave attribution unavailable without changing KV data.
        incarnation = next_ == std::numeric_limits<uint64_t>::max() ? 0 : ++next_;
        return incarnation;
    }
    void release(size_t block) {
        incarnations_.at(block) = 0;
        origins_.erase(block);
    }
    uint64_t incarnation(size_t block) const {
        return incarnations_.at(block);
    }
    bool bind(size_t block, uint64_t expected, Origin origin) {
        if (expected == 0 || incarnation(block) != expected)
            return false;
        if (origin) {
            try {
                origins_[block] = std::move(origin);
            } catch (const std::bad_alloc&) {
                return false;  // Missing telemetry must not fail a completed read.
            }
        } else
            origins_.erase(block);
        return true;
    }
    Origin get(size_t block) const {
        if (incarnation(block) == 0)
            return {};
        const auto found = origins_.find(block);
        return found == origins_.end() ? Origin{} : found->second;
    }

private:
    uint64_t                           next_{0};
    std::vector<uint64_t>              incarnations_;
    std::unordered_map<size_t, Origin> origins_;
};

}  // namespace rtp_llm
