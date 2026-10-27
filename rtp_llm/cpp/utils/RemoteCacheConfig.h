#pragma once

#include <optional>
#include <cstddef>

namespace rtp_llm {

bool remoteCacheGdrEnabled();

// One feature switch for every P CP / D DP topology. No legacy or CP-specific
// flags are read. Publication is disabled unless explicitly enabled.
bool remoteCacheDecodePublicationEnabled();

struct DecodePublicationLimits {
    // Initial deployment defaults: 12.5% of the current 4 GiB/rank test pool.
    // These cap existing pinned pages, rather than allocate another GPU pool.
    size_t bytes = 512ULL * 1024 * 1024;
    size_t jobs = 4;
};

// Strictly parsed at backend initialization. Overrides are deployment budgets,
// not separate feature switches. Invalid values reject initialization.
std::optional<DecodePublicationLimits> remoteCacheDecodePublicationLimits();

// Returns 0 when the environment variable is unset. Invalid or out-of-range
// values return nullopt so backend initialization can fail explicitly.
std::optional<double> remoteCacheQuarantineAbortRatio();

}  // namespace rtp_llm
