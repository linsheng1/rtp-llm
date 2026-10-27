#pragma once

#include <optional>

namespace rtp_llm {

bool remoteCacheGdrEnabled();

// One feature switch for every P CP / D DP topology. Legacy names are read
// only when the generic switch is absent. No CP-specific flags are required.
bool remoteCacheDecodePublicationEnabled();

// Returns 0 when the environment variable is unset. Invalid or out-of-range
// values return nullopt so backend initialization can fail explicitly.
std::optional<double> remoteCacheQuarantineAbortRatio();

}  // namespace rtp_llm
