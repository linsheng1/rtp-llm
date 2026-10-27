#pragma once

#include "kv_cache_manager/manager/write_location_manager.h"

namespace kv_cache_manager {

// The nonempty FinishWrite.locations contract is positional over the selected
// successful session keys. All locations are validated before claiming the
// session. The immutable replacements retain allocation id/version and core URI.
ErrorCode ValidateFinishedLocations(const WriteLocationManager::WriteLocationInfo &allocation,
                                    const std::string &instance_id,
                                    const std::string &session_id,
                                    const BlockMask &success_mask,
                                    const std::vector<CacheLocation> &actual_locations,
                                    CacheLocationVector &serving_locations);

// Authoritative whole-session metric only. Partial masks/CAS failures do not
// describe a completed publication, even if individual locations became serving.
// Call after successful session claim, complete validation and overall EC_OK.
int64_t CountCompletedDecodePublishTokens(bool finish_ok,
                                         size_t allocated_keys,
                                         size_t successful_keys,
                                         const std::string &session_id,
                                         const CacheLocationVector &validated_serving);

} // namespace kv_cache_manager
