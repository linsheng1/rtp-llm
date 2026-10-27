#include "kv_cache_manager/manager/finish_write_validator.h"

#include <limits>
#include <algorithm>
#include <map>
#include <optional>
#include <unordered_set>

#include "kv_cache_manager/common/cache_origin.h"

namespace kv_cache_manager {

int64_t CountCompletedDecodePublishTokens(bool finish_ok,
                                         size_t allocated_keys,
                                         size_t successful_keys,
                                         const std::string &session_id,
                                         const CacheLocationVector &validated_serving) {
    if (!finish_ok || allocated_keys == 0 || successful_keys != allocated_keys ||
        validated_serving.size() != allocated_keys)
        return 0;
    std::vector<std::pair<int64_t, int64_t>> intervals;
    intervals.reserve(validated_serving.size());
    for (const auto &location : validated_serving) {
        if (!location || location->location_specs().empty())
            return 0;
        std::string agreed;
        DecodeCacheOrigin origin;
        for (const auto &spec : location->location_specs()) {
            StandardUri uri(spec.uri());
            const auto value = uri.GetParam(DecodeCacheOrigin::kUriParameter);
            if (!uri.Valid() || !DecodeCacheOrigin::Parse(value, origin) || origin.session != session_id ||
                (!agreed.empty() && agreed != value))
                return 0;
            agreed = value;
        }
        const auto begin = std::max(origin.begin, origin.handoff_end);
        if (origin.end > begin)
            intervals.emplace_back(begin, origin.end);
    }
    std::sort(intervals.begin(), intervals.end());
    int64_t total = 0, previous_end = 0;
    for (const auto &[begin, end] : intervals) {
        const auto uncovered_begin = std::max(begin, previous_end);
        if (end > uncovered_begin)
            total += end - uncovered_begin;
        previous_end = std::max(previous_end, end);
    }
    return total;
}

ErrorCode ValidateFinishedLocations(const WriteLocationManager::WriteLocationInfo &allocation,
                                    const std::string &instance_id,
                                    const std::string &session_id,
                                    const BlockMask &success_mask,
                                    const std::vector<CacheLocation> &actual_locations,
                                    CacheLocationVector &serving_locations) {
    if (allocation.instance_id != instance_id || allocation.keys.size() != allocation.location_ids.size() ||
        allocation.keys.size() != allocation.allocated_locations.size() ||
        !IsBlockMaskValid(success_mask, allocation.keys.size()) || actual_locations.empty()) {
        return EC_BADARGS;
    }
    size_t selected = 0;
    for (size_t i = 0; i < allocation.keys.size(); ++i) {
        selected += IsIndexInMaskRange(success_mask, i);
    }
    if (selected != actual_locations.size()) {
        return EC_BADARGS;
    }
    CacheLocationVector validated;
    validated.reserve(selected);
    std::unordered_set<int64_t> seen_keys;
    std::unordered_set<std::string> seen_ids;
    std::optional<DecodeCacheOrigin> source_scope;
    bool marker_mode = false;
    for (const auto &location : actual_locations) {
        for (const auto &spec : location.location_specs()) {
            marker_mode = marker_mode || HasDecodeOriginParameter(spec.uri());
        }
    }
    for (size_t i = 0; i < allocation.keys.size(); ++i) {
        if (!seen_keys.insert(allocation.keys[i]).second || !seen_ids.insert(allocation.location_ids[i]).second) {
            return EC_BADARGS;
        }
        const auto &original = allocation.allocated_locations[i];
        if (!original || original->status() != CLS_WRITING || original->id() != allocation.location_ids[i] ||
            original->allocation_session_id() != session_id) {
            return EC_BADARGS;
        }
        if (!IsIndexInMaskRange(success_mask, i)) {
            continue;
        }
        const auto &actual = actual_locations[validated.size()];
        if ((actual.type() != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN && actual.type() != original->type()) ||
            (actual.spec_size() != std::numeric_limits<size_t>::max() && actual.spec_size() != original->spec_size()) ||
            original->spec_size() == 0 || original->spec_size() != actual.location_specs().size() ||
            original->spec_size() != original->location_specs().size()) {
            return EC_BADARGS;
        }
        std::map<std::string, const LocationSpec *> by_name;
        for (const auto &spec : actual.location_specs()) {
            if (spec.name().empty() || !by_name.emplace(spec.name(), &spec).second) {
                return EC_BADARGS;
            }
        }
        std::string shared_origin;
        bool first = true;
        std::vector<LocationSpec> specs;
        specs.reserve(original->spec_size());
        for (const auto &spec : original->location_specs()) {
            auto found = by_name.find(spec.name());
            std::string origin;
            if (found == by_name.end() ||
                !ValidateFinishedCacheUri(spec.uri(), found->second->uri(), session_id, origin) ||
                (!first && origin != shared_origin)) {
                return EC_BADARGS;
            }
            shared_origin = std::move(origin);
            first = false;
            specs.push_back(*found->second);
            by_name.erase(found);
        }
        if (!by_name.empty() || (marker_mode && shared_origin.empty())) {
            return EC_BADARGS;
        }
        if (!shared_origin.empty()) {
            DecodeCacheOrigin origin;
            if (!DecodeCacheOrigin::Parse(shared_origin, origin) ||
                (source_scope &&
                 (origin.handoff_end != source_scope->handoff_end ||
                  origin.computed_end != source_scope->computed_end ||
                  origin.sampled_end != source_scope->sampled_end || origin.begin < source_scope->end))) {
                return EC_BADARGS;
            }
            source_scope = std::move(origin);
        }
        auto replacement = std::make_shared<CacheLocation>(*original);
        replacement->set_location_specs(std::move(specs));
        replacement->set_status(CLS_SERVING);
        validated.push_back(std::move(replacement));
    }
    serving_locations = std::move(validated);
    return EC_OK;
}

} // namespace kv_cache_manager
