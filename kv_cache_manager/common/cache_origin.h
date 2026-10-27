#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

inline bool HasDecodeOriginParameter(std::string_view uri) {
    const auto query = uri.find('?');
    if (query == std::string_view::npos)
        return false;
    size_t start = query + 1;
    while (start < uri.size()) {
        auto end = uri.find('&', start);
        if (end == std::string_view::npos)
            end = uri.size();
        auto equal = uri.find('=', start);
        if (equal == std::string_view::npos || equal > end)
            equal = end;
        if (uri.substr(start, equal - start) == "dtop_origin")
            return true;
        start = end + 1;
    }
    return false;
}

// Metadata only. No tensor bytes or sampled-token ownership are inferred here.
// All endpoints are exclusive computed-KV token offsets in one logical history.
// The session binds the assertion to the allocation being finished, not a key
// that may later be allocated by a different writer.
struct DecodeCacheOrigin {
    static constexpr const char *kUriParameter = "dtop_origin";
    std::string session;
    int64_t handoff_end = 0;
    int64_t begin = 0;
    int64_t end = 0;
    int64_t computed_end = 0;
    int64_t sampled_end = 0;

    static bool Parse(std::string_view text, DecodeCacheOrigin &out) {
        // v1.<32 alphanumeric session>.<handoff>.<begin>.<end>.<computed>.<sampled>
        if (text.size() > 140 || text.substr(0, 3) != "v1.") {
            return false;
        }
        std::array<std::string_view, 6> fields;
        size_t offset = 3;
        for (size_t i = 0; i < fields.size(); ++i) {
            const size_t next = text.find('.', offset);
            if ((i + 1 == fields.size()) != (next == std::string_view::npos)) {
                return false;
            }
            const size_t end = next == std::string_view::npos ? text.size() : next;
            fields[i] = text.substr(offset, end - offset);
            offset = end + 1;
        }
        if (fields[0].size() != 32) {
            return false;
        }
        for (char c : fields[0]) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                return false;
            }
        }
        DecodeCacheOrigin result;
        std::array<int64_t *, 5> values{
            &result.handoff_end, &result.begin, &result.end, &result.computed_end, &result.sampled_end};
        for (size_t i = 0; i < values.size(); ++i) {
            const auto field = fields[i + 1];
            if (field.empty() || field.front() < '0' || field.front() > '9' ||
                (field.size() > 1 && field.front() == '0')) {
                return false;
            }
            const auto [last, ec] = std::from_chars(field.data(), field.data() + field.size(), *values[i]);
            if (ec != std::errc{} || last != field.data() + field.size()) {
                return false;
            }
        }
        if (result.begin >= result.end || result.end > result.computed_end ||
            result.handoff_end > result.computed_end || result.computed_end == std::numeric_limits<int64_t>::max() ||
            result.sampled_end != result.computed_end + 1) {
            return false;
        }
        result.session = std::string(fields[0]);
        out = std::move(result);
        return true;
    }
};

// Only add a bounded source assertion to the exact original allocation. Core
// URI parameters (including unknown extension parameters) cannot be remapped.
// Canonical equality with a newly constructed URI also rejects duplicate query
// parameters instead of trusting StandardUri's last-value-wins parsing.
inline bool ValidateFinishedCacheUri(const std::string &allocated,
                                     const std::string &finished,
                                     const std::string &session,
                                     std::string &origin) {
    StandardUri original(allocated);
    StandardUri actual(finished);
    if (!original.Valid() || !actual.Valid() || original.HasParam(DecodeCacheOrigin::kUriParameter)) {
        return false;
    }
    if (finished == allocated) {
        origin.clear();
        return true;
    }
    const std::string value = actual.GetParam(DecodeCacheOrigin::kUriParameter);
    DecodeCacheOrigin parsed;
    if (!DecodeCacheOrigin::Parse(value, parsed) || parsed.session != session ||
        finished != original.ToUriStringWithExtraParam(DecodeCacheOrigin::kUriParameter, value)) {
        return false;
    }
    origin = value;
    return true;
}

} // namespace kv_cache_manager
