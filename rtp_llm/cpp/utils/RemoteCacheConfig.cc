#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace rtp_llm {

bool remoteCacheGdrEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

bool remoteCacheDecodePublicationEnabled() {
    const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH");
    if (value != nullptr)
        return std::string_view(value) == "1";
    const char* legacy = std::getenv("RTP_DTOP_FILE_CP2");
    return legacy != nullptr && std::string_view(legacy) == "1";
}

std::optional<double> remoteCacheQuarantineAbortRatio() {
    const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO");
    if (value == nullptr || *value == '\0') {
        return 0.0;
    }

    errno            = 0;
    char*      end   = nullptr;
    const auto ratio = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        return std::nullopt;
    }
    return ratio;
}

}  // namespace rtp_llm
