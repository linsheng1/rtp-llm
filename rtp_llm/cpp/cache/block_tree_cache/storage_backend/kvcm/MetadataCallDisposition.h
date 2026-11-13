#pragma once

namespace rtp_llm::kvcm {
// Source-buffer completion says nothing about whether metadata executed.
enum class MetadataCallDisposition { NOT_SENT, UNKNOWN, ACKNOWLEDGED };
}  // namespace rtp_llm::kvcm
