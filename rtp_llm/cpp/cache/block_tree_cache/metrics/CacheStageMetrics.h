#pragma once
#include <cstdint>
#include <chrono>
#include <memory>
namespace kmonitor {
class MetricsReporter;
}
namespace rtp_llm {
// Bounded aggregation: producers never allocate or wait for KMonitor/a mutex.
// Contention or excess series drops observations, never business work.
class CacheStageMetrics {
public:
    explicit CacheStageMetrics(std::shared_ptr<kmonitor::MetricsReporter> reporter,
                               std::chrono::milliseconds                  interval = std::chrono::seconds(1));
    ~CacheStageMetrics();
    // latency is the sum over sample_count observations (one batch by default).
    void record(const char* stage,
                int64_t     latency,
                bool        success,
                int64_t     rank,
                const char* scope,
                const char* operation,
                const char* source,
                const char* target,
                uint64_t    sample_count = 1) noexcept;
    void flush();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace rtp_llm
