#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include <gtest/gtest.h>
#include <xgrammar/tokenizer_info.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {
namespace {
class LogitsProcessorFactoryShutdownTest : public ::testing::Test {
protected:
    void SetUp() override {
        kmonitor::MetricsConfig config;
        config.set_inited(true);
        config.set_manually_mode(true);
        config.set_enable_log_file_sink(false);
        ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    }
    void TearDown() override {
        LogitsProcessorFactory::shutdown();
        kmonitor::KMonitorFactory::Shutdown();
    }
    GrammarConfig config() {
        std::vector<std::string> vocab;
        for (int i = 0; i < 128; ++i) { vocab.emplace_back(1, static_cast<char>(i)); }
        GrammarConfig cfg;
        cfg.tokenizer_info_json = xgrammar::TokenizerInfo(
            vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
        cfg.num_workers = 1;
        cfg.compile_concurrency = 1;
        cfg.compile_queue_size = 4;
        cfg.compile_timeout_ms = 2000;
        cfg.compiler_cache_bytes = 1024 * 1024;
        return cfg;
    }
    std::weak_ptr<kmonitor::MetricsReporter> init() {
        auto reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
        std::weak_ptr<kmonitor::MetricsReporter> weak = reporter;
        LogitsProcessorFactory::init(ModelConfig{}, config(), "", std::move(reporter));
        return weak;
    }
};

TEST_F(LogitsProcessorFactoryShutdownTest, ReleasesReporterBeforeMonitorShutdown) {
    auto weak = init();
    ASSERT_FALSE(weak.expired());  // Only the static grammar registry owns it now.
    auto compiled = LogitsProcessorFactory::grammarBackend()->compile({"regex", "[a-z]+"});
    ASSERT_TRUE(compiled.ok());
    LogitsProcessorFactory::shutdown();
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(LogitsProcessorFactory::grammarBackend());
    // The old order freed the underlying KMonitor first, leaving its reporter
    // to dereference a dangling monitor->Name() from the static destructor.
    kmonitor::KMonitorFactory::Shutdown();
    LogitsProcessorFactory::shutdown();
}

TEST_F(LogitsProcessorFactoryShutdownTest, DisabledAndRepeatedShutdownAreSafe) {
    GrammarConfig disabled;
    LogitsProcessorFactory::init(ModelConfig{}, disabled, "");
    LogitsProcessorFactory::shutdown();
    LogitsProcessorFactory::shutdown();
    EXPECT_FALSE(LogitsProcessorFactory::grammarBackend());
}

TEST_F(LogitsProcessorFactoryShutdownTest, CanInitializeAgainAfterShutdown) {
    auto first = init();
    LogitsProcessorFactory::shutdown();
    EXPECT_TRUE(first.expired());
    auto second = init();
    ASSERT_FALSE(second.expired());
    auto compiled = LogitsProcessorFactory::grammarBackend()->compile({"regex", "[0-9]+"});
    ASSERT_TRUE(compiled.ok());
    LogitsProcessorFactory::shutdown();
    EXPECT_TRUE(second.expired());
}
}  // namespace
}  // namespace rtp_llm
