#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "aios/network/anet/metric/KMonitorANetMetricReporter.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include <gtest/gtest.h>
#include <memory>

namespace rtp_llm {
class TcpServerMetricLifetimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        kmonitor::MetricsConfig config;
        config.set_inited(true);
        config.set_manually_mode(true);
        config.set_enable_log_file_sink(false);
        ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    }
    void TearDown() override { kmonitor::KMonitorFactory::Shutdown(); }
};

TEST_F(TcpServerMetricLifetimeTest, OriginalCompositeKeepsAnetLoopWithRetainedRpcOwner) {
    anet::Transport transport(1);
    ASSERT_TRUE(transport.start());
    arpc::KMonitorANetMetricReporterConfig config;
    config.anetConfig.enableANetMetric=true;
    config.arpcConfig.enableArpcMetric=true;
    auto owner=std::make_shared<arpc::KMonitorANetServerMetricReporter>(config);
    ASSERT_TRUE(owner->init(&transport));
    auto retained=owner;
    owner.reset();
    ASSERT_TRUE(retained->_anetReporter);
    EXPECT_TRUE(retained->_anetReporter->_reportThread);
    // Keep the negative ownership observation deterministic without provoking
    // undefined access to a freed monitor in the test process.
    retained->_anetReporter->release();
    EXPECT_FALSE(retained->_anetReporter->_reportThread);
    retained.reset();transport.stop();transport.wait();
}

TEST_F(TcpServerMetricLifetimeTest, ExplicitStopJoinsLoopEvenWithRetainedAnetOwner) {
    TcpServer server;
    ASSERT_TRUE(server.init(1,1,true,8));
    auto witness=server.transport_metrics_;
    ASSERT_TRUE(witness);ASSERT_TRUE(witness->_reportThread);
    server.stop();
    EXPECT_FALSE(witness->_reportThread);
    EXPECT_FALSE(server.transport_metrics_);
    server.stop();
    kmonitor::KMonitorFactory::Shutdown();
    // Its monitor is gone, but no report thread survives and destruction is safe.
    witness.reset();
}

TEST_F(TcpServerMetricLifetimeTest, DisabledMetricsAndReinitializationRemainSafe) {
    TcpServer server;
    ASSERT_TRUE(server.init(1,1,false,8));
    EXPECT_FALSE(server.transport_metrics_);server.stop();
    ASSERT_TRUE(server.init(1,1,true,8));
    EXPECT_TRUE(server.transport_metrics_);server.stop();server.stop();
}
}  // namespace rtp_llm

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include <thread>
namespace rtp_llm {
static std::shared_ptr<NormalCacheStore> networkOnlyStore() {
    auto store=std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    store->request_block_buffer_store_=std::make_shared<RequestBlockBufferStore>(nullptr);
    auto messenger=std::make_shared<TcpMessager>(nullptr,store->request_block_buffer_store_,nullptr);
    messenger->tcp_server_=std::make_shared<TcpServer>();
    if (!messenger->tcp_server_->init(1,1,true,8)) { return nullptr; }
    store->messager_=messenger;
    return store;
}
static std::shared_ptr<anet::KMonitorANetMetricReporter> reporterOf(const std::shared_ptr<NormalCacheStore>& store) {
    return std::static_pointer_cast<TcpMessager>(store->messager_)->tcp_server_->transport_metrics_;
}
TEST_F(TcpServerMetricLifetimeTest, RemoteBaseStopDispatchesStoreQuiescenceDespiteAliases) {
    auto store=networkOnlyStore();ASSERT_TRUE(store);
    auto alias=store;auto witness=reporterOf(store);ASSERT_TRUE(witness->_reportThread);
    RemoteRpcServer remote;
    remote.cache_store_=store;remote.resource_.cache_store=store;
    // Previous nonvirtual base stop only stopped the engine, leaving transport alive.
    remote.LocalRpcServer::stop();EXPECT_TRUE(witness->_reportThread);
    LocalRpcServer* base=&remote;base->stop();
    EXPECT_FALSE(witness->_reportThread);EXPECT_FALSE(remote.cache_store_);
    EXPECT_FALSE(remote.resource_.cache_store);EXPECT_FALSE(store->messager_);
    base->stop();store->stop();
    kmonitor::KMonitorFactory::Shutdown();alias.reset();store.reset();witness.reset();
}
TEST_F(TcpServerMetricLifetimeTest, ConcurrentStoreStopJoinsOneTransportWithRetainedStore) {
    auto store=networkOnlyStore();ASSERT_TRUE(store);auto witness=reporterOf(store);
    std::thread a([store]{store->stop();});std::thread b([store]{store->stop();});a.join();b.join();
    EXPECT_FALSE(witness->_reportThread);EXPECT_FALSE(store->messager_);
    kmonitor::KMonitorFactory::Shutdown();store.reset();witness.reset();
}
TEST_F(TcpServerMetricLifetimeTest, PartiallyInitializedStoreCanStopRepeatedly) {
    auto store=std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    store->stop();store->stop();store.reset();
}
}
