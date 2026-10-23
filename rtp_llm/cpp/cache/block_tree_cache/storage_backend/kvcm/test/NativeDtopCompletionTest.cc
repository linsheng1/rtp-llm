#include <cuda_runtime_api.h>
#include <tuple>
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
namespace rtp_llm { namespace {
// Actual backend completion path with GPU source pages and a recording SDK.
class NativeDtopCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool>> {
protected:
    autil::EnvGuard enable{"RTP_DTOP_FILE_CP2", "1"};
    autil::EnvGuard pace{"RTP_DTOP_PACE", "1"};
    autil::EnvGuard target{"RTP_DTOP_FILE_TARGET_P_TP", "2"};
    autil::EnvGuard capture{"RTP_DTOP_FILE_CAPTURE_BLOCKS", "2"};
    autil::EnvGuard audit{"RTP_DTOP_PAYLOAD_AUDIT", "0"};
    autil::EnvGuard limit{"RTP_DTOP_FILE_MAX_COMPUTED_TOKENS", "2048"};
};

TEST_P(NativeDtopCompletionTest, CompleteCheckpointPublishesAllPSpecsOnlyAtFinishAndOnlyOnce) {
    const int target=std::get<0>(GetParam());
    const bool sharded=std::get<1>(GetParam());
    const int shards=sharded ? target : 1, checkpoints=sharded ? (target==2 ? 3 : 2) : 8;
    const size_t end=128*shards*checkpoints;
    autil::EnvGuard target_layout("RTP_DTOP_FILE_TARGET_P_TP",std::to_string(target));
    autil::EnvGuard capture_layout("RTP_DTOP_FILE_CAPTURE_BLOCKS",std::to_string(target));
    BackendEnvironment environment;
    environment.cache_config = test::makeSimpleMhaCacheConfig(1, 12, 128, DataType::TYPE_FP16, 1, 2);
    initializeEnvironmentPools(environment, "native_dtop_finish", 12);
    const auto tag = environment.cache_config.groupTags().front();
    auto blocks = environment.device_pool->malloc(shards*checkpoints);
    ASSERT_TRUE(blocks.has_value());
    environment.device_pool->incRef(*blocks);
    for (size_t i = 0; i < blocks->size(); ++i) {
        for (const auto& part: environmentBuffers(environment, 0, tag, blocks->at(i))) {
            ASSERT_EQ(cudaMemset(part.addr, static_cast<int>(i+1), part.size_bytes), cudaSuccess);
        }
    }
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    resource.initGroups(environment.cache_config.topologyPtr());
    resource.setBatchBlocks(0, tag, *blocks);
    CacheKeysType source_keys, canonical_keys;
    for(int i=0;i<shards*checkpoints;++i) source_keys.push_back(101+i);
    for(int i=1;i<=checkpoints;++i) canonical_keys.push_back(100+i*shards);
    resource.setBatchCacheKeys(0, source_keys);
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
        [&](const auto& configs, auto, const auto& registrations, const auto& tags) {
            std::vector<std::string> expected_tags;
            for(int rank=0;rank<target;++rank) expected_tags.push_back(tag+"#cp"+std::to_string(rank));
            EXPECT_EQ(tags,expected_tags);
            EXPECT_EQ(registrations.size(),static_cast<size_t>(target));
            const auto json=autil::legacy::ToJsonString(configs.at(""),true);
            EXPECT_NE(json.find("tp0_F"),std::string::npos);
            for(int rank=1;rank<target;++rank) EXPECT_NE(json.find("tp"+std::to_string(rank)+"_F"),std::string::npos);
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    KVCacheConfig config;
    config.kvcm_server_address="unused";
    config.kvcm_instance_group="linshuo_decode_remote_native_completion_test";
    config.kvcm_instance_id_salt="native_completion_v1";
    config.kvcm_model_sdk_config="[{\"type\":\"pace\"}]";
    ParallelismConfig parallel;
    parallel.tp_size=1;parallel.dp_size=target;parallel.ep_size=target;
    parallel.role_type=RoleType::DECODE;
    parallel.prefill_cp_config.prefill_cp_size=target;
    parallel.prefill_cp_config.kv_cache_sharded=sharded;
    RuntimeConfig runtime;runtime.model_name="completion_test";
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config,config,runtime,
        parallel,SpeculativeExecutionConfig{},nullptr,false,nullptr,client));
    ASSERT_TRUE(initSingleRank(*backend.backend,environment));

    bool finishing=false;int starts=0,saves=0,finishes=0;
    kv_cache_manager::WriteLocation location;
    location.write_session_id="finished_session";
    location.block_mask=kv_cache_manager::BlockMaskOffset{0};
    for (int k=0;k<checkpoints;++k) {
        kv_cache_manager::Location one;
        for (int rank=0;rank<target;++rank) one.push_back({kvcm::genLocationSpecName(rank,"F"+tag),"key"+std::to_string(k)+"r"+std::to_string(rank)});
        location.locations.push_back(one);
    }
    EXPECT_CALL(*client,getWriteLocation(_,_,_,_,_,_,_)).WillOnce(Invoke(
        [&](const auto&,const auto&,const auto& keys,const auto&,const auto&,auto,auto) {
            EXPECT_TRUE(finishing);++starts;EXPECT_EQ(keys,canonical_keys);
            return std::make_pair(true,location);
        }));
    EXPECT_CALL(*client,saveKvCachesForTag(_,_,_,_)).Times(target*checkpoints).WillRepeatedly(Invoke(
        [&](const std::string& wiretag,const auto& uris,const auto& buffers,const auto&) {
            EXPECT_TRUE(finishing);++saves;EXPECT_EQ(buffers.size(),1u);EXPECT_EQ(uris.size(),1u);
            EXPECT_EQ(wiretag,tag+"#cp"+uris[0].back());
            EXPECT_EQ(buffers[0].iovs.size(),1u);
            const auto& iov=buffers[0].iovs[0];
            EXPECT_EQ(iov.type,kv_cache_manager::MemoryType::CPU);
            EXPECT_GT(iov.size,0u);
            const int key=uris[0].at(3)-'0',rank=uris[0].back()-'0';
            const auto* bytes=static_cast<const unsigned char*>(iov.base);
            EXPECT_TRUE(std::all_of(bytes,bytes+iov.size,[&](unsigned char b){return b==shards*key+(sharded ? rank : 0)+1;}));
            return std::make_pair(true,kv_cache_manager::UriStrVec{"actual_"+uris[0]});
        }));
    EXPECT_CALL(*client,finishWrite(_,_,"finished_session",_,_)).WillOnce(Invoke(
        [&](const auto&,const auto&,const auto&,const auto&,const auto& actual) {
            ++finishes;EXPECT_EQ(actual.size(),static_cast<size_t>(checkpoints));
            for (const auto& key:actual) for (const auto& unit:key) EXPECT_EQ(unit.uri.rfind("actual_",0),0u);
            return true;
        }));
    ASSERT_TRUE(backend->captureCompletedDecode(resource,end,false));
    EXPECT_EQ(starts,0);EXPECT_EQ(saves,0);ASSERT_NE(resource.decodeCompletionSnapshot(),nullptr);
    // Invalid partial checkpoint must not replace the last complete snapshot.
    ASSERT_TRUE(backend->captureCompletedDecode(resource,end+1,false));
    EXPECT_EQ(resource.decodeCompletionSnapshot()->computed_end,end);
    finishing=true;
    ASSERT_TRUE(backend->captureCompletedDecode(resource,end+132,true));
    EXPECT_EQ(starts,1);EXPECT_EQ(saves,target*checkpoints);EXPECT_EQ(finishes,1);
    ASSERT_EQ(resource.decodeCompletionSnapshot(),nullptr);
    ASSERT_TRUE(backend->captureCompletedDecode(resource,end+132,true));
    EXPECT_EQ(starts,1);EXPECT_EQ(finishes,1);
    environment.device_pool->decRef(*blocks);
}

INSTANTIATE_TEST_SUITE_P(CP2AndCP4BothLayouts,NativeDtopCompletionTest,
    ::testing::Combine(::testing::Values(2,4),::testing::Bool()));

}}
