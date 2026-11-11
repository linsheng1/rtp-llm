#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace {
CompleteTokenIdsPtr tokens(std::initializer_list<int32_t> values) {
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor(std::vector<int32_t>(values), torch::kInt32);
    auto result = std::make_shared<CompleteTokenIds>(1, 1, 32, 2);
    result->init(input);
    return result;
}
BatchKVCacheResourcePtr resource(bool lookahead) {
    auto result = std::make_shared<BatchKVCacheResource>();
    result->resetBatchSize(1);
    result->cache_key_lookahead = lookahead;
    return result;
}
TEST(MtpCacheLookahead, FreshConsumerMatchesPublisherAndSeparatesShiftedDependency) {
    auto published = resource(true), fresh = resource(true), changed = resource(true);
    initCacheKeys(published, tokens({1,2,3,4,5,6,7}), 2, 37);
    initCacheKeys(fresh, tokens({1,2,3,4,5,6,7,8,9}), 2, 37);
    initCacheKeys(changed, tokens({1,2,3,4,90,6,7}), 2, 37);
    EXPECT_EQ(published->cacheKeys()[0], fresh->cacheKeys()[0]);
    EXPECT_EQ(published->cacheKeys()[1], fresh->cacheKeys()[1]);
    EXPECT_EQ(published->cacheKeys()[2], fresh->cacheKeys()[2]);
    EXPECT_EQ(published->cacheKeys()[0], changed->cacheKeys()[0]);
    EXPECT_NE(published->cacheKeys()[1], changed->cacheKeys()[1]);
}
TEST(MtpCacheLookahead, IncrementalKeysKeepRawPrefixSeparateFromLookahead) {
    auto ids = tokens({1,2,3,4,5,6,7}); ids->setSeqLength(3);
    auto incremental = resource(true), fresh = resource(true);
    initCacheKeys(incremental, ids, 2, 37);
    ids->setSeqLength(7);
    updateCacheKeys(incremental, ids, 2, 37);
    initCacheKeys(fresh, ids, 2, 37);
    ASSERT_GE(incremental->cacheKeys().size(), 3u);
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(incremental->cacheKeys()[i], fresh->cacheKeys()[i]);
    // Final token trimming must use the newly retained token history.
    ids->setSeqLength(5); initCacheKeys(incremental, ids, 2, 37);
    initCacheKeys(fresh, tokens({1,2,3,4,5}), 2, 37);
    EXPECT_EQ(incremental->cacheKeys(), fresh->cacheKeys());
}
TEST(MtpCacheLookahead, OrdinaryPrefixKeysRetainOriginalContract) {
    auto a = resource(false), b = resource(false);
    initCacheKeys(a, tokens({1,2,3,4,5}), 2, 37);
    initCacheKeys(b, tokens({1,2,3,4,90}), 2, 37);
    EXPECT_EQ(a->cacheKeys()[1], b->cacheKeys()[1]);
    int32_t prefix[]{1,2,3,4};
    EXPECT_EQ(a->cacheKeys()[1], hashInt64Array(37, prefix, prefix + 4));
}
}
}
