#include "kv_cache_manager/common/cache_origin.h"

#include <vector>

#include "gtest/gtest.h"

namespace kv_cache_manager {
namespace {
const std::string kSession = "0123456789abcdefghijklmnopqrstuv";
const std::string kOrigin = "v1." + kSession + ".7680.7680.8192.8300.8301";
const std::string kAllocated = "pace://pool/page?blkid=17&size=4096";

TEST(CacheOriginTest, ParsesActualComputedEndpointsWithoutMultiplyingCpCopies) {
    DecodeCacheOrigin parsed;
    ASSERT_TRUE(DecodeCacheOrigin::Parse(kOrigin, parsed));
    EXPECT_EQ(parsed.session, kSession);
    EXPECT_EQ(parsed.end - parsed.handoff_end, 512);
    EXPECT_EQ(parsed.computed_end, 8300);
    EXPECT_EQ(parsed.sampled_end, 8301);
}

TEST(CacheOriginTest, MarkerDetectionDoesNotMistakePathOrValuesForTheParameter) {
    EXPECT_TRUE(HasDecodeOriginParameter("pace://p/b?dtop_origin="));
    EXPECT_TRUE(HasDecodeOriginParameter("pace://p/b?size=1&dtop_origin=x"));
    EXPECT_TRUE(HasDecodeOriginParameter("pace://p/b?dtop_origin"));
    EXPECT_FALSE(HasDecodeOriginParameter("pace://p/dtop_origin=foo"));
    EXPECT_FALSE(HasDecodeOriginParameter("pace://p/b?other=dtop_origin=x"));
    EXPECT_FALSE(HasDecodeOriginParameter("pace://p/b?not_dtop_origin=x"));
}

TEST(CacheOriginTest, RejectsMalformedUnboundedOrInconsistentEndpoints) {
    for (const auto &bad : std::vector<std::string>{"", "v2." + kSession + ".0.0.1.1.2",
                           "v1.short.0.0.1.1.2", "v1." + kSession + ".0.0.1.1.2.extra",
                           "v1." + kSession + ".0.0.1.1", "v1." + kSession + ".0.-1.1.1.2",
                           "v1." + kSession + ".0.00.1.1.2", "v1." + kSession + ".2.0.1.1.2",
                           "v1." + kSession + ".0.1.1.1.2", "v1." + kSession + ".0.0.2.1.2",
                           "v1." + kSession + ".0.0.1.1.3",
                           "v1." + kSession + ".0.0.1.9223372036854775807.9223372036854775808",
                           "v1." + kSession + ".0.0.1.18446744073709551616.2"}) {
        SCOPED_TRACE(bad);
        DecodeCacheOrigin parsed;
        parsed.end = 123;
        EXPECT_FALSE(DecodeCacheOrigin::Parse(bad, parsed));
        EXPECT_EQ(parsed.end, 123); // No partially accepted output on rejection.
    }
}

TEST(CacheOriginTest, PreservesExactOldUriAndAcceptsOnlyMetadataExtension) {
    std::string out = "old";
    ASSERT_TRUE(ValidateFinishedCacheUri(kAllocated, kAllocated, kSession, out));
    EXPECT_TRUE(out.empty());
    StandardUri uri(kAllocated);
    const auto actual = uri.ToUriStringWithExtraParam(DecodeCacheOrigin::kUriParameter, kOrigin);
    ASSERT_TRUE(ValidateFinishedCacheUri(kAllocated, actual, kSession, out));
    EXPECT_EQ(out, kOrigin);
}

TEST(CacheOriginTest, RejectsForeignSessionCoreUriChangeDuplicateOrUnknownParams) {
    StandardUri uri(kAllocated);
    const auto actual = uri.ToUriStringWithExtraParam(DecodeCacheOrigin::kUriParameter, kOrigin);
    for (const auto &bad : std::vector<std::string>{actual + "&size=4096", actual + "&unexpected=1",
                           actual + "&dtop_origin=" + kOrigin, "pace://pool/page?blkid=18&size=4096"}) {
        std::string out = "unchanged";
        EXPECT_FALSE(ValidateFinishedCacheUri(kAllocated, bad, kSession, out));
        EXPECT_EQ(out, "unchanged");
    }
    std::string out;
    EXPECT_FALSE(ValidateFinishedCacheUri(kAllocated, actual, std::string(32, 'x'), out));
    EXPECT_FALSE(ValidateFinishedCacheUri(actual, actual, kSession, out));
}
} // namespace
} // namespace kv_cache_manager
