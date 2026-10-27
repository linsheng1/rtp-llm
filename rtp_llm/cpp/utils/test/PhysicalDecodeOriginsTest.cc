#include "rtp_llm/cpp/utils/PhysicalDecodeOrigins.h"
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
PhysicalDecodeOrigins::Origin origin() {
    auto parsed = DecodeCacheOrigin::parse("v1." + std::string(32, 'a') + ".7680.7680.8192.8200.8201");
    return std::make_shared<const DecodeCacheOrigin>(*parsed);
}

TEST(PhysicalDecodeOriginsTest, BindRequiresTheSameLiveAllocation) {
    PhysicalDecodeOrigins table(4);
    EXPECT_EQ(table.incarnation(1), uint64_t{0});
    EXPECT_FALSE(table.bind(1, 0, origin()));
    const auto stamp = table.beginAllocation(1);
    ASSERT_NE(stamp, uint64_t{0});
    EXPECT_FALSE(table.get(1));
    const auto source = origin();
    ASSERT_TRUE(table.bind(1, stamp, source));
    EXPECT_EQ(table.get(1), source);
    EXPECT_FALSE(table.bind(1, stamp + 1, nullptr));
    EXPECT_EQ(table.get(1), source);
}

TEST(PhysicalDecodeOriginsTest, ReallocatedOrReusedDonorRejectsLateCompletion) {
    PhysicalDecodeOrigins table(3);
    const auto            first = table.beginAllocation(1);
    ASSERT_TRUE(table.bind(1, first, origin()));
    const auto donor = table.beginAllocation(1);  // same ID, different value
    EXPECT_NE(first, donor);
    EXPECT_FALSE(table.get(1));
    EXPECT_FALSE(table.bind(1, first, origin()));
    ASSERT_TRUE(table.bind(1, donor, origin()));
    table.release(1);
    EXPECT_EQ(table.incarnation(1), uint64_t{0});
    EXPECT_FALSE(table.get(1));
    EXPECT_FALSE(table.bind(1, donor, origin()));
    const auto next = table.beginAllocation(1);
    EXPECT_NE(next, donor);
    EXPECT_FALSE(table.bind(1, donor, origin()));
}

TEST(PhysicalDecodeOriginsTest, ClearingSourceDoesNotChangePayloadIncarnation) {
    PhysicalDecodeOrigins table(3);
    const auto            first  = table.beginAllocation(1);
    const auto            second = table.beginAllocation(2);
    ASSERT_TRUE(table.bind(1, first, origin()));
    ASSERT_TRUE(table.bind(2, second, origin()));
    EXPECT_TRUE(table.bind(1, first, nullptr));
    EXPECT_EQ(table.incarnation(1), first);
    EXPECT_FALSE(table.get(1));
    EXPECT_TRUE(table.get(2));
}

TEST(PhysicalDecodeOriginsTest, RejectsOutOfRangePhysicalAddresses) {
    PhysicalDecodeOrigins table(3);
    // The pool validates addresses before calling its metadata helper.
    EXPECT_THROW(table.beginAllocation(3), std::out_of_range);
    EXPECT_THROW(table.get(3), std::out_of_range);
}
}  // namespace
}  // namespace rtp_llm
