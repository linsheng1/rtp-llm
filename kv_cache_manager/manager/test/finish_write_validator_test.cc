#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>

#include "kv_cache_manager/common/cache_origin.h"
#include "kv_cache_manager/manager/finish_write_validator.h"

namespace kv_cache_manager {
namespace {
const std::string kSession = "0123456789abcdefghijklmnopqrstuv";

WriteLocationManager::WriteLocationInfo Allocation() {
    WriteLocationManager::WriteLocationInfo result;
    result.instance_id = "instance";
    result.keys = {10, 20};
    result.location_ids = {"a", "b"};
    for (size_t i = 0; i < result.keys.size(); ++i) {
        auto location = std::make_shared<CacheLocation>(
            result.location_ids[i],
            CLS_WRITING,
            DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
            2,
            std::vector<LocationSpec>{{"full", "pace://pool/page?blkid=" + std::to_string(i) + "&size=4096"},
                                      {"state", "pace://pool/state?blkid=" + std::to_string(i) + "&size=512"}});
        location->set_create_time(123);
        location->set_allocation_session_id(kSession);
        result.allocated_locations.push_back(std::move(location));
    }
    return result;
}

std::vector<CacheLocation> Actual(const WriteLocationManager::WriteLocationInfo &allocation) {
    std::vector<CacheLocation> result;
    for (size_t i = 0; i < allocation.keys.size(); ++i) {
        result.push_back(*allocation.allocated_locations[i]);
        const auto origin = "v1." + kSession + ".7680." + std::to_string(7680 + 512 * i) + "." +
                            std::to_string(8192 + 512 * i) + ".8704.8705";
        for (auto &spec : result.back().mutable_location_specs()) {
            spec.set_uri(StandardUri(spec.uri()).ToUriStringWithExtraParam("dtop_origin", origin));
        }
    }
    return result;
}

TEST(FinishWriteValidatorTest, CompleteSpecsBecomeOneImmutableServingValue) {
    const auto allocation = Allocation();
    auto actual = Actual(allocation);
    std::reverse(actual[0].mutable_location_specs().begin(), actual[0].mutable_location_specs().end());
    CacheLocationVector out;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    ASSERT_EQ(out.size(), 2u);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_EQ(out[i]->status(), CLS_SERVING);
        EXPECT_EQ(out[i]->id(), allocation.location_ids[i]);
        EXPECT_EQ(out[i]->create_time(), 123);
        EXPECT_EQ(out[i]->allocation_session_id(), kSession);
        EXPECT_EQ(out[i]->location_specs()[0].name(), "full");
        EXPECT_NE(out[i]->location_specs()[0].uri().find("dtop_origin="), std::string::npos);
        EXPECT_EQ(allocation.allocated_locations[i]->status(), CLS_WRITING);
        EXPECT_EQ(allocation.allocated_locations[i]->location_specs()[0].uri().find("dtop_origin="), std::string::npos);
    }
}

TEST(CompletedDecodePublishMetricTest, CompleteSessionCountsUnionOnceAcrossSpecCopies) {
    const auto allocation = Allocation();
    const auto actual = Actual(allocation);
    CacheLocationVector serving;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, serving));
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, kSession, serving), 1024);
    serving[1] = std::make_shared<CacheLocation>(*serving[0]);
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, kSession, serving), 512);
}

TEST(CompletedDecodePublishMetricTest, PartialMaskAndAnyOverallErrorCountZero) {
    const auto allocation = Allocation();
    CacheLocationVector serving;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, Actual(allocation), serving));
    EXPECT_EQ(CountCompletedDecodePublishTokens(false, 2, 2, kSession, serving), 0);
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 1, kSession, serving), 0);
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 3, 3, kSession, serving), 0);
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 0, 0, kSession, {}), 0);
}

TEST(CompletedDecodePublishMetricTest, LegacyNoSourceAndInconsistentSourcesCountZero) {
    const auto allocation = Allocation();
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, kSession, allocation.allocated_locations), 0);
    CacheLocationVector serving;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, Actual(allocation), serving));
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, std::string(32, 'b'), serving), 0);
    auto mixed = std::make_shared<CacheLocation>(*serving[0]);
    mixed->mutable_location_specs().back().set_uri(allocation.allocated_locations[0]->location_specs()[0].uri());
    serving[0] = std::move(mixed);
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, kSession, serving), 0);
}

TEST(CompletedDecodePublishMetricTest, PrefillOnlyCoverageHasNoDecodePublishedTokens) {
    const auto allocation = Allocation();
    auto actual = Actual(allocation);
    for (size_t i = 0; i < actual.size(); ++i) {
        for (size_t spec = 0; spec < actual[i].location_specs().size(); ++spec) {
            const auto value = "v1." + kSession + ".8704." + std::to_string(7680 + 512 * i) + "." +
                               std::to_string(8192 + 512 * i) + ".8704.8705";
            actual[i].mutable_location_specs()[spec].set_uri(StandardUri(allocation.allocated_locations[i]
                ->location_specs()[spec].uri()).ToUriStringWithExtraParam("dtop_origin", value));
        }
    }
    CacheLocationVector serving;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, serving));
    EXPECT_EQ(CountCompletedDecodePublishTokens(true, 2, 2, kSession, serving), 0);
}

TEST(FinishWriteValidatorTest, ValidatesEntireVectorBeforeProducingAnyReplacement) {
    const auto allocation = Allocation();
    auto actual = Actual(allocation);
    actual.back().mutable_location_specs()[1].set_uri("pace://foreign/page?size=512");
    CacheLocationVector out{allocation.allocated_locations.front()};
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], allocation.allocated_locations.front());
    EXPECT_EQ(out[0]->status(), CLS_WRITING);
}

TEST(FinishWriteValidatorTest, ActualSdkSentinelsUseAuthoritativeAllocationGeometry) {
    const auto allocation = Allocation();
    auto actual = Actual(allocation);
    for (auto &location : actual) {
        location.set_type(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN); // ST_UNSPECIFIED
        location.set_spec_size(std::numeric_limits<size_t>::max());    // wire int32 -1
    }
    CacheLocationVector out;
    ASSERT_EQ(EC_OK, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0]->type(), allocation.allocated_locations[0]->type());
    EXPECT_EQ(out[0]->spec_size(), 2u);
    actual[0].set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    actual[0].set_type(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
    actual[0].set_spec_size(1);
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
}

TEST(FinishWriteValidatorTest, MixedMarkedAndUnmarkedLocationsNeverSilentlyDropOrigin) {
    const auto allocation = Allocation();
    auto actual = Actual(allocation);
    actual[1] = *allocation.allocated_locations[1];
    CacheLocationVector out;
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    EXPECT_TRUE(out.empty());
}

TEST(FinishWriteValidatorTest, RejectsMaskInstanceDuplicatesPartialOrMixedSource) {
    const auto original = Allocation();
    const auto good = Actual(original);
    CacheLocationVector out;
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "foreign", kSession, size_t{2}, good, out));
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "instance", kSession, size_t{3}, good, out));
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "instance", kSession, size_t{1}, good, out));
    auto duplicate = original;
    duplicate.keys[1] = duplicate.keys[0];
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(duplicate, "instance", kSession, size_t{2}, good, out));
    auto partial = good;
    partial[1].mutable_location_specs().pop_back();
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "instance", kSession, size_t{2}, partial, out));
    auto mixed = good;
    mixed[1].mutable_location_specs()[1] = original.allocated_locations[1]->location_specs()[1];
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "instance", kSession, size_t{2}, mixed, out));
    auto same_spec = good;
    same_spec[1].mutable_location_specs()[1].set_name("full");
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(original, "instance", kSession, size_t{2}, same_spec, out));
    EXPECT_TRUE(out.empty());
}

TEST(FinishWriteValidatorTest, SelectedKeysArePositionalAndUnmodifiedUrisAreCompatible) {
    const auto allocation = Allocation();
    std::vector<CacheLocation> actual{*allocation.allocated_locations[1]};
    CacheLocationVector out;
    ASSERT_EQ(EC_OK,
              ValidateFinishedLocations(allocation, "instance", kSession, BlockMaskVector{false, true}, actual, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->id(), "b");
    EXPECT_EQ(out[0]->location_specs()[0].uri(), actual[0].location_specs()[0].uri());
}

TEST(FinishWriteValidatorTest, AllocationVersionSurvivesCopyAndJsonButCannotBeRebound) {
    auto allocation = Allocation();
    const auto actual = Actual(allocation);
    CacheLocation roundtrip;
    ASSERT_TRUE(roundtrip.FromJsonString(allocation.allocated_locations[0]->ToJsonString()));
    EXPECT_EQ(roundtrip.allocation_session_id(), kSession);
    EXPECT_EQ(roundtrip.ToJsonString(), allocation.allocated_locations[0]->ToJsonString());
    auto replacement = std::make_shared<CacheLocation>(roundtrip);
    replacement->set_allocation_session_id(std::string(32, 'x'));
    allocation.allocated_locations[0] = replacement;
    CacheLocationVector out;
    EXPECT_EQ(EC_BADARGS, ValidateFinishedLocations(allocation, "instance", kSession, size_t{2}, actual, out));
    replacement->set_allocation_session_id("");
    EXPECT_EQ(replacement->ToJsonString().find("allocation_session_id"), std::string::npos);
    CacheLocation legacy;
    ASSERT_TRUE(legacy.FromJsonString(replacement->ToJsonString()));
    EXPECT_TRUE(legacy.allocation_session_id().empty());
}

TEST(FinishWriteSessionTest, RejectionKeepsCleanupCapableSessionAndThenClaimsOnce) {
    WriteLocationManager sessions;
    auto allocation = Allocation();
    sessions.Put(kSession, {10, 20}, {"a", "b"}, 10, [](auto) {}, "instance", allocation.allocated_locations);
    WriteLocationManager::WriteLocationInfo out;
    EXPECT_EQ(EC_BADARGS, sessions.TakeValidated(kSession, [](const auto &) { return EC_BADARGS; }, out));
    EXPECT_EQ(sessions.ExpireSize(), 1u);
    EXPECT_TRUE(sessions.HasLocationId("a"));
    EXPECT_EQ(EC_OK, sessions.TakeValidated(kSession, [](const auto &) { return EC_OK; }, out));
    EXPECT_EQ(out.instance_id, "instance");
    EXPECT_EQ(out.allocated_locations, allocation.allocated_locations);
    EXPECT_EQ(EC_NOENT, sessions.TakeValidated(kSession, [](const auto &) { return EC_OK; }, out));
    EXPECT_FALSE(sessions.HasLocationId("a"));
}

TEST(FinishWriteSessionTest, RejectedSessionStillRunsOriginalTimeoutCleanup) {
    WriteLocationManager sessions;
    bool cleaned = false;
    sessions.Put(kSession, {10}, {"a"}, 10, [&](auto info) {
        cleaned = true;
        EXPECT_EQ(info->keys, std::vector<int64_t>{10});
    });
    WriteLocationManager::WriteLocationInfo out;
    EXPECT_EQ(EC_BADARGS, sessions.TakeValidated(kSession, [](const auto &) { return EC_BADARGS; }, out));
    sessions.DoCleanup();
    EXPECT_TRUE(cleaned);
    EXPECT_EQ(sessions.ExpireSize(), 0u);
}

TEST(FinishWriteSessionTest, ConcurrentFinishOrCleanupHasOnlyOneWinner) {
    WriteLocationManager sessions;
    std::atomic<int> winners{0};
    sessions.Put(kSession, {10}, {"a"}, 10, [&](auto) { ++winners; });
    auto claim = [&] {
        WriteLocationManager::WriteLocationInfo out;
        const auto ec = sessions.TakeValidated(kSession, [](const auto &) { return EC_OK; }, out);
        if (ec == EC_OK)
            ++winners;
        else
            EXPECT_EQ(ec, EC_NOENT);
    };
    std::thread first(claim), second(claim), expiry([&] { sessions.DoCleanup(); });
    first.join();
    second.join();
    expiry.join();
    EXPECT_EQ(winners.load(), 1);
    EXPECT_EQ(sessions.ExpireSize(), 0u);
}
} // namespace
} // namespace kv_cache_manager
