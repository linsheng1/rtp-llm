#pragma once

#include <cstdlib>
#include <string>
#include <gtest/gtest.h>

namespace rtp_llm::test {
// Set the environment with the caller's EnvGuard BEFORE entering this helper.
// A threadsafe death test re-execs the binary with that environment, including
// fixture setup; it does not inherit a feature value cached by an earlier test.
// Fatal/nonfatal assertions in the child remain failures in the parent.
template<typename Function>
void withDecodePublicationStartup(Function&& function) {
    struct RestoreStyle {
        std::string previous = ::testing::GTEST_FLAG(death_test_style);
        ~RestoreStyle() { ::testing::GTEST_FLAG(death_test_style) = previous; }
    } restore;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT({
        function();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
    }, ::testing::ExitedWithCode(0), "");
}
}  // namespace rtp_llm::test
