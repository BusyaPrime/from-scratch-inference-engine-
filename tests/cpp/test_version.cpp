// SPDX-License-Identifier: Apache-2.0
#include "engine/version.hpp"

#include <gtest/gtest.h>

TEST(Version, IsNotEmpty) {
    EXPECT_FALSE(engine::version().empty());
}
