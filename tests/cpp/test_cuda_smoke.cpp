// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/device.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// Validates the CUDA toolchain end to end: a device is visible and a launched kernel's result,
// copied back to the host, matches the CPU computation.
TEST(Cuda, DeviceIsVisible) {
    EXPECT_GT(engine::cuda::device_count(), 0);
}

TEST(Cuda, SaxpyMatchesCpu) {
    const int64_t n = 1000;
    std::vector<float> x(static_cast<std::size_t>(n));
    std::vector<float> y(static_cast<std::size_t>(n));
    std::vector<float> out(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>(i) * 0.5f;
        y[static_cast<std::size_t>(i)] = static_cast<float>(n - i);
    }

    engine::cuda::saxpy(2.0f, x.data(), y.data(), out.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        EXPECT_FLOAT_EQ(out[idx], 2.0f * x[idx] + y[idx]);
    }
}
