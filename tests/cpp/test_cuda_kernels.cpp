// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/device.hpp"
#include "engine/nn.hpp"
#include "engine/tensor.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

engine::Tensor random_tensor(std::vector<int64_t> shape, std::mt19937& rng, float sd = 1.0f) {
    engine::Tensor t(std::move(shape));
    std::normal_distribution<float> nd(0.0f, sd);
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data()[i] = nd(rng);
    }
    return t;
}

} // namespace

// The GPU RMSNorm must match the CPU reference, which is itself anchored to transformers.
TEST(CudaKernels, RmsNormMatchesCpu) {
    std::mt19937 rng(7);
    const int64_t rows = 5;
    const int64_t dim = 896; // Qwen2.5-0.5B hidden size
    const double eps = 1e-6;

    const engine::Tensor x = random_tensor({rows, dim}, rng);
    const engine::Tensor weight = random_tensor({dim}, rng, 0.5f);
    const engine::Tensor reference = engine::rms_norm(x, weight, eps);

    std::vector<float> out(static_cast<std::size_t>(rows * dim));
    engine::cuda::rms_norm(x.data(), weight.data(), out.data(), rows, dim, eps);

    for (int64_t i = 0; i < rows * dim; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], reference.data()[i], 1e-4f) << "index " << i;
    }
}

// The GPU SwiGLU must match the CPU reference elementwise.
TEST(CudaKernels, SiluMulMatchesCpu) {
    std::mt19937 rng(11);
    const int64_t n = 4864; // Qwen2.5-0.5B intermediate size

    const engine::Tensor gate = random_tensor({n}, rng);
    const engine::Tensor up = random_tensor({n}, rng);
    const engine::Tensor reference = engine::silu_mul(gate, up);

    std::vector<float> out(static_cast<std::size_t>(n));
    engine::cuda::silu_mul(gate.data(), up.data(), out.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], reference.data()[i], 1e-4f) << "index " << i;
    }
}

// The GPU RoPE must reproduce the CPU half-rotation layout in place.
TEST(CudaKernels, RopeMatchesCpu) {
    std::mt19937 rng(13);
    const int64_t rows = 6;
    const int64_t n_heads = 14; // Qwen2.5-0.5B query heads
    const int64_t head_dim = 64;
    const double theta = 1.0e6;

    const engine::Tensor x = random_tensor({rows, n_heads * head_dim}, rng);
    std::vector<int64_t> positions(static_cast<std::size_t>(rows));
    for (int64_t t = 0; t < rows; ++t) {
        positions[static_cast<std::size_t>(t)] = t + 3; // arbitrary non-zero offset
    }

    engine::Tensor reference = x; // copy, then rotate on the CPU
    engine::rope_inplace(reference, n_heads, head_dim, theta, positions);

    std::vector<float> gpu(x.data(), x.data() + x.numel());
    engine::cuda::rope(gpu.data(), rows, n_heads, head_dim, theta, positions.data());

    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(gpu[static_cast<std::size_t>(i)], reference.data()[i], 1e-4f) << "index " << i;
    }
}
