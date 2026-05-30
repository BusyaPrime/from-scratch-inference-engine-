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

// The cuBLAS GEMM (strict fp32) must match the CPU linear, with and without bias.
TEST(CudaKernels, LinearMatchesCpu) {
    std::mt19937 rng(17);
    const int64_t rows = 4;
    const int64_t in_dim = 896;
    const int64_t out_dim = 4864;

    const engine::Tensor x = random_tensor({rows, in_dim}, rng, 0.05f);
    const engine::Tensor weight = random_tensor({out_dim, in_dim}, rng, 0.05f);
    const engine::Tensor reference = engine::linear(x, weight);

    std::vector<float> out(static_cast<std::size_t>(rows * out_dim));
    engine::cuda::linear(x.data(), weight.data(), nullptr, out.data(), rows, in_dim, out_dim);

    for (int64_t i = 0; i < rows * out_dim; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], reference.data()[i], 1e-3f) << "index " << i;
    }
}

TEST(CudaKernels, LinearWithBiasMatchesCpu) {
    std::mt19937 rng(19);
    const int64_t rows = 3;
    const int64_t in_dim = 896;
    const int64_t out_dim = 128;

    const engine::Tensor x = random_tensor({rows, in_dim}, rng, 0.05f);
    const engine::Tensor weight = random_tensor({out_dim, in_dim}, rng, 0.05f);
    const engine::Tensor bias = random_tensor({out_dim}, rng, 0.1f);
    const engine::Tensor reference = engine::linear(x, weight, bias);

    std::vector<float> out(static_cast<std::size_t>(rows * out_dim));
    engine::cuda::linear(x.data(), weight.data(), bias.data(), out.data(), rows, in_dim, out_dim);

    for (int64_t i = 0; i < rows * out_dim; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], reference.data()[i], 1e-3f) << "index " << i;
    }
}

namespace {

void check_attention(int64_t q_len, int64_t total, int64_t query_offset, unsigned seed) {
    std::mt19937 rng(seed);
    const int64_t n_heads = 14;   // Qwen2.5-0.5B query heads
    const int64_t n_kv_heads = 2; // grouped-query (group size 7)
    const int64_t head_dim = 64;

    const engine::Tensor q = random_tensor({q_len, n_heads * head_dim}, rng);
    const engine::Tensor k = random_tensor({total, n_kv_heads * head_dim}, rng);
    const engine::Tensor v = random_tensor({total, n_kv_heads * head_dim}, rng);
    const engine::Tensor reference =
        engine::attention(q, k, v, n_heads, n_kv_heads, head_dim, query_offset);

    std::vector<float> out(static_cast<std::size_t>(q_len * n_heads * head_dim));
    engine::cuda::attention(q.data(),
                            k.data(),
                            v.data(),
                            out.data(),
                            q_len,
                            total,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            query_offset);

    for (int64_t i = 0; i < q_len * n_heads * head_dim; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], reference.data()[i], 1e-4f) << "index " << i;
    }
}

} // namespace

// Causal grouped-query attention must match the CPU twin for a prefill chunk and a decode step.
TEST(CudaKernels, AttentionPrefillMatchesCpu) {
    check_attention(/*q_len=*/5, /*total=*/12, /*query_offset=*/7, /*seed=*/23);
}

TEST(CudaKernels, AttentionDecodeMatchesCpu) {
    check_attention(/*q_len=*/1, /*total=*/20, /*query_offset=*/19, /*seed=*/29);
}
