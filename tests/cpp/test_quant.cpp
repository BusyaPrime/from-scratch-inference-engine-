// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"
#include "engine/nn.hpp"
#include "engine/quant.hpp"
#include "engine/tensor.hpp"
#include "tiny_model.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

engine::Tensor random_tensor(std::vector<int64_t> shape, std::mt19937& rng, float sd) {
    engine::Tensor t(std::move(shape));
    std::normal_distribution<float> nd(0.0f, sd);
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data()[i] = nd(rng);
    }
    return t;
}

double relative_l2(const engine::Tensor& got, const engine::Tensor& ref) {
    double num = 0.0;
    double den = 0.0;
    for (int64_t i = 0; i < ref.numel(); ++i) {
        const double d = static_cast<double>(got.data()[i]) - static_cast<double>(ref.data()[i]);
        num += d * d;
        den += static_cast<double>(ref.data()[i]) * static_cast<double>(ref.data()[i]);
    }
    return std::sqrt(num / den);
}

} // namespace

// int8-weight linear must track the fp32 linear closely; per-row scales keep the error sub-percent.
TEST(Quant, Int8LinearTracksFp32) {
    std::mt19937 rng(5);
    const engine::Tensor x = random_tensor({4, 256}, rng, 0.1f);
    const engine::Tensor w = random_tensor({512, 256}, rng, 0.02f);

    const engine::Tensor reference = engine::linear(x, w);
    const engine::QuantizedMatrix q = engine::quantize_rowwise_int8(w);
    const engine::Tensor got = engine::linear_int8(x, q);

    EXPECT_LT(relative_l2(got, reference), 0.02);
}

// Round-to-nearest quantization keeps each dequantized weight within half a scale of the original.
TEST(Quant, RoundtripErrorBoundedByScale) {
    std::mt19937 rng(9);
    const int64_t out = 8;
    const int64_t in = 64;
    const engine::Tensor w = random_tensor({out, in}, rng, 0.05f);
    const engine::QuantizedMatrix q = engine::quantize_rowwise_int8(w);

    for (int64_t o = 0; o < out; ++o) {
        const float scale = q.scales[static_cast<std::size_t>(o)];
        for (int64_t i = 0; i < in; ++i) {
            const float dequant =
                static_cast<float>(q.data[static_cast<std::size_t>(o * in + i)]) * scale;
            EXPECT_LE(std::fabs(dequant - w.data()[o * in + i]), 0.5f * scale + 1e-6f);
        }
    }
}

TEST(Quant, Int8LinearWithBias) {
    std::mt19937 rng(11);
    const engine::Tensor x = random_tensor({3, 128}, rng, 0.1f);
    const engine::Tensor w = random_tensor({64, 128}, rng, 0.02f);
    const engine::Tensor bias = random_tensor({64}, rng, 0.1f);

    const engine::Tensor reference = engine::linear(x, w, bias);
    const engine::QuantizedMatrix q = engine::quantize_rowwise_int8(w);
    const engine::Tensor got = engine::linear_int8(x, q, bias);

    EXPECT_LT(relative_l2(got, reference), 0.05);
}

// A quantized model runs its forward through int8 weight-only matmuls and still tracks the
// fp32 logits. The default model stays exact, so fp32 parity is unaffected.
TEST(Quant, QuantizedModelForwardTracksFp32) {
    const std::vector<int64_t> ids = {1, 2, 3, 4, 5};
    const engine::Model fp32 = tiny::tiny_model();
    engine::Model quant = tiny::tiny_model();
    EXPECT_FALSE(quant.is_quantized());
    quant.quantize();
    EXPECT_TRUE(quant.is_quantized());

    const engine::Tensor reference = fp32.forward(ids);
    const engine::Tensor got = quant.forward(ids);
    ASSERT_EQ(got.dim(0), reference.dim(0));
    ASSERT_EQ(got.dim(1), reference.dim(1));
    EXPECT_LT(relative_l2(got, reference), 0.1); // weight-only int8 tracks the fp32 logits
}
