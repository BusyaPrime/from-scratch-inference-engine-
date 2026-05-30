// SPDX-License-Identifier: Apache-2.0
#include "engine/nn.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

TEST(NN, LinearWithBias) {
    const engine::Tensor x({1, 2}, {1.0f, 2.0f});
    const engine::Tensor w({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}); // [out=2, in=2]
    const engine::Tensor b({2}, {10.0f, 20.0f});
    const engine::Tensor y = engine::linear(x, w, b);
    ASSERT_EQ(y.dim(0), 1);
    ASSERT_EQ(y.dim(1), 2);
    EXPECT_FLOAT_EQ(y.data()[0], 13.0f); // 1*1 + 2*1 + 10
    EXPECT_FLOAT_EQ(y.data()[1], 23.0f);
}

TEST(NN, LinearNoBiasSelectsColumns) {
    const engine::Tensor x({1, 3}, {1.0f, 2.0f, 3.0f});
    const engine::Tensor w({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    const engine::Tensor y = engine::linear(x, w);
    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 2.0f);
}

TEST(NN, RmsNorm) {
    const engine::Tensor x({1, 2}, {3.0f, 4.0f});
    const engine::Tensor w({2}, {1.0f, 1.0f});
    const engine::Tensor y = engine::rms_norm(x, w, 0.0);
    const float scale = 1.0f / std::sqrt(12.5f); // mean(9, 16) = 12.5
    EXPECT_NEAR(y.data()[0], 3.0f * scale, 1e-5f);
    EXPECT_NEAR(y.data()[1], 4.0f * scale, 1e-5f);
}

TEST(NN, SiluMul) {
    const engine::Tensor g({2}, {0.0f, 1.0f});
    const engine::Tensor u({2}, {5.0f, 2.0f});
    const engine::Tensor y = engine::silu_mul(g, u);
    EXPECT_NEAR(y.data()[0], 0.0f, 1e-6f); // silu(0) = 0
    EXPECT_NEAR(y.data()[1], (1.0f / (1.0f + std::exp(-1.0f))) * 2.0f, 1e-5f);
}

TEST(NN, Embedding) {
    const engine::Tensor w({3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    const engine::Tensor e = engine::embedding(w, {2, 0});
    EXPECT_FLOAT_EQ(e.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(e.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(e.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(e.data()[3], 1.0f);
}

TEST(NN, EmbeddingRejectsOutOfRange) {
    const engine::Tensor w({2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    EXPECT_THROW((void)engine::embedding(w, {5}), std::out_of_range);
}

TEST(NN, RopeRotatesByAngle) {
    // head_dim=2, single head: x=[1,0], pos=1, inv_freq[0]=theta^0=1, angle=1.
    engine::Tensor x({1, 2}, {1.0f, 0.0f});
    engine::rope_inplace(x, /*n_heads=*/1, /*head_dim=*/2, /*theta=*/10000.0, {1});
    EXPECT_NEAR(x.data()[0], static_cast<float>(std::cos(1.0)), 1e-5f);
    EXPECT_NEAR(x.data()[1], static_cast<float>(std::sin(1.0)), 1e-5f);
}

TEST(NN, AttentionCausalFirstPositionIsValue) {
    // S=2, one head, head_dim=2: position 0 attends only to itself -> out[0] == v[0].
    const engine::Tensor q({2, 2}, {0.3f, -0.1f, 0.5f, 0.2f});
    const engine::Tensor k({2, 2}, {0.1f, 0.4f, -0.2f, 0.3f});
    const engine::Tensor v({2, 2}, {7.0f, 8.0f, 1.0f, 2.0f});
    const engine::Tensor o = engine::attention(q, k, v, /*n_heads=*/1, /*n_kv=*/1, /*head_dim=*/2);
    EXPECT_NEAR(o.data()[0], 7.0f, 1e-5f);
    EXPECT_NEAR(o.data()[1], 8.0f, 1e-5f);
}
