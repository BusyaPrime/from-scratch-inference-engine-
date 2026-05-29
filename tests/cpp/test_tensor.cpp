// SPDX-License-Identifier: Apache-2.0
#include "engine/dtype.hpp"
#include "engine/ops.hpp"
#include "engine/tensor.hpp"

#include <gtest/gtest.h>

TEST(Tensor, ShapeAndNumel) {
    const engine::Tensor t({2, 3});
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
}

TEST(Tensor, RejectsShapeDataMismatch) {
    EXPECT_THROW(engine::Tensor({2, 2}, {1.0f, 2.0f, 3.0f}), std::invalid_argument);
}

TEST(Matmul, MultipliesByIdentity) {
    const engine::Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const engine::Tensor id({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    const engine::Tensor c = engine::matmul(a, id);
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 4.0f);
}

TEST(Matmul, RejectsInnerDimMismatch) {
    const engine::Tensor a({2, 3});
    const engine::Tensor b({2, 2});
    EXPECT_THROW(engine::matmul(a, b), std::invalid_argument);
}

TEST(DType, HalfToFloat) {
    EXPECT_FLOAT_EQ(engine::f16_to_f32(0x3C00), 1.0f);
    EXPECT_FLOAT_EQ(engine::f16_to_f32(0xC000), -2.0f);
    EXPECT_FLOAT_EQ(engine::f16_to_f32(0x0000), 0.0f);
}

TEST(DType, BFloatToFloat) {
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0x3F80), 1.0f);
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0xC000), -2.0f);
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0x0000), 0.0f);
}
