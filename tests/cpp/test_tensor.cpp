// SPDX-License-Identifier: Apache-2.0
#include "engine/dtype.hpp"
#include "engine/ops.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

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

TEST(Matmul, RectangularValues) {
    // a [2,3] * b [3,2]; asymmetric values catch a transposed row/col-major mapping.
    const engine::Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const engine::Tensor b({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    const engine::Tensor c = engine::matmul(a, b);
    ASSERT_EQ(c.dim(0), 2);
    ASSERT_EQ(c.dim(1), 2);
    EXPECT_FLOAT_EQ(c.data()[0], 58.0f);  // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(c.data()[1], 64.0f);  // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(c.data()[2], 139.0f); // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(c.data()[3], 154.0f); // 4*8 + 5*10 + 6*12
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

TEST(DType, HalfSpecialValues) {
    EXPECT_TRUE(std::isinf(engine::f16_to_f32(0x7C00)) && engine::f16_to_f32(0x7C00) > 0.0f);
    EXPECT_TRUE(std::isinf(engine::f16_to_f32(0xFC00)) && engine::f16_to_f32(0xFC00) < 0.0f);
    EXPECT_TRUE(std::isnan(engine::f16_to_f32(0x7E00)));
    EXPECT_NEAR(engine::f16_to_f32(0x0001), 5.9604645e-8f, 1e-12f); // smallest subnormal
}

TEST(DType, BFloatToFloat) {
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0x3F80), 1.0f);
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0xC000), -2.0f);
    EXPECT_FLOAT_EQ(engine::bf16_to_f32(0x0000), 0.0f);
}

TEST(DType, SizeAndFromString) {
    EXPECT_EQ(engine::dtype_size(engine::DType::F32), 4u);
    EXPECT_EQ(engine::dtype_size(engine::DType::BF16), 2u);
    EXPECT_EQ(engine::dtype_from_string("BF16"), engine::DType::BF16);
    EXPECT_THROW((void)engine::dtype_from_string("I64"), std::runtime_error);
}
