// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

namespace engine {

// 2D fp32 matrix multiply: a [M, K] times b [K, N] yields [M, N]. Backed by Eigen.
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace engine
