// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

#include <cstdint>
#include <vector>

namespace engine {

// Row-wise (per-output-channel) symmetric int8 quantization of a [rows, cols] weight matrix:
// weight[o, i] is approximated by data[o, i] * scales[o]. Per-row scales keep the error small for
// the Hugging Face Linear weight layout [out_features, in_features].
struct QuantizedMatrix {
    std::vector<int8_t> data;  // rows * cols, row-major
    std::vector<float> scales; // one scale per row
    int64_t rows = 0;
    int64_t cols = 0;
};

[[nodiscard]] QuantizedMatrix quantize_rowwise_int8(const Tensor& weight);

// y[s, o] = bias[o] + sum_i x[s, i] * (data[o, i] * scales[o]); the Linear convention with an int8
// weight (weight-only quantization; activations stay fp32). bias may be empty.
[[nodiscard]] Tensor
linear_int8(const Tensor& x, const QuantizedMatrix& weight, const Tensor& bias);
[[nodiscard]] Tensor linear_int8(const Tensor& x, const QuantizedMatrix& weight);

} // namespace engine
