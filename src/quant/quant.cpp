// SPDX-License-Identifier: Apache-2.0
#include "engine/quant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {

QuantizedMatrix quantize_rowwise_int8(const Tensor& weight) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("quantize_rowwise_int8: weight must be 2D");
    }
    const int64_t rows = weight.dim(0);
    const int64_t cols = weight.dim(1);

    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.data.resize(static_cast<std::size_t>(rows * cols));
    q.scales.resize(static_cast<std::size_t>(rows));

    const float* w = weight.data();
    for (int64_t o = 0; o < rows; ++o) {
        const float* row = w + o * cols;
        float amax = 0.0f;
        for (int64_t i = 0; i < cols; ++i) {
            amax = std::max(amax, std::fabs(row[i]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        q.scales[static_cast<std::size_t>(o)] = scale;
        const float inv = 1.0f / scale;
        int8_t* qrow = q.data.data() + o * cols;
        for (int64_t i = 0; i < cols; ++i) {
            long r = std::lround(row[i] * inv);
            r = r > 127 ? 127 : (r < -127 ? -127 : r);
            qrow[i] = static_cast<int8_t>(r);
        }
    }
    return q;
}

Tensor linear_int8(const Tensor& x, const QuantizedMatrix& weight, const Tensor& bias) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("linear_int8: x must be 2D");
    }
    const int64_t s = x.dim(0);
    const int64_t in = x.dim(1);
    if (in != weight.cols) {
        throw std::invalid_argument("linear_int8: inner dimension does not match weight");
    }
    const int64_t out = weight.rows;
    if (bias.numel() > 0 && bias.numel() != out) {
        throw std::invalid_argument("linear_int8: bias size does not match output");
    }

    Tensor y({s, out});
    const float* xp = x.data();
    float* yp = y.data();
    const bool has_bias = bias.numel() > 0;
    const float* bp = bias.data();
    for (int64_t row = 0; row < s; ++row) {
        const float* xr = xp + row * in;
        float* yr = yp + row * out;
        for (int64_t o = 0; o < out; ++o) {
            const int8_t* qrow = weight.data.data() + o * in;
            double acc = 0.0;
            for (int64_t i = 0; i < in; ++i) {
                acc += static_cast<double>(xr[i]) * static_cast<double>(qrow[i]);
            }
            float value = static_cast<float>(acc) * weight.scales[static_cast<std::size_t>(o)];
            if (has_bias) {
                value += bp[o];
            }
            yr[o] = value;
        }
    }
    return y;
}

Tensor linear_int8(const Tensor& x, const QuantizedMatrix& weight) {
    return linear_int8(x, weight, Tensor{});
}

} // namespace engine
