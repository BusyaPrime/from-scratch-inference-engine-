// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

#include <cstdint>
#include <vector>

namespace engine {

// y[S, out] = x[S, in] * weight[out, in]^T (+ bias[out]). Hugging Face Linear convention.
[[nodiscard]] Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
[[nodiscard]] Tensor linear(const Tensor& x, const Tensor& weight);

// RMSNorm over the last dimension: out = weight * x * rsqrt(mean(x^2) + eps).
[[nodiscard]] Tensor rms_norm(const Tensor& x, const Tensor& weight, double eps);

// SwiGLU activation: silu(gate) * up, elementwise (shapes must match).
[[nodiscard]] Tensor silu_mul(const Tensor& gate, const Tensor& up);

// Gather rows of weight[vocab, hidden] selected by ids -> [len(ids), hidden].
[[nodiscard]] Tensor embedding(const Tensor& weight, const std::vector<int64_t>& ids);

// Apply rotary position embedding in place to x viewed as [S, n_heads * head_dim]
// using the Hugging Face half-rotation layout.
void rope_inplace(Tensor& x,
                  int64_t n_heads,
                  int64_t head_dim,
                  double theta,
                  const std::vector<int64_t>& positions);

// Causal grouped-query attention. q[S, n_heads*head_dim], k/v[S, n_kv_heads*head_dim]
// -> [S, n_heads*head_dim]. Scale is 1/sqrt(head_dim); softmax in fp64.
[[nodiscard]] Tensor attention(const Tensor& q,
                               const Tensor& k,
                               const Tensor& v,
                               int64_t n_heads,
                               int64_t n_kv_heads,
                               int64_t head_dim);

} // namespace engine
