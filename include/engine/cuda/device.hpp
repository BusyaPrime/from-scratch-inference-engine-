// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Host-callable entry points into the CUDA backend. These signatures are plain C++ (no CUDA
// syntax) so they can be called from translation units compiled by the host compiler and from
// tests, while the implementations and kernels live in the .cu sources.
namespace engine::cuda {

// Number of CUDA devices visible to the process (0 if none / driver unavailable).
[[nodiscard]] int device_count();

// out[i] = a * x[i] + y[i], evaluated on the GPU. A minimal end-to-end exercise of allocation,
// host/device copies, and kernel launch used to validate the toolchain.
void saxpy(float a, const float* x, const float* y, float* out, int64_t n);

// RMSNorm over the last dimension, matching the CPU reference: for each of `rows` rows of length
// `dim`, out[j] = weight[j] * x[j] / sqrt(mean_j(x^2) + eps). Sum of squares is accumulated in
// double to match the CPU twin.
void rms_norm(
    const float* x, const float* weight, float* out, int64_t rows, int64_t dim, double eps);

// SwiGLU: out[i] = silu(gate[i]) * up[i] with silu(x) = x / (1 + exp(-x)), in float to match the
// CPU twin. gate, up, and out are all length n.
void silu_mul(const float* gate, const float* up, float* out, int64_t n);

// In-place rotary position embedding over x[rows, n_heads*head_dim] using the Hugging Face
// half-rotation layout; angles computed in double to match the CPU twin. positions has length
// rows.
void rope(float* x,
          int64_t rows,
          int64_t n_heads,
          int64_t head_dim,
          double theta,
          const int64_t* positions);

// y[rows, out_dim] = x[rows, in_dim] * weight[out_dim, in_dim]^T (+ bias[out_dim] if non-null),
// the Hugging Face Linear convention, via cuBLAS in strict fp32 (TF32 disabled) to match the CPU
// GEMM. bias may be nullptr.
void linear(const float* x,
            const float* weight,
            const float* bias,
            float* y,
            int64_t rows,
            int64_t in_dim,
            int64_t out_dim);

// Causal grouped-query attention matching the CPU twin. q is [q_len, n_heads*head_dim], k and v
// are [total, n_kv_heads*head_dim], out is [q_len, n_heads*head_dim]; query row i attends keys
// 0..query_offset+i. Scores and the softmax are computed in double. head_dim must be <= 256.
void attention(const float* q,
               const float* k,
               const float* v,
               float* out,
               int64_t q_len,
               int64_t total,
               int64_t n_heads,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t query_offset);

// Embedding gather: out[i, :] = weight[ids[i], :]; weight is [vocab, hidden], out is [n_ids,
// hidden].
void embedding(const float* weight,
               int64_t vocab,
               int64_t hidden,
               const int64_t* ids,
               int64_t n_ids,
               float* out);

// x[i] += y[i] over n elements (residual add).
void add_inplace(float* x, const float* y, int64_t n);

// Greedy argmax over each row: out[r] = argmax_j logits[r, j] for r in [0, rows), j in [0, cols).
// Ties resolve to the smallest index, matching the CPU sampler. logits is [rows, cols] on the
// host; out is [rows] on the host.
void argmax(const float* logits, int64_t* out, int64_t rows, int64_t cols);

} // namespace engine::cuda
