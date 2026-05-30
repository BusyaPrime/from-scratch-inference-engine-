// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cublas_v2.h>

// Device-pointer kernel launchers. Every pointer is a device pointer; these launch on the default
// stream and do not synchronize, so a device-resident forward can chain them without host copies.
// The copy-in/out wrappers in device.hpp and the CudaModel forward both build on these.
namespace engine::cuda::kernels {

// out[i, :] = weight[ids[i], :]; gather n_ids rows of a [vocab, hidden] table.
void embedding(const float* weight, const int64_t* ids, float* out, int64_t n_ids, int64_t hidden);

// RMSNorm over the last dim, sum of squares in double (matches the CPU twin).
void rms_norm(
    const float* x, const float* weight, float* out, int64_t rows, int64_t dim, double eps);

// SwiGLU: out[i] = silu(gate[i]) * up[i].
void silu_mul(const float* gate, const float* up, float* out, int64_t n);

// x[i] += y[i] (residual add).
void add_inplace(float* x, const float* y, int64_t n);

// In-place RoPE over x[rows, n_heads*head_dim], half-rotation layout, angles in double.
void rope(float* x,
          int64_t rows,
          int64_t n_heads,
          int64_t head_dim,
          double theta,
          const int64_t* positions);

// y[rows, out_dim] = x[rows, in_dim] * weight[out_dim, in_dim]^T (+ bias). Strict fp32 (no TF32).
void linear(cublasHandle_t handle,
            const float* x,
            const float* weight,
            const float* bias,
            float* y,
            int64_t rows,
            int64_t in_dim,
            int64_t out_dim);

// Causal grouped-query attention; q[q_len, n_heads*head_dim], k/v[total, n_kv_heads*head_dim].
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

// Paged K/V movement through a device block table. The pool is laid out as
// [num_blocks, block_size, kv_dim]; block_table maps logical block index to physical block.
// scatter writes n rows from src at logical rows [start, start+n); gather reads rows [0, rows)
// into a contiguous out buffer in logical order.
void paged_scatter(float* pool,
                   const int64_t* block_table,
                   const float* src,
                   int64_t start,
                   int64_t n,
                   int64_t block_size,
                   int64_t kv_dim);
void paged_gather(const float* pool,
                  const int64_t* block_table,
                  float* out,
                  int64_t rows,
                  int64_t block_size,
                  int64_t kv_dim);

// Greedy next-token selection: out[r] = argmax_j logits[r, j] over cols columns; ties resolve to
// the smallest index, matching the CPU sampler. Reduces on the device so callers copy back rows
// token ids instead of the full [rows, cols] logits.
void argmax(const float* logits, int64_t* out, int64_t rows, int64_t cols);

} // namespace engine::cuda::kernels
