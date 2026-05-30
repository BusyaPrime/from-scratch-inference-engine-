// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/device.hpp"
#include "engine/cuda/kernels.hpp"

#include <cfloat>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace engine::cuda {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(status));
    }
}

void cublas_check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cublas: ") + what);
    }
}

constexpr int kThreads = 256;
constexpr int kMaxHeadDim = 256;

int grid_for(int64_t work, int threads) {
    return static_cast<int>((work + threads - 1) / threads);
}

__global__ void saxpy_kernel(float a, const float* x, const float* y, float* out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

__global__ void embedding_kernel(
    const float* weight, const int64_t* ids, float* out, int64_t n_ids, int64_t hidden) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_ids * hidden) {
        return;
    }
    const int64_t row = idx / hidden;
    const int64_t col = idx % hidden;
    out[idx] = weight[ids[row] * hidden + col];
}

__global__ void add_inplace_kernel(float* x, const float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

__global__ void add_bias_kernel(float* y, const float* bias, int64_t rows, int64_t out_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < rows * out_dim) {
        y[idx] += bias[idx % out_dim];
    }
}

// One block per row. Threads accumulate the sum of squares in double (matching the CPU twin),
// reduce in shared memory, then write the normalised, weighted row. blockDim must be a power of 2.
__global__ void
rms_norm_kernel(const float* x, const float* weight, float* out, int64_t dim, double eps) {
    extern __shared__ double shared[];
    const int64_t row = blockIdx.x;
    const float* xr = x + row * dim;
    float* orow = out + row * dim;

    double partial = 0.0;
    for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
        const double v = static_cast<double>(xr[j]);
        partial += v * v;
    }
    shared[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ double scale;
    if (threadIdx.x == 0) {
        scale = 1.0 / sqrt(shared[0] / static_cast<double>(dim) + eps);
    }
    __syncthreads();

    for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
        orow[j] = static_cast<float>(static_cast<double>(xr[j]) * scale) * weight[j];
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = gate[i];
        out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}

// One thread per rotation pair (row, head, i<half). In place is safe because each thread owns the
// distinct indices i and i+half within its head.
__global__ void rope_kernel(
    float* x, int64_t rows, int64_t n_heads, int64_t head_dim, double theta, const int64_t* pos) {
    const int64_t half = head_dim / 2;
    const int64_t per_row = n_heads * half;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * per_row) {
        return;
    }
    const int64_t t = idx / per_row;
    const int64_t rem = idx % per_row;
    const int64_t hh = rem / half;
    const int64_t i = rem % half;

    const double position = static_cast<double>(pos[t]);
    const double inv_freq =
        pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    const double angle = position * inv_freq;
    const double c = cos(angle);
    const double s = sin(angle);

    float* head = x + t * n_heads * head_dim + hh * head_dim;
    const float x1 = head[i];
    const float x2 = head[i + half];
    head[i] = static_cast<float>(static_cast<double>(x1) * c - static_cast<double>(x2) * s);
    head[i + half] = static_cast<float>(static_cast<double>(x2) * c + static_cast<double>(x1) * s);
}

// One thread per (head, query). Causal softmax-weighted sum over keys 0..query_offset+i, computed
// in double with a single-pass online softmax (algebraically identical to the CPU two-pass).
__global__ void attention_kernel(const float* q,
                                 const float* k,
                                 const float* v,
                                 float* out,
                                 int64_t q_len,
                                 int64_t n_heads,
                                 int64_t n_kv_heads,
                                 int64_t head_dim,
                                 int64_t query_offset,
                                 double scale) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_heads * q_len) {
        return;
    }
    const int64_t h = idx / q_len;
    const int64_t i = idx % q_len;
    const int64_t group = n_heads / n_kv_heads;
    const int64_t kvh = h / group;
    const int64_t last = query_offset + i;
    const float* qvec = q + i * n_heads * head_dim + h * head_dim;

    double acc[kMaxHeadDim];
    for (int64_t d = 0; d < head_dim; ++d) {
        acc[d] = 0.0;
    }
    double running_max = -1.0e308;
    double running_sum = 0.0;

    for (int64_t j = 0; j <= last; ++j) {
        const float* kvec = k + j * n_kv_heads * head_dim + kvh * head_dim;
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(qvec[d]) * static_cast<double>(kvec[d]);
        }
        dot *= scale;
        const double new_max = fmax(running_max, dot);
        const double correction = exp(running_max - new_max);
        const double weight = exp(dot - new_max);
        running_sum = running_sum * correction + weight;
        const float* vvec = v + j * n_kv_heads * head_dim + kvh * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * correction + weight * static_cast<double>(vvec[d]);
        }
        running_max = new_max;
    }

    const double inv_sum = 1.0 / running_sum;
    float* ovec = out + i * n_heads * head_dim + h * head_dim;
    for (int64_t d = 0; d < head_dim; ++d) {
        ovec[d] = static_cast<float>(acc[d] * inv_sum);
    }
}

__global__ void paged_scatter_kernel(float* pool,
                                     const int64_t* block_table,
                                     const float* src,
                                     int64_t start,
                                     int64_t n,
                                     int64_t block_size,
                                     int64_t kv_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * kv_dim) {
        return;
    }
    const int64_t r = idx / kv_dim;
    const int64_t e = idx % kv_dim;
    const int64_t logical = start + r;
    const int64_t pb = block_table[logical / block_size];
    const int64_t slot = logical % block_size;
    pool[(pb * block_size + slot) * kv_dim + e] = src[r * kv_dim + e];
}

__global__ void paged_gather_kernel(const float* pool,
                                    const int64_t* block_table,
                                    float* out,
                                    int64_t rows,
                                    int64_t block_size,
                                    int64_t kv_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * kv_dim) {
        return;
    }
    const int64_t r = idx / kv_dim;
    const int64_t e = idx % kv_dim;
    const int64_t pb = block_table[r / block_size];
    const int64_t slot = r % block_size;
    out[idx] = pool[(pb * block_size + slot) * kv_dim + e];
}

// One block per row. Each thread scans a strided slice of the row for its local max (smallest
// index wins a tie), then the block tree-reduces to the row's argmax. The tie rule matches the CPU
// sampler's first-max-wins. blockDim must be a power of 2; shared memory holds blockDim index/value
// pairs (indices first to keep the int64 array 8-byte aligned).
__global__ void argmax_kernel(const float* logits, int64_t* out, int64_t cols) {
    extern __shared__ unsigned char smem[];
    int64_t* sidx = reinterpret_cast<int64_t*>(smem);
    float* sval = reinterpret_cast<float*>(sidx + blockDim.x);
    const int64_t row = blockIdx.x;
    const float* r = logits + row * cols;

    float best_val = -FLT_MAX;
    int64_t best_idx = 0;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        if (r[j] > best_val) {
            best_val = r[j];
            best_idx = j;
        }
    }
    sidx[threadIdx.x] = best_idx;
    sval[threadIdx.x] = best_val;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_val = sval[threadIdx.x + stride];
            const int64_t other_idx = sidx[threadIdx.x + stride];
            if (other_val > sval[threadIdx.x] ||
                (other_val == sval[threadIdx.x] && other_idx < sidx[threadIdx.x])) {
                sval[threadIdx.x] = other_val;
                sidx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = sidx[0];
    }
}

} // namespace

namespace kernels {

void embedding(const float* weight, const int64_t* ids, float* out, int64_t n_ids, int64_t hidden) {
    if (n_ids <= 0 || hidden <= 0) {
        return;
    }
    embedding_kernel<<<grid_for(n_ids * hidden, kThreads), kThreads>>>(
        weight, ids, out, n_ids, hidden);
    check(cudaGetLastError(), "launch embedding");
}

void rms_norm(
    const float* x, const float* weight, float* out, int64_t rows, int64_t dim, double eps) {
    if (rows <= 0 || dim <= 0) {
        return;
    }
    const auto shared_bytes = sizeof(double) * static_cast<std::size_t>(kThreads);
    rms_norm_kernel<<<static_cast<int>(rows), kThreads, shared_bytes>>>(x, weight, out, dim, eps);
    check(cudaGetLastError(), "launch rms_norm");
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
    if (n <= 0) {
        return;
    }
    silu_mul_kernel<<<grid_for(n, kThreads), kThreads>>>(gate, up, out, n);
    check(cudaGetLastError(), "launch silu_mul");
}

void add_inplace(float* x, const float* y, int64_t n) {
    if (n <= 0) {
        return;
    }
    add_inplace_kernel<<<grid_for(n, kThreads), kThreads>>>(x, y, n);
    check(cudaGetLastError(), "launch add_inplace");
}

void rope(float* x,
          int64_t rows,
          int64_t n_heads,
          int64_t head_dim,
          double theta,
          const int64_t* positions) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) {
        return;
    }
    const int64_t work = rows * n_heads * (head_dim / 2);
    rope_kernel<<<grid_for(work, kThreads), kThreads>>>(
        x, rows, n_heads, head_dim, theta, positions);
    check(cudaGetLastError(), "launch rope");
}

void linear(cublasHandle_t handle,
            const float* x,
            const float* weight,
            const float* bias,
            float* y,
            int64_t rows,
            int64_t in_dim,
            int64_t out_dim) {
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        return;
    }
    // Strict IEEE fp32 (no TF32 tensor cores) so results match the CPU Eigen GEMM. Row-major
    // y = x * weight^T equals, in cuBLAS column-major, y' = weight'^T * x'.
    cublas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "set math mode");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasSgemm(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             static_cast<int>(out_dim),
                             static_cast<int>(rows),
                             static_cast<int>(in_dim),
                             &alpha,
                             weight,
                             static_cast<int>(in_dim),
                             x,
                             static_cast<int>(in_dim),
                             &beta,
                             y,
                             static_cast<int>(out_dim)),
                 "sgemm");
    if (bias != nullptr) {
        add_bias_kernel<<<grid_for(rows * out_dim, kThreads), kThreads>>>(y, bias, rows, out_dim);
        check(cudaGetLastError(), "launch add_bias");
    }
}

void attention(const float* q,
               const float* k,
               const float* v,
               float* out,
               int64_t q_len,
               int64_t total,
               int64_t n_heads,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t query_offset) {
    if (q_len <= 0 || total <= 0) {
        return;
    }
    if (head_dim > kMaxHeadDim) {
        throw std::runtime_error("cuda attention: head_dim exceeds supported maximum");
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    attention_kernel<<<grid_for(n_heads * q_len, kThreads), kThreads>>>(
        q, k, v, out, q_len, n_heads, n_kv_heads, head_dim, query_offset, scale);
    check(cudaGetLastError(), "launch attention");
}

void paged_scatter(float* pool,
                   const int64_t* block_table,
                   const float* src,
                   int64_t start,
                   int64_t n,
                   int64_t block_size,
                   int64_t kv_dim) {
    if (n <= 0 || kv_dim <= 0) {
        return;
    }
    paged_scatter_kernel<<<grid_for(n * kv_dim, kThreads), kThreads>>>(
        pool, block_table, src, start, n, block_size, kv_dim);
    check(cudaGetLastError(), "launch paged_scatter");
}

void paged_gather(const float* pool,
                  const int64_t* block_table,
                  float* out,
                  int64_t rows,
                  int64_t block_size,
                  int64_t kv_dim) {
    if (rows <= 0 || kv_dim <= 0) {
        return;
    }
    paged_gather_kernel<<<grid_for(rows * kv_dim, kThreads), kThreads>>>(
        pool, block_table, out, rows, block_size, kv_dim);
    check(cudaGetLastError(), "launch paged_gather");
}

void argmax(const float* logits, int64_t* out, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const auto shared_bytes =
        static_cast<std::size_t>(kThreads) * (sizeof(int64_t) + sizeof(float));
    argmax_kernel<<<static_cast<int>(rows), kThreads, shared_bytes>>>(logits, out, cols);
    check(cudaGetLastError(), "launch argmax");
}

} // namespace kernels

// --- Host copy-in / copy-out wrappers (used by tests and standalone callers) ---

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

void saxpy(float a, const float* x, const float* y, float* out, int64_t n) {
    if (n <= 0) {
        return;
    }
    const auto bytes = sizeof(float) * static_cast<std::size_t>(n);
    float* dx = nullptr;
    float* dy = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&dx, bytes), "malloc x");
    check(cudaMalloc(&dy, bytes), "malloc y");
    check(cudaMalloc(&dout, bytes), "malloc out");
    check(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice), "copy y");
    saxpy_kernel<<<grid_for(n, kThreads), kThreads>>>(a, dx, dy, dout, n);
    check(cudaGetLastError(), "launch saxpy");
    check(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
}

void embedding(const float* weight,
               int64_t vocab,
               int64_t hidden,
               const int64_t* ids,
               int64_t n_ids,
               float* out) {
    if (n_ids <= 0 || hidden <= 0 || vocab <= 0) {
        return;
    }
    const auto w_bytes = sizeof(float) * static_cast<std::size_t>(vocab * hidden);
    const auto id_bytes = sizeof(int64_t) * static_cast<std::size_t>(n_ids);
    const auto out_bytes = sizeof(float) * static_cast<std::size_t>(n_ids * hidden);
    float* dweight = nullptr;
    int64_t* dids = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&dweight, w_bytes), "malloc weight");
    check(cudaMalloc(&dids, id_bytes), "malloc ids");
    check(cudaMalloc(&dout, out_bytes), "malloc out");
    check(cudaMemcpy(dweight, weight, w_bytes, cudaMemcpyHostToDevice), "copy weight");
    check(cudaMemcpy(dids, ids, id_bytes, cudaMemcpyHostToDevice), "copy ids");
    kernels::embedding(dweight, dids, dout, n_ids, hidden);
    check(cudaMemcpy(out, dout, out_bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dweight);
    cudaFree(dids);
    cudaFree(dout);
}

void add_inplace(float* x, const float* y, int64_t n) {
    if (n <= 0) {
        return;
    }
    const auto bytes = sizeof(float) * static_cast<std::size_t>(n);
    float* dx = nullptr;
    float* dy = nullptr;
    check(cudaMalloc(&dx, bytes), "malloc x");
    check(cudaMalloc(&dy, bytes), "malloc y");
    check(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice), "copy y");
    kernels::add_inplace(dx, dy, n);
    check(cudaMemcpy(x, dx, bytes, cudaMemcpyDeviceToHost), "copy x back");
    cudaFree(dx);
    cudaFree(dy);
}

void rms_norm(
    const float* x, const float* weight, float* out, int64_t rows, int64_t dim, double eps) {
    if (rows <= 0 || dim <= 0) {
        return;
    }
    const auto x_bytes = sizeof(float) * static_cast<std::size_t>(rows * dim);
    const auto w_bytes = sizeof(float) * static_cast<std::size_t>(dim);
    float* dx = nullptr;
    float* dw = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&dx, x_bytes), "malloc x");
    check(cudaMalloc(&dw, w_bytes), "malloc weight");
    check(cudaMalloc(&dout, x_bytes), "malloc out");
    check(cudaMemcpy(dx, x, x_bytes, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(dw, weight, w_bytes, cudaMemcpyHostToDevice), "copy weight");
    kernels::rms_norm(dx, dw, dout, rows, dim, eps);
    check(cudaMemcpy(out, dout, x_bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
    if (n <= 0) {
        return;
    }
    const auto bytes = sizeof(float) * static_cast<std::size_t>(n);
    float* dgate = nullptr;
    float* dup = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&dgate, bytes), "malloc gate");
    check(cudaMalloc(&dup, bytes), "malloc up");
    check(cudaMalloc(&dout, bytes), "malloc out");
    check(cudaMemcpy(dgate, gate, bytes, cudaMemcpyHostToDevice), "copy gate");
    check(cudaMemcpy(dup, up, bytes, cudaMemcpyHostToDevice), "copy up");
    kernels::silu_mul(dgate, dup, dout, n);
    check(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dgate);
    cudaFree(dup);
    cudaFree(dout);
}

void rope(float* x,
          int64_t rows,
          int64_t n_heads,
          int64_t head_dim,
          double theta,
          const int64_t* positions) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) {
        return;
    }
    const int64_t dim = n_heads * head_dim;
    const auto x_bytes = sizeof(float) * static_cast<std::size_t>(rows * dim);
    const auto pos_bytes = sizeof(int64_t) * static_cast<std::size_t>(rows);
    float* dx = nullptr;
    int64_t* dpos = nullptr;
    check(cudaMalloc(&dx, x_bytes), "malloc x");
    check(cudaMalloc(&dpos, pos_bytes), "malloc positions");
    check(cudaMemcpy(dx, x, x_bytes, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(dpos, positions, pos_bytes, cudaMemcpyHostToDevice), "copy positions");
    kernels::rope(dx, rows, n_heads, head_dim, theta, dpos);
    check(cudaMemcpy(x, dx, x_bytes, cudaMemcpyDeviceToHost), "copy x back");
    cudaFree(dx);
    cudaFree(dpos);
}

void linear(const float* x,
            const float* weight,
            const float* bias,
            float* y,
            int64_t rows,
            int64_t in_dim,
            int64_t out_dim) {
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        return;
    }
    const auto x_bytes = sizeof(float) * static_cast<std::size_t>(rows * in_dim);
    const auto w_bytes = sizeof(float) * static_cast<std::size_t>(out_dim * in_dim);
    const auto y_bytes = sizeof(float) * static_cast<std::size_t>(rows * out_dim);
    float* dx = nullptr;
    float* dw = nullptr;
    float* dy = nullptr;
    float* dbias = nullptr;
    check(cudaMalloc(&dx, x_bytes), "malloc x");
    check(cudaMalloc(&dw, w_bytes), "malloc weight");
    check(cudaMalloc(&dy, y_bytes), "malloc y");
    check(cudaMemcpy(dx, x, x_bytes, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(dw, weight, w_bytes, cudaMemcpyHostToDevice), "copy weight");
    if (bias != nullptr) {
        check(cudaMalloc(&dbias, sizeof(float) * static_cast<std::size_t>(out_dim)), "malloc bias");
        check(cudaMemcpy(dbias,
                         bias,
                         sizeof(float) * static_cast<std::size_t>(out_dim),
                         cudaMemcpyHostToDevice),
              "copy bias");
    }
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "create handle");
    kernels::linear(handle, dx, dw, dbias, dy, rows, in_dim, out_dim);
    check(cudaMemcpy(y, dy, y_bytes, cudaMemcpyDeviceToHost), "copy y");
    cublasDestroy(handle);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(dbias);
}

void attention(const float* q,
               const float* k,
               const float* v,
               float* out,
               int64_t q_len,
               int64_t total,
               int64_t n_heads,
               int64_t n_kv_heads,
               int64_t head_dim,
               int64_t query_offset) {
    if (q_len <= 0 || total <= 0) {
        return;
    }
    const int64_t q_dim = n_heads * head_dim;
    const int64_t kv_dim = n_kv_heads * head_dim;
    const auto q_bytes = sizeof(float) * static_cast<std::size_t>(q_len * q_dim);
    const auto kv_bytes = sizeof(float) * static_cast<std::size_t>(total * kv_dim);
    float* dq = nullptr;
    float* dk = nullptr;
    float* dv = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&dq, q_bytes), "malloc q");
    check(cudaMalloc(&dk, kv_bytes), "malloc k");
    check(cudaMalloc(&dv, kv_bytes), "malloc v");
    check(cudaMalloc(&dout, q_bytes), "malloc out");
    check(cudaMemcpy(dq, q, q_bytes, cudaMemcpyHostToDevice), "copy q");
    check(cudaMemcpy(dk, k, kv_bytes, cudaMemcpyHostToDevice), "copy k");
    check(cudaMemcpy(dv, v, kv_bytes, cudaMemcpyHostToDevice), "copy v");
    kernels::attention(dq, dk, dv, dout, q_len, total, n_heads, n_kv_heads, head_dim, query_offset);
    check(cudaMemcpy(out, dout, q_bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
}

void argmax(const float* logits, int64_t* out, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const auto in_bytes = sizeof(float) * static_cast<std::size_t>(rows * cols);
    const auto out_bytes = sizeof(int64_t) * static_cast<std::size_t>(rows);
    float* dlogits = nullptr;
    int64_t* dout = nullptr;
    check(cudaMalloc(&dlogits, in_bytes), "malloc logits");
    check(cudaMalloc(&dout, out_bytes), "malloc out");
    check(cudaMemcpy(dlogits, logits, in_bytes, cudaMemcpyHostToDevice), "copy logits");
    kernels::argmax(dlogits, dout, rows, cols);
    check(cudaMemcpy(out, dout, out_bytes, cudaMemcpyDeviceToHost), "copy out");
    cudaFree(dlogits);
    cudaFree(dout);
}

} // namespace engine::cuda
