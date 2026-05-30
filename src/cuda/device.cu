// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/device.hpp"

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

__global__ void saxpy_kernel(float a, const float* x, const float* y, float* out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
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

} // namespace

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

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    saxpy_kernel<<<blocks, threads>>>(a, dx, dy, dout, n);
    check(cudaGetLastError(), "launch saxpy");
    check(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy out");

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
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

    const int threads = 256;
    const auto shared_bytes = sizeof(double) * static_cast<std::size_t>(threads);
    rms_norm_kernel<<<static_cast<int>(rows), threads, shared_bytes>>>(dx, dw, dout, dim, eps);
    check(cudaGetLastError(), "launch rms_norm");
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

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    silu_mul_kernel<<<blocks, threads>>>(dgate, dup, dout, n);
    check(cudaGetLastError(), "launch silu_mul");
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

    const int64_t work = rows * n_heads * (head_dim / 2);
    const int threads = 256;
    const int blocks = static_cast<int>((work + threads - 1) / threads);
    rope_kernel<<<blocks, threads>>>(dx, rows, n_heads, head_dim, theta, dpos);
    check(cudaGetLastError(), "launch rope");
    check(cudaMemcpy(x, dx, x_bytes, cudaMemcpyDeviceToHost), "copy x back");

    cudaFree(dx);
    cudaFree(dpos);
}

} // namespace engine::cuda
