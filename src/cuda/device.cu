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

} // namespace engine::cuda
