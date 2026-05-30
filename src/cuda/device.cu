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

} // namespace engine::cuda
