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

} // namespace engine::cuda
