// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/kv_cache.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::cuda {
namespace {

void kv_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuda kv: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

GpuKVCache::GpuKVCache(int64_t num_layers, int64_t kv_dim, int64_t capacity)
    : num_layers_(num_layers), kv_dim_(kv_dim), capacity_(capacity),
      keys_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0), nullptr),
      values_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0), nullptr) {
    if (num_layers <= 0 || kv_dim <= 0 || capacity <= 0) {
        throw std::invalid_argument("GpuKVCache: num_layers, kv_dim, capacity must be positive");
    }
    const auto bytes = static_cast<std::size_t>(capacity * kv_dim) * sizeof(float);
    for (std::size_t l = 0; l < keys_.size(); ++l) {
        kv_check(cudaMalloc(&keys_[l], bytes), "malloc keys");
        kv_check(cudaMalloc(&values_[l], bytes), "malloc values");
    }
}

void GpuKVCache::release() noexcept {
    for (float* p : keys_) {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
    for (float* p : values_) {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
    keys_.clear();
    values_.clear();
}

GpuKVCache::~GpuKVCache() {
    release();
}

GpuKVCache::GpuKVCache(GpuKVCache&& other) noexcept
    : num_layers_(other.num_layers_), kv_dim_(other.kv_dim_), capacity_(other.capacity_),
      length_(other.length_), keys_(std::move(other.keys_)), values_(std::move(other.values_)) {
    other.keys_.clear();
    other.values_.clear();
    other.length_ = 0;
}

GpuKVCache& GpuKVCache::operator=(GpuKVCache&& other) noexcept {
    if (this != &other) {
        release();
        num_layers_ = other.num_layers_;
        kv_dim_ = other.kv_dim_;
        capacity_ = other.capacity_;
        length_ = other.length_;
        keys_ = std::move(other.keys_);
        values_ = std::move(other.values_);
        other.keys_.clear();
        other.values_.clear();
        other.length_ = 0;
    }
    return *this;
}

void GpuKVCache::append(int64_t layer, const float* k, const float* v, int64_t n_rows) {
    if (n_rows <= 0) {
        return;
    }
    if (length_ + n_rows > capacity_) {
        throw std::runtime_error("GpuKVCache: capacity exceeded");
    }
    const auto li = static_cast<std::size_t>(layer);
    const std::size_t offset = static_cast<std::size_t>(length_ * kv_dim_);
    const auto bytes = static_cast<std::size_t>(n_rows * kv_dim_) * sizeof(float);
    kv_check(cudaMemcpy(keys_[li] + offset, k, bytes, cudaMemcpyDeviceToDevice), "append keys");
    kv_check(cudaMemcpy(values_[li] + offset, v, bytes, cudaMemcpyDeviceToDevice), "append values");
}

} // namespace engine::cuda
