// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/block_manager.hpp"
#include "engine/cuda/kernels.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace engine::cuda {
namespace {

void bm_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuda block manager: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

GpuBlockManager::GpuBlockManager(int64_t num_layers,
                                 int64_t kv_dim,
                                 int64_t block_size,
                                 int64_t num_blocks)
    : num_layers_(num_layers), kv_dim_(kv_dim), block_size_(block_size), alloc_(num_blocks),
      keys_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0), nullptr),
      values_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0), nullptr) {
    if (num_layers <= 0 || kv_dim <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "GpuBlockManager: num_layers, kv_dim, block_size must be positive");
    }
    const auto bytes = static_cast<std::size_t>(num_blocks * block_size * kv_dim) * sizeof(float);
    for (std::size_t l = 0; l < keys_.size(); ++l) {
        bm_check(cudaMalloc(&keys_[l], bytes), "malloc keys");
        bm_check(cudaMalloc(&values_[l], bytes), "malloc values");
    }
}

void GpuBlockManager::release() noexcept {
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
    if (device_table_ != nullptr) {
        cudaFree(device_table_);
        device_table_ = nullptr;
        device_table_cap_ = 0;
    }
}

GpuBlockManager::~GpuBlockManager() {
    release();
}

GpuBlockManager::GpuBlockManager(GpuBlockManager&& other) noexcept
    : num_layers_(other.num_layers_), kv_dim_(other.kv_dim_), block_size_(other.block_size_),
      alloc_(std::move(other.alloc_)), keys_(std::move(other.keys_)),
      values_(std::move(other.values_)), device_table_(other.device_table_),
      device_table_cap_(other.device_table_cap_) {
    other.keys_.clear();
    other.values_.clear();
    other.device_table_ = nullptr;
    other.device_table_cap_ = 0;
}

GpuBlockManager& GpuBlockManager::operator=(GpuBlockManager&& other) noexcept {
    if (this != &other) {
        release();
        num_layers_ = other.num_layers_;
        kv_dim_ = other.kv_dim_;
        block_size_ = other.block_size_;
        alloc_ = std::move(other.alloc_);
        keys_ = std::move(other.keys_);
        values_ = std::move(other.values_);
        device_table_ = other.device_table_;
        device_table_cap_ = other.device_table_cap_;
        other.keys_.clear();
        other.values_.clear();
        other.device_table_ = nullptr;
        other.device_table_cap_ = 0;
    }
    return *this;
}

void GpuBlockManager::upload_table(const SequenceBlocks& seq) const {
    const auto need = static_cast<int64_t>(seq.block_table.size());
    if (need == 0) {
        return;
    }
    if (need > device_table_cap_) {
        if (device_table_ != nullptr) {
            cudaFree(device_table_);
            device_table_ = nullptr;
        }
        bm_check(cudaMalloc(&device_table_, static_cast<std::size_t>(need) * sizeof(int64_t)),
                 "malloc block table");
        device_table_cap_ = need;
    }
    bm_check(cudaMemcpy(device_table_,
                        seq.block_table.data(),
                        static_cast<std::size_t>(need) * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "copy block table");
}

void GpuBlockManager::reserve(SequenceBlocks& seq, int64_t n_new) {
    const int64_t need = blocks_for(seq.length + n_new);
    while (static_cast<int64_t>(seq.block_table.size()) < need) {
        seq.block_table.push_back(alloc_.allocate());
    }
}

void GpuBlockManager::write(const SequenceBlocks& seq,
                            int64_t layer,
                            int64_t start,
                            const float* k,
                            const float* v,
                            int64_t n) {
    if (n <= 0) {
        return;
    }
    upload_table(seq);
    const auto li = static_cast<std::size_t>(layer);
    kernels::paged_scatter(keys_[li], device_table_, k, start, n, block_size_, kv_dim_);
    kernels::paged_scatter(values_[li], device_table_, v, start, n, block_size_, kv_dim_);
}

void GpuBlockManager::gather(
    const SequenceBlocks& seq, int64_t layer, int64_t rows, float* k_out, float* v_out) const {
    if (rows <= 0) {
        return;
    }
    upload_table(seq);
    const auto li = static_cast<std::size_t>(layer);
    kernels::paged_gather(keys_[li], device_table_, k_out, rows, block_size_, kv_dim_);
    kernels::paged_gather(values_[li], device_table_, v_out, rows, block_size_, kv_dim_);
}

void GpuBlockManager::free(SequenceBlocks& seq) {
    for (const int64_t block : seq.block_table) {
        alloc_.free(block);
    }
    seq.block_table.clear();
    seq.length = 0;
}

} // namespace engine::cuda
