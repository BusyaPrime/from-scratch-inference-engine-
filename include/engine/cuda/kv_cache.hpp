// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::cuda {

// Device-resident contiguous KV cache for a single sequence, the GPU twin of engine::KVCache.
// Each layer owns a preallocated device buffer of capacity*kv_dim floats; append copies new rows
// (already device-resident) to the current length, and key_ptr/value_ptr hand back device pointers
// the attention kernel reads directly. Move-only; owns its device allocations.
class GpuKVCache {
public:
    GpuKVCache(int64_t num_layers, int64_t kv_dim, int64_t capacity);
    ~GpuKVCache();
    GpuKVCache(GpuKVCache&&) noexcept;
    GpuKVCache& operator=(GpuKVCache&&) noexcept;
    GpuKVCache(const GpuKVCache&) = delete;
    GpuKVCache& operator=(const GpuKVCache&) = delete;

    [[nodiscard]] int64_t length() const noexcept { return length_; }
    [[nodiscard]] int64_t kv_dim() const noexcept { return kv_dim_; }
    [[nodiscard]] int64_t capacity() const noexcept { return capacity_; }

    // Append n_rows rows of K and V (device pointers) for one layer at the current length.
    void append(int64_t layer, const float* k, const float* v, int64_t n_rows);

    // Device pointers to a layer's contiguous K/V (rows 0..length()+pending, kv_dim).
    [[nodiscard]] const float* key_ptr(int64_t layer) const {
        return keys_[static_cast<std::size_t>(layer)];
    }
    [[nodiscard]] const float* value_ptr(int64_t layer) const {
        return values_[static_cast<std::size_t>(layer)];
    }

    // Commit n_rows tokens (advance the shared logical length).
    void advance(int64_t n_rows) noexcept { length_ += n_rows; }

private:
    void release() noexcept;

    int64_t num_layers_ = 0;
    int64_t kv_dim_ = 0;
    int64_t capacity_ = 0;
    int64_t length_ = 0;
    std::vector<float*> keys_;   // device pointers, one buffer per layer
    std::vector<float*> values_; // device pointers, one buffer per layer
};

} // namespace engine::cuda
