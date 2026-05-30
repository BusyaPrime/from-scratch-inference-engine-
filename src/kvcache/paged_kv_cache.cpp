// SPDX-License-Identifier: Apache-2.0
#include "engine/paged_kv_cache.hpp"

#include <cstring>
#include <stdexcept>

namespace engine {

PagedKVCache::PagedKVCache(int64_t num_layers,
                           int64_t kv_dim,
                           int64_t block_size,
                           int64_t num_blocks)
    : num_layers_(num_layers), kv_dim_(kv_dim), block_size_(block_size), alloc_(num_blocks),
      keys_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0)),
      values_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0)),
      filled_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0), 0) {
    if (num_layers <= 0 || kv_dim <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "PagedKVCache: num_layers, kv_dim, block_size must be positive");
    }
    const auto pool = static_cast<std::size_t>(num_blocks * block_size * kv_dim);
    for (auto& layer_store : keys_) {
        layer_store.resize(pool);
    }
    for (auto& layer_store : values_) {
        layer_store.resize(pool);
    }
}

void PagedKVCache::ensure_blocks(int64_t rows) {
    const int64_t needed = (rows + block_size_ - 1) / block_size_;
    while (static_cast<int64_t>(block_table_.size()) < needed) {
        block_table_.push_back(alloc_.allocate());
    }
}

void PagedKVCache::write_rows(std::vector<float>& store,
                              int64_t start_row,
                              const float* src,
                              int64_t n_rows) const {
    for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t logical = start_row + r;
        const int64_t block = block_table_[static_cast<std::size_t>(logical / block_size_)];
        const int64_t slot = logical % block_size_;
        const int64_t off = (block * block_size_ + slot) * kv_dim_;
        std::memcpy(store.data() + off,
                    src + r * kv_dim_,
                    sizeof(float) * static_cast<std::size_t>(kv_dim_));
    }
}

void PagedKVCache::append(int64_t layer, const float* k, const float* v, int64_t n_rows) {
    const auto li = static_cast<std::size_t>(layer);
    const int64_t start = filled_[li];
    ensure_blocks(start + n_rows);
    write_rows(keys_[li], start, k, n_rows);
    write_rows(values_[li], start, v, n_rows);
    filled_[li] = start + n_rows;
}

Tensor PagedKVCache::gather(const std::vector<float>& store, int64_t rows) const {
    Tensor out({rows, kv_dim_});
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t block = block_table_[static_cast<std::size_t>(r / block_size_)];
        const int64_t slot = r % block_size_;
        const int64_t off = (block * block_size_ + slot) * kv_dim_;
        std::memcpy(out.data() + r * kv_dim_,
                    store.data() + off,
                    sizeof(float) * static_cast<std::size_t>(kv_dim_));
    }
    return out;
}

Tensor PagedKVCache::key_tensor(int64_t layer) const {
    const auto li = static_cast<std::size_t>(layer);
    return gather(keys_[li], filled_[li]);
}

Tensor PagedKVCache::value_tensor(int64_t layer) const {
    const auto li = static_cast<std::size_t>(layer);
    return gather(values_[li], filled_[li]);
}

} // namespace engine
