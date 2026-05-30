// SPDX-License-Identifier: Apache-2.0
#include "engine/block_manager.hpp"

#include <cstring>
#include <stdexcept>

namespace engine {

BlockManager::BlockManager(int64_t num_layers,
                           int64_t kv_dim,
                           int64_t block_size,
                           int64_t num_blocks)
    : num_layers_(num_layers), kv_dim_(kv_dim), block_size_(block_size), alloc_(num_blocks),
      keys_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0)),
      values_(static_cast<std::size_t>(num_layers > 0 ? num_layers : 0)) {
    if (num_layers <= 0 || kv_dim <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "BlockManager: num_layers, kv_dim, block_size must be positive");
    }
    const auto pool = static_cast<std::size_t>(num_blocks * block_size * kv_dim);
    for (auto& layer_store : keys_) {
        layer_store.resize(pool);
    }
    for (auto& layer_store : values_) {
        layer_store.resize(pool);
    }
}

void BlockManager::reserve(SequenceBlocks& seq, int64_t n_new) {
    const int64_t need = blocks_for(seq.length + n_new);
    while (static_cast<int64_t>(seq.block_table.size()) < need) {
        seq.block_table.push_back(alloc_.allocate());
    }
}

void BlockManager::write_one(std::vector<float>& store,
                             const SequenceBlocks& seq,
                             int64_t start,
                             const float* src,
                             int64_t n) const {
    for (int64_t r = 0; r < n; ++r) {
        const int64_t logical = start + r;
        const int64_t block = seq.block_table[static_cast<std::size_t>(logical / block_size_)];
        const int64_t slot = logical % block_size_;
        const int64_t off = (block * block_size_ + slot) * kv_dim_;
        std::memcpy(store.data() + off,
                    src + r * kv_dim_,
                    sizeof(float) * static_cast<std::size_t>(kv_dim_));
    }
}

void BlockManager::write(const SequenceBlocks& seq,
                         int64_t layer,
                         int64_t start,
                         const float* k,
                         const float* v,
                         int64_t n) {
    const auto li = static_cast<std::size_t>(layer);
    write_one(keys_[li], seq, start, k, n);
    write_one(values_[li], seq, start, v, n);
}

Tensor BlockManager::gather_one(const std::vector<float>& store,
                                const SequenceBlocks& seq,
                                int64_t rows) const {
    Tensor out({rows, kv_dim_});
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t block = seq.block_table[static_cast<std::size_t>(r / block_size_)];
        const int64_t slot = r % block_size_;
        const int64_t off = (block * block_size_ + slot) * kv_dim_;
        std::memcpy(out.data() + r * kv_dim_,
                    store.data() + off,
                    sizeof(float) * static_cast<std::size_t>(kv_dim_));
    }
    return out;
}

Tensor BlockManager::gather_keys(const SequenceBlocks& seq, int64_t layer, int64_t rows) const {
    return gather_one(keys_[static_cast<std::size_t>(layer)], seq, rows);
}

Tensor BlockManager::gather_values(const SequenceBlocks& seq, int64_t layer, int64_t rows) const {
    return gather_one(values_[static_cast<std::size_t>(layer)], seq, rows);
}

void BlockManager::free(SequenceBlocks& seq) {
    for (const int64_t block : seq.block_table) {
        alloc_.free(block);
    }
    seq.block_table.clear();
    seq.length = 0;
}

} // namespace engine
