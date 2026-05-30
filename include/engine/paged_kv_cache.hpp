// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/block_allocator.hpp"
#include "engine/tensor.hpp"

#include <cstdint>
#include <vector>

namespace engine {

// Paged key/value cache for a single sequence. K/V live in fixed-size blocks (block_size
// token slots) drawn from a shared pool; a block table maps logical token positions to
// physical blocks, so storage grows one block at a time and only the final block is ever
// partially filled. The append/key_tensor/value_tensor/advance/length interface matches
// KVCache, so the same cached forward runs over either the contiguous or the paged layout.
class PagedKVCache {
public:
    PagedKVCache(int64_t num_layers, int64_t kv_dim, int64_t block_size, int64_t num_blocks);

    [[nodiscard]] int64_t length() const noexcept { return length_; }
    [[nodiscard]] int64_t kv_dim() const noexcept { return kv_dim_; }
    [[nodiscard]] int64_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] int64_t blocks_used() const noexcept {
        return static_cast<int64_t>(block_table_.size());
    }
    // Physical token slots backing the allocated blocks (>= length(); the gap is the tail waste).
    [[nodiscard]] int64_t reserved_slots() const noexcept {
        return static_cast<int64_t>(block_table_.size()) * block_size_;
    }
    [[nodiscard]] const BlockAllocator& allocator() const noexcept { return alloc_; }

    // Append n_rows rows (each kv_dim long) of K and V for one layer, growing the block table
    // as needed. Rows land at the layer's current fill position.
    void append(int64_t layer, const float* k, const float* v, int64_t n_rows);

    // Gather every row written to a layer into a contiguous [rows, kv_dim] tensor in logical order.
    [[nodiscard]] Tensor key_tensor(int64_t layer) const;
    [[nodiscard]] Tensor value_tensor(int64_t layer) const;

    // Commit n_rows tokens (advance the shared logical length).
    void advance(int64_t n_rows) noexcept { length_ += n_rows; }

private:
    void ensure_blocks(int64_t rows);
    void write_rows(std::vector<float>& store,
                    int64_t start_row,
                    const float* src,
                    int64_t n_rows) const;
    [[nodiscard]] Tensor gather(const std::vector<float>& store, int64_t rows) const;

    int64_t num_layers_;
    int64_t kv_dim_;
    int64_t block_size_;
    BlockAllocator alloc_;
    std::vector<int64_t> block_table_;     // logical block index -> physical block id (shared)
    std::vector<std::vector<float>> keys_; // per layer: num_blocks * block_size * kv_dim
    std::vector<std::vector<float>> values_;
    std::vector<int64_t> filled_; // per layer: rows written so far
    int64_t length_ = 0;          // committed rows (shared logical length)
};

} // namespace engine
