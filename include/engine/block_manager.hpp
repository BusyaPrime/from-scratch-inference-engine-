// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/block_allocator.hpp"
#include "engine/tensor.hpp"

#include <cstdint>
#include <vector>

namespace engine {

// Per-sequence paging state: the ordered physical blocks backing the sequence and the
// number of committed token rows. Many of these draw from one BlockManager pool.
struct SequenceBlocks {
    std::vector<int64_t> block_table; // logical block index -> physical block id
    int64_t length = 0;               // committed token rows
};

// Owns the physical K/V pool (per-layer storage plus a shared block allocator) and serves
// many sequences that draw blocks from the same pool. One global pool with per-sequence
// block tables is what makes paged attention memory-efficient under concurrency.
class BlockManager {
public:
    BlockManager(int64_t num_layers, int64_t kv_dim, int64_t block_size, int64_t num_blocks);

    [[nodiscard]] int64_t num_layers() const noexcept { return num_layers_; }
    [[nodiscard]] int64_t kv_dim() const noexcept { return kv_dim_; }
    [[nodiscard]] int64_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] int64_t free_blocks() const noexcept { return alloc_.num_free(); }

    // Blocks required to store `tokens` tokens.
    [[nodiscard]] int64_t blocks_for(int64_t tokens) const noexcept {
        return (tokens + block_size_ - 1) / block_size_;
    }
    // Whether the free pool can grow `seq` by n_new tokens.
    [[nodiscard]] bool can_append(const SequenceBlocks& seq, int64_t n_new) const noexcept {
        const int64_t have = static_cast<int64_t>(seq.block_table.size());
        return (blocks_for(seq.length + n_new) - have) <= alloc_.num_free();
    }

    // Grow the sequence's block table to hold length + n_new tokens (allocates from the pool;
    // throws if the pool is exhausted, so callers should gate with can_append).
    void reserve(SequenceBlocks& seq, int64_t n_new);

    // Write n rows of K and V for one layer at the sequence's logical rows [start, start+n).
    void write(const SequenceBlocks& seq,
               int64_t layer,
               int64_t start,
               const float* k,
               const float* v,
               int64_t n);

    // Gather logical rows [0, rows) of one layer into a contiguous [rows, kv_dim] tensor.
    [[nodiscard]] Tensor gather_keys(const SequenceBlocks& seq, int64_t layer, int64_t rows) const;
    [[nodiscard]] Tensor
    gather_values(const SequenceBlocks& seq, int64_t layer, int64_t rows) const;

    // Commit n tokens for the sequence.
    void commit(SequenceBlocks& seq, int64_t n) const noexcept { seq.length += n; }

    // Return all of a sequence's blocks to the pool and reset it.
    void free(SequenceBlocks& seq);

private:
    void write_one(std::vector<float>& store,
                   const SequenceBlocks& seq,
                   int64_t start,
                   const float* src,
                   int64_t n) const;
    [[nodiscard]] Tensor
    gather_one(const std::vector<float>& store, const SequenceBlocks& seq, int64_t rows) const;

    int64_t num_layers_;
    int64_t kv_dim_;
    int64_t block_size_;
    BlockAllocator alloc_;
    std::vector<std::vector<float>> keys_;   // per layer: num_blocks * block_size * kv_dim
    std::vector<std::vector<float>> values_; // per layer: num_blocks * block_size * kv_dim
};

} // namespace engine
