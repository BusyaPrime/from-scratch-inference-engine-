// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/block_allocator.hpp"
#include "engine/block_manager.hpp" // SequenceBlocks

#include <cstdint>
#include <vector>

namespace engine::cuda {

// Shared device KV pool serving many sequences, the GPU twin of engine::BlockManager. One device
// pool per layer ([num_blocks, block_size, kv_dim]); per-sequence block tables (engine::
// SequenceBlocks) map logical token rows to physical blocks. write scatters new rows into a
// sequence's blocks; gather materialises a sequence's contiguous K/V for the attention kernel.
// K/V pointers in write/gather are device pointers. Move-only; owns its device pools.
class GpuBlockManager {
public:
    GpuBlockManager(int64_t num_layers, int64_t kv_dim, int64_t block_size, int64_t num_blocks);
    ~GpuBlockManager();
    GpuBlockManager(GpuBlockManager&&) noexcept;
    GpuBlockManager& operator=(GpuBlockManager&&) noexcept;
    GpuBlockManager(const GpuBlockManager&) = delete;
    GpuBlockManager& operator=(const GpuBlockManager&) = delete;

    [[nodiscard]] int64_t kv_dim() const noexcept { return kv_dim_; }
    [[nodiscard]] int64_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] int64_t free_blocks() const noexcept { return alloc_.num_free(); }
    [[nodiscard]] int64_t blocks_for(int64_t tokens) const noexcept {
        return (tokens + block_size_ - 1) / block_size_;
    }
    [[nodiscard]] bool can_append(const SequenceBlocks& seq, int64_t n_new) const noexcept {
        const int64_t have = static_cast<int64_t>(seq.block_table.size());
        return (blocks_for(seq.length + n_new) - have) <= alloc_.num_free();
    }

    // Grow the sequence's block table to hold length + n_new tokens (throws if the pool is empty).
    void reserve(SequenceBlocks& seq, int64_t n_new);

    // Scatter n rows of device K and V for one layer at the sequence's logical rows [start,
    // start+n).
    void write(const SequenceBlocks& seq,
               int64_t layer,
               int64_t start,
               const float* k,
               const float* v,
               int64_t n);

    // Gather logical rows [0, rows) of one layer into contiguous device buffers (caller-allocated).
    void gather(
        const SequenceBlocks& seq, int64_t layer, int64_t rows, float* k_out, float* v_out) const;

    void commit(SequenceBlocks& seq, int64_t n) const noexcept { seq.length += n; }
    void free(SequenceBlocks& seq);

private:
    void release() noexcept;
    void
    upload_table(const SequenceBlocks& seq) const; // refresh device_table_ from seq.block_table

    int64_t num_layers_ = 0;
    int64_t kv_dim_ = 0;
    int64_t block_size_ = 0;
    BlockAllocator alloc_;
    std::vector<float*> keys_;                // device pool per layer
    std::vector<float*> values_;              // device pool per layer
    mutable int64_t* device_table_ = nullptr; // scratch: a sequence's block table on the device
    mutable int64_t device_table_cap_ = 0;
};

} // namespace engine::cuda
