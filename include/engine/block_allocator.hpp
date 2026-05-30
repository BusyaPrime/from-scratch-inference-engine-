// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine {

// Fixed pool of physical KV blocks with reference counting. A block is a unit of
// KV-cache storage (block_size token slots); the allocator hands out block ids and
// reclaims them when their reference count reaches zero. Reference counting lets two
// sequences share a block (e.g. a common prompt prefix) without copying.
class BlockAllocator {
public:
    explicit BlockAllocator(int64_t num_blocks)
        : ref_count_(static_cast<std::size_t>(num_blocks), 0) {
        if (num_blocks < 0) {
            throw std::invalid_argument("BlockAllocator: num_blocks must be non-negative");
        }
        free_list_.reserve(static_cast<std::size_t>(num_blocks));
        for (int64_t b = num_blocks - 1; b >= 0; --b) {
            free_list_.push_back(b);
        }
    }

    [[nodiscard]] int64_t num_blocks() const noexcept {
        return static_cast<int64_t>(ref_count_.size());
    }
    [[nodiscard]] int64_t num_free() const noexcept {
        return static_cast<int64_t>(free_list_.size());
    }

    // Take one free block (reference count becomes 1). Throws when the pool is empty.
    int64_t allocate() {
        if (free_list_.empty()) {
            throw std::runtime_error("BlockAllocator: out of blocks");
        }
        const int64_t b = free_list_.back();
        free_list_.pop_back();
        ref_count_[static_cast<std::size_t>(b)] = 1;
        return b;
    }

    // Add a reference to a live block so it survives until every holder frees it.
    void incref(int64_t block) {
        if (ref_count_[static_cast<std::size_t>(block)] == 0) {
            throw std::runtime_error("BlockAllocator: incref on a free block");
        }
        ref_count_[static_cast<std::size_t>(block)] += 1;
    }

    // Drop one reference; the block returns to the pool when the count hits zero.
    void free(int64_t block) {
        int32_t& rc = ref_count_[static_cast<std::size_t>(block)];
        if (rc == 0) {
            throw std::runtime_error("BlockAllocator: double free");
        }
        rc -= 1;
        if (rc == 0) {
            free_list_.push_back(block);
        }
    }

    [[nodiscard]] int32_t ref_count(int64_t block) const {
        return ref_count_[static_cast<std::size_t>(block)];
    }

private:
    std::vector<int64_t> free_list_;
    std::vector<int32_t> ref_count_;
};

} // namespace engine
