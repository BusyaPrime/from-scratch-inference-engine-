// SPDX-License-Identifier: Apache-2.0
#include "engine/block_manager.hpp"

#include <algorithm>
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
        seq.block_table.push_back(allocate_block());
    }
}

int64_t BlockManager::allocate_block() {
    if (alloc_.num_free() == 0) {
        evict_lru(); // reclaim a cached prefix block rather than fail outright
    }
    return alloc_.allocate();
}

void BlockManager::evict_lru() {
    uint64_t victim_hash = 0;
    std::size_t victim_index = 0;
    int64_t oldest = 0;
    bool found = false;
    for (auto& bucket : prefix_) {
        std::vector<PrefixEntry>& entries = bucket.second;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            // Only blocks held solely by the cache (ref count 1) are safe to drop; a higher count
            // means a live sequence is still attending over them.
            if (alloc_.ref_count(entries[i].block) == 1 &&
                (!found || entries[i].last_used < oldest)) {
                oldest = entries[i].last_used;
                victim_hash = bucket.first;
                victim_index = i;
                found = true;
            }
        }
    }
    if (!found) {
        return;
    }
    std::vector<PrefixEntry>& bucket = prefix_[victim_hash];
    alloc_.free(bucket[victim_index].block); // last reference -> block returns to the pool
    bucket.erase(bucket.begin() + static_cast<std::ptrdiff_t>(victim_index));
    if (bucket.empty()) {
        prefix_.erase(victim_hash);
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

uint64_t BlockManager::hash_block(uint64_t parent, const int64_t* tokens, int64_t n) {
    // FNV-1a over the parent hash and the block's tokens. A collision only affects which bucket is
    // probed; matches are confirmed by comparing the stored tokens, so correctness does not depend
    // on the hash being collision-free.
    uint64_t h = parent ^ 0xcbf29ce484222325ULL;
    for (int64_t i = 0; i < n; ++i) {
        const auto value = static_cast<uint64_t>(tokens[i]);
        for (int shift = 0; shift < 64; shift += 8) {
            h ^= (value >> shift) & 0xffULL;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

int64_t BlockManager::acquire_shared_prefix(SequenceBlocks& seq,
                                            const std::vector<int64_t>& tokens) {
    if (!seq.block_table.empty() || seq.length != 0) {
        return 0; // only meaningful for a fresh sequence
    }
    const int64_t full_blocks = static_cast<int64_t>(tokens.size()) / block_size_;
    uint64_t parent = 0;
    int64_t matched = 0;
    for (int64_t b = 0; b < full_blocks; ++b) {
        const int64_t* span = tokens.data() + b * block_size_;
        const uint64_t h = hash_block(parent, span, block_size_);
        const auto it = prefix_.find(h);
        int64_t block = -1;
        if (it != prefix_.end()) {
            for (PrefixEntry& entry : it->second) {
                if (entry.parent == parent &&
                    static_cast<int64_t>(entry.tokens.size()) == block_size_ &&
                    std::equal(entry.tokens.begin(), entry.tokens.end(), span)) {
                    block = entry.block;
                    entry.last_used = ++tick_; // refresh LRU on reuse
                    break;
                }
            }
        }
        if (block < 0) {
            break; // first miss ends the shareable prefix
        }
        alloc_.incref(block);
        seq.block_table.push_back(block);
        parent = h;
        matched += block_size_;
    }
    seq.length = matched;
    if (matched > 0) {
        ++prefix_hits_;
    }
    return matched;
}

void BlockManager::register_prefix(const SequenceBlocks& seq, const std::vector<int64_t>& tokens) {
    const auto by_blocks = static_cast<int64_t>(seq.block_table.size());
    const int64_t by_tokens = static_cast<int64_t>(tokens.size()) / block_size_;
    const int64_t full_blocks = by_blocks < by_tokens ? by_blocks : by_tokens;
    uint64_t parent = 0;
    for (int64_t b = 0; b < full_blocks; ++b) {
        const int64_t* span = tokens.data() + b * block_size_;
        const uint64_t h = hash_block(parent, span, block_size_);
        std::vector<PrefixEntry>& entries = prefix_[h];
        bool exists = false;
        for (const PrefixEntry& entry : entries) {
            if (entry.parent == parent &&
                static_cast<int64_t>(entry.tokens.size()) == block_size_ &&
                std::equal(entry.tokens.begin(), entry.tokens.end(), span)) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            const int64_t block = seq.block_table[static_cast<std::size_t>(b)];
            alloc_.incref(block); // keep the cached block live independent of the owner
            entries.push_back(
                {parent, std::vector<int64_t>(span, span + block_size_), block, ++tick_});
        }
        parent = h;
    }
}

void BlockManager::clear_prefix_cache() {
    for (const auto& bucket : prefix_) {
        for (const PrefixEntry& entry : bucket.second) {
            alloc_.free(entry.block);
        }
    }
    prefix_.clear();
}

} // namespace engine
