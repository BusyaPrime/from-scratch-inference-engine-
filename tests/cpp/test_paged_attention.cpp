// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"
#include "engine/paged_kv_cache.hpp"
#include "tiny_model.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// Prefill + per-token decode through paged storage must reproduce the full forward exactly,
// proving the block table and slot mapping reassemble K/V in correct logical order.
TEST(PagedKVCache, PagedDecodeMatchesFullAttention) {
    const engine::Model model = tiny::tiny_model();
    const engine::ModelConfig& c = model.config();
    const int64_t vocab = c.vocab_size;
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;

    const std::vector<int64_t> seq = {1, 5, 2, 8, 3, 0, 7};
    const engine::Tensor full = model.forward(seq);

    // block_size 2 forces the 7-token sequence across several blocks.
    engine::PagedKVCache cache(c.num_hidden_layers, kv_dim, /*block_size=*/2, /*num_blocks=*/64);
    const int64_t prefill = 3;

    const std::vector<int64_t> head(seq.begin(), seq.begin() + prefill);
    const engine::Tensor prefill_logits = model.forward_paged(head, cache);
    ASSERT_EQ(prefill_logits.dim(0), prefill);
    for (int64_t i = 0; i < prefill; ++i) {
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(prefill_logits.data()[i * vocab + j], full.data()[i * vocab + j], 1e-4f);
        }
    }

    for (int64_t t = prefill; t < static_cast<int64_t>(seq.size()); ++t) {
        const engine::Tensor step = model.forward_paged({seq[static_cast<std::size_t>(t)]}, cache);
        ASSERT_EQ(step.dim(0), 1);
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(step.data()[j], full.data()[t * vocab + j], 1e-4f);
        }
    }

    EXPECT_EQ(cache.length(), static_cast<int64_t>(seq.size()));
    EXPECT_EQ(cache.blocks_used(), 4); // ceil(7 / 2)
}

// The block table should grow by exactly one block each time a token crosses a block boundary.
TEST(PagedKVCache, BlockTableGrowsOneBlockPerBoundary) {
    engine::PagedKVCache cache(/*num_layers=*/1, /*kv_dim=*/1, /*block_size=*/4, /*num_blocks=*/8);
    const float row = 1.0f;
    for (int64_t t = 0; t < 9; ++t) {
        cache.append(0, &row, &row, 1);
        cache.advance(1);
        const int64_t expected = (t + 1 + 3) / 4; // ceil((t+1)/4)
        EXPECT_EQ(cache.blocks_used(), expected) << "after " << (t + 1) << " tokens";
    }
}

// Internal fragmentation is bounded by one block per sequence, so a realistic workload
// (sequence lengths well above the block size) wastes well under 5% of reserved slots.
TEST(PagedKVCache, InternalFragmentationBelowFivePercent) {
    const int64_t block_size = 16;
    const int64_t kv_dim = 4;
    const std::vector<int64_t> lengths = {200, 256, 312, 180};
    const std::vector<float> row(static_cast<std::size_t>(kv_dim), 0.5f);

    int64_t used = 0;
    int64_t reserved = 0;
    for (const int64_t len : lengths) {
        engine::PagedKVCache cache(1, kv_dim, block_size, /*num_blocks=*/64);
        for (int64_t t = 0; t < len; ++t) {
            cache.append(0, row.data(), row.data(), 1);
            cache.advance(1);
        }
        EXPECT_LT(cache.reserved_slots() - cache.length(), block_size); // <= one partial block
        used += cache.length();
        reserved += cache.reserved_slots();
    }

    const double waste = static_cast<double>(reserved - used) / static_cast<double>(reserved);
    EXPECT_LT(waste, 0.05) << "internal fragmentation " << (waste * 100.0) << "%";
}
