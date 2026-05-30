// SPDX-License-Identifier: Apache-2.0
#include "engine/block_manager.hpp"
#include "engine/model.hpp"
#include "tiny_model.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// Batching several sequences into one prefill step must give each the same last-token logits
// as forwarding it alone: per-item attention keeps the sequences independent.
TEST(ContinuousBatch, BatchedPrefillMatchesIndividualForward) {
    const engine::Model model = tiny::tiny_model();
    const engine::ModelConfig& c = model.config();
    const int64_t vocab = c.vocab_size;
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;

    const std::vector<std::vector<int64_t>> seqs = {{1, 2, 3}, {4, 5}, {6, 7, 8, 9}};

    std::vector<engine::Tensor> refs;
    refs.reserve(seqs.size());
    for (const std::vector<int64_t>& s : seqs) {
        refs.push_back(model.forward(s));
    }

    engine::BlockManager mgr(c.num_hidden_layers, kv_dim, /*block_size=*/2, /*num_blocks=*/64);
    std::vector<engine::SequenceBlocks> blocks(seqs.size());
    std::vector<engine::BatchItem> items;
    items.reserve(seqs.size());
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        items.push_back({&blocks[i], seqs[i]});
    }

    const engine::Tensor logits = model.forward_batch(mgr, items); // [num_seqs, vocab]
    ASSERT_EQ(logits.dim(0), static_cast<int64_t>(seqs.size()));
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        const int64_t last = (static_cast<int64_t>(seqs[i].size()) - 1) * vocab;
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(
                logits.data()[static_cast<int64_t>(i) * vocab + j], refs[i].data()[last + j], 1e-4f)
                << "seq " << i << " logit " << j;
        }
    }
}

// A prefill step followed by a decode step over the same shared pool must match the plain forward
// of each fully extended sequence: the paged cache carries state across steps, and prefill and
// decode items mix freely within a batch.
TEST(ContinuousBatch, MixedPrefillDecodeMatchesSequentialForward) {
    const engine::Model model = tiny::tiny_model();
    const engine::ModelConfig& c = model.config();
    const int64_t vocab = c.vocab_size;
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;

    const std::vector<int64_t> a = {1, 2, 3};
    const std::vector<int64_t> b = {4, 5};
    const int64_t next_a = 7;
    const int64_t next_b = 8;

    const engine::Tensor ref_a = model.forward({1, 2, 3, 7});
    const engine::Tensor ref_b = model.forward({4, 5, 8});

    engine::BlockManager mgr(c.num_hidden_layers, kv_dim, /*block_size=*/2, /*num_blocks=*/64);
    engine::SequenceBlocks blk_a;
    engine::SequenceBlocks blk_b;

    std::vector<engine::BatchItem> prefill = {{&blk_a, a}, {&blk_b, b}};
    static_cast<void>(model.forward_batch(mgr, prefill)); // prefill only populates the cache

    std::vector<engine::BatchItem> decode = {{&blk_a, {next_a}}, {&blk_b, {next_b}}};
    const engine::Tensor logits = model.forward_batch(mgr, decode); // [2, vocab]

    const int64_t last_a = (4 - 1) * vocab; // forward({1,2,3,7}) final row
    const int64_t last_b = (3 - 1) * vocab; // forward({4,5,8}) final row
    for (int64_t j = 0; j < vocab; ++j) {
        EXPECT_NEAR(logits.data()[j], ref_a.data()[last_a + j], 1e-4f) << "a logit " << j;
        EXPECT_NEAR(logits.data()[vocab + j], ref_b.data()[last_b + j], 1e-4f) << "b logit " << j;
    }
    EXPECT_EQ(blk_a.length, 4);
    EXPECT_EQ(blk_b.length, 3);
}
