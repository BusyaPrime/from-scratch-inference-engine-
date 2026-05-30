// SPDX-License-Identifier: Apache-2.0
#include "engine/kv_cache.hpp"
#include "engine/model.hpp"
#include "tiny_model.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

TEST(KVCache, CachedDecodeMatchesFullAttention) {
    const engine::Model model = tiny::tiny_model();
    const engine::ModelConfig& c = model.config();
    const int64_t vocab = c.vocab_size;

    const std::vector<int64_t> seq = {1, 5, 2, 8, 3, 0, 7};
    const engine::Tensor full = model.forward(seq); // [S, vocab]

    // Prefill the first 3 tokens, then decode the rest one at a time through one cache.
    engine::KVCache cache(c.num_hidden_layers, c.num_key_value_heads * c.head_dim);
    const int64_t prefill = 3;

    const std::vector<int64_t> head(seq.begin(), seq.begin() + prefill);
    const engine::Tensor prefill_logits = model.forward_with_cache(head, cache);
    ASSERT_EQ(prefill_logits.dim(0), prefill);
    for (int64_t i = 0; i < prefill; ++i) {
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(prefill_logits.data()[i * vocab + j], full.data()[i * vocab + j], 1e-4f);
        }
    }

    for (int64_t t = prefill; t < static_cast<int64_t>(seq.size()); ++t) {
        const engine::Tensor step =
            model.forward_with_cache({seq[static_cast<std::size_t>(t)]}, cache);
        ASSERT_EQ(step.dim(0), 1);
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(step.data()[j], full.data()[t * vocab + j], 1e-4f);
        }
    }
}
