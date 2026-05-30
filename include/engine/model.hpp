// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/block_manager.hpp"
#include "engine/kv_cache.hpp"
#include "engine/model_config.hpp"
#include "engine/paged_kv_cache.hpp"
#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace engine {

// One sequence's work for a batched step: the new tokens to process now and the paging state
// to extend (non-owning; the caller owns the SequenceBlocks). A prefill item carries the whole
// prompt, a decode item carries a single token. forward_batch runs many of these through one
// set of matmuls.
struct BatchItem {
    SequenceBlocks* blocks;
    std::vector<int64_t> tokens;
};

// A loaded decoder-only transformer (Qwen2 family) for fp32 CPU inference.
class Model {
public:
    static Model from_pretrained(const std::string& model_dir);

    // Construct from a config and an in-memory weight set (e.g. a tiny test model).
    static Model from_safetensors(ModelConfig config, SafeTensors weights) {
        return Model(std::move(config), std::move(weights));
    }

    // Full-attention forward over one sequence of token ids -> logits [seq_len, vocab_size].
    [[nodiscard]] Tensor forward(const std::vector<int64_t>& ids) const;

    // Forward that appends K/V to the cache and attends over the full cache, so a prefill
    // chunk followed by single-token steps reproduces the full forward.
    [[nodiscard]] Tensor forward_with_cache(const std::vector<int64_t>& ids, KVCache& cache) const;

    // The same cached forward over paged (block-table) KV storage.
    [[nodiscard]] Tensor forward_paged(const std::vector<int64_t>& ids, PagedKVCache& cache) const;

    // Continuous-batching forward: process many sequences (mixed prefill and decode) in one set
    // of matmuls over their concatenated tokens, with per-sequence paged attention. Extends each
    // item's paging state and returns next-token logits for the last token of each item
    // [num_items, vocab_size].
    [[nodiscard]] Tensor forward_batch(BlockManager& manager,
                                       const std::vector<BatchItem>& items) const;

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }

private:
    Model(ModelConfig config, SafeTensors weights)
        : config_(std::move(config)), weights_(std::move(weights)) {}

    // Shared cached-forward body; instantiated for KVCache and PagedKVCache.
    template <typename Cache>
    [[nodiscard]] Tensor forward_cached(const std::vector<int64_t>& ids, Cache& cache) const;

    ModelConfig config_;
    SafeTensors weights_;
};

} // namespace engine
