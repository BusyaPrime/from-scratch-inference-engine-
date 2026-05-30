// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/kv_cache.hpp"
#include "engine/model_config.hpp"
#include "engine/paged_kv_cache.hpp"
#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace engine {

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
