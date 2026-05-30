// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/model_config.hpp"
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

    // Full-attention forward over one sequence of token ids -> logits [seq_len, vocab_size].
    [[nodiscard]] Tensor forward(const std::vector<int64_t>& ids) const;

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }

private:
    Model(ModelConfig config, SafeTensors weights)
        : config_(std::move(config)), weights_(std::move(weights)) {}

    ModelConfig config_;
    SafeTensors weights_;
};

} // namespace engine
