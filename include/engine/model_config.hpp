// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

namespace engine {

// Decoder-only transformer configuration parsed from a Hugging Face config.json.
struct ModelConfig {
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0; // explicit when present, else hidden_size / num_attention_heads
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    double rope_theta = 10000.0;
    double rms_norm_eps = 1e-6;
    bool tie_word_embeddings = false;
    bool attention_qkv_bias = false; // Qwen2 adds bias on q/k/v projections; others do not
    std::string hidden_act = "silu";
    std::string model_type;

    // Parse from the raw text of a config.json document.
    static ModelConfig from_json(const std::string& text);
};

} // namespace engine
