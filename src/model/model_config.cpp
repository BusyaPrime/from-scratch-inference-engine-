// SPDX-License-Identifier: Apache-2.0
#include "engine/model_config.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace engine {

ModelConfig ModelConfig::from_json(const std::string& text) {
    const auto j = nlohmann::json::parse(text);

    ModelConfig c;
    c.model_type = j.value("model_type", std::string{});
    c.hidden_size = j.at("hidden_size").get<int64_t>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int64_t>();
    c.num_attention_heads = j.at("num_attention_heads").get<int64_t>();
    if (c.num_attention_heads <= 0) {
        throw std::runtime_error("model config: num_attention_heads must be positive");
    }
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.head_dim = j.contains("head_dim") ? j.at("head_dim").get<int64_t>()
                                        : c.hidden_size / c.num_attention_heads;
    c.intermediate_size = j.at("intermediate_size").get<int64_t>();
    c.vocab_size = j.at("vocab_size").get<int64_t>();
    c.max_position_embeddings = j.value("max_position_embeddings", int64_t{0});
    c.rope_theta = j.value("rope_theta", 10000.0);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.hidden_act = j.value("hidden_act", std::string{"silu"});
    // Qwen2 hardcodes bias on q/k/v; every other architecture follows attention_bias.
    c.attention_qkv_bias = (c.model_type == "qwen2") ? true : j.value("attention_bias", false);
    return c;
}

} // namespace engine
