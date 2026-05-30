// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/model.hpp"
#include "engine/model_config.hpp"
#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny {

inline engine::Tensor random_tensor(std::vector<int64_t> shape, std::mt19937& rng) {
    engine::Tensor t(std::move(shape));
    std::normal_distribution<float> nd(0.0f, 0.02f);
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data()[i] = nd(rng);
    }
    return t;
}

// A tiny Qwen2-family model with deterministic random weights, built entirely in memory.
inline engine::Model tiny_model() {
    engine::ModelConfig c;
    c.model_type = "qwen2";
    c.hidden_size = 8;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 2;
    c.intermediate_size = 16;
    c.vocab_size = 10;
    c.rope_theta = 10000.0;
    c.rms_norm_eps = 1e-6;
    c.attention_qkv_bias = true;
    c.tie_word_embeddings = true;

    const int64_t H = c.hidden_size;
    const int64_t NQ = c.num_attention_heads * c.head_dim;
    const int64_t NKV = c.num_key_value_heads * c.head_dim;
    const int64_t I = c.intermediate_size;

    std::mt19937 rng(1234);
    std::unordered_map<std::string, engine::Tensor> w;
    w["model.embed_tokens.weight"] = random_tensor({c.vocab_size, H}, rng);
    w["model.norm.weight"] = random_tensor({H}, rng);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        w[p + "input_layernorm.weight"] = random_tensor({H}, rng);
        w[p + "post_attention_layernorm.weight"] = random_tensor({H}, rng);
        w[p + "self_attn.q_proj.weight"] = random_tensor({NQ, H}, rng);
        w[p + "self_attn.q_proj.bias"] = random_tensor({NQ}, rng);
        w[p + "self_attn.k_proj.weight"] = random_tensor({NKV, H}, rng);
        w[p + "self_attn.k_proj.bias"] = random_tensor({NKV}, rng);
        w[p + "self_attn.v_proj.weight"] = random_tensor({NKV, H}, rng);
        w[p + "self_attn.v_proj.bias"] = random_tensor({NKV}, rng);
        w[p + "self_attn.o_proj.weight"] = random_tensor({H, NQ}, rng);
        w[p + "mlp.gate_proj.weight"] = random_tensor({I, H}, rng);
        w[p + "mlp.up_proj.weight"] = random_tensor({I, H}, rng);
        w[p + "mlp.down_proj.weight"] = random_tensor({H, I}, rng);
    }
    return engine::Model::from_safetensors(c, engine::SafeTensors::from_tensors(std::move(w)));
}

} // namespace tiny
