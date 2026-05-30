// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"

#include "engine/nn.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("model: cannot open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void add_inplace(Tensor& x, const Tensor& y) {
    const int64_t n = x.numel();
    float* xp = x.data();
    const float* yp = y.data();
    for (int64_t i = 0; i < n; ++i) {
        xp[i] += yp[i];
    }
}

std::string layer_key(int64_t layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

} // namespace

Model Model::from_pretrained(const std::string& model_dir) {
    ModelConfig config = ModelConfig::from_json(read_file(model_dir + "/config.json"));
    SafeTensors weights = SafeTensors::load(model_dir + "/model.safetensors");
    return Model(std::move(config), std::move(weights));
}

Tensor Model::forward(const std::vector<int64_t>& ids) const {
    KVCache cache(config_.num_hidden_layers, config_.num_key_value_heads * config_.head_dim);
    return forward_with_cache(ids, cache);
}

Tensor Model::forward_with_cache(const std::vector<int64_t>& ids, KVCache& cache) const {
    const ModelConfig& c = config_;
    const auto seq = static_cast<int64_t>(ids.size());
    if (seq == 0) {
        throw std::invalid_argument("forward: empty input sequence");
    }

    const int64_t past = cache.length();
    std::vector<int64_t> positions(static_cast<std::size_t>(seq));
    for (int64_t i = 0; i < seq; ++i) {
        positions[static_cast<std::size_t>(i)] = past + i;
    }

    const Tensor& embed = weights_.get("model.embed_tokens.weight");
    Tensor x = embedding(embed, ids); // [S, H]

    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        Tensor normed =
            rms_norm(x, weights_.get(layer_key(l, "input_layernorm.weight")), c.rms_norm_eps);

        const auto project = [&](const std::string& name) {
            const Tensor& weight = weights_.get(layer_key(l, "self_attn." + name + "_proj.weight"));
            if (c.attention_qkv_bias) {
                return linear(
                    normed, weight, weights_.get(layer_key(l, "self_attn." + name + "_proj.bias")));
            }
            return linear(normed, weight);
        };
        Tensor q = project("q");
        Tensor k = project("k");
        Tensor v = project("v");

        rope_inplace(q, c.num_attention_heads, c.head_dim, c.rope_theta, positions);
        rope_inplace(k, c.num_key_value_heads, c.head_dim, c.rope_theta, positions);

        cache.append(l, k.data(), v.data(), seq);
        const Tensor k_all = cache.key_tensor(l);
        const Tensor v_all = cache.value_tensor(l);
        Tensor attn = attention(
            q, k_all, v_all, c.num_attention_heads, c.num_key_value_heads, c.head_dim, past);
        Tensor attn_out = linear(attn, weights_.get(layer_key(l, "self_attn.o_proj.weight")));
        add_inplace(x, attn_out);

        Tensor normed2 = rms_norm(
            x, weights_.get(layer_key(l, "post_attention_layernorm.weight")), c.rms_norm_eps);
        Tensor gate = linear(normed2, weights_.get(layer_key(l, "mlp.gate_proj.weight")));
        Tensor up = linear(normed2, weights_.get(layer_key(l, "mlp.up_proj.weight")));
        Tensor act = silu_mul(gate, up);
        Tensor down = linear(act, weights_.get(layer_key(l, "mlp.down_proj.weight")));
        add_inplace(x, down);
    }
    cache.advance(seq);

    Tensor normed = rms_norm(x, weights_.get("model.norm.weight"), c.rms_norm_eps);
    // Tied embeddings reuse embed_tokens as the output projection unless lm_head is present.
    const Tensor& lm_head =
        weights_.contains("lm_head.weight") ? weights_.get("lm_head.weight") : embed;
    return linear(normed, lm_head); // [S, vocab_size]
}

} // namespace engine
