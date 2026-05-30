// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"

#include "engine/nn.hpp"
#include "engine/paged_kv_cache.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void Model::quantize() {
    const char* attn[] = {"q", "k", "v", "o"};
    const char* mlp[] = {"gate", "up", "down"};
    for (int64_t l = 0; l < config_.num_hidden_layers; ++l) {
        for (const char* n : attn) {
            const std::string key = layer_key(l, std::string("self_attn.") + n + "_proj.weight");
            quant_.emplace(key, quantize_rowwise_int8(weights_.get(key)));
        }
        for (const char* n : mlp) {
            const std::string key = layer_key(l, std::string("mlp.") + n + "_proj.weight");
            quant_.emplace(key, quantize_rowwise_int8(weights_.get(key)));
        }
    }
    quantized_ = true;
}

Tensor Model::linear_w(const Tensor& x, const std::string& weight_name) const {
    if (quantized_) {
        const auto it = quant_.find(weight_name);
        if (it != quant_.end()) {
            return linear_int8(x, it->second);
        }
    }
    return linear(x, weights_.get(weight_name));
}

Tensor Model::linear_w(const Tensor& x, const std::string& weight_name, const Tensor& bias) const {
    if (quantized_) {
        const auto it = quant_.find(weight_name);
        if (it != quant_.end()) {
            return linear_int8(x, it->second, bias);
        }
    }
    return linear(x, weights_.get(weight_name), bias);
}

Tensor Model::forward(const std::vector<int64_t>& ids) const {
    KVCache cache(config_.num_hidden_layers, config_.num_key_value_heads * config_.head_dim);
    return forward_with_cache(ids, cache);
}

template <typename Cache>
Tensor Model::forward_cached(const std::vector<int64_t>& ids, Cache& cache) const {
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
            const std::string wkey = layer_key(l, "self_attn." + name + "_proj.weight");
            if (c.attention_qkv_bias) {
                return linear_w(
                    normed, wkey, weights_.get(layer_key(l, "self_attn." + name + "_proj.bias")));
            }
            return linear_w(normed, wkey);
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
        Tensor attn_out = linear_w(attn, layer_key(l, "self_attn.o_proj.weight"));
        add_inplace(x, attn_out);

        Tensor normed2 = rms_norm(
            x, weights_.get(layer_key(l, "post_attention_layernorm.weight")), c.rms_norm_eps);
        Tensor gate = linear_w(normed2, layer_key(l, "mlp.gate_proj.weight"));
        Tensor up = linear_w(normed2, layer_key(l, "mlp.up_proj.weight"));
        Tensor act = silu_mul(gate, up);
        Tensor down = linear_w(act, layer_key(l, "mlp.down_proj.weight"));
        add_inplace(x, down);
    }
    cache.advance(seq);

    Tensor normed = rms_norm(x, weights_.get("model.norm.weight"), c.rms_norm_eps);
    // Tied embeddings reuse embed_tokens as the output projection unless lm_head is present.
    const Tensor& lm_head =
        weights_.contains("lm_head.weight") ? weights_.get("lm_head.weight") : embed;
    return linear(normed, lm_head); // [S, vocab_size]
}

Tensor Model::forward_with_cache(const std::vector<int64_t>& ids, KVCache& cache) const {
    return forward_cached(ids, cache);
}

Tensor Model::forward_paged(const std::vector<int64_t>& ids, PagedKVCache& cache) const {
    return forward_cached(ids, cache);
}

Tensor Model::forward_batch(BlockManager& manager, const std::vector<BatchItem>& items) const {
    const ModelConfig& c = config_;
    if (items.empty()) {
        throw std::invalid_argument("forward_batch: empty batch");
    }
    const auto num_items = static_cast<int64_t>(items.size());
    const int64_t nq_dim = c.num_attention_heads * c.head_dim;
    const int64_t nkv_dim = c.num_key_value_heads * c.head_dim;

    // Concatenate every item's new tokens into one flat layout and record where each item lives.
    std::vector<int64_t> all_ids;
    std::vector<int64_t> positions;
    std::vector<int64_t> offset(static_cast<std::size_t>(num_items));
    std::vector<int64_t> len(static_cast<std::size_t>(num_items));
    std::vector<int64_t> past(static_cast<std::size_t>(num_items));
    int64_t total_tokens = 0;
    for (int64_t i = 0; i < num_items; ++i) {
        const BatchItem& item = items[static_cast<std::size_t>(i)];
        if (item.blocks == nullptr) {
            throw std::invalid_argument("forward_batch: null sequence blocks");
        }
        if (item.tokens.empty()) {
            throw std::invalid_argument("forward_batch: item has no tokens");
        }
        const auto item_len = static_cast<int64_t>(item.tokens.size());
        const int64_t item_past = item.blocks->length;
        offset[static_cast<std::size_t>(i)] = total_tokens;
        len[static_cast<std::size_t>(i)] = item_len;
        past[static_cast<std::size_t>(i)] = item_past;
        for (int64_t t = 0; t < item_len; ++t) {
            all_ids.push_back(item.tokens[static_cast<std::size_t>(t)]);
            positions.push_back(item_past + t);
        }
        total_tokens += item_len;
        manager.reserve(*item.blocks, item_len); // grow paging for the new tokens
    }

    const Tensor& embed = weights_.get("model.embed_tokens.weight");
    Tensor x = embedding(embed, all_ids); // [T, H]

    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        Tensor normed =
            rms_norm(x, weights_.get(layer_key(l, "input_layernorm.weight")), c.rms_norm_eps);

        const auto project = [&](const std::string& name) {
            const std::string wkey = layer_key(l, "self_attn." + name + "_proj.weight");
            if (c.attention_qkv_bias) {
                return linear_w(
                    normed, wkey, weights_.get(layer_key(l, "self_attn." + name + "_proj.bias")));
            }
            return linear_w(normed, wkey);
        };
        Tensor q = project("q"); // [T, nq_dim], one big GEMM across all items
        Tensor k = project("k"); // [T, nkv_dim]
        Tensor v = project("v"); // [T, nkv_dim]

        rope_inplace(q, c.num_attention_heads, c.head_dim, c.rope_theta, positions);
        rope_inplace(k, c.num_key_value_heads, c.head_dim, c.rope_theta, positions);

        // Attention is per-item: each sequence attends only over its own paged K/V.
        Tensor attn({total_tokens, nq_dim});
        for (int64_t i = 0; i < num_items; ++i) {
            const auto si = static_cast<std::size_t>(i);
            SequenceBlocks& blk = *items[si].blocks;
            manager.write(blk,
                          l,
                          past[si],
                          k.data() + offset[si] * nkv_dim,
                          v.data() + offset[si] * nkv_dim,
                          len[si]);
            const int64_t total = past[si] + len[si];
            const Tensor k_all = manager.gather_keys(blk, l, total);
            const Tensor v_all = manager.gather_values(blk, l, total);

            Tensor q_slice({len[si], nq_dim});
            std::memcpy(q_slice.data(),
                        q.data() + offset[si] * nq_dim,
                        sizeof(float) * static_cast<std::size_t>(len[si] * nq_dim));
            Tensor attn_slice = attention(q_slice,
                                          k_all,
                                          v_all,
                                          c.num_attention_heads,
                                          c.num_key_value_heads,
                                          c.head_dim,
                                          past[si]);
            std::memcpy(attn.data() + offset[si] * nq_dim,
                        attn_slice.data(),
                        sizeof(float) * static_cast<std::size_t>(len[si] * nq_dim));
        }

        Tensor attn_out = linear_w(attn, layer_key(l, "self_attn.o_proj.weight"));
        add_inplace(x, attn_out);

        Tensor normed2 = rms_norm(
            x, weights_.get(layer_key(l, "post_attention_layernorm.weight")), c.rms_norm_eps);
        Tensor gate = linear_w(normed2, layer_key(l, "mlp.gate_proj.weight"));
        Tensor up = linear_w(normed2, layer_key(l, "mlp.up_proj.weight"));
        Tensor act = silu_mul(gate, up);
        Tensor down = linear_w(act, layer_key(l, "mlp.down_proj.weight"));
        add_inplace(x, down);
    }

    for (int64_t i = 0; i < num_items; ++i) {
        manager.commit(*items[static_cast<std::size_t>(i)].blocks,
                       len[static_cast<std::size_t>(i)]);
    }

    Tensor normed = rms_norm(x, weights_.get("model.norm.weight"), c.rms_norm_eps);

    // Only the last token of each item drives next-token prediction, so project just those rows.
    const int64_t h = c.hidden_size;
    Tensor last_hidden({num_items, h});
    for (int64_t i = 0; i < num_items; ++i) {
        const auto si = static_cast<std::size_t>(i);
        const int64_t last_row = offset[si] + len[si] - 1;
        std::memcpy(last_hidden.data() + i * h,
                    normed.data() + last_row * h,
                    sizeof(float) * static_cast<std::size_t>(h));
    }
    const Tensor& lm_head =
        weights_.contains("lm_head.weight") ? weights_.get("lm_head.weight") : embed;
    return linear(last_hidden, lm_head); // [num_items, vocab_size]
}

} // namespace engine
