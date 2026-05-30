// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/kernels.hpp"
#include "engine/cuda/kv_cache.hpp"
#include "engine/cuda/model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::cuda {
namespace {

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(status));
    }
}

void cublas_check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cublas: ") + what);
    }
}

// Move-only owner of a device allocation.
template <class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(int64_t n) : n_(n) {
        if (n_ > 0) {
            cuda_check(cudaMalloc(&ptr_, static_cast<std::size_t>(n_) * sizeof(T)), "cudaMalloc");
        }
    }
    ~DeviceArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    DeviceArray(DeviceArray&& other) noexcept : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* get() const noexcept { return ptr_; }

private:
    T* ptr_ = nullptr;
    int64_t n_ = 0;
};

DeviceArray<float> upload(const Tensor& t) {
    DeviceArray<float> buffer(t.numel());
    if (t.numel() > 0) {
        cuda_check(cudaMemcpy(buffer.get(),
                              t.data(),
                              static_cast<std::size_t>(t.numel()) * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload weight");
    }
    return buffer;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("CudaModel: cannot open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string layer_key(int64_t layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

// Index of the maximum logit in the final row (greedy next token; first max wins ties).
int64_t argmax_last_row(const Tensor& logits) {
    const int64_t rows = logits.dim(0);
    const int64_t vocab = logits.dim(1);
    const float* row = logits.data() + (rows - 1) * vocab;
    int64_t best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (row[j] > row[best]) {
            best = j;
        }
    }
    return best;
}

} // namespace

struct CudaModel::Impl {
    ModelConfig config;
    cublasHandle_t handle = nullptr;
    std::unordered_map<std::string, DeviceArray<float>> weights;

    ~Impl() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }

    [[nodiscard]] const float* weight(const std::string& name) const {
        return weights.at(name).get();
    }
    [[nodiscard]] bool has(const std::string& name) const {
        return weights.find(name) != weights.end();
    }

    [[nodiscard]] Tensor forward(const std::vector<int64_t>& ids) const;
    [[nodiscard]] Tensor forward_cached(const std::vector<int64_t>& ids, GpuKVCache& cache) const;
    [[nodiscard]] Tensor forward_batch(const std::vector<GpuBatchItem>& items) const;
};

Tensor CudaModel::Impl::forward_cached(const std::vector<int64_t>& ids, GpuKVCache& cache) const {
    const ModelConfig& c = config;
    const auto seq = static_cast<int64_t>(ids.size());
    if (seq == 0) {
        throw std::invalid_argument("CudaModel::forward: empty input sequence");
    }
    const int64_t hidden = c.hidden_size;
    const int64_t nq_dim = c.num_attention_heads * c.head_dim;
    const int64_t nkv_dim = c.num_key_value_heads * c.head_dim;
    const int64_t inter = c.intermediate_size;
    const int64_t vocab = c.vocab_size;
    const double eps = c.rms_norm_eps;
    const double theta = c.rope_theta;

    DeviceArray<int64_t> d_ids(seq);
    cuda_check(cudaMemcpy(d_ids.get(),
                          ids.data(),
                          static_cast<std::size_t>(seq) * sizeof(int64_t),
                          cudaMemcpyHostToDevice),
               "copy ids");
    const int64_t past = cache.length();
    std::vector<int64_t> positions(static_cast<std::size_t>(seq));
    for (int64_t i = 0; i < seq; ++i) {
        positions[static_cast<std::size_t>(i)] = past + i;
    }
    DeviceArray<int64_t> d_pos(seq);
    cuda_check(cudaMemcpy(d_pos.get(),
                          positions.data(),
                          static_cast<std::size_t>(seq) * sizeof(int64_t),
                          cudaMemcpyHostToDevice),
               "copy positions");

    DeviceArray<float> d_x(seq * hidden);
    DeviceArray<float> d_normed(seq * hidden);
    DeviceArray<float> d_q(seq * nq_dim);
    DeviceArray<float> d_k(seq * nkv_dim);
    DeviceArray<float> d_v(seq * nkv_dim);
    DeviceArray<float> d_attn(seq * nq_dim);
    DeviceArray<float> d_attn_out(seq * hidden);
    DeviceArray<float> d_normed2(seq * hidden);
    DeviceArray<float> d_gate(seq * inter);
    DeviceArray<float> d_up(seq * inter);
    DeviceArray<float> d_act(seq * inter);
    DeviceArray<float> d_down(seq * hidden);

    const float* embed = weight("model.embed_tokens.weight");
    kernels::embedding(embed, d_ids.get(), d_x.get(), seq, hidden);

    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        kernels::rms_norm(d_x.get(),
                          weight(layer_key(l, "input_layernorm.weight")),
                          d_normed.get(),
                          seq,
                          hidden,
                          eps);

        const float* q_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.q_proj.bias")) : nullptr;
        const float* k_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.k_proj.bias")) : nullptr;
        const float* v_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.v_proj.bias")) : nullptr;
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.q_proj.weight")),
                        q_bias,
                        d_q.get(),
                        seq,
                        hidden,
                        nq_dim);
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.k_proj.weight")),
                        k_bias,
                        d_k.get(),
                        seq,
                        hidden,
                        nkv_dim);
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.v_proj.weight")),
                        v_bias,
                        d_v.get(),
                        seq,
                        hidden,
                        nkv_dim);

        kernels::rope(d_q.get(), seq, c.num_attention_heads, c.head_dim, theta, d_pos.get());
        kernels::rope(d_k.get(), seq, c.num_key_value_heads, c.head_dim, theta, d_pos.get());

        cache.append(l, d_k.get(), d_v.get(), seq);
        kernels::attention(d_q.get(),
                           cache.key_ptr(l),
                           cache.value_ptr(l),
                           d_attn.get(),
                           seq,
                           past + seq,
                           c.num_attention_heads,
                           c.num_key_value_heads,
                           c.head_dim,
                           past);
        kernels::linear(handle,
                        d_attn.get(),
                        weight(layer_key(l, "self_attn.o_proj.weight")),
                        nullptr,
                        d_attn_out.get(),
                        seq,
                        nq_dim,
                        hidden);
        kernels::add_inplace(d_x.get(), d_attn_out.get(), seq * hidden);

        kernels::rms_norm(d_x.get(),
                          weight(layer_key(l, "post_attention_layernorm.weight")),
                          d_normed2.get(),
                          seq,
                          hidden,
                          eps);
        kernels::linear(handle,
                        d_normed2.get(),
                        weight(layer_key(l, "mlp.gate_proj.weight")),
                        nullptr,
                        d_gate.get(),
                        seq,
                        hidden,
                        inter);
        kernels::linear(handle,
                        d_normed2.get(),
                        weight(layer_key(l, "mlp.up_proj.weight")),
                        nullptr,
                        d_up.get(),
                        seq,
                        hidden,
                        inter);
        kernels::silu_mul(d_gate.get(), d_up.get(), d_act.get(), seq * inter);
        kernels::linear(handle,
                        d_act.get(),
                        weight(layer_key(l, "mlp.down_proj.weight")),
                        nullptr,
                        d_down.get(),
                        seq,
                        inter,
                        hidden);
        kernels::add_inplace(d_x.get(), d_down.get(), seq * hidden);
    }
    cache.advance(seq);

    kernels::rms_norm(d_x.get(), weight("model.norm.weight"), d_normed.get(), seq, hidden, eps);
    const float* lm_head = has("lm_head.weight") ? weight("lm_head.weight") : embed;
    DeviceArray<float> d_logits(seq * vocab);
    kernels::linear(handle, d_normed.get(), lm_head, nullptr, d_logits.get(), seq, hidden, vocab);

    cuda_check(cudaDeviceSynchronize(), "synchronize");
    Tensor out({seq, vocab});
    cuda_check(cudaMemcpy(out.data(),
                          d_logits.get(),
                          static_cast<std::size_t>(seq * vocab) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy logits");
    return out;
}

Tensor CudaModel::Impl::forward(const std::vector<int64_t>& ids) const {
    GpuKVCache cache(config.num_hidden_layers,
                     config.num_key_value_heads * config.head_dim,
                     static_cast<int64_t>(ids.size()));
    return forward_cached(ids, cache);
}

Tensor CudaModel::Impl::forward_batch(const std::vector<GpuBatchItem>& items) const {
    const ModelConfig& c = config;
    if (items.empty()) {
        throw std::invalid_argument("CudaModel::forward_batch: empty batch");
    }
    const auto num_items = static_cast<int64_t>(items.size());
    const int64_t hidden = c.hidden_size;
    const int64_t nq_dim = c.num_attention_heads * c.head_dim;
    const int64_t nkv_dim = c.num_key_value_heads * c.head_dim;
    const int64_t inter = c.intermediate_size;
    const int64_t vocab = c.vocab_size;
    const double eps = c.rms_norm_eps;
    const double theta = c.rope_theta;

    std::vector<int64_t> all_ids;
    std::vector<int64_t> positions;
    std::vector<int64_t> offset(static_cast<std::size_t>(num_items));
    std::vector<int64_t> len(static_cast<std::size_t>(num_items));
    std::vector<int64_t> past(static_cast<std::size_t>(num_items));
    int64_t total = 0;
    for (int64_t i = 0; i < num_items; ++i) {
        const GpuBatchItem& item = items[static_cast<std::size_t>(i)];
        if (item.cache == nullptr) {
            throw std::invalid_argument("CudaModel::forward_batch: null cache");
        }
        if (item.tokens.empty()) {
            throw std::invalid_argument("CudaModel::forward_batch: item has no tokens");
        }
        const auto item_len = static_cast<int64_t>(item.tokens.size());
        const int64_t item_past = item.cache->length();
        offset[static_cast<std::size_t>(i)] = total;
        len[static_cast<std::size_t>(i)] = item_len;
        past[static_cast<std::size_t>(i)] = item_past;
        for (int64_t t = 0; t < item_len; ++t) {
            all_ids.push_back(item.tokens[static_cast<std::size_t>(t)]);
            positions.push_back(item_past + t);
        }
        total += item_len;
    }

    DeviceArray<int64_t> d_ids(total);
    cuda_check(cudaMemcpy(d_ids.get(),
                          all_ids.data(),
                          static_cast<std::size_t>(total) * sizeof(int64_t),
                          cudaMemcpyHostToDevice),
               "copy ids");
    DeviceArray<int64_t> d_pos(total);
    cuda_check(cudaMemcpy(d_pos.get(),
                          positions.data(),
                          static_cast<std::size_t>(total) * sizeof(int64_t),
                          cudaMemcpyHostToDevice),
               "copy positions");

    DeviceArray<float> d_x(total * hidden);
    DeviceArray<float> d_normed(total * hidden);
    DeviceArray<float> d_q(total * nq_dim);
    DeviceArray<float> d_k(total * nkv_dim);
    DeviceArray<float> d_v(total * nkv_dim);
    DeviceArray<float> d_attn(total * nq_dim);
    DeviceArray<float> d_attn_out(total * hidden);
    DeviceArray<float> d_normed2(total * hidden);
    DeviceArray<float> d_gate(total * inter);
    DeviceArray<float> d_up(total * inter);
    DeviceArray<float> d_act(total * inter);
    DeviceArray<float> d_down(total * hidden);

    const float* embed = weight("model.embed_tokens.weight");
    kernels::embedding(embed, d_ids.get(), d_x.get(), total, hidden);

    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        kernels::rms_norm(d_x.get(),
                          weight(layer_key(l, "input_layernorm.weight")),
                          d_normed.get(),
                          total,
                          hidden,
                          eps);

        const float* q_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.q_proj.bias")) : nullptr;
        const float* k_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.k_proj.bias")) : nullptr;
        const float* v_bias =
            c.attention_qkv_bias ? weight(layer_key(l, "self_attn.v_proj.bias")) : nullptr;
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.q_proj.weight")),
                        q_bias,
                        d_q.get(),
                        total,
                        hidden,
                        nq_dim);
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.k_proj.weight")),
                        k_bias,
                        d_k.get(),
                        total,
                        hidden,
                        nkv_dim);
        kernels::linear(handle,
                        d_normed.get(),
                        weight(layer_key(l, "self_attn.v_proj.weight")),
                        v_bias,
                        d_v.get(),
                        total,
                        hidden,
                        nkv_dim);

        kernels::rope(d_q.get(), total, c.num_attention_heads, c.head_dim, theta, d_pos.get());
        kernels::rope(d_k.get(), total, c.num_key_value_heads, c.head_dim, theta, d_pos.get());

        for (int64_t i = 0; i < num_items; ++i) {
            const auto si = static_cast<std::size_t>(i);
            GpuKVCache& cache = *items[si].cache;
            cache.append(
                l, d_k.get() + offset[si] * nkv_dim, d_v.get() + offset[si] * nkv_dim, len[si]);
            kernels::attention(d_q.get() + offset[si] * nq_dim,
                               cache.key_ptr(l),
                               cache.value_ptr(l),
                               d_attn.get() + offset[si] * nq_dim,
                               len[si],
                               past[si] + len[si],
                               c.num_attention_heads,
                               c.num_key_value_heads,
                               c.head_dim,
                               past[si]);
        }

        kernels::linear(handle,
                        d_attn.get(),
                        weight(layer_key(l, "self_attn.o_proj.weight")),
                        nullptr,
                        d_attn_out.get(),
                        total,
                        nq_dim,
                        hidden);
        kernels::add_inplace(d_x.get(), d_attn_out.get(), total * hidden);

        kernels::rms_norm(d_x.get(),
                          weight(layer_key(l, "post_attention_layernorm.weight")),
                          d_normed2.get(),
                          total,
                          hidden,
                          eps);
        kernels::linear(handle,
                        d_normed2.get(),
                        weight(layer_key(l, "mlp.gate_proj.weight")),
                        nullptr,
                        d_gate.get(),
                        total,
                        hidden,
                        inter);
        kernels::linear(handle,
                        d_normed2.get(),
                        weight(layer_key(l, "mlp.up_proj.weight")),
                        nullptr,
                        d_up.get(),
                        total,
                        hidden,
                        inter);
        kernels::silu_mul(d_gate.get(), d_up.get(), d_act.get(), total * inter);
        kernels::linear(handle,
                        d_act.get(),
                        weight(layer_key(l, "mlp.down_proj.weight")),
                        nullptr,
                        d_down.get(),
                        total,
                        inter,
                        hidden);
        kernels::add_inplace(d_x.get(), d_down.get(), total * hidden);
    }

    for (int64_t i = 0; i < num_items; ++i) {
        const auto si = static_cast<std::size_t>(i);
        items[si].cache->advance(len[si]);
    }

    kernels::rms_norm(d_x.get(), weight("model.norm.weight"), d_normed.get(), total, hidden, eps);

    // Only the last token of each item drives next-token prediction; gather those rows.
    DeviceArray<float> d_last(num_items * hidden);
    for (int64_t i = 0; i < num_items; ++i) {
        const auto si = static_cast<std::size_t>(i);
        const int64_t last_row = offset[si] + len[si] - 1;
        cuda_check(cudaMemcpy(d_last.get() + i * hidden,
                              d_normed.get() + last_row * hidden,
                              static_cast<std::size_t>(hidden) * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "gather last hidden");
    }
    const float* lm_head = has("lm_head.weight") ? weight("lm_head.weight") : embed;
    DeviceArray<float> d_logits(num_items * vocab);
    kernels::linear(
        handle, d_last.get(), lm_head, nullptr, d_logits.get(), num_items, hidden, vocab);

    cuda_check(cudaDeviceSynchronize(), "synchronize");
    Tensor out({num_items, vocab});
    cuda_check(cudaMemcpy(out.data(),
                          d_logits.get(),
                          static_cast<std::size_t>(num_items * vocab) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy logits");
    return out;
}

CudaModel::CudaModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaModel::~CudaModel() = default;
CudaModel::CudaModel(CudaModel&&) noexcept = default;
CudaModel& CudaModel::operator=(CudaModel&&) noexcept = default;

const ModelConfig& CudaModel::config() const noexcept {
    return impl_->config;
}

Tensor CudaModel::forward(const std::vector<int64_t>& ids) const {
    return impl_->forward(ids);
}

Tensor CudaModel::forward_with_cache(const std::vector<int64_t>& ids, GpuKVCache& cache) const {
    return impl_->forward_cached(ids, cache);
}

Tensor CudaModel::forward_batch(const std::vector<GpuBatchItem>& items) const {
    return impl_->forward_batch(items);
}

std::vector<int64_t>
CudaModel::generate(const std::vector<int64_t>& prompt, int64_t max_tokens, int64_t eos_id) const {
    std::vector<int64_t> out;
    if (max_tokens <= 0 || prompt.empty()) {
        return out;
    }
    const ModelConfig& c = impl_->config;
    GpuKVCache cache(c.num_hidden_layers,
                     c.num_key_value_heads * c.head_dim,
                     static_cast<int64_t>(prompt.size()) + max_tokens);

    Tensor logits = forward_with_cache(prompt, cache);
    int64_t token = argmax_last_row(logits);
    out.push_back(token);
    while (static_cast<int64_t>(out.size()) < max_tokens && token != eos_id) {
        logits = forward_with_cache({token}, cache);
        token = argmax_last_row(logits);
        out.push_back(token);
    }
    return out;
}

CudaModel CudaModel::from_safetensors(ModelConfig config, const SafeTensors& weights) {
    auto impl = std::make_unique<Impl>();
    impl->config = std::move(config);
    cublas_check(cublasCreate(&impl->handle), "create handle");

    const auto add = [&](const std::string& name, bool required) {
        if (!weights.contains(name)) {
            if (required) {
                throw std::runtime_error("CudaModel: missing weight " + name);
            }
            return;
        }
        impl->weights.emplace(name, upload(weights.get(name)));
    };

    add("model.embed_tokens.weight", true);
    add("model.norm.weight", true);
    add("lm_head.weight", false);
    const bool bias = impl->config.attention_qkv_bias;
    for (int64_t l = 0; l < impl->config.num_hidden_layers; ++l) {
        add(layer_key(l, "input_layernorm.weight"), true);
        add(layer_key(l, "post_attention_layernorm.weight"), true);
        add(layer_key(l, "self_attn.q_proj.weight"), true);
        add(layer_key(l, "self_attn.k_proj.weight"), true);
        add(layer_key(l, "self_attn.v_proj.weight"), true);
        add(layer_key(l, "self_attn.o_proj.weight"), true);
        add(layer_key(l, "self_attn.q_proj.bias"), bias);
        add(layer_key(l, "self_attn.k_proj.bias"), bias);
        add(layer_key(l, "self_attn.v_proj.bias"), bias);
        add(layer_key(l, "mlp.gate_proj.weight"), true);
        add(layer_key(l, "mlp.up_proj.weight"), true);
        add(layer_key(l, "mlp.down_proj.weight"), true);
    }

    return CudaModel(std::move(impl));
}

CudaModel CudaModel::from_pretrained(const std::string& model_dir) {
    ModelConfig config = ModelConfig::from_json(read_file(model_dir + "/config.json"));
    const SafeTensors weights = SafeTensors::load(model_dir + "/model.safetensors");
    return from_safetensors(std::move(config), weights);
}

} // namespace engine::cuda
