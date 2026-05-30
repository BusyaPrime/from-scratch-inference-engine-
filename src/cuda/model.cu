// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/kernels.hpp"
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
};

Tensor CudaModel::Impl::forward(const std::vector<int64_t>& ids) const {
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
    std::vector<int64_t> positions(static_cast<std::size_t>(seq));
    for (int64_t i = 0; i < seq; ++i) {
        positions[static_cast<std::size_t>(i)] = i;
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

        kernels::attention(d_q.get(),
                           d_k.get(),
                           d_v.get(),
                           d_attn.get(),
                           seq,
                           seq,
                           c.num_attention_heads,
                           c.num_key_value_heads,
                           c.head_dim,
                           /*query_offset=*/0);
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
