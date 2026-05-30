// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/cuda/kv_cache.hpp"
#include "engine/model_config.hpp"
#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace engine::cuda {

// GPU-resident forward of the Qwen2 model. Weights are uploaded to the device once; forward keeps
// activations on the device across layers and returns host logits. It mirrors the CPU
// Model::forward (full causal attention, no cache) so the two can be compared directly. The
// implementation is hidden behind a pointer so this header stays free of CUDA types.
class CudaModel {
public:
    static CudaModel from_pretrained(const std::string& model_dir);
    static CudaModel from_safetensors(ModelConfig config, const SafeTensors& weights);

    ~CudaModel();
    CudaModel(CudaModel&&) noexcept;
    CudaModel& operator=(CudaModel&&) noexcept;
    CudaModel(const CudaModel&) = delete;
    CudaModel& operator=(const CudaModel&) = delete;

    // Forward one sequence of token ids -> logits [seq_len, vocab_size].
    [[nodiscard]] Tensor forward(const std::vector<int64_t>& ids) const;

    // Cached forward: append this chunk's K/V to the device cache and attend over the whole cache,
    // so a prefill chunk followed by single-token steps reproduces the full forward on the GPU.
    [[nodiscard]] Tensor forward_with_cache(const std::vector<int64_t>& ids,
                                            GpuKVCache& cache) const;

    [[nodiscard]] const ModelConfig& config() const noexcept;

private:
    struct Impl;
    explicit CudaModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::cuda
