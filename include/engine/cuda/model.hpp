// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/cuda/block_manager.hpp"
#include "engine/cuda/kv_cache.hpp"
#include "engine/model_config.hpp"
#include "engine/safetensors.hpp"
#include "engine/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace engine::cuda {

// One sequence's work for a batched GPU step: its device KV cache (extended in place, non-owning)
// and the new tokens to process. A prefill item carries the whole prompt, a decode item one token.
// Mirrors the CPU BatchItem.
struct GpuBatchItem {
    GpuKVCache* cache;
    std::vector<int64_t> tokens;
};

// One sequence's work for a batched paged step: its block table inside a shared GpuBlockManager
// pool (extended in place, non-owning) and the new tokens. The K/V live in the manager, not here.
struct PagedBatchItem {
    SequenceBlocks* blocks;
    std::vector<int64_t> tokens;
};

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

    // Greedy decode using cached forward: returns up to max_tokens generated ids, stopping early at
    // eos_id (-1 disables). The next token is the argmax of the last position's logits.
    [[nodiscard]] std::vector<int64_t>
    generate(const std::vector<int64_t>& prompt, int64_t max_tokens, int64_t eos_id = -1) const;

    // Continuous-batching forward: process many sequences (mixed prefill and decode) in one set of
    // matmuls over their concatenated tokens, with per-sequence attention over each item's device
    // cache. Extends each item's cache and returns last-token logits per item [num_items,
    // vocab_size].
    [[nodiscard]] Tensor forward_batch(const std::vector<GpuBatchItem>& items) const;

    // Like forward_batch but the next token of each item is chosen by a device-side argmax, so only
    // the chosen token ids (not the full [num_items, vocab] logits) are copied back to the host.
    // Greedy only; use forward_batch when stochastic sampling is needed. Extends each item's cache.
    [[nodiscard]] std::vector<int64_t>
    forward_batch_argmax(const std::vector<GpuBatchItem>& items) const;

    // Same as forward_batch but over a shared paged pool (GpuBlockManager): true GPU
    // PagedAttention. Each item's K/V is scattered into its blocks, then gathered for attention.
    [[nodiscard]] Tensor forward_batch_paged(GpuBlockManager& manager,
                                             const std::vector<PagedBatchItem>& items) const;

    [[nodiscard]] const ModelConfig& config() const noexcept;

private:
    struct Impl;
    explicit CudaModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::cuda
