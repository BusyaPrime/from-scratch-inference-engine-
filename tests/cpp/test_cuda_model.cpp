// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/kv_cache.hpp"
#include "engine/cuda/model.hpp"
#include "engine/model.hpp"
#include "engine/safetensors.hpp"
#include "tiny_model.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

// The device-resident GPU forward must match the CPU forward on identical weights. The CPU forward
// is itself anchored to transformers, so this validates the whole GPU path end to end.
TEST(CudaModel, ForwardMatchesCpuForward) {
    const engine::ModelConfig config = tiny::tiny_config();
    engine::SafeTensors weights = engine::SafeTensors::from_tensors(tiny::tiny_weights());

    const engine::cuda::CudaModel gpu = engine::cuda::CudaModel::from_safetensors(config, weights);
    const engine::Model cpu = engine::Model::from_safetensors(config, std::move(weights));

    const std::vector<int64_t> seq = {1, 5, 2, 8, 3, 0, 7};
    const engine::Tensor cpu_logits = cpu.forward(seq);
    const engine::Tensor gpu_logits = gpu.forward(seq);

    ASSERT_EQ(gpu_logits.dim(0), cpu_logits.dim(0));
    ASSERT_EQ(gpu_logits.dim(1), cpu_logits.dim(1));
    const int64_t n = cpu_logits.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_logits.data()[i], cpu_logits.data()[i], 1e-3f) << "index " << i;
    }
}

// A prefill chunk followed by single-token decode steps through the device KV cache must reproduce
// the full GPU forward, proving cached decode is correct.
TEST(CudaModel, CachedDecodeMatchesFullForward) {
    const engine::ModelConfig config = tiny::tiny_config();
    const engine::SafeTensors weights = engine::SafeTensors::from_tensors(tiny::tiny_weights());
    const engine::cuda::CudaModel gpu = engine::cuda::CudaModel::from_safetensors(config, weights);

    const std::vector<int64_t> seq = {1, 5, 2, 8, 3, 0, 7};
    const int64_t vocab = config.vocab_size;
    const engine::Tensor full = gpu.forward(seq);

    engine::cuda::GpuKVCache cache(config.num_hidden_layers,
                                   config.num_key_value_heads * config.head_dim,
                                   /*capacity=*/64);
    const int64_t prefill = 3;
    const std::vector<int64_t> head(seq.begin(), seq.begin() + prefill);
    const engine::Tensor prefill_logits = gpu.forward_with_cache(head, cache);
    ASSERT_EQ(prefill_logits.dim(0), prefill);
    for (int64_t i = 0; i < prefill; ++i) {
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(prefill_logits.data()[i * vocab + j], full.data()[i * vocab + j], 1e-4f);
        }
    }

    for (int64_t t = prefill; t < static_cast<int64_t>(seq.size()); ++t) {
        const engine::Tensor step =
            gpu.forward_with_cache({seq[static_cast<std::size_t>(t)]}, cache);
        ASSERT_EQ(step.dim(0), 1);
        for (int64_t j = 0; j < vocab; ++j) {
            EXPECT_NEAR(step.data()[j], full.data()[t * vocab + j], 1e-4f);
        }
    }
    EXPECT_EQ(cache.length(), static_cast<int64_t>(seq.size()));
}
