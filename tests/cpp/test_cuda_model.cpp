// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/model.hpp"
#include "engine/model.hpp"
#include "engine/safetensors.hpp"
#include "tiny_model.hpp"

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
