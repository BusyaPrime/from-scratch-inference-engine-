// SPDX-License-Identifier: Apache-2.0
#include "engine/nn.hpp"
#include "engine/tensor.hpp"
#include "engine/tensor_parallel.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

engine::Tensor random_tensor(std::vector<int64_t> shape, std::mt19937& rng, float sd) {
    engine::Tensor t(std::move(shape));
    std::normal_distribution<float> nd(0.0f, sd);
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        t.data()[i] = nd(rng);
    }
    return t;
}

} // namespace

// Column-parallel: shard the weight along its output dimension, run each shard, concat the outputs.
TEST(TensorParallel, ColumnParallelMatchesUnsharded) {
    std::mt19937 rng(3);
    const engine::Tensor x = random_tensor({4, 256}, rng, 0.1f);
    const engine::Tensor w = random_tensor({512, 256}, rng, 0.02f);
    const engine::Tensor reference = engine::linear(x, w);

    const std::vector<engine::Tensor> shards = engine::tp::split_rows(w, 4);
    std::vector<engine::Tensor> partials;
    partials.reserve(shards.size());
    for (const engine::Tensor& shard : shards) {
        partials.push_back(engine::linear(x, shard));
    }
    const engine::Tensor combined = engine::tp::concat_columns(partials);

    ASSERT_EQ(combined.dim(0), reference.dim(0));
    ASSERT_EQ(combined.dim(1), reference.dim(1));
    for (int64_t i = 0; i < reference.numel(); ++i) {
        EXPECT_NEAR(combined.data()[i], reference.data()[i], 1e-4f) << "index " << i;
    }
}

// Row-parallel: shard the weight and input along the contraction dimension, run each shard, sum.
TEST(TensorParallel, RowParallelMatchesUnsharded) {
    std::mt19937 rng(7);
    const engine::Tensor x = random_tensor({4, 256}, rng, 0.1f);
    const engine::Tensor w = random_tensor({64, 256}, rng, 0.02f);
    const engine::Tensor reference = engine::linear(x, w);

    const std::vector<engine::Tensor> weight_shards = engine::tp::split_columns(w, 4);
    const std::vector<engine::Tensor> input_shards = engine::tp::split_columns(x, 4);
    std::vector<engine::Tensor> partials;
    partials.reserve(weight_shards.size());
    for (std::size_t s = 0; s < weight_shards.size(); ++s) {
        partials.push_back(engine::linear(input_shards[s], weight_shards[s]));
    }
    const engine::Tensor combined = engine::tp::sum(partials);

    ASSERT_EQ(combined.dim(0), reference.dim(0));
    ASSERT_EQ(combined.dim(1), reference.dim(1));
    for (int64_t i = 0; i < reference.numel(); ++i) {
        EXPECT_NEAR(combined.data()[i], reference.data()[i], 1e-4f) << "index " << i;
    }
}
