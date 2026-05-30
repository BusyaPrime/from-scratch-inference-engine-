// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/engine.hpp"
#include "engine/cuda/model.hpp"
#include "engine/engine.hpp"
#include "engine/sampler.hpp"
#include "tiny_model.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

engine::SamplingParams greedy() {
    engine::SamplingParams p;
    p.temperature = 0.0; // argmax
    return p;
}

// Reference greedy via repeated full GPU forward and the same sampler the engine uses.
std::vector<int64_t>
ref_greedy(const engine::cuda::CudaModel& model, const std::vector<int64_t>& prompt, int64_t n) {
    engine::Sampler sampler(123);
    std::vector<int64_t> ids = prompt;
    std::vector<int64_t> out;
    const int64_t vocab = model.config().vocab_size;
    for (int64_t i = 0; i < n; ++i) {
        const engine::Tensor logits = model.forward(ids);
        const int64_t last = (static_cast<int64_t>(ids.size()) - 1) * vocab;
        const int64_t token = sampler.sample(logits.data() + last, vocab, greedy());
        out.push_back(token);
        ids.push_back(token);
    }
    return out;
}

engine::cuda::CudaModel make_model() {
    return engine::cuda::CudaModel::from_safetensors(
        tiny::tiny_config(), engine::SafeTensors::from_tensors(tiny::tiny_weights()));
}

} // namespace

// The GPU engine's greedy output must match a full-forward greedy reference (same sampler path).
TEST(CudaEngine, GreedyGenerationMatchesReference) {
    const engine::cuda::CudaModel model = make_model();
    const std::vector<int64_t> prompt = {1, 2, 3};
    const int64_t n = 6;
    const std::vector<int64_t> reference = ref_greedy(model, prompt, n);

    engine::cuda::CudaEngine eng(model, /*seed=*/123);
    const std::vector<int64_t> got = eng.generate(prompt, greedy(), n);
    EXPECT_EQ(got, reference);
}

// Two requests scheduled together must each produce exactly what they would produce alone.
TEST(CudaEngine, ConcurrentRequestsAreIndependent) {
    const engine::cuda::CudaModel model = make_model();
    const std::vector<int64_t> pa = {1, 2, 3};
    const std::vector<int64_t> pb = {6, 7, 8, 9};
    const int64_t n = 5;

    engine::cuda::CudaEngine ea(model, 7);
    const std::vector<int64_t> base_a = ea.generate(pa, greedy(), n);
    engine::cuda::CudaEngine eb(model, 7);
    const std::vector<int64_t> base_b = eb.generate(pb, greedy(), n);

    engine::cuda::CudaEngine eng(model, 7);
    const int64_t id_a = eng.add_request({pa, greedy(), n, -1});
    const int64_t id_b = eng.add_request({pb, greedy(), n, -1});
    while (eng.has_work()) {
        eng.step();
    }

    EXPECT_EQ(eng.output(id_a), base_a);
    EXPECT_EQ(eng.output(id_b), base_b);
    EXPECT_EQ(eng.status(id_a), engine::SeqStatus::Finished);
    EXPECT_EQ(eng.status(id_b), engine::SeqStatus::Finished);
}
