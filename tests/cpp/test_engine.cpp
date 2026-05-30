// SPDX-License-Identifier: Apache-2.0
#include "engine/engine.hpp"
#include "engine/model.hpp"
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

// Reference greedy decode: repeated full forward over the growing sequence, same sampler path.
std::vector<int64_t>
ref_greedy(const engine::Model& model, const std::vector<int64_t>& prompt, int64_t n) {
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

} // namespace

TEST(Engine, GreedyGenerationMatchesReference) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> prompt = {1, 2, 3};
    const int64_t n = 6;
    const std::vector<int64_t> ref = ref_greedy(model, prompt, n);

    engine::Engine eng(model, /*block_size=*/4, /*num_blocks=*/64, /*seed=*/123);
    const std::vector<int64_t> got = eng.generate(prompt, greedy(), n);
    EXPECT_EQ(got, ref);
}

// Two requests scheduled together must each produce exactly what they would produce alone.
TEST(Engine, ConcurrentRequestsAreIndependent) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> pa = {1, 2, 3};
    const std::vector<int64_t> pb = {6, 7, 8, 9};
    const int64_t n = 5;

    engine::Engine ea(model, 4, 64, 7);
    const std::vector<int64_t> base_a = ea.generate(pa, greedy(), n);
    engine::Engine eb(model, 4, 64, 7);
    const std::vector<int64_t> base_b = eb.generate(pb, greedy(), n);

    engine::Engine eng(model, 4, 64, 7);
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

// A tight pool (only enough blocks for one sequence) still completes both via FCFS serialization.
TEST(Engine, TightPoolSerializesRequests) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> pa = {1, 2, 3};
    const std::vector<int64_t> pb = {4, 5, 6};
    const int64_t n = 4;

    engine::Engine base_a(model, 8, 64, 5);
    const std::vector<int64_t> ref_a = base_a.generate(pa, greedy(), n);
    engine::Engine base_b(model, 8, 64, 5);
    const std::vector<int64_t> ref_b = base_b.generate(pb, greedy(), n);

    // block_size 8, a single block total: 3-token prompt + 4 tokens = 7 slots fit one sequence,
    // so the second must wait (backpressure) until the first finishes and frees the block.
    engine::Engine eng(model, /*block_size=*/8, /*num_blocks=*/1, /*seed=*/5);
    const int64_t id_a = eng.add_request({pa, greedy(), n, -1});
    const int64_t id_b = eng.add_request({pb, greedy(), n, -1});
    int64_t guard = 0;
    while (eng.has_work() && guard++ < 1000) {
        eng.step();
    }
    EXPECT_EQ(eng.status(id_a), engine::SeqStatus::Finished);
    EXPECT_EQ(eng.status(id_b), engine::SeqStatus::Finished);
    EXPECT_EQ(eng.output(id_a), ref_a);
    EXPECT_EQ(eng.output(id_b), ref_b);
}

// Under memory pressure the engine must preempt a running sequence (free its blocks) so others can
// proceed, then recompute the preempted one from prompt + generated tokens and still match.
TEST(Engine, PreemptionRecomputesAndMatches) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> pa = {1, 2, 3, 4};
    const std::vector<int64_t> pb = {5, 6, 7, 8};
    const int64_t n = 4;

    engine::Engine base_a(model, 4, 64, 9);
    const std::vector<int64_t> ref_a = base_a.generate(pa, greedy(), n);
    engine::Engine base_b(model, 4, 64, 9);
    const std::vector<int64_t> ref_b = base_b.generate(pb, greedy(), n);

    // block_size 4, only 2 blocks: each prompt prefills into one block, then both need a second
    // block on the same step with none free, which forces a preemption.
    engine::Engine eng(model, /*block_size=*/4, /*num_blocks=*/2, /*seed=*/9);
    const int64_t id_a = eng.add_request({pa, greedy(), n, -1});
    const int64_t id_b = eng.add_request({pb, greedy(), n, -1});
    int64_t guard = 0;
    while (eng.has_work() && guard++ < 2000) {
        eng.step();
    }
    EXPECT_GT(eng.preemptions(), 0);
    EXPECT_EQ(eng.status(id_a), engine::SeqStatus::Finished);
    EXPECT_EQ(eng.status(id_b), engine::SeqStatus::Finished);
    EXPECT_EQ(eng.output(id_a), ref_a);
    EXPECT_EQ(eng.output(id_b), ref_b);
}

TEST(Engine, EosStopsGeneration) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> prompt = {2, 4};
    const std::vector<int64_t> full = ref_greedy(model, prompt, 4);
    const int64_t eos = full.front();

    engine::Engine eng(model, 4, 64, 1);
    const std::vector<int64_t> got = eng.generate(prompt, greedy(), 4, eos);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got.back(), eos);
    EXPECT_EQ(got.size(), 1u); // first token is EOS, so generation stops immediately
}

// A request sharing a prompt prefix with an earlier one reuses the cached prefix blocks and still
// produces the same tokens as a run with prefix caching disabled.
TEST(Engine, PrefixCacheReusesSharedPromptAndMatches) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> pa = {1, 2, 3, 4, 5, 6};
    const std::vector<int64_t> pb = {1, 2, 3, 4, 9, 8}; // shares the first two blocks (block_size 2)
    const int64_t n = 5;

    engine::Engine reference(model, /*block_size=*/2, /*num_blocks=*/256, /*seed=*/3,
                             /*max_batch=*/256, /*enable_prefix_cache=*/false);
    const std::vector<int64_t> ref_a = reference.generate(pa, greedy(), n);
    const std::vector<int64_t> ref_b = reference.generate(pb, greedy(), n);

    engine::Engine eng(model, 2, 256, 3, 256, /*enable_prefix_cache=*/true);
    const std::vector<int64_t> got_a = eng.generate(pa, greedy(), n);
    const std::vector<int64_t> got_b = eng.generate(pb, greedy(), n);

    EXPECT_EQ(got_a, ref_a);
    EXPECT_EQ(got_b, ref_b);
    EXPECT_GT(eng.prefix_hits(), 0); // the second request reused the cached prefix
}
