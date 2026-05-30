// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"
#include "engine/speculative.hpp"
#include "tiny_model.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

int64_t argmax_last(const engine::Tensor& logits) {
    const int64_t rows = logits.dim(0);
    const int64_t vocab = logits.dim(1);
    const float* r = logits.data() + (rows - 1) * vocab;
    int64_t best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (r[j] > r[best]) {
            best = j;
        }
    }
    return best;
}

std::vector<int64_t>
plain_greedy(const engine::Model& model, const std::vector<int64_t>& prompt, int64_t n) {
    std::vector<int64_t> ids = prompt;
    std::vector<int64_t> out;
    for (int64_t i = 0; i < n; ++i) {
        const engine::Tensor logits = model.forward(ids);
        const int64_t token = argmax_last(logits);
        out.push_back(token);
        ids.push_back(token);
    }
    return out;
}

} // namespace

// A perfect (self) draft accepts every proposal, but the output must still be plain greedy.
TEST(Speculative, SelfDraftMatchesPlainGreedy) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> prompt = {1, 2, 3};
    const int64_t n = 6;
    const std::vector<int64_t> reference = plain_greedy(model, prompt, n);

    const auto draft = [&](const std::vector<int64_t>& context) {
        return argmax_last(model.forward(context));
    };
    const std::vector<int64_t> got = engine::speculative_greedy(model, draft, prompt, n, 4);
    EXPECT_EQ(got, reference);
}

// An adversarial draft (always proposes token 0) forces rejection every round; the output must
// still equal plain greedy, proving the verify/correct path.
TEST(Speculative, AdversarialDraftStillMatchesPlainGreedy) {
    const engine::Model model = tiny::tiny_model();
    const std::vector<int64_t> prompt = {1, 2, 3};
    const int64_t n = 6;
    const std::vector<int64_t> reference = plain_greedy(model, prompt, n);

    const auto draft = [](const std::vector<int64_t>&) { return static_cast<int64_t>(0); };
    const std::vector<int64_t> got = engine::speculative_greedy(model, draft, prompt, n, 4);
    EXPECT_EQ(got, reference);
    EXPECT_EQ(static_cast<int64_t>(got.size()), n);
}
