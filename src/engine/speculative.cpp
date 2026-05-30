// SPDX-License-Identifier: Apache-2.0
#include "engine/speculative.hpp"

#include <stdexcept>

namespace engine {
namespace {

int64_t argmax_row(const Tensor& logits, int64_t row) {
    const int64_t vocab = logits.dim(1);
    const float* r = logits.data() + row * vocab;
    int64_t best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (r[j] > r[best]) {
            best = j;
        }
    }
    return best;
}

} // namespace

std::vector<int64_t>
speculative_greedy(const Model& target,
                   const std::function<int64_t(const std::vector<int64_t>&)>& draft,
                   const std::vector<int64_t>& prompt,
                   int64_t max_tokens,
                   int64_t lookahead) {
    if (prompt.empty()) {
        throw std::invalid_argument("speculative_greedy: empty prompt");
    }
    if (lookahead < 1) {
        lookahead = 1;
    }

    std::vector<int64_t> out;
    std::vector<int64_t> ids = prompt;
    while (static_cast<int64_t>(out.size()) < max_tokens) {
        // Draft proposes `lookahead` tokens on the growing context.
        std::vector<int64_t> proposed;
        proposed.reserve(static_cast<std::size_t>(lookahead));
        std::vector<int64_t> context = ids;
        for (int64_t j = 0; j < lookahead; ++j) {
            const int64_t token = draft(context);
            proposed.push_back(token);
            context.push_back(token);
        }

        // Target verifies all proposals in a single forward over ids + proposed.
        std::vector<int64_t> full = ids;
        full.insert(full.end(), proposed.begin(), proposed.end());
        const Tensor logits = target.forward(full);
        const int64_t base = static_cast<int64_t>(ids.size());

        // Accept the longest prefix that matches the target's greedy argmax; every emitted token is
        // the target's argmax, so the output equals plain greedy decoding.
        bool rejected = false;
        for (int64_t j = 0; j < lookahead && static_cast<int64_t>(out.size()) < max_tokens; ++j) {
            const int64_t target_token = argmax_row(logits, base - 1 + j);
            out.push_back(target_token);
            ids.push_back(target_token);
            if (target_token != proposed[static_cast<std::size_t>(j)]) {
                rejected = true; // the draft diverged here; later proposals are off-context
                break;
            }
        }
        if (!rejected && static_cast<int64_t>(out.size()) < max_tokens) {
            const int64_t bonus = argmax_row(logits, base - 1 + lookahead);
            out.push_back(bonus);
            ids.push_back(bonus);
        }
    }
    return out;
}

} // namespace engine
