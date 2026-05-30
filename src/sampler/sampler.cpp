// SPDX-License-Identifier: Apache-2.0
#include "engine/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace engine {
namespace {

int64_t argmax(const float* logits, int64_t vocab) {
    int64_t best = 0;
    for (int64_t i = 1; i < vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

} // namespace

int64_t Sampler::sample(const float* logits, int64_t vocab, const SamplingParams& params) {
    if (vocab <= 0) {
        throw std::invalid_argument("sample: vocab must be positive");
    }
    if (params.temperature <= 0.0) {
        return argmax(logits, vocab);
    }

    // Indices ordered by logit, descending.
    std::vector<int64_t> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::sort(order.begin(), order.end(), [logits](int64_t a, int64_t b) {
        return logits[a] > logits[b];
    });

    int64_t keep = vocab;
    if (params.top_k > 0 && params.top_k < keep) {
        keep = params.top_k;
    }

    // Softmax over the kept logits (with temperature), stabilized by the max.
    const double inv_temp = 1.0 / params.temperature;
    const double max_logit = static_cast<double>(logits[order[0]]) * inv_temp;
    std::vector<double> probs(static_cast<std::size_t>(keep));
    double sum = 0.0;
    for (int64_t i = 0; i < keep; ++i) {
        const double e = std::exp(
            static_cast<double>(logits[order[static_cast<std::size_t>(i)]]) * inv_temp - max_logit);
        probs[static_cast<std::size_t>(i)] = e;
        sum += e;
    }
    for (double& v : probs) {
        v /= sum;
    }

    // Nucleus (top-p): smallest prefix whose cumulative probability reaches top_p.
    int64_t nucleus = keep;
    if (params.top_p < 1.0) {
        double cumulative = 0.0;
        for (int64_t i = 0; i < keep; ++i) {
            cumulative += probs[static_cast<std::size_t>(i)];
            if (cumulative >= params.top_p) {
                nucleus = i + 1;
                break;
            }
        }
    }

    double nucleus_sum = 0.0;
    for (int64_t i = 0; i < nucleus; ++i) {
        nucleus_sum += probs[static_cast<std::size_t>(i)];
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double threshold = uniform(rng_) * nucleus_sum;
    double cumulative = 0.0;
    for (int64_t i = 0; i < nucleus; ++i) {
        cumulative += probs[static_cast<std::size_t>(i)];
        if (threshold <= cumulative) {
            return order[static_cast<std::size_t>(i)];
        }
    }
    return order[static_cast<std::size_t>(nucleus - 1)];
}

} // namespace engine
