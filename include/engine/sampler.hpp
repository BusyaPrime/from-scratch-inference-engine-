// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <random>

namespace engine {

struct SamplingParams {
    double temperature = 1.0; // <= 0 selects greedy (argmax)
    int64_t top_k = 0;        // 0 disables top-k
    double top_p = 1.0;       // >= 1 disables top-p (nucleus)
};

// Stateful sampler with a seeded RNG for reproducible sampling.
class Sampler {
public:
    explicit Sampler(uint64_t seed) noexcept : rng_(seed) {}

    // Sample a token id from final-position logits[vocab]. Greedy (argmax) when
    // temperature <= 0; otherwise temperature -> top-k -> top-p -> multinomial.
    [[nodiscard]] int64_t sample(const float* logits, int64_t vocab, const SamplingParams& params);

private:
    std::mt19937_64 rng_;
};

} // namespace engine
