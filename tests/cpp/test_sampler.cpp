// SPDX-License-Identifier: Apache-2.0
#include "engine/sampler.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(Sampler, GreedyWhenTemperatureZero) {
    engine::Sampler s(123);
    const std::vector<float> logits = {0.1f, 0.2f, 5.0f, 0.3f};
    engine::SamplingParams params;
    params.temperature = 0.0;
    EXPECT_EQ(s.sample(logits.data(), 4, params), 2);
}

TEST(Sampler, TopKOneAlwaysPicksArgmax) {
    engine::Sampler s(1);
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f};
    engine::SamplingParams params;
    params.top_k = 1;
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(s.sample(logits.data(), 3, params), 1);
    }
}

TEST(Sampler, SameSeedSameSequence) {
    const std::vector<float> logits = {0.5f, 1.5f, 1.0f, 0.2f};
    engine::SamplingParams params;
    engine::Sampler a(42);
    engine::Sampler b(42);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(a.sample(logits.data(), 4, params), b.sample(logits.data(), 4, params));
    }
}

TEST(Sampler, TopPRestrictsToDominantToken) {
    engine::Sampler s(7);
    const std::vector<float> logits = {0.0f, 10.0f, 0.0f, 0.0f}; // token 1 ~ 0.9999 of mass
    engine::SamplingParams params;
    params.top_p = 0.5;
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(s.sample(logits.data(), 4, params), 1);
    }
}

TEST(Sampler, DistributionFavorsHigherLogit) {
    engine::Sampler s(99);
    const std::vector<float> logits = {0.0f, 2.0f, 0.0f}; // softmax ~ [0.107, 0.787, 0.107]
    engine::SamplingParams params;
    int count_top = 0;
    for (int i = 0; i < 2000; ++i) {
        if (s.sample(logits.data(), 3, params) == 1) {
            ++count_top;
        }
    }
    EXPECT_GT(count_top, 1000);
}
