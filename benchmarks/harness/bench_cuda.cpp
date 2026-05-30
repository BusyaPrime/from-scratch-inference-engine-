// SPDX-License-Identifier: Apache-2.0
// Single-stream GPU latency/throughput benchmark for the CUDA path. Loads a model directory,
// times prefill (TTFT) and cached single-token decode (TPOT, throughput). Token ids are synthetic
// (timing does not depend on token semantics), so no tokenizer is needed here.
//
// Usage: bench_cuda [model_dir] [prompt_len] [max_tokens]

#include "engine/cuda/kv_cache.hpp"
#include "engine/cuda/model.hpp"
#include "engine/model_config.hpp"
#include "engine/tensor.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int64_t argmax_last_row(const engine::Tensor& logits) {
    const int64_t rows = logits.dim(0);
    const int64_t vocab = logits.dim(1);
    const float* row = logits.data() + (rows - 1) * vocab;
    int64_t best = 0;
    for (int64_t j = 1; j < vocab; ++j) {
        if (row[j] > row[best]) {
            best = j;
        }
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    const std::string model_dir = argc > 1 ? argv[1] : "weights/Qwen2.5-0.5B-Instruct";
    const int prompt_len = argc > 2 ? std::atoi(argv[2]) : 16;
    const int max_tokens = argc > 3 ? std::atoi(argv[3]) : 64;

    const engine::cuda::CudaModel model = engine::cuda::CudaModel::from_pretrained(model_dir);
    const engine::ModelConfig& c = model.config();
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;

    std::vector<int64_t> prompt(static_cast<std::size_t>(prompt_len));
    for (int i = 0; i < prompt_len; ++i) {
        prompt[static_cast<std::size_t>(i)] = i % c.vocab_size;
    }

    // Warm up: first launch pays cuBLAS init and kernel load; keep it out of the measurement.
    {
        engine::cuda::GpuKVCache warm(c.num_hidden_layers, kv_dim, 8);
        const engine::Tensor warm_logits = model.forward_with_cache({0, 1}, warm);
        (void)warm_logits;
    }

    engine::cuda::GpuKVCache cache(c.num_hidden_layers, kv_dim, prompt_len + max_tokens);

    const Clock::time_point prefill_start = Clock::now();
    engine::Tensor logits = model.forward_with_cache(prompt, cache);
    const double ttft_ms = ms_since(prefill_start);
    int64_t token = argmax_last_row(logits);

    double decode_ms = 0.0;
    int decode_steps = 0;
    for (int i = 1; i < max_tokens; ++i) {
        const Clock::time_point step_start = Clock::now();
        logits = model.forward_with_cache({token}, cache);
        decode_ms += ms_since(step_start);
        token = argmax_last_row(logits);
        ++decode_steps;
    }

    const double tpot_ms = decode_steps > 0 ? decode_ms / decode_steps : 0.0;
    const double total_s = (ttft_ms + decode_ms) / 1000.0;
    const double throughput = total_s > 0.0 ? max_tokens / total_s : 0.0;

    std::printf("model:        %s\n", model_dir.c_str());
    std::printf("prompt tokens: %d\n", prompt_len);
    std::printf("max tokens:    %d\n", max_tokens);
    std::printf("TTFT:          %.1f ms\n", ttft_ms);
    std::printf("TPOT:          %.1f ms\n", tpot_ms);
    std::printf("decode rate:   %.1f tok/s\n", tpot_ms > 0.0 ? 1000.0 / tpot_ms : 0.0);
    std::printf("throughput:    %.1f tok/s\n", throughput);
    return 0;
}
