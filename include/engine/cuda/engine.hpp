// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/cuda/model.hpp"
#include "engine/engine.hpp"  // Request, SeqStatus
#include "engine/sampler.hpp" // SamplingParams, Sampler

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::cuda {

// Continuous-batching engine on the GPU: schedules many requests, driving CudaModel::forward_batch
// each step with a per-sequence device KV cache. Mirrors the CPU Engine (FCFS admission up to a
// batch cap, per-sequence sampling, EOS/max-token retirement). The referenced CudaModel must
// outlive the engine. Reuses the CPU Request/SeqStatus/Sampler types.
class CudaEngine {
public:
    CudaEngine(const CudaModel& model, uint64_t seed, int64_t max_batch = 256);
    ~CudaEngine();
    CudaEngine(CudaEngine&&) noexcept;
    CudaEngine& operator=(CudaEngine&&) noexcept;
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    // Queue a request; returns its sequence id.
    int64_t add_request(Request request);

    [[nodiscard]] bool has_work() const noexcept;

    // Run one batched step: admit waiting requests, batch prefill/decode for the running set,
    // sample one token each, and retire finished sequences.
    void step();

    [[nodiscard]] const std::vector<int64_t>& output(int64_t id) const;
    [[nodiscard]] SeqStatus status(int64_t id) const;

    // Drive the engine until a single fresh request completes; returns its generated token ids.
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt,
                                  const SamplingParams& params,
                                  int64_t max_tokens,
                                  int64_t eos_id = -1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::cuda
