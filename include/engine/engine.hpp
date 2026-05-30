// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/block_manager.hpp"
#include "engine/model.hpp"
#include "engine/sampler.hpp"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace engine {

// A generation request: prompt tokens, sampling controls, and stop conditions.
struct Request {
    std::vector<int64_t> prompt;
    SamplingParams params;
    int64_t max_tokens = 16;
    int64_t eos_id = -1; // -1 disables EOS stopping
};

enum class SeqStatus { Waiting, Running, Finished };

// A request tracked through scheduling: its paging state, generated tokens, and status.
struct EngineSequence {
    int64_t id = 0;
    Request request;
    SequenceBlocks blocks;
    std::vector<int64_t> output;
    SeqStatus status = SeqStatus::Waiting;
    bool prefilled = false;
};

// Continuous-batching engine. Schedules many requests over one shared block pool, running
// prefill and decode mixed in each batched step. Admission is FCFS and gated on the free pool;
// finished sequences return their blocks immediately. The referenced Model must outlive the
// Engine.
class Engine {
public:
    Engine(const Model& model,
           int64_t block_size,
           int64_t num_blocks,
           uint64_t seed,
           int64_t max_batch = 256);

    // Queue a request; returns its sequence id.
    int64_t add_request(Request request);

    // Whether any sequence is still waiting or running.
    [[nodiscard]] bool has_work() const noexcept { return !waiting_.empty() || !running_.empty(); }

    // Run one batched step: admit waiting requests within budget, batch prefill/decode for the
    // running set, sample one token each, and retire finished sequences.
    void step();

    // Generated tokens for a sequence (complete once finished, partial while running).
    [[nodiscard]] const std::vector<int64_t>& output(int64_t id) const;
    [[nodiscard]] SeqStatus status(int64_t id) const;

    // Total preemptions so far (a sequence evicted under memory pressure and later recomputed).
    [[nodiscard]] int64_t preemptions() const noexcept { return preemptions_; }

    // Convenience: drive the engine until a single fresh request completes; returns its tokens.
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt,
                                  const SamplingParams& params,
                                  int64_t max_tokens,
                                  int64_t eos_id = -1);

private:
    // Free the youngest block-holding running sequence so others can proceed; returns false if no
    // running sequence holds blocks. The freed sequence recomputes from prompt + output on resume.
    bool preempt_youngest();

    const Model& model_;
    BlockManager manager_;
    Sampler sampler_;
    int64_t max_batch_;
    std::deque<int64_t> waiting_;
    std::vector<int64_t> running_;
    std::unordered_map<int64_t, EngineSequence> sequences_;
    int64_t next_id_ = 0;
    int64_t preemptions_ = 0;
};

} // namespace engine
