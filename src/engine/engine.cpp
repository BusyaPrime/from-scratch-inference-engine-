// SPDX-License-Identifier: Apache-2.0
#include "engine/engine.hpp"

#include <iterator>
#include <stdexcept>
#include <utility>

namespace engine {
namespace {

// Tokens to feed a sequence this step. A fresh or recomputed sequence re-runs prompt + everything
// generated so far (recompute rebuilds the freed KV cache); a prefilled sequence decodes its last
// token.
std::vector<int64_t> step_tokens(const EngineSequence& seq) {
    if (seq.prefilled) {
        return {seq.output.back()};
    }
    std::vector<int64_t> tokens = seq.request.prompt;
    tokens.insert(tokens.end(), seq.output.begin(), seq.output.end());
    return tokens;
}

} // namespace

Engine::Engine(
    const Model& model, int64_t block_size, int64_t num_blocks, uint64_t seed, int64_t max_batch)
    : model_(model), manager_(model.config().num_hidden_layers,
                              model.config().num_key_value_heads * model.config().head_dim,
                              block_size,
                              num_blocks),
      sampler_(seed), max_batch_(max_batch) {}

int64_t Engine::add_request(Request request) {
    if (request.prompt.empty()) {
        throw std::invalid_argument("Engine: empty prompt");
    }
    const int64_t id = next_id_++;
    EngineSequence seq;
    seq.id = id;
    seq.request = std::move(request);
    sequences_.emplace(id, std::move(seq));
    waiting_.push_back(id);
    return id;
}

void Engine::step() {
    // Admit waiting requests (FCFS) while the batch cap and free pool allow. A prompt that does
    // not fit the whole pool stays queued; FCFS stops at the first that cannot be admitted.
    while (!waiting_.empty() && static_cast<int64_t>(running_.size()) < max_batch_) {
        const int64_t id = waiting_.front();
        EngineSequence& seq = sequences_.at(id);
        const int64_t need =
            seq.prefilled ? 1 : static_cast<int64_t>(seq.request.prompt.size() + seq.output.size());
        if (!manager_.can_append(seq.blocks, need)) {
            break;
        }
        seq.status = SeqStatus::Running;
        running_.push_back(id);
        waiting_.pop_front();
    }

    // Assemble a batch under a cumulative block budget (every item draws from the same pool this
    // step). If all running sequences are block-starved and more than one is running, preempt the
    // youngest block-holding sequence and retry; it recomputes from prompt + generated tokens when
    // resumed.
    std::vector<BatchItem> items;
    std::vector<int64_t> batched_ids;
    for (;;) {
        items.clear();
        batched_ids.clear();
        int64_t budget = manager_.free_blocks();
        for (const int64_t id : running_) {
            EngineSequence& seq = sequences_.at(id);
            std::vector<int64_t> tokens = step_tokens(seq);
            const auto need = static_cast<int64_t>(tokens.size());
            const auto have = static_cast<int64_t>(seq.blocks.block_table.size());
            const int64_t delta = manager_.blocks_for(seq.blocks.length + need) - have;
            if (delta > budget) {
                continue; // cannot grow within the remaining pool this step
            }
            budget -= delta;
            items.push_back({&seq.blocks, std::move(tokens)});
            batched_ids.push_back(id);
        }
        if (!items.empty() || running_.size() <= 1 || !preempt_youngest()) {
            break;
        }
    }
    if (items.empty()) {
        return;
    }

    const Tensor logits = model_.forward_batch(manager_, items);
    const int64_t vocab = model_.config().vocab_size;

    bool any_finished = false;
    for (std::size_t k = 0; k < batched_ids.size(); ++k) {
        EngineSequence& seq = sequences_.at(batched_ids[k]);
        const int64_t token = sampler_.sample(
            logits.data() + static_cast<int64_t>(k) * vocab, vocab, seq.request.params);
        seq.prefilled = true;
        seq.output.push_back(token);
        const bool hit_eos = seq.request.eos_id >= 0 && token == seq.request.eos_id;
        const bool hit_len = static_cast<int64_t>(seq.output.size()) >= seq.request.max_tokens;
        if (hit_eos || hit_len) {
            seq.status = SeqStatus::Finished;
            manager_.free(seq.blocks);
            any_finished = true;
        }
    }

    if (any_finished) {
        std::vector<int64_t> still_running;
        still_running.reserve(running_.size());
        for (const int64_t id : running_) {
            if (sequences_.at(id).status != SeqStatus::Finished) {
                still_running.push_back(id);
            }
        }
        running_ = std::move(still_running);
    }
}

bool Engine::preempt_youngest() {
    for (auto it = running_.rbegin(); it != running_.rend(); ++it) {
        EngineSequence& seq = sequences_.at(*it);
        if (seq.blocks.block_table.empty()) {
            continue; // holds no blocks; preempting it would free nothing
        }
        manager_.free(seq.blocks);
        seq.prefilled = false;
        seq.status = SeqStatus::Waiting;
        const int64_t id = *it;
        running_.erase(std::next(it).base());
        waiting_.push_front(id);
        ++preemptions_;
        return true;
    }
    return false;
}

const std::vector<int64_t>& Engine::output(int64_t id) const {
    return sequences_.at(id).output;
}

SeqStatus Engine::status(int64_t id) const {
    return sequences_.at(id).status;
}

std::vector<int64_t> Engine::generate(const std::vector<int64_t>& prompt,
                                      const SamplingParams& params,
                                      int64_t max_tokens,
                                      int64_t eos_id) {
    Request request;
    request.prompt = prompt;
    request.params = params;
    request.max_tokens = max_tokens;
    request.eos_id = eos_id;
    const int64_t id = add_request(std::move(request));
    while (sequences_.at(id).status != SeqStatus::Finished) {
        step();
    }
    return sequences_.at(id).output;
}

} // namespace engine
