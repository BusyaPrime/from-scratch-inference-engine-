// SPDX-License-Identifier: Apache-2.0
#include "engine/cuda/engine.hpp"
#include "engine/cuda/kv_cache.hpp"
#include "engine/cuda/model.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::cuda {

struct CudaEngine::Impl {
    struct Seq {
        Request request;
        std::unique_ptr<GpuKVCache> cache;
        std::vector<int64_t> output;
        SeqStatus status = SeqStatus::Waiting;
        bool prefilled = false;
    };

    const CudaModel& model;
    Sampler sampler;
    int64_t max_batch;
    std::deque<int64_t> waiting;
    std::vector<int64_t> running;
    std::unordered_map<int64_t, Seq> sequences;
    int64_t next_id = 0;

    Impl(const CudaModel& m, uint64_t seed, int64_t batch)
        : model(m), sampler(seed), max_batch(batch) {}

    void step();
};

void CudaEngine::Impl::step() {
    const ModelConfig& c = model.config();
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;

    // Admit waiting requests (FCFS) up to the batch cap, giving each its own device cache sized for
    // prompt + generated tokens.
    while (!waiting.empty() && static_cast<int64_t>(running.size()) < max_batch) {
        const int64_t id = waiting.front();
        Seq& seq = sequences.at(id);
        const int64_t capacity =
            static_cast<int64_t>(seq.request.prompt.size()) + seq.request.max_tokens;
        seq.cache = std::make_unique<GpuKVCache>(c.num_hidden_layers, kv_dim, capacity);
        seq.status = SeqStatus::Running;
        running.push_back(id);
        waiting.pop_front();
    }

    std::vector<GpuBatchItem> items;
    std::vector<int64_t> batched_ids;
    items.reserve(running.size());
    batched_ids.reserve(running.size());
    for (const int64_t id : running) {
        Seq& seq = sequences.at(id);
        std::vector<int64_t> tokens =
            seq.prefilled ? std::vector<int64_t>{seq.output.back()} : seq.request.prompt;
        items.push_back({seq.cache.get(), std::move(tokens)});
        batched_ids.push_back(id);
    }
    if (items.empty()) {
        return;
    }

    const Tensor logits = model.forward_batch(items);
    const int64_t vocab = c.vocab_size;

    bool any_finished = false;
    for (std::size_t k = 0; k < batched_ids.size(); ++k) {
        Seq& seq = sequences.at(batched_ids[k]);
        const int64_t token = sampler.sample(
            logits.data() + static_cast<int64_t>(k) * vocab, vocab, seq.request.params);
        seq.prefilled = true;
        seq.output.push_back(token);
        const bool hit_eos = seq.request.eos_id >= 0 && token == seq.request.eos_id;
        const bool hit_len = static_cast<int64_t>(seq.output.size()) >= seq.request.max_tokens;
        if (hit_eos || hit_len) {
            seq.status = SeqStatus::Finished;
            seq.cache.reset(); // free device memory immediately
            any_finished = true;
        }
    }

    if (any_finished) {
        std::vector<int64_t> still_running;
        still_running.reserve(running.size());
        for (const int64_t id : running) {
            if (sequences.at(id).status != SeqStatus::Finished) {
                still_running.push_back(id);
            }
        }
        running = std::move(still_running);
    }
}

CudaEngine::CudaEngine(const CudaModel& model, uint64_t seed, int64_t max_batch)
    : impl_(std::make_unique<Impl>(model, seed, max_batch)) {}
CudaEngine::~CudaEngine() = default;
CudaEngine::CudaEngine(CudaEngine&&) noexcept = default;
CudaEngine& CudaEngine::operator=(CudaEngine&&) noexcept = default;

int64_t CudaEngine::add_request(Request request) {
    if (request.prompt.empty()) {
        throw std::invalid_argument("CudaEngine: empty prompt");
    }
    const int64_t id = impl_->next_id++;
    Impl::Seq seq;
    seq.request = std::move(request);
    impl_->sequences.emplace(id, std::move(seq));
    impl_->waiting.push_back(id);
    return id;
}

bool CudaEngine::has_work() const noexcept {
    return !impl_->waiting.empty() || !impl_->running.empty();
}

void CudaEngine::step() {
    impl_->step();
}

const std::vector<int64_t>& CudaEngine::output(int64_t id) const {
    return impl_->sequences.at(id).output;
}

SeqStatus CudaEngine::status(int64_t id) const {
    return impl_->sequences.at(id).status;
}

std::vector<int64_t> CudaEngine::generate(const std::vector<int64_t>& prompt,
                                          const SamplingParams& params,
                                          int64_t max_tokens,
                                          int64_t eos_id) {
    Request request;
    request.prompt = prompt;
    request.params = params;
    request.max_tokens = max_tokens;
    request.eos_id = eos_id;
    const int64_t id = add_request(std::move(request));
    while (impl_->sequences.at(id).status != SeqStatus::Finished) {
        step();
    }
    return impl_->sequences.at(id).output;
}

} // namespace engine::cuda
