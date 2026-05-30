// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/model.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace engine {

// Speculative greedy decoding. Each round the draft proposes `lookahead` tokens
// (draft(context) -> next token, called repeatedly on the growing context); the target verifies
// them in one forward, accepts the longest prefix matching its own greedy argmax, and emits a
// bonus token when the whole proposal is accepted. The emitted tokens are exactly the target's
// plain greedy decoding -- speculation changes speed (with a cheaper draft), never the output.
// Returns the generated token ids (length max_tokens).
[[nodiscard]] std::vector<int64_t>
speculative_greedy(const Model& target,
                   const std::function<int64_t(const std::vector<int64_t>&)>& draft,
                   const std::vector<int64_t>& prompt,
                   int64_t max_tokens,
                   int64_t lookahead);

} // namespace engine
