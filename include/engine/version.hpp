// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace engine {

// Semantic version of the engine core, surfaced across the binding boundary.
[[nodiscard]] std::string_view version() noexcept;

}  // namespace engine
