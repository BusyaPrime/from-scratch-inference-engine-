// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace engine {

// Source storage formats for weights. The reference path computes in F32;
// F16 and BF16 are converted to F32 on load.
enum class DType { F32, F16, BF16 };

// IEEE half-precision (binary16) bit pattern to float.
[[nodiscard]] float f16_to_f32(uint16_t bits) noexcept;

// bfloat16 (the high 16 bits of a float32) bit pattern to float.
[[nodiscard]] float bf16_to_f32(uint16_t bits) noexcept;

} // namespace engine
