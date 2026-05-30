// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace engine {

// Source storage formats for weights. The reference path computes in F32;
// F16 and BF16 are converted to F32 on load.
enum class DType { F32, F16, BF16 };

// IEEE half-precision (binary16) bit pattern to float. Inline so the per-element
// conversion in the weight loader inlines and vectorizes.
[[nodiscard]] inline float f16_to_f32(uint16_t bits) noexcept {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const int32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FFu;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign; // signed zero
        } else {
            int32_t e = 1; // normalize the subnormal mantissa
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            out = sign | (static_cast<uint32_t>(e + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13); // inf or nan
    } else {
        out = sign | (static_cast<uint32_t>(exp + (127 - 15)) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// bfloat16 (the high 16 bits of a float32) bit pattern to float.
[[nodiscard]] inline float bf16_to_f32(uint16_t bits) noexcept {
    const uint32_t out = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// Stored byte width of one element of a dtype.
[[nodiscard]] inline std::size_t dtype_size(DType dt) noexcept {
    switch (dt) {
    case DType::F32:
        return 4;
    case DType::F16:
    case DType::BF16:
        return 2;
    }
    return 0;
}

// Map a safetensors dtype string to a DType, throwing for unsupported formats.
[[nodiscard]] inline DType dtype_from_string(const std::string& name) {
    if (name == "F32") {
        return DType::F32;
    }
    if (name == "F16") {
        return DType::F16;
    }
    if (name == "BF16") {
        return DType::BF16;
    }
    throw std::runtime_error("unsupported dtype '" + name + "'");
}

} // namespace engine
