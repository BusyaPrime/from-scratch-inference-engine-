// SPDX-License-Identifier: Apache-2.0
#include "engine/dtype.hpp"

#include <cstring>

namespace engine {

float f16_to_f32(uint16_t bits) noexcept {
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

float bf16_to_f32(uint16_t bits) noexcept {
    const uint32_t out = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

} // namespace engine
