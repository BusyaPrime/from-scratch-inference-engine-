// SPDX-License-Identifier: Apache-2.0
#include "engine/safetensors.hpp"

#include "engine/dtype.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace engine {
namespace {

uint16_t little_endian_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

} // namespace

SafeTensors SafeTensors::parse(const uint8_t* data, std::size_t size) {
    if (size < 8) {
        throw std::runtime_error("safetensors: buffer smaller than the 8-byte header length");
    }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    // size >= 8 above, so size - 8 cannot underflow; this avoids the 8 + header_len overflow.
    if (header_len > static_cast<uint64_t>(size) - 8) {
        throw std::runtime_error("safetensors: header length exceeds the buffer");
    }

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(reinterpret_cast<const char*>(data + 8),
                                       reinterpret_cast<const char*>(data + 8) + header_len);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("safetensors: invalid header JSON: ") + e.what());
    }

    const uint8_t* base = data + 8 + header_len;
    const uint64_t region = static_cast<uint64_t>(size) - 8 - header_len;

    std::vector<std::pair<uint64_t, uint64_t>> spans; // for the non-overlap check
    SafeTensors st;
    try {
        for (const auto& item : header.items()) {
            const std::string& name = item.key();
            if (name == "__metadata__") {
                continue;
            }
            const auto& meta = item.value();
            const DType dtype = dtype_from_string(meta.at("dtype").get<std::string>());
            const auto shape = meta.at("shape").get<std::vector<int64_t>>();
            const auto& offsets = meta.at("data_offsets");
            const uint64_t begin = offsets.at(0).get<uint64_t>();
            const uint64_t end = offsets.at(1).get<uint64_t>();

            if (end < begin || end > region) {
                throw std::runtime_error("safetensors: data_offsets out of range for " + name);
            }
            for (const int64_t d : shape) {
                if (d < 0) {
                    throw std::runtime_error("safetensors: negative dimension for " + name);
                }
            }

            const int64_t numel = num_elements(shape);
            const uint64_t elem_size = dtype_size(dtype);
            const uint64_t nbytes = end - begin;
            // Validate via division (never multiplies numel) so an overflowed/negative numel
            // cannot spuriously match, and validate before allocating the output buffer.
            if (nbytes % elem_size != 0 || nbytes / elem_size != static_cast<uint64_t>(numel)) {
                throw std::runtime_error("safetensors: byte count does not match shape for " +
                                         name);
            }

            const uint8_t* p = base + begin;
            std::vector<float> out(static_cast<std::size_t>(numel));
            if (dtype == DType::F32) {
                std::memcpy(out.data(), p, static_cast<std::size_t>(nbytes));
            } else {
                const bool half = (dtype == DType::F16);
                for (int64_t i = 0; i < numel; ++i) {
                    const uint16_t bits = little_endian_u16(p + 2 * i);
                    out[static_cast<std::size_t>(i)] = half ? f16_to_f32(bits) : bf16_to_f32(bits);
                }
            }

            spans.emplace_back(begin, end);
            const bool inserted = st.tensors_.emplace(name, Tensor(shape, std::move(out))).second;
            if (!inserted) {
                throw std::runtime_error("safetensors: duplicate tensor name '" + name + "'");
            }
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("safetensors: malformed header entry: ") + e.what());
    }

    // The format requires sorted, non-overlapping data regions.
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) {
            throw std::runtime_error("safetensors: overlapping tensor data regions");
        }
    }
    return st;
}

SafeTensors SafeTensors::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("safetensors: cannot open " + path);
    }
    const std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    return parse(buf.data(), buf.size());
}

bool SafeTensors::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const Tensor& SafeTensors::get(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("safetensors: missing tensor '" + name + "'");
    }
    return it->second;
}

} // namespace engine
