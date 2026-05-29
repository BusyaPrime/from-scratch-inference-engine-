// SPDX-License-Identifier: Apache-2.0
#include "engine/safetensors.hpp"

#include "engine/dtype.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace engine {
namespace {

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) {
        n *= d;
    }
    return n;
}

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
    if (8 + header_len > size) {
        throw std::runtime_error("safetensors: header length exceeds the buffer");
    }

    const auto header = nlohmann::json::parse(
        std::string(reinterpret_cast<const char*>(data + 8), static_cast<std::size_t>(header_len)));
    const uint8_t* base = data + 8 + header_len;
    const uint64_t region = static_cast<uint64_t>(size) - 8 - header_len;

    SafeTensors st;
    for (const auto& item : header.items()) {
        const std::string& name = item.key();
        if (name == "__metadata__") {
            continue;
        }
        const auto& meta = item.value();
        const std::string dtype = meta.at("dtype").get<std::string>();
        const auto shape = meta.at("shape").get<std::vector<int64_t>>();
        const auto& offsets = meta.at("data_offsets");
        const uint64_t begin = offsets.at(0).get<uint64_t>();
        const uint64_t end = offsets.at(1).get<uint64_t>();
        if (end < begin || end > region) {
            throw std::runtime_error("safetensors: data_offsets out of range for " + name);
        }

        const int64_t numel = numel_of(shape);
        const uint64_t nbytes = end - begin;
        const uint8_t* p = base + begin;
        std::vector<float> out(static_cast<std::size_t>(numel));

        if (dtype == "F32") {
            if (nbytes != static_cast<uint64_t>(numel) * 4) {
                throw std::runtime_error("safetensors: F32 byte count mismatch for " + name);
            }
            std::memcpy(out.data(), p, static_cast<std::size_t>(nbytes));
        } else if (dtype == "F16" || dtype == "BF16") {
            if (nbytes != static_cast<uint64_t>(numel) * 2) {
                throw std::runtime_error("safetensors: 16-bit byte count mismatch for " + name);
            }
            const bool half = (dtype == "F16");
            for (int64_t i = 0; i < numel; ++i) {
                const uint16_t bits = little_endian_u16(p + 2 * i);
                out[static_cast<std::size_t>(i)] = half ? f16_to_f32(bits) : bf16_to_f32(bits);
            }
        } else {
            throw std::runtime_error("safetensors: unsupported dtype '" + dtype + "' for " + name);
        }

        st.tensors_.emplace(name, Tensor(shape, std::move(out)));
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

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        out.push_back(kv.first);
    }
    return out;
}

} // namespace engine
