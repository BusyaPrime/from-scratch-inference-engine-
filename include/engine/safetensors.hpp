// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// Loads tensors from the safetensors format into fp32 Tensors keyed by name.
// F16 and BF16 stored tensors are converted to fp32 on load.
class SafeTensors {
public:
    static SafeTensors load(const std::string& path);
    static SafeTensors parse(const uint8_t* data, std::size_t size);

    [[nodiscard]] bool contains(const std::string& name) const;
    [[nodiscard]] const Tensor& get(const std::string& name) const; // throws if absent
    [[nodiscard]] std::size_t size() const noexcept { return tensors_.size(); }

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

} // namespace engine
