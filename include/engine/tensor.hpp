// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine {

// Row-major, fp32-backed dense tensor. The reference path computes in fp32;
// lower-precision weights are converted to fp32 on load (see dtype.hpp).
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shape)
        : shape_(std::move(shape)), data_(static_cast<std::size_t>(numel_of(shape_)), 0.0f) {}

    Tensor(std::vector<int64_t> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (static_cast<std::size_t>(numel_of(shape_)) != data_.size()) {
            throw std::invalid_argument("Tensor: shape does not match data size");
        }
    }

    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] int64_t dim(std::size_t i) const { return shape_.at(i); }
    [[nodiscard]] int64_t numel() const noexcept { return numel_of(shape_); }

    [[nodiscard]] float* data() noexcept { return data_.data(); }
    [[nodiscard]] const float* data() const noexcept { return data_.data(); }

private:
    static int64_t numel_of(const std::vector<int64_t>& shape) noexcept {
        int64_t n = 1;
        for (const int64_t d : shape) {
            n *= d;
        }
        return n;
    }

    std::vector<int64_t> shape_;
    std::vector<float> data_;
};

} // namespace engine
