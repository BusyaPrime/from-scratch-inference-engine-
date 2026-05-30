// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine {

// Contiguous per-layer key/value cache for incremental (cached) decoding.
class KVCache {
public:
    KVCache(int64_t num_layers, int64_t kv_dim)
        : keys_(static_cast<std::size_t>(num_layers)),
          values_(static_cast<std::size_t>(num_layers)), kv_dim_(kv_dim) {}

    [[nodiscard]] int64_t length() const noexcept { return length_; }
    [[nodiscard]] int64_t kv_dim() const noexcept { return kv_dim_; }

    // Append n_rows rows (each kv_dim long) of K and V for one layer.
    void append(int64_t layer, const float* k, const float* v, int64_t n_rows) {
        const auto n = static_cast<std::size_t>(n_rows * kv_dim_);
        auto& kl = keys_[static_cast<std::size_t>(layer)];
        auto& vl = values_[static_cast<std::size_t>(layer)];
        kl.insert(kl.end(), k, k + n);
        vl.insert(vl.end(), v, v + n);
    }

    [[nodiscard]] int64_t rows(int64_t layer) const {
        return static_cast<int64_t>(keys_[static_cast<std::size_t>(layer)].size()) / kv_dim_;
    }
    [[nodiscard]] Tensor key_tensor(int64_t layer) const {
        return Tensor({rows(layer), kv_dim_}, keys_[static_cast<std::size_t>(layer)]);
    }
    [[nodiscard]] Tensor value_tensor(int64_t layer) const {
        return Tensor({rows(layer), kv_dim_}, values_[static_cast<std::size_t>(layer)]);
    }

    // Advance the logical sequence length after all layers have been appended.
    void advance(int64_t n) noexcept { length_ += n; }

private:
    std::vector<std::vector<float>> keys_;
    std::vector<std::vector<float>> values_;
    int64_t kv_dim_;
    int64_t length_ = 0;
};

} // namespace engine
