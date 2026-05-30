// SPDX-License-Identifier: Apache-2.0
#include "engine/tensor_parallel.hpp"

#include <cstring>
#include <stdexcept>

namespace engine::tp {

std::vector<Tensor> split_rows(const Tensor& m, int64_t shards) {
    if (m.ndim() != 2) {
        throw std::invalid_argument("split_rows: expects a 2D tensor");
    }
    const int64_t rows = m.dim(0);
    const int64_t cols = m.dim(1);
    if (shards <= 0 || rows % shards != 0) {
        throw std::invalid_argument("split_rows: rows must be divisible by shards");
    }
    const int64_t per = rows / shards;
    std::vector<Tensor> parts;
    parts.reserve(static_cast<std::size_t>(shards));
    for (int64_t s = 0; s < shards; ++s) {
        Tensor part({per, cols});
        std::memcpy(part.data(),
                    m.data() + s * per * cols,
                    sizeof(float) * static_cast<std::size_t>(per * cols));
        parts.push_back(std::move(part));
    }
    return parts;
}

std::vector<Tensor> split_columns(const Tensor& m, int64_t shards) {
    if (m.ndim() != 2) {
        throw std::invalid_argument("split_columns: expects a 2D tensor");
    }
    const int64_t rows = m.dim(0);
    const int64_t cols = m.dim(1);
    if (shards <= 0 || cols % shards != 0) {
        throw std::invalid_argument("split_columns: cols must be divisible by shards");
    }
    const int64_t per = cols / shards;
    std::vector<Tensor> parts;
    parts.reserve(static_cast<std::size_t>(shards));
    for (int64_t s = 0; s < shards; ++s) {
        Tensor part({rows, per});
        for (int64_t r = 0; r < rows; ++r) {
            std::memcpy(part.data() + r * per,
                        m.data() + r * cols + s * per,
                        sizeof(float) * static_cast<std::size_t>(per));
        }
        parts.push_back(std::move(part));
    }
    return parts;
}

Tensor concat_columns(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("concat_columns: no parts");
    }
    const int64_t rows = parts.front().dim(0);
    int64_t total_cols = 0;
    for (const Tensor& p : parts) {
        if (p.ndim() != 2 || p.dim(0) != rows) {
            throw std::invalid_argument("concat_columns: parts must share the row count");
        }
        total_cols += p.dim(1);
    }
    Tensor out({rows, total_cols});
    for (int64_t r = 0; r < rows; ++r) {
        int64_t col = 0;
        for (const Tensor& p : parts) {
            const int64_t pc = p.dim(1);
            std::memcpy(out.data() + r * total_cols + col,
                        p.data() + r * pc,
                        sizeof(float) * static_cast<std::size_t>(pc));
            col += pc;
        }
    }
    return out;
}

Tensor sum(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("sum: no parts");
    }
    const int64_t n = parts.front().numel();
    Tensor out(parts.front().shape());
    std::memcpy(out.data(), parts.front().data(), sizeof(float) * static_cast<std::size_t>(n));
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i].numel() != n) {
            throw std::invalid_argument("sum: parts must share shape");
        }
        const float* src = parts[i].data();
        float* dst = out.data();
        for (int64_t j = 0; j < n; ++j) {
            dst[j] += src[j];
        }
    }
    return out;
}

} // namespace engine::tp
