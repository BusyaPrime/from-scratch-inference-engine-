// SPDX-License-Identifier: Apache-2.0
#include "engine/ops.hpp"

#include <Eigen/Dense>
#include <stdexcept>

namespace engine {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul: both operands must be 2D");
    }
    const int64_t m = a.dim(0);
    const int64_t k = a.dim(1);
    const int64_t n = b.dim(1);
    if (k != b.dim(0)) {
        throw std::invalid_argument("matmul: inner dimensions disagree");
    }

    using RowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Map<const RowMajor> am(a.data(), m, k);
    const Eigen::Map<const RowMajor> bm(b.data(), k, n);

    Tensor out({m, n});
    Eigen::Map<RowMajor> om(out.data(), m, n);
    om.noalias() = am * bm;
    return out;
}

} // namespace engine
