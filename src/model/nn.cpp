// SPDX-License-Identifier: Apache-2.0
#include "engine/nn.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace engine {
namespace {
using RowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
} // namespace

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    if (x.ndim() != 2 || weight.ndim() != 2) {
        throw std::invalid_argument("linear: x and weight must be 2D");
    }
    const int64_t s = x.dim(0);
    const int64_t in = x.dim(1);
    const int64_t out = weight.dim(0);
    if (weight.dim(1) != in) {
        throw std::invalid_argument("linear: weight inner dimension does not match x");
    }

    const Eigen::Map<const RowMajor> xm(x.data(), s, in);
    const Eigen::Map<const RowMajor> wm(weight.data(), out, in);
    Tensor y({s, out});
    Eigen::Map<RowMajor> ym(y.data(), s, out);
    ym.noalias() = xm * wm.transpose();

    if (bias.numel() > 0) {
        if (bias.numel() != out) {
            throw std::invalid_argument("linear: bias size does not match output");
        }
        ym.rowwise() += Eigen::Map<const Eigen::RowVectorXf>(bias.data(), out);
    }
    return y;
}

Tensor linear(const Tensor& x, const Tensor& weight) {
    return linear(x, weight, Tensor{});
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, double eps) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("rms_norm: x must be 2D");
    }
    const int64_t s = x.dim(0);
    const int64_t h = x.dim(1);
    if (weight.numel() != h) {
        throw std::invalid_argument("rms_norm: weight size does not match hidden dim");
    }

    Tensor out({s, h});
    const float* xp = x.data();
    const float* wp = weight.data();
    float* op = out.data();
    for (int64_t i = 0; i < s; ++i) {
        const float* row = xp + i * h;
        double sum_sq = 0.0;
        for (int64_t j = 0; j < h; ++j) {
            sum_sq += static_cast<double>(row[j]) * static_cast<double>(row[j]);
        }
        const double scale = 1.0 / std::sqrt(sum_sq / static_cast<double>(h) + eps);
        float* orow = op + i * h;
        for (int64_t j = 0; j < h; ++j) {
            orow[j] = static_cast<float>(static_cast<double>(row[j]) * scale) * wp[j];
        }
    }
    return out;
}

Tensor silu_mul(const Tensor& gate, const Tensor& up) {
    if (gate.shape() != up.shape()) {
        throw std::invalid_argument("silu_mul: gate and up shapes must match");
    }
    Tensor out(gate.shape());
    const int64_t n = gate.numel();
    const float* g = gate.data();
    const float* u = up.data();
    float* o = out.data();
    for (int64_t i = 0; i < n; ++i) {
        const float x = g[i];
        o[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
    return out;
}

Tensor embedding(const Tensor& weight, const std::vector<int64_t>& ids) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("embedding: weight must be 2D");
    }
    const int64_t vocab = weight.dim(0);
    const int64_t h = weight.dim(1);
    const auto s = static_cast<int64_t>(ids.size());
    Tensor out({s, h});
    for (int64_t i = 0; i < s; ++i) {
        const int64_t id = ids[static_cast<std::size_t>(i)];
        if (id < 0 || id >= vocab) {
            throw std::out_of_range("embedding: token id out of range");
        }
        std::memcpy(out.data() + i * h,
                    weight.data() + id * h,
                    sizeof(float) * static_cast<std::size_t>(h));
    }
    return out;
}

void rope_inplace(Tensor& x,
                  int64_t n_heads,
                  int64_t head_dim,
                  double theta,
                  const std::vector<int64_t>& positions) {
    if (x.ndim() != 2 || x.dim(1) != n_heads * head_dim) {
        throw std::invalid_argument("rope: x must be [S, n_heads*head_dim]");
    }
    const int64_t s = x.dim(0);
    if (static_cast<int64_t>(positions.size()) != s) {
        throw std::invalid_argument("rope: positions size must match sequence length");
    }
    const int64_t half = head_dim / 2;

    std::vector<double> inv_freq(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        inv_freq[static_cast<std::size_t>(i)] =
            std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    }

    float* xp = x.data();
    for (int64_t t = 0; t < s; ++t) {
        const auto pos = static_cast<double>(positions[static_cast<std::size_t>(t)]);
        float* row = xp + t * n_heads * head_dim;
        for (int64_t hh = 0; hh < n_heads; ++hh) {
            float* head = row + hh * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const double ang = pos * inv_freq[static_cast<std::size_t>(i)];
                const double c = std::cos(ang);
                const double sn = std::sin(ang);
                const float x1 = head[i];
                const float x2 = head[i + half];
                head[i] =
                    static_cast<float>(static_cast<double>(x1) * c - static_cast<double>(x2) * sn);
                head[i + half] =
                    static_cast<float>(static_cast<double>(x2) * c + static_cast<double>(x1) * sn);
            }
        }
    }
}

Tensor attention(const Tensor& q,
                 const Tensor& k,
                 const Tensor& v,
                 int64_t n_heads,
                 int64_t n_kv_heads,
                 int64_t head_dim) {
    const int64_t s = q.dim(0);
    if (q.dim(1) != n_heads * head_dim) {
        throw std::invalid_argument("attention: q must be [S, n_heads*head_dim]");
    }
    if (k.dim(0) != s || v.dim(0) != s || k.dim(1) != n_kv_heads * head_dim ||
        v.dim(1) != n_kv_heads * head_dim) {
        throw std::invalid_argument("attention: k/v shape mismatch");
    }
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
        throw std::invalid_argument("attention: n_heads must be a multiple of n_kv_heads");
    }

    const int64_t group = n_heads / n_kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    Tensor out({s, n_heads * head_dim});

    const float* qp = q.data();
    const float* kp = k.data();
    const float* vp = v.data();
    float* op = out.data();

    std::vector<double> scores(static_cast<std::size_t>(s));
    std::vector<double> acc(static_cast<std::size_t>(head_dim));

    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t kvh = h / group;
        for (int64_t i = 0; i < s; ++i) { // query position; causal -> keys 0..i
            const float* qvec = qp + i * n_heads * head_dim + h * head_dim;
            double max_score = -std::numeric_limits<double>::infinity();
            for (int64_t j = 0; j <= i; ++j) {
                const float* kvec = kp + j * n_kv_heads * head_dim + kvh * head_dim;
                double dot = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(qvec[d]) * static_cast<double>(kvec[d]);
                }
                dot *= scale;
                scores[static_cast<std::size_t>(j)] = dot;
                max_score = std::max(max_score, dot);
            }
            double sum = 0.0;
            for (int64_t j = 0; j <= i; ++j) {
                const double e = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
                scores[static_cast<std::size_t>(j)] = e;
                sum += e;
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                acc[static_cast<std::size_t>(d)] = 0.0;
            }
            const double inv_sum = 1.0 / sum;
            for (int64_t j = 0; j <= i; ++j) {
                const double p = scores[static_cast<std::size_t>(j)] * inv_sum;
                const float* vvec = vp + j * n_kv_heads * head_dim + kvh * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) {
                    acc[static_cast<std::size_t>(d)] += p * static_cast<double>(vvec[d]);
                }
            }
            float* ovec = op + i * n_heads * head_dim + h * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                ovec[d] = static_cast<float>(acc[static_cast<std::size_t>(d)]);
            }
        }
    }
    return out;
}

} // namespace engine
