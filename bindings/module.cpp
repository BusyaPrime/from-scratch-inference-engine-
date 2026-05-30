// SPDX-License-Identifier: Apache-2.0
#include "engine/model.hpp"
#include "engine/ops.hpp"
#include "engine/tensor.hpp"
#include "engine/version.hpp"

#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// 2D fp32 matmul over NumPy arrays; used to check the engine path against NumPy.
py::array_t<float> matmul_np(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                             py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul expects two 2D arrays");
    }
    const auto m = static_cast<int64_t>(a.shape(0));
    const auto k = static_cast<int64_t>(a.shape(1));
    const auto n = static_cast<int64_t>(b.shape(1));
    if (k != static_cast<int64_t>(b.shape(0))) {
        throw std::invalid_argument("matmul: inner dimensions disagree (a.cols != b.rows)");
    }

    engine::Tensor at({m, k}, std::vector<float>(a.data(), a.data() + a.size()));
    engine::Tensor bt({k, n}, std::vector<float>(b.data(), b.data() + b.size()));
    const engine::Tensor ct = engine::matmul(at, bt);

    py::array_t<float> out({m, n});
    std::memcpy(out.mutable_data(), ct.data(), sizeof(float) * static_cast<std::size_t>(m * n));
    return out;
}

py::array_t<float> model_forward(const engine::Model& model, const std::vector<int64_t>& ids) {
    const engine::Tensor logits = model.forward(ids);
    const auto s = logits.dim(0);
    const auto v = logits.dim(1);
    py::array_t<float> out({s, v});
    std::memcpy(out.mutable_data(),
                logits.data(),
                sizeof(float) * static_cast<std::size_t>(s) * static_cast<std::size_t>(v));
    return out;
}

} // namespace

PYBIND11_MODULE(engine_ext, m) {
    m.doc() = "Native core of the engine package.";
    m.def(
        "version",
        [] { return std::string(engine::version()); },
        "Version string reported by the engine core.");
    m.def("matmul", &matmul_np, "2D float32 matrix multiply (a @ b).");

    py::class_<engine::Model>(m, "Model")
        .def_static("from_pretrained",
                    &engine::Model::from_pretrained,
                    py::arg("model_dir"),
                    "Load config.json + model.safetensors from a directory.")
        .def("forward",
             &model_forward,
             py::arg("ids"),
             "Forward a single sequence of token ids -> logits [seq_len, vocab_size].");
}
