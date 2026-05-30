// SPDX-License-Identifier: Apache-2.0
#include "engine/engine.hpp"
#include "engine/model.hpp"
#include "engine/ops.hpp"
#include "engine/sampler.hpp"
#include "engine/tensor.hpp"
#include "engine/version.hpp"

#include <cstdint>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <utility>
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

    py::class_<engine::SamplingParams>(m, "SamplingParams")
        .def(py::init([](double temperature, int64_t top_k, double top_p) {
                 engine::SamplingParams params;
                 params.temperature = temperature;
                 params.top_k = top_k;
                 params.top_p = top_p;
                 return params;
             }),
             py::arg("temperature") = 1.0,
             py::arg("top_k") = 0,
             py::arg("top_p") = 1.0,
             "Sampling controls; temperature <= 0 selects greedy (argmax).")
        .def_readwrite("temperature", &engine::SamplingParams::temperature)
        .def_readwrite("top_k", &engine::SamplingParams::top_k)
        .def_readwrite("top_p", &engine::SamplingParams::top_p);

    py::enum_<engine::SeqStatus>(m, "SeqStatus")
        .value("Waiting", engine::SeqStatus::Waiting)
        .value("Running", engine::SeqStatus::Running)
        .value("Finished", engine::SeqStatus::Finished);

    // keep_alive<1, 2>: the referenced Model must outlive the Engine that borrows it.
    py::class_<engine::Engine>(m, "Engine")
        .def(py::init<const engine::Model&, int64_t, int64_t, uint64_t, int64_t, bool>(),
             py::arg("model"),
             py::arg("block_size"),
             py::arg("num_blocks"),
             py::arg("seed") = 0,
             py::arg("max_batch") = 256,
             py::arg("enable_prefix_cache") = true,
             py::keep_alive<1, 2>(),
             "Continuous-batching engine over a shared paged KV pool. Prefix caching is on "
             "by default: requests sharing a prompt prefix reuse cached KV blocks.")
        .def(
            "add_request",
            [](engine::Engine& self,
               std::vector<int64_t> prompt,
               const engine::SamplingParams& params,
               int64_t max_tokens,
               int64_t eos_id) {
                engine::Request request;
                request.prompt = std::move(prompt);
                request.params = params;
                request.max_tokens = max_tokens;
                request.eos_id = eos_id;
                return self.add_request(std::move(request));
            },
            py::arg("prompt"),
            py::arg("params") = engine::SamplingParams{},
            py::arg("max_tokens") = 16,
            py::arg("eos_id") = -1,
            "Queue a request (prompt token ids); returns its sequence id.")
        .def("step", &engine::Engine::step, "Run one batched scheduling step.")
        .def("has_work", &engine::Engine::has_work)
        .def("output", &engine::Engine::output, py::arg("id"))
        .def("status", &engine::Engine::status, py::arg("id"))
        .def("preemptions", &engine::Engine::preemptions)
        .def("prefix_hits",
             &engine::Engine::prefix_hits,
             "Number of sequences that reused at least one cached prompt-prefix block.")
        .def("generate",
             &engine::Engine::generate,
             py::arg("prompt"),
             py::arg("params"),
             py::arg("max_tokens"),
             py::arg("eos_id") = -1,
             "Drive the engine until a single fresh request finishes; returns its token ids.");
}
