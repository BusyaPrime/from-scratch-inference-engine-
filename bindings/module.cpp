// SPDX-License-Identifier: Apache-2.0
#include "engine/version.hpp"

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(engine_ext, m) {
    m.doc() = "Native core of the engine package.";
    m.def(
        "version",
        [] { return std::string(engine::version()); },
        "Version string reported by the engine core.");
}
