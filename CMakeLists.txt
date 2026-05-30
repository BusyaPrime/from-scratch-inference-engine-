cmake_minimum_required(VERSION 3.24)

project(engine
    VERSION 0.0.1
    DESCRIPTION "From-scratch LLM inference engine"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENGINE_ENABLE_CUDA "Build CUDA kernels and link cuBLAS" OFF)
option(ENGINE_BUILD_TESTS "Build the C++ test suite" ON)

include(FetchContent)
# Eigen supplies the CPU GEMM. Fetch headers only: pointing SOURCE_SUBDIR at a
# directory Eigen does not contain skips running Eigen's own CMake project.
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _engine_skip_eigen_cmake)
FetchContent_MakeAvailable(eigen)

# nlohmann/json parses safetensors headers and config.json. Headers only, via the
# same SOURCE_SUBDIR trick (its CMake declares an older cmake_minimum_required).
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _engine_skip_json_cmake)
FetchContent_MakeAvailable(nlohmann_json)

add_library(engine_core STATIC
    src/core/version.cpp
    src/core/matmul.cpp
    src/model/model_config.cpp
    src/model/safetensors.cpp
    src/model/nn.cpp
    src/model/model.cpp
    src/kvcache/paged_kv_cache.cpp
    src/kvcache/block_manager.cpp
    src/sampler/sampler.cpp
    src/engine/engine.cpp)
target_include_directories(engine_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(engine_core SYSTEM PRIVATE
    ${eigen_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}/single_include)
target_compile_features(engine_core PUBLIC cxx_std_20)

if(ENGINE_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89) # Ada Lovelace (RTX 40-series)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(engine_core PRIVATE src/cuda/device.cu)
    set_target_properties(engine_core PROPERTIES CUDA_STANDARD 20 CUDA_ARCHITECTURES 89)
    target_compile_definitions(engine_core PUBLIC ENGINE_WITH_CUDA=1)
    target_link_libraries(engine_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Python binding module. Built whenever pybind11 is available, which is always
# true under a scikit-build-core / pip build; a bare cmake configure skips it.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(engine_ext bindings/module.cpp)
    target_link_libraries(engine_ext PRIVATE engine_core)
    install(TARGETS engine_ext LIBRARY DESTINATION engine)
endif()

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
