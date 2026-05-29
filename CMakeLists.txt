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

add_library(engine_core STATIC
    src/core/version.cpp)
target_include_directories(engine_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(engine_core PUBLIC cxx_std_20)

if(ENGINE_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    find_package(CUDAToolkit REQUIRED)
    # Kernel sources and cuBLAS linkage land in Phase 4.
endif()

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
