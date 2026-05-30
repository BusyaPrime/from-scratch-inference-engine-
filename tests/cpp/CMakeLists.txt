include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# Match the parent project's CRT on Windows so linking succeeds.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# GoogleTest compiles its own sources with warnings-as-errors, which trips on
# newer compilers (a char8_t/char32_t conversion in gtest-printers.h under
# Clang 20+). Drop warnings-as-errors for the vendored targets only; our own
# code keeps strict warnings.
if(MSVC)
    set(_engine_gtest_no_werror /WX-)
else()
    set(_engine_gtest_no_werror -Wno-error)
endif()
foreach(_engine_gtest_target gtest gtest_main gmock gmock_main)
    if(TARGET ${_engine_gtest_target})
        target_compile_options(${_engine_gtest_target} PRIVATE ${_engine_gtest_no_werror})
    endif()
endforeach()

add_executable(engine_tests
    test_version.cpp
    test_tensor.cpp
    test_safetensors.cpp
    test_nn.cpp
    test_sampler.cpp
    test_kv_cache.cpp
    test_block_allocator.cpp
    test_paged_attention.cpp
    test_block_manager.cpp
    test_continuous_batch.cpp
    test_engine.cpp
    test_quant.cpp)
if(ENGINE_ENABLE_CUDA)
    target_sources(engine_tests PRIVATE test_cuda_smoke.cpp test_cuda_kernels.cpp
                                         test_cuda_model.cpp test_cuda_engine.cpp
                                         test_cuda_block_manager.cpp)
endif()

target_link_libraries(engine_tests PRIVATE engine_core GTest::gtest_main)
target_compile_definitions(engine_tests PRIVATE ENGINE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(engine_tests)
