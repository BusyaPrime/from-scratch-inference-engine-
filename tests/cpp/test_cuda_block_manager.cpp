// SPDX-License-Identifier: Apache-2.0
#include "engine/block_manager.hpp" // SequenceBlocks
#include "engine/cuda/block_manager.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

float* to_device(const std::vector<float>& host) {
    float* dev = nullptr;
    cudaMalloc(&dev, host.size() * sizeof(float));
    cudaMemcpy(dev, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return dev;
}

std::vector<float> to_host(const float* dev, std::size_t n) {
    std::vector<float> host(n);
    cudaMemcpy(host.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    return host;
}

} // namespace

// Two sequences scattering into one shared pool must gather back exactly their own K/V: the
// per-sequence block tables map them to disjoint physical blocks.
TEST(GpuBlockManager, TwoSequencesDrawFromOnePoolIndependently) {
    engine::cuda::GpuBlockManager m(/*num_layers=*/1,
                                    /*kv_dim=*/2,
                                    /*block_size=*/2,
                                    /*num_blocks=*/8);
    engine::SequenceBlocks a;
    engine::SequenceBlocks b;

    const std::vector<float> ak = {1, 1, 2, 2};
    const std::vector<float> bk = {10, 10, 20, 20, 30, 30};
    float* d_ak = to_device(ak);
    float* d_bk = to_device(bk);

    m.reserve(a, 2);
    m.write(a, 0, 0, d_ak, d_ak, 2);
    m.commit(a, 2);
    m.reserve(b, 3);
    m.write(b, 0, 0, d_bk, d_bk, 3);
    m.commit(b, 3);

    float* d_ga = nullptr;
    float* d_gav = nullptr;
    float* d_gb = nullptr;
    float* d_gbv = nullptr;
    cudaMalloc(&d_ga, 4 * sizeof(float));
    cudaMalloc(&d_gav, 4 * sizeof(float));
    cudaMalloc(&d_gb, 6 * sizeof(float));
    cudaMalloc(&d_gbv, 6 * sizeof(float));

    m.gather(a, 0, a.length, d_ga, d_gav);
    m.gather(b, 0, b.length, d_gb, d_gbv);

    const std::vector<float> ga = to_host(d_ga, 4);
    const std::vector<float> gb = to_host(d_gb, 6);
    for (std::size_t i = 0; i < ak.size(); ++i) {
        EXPECT_FLOAT_EQ(ga[i], ak[i]);
    }
    for (std::size_t i = 0; i < bk.size(); ++i) {
        EXPECT_FLOAT_EQ(gb[i], bk[i]);
    }

    cudaFree(d_ak);
    cudaFree(d_bk);
    cudaFree(d_ga);
    cudaFree(d_gav);
    cudaFree(d_gb);
    cudaFree(d_gbv);
}

TEST(GpuBlockManager, FreeReturnsBlocksToPool) {
    engine::cuda::GpuBlockManager m(1, 2, 2, 4);
    EXPECT_EQ(m.free_blocks(), 4);
    engine::SequenceBlocks s;
    m.reserve(s, 3); // ceil(3 / 2) = 2 blocks
    EXPECT_EQ(m.free_blocks(), 2);
    m.free(s);
    EXPECT_EQ(m.free_blocks(), 4);
    EXPECT_EQ(s.length, 0);
}
