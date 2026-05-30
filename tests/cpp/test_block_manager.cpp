// SPDX-License-Identifier: Apache-2.0
#include "engine/block_manager.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// Two sequences drawing from the same pool must never see each other's K/V: the per-sequence
// block tables map them to disjoint physical blocks.
TEST(BlockManager, TwoSequencesDrawFromOnePoolIndependently) {
    engine::BlockManager m(/*num_layers=*/1, /*kv_dim=*/2, /*block_size=*/2, /*num_blocks=*/8);
    engine::SequenceBlocks a;
    engine::SequenceBlocks b;

    m.reserve(a, 2);
    const std::vector<float> ak = {1, 1, 2, 2};
    m.write(a, 0, 0, ak.data(), ak.data(), 2);
    m.commit(a, 2);

    m.reserve(b, 3);
    const std::vector<float> bk = {10, 10, 20, 20, 30, 30};
    m.write(b, 0, 0, bk.data(), bk.data(), 3);
    m.commit(b, 3);

    const engine::Tensor ga = m.gather_keys(a, 0, a.length);
    const engine::Tensor gb = m.gather_keys(b, 0, b.length);
    ASSERT_EQ(ga.dim(0), 2);
    ASSERT_EQ(gb.dim(0), 3);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(ga.data()[i], ak[static_cast<std::size_t>(i)]);
    }
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(gb.data()[i], bk[static_cast<std::size_t>(i)]);
    }
}

TEST(BlockManager, FreeReturnsBlocksToPool) {
    engine::BlockManager m(1, 2, 2, 4);
    EXPECT_EQ(m.free_blocks(), 4);
    engine::SequenceBlocks s;
    m.reserve(s, 3); // ceil(3 / 2) = 2 blocks
    EXPECT_EQ(s.block_table.size(), 2u);
    EXPECT_EQ(m.free_blocks(), 2);
    m.free(s);
    EXPECT_EQ(m.free_blocks(), 4);
    EXPECT_EQ(s.block_table.size(), 0u);
    EXPECT_EQ(s.length, 0);
}

TEST(BlockManager, CanAppendRespectsBudget) {
    engine::BlockManager m(1, 1, /*block_size=*/4, /*num_blocks=*/2); // 8 slots total
    engine::SequenceBlocks s;
    EXPECT_TRUE(m.can_append(s, 8));  // needs 2 blocks, 2 free
    EXPECT_FALSE(m.can_append(s, 9)); // needs 3 blocks, only 2

    m.reserve(s, 4);
    m.commit(s, 4);                   // 1 block used, 1 free
    EXPECT_TRUE(m.can_append(s, 4));  // total 2 blocks, +1 over the held 1 -> fits in 1 free
    EXPECT_FALSE(m.can_append(s, 5)); // total 3 blocks, +2 over held 1 -> only 1 free
}
