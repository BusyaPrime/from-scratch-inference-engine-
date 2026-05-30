// SPDX-License-Identifier: Apache-2.0
#include "engine/block_allocator.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(BlockAllocator, AllocatesDistinctBlocks) {
    engine::BlockAllocator a(3);
    EXPECT_EQ(a.num_blocks(), 3);
    EXPECT_EQ(a.num_free(), 3);
    const int64_t b0 = a.allocate();
    const int64_t b1 = a.allocate();
    const int64_t b2 = a.allocate();
    EXPECT_NE(b0, b1);
    EXPECT_NE(b1, b2);
    EXPECT_NE(b0, b2);
    EXPECT_EQ(a.num_free(), 0);
}

TEST(BlockAllocator, ThrowsWhenExhausted) {
    engine::BlockAllocator a(1);
    a.allocate();
    EXPECT_THROW(a.allocate(), std::runtime_error);
}

TEST(BlockAllocator, FreeReturnsBlockToPool) {
    engine::BlockAllocator a(2);
    const int64_t b0 = a.allocate();
    EXPECT_EQ(a.num_free(), 1);
    a.free(b0);
    EXPECT_EQ(a.num_free(), 2);
    EXPECT_EQ(a.allocate(), b0); // freed block is reused
}

TEST(BlockAllocator, RefCountDelaysFree) {
    engine::BlockAllocator a(2);
    const int64_t b = a.allocate();
    EXPECT_EQ(a.ref_count(b), 1);
    a.incref(b);
    EXPECT_EQ(a.ref_count(b), 2);
    a.free(b);
    EXPECT_EQ(a.ref_count(b), 1);
    EXPECT_EQ(a.num_free(), 1); // still held by the second reference
    a.free(b);
    EXPECT_EQ(a.ref_count(b), 0);
    EXPECT_EQ(a.num_free(), 2); // released now that both references are gone
}

TEST(BlockAllocator, DoubleFreeThrows) {
    engine::BlockAllocator a(1);
    const int64_t b = a.allocate();
    a.free(b);
    EXPECT_THROW(a.free(b), std::runtime_error);
}
