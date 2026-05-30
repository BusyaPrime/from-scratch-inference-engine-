// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/tensor.hpp"

#include <cstdint>
#include <vector>

namespace engine::tp {

// Tensor-parallel sharding primitives. These split weights/activations and recombine partial
// results so a linear can be partitioned across N model shards:
//   - Column-parallel (output sharded): split the weight [out, in] along `out` with split_rows,
//     run each shard, then concat_columns the partial outputs (an all-gather across shards).
//   - Row-parallel (input sharded): split the weight [out, in] along `in` with split_columns and
//     the input along its columns, run each shard, then sum the partial outputs (an all-reduce).
// On a single device the shards run sequentially and the combine is local; across devices the same
// split/combine maps onto NCCL-style collectives. The arithmetic is identical either way.

// Split a [rows, cols] tensor into `shards` pieces along rows (rows must divide evenly).
[[nodiscard]] std::vector<Tensor> split_rows(const Tensor& m, int64_t shards);

// Split a [rows, cols] tensor into `shards` pieces along columns (cols must divide evenly).
[[nodiscard]] std::vector<Tensor> split_columns(const Tensor& m, int64_t shards);

// Concatenate [rows, *] tensors along columns (all parts must share the row count).
[[nodiscard]] Tensor concat_columns(const std::vector<Tensor>& parts);

// Elementwise sum of identically shaped tensors (the all-reduce).
[[nodiscard]] Tensor sum(const std::vector<Tensor>& parts);

} // namespace engine::tp
