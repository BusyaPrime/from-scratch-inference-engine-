# ADR-0004: CUDA kernel architecture and parity strategy

- Status: Accepted
- Date: 2026-05-30
- Deciders: Akmal Khujdarov

## Context

Phase 4 adds a GPU execution path. The CPU reference path is already correct: its forward
matches Hugging Face `transformers` in fp32 (`max|Δlogit| < 1e-3`, identical argmax, greedy
token-identical to 64 tokens), and cached, paged, and batched decode each reproduce the plain
forward exactly. The GPU path must inherit that correctness rather than re-establish it from
scratch, so every device kernel needs a cheap, deterministic way to be checked against the CPU
op it replaces.

The target GPU is a single 8 GB card. Qwen2.5-0.5B in fp32 is ~2 GB of weights plus the paged
KV pool, which fits, so the first GPU path stays fp32 to preserve the existing parity anchor;
half precision is a later, separately budgeted optimisation.

## Decision

Operation split (consistent with ADR-0003):

- GEMM uses cuBLAS (`cublasSgemm`). Row-major activations and weights are handled by computing
  the transposed product in cuBLAS's column-major convention; no hand-rolled GEMM.
- Hand-written kernels (the differentiator): RMSNorm, SwiGLU (`silu_mul`), RoPE, causal
  grouped-query attention over the paged/contiguous KV layout, embedding gather, the residual
  add, and sampling (argmax plus temperature/top-k/top-p).

Code layout:

- `include/engine/cuda/` exposes host-callable launchers with plain pointer/size signatures
  (e.g. `silu_mul(const float* gate, const float* up, float* out, int64_t n)`), so callers and
  tests never write CUDA syntax.
- `src/cuda/*.cu` holds the kernels and launchers; a small device-buffer RAII type owns
  allocations and host/device copies.
- All CUDA sources compile only under `ENGINE_ENABLE_CUDA`; the CPU build and its sources are
  untouched, and `nvcc` uses the MSVC host compiler (the `cuda-release` preset).

Parity strategy:

- Per kernel: a GoogleTest feeds identical random fp32 inputs to the CPU op (the existing
  `nn.cpp` function) and the GPU launcher and asserts agreement within tolerance. Tolerance
  accounts for GPU reductions running in a different summation order than the CPU's sequential
  fp64 accumulation (RMSNorm sum-of-squares, softmax, attention dot products).
- End to end: a GPU forward is compared against the CPU forward on the same weights; because the
  CPU forward already matches `transformers`, this transitively anchors the GPU path. Argmax
  ties follow the Amendment A near-tie rule.

CI and build:

- GitHub's hosted runners have no GPU, so the default CI gate stays CPU-only. CUDA configure,
  build, and the GPU parity tests run locally and in a separate, non-blocking job, mirroring how
  the `transformers` parity job is partitioned.

## Consequences

- The GPU path is validated incrementally: a kernel is only trusted once it matches its CPU
  twin, and the whole forward is only trusted once it matches the CPU forward.
- Cost: a Windows `nvcc` + MSVC + Ninja toolchain, and the discipline of maintaining two
  implementations of each op. The fp32-first choice trades peak throughput for a clean anchor.
- The 8 GB budget bounds batch size and model size for now; the block manager's pool already
  makes that budget explicit.

## Alternatives considered

- A Triton or other DSL backend. Rejected: the custom CUDA kernels are part of the project's
  signal (ADR-0003).
- Half precision from the start. Rejected for the first path: it would blur the fp32 parity
  anchor that makes kernel bugs obvious. It returns as an explicit, separately measured step.
- One fused mega-kernel. Rejected: per-op kernels are individually testable against the CPU
  twins; fusion is an optimisation to pursue once each op is proven.
