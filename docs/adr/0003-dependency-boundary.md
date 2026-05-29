# ADR-0003: Dependency boundary and "from scratch" scope

- Status: Accepted
- Date: 2026-05-30
- Deciders: Akmal Khujdarov

## Context

"From scratch" has to be defined precisely, or it becomes either dogmatic (reimplementing GEMM and tokenizers adds no signal) or hollow (importing a serving engine defeats the purpose). The value of this project is the engine machinery, not re-deriving numerical or IO primitives.

## Decision

Allowed dependencies, which are not the differentiator:

- GEMM: cuBLAS on GPU; Eigen or a CPU BLAS on the CPU reference path.
- Tokenization: the Hugging Face `tokenizers` library (token ids cross the binding boundary).
- Weight IO: `safetensors` parsing.
- Build, test, and serving libraries: CMake, pybind11, GoogleTest, Google Benchmark, pytest, FastAPI, uvicorn.

Hand-written, which is the differentiator, with no high-level inference library:

- KV cache and paged-attention memory management.
- Attention computation over paged KV.
- The scheduler and continuous batching.
- Sampling, preemption and eviction, prefix caching, and speculative decoding.
- All custom CUDA kernels.

Hard prohibition: the engine must never import a serving or inference engine at runtime, namely `vllm`, `sglang`, `tensorrt_llm`, `transformers` generation, or `text-generation-inference`. `transformers` and `vllm`/`nano-vllm` appear only under `tests/` as a correctness oracle and under `benchmarks/` as comparison baselines, never as a runtime dependency of the engine.

## Consequences

- The original work is concentrated where the signal is, and the boundary is explicit enough to enforce.
- A dependency check can assert that the `engine` package imports none of the prohibited libraries at runtime.
- Cost: reimplementing scheduling, memory management, and kernels that an off-the-shelf engine would otherwise provide.

## Alternatives considered

- Hand-write GEMM and a tokenizer too. Adds effort without changing the project's signal, and a hand-rolled GEMM would lose to cuBLAS regardless.
- Allow a high-level engine for the non-core pieces. Blurs the line the project exists to draw.
