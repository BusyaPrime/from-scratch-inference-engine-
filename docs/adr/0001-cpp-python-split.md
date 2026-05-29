# ADR-0001: C++/CUDA core with a Python frontend

- Status: Accepted
- Date: 2026-05-30
- Deciders: Akmal Khujdarov

## Context

Inference latency is dominated by model compute and memory traffic, not by request orchestration. The reference engines this project benchmarks against split the same way: a large compiled core with a Python frontend. Tokenization, an OpenAI-compatible HTTP API, and the benchmark harness all have mature Python libraries and benefit from staying in Python, where they also match the baselines' measurement surface.

## Decision

The performance-critical core is C++20, with CUDA added in a later phase: the model forward pass, KV-cache and paged-attention memory management, the scheduler and continuous batching, sampling, and custom kernels. It builds as `libengine` with a stable public API under `include/engine/`.

The core is exposed to Python through a single pybind11 extension module, `engine_ext`. A thin Python package, `engine`, wraps it with the FastAPI server, tokenizer orchestration, and the benchmark harness.

The build uses CMake for the core and bindings, driven by scikit-build-core for the Python package, so that `pip install -e .` produces an importable `engine`.

## Consequences

- The hot path stays in C++/CUDA; the cold path stays in Python. Benchmark comparisons against Python-fronted baselines are apples-to-apples.
- CI builds and tests the CPU path with no GPU.
- Cost: a binding boundary to maintain and data to marshal across it (token ids, configs, sampling parameters), plus a two-language build.

## Alternatives considered

- Pure C++ with no Python. Loses the tokenizer and serving ecosystems and the measurement parity with the baselines.
- Pure Python plus Triton, as in nano-vllm. This is the design the project differentiates against; a compiled core is the point.
- A Rust core. Viable, but C++/CUDA is the native environment for cuBLAS and CUDA kernels and matches the reference implementations.
