# engine

A from-scratch LLM inference engine. The performance-critical core (model execution, KV-cache and paged-attention memory management, request scheduling, sampling, and custom kernels) is C++20 and CUDA, exposed to Python through a pybind11 module. A thin Python layer provides an OpenAI-compatible HTTP server and the benchmark harness.

## Status

Early scaffolding. The CPU reference path is under construction; CUDA kernels and the benchmark numbers below land in later phases. This section is replaced by headline results once the engine runs end to end.

## Benchmarks

Pending. Once the CPU and CUDA paths are complete, this section leads with TTFT, TPOT, throughput, and latency percentiles measured against `nano-vllm` and `vllm` on identical hardware, model, and precision, with a full reproducibility manifest. See [benchmarks/](benchmarks/) for the harness and methodology.

## Architecture

```
            HTTP client (OpenAI-compatible)
                        |
        Python frontend (the engine package)
        FastAPI routes, SSE, tokenization
                        |
            pybind11 boundary (engine_ext)
                        |
            C++20 / CUDA core (libengine)
  +-----------+-----------+-----------+----------+
  scheduler   kvcache     attention   sampler   model
 (continuous (paged       (paged KV)            (forward)
  batching)   blocks)
                        |
        device backend (CPU reference / CUDA)
                        |
        GEMM: Eigen/BLAS (CPU) . cuBLAS (GPU)
```

The split keeps the hot path in C++/CUDA while the cold path (HTTP, tokenizer orchestration, benchmarking) stays in Python. Every CUDA kernel has a CPU reference twin, so correctness is checked without a GPU and CI runs the full CPU path.

## Build

Requirements: CMake >= 3.24, Ninja, a C++20 compiler, Python >= 3.11. CUDA builds additionally require the CUDA Toolkit (>= 12.4).

CPU (default):

```
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

Python package (editable install):

```
pip install -e .
```

CUDA (optional, requires an NVIDIA GPU and the toolkit):

```
cmake --preset cuda-release
cmake --build --preset cuda-release
```

## Run the server

Pending the serving layer (Phase 3). The intended entry point starts an OpenAI-compatible server:

```
python -m engine.serving --model <hf-model-id> --port 8000
```

## Reproduce the benchmarks

Pending the benchmark harness (Phase 3). Each run emits a reproducibility manifest (hardware, driver, toolkit, model revision, precision, concurrency, sampling) alongside the metrics.

## Scope and limitations

- Inference only. No training, fine-tuning, or distributed training.
- Single GPU. Tensor parallelism is a possible future extension.
- Small target models: Qwen2.5-0.5B-Instruct (primary), Llama-3.2-1B-Instruct.
- Tokenization uses the Hugging Face `tokenizers` library; GEMM uses cuBLAS (GPU) and Eigen/BLAS (CPU). The engine machinery (paged attention, scheduling, continuous batching, sampling, kernels) is hand-written. See [docs/adr/0003-dependency-boundary.md](docs/adr/0003-dependency-boundary.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
