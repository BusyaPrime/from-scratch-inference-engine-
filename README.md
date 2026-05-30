# engine

A from-scratch LLM inference engine. The performance-critical core (model execution, KV-cache and paged-attention memory management, request scheduling, sampling, and custom kernels) is C++20 and CUDA, exposed to Python through a pybind11 module. A thin Python layer provides an OpenAI-compatible HTTP server and the benchmark harness.

## Status

The CPU reference path runs end to end: safetensors loading, the full Qwen2.5 forward pass, a
contiguous KV cache, paged attention over a shared block pool, a continuous-batching scheduler
with preemption and recompute, and sampling. It is exposed to Python and fronted by an
OpenAI-compatible HTTP server with streaming, plus a latency/throughput benchmark harness.

Correctness is anchored against Hugging Face `transformers` in fp32: per-position logits match
within `max|Δ| < 1e-3` with identical argmax, and greedy decoding is token-identical out to 64
tokens. The C++ suite additionally proves that cached decode, paged attention, and batched
decode each reproduce the plain forward exactly, and that a preempted-and-recomputed sequence
matches its uninterrupted output.

The CUDA path is now in: hand-written kernels for RMSNorm, SwiGLU, RoPE, causal grouped-query
attention, embedding gather, and the residual add, plus cuBLAS for GEMM. On top of them sit a
device-resident forward, a device KV cache with cached decode, greedy generation, a batched
forward (continuous-batching compute across many sequences), a GPU continuous-batching scheduler
(`CudaEngine`), and a shared device block pool (`GpuBlockManager`) with a paged batched forward —
true GPU PagedAttention. Every kernel is checked against its CPU twin, and the full GPU forward,
cached decode, batched forward, and paged forward all match the CPU path (and so, transitively,
`transformers`). A comparison harness benchmarks the engine against the `transformers` reference;
the `nano-vllm`/`vllm` head-to-head uses the same recipe on a Linux + CUDA host where they install
cleanly. Still ahead: half precision and the Phase 5 stretch (speculative decoding, prefix caching,
quantization).

## Benchmarks

Qwen2.5-0.5B-Instruct, fp32, greedy decoding, on an RTX 4060 Laptop (8 GB). CPU figures are from
the Python harness (single-threaded GEMM); GPU figures from the C++ `bench_cuda` harness (single
stream, 16-token prompt, 64 generated). Reproduce with the commands under
[Reproduce the benchmarks](#reproduce-the-benchmarks).

| Path                        | Throughput | TTFT    | TPOT    |
| --------------------------- | ---------- | ------- | ------- |
| CPU, single stream          | ~4.8 tok/s | ~0.8 s  | ~190 ms |
| CPU, 8 concurrent (batched) | ~8.0 tok/s | —       | —       |
| GPU, single stream          | ~43 tok/s  | ~31 ms  | ~23 ms  |

The GPU path runs roughly 9x the CPU decode rate even before optimisation: the attention kernel is
a straightforward one-thread-per-query design and precision is fp32, both chosen for a clean parity
anchor rather than peak speed. On CPU, continuous batching lifts aggregate throughput ~1.7x by
sharing each forward's weight reads across the batch. Still ahead: GPU continuous batching and the
head-to-head against `nano-vllm` and `vllm`. See [benchmarks/](benchmarks/) for the harnesses.

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

Weights are not committed, so fetch them first, then start the OpenAI-compatible server:

```
pip install -e ".[serve,dev]"          # serve: fastapi/uvicorn/jinja2; dev: huggingface_hub
python scripts/fetch_model.py          # downloads Qwen2.5-0.5B-Instruct into weights/
ENGINE_MODEL_DIR=weights/Qwen2.5-0.5B-Instruct \
  uvicorn --factory engine.server:create_app --port 8000
```

Call it like any OpenAI endpoint:

```
curl http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt": "Paged attention is", "max_tokens": 64}'
```

`/v1/chat/completions` (rendered with the model's own chat template) and `"stream": true`
Server-Sent Events are supported, as is `/v1/models`.

## Reproduce the benchmarks

```
pip install -e ".[dev]"
python scripts/fetch_model.py
python benchmarks/harness/bench_engine.py \
  --model-dir weights/Qwen2.5-0.5B-Instruct --num-requests 8 --max-tokens 32
```

Pass `--json` for machine-readable output. The harness reports TTFT, TPOT, throughput, and
requests per second over a batch submitted up front.

For the GPU path (requires the CUDA toolkit and the `cuda-release` build):

```
cmake --preset cuda-release
cmake --build --preset cuda-release --target bench_cuda
build/cuda-release/bench_cuda weights/Qwen2.5-0.5B-Instruct 16 64
```

Head-to-head against the `transformers` reference (same model, precision, greedy):

```
pip install -e ".[dev]" && pip install torch transformers
python benchmarks/harness/compare_baselines.py --model-dir weights/Qwen2.5-0.5B-Instruct
```

## Scope and limitations

- Inference only. No training, fine-tuning, or distributed training.
- Single GPU. Tensor parallelism is a possible future extension.
- Small target models: Qwen2.5-0.5B-Instruct (primary), Llama-3.2-1B-Instruct.
- Tokenization uses the Hugging Face `tokenizers` library; GEMM uses cuBLAS (GPU) and Eigen/BLAS (CPU). The engine machinery (paged attention, scheduling, continuous batching, sampling, kernels) is hand-written. See [docs/adr/0003-dependency-boundary.md](docs/adr/0003-dependency-boundary.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
