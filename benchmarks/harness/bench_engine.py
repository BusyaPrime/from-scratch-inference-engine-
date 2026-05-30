# SPDX-License-Identifier: Apache-2.0
"""Latency/throughput benchmark for the continuous-batching engine.

Submits a batch of requests up front and lets the scheduler batch them, measuring:
  - TTFT  (time to first token, per request)
  - TPOT  (time per output token, per request, after the first token)
  - throughput (total output tokens / wall time) and requests/second

Run: python benchmarks/harness/bench_engine.py --model-dir weights/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter

from engine import engine_ext
from engine.llm import LLM


@dataclass
class BenchmarkResult:
    requests: int
    prompt_tokens: int
    max_tokens: int
    total_output_tokens: int
    wall_seconds: float
    throughput_tokens_per_s: float
    requests_per_s: float
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p90: float
    tpot_ms_mean: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def run_benchmark(
    llm: LLM, prompt_ids: list[list[int]], max_tokens: int, *, temperature: float = 0.0
) -> BenchmarkResult:
    """Submit every prompt, drive the engine to completion, and collect timing metrics."""
    engine = llm.engine
    params = engine_ext.SamplingParams(temperature, 0, 1.0)

    start = perf_counter()
    seq_ids = [engine.add_request(list(p), params, max_tokens, llm.eos_id) for p in prompt_ids]
    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    output_count: dict[int, int] = {}
    awaiting_first = set(seq_ids)
    pending = set(seq_ids)

    while engine.has_work():
        engine.step()
        now = perf_counter()
        for sid in list(awaiting_first):
            if len(engine.output(sid)) > 0:
                first_token_at[sid] = now
                awaiting_first.discard(sid)
        for sid in list(pending):
            if engine.status(sid) == engine_ext.SeqStatus.Finished:
                finished_at[sid] = now
                output_count[sid] = len(engine.output(sid))
                pending.discard(sid)
    end = perf_counter()

    ttfts = [(first_token_at[s] - start) * 1000.0 for s in seq_ids if s in first_token_at]
    tpots = [
        (finished_at[s] - first_token_at[s]) * 1000.0 / (output_count[s] - 1)
        for s in seq_ids
        if s in finished_at and output_count.get(s, 0) > 1
    ]
    total_output = sum(output_count.values())
    wall = end - start

    return BenchmarkResult(
        requests=len(seq_ids),
        prompt_tokens=len(prompt_ids[0]) if prompt_ids else 0,
        max_tokens=max_tokens,
        total_output_tokens=total_output,
        wall_seconds=wall,
        throughput_tokens_per_s=(total_output / wall) if wall > 0 else 0.0,
        requests_per_s=(len(seq_ids) / wall) if wall > 0 else 0.0,
        ttft_ms_mean=statistics.fmean(ttfts) if ttfts else 0.0,
        ttft_ms_p50=_percentile(ttfts, 0.5),
        ttft_ms_p90=_percentile(ttfts, 0.9),
        tpot_ms_mean=statistics.fmean(tpots) if tpots else 0.0,
    )


def format_report(result: BenchmarkResult) -> str:
    return "\n".join(
        [
            f"requests:              {result.requests}",
            f"prompt tokens (each):  {result.prompt_tokens}",
            f"max tokens:            {result.max_tokens}",
            f"output tokens (total): {result.total_output_tokens}",
            f"wall time:             {result.wall_seconds:.3f} s",
            f"throughput:            {result.throughput_tokens_per_s:.1f} tok/s",
            f"requests/s:            {result.requests_per_s:.2f}",
            f"TTFT mean/p50/p90:     "
            f"{result.ttft_ms_mean:.1f} / {result.ttft_ms_p50:.1f} / {result.ttft_ms_p90:.1f} ms",
            f"TPOT mean:             {result.tpot_ms_mean:.1f} ms",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("ENGINE_MODEL_DIR", "weights/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt", default="Explain how a CPU executes an instruction.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="emit metrics as JSON")
    args = parser.parse_args(argv)

    llm = LLM(args.model_dir, block_size=args.block_size, num_blocks=args.num_blocks)
    prompt_ids = [llm.tokenizer.encode(args.prompt) for _ in range(args.num_requests)]
    result = run_benchmark(llm, prompt_ids, args.max_tokens, temperature=args.temperature)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
