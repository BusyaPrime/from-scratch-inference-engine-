# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "weights" / "Qwen2.5-0.5B-Instruct"
_BENCH = _ROOT / "benchmarks" / "harness" / "bench_engine.py"

pytestmark = pytest.mark.skipif(
    not (_MODEL_DIR / "model.safetensors").exists(),
    reason="weights not present; run scripts/fetch_model.py",
)


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_engine", _BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass needs the module resolvable during exec
    spec.loader.exec_module(module)
    return module


def test_benchmark_runs_and_reports_sane_metrics():
    bench = _load_bench()
    from engine.llm import LLM

    llm = LLM(_MODEL_DIR, block_size=16, num_blocks=512)
    prompts = [llm.tokenizer.encode("Hello world") for _ in range(3)]
    result = bench.run_benchmark(llm, prompts, max_tokens=6)

    assert result.requests == 3
    assert result.total_output_tokens >= 3
    assert result.wall_seconds > 0.0
    assert result.throughput_tokens_per_s > 0.0
    assert result.ttft_ms_mean > 0.0
    assert isinstance(bench.format_report(result), str)
