# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from engine.llm import LLM, GenerationConfig

_MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "Qwen2.5-0.5B-Instruct"

pytestmark = pytest.mark.skipif(
    not (_MODEL_DIR / "model.safetensors").exists(),
    reason="weights not present; run scripts/fetch_model.py",
)


def test_greedy_generation_is_deterministic_and_decodes():
    llm = LLM(_MODEL_DIR, block_size=16, num_blocks=512, seed=0)
    config = GenerationConfig(max_tokens=16, temperature=0.0)

    first = llm.generate("The capital of France is", config)
    second = llm.generate("The capital of France is", config)

    assert isinstance(first, str)
    assert first  # produced something
    assert first == second  # greedy decoding is deterministic across runs


def test_concurrent_requests_match_sequential():
    llm = LLM(_MODEL_DIR, block_size=16, num_blocks=512, seed=0)
    config = GenerationConfig(max_tokens=12, temperature=0.0)
    prompts = ["Hello, my name is", "The quick brown fox", "2 + 2 ="]

    sequential = [llm.tokenizer.encode(p) for p in prompts]
    sequential_out = [llm.generate_ids(ids, config) for ids in sequential]

    # Schedule all three together; each must reproduce its sequential greedy output.
    ids = [llm.tokenizer.encode(p) for p in prompts]
    params = llm._params(config)
    seq_ids = [llm.engine.add_request(i, params, config.max_tokens, llm.eos_id) for i in ids]
    while llm.engine.has_work():
        llm.engine.step()
    batched_out = [list(llm.engine.output(s)) for s in seq_ids]

    assert batched_out == sequential_out


def test_prefix_cache_reuses_shared_prompt_and_matches():
    config = GenerationConfig(max_tokens=16, temperature=0.0)
    prompt = "The capital of France is the city of"
    follow = "The capital of France is famous for"  # shares a long token prefix

    plain = LLM(_MODEL_DIR, block_size=16, num_blocks=512, seed=0, enable_prefix_cache=False)
    ref_a = plain.generate(prompt, config)
    ref_b = plain.generate(follow, config)

    cached = LLM(_MODEL_DIR, block_size=16, num_blocks=512, seed=0, enable_prefix_cache=True)
    got_a = cached.generate(prompt, config)
    got_b = cached.generate(follow, config)

    assert got_a == ref_a
    assert got_b == ref_b
    assert cached.engine.prefix_hits() > 0  # the shared prefix was reused from cache


def test_quantized_generation_runs_and_decodes():
    llm = LLM(_MODEL_DIR, block_size=16, num_blocks=512, seed=0, quantize=True)
    assert llm.model.is_quantized()

    config = GenerationConfig(max_tokens=16, temperature=0.0)
    out = llm.generate("The capital of France is", config)
    assert isinstance(out, str) and out  # the quantized serving path produces text
