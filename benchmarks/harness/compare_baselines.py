# SPDX-License-Identifier: Apache-2.0
"""Head-to-head: the engine vs the Hugging Face transformers reference.

Same model directory, fp32, greedy decoding. Reports decode throughput for each and checks that
the generated token ids agree. transformers is used here only as a benchmark baseline, never as a
runtime dependency of the engine (ADR-0003); it is imported lazily so the engine import stays
clean.

A vllm / nano-vllm comparison follows the same recipe (same prompt, model, precision, decode
length) on a Linux + CUDA host where those engines install cleanly; this harness keeps the
transformers reference because it runs anywhere torch does.

Run:
  pip install -e ".[dev]" && pip install torch transformers
  python scripts/fetch_model.py
  python benchmarks/harness/compare_baselines.py --model-dir weights/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import time

from engine.llm import LLM


def _read_eos_id(model_dir: str) -> int:
    import json
    from pathlib import Path

    for name in ("generation_config.json", "config.json"):
        path = Path(model_dir) / name
        if path.exists():
            eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
            if isinstance(eos, list):
                eos = eos[0] if eos else None
            if eos is not None:
                return int(eos)
    return -1


def engine_run(llm: LLM, prompt_ids: list[int], max_tokens: int) -> tuple[list[int], float]:
    from engine import engine_ext

    params = engine_ext.SamplingParams(0.0, 0, 1.0)  # greedy
    start = time.perf_counter()
    out = llm.engine.generate(list(prompt_ids), params, max_tokens, llm.eos_id)
    elapsed = time.perf_counter() - start
    return out, len(out) / elapsed if elapsed > 0 else 0.0


def transformers_run(
    model, prompt_ids: list[int], max_tokens: int, eos_id: int
) -> tuple[list[int], float]:
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    kwargs = {"max_new_tokens": max_tokens, "do_sample": False, "use_cache": True}
    if eos_id >= 0:
        kwargs["eos_token_id"] = eos_id
    with torch.no_grad():
        start = time.perf_counter()
        out = model.generate(input_ids, **kwargs)
        elapsed = time.perf_counter() - start
    generated = out[0][len(prompt_ids) :].tolist()
    return generated, len(generated) / elapsed if elapsed > 0 else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="weights/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    llm = LLM(args.model_dir, block_size=16, num_blocks=4096)
    prompt_ids = llm.tokenizer.encode(args.prompt)
    eos_id = _read_eos_id(args.model_dir)

    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(args.model_dir).float()
    reference.train(False)

    # Warm up both paths so the timed runs exclude one-off costs.
    engine_run(llm, prompt_ids, 2)
    transformers_run(reference, prompt_ids, 2, eos_id)

    engine_out, engine_rate = engine_run(llm, prompt_ids, args.max_tokens)
    ref_out, ref_rate = transformers_run(reference, prompt_ids, args.max_tokens, eos_id)

    leading = 0
    for a, b in zip(engine_out, ref_out, strict=False):
        if a != b:
            break
        leading += 1

    print(f"model:                {args.model_dir}")
    print(f"prompt tokens:        {len(prompt_ids)}")
    print(f"max tokens:           {args.max_tokens}")
    print(f"engine decode:        {engine_rate:.1f} tok/s ({len(engine_out)} tokens)")
    print(f"transformers decode:  {ref_rate:.1f} tok/s ({len(ref_out)} tokens)")
    if ref_rate > 0:
        print(f"engine / reference:   {engine_rate / ref_rate:.2f}x")
    print(f"leading greedy match: {leading}/{args.max_tokens} tokens")
    print("(free-running greedy can diverge on an fp32 near-tie;")
    print(" per-step parity is the test anchor)")


if __name__ == "__main__":
    main()
