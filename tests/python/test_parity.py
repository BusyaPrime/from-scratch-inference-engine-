# SPDX-License-Identifier: Apache-2.0
"""Correctness anchor: the engine's fp32 logits vs transformers (fp32, CPU).

The error metric is the fraction of next-token positions where the engine's
argmax disagrees with the oracle across the prompt set. transformers is
imported only here, never by the engine runtime (ADR-0003).
"""

from pathlib import Path

import numpy as np
import pytest

from engine import engine_ext
from engine.tokenizer import EngineTokenizer

_MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "Qwen2.5-0.5B-Instruct"

# Committed prompt set: short and long inputs, code, and non-ASCII (tokenizer boundary).
PROMPTS = [
    "The capital of France is",
    "Hello, world!",
    "import numpy as np",
    "Once upon a time, in a land far away,",
    "2 + 2 =",
    "The quick brown fox jumps over the lazy dog.",
    "Привет, как дела сегодня?",
    "def add(a, b):\n    return",
    "The mitochondria is the powerhouse of the",
    "Question: What is the speed of light? Answer:",
]

# Per-position next-token argmax error rate must stay at or below this.
MAX_ARGMAX_ERROR = 0.001
# Max absolute logit deviation vs the oracle (Amendment A anchor).
MAX_ABS_LOGIT = 1.0e-3

_needs_weights = pytest.mark.skipif(
    not (_MODEL_DIR / "model.safetensors").exists(),
    reason="weights absent; run scripts/fetch_model.py",
)


def _load_engine_and_oracle():
    from transformers import AutoModelForCausalLM

    engine_model = engine_ext.Model.from_pretrained(str(_MODEL_DIR))
    oracle = AutoModelForCausalLM.from_pretrained(str(_MODEL_DIR)).float()
    oracle.train(False)  # inference mode (disables dropout and similar)
    return engine_model, oracle


def _compare(engine_model, oracle, sequences):
    """Per-position stats over token-id sequences: (positions, argmax_mismatches, max_abs_logit)."""
    import torch

    positions = 0
    mismatches = 0
    max_abs = 0.0
    for ids in sequences:
        engine_logits = np.asarray(engine_model.forward(ids), dtype=np.float32)
        with torch.no_grad():
            ref_logits = oracle(torch.tensor([ids])).logits[0].to(torch.float32).numpy()
        assert engine_logits.shape == ref_logits.shape
        max_abs = max(max_abs, float(np.abs(engine_logits - ref_logits).max()))
        mismatches += int((engine_logits.argmax(-1) != ref_logits.argmax(-1)).sum())
        positions += len(ids)
    return positions, mismatches, max_abs


@pytest.mark.parity
@_needs_weights
def test_logit_and_token_parity_vs_transformers():
    tokenizer = EngineTokenizer.from_model_dir(_MODEL_DIR)
    engine_model, oracle = _load_engine_and_oracle()

    sequences = [tokenizer.encode(prompt) for prompt in PROMPTS]
    positions, mismatches, max_abs = _compare(engine_model, oracle, sequences)

    error_rate = mismatches / positions
    print(
        f"\nparity: positions={positions} argmax_mismatches={mismatches} "
        f"error_rate={error_rate:.4%} max_abs_logit={max_abs:.3e}"
    )
    assert error_rate <= MAX_ARGMAX_ERROR, f"argmax error rate {error_rate:.4%} exceeds 0.1%"
    assert max_abs < MAX_ABS_LOGIT, f"max abs logit diff {max_abs:.3e} too large"


@pytest.mark.parity
@_needs_weights
def test_greedy_continuation_parity_vs_transformers():
    # Generate the oracle's greedy continuation, then teacher-force the engine on the
    # SAME full sequence and compare per position. On identical contexts the logit
    # deviation is the robust drift anchor (Amendment A); exact self-generated token
    # identity is intentionally not asserted, because a single fp32 near-tie flip
    # amplifies autoregressively -- the case Amendment A's fallback rule expects.
    import torch

    new_tokens = 32
    prompts = ["The capital of France is", "import numpy as np", "Once upon a time,"]

    tokenizer = EngineTokenizer.from_model_dir(_MODEL_DIR)
    engine_model, oracle = _load_engine_and_oracle()

    sequences = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        with torch.no_grad():
            generated = oracle.generate(
                torch.tensor([ids]),
                do_sample=False,
                min_new_tokens=new_tokens,
                max_new_tokens=new_tokens,
            )
        sequences.append(generated[0].tolist())  # prompt + oracle greedy continuation

    positions, mismatches, max_abs = _compare(engine_model, oracle, sequences)
    error_rate = mismatches / positions
    print(
        f"\ngreedy-continuation: positions={positions} argmax_mismatches={mismatches} "
        f"error_rate={error_rate:.4%} max_abs_logit={max_abs:.3e}"
    )
    # The logit deviation is the drift anchor; argmax agreement is reported (near-ties relaxed).
    assert max_abs < MAX_ABS_LOGIT, f"logit drift {max_abs:.3e} over a continuation too large"
