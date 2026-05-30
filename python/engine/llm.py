# SPDX-License-Identifier: Apache-2.0
"""High-level text-in / text-out wrapper over the native continuous-batching engine.

Token ids are the unit that crosses into the C++ core; this layer adds the tokenizer and
the model directory's stop-token convention so callers can work in plain text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import engine_ext
from .tokenizer import EngineTokenizer


def _read_eos_id(model_dir: Path) -> int:
    """Read the end-of-sequence token id from the model directory; -1 if unspecified."""
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        eos = data.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        if eos is not None:
            return int(eos)
    return -1


@dataclass
class GenerationConfig:
    max_tokens: int = 64
    temperature: float = 0.0  # <= 0 is greedy
    top_k: int = 0
    top_p: float = 1.0


class LLM:
    """Load a model directory and generate text through the engine."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        block_size: int = 16,
        num_blocks: int = 2048,
        seed: int = 0,
        max_batch: int = 256,
        eos_id: int | None = None,
        enable_prefix_cache: bool = True,
    ):
        model_dir = Path(model_dir)
        self.model = engine_ext.Model.from_pretrained(str(model_dir))
        self.tokenizer = EngineTokenizer.from_model_dir(model_dir)
        self.engine = engine_ext.Engine(
            self.model, block_size, num_blocks, seed, max_batch, enable_prefix_cache
        )
        self.eos_id = _read_eos_id(model_dir) if eos_id is None else eos_id

    def _params(self, config: GenerationConfig) -> engine_ext.SamplingParams:
        return engine_ext.SamplingParams(config.temperature, config.top_k, config.top_p)

    def generate_ids(self, prompt_ids: list[int], config: GenerationConfig) -> list[int]:
        return self.engine.generate(
            prompt_ids, self._params(config), config.max_tokens, self.eos_id
        )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        ids = self.tokenizer.encode(prompt)
        return self.tokenizer.decode(self.generate_ids(ids, config))
