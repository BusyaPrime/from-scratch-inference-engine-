# SPDX-License-Identifier: Apache-2.0
"""Tokenizer wrapper over the Hugging Face tokenizers library.

Token ids are the unit that crosses the binding boundary into the C++ engine.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class EngineTokenizer:
    """Thin encode/decode wrapper around a tokenizers.Tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> EngineTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> EngineTokenizer:
        return cls.from_file(Path(model_dir) / "tokenizer.json")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)
