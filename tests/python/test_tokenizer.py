# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from engine.tokenizer import EngineTokenizer

_MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(
    not (_MODEL_DIR / "tokenizer.json").exists(),
    reason="tokenizer not present; run scripts/fetch_model.py",
)
def test_encode_decode_round_trip():
    tok = EngineTokenizer.from_model_dir(_MODEL_DIR)
    text = "Hello, world! Привет! 你好"

    ids = tok.encode(text)
    assert ids
    assert all(isinstance(i, int) for i in ids)

    # Byte-level BPE is lossless, so re-encoding the decoded text reproduces the ids.
    assert tok.encode(tok.decode(ids)) == ids
