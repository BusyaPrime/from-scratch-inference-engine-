# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest

from engine import engine_ext

_MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(
    not (_MODEL_DIR / "model.safetensors").exists(),
    reason="weights absent; run scripts/fetch_model.py",
)
def test_forward_shape_and_finite():
    model = engine_ext.Model.from_pretrained(str(_MODEL_DIR))
    logits = model.forward([9707, 11, 1879])
    assert logits.shape == (3, 151936)
    assert np.isfinite(logits).all()
