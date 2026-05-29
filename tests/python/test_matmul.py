# SPDX-License-Identifier: Apache-2.0
import numpy as np

from engine import engine_ext


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 5)).astype(np.float32)
    b = rng.standard_normal((5, 3)).astype(np.float32)

    got = engine_ext.matmul(a, b)

    assert got.shape == (7, 3)
    np.testing.assert_allclose(got, a @ b, rtol=1e-5, atol=1e-5)
