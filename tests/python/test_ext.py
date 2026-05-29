# SPDX-License-Identifier: Apache-2.0
import engine


def test_native_core_version_matches_package():
    assert engine.core_version() == engine.__version__
