# SPDX-License-Identifier: Apache-2.0
import engine


def test_version_is_nonempty_string():
    assert isinstance(engine.__version__, str)
    assert engine.__version__
