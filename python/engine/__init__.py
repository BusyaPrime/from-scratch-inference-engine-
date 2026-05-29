# SPDX-License-Identifier: Apache-2.0
"""From-scratch LLM inference engine (Python frontend)."""

from . import engine_ext

__version__ = "0.0.1"


def core_version() -> str:
    """Version string reported by the compiled native core."""
    return engine_ext.version()
