# SPDX-License-Identifier: Apache-2.0
"""Apply a model's own chat template (from tokenizer_config.json) to a message list.

Rendering the model-provided Jinja template keeps the server general across models and keeps
role/template text in the model's (untracked) data files rather than hardcoded in source. The
reply role is detected from the template's generation marker rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_chat_template(model_dir: str | Path) -> str | None:
    """Return the Jinja chat template from tokenizer_config.json, or None if absent."""
    config = Path(model_dir) / "tokenizer_config.json"
    if not config.exists():
        return None
    data = json.loads(config.read_text(encoding="utf-8"))
    template = data.get("chat_template")
    if isinstance(template, list):  # some configs ship a list of named templates
        template = template[0].get("template") if template else None
    return template


class ChatTemplate:
    """A compiled chat template plus the reply role it generates."""

    def __init__(self, template: str):
        from jinja2 import Environment

        env = Environment(trim_blocks=False, lstrip_blocks=False, autoescape=False)
        self._template = env.from_string(template)
        self.response_role = self._detect_response_role()

    def render(self, messages: list[dict], *, add_generation_prompt: bool = True) -> str:
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, tools=None
        )

    def _detect_response_role(self) -> str:
        """Infer the reply role from the text the template appends for the generation turn."""
        probe = [{"role": "user", "content": "x"}]
        with_marker = self.render(probe, add_generation_prompt=True)
        without_marker = self.render(probe, add_generation_prompt=False)
        suffix = with_marker[len(without_marker) :]
        marker = "<|im_start|>"
        if marker in suffix:
            role = suffix.split(marker)[-1].split("\n", 1)[0].strip()
            if role:
                return role
        return "model"
