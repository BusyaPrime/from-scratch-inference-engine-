# SPDX-License-Identifier: Apache-2.0
"""Download model weights and tokenizer files into a local, gitignored cache."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch model weights into ./weights")
    parser.add_argument("model_id", nargs="?", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dest", default="weights")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    local_dir = Path(args.dest) / args.model_id.split("/")[-1]
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=DEFAULT_PATTERNS,
    )
    print(f"downloaded {args.model_id} -> {path}")


if __name__ == "__main__":
    main()
