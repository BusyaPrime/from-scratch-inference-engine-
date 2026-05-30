# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from engine.server import create_app  # noqa: E402

_MODEL_DIR = Path(__file__).resolve().parents[2] / "weights" / "Qwen2.5-0.5B-Instruct"

pytestmark = pytest.mark.skipif(
    not (_MODEL_DIR / "model.safetensors").exists(),
    reason="weights not present; run scripts/fetch_model.py",
)


@pytest.fixture(scope="module")
def client():
    app = create_app(_MODEL_DIR, block_size=16, num_blocks=512, seed=0)
    with TestClient(app) as test_client:
        yield test_client


def test_models_lists_one(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1


def test_completion_non_stream(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "The capital of France is", "max_tokens": 12, "temperature": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"]
    assert body["usage"]["completion_tokens"] >= 1
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_completion_stream_matches_non_stream(client):
    payload = {"prompt": "Hello, my name is", "max_tokens": 12, "temperature": 0.0}
    full = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    chunks: list[str] = []
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as response:
        for raw in response.iter_lines():
            line = raw.rstrip("\r")
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: ") :]
            if payload_str.strip() == "[DONE]":
                break
            chunks.append(json.loads(payload_str)["choices"][0]["text"])

    assert "".join(chunks) == full


def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 12},
    )
    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert isinstance(message["role"], str) and message["role"]  # reply role from the template
    assert isinstance(message["content"], str)
