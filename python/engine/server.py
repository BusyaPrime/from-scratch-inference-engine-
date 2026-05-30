# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server.

A single continuous-batching engine is shared by all clients: a background task steps the
engine and fans newly produced tokens out to per-request queues, so concurrent requests are
batched together. Responses follow the OpenAI /v1/completions and /v1/chat/completions shapes,
with optional Server-Sent Events streaming. Chat requests are rendered with the model's own
template (loaded from its data files) rather than a hardcoded format.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import engine_ext
from .chat import ChatTemplate, load_chat_template
from .llm import LLM


class EngineService:
    """Owns the engine and runs the step loop, fanning tokens out to per-request queues."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self._queues: dict[int, asyncio.Queue] = {}
        self._emitted: dict[int, int] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def submit(
        self, prompt_ids: list[int], params: engine_ext.SamplingParams, max_tokens: int
    ) -> int:
        seq_id = self.llm.engine.add_request(prompt_ids, params, max_tokens, self.llm.eos_id)
        self._queues[seq_id] = asyncio.Queue()
        self._emitted[seq_id] = 0
        self._wakeup.set()
        return seq_id

    async def stream_tokens(self, seq_id: int) -> AsyncIterator[int]:
        queue = self._queues[seq_id]
        try:
            while True:
                kind, token = await queue.get()
                if kind == "done":
                    return
                yield token
        finally:
            self._queues.pop(seq_id, None)
            self._emitted.pop(seq_id, None)

    async def _run(self) -> None:
        while True:
            if not self.llm.engine.has_work():
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            self.llm.engine.step()
            for seq_id in list(self._emitted.keys()):
                output = self.llm.engine.output(seq_id)
                start = self._emitted[seq_id]
                for token in output[start:]:
                    self._queues[seq_id].put_nowait(("token", int(token)))
                self._emitted[seq_id] = len(output)
                if self.llm.engine.status(seq_id) == engine_ext.SeqStatus.Finished:
                    self._queues[seq_id].put_nowait(("done", 0))
                    del self._emitted[seq_id]
            await asyncio.sleep(0)  # let streaming handlers flush between steps


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


def _sampling(req: CompletionRequest | ChatRequest) -> engine_ext.SamplingParams:
    return engine_ext.SamplingParams(req.temperature, req.top_k, req.top_p)


async def _collect_text(service: EngineService, seq_id: int) -> tuple[str, int]:
    out_ids: list[int] = []
    async for token in service.stream_tokens(seq_id):
        out_ids.append(token)
    return service.llm.tokenizer.decode(out_ids), len(out_ids)


async def _sse_stream(
    service: EngineService,
    seq_id: int,
    model_id: str,
    *,
    object_name: str,
    response_id: str,
    role: str | None,
) -> AsyncIterator[str]:
    """Emit OpenAI-style SSE chunks. role!=None selects chat (delta) shape, else completion text."""
    created = int(time.time())

    def envelope(choice: dict) -> str:
        chunk = {
            "id": response_id,
            "object": object_name,
            "created": created,
            "model": model_id,
            "choices": [choice],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    if role is not None:
        yield envelope({"index": 0, "delta": {"role": role}, "finish_reason": None})

    out_ids: list[int] = []
    previous = ""
    async for token in service.stream_tokens(seq_id):
        out_ids.append(token)
        text = service.llm.tokenizer.decode(out_ids)
        delta = text[len(previous) :]
        previous = text
        if not delta:
            continue
        choice = (
            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
            if role is not None
            else {"index": 0, "text": delta, "finish_reason": None}
        )
        yield envelope(choice)

    final = (
        {"index": 0, "delta": {}, "finish_reason": "stop"}
        if role is not None
        else {"index": 0, "text": "", "finish_reason": "stop"}
    )
    yield envelope(final)
    yield "data: [DONE]\n\n"


def create_app(model_dir: str | Path | None = None, **llm_kwargs) -> FastAPI:
    model_dir = model_dir or os.environ.get("ENGINE_MODEL_DIR")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_dir is None:
            raise RuntimeError("set ENGINE_MODEL_DIR or pass model_dir to create_app")
        llm = LLM(model_dir, **llm_kwargs)
        template = load_chat_template(model_dir)
        app.state.service = EngineService(llm)
        app.state.service.start()
        app.state.chat = ChatTemplate(template) if template else None
        app.state.model_id = Path(model_dir).name
        try:
            yield
        finally:
            await app.state.service.stop()

    app = FastAPI(title="from-scratch inference engine", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> dict:
        model_id = request.app.state.model_id
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "engine"}],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        service: EngineService = request.app.state.service
        model_id = request.app.state.model_id
        prompt_ids = service.llm.tokenizer.encode(req.prompt)
        seq_id = service.submit(prompt_ids, _sampling(req), req.max_tokens)
        if req.stream:
            return StreamingResponse(
                _sse_stream(
                    service,
                    seq_id,
                    model_id,
                    object_name="text_completion",
                    response_id="cmpl-" + uuid.uuid4().hex,
                    role=None,
                ),
                media_type="text/event-stream",
            )
        text, completion_tokens = await _collect_text(service, seq_id)
        return JSONResponse(
            {
                "id": "cmpl-" + uuid.uuid4().hex,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_ids) + completion_tokens,
                },
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request):
        service: EngineService = request.app.state.service
        model_id = request.app.state.model_id
        chat: ChatTemplate | None = request.app.state.chat
        if chat is None:
            return JSONResponse(status_code=501, content={"error": "model has no chat template"})

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = chat.render(messages)
        prompt_ids = service.llm.tokenizer.encode(prompt)
        seq_id = service.submit(prompt_ids, _sampling(req), req.max_tokens)
        if req.stream:
            return StreamingResponse(
                _sse_stream(
                    service,
                    seq_id,
                    model_id,
                    object_name="chat.completion.chunk",
                    response_id="chatcmpl-" + uuid.uuid4().hex,
                    role=chat.response_role,
                ),
                media_type="text/event-stream",
            )
        text, completion_tokens = await _collect_text(service, seq_id)
        return JSONResponse(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": chat.response_role, "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_ids) + completion_tokens,
                },
            }
        )

    return app
