"""Minimal LLM layer: one protocol, a local Ollama backend, and a fake for tests.

Kept deliberately small. The agent only ever needs "given a system prompt and a user
prompt, return JSON matching this schema, and tell me the token usage", so that is the
whole interface. A hosted backend (Anthropic) is a second implementation of the same
protocol when the project moves off local models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
# Local generation on 7-8B models is slow; a single call can take minutes.
DEFAULT_TIMEOUT_SECONDS = 300.0
# Ollama defaults to a 4096-token context, which silently truncates CI evidence.
DEFAULT_NUM_CTX = 24_576


class LLMError(Exception):
    """Transport failure, or a response that was not usable JSON."""

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 1

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.calls + other.calls,
        )


class LLMClient(Protocol):
    name: str

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 2048
    ) -> tuple[dict[str, Any], Usage]: ...


class OllamaClient:
    """Local models via Ollama's /api/chat with JSON-schema constrained output."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        think: bool | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self._http = client or httpx.Client(base_url=base_url, timeout=timeout)

    @classmethod
    def from_env(cls) -> OllamaClient:
        return cls(
            model=os.environ.get("CI_TRIAGE_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
            num_ctx=int(os.environ.get("CI_TRIAGE_NUM_CTX", DEFAULT_NUM_CTX)),
        )

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 2048
    ) -> tuple[dict[str, Any], Usage]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": schema,  # Ollama constrains decoding to this JSON schema
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise LLMError(f"ollama returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable at {self._http.base_url}: {exc}") from exc

        content = (body.get("message") or {}).get("content", "")
        usage = Usage(
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return JSON: {exc}", raw_output=content) from exc
        if not isinstance(parsed, dict):
            raise LLMError("model returned JSON that is not an object", raw_output=content)
        return parsed, usage

    def close(self) -> None:
        self._http.close()


@dataclass
class FakeLLM:
    """Scripted client for tests: returns queued responses and records the prompts.

    A queued item may be a dict (returned as-is), a callable (called with system/user),
    or an exception instance (raised) to exercise the repair and failure paths.
    """

    responses: Sequence[Any] = field(default_factory=list)
    name: str = "fake"
    calls: list[dict[str, str]] = field(default_factory=list)
    _index: int = 0

    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 2048
    ) -> tuple[dict[str, Any], Usage]:
        self.calls.append({"system": system, "user": user})
        if self._index >= len(self.responses):
            raise LLMError("FakeLLM ran out of scripted responses")
        item = self.responses[self._index]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Callable):
            item = item(system, user)
        return item, Usage(input_tokens=len(user) // 4, output_tokens=64)
