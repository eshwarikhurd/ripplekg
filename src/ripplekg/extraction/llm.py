"""LLM providers behind one interface, with an on-disk response cache.

Every call is keyed by a hash of (provider, model, prompt, schema). A cached response is never
re-requested, so switching providers, re-running a stage or resuming after a crash never pays
twice for the same answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

# Claude Haiku 5.5, USD per million tokens (prompts under 100K tokens), as of 2026-10.
# Cache reads and writes are approximated at 0.1x and 1.25x the input price.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class LLMResult:
    text: str
    model: str
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cached: bool = False  # served from the local cache: no cost incurred now
    batch: bool = False  # produced by the Batch API, billed at half price

    @property
    def cost_usd(self) -> float:
        if self.cached or self.model not in PRICES:
            return 0.0
        inp, out = PRICES[self.model]
        discount = 0.5 if self.batch else 1.0
        return (
            discount
            * (
                self.input_tokens * inp
                + self.cache_read_tokens * inp * 0.1
                + self.cache_write_tokens * inp * 1.25
                + self.output_tokens * out
            )
            / 1e6
        )


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> LLMResult: ...


class ClaudeProvider:
    """Anthropic Messages API with structured JSON output."""

    name = "claude"

    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "low", client=None):
        self.model = model
        self.effort = effort
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def request_params(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        """Messages API parameters; also used to build Batch API requests."""
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 4096) -> LLMResult:
        resp = self.client.messages.create(**self.request_params(system, user, schema, max_tokens))
        return result_from_message(resp)


class OllamaProvider:
    """A local model served by Ollama, with the JSON schema enforced as a grammar. Free to run."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        timeout: float = 900.0,
        session=None,
    ):
        import requests

        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.session = session or requests.Session()

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 4096) -> LLMResult:
        resp = self.session.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if data.get("done_reason") == "length":
            raise LLMError("response hit num_predict before the JSON was complete")
        return LLMResult(
            text=data["message"]["content"],
            model=self.model,
            stop_reason=data.get("done_reason", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


def result_from_message(resp) -> LLMResult:
    if resp.stop_reason == "refusal":
        raise LLMError("model declined the request (stop_reason=refusal)")
    if resp.stop_reason == "max_tokens":
        raise LLMError("response hit max_tokens before the JSON was complete")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    return LLMResult(
        text=text,
        model=resp.model,
        stop_reason=resp.stop_reason,
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def cache_key(provider: str, model: str, system: str, user: str, schema: dict) -> str:
    blob = json.dumps([provider, model, system, user, schema], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        raw["cached"] = True
        return LLMResult(**raw)

    def put(self, key: str, result: LLMResult) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({**asdict(result), "cached": False}))
        os.replace(tmp, path)


def cached_complete(
    provider: LLMProvider, cache: ResponseCache, system: str, user: str, schema: dict, max_tokens: int = 4096
) -> LLMResult:
    key = cache_key(provider.name, provider.model, system, user, schema)
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = provider.complete_json(system, user, schema, max_tokens)
    cache.put(key, result)
    return result
