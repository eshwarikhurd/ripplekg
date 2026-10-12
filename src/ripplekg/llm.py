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

    @property
    def cost_usd(self) -> float:
        if self.cached or self.model not in PRICES:
            return 0.0
        inp, out = PRICES[self.model]
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_write_tokens * inp * 1.25
            + self.output_tokens * out
        ) / 1e6


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
