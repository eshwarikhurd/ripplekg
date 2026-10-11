"""Test doubles for the EDGAR client: a scripted HTTP session and a fake clock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""


@dataclass
class FakeSession:
    """Returns queued responses per URL, or a routing function's response, and records calls."""

    routes: dict[str, list[FakeResponse]] = field(default_factory=dict)
    router: object = None
    calls: list[tuple[str, dict | None, dict]] = field(default_factory=list)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers or {}))
        if self.router is not None:
            return self.router(url, params)
        return self.routes[url].pop(0)


def json_response(obj, status: int = 200) -> FakeResponse:
    return FakeResponse(status, json.dumps(obj).encode())


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
