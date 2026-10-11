"""Rate-limited, cached HTTP client for SEC EDGAR.

SEC's fair-access policy allows at most 10 requests per second and requires a
User-Agent with a contact email. Responses are cached on disk by URL and query
parameters, so re-running a stage never re-downloads what it already has.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Spaces calls at least 1/rate seconds apart."""

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


class EdgarError(RuntimeError):
    pass


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None,
        rate: float = 8.0,
        max_retries: int = 5,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.session = session or requests.Session()
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(rate, sleep=sleep)
        self.max_retries = max_retries
        self._sleep = sleep
        self.network_calls = 0

    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / digest[:2] / digest

    def get_bytes(self, url: str, params: dict | None = None, use_cache: bool = True) -> bytes:
        path = self._cache_path(url, params)
        if use_cache and path is not None and path.exists():
            return path.read_bytes()

        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            self.network_calls += 1
            try:
                resp = self.session.get(url, params=params, headers=self.headers, timeout=30)
            except requests.ConnectionError as exc:
                if attempt == self.max_retries:
                    raise EdgarError(f"connection failed for {url}: {exc}") from exc
                self._sleep(min(2**attempt, 60))
                continue
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(min(2**attempt, 60))
                continue
            if resp.status_code != 200:
                raise EdgarError(f"HTTP {resp.status_code} for {url} params={params}")
            body = resp.content
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(body)
                tmp.replace(path)
            return body
        raise EdgarError(f"gave up on {url}")  # unreachable; loop always returns or raises

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True) -> Any:
        return json.loads(self.get_bytes(url, params=params, use_cache=use_cache))
