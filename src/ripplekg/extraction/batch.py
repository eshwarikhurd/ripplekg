"""Bulk extraction through the Message Batches API (50% of standard price, asynchronous).

Each request's custom_id is its response-cache key, so collected results land in the same cache
the synchronous path reads: after `collect`, a normal extraction run is all cache hits. Submitted
batch ids are kept in a state file, so submit and collect can run in separate sessions, and a
request already cached or already in a live batch is never submitted twice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ripplekg.extraction.llm import ClaudeProvider, ResponseCache, cache_key, result_from_message
from ripplekg.extraction.prompts import build_prompt, user_message
from ripplekg.ontology.schema import Ontology, load_examples
from ripplekg.ontology.universe import Universe

BATCH_SIZE = 4_000  # keeps each batch far below the 256 MB request limit
MAX_TOKENS = 4096


@dataclass
class Job:
    key: str
    params: dict


def prepare_jobs(
    chunks: pd.DataFrame, provider: ClaudeProvider, cache: ResponseCache, ont: Ontology, universe: Universe
) -> tuple[list[Job], int]:
    """Requests for every (chunk, family) not already cached. Returns (jobs, n_cached)."""
    examples = load_examples()
    prompts = {f: build_prompt(ont, f, examples) for f in {f for fams in chunks["families"] for f in fams}}
    jobs, cached, seen = [], 0, set()
    for row in chunks.to_dict("records"):
        name = universe.by_id[row["company_id"]].name
        user = user_message(
            name, row["company_id"], row["form"], row["published_at"], row["section"], row["text"]
        )
        for fam in row["families"]:
            p = prompts[fam]
            key = cache_key(provider.name, provider.model, p.system, user, p.schema)
            if cache.get(key) is not None:
                cached += 1
                continue
            if key in seen:  # identical text and metadata elsewhere: one request answers both
                continue
            seen.add(key)
            jobs.append(Job(key, provider.request_params(p.system, user, p.schema, MAX_TOKENS)))
    return jobs, cached


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"batches": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def submit(client, jobs: list[Job], state_path: Path, progress: Callable[[str], None] = print) -> list[str]:
    """Submit jobs not already part of a batch that is still open. Returns new batch ids."""
    state = _load_state(state_path)
    pending = {k for b in state["batches"] if not b.get("collected") for k in b["keys"]}
    jobs = [j for j in jobs if j.key not in pending]
    new_ids = []
    for start in range(0, len(jobs), BATCH_SIZE):
        part = jobs[start : start + BATCH_SIZE]
        batch = client.messages.batches.create(
            requests=[{"custom_id": j.key, "params": j.params} for j in part]
        )
        state["batches"].append({"id": batch.id, "keys": [j.key for j in part], "collected": False})
        _save_state(state_path, state)
        new_ids.append(batch.id)
        progress(f"submitted {batch.id} with {len(part)} requests")
    return new_ids


def status(client, state_path: Path) -> list[dict]:
    rows = []
    for b in _load_state(state_path)["batches"]:
        info = client.messages.batches.retrieve(b["id"])
        c = info.request_counts
        rows.append(
            {
                "id": b["id"],
                "status": info.processing_status,
                "requests": len(b["keys"]),
                "succeeded": c.succeeded,
                "errored": c.errored,
                "processing": c.processing,
                "collected": b.get("collected", False),
            }
        )
    return rows


def collect(client, cache: ResponseCache, state_path: Path) -> dict:
    """Write results of every ended, uncollected batch into the cache."""
    state = _load_state(state_path)
    totals = {"succeeded": 0, "failed": 0, "cost_usd": 0.0, "batches_waiting": 0}
    for b in state["batches"]:
        if b.get("collected"):
            continue
        if client.messages.batches.retrieve(b["id"]).processing_status != "ended":
            totals["batches_waiting"] += 1
            continue
        for item in client.messages.batches.results(b["id"]):
            if item.result.type != "succeeded":
                totals["failed"] += 1  # not cached, so the next submit retries it
                continue
            try:
                result = result_from_message(item.result.message)
            except Exception:
                totals["failed"] += 1
                continue
            result.batch = True
            cache.put(item.custom_id, result)
            totals["succeeded"] += 1
            totals["cost_usd"] += result.cost_usd
        b["collected"] = True
        _save_state(state_path, state)
    return totals
