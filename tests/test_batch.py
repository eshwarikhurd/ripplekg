import json
from types import SimpleNamespace

import pandas as pd

from ripplekg.extraction import batch
from ripplekg.extraction.llm import ClaudeProvider, ResponseCache
from ripplekg.ontology.schema import load_ontology
from ripplekg.ontology.universe import load_universe

ONT, UNIVERSE = load_ontology(), load_universe()
CHUNKS = pd.DataFrame(
    [
        {
            "chunk_id": "c1",
            "company_id": "TSLA",
            "form": "10-K",
            "published_at": "2024-01-29T21:00:00+00:00",
            "section": "business",
            "text": "We purchase battery cells from Panasonic.",
            "families": ["trade", "footprint"],
        }
    ]
)


def message(model):
    return SimpleNamespace(
        stop_reason="end_turn",
        model=model,
        content=[SimpleNamespace(type="text", text=json.dumps({"edges": []}))],
        usage=SimpleNamespace(
            input_tokens=3000, output_tokens=400, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )


class FakeBatches:
    def __init__(self):
        self.created = []

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def retrieve(self, batch_id):
        return SimpleNamespace(
            processing_status="ended", request_counts=SimpleNamespace(succeeded=2, errored=0, processing=0)
        )

    def results(self, batch_id):
        reqs = self.created[int(batch_id.split("_")[1]) - 1]
        for r in reqs:
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(type="succeeded", message=message(r["params"]["model"])),
            )


def test_submit_collect_fills_cache_at_half_price(tmp_path):
    batches = FakeBatches()
    provider = ClaudeProvider(client=SimpleNamespace(messages=SimpleNamespace(batches=batches)))
    cache, state = ResponseCache(tmp_path / "cache"), tmp_path / "batches.json"

    jobs, cached = batch.prepare_jobs(CHUNKS, provider, cache, ONT, UNIVERSE)
    assert (len(jobs), cached) == (2, 0)
    assert all(len(j.key) == 64 for j in jobs)  # valid custom_id length

    batch.submit(provider.client, jobs, state, progress=lambda m: None)
    batch.submit(provider.client, jobs, state, progress=lambda m: None)  # still pending: not resubmitted
    assert len(batches.created) == 1

    totals = batch.collect(provider.client, cache, state)
    assert totals["succeeded"] == 2
    full_price = (3000 * 0.10 + 400 * 0.50) / 1e6
    assert abs(totals["cost_usd"] - 2 * full_price / 2) < 1e-12

    jobs, cached = batch.prepare_jobs(CHUNKS, provider, cache, ONT, UNIVERSE)
    assert (len(jobs), cached) == (0, 2)


def test_identical_requests_are_submitted_once(tmp_path):
    provider = ClaudeProvider(client=SimpleNamespace())
    twice = pd.concat([CHUNKS, CHUNKS.assign(chunk_id="c2")])
    jobs, _ = batch.prepare_jobs(twice, provider, ResponseCache(tmp_path), ONT, UNIVERSE)
    assert len(jobs) == 2
