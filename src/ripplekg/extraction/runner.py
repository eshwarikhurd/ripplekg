"""Run extraction over chunks, verify every edge, and summarize cost and quality."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ripplekg.extraction.llm import LLMProvider, ResponseCache
from ripplekg.extraction.prompts import PROMPT_VERSION, ExtractionOutcome, build_prompt, extract
from ripplekg.extraction.verify import Resolver, verify_edge
from ripplekg.ontology.schema import Ontology, load_examples
from ripplekg.ontology.universe import Universe

ANNUAL_FORMS = {"10-K", "20-F", "40-F"}


def sample_chunks(chunks: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A pilot sample spread across relation families: up to n / families chunks per family."""
    tagged = chunks[chunks["families"].map(len) > 0]
    families = sorted({f for fams in tagged["families"] for f in fams})
    per_family = max(1, n // len(families))
    picked: list[str] = []
    for fam in families:
        pool = tagged[tagged["families"].map(lambda f, fam=fam: fam in f) & ~tagged["chunk_id"].isin(picked)]
        picked += pool.sample(min(per_family, len(pool)), random_state=seed)["chunk_id"].tolist()
    rest = tagged[~tagged["chunk_id"].isin(picked)]
    if len(picked) < n and len(rest):
        picked += rest.sample(min(n - len(picked), len(rest)), random_state=seed)["chunk_id"].tolist()
    return tagged[tagged["chunk_id"].isin(picked[:n])]


@dataclass
class RunResult:
    outcomes: pd.DataFrame  # one row per (chunk, family) call
    edges: pd.DataFrame  # one row per extracted edge, with verification verdict


def run_extraction(
    chunks: pd.DataFrame,
    provider: LLMProvider,
    cache: ResponseCache,
    ont: Ontology,
    universe: Universe,
    workers: int = 4,
    progress: Callable[[str], None] = lambda _: None,
) -> RunResult:
    examples = load_examples()
    prompts = {f: build_prompt(ont, f, examples) for f in {f for fams in chunks["families"] for f in fams}}
    resolver = Resolver(ont)
    jobs = [(row, fam) for row in chunks.to_dict("records") for fam in row["families"]]

    def run(job) -> tuple[dict, ExtractionOutcome]:
        row, fam = job
        name = universe.by_id[row["company_id"]].name
        return row, extract(provider, cache, prompts[fam], row, name)

    outcome_rows, edge_rows = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (row, out) in enumerate(pool.map(run, jobs), 1):
            r = out.result
            outcome_rows.append(
                {
                    "chunk_id": out.chunk_id,
                    "family": out.family,
                    "company_id": row["company_id"],
                    "n_edges": len(out.edges),
                    "error": out.error,
                    "input_tokens": r.input_tokens if r else 0,
                    "output_tokens": r.output_tokens if r else 0,
                    "cache_read_tokens": r.cache_read_tokens if r else 0,
                    "cache_write_tokens": r.cache_write_tokens if r else 0,
                    "cost_usd": r.cost_usd if r else 0.0,
                    "from_cache": bool(r and r.cached),
                }
            )
            for e in out.edges:
                v = verify_edge(e, row["text"], ont, resolver)
                edge_rows.append(
                    {
                        **e,
                        "chunk_id": out.chunk_id,
                        "family": out.family,
                        "filer": row["company_id"],
                        "published_at": row["published_at"],
                        "doc_id": row["doc_id"],
                        "ok": v.ok,
                        "reason": v.reason,
                        "canonical_relation": v.relation,
                        "source_ref": v.source_ref,
                        "target_ref": v.target_ref,
                        "model": provider.model,
                        "prompt_version": PROMPT_VERSION,
                    }
                )
            if n % 25 == 0:
                progress(f"{n}/{len(jobs)} calls")
    return RunResult(pd.DataFrame(outcome_rows), pd.DataFrame(edge_rows))


def unresolved_lines(edges: pd.DataFrame, limit: int = 40) -> list[str]:
    """Names that failed to resolve: candidates for config/nodes or universe aliases."""
    if not len(edges):
        return []
    src = edges.loc[edges["reason"] == "unresolved source", "source"]
    tgt = edges.loc[edges["reason"] == "unresolved target", "target"]
    counts = pd.concat([src, tgt]).value_counts().head(limit)
    return [f"- {name} ({n})" for name, n in counts.items()]


def write_pilot_report(path: Path, res: RunResult, n_chunks: int, full_calls: int, model: str) -> str:
    o, e = res.outcomes, res.edges
    live = o[~o["from_cache"] & (o["error"] == "")]
    calls = len(o)
    all_in = live["input_tokens"] + live["cache_read_tokens"] + live["cache_write_tokens"]
    avg_in = all_in.mean() if len(live) else 0
    avg_out = live["output_tokens"].mean() if len(live) else 0
    per_call = live["cost_usd"].mean() if len(live) else 0
    kept = int(e["ok"].sum()) if len(e) else 0
    reasons = e["reason"].value_counts().to_dict() if len(e) else {}

    lines = [
        f"# Extraction pilot ({model}, prompt {PROMPT_VERSION})",
        "",
        f"- Chunks: {n_chunks}, calls: {calls}, errors: {int((o['error'] != '').sum())}",
        f"- Spent this run: ${o['cost_usd'].sum():.4f}",
        f"- Measured per call: {avg_in:,.0f} input tokens, {avg_out:,.0f} output tokens, ${per_call:.5f}",
        f"- Projected for {full_calls:,} calls: ${per_call * full_calls:.2f} standard, "
        f"${per_call * full_calls / 2:.2f} Batch API",
        f"- Edges extracted: {len(e)}, passed verification: {kept}",
        f"- Rejections: {reasons}",
        "",
        "## Names to review (unresolved endpoints, most frequent first)",
        "",
        *unresolved_lines(e),
        "",
        "## Verified edges",
        "",
    ]
    if len(e):
        for r in e[e["ok"]].itertuples():
            strength = f", strength {r.strength:g} ({r.strength_basis})" if pd.notna(r.strength) else ""
            lines.append(
                f"- **{r.source_ref} —{r.canonical_relation}→ {r.target_ref}** ({r.status}{strength})"
            )
            lines.append(f"  > {r.quote}")
        lines += ["", "## Rejected edges", ""]
        for r in e[~e["ok"]].itertuples():
            lines.append(f"- {r.source} —{r.relation}→ {r.target}: *{r.reason}*")
            lines.append(f"  > {r.quote}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines[:9])
