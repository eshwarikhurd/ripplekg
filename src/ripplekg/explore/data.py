"""Load extraction runs for the explorer, re-verified under the current rules.

Raw triples are kept as the model wrote them. Each time the rules change (ontology, aliases,
verification code), re-verifying shows what would pass now without a new LLM call. The result is
cached per run in data/explorer/, keyed by the source file and the current ontology config, so the
app only re-verifies when something changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ripplekg import settings
from ripplekg.extraction.dedupe import dedupe_edges
from ripplekg.extraction.verify import Resolver, verify_edge
from ripplekg.ontology.schema import Ontology, load_ontology

CACHE_DIR = settings.DATA_DIR / "explorer"
CONFIG_FILES = ["relations.yaml", "universe.yaml", "nodes/materials.yaml", "nodes/components.yaml",
                "nodes/regions.yaml", "nodes/authorities.yaml", "nodes/programs.yaml",
                "nodes/facilities.yaml"]  # fmt: skip
VERIFY_VERSION = "2"  # bump when verification code changes in a way config hashing cannot see


@dataclass(frozen=True)
class Run:
    id: str
    label: str
    path: Path


RUNS = [
    Run(
        "v1-annual",
        "v1 prompt · annual reports (full run)",
        settings.DATA_DIR / "extractions" / "annual" / "edges.parquet",
    ),
    Run(
        "v2-pilot",
        "v2 prompt · pilot (50 chunks)",
        settings.RESULTS_DIR / "pilot" / "claude-v2" / "edges.parquet",
    ),
]


def available_runs() -> list[Run]:
    return [r for r in RUNS if r.path.exists()]


def triple_key(row: dict) -> str:
    """Stable id of one extracted triple, used to attach review marks."""
    parts = [str(row.get(k, "")) for k in ("chunk_id", "source", "relation", "target", "quote")]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _fingerprint(run: Run) -> str:
    h = hashlib.sha256(VERIFY_VERSION.encode())
    st = run.path.stat()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    for name in CONFIG_FILES:
        p = settings.CONFIG_DIR / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _filings() -> pd.DataFrame:
    path = settings.DATA_DIR / "index" / "filings.parquet"
    cols = ["accession", "form", "filing_date", "url"]
    return pd.read_parquet(path, columns=cols) if path.exists() else pd.DataFrame(columns=cols)


def reverify(raw: pd.DataFrame, chunk_text: dict[str, str], ont: Ontology) -> pd.DataFrame:
    resolver = Resolver(ont)
    rows = []
    for r in raw.to_dict("records"):
        v = verify_edge(r, chunk_text.get(r["chunk_id"], ""), ont, resolver, filer=r["filer"])
        rows.append(
            {
                **r,
                "ok": v.ok,
                "reason": v.reason,
                "canonical_relation": v.relation,
                "source_ref": v.source_ref,
                "target_ref": v.target_ref,
                "quote_full": v.quote_full,
                "material_ref": v.material_ref,
                "component_ref": v.component_ref,
            }
        )
    return dedupe_edges(pd.DataFrame(rows), ont)


def load_run(run: Run, ont: Ontology | None = None) -> pd.DataFrame:
    """Triples of a run, re-verified under current rules, with filing metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache, meta = CACHE_DIR / f"{run.id}.parquet", CACHE_DIR / f"{run.id}.json"
    fp = _fingerprint(run)
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fp:
        return pd.read_parquet(cache)

    ont = ont or load_ontology()
    raw = pd.read_parquet(run.path)
    raw = raw[[c for c in raw.columns if c not in {"ok", "reason", "canonical_relation", "source_ref",
                                                    "target_ref", "quote_full", "material_ref",
                                                    "component_ref"}]]  # fmt: skip
    chunks = pd.read_parquet(
        settings.DATA_DIR / "chunks" / "chunks.parquet", columns=["chunk_id", "text", "section"]
    )
    chunks = chunks[chunks["chunk_id"].isin(raw["chunk_id"])]
    df = reverify(raw, dict(zip(chunks["chunk_id"], chunks["text"], strict=True)), ont)
    df["section"] = df["chunk_id"].map(dict(zip(chunks["chunk_id"], chunks["section"], strict=True)))
    df["triple_key"] = [triple_key(r) for r in df.to_dict("records")]
    df["accession"] = df["doc_id"].str.removeprefix("edgar:")
    df = df.merge(_filings(), on="accession", how="left")
    df["year"] = df["published_at"].str[:4]
    df["status"] = df["status"].fillna("")

    tmp = cache.with_name(cache.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache)
    meta.write_text(json.dumps({"fingerprint": fp, "rows": len(df)}))
    return df


def chunk_text(chunk_id: str) -> str:
    chunks = pd.read_parquet(
        settings.DATA_DIR / "chunks" / "chunks.parquet",
        columns=["chunk_id", "text"],
        filters=[("chunk_id", "==", chunk_id)],
    )
    return chunks["text"].iloc[0] if len(chunks) else ""


def aggregate_edges(triples: pd.DataFrame) -> pd.DataFrame:
    """Graph edges: passed triples merged by (source, relation, target), with their evidence counts."""
    ok = triples[triples["ok"]]
    if ok.empty:
        return pd.DataFrame(columns=["source_ref", "canonical_relation", "target_ref", "triples", "filings",
                                     "first_seen", "last_seen", "statuses", "filers"])  # fmt: skip
    g = ok.groupby(["source_ref", "canonical_relation", "target_ref"])
    out = g.agg(
        triples=("triple_key", "count"),
        filings=("doc_id", "nunique"),
        first_seen=("published_at", "min"),
        last_seen=("published_at", "max"),
        statuses=("status", lambda s: ", ".join(sorted(set(s)))),
        filers=("filer", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index()
    out["first_seen"] = out["first_seen"].str[:10]
    out["last_seen"] = out["last_seen"].str[:10]
    return out.sort_values(["triples", "filings"], ascending=False, ignore_index=True)


def entity_table(triples: pd.DataFrame, ont: Ontology) -> pd.DataFrame:
    """Nodes that appear in passed triples, with degree and the raw name variants mapped to them."""
    ok = triples[triples["ok"]]
    if ok.empty:
        return pd.DataFrame(columns=["node", "kind", "name", "edges", "name_variants"])
    ends = pd.concat(
        [
            ok[["source_ref", "source"]].set_axis(["node", "written"], axis=1),
            ok[["target_ref", "target"]].set_axis(["node", "written"], axis=1),
        ]
    )
    edges = aggregate_edges(triples)
    degree = pd.concat([edges["source_ref"], edges["target_ref"]]).value_counts()
    rows = []
    for node, g in ends.groupby("node"):
        n = ont.nodes.get(node)
        variants = g["written"].value_counts()
        rows.append(
            {
                "node": node,
                "kind": n.kind if n else "?",
                "name": n.name if n else node,
                "edges": int(degree.get(node, 0)),
                "name_variants": "; ".join(f"{v} ({c})" for v, c in variants.head(6).items()),
            }
        )
    return pd.DataFrame(rows).sort_values("edges", ascending=False, ignore_index=True)


def unresolved_names(triples: pd.DataFrame) -> pd.DataFrame:
    """Names that failed to resolve, with how often and in which relations: the curation queue."""
    src = triples.loc[triples["reason"] == "unresolved source", ["source", "canonical_relation"]]
    tgt = triples.loc[triples["reason"] == "unresolved target", ["target", "canonical_relation"]]
    names = pd.concat(
        [src.set_axis(["name", "relation"], axis=1), tgt.set_axis(["name", "relation"], axis=1)]
    )
    names = names[names["name"].astype(str).str.strip() != ""]
    if names.empty:
        return pd.DataFrame(columns=["name", "count", "relations"])
    out = names.groupby("name").agg(
        count=("relation", "size"), relations=("relation", lambda s: ", ".join(s.value_counts().index[:3]))
    )
    return out.sort_values("count", ascending=False).reset_index()
