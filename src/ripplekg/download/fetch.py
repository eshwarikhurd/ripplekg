"""EDGAR source adapter.

fetch: download each selected filing's primary document, plus EX-99 exhibits for 8-Ks
(press releases) and 40-Fs (the annual information form lives in exhibit 99.1). Raw files are
stored gzipped under data/raw/{company_id}/{accession}/ with a manifest; files already on disk
with the recorded hash are skipped.

to_documents: raw filing -> Document with the sections that describe relationships.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ripplekg.download.client import EdgarClient

EXHIBIT_FORMS = {"8-K", "40-F"}
_EX99 = re.compile(r"ex(?:hibit)?[-_ ]?99", re.I)
# 8-K press releases can be a few KB; 40-F exhibits under 15 KB are auditor consents.
MIN_EXHIBIT_BYTES = {"8-K": 3_000, "40-F": 15_000}
MAX_EXHIBITS = 3

MANIFEST_COLUMNS = ["accession", "company_id", "file", "role", "bytes", "sha256", "fetched_at"]


def _write_gz(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def read_raw(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def exhibit_names(listing: dict, form: str = "8-K") -> list[str]:
    """EX-99 exhibit files in a filing directory listing, largest first, minus tiny ones."""
    items = listing.get("directory", {}).get("item", [])
    candidates = []
    for it in items:
        name = it.get("name", "")
        size = int(it.get("size") or 0)
        if (
            name.lower().endswith((".htm", ".html", ".txt"))
            and _EX99.search(name)
            and size >= MIN_EXHIBIT_BYTES.get(form, 15_000)
        ):
            candidates.append((size, name))
    return [name for _, name in sorted(candidates, reverse=True)[:MAX_EXHIBITS]]


def completed_accessions(manifest: pd.DataFrame, raw_dir: Path) -> set[str]:
    """Accessions whose primary document and every recorded file are on disk."""
    done = set()
    for acc, g in manifest.groupby("accession"):
        if (g["role"] == "primary").any() and all(
            (raw_dir / r.company_id / r.accession / (r.file + ".gz")).exists() for r in g.itertuples()
        ):
            done.add(acc)
    return done


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def fetch_filings(
    client: EdgarClient,
    selected: pd.DataFrame,
    raw_dir: Path,
    manifest_path: Path,
    progress: Callable[[str], None] = lambda _: None,
    save_every: int = 25,
) -> pd.DataFrame:
    """Download selected filings not already complete on disk. Idempotent and resumable:
    a filing's manifest rows are added only after all its files are written, and the
    manifest is saved every `save_every` filings."""
    manifest = load_manifest(manifest_path)
    done = completed_accessions(manifest, raw_dir)
    manifest = manifest[manifest["accession"].isin(done)]  # drop rows of half-finished filings
    rows = manifest.to_dict("records")
    todo = selected[~selected["accession"].isin(done)]
    progress(f"{len(selected) - len(todo)} filings already on disk, {len(todo)} to fetch")

    for n, r in enumerate(todo.itertuples(index=False), 1):
        folder = raw_dir / r.company_id / r.accession
        base = r.url.rsplit("/", 1)[0]
        files = [(r.primary_document, "primary")]
        if r.form in EXHIBIT_FORMS:
            files += [
                (name, "exhibit") for name in exhibit_names(client.get_json(base + "/index.json"), r.form)
            ]
        filing_rows = []
        for name, role in files:
            data = client.get_bytes(f"{base}/{name}", use_cache=False)
            _write_gz(folder / (name + ".gz"), data)
            filing_rows.append(
                {
                    "accession": r.accession,
                    "company_id": r.company_id,
                    "file": name,
                    "role": role,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        rows += filing_rows
        if n % save_every == 0 or n == len(todo):
            save_manifest(pd.DataFrame(rows, columns=MANIFEST_COLUMNS), manifest_path)
            progress(f"fetched {n}/{len(todo)}")
    out = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    save_manifest(out, manifest_path)
    return out
