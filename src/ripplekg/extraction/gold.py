"""Gold set: blind human labels of extracted edges, used to measure extraction precision and to
choose between models (protocol deviation D1).

The label sheet mixes edges from every model on the same chunks, shuffled, without saying which
model produced which edge, so labels cannot favour one model. The key that maps rows back to
models is written to a separate file.

A label answers one question: does the excerpt state this relationship, with this direction and
status? Y = yes, N = no, ? = unclear. Whether a name resolves to a graph node does not matter here;
verification measures that separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ripplekg.extraction.verify import normalize_text

SHEET_COLUMNS = ["row", "filer", "source", "relation", "target", "status", "strength", "quote", "context",
                 "label", "note"]  # fmt: skip
CONTEXT_CHARS = 350


def _context(text: str, quote: str) -> str:
    """The quote with some surrounding text, so the labeller can judge direction and status."""
    flat = " ".join(text.split())
    i = normalize_text(flat).find(normalize_text(quote))
    if i < 0:
        return flat[: 2 * CONTEXT_CHARS] + (" …" if len(flat) > 2 * CONTEXT_CHARS else "")
    start, end = max(0, i - CONTEXT_CHARS), min(len(flat), i + len(quote) + CONTEXT_CHARS)
    return ("… " if start else "") + flat[start:end] + (" …" if end < len(flat) else "")


def build_sheet(
    edges_by_model: dict[str, pd.DataFrame], chunks: pd.DataFrame, per_model: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sheet for the labeller, key mapping rows to models)."""
    text = chunks.set_index("chunk_id")["text"]
    picked = []
    for model, e in sorted(edges_by_model.items()):
        sample = e.sample(min(per_model, len(e)), random_state=seed).copy()
        sample["model"] = model
        picked.append(sample)
    rows = pd.concat(picked, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    rows["row"] = range(1, len(rows) + 1)
    sheet = pd.DataFrame(
        {
            "row": rows["row"],
            "filer": rows["filer"],
            "source": rows["source"],
            "relation": rows["relation"],
            "target": rows["target"],
            "status": rows["status"],
            "strength": rows["strength"].map(lambda v: "" if pd.isna(v) else f"{v:g}"),
            "quote": rows["quote"],
            "context": [_context(text[c], q) for c, q in zip(rows["chunk_id"], rows["quote"], strict=True)],
            "label": "",
            "note": "",
        }
    )[SHEET_COLUMNS]
    key = rows[["row", "model", "chunk_id", "family", "canonical_relation", "ok", "reason"]]
    return sheet, key


def score(labels: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Precision per model: share of labelled edges that are correct, overall and after verification."""
    df = key.merge(labels[["row", "label"]], on="row")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["Y", "N"])]
    out = []
    for model, g in df.groupby("model"):
        kept = g[g["ok"]]
        out.append(
            {
                "model": model,
                "labelled": len(g),
                "precision": round((g["label"] == "Y").mean(), 3),
                "verified": len(kept),
                "precision_after_verification": round((kept["label"] == "Y").mean(), 3)
                if len(kept)
                else None,
            }
        )
    return pd.DataFrame(out)


def write_sheet(sheet: pd.DataFrame, key: pd.DataFrame, folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path = folder / "label_sheet.csv", folder / "key.csv"
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)
    return sheet_path, key_path
