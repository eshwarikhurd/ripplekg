"""Review marks from the explorer: append-only, never deletes data.

Each mark is one JSON line in data/reviews/reviews.jsonl. The latest mark for a triple wins, so
changing your mind is just another line. Marks feed the gold set and node curation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LABELS = {"Y": "correct", "N": "wrong", "?": "unsure"}


@dataclass(frozen=True)
class Mark:
    triple_key: str
    run: str
    label: str
    note: str
    marked_at: str


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path

    def add(self, triple_key: str, run: str, label: str, note: str = "") -> Mark:
        if label not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}")
        mark = Mark(triple_key, run, label, note.strip(), datetime.now(UTC).isoformat(timespec="seconds"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(mark), ensure_ascii=False) + "\n")
        return mark

    def all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=list(Mark.__dataclass_fields__))
        rows = [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return pd.DataFrame(rows, columns=list(Mark.__dataclass_fields__))

    def latest(self) -> pd.DataFrame:
        """One row per triple: its most recent mark."""
        df = self.all()
        if df.empty:
            return df
        return df.sort_values("marked_at").groupby("triple_key", as_index=False).last()
