"""Human review of a curation draft, stored apart from the draft so a rebuilt draft never loses it.

Each decision is one JSON line in config/curation/<round>_review.jsonl; the latest line for an
entry wins, so changing your mind is another line. A decision is approve, reject or edit (approve
with changed fields). Entries are identified by section and group key, which are stable across
draft rebuilds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

SECTIONS = ("aliases", "new_nodes", "skip", "needs_review")
DECISIONS = ("approve", "reject", "edit")
EDITABLE = {
    "aliases": ("node",),
    "new_nodes": ("label", "kind", "parent", "region", "aliases"),
    "skip": (),
    "needs_review": ("action", "node"),  # resolve an undecided entry as an alias or a skip
}


@dataclass(frozen=True)
class Decision:
    section: str
    key: str
    decision: str
    edits: dict = field(default_factory=dict)
    note: str = ""
    reviewed_at: str = ""


def entry_id(section: str, key: str) -> str:
    return f"{section}::{key}"


def load_draft(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return {s: data.get(s) or [] for s in SECTIONS}


def entries_table(draft: dict) -> pd.DataFrame:
    """One row per draft entry, with the fields the review page shows."""
    rows = []
    for section in SECTIONS:
        for e in draft[section]:
            rows.append(
                {
                    "id": entry_id(section, e["key"]),
                    "section": section,
                    "key": e["key"],
                    "name": e.get("label") or e.get("name") or (e.get("variants") or [""])[0],
                    "target": e.get("node", ""),
                    "kind": e.get("kind", ""),
                    "parent": e.get("parent", ""),
                    "region": e.get("region", ""),
                    "seen": e.get("seen", 0),
                    "confidence": e.get("confidence"),
                    "pre_approved": bool(e.get("approve", False)),
                    "issues": len(e.get("issues") or []),
                    "corrected": bool(e.get("review_note")),
                }
            )
    return pd.DataFrame(rows)


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path

    def add(
        self, section: str, key: str, decision: str, edits: dict | None = None, note: str = ""
    ) -> Decision:
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        edits = edits or {}
        bad = set(edits) - set(EDITABLE[section])
        if bad:
            raise ValueError(f"fields {sorted(bad)} cannot be edited in {section}")
        d = Decision(
            section, key, decision, edits, note.strip(), datetime.now(UTC).isoformat(timespec="seconds")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        return d

    def add_many(self, items: list[tuple[str, str]], decision: str, note: str = "") -> int:
        for section, key in items:
            self.add(section, key, decision, note=note)
        return len(items)

    def latest(self) -> dict[str, Decision]:
        """Latest decision per entry id."""
        if not self.path.exists():
            return {}
        out: dict[str, Decision] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = Decision(**json.loads(line))
                out[entry_id(d.section, d.key)] = d
        return out
