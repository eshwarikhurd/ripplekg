"""Company universe: loads and validates config/universe.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ripplekg.settings import CONFIG_DIR

TIERS = {
    "miner_refiner",
    "diversified_miner",
    "materials_components",
    "cells_storage",
    "recycler",
    "automaker",
}


@dataclass(frozen=True)
class Company:
    id: str
    name: str
    tier: str
    search_names: tuple[str, ...]
    cik: int | None = None
    ticker: str | None = None
    listed_until: str | None = None
    successor: str | None = None
    notes: str = ""

    @property
    def is_filer(self) -> bool:
        return self.cik is not None


@dataclass(frozen=True)
class Universe:
    companies: tuple[Company, ...]
    by_id: dict[str, Company] = field(repr=False)

    @property
    def filers(self) -> list[Company]:
        return [c for c in self.companies if c.is_filer]


class UniverseError(ValueError):
    pass


def parse_universe(raw: dict) -> Universe:
    companies = []
    errors = []
    for row in raw.get("companies", []):
        try:
            c = Company(
                id=row["id"],
                name=row["name"],
                tier=row["tier"],
                search_names=tuple(row["search_names"]),
                cik=row.get("cik"),
                ticker=row.get("ticker"),
                listed_until=row.get("listed_until"),
                successor=row.get("successor"),
                notes=row.get("notes", ""),
            )
        except KeyError as exc:
            errors.append(f"{row.get('id', row)}: missing field {exc}")
            continue
        if c.tier not in TIERS:
            errors.append(f"{c.id}: unknown tier {c.tier!r}")
        if not c.search_names:
            errors.append(f"{c.id}: search_names is empty")
        if c.ticker and not c.is_filer:
            errors.append(f"{c.id}: has a ticker but no cik")
        companies.append(c)

    by_id: dict[str, Company] = {}
    for c in companies:
        if c.id in by_id:
            errors.append(f"duplicate id {c.id}")
        by_id[c.id] = c
    seen_ciks: dict[int, str] = {}
    for c in companies:
        if c.cik is not None:
            if c.cik in seen_ciks:
                errors.append(f"{c.id}: cik {c.cik} already used by {seen_ciks[c.cik]}")
            seen_ciks[c.cik] = c.id
        if c.successor and c.successor not in by_id:
            errors.append(f"{c.id}: successor {c.successor!r} is not in the universe")

    if errors:
        raise UniverseError("invalid universe:\n  " + "\n  ".join(errors))
    return Universe(companies=tuple(companies), by_id=by_id)


def load_universe(path: Path = CONFIG_DIR / "universe.yaml") -> Universe:
    with path.open() as f:
        return parse_universe(yaml.safe_load(f))
