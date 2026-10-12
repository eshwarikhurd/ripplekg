"""Benchmark events: loads, validates and splits config/events.yaml.

An event origin is a list of node references. A bare id ("TSLA") refers to a
company in the universe; any other node uses a "kind:id" prefix
("material:lithium", "region:CL", "authority:CN-MOFCOM", "program:US-IRA").
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from ripplekg.ontology.universe import Universe, load_universe
from ripplekg.settings import CONFIG_DIR

NODE_KINDS = {"company", "facility", "material", "component", "region", "authority", "program"}
SHOCK_TYPES = {"supply_shock", "demand_shock", "mixed"}
BUCKETS = {"primary", "macro"}
SIGNS = {"negative", "positive", "mixed"}
TIMINGS = {"before_open", "during_market", "after_close", "non_trading_day", "unknown"}
SPLITS = {"dev", "test"}

# Pre-registered split: seed and per-stratum dev counts are fixed in docs/protocol.md.
SPLIT_SEED = 20261011
DEV_COUNTS = {"company": 4, "non_company": 2}

# Primary event windows [day0, day0 + EVENT_WINDOW_DAYS] must not overlap (protocol 2.1).
EVENT_WINDOW_DAYS = 3


def weekdays_between(a: date, b: date) -> int:
    """Number of weekdays d with a < d <= b (a conservative stand-in for trading days)."""
    if b <= a:
        return -weekdays_between(b, a) if b < a else 0
    full_weeks, rem = divmod((b - a).days, 7)
    count = full_weeks * 5
    for i in range(1, rem + 1):
        if (a.weekday() + i) % 7 < 5:
            count += 1
    return count


_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Event:
    id: str
    title: str
    bucket: str
    origin: tuple[str, ...]
    shock_type: str
    sign: str
    announced: date
    timing: str
    day0: date
    named: tuple[str, ...]
    sources: tuple[str, ...]
    split: str | None
    notes: str = ""

    @property
    def origin_kind(self) -> str:
        """'company' if every origin node is a company, otherwise 'non_company'."""
        return "company" if all(":" not in o for o in self.origin) else "non_company"


class EventsError(ValueError):
    pass


def _as_date(value, field: str, eid: str, errors: list[str]) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        errors.append(f"{eid}: {field} is not an ISO date: {value!r}")
        return None


def parse_events(raw: dict, universe: Universe) -> list[Event]:
    events: list[Event] = []
    errors: list[str] = []
    for row in raw.get("events", []):
        eid = row.get("id", "?")
        try:
            announced = _as_date(row["announced"], "announced", eid, errors)
            day0 = _as_date(row["day0"], "day0", eid, errors)
            e = Event(
                id=eid,
                title=row["title"],
                bucket=row["bucket"],
                origin=tuple(row["origin"]),
                shock_type=row["shock_type"],
                sign=row["sign"],
                announced=announced,
                timing=row["timing"],
                day0=day0,
                named=tuple(row.get("named", [])),
                sources=tuple(row["sources"]),
                split=row.get("split"),
                notes=row.get("notes", ""),
            )
        except KeyError as exc:
            errors.append(f"{eid}: missing field {exc}")
            continue

        if not _ID.match(e.id):
            errors.append(f"{e.id}: id must be lowercase-hyphenated")
        for name, value, allowed in [
            ("bucket", e.bucket, BUCKETS),
            ("shock_type", e.shock_type, SHOCK_TYPES),
            ("sign", e.sign, SIGNS),
            ("timing", e.timing, TIMINGS),
        ]:
            if value not in allowed:
                errors.append(f"{e.id}: {name} {value!r} not in {sorted(allowed)}")
        if not e.origin:
            errors.append(f"{e.id}: origin is empty")
        for node in e.origin:
            if ":" in node:
                kind = node.split(":", 1)[0]
                if kind not in NODE_KINDS:
                    errors.append(f"{e.id}: origin {node!r} has unknown kind {kind!r}")
            elif node not in universe.by_id:
                errors.append(f"{e.id}: origin company {node!r} is not in the universe")
        for c in e.named:
            if c not in universe.by_id:
                errors.append(f"{e.id}: named company {c!r} is not in the universe")
        if not e.sources or not all(s.startswith("https://") for s in e.sources):
            errors.append(f"{e.id}: needs at least one https source")
        if announced and day0:
            if day0 < announced:
                errors.append(f"{e.id}: day0 {day0} is before announced {announced}")
            if day0.weekday() >= 5:
                errors.append(f"{e.id}: day0 {day0} is a weekend")
            if e.timing in {"after_close", "non_trading_day"} and day0 == announced:
                errors.append(f"{e.id}: timing {e.timing} but day0 equals announced date")
        if e.bucket == "primary" and e.split not in SPLITS:
            errors.append(f"{e.id}: primary event needs split dev|test, got {e.split!r}")
        if e.bucket == "macro" and e.split is not None:
            errors.append(f"{e.id}: macro events are not split")
        events.append(e)

    seen: set[str] = set()
    for e in events:
        if e.id in seen:
            errors.append(f"duplicate event id {e.id}")
        seen.add(e.id)

    primary = sorted((e for e in events if e.bucket == "primary" and e.day0), key=lambda e: e.day0)
    for a, b in zip(primary, primary[1:], strict=False):
        if weekdays_between(a.day0, b.day0) <= EVENT_WINDOW_DAYS:
            errors.append(f"{b.id}: event window overlaps {a.id} (drop the later event, protocol 2.1)")

    if errors:
        raise EventsError("invalid events:\n  " + "\n  ".join(errors))
    return events


def assign_split(events: list[Event], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Pre-registered dev/test split of primary events, stratified by origin kind.

    Within each stratum, ids are sorted, shuffled with `seed`, and the first
    DEV_COUNTS[stratum] go to dev. Deterministic, so anyone can recompute it.
    """
    rng = random.Random(seed)
    split: dict[str, str] = {}
    for stratum in sorted(DEV_COUNTS):
        ids = sorted(e.id for e in events if e.bucket == "primary" and e.origin_kind == stratum)
        rng.shuffle(ids)
        for i, eid in enumerate(ids):
            split[eid] = "dev" if i < DEV_COUNTS[stratum] else "test"
    return split


def load_events(path: Path = CONFIG_DIR / "events.yaml", universe: Universe | None = None) -> list[Event]:
    with path.open() as f:
        return parse_events(yaml.safe_load(f), universe or load_universe())
