"""EDGAR full-text search (filings from 2001 onwards).

Used for cheap corpus scouting, such as counting how many of a company's filings
mention another company, before anything is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass

from ripplekg.download.client import EdgarClient

FTS_URL = "https://efts.sec.gov/LATEST/search-index"


@dataclass(frozen=True)
class HitCount:
    value: int
    exact: bool  # False when EDGAR caps the count and reports a lower bound


def build_params(
    query: str,
    ciks: list[int] | None = None,
    forms: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, str]:
    params = {"q": query}
    if ciks:
        params["ciks"] = ",".join(f"{c:010d}" for c in ciks)
    if forms:
        params["forms"] = ",".join(forms)
    if start or end:
        params["dateRange"] = "custom"
        if start:
            params["startdt"] = start
        if end:
            params["enddt"] = end
    return params


def count_hits(
    client: EdgarClient,
    query: str,
    ciks: list[int] | None = None,
    forms: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> HitCount:
    """Number of documents matching `query`. Quote a phrase to match it exactly."""
    data = client.get_json(FTS_URL, params=build_params(query, ciks, forms, start, end))
    total = data["hits"]["total"]
    return HitCount(value=int(total["value"]), exact=total.get("relation", "eq") == "eq")
