"""Week-1 coverage check: is the universe dense enough to build a useful graph?

Uses EDGAR full-text search only (no downloads):

1. Filing counts per filer in the corpus window.
2. Co-mentions: in how many of filer A's documents does company B's name appear?
   This is an upper-bound proxy for A->B edges. It also counts competitor mentions
   and boilerplate, so the real edge count comes later, from extraction.
3. Supply-language phrases per filer (supply agreements, offtake, sole source...).

The go/no-go thresholds come from the plan (section 5).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ripplekg.download.client import EdgarClient
from ripplekg.download.fulltext import count_hits
from ripplekg.download.index import ANNUAL_FORMS, company_submissions, list_filings
from ripplekg.ontology.universe import Universe

SEARCH_FORMS = ["10-K", "10-Q", "8-K", "20-F", "40-F", "6-K"]

SUPPLY_PHRASES = [
    "supply agreement",
    "offtake",
    "sole source",
    "single source",
    "joint venture",
    "of our revenue",
]

MIN_DOCS_PER_PAIR = 2
GO_MIN_SCORED = 30
GO_MIN_PAIRS = 150

Progress = Callable[[str], None]


def check_filers(client: EdgarClient, universe: Universe, start: str, end: str) -> pd.DataFrame:
    """One row per filer: SEC's name for the CIK and filing counts in [start, end]."""
    rows = []
    for c in universe.filers:
        sub = company_submissions(client, c.cik)
        filings = list_filings(client, c.cik, start, end)
        forms = Counter(f.form for f in filings)
        annual = [f for f in filings if f.form in ANNUAL_FORMS]
        rows.append(
            {
                "id": c.id,
                "cik": c.cik,
                "sec_name": sub.get("name", ""),
                "sec_tickers": ",".join(sub.get("tickers", [])),
                "tier": c.tier,
                "annual_form": annual[0].form if annual else "",
                "n_annual": len(annual),
                "n_10q_6k": forms["10-Q"] + forms["6-K"],
                "n_8k": forms["8-K"],
                "first_annual": annual[0].filing_date if annual else "",
                "last_annual": annual[-1].filing_date if annual else "",
            }
        )
    return pd.DataFrame(rows)


def co_mentions(
    client: EdgarClient,
    universe: Universe,
    start: str,
    end: str,
    progress: Progress = lambda _: None,
) -> pd.DataFrame:
    """Long table: filer, target, search name, matching document count."""
    rows = []
    targets = [(t, name) for t in universe.companies for name in t.search_names]
    total = len(universe.filers) * len(targets)
    done = 0
    for filer in universe.filers:
        for target, name in targets:
            done += 1
            if target.id == filer.id:
                continue
            hits = count_hits(client, f'"{name}"', ciks=[filer.cik], forms=SEARCH_FORMS, start=start, end=end)
            rows.append(
                {
                    "filer": filer.id,
                    "target": target.id,
                    "name": name,
                    "docs": hits.value,
                    "exact": hits.exact,
                }
            )
            if done % 200 == 0:
                progress(f"co-mentions: {done}/{total} queries")
    return pd.DataFrame(rows)


def phrase_counts(client: EdgarClient, universe: Universe, start: str, end: str) -> pd.DataFrame:
    rows = []
    for filer in universe.filers:
        row = {"id": filer.id}
        for phrase in SUPPLY_PHRASES:
            hits = count_hits(
                client, f'"{phrase}"', ciks=[filer.cik], forms=SEARCH_FORMS, start=start, end=end
            )
            row[phrase] = hits.value
        rows.append(row)
    return pd.DataFrame(rows)


def pair_table(mentions: pd.DataFrame) -> pd.DataFrame:
    """Directed filer->target pairs, taking the best-matching search name for each target.

    Different names for one target can match overlapping documents, so the max is a
    lower bound on the true count, not a sum.
    """
    if mentions.empty:
        return pd.DataFrame(columns=["filer", "target", "docs"])
    return mentions.groupby(["filer", "target"], as_index=False)["docs"].max()


@dataclass
class Summary:
    n_filers: int
    n_filers_with_annual: int
    n_scored: int
    n_directed_pairs: int
    n_undirected_pairs: int
    go: bool
    scored_ids: list[str]
    isolated_ids: list[str]


def summarise(universe: Universe, filers: pd.DataFrame, pairs: pd.DataFrame) -> Summary:
    strong = pairs[pairs["docs"] >= MIN_DOCS_PER_PAIR]
    undirected = {tuple(sorted(p)) for p in zip(strong["filer"], strong["target"], strict=True)}
    connected = set(strong["filer"]) | set(strong["target"])
    with_annual = set(filers.loc[filers["n_annual"] > 0, "id"])
    listed = {c.id for c in universe.filers if c.ticker}
    scored = sorted(with_annual & listed & connected)
    isolated = sorted(with_annual - connected)
    return Summary(
        n_filers=len(filers),
        n_filers_with_annual=len(with_annual),
        n_scored=len(scored),
        n_directed_pairs=len(strong),
        n_undirected_pairs=len(undirected),
        go=len(scored) >= GO_MIN_SCORED and len(undirected) >= GO_MIN_PAIRS,
        scored_ids=scored,
        isolated_ids=isolated,
    )


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    universe: Universe,
    filers: pd.DataFrame,
    pairs: pd.DataFrame,
    phrases: pd.DataFrame,
    summary: Summary,
    start: str,
    end: str,
) -> None:
    strong = pairs[pairs["docs"] >= MIN_DOCS_PER_PAIR]
    degree = (
        pd.concat([strong["filer"], strong["target"]])
        .value_counts()
        .rename_axis("id")
        .reset_index(name="degree")
    )
    top_pairs = strong.sort_values("docs", ascending=False).head(25)
    non_filer_ids = [c.id for c in universe.companies if not c.is_filer]
    non_filer_reach = (
        strong[strong["target"].isin(non_filer_ids)]
        .groupby("target")
        .agg(mentioned_by=("filer", "nunique"), docs=("docs", "sum"))
        .sort_values("mentioned_by", ascending=False)
        .reset_index()
    )
    per_filer = (
        filers[["id", "tier", "annual_form", "n_annual", "n_10q_6k", "n_8k"]]
        .merge(degree, on="id", how="left")
        .merge(phrases, on="id", how="left")
        .fillna({"degree": 0})
        .astype({"degree": int})
        .sort_values(["tier", "id"])
    )

    verdict = "GO" if summary.go else "NO-GO"
    text = f"""# Coverage check

Generated by `ripplekg coverage` from EDGAR full-text search, for filings dated {start} to {end}
(forms: {", ".join(SEARCH_FORMS)}).

## Verdict: {verdict}

| Check | Value | Threshold |
|---|---|---|
| Filers in universe | {summary.n_filers} | |
| Filers with an annual report in the window | {summary.n_filers_with_annual} | |
| Scorable companies¹ | {summary.n_scored} | ≥ {GO_MIN_SCORED} |
| Co-mentioned company pairs² | {summary.n_undirected_pairs} | ≥ {GO_MIN_PAIRS} |
| Directed filer → company pairs | {summary.n_directed_pairs} | |

¹ Annual report in the window, US listing, and at least one co-mention pair.
² Undirected pairs where one company's name appears in at least {MIN_DOCS_PER_PAIR} of the other's documents.

Filers with no co-mention pair: {", ".join(summary.isolated_ids) or "none"}.

**How to read this.** A co-mention means company B's name appears in a document filed by company A. It is
an upper-bound proxy for a supply-chain edge: it also counts competitor lists, industry commentary and
boilerplate. The real edge count comes later, from extraction. Search-name choices matter: a name that
is too generic inflates counts, one that is too narrow misses them.

## Per filer

`degree` = number of other universe companies linked by a co-mention pair (either direction).
Phrase columns = number of the filer's documents containing that exact phrase.

{_md_table(per_filer)}

## Non-filers reached through filers' documents

{_md_table(non_filer_reach)}

## Strongest co-mention pairs

{_md_table(top_pairs)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
