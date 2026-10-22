"""Deduplication of verified edges within a chunk.

Several relation-family calls see the same chunk, and a model can repeat itself, so one fact
can arrive as several edges. Within a chunk:

1. Exact duplicates of (source, relation, target), after symmetric relations are put in canonical
   order, are kept once.
2. Near-synonyms: for the same pair of nodes (in either direction) supported by the same quote,
   only the most specific relation of each exclusive group in config/relations.yaml is kept; the
   others are marked as superseded. Different sentences may state different facts about a pair
   (a stake, then an acquisition), so they are not merged.

Across chunks nothing is dropped here: repeated sightings of a fact in other chunks and filings
are its evidence, and the graph builder aggregates them into one edge with a history.
"""

from __future__ import annotations

import pandas as pd

from ripplekg.ontology.schema import Ontology


def dedupe_edges(edges: pd.DataFrame, ont: Ontology) -> pd.DataFrame:
    """Returns `edges` with `ok` cleared and `reason` set on dropped rows."""
    if edges.empty:
        return edges
    df = edges.copy()
    ok = df["ok"].astype(bool)

    key = ["chunk_id", "source_ref", "canonical_relation", "target_ref"]
    dup = df.index[ok][df[ok].duplicated(key, keep="first").to_numpy()]
    df.loc[dup, ["ok", "reason"]] = [False, "duplicate in chunk"]

    live = df[df["ok"].astype(bool)]
    if live.empty:
        return df
    pair = live.apply(lambda r: tuple(sorted((r["source_ref"], r["target_ref"]))), axis=1)
    for _key, group in live.groupby([live["chunk_id"], pair, live["quote_full"]]):
        present = set(group["canonical_relation"])
        for name, members in ont.exclusive_groups.items():
            found = [m for m in members if m in present]
            if len(found) < 2:
                continue
            winner = found[0]
            losers = group.index[group["canonical_relation"].isin(found[1:])]
            df.loc[losers, ["ok", "reason"]] = [False, f"superseded by {winner} ({name})"]
    return df
