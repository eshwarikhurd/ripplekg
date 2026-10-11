import pandas as pd

from ripplekg import coverage
from ripplekg.universe import parse_universe


def universe(n_filers=3):
    rows = [
        {
            "id": f"F{i}",
            "name": f"F{i}",
            "tier": "automaker",
            "cik": i + 1,
            "ticker": f"F{i}",
            "search_names": [f"F{i}"],
        }
        for i in range(n_filers)
    ]
    rows.append(
        {"id": "NF", "name": "NonFiler", "tier": "cells_storage", "search_names": ["NonFiler", "NF Ltd"]}
    )
    return parse_universe({"companies": rows})


def filers_df(ids, annual=1):
    return pd.DataFrame({"id": ids, "n_annual": [annual] * len(ids)})


def test_pair_table_takes_max_over_search_names():
    mentions = pd.DataFrame(
        {
            "filer": ["F0", "F0", "F1"],
            "target": ["NF", "NF", "F0"],
            "name": ["NonFiler", "NF Ltd", "F0"],
            "docs": [3, 7, 1],
            "exact": [True] * 3,
        }
    )
    pairs = coverage.pair_table(mentions).set_index(["filer", "target"])["docs"].to_dict()
    assert pairs == {("F0", "NF"): 7, ("F1", "F0"): 1}


def test_summarise_counts_undirected_pairs_and_isolated_filers():
    u = universe(3)
    pairs = pd.DataFrame(
        {
            "filer": ["F0", "F1", "F0", "F2"],
            "target": ["F1", "F0", "NF", "F0"],
            "docs": [5, 4, 2, 1],  # F2->F0 is below MIN_DOCS_PER_PAIR
        }
    )
    s = coverage.summarise(u, filers_df(["F0", "F1", "F2"]), pairs)
    assert s.n_directed_pairs == 3
    assert s.n_undirected_pairs == 2  # {F0,F1} counted once, plus {F0,NF}
    assert s.scored_ids == ["F0", "F1"]
    assert s.isolated_ids == ["F2"]
    assert not s.go


def test_summarise_go_when_thresholds_met(monkeypatch):
    monkeypatch.setattr(coverage, "GO_MIN_SCORED", 2)
    monkeypatch.setattr(coverage, "GO_MIN_PAIRS", 1)
    u = universe(2)
    pairs = pd.DataFrame({"filer": ["F0"], "target": ["F1"], "docs": [9]})
    assert coverage.summarise(u, filers_df(["F0", "F1"]), pairs).go


def test_filer_without_annual_report_is_not_scored():
    u = universe(2)
    pairs = pd.DataFrame({"filer": ["F0"], "target": ["F1"], "docs": [9]})
    filers = pd.DataFrame({"id": ["F0", "F1"], "n_annual": [1, 0]})
    assert coverage.summarise(u, filers, pairs).scored_ids == ["F0"]
