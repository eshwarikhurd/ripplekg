import pytest

from ripplekg.ontology.curation_review import ReviewStore, entries_table, entry_id

DRAFT = {
    "aliases": [
        {"key": "piedmont", "node": "PLL", "seen": 17, "confidence": 0.6, "approve": True, "issues": []}
    ],
    "new_nodes": [
        {
            "key": "atlantic lithium",
            "kind": "company",
            "label": "Atlantic Lithium",
            "seen": 32,
            "confidence": 0.85,
            "approve": True,
            "issues": [],
            "review_note": "checked",
        }
    ],
    "skip": [{"key": "our suppliers", "name": "our suppliers", "seen": 9, "approve": True}],
    "needs_review": [],
}


def test_entries_table_flattens_sections():
    t = entries_table(DRAFT).set_index("id")
    assert list(t.index) == ["aliases::piedmont", "new_nodes::atlantic lithium", "skip::our suppliers"]
    assert t.loc["aliases::piedmont", "target"] == "PLL"
    assert bool(t.loc["new_nodes::atlantic lithium", "corrected"])


def test_latest_decision_wins_and_edits_are_validated(tmp_path):
    store = ReviewStore(tmp_path / "round1_review.jsonl")
    store.add("new_nodes", "atlantic lithium", "reject", note="not sure")
    store.add("new_nodes", "atlantic lithium", "edit", {"label": "Atlantic Lithium Ltd"})
    latest = store.latest()[entry_id("new_nodes", "atlantic lithium")]
    assert latest.decision == "edit" and latest.edits == {"label": "Atlantic Lithium Ltd"}
    with pytest.raises(ValueError, match="cannot be edited"):
        store.add("aliases", "piedmont", "edit", {"label": "x"})
    with pytest.raises(ValueError, match="decision must be"):
        store.add("skip", "our suppliers", "maybe")


def test_bulk_decisions(tmp_path):
    store = ReviewStore(tmp_path / "r.jsonl")
    assert store.add_many([("skip", "a"), ("skip", "b")], "approve", note="bulk") == 2
    assert {d.decision for d in store.latest().values()} == {"approve"}


def test_undecided_entries_resolve_as_alias_or_skip(tmp_path):
    store = ReviewStore(tmp_path / "r.jsonl")
    store.add("needs_review", "agp", "edit", {"action": "skip"})
    assert store.latest()[entry_id("needs_review", "agp")].edits == {"action": "skip"}
