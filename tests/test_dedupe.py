import pandas as pd

from ripplekg.extraction.dedupe import dedupe_edges
from ripplekg.ontology.schema import load_ontology

ONT = load_ontology()
Q = "Albemarle owns a 49% interest in Talison, which operates Greenbushes."


def row(rel, src, tgt, chunk="c1", quote=Q):
    return {"chunk_id": chunk, "source_ref": src, "canonical_relation": rel, "target_ref": tgt,
            "quote_full": quote, "ok": True, "reason": "ok"}  # fmt: skip


def test_exact_duplicates_in_a_chunk_are_kept_once():
    df = pd.DataFrame([row("PRODUCES", "ALB", "material:lithium")] * 3)
    out = dedupe_edges(df, ONT)
    assert out["ok"].tolist() == [True, False, False]
    assert out["reason"].tolist()[1] == "duplicate in chunk"


def test_same_fact_in_another_chunk_is_evidence_not_a_duplicate():
    df = pd.DataFrame(
        [row("PRODUCES", "ALB", "material:lithium"), row("PRODUCES", "ALB", "material:lithium", "c2")]
    )
    assert dedupe_edges(df, ONT)["ok"].all()


def test_most_specific_relation_wins_for_same_pair_and_quote():
    df = pd.DataFrame(
        [
            row("OPERATES", "ALB", "facility:greenbushes"),
            row("JV_PARTNER_IN", "ALB", "facility:greenbushes"),
            row("INVESTS_IN", "PLL", "ELVR"),
            row("SUBSIDIARY_OF", "PLL", "ELVR"),
        ]
    )
    out = dedupe_edges(df, ONT).set_index("canonical_relation")
    assert out.loc["JV_PARTNER_IN", "ok"] and out.loc["SUBSIDIARY_OF", "ok"]
    assert out.loc["OPERATES", "reason"] == "superseded by JV_PARTNER_IN (facility_role)"
    assert out.loc["INVESTS_IN", "reason"] == "superseded by SUBSIDIARY_OF (ownership)"


def test_different_quotes_can_state_different_facts():
    df = pd.DataFrame(
        [row("INVESTS_IN", "PLL", "ELVR", quote="A."), row("ACQUIRES", "PLL", "ELVR", quote="B.")]
    )
    assert dedupe_edges(df, ONT)["ok"].all()


def test_unrelated_relations_on_a_pair_both_stay():
    df = pd.DataFrame([row("SUPPLIES", "LAC", "GM"), row("INVESTS_IN", "GM", "LAC")])
    assert dedupe_edges(df, ONT)["ok"].all()


def test_rejected_row_does_not_hide_a_valid_one():
    bad = {**row("PRODUCES", "ALB", "material:lithium"), "ok": False, "reason": "endpoint not in quote"}
    out = dedupe_edges(pd.DataFrame([bad, row("PRODUCES", "ALB", "material:lithium")]), ONT)
    assert out["ok"].tolist() == [False, True]
