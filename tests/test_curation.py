from collections import Counter

import pandas as pd

from ripplekg.ontology.curation import (
    Group,
    build_draft,
    collect_groups,
    fuzzy_suggestions,
    is_generic,
    variant_key,
)
from ripplekg.ontology.schema import load_ontology

ONT = load_ontology()


def test_variant_key_groups_spelling_variants():
    assert (
        variant_key("Cauchari-Olaroz Project") == variant_key("Caucharí-Olaroz project") == "cauchari olaroz"
    )
    assert variant_key("Millennial Lithium Corp.") == variant_key("Millennial Lithium")
    assert variant_key("Lanxess AG") == variant_key("LANXESS")
    assert variant_key("Minera Exar S.A.") == variant_key("Minera Exar")


def test_generic_phrases():
    assert (
        is_generic("our suppliers")
        and is_generic("lithium production facilities")
        and is_generic("Mill") is False
    )
    assert not is_generic("Atlantic Lithium") and not is_generic("Kellyton Graphite Plant")


def triple(name, end="target", rel="SUPPLIES", filer="PLL", quote="Q."):
    reason = f"unresolved {end}"
    src, tgt = (name, "TSLA") if end == "source" else ("PLL", name)
    return {"reason": reason, "canonical_relation": rel, "relation": rel, "source": src, "target": tgt,
            "filer": filer, "quote_full": quote, "quote": quote}  # fmt: skip


def test_collect_groups_counts_variants_and_kinds():
    rows = [triple("Atlantic Lithium")] * 3 + [triple("Atlantic Lithium Limited")] + [triple("Once Only")]
    groups = collect_groups(pd.DataFrame(rows), ONT, min_count=3)
    assert len(groups) == 1
    g = groups[0]
    assert g.count == 4 and g.label == "Atlantic Lithium" and g.kinds["company"] == 4


def test_customer_of_names_come_from_the_written_side():
    row = {
        **triple("Acme Cells", end="source"),
        "relation": "CUSTOMER_OF",
        "source": "TSLA",
        "target": "Acme Cells",
    }
    (g,) = collect_groups(pd.DataFrame([row] * 3), ONT, min_count=3)
    assert g.label == "Acme Cells"


def test_fuzzy_only_suggests_close_names():
    g = Group("piedmont lithium", Counter({"Piedmont Lithium Inc.": 5}), Counter({"company": 5}))
    assert fuzzy_suggestions([g], ONT)["piedmont lithium"][0] == "PLL"


def group(label, count, filers=("PLL",)):
    return Group(variant_key(label), Counter({label: count}), Counter({"company": count}), Counter(filers))


def decision(action, **kw):
    base = {"action": action, "existing": "", "kind": "", "label": "", "parent": "", "region": "",
            "battery_relevant": True, "reason": "r", "confidence": 0.9}  # fmt: skip
    return {**base, **kw}


def test_build_draft_sections_and_scope_rule():
    groups = [group("our suppliers", 9), group("Piedmont", 12), group("Atlantic Lithium", 29),
              group("Samarco", 25, ("XYZ",)), group("Iron Ore Co", 25)]  # fmt: skip
    decisions = {
        variant_key("Piedmont"): decision("alias", existing="PLL"),
        variant_key("Atlantic Lithium"): decision("new", kind="company", label="Atlantic Lithium"),
        variant_key("Samarco"): decision("new", kind="company", label="Samarco", battery_relevant=False),
        variant_key("Iron Ore Co"): decision(
            "new", kind="company", label="Iron Ore Co", battery_relevant=False, confidence=0.5
        ),  # fmt: skip
    }
    d = build_draft(groups, {}, decisions, ONT)
    assert [s["name"] for s in d["skip"]][:2] == ["our suppliers", "Samarco"]  # generic rule, then off-topic
    assert d["aliases"][0]["node"] == "PLL"
    new = {n["label"]: n for n in d["new_nodes"]}
    assert new["Atlantic Lithium"]["listing"] == "to verify" and new["Atlantic Lithium"]["approve"] is True
    assert new["Iron Ore Co"]["approve"] is False  # off-topic but frequent and linked: kept, low confidence


def test_validation_flags_kind_mismatch_unknown_parent_and_duplicates():
    groups = [group("Lithium Nevada", 10), group("LN Holdco Two", 4), group("Arizona Spoke", 8)]
    groups[2].kinds = Counter({"facility": 8})
    decisions = {
        variant_key("Lithium Nevada"): decision("alias", existing="Thacker Pass"),
        variant_key("LN Holdco Two"): decision("new", kind="company", label="LN Holdco Two", parent="NOPE"),
        variant_key("Arizona Spoke"): decision(
            "new", kind="facility", label="Arizona Spoke", parent="LICY", region=""
        ),
    }  # fmt: skip
    d = build_draft(groups, {}, decisions, ONT)
    alias = d["aliases"][0]
    assert "kind mismatch" in alias["issues"][0] and alias["approve"] is False
    new = {n["label"]: n for n in d["new_nodes"]}
    assert any("parent 'NOPE'" in i for i in new["LN Holdco Two"]["issues"])
    assert any("not a known region" in i for i in new["Arizona Spoke"]["issues"])
    assert all(n["approve"] is False for n in d["new_nodes"])


def test_duplicate_new_nodes_are_merged():
    groups = [group("Atlantic Lithium", 20), group("Atlantic Lithium Plc Group", 5)]
    decisions = {g.key: decision("new", kind="company", label="Atlantic Lithium") for g in groups}
    (n,) = build_draft(groups, {}, decisions, ONT)["new_nodes"]
    assert n["seen"] == 25 and len(n["merged_from"]) == 1


def test_parent_and_region_names_resolve_to_node_labels():
    groups = [group("Kennecott", 30)]
    groups[0].kinds = Counter({"facility": 30})
    decisions = {
        groups[0].key: decision(
            "new", kind="facility", label="Kennecott", parent="Rio Tinto", region="Nevada"
        )
    }
    (n,) = build_draft(groups, {}, decisions, ONT)["new_nodes"]
    assert (n["parent"], n["region"], n["issues"]) == ("RIO", "Nevada", [])
