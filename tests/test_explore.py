import re

import pandas as pd

from ripplekg.explore.data import aggregate_edges, entity_table, triple_key, unresolved_names
from ripplekg.explore.network import neighbourhood, network_html
from ripplekg.explore.reviews import ReviewStore
from ripplekg.ontology.schema import load_ontology

ONT = load_ontology()


def t(
    src,
    rel,
    tgt,
    ok=True,
    reason="ok",
    doc="edgar:1",
    date="2024-01-29T21:00:00+00:00",
    raw_src=None,
    raw_tgt=None,
):
    return {
        "chunk_id": "c1", "doc_id": doc, "filer": "TSLA", "status": "active", "published_at": date,
        "source": raw_src or src, "target": raw_tgt or tgt, "relation": rel, "quote": "q",
        "source_ref": src if ok else None, "target_ref": tgt if ok else None,
        "canonical_relation": rel, "ok": ok, "reason": reason, "triple_key": f"{src}{rel}{tgt}{doc}",
    }  # fmt: skip


TRIPLES = pd.DataFrame(
    [
        t("PANASONIC", "SUPPLIES", "TSLA", raw_src="Panasonic"),
        t(
            "PANASONIC",
            "SUPPLIES",
            "TSLA",
            doc="edgar:2",
            date="2025-01-30T21:00:00+00:00",
            raw_src="Panasonic Holdings",
        ),
        t("TSLA", "CONSUMES", "material:lithium"),
        t("Acme Widgets", "SUPPLIES", "TSLA", ok=False, reason="unresolved source"),
        t("Acme Widgets", "SUPPLIES", "GM", ok=False, reason="unresolved source"),
    ]
)  # fmt: skip


def test_aggregate_merges_triples_into_edges_with_history():
    edges = aggregate_edges(TRIPLES).set_index(["source_ref", "canonical_relation", "target_ref"])
    e = edges.loc[("PANASONIC", "SUPPLIES", "TSLA")]
    assert (e["triples"], e["filings"], e["first_seen"], e["last_seen"]) == (2, 2, "2024-01-29", "2025-01-30")
    assert len(edges) == 2  # failed triples are not edges


def test_entity_table_lists_name_variants_and_degree():
    ents = entity_table(TRIPLES, ONT).set_index("node")
    assert ents.loc["PANASONIC", "kind"] == "company"
    assert "Panasonic Holdings (1)" in ents.loc["PANASONIC", "name_variants"]
    assert ents.loc["TSLA", "edges"] == 2


def test_unresolved_names_form_the_curation_queue():
    q = unresolved_names(TRIPLES)
    assert q.iloc[0]["name"] == "Acme Widgets" and q.iloc[0]["count"] == 2


def test_triple_key_is_stable_and_specific():
    a = {"chunk_id": "c1", "source": "A", "relation": "R", "target": "B", "quote": "q"}
    assert triple_key(a) == triple_key(dict(a)) and triple_key(a) != triple_key({**a, "target": "C"})


def test_review_store_keeps_history_and_latest_wins(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.add("k1", "v1-annual", "N", "wrong direction")
    store.add("k1", "v1-annual", "Y", "")
    store.add("k2", "v1-annual", "?", "")
    assert len(store.all()) == 3
    latest = store.latest().set_index("triple_key")
    assert latest.loc["k1", "label"] == "Y" and latest.loc["k2", "label"] == "?"


def test_network_neighbourhood_and_html():
    edges = aggregate_edges(TRIPLES)
    assert set(neighbourhood(edges, "material:lithium", 1)["source_ref"]) == {"TSLA"}
    assert len(neighbourhood(edges, "material:lithium", 2)) == 2
    html = network_html(edges, ONT, focus="TSLA")
    assert "PANASONIC" in html and "new vis.Network" in html
    loads = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert not [u for u in loads if u.startswith(("http", "../node_modules"))]
