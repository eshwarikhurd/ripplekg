import pytest

from ripplekg.evaluation.events import load_events
from ripplekg.ontology.schema import OntologyError, build_ontology, load_ontology, normalize_name
from ripplekg.ontology.universe import load_universe

ONT = load_ontology()
UNIVERSE = load_universe()


def test_project_ontology_shape():
    assert len(ONT.relations) == 25
    families = {r.family for r in ONT.relations.values()}
    assert families == {"trade", "ownership_capital", "footprint", "competition", "exposure", "disputes"}
    assert ONT.canonical_relation("CUSTOMER_OF") == ("SUPPLIES", True)
    assert ONT.canonical_relation("OFFTAKE") == ("OFFTAKE", False)


def test_every_event_origin_resolves_to_a_node():
    for e in load_events():
        for ref in e.origin:
            assert ref in ONT.nodes, f"{e.id}: {ref}"


def test_edge_signatures():
    assert ONT.edge_error("SUPPLIES", "PANASONIC", "TSLA") is None
    assert ONT.edge_error("OPERATES", "LAC", "facility:thacker-pass") is None
    assert ONT.edge_error("CONSUMES", "TSLA", "material:lithium-hydroxide") is None
    assert ONT.edge_error("PRODUCES", "facility:mountain-pass", "material:ndpr") is None
    assert "target must be" in ONT.edge_error("SUPPLIES", "TSLA", "material:lithium")
    assert "source must be" in ONT.edge_error("LOCATED_IN", "TSLA", "region:US")
    assert ONT.edge_error("SUPPLIES", "TSLA", "TSLA") == "self-loop"
    assert "unknown target" in ONT.edge_error("SUPPLIES", "TSLA", "ACME")
    assert "unknown relation" in ONT.edge_error("LIKES", "TSLA", "GM")


def test_hierarchy_and_lookup():
    assert ONT.ancestors("material:lithium-hydroxide") == ["material:lithium"]
    assert ONT.ancestors("region:US-NV") == ["region:US"]
    assert ONT.lookup("material", "LiOH") == "material:lithium-hydroxide"
    assert ONT.lookup("region", "Québec") == "region:CA-QC"
    assert ONT.lookup("region", "DR Congo") == "region:CD"
    assert ONT.lookup("company", "Contemporary Amperex") == "CATL"
    assert ONT.lookup("material", "unobtainium") is None


def test_normalize_name():
    assert normalize_name("  Québec, Inc. ") == "quebec inc"
    assert normalize_name("Freeport-McMoRan") == "freeport mcmoran"


def raw_relations(**rel_overrides):
    rel = {
        "family": "trade",
        "source": ["company"],
        "target": ["company"],
        "channel": "both",
        "prior": "normal",
        "strength_bases": ["revenue_share"],
    }
    rel.update(rel_overrides)
    return {"relations": {"SUPPLIES": rel}, "strength_bases": {"revenue_share": "x"}}


def empty_nodes():
    return {"material": {"nodes": []}, "region": {"nodes": [{"id": "US", "name": "United States"}]}}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"family": "friendship"}, "unknown family"),
        ({"channel": "sideways"}, "unknown channel"),
        ({"prior": "huge"}, "unknown prior"),
        ({"target": ["planet"]}, "bad target kinds"),
        ({"strength_bases": ["vibes"]}, "undefined strength bases"),
        ({"symmetric": True, "target": ["material"]}, "symmetric relations need identical"),
    ],
)
def test_relation_validation(overrides, message):
    with pytest.raises(OntologyError, match=message):
        build_ontology(raw_relations(**overrides), empty_nodes(), UNIVERSE)


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            {"material": {"nodes": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}]}},
            "duplicate node material:a",
        ),
        ({"material": {"nodes": [{"id": "b", "name": "B", "parent": "zzz"}]}}, "parent 'zzz'"),
        (
            {
                "material": {
                    "nodes": [{"id": "a", "name": "Ore"}, {"id": "b", "name": "B", "aliases": ["ore"]}]
                }
            },
            "already used by material:a",
        ),
        (
            {"facility": {"nodes": [{"id": "f", "name": "F", "operators": ["NOPE"], "region": "US"}]}},
            "operator 'NOPE'",
        ),
    ],
)
def test_node_validation(nodes, message):
    nodes = {**empty_nodes(), **nodes}
    with pytest.raises(OntologyError, match=message):
        build_ontology(raw_relations(), nodes, UNIVERSE)


def test_yaml_boolean_id_is_rejected():
    nodes = {**empty_nodes(), "region": {"nodes": [{"id": False, "name": "Norway"}]}}
    with pytest.raises(OntologyError, match="must be strings"):
        build_ontology(raw_relations(), nodes, UNIVERSE)


def test_relation_examples_are_valid_and_cover_every_relation():
    from ripplekg.ontology.schema import example_errors, load_examples

    assert example_errors(ONT, load_examples()) == []


def test_example_errors_catch_paraphrased_quotes():
    from ripplekg.ontology.schema import Example, example_errors

    ex = Example(
        "SUPPLIES",
        "We buy cells from Acme.",
        "B",
        (
            {
                "source": "Acme",
                "relation": "SUPPLIES",
                "target": "B",
                "status": "active",
                "quote": "We purchase cells",
            },
        ),
    )
    assert any("not verbatim" in e for e in example_errors(ONT, [ex]))


def test_ontology_doc_is_up_to_date():
    from ripplekg.ontology.doc import render
    from ripplekg.ontology.schema import load_examples
    from ripplekg.settings import DOCS_DIR

    assert (DOCS_DIR / "ontology.md").read_text() == render(ONT, load_examples()), (
        "docs/ontology.md is stale: run `uv run ripplekg ontology-doc`"
    )


def test_supply_chain_relations_know_their_upstream_end():
    assert ONT.relations["SUPPLIES"].upstream == "source"
    assert ONT.relations["SOLE_SOURCES_FROM"].upstream == "target"  # buyer -> supplier
    assert ONT.relations["DISTRIBUTES_FOR"].upstream == "target"  # distributor -> maker
    assert ONT.relations["SUBSIDIARY_OF"].upstream is None
    with pytest.raises(OntologyError, match="need upstream"):
        build_ontology(raw_relations(channel="supply_chain"), empty_nodes(), UNIVERSE)
