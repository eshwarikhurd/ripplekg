import json
from types import SimpleNamespace

from ripplekg.extraction.llm import ClaudeProvider, ResponseCache
from ripplekg.extraction.prompts import build_prompt, build_schema, extract
from ripplekg.extraction.verify import Resolver, quote_in_text, strength_backed, verify_edge
from ripplekg.ontology.schema import load_examples, load_ontology

ONT = load_ontology()
EXAMPLES = load_examples()
RESOLVER = Resolver(ONT)


def test_schema_enums_follow_the_family():
    schema = build_schema(ONT, "trade")
    edge = schema["properties"]["edges"]["items"]
    assert "SUPPLIES" in edge["properties"]["relation"]["enum"]
    assert "CUSTOMER_OF" in edge["properties"]["relation"]["enum"]
    assert "OPERATES" not in edge["properties"]["relation"]["enum"]
    assert edge["properties"]["strength_basis"]["enum"][-1] == "none"
    assert edge["additionalProperties"] is False


def test_every_family_prompt_has_its_relations_and_examples():
    for family in ["trade", "ownership_capital", "footprint", "competition", "exposure", "disputes"]:
        p = build_prompt(ONT, family, EXAMPLES)
        rels = [r.name for r in ONT.relations.values() if r.family == family]
        assert all(f"- {r} (" in p.system for r in rels)
        assert "Output:" in p.system
    trade = build_prompt(ONT, "trade", EXAMPLES).system
    assert '"relation": "CUSTOMER_OF"' in trade  # alias example shown as written
    assert "lithium-hydroxide" in trade  # material ids listed for attribute fields


class FakeMessages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def create(self, **params):
        self.calls += 1
        assert params["output_config"]["format"]["type"] == "json_schema"
        return SimpleNamespace(
            stop_reason="end_turn",
            model=params["model"],
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(
                input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
        )


def test_extract_uses_cache_on_second_call(tmp_path):
    edge = {
        "source": "Tesla",
        "relation": "CUSTOMER_OF",
        "target": "Panasonic",
        "status": "active",
        "quote": "We purchase battery cells from Panasonic",
        "strength": None,
        "strength_basis": "none",
        "material": "",
        "component": "battery-cell",
        "hazard": "",
    }
    messages = FakeMessages({"edges": [edge]})
    provider = ClaudeProvider(client=SimpleNamespace(messages=messages))
    cache = ResponseCache(tmp_path)
    prompt = build_prompt(ONT, "trade", EXAMPLES)
    chunk = {
        "chunk_id": "c1",
        "company_id": "TSLA",
        "form": "10-K",
        "published_at": "2024-01-29T21:00:00+00:00",
        "section": "business",
        "text": "We purchase battery cells from Panasonic under a supply agreement.",
    }
    first = extract(provider, cache, prompt, chunk, "Tesla")
    second = extract(provider, cache, prompt, chunk, "Tesla")
    assert first.edges == second.edges == [edge]
    assert messages.calls == 1
    assert first.result.cost_usd > 0 and second.result.cost_usd == 0

    v = verify_edge(edge, chunk["text"], ONT, RESOLVER, filer="TSLA")
    assert v.ok and v.relation == "SUPPLIES"
    assert (v.source_ref, v.target_ref) == ("PANASONIC", "TSLA")  # CUSTOMER_OF reversed


def test_quote_check_tolerates_typography_only():
    text = "Tesla’s  supplier—Panasonic—ships cells."
    assert quote_in_text("Tesla's supplier-Panasonic-ships cells.", text)
    assert not quote_in_text("Tesla's supplier Panasonic ships cells.", text)
    assert not quote_in_text("", text)
    assert quote_in_text("Tesla\\u2019s  supplier", text)  # escape text written by the model


def test_strength_must_be_backed_by_quote():
    assert strength_backed(0.23, "accounted for 23% of revenue")
    assert strength_backed(0.099, "approximately 9.9% of our shares")
    assert not strength_backed(0.4, "accounted for 23% of revenue")
    assert strength_backed(1.0, "a wholly owned subsidiary")
    assert strength_backed(None, "anything")
    assert strength_backed(float("nan"), "anything")


def test_resolver_handles_legal_suffixes_and_kinds():
    assert RESOLVER.resolve("Albemarle Corporation", {"company"}) == "ALB"
    assert RESOLVER.resolve("Panasonic Holdings Corp.", {"company"}) == "PANASONIC"
    assert RESOLVER.resolve("lithium hydroxide", {"material", "component"}) == "material:lithium-hydroxide"
    assert RESOLVER.resolve("Nevada", {"region"}) == "region:US-NV"
    assert RESOLVER.resolve("our suppliers", {"company"}) is None
    assert RESOLVER.resolve("Albemarle (ALB)", {"company"}) == "ALB"
    assert RESOLVER.resolve("POSCO Holdings (PKX)", {"company"}) == "PKX"
    assert RESOLVER.resolve("Vale S.A. (Vale)", {"company"}) == "VALE"


def test_verify_rejects_with_reasons():
    text = "Sales to Tesla accounted for 23% of our revenue."
    base = {"source": "Panasonic", "relation": "SUPPLIES", "target": "Tesla", "status": "active"}
    assert verify_edge({**base, "quote": "Sales to Tesla made up a quarter"}, text, ONT, RESOLVER).reason == (
        "quote not verbatim"
    )
    bad_strength = {**base, "quote": "accounted for 23% of our revenue", "strength": 0.3}
    assert verify_edge(bad_strength, text, ONT, RESOLVER).reason == "strength not in quote"
    unresolved = {**base, "source": "Acme Widgets", "quote": "Sales to Tesla"}
    assert verify_edge(unresolved, text, ONT, RESOLVER).reason == "unresolved source"


def test_ollama_provider_sends_schema_and_reads_counts():
    from ripplekg.extraction.llm import OllamaProvider

    class Session:
        def post(self, url, json, timeout):
            assert url.endswith("/api/chat") and json["format"]["type"] == "object"
            assert json["options"]["temperature"] == 0
            body = {"message": {"content": '{"edges": []}'}, "done_reason": "stop",
                    "prompt_eval_count": 3000, "eval_count": 50}  # fmt: skip
            return SimpleNamespace(status_code=200, json=lambda: body, text="")

    r = OllamaProvider(session=Session()).complete_json("sys", "user", {"type": "object"})
    assert (r.text, r.input_tokens, r.output_tokens, r.cost_usd) == ('{"edges": []}', 3000, 50, 0.0)
