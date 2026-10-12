"""Relation extraction: one LLM call per (chunk, relation family).

The prompt for a family is generated from the ontology (relation definitions, endpoint kinds,
allowed strength bases) and the hand-written examples for that family, so it never drifts from
config. The output is constrained by a JSON schema whose relation and status fields are enums.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ripplekg.llm import LLMProvider, LLMResult, ResponseCache, cached_complete
from ripplekg.ontology import Example, Ontology

PROMPT_VERSION = "v1"
MAX_QUOTE_WORDS = 40

FAMILY_TITLES = {
    "trade": "trade links between companies",
    "ownership_capital": "ownership, joint ventures, investments, financing and acquisitions",
    "footprint": "facilities, locations, and materials or components produced or consumed",
    "competition": "named competitors",
    "exposure": "exposure to regions, trade restrictions, government programs, regulators and hazards",
    "disputes": "litigation, arbitration, recalls and penalties",
}

STATUS_TEXT = """- active: stated as currently true or as having happened.
- planned: agreed, under construction or expected, but not yet in effect.
- ceased: the text says a relationship that existed has ended.
- negated: the text says it does not exist, or that an attempt failed."""


@dataclass
class FamilyPrompt:
    family: str
    system: str
    schema: dict


def _relations(ont: Ontology, family: str) -> list[str]:
    return [r.name for r in ont.relations.values() if r.family == family]


def _labels(ont: Ontology, family: str) -> list[str]:
    labels = _relations(ont, family)
    labels += [a for a, (rel, _) in ont.relation_aliases.items() if rel in labels]
    return labels


def build_schema(ont: Ontology, family: str) -> dict:
    rels = [ont.relations[r] for r in _relations(ont, family)]
    bases = sorted({b for r in rels for b in r.strength_bases}) + ["none"]
    edge = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "relation": {"type": "string", "enum": _labels(ont, family)},
            "target": {"type": "string"},
            "status": {"type": "string", "enum": list(ont.statuses)},
            "quote": {"type": "string"},
            "strength": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "strength_basis": {"type": "string", "enum": bases},
            "material": {"type": "string"},
            "component": {"type": "string"},
            "hazard": {"type": "string"},
        },
        "required": [
            "source",
            "relation",
            "target",
            "status",
            "quote",
            "strength",
            "strength_basis",
            "material",
            "component",
            "hazard",
        ],  # fmt: skip
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"edges": {"type": "array", "items": edge}},
        "required": ["edges"],
        "additionalProperties": False,
    }


def _example_block(ex: Example) -> str:
    edges = [
        {
            "source": e["source"],
            "relation": e["relation"] if ex.relation != "CUSTOMER_OF" else ex.relation,
            "target": e["target"],
            "status": e["status"],
            "quote": e["quote"],
            "strength": e.get("strength"),
            "strength_basis": e.get("strength_basis", "none"),
            "material": e.get("material", ""),
            "component": e.get("component", ""),
            "hazard": e.get("hazard", ""),
        }
        for e in ex.expect
    ]
    if ex.relation == "CUSTOMER_OF":  # show the extractor the label as written, reversed back
        for d in edges:
            d["source"], d["target"] = d["target"], d["source"]
    return f"Filer: {ex.filer}\n<excerpt>\n{ex.text}\n</excerpt>\nOutput: {json.dumps({'edges': edges})}"


def build_prompt(ont: Ontology, family: str, examples: list[Example]) -> FamilyPrompt:
    rels = [ont.relations[r] for r in _relations(ont, family)]
    rel_lines = []
    for r in rels:
        kinds = f"{' or '.join(sorted(r.source))} → {' or '.join(sorted(r.target))}"
        rel_lines.append(f"- {r.name} ({kinds}): {r.description}")
    for alias, (rel, reverse) in ont.relation_aliases.items():
        if rel in {r.name for r in rels}:
            note = " with source and target swapped" if reverse else ""
            rel_lines.append(f"- {alias}: the same fact as {rel}{note}; use whichever matches the wording.")

    attrs = sorted({a for r in rels for a in r.attributes})
    attr_text = ""
    if "material" in attrs or "component" in attrs:
        mats = ", ".join(n.id for n in ont.nodes.values() if n.kind == "material")
        comps = ", ".join(n.id for n in ont.nodes.values() if n.kind == "component")
        attr_text = (
            '\nmaterial / component: what flows along the edge, as one of these ids, or "" if none fits.\n'
            f"Materials: {mats}\nComponents: {comps}\n"
        )
    if "hazard" in attrs:
        attr_text += 'hazard: the natural hazard named (for example "flood", "earthquake"), or "".\n'

    family_examples = [
        ex for ex in examples if ont.canonical_relation(ex.relation)[0] in {r.name for r in rels}
    ]
    shots = "\n\n".join(_example_block(ex) for ex in family_examples)

    system = f"""You extract relationships from SEC filings for a knowledge graph of the electric-vehicle
battery and critical-minerals supply chain. This request covers one family of relationships:
{FAMILY_TITLES[family]}.

You get one excerpt and the name of the company that filed it. Return every relationship of the
types below that the excerpt states explicitly. Most excerpts contain none: then return an empty list.

Relationship types (source kind → target kind):
{chr(10).join(rel_lines)}

Rules:
- Only what the excerpt states. Never add facts from your own knowledge of these companies.
- "We", "our", "us" and "the Company" refer to the filer: write the filer's name.
- Name counterparties as written in the excerpt. Never output a group ("our suppliers", "certain
  customers", "automakers") as a source or target.
- quote: the shortest span from the excerpt that supports the relationship, at most {MAX_QUOTE_WORDS} words,
  copied character for character. Do not paraphrase, shorten with "...", or join separate sentences.
- status:
{STATUS_TEXT}
- strength: only if the quote states a percentage or share for this specific relationship, as a
  fraction (23% → 0.23), with its strength_basis. Otherwise strength is null and strength_basis "none".
{attr_text}
Examples (fictional companies):

{shots}"""
    return FamilyPrompt(family=family, system=system, schema=build_schema(ont, family))


def user_message(
    filer_name: str, filer_id: str, form: str, published_at: str, section: str, text: str
) -> str:
    return (
        f"Filer: {filer_name} ({filer_id})\n"
        f"Document: {form} published {published_at[:10]}, section {section}\n"
        f"<excerpt>\n{text}\n</excerpt>"
    )


@dataclass
class ExtractionOutcome:
    chunk_id: str
    family: str
    edges: list[dict]
    result: LLMResult | None
    error: str = ""


def extract(
    provider: LLMProvider,
    cache: ResponseCache,
    prompt: FamilyPrompt,
    chunk: dict,
    filer_name: str,
    max_tokens: int = 4096,
) -> ExtractionOutcome:
    user = user_message(
        filer_name, chunk["company_id"], chunk["form"], chunk["published_at"], chunk["section"], chunk["text"]
    )
    try:
        result = cached_complete(provider, cache, prompt.system, user, prompt.schema, max_tokens)
        edges = json.loads(result.text)["edges"]
    except Exception as exc:  # recorded per call; one bad call must not stop a run
        return ExtractionOutcome(chunk["chunk_id"], prompt.family, [], None, f"{type(exc).__name__}: {exc}")
    return ExtractionOutcome(chunk["chunk_id"], prompt.family, edges, result)
