"""Node curation: turn frequent unresolved names into reviewed config changes.

Draft (`ripplekg curate draft`):
1. Collect names that failed to resolve, with counts, the node kind expected at that position, the
   filers that mentioned them, and example quotes.
2. Group spelling variants (case, accents, punctuation, legal suffixes, trailing words such as
   "project" or "mine").
3. Keep groups seen at least MIN_COUNT times.
4. Mark generic phrases ("production facilities", "our suppliers") as skip, without an LLM.
5. Suggest close matches to existing nodes with fuzzy scores. Suggestions only: fuzzy matching is
   never applied on its own, because near-identical names can be different companies
   ("Lithium Americas" and "Lithium Argentina").
6. Ask the LLM to propose alias / new node / skip for each remaining group.
7. Apply the scope rule: a new node must be in the battery chain, or appear at least
   OFF_TOPIC_MIN_COUNT times and be mentioned by one of our filers; otherwise it is skipped.

The draft is written to config/curation/<round>.yaml for human review. Tickers are never proposed
here: new public companies are listed for ticker verification after approval.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz, process

from ripplekg.extraction.llm import LLMProvider, ResponseCache, cached_complete
from ripplekg.ontology.schema import Ontology, normalize_name

MIN_COUNT = 3
OFF_TOPIC_MIN_COUNT = 20
FUZZY_SUGGEST = 85
BATCH = 25
CURATION_PROMPT_VERSION = "c1"

_LEGAL = frozenset(
    [
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "plc",
        "ag",
        "sa",
        "nv",
        "se",
        "gmbh",
        "pty",
        "ltda",
        "limitada",
        "spa",
        "the",
        "holdings",
        "group",
    ]
)
_LEGAL_PAIRS = [("s", "a"), ("n", "v")]  # "S.A." and "N.V." normalize to two words
_TRAILING = frozenset(
    [
        "project",
        "projects",
        "mine",
        "mines",
        "operation",
        "operations",
        "facility",
        "facilities",
        "plant",
        "property",
        "deposit",
        "site",
        "complex",
        "asset",
    ]
)
QUALIFIERS = frozenset(
    [
        "our",
        "its",
        "their",
        "certain",
        "other",
        "various",
        "major",
        "key",
        "large",
        "small",
        "new",
        "existing",
        "third",
        "party",
        "third-party",
        "local",
        "international",
        "global",
        "domestic",
        "foreign",
        "several",
        "some",
        "many",
        "all",
        "such",
        "these",
        "those",
        "principal",
        "primary",
        "significant",
        "potential",
        "future",
        "current",
        "battery",
        "ev",
        "lithium",
        "mining",
        "production",
        "manufacturing",
        "processing",
        "chinese",
        "us",
        "u.s.",
        "asian",
        "european",
        "a",
        "an",
        "the",
        "and",
        "of",
        "for",
        "in",
    ]
)
HEAD_NOUNS = frozenset(
    [
        "supplier",
        "suppliers",
        "customer",
        "customers",
        "facility",
        "facilities",
        "operation",
        "operations",
        "project",
        "projects",
        "plant",
        "plants",
        "mine",
        "mines",
        "company",
        "companies",
        "partner",
        "partners",
        "competitor",
        "competitors",
        "manufacturer",
        "manufacturers",
        "producer",
        "producers",
        "automaker",
        "automakers",
        "oem",
        "oems",
        "vendor",
        "vendors",
        "government",
        "governments",
        "authorities",
        "regulators",
        "agencies",
        "jurisdictions",
        "countries",
        "markets",
        "entities",
        "affiliates",
        "subsidiaries",
        "counterparties",
        "buyers",
        "sellers",
        "industry",
    ]
)


def variant_key(name: str) -> str:
    """Key under which spelling variants of one name group together."""
    words = normalize_name(name).split()
    for pair in _LEGAL_PAIRS:
        if tuple(words[-2:]) == pair:
            words = words[:-2]
    words = [w for w in words if w not in _LEGAL]
    while words and words[-1] in _TRAILING:
        words.pop()
    return " ".join(words)


def is_generic(name: str) -> bool:
    """A phrase made only of qualifier words and at least one generic noun ("our suppliers")."""
    words = normalize_name(name).split()
    if not words:
        return True
    has_head = any(w in HEAD_NOUNS for w in words)
    return has_head and all(w in HEAD_NOUNS or w in QUALIFIERS for w in words)


@dataclass
class Group:
    key: str
    variants: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    filers: Counter = field(default_factory=Counter)
    quotes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(self.variants.values())

    @property
    def label(self) -> str:
        return self.variants.most_common(1)[0][0]


def collect_groups(triples: pd.DataFrame, ont: Ontology, min_count: int = MIN_COUNT) -> list[Group]:
    """Unresolved names from re-verified triples, grouped by spelling variant."""
    groups: dict[str, Group] = {}
    for r in triples.itertuples(index=False):
        if r.reason not in ("unresolved source", "unresolved target"):
            continue
        end = "source" if r.reason == "unresolved source" else "target"
        rel = ont.relations.get(r.canonical_relation)
        # Verification reverses alias relations (CUSTOMER_OF -> SUPPLIES), so the canonical source
        # is the written target for those.
        reversed_ = r.relation != r.canonical_relation
        written_source, written_target = (r.target, r.source) if reversed_ else (r.source, r.target)
        name = written_source if end == "source" else written_target
        name = str(name).strip()
        if not name:
            continue
        key = variant_key(name) or normalize_name(name)
        g = groups.setdefault(key, Group(key))
        g.variants[name] += 1
        if rel is not None:
            for k in sorted(getattr(rel, end)):  # sorted: set order changes between runs
                g.kinds[k] += 1
        g.filers[r.filer] += 1
        q = r.quote_full or r.quote
        if q and len(g.quotes) < 2 and q not in g.quotes:
            g.quotes.append(str(q)[:300])
    return sorted((g for g in groups.values() if g.count >= min_count), key=lambda g: -g.count)


def fuzzy_suggestions(groups: list[Group], ont: Ontology) -> dict[str, tuple[str, float]]:
    """Closest existing node (of an expected kind) per group, if similar enough."""
    choices: dict[str, list[tuple[str, str]]] = {}
    for n in ont.nodes.values():
        for text in (n.name, *n.aliases, n.id):
            choices.setdefault(n.kind, []).append((normalize_name(text), n.ref))
    out = {}
    for g in groups:
        best: tuple[str, float] | None = None
        for kind in g.kinds:
            pool = choices.get(kind, [])
            if not pool:
                continue
            hit = process.extractOne(
                normalize_name(g.label), [c for c, _ in pool], scorer=fuzz.token_sort_ratio
            )
            if hit and hit[1] >= FUZZY_SUGGEST and (best is None or hit[1] > best[1]):
                best = (pool[hit[2]][1], float(hit[1]))
        if best:
            out[g.key] = best
    return out


TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group": {"type": "integer"},
                    "action": {"type": "string", "enum": ["alias", "new", "skip"]},
                    "existing": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "company",
                            "facility",
                            "material",
                            "component",
                            "region",
                            "authority",
                            "program",
                            "",
                        ],
                    },  # fmt: skip
                    "label": {"type": "string"},
                    "parent": {"type": "string"},
                    "region": {"type": "string"},
                    "battery_relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "group",
                    "action",
                    "existing",
                    "kind",
                    "label",
                    "parent",
                    "region",
                    "battery_relevant",
                    "reason",
                    "confidence",
                ],  # fmt: skip
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def triage_system(ont: Ontology) -> str:
    directory = "\n".join(
        f"{n.ref} [{n.kind}]" + (f" (parent {n.parent})" if n.parent else "")
        for n in sorted(ont.nodes.values(), key=lambda n: (n.kind, n.ref))
    )
    return f"""You help curate the node list of a knowledge graph of the electric-vehicle battery and
critical-minerals supply chain (lithium, nickel, cobalt, graphite, manganese, rare earths, cathode and
anode materials, battery cells, EVs, energy storage, recycling).

You get groups of names that an extractor found in SEC filings but that match no existing node. For
each group decide one action:
- alias: the group is another name for an existing node. Set "existing" to that node's label exactly
  as listed below.
- new: the group is a real, specific entity that should become a node. Set "kind", and "label" to its
  short canonical name (at most 4 words, no legal suffixes). For a company that is a subsidiary or joint
  venture of an existing company, set "parent" to that company's label. For a facility, set "parent" to
  the operating company's label and "region" to the existing region label it is in, if known.
- skip: a generic phrase, a description rather than a name, a person, a product line, or an entity
  outside the battery and critical-minerals chain that is not an important partner of a listed company.

Set "battery_relevant" true only if the entity is part of the battery or critical-minerals chain.
Use only what the names and quotes show plus widely known facts about the entity; if you are unsure,
choose skip or give a low confidence. Never invent stock tickers. "reason": at most 20 words.
Return one decision per group, in order. Leave unused string fields empty.

Existing nodes (label [kind]):
{directory}"""


def triage_user(batch: list[Group], suggestions: dict[str, tuple[str, float]]) -> str:
    lines = []
    for i, g in enumerate(batch):
        variants = "; ".join(f"{v} ({c})" for v, c in g.variants.most_common(5))
        kinds = ", ".join(k for k, _ in g.kinds.most_common())
        filers = ", ".join(f for f, _ in g.filers.most_common(3))
        lines.append(f"Group {i}: names: {variants} | expected kind: {kinds} | mentioned by: {filers}")
        if g.key in suggestions:
            ref, score = suggestions[g.key]
            lines.append(f"  similar existing node: {ref} (similarity {score:.0f})")
        for q in g.quotes:
            lines.append(f"  quote: {q}")
    return "\n".join(lines)


def triage(
    groups: list[Group],
    suggestions: dict[str, tuple[str, float]],
    ont: Ontology,
    provider: LLMProvider,
    cache: ResponseCache,
    known: dict[str, dict] | None = None,
) -> tuple[dict[str, dict], float]:
    """LLM proposals per group key, and the cost of this run. Groups already in `known` (decisions
    saved by an earlier draft) are not sent again, so a rebuilt draft keeps the same proposals."""
    system = triage_system(ont)
    out, cost = dict(known or {}), 0.0
    groups = [g for g in groups if g.key not in out]
    for start in range(0, len(groups), BATCH):
        batch = groups[start : start + BATCH]
        result = cached_complete(
            provider, cache, system, triage_user(batch, suggestions), TRIAGE_SCHEMA, 8192
        )
        cost += result.cost_usd
        for d in json.loads(result.text)["decisions"]:
            if 0 <= d["group"] < len(batch):
                out[batch[d["group"]].key] = d
    return out, cost


def apply_corrections(decisions: dict[str, dict], corrections: dict) -> dict[str, dict]:
    """Reviewer corrections override the model's decision for a group; each keeps its note.

    A correction for a group the model did not decide (for example a rule-based skip) creates
    the decision. `approve` defaults to true for corrected entries unless the correction says so.
    """
    out = {k: dict(v) for k, v in decisions.items()}
    base = {"action": "skip", "existing": "", "kind": "", "label": "", "parent": "", "region": "",
            "battery_relevant": True, "reason": "", "confidence": 1.0}  # fmt: skip
    for c in corrections.get("corrections", []):
        d = out.setdefault(c["key"], dict(base))
        d.update(c.get("set", {}))
        d["review_note"] = c["note"]
        d["review_approve"] = c.get("set", {}).get("approve", True)
    return out


def _slug(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", label.upper())[:12] or "NODE"


def build_draft(
    groups: list[Group],
    suggestions: dict[str, tuple[str, float]],
    decisions: dict[str, dict],
    ont: Ontology,
    added_nodes: list[dict] | None = None,
) -> dict:
    """The review file's content: aliases, new nodes and skips, each with its evidence."""
    universe_ids = {n.ref for n in ont.nodes.values() if n.kind == "company"}
    new_labels = {d["label"] for d in decisions.values() if d.get("action") == "new" and d.get("label")}
    new_labels |= {x["label"] for x in added_nodes or []}
    aliases, new_nodes, skips, review = [], [], [], []
    used_ids: set[str] = {n.id for n in ont.nodes.values()}
    for g in groups:
        evidence = {
            "key": g.key,
            "seen": g.count,
            "expected_kind": [k for k, _ in g.kinds.most_common()],
            "variants": [v for v, _ in g.variants.most_common(6)],
            "mentioned_by": [f for f, _ in g.filers.most_common(4)],
            "quotes": list(g.quotes),
        }
        d = decisions.get(g.key)
        if is_generic(g.label) and (d is None or "review_note" not in d):
            skips.append({"name": g.label, **evidence, "reason": "generic phrase (rule)", "approve": True})
            continue
        if d is None:
            review.append({"name": g.label, **evidence, "reason": "no LLM decision"})
            continue
        conf = round(float(d["confidence"]), 2)
        base = {**evidence, "confidence": conf, "reason": d["reason"], "approve": conf >= 0.8}
        if "review_note" in d:
            base["review_note"] = d["review_note"]
            base["approve"] = bool(d["review_approve"])
        if g.key in suggestions:
            base["fuzzy_match"] = f"{suggestions[g.key][0]} ({suggestions[g.key][1]:.0f})"
        if d["action"] == "alias":
            if d["existing"] in new_labels:
                aliases.append(
                    {"node": d["existing"], "node_is_new": True, "add_aliases": evidence["variants"], **base}
                )
                continue
            if d["existing"] not in ont.nodes:
                review.append({"name": g.label, **base, "reason": f"alias to unknown node {d['existing']!r}"})
                continue
            aliases.append({"node": d["existing"], "add_aliases": evidence["variants"], **base})
        elif d["action"] == "new":
            linked = any(f in universe_ids for f in g.filers)
            in_scope = d["battery_relevant"] or (g.count >= OFF_TOPIC_MIN_COUNT and linked)
            if not in_scope:
                skips.append({"name": g.label, **base, "reason": f"off-topic (scope rule): {d['reason']}"})
                continue
            node_id = _slug(d["label"])
            while node_id in used_ids:
                node_id += "X"
            used_ids.add(node_id)
            entry = {
                "kind": d["kind"],
                "id": node_id,
                "label": d["label"],
                "aliases": evidence["variants"],
                **base,
            }
            if d["parent"]:
                entry["parent"] = d["parent"]
            if d["kind"] == "facility":
                entry["region"] = d["region"] or "TO FILL"
            if d["kind"] == "company":
                entry["listing"] = "to verify"
            new_nodes.append(entry)
        else:
            skips.append({"name": g.label, **base})
    for extra in added_nodes or []:
        entry = {
            "kind": extra["kind"],
            "id": extra["id"],
            "label": extra["label"],
            "aliases": extra.get("aliases", []),
            "key": f"added:{extra['id']}",
            "seen": 0,
            "expected_kind": [extra["kind"]],
            "variants": [],
            "mentioned_by": [],
            "quotes": [],
            "confidence": 1.0,
            "reason": "added by reviewer",
            "review_note": extra["note"],
            "approve": True,
        }
        if extra.get("parent"):
            entry["parent"] = extra["parent"]
        new_nodes.append(entry)
    draft = {"aliases": aliases, "new_nodes": new_nodes, "skip": skips, "needs_review": review}
    return validate_draft(merge_duplicate_nodes(draft), ont)


def merge_duplicate_nodes(draft: dict) -> dict:
    """Several groups can propose the same new node ("Lithium Nevada" and "Lithium Nevada Corp." if
    grouping missed them). Merge them into one entry with all aliases and evidence."""
    merged: dict[tuple[str, str], dict] = {}
    for e in draft["new_nodes"]:
        key = (e["kind"], normalize_name(e["label"]))
        if key not in merged:
            merged[key] = e
            continue
        m = merged[key]
        m["aliases"] = list(dict.fromkeys(m["aliases"] + e["aliases"]))
        m["variants"] = list(dict.fromkeys(m["variants"] + e["variants"]))
        m["mentioned_by"] = list(dict.fromkeys(m["mentioned_by"] + e["mentioned_by"]))
        m["quotes"] = (m["quotes"] + e["quotes"])[:3]
        m["seen"] += e["seen"]
        m["confidence"] = max(m["confidence"], e["confidence"])
        m.setdefault("merged_from", []).append(e["key"])
    return {**draft, "new_nodes": list(merged.values())}


def validate_draft(draft: dict, ont: Ontology) -> dict:
    """Attach rule-based issues to entries; an entry with issues is never pre-approved."""
    existing = {normalize_name(ref): ref for ref in ont.nodes}
    new_labels = {normalize_name(e["label"]): e for e in draft["new_nodes"]}
    regions = {ref for ref, n in ont.nodes.items() if n.kind == "region"}

    def known(label: str) -> bool:
        return normalize_name(label) in existing or normalize_name(label) in new_labels

    for e in draft["aliases"]:
        issues = []
        if e.get("node_is_new") and normalize_name(e["node"]) not in new_labels:
            issues.append(f"alias target {e['node']!r} is not a proposed new node")
        node = ont.nodes.get(e["node"])
        if node is not None and e["expected_kind"] and node.kind not in e["expected_kind"]:
            issues.append(f"kind mismatch: {e['node']} is a {node.kind}, the name appears where a "
                          f"{'/'.join(e['expected_kind'])} is expected")  # fmt: skip
        e["issues"] = issues
    from ripplekg.extraction.verify import Resolver  # local import: verify depends on this package

    resolver = Resolver(ont)
    for e in draft["new_nodes"]:
        # The model may write a parent or region by name ("Rio Tinto"); store the node label (RIO).
        if e.get("parent"):
            kinds = {"company"} if e["kind"] in ("company", "facility") else {e["kind"]}
            ref = resolver.resolve(e["parent"], kinds)
            if ref and ref != e["parent"]:
                e["parent_written"], e["parent"] = e["parent"], ref
        if e["kind"] == "facility" and e.get("region"):
            ref = resolver.resolve(e["region"], {"region"})
            if ref and ref != e["region"]:
                e["region_written"], e["region"] = e["region"], ref
        issues = []
        if len(e["label"].split()) > 4:
            issues.append("label longer than 4 words")
        if normalize_name(e["label"]) in existing:
            issues.append(
                f"same label as existing node {existing[normalize_name(e['label'])]}: make it an alias"
            )
        if e["expected_kind"] and e["kind"] not in e["expected_kind"]:
            issues.append(
                f"kind {e['kind']} differs from where the name appears ({'/'.join(e['expected_kind'])})"
            )
        if e.get("parent") and not known(e["parent"]):
            issues.append(f"parent {e['parent']!r} is not a node")
        new_regions = {x["label"] for x in draft["new_nodes"] if x["kind"] == "region"}
        if e["kind"] == "facility" and e.get("region") not in regions | new_regions:
            issues.append(f"region {e.get('region')!r} is not a known region")
        e["issues"] = issues
    for section in ("aliases", "new_nodes"):
        for e in draft[section]:
            if e["issues"]:
                e["approve"] = False
    for e in draft["skip"] + draft["needs_review"]:
        e.setdefault("issues", [])
    return draft


DRAFT_HEADER = """# Curation round {round}: proposed changes to the node lists. Nothing here is applied yet.
#
# Review every entry, then run `ripplekg curate apply {round}`. Only entries with approve: true are
# applied. Edit freely: change a label, kind, parent or region, move an entry between sections,
# or set approve to false.
#
#   aliases     names that are another way of writing an existing node; add_aliases are appended
#   new_nodes   new companies, facilities, materials...; companies get tickers verified after approval
#   skip        names to stop reporting as unresolved (generic phrases, off-topic entities)
#   needs_review  the model could not decide; edit into one of the sections above or delete
#
# Each entry shows its evidence: how often it was seen, its spelling variants, which filers
# mentioned it, the model's reason and confidence, and any close existing node (fuzzy_match).
# approve defaults to true only for confidence >= 0.8 and for rule-based generic skips.
"""
