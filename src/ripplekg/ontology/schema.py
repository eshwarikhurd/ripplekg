"""Ontology: relation types and curated node lists (config/relations.yaml, config/nodes/*.yaml).

Every graph node is either a universe company (referenced by its bare id, "TSLA")
or a curated node referenced as "kind:id" ("material:lithium"). Nothing outside
these lists can become a node: unresolved names go to a review table instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from ripplekg.ontology.universe import Universe, load_universe
from ripplekg.settings import CONFIG_DIR

NODE_FILES = {
    "material": "materials.yaml",
    "component": "components.yaml",
    "region": "regions.yaml",
    "authority": "authorities.yaml",
    "program": "programs.yaml",
    "facility": "facilities.yaml",
}
NODE_KINDS = {"company", *NODE_FILES}
FAMILIES = {"trade", "ownership_capital", "footprint", "competition", "exposure", "disputes"}
CHANNELS = {"supply_chain", "both", "hub", "competitive", "event"}
PRIORS = {"strong": 1.0, "normal": 0.6, "weak": 0.3}


@dataclass(frozen=True)
class Relation:
    name: str
    family: str
    source: frozenset[str]
    target: frozenset[str]
    channel: str
    prior: str
    strength_bases: frozenset[str]
    symmetric: bool = False
    attributes: frozenset[str] = frozenset()
    description: str = ""
    upstream: str | None = None  # supply_chain relations: which end supplies the other
    guidance: str = ""  # when to use it, and when a neighbouring relation fits better

    @property
    def prior_weight(self) -> float:
        return PRIORS[self.prior]


MAX_NODE_WORDS = 4


@dataclass(frozen=True)
class Node:
    """A graph node. Its `ref` is the label used everywhere outside config: a company's ticker
    or readable ID, otherwise its short canonical name ("Copper", "Thacker Pass", "China").
    `id` is the short key config files use to refer to it; `kind` is a property, not part of
    the label."""

    kind: str
    id: str
    name: str
    aliases: tuple[str, ...] = ()
    parent: str | None = None  # ref of the parent node, after loading
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def ref(self) -> str:
        return self.id if self.kind == "company" else self.name


class OntologyError(ValueError):
    pass


def normalize_name(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Ontology:
    relations: dict[str, Relation]
    relation_aliases: dict[str, tuple[str, bool]]  # label -> (canonical relation, reverse?)
    statuses: tuple[str, ...]
    strength_bases: dict[str, str]
    exclusive_groups: dict[str, tuple[str, ...]]  # near-synonyms, most specific first
    nodes: dict[str, Node]  # ref -> node
    _names: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)  # (kind, normalized) -> ref
    _by_key: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)  # (kind, config id) -> ref

    def node(self, ref: str) -> Node:
        try:
            return self.nodes[ref]
        except KeyError:
            raise KeyError(f"unknown node {ref!r}") from None

    def kind_of(self, ref: str) -> str:
        return self.node(ref).kind

    def canonical_relation(self, label: str) -> tuple[str, bool]:
        """Map an extracted relation label to (canonical relation, reverse endpoints?)."""
        if label in self.relations:
            return label, False
        if label in self.relation_aliases:
            return self.relation_aliases[label]
        raise KeyError(f"unknown relation {label!r}")

    def edge_error(self, relation: str, source: str, target: str) -> str | None:
        """None if the edge fits the relation's endpoint signature, else the reason it doesn't."""
        rel = self.relations.get(relation)
        if rel is None:
            return f"unknown relation {relation!r}"
        if source not in self.nodes:
            return f"unknown source node {source!r}"
        if target not in self.nodes:
            return f"unknown target node {target!r}"
        if source == target:
            return "self-loop"
        sk, tk = self.kind_of(source), self.kind_of(target)
        if sk not in rel.source:
            return f"{relation} source must be {sorted(rel.source)}, got {sk}"
        if tk not in rel.target:
            return f"{relation} target must be {sorted(rel.target)}, got {tk}"
        return None

    def canonical_endpoints(self, relation: str, source: str, target: str) -> tuple[str, str]:
        """Symmetric relations are stored once, with endpoints in sorted order."""
        if self.relations[relation].symmetric and target < source:
            return target, source
        return source, target

    def ancestors(self, ref: str) -> list[str]:
        """Parent chain of a node, nearest first."""
        out, node = [], self.node(ref)
        while node.parent:
            out.append(node.parent)
            node = self.node(node.parent)
        return out

    def ref_for(self, kind: str, config_id: str) -> str:
        """The node label of a config id, e.g. ("region", "US-NV") -> "Nevada"."""
        try:
            return self._by_key[(kind, config_id)]
        except KeyError:
            raise KeyError(f"unknown {kind} id {config_id!r}") from None

    def lookup(self, kind: str, text: str) -> str | None:
        """Exact (normalized) name or alias match within one node kind."""
        return self._names.get((kind, normalize_name(text)))


def _relation(name: str, raw: dict) -> Relation:
    return Relation(
        name=name,
        family=raw["family"],
        source=frozenset(raw["source"]),
        target=frozenset(raw["target"]),
        channel=raw["channel"],
        prior=raw["prior"],
        strength_bases=frozenset(raw.get("strength_bases", [])),
        symmetric=bool(raw.get("symmetric", False)),
        attributes=frozenset(raw.get("attributes", [])),
        description=raw.get("description", ""),
        guidance=raw.get("guidance", ""),
        upstream=raw.get("upstream"),
    )


def build_ontology(relations_raw: dict, nodes_raw: dict[str, dict], universe: Universe) -> Ontology:
    errors: list[str] = []

    strength_bases = dict(relations_raw.get("strength_bases", {}))
    relations: dict[str, Relation] = {}
    for name, raw in relations_raw.get("relations", {}).items():
        try:
            rel = _relation(name, raw)
        except KeyError as exc:
            errors.append(f"relation {name}: missing field {exc}")
            continue
        if not re.fullmatch(r"[A-Z][A-Z_]*", name):
            errors.append(f"relation {name}: name must be UPPER_SNAKE")
        if rel.family not in FAMILIES:
            errors.append(f"relation {name}: unknown family {rel.family!r}")
        if rel.channel not in CHANNELS:
            errors.append(f"relation {name}: unknown channel {rel.channel!r}")
        if (rel.channel == "supply_chain") != (rel.upstream in {"source", "target"}):
            errors.append(f"relation {name}: supply_chain relations need upstream source|target, others none")
        if rel.prior not in PRIORS:
            errors.append(f"relation {name}: unknown prior {rel.prior!r}")
        for end in ("source", "target"):
            bad = getattr(rel, end) - NODE_KINDS
            if bad or not getattr(rel, end):
                errors.append(f"relation {name}: bad {end} kinds {sorted(bad) or 'empty'}")
        bad_bases = rel.strength_bases - set(strength_bases)
        if bad_bases:
            errors.append(f"relation {name}: undefined strength bases {sorted(bad_bases)}")
        if rel.symmetric and rel.source != rel.target:
            errors.append(f"relation {name}: symmetric relations need identical source and target kinds")
        relations[name] = rel

    exclusive_groups: dict[str, tuple[str, ...]] = {}
    for group, members in (relations_raw.get("exclusive_groups") or {}).items():
        unknown = [m for m in members if m not in relations]
        if unknown:
            errors.append(f"exclusive group {group}: unknown relations {unknown}")
        if len(set(members)) != len(members) or len(members) < 2:
            errors.append(f"exclusive group {group}: needs at least two distinct relations")
        exclusive_groups[group] = tuple(members)
    for name, rel in relations.items():
        if not rel.guidance:
            errors.append(f"relation {name}: missing guidance")

    relation_aliases: dict[str, tuple[str, bool]] = {}
    for label, raw in relations_raw.get("aliases", {}).items():
        if raw["relation"] not in relations:
            errors.append(f"alias {label}: unknown relation {raw['relation']!r}")
        if label in relations:
            errors.append(f"alias {label}: shadows a canonical relation")
        relation_aliases[label] = (raw["relation"], bool(raw.get("reverse", False)))

    nodes: dict[str, Node] = {}
    by_key: dict[tuple[str, str], str] = {}  # (kind, config id) -> ref
    for c in universe.companies:
        n = Node("company", c.id, c.name, tuple(c.search_names))
        nodes[n.ref] = n
        by_key[("company", c.id)] = n.ref
    for kind, raw in nodes_raw.items():
        for row in raw.get("nodes", []):
            try:
                extra = {k: v for k, v in row.items() if k not in {"id", "name", "aliases", "parent"}}
                n = Node(
                    kind, row["id"], row["name"], tuple(row.get("aliases", [])), row.get("parent"), extra
                )
            except KeyError as exc:
                errors.append(f"{kind} node {row}: missing field {exc}")
                continue
            texts = [n.id, n.name, *n.aliases] + ([n.parent] if n.parent is not None else [])
            if not all(isinstance(t, str) for t in texts):
                # YAML reads unquoted NO, YES, ON, OFF as booleans: quote such ids.
                errors.append(f"{kind} node {row}: id, name, aliases and parent must be strings")
                continue
            if n.ref in nodes:
                errors.append(f"node label {n.ref!r} is used by both a {nodes[n.ref].kind} and a {kind}")
                continue
            if (kind, n.id) in by_key:
                errors.append(f"duplicate {kind} id {n.id!r}")
            if len(n.name.split()) > MAX_NODE_WORDS:
                errors.append(f"{kind} {n.id}: name {n.name!r} is longer than {MAX_NODE_WORDS} words")
            nodes[n.ref] = n
            by_key[(kind, n.id)] = n.ref

    # Config files refer to parents by id; from here on the parent is a ref.
    for ref, n in list(nodes.items()):
        if n.parent:
            parent_ref = by_key.get((n.kind, n.parent))
            if parent_ref is None:
                errors.append(f"{ref}: parent {n.parent!r} is not a {n.kind}")
            else:
                nodes[ref] = replace(n, parent=parent_ref)

    labels: dict[str, str] = {}
    for ref in nodes:
        key = normalize_name(ref)
        if key in labels:
            errors.append(f"node labels {labels[key]!r} and {ref!r} differ only in case or punctuation")
        labels[key] = ref

    names: dict[tuple[str, str], str] = {}
    for ref, n in nodes.items():
        for text in (n.id, n.name, *n.aliases):
            key = (n.kind, normalize_name(text))
            if key in names and names[key] != ref:
                errors.append(f"{ref}: name or alias {text!r} already used by {names[key]}")
            names.setdefault(key, ref)
        if n.kind == "facility":
            ops = n.extra.get("operators", [])
            if not ops:
                errors.append(f"{ref}: facility needs operators")
            errors += [
                f"{ref}: operator {o!r} is not a universe company" for o in ops if o not in universe.by_id
            ]
            if ("region", n.extra.get("region")) not in by_key:
                errors.append(f"{ref}: unknown region {n.extra.get('region')!r}")
        if n.kind == "authority" and ("region", n.extra.get("region")) not in by_key:
            errors.append(f"{ref}: unknown region {n.extra.get('region')!r}")
        if n.kind == "program" and ("authority", n.extra.get("authority")) not in by_key:
            errors.append(f"{ref}: unknown authority {n.extra.get('authority')!r}")

    if errors:
        raise OntologyError("invalid ontology:\n  " + "\n  ".join(errors))
    return Ontology(
        relations=relations,
        relation_aliases=relation_aliases,
        exclusive_groups=exclusive_groups,
        statuses=tuple(relations_raw.get("statuses", [])),
        strength_bases=strength_bases,
        nodes=nodes,
        _names=names,
        _by_key=by_key,
    )


def load_ontology(config_dir: Path = CONFIG_DIR, universe: Universe | None = None) -> Ontology:
    with (config_dir / "relations.yaml").open() as f:
        relations_raw = yaml.safe_load(f)
    nodes_raw = {}
    for kind, filename in NODE_FILES.items():
        with (config_dir / "nodes" / filename).open() as f:
            nodes_raw[kind] = yaml.safe_load(f)
    return build_ontology(relations_raw, nodes_raw, universe or load_universe(config_dir / "universe.yaml"))


# ---------------------------------------------------------------- relation examples


@dataclass(frozen=True)
class Example:
    relation: str
    text: str
    filer: str
    expect: tuple[dict, ...]
    why: str = ""


def load_examples(path: Path = CONFIG_DIR / "relation_examples.yaml") -> list[Example]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return [
        Example(r["relation"], r["text"], r["filer"], tuple(r.get("expect") or ()), r.get("why", ""))
        for r in raw["examples"]
    ]


MAX_ENDPOINT_WORDS = 4


def example_errors(ont: Ontology, examples: list[Example]) -> list[str]:
    """Checks that examples agree with the ontology, every quote is a verbatim complete sentence,
    and every endpoint is a short name."""
    from ripplekg.extraction.verify import snap_to_sentences as _snap

    errors = []
    for i, ex in enumerate(examples):
        tag = f"example {i} ({ex.relation})"
        try:
            ont.canonical_relation(ex.relation)
        except KeyError:
            errors.append(f"{tag}: unknown relation {ex.relation!r}")
        if not ex.expect and not ex.why:
            errors.append(f"{tag}: negative example needs a `why`")
        for edge in ex.expect:
            rel = ont.relations.get(edge.get("relation", ""))
            if rel is None:
                errors.append(f"{tag}: expected edge uses non-canonical relation {edge.get('relation')!r}")
                continue
            if edge.get("quote", "") not in ex.text:
                errors.append(f"{tag}: quote is not verbatim in text: {edge.get('quote')!r}")
            elif _snap(edge["quote"], ex.text) != " ".join(edge["quote"].split()):
                errors.append(f"{tag}: quote is not a complete sentence: {edge.get('quote')!r}")
            for end in ("source", "target"):
                if len(str(edge.get(end, "")).split()) > MAX_ENDPOINT_WORDS:
                    errors.append(f"{tag}: {end} longer than {MAX_ENDPOINT_WORDS} words: {edge.get(end)!r}")
            if edge.get("status") not in ont.statuses:
                errors.append(f"{tag}: bad status {edge.get('status')!r}")
            basis = edge.get("strength_basis")
            if basis is not None and basis not in rel.strength_bases:
                errors.append(f"{tag}: strength basis {basis!r} not allowed for {rel.name}")
            strength = edge.get("strength")
            if strength is not None and not 0 < strength <= 1:
                errors.append(f"{tag}: strength {strength} outside (0, 1]")
            for attr in ("material", "component"):
                if attr in edge and (edge[attr] not in ont.nodes or ont.kind_of(edge[attr]) != attr):
                    errors.append(f"{tag}: unknown {attr} {edge[attr]!r}")
            for attr in ("material", "component", "hazard"):
                if attr in edge and attr not in rel.attributes:
                    errors.append(f"{tag}: attribute {attr!r} not allowed for {rel.name}")
    covered = {ont.canonical_relation(ex.relation)[0] for ex in examples if ex.relation in ont.relations}
    errors += [f"relation {r} has no example" for r in sorted(set(ont.relations) - covered)]
    return errors
