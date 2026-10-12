"""Deterministic checks on extracted edges. No LLM is involved.

An edge is kept only if:
1. its quote appears verbatim in the source chunk (after normalizing whitespace, quote marks and
   dashes, which HTML conversion can alter),
2. any strength it claims is backed by a number inside the quote,
3. both endpoints resolve to a universe company or a curated node, and
4. the resolved edge fits its relation's endpoint signature.
Each rejected edge keeps the reason, so the gold set and the data-quality page can report them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ripplekg.ontology.schema import Ontology, normalize_name

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LEGAL_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "plc",
    "ag", "sa", "nv", "se", "gmbh", "holdings", "holding", "group", "the",
}  # fmt: skip


_PAREN = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def normalize_text(text: str) -> str:
    # Models sometimes write a character as its JSON escape text ("\u2019") inside a string.
    text = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    return _WS.sub(" ", text.translate(_TRANSLATE)).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_text(quote)
    return bool(q) and q in normalize_text(text)


def strength_backed(strength: float | None, quote: str) -> bool:
    """A claimed strength must match a number in the quote (as a percentage or a fraction)."""
    if strength is None:
        return True
    numbers = {float(n) for n in _NUMBER.findall(quote.replace(",", ""))}
    return any(abs(n - strength * 100) < 0.051 or abs(n - strength) < 0.0005 for n in numbers) or (
        strength == 1.0 and bool(re.search(r"wholly[- ]owned|100%|sole|all of", quote, re.I))
    )


def _strip_legal(name: str) -> str:
    return " ".join(w for w in normalize_name(name).split() if w not in _LEGAL_WORDS)


class Resolver:
    """Maps a name as written to a node reference, within the kinds a relation allows."""

    def __init__(self, ont: Ontology):
        self.ont = ont
        self._stripped: dict[tuple[str, str], str] = {}
        for ref, n in ont.nodes.items():
            for text in (n.id, n.name, *n.aliases):
                key = (n.kind, _strip_legal(text))
                if key[1]:
                    self._stripped.setdefault(key, ref)

    def _resolve_one(self, name: str, kinds) -> str | None:
        for kind in sorted(kinds):
            ref = self.ont.lookup(kind, name) or self._stripped.get((kind, _strip_legal(name)))
            if ref:
                return ref
        return None

    def resolve(self, name: str, kinds: frozenset[str] | set[str]) -> str | None:
        """Tries the name as written, then without a trailing "(...)", then the bracketed text
        itself, so "Albemarle (ALB)" and "Vale S.A. (Vale)" resolve."""
        candidates = [name]
        m = _PAREN.match(name)
        if m:
            candidates += [m.group(1), m.group(2)]
        for cand in candidates:
            ref = self._resolve_one(cand.strip(), kinds)
            if ref:
                return ref
        return None


@dataclass
class Verdict:
    ok: bool
    reason: str
    relation: str = ""
    source_ref: str | None = None
    target_ref: str | None = None


def verify_edge(edge: dict, chunk_text: str, ont: Ontology, resolver: Resolver) -> Verdict:
    if not quote_in_text(edge.get("quote", ""), chunk_text):
        return Verdict(False, "quote not verbatim")
    if not strength_backed(edge.get("strength"), edge.get("quote", "")):
        return Verdict(False, "strength not in quote")
    try:
        relation, reverse = ont.canonical_relation(edge["relation"])
    except KeyError:
        return Verdict(False, "unknown relation")
    source, target = (edge["target"], edge["source"]) if reverse else (edge["source"], edge["target"])
    rel = ont.relations[relation]
    s = resolver.resolve(source, rel.source)
    t = resolver.resolve(target, rel.target)
    if s is None or t is None:
        missing = "source" if s is None else "target"
        return Verdict(False, f"unresolved {missing}", relation, s, t)
    err = ont.edge_error(relation, s, t)
    if err:
        return Verdict(False, err, relation, s, t)
    return Verdict(True, "ok", relation, s, t)
