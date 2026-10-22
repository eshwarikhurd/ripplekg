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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])[\"'”’)]*\s+(?=[\"'“‘(]*[A-Z0-9•])|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence. Lines are boundaries too, so table rows stay separate."""
    spans, start = [], 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def snap_to_sentences(quote: str, text: str) -> str | None:
    """The complete sentence(s) of `text` that contain `quote`, verbatim from `text`.

    Returns None if the quote is not in the text. Matching ignores case, whitespace and
    typography, so the returned span is located through a normalized-offset map.
    """
    q = normalize_text(quote)
    if not q:
        return None
    norm_chars, index = [], []  # normalized text, and each char's offset in the original
    prev_space = True
    for i, ch in enumerate(text):
        c = normalize_text(ch) if not ch.isspace() else " "
        if c == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        for cc in c or " ":
            norm_chars.append(cc)
            index.append(i)
    norm = "".join(norm_chars)
    pos = norm.find(q)
    if pos < 0:
        return None
    start, end = index[pos], index[min(pos + len(q), len(index)) - 1] + 1
    spans = [sp for sp in sentence_spans(text) if sp[0] < end and sp[1] > start]
    if not spans:
        return None
    return " ".join(text[spans[0][0] : spans[-1][1]].split())


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_text(quote)
    return bool(q) and q in normalize_text(text)


def strength_backed(strength: float | None, quote: str) -> bool:
    """A claimed strength must match a number in the quote (as a percentage or a fraction)."""
    if strength is None or strength != strength:  # None, or NaN read back from parquet
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


MAX_ENDPOINT_WORDS = 4
_FILER_WORDS = re.compile(r"\b(we|our|ours|us|the company)\b", re.I)


def endpoint_words(name: str) -> int:
    """Words in an endpoint name, ignoring a trailing "(TICKER)" and legal suffixes."""
    m = _PAREN.match(name)
    base = m.group(1) if m else name
    return len(_strip_legal(base).split()) if _strip_legal(base) else len(base.split())


def mentioned(ref: str, written: str, quote: str, ont: Ontology, filer: str) -> bool:
    """Is the endpoint named in the quote (as written, or by any curated name, alias or id)?
    The filer may also appear as "we", "our" or "the Company"."""
    q = normalize_text(quote)
    names = {written}
    m = _PAREN.match(written)
    if m:
        names |= {m.group(1), m.group(2)}
    node = ont.nodes.get(ref)
    if node is not None:
        names |= {node.id, node.name, *node.aliases}
    for name in names:
        n = normalize_text(_strip_legal(name) or name)
        if n and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", q):
            return True
    return ref == filer and bool(_FILER_WORDS.search(quote))


@dataclass
class Verdict:
    ok: bool
    reason: str
    relation: str = ""
    source_ref: str | None = None
    target_ref: str | None = None
    quote_full: str = ""
    material_ref: str | None = None
    component_ref: str | None = None


def verify_edge(edge: dict, chunk_text: str, ont: Ontology, resolver: Resolver, filer: str = "") -> Verdict:
    full = snap_to_sentences(edge.get("quote", ""), chunk_text)
    if full is None:
        return Verdict(False, "quote not verbatim")
    if not strength_backed(edge.get("strength"), full):
        return Verdict(False, "strength not in quote", quote_full=full)
    try:
        relation, reverse = ont.canonical_relation(edge["relation"])
    except KeyError:
        return Verdict(False, "unknown relation", quote_full=full)
    source, target = (edge["target"], edge["source"]) if reverse else (edge["source"], edge["target"])
    if max(endpoint_words(source), endpoint_words(target)) > MAX_ENDPOINT_WORDS:
        return Verdict(False, "endpoint too long", relation, quote_full=full)
    rel = ont.relations[relation]
    s = resolver.resolve(source, rel.source)
    t = resolver.resolve(target, rel.target)
    if s is None or t is None:
        missing = "source" if s is None else "target"
        return Verdict(False, f"unresolved {missing}", relation, s, t, full)
    err = ont.edge_error(relation, s, t)
    if err:
        return Verdict(False, err, relation, s, t, full)
    for ref, written in ((s, source), (t, target)):
        if not mentioned(ref, written, full, ont, filer):
            return Verdict(False, "endpoint not in quote", relation, s, t, full)
    s, t = ont.canonical_endpoints(relation, s, t)
    material = resolver.resolve(edge["material"], {"material"}) if edge.get("material") else None
    component = resolver.resolve(edge["component"], {"component"}) if edge.get("component") else None
    return Verdict(True, "ok", relation, s, t, full, material, component)
