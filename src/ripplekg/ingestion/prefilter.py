"""Keyword gate in front of the LLM: which relation families might a chunk contain?

A family is tagged only when its cue words co-occur with something that can be an edge endpoint:
another company (a universe name, or a legal-entity name such as "Halcyon Materials Inc." that is
not the filer's own), a disclosed percentage, or a curated node. Footprint needs a material,
component or facility; exposure accepts any curated node, including regions and authorities.
Untagged chunks never reach the LLM. Each tagged family costs one LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ripplekg.ontology.schema import Ontology

FAMILY_CUES = {
    "trade": r"suppl(?:y|ier|iers|ies|ied)|offtake|off-take|customer|purchase[sd]? from|sole[- ]source|"
    r"single[- ]source|limited (?:number of )?(?:source|supplier)|toll(?:ing| process)|licens|distribut",
    "ownership_capital": r"joint venture|\bJV\b|subsidiar|equity (?:stake|interest|investment)|stake in|"
    r"invest(?:ed|ment) in|prepay|acqui(?:re|red|sition)|merger",
    "footprint": r"\bmine\b|mining|refiner|refining|processing (?:plant|facility)|plant|facility|gigafactory|"
    r"produc(?:e|es|tion) of|source[sd]? (?:of|from)|raw material",
    "competition": r"compet(?:e|es|itor|itors|ition|itive)",
    "exposure": r"export (?:control|licen[cs]e|ban|restriction)|tariff|sanction|section 301|"
    r"entity of concern|\bFEOC\b|inflation reduction act|\b45X\b|\b30D\b|tax credit|loan program|"
    r"earthquake|flood|hurricane|drought|wildfire|water rights",
    "disputes": r"lawsuit|litigation|arbitration|complaint|recall|penalt|\bfined?\b|settlement|breach",
}
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)? percent", re.I)
_LEGAL_SUFFIX = r"(?:Inc|Corp|Corporation|Ltd|Limited|LLC|Co|AG|plc|S\.A|N\.V|GmbH)"
_LEGAL_ENTITY = re.compile(r"\b[A-Z][\w&.\-]*(?: [A-Z][\w&.\-]*)* " + _LEGAL_SUFFIX + r"\b")
_NEEDS_COMPANY = {"trade", "ownership_capital", "competition", "disputes"}
_NEEDS_NODE = {"footprint", "exposure"}


def _alternation(names: set[str]) -> re.Pattern:
    escaped = sorted((re.escape(n) for n in names if len(n) >= 3), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(escaped) + r")(?![\w-])", re.I)


@dataclass
class Prefilter:
    cues: dict[str, re.Pattern]
    companies: dict[str, re.Pattern]  # company id -> its names
    supply_nodes: re.Pattern  # materials, components, facilities
    nodes: re.Pattern  # every curated non-company node name and alias

    @classmethod
    def from_ontology(cls, ont: Ontology) -> Prefilter:
        companies, node_names, supply_names = {}, set(), set()
        for n in ont.nodes.values():
            names = {n.name, *n.aliases}
            if n.kind == "company":
                companies[n.id] = _alternation(names)
                continue
            usable = {x for x in names if len(x) >= 4 or x.isupper()}
            node_names |= usable
            if n.kind in {"material", "component", "facility"}:
                supply_names |= usable
        cues = {fam: re.compile(pat, re.I) for fam, pat in FAMILY_CUES.items()}
        return cls(
            cues=cues,
            companies=companies,
            supply_nodes=_alternation(supply_names),
            nodes=_alternation(node_names),
        )

    def _other_legal_entity(self, text: str, filer: str) -> bool:
        own = self.companies.get(filer)
        return any(not (own and own.search(m.group(0))) for m in _LEGAL_ENTITY.finditer(text))

    def other_companies(self, text: str, filer: str) -> list[str]:
        return [cid for cid, pat in self.companies.items() if cid != filer and pat.search(text)]

    def families(self, text: str, filer: str) -> list[str]:
        cue_hits = [fam for fam, pat in self.cues.items() if pat.search(text)]
        if not cue_hits:
            return []
        has_company = bool(self.other_companies(text, filer)) or self._other_legal_entity(text, filer)
        has_percent = bool(_PERCENT.search(text))
        out = []
        for fam in cue_hits:
            company_ok = fam in _NEEDS_COMPANY and (has_company or (fam == "trade" and has_percent))
            if fam == "footprint":
                node_ok = bool(self.supply_nodes.search(text))
            else:
                node_ok = fam in _NEEDS_NODE and bool(self.nodes.search(text))
            if company_ok or node_ok:
                out.append(fam)
        return out
