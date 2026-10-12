"""Split a filing's text into the sections that describe relationships.

Each form has its own item numbering. Headings are lines that start with "Item N" (or "Item
N.NN" for 8-Ks). A filing's table of contents repeats every heading, so for each item the
**longest** span between consecutive headings is kept: table-of-contents entries are a
line long, real sections are pages long.

10-Q Part I and Part II reuse item numbers, so 10-Q headings are keyed by (part, item).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# form -> {item key -> section name}
WANTED: dict[str, dict[str, str]] = {
    "10-K": {"1": "business", "1A": "risk_factors", "3": "legal_proceedings", "7": "mda"},
    "10-Q": {"I-2": "mda", "II-1": "legal_proceedings", "II-1A": "risk_factors"},
    "20-F": {"3": "key_information", "4": "business", "5": "operating_review", "8": "legal_proceedings"},
    "8-K": {
        "1.01": "material_agreement",
        "1.02": "agreement_termination",
        "1.03": "bankruptcy",
        "2.01": "acquisition_disposition",
        "2.05": "exit_costs",
        "2.06": "impairment",
        "7.01": "reg_fd",
        "8.01": "other_events",
    },
}

MIN_SECTION_CHARS = {"8-K": 50}  # 8-K items are often one or two sentences
DEFAULT_MIN_SECTION_CHARS = 200

_ITEM = re.compile(r"^item\s*(\d{1,2})\s*(?:\.\s*([a-d])\b|([a-d])\b)?[\s.:\-–—]", re.I)
_ITEM_8K = re.compile(r"^item\s*(\d)\.(\d{2})(?!\d)", re.I)
_PART = re.compile(r"^part\s+(i{1,2}|iv|iii)\b", re.I)


@dataclass(frozen=True)
class Section:
    name: str
    text: str


def _headings(lines: list[str], form: str) -> list[tuple[int, str]]:
    """(line index, item key) for every heading line."""
    out = []
    part = ""
    for i, line in enumerate(lines):
        head = line[:80]
        if form == "10-Q":
            m = _PART.match(head)
            if m:
                part = m.group(1).upper()
                continue
        if form == "8-K":
            m = _ITEM_8K.match(head)
            if m:
                out.append((i, f"{m.group(1)}.{m.group(2)}"))
            continue
        m = _ITEM.match(head + " ")
        if m:
            key = m.group(1) + ((m.group(2) or m.group(3) or "").upper())
            if form == "20-F":
                key = m.group(1)  # 20-F sub-items (4.A, 4.B) belong to their parent item
            elif form == "10-Q":
                key = f"{part}-{key}" if part else key
            out.append((i, key))
    return out


def split_sections(text: str, form: str) -> list[Section]:
    """Wanted sections for `form`, or one `full_text` section if the form has no item map
    or none of its items can be found."""
    base_form = form.removesuffix("/A")
    wanted = WANTED.get(base_form)
    if not wanted:
        return [Section("full_text", text)] if text.strip() else []

    lines = text.split("\n")
    heads = _headings(lines, base_form)
    best: dict[str, str] = {}
    for n, (start, key) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        if key not in wanted:
            continue
        body = "\n".join(lines[start:end]).strip()
        if len(body) > len(best.get(key, "")):
            best[key] = body

    sections = [
        Section(wanted[key], body)
        for key, body in best.items()
        if len(body) >= MIN_SECTION_CHARS.get(base_form, DEFAULT_MIN_SECTION_CHARS)
    ]
    if not sections and text.strip():
        return [Section("full_text", text)]
    order = list(wanted)
    keys = {v: k for k, v in wanted.items()}
    return sorted(sections, key=lambda s: order.index(keys[s.name]))
