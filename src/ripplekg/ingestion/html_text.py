"""HTML filing -> plain text with one line per block and tables kept as `cell | cell` rows.

Tables matter: concentration disclosures ("Customer A: 23% of revenue") are often only in
tables. Hidden inline-XBRL facts, scripts and styles are dropped. Page furniture (bare page
numbers, "Table of Contents" links) is removed.
"""

from __future__ import annotations

import re
import unicodedata

from lxml import etree, html

BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "section", "article", "blockquote", "pre", "hr", "dt", "dd",
}  # fmt: skip
DROP_TAGS = {"script", "style", "head", "title", "noscript"}
CELL_TAGS = {"td", "th"}

_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_SPACES = re.compile(r"[ \t  -​  　]+")
_PAGE_FURNITURE = re.compile(
    r"^(?:\d{1,3}|[ivxlc]{1,6}|page \d+|table of contents|index to financial statements)$", re.I
)


def _local(tag) -> str:
    """Tag name without namespace prefix ("ix:header" -> "header"); '' for comments/PIs."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


def _drop(el) -> None:
    """Remove an element but keep its tail text."""
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail
    prev = el.getprevious()
    if tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


def _cell_text(cell) -> str:
    return _SPACES.sub(" ", " ".join(cell.itertext())).strip()


def html_to_text(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return ""
    # lxml rejects unicode strings that carry an XML encoding declaration.
    raw = re.sub(r"^<\?xml[^>]*\?>", "", raw)
    root = html.fromstring(raw)

    for el in list(root.iter()):
        tag = _local(el.tag)
        if not tag:  # comments, processing instructions
            if el.getparent() is not None:
                _drop(el)
            continue
        if (
            tag in DROP_TAGS
            or (tag == "header" and ":" in str(el.tag))
            or _HIDDEN.search(el.get("style", ""))
        ):
            _drop(el)

    # Flatten each table row into a single " | "-joined line.
    for tr in list(root.iter("tr")):
        cells = [_cell_text(c) for c in tr if _local(c.tag) in CELL_TAGS]
        line = " | ".join(c for c in cells if c)
        for child in list(tr):
            tr.remove(child)
        tr.text = line

    for el in root.iter():
        if _local(el.tag) in BLOCK_TAGS:
            el.tail = "\n" + (el.tail or "")
            if _local(el.tag) in {"br", "hr"}:
                continue
            el.text = "\n" + (el.text or "")

    text = (
        root.text_content()
        if hasattr(root, "text_content")
        else etree.tostring(root, method="text", encoding=str)
    )
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for line in text.split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if not line or _PAGE_FURNITURE.match(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip()
