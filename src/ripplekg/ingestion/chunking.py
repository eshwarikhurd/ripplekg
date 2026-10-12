"""Documents -> deduplicated, packed chunks.

Companies repeat most of their filing text from one period to the next. Each line is hashed per
company, in publication order: a line already seen in an earlier document is not sent to the LLM
again, but its new sighting is recorded (`occurrences`) so edges extracted from it can extend
their "last seen" date. Novel lines are packed, in order, into chunks of about TARGET_CHARS, with
the previous chunk's last line repeated for context.

The stage is deterministic: rebuilding from the same documents gives the same chunk ids.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ripplekg.ingestion.documents import Document

TARGET_CHARS = 3000
MIN_DEDUP_CHARS = 40  # shorter lines (headings, labels) are context, not content
MAX_HEADING_CHARS = 120
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def line_hash(line: str) -> str:
    return hashlib.sha256(_WS.sub(" ", line.lower()).strip().encode()).hexdigest()[:20]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company_id: str
    published_at: str
    form: str
    section: str
    idx: int
    text: str
    line_hashes: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class ChunkStats:
    documents: int = 0
    lines: int = 0
    content_lines: int = 0
    repeated_lines: int = 0
    chars_in: int = 0
    chars_novel: int = 0


def _split_long(line: str, limit: int) -> list[str]:
    if len(line) <= limit:
        return [line]
    parts, cur = [], ""
    for sent in _SENTENCE_END.split(line):
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    out = []
    for p in parts:  # sentences longer than the limit are cut hard
        out += [p[i : i + limit] for i in range(0, len(p), limit)]
    return out


def _novel_lines(text: str, seen: set[str], stats: ChunkStats) -> list[tuple[str, str | None]]:
    """Lines to keep, as (line, hash or None for context-only headings)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    kept: list[tuple[str, str | None]] = []
    pending_heading: str | None = None
    for line in lines:
        stats.lines += 1
        stats.chars_in += len(line)
        if len(line) < MIN_DEDUP_CHARS and "|" not in line:
            pending_heading = line if len(line) <= MAX_HEADING_CHARS else None
            continue
        stats.content_lines += 1
        h = line_hash(line)
        if h in seen:
            stats.repeated_lines += 1
            pending_heading = None
            continue
        seen.add(h)
        stats.chars_novel += len(line)
        if pending_heading:
            kept.append((pending_heading, None))
            pending_heading = None
        kept.append((line, h))
    return kept


def _content_hashes(text: str) -> list[str]:
    return [line_hash(ln) for ln in text.split("\n") if len(ln) >= MIN_DEDUP_CHARS or "|" in ln]


def chunk_documents(docs: Iterable[Document]) -> tuple[list[Chunk], list[dict], ChunkStats]:
    """Returns (chunks, occurrences, stats). `docs` may come in any order."""
    stats = ChunkStats()
    chunks: list[Chunk] = []
    occurrences: list[dict] = []
    seen_by_company: dict[str, set[str]] = {}
    for doc in sorted(docs, key=lambda d: (d.company_id, d.published_at, d.doc_id)):
        stats.documents += 1
        seen = seen_by_company.setdefault(doc.company_id, set())
        form = doc.metadata.get("form", "")
        for section in doc.sections:
            before = set(seen)
            kept = _novel_lines(section.text, seen, stats)
            occurrences += [
                {
                    "line_hash": h,
                    "company_id": doc.company_id,
                    "doc_id": doc.doc_id,
                    "published_at": doc.published_at,
                }
                for h in _content_hashes(section.text)
                if h in before
            ]
            chunks += _pack(doc, form, section.name, kept)
    return chunks, occurrences, stats


def _pack(doc: Document, form: str, section: str, kept: list[tuple[str, str | None]]) -> list[Chunk]:
    out: list[Chunk] = []
    buf: list[str] = []
    hashes: list[str] = []
    has_content = False
    carry: str | None = None

    def flush():
        nonlocal buf, hashes, has_content, carry
        if has_content:  # a buffer of headings alone is not worth a chunk
            body = "\n".join(buf)
            text = f"{carry}\n{body}" if carry else body
            idx = len(out)
            cid = hashlib.sha256(f"{doc.doc_id}|{section}|{idx}|{text}".encode()).hexdigest()[:20]
            out.append(
                Chunk(cid, doc.doc_id, doc.company_id, doc.published_at, form, section, idx, text, hashes)
            )
            last = buf[-1]
            carry = last if len(last) <= 400 else None
        buf, hashes, has_content = [], [], False

    for line, h in kept:
        for n, piece in enumerate(_split_long(line, TARGET_CHARS)):
            if buf and sum(len(b) + 1 for b in buf) + len(piece) > TARGET_CHARS:
                flush()
            buf.append(piece)
            if h is not None:
                has_content = True
                if n == 0 or h not in hashes:
                    hashes.append(h)
    flush()
    return out
