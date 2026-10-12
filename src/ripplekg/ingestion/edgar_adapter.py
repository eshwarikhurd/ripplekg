"""EDGAR adapter: a downloaded filing -> a source-agnostic Document with the sections that
describe relationships."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ripplekg.download.fetch import read_raw
from ripplekg.ingestion.documents import DocSection, Document
from ripplekg.ingestion.html_text import html_to_text
from ripplekg.ingestion.sections import split_sections


def source_hash(files: list[dict]) -> str:
    """Hash of a filing's raw files: a document is rebuilt only when this changes."""
    return hashlib.sha256("".join(sorted(f["sha256"] for f in files)).encode()).hexdigest()


def to_document(filing: dict, files: list[dict], raw_dir: Path) -> Document:
    """Build one Document from a filing's index row and its manifest rows."""
    sections: list[DocSection] = []
    for f in sorted(files, key=lambda f: (f["role"] != "primary", f["file"])):
        raw = read_raw(raw_dir / filing["company_id"] / filing["accession"] / (f["file"] + ".gz"))
        text = html_to_text(raw)
        if f["role"] == "primary":
            sections += [DocSection(s.name, s.text) for s in split_sections(text, filing["form"])]
        elif text.strip():
            sections.append(DocSection(f"exhibit:{f['file']}", text))
    et = filing["accepted_et"]
    published = (
        pd.Timestamp(et).tz_localize("America/New_York").tz_convert("UTC").isoformat()
        if et
        else f"{filing['filing_date']}T00:00:00+00:00"
    )
    return Document(
        doc_id=f"edgar:{filing['accession']}",
        source_type="edgar",
        company_id=filing["company_id"],
        published_at=published,
        url=filing["url"],
        title=f"{filing['company_id']} {filing['form']} {filing['filing_date']}",
        sections=tuple(sections),
        source_sha256=source_hash(files),
        metadata={
            "form": filing["form"],
            "filing_date": filing["filing_date"],
            "report_date": filing["report_date"],
            "items": filing["items"],
            "accession": filing["accession"],
        },
    )
