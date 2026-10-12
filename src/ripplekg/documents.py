"""The source-agnostic Document: every source adapter emits these, every later stage reads them.

Stored one file per document as gzipped JSON under data/documents/{company_id}/{doc_id}.json.gz,
written atomically. `source_sha256` records which raw input produced it, so a stage can skip
documents whose raw input has not changed.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocSection:
    name: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str  # stable id within its source, e.g. "edgar:0001318605-24-000010"
    source_type: str  # "edgar", later "press_release", "transcript", ...
    company_id: str  # universe company that published it
    published_at: str  # ISO-8601 UTC time the document became public
    url: str
    title: str
    sections: tuple[DocSection, ...]
    source_sha256: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> Document:
        raw = json.loads(text)
        raw["sections"] = tuple(DocSection(**s) for s in raw["sections"])
        return Document(**raw)


def _filename(doc_id: str) -> str:
    return doc_id.replace(":", "_").replace("/", "_") + ".json.gz"


class DocumentStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, company_id: str, doc_id: str) -> Path:
        return self.root / company_id / _filename(doc_id)

    def write(self, doc: Document) -> Path:
        path = self.path(doc.company_id, doc.doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(doc.to_json())
        os.replace(tmp, path)
        return path

    def read(self, company_id: str, doc_id: str) -> Document | None:
        path = self.path(company_id, doc_id)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return Document.from_json(f.read())

    def __iter__(self) -> Iterator[Document]:
        for path in sorted(self.root.glob("*/*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                yield Document.from_json(f.read())
