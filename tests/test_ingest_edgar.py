import json

import pandas as pd
from fakes import FakeResponse, FakeSession

from ripplekg.download.client import EdgarClient
from ripplekg.download.fetch import exhibit_names, fetch_filings, load_manifest
from ripplekg.ingestion.documents import DocSection, Document, DocumentStore
from ripplekg.ingestion.edgar_adapter import to_document

BASE = "https://www.sec.gov/Archives/edgar/data/1/000000000124000001"
FILLER = "We describe our operations at length. " * 10
PRIMARY_8K = (
    f"<p>Item 1.01 Entry into a Material Definitive Agreement.</p><p>We signed an offtake. {FILLER}</p>"
)
EXHIBIT = "<p>Press release: Acme Lithium signs offtake agreement with Borealis Motors.</p>"


def selected(form="8-K"):
    return pd.DataFrame(
        [
            {
                "company_id": "ACME",
                "accession": "0000000001-24-000001",
                "form": form,
                "filing_date": "2024-03-05",
                "report_date": "",
                "accepted_et": "2024-03-05 08:15:00",
                "items": "1.01,9.01",
                "primary_document": "d8k.htm",
                "url": f"{BASE}/d8k.htm",
            }
        ]
    )


def router(url, params):
    if url.endswith("index.json"):
        listing = {
            "directory": {
                "item": [
                    {"name": "d8k.htm", "size": "4000"},
                    {"name": "dex991.htm", "size": "40000"},
                    {"name": "dex992.htm", "size": "900"},
                    {"name": "img.jpg", "size": "900000"},
                ]
            }
        }
        return FakeResponse(200, json.dumps(listing).encode())
    if url.endswith("d8k.htm"):
        return FakeResponse(200, PRIMARY_8K.encode())
    if url.endswith("dex991.htm"):
        return FakeResponse(200, EXHIBIT.encode())
    raise AssertionError(f"unexpected fetch {url}")


def client(session):
    return EdgarClient("t me@example.com", cache_dir=None, session=session, sleep=lambda s: None)


def test_exhibit_names_keeps_large_ex99_files_only():
    listing = {
        "directory": {
            "item": [
                {"name": "ex99-1.htm", "size": "50000"},
                {"name": "d123dex991.htm", "size": "90000"},
                {"name": "ex99-2.htm", "size": "7000"},
                {"name": "exhibit991toconference.htm", "size": "2000"},
                {"name": "exhibit992toconference.htm", "size": "82000"},
                {"name": "ex10-1.htm", "size": "99999"},
                {"name": "ex99-3.jpg", "size": "99999"},
            ]
        }
    }
    assert exhibit_names(listing, "8-K") == ["d123dex991.htm", "exhibit992toconference.htm", "ex99-1.htm"]
    assert exhibit_names(listing, "40-F") == ["d123dex991.htm", "exhibit992toconference.htm", "ex99-1.htm"]
    small = {"directory": {"item": [{"name": "ex99-2.htm", "size": "7000"}]}}
    assert exhibit_names(small, "8-K") == ["ex99-2.htm"]
    assert exhibit_names(small, "40-F") == []


def test_fetch_is_idempotent_and_builds_document(tmp_path):
    session = FakeSession(router=router)
    raw, manifest_path = tmp_path / "raw", tmp_path / "manifest.parquet"
    m = fetch_filings(client(session), selected(), raw, manifest_path)
    assert sorted(m["file"]) == ["d8k.htm", "dex991.htm"]
    first_calls = len(session.calls)

    fetch_filings(client(session), selected(), raw, manifest_path)
    assert len(session.calls) == first_calls  # nothing re-downloaded
    assert len(load_manifest(manifest_path)) == 2

    filing = selected().iloc[0].to_dict()
    doc = to_document(filing, m.to_dict("records"), raw)
    assert doc.doc_id == "edgar:0000000001-24-000001"
    assert doc.published_at == "2024-03-05T13:15:00+00:00"  # 08:15 EST
    assert [s.name for s in doc.sections] == ["material_agreement", "exhibit:dex991.htm"]
    assert "Borealis Motors" in doc.sections[1].text


def test_half_finished_filing_is_refetched(tmp_path):
    session = FakeSession(router=router)
    raw, manifest_path = tmp_path / "raw", tmp_path / "manifest.parquet"
    fetch_filings(client(session), selected(), raw, manifest_path)
    (raw / "ACME" / "0000000001-24-000001" / "dex991.htm.gz").unlink()
    before = len(session.calls)
    fetch_filings(client(session), selected(), raw, manifest_path)
    assert len(session.calls) > before


def test_document_store_round_trip(tmp_path):
    store = DocumentStore(tmp_path)
    doc = Document(
        doc_id="edgar:0001-24-1",
        source_type="edgar",
        company_id="ACME",
        published_at="2024-03-05T13:15:00+00:00",
        url="https://example.com",
        title="ACME 10-K",
        sections=(DocSection("business", "Text é"),),
        source_sha256="abc",
        metadata={"form": "10-K"},
    )
    store.write(doc)
    assert store.read("ACME", doc.doc_id) == doc
    assert list(store) == [doc]
    assert store.read("ACME", "edgar:missing") is None
