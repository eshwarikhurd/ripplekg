from fakes import FakeSession, json_response

from ripplekg.download.client import EdgarClient
from ripplekg.download.index import Filing, list_filings, load_ticker_map


def block(rows):
    keys = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument", "items"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


RECENT = block(
    [
        ("0000000001-26-000001", "2026-02-10", "2025-12-31", "10-K", "a10k.htm", ""),
        ("0000000001-25-000009", "2025-08-01", "2025-06-30", "10-Q", "q2.htm", ""),
        ("0000000001-25-000003", "2025-03-03", "", "8-K", "ev.htm", "1.01,9.01"),
    ]
)
OLD_IN_RANGE = block([("0000000001-19-000002", "2019-02-20", "2018-12-31", "10-K", "old10k.htm", "")])


def make_client(routes):
    session = FakeSession(router=lambda url, params: json_response(routes[url]))
    return EdgarClient("t me@example.com", cache_dir=None, session=session, sleep=lambda s: None), session


def submissions(files):
    return {"name": "TEST CO", "tickers": ["TST"], "filings": {"recent": RECENT, "files": files}}


def test_filters_by_date_and_form_and_reads_overlapping_pages():
    base = "https://data.sec.gov/submissions/"
    routes = {
        base + "CIK0000000001.json": submissions(
            [
                {"name": "p1.json", "filingFrom": "2018-01-01", "filingTo": "2020-12-31"},
                {"name": "p0.json", "filingFrom": "2001-01-01", "filingTo": "2017-12-31"},
            ]
        ),
        base + "p1.json": OLD_IN_RANGE,
    }
    client, session = make_client(routes)
    got = list_filings(client, 1, "2019-01-01", "2025-12-31", forms={"10-K", "8-K"})
    assert [(f.form, f.filing_date) for f in got] == [("10-K", "2019-02-20"), ("8-K", "2025-03-03")]
    assert got[1].items == "1.01,9.01"
    fetched = [c[0] for c in session.calls]
    assert base + "p0.json" not in fetched  # page entirely before the window is skipped


def test_filing_url_strips_accession_dashes():
    f = Filing(1318605, "0000950170-24-002390", "10-K", "2024-01-29", "2023-12-31", "tsla-20231231.htm", "")
    assert f.url == "https://www.sec.gov/Archives/edgar/data/1318605/000095017024002390/tsla-20231231.htm"


def test_ticker_map():
    routes = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."}
        }
    }
    client, _ = make_client(routes)
    assert load_ticker_map(client) == {"TSLA": (1318605, "Tesla, Inc.")}
