"""Filing index: which filings exist for a company, from the EDGAR submissions API."""

from __future__ import annotations

from dataclasses import dataclass

from ripplekg.edgar.client import EdgarClient

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

ANNUAL_FORMS = ("10-K", "20-F", "40-F")


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    items: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.primary_document
        )


def load_ticker_map(client: EdgarClient) -> dict[str, tuple[int, str]]:
    """ticker -> (cik, SEC company title)."""
    data = client.get_json(TICKER_MAP_URL)
    return {row["ticker"]: (int(row["cik_str"]), row["title"]) for row in data.values()}


def _rows(block: dict, cik: int) -> list[Filing]:
    n = len(block["accessionNumber"])
    items = block.get("items") or [""] * n
    return [
        Filing(
            cik=cik,
            accession=block["accessionNumber"][i],
            form=block["form"][i],
            filing_date=block["filingDate"][i],
            report_date=block["reportDate"][i] or "",
            primary_document=block["primaryDocument"][i],
            items=items[i] or "",
        )
        for i in range(n)
    ]


def company_submissions(client: EdgarClient, cik: int) -> dict:
    return client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))


def list_filings(
    client: EdgarClient,
    cik: int,
    start: str,
    end: str,
    forms: set[str] | None = None,
) -> list[Filing]:
    """All filings for `cik` with start <= filing_date <= end, optionally limited to `forms`.

    The submissions API returns the most recent ~1,000 filings inline and older ones in
    paged files; a page is fetched only if its date range overlaps [start, end].
    """
    sub = company_submissions(client, cik)
    filings = _rows(sub["filings"]["recent"], cik)
    for page in sub["filings"].get("files", []):
        if page["filingTo"] >= start and page["filingFrom"] <= end:
            filings += _rows(client.get_json(SUBMISSIONS_URL.format(name=page["name"])), cik)
    return sorted(
        (f for f in filings if start <= f.filing_date <= end and (forms is None or f.form in forms)),
        key=lambda f: (f.filing_date, f.accession),
    )
