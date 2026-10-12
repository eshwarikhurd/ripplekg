from datetime import date

import pandas as pd
import pytest

from ripplekg.download.corpus import selection, summary_table
from ripplekg.download.index import Filing


def filing(form="10-K", items="", accepted="", filing_date="2024-03-05"):
    return Filing(1, "0000000001-24-000001", form, filing_date, "", "doc.htm", items, accepted)


@pytest.mark.parametrize(
    ("form", "items", "selected", "reason"),
    [
        ("10-K", "", True, "annual report"),
        ("20-F", "", True, "annual report"),
        ("40-F", "", True, "annual report"),
        ("10-Q", "", True, "quarterly report"),
        ("8-K", "1.01,9.01", True, "8-K items 1.01"),
        ("8-K", "2.06,7.01,9.01", True, "8-K items 2.06,7.01"),
        ("8-K", "2.02,9.01", False, "8-K items not selected"),
        ("8-K", "5.07", False, "8-K items not selected"),
        ("10-K/A", "", False, "amendment"),
        ("6-K", "", False, "6-K (no item codes)"),
    ],
)
def test_selection(form, items, selected, reason):
    assert selection(filing(form, items)) == (selected, reason)


def test_acceptance_time_converts_utc_to_eastern_across_dst():
    summer = filing(accepted="2021-08-23T20:22:32.000Z")  # EDT, UTC-4
    winter = filing(accepted="2024-01-11T14:00:00.000Z")  # EST, UTC-5
    assert summer.accepted_et.strftime("%Y-%m-%d %H:%M") == "2021-08-23 16:22"
    assert winter.accepted_et.strftime("%H:%M") == "09:00"


@pytest.mark.parametrize(
    ("accepted", "day", "expected"),
    [
        ("2024-03-05T13:29:00.000Z", date(2024, 3, 5), True),  # 08:29 EST, before the open
        ("2024-03-05T14:31:00.000Z", date(2024, 3, 5), False),  # 09:31 EST, after the open
        ("2024-03-04T21:00:00.000Z", date(2024, 3, 5), True),  # previous evening
        ("2024-03-06T12:00:00.000Z", date(2024, 3, 5), False),  # next day
    ],
)
def test_information_cutoff(accepted, day, expected):
    assert filing(accepted=accepted).public_before_open(day) is expected


def test_information_cutoff_without_acceptance_time_uses_filing_date():
    assert filing(filing_date="2024-03-04").public_before_open(date(2024, 3, 5))
    assert not filing(filing_date="2024-03-05").public_before_open(date(2024, 3, 5))


def test_summary_table_counts_selected_filings_by_group():
    df = pd.DataFrame(
        {
            "company_id": ["A", "A", "A", "A", "B"],
            "form": ["10-K", "10-Q", "8-K", "8-K", "20-F"],
            "accession": list("12345"),
            "selected": [True, True, True, False, True],
        }
    )
    t = summary_table(df).set_index("company_id")
    assert t.loc["A"].to_dict() == {"annual": 1, "10-Q": 1, "8-K": 1, "total": 3}
    assert t.loc["B"].to_dict() == {"annual": 1, "10-Q": 0, "8-K": 0, "total": 1}
