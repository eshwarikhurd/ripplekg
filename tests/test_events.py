from datetime import date

import pytest

from ripplekg.events import DEV_COUNTS, EventsError, assign_split, load_events, parse_events
from ripplekg.universe import load_universe

UNIVERSE = load_universe()


def ev(**kw):
    base = {
        "id": "x-2024",
        "title": "X",
        "bucket": "primary",
        "origin": ["TSLA"],
        "shock_type": "demand_shock",
        "sign": "negative",
        "announced": "2024-03-05",
        "timing": "before_open",
        "day0": "2024-03-05",
        "named": ["TSLA"],
        "sources": ["https://example.com/a"],
        "split": "test",
    }
    base.update(kw)
    return base


def parse(*rows):
    return parse_events({"events": list(rows)}, UNIVERSE)


def test_project_events_are_valid_and_split_matches_preregistration():
    events = load_events()
    primary = [e for e in events if e.bucket == "primary"]
    assert len(primary) >= 20
    expected = assign_split(events)
    assert {e.id: e.split for e in primary} == expected
    assert sum(1 for s in expected.values() if s == "dev") == sum(DEV_COUNTS.values())


def test_project_events_never_score_their_origin_companies():
    for e in load_events():
        companies = [o for o in e.origin if ":" not in o]
        assert set(companies) <= set(e.named), e.id


def test_split_is_deterministic_and_seed_dependent():
    events = load_events()
    assert assign_split(events) == assign_split(events)
    assert assign_split(events, seed=1) != assign_split(events)


def test_parses_dates_and_origin_kind():
    (e,) = parse(ev(origin=["TSLA", "material:lithium"]))
    assert e.day0 == date(2024, 3, 5)
    assert e.origin_kind == "non_company"
    assert parse(ev())[0].origin_kind == "company"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (ev(origin=["NOPE"]), "origin company 'NOPE' is not in the universe"),
        (ev(origin=["planet:mars"]), "unknown kind 'planet'"),
        (ev(named=["NOPE"]), "named company 'NOPE'"),
        (ev(day0="2024-03-09"), "is a weekend"),
        (ev(day0="2024-03-04"), "before announced"),
        (ev(timing="after_close"), "day0 equals announced"),
        (ev(shock_type="vibes"), "shock_type 'vibes'"),
        (ev(split=None), "needs split"),
        (ev(bucket="macro", split="dev"), "macro events are not split"),
        (ev(sources=["http://insecure.example"]), "https source"),
        (ev(id="Bad_Id"), "lowercase-hyphenated"),
        (ev(announced="March 5"), "not an ISO date"),
    ],
)
def test_validation_errors(row, message):
    with pytest.raises(EventsError, match=message):
        parse(row)


def test_duplicate_ids_rejected():
    with pytest.raises(EventsError, match="duplicate event id"):
        parse(ev(), ev())


def test_weekdays_between():
    from ripplekg.events import weekdays_between

    fri, mon = date(2023, 10, 20), date(2023, 10, 23)
    assert weekdays_between(fri, mon) == 1
    assert weekdays_between(fri, date(2023, 10, 27)) == 5
    assert weekdays_between(fri, fri) == 0
    assert weekdays_between(mon, fri) == -1


def test_overlapping_windows_rejected():
    a = ev(id="a-2023", origin=["TSLA"], announced="2023-10-20", day0="2023-10-20")
    b = ev(id="b-2023", origin=["TSLA"], announced="2023-10-23", day0="2023-10-23")
    with pytest.raises(EventsError, match="b-2023: event window overlaps a-2023"):
        parse(a, b)
    c = ev(id="c-2023", origin=["TSLA"], announced="2023-10-26", day0="2023-10-26")
    assert len(parse(a, c)) == 2  # four weekdays apart: windows do not overlap
