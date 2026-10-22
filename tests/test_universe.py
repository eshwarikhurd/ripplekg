import pytest

from ripplekg.ontology.universe import UniverseError, load_universe, parse_universe


def row(**kw):
    base = {
        "id": "A",
        "name": "A Corp",
        "tier": "automaker",
        "cik": 1,
        "ticker": "A",
        "search_names": ["A Corp"],
    }
    base.update(kw)
    return base


def test_project_universe_is_valid():
    u = load_universe()
    assert len(u.filers) >= 30
    assert "TSLA" in u.by_id and u.by_id["TSLA"].is_filer
    assert not u.by_id["CATL"].is_filer


def test_non_filer_has_no_cik():
    u = parse_universe({"companies": [row(id="X", cik=None, ticker=None)]})
    assert not u.companies[0].is_filer


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([row(), row(name="dup")], "duplicate id A"),
        ([row(), row(id="B")], "cik 1 already used by A"),
        ([row(tier="bakery")], "unknown tier"),
        ([row(cik=None)], "has a ticker but no cik"),
        ([row(search_names=[])], "search_names is empty"),
        ([row(successor="ZZZ")], "successor 'ZZZ' is not in the universe"),
    ],
)
def test_validation_errors(rows, message):
    with pytest.raises(UniverseError, match=message):
        parse_universe({"companies": rows})


def test_missing_field_is_reported():
    with pytest.raises(UniverseError, match="missing field 'tier'"):
        parse_universe({"companies": [{"id": "A", "name": "A", "search_names": ["A"]}]})


def test_non_us_companies_have_verified_home_tickers():
    u = load_universe()
    assert u.by_id["CATL"].home_ticker == "300750.SZ"
    assert u.by_id["BYD"].other_tickers == ("1211.HK",)
    assert u.by_id["NORTHVOLT"].listing == "private" and u.by_id["NORTHVOLT"].home_ticker is None
    assert u.by_id["TSLA"].listing == "public" and u.by_id["TSLA"].home_ticker is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"listing": "unicorn"}, "listing 'unicorn'"),
        ({"cik": None, "ticker": None, "listing": "private", "home_ticker": "X.DE"}, "has no home ticker"),
        ({"cik": None, "ticker": None, "home_ticker": 6752}, "must be a string"),
    ],
)
def test_listing_validation(extra, message):
    with pytest.raises(UniverseError, match=message):
        parse_universe({"companies": [row(**extra)]})
