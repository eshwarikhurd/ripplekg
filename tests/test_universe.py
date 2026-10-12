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
