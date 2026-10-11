from fakes import FakeSession, json_response

from ripplekg.edgar.client import EdgarClient
from ripplekg.edgar.fulltext import build_params, count_hits


def test_build_params_pads_ciks_and_sets_custom_range():
    p = build_params(
        '"Panasonic"', ciks=[1318605], forms=["10-K", "8-K"], start="2019-01-01", end="2025-12-31"
    )
    assert p == {
        "q": '"Panasonic"',
        "ciks": "0001318605",
        "forms": "10-K,8-K",
        "dateRange": "custom",
        "startdt": "2019-01-01",
        "enddt": "2025-12-31",
    }


def test_build_params_minimal():
    assert build_params("offtake") == {"q": "offtake"}


def _client(total):
    session = FakeSession(router=lambda url, params: json_response({"hits": {"total": total, "hits": []}}))
    return EdgarClient("t me@example.com", cache_dir=None, session=session, sleep=lambda s: None)


def test_count_hits_exact():
    hits = count_hits(_client({"value": 35, "relation": "eq"}), '"Panasonic"')
    assert (hits.value, hits.exact) == (35, True)


def test_count_hits_lower_bound():
    hits = count_hits(_client({"value": 10000, "relation": "gte"}), "lithium")
    assert (hits.value, hits.exact) == (10000, False)
