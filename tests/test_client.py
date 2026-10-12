import pytest
from fakes import FakeClock, FakeResponse, FakeSession, json_response

from ripplekg.download.client import EdgarClient, EdgarError, RateLimiter

URL = "https://data.sec.gov/x.json"


def make_client(session, tmp_path=None, **kw):
    return EdgarClient(
        "RippleKG test me@example.com", cache_dir=tmp_path, session=session, sleep=lambda s: None, **kw
    )


def test_rate_limiter_spaces_calls():
    clock = FakeClock()
    limiter = RateLimiter(rate=4.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.sleeps == [0.25, 0.25]


def test_rate_limiter_does_not_sleep_when_calls_are_slow():
    clock = FakeClock()
    limiter = RateLimiter(rate=4.0, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 1.0
    limiter.wait()
    assert clock.sleeps == []


def test_sends_user_agent():
    session = FakeSession(routes={URL: [json_response({"a": 1})]})
    assert make_client(session).get_json(URL) == {"a": 1}
    assert session.calls[0][2]["User-Agent"] == "RippleKG test me@example.com"


def test_cache_avoids_second_request(tmp_path):
    session = FakeSession(routes={URL: [json_response({"a": 1})]})
    client = make_client(session, tmp_path)
    assert client.get_json(URL, params={"q": "x"}) == {"a": 1}
    assert client.get_json(URL, params={"q": "x"}) == {"a": 1}
    assert len(session.calls) == 1
    assert client.network_calls == 1


def test_cache_key_includes_params(tmp_path):
    session = FakeSession(routes={URL: [json_response({"a": 1}), json_response({"a": 2})]})
    client = make_client(session, tmp_path)
    assert client.get_json(URL, params={"q": "x"}) == {"a": 1}
    assert client.get_json(URL, params={"q": "y"}) == {"a": 2}


def test_retries_on_429_then_succeeds():
    session = FakeSession(routes={URL: [FakeResponse(429), FakeResponse(503), json_response({"ok": True})]})
    assert make_client(session).get_json(URL) == {"ok": True}
    assert len(session.calls) == 3


def test_raises_on_404_without_retrying():
    session = FakeSession(routes={URL: [FakeResponse(404)]})
    with pytest.raises(EdgarError, match="HTTP 404"):
        make_client(session).get_json(URL)
    assert len(session.calls) == 1


def test_gives_up_after_max_retries():
    session = FakeSession(routes={URL: [FakeResponse(429)] * 3})
    with pytest.raises(EdgarError, match="HTTP 429"):
        make_client(session, max_retries=2).get_json(URL)
    assert len(session.calls) == 3


def test_failed_request_is_not_cached(tmp_path):
    session = FakeSession(routes={URL: [FakeResponse(404), json_response({"a": 1})]})
    client = make_client(session, tmp_path)
    with pytest.raises(EdgarError):
        client.get_json(URL)
    assert client.get_json(URL) == {"a": 1}
