import pytest

from ripplekg import settings


def test_user_agent_requires_email(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "RippleKG research")
    with pytest.raises(settings.SettingsError, match="SEC_USER_AGENT"):
        settings.sec_user_agent()


def test_user_agent_ok(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "RippleKG research me@example.com")
    assert settings.sec_user_agent() == "RippleKG research me@example.com"
