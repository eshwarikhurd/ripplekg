"""Project paths and environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
HTTP_CACHE_DIR = DATA_DIR / "cache" / "http"

# Corpus window: filings published in this range feed the graph.
CORPUS_START = "2019-01-01"
CORPUS_END = "2025-12-31"

load_dotenv(ROOT / ".env")


class SettingsError(RuntimeError):
    pass


def sec_user_agent() -> str:
    """SEC requires a User-Agent that identifies the requester with a contact email."""
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise SettingsError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and set it to "
            '"RippleKG research <your email>" (SEC fair-access policy requires a contact email).'
        )
    return ua
