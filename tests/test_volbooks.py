"""The vol book's skew-z and term-structure legs must reach the Hot Sheet.

Both are computed every morning and, until 2026-08-26, neither had a provider: hotsheet's
discover() scans only the TOP level of src/, so anything in src/strategies/ is invisible to
it. That silence covered 8 flagged skew markets and 4 flagged term markets on the day it was
found — including two vol books agreeing on one product, which is the corroboration the sheet
exists to surface.

These rows reach the client-facing Morning Coffee page, so the prose lock matters as much as
the plumbing: the underlying stores label their signals "Rich — sell skew", and that wording
must never reach a client document.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src import volbooks

BANNED = re.compile(r"\b(buy|sell|recommend|should|target|advise)\b", re.I)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(volbooks, "SIG", tmp_path)
    pd.DataFrame([
        # market, ticker, put/atm/call vols, z, pctl, direction (-1 rich, +1 cheap)
        {"market": "Brazilian Real", "ticker": "BRA Curncy", "put": 10.5, "atm": 11.1,
         "call": 12.1, "z": 3.02, "pctl": 100, "direction": -1, "signal": "Rich — sell skew"},
        {"market": "Feeder Cattle", "ticker": "FCA Comdty", "put": 19.7, "atm": 16.6,
         "call": 19.3, "z": -2.49, "pctl": 0, "direction": 1, "signal": "Cheap — buy skew"},
        {"market": "Quiet Thing", "ticker": "QQ Comdty", "put": 1.0, "atm": 1.0,
         "call": 1.0, "z": 0.1, "pctl": 50, "direction": 0, "signal": "—"},
    ]).to_parquet(tmp_path / "skew.parquet")
    pd.DataFrame([
        {"market": "Iron Ore", "ticker": "SCOA Comdty", "iv_1m": 11.8, "iv_3m": 15.4,
         "z": 2.31, "pctl": 94, "direction": 1, "signal": "Steep — front cheap"},
        {"market": "Live Cattle", "ticker": "LCA Comdty", "iv_1m": 19.2, "iv_3m": 18.1,
         "z": -1.87, "pctl": 6, "direction": -1, "signal": "Inverted"},
    ]).to_parquet(tmp_path / "termstructure.parquet")
    return tmp_path


def test_both_books_emit(stores):
    items = volbooks.radar_items()
    tags = [it["tag"] for it in items]
    assert "SKEW-Z" in tags and "TERM" in tags
    assert all(it["book"] == "ficc" for it in items)


def test_unflagged_rows_are_left_out(stores):
    """direction == 0 is the book saying "nothing here" — it must not reach the sheet."""
    assert not any("Quiet Thing" in it["text"] for it in volbooks.radar_items())


def test_prose_is_client_safe(stores):
    """The stores' own signal strings say "sell skew" / "buy skew". None of that may
    survive into a row that reaches a client PDF."""
    for it in volbooks.radar_items():
        assert not BANNED.search(it["text"]), f"advice language in: {it['text']}"
        assert "**" in it["text"], "the product name must be bold, as every provider does"


def test_rich_and_cheap_are_the_right_way_round(stores):
    """direction -1 is the book's "rich"; getting this backwards would invert every row."""
    by = {it["key"]: it["text"] for it in volbooks.radar_items()}
    assert "screens **rich**" in by["SKEW-Z:BRA Curncy:rich"]
    assert "screens **cheap**" in by["SKEW-Z:FCA Comdty:cheap"]


def test_heat_tracks_the_z_and_does_not_saturate(stores):
    """A z-based heat must discriminate. The percentile version saturates at 100 for BOTH
    tails, which is how a provider ends up pinning every row at the top of the sheet."""
    heats = {it["key"]: it["heat"] for it in volbooks.radar_items()}
    assert heats["SKEW-Z:BRA Curncy:rich"] > heats["SKEW-Z:FCA Comdty:cheap"]
    assert all(h < 100 for h in heats.values())


def test_provider_never_raises_without_stores(tmp_path, monkeypatch):
    """The provider contract: a missing store means quiet, not a traceback."""
    monkeypatch.setattr(volbooks, "SIG", tmp_path / "nothing")
    assert volbooks.radar_items() == []
