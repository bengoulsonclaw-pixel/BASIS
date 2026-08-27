"""The FICC product tearsheet assembles eight desk books onto one client-facing page.

Two things can go wrong that a rendered PDF will not shout about, so both are locked here.

The first is prose. Every store on this page is written for the DESK and carries the trade to
put on — "Cheap — buy skew", "Short (sell the rally)", "· sell the bond". Client copy is
neutral observation, never advice, and this is the first report that renders store strings
verbatim rather than feeding numbers to the AI writer, so the scrubber is the only thing
standing between those strings and a client's inbox.

The second is the headline level. The stores hold two different generics — the deep store's
'1' (front contract) and the snapshot's 'A' (most active) — and mid-roll they are different
contracts: on 2026-08-27 coffee sat 35.45 points apart and SOFR 32bp. Every technical read on
the page is computed on get_ta, so the big number at the top has to come from the same series,
or the sheet contradicts its own table.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src import ficcsheet
from src.reportkit import client_safe

# "target"/"should" are deliberately NOT banned: the technical stores quote a measured
# projection ("target 377.75, R:R 3.0") that the Technical Analysis Report already shows
# clients. What is banned is telling someone to transact.
BANNED = re.compile(r"\b(buy|sell|bought|sold|recommend\w*|advise\w*)\b", re.I)


# ── the scrubber ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw, want", [
    ("Cheap — buy skew", "Cheap"),
    ("Rich — sell vol", "Rich"),
    ("Long (buy the dip) · sell the bond", "Long"),
    ("Short (sell the rally) · buy the future", "Short"),
    ("Wave 3 underway ▲ · sell the bond", "Wave 3 underway ▲"),
    ("Resistance 359.25 (3 touches) 0.5% above — sell-rally zone",
     "Resistance 359.25 (3 touches) 0.5% above"),
    ("Support 5.13 (4 touches) 0.6% below — buy-dip zone",
     "Support 5.13 (4 touches) 0.6% below"),
    ("mean 0.23 ± 0.01, half-life ≈ 69d — Buy Soybean Oil / Sell Soybean Meal",
     "mean 0.23 ± 0.01, half-life ≈ 69d"),
])
def test_client_safe_strips_the_instruction_and_keeps_the_observation(raw, want):
    got = client_safe(raw)
    assert got == want
    assert not BANNED.search(got)


def test_client_safe_refuses_rather_than_ships_unknown_advice():
    """A phrase the map doesn't cover must fail loudly. Silently letting it through is how
    advice reaches a client page, and the caller can extend the map in one line."""
    with pytest.raises(ValueError):
        client_safe("Buy Soybean Oil / Sell Soybean Meal")
    with pytest.raises(ValueError):
        client_safe("we recommend buying the front contract")


def test_client_safe_keeps_an_analyst_rating():
    """'Strong Buy' here is the NAME of a broker's rating — a reportable fact, not us telling
    a client to transact. Blocking it would drop real content off the equities sheet."""
    out = client_safe("**AMD**: Raymond James moved to **Strong Buy** from Outperform (25 Aug).")
    assert "Strong Buy" in out


# ── the page ─────────────────────────────────────────────────────────────────
@pytest.fixture
def stores(tmp_path, monkeypatch):
    """A minimal set of daily stores for one product, with the desk wording left in."""
    monkeypatch.setattr(ficcsheet, "SIG", tmp_path)
    pd.DataFrame([
        {"strategy": "Support & Resistance", "market": "Coffee", "instruments": "KCA Comdty",
         "signal": "Short (sell the rally)", "direction": -1, "metric": -76.9,
         "metric_label": "level proximity",
         "context": "Resistance 359.25 (3 touches) 0.5% above — sell-rally zone"},
        {"strategy": "Trend", "market": "Coffee", "instruments": "KCA Comdty",
         "signal": "Long", "direction": 1, "metric": 42.6, "metric_label": "3m return %",
         "context": "MA20 vs MA100: +16.5%"},
        # already has its own Options section further down the page
        {"strategy": "Skew Volatility", "market": "Coffee", "instruments": "KCA Comdty",
         "signal": "Cheap — buy skew", "direction": 1, "metric": -2.24,
         "metric_label": "skew z (1y)", "context": "put 40.6 / call 45.4 / ATM 42.9"},
    ]).to_parquet(tmp_path / "opportunities.parquet")
    pd.DataFrame([{"market": "Coffee", "ticker": "KCA Comdty", "asset": "Softs", "region": "",
                   "iv": 42.9, "rv": 43.7, "spread": -0.8, "z": 0.09, "pctl": 45.0,
                   "signal": "—", "direction": 0, "iv_sd": 9.66, "px_dec": 2.0}]
                 ).to_parquet(tmp_path / "volatility.parquet")
    pd.DataFrame([{"market": "Coffee", "ticker": "KCA Comdty", "asset": "Softs", "region": "",
                   "put": 40.6, "call": 45.4, "atm": 42.9, "skew": -4.8, "z": -2.24,
                   "pctl": 1.0, "signal": "Cheap — buy skew", "direction": 1}]
                 ).to_parquet(tmp_path / "skew.parquet")
    return tmp_path


def test_a_book_with_its_own_section_is_not_also_listed_as_a_technical(stores):
    """Skew appears in the Options table. Repeating it in the technical column would say the
    same thing twice AND change dimension mid-column: "Skew Volatility — Long" means long
    SKEW, not long coffee, which a client reading a column of Long/Short cannot know."""
    tech = ficcsheet._tech_block("KCA Comdty")
    names = {r["strategy"] for r in tech["rows"]}
    assert "Skew Volatility" not in names
    assert names == {"Support & Resistance", "Trend"}
    assert (tech["up"], tech["down"]) == (1, 1)


def test_no_technical_row_carries_desk_instructions(stores):
    tech = ficcsheet._tech_block("KCA Comdty")
    for r in tech["rows"]:
        assert not BANNED.search(r["context"]), r
        assert r["read"] in {"Long", "Short", "Neutral"}


def test_the_options_table_states_cheap_without_saying_buy(stores):
    rows = ficcsheet._vol_block("KCA Comdty")["rows"]
    skew = [r for r in rows if r["label"].startswith("Skew")][0]
    assert skew["note"] == "Cheap"
    assert not BANNED.search(skew["note"])


def test_a_missing_book_is_left_out_not_drawn_empty(stores):
    """putcall drops products whose option legs are too thin to quote, and cot only covers the
    CFTC-reported markets. Absent is normal; it must not raise or render a hollow table."""
    assert ficcsheet._pos_block("KCA Comdty") is None
    assert ficcsheet._seas_block("KCA Comdty") is None
    assert ficcsheet._mentions("KCA Comdty") == []


def test_internal_only_highlights_never_reach_the_sheet(stores, tmp_path):
    """The Hot Sheet marks some rows internal_only precisely because they are not for
    clients; this page is a client document, so that flag has to be honoured here too."""
    pd.DataFrame([
        {"date": pd.Timestamp("2026-08-26"), "key": "a", "tag": "VOL", "text": "public row",
         "ticker": "KCA Comdty", "internal_only": False},
        {"date": pd.Timestamp("2026-08-26"), "key": "b", "tag": "FLOW", "text": "desk eyes only",
         "ticker": "KCA Comdty", "internal_only": True},
    ]).to_parquet(tmp_path / "hotsheet_history.parquet")
    got = ficcsheet._mentions("KCA Comdty")
    assert [m["text"] for m in got] == ["public row"]


def test_the_headline_level_comes_from_the_series_the_page_is_computed_on(monkeypatch):
    """get_ta, not get_raw and not the snapshot. Mid-roll the generics are different
    contracts — coffee's front and most-active sat 35.45 points apart on 2026-08-27 — and the
    technical rows underneath ("65% up the [319.5 - 377.75] channel") are only true of one."""
    idx = pd.bdate_range("2026-08-20", periods=4)
    monkeypatch.setattr("src.deepstore.get_ta",
                        lambda tk, *a, **k: pd.DataFrame({tk[0]: [350.0, 352.0, 371.4, 357.6]},
                                                         index=idx))
    monkeypatch.setattr("src.deepstore.get_raw",
                        lambda tk, *a, **k: pd.DataFrame({tk[0]: [1.0, 2.0, 3.0, 322.15]},
                                                         index=idx))
    last, chg = ficcsheet._level("KCA Comdty")
    assert last == pytest.approx(357.6)
    assert chg == pytest.approx(-13.8)


def test_fixed_income_is_quoted_in_yield_and_moves_in_basis_points(monkeypatch, stores):
    """The TA engine runs on yields for fixed income, so the sheet does too — and a percent
    change OF a yield is meaningless, so the move belongs in bp."""
    monkeypatch.setattr(ficcsheet.u, "is_fixed_income", lambda t: True)
    monkeypatch.setattr(ficcsheet, "_level", lambda t: (4.6462, 0.018))
    monkeypatch.setattr(ficcsheet, "_price_png", lambda *a: "")
    monkeypatch.setattr(ficcsheet, "_vol_png", lambda *a: "")
    d = ficcsheet.gather("TYA Comdty")
    assert d["last_txt"] == "4.646%"
    assert d["chg_txt"] == "+1.8 bp"
    # iv_sd is in futures PRICE points; converting it against a yield move needs the
    # contract's DV01, which is in no store here, so the multiple is withheld rather than
    # printed wrong.
    assert d["sd_mult"] is None
