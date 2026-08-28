"""The roll board says when each product's front contract rolls and what the roll costs.

Its dates are observed rather than assumed — deep_contract.parquet holds the actual contract
behind the front generic every day since 2016 — so the tests that matter are about the four
judgement calls, each of which was measured rather than guessed:

  * the roll offset is anchored to the EXPIRING contract's month, not the incoming one;
  * a spread is taken off RAW legs on BOTH sides, never a panama-adjusted one;
  * every product is struck on ONE session, not on each column's own last print;
  * an unknown contract size is blank, never a zero that reads as "this roll is free".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import rollboard as rb


# ── decoding a contract symbol ───────────────────────────────────────────────
@pytest.mark.parametrize("sym, ref, want", [
    # two-digit years are unambiguous; the store used them until Bloomberg changed format
    ("KCZ16", "2016-09-21", "2016-12-01"),
    ("CLV26", "2026-08-21", "2026-10-01"),
    # a root carrying a trailing space — corn, wheat, soybeans are all "C ", "W ", "S "
    ("C Z16", "2016-09-01", "2016-12-01"),
    # one digit is ambiguous: '6' is 2016 as readily as 2026, and only the observation
    # date separates them
    ("CLV6", "2026-08-21", "2026-10-01"),
    ("CLV6", "2016-08-21", "2016-10-01"),
])
def test_decode_contract(sym, ref, want):
    got = rb.decode_contract(sym, pd.Timestamp(ref))
    assert got == pd.Timestamp(want)


def test_decode_rejects_a_non_contract():
    assert rb.decode_contract("ESA Index", pd.Timestamp("2026-08-27")) is None
    assert rb.decode_contract("", pd.Timestamp("2026-08-27")) is None


# ── the roll offset ──────────────────────────────────────────────────────────
def _contract_frame(pairs):
    """A daily contract series from [(date, symbol), ...] step changes."""
    idx = pd.bdate_range("2025-01-01", "2026-12-31")
    s = pd.Series(index=idx, dtype=object)
    for d, sym in pairs:
        s.loc[pd.Timestamp(d):] = sym
    return pd.DataFrame({"XX Comdty": s.dropna()})


def test_offset_is_measured_against_the_expiring_contract():
    """The roll is driven by the contract being rolled OUT of, so that is the anchor.

    Across all 4,770 observed rolls the expiring anchor holds a median inter-quartile spread
    of 1 business day with 99% of products inside 5, against 2 days and 83% for the incoming
    contract — because an irregular cycle gap (sugar's Oct->Mar is five months) corrupts the
    incoming anchor and leaves the expiring one untouched.
    """
    # each roll sits ~14 business days into the EXPIRING contract's own month, while the gap
    # to the incoming month varies (Dec -> Mar skips a quarter).
    c = _contract_frame([("2025-01-02", "XXH25"), ("2025-03-20", "XXM25"),
                         ("2025-06-20", "XXU25"), ("2025-09-19", "XXZ25"),
                         ("2025-12-19", "XXH26"), ("2026-03-20", "XXM26")])
    prof = rb.profile("XX Comdty", c)
    assert prof["n"] == 5
    assert prof["offset"] == pytest.approx(13, abs=2)
    assert prof["band"] <= 2


def test_next_roll_projects_forward_from_the_current_front():
    c = _contract_frame([("2025-01-02", "XXH25"), ("2025-03-20", "XXM25"),
                         ("2025-06-20", "XXU25"), ("2025-09-19", "XXZ25"),
                         ("2025-12-19", "XXH26"), ("2026-06-19", "XXU26")])
    nr = rb.next_roll("XX Comdty", today="2026-08-27", contracts=c)
    assert nr["front"] == "XXU26"
    # projected from the front contract's own month plus the measured offset
    assert nr["date"].month == 9 and nr["date"].year == 2026
    assert nr["bd"] > 0                     # a roll that has not happened yet
    assert nr["band"] <= 3


def test_too_few_observed_rolls_yields_no_projection():
    """A product we have barely seen roll gets no date at all, rather than one built on two
    observations and presented with the same confidence as one built on fifty."""
    c = _contract_frame([("2025-01-02", "XXH25"), ("2025-03-20", "XXM25")])
    assert rb.profile("XX Comdty", c) is None
    assert rb.next_roll("XX Comdty", today="2026-08-27", contracts=c) is None


# ── the spread ───────────────────────────────────────────────────────────────
@pytest.fixture
def legs(monkeypatch):
    """Three products: the majority share a session and SLOW lags it by one, mirroring the
    real split where 78 products print one date and the Asian contracts print another."""
    idx = pd.bdate_range("2026-08-20", "2026-08-27")
    p1 = pd.DataFrame({"SLOW Comdty": np.linspace(100.0, 107.0, len(idx)),
                       "FAST Index": np.linspace(200.0, 207.0, len(idx)),
                       "ALSO Index": np.linspace(300.0, 307.0, len(idx))}, index=idx)
    p2 = p1 - 1.0
    p1.loc[idx[-1], "SLOW Comdty"] = np.nan       # SLOW has not printed today
    p2.loc[idx[-1], "SLOW Comdty"] = np.nan
    monkeypatch.setattr("src.deepstore.get_raw", lambda tk, *a, **k: p1[[c for c in tk if c in p1]])
    monkeypatch.setattr("src.deepstore.get_front2", lambda tk, *a, **k: p2[[c for c in tk if c in p2]])
    return idx


def test_every_product_is_struck_on_one_session(legs, monkeypatch):
    """Taking each column's own last print mixes sessions — the Asian contracts run a day
    ahead of the other 78 — so a board built column by column compares yesterday's Europe
    against today's Asia."""
    monkeypatch.setattr(rb, "_is_fi", lambda t: False)
    monkeypatch.setattr("src.volbt.point_value", lambda t: 1.0)
    monkeypatch.setattr("src.volbt.currency", lambda t: "USD")
    out = rb.spreads(["SLOW Comdty", "FAST Index", "ALSO Index"])
    # the session is set by the majority, and everything is clipped to it — nothing is taken
    # from a LATER session than the board is struck on
    assert out["FAST Index"]["asof"] == out["ALSO Index"]["asof"]
    assert out["SLOW Comdty"]["asof"] < out["FAST Index"]["asof"]
    # a product with no print on that session is carried forward and MARKED, not silently
    # mixed in as though it were current
    assert out["SLOW Comdty"]["stale_days"] >= 1
    assert out["FAST Index"]["stale_days"] == 0


def test_an_unknown_contract_size_is_blank_not_zero(legs, monkeypatch):
    """volbt.point_value returns 0.0 for the 19 products absent from its table. Multiplying
    through gives -0.00, which reads as "this roll costs you nothing" when it means "we do
    not know the contract size"."""
    monkeypatch.setattr(rb, "_is_fi", lambda t: False)
    monkeypatch.setattr("src.volbt.point_value", lambda t: 0.0)
    monkeypatch.setattr("src.volbt.currency", lambda t: "USD")
    out = rb.spreads(["FAST Index"])
    assert out["FAST Index"]["cash"] is None


def test_stirs_are_quoted_in_basis_points_and_never_as_percent_of_price(legs, monkeypatch):
    """A STIR price is 100 - rate, so points ARE a rate differential and x100 is basis
    points. "0.16% of the price" answers no question anyone asks."""
    monkeypatch.setattr(rb, "_is_stir", lambda t: True)
    monkeypatch.setattr(rb, "_is_fi", lambda t: True)
    monkeypatch.setattr("src.volbt.point_value", lambda t: 2500.0)
    monkeypatch.setattr("src.volbt.currency", lambda t: "USD")
    out = rb.spreads(["FAST Index"])["FAST Index"]
    assert out["pct"] is None
    assert out["bp"] == pytest.approx(out["pts"] * 100.0)


def test_non_fi_carries_percent_of_front(legs, monkeypatch):
    monkeypatch.setattr(rb, "_is_stir", lambda t: False)
    monkeypatch.setattr(rb, "_is_fi", lambda t: False)
    monkeypatch.setattr("src.volbt.point_value", lambda t: 1.0)
    monkeypatch.setattr("src.volbt.currency", lambda t: "USD")
    out = rb.spreads(["FAST Index"])["FAST Index"]
    assert out["bp"] is None
    assert out["pct"] == pytest.approx(out["pts"] / abs(out["front_px"]) * 100.0)


# ── normalisation: the reason the board carries two z's ──────────────────────
def test_normalised_z_uses_percent_for_non_fi_and_points_for_fixed_income(monkeypatch):
    """Ranking a cross-product board on raw points ranks by which products re-rated. Gold's
    spread reads +1.4 sigma in points and +2.2 in percent, so a points board misses it; the
    -1.79 sigma figure that circulated came from differencing a panama-ADJUSTED front against
    a raw second, a series whose 2y mean spread is +67 against a true -26.
    """
    idx = pd.bdate_range("2024-08-27", "2026-08-27")
    n = len(idx)
    # a front that doubles, with a spread that is constant as a FRACTION of it
    front = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
    second = front * 1.01                      # always 1% contango
    p1 = pd.DataFrame({"RERATED Comdty": front})
    p2 = pd.DataFrame({"RERATED Comdty": second})
    monkeypatch.setattr("src.deepstore.get_raw", lambda tk, *a, **k: p1)
    monkeypatch.setattr("src.deepstore.get_front2", lambda tk, *a, **k: p2)
    monkeypatch.setattr(rb, "_is_fi", lambda t: False)
    z = rb.norm_z(["RERATED Comdty"])
    # in percent the spread never moved, so it cannot be an outlier; in points it grew
    # steadily with the price and today would be the largest reading in the window.
    assert abs(z["RERATED Comdty"]) < 0.5

    monkeypatch.setattr(rb, "_is_fi", lambda t: True)
    z_pts = rb.norm_z(["RERATED Comdty"])
    # in points the spread widened monotonically with the price, so today is the extreme of
    # the window (negative here: the spread is -1% of a front that doubled).
    assert abs(z_pts["RERATED Comdty"]) > 1.5


# ── the Hot Sheet provider ───────────────────────────────────────────────────
def test_a_roll_alone_never_earns_a_hot_sheet_row(monkeypatch):
    """A roll on its own is a diary entry. It takes a roll AND a spread stretched against its
    own season to be worth a call."""
    df = pd.DataFrame([{"ticker": "AA Comdty", "name": "Quiet", "asset": "Softs",
                        "bd": 2, "seas_z": 0.2, "seas_norm": 0.0, "seas_years": 9.0,
                        "pts": 1.0, "unit": "pts", "state": "contango", "cash": 10.0,
                        "ccy": "USD"}])
    monkeypatch.setattr(rb, "board", lambda *a, **k: df)
    assert rb.radar_items() == []

    df.loc[0, "seas_z"] = 3.4
    monkeypatch.setattr(rb, "board", lambda *a, **k: df)
    got = rb.radar_items()
    assert len(got) == 1 and got[0]["tag"] == "ROLL"
    assert "Quiet" in got[0]["text"]


def test_a_distant_roll_is_not_flagged_however_stretched(monkeypatch):
    df = pd.DataFrame([{"ticker": "AA Comdty", "name": "Later", "asset": "Softs",
                        "bd": 40, "seas_z": 6.0, "seas_norm": 0.0, "seas_years": 9.0,
                        "pts": 1.0, "unit": "pts", "state": "contango", "cash": 10.0,
                        "ccy": "USD"}])
    monkeypatch.setattr(rb, "board", lambda *a, **k: df)
    assert rb.radar_items() == []


def test_heat_discriminates_across_the_range_this_book_actually_produces(monkeypatch):
    """Calendar spreads routinely run z of 3-6 against their seasonal, so saturating at the
    usual 4 sigma would tie most of the book at 100 and say nothing about which roll is the
    unusual one."""
    rows = [{"ticker": f"{i} Comdty", "name": f"P{i}", "asset": "Softs", "bd": 2,
             "seas_z": z, "seas_norm": 0.0, "seas_years": 9.0, "pts": 1.0, "unit": "pts",
             "state": "contango", "cash": 10.0, "ccy": "USD"}
            for i, z in enumerate([5.9, 4.7, 3.6, 2.4])]
    monkeypatch.setattr(rb, "board", lambda *a, **k: pd.DataFrame(rows))
    heats = [i["heat"] for i in rb.radar_items()]
    assert heats == sorted(heats, reverse=True)
    assert len(set(round(h) for h in heats)) == len(heats)      # no ties
    assert max(heats) <= 100.0
