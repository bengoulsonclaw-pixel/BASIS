"""RV Trade Tickets turn the Curve / RV book into backtested, sized tickets.

The numbers that must not drift are the ones a trader would act on:

  * a fade episode is counted ONCE per stretched stint, entered on a fresh crossing, and
    followed to a FROZEN target / stop — not re-entered every day |z| stays beyond the band;
  * a day that gaps through both levels is scored as a loss (stop first), never a free win;
  * sizing is honest — a reconciled point value gives a lot count, a single-market curve is
    DV01-weighted off the editable CTD table, and anything needing an FX or notional ratio we
    don't hold is returned "manual" with no fabricated contract count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import curvemon, futyield, rvtickets


# ── fade-rule backtest ────────────────────────────────────────────────────────
def _series(segments: list[float]) -> pd.Series:
    return pd.Series(segments, index=pd.bdate_range("2016-01-01", periods=len(segments)))


def _warm(window: int, reps: int = 2) -> list[float]:
    """±1 noise → rolling mean ≈ 0, std ≈ 1.2 once a spike enters the window, so a push to
    +3.0 lands at z ≈ 2.4 — inside the [threshold, INVAL_SIGMA) band where the stop still
    sits ABOVE the entry."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(window * reps)]


def test_a_clean_reversion_is_a_win_and_pays_positive_expectancy():
    """A spread that spikes to ~+2.4σ and falls back through its mean is the textbook fade:
    target (the mean) is reached before the +3σ stop, so it must score as a win with a
    positive per-trade expectancy."""
    window = 20
    seg = _warm(window)
    for _ in range(3):                                   # three identical, cleanly reverting spikes
        seg += [3.0] * 6 + [1.0, 0.0, 0.0] + _warm(window, 1)
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] >= 2
    assert bt["wins"] == bt["n"] and bt["losses"] == 0          # every spike reverted
    assert bt["win_rate"] == 1.0
    assert bt["expectancy_sigma"] > 0
    assert bt["median_days_win"] == bt["median_days_win"]       # finite


def test_a_runaway_stretch_is_a_loss_not_a_win():
    """Fading works until it doesn't: a spread that keeps going once stretched hits the stop.
    The rule must book that as a loss with negative expectancy, or the screen would flatter
    every trend."""
    window = 20
    seg = _warm(window) + [3.0, 4.5, 6.0, 8.0, 10.0]            # crosses +2σ then runs away
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] >= 1
    assert bt["losses"] >= 1 and bt["wins"] == 0
    assert bt["win_rate"] == 0.0
    assert bt["expectancy_sigma"] < 0


def test_one_long_stretched_stint_is_a_single_episode():
    """The whole point of entering on a FRESH crossing and holding one position: a spread that
    sits beyond the band for forty sessions is one trade, not forty. Counting every day would
    manufacture a sample out of a single event."""
    window = 20
    seg = _warm(window) + [3.0] * 40 + [1.0, 0.0, 0.0] + _warm(window, 1)
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=12.0)
    assert bt["n"] == 1


def test_a_stretch_that_never_resolves_times_out_rather_than_scoring():
    """A spread that stays wedged between target and stop past the hold cap is marked out at
    the market as a timeout — a non-win that is neither a free win nor a stop-out."""
    window = 20
    seg = _warm(window) + [3.0] * 60                           # held ~60 bars; cap at 5×10 = 50
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] == 1 and bt["timeouts"] == 1 and bt["wins"] == 0


def test_an_unresolved_tail_is_dropped_not_counted():
    """A fresh stretch in the last few sessions can't be followed to target OR stop, so it
    must be excluded — counting it would bias the sample toward whatever the data happens to
    end on."""
    window = 20
    seg = _warm(window) + [3.0, 3.0]                           # crosses at the very end, no room
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] == 0


# ── sizing: the three honest regimes ──────────────────────────────────────────
def test_calendar_sizes_one_lot_per_leg_off_the_reconciled_point_value():
    """Same-product calendars reconcile to a single $/unit, so they get a real lot count —
    and a rich spread sells the front, buys the second."""
    spec = {"key": "x_cal", "group": "STIR Calendars", "unit": "bp", "dp": 1, "scale": 100.0,
            "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")]}
    row = {"level": 20.0, "invalidation": 25.0, "sigma": 5.0, "direction": -1,
           "dollar_sigma": 125.0, "dsig_sym": "$"}
    t = rvtickets.size_ticket(spec, row, risk_budget=2500.0)
    assert t["method"] == "calendar"
    assert t["lots"] == 20                                   # risk 2500 / (5 unit × $25/unit)
    assert t["risk_money"] == pytest.approx(2500.0)
    sides = {lg["label"]: lg["side"] for lg in t["structure_legs"]}
    assert sides == {"front": "Sell", "2nd": "Buy"}          # fade a rich calendar


def test_single_market_curve_is_dv01_weighted_into_two_futures():
    """A 2s10s ticket must read as two real futures in a DV01 ratio off the editable CTD
    table — buy the long end / sell the short end to fade a steep (rich) curve."""
    spec = curvemon.SPREAD_BY_KEY["us_2s10s"]
    assert rvtickets._is_single_market_curve(spec) is True
    row = {"level": 58.0, "invalidation": 78.0, "sigma": 10.0, "direction": -1,
           "dollar_sigma": None, "dsig_sym": "$"}
    raw_last = {"TUA Comdty": 104.0, "TYA Comdty": 110.0}
    t = rvtickets.size_ticket(spec, row, risk_budget=10_000.0,
                              raw_last=raw_last, ctd=futyield.load_ctd())
    assert t["method"] == "curve_dv01"
    sides = {lg["label"]: lg["side"] for lg in t["structure_legs"]}
    assert sides.get("TY") == "Buy" and sides.get("TU") == "Sell"   # flattener
    assert all(lg["lots"] >= 1 for lg in t["structure_legs"])
    assert t["per_bp"] > 0 and t["risk_money"] > 0


def test_boxes_and_ratios_refuse_to_fake_a_contract_count():
    """A cross-currency box needs an FX ratio and a metal ratio needs a notional split —
    neither of which we hold — so both come back 'manual', never a made-up lot count (the same
    discipline as curvemon._dollar_sigma)."""
    base = {"level": 10.0, "invalidation": 16.0, "sigma": 2.0, "dollar_sigma": None,
            "direction": -1, "dsig_sym": "$"}
    box = rvtickets.size_ticket(curvemon.SPREAD_BY_KEY["box_2s10s"], base, 10_000.0)
    assert box["method"] == "manual" and box["lots"] is None

    ratio = rvtickets.size_ticket(curvemon.SPREAD_BY_KEY["gc_si"], base, 10_000.0)
    assert ratio["method"] == "manual" and "ratio" in ratio["note"].lower()


def test_no_stop_distance_sizes_to_nothing():
    """If the invalidation sits on the entry there is no risk to size against — the sizer must
    say so, not divide by zero."""
    spec = {"key": "x_cal", "group": "STIR Calendars", "unit": "bp", "dp": 1, "scale": 100.0,
            "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")]}
    row = {"level": 25.0, "invalidation": 25.0, "sigma": 5.0, "direction": -1,
           "dollar_sigma": 125.0, "dsig_sym": "$"}
    assert rvtickets.size_ticket(spec, row, 2500.0)["method"] == "none"


# ── the assembled, ranked book (needs the deep store) ─────────────────────────
def test_tickets_rank_by_edge_and_carry_a_full_ticket():
    """End to end on whatever history the machine holds: the book must come back sorted best
    edge first, and every ticket must carry the levels and structure a desk would act on."""
    hist = curvemon.load_history()
    if hist is None or hist.empty:
        pytest.skip("no deep store on this machine")
    ts = rvtickets.tickets(window=252, threshold=1.0, history=hist)
    assert isinstance(ts, list)
    if not ts:
        pytest.skip("calm day — nothing stretched beyond 1σ")
    assert [t["score"] for t in ts] == sorted((t["score"] for t in ts), reverse=True)
    for t in ts:
        for k in ("name", "level", "objective", "invalidation", "structure", "action",
                  "rr", "ev_sigma", "bt_n", "conf", "risk_units"):
            assert k in t
        assert t["risk_units"] > 0
        assert t["structure"]
        assert t["size_method"] in {"calendar", "curve_dv01", "manual", "none"}
