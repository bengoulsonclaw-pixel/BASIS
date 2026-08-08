"""Golden locks on the Vol Backtester's maths (src/volbt.py): Black-76 straddle
greeks, the vol-surface interpolation, the FX quote-convention divisors and the
repo's ONLY contract point-value table (which tabt's $ P&L also rides)."""
from __future__ import annotations

import math
from datetime import date

import pytest

import pandas as pd

from src import universe, volbt


# ── Black-76 straddle ────────────────────────────────────────────────────────
def test_straddle_greeks_atm_golden():
    """F=100, K=100, vol=20, τ=0.25y — hand-verified: d1 = 0.05, call = 100·(2N(.05)−1),
    put = call (parity, r=0), vega = 2·F·φ(.05)·√τ/100 = φ(.05)."""
    g = volbt.straddle_greeks(100.0, 100.0, 20.0, 0.25)
    assert g.value == pytest.approx(7.975522335348984, rel=1e-12)
    assert g.call == pytest.approx(g.put, rel=1e-12)              # ATM: call == put exactly
    assert g.call + g.put == pytest.approx(g.value, rel=1e-12)
    assert g.delta == pytest.approx(0.03987761167674497, rel=1e-12)
    assert g.gamma == pytest.approx(0.07968878281895281, rel=1e-12)
    assert g.vega == pytest.approx(0.398443914094764, rel=1e-12)
    assert g.theta == pytest.approx(-15.937756563790561, rel=1e-12)


def test_straddle_parity_and_monotonicity():
    g = volbt.straddle_greeks(100.0, 110.0, 20.0, 0.25)
    assert g.put - g.call == pytest.approx(10.0, abs=1e-9)        # put − call = K − F (r=0)
    assert g.value == pytest.approx(11.90789478371444, rel=1e-12)
    # more vol / more time -> dearer straddle; vol floor keeps τ→0 finite
    assert volbt.straddle_greeks(100, 100, 30, 0.25).value > g_atm().value
    assert volbt.straddle_greeks(100, 100, 20, 0.5).value > g_atm().value
    assert volbt.straddle_greeks(100, 100, 0.0, 1e-9).value > 0


def g_atm():
    return volbt.straddle_greeks(100.0, 100.0, 20.0, 0.25)


# ── VolSurface: variance-linear tenor interp + 1M smile ──────────────────────
def _surface(term_vols: dict, smile: tuple | None = None) -> volbt.VolSurface:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-01")])
    t = "CLA Comdty"
    term = {lab: pd.DataFrame({t: [v]}, index=idx) for lab, v in term_vols.items()}
    skew = {}
    if smile:
        put, call, atm = smile
        skew = {"put": pd.DataFrame({t: [put]}, index=idx),
                "call": pd.DataFrame({t: [call]}, index=idx),
                "atm": pd.DataFrame({t: [atm]}, index=idx)}
    return volbt.VolSurface(t, term, skew, idx)


def test_volsurface_tenor_interpolation():
    vs = _surface({"1M": 20.0, "3M": 25.0})
    day = pd.Timestamp("2026-06-01")
    assert vs.ok()
    assert vs.atm(day, 30) == pytest.approx(20.0)
    assert vs.atm(day, 91) == pytest.approx(25.0)
    # variance-linear between pillars: hand-computed at 60d
    assert vs.atm(day, 60) == pytest.approx(23.829139070735625, rel=1e-12)
    assert vs.atm(day, 10) == pytest.approx(20.0)     # flat below the short end
    assert vs.atm(day, 400) == pytest.approx(25.0)    # flat beyond the long end


def test_volsurface_smile():
    """Wings 22/21 around a 20 ATM: at the 90% strike the quadratic reproduces the put
    wing EXACTLY at 30d, decays by √(30/τ) further out, and clamps moneyness at ±20%."""
    vs = _surface({"1M": 20.0}, smile=(22.0, 21.0, 20.0))
    day = pd.Timestamp("2026-06-01")
    assert vs.vol(day, 30, 100.0, 90.0) == pytest.approx(22.0, rel=1e-12)
    assert vs.vol(day, 30, 100.0, 110.0) == pytest.approx(21.0, rel=1e-12)
    assert vs.vol(day, 30, 100.0, 100.0) == pytest.approx(20.0, rel=1e-12)
    assert vs.vol(day, 120, 100.0, 90.0) == pytest.approx(21.0, rel=1e-12)   # 20 + 2·√(30/120)
    # deep OTM clamps at x = −0.20 — same value as a strike exactly 20% below spot
    assert vs.vol(day, 30, 100.0, 70.0) == pytest.approx(vs.vol(day, 30, 100.0, math.exp(-0.2) * 100))


# ── FX quote conventions + the point-value table ─────────────────────────────
def test_fx_usd_rate_quote_divisors():
    """Verified against QUOTE_UNITS 2026-08-08 — raw-price-as-USD was 100–10,000× off
    for GBP/CHF/AUD/CAD (cents) and JPY (cents per 100 yen)."""
    assert volbt.fx_usd_rate("ECA Curncy", 1.1581) == pytest.approx(1.1581)
    assert volbt.fx_usd_rate("BPA Curncy", 134.99) == pytest.approx(1.3499)
    assert volbt.fx_usd_rate("JYA Curncy", 63.65) == pytest.approx(0.006365)
    assert volbt.fx_usd_rate("SFA Curncy", 123.45) == pytest.approx(1.2345)
    assert volbt.fx_usd_rate("XXX Curncy", 2.5) == pytest.approx(2.5)   # unknown: raw


def test_currency_map():
    assert volbt.currency("CLA Comdty") == "USD"
    assert volbt.currency("RXA Comdty") == "EUR"
    assert volbt.currency("G A Comdty") == "GBP"
    assert volbt.currency("TKYA Comdty") == "EUR"   # ICE 3M €STR — Tokyo-looking root, EUR contract


def test_point_values_quote_units():
    """The 2026-08-08 FUT_VAL_PT audit: cent-quoted contracts carry contract-size÷100
    (÷10,000 for JY) — these regressing to contract SIZES means 100× P&L again."""
    expect = {"HGA Comdty": 250.0, "XBA Comdty": 420.0, "HOA Comdty": 420.0,
              "SIA Comdty": 5000.0, "TUA Comdty": 2000.0, "SERA Comdty": 4167.0,
              "NGA Comdty": 10000.0, "ESA Index": 50.0,
              "BPA Curncy": 625.0, "JYA Curncy": 1250.0}
    for tk, pv in expect.items():
        assert volbt.point_value(tk) == pv, tk
    assert all(v > 0 for v in volbt.POINT_VALUE.values())


def test_point_value_universe_coverage(golden):
    """Lock the LIST of universe tickers with no point value on file. A new name
    appearing here should be given a PV (or knowingly accepted); one silently
    VANISHING from POINT_VALUE would flip its backtests to $0 P&L."""
    missing = sorted(t for t in universe.INSTRUMENTS if volbt.point_value(t) <= 0)
    golden("volbt_missing_point_values", missing)


# ── expiry calendar helpers ──────────────────────────────────────────────────
def test_third_friday_and_quarterlies():
    assert volbt.third_friday(2026, 3) == date(2026, 3, 20)
    assert volbt.third_friday(2026, 9) == date(2026, 9, 18)
    assert volbt.quarterly_expiries(date(2026, 8, 7), 3) == \
        [date(2026, 9, 18), date(2026, 12, 18), date(2027, 3, 19)]
    # strictly-after rule: asking ON an expiry day skips to the next quarter
    assert volbt.quarterly_expiries(date(2026, 9, 18), 1) == [date(2026, 12, 18)]
