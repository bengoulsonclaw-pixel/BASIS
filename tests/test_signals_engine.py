"""Signal-direction fixtures: Mean Reversion (pair z-score) + tabt engine invariants.

Layer 1 — src/strategies/mean_reversion.py on engineered pair history: a spread pushed
far below / above its 90-day mean must flag Long spread (+1) / Short spread (−1) on the
real universe.PAIRS WTI–Brent pair, and an unstretched spread must stay silent (0).

Layer 2 — src/tabt.py engine invariants EXTENDING tests/test_tabt.py (no duplication):
  * FI sign flip END-TO-END through run_backtest(): a rising-yield (yield-space Long)
    STIR history must open a SHORT futures position and earn positive P&L as the price falls;
  * Mean-Reversion leg translation through the LIVE module (_raw_frames): 'Long spread'
    lands as +1 on leg A's own key and −1 on leg B's;
  * the stop/take overlay claims an exit BEFORE the primary rule on the same day
    (take vs hold_days; stop vs score_drop — test_tabt already locks stop vs reversal);
  * hold_days is CALENDAR days: a Friday entry with hold_days=2 exits Monday (3 days);
  * cost bookkeeping: frictionless total − with-costs total == n_trades × 2 × side_cost
    exactly, and the equity-curve end equals the summary total in both runs.

All frames are synthetic, all RNGs seeded — deterministic forever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import tabt, universe
from src.strategies import mean_reversion


# ── synthetic pair history ──────────────────────────────────────────────────

PAIR_TICKERS = sorted({t for p in universe.PAIRS for t in (p["a"], p["b"])})
WTI, BRENT = "CLA Comdty", "COA Comdty"          # the universe's WTI–Brent diff pair


def _pairs_frame(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """All PAIRS tickers (mean_reversion loops the full pair universe, so every leg
    must be present), each a distinct positive level + small seeded noise. The FINAL
    bar of the WTI–Brent legs is pinned to its exact base level so the unmodified
    frame is deterministically UNstretched; the long/short tests then displace it."""
    idx = pd.bdate_range(end="2026-06-30", periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for i, t in enumerate(PAIR_TICKERS):
        base = 50.0 + 7.0 * i
        px = base + rng.normal(0.0, 0.2, n)
        if t in (WTI, BRENT):
            px[-1] = base                        # last bar exactly on base level
        data[t] = px
    return pd.DataFrame(data, index=idx)


def _wti_brent_row(df: pd.DataFrame) -> pd.Series:
    row = df[df["instruments"] == f"{WTI} / {BRENT}"]
    assert len(row) == 1, "expected exactly one WTI–Brent row"
    return row.iloc[0]


def test_mean_reversion_long_spread():
    """Spread crushed ~10σ below its 90d mean -> Long spread, direction +1."""
    hist = _pairs_frame()
    hist.iloc[-1, hist.columns.get_loc(WTI)] -= 3.0     # spread ≈ −10z vs σ≈0.28
    row = _wti_brent_row(mean_reversion.find_opportunities(history=hist))
    assert row["signal"] == "Long spread"
    assert row["direction"] == 1
    assert row["metric"] <= -mean_reversion.Z_THRESHOLD
    assert row["metric_label"] == "z-score"
    assert "Buy" in row["context"] and "Sell" in row["context"]   # buy A / sell B hint
    assert row["strategy"] == "Mean Reversion"


def test_mean_reversion_short_spread():
    """Spread stretched ~10σ ABOVE its 90d mean -> Short spread, direction −1."""
    hist = _pairs_frame()
    hist.iloc[-1, hist.columns.get_loc(WTI)] += 3.0
    row = _wti_brent_row(mean_reversion.find_opportunities(history=hist))
    assert row["signal"] == "Short spread"
    assert row["direction"] == -1
    assert row["metric"] >= mean_reversion.Z_THRESHOLD
    assert "Sell" in row["context"]


def test_mean_reversion_neutral():
    """Spread sitting on its mean -> no flag: signal '—', direction 0."""
    row = _wti_brent_row(mean_reversion.find_opportunities(history=_pairs_frame()))
    assert abs(row["metric"]) < mean_reversion.Z_THRESHOLD
    assert row["direction"] == 0
    assert row["signal"] == "—"


def test_mean_reversion_ratio_pair_long():
    """Ratio-kind pair (Gold/Silver): same direction convention as the diff pairs."""
    hist = _pairs_frame()
    gold = "GCA Comdty"
    hist.iloc[-1, hist.columns.get_loc(gold)] *= 0.90    # ratio well below its mean
    df = mean_reversion.find_opportunities(history=hist)
    row = df[df["instruments"] == "GCA Comdty / SIA Comdty"].iloc[0]
    assert row["direction"] == 1 and row["signal"] == "Long spread"
    assert row["metric"] <= -mean_reversion.Z_THRESHOLD


# ── tabt: Mean-Reversion leg translation through the LIVE module ────────────

def test_raw_frames_mr_leg_translation():
    """'Long spread' (buy A / sell B) computed by the real module must land as
    direction +1 keyed on leg A and −1 keyed on leg B, z-sign flipped with it."""
    hist = _pairs_frame()
    hist.iloc[-1, hist.columns.get_loc(WTI)] -= 3.0      # Long spread on WTI–Brent

    row_a = tabt._raw_frames(WTI, ["Mean Reversion"], hist, None)
    assert len(row_a) == 1
    a = row_a.iloc[0]
    assert a["instruments"] == WTI and a["direction"] == 1 and a["metric"] < 0

    row_b = tabt._raw_frames(BRENT, ["Mean Reversion"], hist, None)
    assert len(row_b) == 1
    b = row_b.iloc[0]
    assert b["instruments"] == BRENT and b["direction"] == -1 and b["metric"] > 0
    assert b["metric"] == pytest.approx(-a["metric"])    # exact leg-sign mirror


# ── tabt: FI sign flip end-to-end through run_backtest ──────────────────────

def _fi_frozen_history():
    """420 bdays ending 2026-06-30: a STIR's implied YIELD grinding up +0.3%/day
    (seeded whisper of noise), price = 100 − yield falling with it. The 3-month
    yield return clears Trend's +10% hub bar from the scored window's first day."""
    idx = pd.bdate_range(end="2026-06-30", periods=420)
    r = 0.003 + np.random.default_rng(11).normal(0, 5e-5, len(idx))
    yld = 2.0 * np.exp(np.cumsum(r))
    yld_df = pd.DataFrame({"SERA Comdty": yld}, index=idx)
    px_df = pd.DataFrame({"SERA Comdty": 100.0 - yld}, index=idx)
    return yld_df, px_df


def test_run_backtest_fi_sign_flip(monkeypatch):
    """THE FI convention, end-to-end: yields rising (yield-space Long) on a STIR must
    open a SHORT futures position in the blotter and profit as the price falls."""
    assert universe.is_fixed_income("SERA Comdty")
    yld_df, px_df = _fi_frozen_history()
    buf_start, end_ts = yld_df.index[0], yld_df.index[-1]

    def _fake_load(scope, ticker, strategies, start, end):
        return yld_df, px_df, None, [ticker], buf_start, end_ts, []

    monkeypatch.setattr(tabt, "_load_history", _fake_load)
    monkeypatch.setattr(tabt, "_cached_day_rows", lambda *a, **k: {})

    res = tabt.run_backtest("ficc", "SERA Comdty", ["Trend"],
                            start=pd.Timestamp("2026-04-01").date(),
                            end=pd.Timestamp("2026-06-30").date())
    assert res.summary["fi"] is True
    assert res.summary["n_trades"] == 1
    t = res.trades.iloc[0]
    assert t["direction"] == "Short"                      # yield Long -> futures SHORT
    assert t["exit_reason"] == "period_end"
    assert t["exit_price"] < t["entry_price"]             # price fell as yields rose
    assert t["pnl"] > 0                                   # the short made money
    # in position (short) every day, force-flat on the final bar
    assert (res.daily["position"].iloc[:-1] == -1).all()
    assert res.daily["position"].iloc[-1] == 0
    # the yield-space Long is visible as a NEGATIVE price-space score all window
    assert (res.daily["score"].dropna() < 0).all()
    # signal_level charts the YIELD, price column the future
    assert (res.daily["signal_level"] < 10.0).all()
    assert (res.daily["price"] > 90.0).all()


# ── tabt: stop/take overlay precedence + hold_days calendar semantics ───────

def _sig(d, conviction=80.0, score=50.0):
    return {"direction": d, "conviction": conviction, "score": score * d}


_KW = dict(exit_rule="reversal", hold_days=None, stop_pct=None, take_pct=None,
           min_conviction=0.0, min_score=0.0, direction_filter="both",
           size_mult=1000.0, size=2.0, side_cost=25.0)


def _walk(seq, **overrides):
    kw = {**_KW, **overrides}
    pos, trades, states = None, [], []
    for i, (day, price, sig) in enumerate(seq):
        pos, trade = tabt._step(pos, sig, price, day, i == len(seq) - 1, **kw)
        states.append(pos["direction"] if pos else 0)
        if trade:
            trades.append(trade)
    return trades, states


def test_step_take_beats_hold_days_same_day():
    """Day 1 satisfies BOTH the take overlay (+5% vs +4% take) and the hold_days rule
    (1 day elapsed) — the overlay is checked first and must claim the exit."""
    days = pd.date_range("2026-01-05", periods=3, freq="B")
    seq = [(days[0], 100.0, _sig(+1)), (days[1], 105.0, None), (days[2], 105.0, None)]
    trades, _ = _walk(seq, exit_rule="hold_days", hold_days=1, take_pct=4.0)
    assert [t["exit_reason"] for t in trades] == ["take"]


def test_step_stop_beats_score_drop_same_day():
    """Day 1 prints −3% AND sags below the conviction bar — the stop must claim it."""
    days = pd.date_range("2026-01-05", periods=3, freq="B")
    seq = [(days[0], 100.0, _sig(+1, conviction=60.0)),
           (days[1], 97.0, _sig(+1, conviction=40.0)),
           (days[2], 97.0, None)]
    trades, _ = _walk(seq, exit_rule="score_drop", min_conviction=50.0, stop_pct=2.0)
    assert [t["exit_reason"] for t in trades] == ["stop"]


def test_step_hold_days_counts_calendar_days_over_weekend():
    """hold_days is CALENDAR days: enter Friday with hold_days=2 -> the first bar with
    ≥2 elapsed calendar days is Monday (3 days), so it exits there, not Tuesday."""
    days = pd.bdate_range("2026-01-09", periods=4)          # Fri, Mon, Tue, Wed
    assert days[0].day_name() == "Friday"
    seq = [(days[0], 10.0, _sig(+1))] + [(d, 10.0, None) for d in days[1:]]
    trades, states = _walk(seq, exit_rule="hold_days", hold_days=2)
    assert [t["exit_reason"] for t in trades] == ["hold_days"]
    assert trades[0]["exit_date"] == days[1].date()          # Monday
    assert trades[0]["holding_days"] == 3                    # calendar, not trading, days
    assert states[0] == 1 and states[1] == 0


# ── tabt: cost bookkeeping invariant ────────────────────────────────────────

def _cla_frozen_history():
    """Same shape as test_tabt's frozen CLA feed (up-leg then a late down-leg) so the
    scored quarter contains a flip -> exactly two closed trades to audit costs on."""
    idx = pd.bdate_range(end="2026-06-30", periods=420)
    i = np.arange(len(idx))
    r = np.where(i < 390, 0.003, -0.008) + np.random.default_rng(7).normal(0, 5e-4, len(idx))
    px = 60.0 * np.exp(np.cumsum(r))
    return pd.DataFrame({"CLA Comdty": px}, index=idx)


def test_run_backtest_cost_invariant(monkeypatch):
    """Frictionless total − with-costs total == n_trades × 2 × side_cost EXACTLY
    (costs never change entries/exits), and the equity-curve end == summary total."""
    hist = _cla_frozen_history()
    buf_start, end_ts = hist.index[0], hist.index[-1]

    def _fake_load(scope, ticker, strategies, start, end):
        return hist, hist, None, [ticker], buf_start, end_ts, []

    monkeypatch.setattr(tabt, "_load_history", _fake_load)
    monkeypatch.setattr(tabt, "_cached_day_rows", lambda *a, **k: {})

    kw = dict(start=pd.Timestamp("2026-04-01").date(), end=pd.Timestamp("2026-06-30").date(),
              size=2.0)
    free = tabt.run_backtest("ficc", "CLA Comdty", ["Trend"], **kw)
    cost = tabt.run_backtest("ficc", "CLA Comdty", ["Trend"],
                             commission=12.5, slippage_pts=0.01, **kw)

    # identical trading — costs are pure friction, never signal
    assert cost.summary["n_trades"] == free.summary["n_trades"] > 0
    assert list(cost.trades["entry_date"]) == list(free.trades["entry_date"])
    assert list(cost.trades["exit_date"]) == list(free.trades["exit_date"])

    # side_cost = commission×size + slippage_pts×point_value×fx×size (CLA: $1000/pt, USD)
    side_cost = 12.5 * 2.0 + 0.01 * 1000.0 * 2.0          # = 45.0
    n = free.summary["n_trades"]
    assert free.summary["total_pnl"] - cost.summary["total_pnl"] == pytest.approx(
        n * 2.0 * side_cost, abs=1e-6)
    assert cost.summary["costs"] == pytest.approx(n * 2.0 * side_cost, abs=1e-9)
    assert free.summary["costs"] == 0.0

    # equity-curve end == summary total (flat at period end, all costs paid as incurred)
    assert free.daily["cum_pnl"].iloc[-1] == pytest.approx(free.summary["total_pnl"], abs=1e-6)
    assert cost.daily["cum_pnl"].iloc[-1] == pytest.approx(cost.summary["total_pnl"], abs=1e-6)
    assert free.daily["position"].iloc[-1] == 0
