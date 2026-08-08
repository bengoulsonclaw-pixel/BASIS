"""Golden locks on the TA Signal Backtester engine (src/tabt.py).

Three layers, most-critical first:
  * the flat/long/short state machine (_step) — every exit rule, cost accounting;
  * the scoring seams — the FIXED-INCOME SIGN FLIP (a yield-space Long must trade the
    future SHORT) and the Mean-Reversion pair-leg translation;
  * one full run_backtest() walk-forward over a frozen synthetic history (feed layer
    monkeypatched out), locking the whole loop: window slicing, the live strategy
    module, hub-trigger re-flagging, harmonic de-dup, blotter and summary stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import tabt, universe


def _sig(d, conviction=80.0, score=50.0):
    return {"direction": d, "conviction": conviction, "score": score * d}


_KW = dict(exit_rule="reversal", hold_days=None, stop_pct=None, take_pct=None,
           min_conviction=0.0, min_score=0.0, direction_filter="both",
           size_mult=1000.0, size=2.0, side_cost=25.0)


def _walk(seq, **overrides):
    """Run a scripted [(day, price, sig), …] through _step; returns (trades, states)."""
    kw = {**_KW, **overrides}
    pos, trades, states = None, [], []
    for i, (day, price, sig) in enumerate(seq):
        pos, trade = tabt._step(pos, sig, price, day, i == len(seq) - 1, **kw)
        states.append(pos["direction"] if pos else 0)
        if trade:
            trades.append(trade)
    return trades, states


def test_step_reversal_roundtrip(golden):
    days = pd.date_range("2026-01-05", periods=5, freq="B")
    seq = [(days[0], 100.0, _sig(+1)),   # open long @100
           (days[1], 102.0, None),       # hold through a silent day
           (days[2], 104.0, _sig(-1)),   # reversal: close +4 pts, flip short @104
           (days[3], 103.0, None),
           (days[4], 101.0, None)]       # period end: close short +3 pts
    trades, states = _walk(seq)
    assert states == [1, 1, -1, -1, 0]
    assert len(trades) == 2
    t1, t2 = trades
    # 4 pts × $1000/pt × 2 lots − $25/side round trip
    assert t1["pnl"] == pytest.approx(4 * 1000 * 2 - 50.0)
    assert t1["exit_reason"] == "reversal" and t1["direction"] == "Long"
    assert t1["holding_days"] == (days[2] - days[0]).days
    assert t2["pnl"] == pytest.approx(3 * 1000 * 2 - 50.0)
    assert t2["exit_reason"] == "period_end" and t2["direction"] == "Short"
    golden("tabt_step_reversal", trades)


def test_step_score_drop_and_entry_bar():
    days = pd.date_range("2026-01-05", periods=4, freq="B")
    seq = [(days[0], 50.0, _sig(+1, conviction=40.0)),   # below the bar — no entry
           (days[1], 50.0, _sig(+1, conviction=60.0)),   # clears it — open
           (days[2], 51.0, _sig(+1, conviction=40.0)),   # sags below — score_drop exit
           (days[3], 52.0, None)]
    trades, states = _walk(seq, exit_rule="score_drop", min_conviction=50.0)
    assert states == [0, 1, 0, 0]
    assert [t["exit_reason"] for t in trades] == ["score_drop"]


def test_step_hold_days():
    days = pd.date_range("2026-01-05", periods=6, freq="B")
    seq = [(days[0], 10.0, _sig(+1))] + [(d, 10.0, None) for d in days[1:]]
    trades, _ = _walk(seq, exit_rule="hold_days", hold_days=3)   # calendar days, not bars
    assert [t["exit_reason"] for t in trades] == ["hold_days"]
    assert trades[0]["holding_days"] == 3
    assert trades[0]["exit_date"] == days[3].date()


def test_step_stop_beats_primary_rule():
    days = pd.date_range("2026-01-05", periods=3, freq="B")
    # long @100; day 2 prints 97 (−3%) AND flips the signal — the stop must claim it
    seq = [(days[0], 100.0, _sig(+1)), (days[1], 97.0, _sig(-1)), (days[2], 97.0, None)]
    trades, states = _walk(seq, stop_pct=2.0)
    assert trades[0]["exit_reason"] == "stop"
    assert states[1] == -1        # …and the fresh reversal signal still opens the short


def test_step_take_profit_and_direction_filter():
    days = pd.date_range("2026-01-05", periods=3, freq="B")
    seq = [(days[0], 100.0, _sig(+1)), (days[1], 105.0, None), (days[2], 105.0, None)]
    trades, _ = _walk(seq, take_pct=4.0)
    assert trades[0]["exit_reason"] == "take"
    # long-only filter must ignore a short signal entirely
    seq = [(days[0], 100.0, _sig(-1)), (days[1], 99.0, None), (days[2], 98.0, None)]
    trades, states = _walk(seq, direction_filter="long")
    assert trades == [] and states == [0, 0, 0]


def test_summarize_stats(golden):
    trades = pd.DataFrame([
        {"pnl": 500.0, "holding_days": 3, "cost": 50.0},
        {"pnl": -200.0, "holding_days": 2, "cost": 50.0},
        {"pnl": 300.0, "holding_days": 5, "cost": 50.0},
    ])
    daily = pd.DataFrame({"cum_pnl": [0.0, 500.0, 300.0, 800.0, 200.0, 600.0]})
    s = tabt._summarize(trades, daily, "USD")
    assert s["total_pnl"] == pytest.approx(600.0)
    assert s["win_rate"] == pytest.approx(200.0 / 3)
    assert s["profit_factor"] == pytest.approx(800.0 / 200.0)
    assert s["max_drawdown"] == pytest.approx(600.0)         # 800 -> 200
    assert s["costs"] == pytest.approx(150.0)
    golden("tabt_summarize", s)
    empty = tabt._summarize(pd.DataFrame(), None, "USD")
    assert empty["n_trades"] == 0 and empty["total_pnl"] == 0.0


def test_select_rows_translates_mean_reversion_leg(golden):
    """A 'Long spread' on WTI–Brent is buy CLA / sell COA: selecting for the B leg must
    flip metric AND direction onto COA's own key."""
    pair = tabt.pair_for("COA Comdty")
    assert pair and pair["a"] == "CLA Comdty"                # the universe's WTI–Brent pair
    raw = pd.DataFrame([
        {"strategy": "Mean Reversion", "market": "WTI – Brent",
         "instruments": "CLA Comdty / COA Comdty", "signal": "Long spread",
         "direction": 1, "metric": 2.0},
        {"strategy": "Trend", "market": "Brent", "instruments": "COA Comdty",
         "signal": "Long", "direction": 1, "metric": 12.0},
    ])
    out = tabt._select_rows(raw, "COA Comdty", ["Trend", "Mean Reversion"])
    mr = out[out["strategy"] == "Mean Reversion"].iloc[0]
    assert mr["instruments"] == "COA Comdty"
    assert mr["direction"] == -1 and mr["metric"] == pytest.approx(-2.0)
    golden("tabt_select_rows_mr", out)


def test_score_set_fi_sign_flip():
    """THE key convention: FI strategies score YIELDS, so a yield-space Long (rates up)
    must come out as direction −1 on the bond FUTURE, score sign flipped with it."""
    assert universe.is_fixed_income("TYA Comdty")
    raw = pd.DataFrame([{"strategy": "Trend", "market": "US 10yr", "instruments": "TYA Comdty",
                         "signal": "—", "direction": 0, "metric": 15.0}])   # strength 60
    fut = tabt._score_set(raw, ["Trend"], "TYA Comdty", sign=-1)
    yld = tabt._score_set(raw, ["Trend"], "TYA Comdty", sign=+1)
    assert yld == {"direction": 1, "conviction": 60.0, "score": 60.0}
    assert fut == {"direction": -1, "conviction": 60.0, "score": -60.0}


def _frozen_history():
    """420 business days ending 2026-06-30, frozen forever: a +0.3%/day up-leg, then a
    −0.8%/day down-leg over the final 30 sessions, plus a whisper of seeded noise. By
    construction the 3-month return clears Trend's +10% hub bar from the window's first
    day (Long) and crosses −10% in late June (reversal to Short) — so the walk-forward
    must produce exactly one flip inside the scored quarter."""
    idx = pd.bdate_range(end="2026-06-30", periods=420)
    i = np.arange(len(idx))
    r = np.where(i < 390, 0.003, -0.008) + np.random.default_rng(7).normal(0, 5e-4, len(idx))
    px = 60.0 * np.exp(np.cumsum(r))
    return pd.DataFrame({"CLA Comdty": px}, index=idx)


def test_run_backtest_end_to_end(golden, monkeypatch):
    """One full walk-forward on the real Trend module over a frozen synthetic feed —
    the whole loop locked: fixed 400-session window slicing, find_opportunities, the
    hub scoring pipeline, the state machine, blotter and summary."""
    hist = _frozen_history()
    buf_start, end_ts = hist.index[0], hist.index[-1]

    def _fake_load(scope, ticker, strategies, start, end):
        return hist, hist, None, [ticker], buf_start, end_ts, []

    monkeypatch.setattr(tabt, "_load_history", _fake_load)
    monkeypatch.setattr(tabt, "_cached_day_rows", lambda *a, **k: {})   # no repo signal cache

    res = tabt.run_backtest("ficc", "CLA Comdty", ["Trend"],
                            start=pd.Timestamp("2026-04-01").date(),
                            end=pd.Timestamp("2026-06-30").date())
    days = hist.loc["2026-04-01":"2026-06-30"].index
    assert len(res.daily) == len(days)
    assert res.summary["cached_days"] == 0
    assert res.summary["fi"] is False
    assert np.isfinite(res.daily["cum_pnl"]).all()
    # the constructed history guarantees a Long through the up-leg, a reversal to Short
    # in late June, and a period-end close: two trades, both position signs seen
    assert res.summary["n_trades"] == 2
    assert list(res.trades["exit_reason"]) == ["reversal", "period_end"]
    assert set(res.daily["position"].iloc[:-1]) == {1, -1}
    assert res.daily["position"].iloc[-1] == 0            # period_end closes flat
    golden("tabt_run_backtest", {
        "summary": res.summary,
        "trades": res.trades,
        "daily_tail": res.daily.tail(5).reset_index().rename(columns={"index": "date"}),
        "n_daily": len(res.daily),
    })
