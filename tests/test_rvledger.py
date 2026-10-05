"""The RV ledger tracks the fade rule forward as an append-only, frozen record.

The behaviours that must hold (the Signal Ledger's hard-won discipline, applied here):

  * a position that is open when tracking begins is taken on and followed to its outcome;
  * an open position SETTLES once forward data resolves it;
  * a SETTLED episode is a frozen fact — a later re-run never re-measures it, even if the
    recomputed series would now disagree (the 2026-08 freeze lesson: a transient bad frame must
    not be able to silently rewrite the track record);
  * the scorecard shows the full-history backtest baseline next to the live forward track.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import curvemon, rvledger


def _series(vals: list[float]) -> pd.Series:
    return pd.Series(vals, index=pd.bdate_range("2016-01-01", periods=len(vals)))


def _warm(w: int, reps: int = 2) -> list[float]:
    return [1.0 if i % 2 == 0 else -1.0 for i in range(w * reps)]


_SPEC = {"key": "xx", "name": "Test Spread", "group": "Rates — Curve", "unit": "bp",
         "dp": 1, "legs": [(1, "yield", "AAA Comdty")], "scale": 1.0}


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """One synthetic spread, a short z-window, and a throwaway store, so update()/scorecard()
    are fully deterministic and never touch the real ledger."""
    monkeypatch.setattr(rvledger, "STORE", tmp_path / "rv_ledger.json")
    monkeypatch.setattr(rvledger, "WINDOW", 20)
    monkeypatch.setattr(curvemon, "SPREADS", [_SPEC])
    holder: dict = {}
    monkeypatch.setattr(curvemon, "_build_spread", lambda spec, h: holder.get("series"))
    return lambda s: holder.__setitem__("series", s)


def test_first_run_takes_on_the_open_position(ledger_env):
    """On the first update a currently-stretched, unresolved spread is logged as an open
    position and the forward track's start date is stamped."""
    ledger_env(_series(_warm(20) + [3.0, 3.0, 3.0]))          # stretched and unresolved at the end
    store = rvledger.update(history=pd.DataFrame(), today="2026-01-10")
    assert store["config"]["since"]
    assert len(store["episodes"]) == 1
    e = store["episodes"][0]
    assert e["key"] == "xx" and e["outcome"] == "open" and e["exit_date"] is None


def test_open_position_settles_then_is_frozen_forever(ledger_env):
    """An open position settles when the data resolves it — and once settled it is never
    re-measured, even if the stored outcome is later corrupted and the recompute would say win."""
    set_series = ledger_env
    set_series(_series(_warm(20) + [3.0, 3.0, 3.0]))
    rvledger.update(history=pd.DataFrame(), today="2026-01-10")
    assert rvledger.load()["episodes"][0]["outcome"] == "open"

    # forward data arrives and the spread reverts to the mean → the position wins
    set_series(_series(_warm(20) + [3.0, 3.0, 3.0] + [0.0] * 12))
    st = rvledger.update(history=pd.DataFrame(), today="2026-02-10")
    assert len(st["episodes"]) == 1 and st["episodes"][0]["outcome"] == "win"

    # corrupt the settled fact, then re-run with the SAME (win-producing) series
    st["episodes"][0]["outcome"] = "loss"
    st["episodes"][0]["pnl_sigma"] = -42.0
    rvledger._save(st)
    frozen = rvledger.update(history=pd.DataFrame(), today="2026-03-10")["episodes"][0]
    assert frozen["outcome"] == "loss" and frozen["pnl_sigma"] == -42.0   # never rewritten


def test_scorecard_reports_the_backtest_baseline_beside_the_live_track(ledger_env):
    """The scorecard carries the full-history backtest (the 'expected') for every spread with
    history, independent of how much the live forward track has accrued."""
    seg = _warm(20)
    for _ in range(3):                                        # three cleanly-reverting episodes
        seg += [3.0] * 5 + [0.0, 0.0] + _warm(20, 1)
    ledger_env(_series(seg))
    st = rvledger.update(history=pd.DataFrame(), today="2026-06-01")
    sc = rvledger.scorecard(store=st, history=pd.DataFrame())
    assert sc["overall"]["backtest"]["n"] >= 2               # resolved episodes over full history
    assert sc["overall"]["backtest"]["hit_rate"] == sc["overall"]["backtest"]["hit_rate"]  # finite
    assert len(sc["rows"]) == 1 and sc["rows"][0]["key"] == "xx"
    assert sc["since"]
