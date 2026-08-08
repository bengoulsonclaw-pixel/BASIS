"""Signal-direction fixture tests for the pattern strategies.

Covers: Support & Resistance, Fibonacci Retracement, Breakout & Retest,
Flag Breakout, Elliott Wave. Each test engineers a synthetic close series where
the textbook LONG / SHORT / NEUTRAL condition clearly holds and asserts
find_opportunities() flags it the right way round (direction +1 / -1 / 0-or-no-row).

All frames are built in the test and are fully deterministic (the shapes are
exact — no RNG). Tickers are deliberately NOT in the real universe, and Flag
Breakout is always called with an injected history= frame so its FICC persist
path never writes to data/signals/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import breakout_retest as br
from src.strategies import elliott_wave as ew
from src.strategies import fibonacci as fib
from src.strategies import flag_breakout as fb
from src.strategies import support_resistance as sr

LONG_T, SHORT_T, NEUT_T = "TESTLONG X", "TESTSHORT X", "TESTNEUT X"


def _frame(**cols) -> pd.DataFrame:
    n = max(len(v) for v in cols.values())
    assert all(len(v) == n for v in cols.values()), "columns must be equal length"
    idx = pd.bdate_range("2020-01-06", periods=n)
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=idx)


def _row(df: pd.DataFrame, ticker: str) -> pd.Series:
    sub = df[df["instruments"] == ticker]
    assert len(sub) == 1, f"expected exactly one row for {ticker}, got {len(sub)}"
    return sub.iloc[0]


# ── Support & Resistance ─────────────────────────────────────────────────────
# A clean triangle wave mints dozens of exact pivot touches at its two extremes,
# which cluster into one strong support and one strong resistance level. End the
# series 0.3% from one of them: level proximity ≈ 85 vs the 50 default trigger.

def _sr_series(cycle: list, last: float) -> np.ndarray:
    return np.append(np.tile(cycle, 39), last)                   # 391 bars


def test_support_resistance_directions():
    up_cycle = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102]   # lows 100 / highs 110
    dn_cycle = [100, 98, 96, 94, 92, 90, 92, 94, 96, 98]            # highs 100 / lows 90
    hist = _frame(**{
        LONG_T: _sr_series(up_cycle, 100.3),    # 0.3% ABOVE tested support 100 → buy the dip
        SHORT_T: _sr_series(dn_cycle, 99.7),    # 0.3% BELOW tested resistance 100 → sell the rally
        NEUT_T: _sr_series(up_cycle, 104.9),    # mid-range: no level within the 2% band
    })

    df = sr.find_opportunities(history=hist)

    r = _row(df, LONG_T)
    assert r["direction"] == 1 and r["metric"] > 50
    assert r["signal"] == "Long (buy the dip)"
    assert "Support" in r["context"]

    r = _row(df, SHORT_T)
    assert r["direction"] == -1 and r["metric"] < -50
    assert r["signal"] == "Short (sell the rally)"
    assert "Resistance" in r["context"]

    r = _row(df, NEUT_T)
    assert r["direction"] == 0 and r["signal"] == "—" and r["metric"] == 0.0


# ── Fibonacci Retracement ────────────────────────────────────────────────────
# A 30% dominant leg, then price parked EXACTLY on the 61.8% retracement:
# Fib proximity 100 vs the 60 default trigger. Up-leg → long, down-leg → short.

FIB_UP_618 = 130.0 - 0.618 * 30.0      # retrace down from the 130 high (≈111.46)
FIB_DN_618 = 100.0 + 0.618 * 30.0      # bounce up from the 100 low (≈118.54)


def _fib_up() -> np.ndarray:
    return np.concatenate([
        np.full(220, 115.0),                    # outside the 180-bar swing window
        np.linspace(106.0, 100.0, 20),          # drift into the swing low (100)
        np.linspace(100.0, 130.0, 60),          # the dominant up-leg (+30%)
        np.linspace(130.0, FIB_UP_618, 40),     # pull back to the golden level
        np.full(60, FIB_UP_618),                # sit on it
    ])                                          # 400 bars


def _fib_down() -> np.ndarray:
    return np.concatenate([
        np.full(220, 115.0),
        np.linspace(124.0, 130.0, 20),          # drift into the swing high (130)
        np.linspace(130.0, 100.0, 60),          # the dominant down-leg
        np.linspace(100.0, FIB_DN_618, 40),     # bounce up to the golden level
        np.full(60, FIB_DN_618),
    ])


def test_fibonacci_directions():
    wiggle = 100.0 + 0.5 * np.sin(np.arange(400.0))   # ~1% total range < 5% minimum leg
    hist = _frame(**{LONG_T: _fib_up(), SHORT_T: _fib_down(), NEUT_T: wiggle})

    df = fib.find_opportunities(history=hist)

    r = _row(df, LONG_T)
    assert r["direction"] == 1 and r["metric"] >= 60
    assert r["signal"] == "Long (buy the dip)"
    assert "up-leg" in r["context"] and "61.8% retrace" in r["context"]

    r = _row(df, SHORT_T)
    assert r["direction"] == -1 and r["metric"] <= -60
    assert r["signal"] == "Short (sell the rally)"
    assert "down-leg" in r["context"]

    assert NEUT_T not in set(df["instruments"])       # sub-5% swing → no row at all


# ── Breakout & Retest ────────────────────────────────────────────────────────
# Oscillation builds a many-touch horizontal level at 100; a decisive thrust
# through it a few bars ago, then a pull-back to ~0.3% from the level (retest
# proximity ≈ 80 vs the 60 default). Conditional: the neutral name yields NO row.

BR_CYCLE_UP = [90, 92, 94, 96, 98, 100, 98, 96, 94, 92]      # resistance minted at 100
BR_CYCLE_DN = [110, 108, 106, 104, 102, 100, 102, 104, 106, 108]   # support minted at 100


def _br_long() -> np.ndarray:
    tail = [90, 92, 94, 96, 98, 100, 102, 103.5,             # decisive upside break of 100
            102.5, 101.5, 100.8, 100.4, 100.3]               # ...pull back onto it from above
    return np.append(np.tile(BR_CYCLE_UP, 38), tail)         # 393 bars


def _br_short() -> np.ndarray:
    tail = [110, 108, 106, 104, 102, 100, 98, 96.5,          # decisive downside break of 100
            97.5, 98.5, 99.2, 99.6, 99.7]                    # ...bounce back onto it from below
    return np.append(np.tile(BR_CYCLE_DN, 38), tail)


def test_breakout_retest_directions():
    neut = np.append(np.tile(BR_CYCLE_UP, 39), [90, 92, 94])   # keeps oscillating, never breaks
    hist = _frame(**{LONG_T: _br_long(), SHORT_T: _br_short(), NEUT_T: neut})

    df = br.find_opportunities(history=hist, volume=pd.DataFrame())

    r = _row(df, LONG_T)
    assert r["direction"] == 1 and r["metric"] >= 60
    assert r["signal"] == "Long — retest"
    assert "Broke 100" in r["context"] and "retesting from above" in r["context"]

    r = _row(df, SHORT_T)
    assert r["direction"] == -1 and r["metric"] <= -60
    assert r["signal"] == "Short — retest"
    assert "retesting from below" in r["context"]

    assert NEUT_T not in set(df["instruments"])       # no recent break → conditional no-row


# ── Flag Breakout ────────────────────────────────────────────────────────────
# A 10% pole in 10 bars, a 10-bar flag drifting gently against it with a tight
# zigzag (the channel needs residual σ > 0), then a final bar popping through
# the breakout line — readiness clears the 70 default trigger decisively.
# history= is ALWAYS injected so the FICC persist path never writes parquet.

def _flag_series(sign: int) -> np.ndarray:
    tip = 100.0 + sign * 10.0                                # pole tip: 110 bull / 90 bear
    pole = 100.0 + sign * np.arange(1.0, 11.0)               # 10 bars of fast pole
    x = np.arange(1.0, 10.0)                                 # 9 flag bars after the tip bar
    resid = sign * np.where(x % 2 == 1, -0.25, 0.25)         # tight alternating zigzag
    body = tip - sign * 0.2 * x + resid                      # drifts against the pole
    pop = tip + sign * 1.0                                   # final close beyond the pole tip
    return np.concatenate([np.full(380, 100.0), pole, body, [pop]])   # 400 bars


def test_flag_breakout_directions():
    neut = 100.0 + 0.1 * np.where(np.arange(400) % 2 == 0, 1.0, -1.0)   # chop, no pole
    hist = _frame(**{LONG_T: _flag_series(+1), SHORT_T: _flag_series(-1), NEUT_T: neut})

    df = fb.find_opportunities(history=hist, volume=pd.DataFrame())

    r = _row(df, LONG_T)
    assert r["direction"] == 1 and r["metric"] >= 70
    assert r["signal"] == "Bull breakout setup"
    assert "Bull" in r["context"]

    r = _row(df, SHORT_T)
    assert r["direction"] == -1 and r["metric"] <= -70
    assert r["signal"] == "Bear breakout setup"
    assert "Bear" in r["context"]

    assert NEUT_T not in set(df["instruments"])       # no qualifying pole → no row


def test_flag_breakout_injected_history_never_persists(tmp_path, monkeypatch):
    """With history= injected, neither flag cache file may be (re)written."""
    monkeypatch.setattr(fb, "DETAIL_FILE", tmp_path / "detail.parquet")
    monkeypatch.setattr(fb, "HISTORY_FILE", tmp_path / "history.parquet")
    fb.find_opportunities(history=_frame(**{LONG_T: _flag_series(+1)}),
                          volume=pd.DataFrame())
    assert not (tmp_path / "detail.parquet").exists()
    assert not (tmp_path / "history.parquet").exists()


# ── Elliott Wave ─────────────────────────────────────────────────────────────
# A textbook wave-3 juncture: 20% wave 1, exact 61.8% wave-2 retrace, price
# advancing off the wave-2 pivot → wave fit ≈ 100 vs the 55 flag bar.
# Up-impulse → +1 ("Wave 3 underway ▲"), mirrored down-impulse → −1.

def _ew_series(sign: int) -> np.ndarray:
    w1_end = 100.0 + sign * 20.0                     # 120 up / 80 down
    w2_end = w1_end - sign * 0.618 * 20.0            # 61.8% retrace of wave 1
    w3_now = w2_end + sign * 4.36                    # advancing off the wave-2 pivot
    return np.concatenate([
        np.full(347, 100.0),                         # quiet base (ZigZag thr floors at 2.5%)
        np.linspace(100.0, w1_end, 31)[1:],          # wave 1 (30 bars)
        np.linspace(w1_end, w2_end, 16)[1:],         # wave 2 (15 bars)
        np.linspace(w2_end, w3_now, 9)[1:],          # wave 3 underway (8 bars)
    ])                                               # 400 bars


def test_elliott_wave_directions():
    hist = _frame(**{LONG_T: _ew_series(+1), SHORT_T: _ew_series(-1),
                     NEUT_T: np.full(400, 100.0)})

    df = ew.find_opportunities(history=hist)

    r = _row(df, LONG_T)
    assert r["direction"] == 1 and r["metric"] >= 55
    assert r["signal"].startswith("Wave 3 underway")
    assert "up-impulse" in r["context"] and "W2 62% retrace" in r["context"]

    r = _row(df, SHORT_T)
    assert r["direction"] == -1 and r["metric"] <= -55
    assert r["signal"].startswith("Wave 3 underway")
    assert "down-impulse" in r["context"]

    assert NEUT_T not in set(df["instruments"])       # flat series → no count, no row
