"""Signal-direction fixture tests — trend-family strategies.

For each strategy an engineered synthetic history where the textbook LONG condition
clearly holds must yield direction +1 for that ticker, the mirrored SHORT fixture
direction -1, and (where the strategy has a no-trade state) a NEUTRAL fixture must
yield direction 0 / signal "—". Frames are built in-test (deterministic, no RNG
needed — all fixtures are analytic ramps/steps), tickers are outside the real
universe so nothing else interferes, and every strategy is called with an explicit
history= frame (never the mock default book).

Covers: trend.py (Trend), ma_crossover.py (MA Crossover), ma_crossover_swing.py
(MA Swing), ichimoku.py (Ichimoku Cloud), donchian.py (Donchian Channel),
aroon.py (Aroon).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies import (aroon, donchian, ichimoku, ma_crossover,
                            ma_crossover_swing, trend)

LONG_T = "TESTLONG X"
SHORT_T = "TESTSHORT X"
FLAT_T = "TESTFLAT X"

N = 420  # > every MIN_HISTORY gate in this family (largest is 200-SMA + 5)


def _idx(n: int = N) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def _hist(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=_idx(n))


def _row(df: pd.DataFrame, ticker: str) -> pd.Series:
    sub = df[df["instruments"] == ticker]
    assert len(sub) == 1, f"expected exactly one row for {ticker}, got {len(sub)}\n{df}"
    return sub.iloc[0]


# ---------------------------------------------------------------- Trend (20/100 MA + 63d mom)

def _ramp_up(n: int = N) -> np.ndarray:
    return 100.0 + 0.2 * np.arange(n)          # clean uptrend, 63d return > 0


def _ramp_dn(n: int = N) -> np.ndarray:
    return 200.0 - 0.2 * np.arange(n)          # clean downtrend, stays positive


def test_trend_long_short():
    df = trend.find_opportunities(history=_hist(**{LONG_T: _ramp_up(), SHORT_T: _ramp_dn()}))
    lo, sh = _row(df, LONG_T), _row(df, SHORT_T)
    assert lo["direction"] == 1 and lo["signal"] == "Long"
    assert lo["metric"] > 0                                 # 3m return %, positive
    assert lo["context"].startswith("MA20 vs MA100")
    assert "disagrees" not in lo["context"]                 # MAs agree with the return sign
    assert sh["direction"] == -1 and sh["signal"] == "Short"
    assert sh["metric"] < 0
    # Trend has no neutral state by design: direction is the sign of the 3m return,
    # so every market with enough history gets a Long/Short stance (page default
    # trigger is 0; the hub re-flags at |3m return| >= 10%).


# ------------------------------------------- MA Crossover (50/200 SMA, 15-EMA + 3m confirm)

def test_ma_crossover_long_short_neutral():
    hist = _hist(**{LONG_T: _ramp_up(), SHORT_T: _ramp_dn(),
                    FLAT_T: np.full(N, 100.0)})
    df = ma_crossover.find_opportunities(history=hist)
    lo, sh, fl = _row(df, LONG_T), _row(df, SHORT_T), _row(df, FLAT_T)
    # golden cross + EMA15 above the 50 + positive 3m return + gap well past 2% -> Long
    assert lo["direction"] == 1 and lo["signal"] == "Long"
    assert lo["metric"] > ma_crossover.POSITION.min_gap     # MA gap % comfortably past the filter
    assert "golden cross" in lo["context"]
    # death cross, all confirmations negative -> Short
    assert sh["direction"] == -1 and sh["signal"] == "Short"
    assert sh["metric"] < -ma_crossover.POSITION.min_gap
    assert "death cross" in sh["context"]
    # dead-flat series: no separation, no momentum -> no trade
    assert fl["direction"] == 0 and fl["signal"] == "—"


# ---------------------------------------------- MA Swing (20/50 SMA, 9-EMA + 1m confirm)

def test_ma_swing_long_short_neutral():
    hist = _hist(**{LONG_T: _ramp_up(), SHORT_T: _ramp_dn(),
                    FLAT_T: np.full(N, 100.0)})
    df = ma_crossover_swing.find_opportunities(history=hist)
    lo, sh, fl = _row(df, LONG_T), _row(df, SHORT_T), _row(df, FLAT_T)
    assert lo["direction"] == 1 and lo["signal"] == "Long"
    assert lo["metric"] > ma_crossover.SWING.min_gap
    assert "golden cross" in lo["context"]
    assert sh["direction"] == -1 and sh["signal"] == "Short"
    assert sh["metric"] < -ma_crossover.SWING.min_gap
    assert "death cross" in sh["context"]
    assert fl["direction"] == 0 and fl["signal"] == "—"


# ------------------------------------------------- Ichimoku Cloud (9/26/52, fresh events)

def _ichi_base(n: int = 400, phase: float = 0.0) -> np.ndarray:
    # flat wiggle: the cloud hugs ~[100.0, 100.6] (sinusoid midpoints), price
    # oscillates through it. Freshness needs the pre-move bars on the cloud side the
    # move leaves FROM: phase=0 ends the base near the wiggle bottom (below the
    # cloud -> a surge up is a fresh break UP), phase=11 ends it at the wiggle top
    # (above the bot -> a plunge is a fresh break DOWN).
    i = np.arange(n, dtype=float)
    return 100.0 + 1.5 * np.sin(2 * np.pi * (i + phase) / 40.0)


def test_ichimoku_long_breakout():
    px = _ichi_base()
    px[-3:] = [px[-4] + 6, px[-4] + 12, px[-4] + 18]        # 3-day surge through the cloud
    df = ichimoku.find_opportunities(history=_hist(**{LONG_T: px}))
    lo = _row(df, LONG_T)
    assert lo["direction"] == 1
    assert lo["signal"].startswith("Cloud breakout up")
    assert lo["metric"] >= ichimoku.FLAG_SCORE              # signed score, flagged at >= 58
    assert "above the cloud" in lo["context"]


def test_ichimoku_short_breakdown():
    px = _ichi_base(phase=11.0)                             # base ends at the wiggle top
    px[-3:] = [px[-4] - 6, px[-4] - 12, px[-4] - 18]        # 3-day break below the cloud
    df = ichimoku.find_opportunities(history=_hist(**{SHORT_T: px}))
    sh = _row(df, SHORT_T)
    assert sh["direction"] == -1
    assert sh["signal"].startswith("Cloud breakout down")
    assert sh["metric"] <= -ichimoku.FLAG_SCORE
    assert "below the cloud" in sh["context"]


def test_ichimoku_neutral_inside_cloud():
    px = 100.0 + 2.0 * np.sin(2 * np.pi * np.arange(400, dtype=float) / 40.0)
    px[-1] = 100.0                                          # park the close mid-cloud
    df = ichimoku.find_opportunities(history=_hist(**{FLAT_T: px}))
    fl = _row(df, FLAT_T)
    assert fl["direction"] == 0 and fl["signal"] == "—"
    assert "between cloud" in fl["context"]


# --------------------------------------------- Donchian Channel (20d, trigger |pos| >= 80)

def _donchian_px(last: float, n: int = 60) -> np.ndarray:
    px = np.where(np.arange(n) % 2 == 0, 100.0, 110.0).astype(float)  # 20d channel 100-110
    px[-1] = last
    return px


def test_donchian_long_short_neutral():
    hist = _hist(**{LONG_T: _donchian_px(120.0),            # close through the roof
                    SHORT_T: _donchian_px(90.0),            # close through the floor
                    FLAT_T: _donchian_px(105.0)})           # mid-channel
    df = donchian.find_opportunities(history=hist)
    lo, sh, fl = _row(df, LONG_T), _row(df, SHORT_T), _row(df, FLAT_T)
    assert lo["direction"] == 1 and lo["signal"].startswith("Bullish channel breakout")
    assert lo["metric"] >= donchian.DEFAULT_TRIGGER         # well beyond +80 (clipped at +140)
    assert "new 20-day high" in lo["context"]
    assert sh["direction"] == -1 and sh["signal"].startswith("Bearish channel breakout")
    assert sh["metric"] <= -donchian.DEFAULT_TRIGGER
    assert "new 20-day low" in sh["context"]
    assert fl["direction"] == 0 and fl["signal"] == "—"
    assert abs(fl["metric"]) < donchian.DEFAULT_TRIGGER


# ------------------------------------------------------- Aroon (25d, trigger |osc| >= 50)

def test_aroon_long_short_neutral():
    n = 60
    up = 100.0 + 0.5 * np.arange(n)                         # new 25d high today -> osc +100
    dn = 150.0 - 0.5 * np.arange(n)                         # new 25d low today  -> osc -100
    flat = np.full(n, 100.0)                                # extremes adjacent mid-window
    flat[-14] = 110.0                                       # 25d high 13 bars ago
    flat[-13] = 90.0                                        # 25d low  12 bars ago -> osc -4
    hist = _hist(**{LONG_T: up, SHORT_T: dn, FLAT_T: flat})
    df = aroon.find_opportunities(history=hist)
    lo, sh, fl = _row(df, LONG_T), _row(df, SHORT_T), _row(df, FLAT_T)
    assert lo["direction"] == 1 and lo["signal"].startswith("Uptrend (Aroon)")
    assert lo["metric"] == 100.0                            # Up 100 / Down 0
    assert sh["direction"] == -1 and sh["signal"].startswith("Downtrend (Aroon)")
    assert sh["metric"] == -100.0
    assert fl["direction"] == 0 and fl["signal"] == "—"
    assert abs(fl["metric"]) < aroon.DEFAULT_TRIGGER
