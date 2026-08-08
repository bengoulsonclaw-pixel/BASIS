"""Signal-direction fixtures for the momentum / volume family.

Each test engineers a synthetic close (and, for the volume-aware strategies, volume)
history where a textbook setup unambiguously holds, then asserts find_opportunities
flags that ticker with the right SIGN:

  * momentum.py  — Momentum (RSI/MACD): fresh MACD cross both ways, RSI divergence
                   both ways, and a no-event neutral.
  * bollinger.py — Bollinger Squeeze: close through the upper/lower band, a coiled
                   squeeze leaning up, and a wide-band mid-price neutral.
  * obv.py       — On-Balance Volume: volume-confirmed breakout (long), bearish OBV
                   divergence (price higher high on fading volume-pressure, short),
                   and a dead-flat neutral that produces no read.
  * mfi.py       — Money Flow Index: one-way flow washouts both ways and a balanced
                   chop neutral that produces no read.

Everything is deterministic and built in-test (conftest forces DATAFEED_MODE=mock);
ticker names are synthetic ("TESTLONG X", ...) so nothing in the real universe can
interfere — the strategies iterate history columns and fall back to the raw ticker
for unknown names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies import bollinger, mfi, momentum, obv

N_MOM = 418          # enough for MACD warm-up + a 40-bar divergence window, with margin


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _row(df: pd.DataFrame, ticker: str) -> pd.Series:
    sub = df[df["instruments"] == ticker]
    assert len(sub) == 1, f"expected exactly one row for {ticker}, got {len(sub)}"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Momentum (RSI/MACD)
# ---------------------------------------------------------------------------

def _momentum_history() -> pd.DataFrame:
    t = np.arange(N_MOM)

    # LONG — accelerating downtrend keeps the MACD line pinned below its signal
    # (and RSI depressed), then a sharp 3-bar rally forces a FRESH bullish cross.
    down = 300.0 - 0.1 * t - 0.001 * t ** 2
    long_px = down.copy()
    long_px[-3:] = long_px[-4] + 1.5 * np.arange(1, 4)

    # SHORT — the mirror: accelerating uptrend, then a 3-bar break = fresh bearish cross.
    up = 100.0 + 0.1 * t + 0.001 * t ** 2
    short_px = up.copy()
    short_px[-3:] = short_px[-4] - 1.5 * np.arange(1, 4)

    # NEUTRAL — a slow period-60 sine: at N_MOM=418 bars the latest MACD cross is 8
    # bars old (stale, > FRESH_BARS) and the equal sine peaks defeat the strict
    # divergence comparisons, so there is no EVENT -> flat. (A linear trend is NOT a
    # safe neutral here: MACD line - signal converges to ~0 and floating-point jitter
    # can manufacture a phantom "fresh" cross.)
    neut_px = 100.0 + 5.0 * np.sin(2 * np.pi * t / 60)

    # DIV SHORT — textbook bearish RSI divergence: a steep rally to high #1 (RSI hot),
    # a pullback, then a slow grind to a marginally HIGHER price high on visibly less
    # momentum (RSI lower), confirmed by three down bars (the swing needs SWING_K bars
    # after it, and must land within DIV_MAX_AGE of the end).
    div_short = 100.0 + 1.0 * np.sin(2 * np.pi * t / 30)
    div_short[380:396] = np.linspace(div_short[379], 120.0, 17)[1:]
    div_short[396:404] = np.linspace(120.0, 112.0, 9)[1:]
    div_short[404:415] = np.linspace(112.0, 121.0, 12)[1:]
    div_short[415:] = 121.0 - 0.3 * np.arange(1, 4)

    # DIV LONG — the mirror bullish divergence: washout low, bounce, slow slide to a
    # marginally LOWER low with RSI making a higher low.
    div_long = 200.0 - 1.0 * np.sin(2 * np.pi * t / 30)
    div_long[380:396] = np.linspace(div_long[379], 180.0, 17)[1:]
    div_long[396:404] = np.linspace(180.0, 188.0, 9)[1:]
    div_long[404:415] = np.linspace(188.0, 179.0, 12)[1:]
    div_long[415:] = 179.0 + 0.3 * np.arange(1, 4)

    return pd.DataFrame({
        "TESTLONG X": long_px,
        "TESTSHORT X": short_px,
        "TESTFLAT X": neut_px,
        "TESTDIVSHORT X": div_short,
        "TESTDIVLONG X": div_long,
    }, index=_idx(N_MOM))


def test_momentum_fresh_bull_cross_flags_long():
    row = _row(momentum.find_opportunities(history=_momentum_history()), "TESTLONG X")
    assert row["direction"] == 1
    assert row["signal"] == "Bullish momentum"
    assert row["metric"] >= momentum.DEFAULT_TRIGGER
    assert "bullish MACD cross" in row["context"]


def test_momentum_fresh_bear_cross_flags_short():
    row = _row(momentum.find_opportunities(history=_momentum_history()), "TESTSHORT X")
    assert row["direction"] == -1
    assert row["signal"] == "Bearish momentum"
    assert row["metric"] <= -momentum.DEFAULT_TRIGGER
    assert "bearish MACD cross" in row["context"]


def test_momentum_bullish_rsi_divergence_flags_long():
    row = _row(momentum.find_opportunities(history=_momentum_history()), "TESTDIVLONG X")
    assert row["direction"] == 1
    assert row["signal"] == "Bullish momentum"
    assert "Bullish RSI divergence" in row["context"]


def test_momentum_bearish_rsi_divergence_flags_short():
    row = _row(momentum.find_opportunities(history=_momentum_history()), "TESTDIVSHORT X")
    assert row["direction"] == -1
    assert row["signal"] == "Bearish momentum"
    assert "Bearish RSI divergence" in row["context"]


def test_momentum_no_event_stays_flat():
    # A steady trend with no fresh cross and no divergence is NOT a momentum event —
    # the row publishes flat (the RSI-extreme-alone false signal was deliberately removed).
    row = _row(momentum.find_opportunities(history=_momentum_history()), "TESTFLAT X")
    assert row["direction"] == 0
    assert row["signal"] == "—"
    assert row["metric"] == 0.0


# ---------------------------------------------------------------------------
# Bollinger Squeeze
# ---------------------------------------------------------------------------

N_BB = 421   # >= MIN_HISTORY (PCTL_WINDOW 252 + BB_WINDOW 20)


def _bollinger_history() -> pd.DataFrame:
    t = np.arange(N_BB)
    # Period-20 sine: every rolling 20-bar window spans one full cycle, so the
    # bandwidth is constant within a regime — no accidental squeezes.
    wide = 100.0 + 8.0 * np.sin(2 * np.pi * t / 20)

    quiet = wide.copy()
    quiet[150:] = 100.0 + 0.15 * np.sin(2 * np.pi * t[150:] / 20)

    long_px = quiet.copy();  long_px[-1] = 102.0    # last close far above the upper band
    short_px = quiet.copy(); short_px[-1] = 98.0    # ... and far below the lower band

    # Squeeze-watch: volatile first year, then a concave rise (shrinking daily gains)
    # drives bandwidth to its 1-year minimum while price rides the upper half of the band.
    squeeze = wide.copy()
    squeeze[250:] = squeeze[249] + np.cumsum(np.linspace(0.2, 0.02, N_BB - 250))

    return pd.DataFrame({
        "TESTLONG X": long_px,
        "TESTSHORT X": short_px,
        "TESTFLAT X": wide,
        "TESTSQUEEZE X": squeeze,
    }, index=_idx(N_BB))


def test_bollinger_upper_band_break_flags_long():
    row = _row(bollinger.find_opportunities(history=_bollinger_history()), "TESTLONG X")
    assert row["direction"] == 1
    assert "Upside break" in row["signal"]
    assert "ABOVE upper band" in row["context"]


def test_bollinger_lower_band_break_flags_short():
    row = _row(bollinger.find_opportunities(history=_bollinger_history()), "TESTSHORT X")
    assert row["direction"] == -1
    assert "Downside break" in row["signal"]
    assert "BELOW lower band" in row["context"]


def test_bollinger_tight_coil_upper_half_leans_long():
    row = _row(bollinger.find_opportunities(history=_bollinger_history()), "TESTSQUEEZE X")
    assert row["direction"] == 1
    assert row["signal"] == "Squeeze — upside watch"
    # bandwidth at its 1-year low -> intensity ~100, comfortably over the page trigger
    assert row["metric"] >= bollinger.DEFAULT_TRIGGER


def test_bollinger_wide_bands_mid_price_stays_flat():
    row = _row(bollinger.find_opportunities(history=_bollinger_history()), "TESTFLAT X")
    assert row["direction"] == 0
    assert row["signal"] == "—"
    assert row["metric"] == 0.0


# ---------------------------------------------------------------------------
# On-Balance Volume
# ---------------------------------------------------------------------------

N_VOL = 400   # analysis runs on the 120-bar tail; MIN_HISTORY is 90


def _obv_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = N_VOL
    # LONG — steady uptrend on constant volume: price AND OBV both close at their
    # 55-day highs -> "Breakout confirmed by volume".
    p_long = 100.0 + 0.1 * np.arange(n)
    v_long = np.full(n, 1000.0)

    # SHORT — bearish OBV divergence in the 120-bar analysis tail: rally to 110 on
    # solid volume, a HEAVY-volume pullback (OBV gives most of it back), then a thin
    # low-volume grind to a marginally higher price high — price HH, OBV LH — with a
    # few down bars to confirm the swing inside the 15-bar recency gate.
    p_short = np.full(n, 100.0)
    v_short = np.full(n, 1000.0)
    q = n - 120
    p_short[q:q + 61] = np.linspace(100.0, 110.0, 61)
    p_short[q + 61:q + 76] = np.linspace(110.0, 105.0, 16)[1:]
    v_short[q + 61:q + 76] = 2000.0
    p_short[q + 76:q + 114] = np.linspace(105.0, 112.0, 39)[1:]
    v_short[q + 76:q + 114] = 300.0
    p_short[q + 114:] = 112.0 - 0.1 * np.arange(1, 7)
    v_short[q + 114:] = 500.0

    # NEUTRAL — dead-flat price: no divergence, no breakout, no hidden flow -> no read.
    p_flat = np.full(n, 100.0)
    v_flat = np.full(n, 1000.0)

    hist = pd.DataFrame({"TESTLONG X": p_long, "TESTSHORT X": p_short,
                         "TESTFLAT X": p_flat}, index=_idx(n))
    vols = pd.DataFrame({"TESTLONG X": v_long, "TESTSHORT X": v_short,
                         "TESTFLAT X": v_flat}, index=_idx(n))
    return hist, vols


def test_obv_volume_confirmed_breakout_flags_long():
    hist, vols = _obv_frames()
    row = _row(obv.find_opportunities(history=hist, volume=vols), "TESTLONG X")
    assert row["direction"] == 1
    assert "Breakout confirmed by volume" in row["signal"]
    assert row["metric"] >= obv.FLAG_SCORE


def test_obv_bearish_divergence_flags_short():
    hist, vols = _obv_frames()
    row = _row(obv.find_opportunities(history=hist, volume=vols), "TESTSHORT X")
    assert row["direction"] == -1
    assert "Bearish OBV divergence" in row["signal"]
    assert row["metric"] <= -obv.FLAG_SCORE
    assert "HH" in row["context"] and "LH" in row["context"]


def test_obv_flat_tape_produces_no_read():
    hist, vols = _obv_frames()
    df = obv.find_opportunities(history=hist, volume=vols)
    assert "TESTFLAT X" not in set(df["instruments"])


# ---------------------------------------------------------------------------
# Money Flow Index
# ---------------------------------------------------------------------------

def _mfi_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = N_VOL
    t = np.arange(n)
    # LONG — relentless one-way selling: every dollar of flow is negative, MFI pins
    # at 0 (deep under the OS=20 band) and RSI agrees -> "Oversold on real flow".
    p_long = 200.0 - 0.2 * t

    # SHORT — strong buying with a token down-tick every 10th bar (a spotless uptrend
    # makes Wilder's RSI 0/0 -> NaN and the market is skipped): MFI ~86 over the OB=80
    # band with RSI validating -> "Overbought on real flow".
    p_short = 100.0 + np.cumsum(np.where(t % 10 == 9, -0.05, 0.5))

    # NEUTRAL — balanced chop: up- and down-flow match, MFI ~50, no extreme, no
    # divergence-style read -> the market produces no row at all.
    p_flat = 100.0 + 0.25 * ((-1.0) ** t)

    v = np.full(n, 1000.0)
    hist = pd.DataFrame({"TESTLONG X": p_long, "TESTSHORT X": p_short,
                         "TESTFLAT X": p_flat}, index=_idx(n))
    vols = pd.DataFrame({c: v for c in hist.columns}, index=_idx(n))
    return hist, vols


def test_mfi_oversold_on_real_flow_flags_long():
    hist, vols = _mfi_frames()
    row = _row(mfi.find_opportunities(history=hist, volume=vols), "TESTLONG X")
    assert row["direction"] == 1
    assert "Oversold on real flow" in row["signal"]
    assert row["metric"] >= mfi.FLAG_SCORE
    assert "validates RSI" in row["context"]


def test_mfi_overbought_on_real_flow_flags_short():
    hist, vols = _mfi_frames()
    row = _row(mfi.find_opportunities(history=hist, volume=vols), "TESTSHORT X")
    assert row["direction"] == -1
    assert "Overbought on real flow" in row["signal"]
    assert row["metric"] <= -mfi.FLAG_SCORE
    assert "validates RSI" in row["context"]


def test_mfi_balanced_chop_produces_no_read():
    hist, vols = _mfi_frames()
    df = mfi.find_opportunities(history=hist, volume=vols)
    assert "TESTFLAT X" not in set(df["instruments"])
