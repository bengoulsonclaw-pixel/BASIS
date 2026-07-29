"""Momentum oscillator monitor — RSI + MACD, with the divergence as the headline.

Two classic momentum reads on each product's close series, combined into one
signed score:

  * **RSI** — Wilder's 14-period Relative Strength Index. Above 70 is overbought,
    below 30 oversold. (Wilder's smoothing = an EMA with `alpha = 1/period`, so
    the average gain/loss is `ewm(alpha=1/14, adjust=False)` on the up/down moves.)
  * **MACD** — the 12/26 EMA difference, its 9-EMA signal line, and the histogram
    (line − signal). The **cross** is the trade trigger: the line crossing *above*
    the signal is bullish, *below* is bearish; "fresh" means it happened within the
    last ~3 bars.
  * **RSI divergence** — the headline warning. Over the trailing ~40 bars, price
    making a **higher high while RSI makes a lower high** is a *bearish* divergence
    (the rally is running out of momentum); price making a **lower low while RSI
    makes a higher low** is *bullish*. Detected from the two most recent swing
    highs / lows in the window.

The signal direction is **bullish (+1)** if there's a bullish divergence OR RSI is
oversold OR a fresh bullish MACD cross — **bearish (−1)** the mirror — else 0;
divergence is prioritised (mentioned first) because it leads the oscillators.

The headline `metric` is a **signed momentum score** (magnitude ~0..100): how far
RSI sits from its 50 midline (`2 × |RSI − 50|`), plus a divergence bonus and a
fresh-cross bonus, **signed by the setup direction (+ bullish / − bearish)** and
clipped to ~[−100, 100]. Signing it lets the table drop straight into the
dashboard's symmetric ±trigger machinery (see src/specs.py): `metric ≥ +t` flags a
bullish setup, `≤ −t` a bearish one.

Close-only (RSI, MACD and the divergence are all built from settlement closes, the
same get_history seam in mock and live) and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Momentum (RSI/MACD)"

# --- oscillator knobs (trading days, all tunable) ---
RSI_LEN = 14          # Wilder's RSI lookback
RSI_OB, RSI_OS = 70.0, 30.0    # overbought / oversold thresholds
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9    # MACD EMAs: fast, slow, signal
FRESH_BARS = 3        # a MACD cross this-many bars old (or newer) still counts as "fresh"
DIV_WIN = 40          # window for the RSI-divergence swing-high/low comparison
SWING_K = 3           # a swing point is the extreme of a ±SWING_K-bar fractal
DIV_MAX_AGE = 10      # the confirming swing must be this recent — an older divergence has
                      # already played out and describes the move that HAS happened, not a setup
DIV_MIN_RSI_GAP = 3.0  # …and the RSI gap must clear this, so a 1-2 point gap (an artefact of
                      # which bars happen to be picked as swings) doesn't count as a divergence
DIV_BONUS = 25.0      # momentum-score bonus when a divergence is present
CROSS_BONUS = 15.0    # …and when a fresh MACD cross is present
MIN_HISTORY = MACD_SLOW + MACD_SIG + DIV_WIN   # enough bars for a settled MACD + a divergence window

# The page's default momentum-score trigger (mirrors specs.py["Momentum (RSI/MACD)"]).
DEFAULT_TRIGGER = 40.0


def _rsi(px: pd.Series, length: int = RSI_LEN) -> pd.Series:
    """Wilder's RSI on a close series. The average gain / loss use Wilder's
    smoothing (an EMA with `alpha = 1/length`), so RSI = 100 − 100/(1 + RS)
    with `RS = avg_gain / avg_loss`."""
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0, 100.0)         # all-gains window → RSI 100 (avoid 0/0)


def _macd(px: pd.Series):
    """(line, signal, hist): line = EMA(fast) − EMA(slow), signal = EMA(line),
    histogram = line − signal — the standard 12/26/9 MACD."""
    ema_fast = px.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = px.ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=MACD_SIG, adjust=False).mean()
    return line, signal, line - signal


def _cross_age(line: pd.Series, signal: pd.Series) -> tuple[int, int | None]:
    """(cross_dir, bars_since) for the latest MACD line/signal cross. `cross_dir` is
    the sign of (line − signal) today (+1 line above, −1 below); `bars_since` is how
    many bars ago they last crossed (0 = crossed on the latest bar), or None if they
    never cross in the overlapping window."""
    diff = (line - signal).dropna()
    if len(diff) < 2:
        return (1 if (len(diff) and diff.iloc[-1] >= 0) else -1), None
    s = np.sign(diff.to_numpy())
    s[s == 0] = 1
    cross_dir = int(s[-1])
    flips = np.where(s[1:] != s[:-1])[0]            # i where the sign flips between bar i and i+1
    if len(flips) == 0:
        return cross_dir, None
    return cross_dir, int(len(s) - 1 - (flips[-1] + 1))


def _swing_points(v: np.ndarray, k: int = SWING_K) -> tuple[list[int], list[int]]:
    """Indices of swing highs and lows in array `v`: a swing high is a bar that is the
    strict max of its ±k-bar neighbourhood (a low the strict min). Endpoints can't be
    confirmed as fractals, so they're excluded."""
    highs, lows = [], []
    n = len(v)
    for i in range(k, n - k):
        win = v[i - k:i + k + 1]
        if v[i] == win.max() and (win == v[i]).sum() == 1:
            highs.append(i)
        elif v[i] == win.min() and (win == v[i]).sum() == 1:
            lows.append(i)
    return highs, lows


def _divergence(px: np.ndarray, rsi: np.ndarray) -> int:
    """RSI divergence over the last DIV_WIN bars: +1 bullish, −1 bearish, 0 none.

    Compares the two most recent swing highs/lows of price against the matching RSI
    swings: a **bearish** divergence is price higher-high + RSI lower-high; a
    **bullish** one is price lower-low + RSI higher-low. Falls back to comparing the
    window's two halves (front vs back extreme) when fewer than two clean swings are
    present, so a divergence still surfaces on a smooth run.

    Two gates keep it ACTIONABLE rather than merely historical: the confirming swing must be
    within DIV_MAX_AGE bars, and the RSI gap must clear DIV_MIN_RSI_GAP. Without them a
    divergence formed weeks ago — and long since resolved in the rally it predicted — kept
    being reported as a live setup, and a 1-2 point RSI gap that flips with the choice of
    swing bars counted the same as a decisive one."""
    n = min(DIV_WIN, len(px))
    if n < 2 * SWING_K + 3:
        return 0
    p, r = px[-n:], rsi[-n:]
    highs, lows = _swing_points(p)

    def _actionable(j: int, gap: float) -> bool:
        return (n - 1 - j) <= DIV_MAX_AGE and abs(gap) >= DIV_MIN_RSI_GAP

    # Bearish: the two most recent swing highs — price up, RSI down.
    if len(highs) >= 2:
        i, j = highs[-2], highs[-1]
        if p[j] > p[i] and r[j] < r[i] and _actionable(j, r[j] - r[i]):
            return -1
    # Bullish: the two most recent swing lows — price down, RSI up.
    if len(lows) >= 2:
        i, j = lows[-2], lows[-1]
        if p[j] < p[i] and r[j] > r[i] and _actionable(j, r[j] - r[i]):
            return 1

    # Fallback: split the window in half and compare each half's extreme.
    half = n // 2
    fi, bi = int(np.argmax(p[:half])), half + int(np.argmax(p[half:]))
    if p[bi] > p[fi] and r[bi] < r[fi] and _actionable(bi, r[bi] - r[fi]):
        return -1
    fi, bi = int(np.argmin(p[:half])), half + int(np.argmin(p[half:]))
    if p[bi] < p[fi] and r[bi] > r[fi] and _actionable(bi, r[bi] - r[fi]):
        return 1
    return 0


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a momentum-score trigger — matches specs.reflag_rows so
    the table re-flags identically when the user moves the slider."""
    if metric >= trigger:
        return "Bullish momentum", 1
    if metric <= -trigger:
        return "Bearish momentum", -1
    return "—", 0


def _assess(px: pd.Series) -> dict | None:
    """Full momentum read on one product's close series, or None if too short. Returns
    the latest RSI / MACD state, the divergence, the signed setup direction and the
    signed momentum score (the dashboard `metric`)."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    rsi_s = _rsi(s)
    line, sig, hist = _macd(s)
    rsi = float(rsi_s.iloc[-1])
    if not np.isfinite(rsi):
        return None

    cross_dir, bars_since = _cross_age(line, sig)
    fresh = bars_since is not None and bars_since <= FRESH_BARS
    fresh_bull = fresh and cross_dir > 0
    fresh_bear = fresh and cross_dir < 0
    div = _divergence(s.to_numpy(dtype=float), rsi_s.to_numpy(dtype=float))

    # Setup direction. Only an EVENT sets a side — an RSI divergence or a fresh MACD cross.
    # An RSI extreme on its own does NOT: "RSI is over 70 so it's a sell" is the classic false
    # signal in a trend, and it was firing Bearish momentum on markets printing 55-day highs,
    # flatly contradicting the trend and OBV reads on the same chart. Extension instead acts as
    # a VETO — it blocks a signal pointing further into the move already made, which is also
    # what now resolves a bullish-divergence-into-overbought clash to flat.
    bull = ((div > 0) or fresh_bull) and not (rsi > RSI_OB)
    bear = ((div < 0) or fresh_bear) and not (rsi < RSI_OS)
    direction = 1 if (bull and not bear) else -1 if (bear and not bull) else 0

    # Signed momentum score, magnitude ~0..100. The RSI term is how FAVOURABLY RSI sits for
    # THIS direction — room below the midline for a long, above it for a short — so a market
    # already extended the way it is being read scores nothing for it. (It used to be
    # 2*|RSI-50|, which is direction-blind: an RSI of 75 added 50 points to a BULLISH score,
    # i.e. the more overbought a market got the stronger a buy it looked.) The bonuses count
    # only when they agree with the direction.
    magnitude = 2.0 * max(0.0, direction * (50.0 - rsi))
    if div * direction > 0:
        magnitude += DIV_BONUS
    if (fresh_bull and direction > 0) or (fresh_bear and direction < 0):
        magnitude += CROSS_BONUS
    # No event, or an event vetoed by extension, means direction == 0 — which zeroes the
    # magnitude above, so it publishes flat. The old sign fallback (`1 if rsi >= 50`) re-signed
    # precisely those cases into confident calls, so a "flat" verdict never survived to the table.
    metric = float(np.clip(direction * magnitude, -100.0, 100.0))

    return {
        "rsi": rsi,
        "macd_line": float(line.iloc[-1]),
        "macd_signal": float(sig.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "macd_state": "bullish" if cross_dir > 0 else "bearish",
        "bars_since": bars_since,
        "fresh": bool(fresh),
        "fresh_bull": bool(fresh_bull),
        "fresh_bear": bool(fresh_bear),
        "divergence": "bullish" if div > 0 else "bearish" if div < 0 else "none",
        "direction": direction,
        "metric": metric,
        "level": float(s.iloc[-1]),
    }


def _context(d: dict) -> str:
    """Human note for an assessment dict — divergence first (it's the headline), then
    the RSI level and the MACD state."""
    parts = []
    if d["divergence"] != "none":
        parts.append(f"{d['divergence'].capitalize()} RSI divergence")
    if d["rsi"] >= RSI_OB:
        parts.append(f"Overbought RSI {d['rsi']:.0f}")
    elif d["rsi"] <= RSI_OS:
        parts.append(f"Oversold RSI {d['rsi']:.0f}")
    else:
        parts.append(f"RSI {d['rsi']:.0f}")

    if d["fresh_bull"]:
        parts.append("bullish MACD cross")
    elif d["fresh_bear"]:
        parts.append("bearish MACD cross")
    elif d["macd_state"] == "bullish":
        parts.append("MACD above signal")
    else:
        parts.append("MACD rolling over" if d["divergence"] == "bearish" else "MACD below signal")
    return "; ".join(parts)


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in list(history.columns):        # universe-agnostic (was TREND_UNIVERSE)
        if t not in history.columns:
            continue
        d = _assess(history[t])
        if d is None:
            continue
        signal, direction = _signal(d["metric"], DEFAULT_TRIGGER)
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": signal,
            "direction": direction,
            "metric": round(d["metric"], 1),
            "metric_label": "momentum score",
            "level": round(d["level"], 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    # Rank by how strong the momentum setup is (|score|).
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def momentum_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price + the RSI and MACD oscillators over the last ~180 bars, plus an info dict
    for the dashboard. (DataFrame, info) or (None, None) if there isn't enough history.

    df columns : date, price, rsi, macd, macd_signal, macd_hist
    info keys  : rsi, macd_state, divergence, signal, direction, metric, price
    """
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _assess(history[ticker])
    if d is None:
        return None, None

    s = history[ticker].dropna()
    rsi_s = _rsi(s)
    line, sig, hist = _macd(s)
    out = pd.DataFrame({
        "date": s.index,
        "price": s.to_numpy(dtype=float),
        "rsi": rsi_s.to_numpy(dtype=float),
        "macd": line.to_numpy(dtype=float),
        "macd_signal": sig.to_numpy(dtype=float),
        "macd_hist": hist.to_numpy(dtype=float),
    }).tail(180).reset_index(drop=True)

    signal, _ = _signal(d["metric"], DEFAULT_TRIGGER)
    info = {
        "rsi": d["rsi"],
        "macd_state": d["macd_state"],
        "divergence": d["divergence"],
        "signal": signal,
        "direction": d["direction"],
        "metric": d["metric"],
        "price": d["level"],
    }
    return out, info
