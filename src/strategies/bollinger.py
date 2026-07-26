"""Bollinger Squeeze monitor — spots products coiling inside compressed bands.

*Bollinger Bands* wrap a moving average in a ±2σ envelope: the **mid** is the
20-day SMA of the close and the **upper/lower** bands sit `K_SIGMA` rolling
standard deviations either side. **Bandwidth** = (upper − lower) / mid is the
band's width as a fraction of price — a direct read on how volatile the market
has been.

A **squeeze** is bandwidth sitting in a LOW percentile of its own trailing year
(percentile over ~252 bars): volatility has wound itself tight, and a tight coil
tends to release in a sharp directional move. The textbook trade is the **band
breakout** that follows — a close back outside the envelope, especially fresh out
of a squeeze:

    close > upper  -> upside break   (dir +1)
    close < lower  -> downside break (dir −1)

This page answers "what's coiled and about to go?" It scans the book, scores each
product's squeeze intensity as a 0–100 number (100 = bandwidth at its tightest of
the year) and signs it by the lean — **+ for an upside break / upper-half coil, −
for a downside one** — so it drops straight into the dashboard's symmetric ±trigger
machinery (see src/specs.py): intensity ≥ +t flags an upside setup, ≤ −t a downside
one.

Close-based: the bands, the bandwidth percentile and the breakout are all built
from settlement closes (the same get_history seam in mock and live). The rolling
std is the SAMPLE standard deviation (pandas default, ddof=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Bollinger Squeeze"

# --- band + squeeze knobs (trading days, all tunable) ---
BB_WINDOW = 20        # SMA / rolling-std window for the bands (classic Bollinger 20)
K_SIGMA = 2.0         # band half-width in rolling standard deviations (classic 2σ)
PCTL_WINDOW = 252     # trailing window the bandwidth is percentile-ranked against (~1y)
SQUEEZE_PCTL = 20.0   # bandwidth ≤ this percentile of its own year => "in a squeeze"
MID_BAND = 0.10       # %B within ±this of the mid counts as "genuinely mid" (no lean)

# The page's default intensity trigger (mirrors specs.py["Bollinger Squeeze"]["default"]).
DEFAULT_TRIGGER = 80.0

# Minimum history: a full percentile window plus a band window to seed it.
MIN_HISTORY = PCTL_WINDOW + BB_WINDOW


def _bands(px: pd.Series):
    """Mid / upper / lower / bandwidth series for a close series `px`.

    mid = SMA(BB_WINDOW); upper, lower = mid ± K_SIGMA·rolling std(BB_WINDOW)
    (sample std, ddof=1); bandwidth = (upper − lower) / mid."""
    mid = px.rolling(BB_WINDOW).mean()
    sd = px.rolling(BB_WINDOW).std()                       # sample std (ddof=1)
    upper, lower = mid + K_SIGMA * sd, mid - K_SIGMA * sd
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, bandwidth


def _pctl_rank(s: pd.Series, window: int = PCTL_WINDOW) -> float:
    """Percentile (0..100) of the LAST value of `s` within its trailing `window`
    (inclusive). 0 = lowest of the window (tightest bands), 100 = highest."""
    tail = s.dropna().tail(window)
    if len(tail) < 2:
        return float("nan")
    last = tail.iloc[-1]
    return float((tail <= last).mean() * 100.0)


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at an intensity trigger — matches specs.reflag_rows so the
    table re-flags identically when the user moves the slider. The metric is SIGNED
    squeeze intensity (+ upside lean / − downside lean); the sign already carries the
    breakout-vs-coil distinction, which `_describe` spells out in words."""
    if metric >= trigger:
        return "Squeeze (coiled)", 1
    if metric <= -trigger:
        return "Squeeze (coiled)", -1
    return "—", 0


def _scan_one(px: pd.Series) -> dict | None:
    """Bollinger state for one product's close series, or None without enough history.

    Returns the band levels, the bandwidth percentile, the squeeze flag, the breakout
    state and the SIGNED intensity metric, plus the human signal/context the dashboard
    and report share."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    mid_s, upper_s, lower_s, bw_s = _bands(s)
    bw = bw_s.dropna()
    if bw.empty:
        return None

    close = float(s.iloc[-1])
    mid, upper, lower = float(mid_s.iloc[-1]), float(upper_s.iloc[-1]), float(lower_s.iloc[-1])
    if not np.isfinite(mid) or mid <= 0 or not np.isfinite(upper - lower):
        return None

    pctl = _pctl_rank(bw_s)                                # 0..100, low = tight bands
    if not np.isfinite(pctl):
        return None
    squeeze = pctl <= SQUEEZE_PCTL
    intensity = 100.0 - pctl                               # 0..100, high = very squeezed

    # %B: where the close sits across the band (0 = lower, 0.5 = mid, 1 = upper). Used to
    # lean a coiled-but-not-broken market and to spot the breakout (%B > 1 / < 0).
    span = upper - lower
    pctb = (close - lower) / span if span > 0 else 0.5
    breakout = "up" if close > upper else "down" if close < lower else "none"

    # Direction: a firing breakout wins; else lean a squeeze by which half price sits in;
    # else (genuinely mid, or no squeeze) flat. The metric is signed by this same lean.
    if breakout == "up":
        direction = 1
    elif breakout == "down":
        direction = -1
    elif squeeze and pctb >= 0.5 + MID_BAND:
        direction = 1
    elif squeeze and pctb <= 0.5 - MID_BAND:
        direction = -1
    else:
        direction = 0
    metric = intensity * direction                        # SIGNED squeeze intensity

    signal = _signal_label(breakout, squeeze, direction)
    context = _describe(pctl, squeeze, breakout, pctb, direction)
    return {"close": close, "mid": mid, "upper": upper, "lower": lower,
            "bandwidth": float(bw_s.iloc[-1]), "bandwidth_pctl": round(pctl, 1),
            "squeeze": bool(squeeze), "breakout": breakout, "pctb": float(pctb),
            "direction": int(direction), "metric": round(float(metric), 1),
            "signal": signal, "context": context}


def _signal_label(breakout: str, squeeze: bool, direction: int) -> str:
    """Human signal cell. A live breakout reads as a break (noting if it sprang from a
    squeeze); a coiled market reads as a watch by its lean; everything else is flat."""
    if breakout == "up":
        return "Upside break (from squeeze)" if squeeze else "Upside break"
    if breakout == "down":
        return "Downside break (from squeeze)" if squeeze else "Downside break"
    if squeeze:
        if direction > 0:
            return "Squeeze — upside watch"
        if direction < 0:
            return "Squeeze — downside watch"
        return "Squeeze (coiled)"
    return "—"


def _describe(pctl: float, squeeze: bool, breakout: str, pctb: float, direction: int) -> str:
    """Short human note for the table: the bandwidth percentile, the squeeze/breakout
    state and where price sits in the band."""
    state = f"Bandwidth {_ord(pctl)} pctl"
    state += " (squeeze)" if squeeze else " (bands wide)"
    if breakout == "up":
        return f"{state}; close broke ABOVE upper band — upside break" + (" from squeeze" if squeeze else "")
    if breakout == "down":
        return f"{state}; close broke BELOW lower band — downside break" + (" from squeeze" if squeeze else "")
    if squeeze:
        where = ("price at upper half — watch upside break" if direction > 0
                 else "price at lower half — watch downside break" if direction < 0
                 else "price mid-band — direction unclear")
        return f"{state}; {where}"
    return f"{state}; no squeeze, no breakout"


def _ord(p: float) -> str:
    """Percentile as an ordinal, e.g. 8.0 -> '8th', 21.0 -> '21st'."""
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in TREND_UNIVERSE:
        if t not in history.columns:
            continue
        d = _scan_one(history[t])
        if d is None:
            continue
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": d["signal"],
            "direction": d["direction"],
            "metric": d["metric"],
            "metric_label": "squeeze (0-100)",
            "level": round(d["close"], 4),
            "context": d["context"],
        })

    df = frame(rows)
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def bollinger_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price + the Bollinger envelope (mid / upper / lower) and bandwidth over the last
    ~180 bars, plus an info dict for the dashboard. (DataFrame, info) or (None, None) if
    there isn't enough history.

    df columns: ['date','price','mid','upper','lower','bandwidth'].
    info keys: 'bandwidth_pctl','squeeze','breakout','signal','direction','metric','price'.
    """
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _scan_one(history[ticker])
    if d is None:
        return None, None

    s = history[ticker].dropna()
    mid_s, upper_s, lower_s, bw_s = _bands(s)
    tail = s.tail(180).index
    df = pd.DataFrame({
        "date": tail,
        "price": s.reindex(tail).to_numpy(dtype=float),
        "mid": mid_s.reindex(tail).to_numpy(dtype=float),
        "upper": upper_s.reindex(tail).to_numpy(dtype=float),
        "lower": lower_s.reindex(tail).to_numpy(dtype=float),
        "bandwidth": bw_s.reindex(tail).to_numpy(dtype=float),
    })
    info = {
        "bandwidth_pctl": d["bandwidth_pctl"],
        "squeeze": d["squeeze"],
        "breakout": d["breakout"],
        "signal": d["signal"],
        "direction": d["direction"],
        "metric": d["metric"],
        "price": d["close"],
    }
    return df, info
