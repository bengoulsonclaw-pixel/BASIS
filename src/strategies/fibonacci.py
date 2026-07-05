"""Fibonacci retracement monitor — price reacting at a key Fib level of the dominant swing.

After a strong directional leg, markets often pull back to a **Fibonacci retracement** of
that move — 23.6 / 38.2 / 50 / 61.8 / 78.6 % — before the trend resumes. The "golden zone"
(38.2–61.8 %, the 61.8 % especially) is where pullbacks most often find support in an uptrend
or resistance in a downtrend. This page draws the retracement of each product's dominant
recent swing and flags the ones whose latest close is sitting on a key level:

    up-leg (uptrend)    -> pullback to a key retracement = SUPPORT    -> Long  (buy the dip)
    down-leg (downtrend)-> bounce to a key retracement   = RESISTANCE -> Short (sell the rally)

The dominant swing is the highest high and lowest low over the lookback; whichever is the more
recent sets the leg direction (high after low = up-leg, and vice-versa). The headline is a
0–100 **Fib proximity** (100 = price exactly on a key level), signed **+ for a long (up-leg)
setup, − for a short (down-leg) one**, so it drops straight into the dashboard's symmetric
±trigger machinery (see src/specs.py). Each setup also carries a **target** (the prior swing
extreme — a full retrace-and-resume), a **stop** just beyond the 78.6 % level (a deeper close
breaks the setup) and the reward:risk.

Close-only (the swing, the levels and the distance read are all from settlement closes) and
deterministic. The 50 % level isn't a true Fibonacci ratio but is included by convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Fibonacci Retracement"

# --- detection knobs (trading days / ratios, all tunable) ---
LOOKBACK = 180        # window the dominant swing (high & low) is taken from
MIN_HISTORY = 60      # need a real leg before a retracement means anything
LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]   # the retracement grid (0.5 by convention; 0/100 = the swing anchors)
KEY = [0.382, 0.5, 0.618]                     # the actionable retracements (golden zone)
FAIL = 0.786          # a close beyond this retraces "too far" — the stop reference
NEAR_PCT = 1.0        # within this % of a key level = sitting "on" it (proximity band)
LEG_MIN_PCT = 0.05    # the dominant swing must be at least this big a move to bother

# The page's default proximity trigger (mirrors specs.py["Fibonacci Retracement"]["default"]).
DEFAULT_TRIGGER = 60.0


def _proximity(price: float, level: float) -> float:
    """Unsigned 0..100 closeness of `price` to `level`: 100 = on it, fading to 0 at NEAR_PCT."""
    band = NEAR_PCT / 100.0 * price
    if band <= 0:
        return 0.0
    return float(np.clip(100.0 * (1.0 - abs(price - level) / band), 0.0, 100.0))


def _level_prices(hi: float, lo: float, leg: str) -> dict:
    """Retracement price for each ratio. Up-leg retraces DOWN from the high (0 % = high,
    100 % = low); down-leg retraces UP from the low (0 % = low, 100 % = high)."""
    rng = hi - lo
    if leg == "up":
        return {r: hi - r * rng for r in LEVELS}
    return {r: lo + r * rng for r in LEVELS}


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a proximity trigger — matches specs.reflag_rows so the table
    re-flags identically when the slider moves. + = up-leg pullback (long), − = down-leg bounce."""
    if metric >= trigger:
        return "Long (buy the dip)", 1
    if metric <= -trigger:
        return "Short (sell the rally)", -1
    return "—", 0


def _analyse(px: pd.Series) -> dict | None:
    """The dominant swing's retracement read for a product, or None without enough history."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    s = s.tail(LOOKBACK)
    c = s.to_numpy(dtype=float)
    price = float(c[-1])
    hi_i, lo_i = int(np.argmax(c)), int(np.argmin(c))
    hi, lo = float(c[hi_i]), float(c[lo_i])
    rng = hi - lo
    if hi_i == lo_i or rng <= 0 or rng / price < LEG_MIN_PCT:
        return None

    leg = "up" if hi_i > lo_i else "down"          # which extreme is the more recent
    levels = _level_prices(hi, lo, leg)
    retr = (hi - price) / rng if leg == "up" else (price - lo) / rng   # how far it's pulled back (0..1)

    nearest_ratio = min(KEY, key=lambda r: abs(levels[r] - price))
    nearest = levels[nearest_ratio]
    sign = 1 if leg == "up" else -1
    prox = _proximity(price, nearest)
    metric = sign * prox
    dist_pct = (price - nearest) / price * 100.0

    target = hi if leg == "up" else lo             # retrace then resume to the prior extreme
    stop = levels[FAIL]                            # a close beyond 78.6% breaks the setup
    entry = nearest
    reward, risk = abs(target - entry), abs(entry - stop)
    rr = reward / risk if risk > 1e-12 else float("nan")
    signal, direction = _signal(metric, DEFAULT_TRIGGER)

    return {"series": s, "levels": levels, "leg": leg, "hi": hi, "lo": lo, "hi_i": hi_i, "lo_i": lo_i,
            "price": price, "nearest_ratio": nearest_ratio, "nearest": nearest, "dist_pct": float(dist_pct),
            "proximity": float(prox), "metric": float(metric), "signal": signal, "direction": direction,
            "target": float(target), "stop": float(stop), "rr": float(rr), "retr": float(retr)}


def _context(d: dict) -> str:
    leg = "up-leg" if d["leg"] == "up" else "down-leg"
    zone = "buy-dip" if d["leg"] == "up" else "sell-rally"
    where = "above" if d["dist_pct"] >= 0 else "below"
    rr = f", R:R {d['rr']:.1f}" if np.isfinite(d["rr"]) else ""
    return (f"{leg} {d['lo']:g}→{d['hi']:g}; at {d['nearest_ratio'] * 100:.1f}% retrace ({d['nearest']:g}) "
            f"{abs(d['dist_pct']):.1f}% {where} — {zone}; target {d['target']:g}{rr}")


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in TREND_UNIVERSE:
        if t not in history.columns:
            continue
        d = _analyse(history[t])
        if d is None:
            continue
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": d["signal"],
            "direction": d["direction"],
            "metric": round(d["metric"], 1),       # SIGNED: + up-leg/long, − down-leg/short
            "metric_label": "Fib proximity",
            "level": round(d["price"], 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def fib_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price over the lookback + the Fibonacci retracement grid of the dominant swing, plus an
    info dict for the dashboard. (price_df, info) or (None, None) if there isn't enough history.

    price_df: ['date','price']. info: 'levels' (list of {'ratio','price','key'}), 'leg'
    ('up'/'down'), 'hi','lo','hi_date','lo_date','nearest_ratio','nearest','dist_pct',
    'proximity','signal','direction','target','stop','rr','price'."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _analyse(history[ticker])
    if d is None:
        return None, None

    s = d["series"]
    price_df = pd.DataFrame({"date": s.index, "price": s.to_numpy(dtype=float)})
    info = {
        "levels": [{"ratio": r, "price": d["levels"][r], "key": r in KEY} for r in LEVELS],
        "leg": d["leg"], "hi": d["hi"], "lo": d["lo"],
        "hi_date": s.index[d["hi_i"]], "lo_date": s.index[d["lo_i"]],
        "nearest_ratio": d["nearest_ratio"], "nearest": d["nearest"], "dist_pct": d["dist_pct"],
        "proximity": d["proximity"], "signal": d["signal"], "direction": d["direction"],
        "target": d["target"], "stop": d["stop"], "rr": d["rr"], "price": d["price"],
    }
    return price_df, info
