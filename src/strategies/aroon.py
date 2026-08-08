"""Aroon — trend strength and freshness from how recently price made a new extreme.

Aroon (Chande) answers a question none of the other trend reads do: *how strong is the trend, and
how recently did it last make a new high or low?* Over the last N sessions it measures the time
since the highest and lowest close:

    Aroon Up   = 100 × (N − sessions since the N-day high)  ÷ N   # 100 = a new high today
    Aroon Down = 100 × (N − sessions since the N-day low)   ÷ N   # 100 = a new low today

A market powering higher keeps printing fresh highs → Aroon Up pinned near 100, Down near 0. A
market going nowhere makes neither → both drift low. The **Aroon oscillator** = Up − Down runs
−100…+100: strongly positive = a healthy uptrend, strongly negative = a downtrend, near zero =
no trend (chop). That oscillator is the signed 0–100 **metric**, so it drops straight into the
dashboard's symmetric ±trigger machinery (specs.py): ≥ +t flags a strong uptrend, ≤ −t a strong
downtrend; |value| is the trend strength.

Close-based (settlement closes — the same get_history seam in mock and live), so it runs
identically on the futures book and the equity universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (suite convention)
from ..universe import TREND_UNIVERSE, name
from .base import frame

STRATEGY = "Aroon"

WINDOW = 25           # Aroon look-back (sessions) — the standard period
MIN_HISTORY = WINDOW + 5
DEFAULT_TRIGGER = 50.0  # flag when |Aroon oscillator| ≥ this (mirrors specs.py["Aroon"])


def _aroon_last(c: np.ndarray) -> tuple[float, float, int, int]:
    """(Aroon Up, Aroon Down, bars-since-high, bars-since-low) on the final N-window of closes."""
    w = c[-(WINDOW + 1):]                                # last N+1 closes; w[-1] = today
    hi_idx = int(np.argmax(w))                           # 0 = N bars ago … WINDOW = today
    lo_idx = int(np.argmin(w))
    up = 100.0 * hi_idx / WINDOW
    dn = 100.0 * lo_idx / WINDOW
    return up, dn, WINDOW - hi_idx, WINDOW - lo_idx      # bars-since = N − idx


def _analyse(px: pd.Series) -> dict | None:
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    c = s.to_numpy(dtype=float)
    up, dn, since_hi, since_lo = _aroon_last(c)
    osc = up - dn
    if not np.isfinite(osc):
        return None
    return {"s": s, "up": up, "dn": dn, "osc": float(osc), "since_hi": since_hi,
            "since_lo": since_lo, "price": float(c[-1])}


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at an Aroon-oscillator trigger — matches specs.reflag_rows."""
    if metric >= trigger:
        return "Uptrend (Aroon)", 1
    if metric <= -trigger:
        return "Downtrend (Aroon)", -1
    return "—", 0


def _context(d: dict) -> str:
    return (f"Aroon Up {d['up']:.0f} / Down {d['dn']:.0f} (osc {d['osc']:+.0f}); "
            f"last {WINDOW}-day high {d['since_hi']}d ago, low {d['since_lo']}d ago")


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    uni = list(TREND_UNIVERSE)
    if history is None:
        history = get_history(uni)

    rows = []
    for t in list(history.columns):                     # universe-agnostic (equities pass their own)
        try:
            d = _analyse(history[t])
        except Exception:
            continue
        if d is None:
            continue
        flagged = abs(d["osc"]) >= DEFAULT_TRIGGER
        signal, direction = _signal(d["osc"], DEFAULT_TRIGGER)
        arrow = " ▲" if direction > 0 else " ▼" if direction < 0 else ""
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": (signal + arrow) if flagged else "—",
            "direction": direction if flagged else 0,
            "metric": round(d["osc"], 1),               # SIGNED Aroon oscillator (−100 … +100)
            "metric_label": "Aroon oscillator",
            "level": round(d["price"], 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def aroon_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """(price_df, info): price_df ['date','price','aroon_up','aroon_down'] over the series; info
    {'signal','direction','metric','up','down','price', ...} — or (None, None)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _analyse(history[ticker])
    if d is None:
        return None, None
    s = d["s"]
    c = pd.Series(s.to_numpy(dtype=float), index=s.index)
    # Rolling Aroon Up/Down over the series (position of the high/low within each trailing window).
    up = c.rolling(WINDOW + 1).apply(lambda x: 100.0 * int(np.argmax(x)) / WINDOW, raw=True)
    dn = c.rolling(WINDOW + 1).apply(lambda x: 100.0 * int(np.argmin(x)) / WINDOW, raw=True)
    price_df = pd.DataFrame({"date": s.index, "price": c.to_numpy(dtype=float),
                             "aroon_up": up.to_numpy(dtype=float),
                             "aroon_down": dn.to_numpy(dtype=float)})
    signal, direction = _signal(d["osc"], DEFAULT_TRIGGER)
    arrow = " ▲" if direction > 0 else " ▼" if direction < 0 else ""
    info = {"signal": (signal + arrow) if direction else signal, "direction": direction,
            "metric": d["osc"], "up": d["up"], "down": d["dn"], "osc": d["osc"],
            "price": d["price"], "window": WINDOW,
            "since_hi": d["since_hi"], "since_lo": d["since_lo"]}
    return price_df, info
