"""Trend / time-series momentum monitor.

Classic CTA-style signal: a fast vs slow moving-average crossover, confirmed
by the trailing ~3-month return. Long if fast > slow, short if fast < slow.
"""
from __future__ import annotations

import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name
from .base import frame

FAST, SLOW, MOM_WINDOW = 20, 100, 63
STRATEGY = "Trend"


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in TREND_UNIVERSE:
        px = history[t].dropna()
        if len(px) < SLOW + 5:
            continue
        fast = px.rolling(FAST).mean().iloc[-1]
        slow = px.rolling(SLOW).mean().iloc[-1]
        mom = px.iloc[-1] / px.iloc[-MOM_WINDOW] - 1.0
        ma_gap = (fast / slow - 1) * 100
        direction = 1 if mom > 0 else -1            # time-series momentum: sign of the 3m return
        context = f"MA{FAST} vs MA{SLOW}: {ma_gap:+.1f}%"
        if (ma_gap > 0) != (mom > 0):
            context += "  (MA trend disagrees - mixed)"
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": "Long" if direction == 1 else "Short",
            "direction": direction,
            "metric": round(mom * 100, 1),
            "metric_label": "3m return %",
            "level": round(float(px.iloc[-1]), 3),
            "context": context,
        })

    df = frame(rows)
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def trend_chart_data(ticker: str, history: pd.DataFrame | None = None, window: int = 252):
    """Price + MA20/MA100 history and the headline trend stats for one product, for the
    on-page Trend chart (mirrors the other strategies' *_chart_data helpers). Returns
    (cdata, info) or (None, None) if there isn't enough history.

    cdata columns: date, price, fast (MA{FAST}), slow (MA{SLOW}) — trimmed to the last
    `window` sessions for display, but the MAs are computed on the FULL series so they're
    correct from the left edge. info carries the signal, the 3-month return (the trigger
    metric), the MA gap, and the start point of the 3-month window (to draw it)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in getattr(history, "columns", []):
        return None, None
    px = history[ticker].dropna()
    if len(px) < SLOW + 5:
        return None, None

    fast_s = px.rolling(FAST).mean()
    slow_s = px.rolling(SLOW).mean()
    full = pd.DataFrame({"date": px.index, "price": px.to_numpy(),
                         "fast": fast_s.to_numpy(), "slow": slow_s.to_numpy()})

    last = float(px.iloc[-1])
    base = float(px.iloc[-MOM_WINDOW])
    mom = last / base - 1.0
    f, s = float(fast_s.iloc[-1]), float(slow_s.iloc[-1])
    ma_gap = (f / s - 1.0) * 100.0
    direction = 1 if mom > 0 else -1
    info = {
        "name": name(ticker), "ticker": ticker,
        "last": last, "fast": f, "slow": s,
        "mom": mom, "ma_gap": ma_gap, "direction": direction,
        "mixed": (ma_gap > 0) != (mom > 0),
        "signal": "Long" if direction == 1 else "Short",
        "fast_w": FAST, "slow_w": SLOW, "mom_window": MOM_WINDOW,
        "mom_date": px.index[-MOM_WINDOW], "mom_price": base,
    }
    cdata = full.tail(window).reset_index(drop=True)
    return cdata, info
