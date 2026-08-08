"""Donchian channel breakout — the classic price-channel / turtle trend-following signal.

A *Donchian channel* is simply the highest and lowest close over the last N sessions. The trade
is the **breakout**: a close through the top of the channel is the canonical long entry, a close
through the bottom the short — the bread-and-butter signal of systematic trend-followers (the
"turtle" system, and most managed-futures models to this day).

Per product, on the close series, we take the channel over the **prior** N closes (excluding
today, so "a new N-day high" is genuinely new) and score where today's close sits across it:

    position = (close − lower) ÷ (upper − lower)        # 0 = on the floor, 1 = on the roof

reported as a signed 0–100 **channel position**: `metric = 100 × (2·position − 1)` — so −100 sits
on the lower band, +100 on the upper band, and beyond ±100 price has already broken out. Signed
this way it drops straight into the dashboard's symmetric ±trigger machinery (specs.py): a metric
≥ +t flags a bullish breakout setup (at/near a new N-day high), ≤ −t a bearish one.

Close-based (settlement closes — the same get_history seam in mock and live), so it runs
identically on the futures book and the equity universe. No volume input: the turtle breakout is
a pure price signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (suite convention)
from ..universe import TREND_UNIVERSE, name
from .base import frame

STRATEGY = "Donchian Channel"

WINDOW = 20           # channel look-back (sessions) — the classic short-term turtle entry channel
MIN_HISTORY = WINDOW + 5
DEFAULT_TRIGGER = 80.0  # flag when |channel position| ≥ this (mirrors specs.py["Donchian Channel"])


def _analyse(px: pd.Series) -> dict | None:
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    c = s.to_numpy(dtype=float)
    prior = c[-(WINDOW + 1):-1]                          # the N closes BEFORE today
    upper, lower = float(prior.max()), float(prior.min())
    rng = upper - lower
    if not np.isfinite(rng) or rng <= 0:
        return None
    now = float(c[-1])
    pos = (now - lower) / rng                            # 0 = lower band, 1 = upper band, >1 / <0 = broken out
    metric = float(np.clip(100.0 * (2.0 * pos - 1.0), -140.0, 140.0))
    broke_up, broke_dn = now > upper, now < lower
    mid = 0.5 * (upper + lower)
    # Measured objective = the channel height projected from the broken band; stop = the opposite
    # band (a turtle exits on the reverse channel). Direction-agnostic here; the sign sets which.
    up_target, dn_target = upper + rng, lower - rng
    return {"s": s, "now": now, "upper": upper, "lower": lower, "mid": mid, "rng": rng,
            "pos": pos, "metric": metric, "broke_up": broke_up, "broke_dn": broke_dn,
            "up_target": up_target, "dn_target": dn_target}


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a channel-position trigger — matches specs.reflag_rows."""
    if metric >= trigger:
        return "Bullish channel breakout", 1
    if metric <= -trigger:
        return "Bearish channel breakout", -1
    return "—", 0


def _context(d: dict, direction: int) -> str:
    where = (f"new {WINDOW}-day high (broke above {d['upper']:g})" if d["broke_up"]
             else f"new {WINDOW}-day low (broke below {d['lower']:g})" if d["broke_dn"]
             else (f"{d['pos'] * 100:.0f}% up the {WINDOW}-day channel "
                   f"[{d['lower']:g} – {d['upper']:g}]"))
    tail = ""
    if direction > 0:
        tail = f"; objective {d['up_target']:g}, invalidation {d['lower']:g}"
    elif direction < 0:
        tail = f"; objective {d['dn_target']:g}, invalidation {d['upper']:g}"
    return where + tail


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
        flagged = abs(d["metric"]) >= DEFAULT_TRIGGER
        signal, direction = _signal(d["metric"], DEFAULT_TRIGGER)
        arrow = " ▲" if direction > 0 else " ▼" if direction < 0 else ""
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": (signal + arrow) if flagged else "—",
            "direction": direction if flagged else 0,
            "metric": round(d["metric"], 1),            # SIGNED channel position (−100 floor … +100 roof)
            "metric_label": "channel position (0-100)",
            "level": round(d["now"], 4),
            "context": _context(d, direction if flagged else 0),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def donchian_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """(price_df, info): price_df ['date','price','upper','lower'] over the shown window (the
    channel bands are the rolling prior-N high/low); info {'signal','direction','metric','upper',
    'lower','pos','price', ...} — or (None, None)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _analyse(history[ticker])
    if d is None:
        return None, None
    s = d["s"]
    c = s.to_numpy(dtype=float)
    # Rolling channel over the whole series (prior-N high/low), NaN until enough history, so the
    # chart brackets each bar by the channel it faced.
    roll = pd.Series(c, index=s.index)
    upper = roll.shift(1).rolling(WINDOW).max()
    lower = roll.shift(1).rolling(WINDOW).min()
    price_df = pd.DataFrame({"date": s.index, "price": c,
                             "upper": upper.to_numpy(dtype=float),
                             "lower": lower.to_numpy(dtype=float)})
    signal, direction = _signal(d["metric"], DEFAULT_TRIGGER)
    arrow = " ▲" if direction > 0 else " ▼" if direction < 0 else ""
    info = {"signal": (signal + arrow) if direction else signal, "direction": direction,
            "metric": d["metric"], "upper": d["upper"], "lower": d["lower"], "mid": d["mid"],
            "pos": d["pos"], "price": d["now"], "window": WINDOW,
            "target": d["up_target"] if direction > 0 else d["dn_target"] if direction < 0 else None,
            "stop": d["lower"] if direction > 0 else d["upper"] if direction < 0 else None}
    return price_df, info
