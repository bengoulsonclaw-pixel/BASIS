"""Ichimoku Kinko Hyo (the cloud) — trend regime + multi-line confirmation.

Ichimoku reads a market off five lines, and its edge is *confluence*: price clear of the
cloud, the Tenkan/Kijun cross agreeing, the future cloud the right colour, and the lagging
span confirming — the more that align, the stronger the regime. Standard 9 / 26 / 52 / 26
settings.

  Tenkan-sen (conversion)  = (9-high + 9-low) / 2
  Kijun-sen (base)         = (26-high + 26-low) / 2
  Senkou Span A (lead A)   = (Tenkan + Kijun) / 2, plotted 26 ahead   } the cloud (Kumo)
  Senkou Span B (lead B)   = (52-high + 52-low) / 2, plotted 26 ahead  }
  Chikou (lagging)         = close, plotted 26 behind

This suite is **close-only** (settlement series; fixed income on yields), so the n-period
high/low use the rolling max/min of closes — a standard close-based Ichimoku. The signal is
the price's position relative to the cloud (Ichimoku's first rule: **no trade inside the
cloud**), scored by how many of the three other elements confirm it:

  * price above the cloud → constructive; below → cautious; inside → no signal
  * **score 0–100** = 40 + 16 × (# of {TK cross, future-cloud colour, lagging span} agreeing)
    + a small clear-of-cloud bonus; a **fresh TK cross** or a **cloud breakout** in the last
    ~6 sessions is highlighted (the actionable trigger). Flagged at score ≥ 58.

Signed by direction (+ constructive / − cautious). Fixed income counted on yields
(constructive = yields above the cloud = the future lower). Close-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name
from .base import frame

STRATEGY = "Ichimoku Cloud"

TENKAN, KIJUN, SPAN_B, SHIFT = 9, 26, 52, 26
LOOKBACK = 170        # sessions shown on the chart
MIN_HISTORY = SPAN_B + SHIFT + 10
FRESH = 6             # a TK cross / cloud breakout this recent is "fresh" (the trigger)
FLAG_SCORE = 58.0


def _lines(s: pd.Series) -> dict:
    """The five Ichimoku lines from a close series (n-high/low = rolling max/min of closes)."""
    tenkan = (s.rolling(TENKAN).max() + s.rolling(TENKAN).min()) / 2
    kijun = (s.rolling(KIJUN).max() + s.rolling(KIJUN).min()) / 2
    span_a_base = (tenkan + kijun) / 2
    span_b_base = (s.rolling(SPAN_B).max() + s.rolling(SPAN_B).min()) / 2
    return {"tenkan": tenkan, "kijun": kijun,
            "span_a_base": span_a_base, "span_b_base": span_b_base,
            "span_a": span_a_base.shift(SHIFT), "span_b": span_b_base.shift(SHIFT),
            "chikou": s.shift(-SHIFT)}


def _analyse(series: pd.Series) -> dict | None:
    s = series.dropna()
    if len(s) < MIN_HISTORY:
        return None
    L = _lines(s)
    px = float(s.iloc[-1])
    tn, kj = float(L["tenkan"].iloc[-1]), float(L["kijun"].iloc[-1])
    sa, sb = float(L["span_a"].iloc[-1]), float(L["span_b"].iloc[-1])          # cloud at today
    fa, fb = float(L["span_a_base"].iloc[-1]), float(L["span_b_base"].iloc[-1])  # cloud 26 ahead
    if not all(np.isfinite(x) for x in (px, tn, kj, sa, sb, fa, fb)):
        return None
    top, bot = max(sa, sb), min(sa, sb)
    px26 = float(s.iloc[-1 - SHIFT])

    c_price = 1 if px > top else -1 if px < bot else 0
    c_tk = 1 if tn > kj else -1
    c_future = 1 if fa > fb else -1                    # green cloud ahead vs red
    c_chikou = 1 if px > px26 else -1                  # lagging span above / below past price

    # fresh Tenkan/Kijun cross, and fresh cloud breakout, in the last FRESH sessions
    tk_now, tk_prev = L["tenkan"].tail(FRESH + 1), L["kijun"].tail(FRESH + 1)
    diff = (tk_now - tk_prev)
    cross_up = bool(((diff.shift(1) <= 0) & (diff > 0)).any())
    cross_dn = bool(((diff.shift(1) >= 0) & (diff < 0)).any())
    top_s = pd.concat([L["span_a"], L["span_b"]], axis=1).max(axis=1)
    bot_s = pd.concat([L["span_a"], L["span_b"]], axis=1).min(axis=1)
    recent = s.tail(FRESH + 1)
    break_up = bool(c_price > 0 and (recent.iloc[:-1].to_numpy() <= top_s.tail(FRESH + 1).iloc[:-1].to_numpy()).any())
    break_dn = bool(c_price < 0 and (recent.iloc[:-1].to_numpy() >= bot_s.tail(FRESH + 1).iloc[:-1].to_numpy()).any())

    event = False
    if c_price == 0:                                   # inside the cloud → no trade (Ichimoku rule 1)
        label, direction, score = "Inside the cloud (no trend)", 0, 30.0
        note = f"price {px:g} between cloud {bot:g}–{top:g}"
    else:
        agree = sum(1 for c in (c_tk, c_future, c_chikou) if c == c_price)
        edge = top if c_price > 0 else bot
        clear = min(8.0, abs(px - edge) / abs(px) / 0.03 * 8.0) if px else 0.0   # 3% clear = full
        score = float(np.clip(40 + 16 * agree + clear, 0, 100))
        up = c_price > 0
        parts = []
        if (up and c_tk > 0) or (not up and c_tk < 0):
            parts.append("TK")
        if (up and c_future > 0) or (not up and c_future < 0):
            parts.append("cloud")
        if (up and c_chikou > 0) or (not up and c_chikou < 0):
            parts.append("chikou")
        note = (f"price {'above' if up else 'below'} the cloud; "
                f"{agree}/3 confirm ({'/'.join(parts) or 'none yet'})")
        # Only FRESH events are actionable signals (a persistent trend above the cloud is already
        # captured by Trend/MA); a strong-but-stale alignment is shown but not flagged.
        if (up and break_up) or (not up and break_dn):
            label = "Cloud breakout " + ("up" if up else "down")
            score, event = min(100.0, score + 10.0), True
        elif (up and cross_up) or (not up and cross_dn):
            label = "Bullish TK cross" if up else "Bearish TK cross"
            score, event = min(100.0, score + 6.0), True
        elif agree == 3:
            label = "Strong bull — above cloud" if up else "Strong bear — below cloud"
        else:
            label = "Bullish — above cloud" if up else "Bearish — below cloud"
        direction = c_price

    return {"series": s, "lines": L, "score": float(score), "direction": int(direction),
            "label": label, "note": note, "price": px, "tenkan": tn, "kijun": kj,
            "cloud_top": top, "cloud_bot": bot, "event": event,
            "metric": float(np.copysign(score, direction) if direction else 0.0)}


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)
    rows = []
    for t in list(history.columns):        # universe-agnostic (was TREND_UNIVERSE)
        if t not in history.columns:
            continue
        try:
            d = _analyse(history[t])
        except Exception:
            continue
        if d is None:
            continue
        flagged = d.get("event") and d["score"] >= FLAG_SCORE and d["direction"] != 0
        arrow = " ▲" if d["direction"] > 0 else " ▼" if d["direction"] < 0 else ""
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": (d["label"] + arrow) if flagged else "—",
            "direction": d["direction"] if flagged else 0,
            "metric": round(d["metric"], 1),       # SIGNED Ichimoku score 0–100
            "metric_label": "Ichimoku score (0-100)",
            "level": round(d["price"], 4),
            "context": d["note"] + f"; score {d['score']:.0f}/100",
        })
    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def ichimoku_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """(price_df, info). price_df ['date','price']. info carries the five lines as record
    lists (cloud incl. the 26-session forward projection) + the scalar read."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _analyse(history[ticker])
    if d is None:
        return None, None
    s = d["series"].tail(LOOKBACK)
    L = d["lines"]
    idx = s.index
    price_df = pd.DataFrame({"date": idx, "price": s.to_numpy(dtype=float)})

    def _rec(series):
        v = series.reindex(idx)
        return [{"date": t, "val": (float(x) if pd.notna(x) else None)} for t, x in v.items()]

    # the cloud, extended SHIFT sessions into the future (the leading span's projection)
    full = d["series"]
    fut_dates = pd.bdate_range(full.index[-1], periods=SHIFT + 1)[1:]
    a_base, b_base = L["span_a_base"], L["span_b_base"]
    cloud = []
    for t in idx:                                       # historical portion (already shifted)
        a, b = L["span_a"].get(t), L["span_b"].get(t)
        cloud.append({"date": t, "a": (float(a) if pd.notna(a) else None),
                      "b": (float(b) if pd.notna(b) else None)})
    tail_a, tail_b = a_base.tail(SHIFT).to_numpy(), b_base.tail(SHIFT).to_numpy()
    for t, a, b in zip(fut_dates, tail_a, tail_b):      # future projection
        cloud.append({"date": t, "a": (float(a) if np.isfinite(a) else None),
                      "b": (float(b) if np.isfinite(b) else None)})

    arrow = " ▲" if d["direction"] > 0 else " ▼" if d["direction"] < 0 else ""
    info = {
        "tenkan": _rec(L["tenkan"]), "kijun": _rec(L["kijun"]),
        "chikou": _rec(L["chikou"]), "cloud": cloud,
        "signal": d["label"] + arrow, "direction": d["direction"], "score": d["score"],
        "label": d["label"], "note": d["note"], "price": d["price"],
        "tenkan_now": d["tenkan"], "kijun_now": d["kijun"],
        "cloud_top": d["cloud_top"], "cloud_bot": d["cloud_bot"],
    }
    return price_df, info
