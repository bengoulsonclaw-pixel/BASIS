"""On-Balance Volume — volume confirmation, hidden accumulation, and trend-failure warnings.

OBV (Granville) accumulates volume signed by the day's direction: up day + volume, down
day − volume. Price moves the tape, but OBV shows whether volume is BEHIND the move.
Three reads, in priority order:

  1. **OBV divergence** — price makes a higher high while OBV makes a lower high (bearish;
     the rally is running on fumes) or price a lower low while OBV a higher low (bullish) —
     the classic trend-failure warning.
  2. **Breakout (un)confirmation** — price at/through a fresh ~55-day extreme: OBV at its
     own extreme too → the breakout is volume-backed; OBV well short → suspect.
  3. **Hidden accumulation / distribution** — price range-bound while OBV climbs (someone
     is quietly buying) or slides (quietly selling).

**FX futures are EXCLUDED**: the vast majority of FX turnover trades OTC (swaps/forwards),
so CME FX futures volume is a small venue-specific slice — OBV there reads roll cycles and
listed-market quirks, not the market's real flow. Rates stay in: Treasury/STIR futures ARE
the volume centre of their markets. Fixed income runs on yields (suite convention), with
the futures volume signed by the YIELD direction — constructive = yields up on volume.
Close-only; volume = FUT_AGGTE_VOL (all listed contracts); no volume series → skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..datafeed import get_volume_history
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "On-Balance Volume"

WINDOW = 120          # sessions the reads are taken from
MIN_HISTORY = 90
BREAK_LOOKBACK = 55   # the "fresh extreme" window for breakout confirmation
PIVOT_W = 4           # swing pivots for divergence = strict extreme within ±this
FLAG_SCORE = 55.0     # flag when |score| ≥ this


def _universe() -> list:
    """The volume universe: everything except FX futures (OTC swaps carry the real FX
    volume — listed futures volume is a biased sliver) and markets with no volume series."""
    return [t for t in TREND_UNIVERSE if asset(t) != "FX"]


def _pivots(v: np.ndarray, w: int, high: bool) -> list:
    """Indices of swing highs (or lows) — strict extremes within ±w bars."""
    out = []
    for i in range(w, len(v) - w):
        seg = v[i - w:i + w + 1]
        if (high and v[i] == seg.max()) or (not high and v[i] == seg.min()):
            out.append(i)
    return out


def _analyse(px: pd.Series, vol: pd.Series) -> dict | None:
    df = pd.concat({"px": px, "vol": vol}, axis=1).dropna()
    if len(df) < MIN_HISTORY:
        return None
    df = df.tail(WINDOW)
    p = df["px"].to_numpy(dtype=float)
    v = df["vol"].to_numpy(dtype=float)
    if np.nanmedian(v) <= 0:
        return None
    obv = np.cumsum(np.sign(np.diff(p, prepend=p[0])) * v)
    price_now = float(p[-1])

    reads = []                                       # (|score|, dir, label, note)

    # 1 — divergence off the last two swing pairs (price vs OBV at the same pivots)
    for high, ddir, lbl in ((True, -1, "Bearish OBV divergence"),
                            (False, 1, "Bullish OBV divergence")):
        piv = _pivots(p, PIVOT_W, high)
        if len(piv) >= 2:
            i1, i2 = piv[-2], piv[-1]
            newer = (p[i2] > p[i1]) if high else (p[i2] < p[i1])
            obv_disagrees = (obv[i2] < obv[i1]) if high else (obv[i2] > obv[i1])
            recent = (len(p) - 1 - i2) <= 15
            if newer and obv_disagrees and recent:
                rng = obv[-BREAK_LOOKBACK:].max() - obv[-BREAK_LOOKBACK:].min()
                gap = abs(obv[i2] - obv[i1]) / rng * 100.0 if rng > 0 else 0.0
                sc = float(np.clip(70.0 + gap * 0.6, 0.0, 100.0))
                reads.append((sc, ddir, lbl,
                              f"price {'HH' if high else 'LL'} vs OBV {'LH' if high else 'HL'}"))

    # 2 — breakout confirmation at a fresh ~55d extreme
    lb = min(BREAK_LOOKBACK, len(p))
    p_win, o_win = p[-lb:], obv[-lb:]
    p_rng, o_rng = p_win.max() - p_win.min(), o_win.max() - o_win.min()
    if p_rng > 0 and o_rng > 0:
        p_pos = (price_now - p_win.min()) / p_rng          # 0..1 where price sits in range
        o_pos = (obv[-1] - o_win.min()) / o_rng
        broke_up = p_pos >= 0.99 or p_win.argmax() >= lb - 5
        broke_dn = p_pos <= 0.01 or p_win.argmin() >= lb - 5
        if broke_up:                                  # capped below divergence: confirmation is
            if o_pos >= 0.90:                         # the routine read, divergence the headline
                reads.append((50.0 + 35.0 * o_pos, 1, "Breakout confirmed by volume",
                              f"price & OBV both at {lb}d highs"))
            elif o_pos <= 0.70:
                reads.append((60.0, -1, "Breakout lacks volume",
                              f"price at {lb}d high, OBV only {o_pos * 100:.0f}% of its range"))
        elif broke_dn:
            if o_pos <= 0.10:
                reads.append((50.0 + 35.0 * (1 - o_pos), -1, "Breakdown confirmed by volume",
                              f"price & OBV both at {lb}d lows"))
            elif o_pos >= 0.30:
                reads.append((60.0, 1, "Breakdown lacks volume",
                              f"price at {lb}d low, OBV holding {o_pos * 100:.0f}% up its range"))

    # 3 — hidden accumulation / distribution: price sideways, OBV trending
    r20 = price_now / p[-21] - 1.0 if len(p) > 21 else 0.0
    sig20 = float(pd.Series(p).pct_change().std() * np.sqrt(20))
    if sig20 > 0 and abs(r20) < 0.75 * sig20:            # price going nowhere
        d_obv = pd.Series(obv).diff(20).dropna()
        sd = float(d_obv.std())
        if sd > 0:
            z = float((obv[-1] - obv[-21]) / sd) if len(obv) > 21 else 0.0
            if z >= 1.2:
                reads.append((float(np.clip(45 + z * 18, 0, 100)), 1, "Hidden accumulation",
                              f"price flat ({r20 * 100:+.1f}%/20d), OBV climbing (z {z:.1f})"))
            elif z <= -1.2:
                reads.append((float(np.clip(45 - z * 18, 0, 100)), -1, "Hidden distribution",
                              f"price flat ({r20 * 100:+.1f}%/20d), OBV sliding (z {z:.1f})"))

    if not reads:
        return None
    sc, ddir, lbl, note = max(reads, key=lambda r: r[0])
    return {"df": df, "obv": obv, "score": float(sc), "direction": int(ddir),
            "label": lbl, "note": note, "price": price_now,
            "metric": float(np.copysign(sc, ddir))}


def find_opportunities(history: pd.DataFrame | None = None,
                       volume: pd.DataFrame | None = None) -> pd.DataFrame:
    uni = _universe()
    if history is None:
        history = get_history(uni)
    vols = volume if volume is not None else get_volume_history(uni)   # equities pass their own

    rows = []
    for t in list(history.columns):        # universe-agnostic (was `uni`)
        if t not in getattr(vols, "columns", []):
            continue
        try:
            d = _analyse(history[t], vols[t])
        except Exception:
            continue
        if d is None:
            continue
        flagged = d["score"] >= FLAG_SCORE
        arrow = " ▲" if d["direction"] > 0 else " ▼"
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": (d["label"] + arrow) if flagged else "—",
            "direction": d["direction"] if flagged else 0,
            "metric": round(d["metric"], 1),       # SIGNED OBV score 0–100
            "metric_label": "OBV score (0-100)",
            "level": round(d["price"], 4),
            "context": d["note"] + f"; score {d['score']:.0f}/100",
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def obv_chart_data(ticker: str, history: pd.DataFrame | None = None,
                   volume: pd.DataFrame | None = None):
    """(price_df, info): price_df ['date','price','obv']; info {'signal','direction',
    'score','label','note','price'} — or (None, None)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    vols = volume if volume is not None else get_volume_history([ticker])   # equities pass their own
    if ticker not in getattr(vols, "columns", []):
        return None, None
    d = _analyse(history[ticker], vols[ticker])
    if d is None:
        return None, None
    df = d["df"]
    price_df = pd.DataFrame({"date": df.index, "price": df["px"].to_numpy(dtype=float),
                             "obv": d["obv"]})
    arrow = " ▲" if d["direction"] > 0 else " ▼"
    info = {"signal": d["label"] + arrow, "direction": d["direction"], "score": d["score"],
            "label": d["label"], "note": d["note"], "price": d["price"]}
    return price_df, info
