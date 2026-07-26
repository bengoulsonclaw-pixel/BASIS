"""Money Flow Index — volume-weighted RSI: validation, distribution, and flow shifts.

MFI is RSI built on money flow (price × volume) instead of price alone — so it answers
"is the momentum carried by actual volume?". Close-only variant (the suite is settlement-
based): flow = close × volume, signed by the day's direction, ratioed over 14 sessions.
Four reads, in priority order:

  1. **Volume-confirmed extreme** — MFI ≥ 80 (overbought on real flow → distribution risk)
     or ≤ 20 (oversold → accumulation zone). When RSI agrees (>70 / <30) the read is
     tagged as *validating* the RSI signal; momentum extremes without flow are weaker.
  2. **RSI unconfirmed by flow** — RSI stretched but MFI nowhere near its extreme: the
     price momentum isn't volume-backed (an early warning the move may not carry).
  3. **Early distribution / accumulation** — price still trending up while MFI sags below
     50 and falls (sellers working into strength), or the mirror.
  4. **Flow shift** — MFI crossing 50 on a steep 10-day slope: momentum changing hands
     WITH volume support.

**FX futures are EXCLUDED** — most FX turnover is OTC (swaps/forwards); CME FX futures
volume is a thin venue-specific slice, so flow measures there mislead. Rates stay in
(Treasury/STIR futures are the volume centre of their markets); fixed income runs on
yields with futures volume, keeping the suite's yield convention. No volume → skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..datafeed import get_volume_history
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Money Flow Index"

WINDOW = 120
MIN_HISTORY = 90
MFI_N = 14
RSI_N = 14
OB, OS = 80.0, 20.0   # MFI extreme bands (wider than RSI's 70/30 by convention)
FLAG_SCORE = 55.0


def _universe() -> list:
    """Everything except FX futures (the real FX volume is OTC; listed futures volume is a
    biased sliver) and markets with no volume series."""
    return [t for t in TREND_UNIVERSE if asset(t) != "FX"]


def _rsi(p: pd.Series, n: int = RSI_N) -> pd.Series:
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _mfi(p: pd.Series, v: pd.Series, n: int = MFI_N) -> pd.Series:
    flow = p * v
    sign = np.sign(p.diff())
    pos = flow.where(sign > 0, 0.0).rolling(n).sum()
    neg = flow.where(sign < 0, 0.0).rolling(n).sum()
    tot = pos + neg
    return (100.0 * pos / tot.replace(0, np.nan)).astype(float)


def _analyse(px: pd.Series, vol: pd.Series) -> dict | None:
    df = pd.concat({"px": px, "vol": vol}, axis=1).dropna()
    if len(df) < MIN_HISTORY:
        return None
    df = df.tail(WINDOW)
    p, v = df["px"], df["vol"]
    if float(np.nanmedian(v)) <= 0:
        return None
    mfi = _mfi(p, v)
    rsi = _rsi(p)
    m, r = float(mfi.iloc[-1]), float(rsi.iloc[-1])
    if not (np.isfinite(m) and np.isfinite(r)):
        return None
    d10 = float(mfi.iloc[-1] - mfi.iloc[-11]) if len(mfi) > 11 else 0.0
    r20 = float(p.iloc[-1] / p.iloc[-21] - 1.0) if len(p) > 21 else 0.0

    reads = []                                       # (|score|, dir, label, note)

    # 1 — volume-confirmed extreme (± RSI validation)
    if m >= OB:
        sc = 2.0 * (m - 50.0)
        note = f"MFI {m:.0f}"
        if r >= 70:
            sc, note = min(100.0, sc + 12.0), note + f", validates RSI {r:.0f} overbought"
        reads.append((sc, -1, "Overbought on real flow", note))
    elif m <= OS:
        sc = 2.0 * (50.0 - m)
        note = f"MFI {m:.0f}"
        if r <= 30:
            sc, note = min(100.0, sc + 12.0), note + f", validates RSI {r:.0f} oversold"
        reads.append((sc, 1, "Oversold on real flow", note))

    # 2 — RSI stretched but flow missing (early warning, against the price momentum)
    if r >= 70 and m < 60:
        reads.append((58.0 + (r - 70) * 0.8, -1, "RSI overbought, flow absent",
                      f"RSI {r:.0f} vs MFI only {m:.0f} — momentum not volume-backed"))
    elif r <= 30 and m > 40:
        reads.append((58.0 + (30 - r) * 0.8, 1, "RSI oversold, flow absent",
                      f"RSI {r:.0f} vs MFI {m:.0f} — selling not volume-backed"))

    # 3 — early distribution / accumulation: price trending, flow leaking the other way
    if r20 >= 0.02 and m < 45 and d10 < -5:
        reads.append((float(np.clip(55 + (45 - m) * 1.2, 0, 100)), -1, "Early distribution",
                      f"price {r20 * 100:+.1f}%/20d but MFI {m:.0f} and falling ({d10:+.0f}/10d)"))
    elif r20 <= -0.02 and m > 55 and d10 > 5:
        reads.append((float(np.clip(55 + (m - 55) * 1.2, 0, 100)), 1, "Early accumulation",
                      f"price {r20 * 100:+.1f}%/20d but MFI {m:.0f} and rising ({d10:+.0f}/10d)"))

    # 4 — flow shift: MFI through 50 on a steep slope that AGREES with the cross direction
    m3 = float(mfi.iloc[-4]) if len(mfi) > 4 else m
    if abs(d10) >= 15 and ((m3 < 50 <= m and d10 > 0) or (m3 > 50 >= m and d10 < 0)):
        up = m >= 50
        reads.append((float(np.clip(45 + abs(d10), 0, 100)), 1 if up else -1,
                      "Flow shift " + ("up" if up else "down"),
                      f"MFI through 50 ({m3:.0f}→{m:.0f}), {d10:+.0f} over 10d"))

    if not reads:
        return None
    sc, ddir, lbl, note = max(reads, key=lambda x: x[0])
    return {"df": df, "mfi": mfi, "rsi": rsi, "score": float(sc), "direction": int(ddir),
            "label": lbl, "note": note, "price": float(p.iloc[-1]),
            "mfi_now": m, "rsi_now": r, "metric": float(np.copysign(sc, ddir))}


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    uni = _universe()
    if history is None:
        history = get_history(uni)
    vols = get_volume_history(uni)

    rows = []
    for t in uni:
        if t not in history.columns or t not in getattr(vols, "columns", []):
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
            "metric": round(d["metric"], 1),       # SIGNED MFI score 0–100
            "metric_label": "MFI score (0-100)",
            "level": round(d["price"], 4),
            "context": d["note"] + f"; score {d['score']:.0f}/100",
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def mfi_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """(price_df, info): price_df ['date','price','mfi','rsi']; info {'signal','direction',
    'score','label','note','mfi','rsi','price'} — or (None, None)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    vols = get_volume_history([ticker])
    if ticker not in getattr(vols, "columns", []):
        return None, None
    d = _analyse(history[ticker], vols[ticker])
    if d is None:
        return None, None
    df = d["df"]
    price_df = pd.DataFrame({"date": df.index, "price": df["px"].to_numpy(dtype=float),
                             "mfi": d["mfi"].to_numpy(dtype=float),
                             "rsi": d["rsi"].to_numpy(dtype=float)})
    arrow = " ▲" if d["direction"] > 0 else " ▼"
    info = {"signal": d["label"] + arrow, "direction": d["direction"], "score": d["score"],
            "label": d["label"], "note": d["note"], "mfi": d["mfi_now"], "rsi": d["rsi_now"],
            "price": d["price"]}
    return price_df, info
