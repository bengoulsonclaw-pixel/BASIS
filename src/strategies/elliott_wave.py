"""Elliott Wave — rules-based impulse wave count.

Elliott counting is notoriously subjective, so this is the DETERMINISTIC version: swing
pivots from an adaptive ZigZag (reversal threshold scaled to each market's own volatility),
validated against the three hard impulse rules —

  R1  wave 2 never retraces past the start of wave 1
  R2  wave 3 pushes beyond the top of wave 1 (and is never the shortest of 1/3/5)
  R3  wave 4 never overlaps wave 1's price territory

— then the product is flagged when it sits at one of three recognizable junctures:

  * wave-2 pullback complete → **wave 3 underway** (with the impulse; the strongest setup)
  * wave-4 pullback complete → **wave 5 underway** (with the impulse, maturing)
  * five waves complete      → **corrective phase likely** (against the impulse)

The headline **wave fit** (0–100) scores how textbook the count is: wave-2 retrace near
61.8%, wave-4 near 38.2%, wave-3 extension toward 1.618×, plus an alternation bonus.
Signed by direction (+ constructive / − cautious). Close-only; fixed income runs on yields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name
from .base import frame

STRATEGY = "Elliott Wave"

# --- detection knobs (all tunable) ---
LOOKBACK = 240        # sessions the count is built from
MIN_HISTORY = 120
ZZ_VOL_MULT = 2.2     # reversal threshold = this × weekly-scale vol (√5 × daily σ%)
ZZ_MIN_PCT = 2.5      # ... floored / capped so quiet & wild markets both give sane pivots
ZZ_MAX_PCT = 12.0
MAX_PIVOT_AGE = 45    # the count's last pivot must be this recent (sessions) to matter
FLAG_FIT = 55.0       # flag when wave fit ≥ this


# ── ZigZag pivots ────────────────────────────────────────────────────────────
def _zigzag(vals: np.ndarray, thr: float):
    """Alternating swing pivots [(index, price, 'H'|'L')]; `thr` = reversal fraction."""
    piv = []
    hi_i = lo_i = 0
    trend = 0                                    # 0 unset, +1 rising leg, −1 falling leg
    ce = 0                                       # index of the current leg's extreme
    for i in range(1, len(vals)):
        v = vals[i]
        if trend == 0:
            if v > vals[hi_i]:
                hi_i = i
            if v < vals[lo_i]:
                lo_i = i
            if v <= vals[hi_i] * (1 - thr):
                piv.append((hi_i, float(vals[hi_i]), "H")); trend, ce = -1, i
            elif v >= vals[lo_i] * (1 + thr):
                piv.append((lo_i, float(vals[lo_i]), "L")); trend, ce = 1, i
        elif trend > 0:
            if v > vals[ce]:
                ce = i
            elif v <= vals[ce] * (1 - thr):
                piv.append((ce, float(vals[ce]), "H")); trend, ce = -1, i
        else:
            if v < vals[ce]:
                ce = i
            elif v >= vals[ce] * (1 + thr):
                piv.append((ce, float(vals[ce]), "L")); trend, ce = 1, i
    return piv


def _fit_band(x: float, lo: float, ideal: float, hi: float) -> float:
    """0–100: 100 at `ideal`, fading linearly to 0 at `lo`/`hi` (0 outside)."""
    if not np.isfinite(x) or x <= lo or x >= hi:
        return 0.0
    if x <= ideal:
        return 100.0 * (x - lo) / (ideal - lo)
    return 100.0 * (hi - x) / (hi - ideal)


# ── the wave count ───────────────────────────────────────────────────────────
def _count(piv: list, px: float, m: int) -> dict | None:
    """Try to read the most recent pivots as an impulse in direction `m` (+1 up / −1 down).
    Works on m×price so one set of comparisons covers both. Returns the best juncture, or
    None. Pivot labels use the classic 0..5 wave numbering (0 = start of wave 1)."""
    # In m-space every impulse looks UP; for a down-count real-space kinds flip (a price
    # low is the m-space high), so remap kinds along with the prices.
    q = [(i, m * p, k if m > 0 else ("H" if k == "L" else "L")) for i, p, k in piv]
    want_last = "L"                              # L in m-space = W2/W4 pullback low
    out = None

    def _w1_quality(w1: float, base: float) -> float:
        return float(np.clip(100.0 * (w1 / base) / 0.12, 0.0, 100.0))   # 12% leg = full marks

    # -- five waves complete: ...L0 H1 L2 H3 L4 H5, last pivot the wave-5 top --
    if len(q) >= 6 and q[-1][2] == ("H" if want_last == "L" else "L"):
        i0, p0, _ = q[-6]; i1, p1, _ = q[-5]; i2, p2, _ = q[-4]
        i3, p3, _ = q[-3]; i4, p4, _ = q[-2]; i5, p5, _ = q[-1]
        w1, w3, w5 = p1 - p0, p3 - p2, p5 - p4
        if (min(w1, w3, w5) > 0 and p2 > p0 and p3 > p1 and p4 > p1 and p5 > p3
                and w3 >= min(w1, w5)):          # R1 / R2 / R3 / W3-not-shortest
            r2, r4 = (p1 - p2) / w1, (p3 - p4) / w3
            fit = (0.35 * _fit_band(r2, 0.236, 0.618, 0.886)
                   + 0.30 * _fit_band(r4, 0.10, 0.382, 0.65)
                   + 0.35 * float(np.clip(w3 / w1 / 1.618 * 100.0, 0.0, 100.0)))
            if abs(r2 - r4) >= 0.20:
                fit = min(100.0, fit + 10.0)     # alternation bonus
            out = {"phase": "done", "fit": fit * 0.85, "idx": [i0, i1, i2, i3, i4, i5],
                   "pxs": [m * p for p in (p0, p1, p2, p3, p4, p5)],
                   "r2": r2, "r4": r4, "ext": w3 / w1}
    # -- wave 5 underway: ...L0 H1 L2 H3 L4, price advancing off the wave-4 low --
    if out is None and len(q) >= 5 and q[-1][2] == want_last:
        i0, p0, _ = q[-5]; i1, p1, _ = q[-4]; i2, p2, _ = q[-3]
        i3, p3, _ = q[-2]; i4, p4, _ = q[-1]
        w1, w3 = p1 - p0, p3 - p2
        if (min(w1, w3) > 0 and p2 > p0 and p3 > p1 and p4 > p1 and m * px > p4):
            r2, r4 = (p1 - p2) / w1, (p3 - p4) / w3
            fit = (0.40 * _fit_band(r2, 0.236, 0.618, 0.886)
                   + 0.35 * _fit_band(r4, 0.10, 0.382, 0.65)
                   + 0.25 * float(np.clip(w3 / w1 / 1.618 * 100.0, 0.0, 100.0)))
            if abs(r2 - r4) >= 0.20:
                fit = min(100.0, fit + 10.0)
            out = {"phase": "W5", "fit": fit, "idx": [i0, i1, i2, i3, i4],
                   "pxs": [m * p for p in (p0, p1, p2, p3, p4)],
                   "r2": r2, "r4": r4, "ext": w3 / w1}
    # -- wave 3 underway: ...L0 H1 L2, price advancing off the wave-2 low --
    if out is None and len(q) >= 3 and q[-1][2] == want_last:
        i0, p0, _ = q[-3]; i1, p1, _ = q[-2]; i2, p2, _ = q[-1]
        w1 = p1 - p0
        if w1 > 0 and p2 > p0 and m * px > p2:   # R1 + advancing
            r2 = (p1 - p2) / w1
            fit = (0.65 * _fit_band(r2, 0.236, 0.618, 0.886)
                   + 0.35 * _w1_quality(w1, abs(p0) if p0 else 1.0))
            out = {"phase": "W3", "fit": fit, "idx": [i0, i1, i2],
                   "pxs": [m * p for p in (p0, p1, p2)], "r2": r2, "r4": np.nan,
                   "ext": np.nan}
    return out


_PHASE_SIG = {  # (signal template, direction multiplier vs the impulse, label sequence)
    "W3":   ("Wave 3 underway", 1, ["0", "1", "2"]),
    "W5":   ("Wave 5 underway", 1, ["0", "1", "2", "3", "4"]),
    "done": ("Five waves complete", -1, ["0", "1", "2", "3", "4", "5"]),
}


def _analyse(series: pd.Series) -> dict | None:
    s = series.dropna()
    if len(s) < MIN_HISTORY:
        return None
    s = s.tail(LOOKBACK)
    vals = s.to_numpy(dtype=float)
    if np.any(vals <= 0):                        # % ZigZag needs a positive series
        return None
    px = float(vals[-1])
    sig_pct = float(pd.Series(vals).pct_change().std() * 100.0)
    if not np.isfinite(sig_pct) or sig_pct <= 0:
        return None
    thr = float(np.clip(ZZ_VOL_MULT * sig_pct * np.sqrt(5.0), ZZ_MIN_PCT, ZZ_MAX_PCT)) / 100.0
    piv = _zigzag(vals, thr)
    if len(piv) < 3:
        return None

    best = None
    for m in (+1, -1):                           # try an up-count and a down-count
        c = _count(piv, px, m)
        if c and c["fit"] >= 1.0 and (best is None or c["fit"] > best["fit"]):
            c["m"] = m
            best = c
    if best is None or (len(vals) - 1 - best["idx"][-1]) > MAX_PIVOT_AGE:
        return None

    sig_txt, dmul, labels = _PHASE_SIG[best["phase"]]
    m = best["m"]
    direction = int(m * dmul)
    arrow = " ▲" if direction > 0 else " ▼"

    # objective / invalidation for the juncture (classic projections)
    p = best["pxs"]
    if best["phase"] == "W3":
        target = p[2] + m * abs(p[1] - p[0]) * 1.618   # W3 ≈ 1.618 × W1 off the W2 low
        stop = p[2]                                     # back through the wave-2 pivot
    elif best["phase"] == "W5":
        target = p[4] + m * abs(p[1] - p[0])            # W5 ≈ W1 off the W4 low
        stop = p[4]
    else:                                               # correction toward the prior wave 4
        target = p[4]
        stop = p[5]
    risk = abs(px - stop)
    rr = abs(target - px) / risk if risk > 0 else np.nan

    return {"series": s, "pivots": piv, "count": best, "phase": best["phase"],
            "labels": labels, "direction": direction, "fit": float(best["fit"]),
            "metric": float(np.copysign(best["fit"], direction)),
            "signal": sig_txt + arrow, "price": px, "thr_pct": thr * 100.0,
            "target": float(target), "stop": float(stop), "rr": float(rr)}


def _context(d: dict) -> str:
    c = d["count"]
    bits = [f"{'up' if c['m'] > 0 else 'down'}-impulse", f"fit {d['fit']:.0f}/100"]
    if np.isfinite(c.get("r2", np.nan)):
        bits.append(f"W2 {c['r2'] * 100:.0f}% retrace")
    if np.isfinite(c.get("r4", np.nan)):
        bits.append(f"W4 {c['r4'] * 100:.0f}%")
    if np.isfinite(c.get("ext", np.nan)):
        bits.append(f"W3 {c['ext']:.1f}× W1")
    if np.isfinite(d["rr"]):
        bits.append(f"target {d['target']:g}, R:R {d['rr']:.1f}")
    return "; ".join(bits)


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in TREND_UNIVERSE:
        if t not in history.columns:
            continue
        try:
            d = _analyse(history[t])
        except Exception:
            continue
        if d is None:
            continue
        flagged = d["fit"] >= FLAG_FIT
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": d["signal"] if flagged else "—",
            "direction": d["direction"] if flagged else 0,
            "metric": round(d["metric"], 1),       # SIGNED wave fit: + constructive / − cautious
            "metric_label": "wave fit (0-100)",
            "level": round(d["price"], 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def elliott_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price over the lookback + the labelled wave count. (price_df, info) or (None, None).

    price_df: ['date','price']. info: 'pivots' (the counted pivots, [{date, price, label,
    kind}]), 'zigzag' (every ZigZag pivot [{date, price}] for context), 'phase' ('W3'|'W5'|
    'done'), 'signal', 'direction', 'fit', 'thr_pct', 'target', 'stop', 'rr', 'price'."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    d = _analyse(history[ticker])
    if d is None:
        return None, None
    s = d["series"]
    idx = d["count"]["idx"]
    m = d["count"]["m"]
    price_df = pd.DataFrame({"date": s.index, "price": s.to_numpy(dtype=float)})
    # The count alternates from its start: up-impulse 0=L,1=H,2=L…; down-impulse mirrored.
    counted = [{"date": s.index[i], "price": float(p), "label": lab,
                "kind": ("L" if j % 2 == 0 else "H") if m > 0 else ("H" if j % 2 == 0 else "L")}
               for j, (i, p, lab) in enumerate(zip(idx, d["count"]["pxs"], d["labels"]))]
    info = {
        "pivots": counted,
        "zigzag": [{"date": s.index[i], "price": float(p)} for i, p, _k in d["pivots"]],
        "phase": d["phase"], "signal": d["signal"], "direction": d["direction"],
        "fit": d["fit"], "thr_pct": d["thr_pct"],
        "target": d["target"], "stop": d["stop"], "rr": d["rr"], "price": d["price"],
    }
    return price_df, info
