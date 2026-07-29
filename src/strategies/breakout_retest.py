"""Breakout-&-retest monitor — spots products retesting a level they just broke.

One of the cleanest entries on the desk: a horizontal level — a prior swing high
(resistance) or swing low (support) — gives way on a **decisive close** through it,
ideally on a **volume surge**; price then **pulls back to retest** the broken level,
which has *flipped role* (broken resistance becomes new support; broken support
becomes new resistance). The trade is taken on the retest, in the direction of the
break, with a stop just back through the level:

    Upside break : close above a swing high  -> pull back DOWN to it -> "Long — retest"
    Downside break: close below a swing low   -> pull back UP to it   -> "Short — retest"

This page answers "what just broke and is now being retested?" It scans the book,
finds on each product's close history the most recent decisively-broken horizontal
level, and — when price has pulled back to within RETEST_PCT of that level — flags
the active retest. Proximity to the level is scored 0–100 (**retest proximity**, 100
= price sitting exactly on the flipped level) and signed by direction — **+ for a
long (upside) retest, − for a short (downside) retest** — so it drops straight into
the dashboard's symmetric ±trigger machinery (see src/specs.py): proximity ≥ +t flags
a long retest, ≤ −t a short one.

**Volume confirmation.** A genuine break trades heavy: the textbook setup surges on
the breakout bar. With a volume series available (`datafeed.get_volume_history`) the
breakout bar's volume is compared to its trailing average and the level only counts
as broken when that ratio clears VOL_MULT (a **vol_confirm** flag rides along). Volume
is a *gate on the break here* (an unconfirmed thrust is more likely a fakeout), but a
missing series is not held against the setup: no volume data just leaves vol_confirm
None and the price geometry stands on its own.

This is a CONDITIONAL pattern, like the flag page: only products with a qualifying
recent break being actively retested produce a row, so the table is often short.

Close-based: the swing levels, the break and the retest are all built from settlement
closes (the same get_history seam in mock and live), and ATR is a close-to-close proxy
so it needs no high/low feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history, get_volume_history   # FI → yields
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Breakout & Retest"

# --- detection knobs (trading days, all tunable) ------------------------------
LOOKBACK = 150        # closes scanned per product (the recent regime)
MIN_HISTORY = 60      # need enough bars to find a swing, a break and a retest
PIVOT_W = 3           # a swing high/low is an extreme vs ±this many neighbours (fractal)
ATR_WINDOW = 14       # window for the close-to-close ATR proxy (true-range needs no H/L)
BREAK_MAX = 20        # the break must have happened within this many days (still "recent")
BREAK_BUFFER = 0.30   # decisive close = beyond the level by > this × ATR (not a graze)
VOL_WINDOW = 20       # trailing window for the breakout-bar volume comparison
VOL_MULT = 1.30       # breakout bar "confirms" when its volume ≥ this × the trailing avg
RETEST_PCT = 1.5      # active retest = price now within this % of the broken level

# The page's default proximity trigger (mirrors specs.py["Breakout & Retest"]["default"]).
DEFAULT_TRIGGER = 60.0


def _atr_proxy(c: np.ndarray, window: int = ATR_WINDOW) -> float:
    """Close-to-close ATR stand-in: mean absolute daily change over the last `window`
    bars. We have settlement closes only (no high/low), so |Δclose| is the available
    true-range proxy — enough to scale the "decisive" break buffer to each product."""
    if len(c) <= window:
        return float(np.mean(np.abs(np.diff(c)))) if len(c) > 1 else 0.0
    return float(np.mean(np.abs(np.diff(c[-window - 1:]))))


def _pivots(c: np.ndarray, w: int, kind: int) -> list[int]:
    """Indices of confirmed swing pivots on close array `c`. `kind` +1 hunts swing HIGHs
    (a close strictly ≥ its `w` neighbours each side), −1 swing LOWs. The last `w` bars
    can't be confirmed pivots (no right-hand shoulder yet), so they're excluded — which
    is what we want: the level must pre-date the break."""
    out = []
    n = len(c)
    for i in range(w, n - w):
        seg = c[i - w:i + w + 1]
        if kind > 0 and c[i] >= seg.max() - 1e-9:
            out.append(i)
        elif kind < 0 and c[i] <= seg.min() + 1e-9:
            out.append(i)
    return out


def _vol_confirm(va: np.ndarray | None, brk_idx: int) -> bool | None:
    """Did the breakout bar trade heavy? Ratio of the breakout-bar volume to the trailing
    VOL_WINDOW average vs VOL_MULT. Returns None when no usable volume is available (no
    series, or the trailing window is all-NaN) — the caller treats None as "no gate"."""
    if va is None:
        return None
    last = va[brk_idx]
    base = va[max(0, brk_idx - VOL_WINDOW):brk_idx]
    base = base[np.isfinite(base)]
    if not np.isfinite(last) or last <= 0 or base.size == 0:
        return None
    avg = float(np.mean(base))
    if avg <= 0:
        return None
    return bool(last >= VOL_MULT * avg)


def _best_break(c: np.ndarray, va: np.ndarray | None, sign: int, atr: float) -> dict | None:
    """Most recent decisively-broken level of one polarity on close array `c`, or None.

    `sign` +1 hunts an UPSIDE break of a prior swing HIGH (resistance → support); −1 a
    DOWNSIDE break of a prior swing LOW (support → resistance). A level qualifies when a
    close in the last BREAK_MAX days cleared it by > BREAK_BUFFER·ATR and (if volume is
    available) the breakout bar cleared the VOL_MULT volume gate. We keep the MOST RECENT
    such break and pick, among levels broken that day, the one nearest the breakout price
    (the tightest, freshest flip). Returns a geometry dict or None."""
    n = len(c)
    if atr <= 0:
        return None
    buffer = BREAK_BUFFER * atr
    pivots = _pivots(c, PIVOT_W, sign)
    if not pivots:
        return None

    # Walk back from today over the recent window; the first day whose close clears an
    # earlier pivot level by the buffer (with volume, if any) is the break we report.
    first = max(PIVOT_W + 1, n - BREAK_MAX)
    for j in range(n - 1, first - 1, -1):
        # Earlier pivots only (the level must pre-date the break by at least one bar).
        levels = [c[p] for p in pivots if p < j]
        if not levels:
            continue
        if sign > 0:
            cleared = [lv for lv in levels if c[j] > lv + buffer]
            level = max(cleared) if cleared else None       # nearest broken resistance from below
        else:
            cleared = [lv for lv in levels if c[j] < lv - buffer]
            level = min(cleared) if cleared else None        # nearest broken support from above
        if level is None:
            continue
        confirm = _vol_confirm(va, j)
        if confirm is False:                                 # volume present but failed the gate
            continue                                         # not a clean break — keep looking back
        return {"sign": sign, "level": float(level), "brk_idx": int(j),
                "vol_confirm": confirm, "atr": float(atr)}
    return None


def _retest(c: np.ndarray, d: dict) -> dict:
    """Score the retest of the broken level in geometry `d` against the latest close.

    Distance from price to the level as a signed %, and a 0..100 proximity (100 = price
    exactly on the level, 0 = RETEST_PCT away or further). `active` is True once price has
    pulled back to within RETEST_PCT *and* sits on the correct side for a role-flip retest
    (an upside break is retested from ABOVE → support; a downside break from BELOW →
    resistance). If price is still riding past the level (hasn't pulled back yet) it's not
    an active retest — it's a watch."""
    price = float(c[-1])
    level = d["level"]
    sign = d["sign"]
    dist_pct = (level - price) / price * 100.0               # signed: where the level is vs price
    proximity = float(np.clip(100.0 * (1.0 - abs(dist_pct) / RETEST_PCT), 0.0, 100.0))
    # Role-flip side: upside break retested from above (price ≥ level), downside from below.
    on_side = (price >= level) if sign > 0 else (price <= level)
    active = bool(abs(dist_pct) <= RETEST_PCT and on_side)
    return {"price": price, "dist_pct": float(dist_pct), "proximity": proximity,
            "active": active, "on_side": bool(on_side)}


def _detect(px: pd.Series, vol: pd.Series | None = None) -> dict | None:
    """The most relevant recent break-and-retest (up or down) on a product's close series,
    or None. `vol` (optional) is the product's daily volume, aligned to the close index and
    cleaned (≤0 → NaN, a roll/lag artifact). Of the two polarities we keep whichever break
    is the *more recent*, tie-broken by tighter retest proximity."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    s = s.iloc[-LOOKBACK:]
    c = s.to_numpy(dtype=float)
    va = None
    if vol is not None:
        va = vol.reindex(s.index).to_numpy(dtype=float).copy()
        va[~(va > 0)] = np.nan       # treat 0/blank contract volume as missing (roll/lag artifacts)
    atr = _atr_proxy(c)

    cands = []
    for sign in (+1, -1):
        d = _best_break(c, va, sign, atr)
        if d is None:
            continue
        d.update(_retest(c, d))
        cands.append(d)
    if not cands:
        return None
    # Prefer the fresher break; if both broke the same day, the tighter retest.
    return max(cands, key=lambda d: (d["brk_idx"], d["proximity"]))


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a proximity trigger — matches specs.reflag_rows so the table
    re-flags identically when the user moves the slider. A long retest (metric ≥ +t) is the
    pullback to a broken-resistance-turned-support; a short retest (≤ −t) the mirror."""
    if metric >= trigger:
        return "Long — retest", 1
    if metric <= -trigger:
        return "Short — retest", -1
    return "—", 0


def _context(d: dict) -> str:
    """Human note for a detected setup `d`: what broke, how long ago, on what volume, and
    where price sits in the retest (which side, how close)."""
    age = d["age"]
    side = "from above" if d["sign"] > 0 else "from below"
    if d["vol_confirm"] is True:
        vol = f" on {d['vol_ratio']:+.0%} vol" if np.isfinite(d.get("vol_ratio", np.nan)) else " on heavy vol"
    elif d["vol_confirm"] is False:
        vol = " (vol unconfirmed)"
    else:
        vol = ""
    where = (f"retesting {side} ({abs(d['dist_pct']):.1f}%)" if d["active"]
             else f"not yet retesting ({abs(d['dist_pct']):.1f}% {side})")
    return f"Broke {d['level']:g} {age}d ago{vol}; {where}"


def _vol_ratio(va: np.ndarray | None, brk_idx: int) -> float:
    """Breakout-bar volume as a multiple of its trailing average minus 1 (i.e. the % surge),
    for the context note; NaN when volume isn't usable."""
    if va is None:
        return float("nan")
    last = va[brk_idx]
    base = va[max(0, brk_idx - VOL_WINDOW):brk_idx]
    base = base[np.isfinite(base)]
    if not np.isfinite(last) or last <= 0 or base.size == 0:
        return float("nan")
    avg = float(np.mean(base))
    return last / avg - 1.0 if avg > 0 else float("nan")


def find_opportunities(history: pd.DataFrame | None = None,
                       volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """Currently-active break-and-retests across the book, one row per product, ranked by
    retest proximity (tightest first). Conditional: only products with a qualifying recent
    break being retested appear — the frame is often short and may be empty."""
    if history is None:
        history = get_history(TREND_UNIVERSE)
    if volume is None:                                # equities pass their own volume frame
        try:
            volume = get_volume_history(TREND_UNIVERSE)
        except Exception:
            volume = pd.DataFrame()

    rows = []
    for t in list(history.columns):        # universe-agnostic (was TREND_UNIVERSE)
        vser = volume[t] if (volume is not None and not volume.empty and t in volume.columns) else None
        d = _detect(history[t], vser)
        if d is None or not d["active"]:        # only the live retests make the table
            continue
        s = history[t].dropna().iloc[-LOOKBACK:]
        va = (vser.reindex(s.index).to_numpy(dtype=float) if vser is not None else None)
        if va is not None:
            va = va.copy()
            va[~(va > 0)] = np.nan
        d["age"] = int(len(s) - 1 - d["brk_idx"])
        d["vol_ratio"] = _vol_ratio(va, d["brk_idx"])
        metric = d["sign"] * d["proximity"]                  # signed: + long retest / − short
        signal, direction = _signal(metric, DEFAULT_TRIGGER)
        rows.append({
            "strategy": STRATEGY,
            "market": name(t),
            "instruments": t,
            "signal": signal,
            "direction": direction,
            "metric": round(metric, 1),
            "metric_label": "retest proximity",
            "level": round(float(s.iloc[-1]), 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def retest_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price over the recent window plus an info dict for the dashboard chart: the broken
    level, the breakout day, and the retest read-out. (DataFrame, info) or (None, None) if
    there's no qualifying break to draw.

    price_df: columns ['date','price'] over the last ~LOOKBACK bars.
    info keys: 'level' (broken level price), 'broke_date' (breakout-day timestamp),
    'kind' ('up'|'down'), 'vol_confirm' (bool|None), 'signal', 'direction',
    'dist_pct' (signed % price→level), 'proximity' (0..100), 'price' (last close)."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    try:
        vser = get_volume_history([ticker])
        vser = vser[ticker] if ticker in vser.columns else None
    except Exception:
        vser = None
    d = _detect(history[ticker], vser)
    if d is None:
        return None, None

    s = history[ticker].dropna().iloc[-LOOKBACK:]
    price_df = pd.DataFrame({"date": s.index, "price": s.to_numpy(dtype=float)})
    metric = d["sign"] * d["proximity"]
    signal, direction = _signal(metric, DEFAULT_TRIGGER)
    info = {
        "level": d["level"],
        "broke_date": s.index[d["brk_idx"]],
        "kind": "up" if d["sign"] > 0 else "down",
        "vol_confirm": d["vol_confirm"],
        "signal": signal,
        "direction": direction,
        "dist_pct": d["dist_pct"],                            # signed % from price to the level
        "proximity": d["proximity"],
        "price": d["price"],
    }
    return price_df, info
