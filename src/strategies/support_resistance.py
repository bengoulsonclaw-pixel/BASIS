"""Support & Resistance monitor — price sitting on a tested horizontal level.

Pure price action, no oscillators: map the horizontal zones where price has
repeatedly *stopped falling* (**support**) or *stopped rising* (**resistance**),
then flag the products whose latest close is parked right on a strong one — the
classic "buy near support / sell near resistance" trade:

    near strong SUPPORT    -> Long  (buy the dip)    dir +1
    near strong RESISTANCE -> Short (sell the rally)  dir -1

Levels are built from **swing pivots** on the close series (a pivot high/low is a
bar that is the max/min within ±W bars), then pivots at a similar price are
**clustered** into a level whose strength = the number of touches.

**Role reversal.** A level's role is read from where it sits relative to the LATEST
close — *below price = support, above price = resistance* — NOT from the pivots that
formed it. Once price breaks through a ceiling, that old resistance becomes support
(and vice-versa), which is the classic broken-and-retested zone. Each level keeps its
pivot-derived role as `origin` and is marked `flipped` when the two disagree. (Without
this a months-old, long-broken ceiling would still be drawn as resistance below the
market, and — worse — would be skipped entirely by the actionable-level search.)

The headline
number is a 0–100 **level proximity** (100 = price exactly on the level, fading to
0 once it sits more than NEAR% away). It is signed by side — **+ for support
(long), − for resistance (short)** — so it drops straight into the dashboard's
symmetric ±trigger machinery (see src/specs.py): proximity ≥ +t flags a buy-dip
setup, ≤ −t a sell-rally one.

Close-only: pivots, levels and the distance read are all built from settlement
closes (the same get_history seam in mock and live). There is no intraday high/low
or volume, so "ATR" is approximated by the rolling mean absolute daily close
change — used only to size the clustering tolerance. Deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history_ta as get_history   # fixed income → yields (see universe)
from ..universe import TREND_UNIVERSE, name, asset
from .base import frame

STRATEGY = "Support & Resistance"

# --- detection knobs (trading days / fractions, all tunable) ---
LOOKBACK = 252        # closes scanned for pivots (~1Y of sessions)
MIN_HISTORY = 60      # need a decent run before any level is meaningful
PIVOT_W = 4           # swing window: a pivot is the max/min within ±this many bars
ATR_WINDOW = 14       # bars for the close-only ATR proxy (mean |daily change|)
TOL_ATR = 1.0         # cluster tolerance = max(this × ATRproxy, …)
TOL_PCT = 0.015       #                     …, this fraction of price)   → ~1.5%
MIN_TOUCHES = 2       # a level needs at least this many pivots (more = stronger)
NEAR_PCT = 0.02       # "actionable" band around the last close → ~2%

# The page's default proximity trigger (mirrors specs.py["Support & Resistance"]["default"]).
DEFAULT_TRIGGER = 50.0


def _atr_proxy(c: np.ndarray, window: int = ATR_WINDOW) -> float:
    """Close-only ATR stand-in: the mean absolute day-to-day close change over the
    last `window` bars (no intraday range available). Used only to size the level
    clustering tolerance, so a rough scale is fine."""
    if len(c) < 2:
        return 0.0
    d = np.abs(np.diff(c))
    return float(np.mean(d[-window:])) if len(d) else 0.0


def _pivots(c: np.ndarray, w: int = PIVOT_W) -> tuple[list[int], list[int]]:
    """Swing pivot indices on close array `c`. A pivot HIGH is a bar whose close is
    the STRICT max within ±w bars (it must turn back down on both sides); a pivot LOW
    the strict min. The strictness matters: a bar inside a long flat run is not a
    swing — requiring a genuine turning point keeps plateaus (and a degenerate series
    pinned at a price floor) from minting hundreds of phantom touches. The first/last
    w bars can't be confirmed (no full window), so they're skipped. Returns
    (highs, lows)."""
    highs, lows = [], []
    n = len(c)
    for i in range(w, n - w):
        left, right = c[i - w:i], c[i + 1:i + w + 1]
        if c[i] > left.max() and c[i] > right.max():
            highs.append(i)
        elif c[i] < left.min() and c[i] < right.min():
            lows.append(i)
    return highs, lows


def _cluster(pivots: list[tuple[float, int]], tol: float) -> list[dict]:
    """Group pivots whose PRICES are within `tol` of each other into levels.

    `pivots` is a list of (price, kind) where kind is +1 for a pivot high, −1 for a
    pivot low. Walking the price-sorted pivots, a new member joins the running
    cluster while it stays within `tol` of the cluster's mean; otherwise the cluster
    is closed and a fresh one starts. Each level carries its mean price, touch count and
    its `origin` — the pivot-derived role ('resistance' if built mostly from highs, else
    'support'). The EFFECTIVE role is set later, against the latest close (see _levels)."""
    if not pivots:
        return []
    pivots = sorted(pivots, key=lambda p: p[0])
    levels = []
    grp = [pivots[0]]
    for price, kind in pivots[1:]:
        mean = np.mean([p[0] for p in grp])
        if abs(price - mean) <= tol:
            grp.append((price, kind))
        else:
            levels.append(_level_from_group(grp))
            grp = [(price, kind)]
    levels.append(_level_from_group(grp))
    return levels


def _level_from_group(grp: list[tuple[float, int]]) -> dict:
    """Collapse a cluster of pivots into one level dict (price = member mean, touches =
    member count, origin = the majority pivot type; ties → resistance). `origin` is the
    HISTORICAL role — the live role is assigned against the latest close in _levels."""
    prices = [p[0] for p in grp]
    highs = sum(1 for _, k in grp if k > 0)
    origin = "resistance" if highs >= len(grp) - highs else "support"
    return {"price": float(np.mean(prices)), "touches": len(grp), "origin": origin}


def _levels(c: np.ndarray) -> list[dict]:
    """Strong horizontal levels (touches ≥ MIN_TOUCHES) on close array `c`, sorted by
    price. Clustering tolerance is max(TOL_ATR·ATRproxy, TOL_PCT·price).

    ROLE REVERSAL is applied here: `kind` comes from the level's position against the
    LATEST close (below → support, above → resistance), `origin` keeps the pivot-derived
    role, and `flipped` marks a level whose role has reversed (a broken-and-retested
    zone — usually the better one to trade)."""
    highs, lows = _pivots(c)
    pivots = [(c[i], 1) for i in highs] + [(c[i], -1) for i in lows]
    if not pivots:
        return []
    price = float(c[-1])
    tol = max(TOL_ATR * _atr_proxy(c), TOL_PCT * price)
    levels = [lv for lv in _cluster(pivots, tol) if lv["touches"] >= MIN_TOUCHES]
    for lv in levels:
        lv["kind"] = "support" if lv["price"] <= price else "resistance"
        lv["flipped"] = lv["kind"] != lv["origin"]
    return sorted(levels, key=lambda lv: lv["price"])


def _nearest_actionable(levels: list[dict], price: float) -> dict | None:
    """The strong level the last close is sitting on, or None.

    A SUPPORT counts only when it is within NEAR_PCT at/below price (price hasn't
    broken through it yet — still a floor to buy); a RESISTANCE only within NEAR_PCT
    at/above price (still a ceiling to sell). The closest qualifying level wins; ties
    break toward more touches (the stronger zone)."""
    band = NEAR_PCT * price
    cands = []
    for lv in levels:
        d = price - lv["price"]                                 # >0: level below price
        if lv["kind"] == "support" and 0 <= d <= band:
            cands.append((abs(d), lv))
        elif lv["kind"] == "resistance" and 0 <= -d <= band:
            cands.append((abs(d), lv))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], -x[1]["touches"]))
    return cands[0][1]


def _proximity(price: float, level: float) -> float:
    """Unsigned 0..100 closeness of `price` to `level`: 100 = exactly on it, fading
    linearly to 0 at the edge of the NEAR_PCT band (clipped ≥ 0)."""
    band = NEAR_PCT * price
    if band <= 0:
        return 0.0
    return float(np.clip(100.0 * (1.0 - abs(price - level) / band), 0.0, 100.0))


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a proximity trigger — matches specs.reflag_rows so the
    table re-flags identically when the user moves the slider. Signed metric: + sits
    on support (buy the dip), − on resistance (sell the rally)."""
    if metric >= trigger:
        return "Long (buy the dip)", 1
    if metric <= -trigger:
        return "Short (sell the rally)", -1
    return "—", 0


def _analyse(px: pd.Series) -> dict | None:
    """Levels + the nearest actionable one for a product's close series, or None if
    there isn't enough history. Returns the geometry the table and the chart share."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    s = s.tail(LOOKBACK)
    c = s.to_numpy(dtype=float)
    price = float(c[-1])
    levels = _levels(c)

    lv = _nearest_actionable(levels, price)
    if lv is None:
        nearest, dist_pct, proximity, metric = np.nan, np.nan, 0.0, 0.0
    else:
        nearest = lv["price"]
        dist_pct = (price - nearest) / price * 100.0          # signed: >0 level below price
        sign = 1 if lv["kind"] == "support" else -1           # support → long(+), resistance → short(−)
        metric = sign * _proximity(price, nearest)
        proximity = abs(metric)
    signal, direction = _signal(metric, DEFAULT_TRIGGER)
    return {"series": s, "levels": levels, "price": price, "nearest": nearest,
            "kind": (lv["kind"] if lv else None), "touches": (lv["touches"] if lv else 0),
            "flipped": bool(lv.get("flipped")) if lv else False,
            "dist_pct": float(dist_pct), "proximity": float(proximity),
            "metric": float(metric), "signal": signal, "direction": direction}


def _context(d: dict) -> str:
    """Human note for an analysed product `d`."""
    if d["nearest"] is None or not np.isfinite(d["nearest"]):
        n = len(d["levels"])
        return f"no level within {NEAR_PCT * 100:.0f}% ({n} level{'s' if n != 1 else ''} mapped)"
    where = "below" if d["dist_pct"] >= 0 else "above"
    zone = "buy-dip zone" if d["kind"] == "support" else "sell-rally zone"
    # a role-reversed level (broken then retested from the other side) is the stronger read
    rev = (f", role-reversed from {'resistance' if d['kind'] == 'support' else 'support'}"
           if d.get("flipped") else "")
    return (f"{d['kind'].capitalize()} {d['nearest']:g} ({d['touches']} touches{rev}) "
            f"{abs(d['dist_pct']):.1f}% {where} — {zone}")


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = get_history(TREND_UNIVERSE)

    rows = []
    for t in list(history.columns):        # universe-agnostic (was TREND_UNIVERSE)
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
            "metric": round(d["metric"], 1),       # SIGNED: + support/long, − resistance/short
            "metric_label": "level proximity",
            "level": round(d["price"], 4),
            "context": _context(d),
        })

    df = frame(rows)
    if df.empty:
        return df
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def sr_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price over the lookback window + the mapped support/resistance levels, plus an
    info dict for the dashboard. (price_df, info) or (None, None) if there isn't
    enough history. `price_df` has columns ['date','price']; `info` carries the
    levels and the nearest-level read."""
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
        "levels": [{"price": lv["price"], "kind": lv["kind"], "touches": lv["touches"],
                    "origin": lv.get("origin"), "flipped": bool(lv.get("flipped"))}
                   for lv in d["levels"]],
        "signal": d["signal"],
        "direction": d["direction"],
        "nearest": d["nearest"],
        "flipped": d["flipped"],
        "dist_pct": d["dist_pct"],          # signed % from price to the nearest level
        "proximity": d["proximity"],        # unsigned 0..100
        "price": d["price"],
    }
    return price_df, info
