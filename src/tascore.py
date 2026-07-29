"""Cross-strategy conviction scoring for the Technical Analysis overview + report.

Each technical strategy reports its setup on its OWN metric scale (a z-score, a 0–100
readiness/proximity, a momentum score, a 3-month return %, an MA gap %, …), so the raw
metrics aren't comparable across strategies. This module:

  * maps every flagged signal onto a common 0–100 **strength**,
  * aggregates a product's signals into one signed **conviction score** = how many
    strategies agree × how strong they are, netting longs against shorts so conflicting
    calls cancel (|score| ranks; its sign is the net side), and
  * carries the shared list of technical strategies + the overview's re-flag helper, so
    the dashboard hub (app.py) and the PDF report (convreport.py) score identically.

It is import-light (pandas/numpy + src.specs only — no Streamlit), so the headless report
subprocess can use it too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import SPECS, reflag_rows

# The price-based technical strategies the overview aggregates (the app's nav group mirrors
# this list). Order = display order.
TA_STRATEGIES = [
    "Mean Reversion", "Trend", "MA Crossover", "MA Swing", "Flag Breakout",
    "Support & Resistance", "Fibonacci Retracement", "Breakout & Retest",
    "Momentum (RSI/MACD)", "Bollinger Squeeze", "Elliott Wave", "Ichimoku Cloud",
    "On-Balance Volume", "Money Flow Index",
]

# ── the five axes of technical analysis ──────────────────────────────────────────────────────
# Every method belongs to exactly ONE axis — an independent DIMENSION of the market (direction /
# momentum / participation / location / structure). Agreement ACROSS axes is genuine corroboration;
# several methods WITHIN one axis are de-duplicated in the score (harmonic weights — strongest full,
# next ½, ⅓, …) so a single dimension can't vote twice. The app groups the confluence picker by these
# axes, and the report prints which method(s) represent each axis. Order = the axes' display order.
TA_AXES = {
    "Trend": ["Trend", "MA Crossover", "MA Swing", "Ichimoku Cloud"],
    "Momentum / Oscillators": ["Momentum (RSI/MACD)", "Mean Reversion"],
    "Volume": ["On-Balance Volume", "Money Flow Index"],
    "Support & Resistance": ["Support & Resistance", "Fibonacci Retracement"],
    "Patterns & Breakouts": ["Flag Breakout", "Breakout & Retest", "Bollinger Squeeze", "Elliott Wave"],
}
assert {s for methods in TA_AXES.values() for s in methods} == set(TA_STRATEGIES), \
    "TA_AXES must partition TA_STRATEGIES — every method in exactly one axis"

_AXIS_OF = {s: ax for ax, methods in TA_AXES.items() for s in methods}


def axis_of(strategy: str) -> str:
    """The axis a strategy sits in — or its own name if ungrouped, so an unknown method
    de-duplicates alone rather than being folded into another axis."""
    return _AXIS_OF.get(strategy, strategy)


def axis_breakdown(strategies):
    """[(axis, [methods in it]), …] over the axes actually represented in `strategies`, in axis
    display order — what the app dropdowns and the report's per-axis list both render from."""
    chosen = set(strategies)
    return [(ax, [m for m in methods if m in chosen])
            for ax, methods in TA_AXES.items() if any(m in chosen for m in methods)]


def has_intra_axis_dup(strategies) -> bool:
    """True if the set puts >1 method in any single axis — i.e. the within-axis de-dup will bite."""
    counts = {}
    for s in strategies:
        ax = axis_of(s)
        counts[ax] = counts.get(ax, 0) + 1
    return any(c > 1 for c in counts.values())

# The CONFLUENCE SET — the strategies whose agreement feeds the cross-strategy score. Chosen for
# INDEPENDENCE (one per axis — trend / momentum / volume / location / pattern) so agreement is
# genuine corroboration, not the same read echoed by several correlated strategies. EVERY other TA
# strategy keeps its own page and its chart overlays; it's just display/context, out of the score.
# The desk edits this set in the app; it persists to data/confluence_set.json.
CONFLUENCE_DEFAULT = [
    "Trend",                    # direction / regime
    "Momentum (RSI/MACD)",      # momentum & divergence
    "On-Balance Volume",        # volume / participation
    "Support & Resistance",     # location / tested levels
    "Flag Breakout",            # pattern + measured target
]
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# One confluence set PER BOOK — FICC futures and equities keep independent selections, so each TA
# page can score on its own axes/methods. `scope` selects the file; default "ficc" = the futures set.
_CONFLUENCE_FILES = {"ficc": _DATA_DIR / "confluence_set.json",
                     "equities": _DATA_DIR / "eq_confluence_set.json"}

# Back-compat alias — the de-dup below now covers EVERY axis (via axis_of), not just trend.
TREND_FAMILY = frozenset(TA_AXES["Trend"])


def confluence_set(scope: str = "ficc"):
    """The scored strategies for `scope` ('ficc' | 'equities'), from its confluence_set json (falls
    back to CONFLUENCE_DEFAULT). Ordered by TA_STRATEGIES (display order); unknown names dropped."""
    f = _CONFLUENCE_FILES.get(scope, _CONFLUENCE_FILES["ficc"])
    try:
        saved = set(json.loads(f.read_text(encoding="utf-8")).get("strategies", []))
    except Exception:
        saved = set()
    chosen = saved or set(CONFLUENCE_DEFAULT)
    return [s for s in TA_STRATEGIES if s in chosen]


def save_confluence_set(strategies, scope: str = "ficc") -> None:
    """Persist the confluence set (the scored strategies) for `scope` ('ficc' | 'equities')."""
    f = _CONFLUENCE_FILES.get(scope, _CONFLUENCE_FILES["ficc"])
    keep = [s for s in TA_STRATEGIES if s in set(strategies)]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"strategies": keep}, indent=2), encoding="utf-8")


def scoring_strategies(override=None):
    """The strategies that feed the score: the saved confluence set, or an explicit override."""
    if override is not None:
        ov = set(override)
        return [s for s in TA_STRATEGIES if s in ov]
    return confluence_set()


# Overview re-flag triggers. Most strategies use their page default (from SPECS); Trend's
# page default is 0 (it ranks the whole book), so the overview holds it to a selective bar so
# it lists genuine trends, not every market's stance. Tune here.
TA_HUB_TRIGGER = {"Trend": 10.0}

# Metric value that maps to full strength (100), per strategy. Metrics already on a 0–100
# scale (readiness / proximity / momentum / squeeze) use 100 (identity, clipped); z-scores
# top out near 3, the 3-month trend return near 25%, the MA gap near 10%.
STRENGTH_SCALE = {
    "Mean Reversion": 3.0,          # |z-score|
    "Trend": 25.0,                  # |3-month return %|
    "MA Crossover": 10.0,           # |MA gap %|
    "MA Swing": 10.0,               # |MA gap %|
    "Flag Breakout": 100.0,         # breakout readiness 0–100
    "Support & Resistance": 100.0,  # level proximity 0–100
    "Fibonacci Retracement": 100.0,  # Fib proximity 0–100
    "Breakout & Retest": 100.0,     # retest proximity 0–100
    "Momentum (RSI/MACD)": 100.0,   # momentum score 0–100
    "Bollinger Squeeze": 100.0,     # squeeze intensity 0–100
    "Elliott Wave": 100.0,          # wave fit 0–100
    "Ichimoku Cloud": 100.0,        # Ichimoku score 0–100
    "On-Balance Volume": 100.0,     # OBV score 0–100
    "Money Flow Index": 100.0,      # MFI score 0–100
}
DEFAULT_SCALE = 100.0


def strength(strategy: str, metric: float) -> float:
    """0–100 conviction for one signal, from its strategy-native metric magnitude."""
    try:
        metric = float(metric)
    except (TypeError, ValueError):
        return 0.0
    scale = STRENGTH_SCALE.get(strategy, DEFAULT_SCALE)
    if not np.isfinite(metric) or scale <= 0:
        return 0.0
    return float(np.clip(abs(metric) / scale * 100.0, 0.0, 100.0))


def ta_flagged(df: pd.DataFrame, strategies=None) -> pd.DataFrame:
    """The flagged technical signals across the CONFLUENCE SET (the scored strategies; pass
    `strategies` to override the saved set), re-flagged at the overview triggers. `df` is the
    cached opportunities frame (one row per strategy×product). Returns the subset that is flagged
    (signal ≠ "—", direction ≠ 0) with the standard columns; strategies without a symmetric
    ±trigger (MA crossover/swing, Bollinger) keep their own cached labels."""
    if df is None or "strategy" not in getattr(df, "columns", []):
        return pd.DataFrame()
    parts = []
    for s in scoring_strategies(strategies):
        sub = df[df["strategy"] == s].copy()
        if sub.empty:
            continue
        spec = SPECS.get(s, {})
        thr = TA_HUB_TRIGGER.get(s, spec.get("default"))
        if spec.get("hi") and thr is not None:
            sub = reflag_rows(sub, float(thr), spec["hi"], spec["lo"], fi_yield=True)
        parts.append(sub)
    if not parts:
        return df.iloc[0:0]
    ta = pd.concat(parts, ignore_index=True)
    return ta[ta["signal"].ne("—") & ta["direction"].ne(0)].copy()


def score_products(flagged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate flagged signals (one row per strategy×product) into one row per product.

    `flagged` needs columns: instruments, market, strategy, direction, metric. Returns one
    row per product with: instruments, market, n (# strategies), longs, shorts, conflict
    (both sides present), net_dir (+1/−1/0), conviction (mean strength 0–100), score (signed
    Σ strength → |score| ranks, sign = net side), tags (list of (strategy, direction,
    strength)). Sorted by |score| descending (highest conviction first)."""
    if flagged is None or flagged.empty:
        return pd.DataFrame(columns=["instruments", "market", "n", "longs", "shorts",
                                     "conflict", "net_dir", "conviction", "score", "tags"])
    rows = []
    for key, sub in flagged.groupby("instruments"):
        sig = [(r.strategy, int(r.direction), strength(r.strategy, r.metric))
               for r in sub.itertuples(index=False)]
        longs = sum(1 for _, d, _ in sig if d > 0)
        shorts = sum(1 for _, d, _ in sig if d < 0)
        # De-duplicate WITHIN each axis: several methods reading the same dimension (two trend
        # confirmations, OBV + MFI on volume, …) shouldn't each count in full. Per axis the
        # strongest counts full, the next at 1/2, 1/3, … (harmonic). Agreement ACROSS axes still
        # counts fully — that's the genuine confluence. (n / conviction stay raw.)
        axis_w = {}
        for ax in {axis_of(s) for s, _d, _st in sig}:
            members = sorted(((s, st) for s, _d, st in sig if axis_of(s) == ax), key=lambda x: -x[1])
            for k, (s, _st) in enumerate(members):
                axis_w[s] = 1.0 / (k + 1)
        signed = float(sum(d * st * axis_w.get(s, 1.0) for s, d, st in sig))
        mkt = sub["market"].mode()
        rows.append({
            "instruments": key,
            "market": mkt.iat[0] if not mkt.empty else sub["market"].iat[0],
            "n": len({s for s, _, _ in sig}),
            "longs": longs, "shorts": shorts, "conflict": bool(longs and shorts),
            "net_dir": 1 if signed > 0 else -1 if signed < 0 else 0,
            "conviction": round(float(np.mean([st for _, _, st in sig])), 1) if sig else 0.0,
            "score": round(signed, 1), "tags": sig,
        })
    out = pd.DataFrame(rows)
    return out.reindex(out["score"].abs().sort_values(ascending=False).index).reset_index(drop=True)
