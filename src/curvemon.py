"""Curve / RV monitor — the desk's spread book on the deep price store.

A fixed book of curve and relative-value spreads — rate curves, cross-market yield
spreads, STIR calendars, energy time-spreads, metal ratios — each measured against
its own history: a rolling z-score (how stretched vs the recent regime) plus a
percentile over the FULL ~10-year deep store (how stretched vs everything the last
decade has shown). The 'A'-generic live feed could never do this — Treasuries hold
~6 months of 'A' history — so this module reads the deep store directly.

SERIES CONVENTIONS (each chosen so the level is a real market observable):
  • Curve / cross-market rate spreads run on BENCHMARK YIELDS (deepstore.get_yields,
    the same series the TA book charts) in bp. No futures-price inversion, no roll
    gaps, directly comparable across a decade.
  • STIR calendars are front − second PRICE spread × 100 = bp (price convention:
    positive = front priced above next = tightening priced OUT the front… falls out
    of 100 − rate). Energy/metal calendars are front − second in native points —
    backwardation positive, contango negative.
  • Cross-product diffs and ratios use RAW front levels (actual traded prices, not
    panama-adjusted — an adjusted level is continuation fiction and would distort a
    ratio). The legs each jump at their own rolls; the spread of two fronts is the
    standard curve-shape observable and is what a 10-year percentile should rank.

Entry/objective/invalidation are mechanical, not advice: objective = the rolling
mean the z-score reverts to; invalidation = the level at ±INVAL_SIGMA σ on the
stretched side (a further-stretch stop). Client prose stays neutral.

Fallback: with no deep store on disk (fresh clone pre-backfill), yields and raw
fronts fall back to the live feed's ~400-day window; calendar spreads need the
stored '2' generic and simply drop. Depth is reported per spread so the page can
show what history each number stands on.

No Streamlit here — the page and the PDF CLI drive this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import deepstore, universe
from .volbt import point_value

REV = 2               # bump when the book/row schema changes — busts the page's st.cache_data
WINDOW = 252          # default rolling window (sessions) for the z-score
Z_THRESHOLD = 2.0     # |z| beyond this flags the spread as stretched
INVAL_SIGMA = 3.0     # invalidation level: mean ± this many rolling σ
MIN_OVERLAP = 60      # a spread needs at least this many joint sessions

GROUPS = ["Rates — Curve", "Rates — Cross-market", "STIR Calendars", "Energy", "Metals"]

# ── the spread book ─────────────────────────────────────────────────────────
# legs: [(weight, kind, ticker)] summed after each series is built, except
# kind_of_spread == "ratio" which divides the first leg by the second.
# kinds: "yield" (benchmark %, deep), "raw" ('1' actual level), "front2" ('2' level).
# scale multiplies the combined spread (100 turns % / price-points into bp).
#
# The rate-curve group carries the FULL 2/5/10/30 ladder for each market (every
# pairwise spread, generated below); `bench` marks the pair that market's desk
# actually quotes — US trades 2s10s and 5s30s (the 5Y is the policy pivot vs the
# term-premium long end), the euro market quotes off the Bund so its long-end
# steepener is 10s30s (Bund–Buxl). Starred on the page and in the PDF.
_CURVE_LADDERS = [
    ("us", "US", [("2", "TUA Comdty"), ("5", "FVA Comdty"),
                  ("10", "TYA Comdty"), ("30", "USA Comdty")]),
    ("de", "Germany", [("2", "DUA Comdty"), ("5", "OEA Comdty"),
                       ("10", "RXA Comdty"), ("30", "UBA Comdty")]),
]
BENCHMARKS = {"us_2s10s", "us_5s30s", "de_2s10s", "de_10s30s"}


def _curve_specs() -> list:
    out = []
    for mkey, mname, ladder in _CURVE_LADDERS:
        for i in range(len(ladder)):
            for j in range(i + 1, len(ladder)):
                (ta, fa), (tb, fb) = ladder[i], ladder[j]
                key = f"{mkey}_{ta}s{tb}s"
                bench = key in BENCHMARKS
                out.append({
                    "key": key, "name": f"{mname} {ta}s{tb}s", "group": "Rates — Curve",
                    "unit": "bp", "dp": 1, "bench": bench,
                    "legs": [(1, "yield", fb), (-1, "yield", fa)], "scale": 100.0,
                    "desc": f"{mname} {tb}Y minus {ta}Y benchmark yield — how steep the curve "
                            f"is between those two points (futures legs "
                            f"{fb.split()[0][:-1]} / {fa.split()[0][:-1]})."
                            + (" The benchmark quote for this market's curve." if bench else ""),
                })
    return out


SPREADS = _curve_specs() + [
    {"key": "box_2s10s", "name": "US − Germany 2s10s box", "group": "Rates — Curve", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "TYA Comdty"), (-1, "yield", "TUA Comdty"),
              (-1, "yield", "RXA Comdty"), (1, "yield", "DUA Comdty")], "scale": 100.0,
     "desc": "US 2s10s minus German 2s10s — relative curve shape between the two markets, "
             "four futures legs."},

    # Rates — cross-market 10Y spreads, in bp
    {"key": "ust_bund",  "name": "10Y Treasury − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "TYA Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "US 10Y minus German 10Y benchmark yield — the transatlantic spread, TY vs RX."},
    {"key": "oat_bund",  "name": "10Y OAT − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "OATA Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "France over Germany 10Y — the euro-area sovereign risk barometer, OAT vs RX."},
    {"key": "gilt_bund", "name": "10Y Gilt − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "G A Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "UK 10Y minus German 10Y benchmark yield — Gilt vs RX."},

    # STIR calendars — front − second price spread in bp
    {"key": "sfr_cal", "name": "3M SOFR cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")], "scale": 100.0,
     "desc": "Front SOFR quarterly minus the next — the pace of easing/tightening priced "
             "between the first two IMM dates (positive = lower front rate than next)."},
    {"key": "er_cal",  "name": "3M Euribor cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "ERA Comdty"), (-1, "front2", "ERA Comdty")], "scale": 100.0,
     "desc": "Front Euribor quarterly minus the next — the ECB path between the first two IMMs."},
    {"key": "sfi_cal", "name": "3M SONIA cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "SFIA Comdty"), (-1, "front2", "SFIA Comdty")], "scale": 100.0,
     "desc": "Front SONIA quarterly minus the next — the BoE path between the first two IMMs."},

    # Energy — time spreads (backwardation positive) + the flagship product spread
    {"key": "cl_cal", "name": "WTI M1−M2", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "CLA Comdty"), (-1, "front2", "CLA Comdty")], "scale": 1.0,
     "desc": "WTI prompt spread — positive = backwardation (tight prompt barrels)."},
    {"key": "co_cal", "name": "Brent M1−M2", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "COA Comdty"), (-1, "front2", "COA Comdty")], "scale": 1.0,
     "desc": "Brent prompt spread — positive = backwardation."},
    {"key": "ng_cal", "name": "Henry Hub M1−M2", "group": "Energy", "unit": "$/MMBtu", "dp": 3,
     "legs": [(1, "raw", "NGA Comdty"), (-1, "front2", "NGA Comdty")], "scale": 1.0,
     "desc": "US nat-gas prompt spread — seasonal storage economics at the front."},
    {"key": "ttf_cal", "name": "TTF M1−M2", "group": "Energy", "unit": "€/MWh", "dp": 2,
     "legs": [(1, "raw", "FJSA Comdty"), (-1, "front2", "FJSA Comdty")], "scale": 1.0,
     "desc": "European gas prompt spread — the storage-injection signal."},
    {"key": "wti_brent", "name": "WTI − Brent", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "CLA Comdty"), (-1, "raw", "COA Comdty")], "scale": 1.0,
     "desc": "Front WTI minus front Brent — the Atlantic-basin arb, actual traded levels."},

    # Metals — ratios on actual levels + the copper time spread
    {"key": "gc_si", "name": "Gold / Silver ratio", "group": "Metals", "unit": "×", "dp": 1,
     "kind_of_spread": "ratio",
     "legs": [(1, "raw", "GCA Comdty"), (1, "raw", "SIA Comdty")], "scale": 1.0,
     "desc": "Ounces of silver per ounce of gold, front futures — the precious relative-value dial."},
    {"key": "gc_pl", "name": "Gold / Platinum ratio", "group": "Metals", "unit": "×", "dp": 2,
     "kind_of_spread": "ratio",
     "legs": [(1, "raw", "GCA Comdty"), (1, "raw", "PLA Comdty")], "scale": 1.0,
     "desc": "Gold over platinum, front futures — precious vs industrial-precious."},
    # HG quotes in ¢/lb (659.10, not 6.59) — volbt's point value is per ¢-point ($250).
    {"key": "hg_cal", "name": "Copper M1−M2", "group": "Metals", "unit": "¢/lb", "dp": 2,
     "legs": [(1, "raw", "HGA Comdty"), (-1, "front2", "HGA Comdty")], "scale": 1.0,
     "desc": "COMEX copper prompt spread — positive = backwardation (tight metal)."},
]

SPREAD_BY_KEY = {s["key"]: s for s in SPREADS}


# ── history assembly ────────────────────────────────────────────────────────
def _needed(specs) -> dict:
    """{kind: sorted tickers} across `specs`."""
    need: dict = {}
    for s in specs:
        for _, kind, tkr in s["legs"]:
            need.setdefault(kind, set()).add(tkr)
    return {k: sorted(v) for k, v in need.items()}


def load_history(specs=None) -> pd.DataFrame:
    """One wide frame holding every leg series the book needs, columns keyed
    '<kind>:<ticker>'. Deep store first; yields and raw fronts fall back to the
    live feed's window when the store has nothing (fresh clone), '2' generics
    are store-only."""
    need = _needed(specs or SPREADS)
    frames = []

    def _add(df: pd.DataFrame, kind: str):
        if df is not None and not df.empty:
            frames.append(df.rename(columns={c: f"{kind}:{c}" for c in df.columns}))

    if "yield" in need:
        y = deepstore.get_yields(need["yield"])
        missing = [t for t in need["yield"] if t not in getattr(y, "columns", [])]
        _add(y, "yield")
        if missing:
            from .datafeed import get_yield_history
            _add(get_yield_history(missing), "yield")
    if "raw" in need:
        r = deepstore.get_raw(need["raw"])
        missing = [t for t in need["raw"] if t not in getattr(r, "columns", [])]
        _add(r, "raw")
        if missing:
            from .datafeed import get_history
            _add(get_history(missing, raw=True), "raw")
    if "front2" in need:
        _add(deepstore.get_front2(need["front2"]), "front2")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()].sort_index()


def _build_spread(spec: dict, history: pd.DataFrame) -> pd.Series | None:
    cols = [f"{kind}:{tkr}" for _, kind, tkr in spec["legs"]]
    if any(c not in history.columns for c in cols):
        return None
    legs = history[cols].dropna()
    if len(legs) < MIN_OVERLAP:
        return None
    if spec.get("kind_of_spread") == "ratio":
        s = legs.iloc[:, 0] / legs.iloc[:, 1]
    else:
        s = sum(w * legs[f"{kind}:{tkr}"] for w, kind, tkr in spec["legs"])
    return (s * spec.get("scale", 1.0)).rename(spec["key"])


def _half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life (same estimator as the Mean Reversion book)."""
    s = spread.dropna()
    if len(s) < 30:
        return float("nan")
    d = pd.concat([s.diff(), s.shift(1)], axis=1).dropna()
    if d.empty:
        return float("nan")
    b = np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("nan")


def _dollar_sigma(spec: dict, sigma: float) -> float | None:
    """1σ of the spread in $ per 1-lot-per-leg, ONLY where the volbt point value
    reconciles cleanly: same-product calendars and unit-weight diffs whose legs share
    one point value. Yield-space and ratio spreads return None ($ needs DV01 / lot
    ratios — we don't fake it)."""
    if "pv_unit" in spec:
        return abs(sigma) * spec["pv_unit"]
    if spec.get("kind_of_spread") == "ratio" or spec["unit"] == "bp" and any(
            kind == "yield" for _, kind, _ in spec["legs"]):
        return None
    pvs = {point_value(tkr) for _, _, tkr in spec["legs"]}
    if len(pvs) != 1 or not pvs or 0.0 in pvs:
        return None
    pv = pvs.pop() / spec.get("scale", 1.0)      # $ per displayed spread unit
    return abs(sigma) * pv


def _row(spec: dict, s: pd.Series, window: int, threshold: float) -> dict | None:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std
    if pd.isna(z.iloc[-1]):
        # not enough history for the window — fall back to whatever we have
        if len(s) < MIN_OVERLAP:
            return None
        mean = s.expanding(MIN_OVERLAP // 2).mean()
        std = s.expanding(MIN_OVERLAP // 2).std()
        z = (s - mean) / std
        if pd.isna(z.iloc[-1]):
            return None
    level, m, sd, zi = float(s.iloc[-1]), float(mean.iloc[-1]), float(std.iloc[-1]), float(z.iloc[-1])
    pctl = float((s <= level).mean() * 100.0)
    if zi >= threshold:
        signal, direction = "Rich", -1
    elif zi <= -threshold:
        signal, direction = "Cheap", 1
    else:
        signal, direction = "—", 0
    inval = m + INVAL_SIGMA * sd * (1 if zi >= 0 else -1)
    dsig = _dollar_sigma(spec, sd)
    return {
        "key": spec["key"], "name": spec["name"], "group": spec["group"],
        "unit": spec["unit"], "dp": spec["dp"], "desc": spec["desc"],
        "bench": bool(spec.get("bench", False)),
        "level": level, "chg1d": float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else float("nan"),
        "z": zi, "pctl": pctl, "mean": m, "sigma": sd,
        "hi": float(s.max()), "lo": float(s.min()),
        "half_life": _half_life(s.tail(window * 2)),
        "signal": signal, "direction": direction,
        "objective": m, "invalidation": inval, "dollar_sigma": dsig,
        "first": s.index.min().date().isoformat(), "days": int(len(s)),
        "asof": s.index.max().date().isoformat(),
    }


def monitor(window: int = WINDOW, threshold: float = Z_THRESHOLD,
            history: pd.DataFrame | None = None) -> pd.DataFrame:
    """The whole book, one row per spread, grouped in GROUPS order then by |z|."""
    if history is None:
        history = load_history()
    rows = []
    for spec in SPREADS:
        s = _build_spread(spec, history)
        if s is None:
            continue
        r = _row(spec, s, window, threshold)
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_g"] = df["group"].map({g: i for i, g in enumerate(GROUPS)})
    return (df.sort_values(["_g", "z"], key=lambda c: c.abs() if c.name == "z" else c,
                           ascending=[True, False])
              .drop(columns="_g").reset_index(drop=True))


def spread_chart_data(key: str, window: int = WINDOW, threshold: float = Z_THRESHOLD,
                      history: pd.DataFrame | None = None, years: float | None = None):
    """Per-spread series + summary for the detail chart (mean_reversion.pair_chart_data
    shape): DataFrame(date, spread, mean, upper, lower, z) + info dict. `years` trims
    the returned frame's left edge AFTER the stats are computed on full depth."""
    spec = SPREAD_BY_KEY[key]
    if history is None:
        history = load_history([spec])
    s = _build_spread(spec, history)
    if s is None:
        return pd.DataFrame(), {}
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std
    out = pd.DataFrame({
        "date": s.index, "spread": s.to_numpy(), "mean": mean.to_numpy(),
        "upper": (mean + threshold * std).to_numpy(),
        "lower": (mean - threshold * std).to_numpy(), "z": z.to_numpy(),
    })
    info = _row(spec, s, window, threshold) or {}
    info["threshold"] = threshold
    info["window"] = window
    if years:
        cut = s.index.max() - pd.DateOffset(days=int(years * 365.25))
        out = out[out["date"] >= cut].reset_index(drop=True)
    return out, info
