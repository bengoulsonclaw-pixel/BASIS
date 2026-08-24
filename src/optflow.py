"""Unusual option flow — a product traded far more puts (or calls) today than it
normally does (Ben, 2026-08-23: "if a product has had a substantial amount of puts or
calls traded in a single day relative to what it normally gets, that should be pointed
out on the Hot Sheet").

This is a DIFFERENT KIND of row from the rest of the sheet, and is deliberately built
and scored that way. Every other provider reports a DISLOCATION — positioning is
crowded, a correlation has broken, a wing is mispriced, a spread sits at an extreme —
each of which implies something is out of line and is scored by how extreme it is. A
flow row implies nothing. It says a lot of contracts changed hands here today, go and
look. So:

  • the prose is pure OBSERVATION — no direction, no rich/cheap, no buy/sell (it flows
    into client PDFs, see [[client-commentary-not-advice]]);
  • heat is capped at FLOW_HEAT_CEIL so a flow row can never push a dislocation row out
    of the sheet's top 10 (measured: the tightest observed day needs heat > 99.2 to
    enter the strip, and 27-29 FICC rows already sit above 45 every day);
  • at most MAX_ROWS a day, one per product.

WHY THE GATE IS A PERCENTILE AND THE DISPLAY IS A MULTIPLE (measured 2026-08-23 over
400 sessions x 89 tickers). "N times normal" is what a broker wants to read, but it
cannot be the selector: the per-product 95th percentile of that multiple spans 1.76x
(DAX) to 173x (Mexican Peso), a 98-fold spread, so one global cutoff fires 42% of days
in Japanese Yen and literally never in DAX, KOSPI or Euro Stoxx 50. Gating on each
product's OWN trailing percentile — the same normalisation the COT index uses — spreads
the rows across 21 products with the busiest owning 8.8%. So: gate on the percentile,
show the multiple, rank by the multiple (ranking by a z-score puts a smaller displayed
multiple above a larger one on half of multi-fire days, which reads as broken).

The expiry roll turned out NOT to contaminate this: firing days sit near an option
expiry at 0.56-1.13x the base rate, because a 60-session baseline already spans ~3
monthly expiries and prices the roll in. The one real pocket is equity indices around
quarterly triple-witching (lift 2.8x, and demonstrably churn — next-day open-interest
change of 0.19% against 1.10% for the rows that survive), so those are carved out.
STIRs are NOT carved out: their quarterly lift is the same size but carries real news
(same-day move 2.13 sigma in-window against 0.82 out) — those are IMM/FOMC dates.

Reads the cached snapshot stores only, never a fetch (the provider contract in
hotsheet.py). Ratios/percentiles for the client-facing page live in
src/strategies/putcall.py; this module is the Hot Sheet seam, mirroring the way
src/cotdata.py provides for src/strategies/cot.py — hotsheet.discover() globs only the
TOP level of src/, so a provider inside src/strategies/ would never be found.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot"

# ── metric ──────────────────────────────────────────────────────────────────────
BASE_WINDOW = 60         # non-null sessions in the median baseline (120 halves the
                         # signal rate and drops the universe from 53 tickers to 37)
PCTL_WINDOW = 252        # own-history lookback for the percentile gate
PCTL_MINP = 60           # ratios needed before that percentile means anything

# ── gates ───────────────────────────────────────────────────────────────────────
MIN_PCTL = 98.0          # the selector: today's multiple vs the product's own year
MIN_RATIO = 3.0          # and it must still be a big multiple in absolute terms
MIN_LOTS = 1_000         # contracts traded today — insurance against a silly row
MIN_BASE = 1_000         # baseline contracts/day: kills the 3-lots-a-day products
MIN_PRINTS_60 = 45       # of the last 60 sessions — a mechanical liveness test that
                         # drops a dead feed by itself (3M SONIA froze in June 2026)
MAX_GAP = 3              # sessions since the product's previous print
PARTIAL_FRAC = 0.25      # a session whose panel volume is below this share of its
                         # trailing median is a partial morning capture, not a session
MAX_ROWS = 2             # hard per-day cap

# ── heat: bounded so a flow row never displaces a dislocation row ───────────────
FLOW_HEAT_CEIL = 45.0    # 3x -> 16.5, 5x -> 24.2, 10x -> 34.6, 20x+ -> 45.0
HEAT_FULL_X = 20.0       # multiple at which the gauge saturates

SPARK_N = 60             # sessions of the product's own daily volume carried per spark
_INDEX_ASSET = "Indices"  # the only book carved out around quarterly expiry
_QUARTERLY_MONTHS = (3, 6, 9, 12)
_WITCH_WINDOW = 2        # sessions either side of quarterly expiry to suppress


def _read(name: str) -> pd.DataFrame | None:
    """One cached store, or None — a provider never takes the sheet down."""
    try:
        df = pd.read_parquet(SNAP / name)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if "date" in df.columns:                      # stored with date as a column
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    first_fri = d + pd.Timedelta(days=(4 - d.dayofweek) % 7)
    return first_fri + pd.Timedelta(days=14)


def _in_witching_window(day: pd.Timestamp, sessions: pd.DatetimeIndex) -> bool:
    """True when `day` sits within +/-_WITCH_WINDOW SESSIONS of a quarterly
    (Mar/Jun/Sep/Dec) third-Friday expiry — counted in sessions, not calendar days,
    so a holiday cannot slide the window off."""
    try:
        pos = sessions.get_indexer([day])[0]
    except Exception:
        return False
    if pos < 0:
        return False
    lo, hi = max(0, pos - _WITCH_WINDOW), min(len(sessions) - 1, pos + _WITCH_WINDOW)
    for d in sessions[lo:hi + 1]:
        if d.month in _QUARTERLY_MONTHS:
            wf = _third_friday(d.year, d.month)
            # the roll lands on the third Friday, or the session nearest it
            if abs((d - wf).days) <= 1:
                return True
    return False


def _anchor(vol: pd.DataFrame, oi: pd.DataFrame | None) -> pd.Timestamp | None:
    """The last REAL session. The morning pull runs ~06:45, so the final volume row is
    routinely a fraction of a session (2026-08-21 printed 2.1% of normal panel volume);
    scoring it would read as a collapse in flow, and the feature would go blank every
    pull morning. Anchor on the last date that carries open interest AND a normal share
    of the panel's usual volume."""
    if vol is None or vol.empty:
        return None
    total = vol.sum(axis=1, skipna=True)
    typical = float(total.tail(BASE_WINDOW).median() or 0.0)
    ok = total.index if typical <= 0 else total[total >= PARTIAL_FRAC * typical].index
    if oi is not None and not oi.empty:
        ok = ok[ok <= oi.index.max()]              # OI lands a session behind volume
    return ok[-1] if len(ok) else None


def _score_side(s: pd.Series, day: pd.Timestamp) -> dict | None:
    """One product, one side. `s` is that side's daily contract volume, oldest→newest,
    truncated at the anchor session. NaN means the feed was dark, NOT that nothing
    traded (the stores hold no exact zeros at all), so gaps are compressed out rather
    than imputed — imputing zero would manufacture spikes."""
    s = s.dropna()
    if len(s) < BASE_WINDOW + PCTL_MINP + 1 or s.index[-1] != day:
        return None                                # not scoreable, or didn't print today
    # liveness: enough recent prints, and no long gap right before today
    recent = s.index[s.index > day - pd.Timedelta(days=90)]
    if len(recent) < MIN_PRINTS_60 or (day - s.index[-2]).days > MAX_GAP * 2:
        return None
    # baseline EXCLUDES today (shift 1) — today's volume must never enter its own normal
    base_s = s.shift(1).rolling(BASE_WINDOW, min_periods=BASE_WINDOW).median()
    ratio_s = (s / base_s).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio_s) < PCTL_MINP or ratio_s.index[-1] != day:
        return None
    vol_now, base_now, ratio = float(s.iloc[-1]), float(base_s.iloc[-1]), float(ratio_s.iloc[-1])
    if not np.isfinite(base_now) or base_now < MIN_BASE or vol_now < MIN_LOTS or ratio < MIN_RATIO:
        return None
    hist = ratio_s.iloc[-PCTL_WINDOW:]
    if hist.nunique() <= 1:
        return None
    pctl = float((hist <= ratio).mean() * 100.0)
    if pctl < MIN_PCTL:
        return None
    return {"vol": vol_now, "base": base_now, "ratio": ratio, "pctl": pctl,
            "spark": s.iloc[-SPARK_N:].tolist()}


def candidates(day: pd.Timestamp | None = None) -> list[dict]:
    """Every product/side clearing the gates on the anchor session, biggest multiple
    first, one row per product. Separated from radar_items() so it can be tested and
    replayed on a past date without the Hot Sheet in the way."""
    pv, cv = _read("putcall_put_vol.parquet"), _read("putcall_call_vol.parquet")
    if pv is None or cv is None:
        return []
    oi = _read("putcall_put_oi.parquet")
    both = pv.join(cv, how="outer", lsuffix="_p", rsuffix="_c").index
    sessions = pd.DatetimeIndex(sorted(set(both)))
    day = pd.Timestamp(day) if day is not None else _anchor(pv.add(cv, fill_value=0), oi)
    if day is None:
        return []
    pv, cv = pv.loc[:day], cv.loc[:day]

    from src.universe import INSTRUMENTS, asset
    out = []
    for t in INSTRUMENTS:
        if asset(t) == _INDEX_ASSET and _in_witching_window(day, sessions):
            continue                               # triple-witching churn, not news
        best = None
        for side, frame in (("put", pv), ("call", cv)):
            if t not in frame.columns:
                continue
            try:
                sc = _score_side(frame[t], day)
            except Exception:
                sc = None
            if sc and (best is None or sc["ratio"] > best["ratio"]):
                best = {**sc, "ticker": t, "side": side}
        if best:
            out.append(best)
    out.sort(key=lambda r: -r["ratio"])
    return out


def radar_items() -> list:
    """Hot Sheet provider — the day's unusual option activity. Cache-only; returns []
    rather than raising on any missing or malformed store."""
    try:
        from src import hotsheet
        from src.reportkit import ordinal
        from src.universe import name

        rows = candidates()[:MAX_ROWS]
        out = []
        for c in rows:
            heat = min(FLOW_HEAT_CEIL,
                       FLOW_HEAT_CEIL * math.log10(max(c["ratio"], 1.0)) / math.log10(HEAT_FULL_X))
            out.append(hotsheet.item(
                tag="FLOW", key=f"{c['ticker']}:{c['side']}", section="Option Flow",
                text=(f"**{name(c['ticker'])}** traded {c['vol']:,.0f} {c['side']} option "
                      f"contracts — **{c['ratio']:.1f}x** its normal daily {c['side']} "
                      f"volume ({ordinal(int(round(c['pctl'])))} percentile of the year)."),
                heat=heat, metric=f"{c['ratio']:.1f}x normal",
                sub=f"vs {c['base']:,.0f}/day, 60-session median",
                value=float(c["vol"]), ticker=c["ticker"],
                page="Put/Call Ratios", book="ficc", spark=c["spark"]))
        return out
    except Exception:
        return []
