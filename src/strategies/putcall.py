"""Put/Call Ratios — options put/call open-interest & volume monitor (its own
dashboard page + client report).

The put/call ratio (puts ÷ calls) is a sentiment / positioning gauge. We carry it on
two bases:
  • OPEN INTEREST  — standing positioning; the HEADLINE signal that drives the ranking
                     and the put-heavy / call-heavy flag.
  • VOLUME         — today's traded flow; used for the day-over-day spike and the
                     flow-vs-OI divergence flags.

A raw P/C level isn't comparable across products (index options sit structurally
put-heavy from hedging; others sit nearer 1), so each product is normalized against
its OWN trailing-year history as a 0–100 percentile (the "P/C Percentile") plus a
z-score — the same idea as the COT Index. High percentile = unusually put-heavy
(defensive / hedging demand); low = unusually call-heavy (bullish). Extremes are
frequently read as contrarian. Data flows through the datafeed put/call seam (mock
today, Bloomberg once live); the visual client report is src/pcreport.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datafeed import get_putcall, get_history
from ..universe import INSTRUMENTS, name, asset, region
from .base import frame
from .volatility import _ord                      # shared English-ordinal helper

STRATEGY = "Put/Call Ratios"
STAT_WINDOW = 252        # ~1y of obs for the percentile / z-score
MINP = 60                # need ~a quarter-year before the percentile is meaningful
EXTREME_HI = 80.0        # OI P/C percentile ≥ this → put-heavy extreme
EXTREME_LO = 20.0        # OI P/C percentile ≤ this → call-heavy extreme
SPIKE_Z = 2.0            # |z| of the 1-day OI P/C change → flag a fresh shift
DIVERGENCE = 35.0        # |vol %ile − OI %ile| ≥ this → flow running away from positioning
HISTORY_WINDOW = 252     # days of P/C history cached for the report charts

# ── Data-quality guards (2026-08-23) ────────────────────────────────────────────
# A ratio is only meaningful when BOTH legs carry real size. The old guard masked
# exact zeros only, which never fired once — the store holds no exact zeros at all —
# while the real failure mode is a 1-to-5-contract leg: Coffee (Arabica) published
# 17,385.00 (= 52,155 puts ÷ 3 calls, the put leg stale-filled) and Brazilian Real
# published 0.01 (= 10 ÷ 2,000, its put leg frozen at 10.0 for 271 sessions). Both
# reached the client PDF. Floors are symmetric because BRL fails on the NUMERATOR —
# a denominator-only guard misses it.
MIN_LEG_OI = 100         # contracts of open interest required on each leg
MIN_LEG_VOL = 100        # same floor on traded volume: measured 2026-08-23, dropping it to
                         # 25 buys 3 more products priced but doubles the implausible
                         # volume ratios (8 vs 4), and the volume ratio is the secondary read
MAX_STALE = 5            # sessions: a ratio older than this is dropped, not carried forward
PARTIAL_FRAC = 0.55      # a session printing < this share of its recent product count is
                         # a partial capture (the 06:45 pull) — not a real session

DATA = Path(__file__).resolve().parents[2] / "data" / "signals"
DETAIL_FILE = DATA / "putcall.parquet"
HISTORY_FILE = DATA / "putcall_history.parquet"

DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "pc_oi", "pc_vol",
                  "oi_pctl", "oi_z", "vol_pctl", "tot_call", "tot_put", "avg_day",
                  "call_last", "put_last", "avg_call", "avg_put", "vol_days",
                  "oi_chg_z", "divergence", "signal", "direction"]
# tot_call / tot_put = total call / put contracts TRADED over the trailing window;
# avg_day = average total option contracts (calls + puts) traded per day. The window
# is STAT_WINDOW sessions (~1y) so it lines up with the percentile basis.
# call_last / put_last = the PREVIOUS DAY's call / put contracts traded; avg_call /
# avg_put = each side's trailing-1y daily average — for the daily activity leaderboard.
HISTORY_COLUMNS = ["date", "ticker", "market", "asset", "pc_oi", "pc_vol",
                   "call_vol", "put_vol", "oi_pctl", "price"]


def _ratio(num: pd.DataFrame, den: pd.DataFrame, tickers, min_leg: float = MIN_LEG_OI) -> pd.DataFrame:
    """puts ÷ calls, element-wise, blanking any cell where EITHER leg is below
    `min_leg` contracts — a ratio off a 3-contract leg is arithmetic, not positioning."""
    n = num.reindex(columns=tickers)
    d = den.reindex(columns=tickers)
    thin = (n < min_leg) | (d < min_leg) | n.isna() | d.isna()
    return (n / d).mask(thin)


def _pctl_now(series: pd.Series) -> float:
    """Percentile rank (0–100) of the latest value within the trailing STAT_WINDOW.

    A frozen series has no percentile: `(s <= s.iloc[-1]).mean()` is 1.0 when every
    value is identical, which stamped dead tickers 100th-percentile "Put-heavy"
    indefinitely (3M SONIA sat there for 51 sessions after its feed died)."""
    s = series.dropna().iloc[-STAT_WINDOW:]
    if len(s) < MINP or s.nunique() <= 1:
        return float("nan")
    return float((s <= s.iloc[-1]).mean() * 100.0)


def _z_now(series: pd.Series) -> float:
    """z-score of the latest value over the trailing STAT_WINDOW."""
    s = series.dropna().iloc[-STAT_WINDOW:]
    if len(s) < MINP:
        return float("nan")
    sd = float(s.std())
    return (float(s.iloc[-1]) - float(s.mean())) / sd if sd > 1e-9 else float("nan")


def _rolling_pctl(series: pd.Series, window: int = STAT_WINDOW, minp: int = MINP) -> pd.Series:
    """Trailing percentile rank (0–100) of each point within its prior `window` — the
    series that drives the heatmap and the per-product oscillator in the report.
    A frozen window yields NaN rather than a spurious 100 (see `_pctl_now`)."""
    return series.rolling(window, min_periods=minp).apply(
        lambda w: np.nan if np.nanmin(w) == np.nanmax(w) else (w <= w[-1]).mean() * 100.0,
        raw=True)


def _last_complete(comp: dict) -> pd.Timestamp | None:
    """The last session that is a REAL session rather than a partial capture.

    The morning pull runs at ~06:45 and the final row can be a fraction of a session
    (2026-08-21 printed 2% of normal panel volume across 30 of 89 products). Such a row
    would read as a collapse in flow, so anchor on the last date that (a) carries open
    interest and (b) prints for a normal share of the panel."""
    oi = comp.get("put_oi")
    if oi is None or getattr(oi, "empty", True):
        return None
    counts = oi.notna().sum(axis=1)
    typical = float(counts.tail(40).median() or 0)
    if typical <= 0:
        return oi.index[-1]
    ok = counts[counts >= PARTIAL_FRAC * typical]
    return ok.index[-1] if len(ok) else oi.index[-1]


def _persist_history(oi: pd.DataFrame, vol: pd.DataFrame, cv: pd.DataFrame, pv: pd.DataFrame,
                     px: pd.DataFrame, window: int = HISTORY_WINDOW) -> None:
    """Cache ~1y of OI & volume P/C, the raw daily call/put volumes traded, the trailing
    OI-P/C percentile, and the underlying price per market (long format) for the report's
    time-series charts + heatmap."""
    frames = []
    pxcols = getattr(px, "columns", [])
    for t in INSTRUMENTS:
        if t not in oi.columns:
            continue
        h = pd.DataFrame({
            "pc_oi": oi[t],
            "pc_vol": vol[t] if t in vol.columns else np.nan,
            "call_vol": cv[t] if t in cv.columns else np.nan,
            "put_vol": pv[t] if t in pv.columns else np.nan,
            "oi_pctl": _rolling_pctl(oi[t]),
            "price": px[t] if t in pxcols else np.nan,
        }).iloc[-window:]
        h = h.dropna(subset=["pc_oi"])
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "pc_oi", "pc_vol", "call_vol", "put_vol", "oi_pctl", "price"]
        h["ticker"], h["market"], h["asset"] = t, name(t), asset(t)
        frames.append(h[HISTORY_COLUMNS])
    if frames:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(HISTORY_FILE, index=False)
        except Exception:
            pass


def compute_table() -> pd.DataFrame:
    """Full cross-section: one row per market with OI & volume P/C, the OI-P/C
    percentile (headline) + z, the volume percentile, the 1-day OI-P/C change z (spike)
    and the flow-vs-OI divergence, sorted most-extreme first. Source of truth for both
    the dashboard rows and the visual report."""
    tickers = list(INSTRUMENTS)
    comp = get_putcall(tickers)
    cutoff = _last_complete(comp)                     # ignore a partial morning capture
    if cutoff is not None:
        comp = {k: v.loc[:cutoff] for k, v in comp.items() if v is not None}
    oi = _ratio(comp["put_oi"], comp["call_oi"], tickers)
    vol = _ratio(comp["put_vol"], comp["call_vol"], tickers, min_leg=MIN_LEG_VOL)
    call_vol, put_vol = comp["call_vol"], comp["put_vol"]   # raw traded volumes (for the totals)
    n_sessions = len(oi.index)
    try:
        _persist_history(oi, vol, call_vol, put_vol, get_history(tickers))
    except Exception:
        pass

    recs = []
    for t in tickers:
        oi_s = oi[t].dropna() if t in oi.columns else pd.Series(dtype=float)
        vol_s = vol[t].dropna() if t in vol.columns else pd.Series(dtype=float)
        if oi_s.empty:
            continue
        # Staleness: `dropna` would otherwise republish a ratio from weeks ago as today's
        # — which is how a dead feed keeps printing (3M SONIA froze for 51 sessions).
        if n_sessions - 1 - oi.index.get_loc(oi_s.index[-1]) > MAX_STALE:
            continue
        pc_oi = float(oi_s.iloc[-1])
        pc_vol = float(vol_s.iloc[-1]) if not vol_s.empty else float("nan")
        oi_pctl, oi_z = _pctl_now(oi_s), _z_now(oi_s)
        vol_pctl = _pctl_now(vol_s) if not vol_s.empty else float("nan")
        oi_chg = oi_s.diff().dropna()
        oi_chg_z = _z_now(oi_chg) if len(oi_chg) >= MINP else float("nan")
        divergence = (vol_pctl - oi_pctl) if (np.isfinite(vol_pctl) and np.isfinite(oi_pctl)) else float("nan")

        # Traded-volume totals + daily average over the trailing window (~1y), plus the
        # previous day's volumes and each side's daily average (for the activity leaderboard).
        cv = call_vol[t].dropna().iloc[-STAT_WINDOW:] if t in call_vol.columns else pd.Series(dtype=float)
        pv = put_vol[t].dropna().iloc[-STAT_WINDOW:] if t in put_vol.columns else pd.Series(dtype=float)
        tot_call = float(cv.sum()) if not cv.empty else float("nan")
        tot_put = float(pv.sum()) if not pv.empty else float("nan")
        call_last = float(cv.iloc[-1]) if not cv.empty else float("nan")
        put_last = float(pv.iloc[-1]) if not pv.empty else float("nan")
        avg_call = float(cv.mean()) if not cv.empty else float("nan")
        avg_put = float(pv.mean()) if not pv.empty else float("nan")
        if t in call_vol.columns and t in put_vol.columns:
            day = (call_vol[t] + put_vol[t]).dropna().iloc[-STAT_WINDOW:]
            avg_day = float(day.mean()) if not day.empty else float("nan")
        else:
            avg_day = float("nan")

        if np.isfinite(oi_pctl) and oi_pctl >= EXTREME_HI:
            signal, direction = "Put-heavy", -1
        elif np.isfinite(oi_pctl) and oi_pctl <= EXTREME_LO:
            signal, direction = "Call-heavy", 1
        else:
            signal, direction = "—", 0

        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "pc_oi": round(pc_oi, 2),
            "pc_vol": round(pc_vol, 2) if np.isfinite(pc_vol) else np.nan,
            "oi_pctl": round(oi_pctl) if np.isfinite(oi_pctl) else np.nan,
            "oi_z": round(oi_z, 2) if np.isfinite(oi_z) else np.nan,
            "vol_pctl": round(vol_pctl) if np.isfinite(vol_pctl) else np.nan,
            "tot_call": round(tot_call) if np.isfinite(tot_call) else np.nan,
            "tot_put": round(tot_put) if np.isfinite(tot_put) else np.nan,
            "avg_day": round(avg_day) if np.isfinite(avg_day) else np.nan,
            "call_last": round(call_last) if np.isfinite(call_last) else np.nan,
            "put_last": round(put_last) if np.isfinite(put_last) else np.nan,
            "avg_call": round(avg_call) if np.isfinite(avg_call) else np.nan,
            "avg_put": round(avg_put) if np.isfinite(avg_put) else np.nan,
            "vol_days": int(len(cv)),       # days of put/call volume the average is built on
            "oi_chg_z": round(oi_chg_z, 2) if np.isfinite(oi_chg_z) else np.nan,
            "divergence": round(divergence) if np.isfinite(divergence) else np.nan,
            "signal": signal, "direction": direction,
        })

    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = (df["oi_pctl"].fillna(50) - 50).abs().sort_values(ascending=False).index
        df = df.reindex(order).reset_index(drop=True)
    return df


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    tbl = compute_table()

    # Cache the rich table for the visual report (compute-once-daily, like the rest).
    try:
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        ctx = f"OI P/C {r.pc_oi:.2f}"
        if pd.notna(r.pc_vol):
            ctx += f" · vol P/C {r.pc_vol:.2f}"
        if pd.notna(r.oi_pctl):
            ctx += f" · {_ord(r.oi_pctl)} %ile (1y)"
        if pd.notna(r.avg_day):
            ctx += f" · ~{r.avg_day:,.0f} contracts/day"
        extra = []
        if pd.notna(r.oi_chg_z) and abs(r.oi_chg_z) >= SPIKE_Z:
            extra.append("1d shift")
        if pd.notna(r.divergence) and abs(r.divergence) >= DIVERGENCE:
            extra.append("flow≠OI")
        if extra:
            ctx += " · " + " · ".join(extra)
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.oi_pctl) if pd.notna(r.oi_pctl) else 0.0,
            "metric_label": "OI P/C %ile (1y)",
            "level": float(r.pc_oi),
            "context": ctx,
        })
    return frame(rows)   # already sorted most-extreme first
