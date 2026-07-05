"""Implied-vs-realized volatility monitor — the Volatility Report.

For every market with listed options we compare 1-month at-the-money IMPLIED
vol against ~1-month (21 trading day) REALIZED vol. The headline number is the
**z-score of the implied−realized spread over the past year** — i.e. how far
today's vol-risk premium has stretched from its own norm. That is the "spread
has reached a high number" signal:

  • z >= +1.5  -> implied unusually rich vs realized -> SELL vol (sell premium)
  • z <= -1.5  -> implied unusually cheap vs realized -> BUY  vol

Both legs come from datafeed (mock today, Bloomberg once live): realized is
computed from settlement prices; implied is the constant-maturity 1M ATM surface
(FX from the OTC pair vol). `compute_table()` returns the full rich cross-section
(IV, RV, spread, z, percentile) and is also what the visual client report reads;
`find_opportunities()` maps it onto the shared strategy schema for the dashboard.

Note for fixed income: STIR/bond option vol may be quoted normal (bp) not
lognormal/price vol — confirm the convention so the implied leg matches this
price-return realized leg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datafeed import (get_realized_vol_history, get_implied_vol_history,
                        get_skew_components, get_history, stale_iv_reasons, warn_stale,
                        price_decimals)
from ..universe import INSTRUMENTS, name, asset, region
from .base import frame

STRATEGY = "Volatility"
STAT_WINDOW = 252       # ~1 year of obs for the z-score / percentile
Z_FLAG = 1.5            # |z| beyond this flags an opportunity (matches mean_reversion)
_SQRT252 = np.sqrt(252.0)   # de-annualiser for the daily 1σ move shown next to each vol

# STIR futures are pinned near 100, so their *price* realized vol is ~0 and can't
# be compared to the option-surface implied (which isn't price vol). Rate vol
# needs a dedicated normal/basis-point treatment — excluded here until then.
EXCLUDE_ASSETS = {"STIRs"}

# Non-US index FUTURES whose option surface returns nothing on the Terminal stay
# excluded; the EMEA CASH indices (SX5E/SX7E/DAX/UKX Index, added 2026-06-04) DO
# have surface vol, so they are absent here and flow into the vol reports.
EXCLUDE_INDEX = {"VGA Index", "CAA Index", "Z A Index", "GXA Index", "SMA Index",
                 "NKA Index", "KMA Index", "XPA Index"}

# Skew uses fixed 90/110% moneyness wings for listed products (FX uses the OTC
# 25-delta risk reversal — it has no moneyness surface). When ATM price vol is low
# those strikes sit tens of σ OTM where the surface is extrapolated garbage, so
# BONDS are dropped from SKEW (their ATM vol page is fine — that's 100% moneyness)
# and STIRs too (via EXCLUDE_ASSETS). Any |skew| beyond the cap is treated as bad
# surface data.
SKEW_EXCLUDE_EXTRA = frozenset({"Bonds"})
SKEW_CAP = 2.5


def _excluded(t: str, extra_assets=frozenset()) -> bool:
    a = asset(t)
    if a in EXCLUDE_ASSETS or a in extra_assets:
        return True
    if a == "Indices" and t in EXCLUDE_INDEX:        # dead non-US index futures
        return True
    return False

# Rich per-market tables cached here for the visual client report (src/volreport.py).
DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "volatility.parquet"
SKEW_DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "skew.parquet"
HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "volatility_history.parquet"
SKEW_HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "skew_history.parquet"

DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "iv", "rv",
                  "spread", "z", "pctl", "signal", "direction",
                  "iv_sd", "rv_sd", "px_dec"]   # daily 1σ price move + native decimals
SKEW_DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "put", "call", "atm",
                       "skew", "z", "pctl", "signal", "direction"]


def _ord(n: float) -> str:
    """English ordinal: 1->1st, 2->2nd, 3->3rd, 11->11th, 92->92nd."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _z_and_pctl(series: pd.Series):
    """Current value, its z-score and percentile over the trailing STAT_WINDOW."""
    s = series.dropna()
    if s.empty:
        return float("nan"), float("nan"), float("nan")
    now = float(s.iloc[-1])
    win = s.iloc[-STAT_WINDOW:]
    sd = float(win.std())
    z = (now - float(win.mean())) / sd if sd > 1e-9 else float("nan")
    pctl = float((win <= now).mean() * 100.0) if len(win) >= 20 else float("nan")
    return now, z, pctl


def _last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _persist_history(iv, rv, px, window: int = STAT_WINDOW):
    """Cache ~1y of the implied−realized spread and the underlying price per market
    (long format) for the report's spread-vs-underlying time-series chart."""
    frames = []
    for t in INSTRUMENTS:
        if _excluded(t) or not all(t in d.columns for d in (iv, rv, px)):
            continue
        h = pd.DataFrame({"spread": iv[t] - rv[t], "price": px[t]}).dropna().iloc[-window:]
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "spread", "price"]
        h["ticker"] = t
        frames.append(h[["date", "ticker", "spread", "price"]])
    if frames:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(HISTORY_FILE, index=False)
        except Exception:
            pass


def _persist_skew_history(put, call, atm, px, window: int = STAT_WINDOW):
    """Cache ~1y of the (90% put − 110% call)/ATM skew and the underlying price per
    market (long format) for the skew report's skew-vs-underlying time-series chart."""
    frames = []
    for t in INSTRUMENTS:
        if _excluded(t, SKEW_EXCLUDE_EXTRA) or not all(t in d.columns for d in (put, call, atm, px)):
            continue
        skew = (put[t] - call[t]) / atm[t].replace(0, np.nan)
        h = pd.DataFrame({"skew": skew, "price": px[t]}).dropna().iloc[-window:]
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "skew", "price"]
        h["ticker"] = t
        frames.append(h[["date", "ticker", "skew", "price"]])
    if frames:
        try:
            SKEW_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(SKEW_HISTORY_FILE, index=False)
        except Exception:
            pass


def compute_table() -> pd.DataFrame:
    """Full cross-section: one row per market with IV, RV, spread, z and percentile,
    sorted by |z| (most stretched first). The single source of truth for both the
    dashboard rows and the visual report."""
    tickers = list(INSTRUMENTS)
    rv = get_realized_vol_history(tickers)
    iv = get_implied_vol_history(tickers)
    px = get_history(tickers)                            # underlying settlement — SD + history
    try:
        _persist_history(iv, rv, px)                     # 1y spread + underlying for the report
    except Exception:
        pass

    stale = stale_iv_reasons(iv)                     # surfaces that aren't a live quote
    recs, skipped_stale = [], []
    for t in tickers:
        if _excluded(t):
            continue
        if t not in rv.columns or t not in iv.columns:
            continue
        if t in stale:                               # frozen/dead surface — don't score it
            skipped_stale.append(t)
            continue
        rv_s, iv_s = rv[t].dropna(), iv[t].dropna()
        spread_s = (iv[t] - rv[t]).dropna()
        if rv_s.empty or iv_s.empty or spread_s.empty:
            continue
        iv_now, rv_now = float(iv_s.iloc[-1]), float(rv_s.iloc[-1])
        if not (np.isfinite(iv_now) and np.isfinite(rv_now)):
            continue
        spread, z, pctl = _z_and_pctl(spread_s)

        if np.isfinite(z) and z >= Z_FLAG:
            signal, direction = "Rich — sell vol", -1
        elif np.isfinite(z) and z <= -Z_FLAG:
            signal, direction = "Cheap — buy vol", 1
        else:
            signal, direction = "—", 0

        # Daily 1σ move in the contract's own price units — vol ÷ √252 × price — shown
        # in brackets next to each vol, to the underlying's native number of decimals.
        ps = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        S = float(ps.iloc[-1]) if not ps.empty else float("nan")
        dec = price_decimals(ps, asset(t)) if not ps.empty else 1
        iv_sd = round(S * iv_now / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan
        rv_sd = round(S * rv_now / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan

        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "iv": round(iv_now, 1), "rv": round(rv_now, 1), "spread": round(spread, 1),
            "z": round(z, 2) if np.isfinite(z) else np.nan,
            "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
            "signal": signal, "direction": direction,
            "iv_sd": iv_sd, "rv_sd": rv_sd, "px_dec": dec,
        })

    warn_stale("Volatility", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


def compute_skew_table() -> pd.DataFrame:
    """Cross-section of the normalized skew (90% put − 110% call)/ATM, z-scored over
    the same 252-day window (positive = puts richer than calls). Listed markets use
    the 90/110% moneyness wings off the surface, FX the OTC 25-delta risk reversal.
    Feeds the report's skew page. z >= +1.5 -> skew rich (sell); z <= -1.5 -> cheap
    (buy)."""
    tickers = list(INSTRUMENTS)
    comp = get_skew_components(tickers)
    put, call, atm = comp["put"], comp["call"], comp["atm"]
    try:
        _persist_skew_history(put, call, atm, get_history(tickers))   # 1y skew + underlying
    except Exception:
        pass

    stale = stale_iv_reasons(atm)                    # skew rides on the ATM surface
    recs, skipped_stale = [], []
    for t in tickers:
        if _excluded(t, SKEW_EXCLUDE_EXTRA):
            continue
        if not all(t in d.columns for d in (put, call, atm)):
            continue
        if t in stale:                               # frozen/dead surface — don't score it
            skipped_stale.append(t)
            continue
        skew_s = ((put[t] - call[t]) / atm[t].replace(0, np.nan)).dropna()
        if skew_s.empty:
            continue
        put_now, call_now, atm_now = _last(put[t]), _last(call[t]), _last(atm[t])
        if not np.isfinite(atm_now) or atm_now == 0:
            continue
        skew_now, z, pctl = _z_and_pctl(skew_s)
        if not np.isfinite(skew_now) or abs(skew_now) > SKEW_CAP:
            continue   # unreliable surface (e.g. low-vol wings far OTM)

        if np.isfinite(z) and z >= Z_FLAG:
            signal, direction = "Rich — sell skew", -1
        elif np.isfinite(z) and z <= -Z_FLAG:
            signal, direction = "Cheap — buy skew", 1
        else:
            signal, direction = "—", 0

        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "put": round(put_now, 1), "call": round(call_now, 1), "atm": round(atm_now, 1),
            "skew": round(skew_now, 3),
            "z": round(z, 2) if np.isfinite(z) else np.nan,
            "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
            "signal": signal, "direction": direction,
        })

    warn_stale("Skew Volatility", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
    df = pd.DataFrame(recs, columns=SKEW_DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    tbl = compute_table()

    # Cache the rich table for the visual report (compute-once-daily, like the rest).
    # (The skew table is computed + cached by the separate `skew` strategy.)
    try:
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        ctx = f"IV {r.iv:.1f} / RV {r.rv:.1f}"
        if pd.notna(r.pctl):
            ctx += f" · spread {_ord(r.pctl)} %ile (1y)"
        if r.rv:
            ctx += f" · IV/RV {r.iv / r.rv:.2f}"
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.z) if pd.notna(r.z) else 0.0,
            "metric_label": "spread z (1y)",
            "level": float(r.spread),          # IV−RV spread, vol points
            "context": ctx,
        })
    return frame(rows)   # already sorted by |z| (most stretched first)
