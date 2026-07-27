"""Vol Term Structure — ATM implied-vol curve monitor (own report + dashboard page).

For every market we read ATM implied vol at 1M/3M/6M/12M and track the curve's
**3M − 1M slope**, z-scored over the same 252-day window: high z = steep contango
(front cheap vs back), low z = backwardation (front bid/rich). |z| >= 1.5 flags a
calendar-spread opportunity (fade the slope). We also carry each tenor's
implied-vs-realized premium (1M↔21d, 3M↔63d, 6M↔126d, 12M↔252d). Bonds are kept
(ATM curve is fine); STIRs and non-US indices are excluded as elsewhere. The visual
client report is src/termreport.py.

Since 2026-07-27 the tenors are OUR OWN curve wherever it exists (_own_term_overlay:
src/owncurve.py evaluated at 30/91/182/365d off the same settlement-built fit as the
vol book, coverage-gated per tenor), with the vendor surface as per-date fallback —
built after the vendor's tenor family started publishing hours late. The slope z's
history is effectively vendor until own_term_history deepens (accrues daily; no
backfill yet). volbt deliberately stays on vendor frames for backtest depth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datafeed import (get_term_structure, get_history, TENORS, stale_iv_reasons,
                        warn_stale, price_decimals)
from ..universe import INSTRUMENTS, name, asset, region
from .base import frame
from .volatility import _z_and_pctl, _last, _excluded, _ord, Z_FLAG, _SQRT252

STRATEGY = "Vol Term Structure"
DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "termstructure.parquet"
HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "termstructure_history.parquet"

_LABS = [lab for lab, *_ in TENORS]
DETAIL_COLUMNS = (["market", "ticker", "asset", "region"]
                  + [f"iv_{l.lower()}" for l in _LABS]
                  + ["slope", "ratio", "z", "pctl", "signal", "direction"]
                  + [f"rv_{l.lower()}" for l in _LABS]
                  + [f"vrp_{l.lower()}" for l in _LABS]
                  + [f"iv_sd_{l.lower()}" for l in _LABS] + ["px_dec"]   # daily 1σ move + native dp
                  + ["src", "vs_bbg"])                     # tenor source (own/bbg/otc) + 1M gap

OWN_TERM_FILE = Path(__file__).resolve().parents[2] / "data" / "snapshot" / "own_term_history.parquet"


def _own_term_overlay(ts: dict):
    """Run the term book on OUR curve where it exists. Overlay the settle-dated
    per-tenor values from src/owncurve.py (own_term_history: date/ticker/tenor/vol)
    onto the vendor tenor frames — own-first, vendor-fallback, per date. Own term
    history accrues from 2026-07-27 (no per-tenor backfill yet), so the slope z's
    252-day window stays effectively vendor-based today and converges to ours as the
    file deepens — the same splice semantics the 30d vol book launched with. Returns
    the blended frames + per-ticker {last own date, vendor gap per tenor} metadata."""
    meta: dict = {}
    try:
        h = pd.read_parquet(OWN_TERM_FILE)
    except Exception:
        return ts, meta
    if h.empty:
        return ts, meta
    h = h.assign(date=pd.to_datetime(h["date"]))
    out = {}
    for lab, vend in ts.items():
        sub = h[h["tenor"] == lab]
        if sub.empty:
            out[lab] = vend
            continue
        piv = sub.pivot_table(index="date", columns="ticker", values="vol", aggfunc="last")
        f2 = vend.reindex(vend.index.union(piv.index)).sort_index()
        for t in piv.columns:
            s = piv[t].dropna()
            if s.empty:
                continue
            if t not in f2.columns:
                f2[t] = np.nan
            gap = float("nan")
            if t in vend.columns:
                v_same = vend[t].get(s.index[-1])
                if pd.notna(v_same):
                    gap = float(s.iloc[-1] - v_same)
            f2.loc[s.index, t] = s
            m = meta.setdefault(t, {"last": s.index[-1], "gaps": {}})
            m["last"] = max(m["last"], s.index[-1])
            m["gaps"][lab] = gap
        out[lab] = f2
    return out, meta


def _persist_term_history(ts, px, window: int = 252):
    """Cache ~1y of the 3M−1M slope, its two legs and the underlying price per market
    (long format) for the term report's slope-vs-underlying panels + captions."""
    frames = []
    for t in px.columns:
        if _excluded(t) or not all(t in ts[l].columns for l in ("1M", "3M")):
            continue
        h = pd.DataFrame({"slope": ts["3M"][t] - ts["1M"][t], "price": px[t],
                          "iv1m": ts["1M"][t], "iv3m": ts["3M"][t]}).dropna().iloc[-window:]
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "slope", "price", "iv1m", "iv3m"]
        h["ticker"] = t
        frames.append(h[["date", "ticker", "slope", "price", "iv1m", "iv3m"]])
    if frames:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(HISTORY_FILE, index=False)
        except Exception:
            pass


def compute_termstructure_table() -> pd.DataFrame:
    """Cross-section: per market the 1M/3M/6M/12M ATM curve, the 3M−1M slope and its
    z-score/percentile, and each tenor's implied-vs-realized premium. Sorted by |z|."""
    tickers = list(INSTRUMENTS)
    ts = get_term_structure(tickers)                       # {label: date×ticker df}
    ts, own_meta = _own_term_overlay(ts)                   # own-first, vendor-fallback
    px = get_history(tickers)                               # underlying settlement — SD + realized
    try:
        _persist_term_history(ts, px)                      # 1y slope + underlying for the report
    except Exception:
        pass
    logret = np.log(px).diff()
    rvw = {win: logret.rolling(win).std() * np.sqrt(252) * 100.0
           for _, _, _, win in TENORS}                     # realized at each matched window

    stale = stale_iv_reasons(ts["1M"])               # gate on the (overlaid) 1M ATM leg
    recs, skipped_stale, own_stale = [], [], []
    for t in tickers:
        if _excluded(t):
            continue
        if not all(t in ts[l].columns for l in _LABS):
            continue
        if t in stale:                               # frozen/dead surface — don't score it
            skipped_stale.append(t)
            continue
        iv = {l: _last(ts[l][t]) for l in _LABS}
        if not all(np.isfinite(v) for v in iv.values()):
            continue
        slope_series = (ts["3M"][t] - ts["1M"][t]).dropna()
        if slope_series.empty:
            continue
        slope_now, z, pctl = _z_and_pctl(slope_series)
        ratio = iv["3M"] / iv["1M"] if iv["1M"] else float("nan")

        rv, vrp = {}, {}
        for lab, _, _, win in TENORS:
            r = _last(rvw[win][t]) if t in rvw[win].columns else float("nan")
            rv[lab] = r
            vrp[lab] = iv[lab] - r if np.isfinite(r) else float("nan")

        if np.isfinite(z) and z >= Z_FLAG:
            signal, direction = "Steep — front cheap", 1
        elif np.isfinite(z) and z <= -Z_FLAG:
            signal, direction = "Inverted — front rich", -1
        else:
            signal, direction = "—", 0

        ps = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        S = float(ps.iloc[-1]) if not ps.empty else float("nan")
        dec = price_decimals(ps, asset(t)) if not ps.empty else 1

        # which construction produced today's tenor values (mirrors the vol book):
        # ours where the own curve covers today's 1M leg, OTC for FX, vendor else
        meta = own_meta.get(t)
        one_m = ts["1M"][t].dropna()
        if asset(t) == "FX":
            src = "otc"
        elif meta is not None and len(one_m) and meta["last"] >= one_m.index[-1]:
            src = "own"
        else:
            src = "bbg"
            if meta is not None:
                own_stale.append(name(t))
        vs_bbg = meta["gaps"].get("1M", float("nan")) if (src == "own" and meta) else float("nan")

        rec = {"market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
               "slope": round(slope_now, 2),
               "ratio": round(ratio, 2) if np.isfinite(ratio) else np.nan,
               "z": round(z, 2) if np.isfinite(z) else np.nan,
               "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
               "signal": signal, "direction": direction, "px_dec": dec,
               "src": src,
               "vs_bbg": round(vs_bbg, 1) if np.isfinite(vs_bbg) else np.nan}
        for l in _LABS:
            rec[f"iv_{l.lower()}"] = round(iv[l], 1)
            rec[f"iv_sd_{l.lower()}"] = round(S * iv[l] / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan
            rec[f"rv_{l.lower()}"] = round(rv[l], 1) if np.isfinite(rv[l]) else np.nan
            rec[f"vrp_{l.lower()}"] = round(vrp[l], 1) if np.isfinite(vrp[l]) else np.nan
        recs.append(rec)

    warn_stale("Vol Term Structure", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
    warn_stale("Own term curve", [f"{m} (fell back to vendor tenors)" for m in own_stale])
    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


def find_opportunities(history=None) -> pd.DataFrame:
    tbl = compute_termstructure_table()
    try:
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        curve = " / ".join(f"{l} {getattr(r, 'iv_' + l.lower()):.0f}" for l in _LABS)
        ctx = f"{curve} · slope {r.slope:+.1f}"
        if pd.notna(r.pctl):
            ctx += f" ({_ord(r.pctl)} %ile)"
        if r.direction > 0:
            ctx += " — buy 1M / sell 3M"
        elif r.direction < 0:
            ctx += " — sell 1M / buy 3M"
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.z) if pd.notna(r.z) else 0.0,
            "metric_label": "slope z (1y)",
            "level": float(r.slope),          # 3M − 1M, vol points
            "context": ctx,
        })
    return frame(rows)
