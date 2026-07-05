"""Vol Term Structure — ATM implied-vol curve monitor (own report + dashboard page).

For every market we read ATM implied vol at 1M/3M/6M/12M and track the curve's
**3M − 1M slope**, z-scored over the same 252-day window: high z = steep contango
(front cheap vs back), low z = backwardation (front bid/rich). |z| >= 1.5 flags a
calendar-spread opportunity (fade the slope). We also carry each tenor's
implied-vs-realized premium (1M↔21d, 3M↔63d, 6M↔126d, 12M↔252d). Bonds are kept
(ATM curve is fine); STIRs and non-US indices are excluded as elsewhere. The visual
client report is src/termreport.py.
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

_LABS = [lab for lab, *_ in TENORS]
DETAIL_COLUMNS = (["market", "ticker", "asset", "region"]
                  + [f"iv_{l.lower()}" for l in _LABS]
                  + ["slope", "ratio", "z", "pctl", "signal", "direction"]
                  + [f"rv_{l.lower()}" for l in _LABS]
                  + [f"vrp_{l.lower()}" for l in _LABS]
                  + [f"iv_sd_{l.lower()}" for l in _LABS] + ["px_dec"])   # daily 1σ move + native dp


def compute_termstructure_table() -> pd.DataFrame:
    """Cross-section: per market the 1M/3M/6M/12M ATM curve, the 3M−1M slope and its
    z-score/percentile, and each tenor's implied-vs-realized premium. Sorted by |z|."""
    tickers = list(INSTRUMENTS)
    ts = get_term_structure(tickers)                       # {label: date×ticker df}
    px = get_history(tickers)                               # underlying settlement — SD + realized
    logret = np.log(px).diff()
    rvw = {win: logret.rolling(win).std() * np.sqrt(252) * 100.0
           for _, _, _, win in TENORS}                     # realized at each matched window

    stale = stale_iv_reasons(ts["1M"])               # gate on the 1M ATM leg
    recs, skipped_stale = [], []
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

        rec = {"market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
               "slope": round(slope_now, 2),
               "ratio": round(ratio, 2) if np.isfinite(ratio) else np.nan,
               "z": round(z, 2) if np.isfinite(z) else np.nan,
               "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
               "signal": signal, "direction": direction, "px_dec": dec}
        for l in _LABS:
            rec[f"iv_{l.lower()}"] = round(iv[l], 1)
            rec[f"iv_sd_{l.lower()}"] = round(S * iv[l] / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan
            rec[f"rv_{l.lower()}"] = round(rv[l], 1) if np.isfinite(rv[l]) else np.nan
            rec[f"vrp_{l.lower()}"] = round(vrp[l], 1) if np.isfinite(vrp[l]) else np.nan
        recs.append(rec)

    warn_stale("Vol Term Structure", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
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
