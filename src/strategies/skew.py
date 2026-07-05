"""Skew Volatility — option skew monitor (its own dashboard page + client report).

The metric is the normalized skew (90% put − 110% call)/ATM for listed products
(FX uses its OTC 25Δ risk reversal), z-scored over the same 252-day window as the
Volatility strategy (positive = puts richer than calls; z >= +1.5 -> skew rich /
sell, z <= -1.5 -> cheap / buy). The computation
lives in volatility.compute_skew_table() (it shares the z-score machinery); here we
cache it and map it onto the shared dashboard schema. The visual client report is
src/skewreport.py.
"""
from __future__ import annotations

import pandas as pd

from .base import frame
from .volatility import compute_skew_table, SKEW_DETAIL_FILE, _ord

STRATEGY = "Skew Volatility"


def find_opportunities(history=None) -> pd.DataFrame:
    tbl = compute_skew_table()

    # Cache the rich table for the visual report (compute-once-daily, like the rest).
    try:
        SKEW_DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(SKEW_DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        ctx = f"put {r.put:.1f} / call {r.call:.1f} / ATM {r.atm:.1f}"
        if pd.notna(r.pctl):
            ctx += f" · skew {_ord(r.pctl)} %ile (1y)"
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.z) if pd.notna(r.z) else 0.0,
            "metric_label": "skew z (1y)",
            "level": float(r.skew),          # (90% put − 110% call) / ATM  (FX: −RR/ATM)
            "context": ctx,
        })
    return frame(rows)   # already sorted by |z| (most stretched first)
