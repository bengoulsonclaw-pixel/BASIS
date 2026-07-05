"""COT Positioning strategy — managed-money / leveraged-funds net positioning from
the CFTC Commitments of Traders report, surfaced as the **COT Index** (0–100, a
3-year stochastic of the net position). Crowded long (≥80) / crowded short (≤20)
flag a positioning extreme — often read contrarian.

Thin wrapper over `cotdata.compute()` (which fetches the CFTC API + caches the
weekly history and latest cross-section); here we just map the cross-section onto
the shared strategy schema for the dashboard table + main opportunities feed.
"""
from __future__ import annotations

import pandas as pd

from .. import cotdata
from .base import frame

STRATEGY = "COT Reports"


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        detail, _ = cotdata.compute()
    except Exception:
        detail = pd.DataFrame()
    if detail is None or detail.empty:
        return frame([])

    rows = []
    for r in detail.itertuples(index=False):
        pct = f"{r.net_pct_oi:+.0f}% OI" if pd.notna(r.net_pct_oi) else "—"
        ctx = f"{r.category} net {r.net:+,.0f} ({pct}) · L {r.long:,.0f} / S {r.short:,.0f}"
        if pd.notna(r.chg_net):
            ctx += f" · Δwk {r.chg_net:+,.0f}"
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.cot_index) if pd.notna(r.cot_index) else 0.0,
            "metric_label": "COT Index (0–100)",
            "level": float(r.net),
            "context": ctx,
        })
    return frame(rows)   # already sorted most-extreme first by cotdata
