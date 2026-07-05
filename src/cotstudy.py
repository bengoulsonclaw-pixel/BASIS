"""Forward-return study for COT positioning extremes — shared by the PDF report and the
dashboard. Pure pandas with NO project-relative imports, so it can be imported both as
`from src import cotstudy` (app) and as `import cotstudy` (the standalone report script).

For each market it measures how the underlying price behaved AFTER positioning reached a
crowded extreme, versus a baseline of all weeks:

  • episodes_long / episodes_short — independent ENTRIES into the extreme. A multi-week
    crowded run counts ONCE (the week it crosses in), so heavily-overlapping weekly
    observations aren't double-counted — the honest sample size.
  • long_h / short_h — mean forward return (%) over h weeks, measured from each entry.
  • base_h — mean forward return (%) over h weeks across ALL weeks (the benchmark).
    Comparing long_h / short_h to base_h shows whether the extreme actually added anything.
"""
from __future__ import annotations

import pandas as pd

HORIZONS = (4, 13)        # weeks (~1 month / ~1 quarter)
MIN_EPISODES = 3          # markets with >= this many entries (either side) are shown in the report


def _entries(extreme: pd.Series) -> pd.Series:
    """First week of each contiguous run of `extreme` (independent episode entries)."""
    extreme = extreme.fillna(False).astype(bool)
    return extreme & ~extreme.shift(1, fill_value=False)


def forward_return_study(hist: pd.DataFrame, hi: float = 80.0, lo: float = 20.0,
                         horizons=HORIZONS, min_episodes: int = MIN_EPISODES) -> pd.DataFrame:
    recs = []
    for tkr, g in hist.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        price = pd.to_numeric(g.get("price"), errors="coerce")
        ci = pd.to_numeric(g.get("cot_index"), errors="coerce")
        if price.notna().sum() < 30:           # too little aligned price to say anything
            continue
        # Count episodes ONLY within the window we actually hold price for, so n and the
        # return averages are drawn from the SAME span. (Otherwise n could span ~10y of COT
        # positioning while the returns only cover the months of price history we have.)
        priced = price.notna()
        long_entry = _entries(ci >= hi) & priced
        short_entry = _entries(ci <= lo) & priced
        rec = {
            "ticker": tkr, "market": str(g["market"].iloc[-1]), "asset": str(g["asset"].iloc[-1]),
            "episodes_long": int(long_entry.sum()), "episodes_short": int(short_entry.sum()),
            "weeks": int(price.notna().sum()),                       # price-history depth
            "years": round(int(price.notna().sum()) / 52.0, 1),      # ... drawn on for this market
        }
        for h in horizons:
            fwd = price.shift(-h) / price - 1.0
            base = fwd.dropna()
            le = fwd[long_entry & fwd.notna()]
            se = fwd[short_entry & fwd.notna()]
            rec[f"base_{h}"] = float(base.mean()) * 100 if len(base) else float("nan")
            rec[f"long_{h}"] = float(le.mean()) * 100 if len(le) else float("nan")
            rec[f"short_{h}"] = float(se.mean()) * 100 if len(se) else float("nan")
            rec[f"nL_{h}"] = int(len(le))       # entries that have an h-week forward return
            rec[f"nS_{h}"] = int(len(se))
        rec["ok"] = (rec["episodes_long"] >= min_episodes) or (rec["episodes_short"] >= min_episodes)
        recs.append(rec)
    return pd.DataFrame(recs)
