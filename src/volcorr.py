"""Return co-movement — the Volatility page's relative-value peer finder.

For every pair of products we correlate their DAILY LOG RETURNS over the trailing
year — the SAME calculation as the Product Correlations page ('Returns' toggle,
1-year matrix), so the two always agree. Two products whose prices move together
are natural pair candidates: when one screens cheap (buy vol) and a
highly-correlated one screens rich (sell vol), trading the pair is cleaner than
either leg outright — you are long the spread, not a naked vol view.

The matrix is precomputed with the daily signals (it reuses sectorcorr, which needs
the data feed) and cached to data/signals/volcorr.parquet. The app and the PDF
report both READ that cache and apply the (adjustable) correlation threshold at
DISPLAY time, so raising or lowering the bar never needs a re-pull. The heavy import
is lazy inside build_and_cache so this module stays importable (for load()/peers())
from the report subprocess, which has no package context.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SIGNALS = Path(__file__).resolve().parents[1] / "data" / "signals"
MATRIX_FILES = {"returns": _SIGNALS / "volcorr.parquet",        # returns keeps the original file name
                "iv": _SIGNALS / "volcorr_iv.parquet",          # implied-vol changes
                "realized": _SIGNALS / "volcorr_realized.parquet"}   # realized-vol changes
# Trailing 1-MONTH twins — shown next to the 1-year number so a relationship that has
# recently loosened is visible (the "is it currently holding?" check).
MATRIX_FILES_1M = {m: p.with_name(p.stem + "_1m.parquet") for m, p in MATRIX_FILES.items()}


def build_and_cache(tickers=None, asof=None) -> "pd.DataFrame | None":
    """Compute + cache the correlation matrices for every metric (returns / iv /
    realized), each over BOTH windows — trailing 1-year and 1-month — EXACTLY as the
    Product Correlations page computes them (reuses sectorcorr), so all surfaces
    always agree. Runs inside the daily signals build."""
    from .sectorcorr import instrument_changes, LONG, SHORT     # lazy: needs the feed
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    out = {}
    for metric, path in MATRIX_FILES.items():
        try:
            chg, _ = instrument_changes(metric, asof)
            chg = chg.dropna(how="all")
            cm = chg.iloc[-LONG:].corr(min_periods=SHORT * 2)
            cm1m = chg.iloc[-SHORT:].corr(min_periods=int(SHORT * 0.7))
            path.parent.mkdir(parents=True, exist_ok=True)
            cm.to_parquet(path)
            cm1m.to_parquet(MATRIX_FILES_1M[metric])
            out[metric] = cm
        except Exception:
            pass
    return out.get("returns")


def load(metric: str = "returns", window: str = "1y") -> "pd.DataFrame | None":
    """The cached ticker×ticker correlation matrix for `metric` ('returns' / 'iv' /
    'realized') and `window` ('1y' or '1m'), or None if not built yet. Read-only (no
    data-feed dependency) so the report subprocess can use it too."""
    path = (MATRIX_FILES_1M if window == "1m" else MATRIX_FILES).get(metric)
    if path is None or not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def peers(ticker: str, cm, threshold: float, universe=None):
    """(peer_ticker, corr) for peers with corr >= threshold, self excluded, strongest
    first. If `universe` is given, keep only peers in it (e.g. the products actually
    shown on the page)."""
    if cm is None or ticker not in cm.columns:
        return []
    s = cm[ticker].dropna()
    s = s[s.index != ticker]
    s = s[s >= threshold]
    if universe is not None:
        s = s[s.index.isin(set(universe))]
    return list(s.sort_values(ascending=False).items())
