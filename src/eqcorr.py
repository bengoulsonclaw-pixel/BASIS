"""Single Stock Correlations — engine.

Correlates the DAILY LOG RETURNS of every pair of index-constituent stocks over
two selectable trailing windows (1M / 3M / 6M / 1Y / 5Y of sessions). Window 1
is the SCREEN: only pairs whose window-1 correlation clears a threshold
(default 0.90) survive, and the window-2 and difference matrices are computed
for exactly those companies — the names that traded as one, and whether they
still do. Window 2 − window 1 is the regime map: a strongly negative cell is a
pair whose lockstep has broken down.

Prices are mode-branched like the rest of the Equities side (equities.py):
live Bloomberg on the work PC (each pull also tops up an on-disk long-history
cache that snapshot mode reads), deterministic synthetic prices offline. The
demo prices come from a market + GICS-sector factor model — NOT independent
walks — so the offline matrices show genuine structure and a handful of
same-sector pairs really do clear the 0.90 default.

No Streamlit here — the page drives this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import datafeed, equities, yfin

# label -> trailing sessions (21 per month, 252 per year — same convention as sectorcorr)
PERIODS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252, "5 years": 1260}
PERIOD_ORDER = list(PERIODS)
DEFAULT_P1, DEFAULT_P2 = "1 year", "1 month"
DEFAULT_THRESHOLD = 0.90

_HIST_FILE = Path(__file__).resolve().parents[1] / "data" / "equities" / "corr_history.parquet"


@dataclass
class StockCorrSet:
    """The three matrices behind the page: the window-1 ('screen') correlation on
    the kept companies, the window-2 one on the same companies, and their
    difference — plus the qualifying pairs behind the maps."""
    asof: pd.Timestamp
    p1: str                       # window labels (PERIODS keys)
    p2: str
    threshold: float
    source: str                   # where the prices came from (for the caption)
    n_universe: int               # stocks screened over window 1
    names: dict                   # ticker -> display label (collisions carry the ticker root)
    sectors: dict                 # ticker -> GICS sector
    corr1: pd.DataFrame           # kept x kept, window-1 correlation
    corr2: pd.DataFrame           # same companies, window-2
    diff: pd.DataFrame            # corr2 − corr1 (diagonal blanked)
    pairs: pd.DataFrame           # a, b, sector_a, sector_b, c1, c2, d — sorted by c1 desc
    dropped: list                 # tickers excluded (no prices / too little history)


def _min_obs(n: int) -> int:
    """Non-NaN daily returns a window of n sessions must overlap on before a pair
    correlation is trusted — loose enough for mixed US/EU holiday calendars."""
    return max(15, int(n * 0.6))


# ── demo prices: market + sector factor model ─────────────────────────────────
# Per-ticker loading style (market, sector); the residual weight makes the three
# sum to unit variance. Style 0 is the 'index hugger' — two huggers in the same
# sector correlate ~0.94, so the 0.90 default genuinely lights up offline.
_STYLES = [(0.60, 0.76), (0.52, 0.58), (0.45, 0.42), (0.55, 0.30), (0.36, 0.48)]


def _mock_history(tickers: list, sectors: dict, sessions: int) -> pd.DataFrame:
    days = sessions + 10
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    mkt = np.random.default_rng(equities._seed("EQC-MKT")).normal(0.0, 1.0, days)
    fac = {s: np.random.default_rng(equities._seed("EQC-" + s)).normal(0.0, 1.0, days)
           for s in set(sectors.values())}
    data = {}
    for t in tickers:
        seed = equities._seed(t)
        a, b = _STYLES[seed % len(_STYLES)]
        c = float(np.sqrt(max(1e-6, 1.0 - a * a - b * b)))
        eps = np.random.default_rng(seed ^ 0xC0DE).normal(0.0, 1.0, days)
        sec = fac.get(sectors.get(t, ""), np.zeros(days))
        rets = 0.014 * (a * mkt + b * sec + c * eps)
        data[t] = equities._mock_base(t) * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


# ── Bloomberg (live via xbbg) + the long-history cache ────────────────────────
def _bloomberg_history(tickers: list, sessions: int) -> pd.DataFrame:
    from . import bbg as blp
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(sessions * 7 / 5) + 40)   # sessions -> calendar + slack
    wide = datafeed._bdh_to_wide(blp.bdh(tickers=list(tickers), flds="px_last",
                                         start_date=start.strftime("%Y-%m-%d"),
                                         end_date=end.strftime("%Y-%m-%d")))
    return wide if wide is not None else pd.DataFrame()


def _merge_cache(new: pd.DataFrame) -> None:
    """Fold a live pull into corr_history.parquet (union of dates/tickers, newest
    values win) so snapshot mode keeps whatever depth has ever been pulled. A
    short pull never truncates a long one; failures never break the live path."""
    try:
        old = pd.read_parquet(_HIST_FILE) if _HIST_FILE.exists() else None
    except Exception:
        old = None
    try:
        merged = new if old is None else new.combine_first(old)
        merged = merged.sort_index().iloc[-(max(PERIODS.values()) + 60):]
        _HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        merged.rename_axis("date").to_parquet(_HIST_FILE)
    except Exception:
        pass


def price_history(tickers: list, sectors: dict, sessions: int) -> tuple:
    """(prices date x ticker, source label) covering ~`sessions` trading days back.
    Yahoo Finance first (free, deep history — also tops up the cache), then live
    Bloomberg, then the cache when it covers enough of the window; otherwise
    deterministic demo prices."""
    if equities._use_yf():
        try:
            px = yfin.get_history(tickers, sessions=sessions)
            if px is not None and not px.empty:
                _merge_cache(px)
                return px, "live Yahoo Finance"
        except Exception:
            pass
    if datafeed.MODE == "bloomberg":
        try:
            px = _bloomberg_history(tickers, sessions)
            if px is not None and not px.empty:
                _merge_cache(px)
                return px, "live Bloomberg"
        except Exception:
            pass
    # offline: whatever the last live pull (Yahoo or Bloomberg) left in the cache
    try:
        cached = pd.read_parquet(_HIST_FILE) if _HIST_FILE.exists() else None
    except Exception:
        cached = None
    if cached is not None and len(cached) >= _min_obs(sessions):
        have = [t for t in tickers if t in cached.columns]
        if len(have) >= max(2, len(tickers) // 2):
            px = cached[have].sort_index()
            return px, f"cached pull · to {px.index.max():%Y-%m-%d}"
    return _mock_history(list(tickers), sectors, sessions), "demo (synthetic) data"


# ── the screen ────────────────────────────────────────────────────────────────
def compute(universe: dict, index_keys: list, p1: str, p2: str,
            threshold: float) -> StockCorrSet | None:
    """StockCorrSet for the stocks of `index_keys` (a stock in several indices
    counts once): screen every pair on the window-1 correlation, then re-correlate
    the survivors over window 2. None when there's nothing to correlate."""
    n1, n2 = PERIODS[p1], PERIODS[p2]
    rows: dict = {}
    for k in index_keys:
        for c in universe.get(k, []):
            rows.setdefault(c["ticker"], {"name": c.get("name", c["ticker"]),
                                          "sector": c.get("sector", "—")})
    tickers = sorted(rows)
    if len(tickers) < 2:
        return None
    sectors = {t: rows[t]["sector"] for t in tickers}

    px, source = price_history(tickers, sectors, max(n1, n2))
    px = px.reindex(columns=[t for t in tickers if t in px.columns]).dropna(how="all")
    rets = np.log(px.where(px > 0)).diff()
    if rets.empty:
        return None
    asof = rets.index.max()

    w1 = rets.iloc[-n1:]
    thin = [c for c in rets.columns if w1[c].notna().sum() < _min_obs(n1)]
    live = [c for c in rets.columns if c not in thin]
    dropped = sorted(set(thin) | (set(tickers) - set(rets.columns)))
    if len(live) < 2:
        return None
    corr1_full = w1[live].corr(min_periods=_min_obs(n1))

    # qualifying pairs: upper triangle at/above the threshold over window 1
    upper = corr1_full.where(np.triu(np.ones(corr1_full.shape, dtype=bool), k=1))
    prs = upper.stack()
    prs = prs[prs >= threshold].sort_values(ascending=False)

    # companies behind those pairs, ordered sector -> name so the maps cluster
    kept = sorted({t for ab in prs.index for t in ab},
                  key=lambda t: (sectors.get(t, ""), rows[t]["name"], t))
    # duplicate display names (dual listings — Airbus AIR FP / AIR GY) carry root + exchange
    counts = pd.Series([rows[t]["name"] for t in kept]).value_counts()
    names = {t: (f"{rows[t]['name']} ({' '.join(t.split()[:2])})"
                 if counts.get(rows[t]["name"], 0) > 1 else rows[t]["name"])
             for t in kept}

    if kept:
        corr1 = corr1_full.loc[kept, kept]
        corr2 = rets[kept].iloc[-n2:].corr(min_periods=_min_obs(n2))
        diff = (corr2 - corr1).mask(np.eye(len(kept), dtype=bool))
    else:
        corr1 = corr2 = diff = pd.DataFrame()

    pairs = pd.DataFrame([{
        "a": a, "b": b, "sector_a": sectors.get(a, "—"), "sector_b": sectors.get(b, "—"),
        "c1": float(c1v),
        "c2": float(corr2.loc[a, b]) if pd.notna(corr2.loc[a, b]) else np.nan,
        "d": float(corr2.loc[a, b] - c1v) if pd.notna(corr2.loc[a, b]) else np.nan,
    } for (a, b), c1v in prs.items()]) if kept else pd.DataFrame(
        columns=["a", "b", "sector_a", "sector_b", "c1", "c2", "d"])

    return StockCorrSet(asof=pd.Timestamp(asof), p1=p1, p2=p2, threshold=float(threshold),
                        source=source, n_universe=len(live), names=names, sectors=sectors,
                        corr1=corr1, corr2=corr2, diff=diff, pairs=pairs, dropped=dropped)
