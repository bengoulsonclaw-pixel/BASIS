"""Signal Ledger — every TA signal the hub would have flagged, tracked forward.

Turns the signal cache (src/sigcache.py — per-day raw strategy rows, whole FICC book,
~10y) into an accountability record: for each day each strategy cleared its own hub
trigger, did the market then go the signal's way? Hit rates and σ-normalized moves by
strategy, by product, and for the desk's confluence composite — the "our Ichimoku reads
on SOFR hit 64% over 10 years" numbers.

HOW SIGNALS ARE SELECTED. Exactly the hub's pipeline, run over ALL cached days at once:
`tascore.ta_flagged` re-flags every raw row at the hub's own triggers (it's vectorized —
it never looks at the date column), and the confluence composite re-implements
`tascore.score_products`' harmonic within-axis de-dup VECTORIZED per (date, product)
(verified equal against score_products row-by-row in tests) over the book's saved
confluence set.

HOW OUTCOMES ARE MEASURED. In SIGNAL SPACE, at 5 / 10 / 21 sessions: yields for FI (a
"Long" there means yields UP — the same convention every FI strategy page states), pair
spreads for Mean Reversion (rebuilt from the same signal-space legs the strategy used),
prices elsewhere. hit = the signal-space move went the signal's way; σ-move normalizes
by the product's trailing-21d daily σ × √h so a 2bp SOFR move and a $40 gold move
compare fairly. A signal too recent to evaluate keeps NaN outcomes ("pending" in the
blotter). P&L-in-dollars deliberately does NOT live here — that's the TA Backtester's
job (real point values, exits, costs); the ledger answers the cleaner question "was the
CALL right?", free of exit-rule choices.

`rebuild()` (snapshot compute phase, seconds — pure pandas) writes
data/signal_cache/ledger_outcomes.parquet; the app page aggregates from disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sigcache, tascore, universe

DATA = Path(__file__).resolve().parents[1] / "data"
OUTCOMES_FILE = DATA / "signal_cache" / "ledger_outcomes.parquet"

HORIZONS = (5, 10, 21)     # sessions ahead a signal is judged at
SIGMA_WINDOW = 21          # trailing sessions for the daily-σ normalizer
CONFLUENCE = "Confluence"  # the composite's pseudo-strategy name in the ledger


# ---------------------------------------------------------------------------
# signal-space series — products + Mean Reversion pair spreads
# ---------------------------------------------------------------------------
def _signal_space() -> pd.DataFrame:
    """The signal-space frame outcomes are measured on: every product's series (yields
    for FI) plus one 'A / B' column per universe.PAIRS spread, built from the SAME legs
    Mean Reversion scored on."""
    sig, _ = sigcache.book_frames()
    out = sig.copy()
    for p in universe.PAIRS:
        a, b = p["a"], p["b"]
        if a in sig.columns and b in sig.columns:
            out[f"{a} / {b}"] = (sig[a] / sig[b]) if p["kind"] == "ratio" else (sig[a] - sig[b])
    return out


# ---------------------------------------------------------------------------
# the confluence composite, vectorized (score_products semantics over all days)
# ---------------------------------------------------------------------------
def _confluence_rows(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, product) with the composite's net direction + score, from the
    hub-flagged rows of the confluence set — the same harmonic within-axis de-dup as
    tascore.score_products (strongest in an axis full, next ½, ⅓ …; agreement across
    axes counts in full), computed with groupby.rank instead of a per-day loop."""
    if flagged is None or flagged.empty:
        return pd.DataFrame()
    f = flagged.copy()
    f["strength"] = [tascore.strength(s, m) for s, m in zip(f["strategy"], f["metric"])]
    f["axis"] = f["strategy"].map(tascore.axis_of)
    rank = (f.groupby(["date", "instruments", "axis"])["strength"]
            .rank(ascending=False, method="first"))
    f["contrib"] = f["direction"] * f["strength"] / rank
    g = f.groupby(["date", "instruments"], as_index=False).agg(
        market=("market", "first"), metric=("contrib", "sum"),
        n=("strategy", "nunique"), conviction=("strength", "mean"))
    g["direction"] = np.sign(g["metric"]).astype(int)
    g = g[g["direction"] != 0]
    g["strategy"] = CONFLUENCE
    g["signal"] = np.where(g["direction"] > 0, "Long", "Short")
    g["level"] = np.nan
    return g[["date", "strategy", "market", "instruments", "signal", "direction",
              "metric", "level"]]


# ---------------------------------------------------------------------------
# rebuild — flag everything, measure everything, persist
# ---------------------------------------------------------------------------
def rebuild(log=print) -> pd.DataFrame:
    cov = sigcache.coverage()
    if cov.empty:
        log("  signal ledger: no signal cache on disk yet — run backfill_signals.py")
        return pd.DataFrame()
    lo, hi = cov["date"].min(), cov["date"].max()
    rows = sigcache.rows_for(lo, hi)
    if rows.empty:
        return pd.DataFrame()

    per_strat = tascore.ta_flagged(rows, strategies=sigcache.cacheable_strategies())
    conf_in = tascore.ta_flagged(rows, strategies=tascore.confluence_set("ficc"))
    flagged = pd.concat([per_strat, _confluence_rows(conf_in)], ignore_index=True)
    keep = ["date", "strategy", "market", "instruments", "signal", "direction", "metric", "level"]
    flagged = flagged[[c for c in keep if c in flagged.columns]].copy()
    flagged["date"] = pd.to_datetime(flagged["date"])

    lv = _signal_space()
    # σ=0 happens on ffilled holiday runs / pinned series — a 0-σ normalizer turns a
    # finite move into ±inf and poisons every mean downstream; treat as "no σ available".
    sigma = lv.diff().rolling(SIGMA_WINDOW).std().replace(0.0, np.nan)
    parts = {"entry_level": lv, "sigma": sigma}
    for h in HORIZONS:
        parts[f"chg{h}"] = lv.shift(-h) - lv
    melted = None
    for name_, frame in parts.items():
        m = frame.stack().rename(name_).reset_index()
        m.columns = ["date", "instruments", name_]
        melted = m if melted is None else melted.merge(m, on=["date", "instruments"], how="outer")

    out = flagged.merge(melted, on=["date", "instruments"], how="left")
    for h in HORIZONS:
        out[f"move{h}"] = out[f"chg{h}"] * out["direction"]              # signed: + = went our way
        out[f"sig{h}"] = out[f"move{h}"] / (out["sigma"] * np.sqrt(h))   # in trailing-σ units
        out[f"hit{h}"] = np.where(out[f"move{h}"].notna(), out[f"move{h}"] > 0, np.nan)
        out = out.drop(columns=[f"chg{h}"])
    out["level"] = out["level"].fillna(out["entry_level"])

    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["date", "strategy", "instruments"]).to_parquet(OUTCOMES_FILE, index=False)
    log(f"  signal ledger: {len(out):,} flagged signals evaluated "
        f"({out['date'].min().date()} -> {out['date'].max().date()})")
    return out


def load() -> pd.DataFrame:
    """The evaluated ledger from disk (empty frame if never rebuilt)."""
    if not OUTCOMES_FILE.exists():
        return pd.DataFrame()
    out = pd.read_parquet(OUTCOMES_FILE)
    out["date"] = pd.to_datetime(out["date"])
    return out


# ---------------------------------------------------------------------------
# aggregations for the page
# ---------------------------------------------------------------------------
def league(out: pd.DataFrame, by: str = "strategy") -> pd.DataFrame:
    """Hit-rate league table grouped by `by` ('strategy' | 'market' | 'instruments'):
    n signals, hit % and mean σ-move per horizon, sorted by the 21-session hit rate."""
    if out is None or out.empty:
        return pd.DataFrame()
    aggs = {"n": (by, "size")}
    for h in HORIZONS:
        aggs[f"hit{h}"] = (f"hit{h}", "mean")
        aggs[f"sig{h}"] = (f"sig{h}", "mean")
    g = out.groupby(by, as_index=False).agg(**aggs)
    for h in HORIZONS:
        g[f"hit{h}"] = g[f"hit{h}"] * 100.0
    return g.sort_values(f"hit{HORIZONS[-1]}", ascending=False).reset_index(drop=True)


def heat(out: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """strategy × market hit-% grid (long form: strategy, market, n, hit) at `horizon`."""
    if out is None or out.empty:
        return pd.DataFrame()
    g = (out.groupby(["strategy", "market"], as_index=False)
         .agg(n=("strategy", "size"), hit=(f"hit{horizon}", "mean")))
    g["hit"] = g["hit"] * 100.0
    return g.dropna(subset=["hit"])
