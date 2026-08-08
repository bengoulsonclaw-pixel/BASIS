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
    # Near-zero σ (ffilled holiday runs, pinned/collapsed series) turns a finite move
    # into a ±hundreds-of-σ artifact that poisons every mean downstream. A σ is only a
    # usable normalizer when it's in the product's NORMAL range — require at least 5% of
    # the column's own median σ (exact zeros fall out with everything else).
    sigma = lv.diff().rolling(SIGMA_WINDOW).std()
    floor = sigma.median() * 0.05
    sigma = sigma.where(sigma.ge(floor, axis=1))
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


def regime_read(out: pd.DataFrame, recent_years: int = 2, horizon: int = 21,
                min_n_past: int = 400, min_n_recent: int = 120) -> dict | None:
    """The page's auto-written "what's working in THIS era vs what worked before" note —
    deterministic prose recomputed from the ledger itself, so when the regime turns the
    paragraph turns with it (Ben's ask 2026-08-08: a talking point he'd otherwise forget).

    Splits the ledger at `recent_years` before its last date, takes 21-session hit rates
    per strategy in each era (min-n gated so thin samples can't lead the narrative), and
    names: the current era's leaders, the prior era's leaders, and the biggest warmers /
    coolers between the two (≥3pp swing, measured in both eras). The composite gets its
    own sentence. Returns {"text": ..., "asof", "cut"} or None while the ledger is too
    short to split into two meaningful eras. Client-safe phrasing: observations with
    dates and sample sizes, never advice."""
    if out is None or out.empty:
        return None
    hcol = f"hit{horizon}"
    lo, hi = out["date"].min(), out["date"].max()
    cut = hi - pd.DateOffset(years=recent_years)
    if cut <= lo + pd.DateOffset(years=1):
        return None                                     # not enough history for a "prior era"
    core = out[out["strategy"] != CONFLUENCE]
    eras = {}
    for key, sub, mn in (("now", core[core["date"] >= cut], min_n_recent),
                         ("past", core[core["date"] < cut], min_n_past)):
        g = sub.groupby("strategy").agg(n=(hcol, "count"), hit=(hcol, "mean"))
        g["hit"] = g["hit"] * 100.0
        eras[key] = g[g["n"] >= mn]
    now, past = eras["now"], eras["past"]
    if now.empty or past.empty:
        return None

    def _fmt(g, names):
        return ", ".join(f"{s} ({g.loc[s, 'hit']:.0f}%, n={int(g.loc[s, 'n']):,})" for s in names)

    lead_now = list(now.sort_values("hit", ascending=False).head(3).index)
    lead_past = list(past.sort_values("hit", ascending=False).head(3).index)
    both = now.join(past, lsuffix="_now", rsuffix="_past", how="inner")
    both["delta"] = both["hit_now"] - both["hit_past"]
    risers = both[both["delta"] >= 3.0].sort_values("delta", ascending=False)
    faders = both[both["delta"] <= -3.0].sort_values("delta")

    y0, y1 = cut.year, hi.year
    text = (f"**Current era ({y0}–{y1}, last {recent_years}y):** the book's most reliable reads "
            f"at {horizon} sessions have been {_fmt(now, lead_now)}. "
            f"**Prior era ({lo.year}–{y0}):** leadership sat with {_fmt(past, lead_past)}.")
    if len(risers):
        text += (" **Warming into this era:** "
                 + ", ".join(f"{s} (+{both.loc[s, 'delta']:.0f}pp vs its prior-era hit rate)"
                             for s in risers.index[:3]) + ".")
    if len(faders):
        text += (" **Cooling:** "
                 + ", ".join(f"{s} ({both.loc[s, 'delta']:.0f}pp)"
                             for s in faders.index[:3]) + ".")
    conf = out[(out["strategy"] == CONFLUENCE) & (out["date"] >= cut)]
    if conf[hcol].notna().sum() >= min_n_recent:
        text += (f" The combined confluence read is hitting {conf[hcol].mean() * 100:.0f}% "
                 f"over the current era.")
    text += (" Signal leadership rotates with the market regime — when this paragraph "
             "changes, that rotation is the story.")
    return {"text": text, "asof": hi, "cut": cut}


def heat(out: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """strategy × market hit-% grid (long form: strategy, market, n, hit) at `horizon`."""
    if out is None or out.empty:
        return pd.DataFrame()
    g = (out.groupby(["strategy", "market"], as_index=False)
         .agg(n=("strategy", "size"), hit=(f"hit{horizon}", "mean")))
    g["hit"] = g["hit"] * 100.0
    return g.dropna(subset=["hit"])
