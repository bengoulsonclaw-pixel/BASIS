"""sigaudit.py — BASIS's own TA strategies, held to the gold engine's standard.

Track C. The gold work concluded that no macro driver forecasts gold's direction and
that no model built on them beat simply holding the metal. That conclusion only meant
anything because of the controls around it. This module points the same controls at
the strategies BASIS already runs.

WHAT tabt ALREADY DOES, AND WHAT IT DOES NOT

To be fair to the existing backtester: it is an event-driven trade simulator with
FIXED rules, not a walk-forward model fit. Several of the gold harness's controls do
not apply to it and it is not a criticism that they are absent —

  * purge and embargo guard against a model trained on data adjacent to its test set.
    tabt fits nothing; the strategy parameters are fixed in advance, so there is no
    in-sample leakage to purge.
  * phase-averaging guards against sampling one of h equally valid offsets when
    forward windows overlap. tabt takes DISCRETE trades, so there is no phase to
    choose.

Three gaps are real, and they are the ones that decide whether a P&L number means
anything:

  1. NO BENCHMARK. `_summarize` returns total_pnl, win_rate, profit_factor and
     drawdown, and nothing to compare them against. A strategy that made money in a
     rising market may still have made less than holding the contract. This is
     exactly what the gold work found: every model was profitable in isolation and
     every one of them lost to buy-and-hold.
  2. NO SIGNIFICANCE TEST. `win_rate` is a bare percentage with no interval and no
     null. Sixty trades at 55% and six hundred at 55% are different claims.
  3. NO CORRECTION FOR SELECTION. `compare_strategies` runs 15 strategies plus the
     Confluence composite and returns them SORTED BY P&L. Reading the top row is
     picking the best of sixteen, which is the most reliable way there is to find a
     pattern in noise.

HOW EACH GAP IS CLOSED

  Benchmark — buy-and-hold over the identical window, in the identical units, taken
  from tabt's own `_setup` so the point value, size multiplier and FX conversion are
  the ones the strategy P&L already used. Comparing against a hand-built benchmark
  with different point values would be worse than no benchmark at all.

  Significance — the comparison is made SCALE-FREE and reduced to a binary: did this
  strategy beat buy-and-hold on this contract, yes or no. Summing P&L across a
  universe priced in five currencies with different point values is meaningless, and
  the binary avoids it entirely. Across N contracts, a strategy with no edge beats
  buy-and-hold about half the time, so the count is Binomial(N, 0.5) under the null —
  an exact test, no distributional assumption.

  Selection — Holm across the sixteen variants, and the number of variants tested is
  carried into the output so a winner can never be quoted without it.

WHAT THIS CANNOT SETTLE

Beating buy-and-hold is not the only reason to run a strategy: a signal with lower
drawdown, or one that is flat when you want to be flat, has value this test does not
score. Time in market is reported alongside so that a strategy which is out of the
market most of the time is not judged as though it were fully invested. What the test
does settle is the specific claim a P&L column invites — that running the strategy
beat not running it.
"""
from __future__ import annotations

import sys
from datetime import date
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_START = date(2016, 1, 1)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def binom_p_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p. No normal approximation — n is small here."""
    if n <= 0:
        return float("nan")
    probs = [comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= obs * (1 + 1e-12))))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson interval — behaves at the extremes where the normal one does not."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def holm(pvals: dict) -> dict:
    """Holm step-down. Same implementation the gold event study uses."""
    items = sorted(pvals.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        if np.isnan(p):
            out[k] = float("nan")
            continue
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------
def buy_and_hold(scope: str, ticker: str, start: date, end: date,
                 size: float = 1.0) -> dict:
    """Long the contract for the whole window, in tabt's own units.

    Uses tabt._setup so the price series, size multiplier and FX conversion are
    IDENTICAL to the ones the strategy P&L was computed with. The point-value table
    is quote-denominated (cents for HG/XB/HO, and the FX contracts divide), and
    rebuilding it here would reintroduce the 100-to-10,000x errors that table exists
    to prevent.

    THE SERIES IS ROLL-ADJUSTED, AND THAT IS THE POINT. On the deep panama-adjusted
    store, gold runs 1875.60 -> 2315.61 across 2020-2026: +440 points, where SPOT gold
    rose about +2,966. The gap is 6.6 years of accumulated roll cost in a contango
    market, and it is real money. Benchmarking against spot would credit buy-and-hold
    with $296,600 on a position nobody could actually have held without rolling, make
    the benchmark 6.7x too hard, and mark every strategy down against a fiction. Both
    the strategy and the benchmark trade the same rolled contract and pay the same
    carry, which is the only comparison that means anything.

    (I initially took the 44,000 figure for a point-value bug, because it is 6.7x
    smaller than the spot arithmetic suggests. It is not a bug. The check is recorded
    because the number looks wrong and is right.)
    """
    from src import tabt
    from src.strategies import trend  # noqa: F401  (import side-effects in tabt)

    strategies = tabt.strategies_for_scope(scope)[:1]
    _sig, pnl_hist, _vol, days, size_mult, ccy, _sign, warnings = tabt._setup(
        scope, ticker, strategies, start, end, "reversal", None)
    if pnl_hist is None or not len(days):
        return {"pnl": float("nan"), "ccy": ccy, "n_days": 0,
                "warnings": list(warnings or [])}
    px = pnl_hist.loc[days, ticker].dropna()
    if len(px) < 2:
        return {"pnl": float("nan"), "ccy": ccy, "n_days": int(len(px)),
                "warnings": list(warnings or [])}
    pnl = float((px.iloc[-1] - px.iloc[0]) * size_mult * size)
    curve = (px - px.iloc[0]) * size_mult * size
    return {"pnl": pnl, "ccy": ccy, "n_days": int(len(px)),
            "max_drawdown": float((curve.cummax() - curve).max()),
            "first": float(px.iloc[0]), "last": float(px.iloc[-1]),
            "warnings": list(warnings or [])}


# ---------------------------------------------------------------------------
# per contract
# ---------------------------------------------------------------------------
def audit_ticker(scope: str, ticker: str, start: date = DEFAULT_START,
                 end: date | None = None, **kw) -> pd.DataFrame:
    """Every strategy on one contract, with the benchmark attached.

    Returns one row per variant. `beats_bh` is the binary the universe-level test
    aggregates; `excess_pnl` is kept for reading but never summed across contracts,
    because they are not in the same currency or point value.
    """
    from src import tabt
    end = end or date.today()
    res = tabt.compare_strategies(scope, ticker, start=start, end=end, **kw)
    bh = buy_and_hold(scope, ticker, start, end, size=kw.get("size", 1.0))
    res = res.copy()
    res["ticker"] = ticker
    res["bh_pnl"] = bh["pnl"]
    res["excess_pnl"] = res["total_pnl"] - bh["pnl"]
    res["beats_bh"] = res["total_pnl"] > bh["pnl"]
    res["n_variants_tested"] = len(res)
    res["bh_drawdown"] = bh.get("max_drawdown", np.nan)
    return res


# ---------------------------------------------------------------------------
# across the universe — where the power is
# ---------------------------------------------------------------------------
def stratified_sample(per_class: int = 3, seed: int = 20260823) -> list:
    """A few contracts from every asset class, rather than the first N alphabetically.

    ~12 minutes per contract means the full 89 cannot be run here, and an unstratified
    head-of-list sample would be mostly Indices and FX (39 of 89) — a result driven by
    two asset classes reported as though it covered the book. Stratifying makes the
    coverage explicit and the shortfall honest.
    """
    from src import universe
    rows = pd.DataFrame(universe.load_rows())
    rng = np.random.default_rng(seed)
    picked = []
    for cls, g in rows.groupby("asset"):
        t = list(g["ticker"])
        take = min(per_class, len(t))
        picked += list(rng.choice(t, take, replace=False))
    return picked


def audit_universe(scope: str, tickers, start: date = DEFAULT_START,
                   end: date | None = None, progress=print,
                   checkpoint: Path | None = None, **kw) -> pd.DataFrame:
    """audit_ticker across many contracts, stacked.

    Writes after every contract when `checkpoint` is given. A run of this length that
    has to be abandoned should leave usable partial results rather than nothing, and
    a partial result with its contract count attached is still a real measurement —
    it just has less power, which the binomial test accounts for by itself.
    """
    frames = []
    for i, t in enumerate(tickers, 1):
        try:
            frames.append(audit_ticker(scope, t, start, end, **kw))
            if progress:
                progress(f"  [{i}/{len(tickers)}] {t}")
        except Exception as e:
            if progress:
                progress(f"  [{i}/{len(tickers)}] {t} FAILED: {type(e).__name__}: {e}")
        if checkpoint and frames:
            try:
                pd.concat(frames, ignore_index=True).to_parquet(checkpoint, index=False)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def strategy_scorecard(stacked: pd.DataFrame) -> pd.DataFrame:
    """The headline: per strategy, on how many contracts did it beat buy-and-hold?

    Under a null of no skill a strategy beats the benchmark on about half the
    contracts, so the count is Binomial(N, 0.5) — exact, and it needs no assumption
    about the distribution of P&L, which is heavy-tailed and differently scaled on
    every contract.

    Holm-corrected across the strategies compared, because they were compared.
    """
    if stacked.empty:
        return pd.DataFrame()
    rows = []
    for strat, g in stacked.groupby("strategy"):
        n = int(g["beats_bh"].notna().sum())
        k = int(g["beats_bh"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "strategy": strat, "contracts": n, "beat_bh": k,
            "share": k / n if n else np.nan,
            "ci_low": lo, "ci_high": hi,
            "p_raw": binom_p_two_sided(k, n),
            "median_trades": float(g["n_trades"].median()),
            "median_win_rate": float(g["win_rate"].median()),
        })
    out = pd.DataFrame(rows)
    adj = holm({r.strategy: r.p_raw for r in out.itertuples()})
    out["p_holm"] = out["strategy"].map(adj)
    out["survives"] = (out["p_holm"] < 0.05) & (out["share"] > 0.5)
    return out.sort_values("share", ascending=False).reset_index(drop=True)


def verdict(card: pd.DataFrame) -> str:
    """Derived from the table, never asserted alongside it."""
    if card.empty:
        return "no results"
    n = int(card["contracts"].max())
    win = card[card["survives"]]
    if win.empty:
        best = card.iloc[0]
        return (f"No strategy beats buy-and-hold on more contracts than chance would "
                f"give. The best, {best['strategy']}, wins on {best['beat_bh']} of "
                f"{best['contracts']} ({best['share']:.0%}, Holm p={best['p_holm']:.2f}) "
                f"— and {len(card)} strategies were compared, so the top of a ranked "
                f"list is where the luckiest one lands whether or not any has an edge")
    names = ", ".join(f"{r.strategy} ({r.share:.0%})" for r in win.itertuples())
    return (f"{len(win)} of {len(card)} strategies beat buy-and-hold on more than half "
            f"of {n} contracts after Holm correction: {names}")


def main() -> int:
    from src import universe
    scope = "FICC"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        scope = args[0]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])

    rows = pd.DataFrame(universe.load_rows())
    tickers = list(rows["ticker"])
    if limit:
        tickers = tickers[:limit]
    print(f"Auditing {len(tickers)} contracts x 16 variants from {DEFAULT_START}\n")
    stacked = audit_universe(scope, tickers)
    if stacked.empty:
        print("no results")
        return 1
    out = ROOT / "data" / "gold_store" / "sigaudit_ficc.parquet"
    stacked.to_parquet(out, index=False)

    card = strategy_scorecard(stacked)
    print("\nBeat buy-and-hold, by strategy:\n")
    show = card[["strategy", "contracts", "beat_bh", "share", "ci_low", "ci_high",
                 "p_raw", "p_holm", "survives"]]
    print(show.to_string(index=False,
                         formatters={"share": "{:.0%}".format,
                                     "ci_low": "{:.2f}".format,
                                     "ci_high": "{:.2f}".format,
                                     "p_raw": "{:.4f}".format,
                                     "p_holm": "{:.4f}".format}))
    print(f"\n  {verdict(card)}.")
    print(f"\n  written to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
