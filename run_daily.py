"""Daily job: pull data, run every strategy, cache the opportunities.

Schedule this with Windows Task Scheduler to run after settlements. The
dashboard only ever READS the cached file produced here — it never computes
on a button click.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src import universe, tascore, specs
from src.strategies import (mean_reversion, trend, ma_crossover, ma_crossover_swing,
                            flag_breakout, support_resistance, fibonacci, breakout_retest,
                            momentum, bollinger, elliott_wave, ichimoku, obv, mfi,
                            donchian, aroon, carry,
                            volatility, skew, termstructure, cot, putcall, ag_fundamentals)

SIGNALS_DIR = Path(__file__).parent / "data" / "signals"
SIGNALS_FILE = SIGNALS_DIR / "opportunities.parquet"
META_FILE = SIGNALS_DIR / "meta.json"

STRATEGIES = [mean_reversion, trend, ma_crossover, ma_crossover_swing, flag_breakout,
              support_resistance, fibonacci, breakout_retest, momentum, bollinger,
              elliott_wave, ichimoku, obv, mfi, donchian, aroon, carry, volatility, skew,
              termstructure, cot, putcall, ag_fundamentals]



# ── Gold-store leg cadence ────────────────────────────────────────────────────
# ALFRED vintages and the permutation study are minutes, not seconds, and neither
# changes intraday. Stamps live beside the store so a fresh checkout runs every
# leg once on its first pull.
_GOLD_STAMPS = Path(__file__).resolve().parent / "data" / "gold_store" / "leg_stamps.json"


def _gold_leg_due(leg: str, every_days: int) -> bool:
    try:
        last = json.loads(_GOLD_STAMPS.read_text(encoding="utf-8")).get(leg)
        if not last:
            return True
        return (dt.date.today() - dt.date.fromisoformat(last)).days >= every_days
    except Exception:
        return True


def _gold_leg_stamp(leg: str) -> None:
    try:
        d = {}
        if _GOLD_STAMPS.exists():
            d = json.loads(_GOLD_STAMPS.read_text(encoding="utf-8"))
        d[leg] = dt.date.today().isoformat()
        _GOLD_STAMPS.parent.mkdir(parents=True, exist_ok=True)
        _GOLD_STAMPS.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def run() -> pd.DataFrame:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    universe.reload()                                 # pick up any edits to data/universe.json
    frames = [mod.find_opportunities() for mod in STRATEGIES]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Fixed income runs on YIELDS in the technical strategies, so relabel those rows' market
    # name to the yield/rate (e.g. "US 10Y Note (yield)", "3M SOFR (rate)") — one chokepoint
    # so every table, chart and report reads in yield terms. Single FI instruments only;
    # pairs and the price/vol/positioning strategies are left as-is.
    if not df.empty and {"strategy", "instruments", "market"} <= set(df.columns):
        _ta = set(tascore.TA_STRATEGIES)
        _fi = df["strategy"].isin(_ta) & df["instruments"].map(universe.is_fixed_income)
        df.loc[_fi, "market"] = df.loc[_fi, "instruments"].map(universe.yield_name)
        if {"signal", "direction"} <= set(df.columns):   # add the futures action ("· sell the bond")
            df.loc[_fi, "signal"] = [specs.fi_action(s, d, tk) for s, d, tk in
                                     zip(df.loc[_fi, "signal"], df.loc[_fi, "direction"],
                                         df.loc[_fi, "instruments"])]
    df.to_parquet(SIGNALS_FILE, index=False)
    META_FILE.write_text(json.dumps({
        "as_of": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
        "rows": int(len(df)),
    }))
    # Hot Sheet: this is the moment the signal cross-section becomes FINAL (the
    # snapshot compute's stamp runs BEFORE this rebuild in the app pull flow, so it
    # sees yesterday's opportunities) — re-stamp today + re-persist the sheet here.
    # Same-day re-stamps replace today only; a failure can never fail the rebuild.
    try:
        from src import hotsheet
        hotsheet.stamp_today(log=print)
    except Exception as e:
        print(f"  (Hot Sheet refresh skipped: {e})")
    # Gold Driver Model: eight external sources and a walk-forward fit, ~40s cold —
    # far too slow to run on page-open, so it lands on disk here (gold_features.parquet
    # + gold_model.json) like seasonality and the Hot Sheet. Never fails the rebuild;
    # a dead source degrades to the last good cache inside golddata itself.
    # Economic-surprise accrual. This is the ONLY store in the repo that cannot be
    # backfilled: the free calendar feed carries the current week alone, and no free
    # source has consensus-forecast history. A week the machine never runs is a
    # permanent hole, so this goes first and never raises.
    try:
        from src import macrosurprise
        r = macrosurprise.refresh()
        print(f"  Surprise accrual: +{r['added']} new, {r['total_stored']} stored "
              f"({r['not_yet_printed']} awaiting print)")
    except Exception as e:
        print(f"  (Surprise accrual skipped: {e})")
    # Gold store. Every leg refreshes here, and each is guarded separately: a dead
    # FRED key must not cost us the day's market and COT observations.
    #
    # Only market+COT ran until the audit. FRED, the IMF reserves and the synthetic
    # pre-2003 real yield were ingested once by hand, so 8 of 28 features quietly
    # froze and were forward-filled forward forever — which is what made the client
    # PDF's fair-value gap 19 business days stale while printing "now stands at".
    for _leg, _fn, _every in (
            ("market",     "ingest_market",              1),
            ("COT",        "ingest_cot",                 1),
            # ALFRED vintages: ~40 series x full revision history. Slow, and nothing
            # in it changes intraday, so it runs weekly unless forced.
            ("FRED",       "ingest_fred",                7),
            ("IMF",        "ingest_central_banks",      30),
            ("real-yield", "ingest_synthetic_real_yield", 30)):
        try:
            from src import goldingest
            if _every > 1 and not _gold_leg_due(_leg, _every):
                continue
            n = getattr(goldingest, _fn)()
            _gold_leg_stamp(_leg)
            print(f"  Gold ingest [{_leg}]: {n} observations")
        except Exception as e:
            print(f"  (Gold ingest [{_leg}] skipped: {e})")
    try:
        from src import goldfeatures
        feats, _ = goldfeatures.build()
        print(f"  Gold features: {feats.shape[1]} x {feats.shape[0]} dates")
    except Exception as e:
        print(f"  (Gold features skipped: {e})")
    # The three artifacts the BASIS page and the weekly PDF actually read. Before
    # the audit the pull rebuilt goldmodel (the superseded look-ahead fit, which
    # nothing client-facing reads) and left these to whenever they were last run
    # by hand.
    try:
        from src import goldsens
        o = goldsens.compute()
        print(f"  Gold sensitivities: R2 {o['r2_macro']:.2f}, "
              f"fair-value gap {o['fair_value_gap_pct']:+.1f}%")
    except Exception as e:
        print(f"  (Gold sensitivities skipped: {e})")
    try:
        from src import goldevents
        # 20k permutation draws is ~3 min. The release calendar moves once a week
        # at most, so this runs weekly like the FRED leg.
        if _gold_leg_due("events", 7):
            ev = goldevents.compute(draws=20000)
            _gold_leg_stamp("events")
            print(f"  Gold event study: {ev['n_trading_days']} days, {ev['sample']}")
    except Exception as e:
        print(f"  (Gold event study skipped: {e})")
    # CVM fund store (🇧🇷 Brazil Funds). Only the current and previous month are
    # re-downloaded — CVM revises those two daily and freezes the rest — so a routine
    # refresh moves ~25MB, not the ~160MB a cold build costs. Guarded like every other
    # leg: dados.cvm.gov.br truncates downloads often enough that a bad afternoon must
    # not cost us the day's signals.
    try:
        from src import cvmfunds
        met = cvmfunds.build()
        print(f"  CVM funds: {len(met):,} share classes, "
              f"{met['gestor'].nunique():,} gestores")
    except Exception as e:
        print(f"  (CVM fund store skipped: {e})")
    return df


if __name__ == "__main__":
    out = run()
    print(f"Wrote {len(out)} opportunities to {SIGNALS_FILE}")
