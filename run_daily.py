"""Daily job: pull data, run every strategy, cache the opportunities.

Schedule this with Windows Task Scheduler to run after settlements. The
dashboard only ever READS the cached file produced here — it never computes
on a button click.
"""
from __future__ import annotations

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
    return df


if __name__ == "__main__":
    out = run()
    print(f"Wrote {len(out)} opportunities to {SIGNALS_FILE}")
