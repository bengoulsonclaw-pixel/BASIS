"""Daily job: pull data, run every strategy, cache the opportunities.

Schedule this with Windows Task Scheduler to run after settlements. The
dashboard only ever READS the cached file produced here — it never computes
on a button click.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import universe
from src.strategies import (mean_reversion, trend, ma_crossover, ma_crossover_swing,
                            flag_breakout, support_resistance, fibonacci, breakout_retest,
                            momentum, bollinger, carry, volatility, skew, termstructure, cot,
                            putcall, ag_fundamentals)

SIGNALS_DIR = Path(__file__).parent / "data" / "signals"
SIGNALS_FILE = SIGNALS_DIR / "opportunities.parquet"
META_FILE = SIGNALS_DIR / "meta.json"

STRATEGIES = [mean_reversion, trend, ma_crossover, ma_crossover_swing, flag_breakout,
              support_resistance, fibonacci, breakout_retest, momentum, bollinger,
              carry, volatility, skew, termstructure, cot, putcall, ag_fundamentals]


def run() -> pd.DataFrame:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    universe.reload()                                 # pick up any edits to data/universe.json
    frames = [mod.find_opportunities() for mod in STRATEGIES]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_parquet(SIGNALS_FILE, index=False)
    META_FILE.write_text(json.dumps({
        "as_of": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
        "rows": int(len(df)),
    }))
    return df


if __name__ == "__main__":
    out = run()
    print(f"Wrote {len(out)} opportunities to {SIGNALS_FILE}")
