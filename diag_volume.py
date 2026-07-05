"""Verify the live daily-VOLUME field for the Flag Breakout page.

Probes PX_VOLUME (what datafeed.get_volume_history uses) against a couple of
alternatives across every asset class on the live Terminal, so we can confirm the
right field — and where it's simply absent (e.g. cash indices) — before relying on
it for the flag volume-confirmation. Run with the Terminal open:

    $env:DATAFEED_MODE='bloomberg'; .venv\\Scripts\\python.exe diag_volume.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from src.universe import INSTRUMENTS, asset

ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
FIELDS = ["PX_VOLUME", "VOLUME", "FUT_AGGTE_VOL"]   # active-contract / generic / all-contracts aggregate
PER_ASSET = 3


def sample_tickers() -> list:
    by: dict = {}
    for t in INSTRUMENTS:
        by.setdefault(asset(t), []).append(t)
    order = [a for a in ASSET_ORDER if a in by] + [a for a in by if a not in ASSET_ORDER]
    picks = []
    for a in order:
        picks += by[a][:PER_ASSET]
    return picks


def main() -> int:
    from xbbg import blp
    from src.datafeed import _bdh_to_wide, MODE

    print(f"DATAFEED_MODE = {MODE}")
    if MODE != "bloomberg":
        print("WARN: not in bloomberg mode — set $env:DATAFEED_MODE='bloomberg' first.\n")

    tks = sample_tickers()
    end = pd.Timestamp.today().normalize()
    start = (end - pd.tseries.offsets.BDay(15)).strftime("%Y-%m-%d")
    ends = end.strftime("%Y-%m-%d")

    cov = {f: {} for f in FIELDS}
    zeros = {f: {} for f in FIELDS}
    last = {f: {} for f in FIELDS}
    for f in FIELDS:
        try:
            w = _bdh_to_wide(blp.bdh(tickers=tks, flds=f, start_date=start, end_date=ends))
        except Exception as e:
            print(f"{f}: query failed: {e!r}")
            w = None
        for t in tks:
            s = w[t].dropna() if (w is not None and t in w.columns) else pd.Series(dtype=float)
            cov[f][t] = len(s)
            zeros[f][t] = int((s == 0).sum())
            last[f][t] = float(s.iloc[-1]) if len(s) else float("nan")

    rows = [{"ticker": t, "asset": asset(t), "name": INSTRUMENTS[t][0],
             "PXVOL(n)": cov["PX_VOLUME"][t], "PXVOL=0": zeros["PX_VOLUME"][t],
             "PXVOL last": last["PX_VOLUME"][t], "AGG last": last["FUT_AGGTE_VOL"][t],
             "AGG=0": zeros["FUT_AGGTE_VOL"][t]} for t in tks]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(f"\nProbed {len(tks)} tickers, {start} → {ends} (≤15 sessions each):\n")
    print(df.to_string(index=False))

    print("\nVerdict by asset — PX_VOLUME zero-last issues vs FUT_AGGTE_VOL coverage:")
    order = [a for a in ASSET_ORDER if a in df["asset"].values] + \
            [a for a in df["asset"].unique() if a not in ASSET_ORDER]
    for a in order:
        sub = df[df["asset"] == a]
        pxbad = int((sub["PXVOL last"] == 0).sum())
        aggok = int((sub["AGG last"] > 0).sum())
        print(f"  {a:12s} PX_VOLUME zero-last {pxbad}/{len(sub)}   ·   FUT_AGGTE_VOL ok {aggok}/{len(sub)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
