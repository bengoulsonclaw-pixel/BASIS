"""Diagnose how much DAILY history Bloomberg actually serves for each COT market, by
field. Run with the Terminal open + logged in:

    (PowerShell)
    cd C:\\Users\\Ben\\OneDrive\\Desktop\\AI\\strategy-dashboard
    $env:DATAFEED_MODE="bloomberg"; .\\.venv\\Scripts\\python.exe diag_history.py

For each ticker it pulls a 10-year window on PX_SETTLE and PX_LAST and prints the earliest
date + row count returned. Markets that come back shallow on PX_SETTLE but deep on PX_LAST
(or vice-versa) tell us which field/ticker to use for the price database. Paste the output
back and I'll wire the right per-market override.
"""
from __future__ import annotations

import os
os.environ.setdefault("DATAFEED_MODE", "bloomberg")

import pandas as pd
from xbbg import blp

from src.cotdata import COT_MAP
from src.datafeed import _bdh_to_wide

START = (pd.Timestamp.now().normalize() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
END = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")


def depth(tkr: str, fld: str):
    try:
        wide = _bdh_to_wide(blp.bdh(tickers=tkr, flds=fld, start_date=START, end_date=END))
        if wide is None or wide.shape[1] == 0:
            return "(none)", 0
        s = wide[wide.columns[0]].dropna()
        return (str(s.index.min().date()), len(s)) if len(s) else ("(none)", 0)
    except Exception as e:
        return f"ERR:{type(e).__name__}", 0


def gen1(tkr: str) -> str:
    """Guess the 1st-generic form: 'GCA Comdty' -> 'GC1 Comdty', 'C A Comdty' -> 'C 1 Comdty'."""
    gen, key = tkr.rsplit(" ", 1)
    if gen.endswith("A"):
        gen = gen[:-1] + "1"
    return f"{gen} {key}"


from src.universe import name as _nm
print(f"{'ticker':12} {'name':20} | {'SETTLE n':>8} {'LAST n':>7} | {'1st-gen':12} {'SETTLE n':>8}")
print("-" * 82)
for tkr in COT_MAP:
    _, sn = depth(tkr, "PX_SETTLE")
    _, ln = depth(tkr, "PX_LAST")
    g = gen1(tkr)
    _, gn = depth(g, "PX_SETTLE")
    flag = "  <-- ALL shallow" if max(sn, ln, gn) < 750 else (
        "  (gen1 deeper)" if gn > max(sn, ln) + 250 else (
            "  (PX_LAST deeper)" if ln > sn + 250 else ""))
    print(f"{tkr:12} {_nm(tkr)[:20]:20} | {sn:8} {ln:7} | {g:12} {gn:8}{flag}")
print("\n~252 rows = 1yr, ~2500 = 10yr. Flags show where the deeper history is.")
