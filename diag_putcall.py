"""Diagnose how much PUT/CALL history Bloomberg serves for each product, by ticker form.

Run with the Terminal OPEN + LOGGED IN (same as a snapshot):

    (PowerShell)
    cd C:\\Users\\Ben\\OneDrive\\Desktop\\AI\\strategy-dashboard
    $env:DATAFEED_MODE="bloomberg"; .\\.venv\\Scripts\\python.exe diag_putcall.py

  ...or just double-click diag_putcall.bat.

For every universe product it pulls a 3-YEAR window of the aggregated option fields
(OPEN_INT_TOTAL_PUT/CALL, VOLUME_TOTAL_PUT/CALL) on three ticker forms and prints how many
days come back for each:
  • active  = the 'A' active-generic the app uses now (e.g. TYA Comdty)
  • gen1    = the 1st-generic (e.g. TY1 Comdty) — the form that fixed the COT *price* history
  • source = the put/call source override, where one exists (CAC/Nikkei -> cash index)

If a shallow product comes back much DEEPER on gen1 (or source), that's the ticker to switch
its put/call pull to. Paste the output back and I'll wire the per-market override.
~252 rows ≈ 1 year, ~756 ≈ 3 years.
"""
from __future__ import annotations

import os
os.environ.setdefault("DATAFEED_MODE", "bloomberg")

import pandas as pd
from xbbg import blp

from src.universe import INSTRUMENTS, name as _nm
from src.datafeed import _bdh_to_wide, PUTCALL_FIELDS, PUTCALL_SOURCE

START = (pd.Timestamp.now().normalize() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
END = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
THIN = 120


def gen1(tkr: str) -> str:
    """1st-generic form: 'TYA Comdty' -> 'TY1 Comdty', 'C A Comdty' -> 'C 1 Comdty'.
    Cash/other tickers not ending in 'A' (SX5E/DAX/UKX Index) are returned unchanged."""
    gen, key = tkr.rsplit(" ", 1)
    if gen.endswith("A"):
        gen = gen[:-1] + "1"
    return f"{gen} {key}"


def _depths(probe_map: dict) -> dict:
    """probe_map = {probe_ticker: original_ticker}. Returns {original: {field_key: day_count}}
    by pulling each field once (batched across all probe tickers in the map)."""
    out = {orig: {} for orig in probe_map.values()}
    probes = sorted(probe_map)
    for key, fld in PUTCALL_FIELDS.items():
        try:
            wide = _bdh_to_wide(blp.bdh(tickers=probes, flds=fld, start_date=START, end_date=END))
        except Exception:
            wide = None
        for p, orig in probe_map.items():
            cnt = int(wide[p].notna().sum()) if (wide is not None and p in wide.columns) else 0
            out[orig][key] = cnt
    return out


def _oi_vol(d: dict) -> tuple[int, int]:
    """Max OI days and max volume days across put/call for one form's result dict."""
    oi = max(d.get("put_oi", 0), d.get("call_oi", 0))
    vol = max(d.get("put_vol", 0), d.get("call_vol", 0))
    return oi, vol


tickers = list(INSTRUMENTS)
print(f"Pulling 3y of put/call fields per ticker form ({START} -> {END}) ... Terminal must be live.\n")

active = _depths({t: t for t in tickers})
gmap = {gen1(t): t for t in tickers if gen1(t) != t}
genone = _depths(gmap)
src = _depths({s: t for t, s in PUTCALL_SOURCE.items()}) if PUTCALL_SOURCE else {}

hdr = f"{'product':24} {'active tkr':12} {'OI':>5} {'VOL':>5} | {'gen1 tkr':12} {'OI':>5} {'VOL':>5} | note"
print(hdr); print("-" * len(hdr))

rows = []
for t in tickers:
    a_oi, a_vol = _oi_vol(active.get(t, {}))
    rows.append((t, max(a_oi, a_vol)))
rows.sort(key=lambda r: r[1])                       # shallowest current depth first

for t, a_best in rows:
    a_oi, a_vol = _oi_vol(active.get(t, {}))
    g1 = gen1(t)
    has_g1 = g1 != t
    g_oi, g_vol = _oi_vol(genone.get(t, {})) if has_g1 else (0, 0)
    g_best = max(g_oi, g_vol)
    note = ""
    if t in src:
        s_oi, s_vol = _oi_vol(src.get(t, {}))
        note = f"source {PUTCALL_SOURCE[t]}: OI {s_oi} VOL {s_vol}"
    elif has_g1 and g_best > a_best + 60:
        note = "<-- gen1 deeper"
    elif a_best < THIN:
        note = "<-- shallow (no deeper form found)" if (not has_g1 or g_best <= a_best + 60) else ""
    g_show = (f"{g1:12} {g_oi:5} {g_vol:5}" if has_g1 else f"{'(n/a)':12} {'':5} {'':5}")
    print(f"{_nm(t)[:24]:24} {t:12} {a_oi:5} {a_vol:5} | {g_show} | {note}")

print(f"\n~252 ≈ 1yr · ~756 ≈ 3yr · OI = open-interest fields, VOL = volume fields.")
print("Lines flagged '<-- gen1 deeper' (or with a deeper 'source') are the ones to switch.")
