"""Morning snapshot — pull every Bloomberg input the reports need, ONCE, and cache
it to data/snapshot/ as parquet. Run it in the morning with the Terminal open:

    (PowerShell)  $env:DATAFEED_MODE="bloomberg"; python snapshot.py

Then run the app with DATAFEED_MODE=snapshot and everything (signals, reports,
threshold tweaks, report-code edits) works all day from these files — Terminal
closed. Run with DATAFEED_MODE=mock to snapshot synthetic data (an offline test
of the whole pipeline).

    python snapshot.py --excel out.xlsx   # dump the current snapshot to one .xlsx
    python snapshot.py --oi               # WEEKLY (Mon): capture ONLY the fixed-income OI chains
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.datafeed import (MODE, get_history, get_volume_history, get_implied_vol_history,
                          get_live_quote, get_skew_components, get_term_structure, get_putcall,
                          get_oi_chain, TENOR_LABELS, _PUTCALL_KEYS, _OI_COLS, OI_SNAPSHOT_TICKERS)
from src.universe import INSTRUMENTS

SNAP = Path(__file__).parent / "data" / "snapshot"

# How many forward expiries of each option chain to cache (the report shows ≤10; this gives
# headroom while bounding the file). All strikes are kept — the report windows them at read.
OI_CHAIN_CAPTURE_EXPIRIES = 24

# Snapshot frames, in workbook order (one parquet + one Excel sheet each). The option chain
# (oi_chain) is the odd one out — a tidy LONG grid with a `ticker` column, not date-indexed —
# so it's saved/read directly rather than through _save.
FRAMES = (["prices", "volume", "implied_vol", "skew_put", "skew_call", "skew_atm"]
          + [f"term_{lab.lower()}" for lab in TENOR_LABELS]
          + [f"putcall_{k}" for k in _PUTCALL_KEYS] + ["oi_chain", "live"])


def _save(df: pd.DataFrame, name: str) -> int:
    SNAP.mkdir(parents=True, exist_ok=True)
    df.rename_axis("date").reset_index().to_parquet(SNAP / f"{name}.parquet", index=False)
    return int(df.shape[0])


def _oi_chain_frame(tickers) -> pd.DataFrame:
    """The 11 fixed-income products' listed-option chains (full strike grid, the strip of
    expiries) as one tidy LONG frame with a `ticker` column — the shape _read_oi_snapshot
    expects. Only OI_SNAPSHOT_TICKERS are captured (the Fixed Income book): this is a rates
    tool and pulling more / more often would burn data limits — other products pull their chain
    LIVE on demand. This seam lets the Open Interest report show REAL numbers in snapshot mode."""
    fi = [t for t in tickers if t in OI_SNAPSHOT_TICKERS]
    frames = []
    for t in fi:
        try:
            c = get_oi_chain(t, n_expiries=OI_CHAIN_CAPTURE_EXPIRIES, n_strikes=None)
        except Exception:
            c = None
        if c is not None and not c.empty:
            frames.append(c.assign(ticker=t))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["ticker", *_OI_COLS]))


def _existing_manifest() -> dict:
    p = SNAP / "manifest.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _oi_markets_cached() -> int:
    """How many products the LAST weekly OI capture wrote (read from oi_chain.parquet)."""
    p = SNAP / "oi_chain.parquet"
    if not p.exists():
        return 0
    try:
        d = pd.read_parquet(p)
        return int(d["ticker"].nunique()) if ("ticker" in d.columns and len(d)) else 0
    except Exception:
        return 0


def run_oi() -> int:
    """WEEKLY (Monday) job — capture ONLY the 11 fixed-income option chains and write
    oi_chain.parquet, leaving every other cached input untouched. Run with the Terminal up:
        (PowerShell)  $env:DATAFEED_MODE="bloomberg"; python snapshot.py --oi
    """
    SNAP.mkdir(parents=True, exist_ok=True)
    oi = _oi_chain_frame(list(INSTRUMENTS))
    n = int(oi["ticker"].nunique()) if not oi.empty else 0
    if n == 0:                                          # Bloomberg down / not connected — never WIPE
        print("OI capture returned nothing (is the Terminal/API up?) — kept the existing oi_chain.parquet.")
        return 0
    oi.to_parquet(SNAP / "oi_chain.parquet", index=False)
    m = _existing_manifest()
    m["oi_markets"] = n
    m["oi_as_of"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    (SNAP / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"OI chains captured (weekly): {n} fixed-income products -> {SNAP / 'oi_chain.parquet'}")
    return n


def run() -> dict:
    """Pull the DAILY inputs (LIVE if DATAFEED_MODE=bloomberg) and cache to data/snapshot/.
    Option OI chains are NOT pulled here — they're a separate weekly job (run_oi / --oi)."""
    tickers = list(INSTRUMENTS)
    prices = get_history(tickers)
    volume = get_volume_history(tickers)              # daily contract volume (flag confirmation)
    iv = get_implied_vol_history(tickers)
    skew = get_skew_components(tickers)
    ts = get_term_structure(tickers)
    pc = get_putcall(tickers)                         # options OI & volume, puts vs calls
    live = get_live_quote(tickers)                    # current px vs prior settle (CHG_*_1D)
    live_as_of = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    _save(prices, "prices")
    _save(volume, "volume")
    _save(iv, "implied_vol")
    _save(skew["put"], "skew_put")
    _save(skew["call"], "skew_call")
    _save(skew["atm"], "skew_atm")
    for lab in TENOR_LABELS:
        _save(ts[lab], f"term_{lab.lower()}")
    for k in _PUTCALL_KEYS:
        _save(pc[k], f"putcall_{k}")
    SNAP.mkdir(parents=True, exist_ok=True)
    # Option OI chains are a SEPARATE weekly job (run_oi / --oi); the daily snapshot leaves the
    # last capture in place and carries its count + date forward into the manifest below.
    prev = _existing_manifest()
    live.rename_axis("ticker").reset_index().to_parquet(SNAP / "live.parquet", index=False)  # ticker-indexed

    # Extend the persistent ~10y COT price DB (incremental: only the new dates). This is
    # the Bloomberg-connected moment, so it's where the DB grows; it's a no-op off-Bloomberg.
    try:
        from src import cotdata
        store = cotdata.update_price_store(list(cotdata.COT_MAP))
        print(f"  COT price DB: {store.shape[0]} dates x {store.shape[1]} markets")
        cotdata.compute(force=True)     # rebuild COT history/signals so they ingest the fresh deep prices
        print("  COT signals rebuilt from the price DB")
    except Exception as e:
        print(f"  (COT price DB update skipped: {e})")

    # Equities side — cache the index membership + overnight quotes/history so the Equities pages run
    # off the same daily snapshot (Terminal-closed). Guarded: a failure never blocks the snapshot.
    eq = {}
    try:
        from src import equities
        eq = equities.build_snapshot()
        print(f"  Equities: {eq.get('n_memberships', 0)} constituents / {eq.get('n_unique', 0)} "
              f"unique across {len(eq.get('indices', {}))} indices"
              if eq.get("ok") else f"  Equities snapshot skipped: {eq.get('reason')}")
    except Exception as e:
        print(f"  (Equities snapshot skipped: {e})")

    manifest = {
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": str(prices.index.max().date()) if len(prices) else "",
        "source": MODE,                       # "bloomberg" for a real morning pull
        "n_tickers": len(tickers),
        "price_rows": int(len(prices)),
        "iv_markets": int(iv.notna().any().sum()),
        "oi_markets": _oi_markets_cached(),    # from the last weekly OI capture (run_oi / --oi)
        "oi_as_of": prev.get("oi_as_of", ""),  # when that weekly OI capture last ran
        "live_as_of": live_as_of,             # when the prior-settle->now quote was pulled
        "live_n": int(live["pct"].notna().sum()) if "pct" in live.columns else 0,
        "equities": eq.get("indices", {}) if eq.get("ok") else {},
    }
    SNAP.mkdir(parents=True, exist_ok=True)
    (SNAP / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_excel(out_path: str) -> str:
    """Dump the cached snapshot parquets to one multi-sheet .xlsx for eyeballing."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        wrote = False
        for name in FRAMES:
            p = SNAP / f"{name}.parquet"
            if p.exists():
                pd.read_parquet(p).to_excel(xl, sheet_name=name[:31], index=False)
                wrote = True
        if not wrote:                         # ExcelWriter needs at least one sheet
            pd.DataFrame({"note": ["no snapshot yet — run snapshot.py first"]}).to_excel(
                xl, sheet_name="empty", index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="", help="dump the cached snapshot to this .xlsx and exit")
    ap.add_argument("--oi", action="store_true",
                    help="WEEKLY job: capture ONLY the fixed-income option chains (run Mondays)")
    args = ap.parse_args()
    if args.excel:
        export_excel(args.excel)
        print("Wrote", args.excel)
        return
    if args.oi:
        run_oi()
        return
    m = run()
    print("Snapshot written to", SNAP)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
