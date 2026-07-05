"""Probe the LIVE option-chain pull — run with the Bloomberg Terminal open:

    (PowerShell)  $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_oi_chain.py

The morning snapshot pulled everything EXCEPT the option chains (manifest oi_markets: 0),
so `_bloomberg_oi_chain` is returning nothing. The most likely reason: the universe uses
GENERIC continuation tickers (ESA Index, TYA Comdty, …) and `OPT_CHAIN` is empty on a
generic — the options live on the active contract. This prints, per representative ticker,
what each step actually returns so we can wire the correct path. Paste the output back.

Nothing here is cached or written; it's read-only diagnostics.
"""
from __future__ import annotations

import traceback

import pandas as pd

# A spread across asset classes (XPA = the one from the screenshot; TYA = the FI case).
TICKERS = ["XPA Index", "ESA Index", "TYA Comdty", "CLA Comdty", "GCA Comdty"]
CHAIN_FIELDS = ["OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPEN_INT"]


def _show(label, obj):
    print(f"  {label}:")
    try:
        df = obj.to_pandas() if hasattr(obj, "to_pandas") else obj
    except Exception:
        df = obj
    if isinstance(df, pd.DataFrame):
        print(f"    shape={df.shape}  columns={list(df.columns)}")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("    " + df.head(6).to_string().replace("\n", "\n    "))
    else:
        print(f"    {df!r}")


def main():
    from xbbg import blp

    for t in TICKERS:
        print("=" * 78)
        print(t)

        # 1) Sanity: is the generic resolving at all, and what's its active contract?
        try:
            _show("bdp FUT_CUR_GEN_TICKER / NAME / PX_LAST",
                  blp.bdp(t, ["FUT_CUR_GEN_TICKER", "NAME", "PX_LAST"]))
        except Exception:
            print("    FUT_CUR_GEN_TICKER bdp raised:\n" + traceback.format_exc())

        # 2) OPT_CHAIN on the GENERIC ticker (what the code does today).
        active = None
        try:
            ch = blp.bdp(t, "FUT_CUR_GEN_TICKER")
            active = None
            if hasattr(ch, "to_pandas"):
                ch = ch.to_pandas()
            if isinstance(ch, pd.DataFrame) and ch.size:
                active = str(ch.iloc[0, 0])
        except Exception:
            pass
        for label, sec in [("OPT_CHAIN on generic", t),
                           (f"OPT_CHAIN on active ({active})", active)]:
            if not sec:
                continue
            try:
                _show(label, blp.bds(sec, "OPT_CHAIN"))
            except Exception:
                print(f"    {label} raised:\n" + traceback.format_exc())

        # 3) For an index future, options may sit on the cash index / a different root —
        #    try CHAIN_TICKERS too (some securities use that bulk field).
        try:
            _show("bds CHAIN_TICKERS on generic", blp.bds(t, "CHAIN_TICKERS"))
        except Exception:
            print("    CHAIN_TICKERS raised:\n" + traceback.format_exc())

    print("=" * 78)
    print("If OPT_CHAIN is empty on the generic but populated on the active contract, the "
          "fix is to resolve generic -> active before pulling the chain. Paste this output back.")


if __name__ == "__main__":
    main()
