"""LIVE probe (Bloomberg) for the option-skew surface. Answers two questions:

  A. Which 30DAY_IMPVOL_<M>%MNY_DF moneyness pillars actually return data, per
     asset class?  (validates / refines SKEW_MONEYNESS_GRID — especially the
     near-ATM pillars the low-vol products lean on.)
  B. Does Bloomberg expose the surface BY DELTA directly (a cleaner alternative
     to solving the 25Δ strike + interpolating)?  We try a set of candidate
     delta-parametrized fields and, if available, a BQL surface-by-delta query.

Needs the Terminal logged in.  Run:
    $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_skew_surface.py
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from src.datafeed import _coerce_pd, _bdh_to_wide, _mny_field

# One liquid, vol-bearing future per asset class (+ a couple of low-vol bonds,
# which are the whole reason for vol-scaling).
SAMPLES = {
    "Equity index": "ESA Index",
    "Energy":       "CLA Comdty",
    "Metals":       "GCA Comdty",
    "Agriculture":  "W A Comdty",
    "Bond 10Y":     "TYA Comdty",
    "Bond 2Y":      "TUA Comdty",
    "Bond Bund":    "RXA Comdty",
}

# Superset of moneyness pillars — finer than the production grid so we learn the
# true granularity available (the near-ATM 97.5/102.5/95/105 matter most for bonds).
LADDER = [80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 100.0,
          102.5, 105.0, 107.5, 110.0, 115.0, 120.0]

# Candidate fields that might expose 25Δ vols / risk-reversal directly. Mnemonics
# are best-guesses across the IMPVOL family + RR/BF; the probe reports which resolve.
DELTA_FIELDS = [
    "30DAY_IMPVOL_25DELTA_DF",
    "30DAY_IMPVOL_25.0DELTA_DF",
    "30DAY_CALL_IMPVOL_25DELTA_DF",
    "30DAY_PUT_IMPVOL_25DELTA_DF",
    "30DAY_IMPVOL_25DELTA_CALL_DF",
    "30DAY_IMPVOL_25DELTA_PUT_DF",
    "1MTH_IMPVOL_25DELTA_DF",
    "30DAY_IMPVOL_DELTA_25_DF",
    "25_DELTA_RISK_REVERSAL_1M",
    "1M_25D_RISK_REVERSAL",
    "30DAY_25DELTA_RISK_REVERSAL_DF",
    "30DAY_25DELTA_BUTTERFLY_DF",
]


def bdh_last(blp, tickers, fld, days=10):
    """{ticker: last value or None} for one field, via the VERIFIED bdh path
    (these surface fields are served historically, not by bdp). Reuses production's
    _bdh_to_wide normalization. Returns {'__error__': msg} if the request raised."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.tseries.offsets.BDay(days)
    try:
        wide = _bdh_to_wide(blp.bdh(tickers=list(tickers), flds=fld,
                                    start_date=start, end_date=end))
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}
    out = {}
    for t in tickers:
        if wide is not None and t in wide.columns:
            v = wide[t].dropna()
            out[t] = float(v.iloc[-1]) if len(v) else None
        else:
            out[t] = None
    return out


def main() -> int:
    from xbbg import blp

    print("=" * 74)
    print("A. MONEYNESS LADDER — which 30DAY_IMPVOL_<M>%MNY_DF pillars return data")
    print("   (via bdh — the access path these surface fields actually use)")
    print("=" * 74)
    samples = list(SAMPLES.values())
    header = "  ".join(f"{M:>5g}" for M in LADDER)
    print(f"{'asset':<14}{'ticker':<12}  {header}")
    grid = {M: bdh_last(blp, samples, _mny_field(M)) for M in LADDER}     # one call per pillar
    grid_hits = {M: 0 for M in LADDER}
    for label, tk in SAMPLES.items():
        cells = []
        for M in LADDER:
            v = grid[M].get(tk)
            ok = v is not None and pd.notna(v)
            cells.append(f"{float(v):>5.1f}" if ok else "    ·")
            if ok:
                grid_hits[M] += 1
        print(f"{label:<14}{tk:<12}  " + "  ".join(cells))
    present = [M for M in LADDER if grid_hits[M] == len(SAMPLES)]
    partial = [M for M in LADDER if 0 < grid_hits[M] < len(SAMPLES)]
    print(f"\n  pillars present for ALL samples : {present}")
    print(f"  pillars present for SOME samples: {partial}")
    print("  -> set SKEW_MONEYNESS_GRID to the 'ALL' set (plus any partials you want).")

    print("\n" + "=" * 74)
    print("B. DELTA-PARAMETRIZED SURFACE — is 25Δ exposed directly?")
    print("=" * 74)
    any_delta = False
    dsamples = ["ESA Index", "CLA Comdty", "TYA Comdty"]
    for f in DELTA_FIELDS:
        res = bdh_last(blp, dsamples, f)
        if "__error__" in res:
            print(f"  {f:<32} error: {res['__error__']}")
            continue
        hits = {t: v for t, v in res.items() if v is not None and pd.notna(v)}
        if hits:
            any_delta = True
            print(f"  {f:<32} RESOLVED: " + ", ".join(f"{t}={v:.2f}" for t, v in hits.items()))
        else:
            print(f"  {f:<32} —")
    print(f"\n  delta-field shortcut available: {'YES' if any_delta else 'NO'}")

    print("\n" + "-" * 74)
    print("  BQL surface-by-delta probe")
    if not hasattr(blp, "bql"):
        print("  this xbbg build has no blp.bql — BQL not reachable over the desktop API here.")
    else:
        try:
            q = ("get(impliedVolatility(expiry='30D', strikeReference='DELTA', "
                 "delta=25, optionType='PUT')) for('ESA Index')")
            res = _coerce_pd(blp.bql(q))
            print("  BQL OK — sample:\n", res.head().to_string() if res is not None else "  (empty)")
        except Exception as e:
            print(f"  BQL query failed: {type(e).__name__}: {e}")

    print("\nDONE. Use section A to refine SKEW_MONEYNESS_GRID; if B is YES the listed")
    print("branch can read 25Δ straight from the field instead of interpolating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
