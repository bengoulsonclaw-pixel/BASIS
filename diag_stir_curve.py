# -*- coding: utf-8 -*-
"""Build-our-own constant-maturity STIR ATM vol — LIVE probe + validation.

For each 3M STIR (SOFR / SONIA / Euribor):
  1. Walk the futures chain, keep the front quarterly contracts (H/M/U/Z).
  2. For each quarterly: its own price, its listed option chain, the ATM strike
     (nearest the future), the ATM call + put implied vols -> their MID.
  3. Interpolate linearly in TOTAL VARIANCE to a constant 90-day point.
  4. Compare against Bloomberg's own 3MTH_IMPVOL_100.0%MNY_DF where it exists
     (SOFR / SONIA) — the validation of our maths the desk asked for.

Run with the Terminal up:  .venv\\Scripts\\python.exe diag_stir_curve.py
Prints every input (expiry, dte, strike, call/put IV, future price) so the whole
construction is auditable line by line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from xbbg import blp

from src.datafeed import _coerce_pd

PRODUCTS = {"SFRA Comdty": "3M SOFR", "SFIA Comdty": "3M SONIA", "ERA Comdty": "3M Euribor"}
QUARTERLY = set("HMUZ")          # Mar / Jun / Sep / Dec month codes
N_QUARTERLIES = 4                # enough to bracket 90 days with margin
TARGET_DAYS = 90.0
IV_FIELDS = ["IVOL_MID", "IVOL", "OPT_IMPLIED_VOLATILITY_MID"]   # first that returns wins


def _rows(raw):
    """bds result -> list of dicts with lower-cased keys (xbbg may return narwhals)."""
    pdf = _coerce_pd(raw)
    if pdf is None or len(pdf) == 0:
        return []
    pdf.columns = [str(c).strip().lower().replace(" ", "_") for c in pdf.columns]
    return pdf.to_dict("records")


def _first_col(d, *names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def fut_chain(generic):
    """Front quarterly futures tickers from the generic's chain."""
    out = []
    for r in _rows(blp.bds(generic, "FUT_CHAIN")):
        tk = str(_first_col(r, "security_description", "security", "value") or "").strip()
        if not tk:
            continue
        root = tk.replace(" Comdty", "")
        code = root[-2] if len(root) >= 2 and root[-1].isdigit() else (root[-3] if len(root) >= 3 else "")
        if code in QUARTERLY:
            out.append(tk if tk.endswith("Comdty") else tk + " Comdty")
        if len(out) >= N_QUARTERLIES:
            break
    return out


def bdp_one(ticker, fld):
    try:
        pdf = _coerce_pd(blp.bdp(ticker, fld))
        if pdf is None or pdf.empty:
            return None
        return pdf.iloc[0, -1]
    except Exception:
        return None


def opt_chain(fut):
    """(calls, puts) as {strike: option_ticker} parsed from the option chain."""
    calls, puts = {}, {}
    for r in _rows(blp.bds(fut, "OPT_CHAIN")):
        tk = str(_first_col(r, "security_description", "security", "value") or "").strip()
        # e.g. 'SFRU6C 95.750 Comdty' / 'ERU6P 97.875 Comdty'
        parts = tk.replace(" Comdty", "").split()
        if len(parts) != 2:
            continue
        root, strike_s = parts
        try:
            strike = float(strike_s)
        except ValueError:
            continue
        cp = next((ch for ch in root[::-1] if ch in "CP"), "")
        (calls if cp == "C" else puts if cp == "P" else {})[strike] = tk + (" Comdty" if not tk.endswith("Comdty") else "")
    return calls, puts


def atm_iv(fut):
    """(dte, strike, call_iv, put_iv, fut_px, iv_field) for the quarterly's ATM pair."""
    px = bdp_one(fut, "PX_LAST")
    if px is None:
        return None
    calls, puts = opt_chain(fut)
    both = sorted(set(calls) & set(puts), key=lambda k: abs(k - float(px)))
    if not both:
        return None
    strike = both[0]
    expiry = bdp_one(calls[strike], "OPT_EXPIRE_DT")
    dte = (pd.to_datetime(expiry) - pd.Timestamp.today().normalize()).days if expiry is not None else None
    if not dte or dte <= 0:
        return None
    for fld in IV_FIELDS:
        civ, piv = bdp_one(calls[strike], fld), bdp_one(puts[strike], fld)
        try:
            civ, piv = float(civ), float(piv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(civ) and np.isfinite(piv) and civ > 0 and piv > 0:
            return {"dte": int(dte), "strike": strike, "call": civ, "put": piv,
                    "fut": float(px), "field": fld}
    return None


def const_maturity(points, target=TARGET_DAYS):
    """Linear in TOTAL VARIANCE between the expiries bracketing `target` days."""
    pts = sorted(((p["dte"], (p["call"] + p["put"]) / 2.0) for p in points))
    if not pts:
        return None
    if len(pts) == 1 or target <= pts[0][0]:
        return pts[0][1]                                  # flat before the front expiry
    for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
        if t1 <= target <= t2:
            w1, w2 = v1 ** 2 * t1, v2 ** 2 * t2           # total variance at each pillar
            tv = w1 + (w2 - w1) * (target - t1) / (t2 - t1)
            return float(np.sqrt(tv / target))
    return pts[-1][1]                                     # beyond the back pillar: flat


def main():
    for gen, name in PRODUCTS.items():
        print("=" * 74)
        print(f"{name}  ({gen})")
        futs = fut_chain(gen)
        print("  quarterlies:", futs or "NONE FOUND")
        points = []
        for f in futs:
            got = atm_iv(f)
            if got:
                points.append(got)
                print(f"  {f:16s} dte={got['dte']:4d}  fut={got['fut']:.3f}  K={got['strike']:.3f}"
                      f"  C={got['call']:.2f}  P={got['put']:.2f}  mid={(got['call']+got['put'])/2:.2f}"
                      f"  [{got['field']}]")
            else:
                print(f"  {f:16s} no usable ATM pair")
        ours = const_maturity(points)
        bbg = None
        try:
            w = _coerce_pd(blp.bdp(gen, "3MTH_IMPVOL_100.0%MNY_DF"))
            bbg = float(w.iloc[0, -1]) if w is not None and not w.empty else None
        except Exception:
            pass
        print(f"  -> OUR constant-90d ATM: {ours:.2f}" if ours else "  -> could not build")
        if ours and bbg:
            print(f"     Bloomberg 3MTH point: {bbg:.2f}   diff {ours - bbg:+.2f}")
        elif ours:
            print("     Bloomberg 3MTH point: (not published — no benchmark, as for Euribor)")


if __name__ == "__main__":
    main()
