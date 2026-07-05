"""Calibration / follow-through study for the Flag Breakout detector.

Walks history forward and asks the only question that matters for a continuation
pattern: **after a flag breakout setup, did price actually continue in the pole's
direction?** It reuses the exact detector the app uses (src/strategies/flag_breakout),
so tuning here tunes the live page.

Method (descriptive, not a tradeable backtest):
  * Step through history every --step trading days.
  * At each step, for each product, detect the current best flag (same code as live).
  * Count a **breakout episode** when readiness ≥ a trigger and the product wasn't
    already in a same-direction setup within --cooldown days (so a multi-day run counts
    once — a fresh entry).
  * Measure the **forward return in the breakout direction** over several horizons.
  * Report hit rate (% that continued) and average/median move, split by pattern,
    horizon and volume-confirmation, vs an unconditional baseline (the typical move of
    the same size) — then sweep the trigger so you can see where the edge concentrates.

Run with the Terminal open for real depth (a random-walk demo feed has no genuine
continuation edge, so expect ~50% there — the point offline is that it RUNS):
    $env:DATAFEED_MODE='bloomberg'; .venv\\Scripts\\python.exe backtest_flags.py --years 10
Offline (mock / snapshot):
    .venv\\Scripts\\python.exe backtest_flags.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:                                          # print →/…/≥ on a legacy (cp1252) console too
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.datafeed import MODE, get_history, get_volume_history
from src.universe import TREND_UNIVERSE, asset
from src.strategies import flag_breakout as fb

HORIZONS = [5, 10, 20]            # trading days forward to measure continuation
SENSITIVITY = [60, 70, 80]        # readiness triggers to sweep
ENTRY_MIN = 50                    # record episodes at/above this readiness (≤ smallest sweep trigger)
ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]


def _detect_on(c: np.ndarray, va: np.ndarray | None):
    """Best flag (either polarity) on a close slice — same internals as the live page."""
    if len(c) < fb.MIN_HISTORY:
        return None
    ret = np.diff(c) / c[:-1]
    dv = float(np.std(ret[-fb.VOL_WINDOW:])) if len(ret) >= 2 else float("nan")
    if not np.isfinite(dv) or dv <= 0:
        return None
    cands = [d for d in (fb._best_flag(c, dv, +1, va), fb._best_flag(c, dv, -1, va)) if d is not None]
    return max(cands, key=lambda d: d["proximity"]) if cands else None


def episodes(prices: pd.DataFrame, volume: pd.DataFrame, step: int, cooldown: int) -> pd.DataFrame:
    """One row per fresh breakout episode, with forward returns in the pole direction."""
    recs = []
    for t in TREND_UNIVERSE:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if len(s) < fb.MIN_HISTORY + max(HORIZONS):
            continue
        c = s.to_numpy(dtype=float)
        n = len(c)
        va_full = (volume[t].reindex(s.index).to_numpy(dtype=float)
                   if (not volume.empty and t in volume.columns) else None)
        last_entry = {1: -10 ** 9, -1: -10 ** 9}
        for i in range(fb.MIN_HISTORY, n):
            if (i - fb.MIN_HISTORY) % step:
                continue
            d = _detect_on(c[:i + 1], va_full[:i + 1] if va_full is not None else None)
            if d is None or d["proximity"] < ENTRY_MIN:
                continue
            sign = d["sign"]
            if i - last_entry[sign] < cooldown:
                continue
            last_entry[sign] = i
            rec = {"date": s.index[i], "ticker": t, "asset": asset(t), "sign": sign,
                   "pattern": "Bull" if sign > 0 else "Bear", "readiness": d["proximity"],
                   "vol_confirms": d["vol_confirms"]}
            for h in HORIZONS:
                rec[f"fwd_{h}"] = sign * (c[i + h] / c[i] - 1.0) * 100.0 if i + h < n else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def _agg(sub: pd.DataFrame) -> dict:
    """n / hit% / mean% / median% over the horizon columns of a slice."""
    out = {"n": len(sub)}
    for h in HORIZONS:
        col = sub[f"fwd_{h}"].dropna()
        out[f"hit_{h}"] = float((col > 0).mean() * 100) if len(col) else float("nan")
        out[f"mean_{h}"] = float(col.mean()) if len(col) else float("nan")
        out[f"med_{h}"] = float(col.median()) if len(col) else float("nan")
    return out


def _baseline(prices: pd.DataFrame) -> dict:
    """Unconditional |H-day move| across the book — the 'typical' move for scale."""
    rets = prices.pct_change()
    out = {}
    for h in HORIZONS:
        fwd = (prices.shift(-h) / prices - 1.0).abs().to_numpy().ravel() * 100.0
        fwd = fwd[np.isfinite(fwd)]
        out[h] = float(np.mean(fwd)) if len(fwd) else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=3, help="trading days between detections (raise to speed up)")
    ap.add_argument("--cooldown", type=int, default=20, help="min days between episodes of the same direction")
    args = ap.parse_args()

    start = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(int(args.years * 252))
    print(f"Mode: {MODE} | universe: {len(TREND_UNIVERSE)} | requested ~{args.years:g}y from {start:%Y-%m-%d} "
          f"| step {args.step}d\nPulling history…")
    prices = get_history(TREND_UNIVERSE, start=start).sort_index()
    try:
        volume = get_volume_history(TREND_UNIVERSE, start=start).sort_index()
    except Exception:
        volume = pd.DataFrame()
    print(f"Got {prices.shape[0]} rows × {prices.shape[1]} tickers "
          f"({prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d}). Detecting flags…")

    ep = episodes(prices, volume, args.step, args.cooldown)
    base = _baseline(prices)
    pd.set_option("display.width", 200)

    if ep.empty:
        print("\nNo breakout episodes detected in this window/feed.")
        return 0

    print(f"\n{len(ep)} breakout episodes  ·  {int((ep['sign'] > 0).sum())} bull / "
          f"{int((ep['sign'] < 0).sum())} bear  ·  readiness ≥ {ENTRY_MIN}\n")
    print("Baseline — typical |move| across the book (for scale): "
          + " · ".join(f"{h}d {base[h]:.1f}%" for h in HORIZONS))

    # By pattern × horizon.
    rows = []
    for pat, sub in [("Bull", ep[ep["sign"] > 0]), ("Bear", ep[ep["sign"] < 0]), ("All", ep)]:
        a = _agg(sub)
        for h in HORIZONS:
            rows.append({"Pattern": pat, "Horizon": f"{h}d", "n": a["n"],
                         "Hit%": a[f"hit_{h}"], "Mean%": a[f"mean_{h}"], "Median%": a[f"med_{h}"]})
    print("\nForward continuation by pattern (directional move after the breakout):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # Volume-confirmed vs not (full episode set).
    vc = ep["vol_confirms"].map(lambda v: "confirmed" if v is True else ("no dry-up" if v is False else "no volume"))
    vrows = []
    for label in ["confirmed", "no dry-up", "no volume"]:
        sub = ep[vc == label]
        if sub.empty:
            continue
        a = _agg(sub)
        vrows.append({"Volume": label, "n": a["n"], **{f"Hit{h}d%": a[f"hit_{h}"] for h in HORIZONS},
                      **{f"Mean{h}d%": a[f"mean_{h}"] for h in HORIZONS}})
    if vrows:
        print("\nDoes volume confirmation help? (dry-up through the flag):")
        print(pd.DataFrame(vrows).round(2).to_string(index=False))

    # Trigger sensitivity — where does the edge concentrate?
    srows = []
    for trig in SENSITIVITY:
        sub = ep[ep["readiness"] >= trig]
        a = _agg(sub)
        srows.append({"Trigger": trig, "n": a["n"], **{f"Hit{h}d%": a[f"hit_{h}"] for h in HORIZONS},
                      **{f"Mean{h}d%": a[f"mean_{h}"] for h in HORIZONS}})
    print("\nTrigger sensitivity (episodes with readiness ≥ trigger):")
    print(pd.DataFrame(srows).round(2).to_string(index=False))

    out_csv = Path(__file__).parent / "data" / "signals" / "flag_backtest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    num = ep.select_dtypes("number").columns
    ep.assign(**{c: ep[c].round(4) for c in num}).to_csv(out_csv, index=False)
    print(f"\nSaved {len(ep)} episodes → {out_csv}")
    if MODE == "mock":
        print("NOTE: demo feed is a mean-reverting random walk — no genuine continuation edge is "
              "expected here. Run with DATAFEED_MODE=bloomberg for a real calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
