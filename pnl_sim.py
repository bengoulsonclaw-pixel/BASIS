"""Per-product scorecard for the MA Crossover strategy — which products does it
actually trade best?

Flat dollar P&L on 10 lots is a poor cross-product comparison: it's dominated by
contract size (point value) and notional, not signal quality. So this leads with
SCALE-FREE, risk-adjusted metrics and a VOL-TARGETED dollar P&L (every product
sized to the same daily $ risk), and keeps the raw 10-lot dollar figure alongside
only to show how misleading it is.

Position = our LIVE MA Crossover rule (50/200 cross + 15-EMA + 3-month return).
Bonds excluded, same as the strategy. Point values + currency from Bloomberg.

NB on roll: prices are the front generic (not roll-adjusted). Risk-adjusted /
relative rankings are robust to that; absolute flat-lot dollars are not — which is
exactly why the comparison below is risk-based, not raw-dollar-based.

Run with the Terminal open:
    $env:DATAFEED_MODE='bloomberg'; .venv\\Scripts\\python.exe pnl_sim.py --years 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xbbg import blp

sys.path.insert(0, str(Path(__file__).parent))
from src import universe
from src.datafeed import MODE, get_history
from src.strategies.ma_crossover import EXCLUDE_ASSETS, POSITION, SWING
from src.universe import TREND_UNIVERSE, name

CONTRACTS = 10
TARGET_DAILY_VOL = 10_000     # USD: size each product to this daily 1σ for the vol-targeted P&L
TRADING_DAYS = 252
CCY_FX = {"EUR": ("EURUSD Curncy", False), "GBP": ("GBPUSD Curncy", False),
          "AUD": ("AUDUSD Curncy", False), "CHF": ("USDCHF Curncy", True),
          "JPY": ("USDJPY Curncy", True), "KRW": ("USDKRW Curncy", True)}


def position(px: pd.Series, cfg) -> pd.Series:
    """Exact live MA Crossover position for a given config: +1 / -1 / 0 per day."""
    fast, slow = px.rolling(cfg.fast).mean(), px.rolling(cfg.slow).mean()
    ema = px.ewm(span=cfg.ema, adjust=False).mean()
    mom = px / px.shift(cfg.mom) - 1.0
    cd = pd.Series(np.where(fast > slow, 1.0, -1.0), index=px.index).where(fast.notna() & slow.notna())
    ema_ok = ((cd > 0) & (ema > fast)) | ((cd < 0) & (ema < fast))
    mom_ok = ((cd > 0) & (mom > 0)) | ((cd < 0) & (mom < 0))
    return cd.where(ema_ok & mom_ok, 0.0).fillna(0.0)


def scorecard(px: pd.Series, pointval: float, usd_fx: float, cfg) -> dict | None:
    pos = position(px, cfg)
    ret = px.pct_change()
    sr = (pos.shift(1) * ret).dropna()
    if sr.empty or sr.std() == 0:
        return None
    sharpe = sr.mean() * TRADING_DAYS / (sr.std() * np.sqrt(TRADING_DAYS))
    total = (1 + sr).prod() - 1

    # per-trade stats (a "trade" = a run of constant non-zero position)
    run = (pos != pos.shift()).cumsum()
    first = pos.groupby(run).first()
    lens = pos.groupby(run).size()
    tret = (pos.shift(1) * ret).groupby(run).apply(lambda x: (1 + x.fillna(0)).prod() - 1)
    keep = first.ne(0) & first.notna()
    tret, lens = tret[keep], lens[keep]
    wins, losses = tret[tret > 0].sum(), -tret[tret < 0].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    hit = (tret > 0).mean() * 100 if len(tret) else float("nan")

    # vol-targeted $ (USD): size to TARGET_DAILY_VOL of daily 1σ, then run the signal
    d_dollar_vol = px.diff().std() * pointval * usd_fx           # USD daily σ per 1 contract
    k = (TARGET_DAILY_VOL / d_dollar_vol) if d_dollar_vol > 0 else 0.0
    vt = float((pos.shift(1) * px.diff() * pointval * usd_fx * k).sum())
    raw10 = float((CONTRACTS * pos.shift(1) * px.diff() * pointval * usd_fx).sum())
    return {"sharpe": sharpe, "total_ret": total * 100, "pf": pf, "hit": hit,
            "trades": int(len(tret)), "avg_hold": float(lens.mean()) if len(lens) else float("nan"),
            "vt_usd": vt, "raw10_usd": raw10}


def spot_fx(currencies, start) -> dict:
    need = sorted({c for c in currencies if c != "USD" and c in CCY_FX})
    out = {"USD": 1.0}
    if need:
        fx = get_history([CCY_FX[c][0] for c in need], start=start, field="PX_LAST")
        for c in need:
            tk, inv = CCY_FX[c]
            v = pd.to_numeric(fx[tk], errors="coerce").dropna().iloc[-1]
            out[c] = (1.0 / v) if inv else v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--variant", choices=["position", "swing"], default="position",
                    help="position = 50/200 (few trades); swing = 20/50 (more trades, better per-product stats)")
    ap.add_argument("--min-trades", type=int, default=0, help="hide products with fewer than N trades")
    args = ap.parse_args()
    cfg = {"position": POSITION, "swing": SWING}[args.variant]

    start = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(int(args.years * TRADING_DAYS))
    products = [t for t in TREND_UNIVERSE if universe.asset(t) not in EXCLUDE_ASSETS]
    print(f"Mode: {MODE} | variant: {cfg.name} ({cfg.fast}/{cfg.slow}) | products: {len(products)} "
          f"(bonds excluded) | ~{args.years:g}y\nPulling…")
    prices = get_history(products, start=start).sort_index()
    products = [t for t in products if t in prices.columns]

    raw = blp.bdp(products, ["FUT_VAL_PT", "CRNCY"]).to_pandas()
    mult = raw.pivot(index="ticker", columns="field", values="value").reindex(products)
    mult.columns = [c.lower() for c in mult.columns]
    mult["fut_val_pt"] = pd.to_numeric(mult["fut_val_pt"], errors="coerce")
    fx = spot_fx(mult["crncy"].dropna().unique(), start)

    rows = []
    for t in products:
        pv, ccy = mult.at[t, "fut_val_pt"], mult.at[t, "crncy"]
        if pd.isna(pv) or ccy not in fx:
            continue
        m = scorecard(prices[t], pv, fx[ccy], cfg)
        if m and m["trades"] >= args.min_trades:
            rows.append({"market": name(t), "asset": universe.asset(t), **m})

    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    span = f"{prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d}"
    print(f"Got {prices.shape[0]} days × {len(res)} products, {span}.\n")

    pd.set_option("display.width", 220)
    fmt = {"sharpe": "{:+.2f}".format, "total_ret": "{:+.0f}%".format, "pf": "{:.2f}".format,
           "hit": "{:.0f}".format, "avg_hold": "{:.0f}".format,
           "vt_usd": "{:+,.0f}".format, "raw10_usd": "{:+,.0f}".format}
    cols = ["market", "asset", "sharpe", "pf", "hit", "trades", "avg_hold", "vt_usd", "raw10_usd"]
    print("Ranked by Sharpe — the products this strategy trades BEST (scale-free).")
    print("  vt_usd = P&L if every product sized to $10k/day risk · raw10_usd = flat 10 lots (distorted)\n")
    print(res[cols].head(15).to_string(index=False, formatters=fmt))
    print("\n…bottom 8:")
    print(res[cols].tail(8).to_string(index=False, formatters=fmt))

    print("\nBy asset class (mean Sharpe):")
    a = res.groupby("asset")["sharpe"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for asset, r in a.iterrows():
        print(f"  {asset:<14} {r['mean']:+.2f}   ({int(r['count'])} products)")

    pos_n = int((res["sharpe"] > 0).sum())
    print(f"\nPositive Sharpe on {pos_n}/{len(res)} products. "
          f"Vol-targeted book total (equal risk each): ${res['vt_usd'].sum():,.0f}.")
    out = Path(__file__).parent / "data" / "signals" / "ma_product_scorecard.csv"
    res.to_csv(out, index=False)
    print(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
