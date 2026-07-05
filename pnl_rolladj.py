"""Roll-adjusted dollar P&L for the MA Crossover strategy.

Builds a back-adjusted continuous series per product by stitching the actual
futures contracts (FUT_CHAIN) on their LAST_TRADEABLE_DT, so daily P&L reflects
only real within-contract moves — no roll gaps. Then runs the exact live MA
position and books 10 contracts × position × within-contract ΔP × point value,
converted to USD. This is the roll-clean version of pnl_sim's flat-lot figure.

    $env:DATAFEED_MODE='bloomberg'; .venv\\Scripts\\python.exe pnl_rolladj.py --only "CLA Comdty,ESA Index,GCA Comdty"
    (no --only = whole non-bond book)
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
from src.datafeed import _bdh_to_wide, get_history
from src.strategies.ma_crossover import EXCLUDE_ASSETS, POSITION as CFG
from src.universe import TREND_UNIVERSE, name

CONTRACTS, BUFFER = 10, 4          # roll BUFFER business days before a contract's last trade
CCY_FX = {"EUR": ("EURUSD Curncy", False), "GBP": ("GBPUSD Curncy", False),
          "AUD": ("AUDUSD Curncy", False), "CHF": ("USDCHF Curncy", True),
          "JPY": ("USDJPY Curncy", True), "KRW": ("USDKRW Curncy", True)}


def _pd(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def continuous_pnl_pts(root: str, start, end) -> pd.Series | None:
    """Within-held-contract daily price change (points) — the roll-clean P&L driver."""
    try:
        chain = _pd(blp.bds(root, "FUT_CHAIN", INCLUDE_EXPIRED_CONTRACTS="Y", CHAIN_DATE=end.strftime("%Y%m%d")))
        contracts = chain.iloc[:, -1].tolist()
    except Exception:
        return None
    if not contracts:
        return None
    exp = _pd(blp.bdp(contracts, "LAST_TRADEABLE_DT"))
    exp = exp.pivot(index="ticker", columns="field", values="value") if "field" in exp.columns else exp
    exp.columns = [c.lower() for c in exp.columns]
    exp["last_tradeable_dt"] = pd.to_datetime(exp["last_tradeable_dt"], errors="coerce")
    exp = exp.dropna(subset=["last_tradeable_dt"])
    # contracts whose expiry is inside [start, end+1y] cover a 10y hold window
    exp = exp[(exp["last_tradeable_dt"] >= start - pd.Timedelta(days=40)) &
              (exp["last_tradeable_dt"] <= end + pd.Timedelta(days=400))].sort_values("last_tradeable_dt")
    keep = exp.index.tolist()
    if len(keep) < 2:
        return None
    px = _bdh_to_wide(blp.bdh(tickers=keep, flds="PX_SETTLE",
                              start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")))
    if px is None or px.shape[1] < 2:
        return None
    px = px.sort_index().reindex(columns=[c for c in keep if c in px.columns])

    # roll schedule: on each date hold the earliest contract whose last-trade is > date + BUFFER bdays
    roll_by = (exp["last_tradeable_dt"] - pd.tseries.offsets.BDay(BUFFER)).reindex(px.columns)
    held = pd.Series(index=px.index, dtype=object)
    for c in px.columns:                      # later contracts overwrite earlier → earliest valid wins last
        pass
    # earliest-expiry-not-yet-rolled: iterate contracts in expiry order, assign dates not yet assigned
    unassigned = pd.Series(True, index=px.index)
    for c in px.columns:                      # px.columns already in expiry order
        mask = unassigned & (px.index <= roll_by[c])
        held[mask] = c
        unassigned &= ~mask
    held = held.dropna()
    if held.empty:
        return None

    pnl = pd.Series(0.0, index=held.index)
    for c in held.unique():
        m = held == c
        seg = (px[c] - px[c].shift(1))[m]     # within-contract change (no roll gap at handover)
        pnl.loc[m] = seg.values
    return pnl.astype(float)


def position(px: pd.Series) -> pd.Series:
    fast, slow = px.rolling(CFG.fast).mean(), px.rolling(CFG.slow).mean()
    ema = px.ewm(span=CFG.ema, adjust=False).mean()
    mom = px / px.shift(CFG.mom) - 1.0
    cd = pd.Series(np.where(fast > slow, 1.0, -1.0), index=px.index).where(fast.notna() & slow.notna())
    ema_ok = ((cd > 0) & (ema > fast)) | ((cd < 0) & (ema < fast))
    mom_ok = ((cd > 0) & (mom > 0)) | ((cd < 0) & (mom < 0))
    return cd.where(ema_ok & mom_ok, 0.0).fillna(0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    end = pd.Timestamp.today().normalize()
    start = end - pd.tseries.offsets.BDay(int(args.years * 252))

    prods = [t for t in TREND_UNIVERSE if universe.asset(t) not in EXCLUDE_ASSETS]
    if args.only:
        prods = [t.strip() for t in args.only.split(",")]

    front = get_history(prods, start=start).sort_index()      # for point values + the front-generic comparison
    raw = _pd(blp.bdp(prods, ["FUT_VAL_PT", "CRNCY"]))
    mult = raw.pivot(index="ticker", columns="field", values="value").reindex(prods)
    mult.columns = [c.lower() for c in mult.columns]
    mult["fut_val_pt"] = pd.to_numeric(mult["fut_val_pt"], errors="coerce")
    need = sorted({c for c in mult["crncy"].dropna().unique() if c != "USD" and c in CCY_FX})
    fxh = get_history([CCY_FX[c][0] for c in need], start=start, field="PX_LAST") if need else pd.DataFrame()
    fx = {"USD": 1.0}
    for c in need:
        tk, inv = CCY_FX[c]
        v = pd.to_numeric(fxh[tk], errors="coerce").dropna().iloc[-1]
        fx[c] = (1.0 / v) if inv else v

    print(f"{'product':<28}{'roll-adj $':>16}{'front-generic $':>18}  (10 lots)")
    tot_adj = tot_front = 0.0
    for t in prods:
        pv, ccy = mult.at[t, "fut_val_pt"], mult.at[t, "crncy"]
        if pd.isna(pv) or ccy not in fx:
            continue
        pnl_pts = continuous_pnl_pts(t, start, end)
        if pnl_pts is None or pnl_pts.empty:
            print(f"{name(t):<28}{'— no chain':>16}")
            continue
        # back-adjusted continuous price (difference method), anchored to REAL levels
        # (end = current front price) so prices stay positive and the 3-month-return
        # filter is correct — anchoring to +current_price distorts net-decliners.
        cs = pnl_pts.cumsum()
        if cs.dropna().empty:
            continue
        anchor = float(front[t].dropna().iloc[-1]) if (t in front and front[t].notna().any()) else 100.0
        bp = cs + (anchor - cs.dropna().iloc[-1])      # real levels; .dropna() guards trailing NaN
        pos = position(bp)
        adj = float((CONTRACTS * pos.shift(1) * pnl_pts * pv * fx[ccy]).sum())
        # front-generic comparison (the distorted version)
        fpos = position(front[t]) if t in front else pos
        fr = float((CONTRACTS * fpos.shift(1) * front[t].diff() * pv * fx[ccy]).sum()) if t in front else float("nan")
        tot_adj += adj; tot_front += (fr if fr == fr else 0.0)
        print(f"{name(t):<28}{adj:>16,.0f}{fr:>18,.0f}")

    print(f"\n{'TOTAL':<28}{tot_adj:>16,.0f}{tot_front:>18,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
