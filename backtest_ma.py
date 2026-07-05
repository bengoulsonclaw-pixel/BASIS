"""Parameter sweep / backtest for MA-crossover trigger settings.

Pulls deep settlement history for the trend universe and backtests the candidate
fast/slow MA pairs the literature pairs with different horizons, ranking them by
signal quality across the whole book AND per asset class — so we can see which
"trigger" has actually worked best on this book rather than guessing from an article.

For each pair it runs two rules:
  * raw      — go long when fast>slow, short when fast<slow (the pure crossover)
  * filtered — only hold when the trailing return (the pair's slow window) agrees
               in sign (our "confirm with momentum" approach)

Positions are lagged one day (signal known at close t-1, applied to t's return),
and a per-turn transaction cost is charged so whipsaw-heavy fast pairs are penalised.

Run with the Bloomberg Terminal open for real depth:
    $env:DATAFEED_MODE='bloomberg'; .venv\\Scripts\\python.exe backtest_ma.py --years 10 --pdf
Offline (thin — the snapshot is only ~400 days, too short for 50/200):
    $env:DATAFEED_MODE='snapshot';  .venv\\Scripts\\python.exe backtest_ma.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src import universe
from src.datafeed import MODE, get_history
from src.universe import TREND_UNIVERSE

# (label, fast, slow, kind) — the article's pairs by horizon + our 50/100 midpoint.
CANDIDATES = [
    ("9 / 21 EMA",   9,  21,  "ema"),
    ("12 / 26 EMA",  12, 26,  "ema"),
    ("20 / 50 SMA",  20, 50,  "sma"),
    ("50 / 100 SMA", 50, 100, "sma"),
    ("50 / 200 SMA", 50, 200, "sma"),
]
ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
TRADING_DAYS = 252


def _ma(prices: pd.DataFrame, n: int, kind: str) -> pd.DataFrame:
    return prices.ewm(span=n, adjust=False).mean() if kind == "ema" else prices.rolling(n).mean()


def _signals(prices: pd.DataFrame, fast: int, slow: int, kind: str):
    """(raw, filtered) position frames in {-1,0,+1}. Raw = sign(fast−slow); filtered
    keeps it only when the trailing slow-window return agrees in sign."""
    f, s = _ma(prices, fast, kind), _ma(prices, slow, kind)
    raw = np.sign(f - s).replace(0, np.nan).ffill()
    mom = prices / prices.shift(slow) - 1.0
    return raw, raw.where(np.sign(mom) == raw, 0.0)


def _strat(prices: pd.DataFrame, pos: pd.DataFrame, cost_bps: float):
    """Per-instrument daily strategy return (position lagged a day, cost on turnover)."""
    p = pos.shift(1)
    return p * prices.pct_change() - (cost_bps / 1e4) * p.diff().abs(), p


def _sharpe(strat: pd.DataFrame) -> float:
    port = strat.mean(axis=1, skipna=True).dropna()
    if port.empty or port.std() == 0:
        return float("nan")
    return (port.mean() * TRADING_DAYS) / (port.std() * np.sqrt(TRADING_DAYS))


def _metrics(strat: pd.DataFrame, pos: pd.DataFrame) -> dict:
    """Whole-book metrics: equal-weight across instruments holding a position that day."""
    port = strat.mean(axis=1, skipna=True).dropna()
    if port.empty or port.std() == 0:
        return {}
    ann_ret, ann_vol = port.mean() * TRADING_DAYS, port.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + port).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    flips = pos.diff().abs().fillna(0).gt(0).sum()
    years = pos.notna().sum() / TRADING_DAYS
    trades_yr = (flips / years.replace(0, np.nan)).mean()
    in_mkt = pos.ne(0).where(pos.notna()).mean().mean()
    wins = total = 0
    for t in pos.columns:
        p = pos[t]
        run = (p != p.shift()).cumsum()
        first = p.groupby(run).first()
        tr = strat[t].groupby(run).sum()[first.ne(0) & first.notna()]
        wins += int((tr > 0).sum()); total += int(tr.notna().sum())
    return {"sharpe": ann_ret / ann_vol, "ann_ret": ann_ret * 100, "maxdd": maxdd * 100,
            "trades_yr": trades_yr, "in_mkt": in_mkt * 100,
            "hit": (wins / total * 100) if total else float("nan"), "trades": total}


def run_sweep(prices: pd.DataFrame, cost_bps: float):
    """Returns (book_table, asset_matrix, meta). book_table = one row per pair with the
    filtered metrics + the raw Sharpe; asset_matrix = filtered Sharpe per asset class × pair."""
    groups = {}
    for t in TREND_UNIVERSE:
        if t in prices.columns:
            groups.setdefault(universe.asset(t), []).append(t)
    assets = [a for a in ASSET_ORDER if a in groups] + [a for a in groups if a not in ASSET_ORDER]

    book_rows, matrix = [], {}
    for label, fast, slow, kind in CANDIDATES:
        raw, filt = _signals(prices, fast, slow, kind)
        strat_f, pos_f = _strat(prices, filt, cost_bps)
        strat_r, pos_r = _strat(prices, raw, cost_bps)
        m = _metrics(strat_f, pos_f)
        if m:
            book_rows.append({"Pair": label, "sharpe_filt": m["sharpe"], "sharpe_raw": _sharpe(strat_r),
                              "ann_ret": m["ann_ret"], "maxdd": m["maxdd"], "trades_yr": m["trades_yr"],
                              "in_mkt": m["in_mkt"], "hit": m["hit"], "trades": m["trades"]})
        matrix[label] = {a: _sharpe(strat_f[[c for c in groups[a] if c in strat_f.columns]]) for a in assets}

    book = pd.DataFrame(book_rows).sort_values("sharpe_filt", ascending=False).reset_index(drop=True)
    mat = pd.DataFrame(matrix).reindex(index=assets, columns=[c[0] for c in CANDIDATES])
    n_by_asset = {a: len(groups[a]) for a in assets}
    bh = prices.pct_change().mean(axis=1).dropna()
    meta = {"bh_sharpe": (bh.mean() * TRADING_DAYS) / (bh.std() * np.sqrt(TRADING_DAYS)),
            "n_markets": prices.shape[1], "n_by_asset": n_by_asset, "cost_bps": cost_bps,
            "start": f"{prices.index.min():%Y-%m-%d}", "end": f"{prices.index.max():%Y-%m-%d}",
            "years": (prices.index.max() - prices.index.min()).days / 365.0}
    return book, mat, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--cost-bps", type=float, default=2.0, help="round-turn cost in bps per position change")
    ap.add_argument("--pdf", nargs="?", const="data/MA_Backtest_Report.pdf", default=None,
                    help="also render the branded PDF (optional path)")
    args = ap.parse_args()

    start = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(int(args.years * TRADING_DAYS))
    print(f"Mode: {MODE} | universe: {len(TREND_UNIVERSE)} | requested ~{args.years:g}y from {start:%Y-%m-%d} "
          f"| cost {args.cost_bps:g}bp/turn\nPulling history…")
    prices = get_history(TREND_UNIVERSE, start=start).sort_index()
    book, mat, meta = run_sweep(prices, args.cost_bps)
    print(f"Got {prices.shape[0]} rows × {prices.shape[1]} tickers, {meta['start']} → {meta['end']} "
          f"(~{meta['years']:.1f}y). Buy&hold book Sharpe = {meta['bh_sharpe']:.2f}.\n")

    pd.set_option("display.width", 200)
    show = book.copy()
    show.columns = ["Pair", "Sharpe(filt)", "Sharpe(raw)", "Ann.Ret%", "MaxDD%", "Trades/yr", "In-mkt%", "Hit%", "Trades"]
    print("Ranked by FILTERED Sharpe (how we'd trade it):")
    print(show.round(2).to_string(index=False))
    print("\nFiltered Sharpe by asset class × pair:")
    print(mat.round(2).to_string())

    out_csv = Path(__file__).parent / "data" / "signals" / "ma_backtest.csv"
    book.round(4).to_csv(out_csv, index=False)
    mat.round(4).to_csv(out_csv.with_name("ma_backtest_by_asset.csv"))
    print(f"\nSaved tables → {out_csv} (+ _by_asset.csv)")

    if args.pdf is not None:
        from src.btreport import build_pdf
        out = Path(args.pdf)
        if not out.is_absolute():                       # resolve against the project, not the shell CWD
            out = Path(__file__).parent / out
        out.parent.mkdir(parents=True, exist_ok=True)
        build_pdf(book, mat, meta, out)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
