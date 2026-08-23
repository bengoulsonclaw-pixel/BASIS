"""goldsens.py — the explanatory engine, on the point-in-time store.

This is the part of the gold work that actually holds up. Seven milestones established
that gold's DIRECTION is not forecastable from these drivers at any horizon we can
test. The same work established that the drivers EXPLAIN gold's moves well, and that
is a different and useful thing:

    sensitivities   gold % per unit move in each driver, fitted jointly
    scenario        run a view on the drivers, get the implied gold move
    fair_value      cumulative move NOT explained by the drivers — the official-bid
                    and debasement residual, made visible
    attribution     decompose the month just gone into per-driver pieces

Why this exists rather than src/goldmodel.py
---------------------------------------------
`goldmodel.py` has the same four functions and was written first, but it reads
`golddata`'s forward-filled frame, which carries no publication timestamps. Every
number it produces is computed on data that includes figures the market had not yet
seen — the identical flaw that inflated the Stage 3 headline until the audit caught
it. Anything client-facing has to come off `goldfeatures`, which reads the
point-in-time store. So this is a port, not a duplicate, and goldmodel is superseded.

EXOGENOUS DRIVERS ONLY
----------------------
ETF tonnage and managed-money positioning are excluded from the fit even though they
raise R² substantially. They are co-movers: tonnage rises BECAUSE gold rallied at
least as much as the reverse, and including them drags the 10y real-yield coefficient
from about -0.51% per 10bp to -0.12% — the flow variables absorb the very effect the
model is trying to price. They belong in a description of a move, never in a scenario.

CLI:  python src/goldsens.py [--horizon 21] [--scenario real=-0.20,dxy=-0.02]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import goldfeatures  # noqa: E402
from src.goldmodels import newey_west_se  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
OUT_FILE = STORE_DIR / "sensitivities.json"

DEFAULT_H = 21          # ~one month of trading days
RIDGE_LAMBDA = 3.0

# Exogenous only — see the module docstring.
MACRO_DRIVERS = ["real_yield_10y_chg_20d", "breakeven_10y_chg_20d",
                 "fed_cut_odds_chg_20d", "dxy_chg_20d", "hy_spread_chg_20d",
                 "vix_z_1y"]
FLOW_DRIVERS = ["etf_tonnage_chg_4w", "cot_mm_net_chg_4w"]

# feature -> (label, unit step, unit text, whether a rise is usually good for gold)
UNITS = {
    "real_yield_10y_chg_20d": ("10y real yield", 0.10, "+10bp"),
    "breakeven_10y_chg_20d":  ("10y breakeven", 0.10, "+10bp"),
    "fed_cut_odds_chg_20d":   ("Cuts priced (2y less funds)", 0.10, "+10bp"),
    "dxy_chg_20d":            ("Dollar (DXY)", 0.01, "+1%"),
    "hy_spread_chg_20d":      ("Baa credit spread", 0.10, "+10bp"),
    "vix_z_1y":               ("VIX (vs 1y range)", 1.0, "+1 sd"),
    "etf_tonnage_chg_4w":     ("SPDR GLD tonnes", 0.01, "+1%"),
    "cot_mm_net_chg_4w":      ("Managed-money net", 10000.0, "+10k lots"),
}


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def trailing_return(price: pd.Series, h: int) -> pd.Series:
    return np.log(price / price.shift(h))


def fit(feats: pd.DataFrame, targets: pd.DataFrame, h: int = DEFAULT_H,
        cols: list | None = None, lam: float = RIDGE_LAMBDA) -> dict:
    """Ridge of the h-day gold return on the SAME period's driver changes.

    A same-period relationship, not a forecast. Standard errors are Newey-West at lag
    h, because the residuals of an overlapping h-day regression are autocorrelated by
    construction and plain OLS errors understate them by roughly sqrt(h)."""
    cols = cols or [c for c in MACRO_DRIVERS if c in feats.columns]
    px = targets["realised_vol_60d"]        # index carrier only
    y = targets["fwd_ret_5d"]               # placeholder, replaced below
    # the h-day TRAILING return is the left-hand side; rebuild it from the target
    # price the feature layer already used, so the two cannot drift apart
    from src import goldstore
    panel = goldstore.daily_panel([goldfeatures.TARGET_PRICE],
                                  start=str(feats.index.min().date()))
    price = panel[goldfeatures.TARGET_PRICE].reindex(feats.index).ffill()
    y = trailing_return(price, h).rename("y")

    d = pd.concat([feats[cols], y], axis=1).dropna()
    if len(d) < 250:
        return {}
    mu, sd = d[cols].mean(), d[cols].std().replace(0, np.nan)
    Z = ((d[cols] - mu) / sd).fillna(0.0)
    ymu = float(d["y"].mean())
    beta = pd.Series(_ridge(Z.to_numpy(), (d["y"] - ymu).to_numpy(), lam), index=cols)

    fitted = Z.to_numpy() @ beta.to_numpy() + ymu
    resid = pd.Series(d["y"].to_numpy() - fitted, index=d.index)
    r2 = float(1 - resid.var() / d["y"].var())
    se = pd.Series(newey_west_se(Z.to_numpy(), resid.to_numpy(), lags=h), index=cols)
    return {"cols": cols, "beta": beta, "se": se, "mu": mu, "sd": sd, "ymu": ymu,
            "r2": r2, "resid": resid, "n": int(len(d)), "h": h,
            "resid_sd": float(resid.std()), "price": price}


def sensitivities(f: dict) -> pd.DataFrame:
    """Gold % per unit move in each driver — the desk-usable form."""
    rows = []
    for c in f["cols"]:
        label, step, unit = UNITS.get(c, (c, 1.0, "+1 unit"))
        per_nat = float(f["beta"][c] / f["sd"][c])
        se = float(f["se"][c])
        rows.append({"driver": label, "move": unit, "feature": c,
                     "gold_pct": float(np.expm1(per_nat * step) * 100),
                     "t_stat": float(f["beta"][c] / se) if se else np.nan})
    out = pd.DataFrame(rows)
    return out.reindex(out["gold_pct"].abs().sort_values(ascending=False).index)


def scenario(f: dict, moves: dict) -> dict:
    """Run a view on the drivers through the fitted sensitivities.

    `moves` is in NATURAL units: {'real_yield_10y_chg_20d': -0.20} = real yields fall
    20bp over the window. Unspecified drivers are held at ZERO change, not at their
    sample mean — a scenario should state what it assumes rather than smuggle in a
    drift."""
    total, parts = 0.0, {}
    for c in f["cols"]:
        contrib = float(f["beta"][c] / f["sd"][c] * float(moves.get(c, 0.0)))
        parts[UNITS.get(c, (c,))[0]] = float(np.expm1(contrib) * 100)
        total += contrib
    return {"gold_pct": float(np.expm1(total) * 100),
            "parts_pct": parts,
            "band_1sd_pct": float(f["resid_sd"] * 100)}


def attribution(f: dict, feats: pd.DataFrame) -> dict:
    """Decompose the h-day move just realised into per-driver contributions."""
    row = feats.iloc[-1][f["cols"]]
    z = ((row - f["mu"]) / f["sd"]).fillna(0.0)
    contrib = z * f["beta"]
    actual = float(trailing_return(f["price"], f["h"]).iloc[-1])
    explained = float(contrib.sum() + f["ymu"])
    return {"window_days": f["h"],
            "actual_pct": float(np.expm1(actual) * 100),
            "explained_pct": float(np.expm1(explained) * 100),
            "unexplained_pct": float(np.expm1(actual - explained) * 100),
            "parts_pct": {UNITS.get(c, (c,))[0]: float(np.expm1(v) * 100)
                          for c, v in contrib.items()}}


def fair_value(feats: pd.DataFrame, h: int = DEFAULT_H, min_train: int = 750,
               months: int = 12) -> pd.Series:
    """Cumulative unexplained move over the trailing `months` windows, walk-forward.

    Each point is the residual of one h-day window fitted only on data that closed
    before that window opened, sampled h days apart so no return is counted twice.
    This is the number that answers the question the framework raises but cannot
    price: how much of gold's run is NOT explained by rates, the dollar and risk."""
    from src import goldstore
    panel = goldstore.daily_panel([goldfeatures.TARGET_PRICE],
                                  start=str(feats.index.min().date()))
    price = panel[goldfeatures.TARGET_PRICE].reindex(feats.index).ffill()
    cols = [c for c in MACRO_DRIVERS if c in feats.columns]
    y = trailing_return(price, h)
    d = pd.concat([feats[cols], y.rename("y")], axis=1).dropna()
    if len(d) <= min_train + h:
        return pd.Series(dtype=float)
    out, idx = {}, d.index
    for i in range(min_train, len(idx), h):
        tr = d.iloc[:max(i - h + 1, 1)]
        if len(tr) < min_train:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, np.nan)
        Z = ((tr[cols] - mu) / sd).fillna(0.0)
        ymu = float(tr["y"].mean())
        b = _ridge(Z.to_numpy(), (tr["y"] - ymu).to_numpy(), RIDGE_LAMBDA)
        z = ((d[cols].iloc[i] - mu) / sd).fillna(0.0)
        out[idx[i]] = float(d["y"].iloc[i] - (float(z.to_numpy() @ b) + ymu))
    r = pd.Series(out).sort_index()
    return (r.rolling(months, min_periods=max(3, months // 2)).sum() * 100
            ).rename("fair_value_gap_pct")


def compute(h: int = DEFAULT_H) -> dict:
    feats, targs = goldfeatures.load()
    f = fit(feats, targs, h)
    if not f:
        return {"error": "insufficient data"}
    full = fit(feats, targs, h,
               cols=[c for c in MACRO_DRIVERS + FLOW_DRIVERS if c in feats.columns])
    gap = fair_value(feats, h)
    out = {
        "asof": str(feats.index[-1].date()),
        "horizon_days": h,
        "r2_macro": f["r2"], "n": f["n"],
        "r2_with_flows": full.get("r2"),
        "unexplained_sd_pct": f["resid_sd"] * 100,
        "sensitivities": json.loads(sensitivities(f).to_json(orient="records")),
        "attribution": attribution(f, feats),
        "fair_value_gap_pct": float(gap.iloc[-1]) if len(gap) else None,
        "fair_value_gap_pctile": (float((gap <= gap.iloc[-1]).mean() * 100)
                                  if len(gap) else None),
        "built": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    h = DEFAULT_H
    if "--horizon" in sys.argv:
        h = int(sys.argv[sys.argv.index("--horizon") + 1])
    o = compute(h)
    if "error" in o:
        print(o["error"])
        return 1
    print(f"Gold sensitivities — {o['asof']}, {h}-day window, point-in-time data")
    print(f"  explains R2 {o['r2_macro']:.2f} from exogenous drivers "
          f"({o['r2_with_flows']:.2f} if co-moving flows are added back), "
          f"n={o['n']:,}")
    print(f"  unexplained move: +/-{o['unexplained_sd_pct']:.1f}% at 1 sd\n")
    print(f"    {'driver':30s} {'move':>8s} {'gold':>8s} {'t':>7s}")
    for s in o["sensitivities"]:
        print(f"    {s['driver']:30s} {s['move']:>8s} {s['gold_pct']:+7.2f}% "
              f"{s['t_stat']:+7.2f}")
    a = o["attribution"]
    print(f"\n  last {a['window_days']}d: actual {a['actual_pct']:+.2f}% = "
          f"explained {a['explained_pct']:+.2f}% + unexplained {a['unexplained_pct']:+.2f}%")
    if o["fair_value_gap_pct"] is not None:
        print(f"  fair value: gold {o['fair_value_gap_pct']:+.1f}% vs drivers over 12m "
              f"({o['fair_value_gap_pctile']:.0f}th pctile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
