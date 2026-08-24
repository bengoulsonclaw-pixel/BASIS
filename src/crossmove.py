"""crossmove.py — translate a move in one macro instrument into the others.

"The dollar rallies 1%. What has gold typically done, and the 10y, over the last five
years?" That is the question this answers, and it is deliberately a question about
what HAS happened rather than what will.

HOW IT WORKS

For a chosen lookback and change horizon, every instrument is regressed on the driver
one pair at a time:

    move_in_Y  =  a  +  beta * move_in_X  +  e

`beta` is the average move in Y that has accompanied a one-unit move in X. Pairwise
rather than joint, on purpose: the question "what has gold done when the dollar moved
1%" is about the TOTAL association, including whatever else moved alongside the
dollar. A joint fit would answer a different question — gold's response holding
yields fixed — which is the right tool for attribution and the wrong one here.
macrochain.py does the joint version and the two disagree by design.

UNITS

Prices move in percent, yields and spreads in basis points, VIX in points. Mixing
them silently is how a dollar slider ends up labelled 100x wrong, so every instrument
declares its unit and every output carries it.

THREE THINGS TRAVEL WITH EVERY NUMBER

  t-statistic      Newey-West, because overlapping h-day changes are autocorrelated
                   by construction and plain errors overstate certainty by ~sqrt(h).
  r_squared        how much of Y's variation the driver accounts for at all.
  band_1sd         one standard deviation of what the relationship does NOT explain.

The band is the one that decides whether a number is usable. Over a month, gold's
unexplained variation is around 4%, against roughly 0.8% for a 1% dollar move. A
relationship can be statistically overwhelming and still be swamped by noise on any
single occasion, and both facts have to be visible at once.

WHAT THIS IS NOT

Not a forecast. Everything is CONTEMPORANEOUS — the same window on both sides. The
lead-lag work in golddiag tested whether any of these drivers leads gold, silver,
platinum or palladium at any horizon out to 60 days, and across four metals not one
does. This tool sizes a move that has already happened; it does not anticipate one.

Not causation either. A dollar move and a gold move share drivers; the beta records
that they travel together, not that one pushes the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldstore                                            # noqa: E402

# label -> (store series, unit, display step for the input box)
#
# `unit` drives BOTH the arithmetic and the label. "pct" instruments are differenced
# in logs and shown as percentages; "bp" instruments are already in percent in the
# store, so a diff is multiplied by 100; "pt" is a plain level difference.
INSTRUMENTS = {
    "Gold":            ("LBMA_GOLD_PM_USD", "pct", 1.0),
    "Silver":          ("LBMA_SILVER_USD", "pct", 1.0),
    "Platinum":        ("LBMA_PLATINUM_PM_USD", "pct", 1.0),
    "Palladium":       ("LBMA_PALLADIUM_PM_USD", "pct", 1.0),
    "Dollar (DXY)":    ("DXY", "pct", 0.5),
    "S&P 500":         ("SPX", "pct", 1.0),
    "EURUSD":          ("EURUSD", "pct", 0.5),
    "USDJPY":          ("USDJPY", "pct", 0.5),
    "US 2y yield":     ("NOMINAL_2Y", "bp", 10.0),
    "US 10y yield":    ("NOMINAL_10Y", "bp", 10.0),
    "US 10y real":     ("REAL_10Y_SPLICED", "bp", 10.0),
    "10y breakeven":   ("BREAKEVEN_10Y", "bp", 10.0),
    "HY spread":       ("HY_SPREAD", "bp", 10.0),
    "Baa spread":      ("BAA_SPREAD", "bp", 10.0),
    "VIX":             ("VIX", "pt", 1.0),
}

HORIZONS = {"1 day": 1, "1 week": 5, "1 month": 20, "3 months": 60}
LOOKBACKS = {"1 year": 1, "2 years": 2, "5 years": 5, "10 years": 10,
             "20 years": 20, "Everything": None}

UNIT_LABEL = {"pct": "%", "bp": "bp", "pt": "pts"}


def _series(label: str) -> pd.Series:
    sid, _unit, _step = INSTRUMENTS[label]
    s = goldstore.get_series(sid)
    return s[s.notna()]


def changes(labels=None, horizon: int = 20, years: float | None = 5) -> pd.DataFrame:
    """Aligned h-day changes for the requested instruments, in their own units."""
    labels = list(labels or INSTRUMENTS)
    panel = goldstore.daily_panel([INSTRUMENTS[l][0] for l in labels])
    panel = panel.rename(columns={INSTRUMENTS[l][0]: l for l in labels})
    if years:
        cut = panel.index.max() - pd.DateOffset(years=years)
        panel = panel[panel.index >= cut]
    out = pd.DataFrame(index=panel.index)
    for l in labels:
        unit = INSTRUMENTS[l][1]
        s = panel[l]
        if unit == "pct":
            out[l] = np.log(s.where(s > 0)).diff(horizon) * 100
        elif unit == "bp":
            out[l] = s.diff(horizon) * 100
        else:
            out[l] = s.diff(horizon)
    return out


def _nw(y: np.ndarray, x: np.ndarray, lags: int) -> tuple:
    """Bivariate slope with Newey-West standard error."""
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    S = (X * r[:, None]).T @ (X * r[:, None])
    for L in range(1, max(lags, 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * r[L:, None]).T @ (X[:-L] * r[:-L, None])
        S += w * (A + A.T)
    XtX = np.linalg.pinv(X.T @ X)
    cov = XtX @ S @ XtX
    se = float(np.sqrt(max(cov[1, 1], 0)))
    r2 = float(1 - r.var() / y.var()) if y.var() > 0 else np.nan
    return float(b[1]), se, r2, float(np.std(r))


def translate(driver: str, move: float, horizon: int = 20,
              years: float | None = 5, labels=None) -> pd.DataFrame:
    """Given `move` in `driver`, what have the others typically done?

    Returns one row per instrument with the implied move, its t-statistic, the share
    of variation explained, and the one-standard-deviation band of what is left over.
    """
    labels = [l for l in (labels or INSTRUMENTS) if l != driver]
    d = changes([driver] + labels, horizon, years)
    rows = []
    for l in labels:
        f = d[[driver, l]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(f) < 60:
            continue
        beta, se, r2, resid = _nw(f[l].to_numpy(), f[driver].to_numpy(), horizon)
        unit = INSTRUMENTS[l][1]
        rows.append({
            "instrument": l,
            "implied": beta * move,
            "unit": UNIT_LABEL[unit],
            "beta": beta,
            "t": beta / se if se > 0 else np.nan,
            "r_squared": r2,
            "band_1sd": resid,
            "n": int(len(f)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_t"] = out["t"].abs()
    return out.sort_values("abs_t", ascending=False).drop(columns="abs_t") \
              .reset_index(drop=True)


def matrix(horizon: int = 20, years: float | None = 5, labels=None) -> pd.DataFrame:
    """Every pair: beta of the ROW instrument per one unit of the COLUMN instrument."""
    labels = list(labels or INSTRUMENTS)
    d = changes(labels, horizon, years)
    m = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for col in labels:
        for row in labels:
            if row == col:
                m.loc[row, col] = 1.0
                continue
            f = d[[col, row]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(f) < 60:
                continue
            beta, _se, _r2, _sd = _nw(f[row].to_numpy(), f[col].to_numpy(), horizon)
            m.loc[row, col] = beta
    return m


def unit_of(label: str) -> str:
    return UNIT_LABEL[INSTRUMENTS[label][1]]


def default_move(label: str) -> float:
    """A sensible starting shock — 1% for prices, 25bp for yields, 1pt for VIX."""
    unit = INSTRUMENTS[label][1]
    return {"pct": 1.0, "bp": 25.0, "pt": 1.0}[unit]


def caveat(driver: str, horizon: int, years) -> str:
    """The sentence that has to appear next to the numbers, generated not asserted."""
    span = f"the last {years} years" if years else "the full history"
    hz = {1: "one day", 5: "one week", 20: "one month", 60: "three months"}.get(
        horizon, f"{horizon} trading days")
    return (f"Average co-movement over {span}, measured on {hz} changes. "
            f"Contemporaneous — this sizes a move that has already happened and does "
            f"not forecast one. Each row's band is one standard deviation of what the "
            f"relationship leaves unexplained; where that band is larger than the "
            f"implied move, the typical outcome is dominated by everything else going "
            f"on.")


def main() -> int:
    driver = sys.argv[1] if len(sys.argv) > 1 else "Dollar (DXY)"
    move = float(sys.argv[2]) if len(sys.argv) > 2 else default_move(driver)
    years = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    t = translate(driver, move, horizon=20, years=years)
    print(f"\n{driver} {move:+g}{unit_of(driver)} over one month, last {years:g} years\n")
    print(f"  {'instrument':16s} {'implied':>12s} {'t':>7s} {'R2':>6s} {'+/-1sd':>10s}")
    for r in t.itertuples():
        print(f"  {r.instrument:16s} {r.implied:+10.2f}{r.unit:<2s} {r.t:+7.1f} "
              f"{r.r_squared:6.2f} {r.band_1sd:9.2f}{r.unit}")
    print("\n  " + caveat(driver, 20, years))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
