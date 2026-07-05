"""Mean-reversion / rich-cheap monitor.

For each configured pair we build a spread (a ratio, or an OLS-hedged
difference), measure how far today's spread sits from its recent mean in
standard deviations (a z-score), estimate the mean-reversion half-life, and
flag pairs that look stretched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datafeed import get_history
from ..universe import PAIRS, name
from .base import frame

LOOKBACK = 90        # business days for the rolling mean / std
Z_THRESHOLD = 1.5    # |z| beyond this is flagged as "stretched"
STRATEGY = "Mean Reversion"


def _half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life: regress d(spread) on lagged spread."""
    s = spread.dropna()
    if len(s) < 30:
        return float("nan")
    d = pd.concat([s.diff(), s.shift(1)], axis=1).dropna()
    if d.empty:
        return float("nan")
    b = np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("nan")


def _build_spread(a: pd.Series, b: pd.Series, kind: str):
    if kind == "ratio":
        return a / b
    return a - b   # same-unit legs (WTI/Brent, Chi/KC wheat, Bund/OAT) -> simple differential


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    tickers = sorted({t for p in PAIRS for t in (p["a"], p["b"])})
    if history is None:
        history = get_history(tickers)

    rows = []
    for p in PAIRS:
        a, b = history[p["a"]], history[p["b"]]
        if pd.concat([a, b], axis=1).dropna().shape[0] < LOOKBACK + 5:
            continue   # not enough overlapping history (e.g. a ticker returned no data)
        spread = _build_spread(a, b, p["kind"])
        roll_mean = spread.rolling(LOOKBACK).mean()
        roll_std = spread.rolling(LOOKBACK).std()
        z_now = float((spread.iloc[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1])
        level = float(spread.iloc[-1])
        hl = _half_life(spread)

        if z_now <= -Z_THRESHOLD:
            signal, direction = "Long spread", 1     # cheap -> buy A / sell B
            hint = f"Buy {name(p['a'])} / Sell {name(p['b'])}"
        elif z_now >= Z_THRESHOLD:
            signal, direction = "Short spread", -1    # rich -> sell A / buy B
            hint = f"Sell {name(p['a'])} / Buy {name(p['b'])}"
        else:
            signal, direction, hint = "—", 0, ""

        context = f"mean {roll_mean.iloc[-1]:.2f} ± {roll_std.iloc[-1]:.2f}"
        if not np.isnan(hl):
            context += f", half-life ≈ {hl:.0f}d"
        if hint:
            context += f" — {hint}"

        rows.append({
            "strategy": STRATEGY,
            "market": p["name"],
            "instruments": f"{p['a']} / {p['b']}",
            "signal": signal,
            "direction": direction,
            "metric": round(z_now, 2),
            "metric_label": "z-score",
            "level": round(level, 3),
            "context": context,
        })

    df = frame(rows)
    # rank by how stretched each pair is
    order = df["metric"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def pair_chart_data(pair, threshold=None, history=None):
    """Per-pair series + summary for the dashboard spread chart. Reuses the exact
    strategy maths so the chart matches the table:
      - a_idx / b_idx : the two legs rebased to 100 (their relationship over time)
      - spread, mean, upper, lower : the spread (A-B or A/B), its 90-day mean and
        a ±threshold*sigma band
      - z : the rolling z-score
    Returns (DataFrame, info dict). Reads the snapshot in snapshot mode (no Bloomberg)."""
    thr = Z_THRESHOLD if threshold is None else float(threshold)
    a_t, b_t = pair["a"], pair["b"]
    if history is None:
        history = get_history([a_t, b_t])
    pair_df = pd.concat([history[a_t].rename("a"), history[b_t].rename("b")], axis=1).dropna()
    if pair_df.empty:
        return pd.DataFrame(), {}
    spread = _build_spread(pair_df["a"], pair_df["b"], pair["kind"])
    roll_mean = spread.rolling(LOOKBACK).mean()
    roll_std = spread.rolling(LOOKBACK).std()
    z = (spread - roll_mean) / roll_std
    out = pd.DataFrame({
        "date": pair_df.index,
        "a_idx": (pair_df["a"] / pair_df["a"].iloc[0] * 100.0).to_numpy(),
        "b_idx": (pair_df["b"] / pair_df["b"].iloc[0] * 100.0).to_numpy(),
        "spread": spread.to_numpy(),
        "mean": roll_mean.to_numpy(),
        "upper": (roll_mean + thr * roll_std).to_numpy(),
        "lower": (roll_mean - thr * roll_std).to_numpy(),
        "z": z.to_numpy(),
    })
    valid = out.dropna(subset=["z"])
    cur = valid.iloc[-1] if not valid.empty else out.iloc[-1]
    sp = out["spread"].dropna()
    cur_spread = float(cur["spread"]) if pd.notna(cur["spread"]) else float("nan")
    pctl = (float((sp.tail(252) <= cur_spread).mean() * 100.0)
            if len(sp) and pd.notna(cur_spread) else float("nan"))
    info = {
        "a_name": name(a_t), "b_name": name(b_t), "kind": pair["kind"],
        "z": float(cur["z"]) if pd.notna(cur["z"]) else float("nan"),
        "spread": cur_spread,
        "mean": float(cur["mean"]) if pd.notna(cur["mean"]) else float("nan"),
        "half_life": _half_life(spread), "pctl": pctl, "threshold": thr,
    }
    return out, info
