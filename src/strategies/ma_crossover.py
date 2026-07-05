"""Moving-average crossover (golden / death cross) — EMA + momentum confirmed.

The signal is anchored on a fast/slow SMA *golden cross* / *death cross*, and a
trade is taken only when BOTH confirmations agree:

    fast-SMA above slow-SMA  (golden cross)  ->  Long  — if fast-EMA > fast-SMA AND mom > 0
    fast-SMA below slow-SMA  (death cross)   ->  Short — if fast-EMA < fast-SMA AND mom < 0

The fast EMA on the trend side confirms short-term momentum still leads the fast
SMA; the trailing return confirms the move has momentum behind it. If either
disagrees the trade is skipped ("—" / unconfirmed).

Two configs run off this one module as separate dashboard buttons:
  * POSITION — 50/200 cross, 15-EMA confirm, 3-month (63d) return  (golden/death cross)
  * SWING    — 20/50 cross,  9-EMA confirm,  1-month (21d) return  (faster, swing horizon)

An extra context SMA (100 for position, 200 for swing) and the fast EMA are also
returned for the dashboard chart (a four-MA "ribbon"; the fast SMA is coloured by
its slope), but only the fast/slow cross, the fast EMA and the return drive the
signal — the context SMA is not a gate.

Distinct from the Trend page, which leads on the sign of the 3-month return with a
20/100 MA as context only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..datafeed import get_history
from ..universe import TREND_UNIVERSE, asset, name
from .base import frame

# Asset classes these crossover strategies skip. Bonds/rates were a consistent
# money-loser across EVERY pair in the 9.7-year backtest (Sharpe −0.6 to −1.6),
# so the MA pages don't signal on them — they stay in every other strategy.
EXCLUDE_ASSETS = {"Bonds"}


@dataclass(frozen=True)
class MAConfig:
    name: str            # strategy / button name
    fast: int            # fast SMA — the slope-coloured "signal" line + half the cross
    slow: int            # slow SMA — the other half of the cross
    ema: int             # fast EMA confirmation
    ctx: int             # extra context SMA (drawn on the chart only)
    mom: int             # momentum-filter window, trading days
    mom_label: str       # short label for that window, e.g. "3m" / "1m"
    lookback: int        # chart window, trading days
    horizon: str         # "position" / "swing"


POSITION = MAConfig("MA Crossover", fast=50, slow=200, ema=15, ctx=100,
                    mom=63, mom_label="3m", lookback=504, horizon="position")
SWING = MAConfig("MA Swing", fast=20, slow=50, ema=9, ctx=200,
                 mom=21, mom_label="1m", lookback=252, horizon="swing")

# Module default = the position (50/200) variant; the swing module reuses _find/_chart.
CFG = POSITION
STRATEGY = CFG.name


def _days_since_cross(fast: pd.Series, slow: pd.Series) -> int | None:
    """Trading days since the fast MA last crossed the slow MA (0 = crossed on the
    latest bar). None if the two never cross in the overlapping (non-NaN) window."""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None
    s = np.sign(diff.to_numpy())
    s[s == 0] = 1
    flips = np.where(s[1:] != s[:-1])[0]    # i where sign changed between bar i and i+1
    if len(flips) == 0:
        return None
    return int(len(s) - 1 - (flips[-1] + 1))


def _signal(ma_fast: float, ma_slow: float, ema_fast: float, mom: float):
    """(signal, direction, ema_ok, mom_ok). The fast/slow cross sets the regime; the
    trade needs BOTH the fast EMA on the trend side and the return agreeing in sign."""
    cross_dir = 1 if ma_fast > ma_slow else -1
    ema_ok = (ema_fast > ma_fast) if cross_dir == 1 else (ema_fast < ma_fast)
    mom_ok = (mom > 0) if cross_dir == 1 else (mom < 0)
    if ema_ok and mom_ok:
        return ("Long", 1, ema_ok, mom_ok) if cross_dir == 1 else ("Short", -1, ema_ok, mom_ok)
    return "—", 0, ema_ok, mom_ok


def _find(history: pd.DataFrame, cfg: MAConfig) -> pd.DataFrame:
    rows = []
    for t in TREND_UNIVERSE:
        if asset(t) in EXCLUDE_ASSETS:           # bonds/rates — see EXCLUDE_ASSETS note
            continue
        px = history[t].dropna()
        if len(px) < cfg.slow + 5:
            continue
        fast_s = px.rolling(cfg.fast).mean()
        slow_s = px.rolling(cfg.slow).mean()
        ema_f = px.ewm(span=cfg.ema, adjust=False).mean()
        fast, slow, ef = fast_s.iloc[-1], slow_s.iloc[-1], ema_f.iloc[-1]
        if pd.isna(fast) or pd.isna(slow):
            continue

        mom = px.iloc[-1] / px.iloc[-cfg.mom] - 1.0
        ma_gap = (fast / slow - 1) * 100                     # crossover strength / conviction
        signal, direction, ema_ok, mom_ok = _signal(fast, slow, ef, mom)

        cross = "golden cross" if fast > slow else "death cross"
        days = _days_since_cross(fast_s, slow_s)
        state = f"fresh {cross} ({days}d ago)" if days is not None and days <= cfg.fast else cross
        ema_sym = ">" if ef > fast else "<"
        note = (f"{state}; {cfg.ema}-EMA{ema_sym}{cfg.fast} {'✓' if ema_ok else '✗'}; "
                f"{cfg.mom_label} {mom * 100:+.1f}% {'✓' if mom_ok else '✗'}")
        if direction == 0:
            note += " — unconfirmed"

        rows.append({
            "strategy": cfg.name,
            "market": name(t),
            "instruments": t,
            "signal": signal,
            "direction": direction,
            "metric": round(ma_gap, 2),
            "metric_label": f"MA gap % ({cfg.fast}/{cfg.slow})",
            "level": round(float(px.iloc[-1]), 3),
            "context": note,
        })

    df = frame(rows)
    if df.empty:
        return df
    # Confirmed signals first, then by crossover strength (|MA gap|).
    df["_conf"] = (df["direction"] != 0).astype(int)
    df["_abs"] = df["metric"].abs()
    df = df.sort_values(["_conf", "_abs"], ascending=[False, False]).drop(columns=["_conf", "_abs"])
    return df.reset_index(drop=True)


def _chart(ticker: str, cfg: MAConfig, history: pd.DataFrame | None = None):
    """Price + the four-MA ribbon (EMA / fast SMA / context SMA / slow SMA) over the
    last cfg.lookback days, plus an info dict. `fast_rising` flags the fast SMA's
    slope for green/red colouring. Returns (DataFrame, info) or (None, None)."""
    if history is None:
        history = get_history([ticker])
    px = history[ticker].dropna()
    if len(px) < cfg.slow + 5:
        return None, None

    fast_s = px.rolling(cfg.fast).mean()
    ctx_s = px.rolling(cfg.ctx).mean()
    slow_s = px.rolling(cfg.slow).mean()
    ema_f = px.ewm(span=cfg.ema, adjust=False).mean()
    data = pd.DataFrame({
        "date": px.index, "price": px.to_numpy(), "ema": ema_f.to_numpy(),
        "fast": fast_s.to_numpy(), "ctx": ctx_s.to_numpy(), "slow": slow_s.to_numpy(),
    }).tail(cfg.lookback)
    data["fast_rising"] = data["fast"].diff() > 0        # fast-SMA slope → green/red on the chart

    mom = px.iloc[-1] / px.iloc[-cfg.mom] - 1.0
    signal, direction, ema_ok, mom_ok = _signal(fast_s.iloc[-1], slow_s.iloc[-1], ema_f.iloc[-1], mom)
    info = {
        "state": "golden cross" if fast_s.iloc[-1] > slow_s.iloc[-1] else "death cross",
        "mom": float(mom),
        "ema_above": bool(ema_f.iloc[-1] > fast_s.iloc[-1]),
        "ema_ok": bool(ema_ok),
        "mom_ok": bool(mom_ok),
        "confirmed": bool(direction != 0),
        "signal": signal,
        "days_since": _days_since_cross(fast_s, slow_s),
        "price": float(px.iloc[-1]),
    }
    return data, info


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    return _find(history if history is not None else get_history(TREND_UNIVERSE), CFG)


def crossover_chart_data(ticker: str, history: pd.DataFrame | None = None):
    return _chart(ticker, CFG, history)
