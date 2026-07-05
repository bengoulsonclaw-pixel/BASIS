"""MA Crossover — swing variant (20/50 SMA, 9-EMA + 1-month confirmed).

A faster-horizon sibling of the position-trade 50/200 MA Crossover. Same logic
(see ma_crossover.py), just the shorter periods the literature pairs with swing
trading: a 20/50 SMA golden/death cross, confirmed by the 9-day EMA on the trend
side of the 20 and the trailing 1-month (21-day) return. The 200-day SMA is drawn
on the chart as the big-picture trend, but does not gate the signal.
"""
from __future__ import annotations

import pandas as pd

from ..datafeed import get_history
from ..universe import TREND_UNIVERSE
from .ma_crossover import SWING, _chart, _find

CFG = SWING
STRATEGY = CFG.name


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    return _find(history if history is not None else get_history(TREND_UNIVERSE), CFG)


def crossover_chart_data(ticker: str, history: pd.DataFrame | None = None):
    return _chart(ticker, CFG, history)
