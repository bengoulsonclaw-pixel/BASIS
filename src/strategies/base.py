"""Common output schema shared by every strategy.

Each strategy returns a pandas DataFrame with these columns, so the dashboard
and the PDF report can treat all strategies uniformly.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = [
    "strategy",      # strategy name
    "market",        # human-readable market / spread, e.g. "Gold / Silver ratio"
    "instruments",   # underlying Bloomberg tickers involved
    "signal",        # "Long spread", "Short", "Long", "Short", "—", ...
    "direction",     # +1 long / -1 short / 0 none  (used for colour-coding)
    "metric",        # headline number (z-score, % return, ...)
    "metric_label",  # what `metric` means
    "level",         # current spread / price level
    "context",       # short human note (mean, std, half-life, trade hint, ...)
]


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return empty()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]
