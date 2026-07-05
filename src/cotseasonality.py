"""Positioning seasonality (Bloomberg SEAG-style) for one market.

How speculative net %OI typically moves across the calendar year: every year is
aligned on ISO week-of-year (1..52) and, for each week, we take the median and the
25-75% range across all available years — the "typical" seasonal shape and how
dispersed (i.e. reliable) it is. The current calendar year's path is returned
separately so it can be overlaid: where positioning sits versus its seasonal norm now.

Pure pandas, NO project-relative imports, so it imports BOTH ways — `from src import
cotseasonality` (dashboard) and a bare `import cotseasonality` (the standalone
cotreport.py, which puts src/ on sys.path). Same dual-import contract as cotstudy.py.
"""
from __future__ import annotations

import pandas as pd

# Week-of-year anchors (≈ first week of each month) for labelling a 1..52 axis.
MONTH_WEEKS = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]
MONTH_LABELS = list("JFMAMJJASOND")
MIN_YEARS = 4          # below this the seasonal profile is too thin to be worth showing


def seasonal_profile(hist_one: pd.DataFrame, metric: str = "net_pct_oi", smooth: int = 3):
    """Seasonal profile of ONE market's weekly history.

    Returns (seas, current, info):
      seas    : DataFrame indexed by week-of-year 1..52 with columns med/p25/p75/n,
                or None if there isn't enough history (< MIN_YEARS).
      current : Series of the latest calendar year's `metric` by week-of-year (NaN-padded),
                or None.
      info    : {"years": int, "cur_year": int|None, "weeks": int}
    """
    info = {"years": 0, "cur_year": None, "weeks": 0}
    if hist_one is None or hist_one.empty or metric not in hist_one.columns:
        return None, None, info
    d = hist_one.dropna(subset=[metric]).copy()
    if d.empty:
        return None, None, info
    d["date"] = pd.to_datetime(d["date"])
    iso = d["date"].dt.isocalendar()
    d["woy"] = iso["week"].astype(int)
    d["iso_year"] = iso["year"].astype(int)
    d = d[d["woy"].between(1, 52)]                       # fold the rare week 53 away
    info["years"] = int(d["iso_year"].nunique())
    info["weeks"] = int(len(d))
    if d.empty or info["years"] < MIN_YEARS:
        return None, None, info

    g = d.groupby("woy")[metric]
    seas = pd.DataFrame({
        "med": g.median(), "p25": g.quantile(0.25),
        "p75": g.quantile(0.75), "n": g.count(),
    }).rename_axis("woy").reindex(range(1, 53))
    if smooth and smooth > 1:                            # wrap-around smooth (Dec→Jan continuous)
        for c in ("med", "p25", "p75"):
            s = seas[c]
            wrapped = pd.concat([s, s, s]).rolling(smooth, center=True, min_periods=1).mean()
            seas[c] = wrapped.iloc[len(s):2 * len(s)].to_numpy()

    cur_year = int(d["iso_year"].max())
    info["cur_year"] = cur_year
    current = (d[d["iso_year"] == cur_year]
               .drop_duplicates("woy", keep="last")
               .set_index("woy")[metric].reindex(range(1, 53)))
    return seas, current, info


def seasonal_long(hist_one: pd.DataFrame, metric: str = "net_pct_oi", smooth: int = 3):
    """The profile as a tidy long frame for Altair: columns woy, med, p25, p75, n, current
    (current is NaN where the current year has no data yet). Returns (df, info); df is
    empty when there's insufficient history."""
    seas, current, info = seasonal_profile(hist_one, metric=metric, smooth=smooth)
    if seas is None:
        return pd.DataFrame(columns=["woy", "med", "p25", "p75", "n", "current"]), info
    out = seas.reset_index()
    out["current"] = current.to_numpy()
    # a date in a fixed non-leap reference year per week-of-year, so Altair can use a native
    # temporal axis (month labels via format='%b') instead of a brittle label expression.
    out["wdate"] = pd.Timestamp("2001-01-01") + pd.to_timedelta((out["woy"] - 1) * 7, unit="D")
    return out, info
