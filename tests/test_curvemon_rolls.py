"""Curve / RV calendars must be like-for-like across contract rolls.

The 2026-10-06 audit: every one of the five largest daily moves in HG1 − HG2 over the past year
fell exactly on a roll date. HG's chain is H/K/N/U/Z, so the 1st/2nd pair alternates 2- and
3-month gaps and the raw spread steps at every roll — the z-score and 10y percentile were ranking
different contract pairs against each other (−2.2σ / 0th percentile raw; −1.05σ / 3rd on carry).

What must hold:
  * a calendar with CONSTANT annualised carry reads flat through every roll, whatever the gap,
    and today's value is exactly today's live F1 − F2;
  * roll_breaks flags exactly the sessions where the contract pair changed;
  * a front contract stamped implausibly far from its date (TTF's 'FJSU20' back to 2016) is
    unknown, and the spread drops those days rather than guess its rolls;
  * a seasonal gas calendar nets out its front month's norm from PRIOR years only;
  * a roll day's '1d Δ' is not reported as a move on the day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import curvemon

_MON = {3: "H", 5: "K", 7: "N", 9: "U", 12: "Z"}
_ALL = dict(zip(range(1, 13), "FGHJKMNQUVXZ"))


def _chain(dates, cycle: list[int]) -> list[tuple]:
    """(year, month) of the front contract per date: the first cycle month whose 1st is no
    more than 27 days behind the date — a front holds until ~4 weeks into its own month."""
    out = []
    for d in dates:
        for y in (d.year, d.year + 1):
            hit = [m for m in cycle if pd.Timestamp(y, m, 1) >= d - pd.Timedelta(days=27)]
            if hit:
                out.append((y, hit[0]))
                break
    return out


def _next(ym: tuple, cycle: list[int]) -> tuple:
    y, m = ym
    later = [c for c in cycle if c > m]
    return (y, later[0]) if later else (y + 1, cycle[0])


def _history(tkr: str, root: str, codes: dict, carry_of, start="2022-01-03", end="2026-10-05",
             f1=500.0) -> pd.DataFrame:
    """Synthetic raw/front2/contract history: F1 flat at `f1`, F2 set so the pair's annualised
    carry is exactly carry_of(front_month)."""
    cycle = sorted(codes)
    dates = pd.bdate_range(start, end)
    fronts = _chain(dates, cycle)
    p1, p2, ct = [], [], []
    for (y, m) in fronts:
        ny, nm = _next((y, m), cycle)
        gap = (ny - y) * 12 + nm - m
        p1.append(f1)
        p2.append(f1 - f1 * carry_of(m) * gap / 12.0)          # (F1 − F2)/F1 × 12/gap = carry
        ct.append(f"{root}{codes[m]}{y % 100:02d}")
    return pd.DataFrame({f"raw:{tkr}": p1, f"front2:{tkr}": p2, f"contract:{tkr}": ct},
                        index=dates)


_HG = {"key": "hg_t", "name": "t", "group": "Metals", "unit": "¢/lb", "dp": 2, "desc": "",
       "legs": [(1, "raw", "HGA Comdty"), (-1, "front2", "HGA Comdty")], "scale": 1.0,
       "basis": "carry"}


def test_constant_carry_reads_flat_through_alternating_gap_rolls():
    h = _history("HGA Comdty", "HG", _MON, lambda m: -0.04)
    raw = h["raw:HGA Comdty"] - h["front2:HGA Comdty"]
    assert raw.round(6).nunique() == 2                        # the raw level steps: 2- vs 3-month pairs
    s = curvemon._build_spread(_HG, h)
    assert s is not None
    assert s.std() < 1e-9                                     # carry basis: one flat line
    assert abs(s.iloc[-1] - raw.iloc[-1]) < 1e-9              # and today IS the live spread


def test_roll_breaks_flag_exactly_the_pair_switches():
    h = _history("HGA Comdty", "HG", _MON, lambda m: -0.04)
    s, brk = curvemon.spread_with_breaks(_HG, h)
    ct = h["contract:HGA Comdty"].reindex(s.index)
    want = (ct != ct.shift()).iloc[1:]
    assert not brk.iloc[0]                                    # the first bar is never a roll
    assert (brk.iloc[1:] == want).all()
    assert int(brk.sum()) == int(want.sum()) > 10


def test_an_implausible_front_contract_is_unknown_and_dropped():
    """TTF's generic carries 'FJSU20' on every day from 2016 to Aug 2020 — not a real front, so
    those days can't be placed against a roll and must not enter the series."""
    h = _history("HGA Comdty", "HG", _MON, lambda m: -0.04)
    h.iloc[:150, h.columns.get_loc("contract:HGA Comdty")] = "HGZ30"
    s = curvemon._build_spread(_HG, h)
    assert s.index.min() > h.index[149]
    plain = {**_HG, "basis": None}                             # a level-basis spread drops them too
    assert curvemon._build_spread(plain, h).index.min() > h.index[149]


def test_a_blanked_run_never_feeds_the_live_gap():
    """A blanked run's month is NaT in the runs frame, not None. Read as a month it put a NaN
    into the chain's cycle, and a NaN's hash is its id, so sorted() handed it back as the LIVE
    gap on some runs and not others — the spread above came back None (the pre-push flake of
    2026-10-06). The blanked run gets no gap; every known run gets the chain's own."""
    h = _history("HGA Comdty", "HG", _MON, lambda m: -0.04)
    h.iloc[:150, h.columns.get_loc("contract:HGA Comdty")] = "HGZ30"
    runs = curvemon._runs_for(h, "HGA Comdty")
    gaps = curvemon._next_gap(runs)
    assert pd.isna(runs["month"].iloc[0]) and gaps[0] is None
    assert all(g in (2, 3) for g in gaps[1:]), gaps          # H/K/N/U/Z: 2- and 3-month pairs
    assert runs["month"].iloc[-1].month == 12 and gaps[-1] == 3   # live Z → next H


def test_seasonal_norm_uses_prior_years_only():
    """A gas calendar whose carry is a pure function of the front month is ALL season: after two
    prior years of each month the de-seasonalised series is flat, and nothing before that point
    (when no instance has two prior same-month years) is ranked."""
    seas = {m: 0.5 * np.sin(m / 12 * 2 * np.pi) for m in range(1, 13)}
    h = _history("NGA Comdty", "NG", _ALL, lambda m: seas[m], start="2020-01-02", f1=3.0)
    spec = {"key": "ng_t", "name": "t", "group": "Energy", "unit": "$/MMBtu", "dp": 3, "desc": "",
            "legs": [(1, "raw", "NGA Comdty"), (-1, "front2", "NGA Comdty")], "scale": 1.0,
            "basis": "carry_seasonal"}
    s = curvemon._build_spread(spec, h)
    assert s is not None
    assert s.index.min() >= pd.Timestamp("2021-12-01")         # 2 prior same-month years needed
    assert s.std() < 1e-9
    raw = h["raw:NGA Comdty"] - h["front2:NGA Comdty"]
    assert abs(s.iloc[-1] - raw.iloc[-1]) < 1e-9


def test_a_roll_day_reports_no_one_day_change():
    h = _history("HGA Comdty", "HG", _MON, lambda m: -0.04)
    plain = {**_HG, "basis": None}
    ct = h["contract:HGA Comdty"]
    last_roll = ct.index[(ct != ct.shift())][-1]
    h = h.loc[:last_roll]                                      # 'today' is a roll day
    s, brk = curvemon.spread_with_breaks(plain, h)
    assert bool(brk.iloc[-1])
    row = curvemon._row(plain, s, 60, 2.0, brk)
    assert row["rolled_today"] and row["chg1d"] != row["chg1d"]   # NaN, not the pair-switch jump
