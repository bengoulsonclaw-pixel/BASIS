"""Locks over the put/call data-quality guards (src/strategies/putcall.py, 2026-08-23).

These exist because the Put/Call page and its client PDF published two nonsense rows for
weeks: Coffee (Arabica) at a put/call ratio of 17,385.00 (= 52,155 puts ÷ 3 calls, the put
leg stale-filled) and Brazilian Real at 0.01 (= 10 ÷ 2,000, its put leg frozen at 10.0 for
271 sessions) — both stamped "Put-heavy, 100th percentile". Three separate faults let that
through, and each has a lock here:

  • the old divide guard masked EXACT zeros, which never once fired (the stores hold no
    exact zeros at all) while the real failure was a 3-contract leg;
  • the percentile of a FROZEN series is 100.0 by construction, so the deadest data on the
    sheet scored the maximum extreme;
  • `dropna()` republished a weeks-old ratio as today's, so a dead feed kept printing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import putcall as pc


def _frames(pairs):
    """pairs = {ticker: (numerator_series, denominator_series)}"""
    num = pd.DataFrame({t: n for t, (n, _) in pairs.items()})
    den = pd.DataFrame({t: d for t, (_, d) in pairs.items()})
    return num, den


def test_thin_leg_blanks_the_ratio_on_either_side():
    """Symmetric floor. A denominator-only guard misses BRL, which fails on the
    NUMERATOR — that is exactly how 0.01 reached the client PDF."""
    idx = pd.bdate_range(end="2026-08-20", periods=3)
    num, den = _frames({
        "KCA Comdty": (pd.Series(52_155.0, idx), pd.Series(3.0, idx)),        # Coffee: tiny call leg
        "BRA Comdty": (pd.Series(10.0, idx), pd.Series(2_000.0, idx)),        # BRL: tiny put leg
        "ESA Index": (pd.Series(60_000.0, idx), pd.Series(40_000.0, idx)),    # healthy
    })
    r = pc._ratio(num, den, list(num.columns))
    assert r["KCA Comdty"].isna().all(), "a 3-contract denominator must not price a ratio"
    assert r["BRA Comdty"].isna().all(), "a 10-contract numerator must not price a ratio"
    assert r["ESA Index"].iloc[-1] == pytest.approx(1.5), "healthy legs are untouched"


def test_floor_is_configurable_and_defaults_differ_for_oi_and_volume():
    idx = pd.bdate_range(end="2026-08-20", periods=2)
    num, den = _frames({"X": (pd.Series(50.0, idx), pd.Series(50.0, idx))})
    assert pc._ratio(num, den, ["X"], min_leg=10)["X"].iloc[-1] == pytest.approx(1.0)
    assert pc._ratio(num, den, ["X"], min_leg=100)["X"].isna().all()
    assert pc.MIN_LEG_OI >= 100 and pc.MIN_LEG_VOL >= 100


def test_frozen_series_has_no_percentile():
    """`(s <= s.iloc[-1]).mean()` is 1.0 for a constant series, which is why a dead
    ticker was stamped 100th-percentile Put-heavy indefinitely (3M SONIA, 51 sessions)."""
    frozen = pd.Series([2.5] * 263)
    assert np.isnan(pc._pctl_now(frozen))
    live = pd.Series(np.linspace(1.0, 2.0, 263))
    assert pc._pctl_now(live) == pytest.approx(100.0)      # a genuine high still scores


def test_rolling_percentile_also_guards_frozen_windows():
    """Same fault in the series that drives the report heatmap."""
    s = pd.Series([1.0] * 120 + list(np.linspace(1.0, 3.0, 140)))
    roll = pc._rolling_pctl(s)
    assert np.isnan(roll.iloc[119]), "a flat window must not read as a 100th percentile"
    assert np.isfinite(roll.iloc[-1])


def test_partial_session_is_not_the_anchor():
    """The ~06:45 pull lands a fraction of a session; anchoring on it would read as a
    collapse in positioning across the whole book."""
    idx = pd.bdate_range(end="2026-08-21", periods=60)
    full = pd.DataFrame({f"T{i}": pd.Series(1_000.0, idx) for i in range(10)})
    full.iloc[-1, 3:] = np.nan                             # only 3 of 10 products printed
    assert pc._last_complete({"put_oi": full}) == idx[-2]


def test_complete_session_is_kept_as_the_anchor():
    idx = pd.bdate_range(end="2026-08-21", periods=60)
    full = pd.DataFrame({f"T{i}": pd.Series(1_000.0, idx) for i in range(10)})
    assert pc._last_complete({"put_oi": full}) == idx[-1]


def test_guard_constants_are_present():
    """Tripwire: these are the knobs the fix rests on — renaming one silently
    disarms the guard."""
    for attr in ("MIN_LEG_OI", "MIN_LEG_VOL", "MAX_STALE", "PARTIAL_FRAC"):
        assert hasattr(pc, attr), f"putcall.{attr} went missing"
    assert 0 < pc.PARTIAL_FRAC < 1 and pc.MAX_STALE >= 1
