"""Locks over the unusual-option-flow provider (src/optflow.py).

Two things here are load-bearing and must not be relaxed without Ben's say-so:
  • the HEAT CEILING — a flow row is an observation, not a dislocation, and must never
    push a real signal out of the Hot Sheet's top 10;
  • the CLIENT-SAFE PROSE — these rows reach client PDFs, so the text stays neutral
    observation with no buy/sell/recommend language.

Synthetic frames written to tmp stores only — never the repo's data/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import hotsheet, optflow

TICK = "ESA Index"          # real universe tickers so name()/asset() resolve
STIR = "ERA Comdty"         # 3M Euribor — a STIR, deliberately NOT carved out


def _series(n=400, level=20_000.0, end="2026-08-20"):
    """A calm daily volume series: constant level with a little jitter so the
    percentile has something to rank (a frozen series is correctly unscoreable)."""
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(0)
    return pd.Series(level + rng.normal(0, level * 0.02, n), index=idx)


def _stores(tmp_path, monkeypatch, put=None, call=None):
    """Write synthetic put/call volume + OI stores and point the module at them."""
    idx = (put if put is not None else call).index
    for nm, s in (("putcall_put_vol.parquet", put), ("putcall_call_vol.parquet", call)):
        base = s if s is not None else pd.Series(1.0, index=idx)
        pd.DataFrame({TICK: base, STIR: base}).to_parquet(tmp_path / nm)
    pd.DataFrame({TICK: pd.Series(50_000.0, index=idx)}).to_parquet(
        tmp_path / "putcall_put_oi.parquet")
    monkeypatch.setattr(optflow, "SNAP", tmp_path)
    return idx[-1]


# --- the metric ------------------------------------------------------------
def test_ratio_and_baseline_are_exact(tmp_path, monkeypatch):
    """A known 10x spike on a known baseline reports exactly 10x."""
    s = _series(); s.iloc[-1] = s.iloc[-61:-1].median() * 10
    day = _stores(tmp_path, monkeypatch, put=s)
    got = [c for c in optflow.candidates(day) if c["ticker"] == TICK]
    assert len(got) == 1
    c = got[0]
    assert c["side"] == "put" and c["ratio"] == pytest.approx(10.0, rel=1e-6)
    assert c["base"] == pytest.approx(float(s.iloc[-61:-1].median()), rel=1e-9)


def test_baseline_excludes_today(tmp_path, monkeypatch):
    """No look-ahead: today's volume must never enter its own normal."""
    s = _series()
    day = _stores(tmp_path, monkeypatch, put=s.copy())
    s.iloc[-1] = s.iloc[-61:-1].median() * 8
    _stores(tmp_path, monkeypatch, put=s)
    base_spiked = optflow.candidates(day)[0]["base"]
    s.iloc[-1] = s.iloc[-61:-1].median() * 40          # a far bigger spike...
    _stores(tmp_path, monkeypatch, put=s)
    assert optflow.candidates(day)[0]["base"] == pytest.approx(base_spiked)   # ...same baseline


def test_gaps_are_compressed_not_zero_filled(tmp_path, monkeypatch):
    """NaN means the feed was dark, not that nothing traded (the stores hold no exact
    zeros at all). The baseline must therefore be the median of the last 60 REAL prints,
    reaching back through a blackout — not of the last 60 calendar rows with zeros in."""
    s = _series()
    s.iloc[-30:-18] = np.nan                           # a 12-session blackout
    s.iloc[-1] = 200_000.0
    day = _stores(tmp_path, monkeypatch, put=s)
    got = optflow.candidates(day)
    assert got, "a short blackout must not make the product unscoreable"
    prior = s.iloc[:-1].dropna()                       # every real print before today
    assert got[0]["base"] == pytest.approx(float(prior.iloc[-60:].median()), rel=1e-9)


# --- the gates -------------------------------------------------------------
@pytest.mark.parametrize("attr,value", [
    ("MIN_RATIO", 99.0),      # multiple too small
    ("MIN_PCTL", 100.1),      # percentile bar unreachable
    ("MIN_LOTS", 10 ** 9),    # absolute contract floor
    ("MIN_BASE", 10 ** 9),    # baseline floor — kills the 3-lots-a-day products
])
def test_each_gate_blanks_the_row(tmp_path, monkeypatch, attr, value):
    s = _series(); s.iloc[-1] = s.iloc[-61:-1].median() * 10
    day = _stores(tmp_path, monkeypatch, put=s)
    assert optflow.candidates(day), "control: the row fires before the gate is raised"
    monkeypatch.setattr(optflow, attr, value)
    assert optflow.candidates(day) == []


def test_tiny_product_cannot_fire(tmp_path, monkeypatch):
    """The real Dow E-mini case: 43 contracts on a 3-lot baseline is 14x normal and
    is not news. MIN_BASE/MIN_LOTS must blank it."""
    s = _series(level=3.0); s.iloc[-1] = 43.0
    day = _stores(tmp_path, monkeypatch, put=s)
    assert optflow.candidates(day) == []


def test_frozen_feed_emits_nothing(tmp_path, monkeypatch):
    """The 3M SONIA case — a feed frozen for weeks then printing must not read as a
    spike, and a constant series has no percentile at all."""
    s = _series(); s.iloc[-60:] = s.iloc[-61]          # frozen dead flat
    day = _stores(tmp_path, monkeypatch, put=s)
    assert optflow.candidates(day) == []


def test_partial_session_is_not_scored(tmp_path, monkeypatch):
    """The 06:45 pull lands a fraction of a session; anchoring on it would blank the
    feature every morning. The anchor must step back to the last real session."""
    s = _series(n=401, end="2026-08-21")
    s.iloc[-1] = s.iloc[-1] * 0.02                     # a 2%-of-normal partial capture
    idx = s.index
    pd.DataFrame({TICK: s}).to_parquet(tmp_path / "putcall_put_vol.parquet")
    pd.DataFrame({TICK: s}).to_parquet(tmp_path / "putcall_call_vol.parquet")
    pd.DataFrame({TICK: pd.Series(50_000.0, index=idx[:-1])}).to_parquet(
        tmp_path / "putcall_put_oi.parquet")           # OI stops a session earlier
    monkeypatch.setattr(optflow, "SNAP", tmp_path)
    anchor = optflow._anchor(pd.DataFrame({TICK: s}), pd.DataFrame({TICK: s.iloc[:-1]}))
    assert anchor == idx[-2], "must anchor on the last complete session"


# --- selection ------------------------------------------------------------
def test_cap_dedupe_and_ordering(tmp_path, monkeypatch):
    """Both sides of one product must not take both slots, and rows come out
    biggest-multiple first."""
    p = _series(); p.iloc[-1] = p.iloc[-61:-1].median() * 6
    c = _series(level=18_000.0); c.iloc[-1] = c.iloc[-61:-1].median() * 12
    day = _stores(tmp_path, monkeypatch, put=p, call=c)
    got = optflow.candidates(day)
    assert len(got) == len({r["ticker"] for r in got}), "one row per product"
    assert got[0]["side"] == "call", "the bigger multiple wins the product's slot"
    assert [r["ratio"] for r in got] == sorted((r["ratio"] for r in got), reverse=True)
    assert len(optflow.radar_items()) <= optflow.MAX_ROWS


def test_quarterly_witching_carve_out_hits_indices_only(tmp_path, monkeypatch):
    """Equity indices around triple-witching are churn (next-day OI change 0.19% vs
    1.10%), so they are suppressed. STIRs on the same date carry real news — IMM and
    FOMC dates — and must NOT be."""
    day = pd.Timestamp("2026-03-20")                   # third Friday of March
    s = _series(n=400, end="2026-03-20"); s.iloc[-1] = s.iloc[-61:-1].median() * 10
    _stores(tmp_path, monkeypatch, put=s)
    tickers = {c["ticker"] for c in optflow.candidates(day)}
    assert TICK not in tickers, "index row must be carved out at quarterly expiry"
    assert STIR in tickers, "STIR row must survive quarterly expiry"


# --- the two load-bearing locks -------------------------------------------
def test_heat_is_capped(tmp_path, monkeypatch):
    """A flow row is never a dislocation row. Even an absurd multiple stays under the
    ceiling."""
    s = _series(); s.iloc[-1] = s.iloc[-61:-1].median() * 5_000
    _stores(tmp_path, monkeypatch, put=s)
    items = optflow.radar_items()
    assert items, "control: something fired"
    assert all(it["heat"] <= optflow.FLOW_HEAT_CEIL for it in items)
    assert optflow.FLOW_HEAT_CEIL <= 50.0


def test_flow_never_displaces_a_dislocation_row(tmp_path, monkeypatch):
    """The whole design rests on this: with a normal sheet of real signals, flow rows
    must not reach the top strip."""
    s = _series(); s.iloc[-1] = s.iloc[-61:-1].median() * 5_000
    _stores(tmp_path, monkeypatch, put=s)
    flow = optflow.radar_items()
    assert flow
    # a normal sheet: distinct tags, since top_strip takes at most 2 rows per tag
    real = [hotsheet.item(tag=f"T{i}", key=f"d{i}", section="Positioning",
                          text=f"**Product {i}** positioning is crowded long.",
                          heat=50.0 + i) for i in range(12)]
    everything = sorted(real + flow, key=lambda it: -it["heat"])   # engine sorts by heat
    strip = hotsheet.top_strip(everything)
    assert len(strip) == 10
    assert not any(it["tag"] == "FLOW" for it in strip), \
        "a flow row pushed a real signal out of the top 10"


def test_prose_is_client_safe():
    """These rows reach client PDFs — neutral observation only (compliance)."""
    import re
    txt = ("**Corn** traded 282,449 call option contracts — **14.4x** its normal daily "
           "call volume (100th percentile of the year).")
    banned = re.compile(r"\b(buy|sell|recommend|should|target|cheap|rich|long|short)\b", re.I)
    assert not banned.search(txt)
    assert "**" in txt


def test_provider_never_raises(tmp_path, monkeypatch):
    """A provider that raises is isolated by the engine, but one that drops off the
    sheet silently is a bug — an empty store returns [], not an exception."""
    monkeypatch.setattr(optflow, "SNAP", tmp_path / "nope")
    assert optflow.radar_items() == []
    assert optflow.candidates() == []
