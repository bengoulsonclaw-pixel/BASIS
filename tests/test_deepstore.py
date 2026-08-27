"""Golden locks on the deep price store (src/deepstore.py) — the panama/difference
roll-adjustment maths and the read conventions the WHOLE app now rides (today's
switchover moved ~122 live signals onto these code paths):

  * _offsets/_adjust — the roll gap is ('2' − '1') on the last old-front day, applied
    cumulatively to everything at or before it; point moves preserved exactly;
  * get_ta — STIRs as 100 − adjusted, bonds as their deep benchmark YIELD (dropped
    when absent, never served as price), non-FI trimmed to the positive tail;
  * overlay — the tabt seam's guard rails: never a shallower store, never a stale one,
    ffill-heal holidays, volume never ffilled.

All store reads run against a THROWAWAY store in tmp (STORE_DIR monkeypatched) —
the repo's real deep store is never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import deepstore


# ── the adjustment maths on bare series ──────────────────────────────────────
def _roll_fixture():
    idx = pd.bdate_range("2026-03-02", periods=6)
    px1 = pd.Series([50.0, 51.0, 52.0, 49.0, 49.5, 50.0], index=idx)
    # front contract flips on day 4 — the '1' generic drops 3 points on the roll itself
    ct = pd.Series(["CLK6"] * 3 + ["CLM6"] * 3, index=idx)
    px2 = pd.Series([48.5, 48.6, 48.7, 47.9, np.nan, np.nan], index=idx)  # the incoming contract
    return idx, px1, px2, ct


def test_offsets_from_front2_gap():
    idx, px1, px2, ct = _roll_fixture()
    offs = deepstore._offsets(px1, px2, ct)
    assert len(offs) == 1
    day, gap = offs[0]
    assert day == idx[2]                          # last day the OLD contract was front
    assert gap == pytest.approx(48.7 - 52.0)      # '2' − '1' on that day, NOT the raw jump


def test_adjust_preserves_point_moves():
    idx, px1, px2, ct = _roll_fixture()
    adj = deepstore._adjust(px1, deepstore._offsets(px1, px2, ct))
    # pre-roll levels shifted onto the new contract's curve …
    assert list(adj.values) == pytest.approx([46.7, 47.7, 48.7, 49.0, 49.5, 50.0])
    # … so the roll-day "move" is the real 48.7 -> 49.0, and today's level is untouched
    assert adj.iloc[-1] == px1.iloc[-1]
    real_moves = np.diff(adj.values)
    assert real_moves[2] == pytest.approx(0.3)    # not the raw fake −3.0
    assert np.diff(px1.values)[0] == pytest.approx(real_moves[0])   # non-roll days identical


def test_adjust_falls_back_to_observed_jump_without_front2():
    idx, px1, _px2, ct = _roll_fixture()
    offs = deepstore._offsets(px1, None, ct)
    assert offs[0][1] == pytest.approx(49.0 - 52.0)     # the whole jump becomes the gap
    adj = deepstore._adjust(px1, offs)
    assert adj.iloc[3] - adj.iloc[2] == pytest.approx(0.0)   # roll boundary flattened


def test_no_contract_series_means_no_offsets():
    idx, px1, px2, _ct = _roll_fixture()
    assert deepstore._offsets(px1, px2, None) == []


# ── store readers on a throwaway tmp store ───────────────────────────────────
@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(deepstore, "STORE_DIR", tmp_path)
    idx = pd.bdate_range("2026-03-02", periods=6)
    _, px1, px2, ct = _roll_fixture()
    prices = pd.DataFrame({
        "CLA Comdty": px1,                                     # rolls (adjusted on read)
        "SERA Comdty": [95.8, 95.9, 96.0, 96.1, 96.0, 96.2],   # STIR
        "TYA Comdty": [110.0, 110.5, 111.0, 111.2, 111.1, 111.3],   # bond WITH deep yield
        "RXA Comdty": [130.0, 130.5, 131.0, 131.2, 131.1, 131.3],   # bond WITHOUT deep yield
        "NGA Comdty": [-1.0, -0.5, 2.5, 2.6, 2.7, 2.8],        # dragged through zero deep back
    }, index=idx)
    deepstore._write(prices, "prices")
    deepstore._write(pd.DataFrame({"CLA Comdty": px2}, index=idx), "front2")
    deepstore._write(pd.DataFrame({"CLA Comdty": ct}, index=idx), "contract")
    deepstore._write(pd.DataFrame({"TYA Comdty": [4.20, 4.22, 4.25, 4.24, 4.23, 4.26]},
                                  index=idx), "yields")
    deepstore._write(pd.DataFrame({"CLA Comdty": [1000.0] * 6}, index=idx), "volume")
    return idx


def test_get_raw_vs_adjusted(tmp_store):
    raw = deepstore.get_raw(["CLA Comdty"])
    adj = deepstore.get_adjusted(["CLA Comdty"])
    assert list(raw["CLA Comdty"]) == pytest.approx([50.0, 51.0, 52.0, 49.0, 49.5, 50.0])
    assert list(adj["CLA Comdty"]) == pytest.approx([46.7, 47.7, 48.7, 49.0, 49.5, 50.0])


def test_get_ta_conventions(tmp_store):
    ta = deepstore.get_ta(["CLA Comdty", "SERA Comdty", "TYA Comdty",
                           "RXA Comdty", "NGA Comdty"])
    # STIR: 100 − adjusted price = the implied rate
    assert list(ta["SERA Comdty"]) == pytest.approx([4.2, 4.1, 4.0, 3.9, 4.0, 3.8])
    # bond WITH a deep yield series: the strategies see the YIELD, not the price
    assert list(ta["TYA Comdty"]) == pytest.approx([4.20, 4.22, 4.25, 4.24, 4.23, 4.26])
    # bond WITHOUT one: DROPPED — never served as a price to yield-space strategies
    assert "RXA Comdty" not in ta.columns
    # non-FI trimmed to its positive tail: %-maths is meaningless across a zero crossing
    nga = ta["NGA Comdty"]
    assert nga.iloc[:2].isna().all() and list(nga.iloc[2:]) == pytest.approx([2.5, 2.6, 2.7, 2.8])
    # plain non-FI: the adjusted series straight through
    assert list(ta["CLA Comdty"]) == pytest.approx([46.7, 47.7, 48.7, 49.0, 49.5, 50.0])


def test_get_ta_slice_after_adjustment(tmp_store):
    """Adjustment must be computed on the FULL series, then sliced — a slice that
    dropped the roll would re-anchor everything before it."""
    idx = tmp_store
    ta = deepstore.get_ta(["CLA Comdty"], start=idx[1], end=idx[2])
    assert list(ta["CLA Comdty"]) == pytest.approx([47.7, 48.7])


def test_coverage_counts_rolls(tmp_store):
    cov = deepstore.coverage().set_index("ticker")
    assert cov.loc["CLA Comdty", "rolls"] == 1
    assert cov.loc["SERA Comdty", "rolls"] == 0
    assert cov.loc["CLA Comdty", "days"] == 6


# ── the tabt overlay seam ────────────────────────────────────────────────────
def _feed(idx, col="CLA Comdty", base=49.0):
    return pd.DataFrame({col: base + np.linspace(0, 1, len(idx))}, index=idx)


def test_overlay_skips_stale_store(tmp_store):
    """The store ends 2026-03-09; a feed running ~3 weeks past it must be left alone —
    a lapsed store silently truncating a backtest's right edge is the failure mode."""
    idx = pd.bdate_range("2026-02-02", periods=40)            # ends 2026-03-27
    pnl, sig = _feed(idx), _feed(idx)
    used, pnl2, sig2, _ = deepstore.overlay(["CLA Comdty"], idx[0], idx[-1], pnl, sig, None)
    assert used == []
    assert pnl2 is pnl and sig2 is sig


def test_overlay_skips_shallower_store(tmp_store):
    idx = pd.bdate_range("2026-01-05", periods=48)            # starts BEFORE the store's 03-02
    idx = idx[idx <= "2026-03-10"]
    pnl, sig = _feed(idx), _feed(idx)
    used, *_ = deepstore.overlay(["CLA Comdty"], idx[0], idx[-1], pnl, sig, None)
    assert used == []                                         # store adds no depth -> untouched


def test_overlay_upgrades_and_heals(tmp_store):
    store_idx = tmp_store                                     # 2026-03-02 … 2026-03-09
    feed_idx = pd.bdate_range("2026-03-04", periods=5)        # shallower, one day fresher
    assert feed_idx[-1] > store_idx[-1]
    pnl, sig = _feed(feed_idx, base=50.0), _feed(feed_idx, base=50.0)
    vol = pd.DataFrame({"CLA Comdty": [np.nan, 2000, 2000, 2000, 2000]}, index=feed_idx)
    used, pnl2, sig2, vol2 = deepstore.overlay(
        ["CLA Comdty"], store_idx[0], feed_idx[-1], pnl, sig, vol)
    assert used == ["CLA Comdty"]
    # deep depth gained: the union index reaches back to the store's first day
    assert pnl2.index[0] == store_idx[0]
    assert pnl2.loc[store_idx[0], "CLA Comdty"] == pytest.approx(46.7)   # adjusted level
    # the tail past the store's last settle keeps the FEED's value ("today" never vanishes)
    tail_day = feed_idx[-1]
    assert pnl2.loc[tail_day, "CLA Comdty"] == pytest.approx(pnl.loc[tail_day, "CLA Comdty"])
    # volume is swapped but NEVER ffilled — a flow's hole must stay a hole
    assert vol2.loc[store_idx[0], "CLA Comdty"] == pytest.approx(1000.0)


def test_stir_rate_level_is_anchored_to_the_front_contract(monkeypatch, tmp_path):
    """A STIR's charted "yield" must equal the front contract's implied rate TODAY.

    Regression for 2026-08-26: get_ta returned 100 - the difference-adjusted price, which
    is a rate path plus every roll gap the adjustment has removed - SOFR read 3.6450 while
    the front implied 3.9600, 31.5bp adrift, under an axis labelled "Yield (%)" and with
    those levels quoted to clients. Re-anchoring is a CONSTANT shift, so every difference
    (and therefore every signal, hit and backtest P&L) must come through untouched."""
    root = tmp_path / "iso"                       # own root: the anchor sits beside the store
    store, snap = root / "price_store", root / "snapshot"
    store.mkdir(parents=True); snap.mkdir(parents=True)
    monkeypatch.setattr(deepstore, "STORE_DIR", store)
    idx = pd.bdate_range("2026-03-02", periods=6)
    deepstore._write(pd.DataFrame(
        {"SERA Comdty": [95.8, 95.9, 96.0, 96.1, 96.0, 96.2]}, index=idx), "prices")
    before = deepstore.get_ta(["SERA Comdty"])["SERA Comdty"]      # no anchor file yet
    # now publish an 'A' generic priced 40bp away from the store's own series
    pd.DataFrame({"SERA Comdty": [96.2, 96.3, 96.4, 96.5, 96.4, 96.6]},
                 index=idx).rename_axis("date").reset_index().to_parquet(
                     snap / "prices.parquet", index=False)
    ta = deepstore.get_ta(["SERA Comdty"])["SERA Comdty"]
    assert ta.iloc[-1] == pytest.approx(100.0 - 96.6), "last level must equal the front rate"
    assert ta.iloc[-1] != pytest.approx(before.iloc[-1]), "control: the anchor moved it"
    assert list(ta.diff().dropna()) == pytest.approx(list(before.diff().dropna())),         "a constant shift must leave every move identical"


def test_anchor_never_reaches_outside_a_relocated_store(monkeypatch, tmp_path):
    """The anchor is found relative to STORE_DIR. A relocated store (any test, or a second
    install) must not silently read the live data/ directory - the first cut did, and it
    re-anchored a synthetic fixture to real SOFR."""
    store = tmp_path / "elsewhere" / "price_store"
    store.mkdir(parents=True)
    monkeypatch.setattr(deepstore, "STORE_DIR", store)
    assert deepstore._front_price_anchor() is None
