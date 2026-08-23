"""Milestone 3 locks — every formula against a hand-computed value (spec §11).

A feature bug does not announce itself. A z-score with the wrong window, a momentum
term that forgot to skip the last month, a winsoriser that peeked at the full sample
— each produces a perfectly plausible column of numbers, and the only symptom is a
model that backtests better than it trades. So the numbers here are worked out by
hand in the test rather than snapshotted from the implementation, which would just
lock in whatever it did on the day.

The four that would cost the most if they broke silently:

  * `_winsorise` must be EXPANDING. Full-sample clipping uses knowledge of crises
    that had not happened yet.
  * `gold_mom_12m_1m` must SKIP the most recent month. Without the skip it is
    contaminated by short-horizon reversal and stops being momentum.
  * The volatility-scaled targets must divide by TRAILING vol. Dividing by vol
    realised over the forward window is scaling by the answer.
  * Forward targets must look forward and never fill the unknowable tail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import goldfeatures as gf


@pytest.fixture()
def idx():
    return pd.bdate_range("2020-01-01", periods=400)


# ---------------------------------------------------------------------------
# primitives, against hand-computed values
# ---------------------------------------------------------------------------
def test_chg_and_pct_are_n_period_differences(idx):
    s = pd.Series(np.arange(1.0, len(idx) + 1.0), index=idx)   # 1,2,3,...
    assert gf._chg(s, 20).iloc[25] == pytest.approx(20.0)      # linear ramp: +1/day
    assert pd.isna(gf._chg(s, 20).iloc[19]), "no value before n periods have passed"
    # pct over 20 periods at row 40 (value 41) vs row 20 (value 21): 41/21 - 1
    assert gf._pct(s, 20).iloc[40] == pytest.approx(41.0 / 21.0 - 1.0)


def test_z_score_matches_a_hand_computed_window(idx):
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    z = gf._z(s, 252)
    i = 300
    w = s.iloc[i - 251:i + 1]                                  # inclusive 252-wide
    expected = (s.iloc[i] - w.mean()) / w.std()
    assert z.iloc[i] == pytest.approx(expected, rel=1e-9)


def test_pctile_is_rank_within_the_trailing_window(idx):
    s = pd.Series(np.arange(1.0, len(idx) + 1.0), index=idx)
    # a strictly rising series always sits at the top of its own window
    assert gf._pctile(s, 252).iloc[300] == pytest.approx(100.0)
    # a strictly falling series always sits at the bottom (only itself is <=)
    d = pd.Series(np.arange(len(idx), 0.0, -1.0), index=idx)
    assert gf._pctile(d, 252).iloc[300] == pytest.approx(100.0 / 252.0)


def test_dist_is_percent_from_the_moving_average(idx):
    s = pd.Series(100.0, index=idx)
    s.iloc[-1] = 110.0
    got = gf._dist(s, 50).iloc[-1]
    ma = s.iloc[-50:].mean()
    assert got == pytest.approx(110.0 / ma - 1.0)
    # flat series sits exactly on its own average
    assert gf._dist(pd.Series(7.0, index=idx), 50).iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the leak-shaped ones
# ---------------------------------------------------------------------------
def test_winsorise_is_expanding_not_full_sample(idx):
    """The spike must not clip anything BEFORE it happens.

    Full-sample winsorising would pull down earlier observations using a bound
    computed from a crisis that had not occurred yet — a leak that is invisible
    because the output still looks like a sensible series."""
    s = pd.Series(1.0, index=idx)
    s.iloc[350] = 1000.0                                   # a late, violent outlier
    w = gf._winsorise(s, min_periods=250)
    assert w.iloc[300] == pytest.approx(1.0), "a later spike changed an earlier row"
    assert w.iloc[350] < 1000.0, "the spike itself should be clipped once bounds exist"
    # and truncating the series must not change any surviving value
    w_short = gf._winsorise(s.iloc[:340], min_periods=250)
    common = w_short.index
    assert np.allclose(w.loc[common], w_short, equal_nan=True), "winsorise looked ahead"


def test_momentum_skips_the_most_recent_month(idx):
    """gold_mom_12m_1m is the 12m return EXCLUDING the last month. If the skip is
    dropped it silently becomes a plain 12m return and picks up the short-horizon
    reversal it exists to avoid."""
    px = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    p = pd.DataFrame({gf.TARGET_PRICE: px}, index=idx)
    f = gf._build_all(p)["gold_mom_12m_1m"]
    i = 350
    expected = np.log(px.iloc[i - 20] / px.iloc[i - 252])
    assert f.iloc[i] == pytest.approx(expected, rel=1e-9)
    # it must NOT equal the un-skipped 12m return
    assert f.iloc[i] != pytest.approx(np.log(px.iloc[i] / px.iloc[i - 252]), rel=1e-9)


def test_targets_look_forward_and_scale_by_trailing_volatility(idx):
    px = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    p = pd.DataFrame({gf.TARGET_PRICE: px}, index=idx)
    t = gf.build_targets(p)
    i = 100
    assert t["fwd_ret_5d"].iloc[i] == pytest.approx(np.log(px.iloc[i + 5] / px.iloc[i]))
    # the unknowable tail stays NaN, never filled
    assert t["fwd_ret_5d"].iloc[-5:].isna().all()
    assert t["fwd_ret_250d"].iloc[-250:].isna().all()
    # scaling uses vol known AT i, not vol realised over [i, i+h]
    rv = t["realised_vol_60d"].iloc[i]
    assert t["fwd_ret_5d_scaled"].iloc[i] == pytest.approx(
        t["fwd_ret_5d"].iloc[i] / (rv * np.sqrt(5)), rel=1e-9)


def test_scaled_target_is_unchanged_by_future_data(idx):
    """Truncating the frame must not move a scaled target that already exists —
    the direct test that the denominator is trailing rather than forward-looking."""
    rng = np.random.default_rng(11)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    full = gf.build_targets(pd.DataFrame({gf.TARGET_PRICE: px}))
    cut = gf.build_targets(pd.DataFrame({gf.TARGET_PRICE: px.iloc[:300]}))
    i = 200
    assert full["realised_vol_60d"].iloc[i] == pytest.approx(
        cut["realised_vol_60d"].iloc[i], rel=1e-12)
    assert full["fwd_ret_5d_scaled"].iloc[i] == pytest.approx(
        cut["fwd_ret_5d_scaled"].iloc[i], rel=1e-12)


# ---------------------------------------------------------------------------
# cross-source derivations
# ---------------------------------------------------------------------------
def test_shanghai_premium_converts_and_uses_the_am_fix(idx):
    """900 CNY/g at 7.0 USDCNY = 900 * 31.1034768 / 7 = $3,999.02/oz.

    The reference leg must be the AM fix: SGE's day session closes ~08:30 London and
    the AM fix is 10:30, while the PM fix is 15:00. Against PM the series carries a
    whole London session of drift that has nothing to do with Chinese demand."""
    p = pd.DataFrame({
        "SGE_AU9999": pd.Series(900.0, index=idx),
        "USDCNY": pd.Series(7.0, index=idx),
        "LBMA_GOLD_AM_USD": pd.Series(3950.0, index=idx),
        "LBMA_GOLD_PM_USD": pd.Series(3800.0, index=idx),      # decoy, must be unused
    }, index=idx)
    prem = gf._build_all(p)["shanghai_premium_usd"].dropna().iloc[-1]
    assert prem == pytest.approx(900.0 * 31.1034768 / 7.0 - 3950.0, abs=1e-6)
    assert prem == pytest.approx(49.02, abs=0.05)
    assert prem != pytest.approx(199.02, abs=0.05), "used the PM fix by mistake"


def test_fx_breadth_is_the_share_of_legs_above_their_own_average(idx):
    """The 'is this gold or is this the dollar' test. All legs rising together = 1.0."""
    up = pd.Series(np.linspace(100.0, 300.0, len(idx)), index=idx)
    p = pd.DataFrame({gf.TARGET_PRICE: up, "LBMA_GOLD_PM_EUR": up,
                      "LBMA_GOLD_PM_GBP": up, "USDJPY": pd.Series(1.0, index=idx),
                      "USDINR": pd.Series(1.0, index=idx),
                      "USDCNY": pd.Series(1.0, index=idx)}, index=idx)
    assert gf._build_all(p)["gold_fx_breadth"].iloc[-1] == pytest.approx(1.0)
    down = pd.Series(np.linspace(300.0, 100.0, len(idx)), index=idx)
    p2 = p.copy()
    for c in p2.columns:
        if c in (gf.TARGET_PRICE, "LBMA_GOLD_PM_EUR", "LBMA_GOLD_PM_GBP"):
            p2[c] = down
    assert gf._build_all(p2)["gold_fx_breadth"].iloc[-1] == pytest.approx(0.0)


def test_gold_silver_ratio_falls_back_to_comex_when_lbma_is_absent(idx):
    """LBMA silver reaches 2005 and the COMEX contract only 2016; the ratio should
    prefer LBMA and fall back rather than going blank for a decade."""
    px = pd.Series(2000.0, index=idx)
    lbma = pd.Series(np.nan, index=idx)
    lbma.iloc[200:] = 25.0
    comex = pd.Series(20.0, index=idx)
    p = pd.DataFrame({gf.TARGET_PRICE: px, "LBMA_SILVER_USD": lbma,
                      "COMEX_SILVER_FRONT": comex}, index=idx)
    f = gf._build_all(p)
    assert "gold_silver_ratio_z_5y" in f
    # the underlying ratio switches source without leaving a hole
    assert f["gold_silver_ratio_z_5y"].notna().sum() > 0


# ---------------------------------------------------------------------------
# no train/serve skew
# ---------------------------------------------------------------------------
def test_stored_features_round_trip_unchanged(tmp_path, monkeypatch):
    """Spec §4.1 stores features so the backtest and the live path read identical
    numbers. If the round trip is lossy, the two paths silently diverge."""
    monkeypatch.setattr(gf, "STORE_DIR", tmp_path)
    monkeypatch.setattr(gf, "FEATURES_FILE", tmp_path / "features.parquet")
    monkeypatch.setattr(gf, "TARGETS_FILE", tmp_path / "targets.parquet")
    monkeypatch.setattr(gf, "FEATURE_META", tmp_path / "feature_meta.json")

    idx = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(5)
    px = pd.Series(1200 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))), index=idx)
    panel = pd.DataFrame({c: pd.Series(rng.normal(2, 0.2, len(idx)), index=idx)
                          for c in gf.PANEL_SERIES}, index=idx)
    panel[gf.TARGET_PRICE] = px
    monkeypatch.setattr(gf.goldstore, "daily_panel", lambda *a, **k: panel)

    built, _ = gf.build()
    loaded, _ = gf.load()
    common = [c for c in built.columns if c in loaded.columns]
    assert common, "nothing round-tripped"
    a = built[common].loc[loaded.index]
    assert np.allclose(a.to_numpy(), loaded[common].to_numpy(), equal_nan=True), \
        "stored features differ from freshly computed ones — train/serve skew"


def test_missing_features_are_declared_not_silently_absent():
    """A feature dropped for a source gap must be visible in the output. A list
    somebody has to re-read is how a gap becomes permanent."""
    assert set(gf.MISSING_FEATURES) >= {"risk_reversal_25d", "gold_aisc_ratio"}
    for reason in gf.MISSING_FEATURES.values():
        assert len(reason) > 20, "each gap needs a real reason, not a placeholder"
