"""Milestone 5 locks — Stage 1 diagnostics (spec §6 Stage 1).

The diagnostics decide which features are worth fitting, so a fault here misdirects
every later milestone. Three things must be provably right:

  * **A planted lead is found at the right lag.** If the cross-correlation is off by
    a sign or a shift, a coincident feature reads as leading and the whole "which
    features carry signal" conclusion inverts.
  * **The multiple-comparison threshold is applied.** 28 features x 61 lags is 1,708
    simultaneous tests; at the naive 5% bar roughly 85 pure-noise cells clear by
    construction, and the largest of them looks exactly like a discovery.
  * **The stability metric survives a near-zero correlation.** Agreement-with-the-
    full-sample-sign reads as 6% instability when the full-sample number is noise
    about zero and its sign is arbitrary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import golddiag as gd


@pytest.fixture()
def price():
    idx = pd.bdate_range("2010-01-01", periods=2500)
    rng = np.random.default_rng(21)
    return pd.Series(1200 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)


# ---------------------------------------------------------------------------
# lead-lag detection
# ---------------------------------------------------------------------------
def test_a_planted_lead_is_found_at_the_right_lag(price):
    """A feature built to predict the return 10 days ahead must peak at lag 10."""
    ret = np.log(price / price.shift(1))
    lead = 10
    rng = np.random.default_rng(2)
    # feature[t] carries the return at t+lead, plus noise to keep it realistic
    feat = ret.shift(-lead) + pd.Series(rng.normal(0, 0.004, len(price)), index=price.index)
    feats = pd.DataFrame({"planted_lead": feat})
    xc, counts = gd.cross_correlations(feats, price, max_lag=30)
    summary, raw, adj = gd.lead_lag_summary(xc, counts, n_tests=len(xc) * xc.shape[1])
    row = summary.set_index("feature").loc["planted_lead"]
    assert int(row["peak_lag"]) == lead, f"peak found at {row['peak_lag']}, expected {lead}"
    assert not row["coincident"]
    assert row["clears_bonferroni"], "a real lead must survive the adjusted threshold"


def test_a_coincident_feature_peaks_at_lag_zero(price):
    ret = np.log(price / price.shift(1))
    rng = np.random.default_rng(3)
    feats = pd.DataFrame({
        "coincident": ret + pd.Series(rng.normal(0, 0.004, len(price)), index=price.index)})
    xc, counts = gd.cross_correlations(feats, price, max_lag=30)
    summary, _, _ = gd.lead_lag_summary(xc, counts, n_tests=len(xc) * xc.shape[1])
    row = summary.set_index("feature").loc["coincident"]
    assert int(row["peak_lag"]) == 0
    assert row["coincident"], "a same-day relationship must be labelled coincident"


def test_pure_noise_does_not_clear_the_adjusted_threshold(price):
    """The control. Noise features must not be promoted to findings — this is the
    test that would fail if the Bonferroni adjustment were dropped."""
    rng = np.random.default_rng(7)
    feats = pd.DataFrame({f"noise_{i}": rng.normal(0, 1, len(price))
                          for i in range(12)}, index=price.index)
    xc, counts = gd.cross_correlations(feats, price, max_lag=60)
    n_tests = len(xc) * xc.shape[1]
    summary, raw, adj = gd.lead_lag_summary(xc, counts, n_tests=n_tests)
    assert adj > raw, "the adjusted threshold must be stricter than the raw one"
    assert summary["clears_bonferroni"].sum() == 0, \
        "pure noise cleared the adjusted threshold"


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
def test_r_threshold_matches_the_textbook_value():
    """At n = 4,500 the 5% two-sided threshold on a correlation is ~0.029."""
    assert gd.r_threshold(4500, 0.05) == pytest.approx(0.029, abs=0.003)
    # Bonferroni over 1,708 tests is far stricter
    strict = gd.r_threshold(4500, 0.05 / 1708)
    assert strict > gd.r_threshold(4500, 0.05) * 1.7
    # a smaller sample needs a bigger correlation to say anything
    assert gd.r_threshold(300, 0.05) > gd.r_threshold(4500, 0.05)


def test_each_feature_is_judged_at_its_own_sample_size(price):
    """Taking the global minimum n let the shortest series set the bar for every
    feature — needlessly conservative for the long ones."""
    rng = np.random.default_rng(9)
    long_f = pd.Series(rng.normal(0, 1, len(price)), index=price.index)
    short_f = long_f.copy()
    short_f.iloc[:2000] = np.nan                       # only ~500 usable rows
    feats = pd.DataFrame({"long_feature": long_f, "short_feature": short_f})
    xc, counts = gd.cross_correlations(feats, price, max_lag=10)
    assert counts["long_feature"] > counts["short_feature"] * 2
    summary, _, _ = gd.lead_lag_summary(xc, counts, n_tests=len(xc) * xc.shape[1])
    s = summary.set_index("feature")
    assert s.loc["long_feature", "threshold_raw"] < s.loc["short_feature", "threshold_raw"], \
        "the longer series must earn a lower significance bar"


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------
def _stability_of(feature: pd.Series, target: pd.Series) -> pd.Series:
    feats = pd.DataFrame({"f": feature})
    targets = pd.DataFrame({"fwd_ret_5d": target, "fwd_ret_60d": target,
                            "fwd_ret_250d": target})
    out = gd.rolling_stability(feats, targets, window=252)
    return out[out["target"] == "fwd_ret_60d"].iloc[0]


def test_one_sided_metric_distinguishes_stable_from_flipping(price):
    idx = price.index
    rng = np.random.default_rng(13)
    y = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)

    stable = _stability_of(y * 2 + rng.normal(0, 0.002, len(idx)), y)
    assert stable["one_sided"] > 0.95, "a persistent relationship must read as one-sided"

    # a relationship whose sign alternates by year
    flip = pd.Series(np.where((np.arange(len(idx)) // 252) % 2 == 0, 1.0, -1.0), index=idx)
    flipping = _stability_of(y * flip, y)
    assert flipping["one_sided"] < 0.75, "an alternating relationship must not read stable"
    assert 0.0 <= flipping["share_positive"] <= 1.0


def test_stability_is_nan_safe_for_a_constant_feature(price):
    """A feature that is constant over a window gives an undefined correlation.
    gold_fx_breadth does this in real data and produced NaN min/max before the fix."""
    idx = price.index
    rng = np.random.default_rng(17)
    y = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    const = pd.Series(1.0, index=idx)
    const.iloc[1500:] = 0.0                            # constant within most windows
    out = gd.rolling_stability(pd.DataFrame({"f": const}),
                               pd.DataFrame({"fwd_ret_5d": y, "fwd_ret_60d": y,
                                             "fwd_ret_250d": y}), window=252)
    if len(out):
        assert np.isfinite(out["roll_min"]).all(), "NaN leaked into the stability stats"
        assert np.isfinite(out["one_sided"]).all()


def test_priors_are_recorded_but_never_used_to_flip_a_result():
    """The prior sign is reported for comparison only. Refitting a prior to the
    sample is how a backtest gets fitted to its own noise."""
    assert gd.PRIOR_SIGN["real_yield_10y_chg_20d"] == -1
    assert gd.PRIOR_SIGN["etf_tonnage_chg_4w"] == +1
    src = (gd.__file__)
    text = open(src, encoding="utf-8").read()
    assert "PRIOR_SIGN" in text
    # the diagnostics must not multiply any correlation by its prior
    assert "* PRIOR_SIGN" not in text and "PRIOR_SIGN[f] *" not in text
