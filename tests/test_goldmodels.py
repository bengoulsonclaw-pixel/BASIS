"""Milestone 6 locks — Stage 2 elastic net (spec §6 Stage 2, §7.3).

Two failures here would be invisible and expensive:

  * **An unpurged CV split.** Tuning alpha on folds whose labels overlap the
    validation block selects a weaker penalty than the data supports, and the model
    then looks better in-sample than it can ever be out of it.
  * **OLS standard errors on overlapping returns.** Residuals of an h-day regression
    are autocorrelated by construction; without the Newey-West correction the errors
    are understated by roughly sqrt(h), turning a t of 1.2 into a t of 4.

Also locked: that a boundary CV solution is flagged rather than reported as a choice,
and that everything the model learns is computed on the training slice alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import goldmodels as gm


# ---------------------------------------------------------------------------
# purged cross-validation
# ---------------------------------------------------------------------------
def test_purged_splits_never_let_a_label_reach_the_validation_block():
    """A training row at position p owns the label window [p, p+h]. If any part of
    that window touches the validation block, the row must be excluded."""
    n, h = 2000, 60
    splits = list(gm.purged_cv_splits(n, n_splits=5, horizon=h, embargo=h))
    assert splits, "no splits produced"
    for train, val in splits:
        v0, v1 = int(val.min()), int(val.max())
        assert len(np.intersect1d(train, val)) == 0, "train and val overlap directly"
        for p in train:
            label_end = p + h
            # every training label must close before the block or start after it,
            # with the embargo respected on both sides
            assert label_end < v0 - h or p > v1 + h, (
                f"row {p} (label to {label_end}) leaks into validation [{v0},{v1}]")


def test_purge_removes_more_rows_than_a_naive_split():
    """The purge must actually cost sample. If it doesn't, it isn't doing anything."""
    n, h = 2000, 60
    purged = list(gm.purged_cv_splits(n, 5, h, embargo=h))
    light = list(gm.purged_cv_splits(n, 5, h, embargo=0))
    assert sum(len(t) for t, _ in purged) < sum(len(t) for t, _ in light)


def test_splits_are_contiguous_blocks_not_shuffled():
    """Spec §12 forbids random splits anywhere. Shuffling a time series destroys the
    dependence the purge exists to respect."""
    for _, val in gm.purged_cv_splits(2000, 5, 60):
        assert np.array_equal(val, np.arange(val.min(), val.max() + 1)), \
            "validation block is not contiguous"


def test_no_splits_when_the_window_is_too_short():
    """Better to return nothing than to return folds too small to mean anything —
    the model then falls back to a default alpha rather than tuning on noise."""
    assert list(gm.purged_cv_splits(100, 5, 60)) == []


# ---------------------------------------------------------------------------
# Newey-West
# ---------------------------------------------------------------------------
def test_newey_west_exceeds_ols_when_regressors_and_residuals_are_both_persistent():
    """The whole reason the correction exists — and the setup matters.

    HAC corrects autocorrelation in the SCORE, X_t * u_t, not in the residual alone.
    With white-noise regressors the score stays near-white even when residuals are
    heavily autocorrelated, because X flips sign randomly and the products decorrelate
    — an earlier version of this test used i.i.d. X and found HAC errors SMALLER than
    OLS, correctly.

    The real feature matrix is nothing like that: rolling z-scores, distances from
    moving averages and 20-day changes are all strongly persistent, and they are
    regressed on overlapping returns. Both legs persistent is exactly when the
    correction bites, which is why it is not optional here."""
    rng = np.random.default_rng(5)
    n, h = 1500, 60
    # persistent regressors, like every feature in the real matrix
    raw_x = rng.normal(0, 1, (n + h, 3))
    X = np.array([raw_x[i:i + h].mean(axis=0) for i in range(n)])
    # overlapping-return residuals
    raw_u = rng.normal(0, 1, n + h)
    resid = np.array([raw_u[i:i + h].mean() for i in range(n)])
    nw = gm.newey_west_se(X, resid, lags=h)
    s2 = float(resid @ resid) / (n - X.shape[1])
    ols = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    assert np.all(nw > ols), "HAC errors must exceed OLS when both legs are persistent"
    assert np.mean(nw / ols) > 2.0, "correction is implausibly small for h=60"


def test_newey_west_matches_ols_when_residuals_are_white():
    """At lag 0 with independent residuals the two must agree closely — otherwise the
    estimator is biased, not just conservative."""
    rng = np.random.default_rng(6)
    n = 4000
    X = rng.normal(0, 1, (n, 3))
    resid = rng.normal(0, 1, n)
    nw = gm.newey_west_se(X, resid, lags=0)
    s2 = float(resid @ resid) / n
    ols = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    assert np.allclose(nw, ols, rtol=0.05)


def test_newey_west_is_degenerate_safe():
    X = np.ones((3, 5))
    assert np.isnan(gm.newey_west_se(X, np.zeros(3), lags=10)).all()


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@pytest.fixture()
def frame():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    rng = np.random.default_rng(11)
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, len(idx)) for i in range(6)}, index=idx)
    # f0 genuinely drives y; the rest are noise
    y = pd.Series(0.6 * X["f0"] + rng.normal(0, 0.5, len(idx)), index=idx)
    return X, y


def test_elastic_net_recovers_a_real_driver_and_drops_the_noise(frame):
    X, y = frame
    m = gm.elastic_net_for(20)()
    m.fit(X, y)
    assert m.coef_["f0"] > 0.2, "the real driver should survive regularisation"
    noise = m.coef_.drop("f0").abs()
    assert noise.max() < m.coef_["f0"] / 2, "noise features carry too much weight"
    assert abs(m.t_["f0"]) > 2, "a genuine driver should clear |t| > 2"


def test_everything_learned_comes_from_the_training_slice(frame):
    """The gap-filling median, the standardisation and the chosen alpha must all be
    fitted on the rows the harness handed over — never on the full frame."""
    X, y = frame
    cut = 900
    m = gm.elastic_net_for(20)()
    m.fit(X.iloc[:cut], y.iloc[:cut])
    assert np.allclose(m.median_.to_numpy(),
                       X.iloc[:cut][m.cols_].median().to_numpy(), equal_nan=True)
    assert np.allclose(m.mu_x_.to_numpy(),
                       X.iloc[:cut][m.cols_].fillna(m.median_).mean().to_numpy())
    # predicting on later rows must not change anything the model learned
    before = m.coef_.copy()
    m.predict(X.iloc[cut:])
    pd.testing.assert_series_equal(before, m.coef_)


def test_a_boundary_cv_solution_is_flagged(frame):
    """If the CV optimum sits at the edge of the grid, the grid chose the answer, not
    the data. The first version stopped at alpha=0.1 and CV picked exactly 0.1."""
    X, y = frame
    Narrow = gm.elastic_net_for(20, _alphas=(0.001, 0.003))
    m = Narrow()
    m.fit(X, y)
    assert m.alpha_at_boundary_, "a boundary solution must be flagged"

    Wide = gm.elastic_net_for(20, _alphas=gm.ALPHAS)
    w = Wide()
    w.fit(X, y)
    assert w.alpha_ in gm.ALPHAS


def test_alpha_grid_reaches_far_enough_to_shrink_everything():
    """The grid must be able to express 'use no features'. With every robust feature
    coincident, that is a live answer and the model has to be allowed to give it."""
    assert max(gm.ALPHAS) >= 1.0
    assert min(gm.ALPHAS) <= 0.001


def test_model_plugs_into_the_harness_interface():
    """Spec §12 compares every model against the four benchmarks through one harness,
    so the interface has to match exactly."""
    from src import goldbacktest as bt
    cls = gm.elastic_net_for(60)
    assert issubclass(cls, bt._Base)
    assert hasattr(cls, "name") and callable(cls().fit) and callable(cls().predict)
    assert cls._horizon == 60, "the Newey-West lag and purge must follow the horizon"
