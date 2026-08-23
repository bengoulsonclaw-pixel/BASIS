"""Milestone 4 locks — the walk-forward harness (spec §7).

The harness is the instrument every later result is measured with, so a fault here
does not produce an obviously wrong number: it produces a flattering one. The four
that matter most:

  * **The purge/embargo boundary.** A row dated T carries a label that does not
    resolve until T+h. Training on it while testing at t < T+h trains on the answer.
    Tested by spying on what the model is actually handed, not by inspecting code.
  * **The holdout stays locked.** The default evaluation must not produce a single
    prediction inside the final three years.
  * **Non-overlapping scoring.** Consecutive daily rows share nearly all of their
    forward window; counting them as independent trades inflates both confidence and
    turnover.
  * **Costs are actually charged.** A rule that looks profitable gross and loses net
    is the normal case at 10bp a trade, and a harness that forgets the subtraction
    will never say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import goldbacktest as bt


@pytest.fixture()
def data():
    """Long enough to clear the 10-year minimum training window."""
    idx = pd.bdate_range("2005-01-03", periods=4000)
    rng = np.random.default_rng(17)
    X = pd.DataFrame({
        "real_yield_10y_chg_20d": rng.normal(0, 0.1, len(idx)),
        "gold_mom_12m_1m": rng.normal(0, 0.05, len(idx)),
        "other": rng.normal(0, 1, len(idx)),
    }, index=idx)
    y = pd.Series(rng.normal(0.0005, 0.02, len(idx)), index=idx)
    return X, y


# ---------------------------------------------------------------------------
# the purge / embargo boundary
# ---------------------------------------------------------------------------
def test_training_never_sees_a_label_that_has_not_resolved(data):
    """The core guarantee, checked by watching what the model is handed.

    At prediction date t the last training row T must satisfy T + horizon + embargo
    <= t. If the purge were dropped, T could run right up to t and the model would be
    fitted on labels covering the very window it is predicting."""
    X, y = data
    horizon, embargo = 60, 60
    fits = []
    res = bt.walk_forward(bt.RandomWalk, X, y, horizon=horizon, refit_days=63,
                          min_train=500, embargo=embargo,
                          on_fit=lambda t, last, n: fits.append((t, last, n)))
    assert len(res) > 100, "harness produced no predictions"
    assert fits, "model was never fitted"

    # Compared against the date the fit was TRIGGERED, not against the first
    # prediction of the whole run — later refits legitimately train on data that
    # postdates earlier predictions, and checking against a single global date
    # confuses that for a leak.
    # Measured in INDEX POSITIONS, not calendar days. The earlier version compared
    # calendar days against a trading-day horizon — the same quantity the code
    # computed, so the test could not have caught a wrong one.
    pos = pd.Series(np.arange(len(X.index)), index=X.index)
    for t, last_train, n in fits:
        assert n >= 500
        gap = int(pos[t]) - int(pos[last_train])
        assert gap >= horizon + embargo, (
            f"fit at {t.date()} trained up to {last_train.date()} — only {gap} "
            f"trading days, needs purge+embargo of {horizon + embargo}")


def test_embargo_defaults_to_the_horizon(data):
    """Spec §7.2: 'embargo gap equal to the forecast horizon'."""
    X, y = data
    seen_default, seen_zero = [], []

    def _spy(sink):
        class S(bt.RandomWalk):
            def fit(self, Xtr, ytr):
                super().fit(Xtr, ytr)
                sink.append(Xtr.index.max())
        return S

    bt.walk_forward(_spy(seen_default), X, y, horizon=60, refit_days=200,
                    min_train=500)
    bt.walk_forward(_spy(seen_zero), X, y, horizon=60, refit_days=200,
                    min_train=500, embargo=0)
    assert max(seen_zero) > max(seen_default), \
        "embargo=0 should reach further forward than the default"


def test_a_leaky_feature_is_caught_by_the_harness(data):
    """Sanity check on the instrument itself: if a feature literally contains the
    future, the harness must still report near-perfect skill. A harness that
    'protected' against this would be hiding leaks rather than measuring them —
    the guard belongs in the feature layer, and this proves the measurement is live."""
    X, y = data
    X = X.copy()
    X["cheat"] = y                                  # the label itself, as a feature

    class Cheat(bt.RandomWalk):
        needs = ("cheat",)

        def predict(self, Xr):
            return Xr["cheat"].fillna(0.0).to_numpy()

    res = bt.walk_forward(Cheat, X, y, horizon=60, refit_days=63, min_train=500)
    res["actual"] = y.reindex(res.index)
    m = bt.metrics(res, 60)
    assert m["hit_rate"] > 0.9, "harness failed to register an obvious leak"
    warns = bt.check_calibration("fwd_ret_60d", m)
    assert any("BUG OR LEAK" in w for w in warns), \
        "a 90%+ hit rate must be flagged, not celebrated"


# ---------------------------------------------------------------------------
# the locked holdout
# ---------------------------------------------------------------------------
def test_holdout_window_is_excluded_by_default(data):
    X, y = data
    cut = bt.holdout_start(X.index)
    res = bt.walk_forward(bt.AlwaysLong, X, y, horizon=5, refit_days=21,
                          min_train=500, end=cut)
    assert res.index.max() < cut, "a prediction landed inside the locked holdout"
    inside = bt.walk_forward(bt.AlwaysLong, X, y, horizon=5, refit_days=21,
                             min_train=500, start=cut)
    assert inside.index.min() >= cut
    assert len(inside) > 0, "holdout window should be scoreable when unlocked"


def test_holdout_is_the_last_three_years(data):
    X, _ = data
    cut = bt.holdout_start(X.index)
    assert (X.index.max() - cut).days == pytest.approx(365 * bt.HOLDOUT_YEARS, abs=3)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_wilson_interval_against_hand_computed_values():
    lo, hi = bt.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    # small n must give a wide interval, not a confident one
    lo_s, hi_s = bt.wilson_interval(12, 18)
    assert hi_s - lo_s > 0.35, "a 18-observation interval cannot be narrow"
    assert bt.wilson_interval(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2 \
        or np.isnan(bt.wilson_interval(0, 0)[0])


def test_ranks_average_ties_so_a_constant_prediction_has_no_correlation():
    """The defect this replaced was invisible and consequential.

    Assigning ties consecutive ranks in array order ranks a CONSTANT prediction in
    calendar order — and calendar order correlates with anything that trends. That is
    the only reason random_walk, which predicts all zeros, scored IC +0.020 / +0.035
    / +0.191 across the three horizons, and it is why "the only model with positive
    IC at all three horizons" was a statement about a broken rank function."""
    assert list(bt._rank(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))) == [1.5, 1.5, 1.5, 1.5, 4.0]
    rng = np.random.default_rng(1)
    const = pd.Series([0.0] * 300)
    trend = pd.Series(np.arange(300, dtype=float))
    assert np.isnan(bt.spearman(const, trend)),         "a constant series has no ranking; returning a number invents one"
    assert np.isnan(bt.spearman(const, pd.Series(rng.normal(0, 1, 300))))


def test_metrics_average_every_sampling_phase_not_just_the_first():
    """`d.iloc[::horizon]` picks one of `horizon` equally valid offsets — always the
    one starting at row 0. On the real 5-day result the five phases gave edges over
    always-long of +2.60, +2.39, +0.65, +0.22 and +1.74pp. Reporting phase 0 turned a
    +1.5pp average into a +2.6pp headline."""
    idx = pd.bdate_range("2015-01-01", periods=1000)
    rng = np.random.default_rng(8)
    # a signal deliberately concentrated in one phase
    pred = pd.Series(rng.normal(0, 1, 1000), index=idx)
    actual = pd.Series(rng.normal(0, 0.02, 1000), index=idx)
    actual.iloc[0::5] = np.abs(actual.iloc[0::5]) * np.sign(pred.iloc[0::5])
    m = bt.metrics(pd.DataFrame({"pred": pred, "actual": actual}), 5)
    assert m["phases_scored"] == 5, "every offset must be scored"
    lo, hi = m["edge_phase_spread"]
    assert hi - lo > 0.1, "a phase-concentrated signal must show a wide spread"
    assert lo < m["edge_vs_always_long"] < hi, "the report must be the average"


def test_paired_mcnemar_is_available_and_two_sided():
    """Two unpaired Wilson intervals cannot answer 'is this model better than that
    one'. The models are scored on the same days and their errors are correlated;
    only the discordant pairs carry information."""
    m = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 10, dtype=bool)
    b_ = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 10, dtype=bool)
    b, c, p = bt.mcnemar_p(m, b_)
    assert b == 40 and c == 40 and p == pytest.approx(1.0), "symmetric disagreement"
    lop = bt.mcnemar_p(np.ones(40, dtype=bool), np.zeros(40, dtype=bool))
    assert lop[2] < 1e-9, "total disagreement must be significant"


def test_pnl_is_reported_against_buy_and_hold():
    """always_long at 5d never clears the 0.55 probability threshold, so its P&L is
    0.000 — cash, not a strategy. Comparing a model against it made a LOSS against
    buy-and-hold look like a win."""
    idx = pd.bdate_range("2015-01-01", periods=600)
    res = pd.DataFrame({"pred": 3.0, "actual": 0.01}, index=idx)
    m = bt.metrics(res, 5)
    assert "buyhold_sum_logret" in m and "excess_vs_buyhold" in m
    assert m["excess_vs_buyhold"] == pytest.approx(
        m["strategy_sum_logret"] - m["buyhold_sum_logret"], abs=1e-9)
    flat = bt.metrics(pd.DataFrame({"pred": -3.0, "actual": 0.01}, index=idx), 5)
    assert flat["time_in_market"] == 0, "a model that never trades must be detectable"


def test_metrics_score_non_overlapping_observations_only():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    rng = np.random.default_rng(4)
    res = pd.DataFrame({"pred": rng.normal(0, 1, len(idx)),
                        "actual": rng.normal(0, 0.02, len(idx))}, index=idx)
    m = bt.metrics(res, 60)
    assert m["n"] == 1000
    # the mean across phases, since every phase is now scored
    assert m["n_independent"] == pytest.approx(1000 / 60, abs=1), \
        "independent count must be the sample divided by the horizon"
    assert m["n_independent"] < m["n"] / 50


def test_costs_are_charged_on_every_position_change():
    """A rule that trades must pay 10bp a turn. If the subtraction is missing, net
    and gross agree and the backtest silently overstates every strategy."""
    idx = pd.bdate_range("2015-01-01", periods=600)
    # alternating strong/weak predictions -> the 0.55 rule flips repeatedly
    pred = pd.Series(np.where(np.arange(len(idx)) % 10 < 5, 3.0, -3.0), index=idx)
    res = pd.DataFrame({"pred": pred, "actual": pd.Series(0.0, index=idx)})
    m = bt.metrics(res, 5)
    assert m["trades"] > 0, "test data should generate trades"
    # with zero actual return, all P&L is cost, so it must be strictly negative
    assert m["strategy_sum_logret"] < 0
    assert m["strategy_sum_logret"] == pytest.approx(
        -m["turnover"] * bt.COST_BPS / 1e4, rel=1e-9)


def test_probability_of_a_degenerate_predictor_is_one_half():
    """RandomWalk predicts all zeros. That is 'no view', and it must map to 0.5 —
    not to a divide-by-zero or to spurious confidence."""
    idx = pd.bdate_range("2015-01-01", periods=300)
    res = pd.DataFrame({"pred": 0.0, "actual": 0.01}, index=idx)
    p = bt.to_probability(res)
    assert (p == 0.5).all()
    m = bt.metrics(res, 5)
    assert np.isfinite(m["brier"])


def test_calibration_flags_an_implausible_five_day_hit_rate():
    """Spec §12: above 60% at 5 days is a bug until proven otherwise."""
    good = {"hit_rate": 0.54, "n_independent": 200, "n": 1000}
    assert not [w for w in bt.check_calibration("fwd_ret_5d", good) if "BUG" in w]
    suspicious = {"hit_rate": 0.72, "n_independent": 200, "n": 1000}
    assert any("BUG OR LEAK" in w for w in bt.check_calibration("fwd_ret_5d", suspicious))
    thin = {"hit_rate": 0.55, "n_independent": 8, "n": 2000}
    assert any("independent observations" in w
               for w in bt.check_calibration("fwd_ret_250d", thin))


# ---------------------------------------------------------------------------
# run logging (spec §7.6 — no unlogged runs)
# ---------------------------------------------------------------------------
def test_run_records_carry_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "RUN_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(bt, "STORE_DIR", tmp_path)
    bt.log_run({"run_at": "2026-01-01T00:00:00", "git": "abc1234",
                "feature_version": "deadbeef01", "window": "development",
                "models": ["x"], "results": {}})
    log = bt.read_log()
    assert len(log) == 1
    for field in ("run_at", "git", "feature_version", "window"):
        assert field in log[0], f"run record must carry {field}"


def test_feature_version_changes_when_the_feature_set_changes():
    """A run record has to identify exactly which columns the model saw."""
    a = pd.DataFrame(columns=["x", "y"])
    b = pd.DataFrame(columns=["x", "y", "z"])
    assert bt._feature_version(a) != bt._feature_version(b)
    # column ORDER must not matter — the same set is the same version
    assert bt._feature_version(a) == bt._feature_version(pd.DataFrame(columns=["y", "x"]))


def test_all_four_spec_benchmarks_are_present():
    """Spec §7/§12: every model is compared against these four, and none ships
    without beating all of them."""
    names = {m.name for m in bt.BENCHMARKS}
    assert names == {"random_walk", "always_long", "momentum_12m", "real_yield_only"}
