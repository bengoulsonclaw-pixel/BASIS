"""Milestone 7 locks — Stage 3 bucket scores and the §8 output contract.

Stage 3's whole claim is that it needs less sample than Stage 2 because the
equal-weight variant fits nothing. That claim only holds if the code really does fit
nothing, so the first test asserts it directly rather than trusting the description.

The §8 contract is a machine-readable payload that a human will act on, so the tests
also cover the parts that would mislead rather than crash: a bounded score that isn't
bounded, a disagreement figure that doesn't measure disagreement, and — most
importantly — the flag that says this model does not beat buy-and-hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import goldbuckets as gb


@pytest.fixture()
def frame():
    idx = pd.bdate_range("2012-01-02", periods=1200)
    rng = np.random.default_rng(23)
    cols = {}
    for f, b in gb.goldfeatures.FEATURE_BUCKETS.items():
        cols[f] = rng.normal(0, 1, len(idx))
    X = pd.DataFrame(cols, index=idx)
    y = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    return X, y


# ---------------------------------------------------------------------------
# the equal-weight variant fits nothing
# ---------------------------------------------------------------------------
def test_equal_weight_scores_do_not_depend_on_the_target(frame):
    """The core claim of Stage 3. If the equal-weight scores move when the target
    changes, something is being fitted and the 'no parameters, no sample needed'
    argument collapses."""
    X, y = frame
    m1 = gb.bucket_model_for(60, "equal", refit=False)()
    m1.fit(X, y)
    s1 = m1.bucket_scores(X)

    m2 = gb.bucket_model_for(60, "equal", refit=False)()
    m2.fit(X, -y * 7.0 + 3.0)               # a completely different target
    s2 = m2.bucket_scores(X)
    pd.testing.assert_frame_equal(s1, s2)


def test_net_variant_does_depend_on_the_target(frame):
    """The contrast case — the per-bucket elastic net is supposed to fit."""
    X, y = frame
    a = gb.bucket_model_for(60, "net", refit=False)()
    a.fit(X, y)
    b = gb.bucket_model_for(60, "net", refit=False)()
    b.fit(X, X[list(X.columns)[0]] * 3.0)
    assert not np.allclose(a.bucket_scores(X).to_numpy(),
                           b.bucket_scores(X).to_numpy())


def test_scores_are_bounded_to_minus_one_plus_one(frame):
    """Spec §6 Stage 3: 'five bucket scores in the range minus one to plus one'.
    Tested with an absurd input, because the bound is what makes the weights
    interpretable as weights."""
    X, y = frame
    m = gb.bucket_model_for(60, "equal", refit=False)()
    m.fit(X, y)
    extreme = X.copy() * 500.0
    s = m.bucket_scores(extreme)
    assert s.to_numpy().min() >= -1.0 and s.to_numpy().max() <= 1.0
    assert set(s.columns) == set(gb.BUCKETS)


def test_prior_signs_are_applied_within_a_bucket(frame):
    """A feature whose prior is negative must push its bucket DOWN when it rises."""
    X, y = frame
    m = gb.bucket_model_for(60, "equal", refit=False)()
    m.fit(X, y)
    neg = [c for c in m.cols_
           if m.buckets_.get(c) == "monetary" and gb.PRIOR_SIGN.get(c, 0) == -1]
    assert neg, "no negative-prior monetary feature to test with"
    base = m.bucket_scores(X.iloc[[-1]])["monetary"].iloc[0]
    bumped = X.iloc[[-1]].copy()
    bumped[neg] = bumped[neg] + 5.0
    after = m.bucket_scores(bumped)["monetary"].iloc[0]
    assert after < base, "raising a negative-prior feature should lower its bucket"


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def test_fixed_weights_match_the_spec_table():
    """Spec §6 Stage 3's starting values, per horizon."""
    assert gb.FIXED_WEIGHTS[5]["flows"] == 0.30
    assert gb.FIXED_WEIGHTS[60]["monetary"] == 0.40
    assert gb.FIXED_WEIGHTS[250]["physical"] == 0.30
    for h, w in gb.FIXED_WEIGHTS.items():
        assert set(w) == set(gb.BUCKETS)
        assert sum(w.values()) == pytest.approx(1.0), f"horizon {h} weights must sum to 1"


def test_refitted_weights_are_normalised_and_can_go_negative(frame):
    """Normalised so the fitted set is comparable to the fixed set in shape rather
    than magnitude. Negative weights are allowed on purpose: forcing non-negativity
    would hide a bucket that works backwards, which is exactly what the comparison
    against the spec's table is meant to reveal."""
    X, y = frame
    m = gb.bucket_model_for(60, "equal", refit=True)()
    m.fit(X, y)
    assert sum(abs(v) for v in m.weights_.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(m.weights_) == set(gb.BUCKETS)
    assert m.weights_fixed_ == gb.FIXED_WEIGHTS[60], "the fixed set must be preserved"


def test_fixed_variant_leaves_the_spec_weights_alone(frame):
    X, y = frame
    m = gb.bucket_model_for(60, "equal", refit=False)()
    m.fit(X, y)
    assert m.weights_ == gb.FIXED_WEIGHTS[60]


def test_everything_is_learned_from_the_training_slice(frame):
    X, y = frame
    cut = 800
    m = gb.bucket_model_for(60, "equal", refit=True)()
    m.fit(X.iloc[:cut], y.iloc[:cut])
    assert np.allclose(m.mu_.to_numpy(),
                       X.iloc[:cut][m.cols_].fillna(m.median_).mean().to_numpy())
    before = dict(m.weights_)
    m.predict(X.iloc[cut:])
    assert m.weights_ == before, "predicting must not update anything fitted"


# ---------------------------------------------------------------------------
# the §8 contract
# ---------------------------------------------------------------------------
def _contract(monkeypatch, beats: bool = False) -> dict:
    return gb.build_contract("fwd_ret_60d", beats_benchmark=beats)


def test_contract_carries_every_field_the_spec_requires():
    c = gb.build_contract("fwd_ret_60d")
    for field in ("as_of", "horizon", "probability_up", "expected_return",
                  "confidence", "regime", "buckets", "disagreement", "flags"):
        assert field in c, f"§8 requires {field}"
    assert set(c["buckets"]) == set(gb.BUCKETS)
    for b, blk in c["buckets"].items():
        for k in ("score", "weight", "contribution", "top_drivers"):
            assert k in blk, f"bucket {b} missing {k}"
        assert blk["contribution"] == pytest.approx(blk["score"] * blk["weight"],
                                                    abs=1e-3)
    assert 0.0 <= c["probability_up"] <= 1.0


def test_disagreement_is_the_spread_of_bucket_scores():
    """Spec §8: 'disagreement is the standard deviation of the bucket scores'. It is
    a sizing signal, not a direction, so it must actually measure spread."""
    c = gb.build_contract("fwd_ret_60d")
    scores = [c["buckets"][b]["score"] for b in gb.BUCKETS]
    assert c["disagreement"] == pytest.approx(float(np.std(scores)), abs=1e-3)


def test_contract_flags_that_the_model_does_not_beat_buy_and_hold():
    """The single most important field in the payload. Every §12 comparison so far
    says this engine does not beat always-long, and a consumer reading only
    probability_up would never learn that."""
    c = gb.build_contract("fwd_ret_60d", beats_benchmark=False)
    assert "model_does_not_beat_always_long_benchmark" in c["flags"]
    assert c["flags"][0] == "model_does_not_beat_always_long_benchmark", \
        "the caveat must lead, not be buried mid-list"
    assert c["confidence"] == "low", "a model that loses to buy-and-hold is not confident"


def test_regime_is_not_invented_before_the_regime_layer_exists():
    """Spec §8's example shows regime='central_bank_led'. Emitting a plausible label
    from a classifier that has not been built would be fabrication."""
    c = gb.build_contract("fwd_ret_60d")
    assert "unclassified" in c["regime"].lower()


def test_missing_features_surface_as_flags():
    c = gb.build_contract("fwd_ret_60d")
    joined = " ".join(c["flags"])
    for fid in gb.goldfeatures.MISSING_FEATURES:
        assert f"feature_unavailable:{fid}" in joined


def test_all_four_variants_plug_into_the_harness():
    from src import goldbacktest as bt
    models = gb.all_bucket_models(60)
    assert len(models) == 4
    names = {m.name for m in models}
    assert names == {"bucket_equal_fixed", "bucket_equal_fitted",
                     "bucket_net_fixed", "bucket_net_fitted"}
    for m in models:
        assert issubclass(m, bt._Base)
        assert m._horizon == 60
