"""Locks on the Gold Driver Model's maths (2026-08-22 build).

The four things here that would cost real money if they broke silently:

  * NO LOOKAHEAD in the feature matrix. Every column is built from rolling/shifted
    operations; if one of them ever picks up a centred window or a bfill, the fit
    starts scoring on information it could not have had and every backtest number
    in the module becomes a lie that still looks plausible.
  * THE OVERLAP CORRECTION on t-stats. Overlapping 21-day returns inflate naive
    t-stats by ~sqrt(21). Losing this turns "no reliable edge" into "t = 4".
  * SENSITIVITY RECOVERY. The fit must return the coefficient that generated the
    data, in the units the sensitivity table quotes.
  * THE SHANGHAI UNIT CONVERSION. CNY/gram to USD/oz is two multiplications and a
    divide, and getting it wrong produces a premium series that is wrong by a
    factor of 31 while still looking like a plausible time series.

Pure maths on synthetic frames — no network, no repo stores, no Bloomberg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import golddata, goldmodel as gm


# ---------------------------------------------------------------------------
# a deterministic synthetic driver frame
# ---------------------------------------------------------------------------
def _frame(n: int = 1600, seed: int = 7) -> pd.DataFrame:
    """Gold built so that its 21-day return is a KNOWN linear function of the
    21-day changes in real yields and the dollar, plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    real = pd.Series(2.0 + np.cumsum(rng.normal(0, 0.02, n)), index=idx)
    dxy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=idx)

    # Daily gold log return: -0.05 per pp of real yield (i.e. -0.50% per 10bp, the
    # magnitude the live fit actually returns) and a -0.8 beta to the dollar.
    # Calibrating to the real coefficient matters: at an arbitrary -5.0 per pp a
    # 10bp move implies a 39% gold move, and the test would be locking maths that
    # never occurs in the data it guards.
    d_real = real.diff().fillna(0.0)
    d_dxy = np.log(dxy / dxy.shift(1)).fillna(0.0)
    g = np.exp(np.cumsum(-0.05 * d_real - 0.8 * d_dxy + rng.normal(0, 0.0004, n)))
    gold = pd.Series(1500.0 * g, index=idx)

    F = pd.DataFrame({
        "gold": gold, "gold_raw": gold, "silver": gold / 80.0,
        "copper": pd.Series(400.0, index=idx),
        "gold_silver_ratio": pd.Series(80.0, index=idx),
        "real_10y": real, "real_5y": real - 0.2,
        "breakeven_10y": pd.Series(2.3 + np.cumsum(rng.normal(0, 0.005, n)), index=idx),
        "nominal_10y": real + 2.3, "nominal_2y": real + 1.9,
        "fed_funds": pd.Series(3.5, index=idx),
        "cuts_priced": pd.Series(np.cumsum(rng.normal(0, 0.004, n)), index=idx),
        "dxy": dxy, "usdcny": pd.Series(7.0, index=idx),
        "vix": pd.Series(18.0 + np.cumsum(rng.normal(0, 0.05, n)), index=idx),
        "credit_baa": pd.Series(1.8 + np.cumsum(rng.normal(0, 0.004, n)), index=idx),
        "gld_tonnes": pd.Series(900.0 + np.cumsum(rng.normal(0, 1.0, n)), index=idx),
        "mm_net": pd.Series(150000.0 + np.cumsum(rng.normal(0, 900, n)), index=idx),
        "cot_index": pd.Series(50.0 + np.cumsum(rng.normal(0, 0.4, n)), index=idx).clip(0, 100),
        "cb_net_12m": pd.Series(400.0 + np.cumsum(rng.normal(0, 1.0, n)), index=idx),
        "sge_premium": pd.Series(np.cumsum(rng.normal(0, 0.4, n)), index=idx),
        "gc_carry": pd.Series(20.0 + np.cumsum(rng.normal(0, 0.05, n)), index=idx),
        "efp": pd.Series(np.cumsum(rng.normal(0, 0.6, n)), index=idx),
        "miners_vs_bullion": pd.Series(0.2 * g, index=idx),
        "lbma_usd": gold, "gold_eur": gold / 1.1, "gold_gbp": gold / 1.27,
        "gold_jpy": gold * 150.0, "gold_inr": gold * 85.0,
    })
    return F


# ---------------------------------------------------------------------------
# 1. no lookahead
# ---------------------------------------------------------------------------
def test_features_use_no_future_information():
    """Truncating the frame must not change any feature value that survives.

    This is the single most important lock in the file. Every feature is built
    from rolling/shifted operations, so a row's value can depend only on that row
    and earlier ones. If a centred window or a bfill ever creeps in, this fails."""
    F = _frame()
    full = gm.features(F)
    cut = gm.features(F.iloc[:1200])
    common = cut.dropna(how="all").index
    a, b = full.loc[common], cut.loc[common]
    for col in a.columns:
        pair = pd.concat([a[col], b[col]], axis=1).dropna()
        if pair.empty:
            continue
        assert np.allclose(pair.iloc[:, 0], pair.iloc[:, 1], equal_nan=True), \
            f"{col} changed when later data was removed — lookahead"


def test_target_looks_forward_and_trailing_looks_back():
    F = _frame(400)
    h = 21
    fwd, trail = gm.target(F, h), gm.trailing(F, h)
    g = F["gold"]
    assert fwd.iloc[0] == pytest.approx(np.log(g.iloc[h] / g.iloc[0]))
    assert trail.iloc[h] == pytest.approx(np.log(g.iloc[h] / g.iloc[0]))
    # the last h targets are unknowable and must be NaN, never filled
    assert fwd.iloc[-h:].isna().all()


# ---------------------------------------------------------------------------
# 2. the overlap correction
# ---------------------------------------------------------------------------
def test_overlap_t_uses_effective_sample_not_row_count():
    """A 0.15 IC on 2,100 overlapping 21-day rows is t≈1.5, not t≈6.9."""
    naive = 0.15 * np.sqrt(2100 - 2) / np.sqrt(1 - 0.15 ** 2)
    corrected = gm.overlap_t(0.15, 2100, 21)
    assert naive > 6.5, "test's own naive baseline drifted"
    assert corrected == pytest.approx(0.15 * np.sqrt(100 - 2) / np.sqrt(1 - 0.15 ** 2))
    assert abs(corrected) < 2.0, "overlap correction is not being applied"
    # h=1 (non-overlapping) must reduce to the naive statistic
    assert gm.overlap_t(0.15, 2100, 1) == pytest.approx(naive)


# ---------------------------------------------------------------------------
# 3. sensitivity recovery
# ---------------------------------------------------------------------------
def test_fit_recovers_the_generating_coefficients():
    """Gold was generated at -0.05 per pp of real yield => -0.50% per 10bp, and
    with a -0.8 beta to the dollar => -0.80% per 1%. The ridge shrinks slightly,
    so the tolerance is loose on magnitude and strict on sign."""
    F = _frame()
    X = gm.features(F)
    fit = gm.fit_coincident(F, X, 21, cols=["real10_chg21", "dxy_ret21"])
    s = gm.sensitivities(fit).set_index("feature")
    real = s.loc["real10_chg21", "gold_pct"]
    dxy = s.loc["dxy_ret21", "gold_pct"]
    assert real < 0 and dxy < 0
    assert real == pytest.approx(-0.50, abs=0.15), f"real yield sensitivity {real}"
    assert dxy == pytest.approx(-0.80, abs=0.25), f"dollar sensitivity {dxy}"
    assert fit["r2"] > 0.8, "synthetic data is nearly deterministic; fit should be tight"


def test_scenario_is_linear_and_matches_the_sensitivity_table():
    F = _frame()
    X = gm.features(F)
    fit = gm.fit_coincident(F, X, 21, cols=["real10_chg21", "dxy_ret21"])
    one = gm.scenario(fit, {"real10_chg21": 0.10})["gold_pct"]
    table = gm.sensitivities(fit).set_index("feature").loc["real10_chg21", "gold_pct"]
    assert one == pytest.approx(table, abs=1e-9)
    # doubling the shock doubles the log contribution (not the percentage)
    two = gm.scenario(fit, {"real10_chg21": 0.20})["gold_pct"]
    assert np.log1p(two / 100) == pytest.approx(2 * np.log1p(one / 100), rel=1e-9)
    # an unspecified driver contributes exactly zero, never a sample mean
    assert gm.scenario(fit, {})["gold_pct"] == pytest.approx(0.0, abs=1e-9)


def test_flow_drivers_are_excluded_from_the_default_fit():
    """The scenario/sensitivity fit must stay exogenous. ETF tonnage and
    managed-money net are co-movers; letting them in halves the measured
    real-yield coefficient (0.31 -> 0.47 R2, -0.51%/10bp -> -0.12%)."""
    assert "gld_chg21" not in gm.MACRO_DRIVERS
    assert "mm_chg21" not in gm.MACRO_DRIVERS
    F = _frame()
    fit = gm.fit_coincident(F, gm.features(F), 21)
    assert set(fit["cols"]).isdisjoint(gm.FLOW_DRIVERS)


# ---------------------------------------------------------------------------
# 4. the Shanghai unit conversion
# ---------------------------------------------------------------------------
def test_sge_premium_converts_cny_per_gram_to_usd_per_ounce():
    """900 CNY/g at 7.0 USDCNY is 900 * 31.1034768 / 7 = $3,999.02/oz. Against a
    $3,950 London fix that is a +$49.02 premium — not $1.58 (grams left in) and
    not $27,993 (the divide dropped)."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    sge = pd.DataFrame({"sge_cny_g": pd.Series(900.0, index=idx)})
    lbma = pd.DataFrame({"lbma_am_usd": pd.Series(3950.0, index=idx)},)
    cny = pd.Series(7.0, index=idx)
    prem = golddata.sge_premium(sge, lbma, cny, smooth=5)
    expected = 900.0 * golddata.OZ_PER_GRAM / 7.0 - 3950.0
    assert expected == pytest.approx(49.02, abs=0.05)
    assert prem.dropna().iloc[-1] == pytest.approx(expected, abs=1e-6)


def test_sge_premium_survives_a_missing_leg():
    """A dead source must return empty, never a garbage series — the model can
    drop a feature but must not fit on nonsense."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    assert golddata.sge_premium(pd.DataFrame(), pd.DataFrame(),
                                pd.Series(dtype=float)).empty
    assert golddata.sge_premium(pd.DataFrame({"sge_cny_g": pd.Series(900.0, index=idx)}),
                                pd.DataFrame(), None).empty


# ---------------------------------------------------------------------------
# 5. the composite and the fair-value gap
# ---------------------------------------------------------------------------
def test_composite_has_no_fitted_parameters():
    """Flipping every feature's sign must flip the composite exactly. If any
    coefficient were fitted, the mapping would not be a clean sign flip."""
    F = _frame()
    X = gm.features(F)
    comp, blocks = gm.composite(X)
    assert set(blocks.columns) <= set(gm.BLOCK_ORDER)
    flipped, _ = gm.composite(-X)
    both = pd.concat([comp, flipped], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], -both.iloc[:, 1], atol=1e-9)


def test_fair_value_gap_is_walk_forward_only():
    """Every residual must come from a fit whose training window closed before
    the measured window opened — so appending future rows cannot change a gap
    value that already exists."""
    F = _frame(1900)
    X = gm.features(F)
    gap_full = gm.fair_value(F, X, 21, min_train=600, months=6)
    F2, X2 = F.iloc[:1500], gm.features(F.iloc[:1500])
    gap_cut = gm.fair_value(F2, X2, 21, min_train=600, months=6)
    common = gap_full.index.intersection(gap_cut.index)
    assert len(common) > 5, "not enough overlap to test"
    # equal_nan: the first `months`-1 points are legitimately NaN in both series
    # (the rolling sum has no full window yet), and NaN != NaN would fail a match
    # that is in fact exact.
    assert np.allclose(gap_full.loc[common], gap_cut.loc[common],
                       atol=1e-9, equal_nan=True), \
        "fair-value gap changed when future data was appended — lookahead"
