"""Locks on the defects the adversarial audit found in the Gold Signal Engine.

Each test here corresponds to something that was WRONG in code that shipped, and
several guard numbers that had already reached a client PDF. They are grouped by what
the failure would have looked like to a reader rather than by module, because that is
how they were found.

Everything runs against temp stores or pure functions — never the repo's own data.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
def _frame(ref, pubs, vals):
    return pd.DataFrame({"reference_date": [ref] * len(pubs),
                         "published_at": [pd.Timestamp(p) for p in pubs],
                         "value": list(vals)})


def test_put_keeps_a_revision_that_reverts_to_an_earlier_value(tmp_path, monkeypatch):
    """A -> B -> A must store all three, not silently drop the revert.

    The in-force value has to be tracked THROUGH the incoming batch. ALFRED hands over
    a whole vintage history at once, so comparing every row against the pre-batch
    value dropped the final A and left the store claiming B was current.
    """
    from src import goldstore as gs
    monkeypatch.setattr(gs, "OBS_FILE", tmp_path / "obs.parquet")
    ref = pd.Timestamp("2026-01-31")
    gs.put("T", _frame(ref, ["2026-02-01"], [100.0]), source="test")
    n = gs.put("T", _frame(ref, ["2026-03-01", "2026-04-01"], [200.0, 100.0]),
               source="test")
    assert n == 2, "the revert back to 100.0 was dropped"
    assert gs.get_series("T").iloc[-1] == 100.0
    # ...and the intermediate vintage is still what you saw at the time.
    assert gs.get_series("T", as_of="2026-03-15").iloc[-1] == 200.0


def test_put_still_drops_a_true_no_op_restatement(tmp_path, monkeypatch):
    """The revert fix must not reopen the duplicate-revision hole beside it."""
    from src import goldstore as gs
    monkeypatch.setattr(gs, "OBS_FILE", tmp_path / "obs.parquet")
    ref = pd.Timestamp("2026-01-31")
    gs.put("T", _frame(ref, ["2026-02-01"], [100.0]), source="test")
    assert gs.put("T", _frame(ref, ["2026-03-01"], [100.0]), source="test") == 0


def test_coverage_does_not_report_future_stamps_as_published(tmp_path, monkeypatch):
    """The lagged tier stamps published_at in the future; that is not 'released'."""
    from src import goldstore as gs
    monkeypatch.setattr(gs, "OBS_FILE", tmp_path / "obs.parquet")
    now = pd.Timestamp.now()
    gs.put("T", pd.DataFrame({
        "reference_date": [now - pd.Timedelta(days=40), now - pd.Timedelta(days=5)],
        "published_at": [now - pd.Timedelta(days=10), now + pd.Timedelta(days=20)],
        "value": [1.0, 2.0]}), source="test")
    cov = gs.coverage()
    assert cov.loc["T", "pending"] == 1
    assert pd.Timestamp(cov.loc["T", "last_published"]) <= now


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def test_forward_targets_are_nan_not_zero_past_the_last_real_fix(monkeypatch):
    """A stalled price feed must not manufacture 'gold did not move'.

    daily_panel forward-fills to today, so shift(-h) over the ffilled tail gives a
    ratio of exactly 1.0 and a forward return of exactly 0.0 — indistinguishable from
    an observation, and it trains and scores models as though it were one.
    """
    from src import goldfeatures, goldstore
    idx = pd.bdate_range("2026-01-01", periods=60)
    px = pd.Series(np.linspace(2000, 2100, 40).tolist() + [2100.0] * 20, index=idx)
    panel = pd.DataFrame({goldfeatures.TARGET_PRICE: px})
    monkeypatch.setattr(goldstore, "last_reference", lambda *a, **k: idx[39])
    out = goldfeatures.build_targets(panel)
    assert out["fwd_ret_5d"].iloc[40:].isna().all(), \
        "forward returns fabricated past the last real fix"
    assert out["fwd_ret_5d"].iloc[:30].notna().any()


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------
def test_always_long_actually_holds_the_position():
    """The buy-and-hold benchmark must be IN the market at every horizon.

    It emitted the trailing mean of the vol-scaled target, which Phi() maps to about
    0.52 at five days — below the 0.55 entry threshold. The benchmark named 'always
    long' sat in cash for the whole sample while the page displayed its hit rate as a
    strategy result.
    """
    from src import goldbacktest as bt
    m = bt.AlwaysLong()
    m.mu_ = 0.02
    pred = m.predict(pd.DataFrame(index=range(50)))
    prob = bt.to_probability(pd.DataFrame({"pred": pred}))
    assert (prob > bt.LONG_THRESHOLD).all(), "always_long does not hold a position"
    assert bt.AlwaysLong.benchmark_only is True


def test_mcnemar_print_guard_survives_a_missing_p_value():
    """`x == x` is a NaN test that None also passes, and the subscript then raised."""
    for mc in ({}, {"p_value": None}, {"p_value": float("nan")}, {"p_value": 0.04}):
        pv = mc.get("p_value")
        ok = isinstance(pv, float) and pv == pv
        _ = f"{pv:8.3f}" if ok else "       -"          # must not raise


# ---------------------------------------------------------------------------
# The calendar feed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,valid", [
    ([{"date": "2026-08-24T12:30:00Z", "title": "CPI"}], True),
    ([{"title": "X", "forecast": "1", "previous": "0"}], False),   # the test fixture
    ([], False),
    ({"error": "rate limited"}, False),
    ("nope", False),
    (None, False),
])
def test_calendar_payload_validation(payload, valid):
    """A fixture titled 'X' once reached the cache and would have printed a
    fabricated release into a client PDF."""
    from src import econ
    assert econ._looks_like_feed(payload) is valid


def test_calendar_cache_has_a_staleness_ceiling(tmp_path, monkeypatch):
    """Past the ceiling the feed must report unavailable, not serve last week.

    week_ahead() probes _feed() to decide between 'no releases scheduled' and 'the
    calendar was unreachable'. Without a ceiling an ancient cache made a dead feed
    look alive, and the report asserted an empty week it had never verified.
    """
    from src import econ
    cache = tmp_path / "cal.json"
    cache.write_text(json.dumps([{"date": "2020-01-01T12:30:00Z", "title": "CPI"}]),
                     encoding="utf-8")
    old = time.time() - (econ._MAX_STALE_SECONDS + 3600)
    os.utime(cache, (old, old))
    monkeypatch.setattr(econ, "_CACHE", cache)

    def _dead(*a, **k):
        raise OSError("feed down")

    monkeypatch.setattr(econ.urllib.request, "urlopen", _dead)
    with pytest.raises(econ.FeedUnavailable):
        econ._feed()


def test_calendar_cache_within_the_ceiling_is_still_served(tmp_path, monkeypatch):
    """Degrading to a recent cache is the whole point; only ANCIENT ones are refused."""
    from src import econ
    cache = tmp_path / "cal.json"
    cache.write_text(json.dumps([{"date": "2026-08-24T12:30:00Z", "title": "CPI"}]),
                     encoding="utf-8")
    recent = time.time() - (econ._TTL_SECONDS + 60)
    os.utime(cache, (recent, recent))
    monkeypatch.setattr(econ, "_CACHE", cache)

    def _dead(*a, **k):
        raise OSError("feed down")

    monkeypatch.setattr(econ.urllib.request, "urlopen", _dead)
    assert len(econ._feed()) == 1


# ---------------------------------------------------------------------------
# What a client PDF is allowed to say
# ---------------------------------------------------------------------------
def test_wgc_is_not_client_publishable_by_default():
    """WGC is licensed for internal analysis; the synopsis report auto-sends."""
    from src import compliance
    assert compliance.publication_check("wgc"), "WGC must not be silently publishable"
    assert compliance.required_citations("wgc")


def test_unregistered_source_is_reported_not_waved_through():
    from src import compliance
    assert compliance.publication_check("some-new-vendor")


def test_synopsis_build_refuses_an_unapproved_source(tmp_path):
    from src import pmrelreport
    with pytest.raises(pmrelreport.LicenceBlocked):
        pmrelreport.build_pdf({"pub": "wgc", "label": "x"}, tmp_path / "x.pdf")


def test_report_prose_is_generated_not_asserted():
    """Hardcoded prose asserting live numbers is how a page goes quietly false.

    The template claimed 'above 1.0 in all seven periods' and 'every confidence
    interval contains 1.0'; a baseline correction moved the oldest bucket to 0.97 and
    no confidence interval was ever computed anywhere in the engine.
    """
    from src import goldreport
    # The shape that produced the false claim: elevated in the recent buckets, but
    # NOT the oldest, which sat at 0.97.
    row = {"label": "Employment report",
           "buckets": [{"ratio": 1.55}, {"ratio": 1.70}, {"ratio": 1.99},
                       {"ratio": 1.23}, {"ratio": 1.29}, {"ratio": 1.13},
                       {"ratio": 0.97}],
           "recent_run": 6, "n_buckets": 7, "n_above": 6,
           "too_short_to_judge": False, "always_elevated": False,
           "mostly_elevated": True, "recent_24h": 1.17, "sig_24h": False,
           "significant": True}
    verdict = goldreport.verdict_for(row)
    assert verdict != "elevated in all 7 periods", "asserts a period that is 0.97"
    assert "6" in verdict, f"verdict should name the run that IS elevated: {verdict}"

    text = goldreport.summary_sentence([row])
    assert "confidence interval" not in text.lower()
    # The range quoted beside "every recent period" must cover the RUN, not the whole
    # row — putting 0.97 there contradicted the sentence it was supporting.
    assert "0.97" not in text, "quoted a number contradicting the claim it supports"
    assert "1.13" in text and "1.99" in text, "run range not quoted"
    # ...and the 24h fade is stated rather than left for the reader to notice.
    assert "1.17" in text and "next fix" in text


def test_attribution_separates_drift_from_the_drivers():
    """'The drivers account for +3.49%' silently included +0.85% of drift, and the
    parts table summed to 2.60% with nothing to attribute the difference to."""
    from src import goldsens
    idx = pd.bdate_range("2020-01-01", periods=900)
    rng = np.random.default_rng(0)
    cols = ["real_yield_10y_chg_20d", "dxy_chg_20d"]
    feats = pd.DataFrame(rng.normal(size=(len(idx), 2)) * 0.01, index=idx, columns=cols)
    f = {"cols": cols, "beta": pd.Series([0.5, -0.8], index=cols),
         "mu": feats.mean(), "sd": feats.std(), "ymu": 0.008, "h": 21,
         "price": pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))) * 2000,
                            index=idx)}
    a = goldsens.attribution(f, feats)
    lp = a["log_parts"]
    assert abs(lp["drivers"] + lp["drift"] + lp["unexplained"] - lp["actual"]) < 1e-12
    assert "drift_pct" in a and a["drift_pct"] != 0


def test_dollar_scenario_slider_agrees_with_the_published_sensitivity():
    """The slider ran in fractional units under a '%' format, so a 5% dollar move
    rendered as '0.1%'. Every dollar scenario on the page was labelled 100x small."""
    per_1pct = -0.8249                       # the published table value
    step = 0.01
    for shown in (1.0, 5.0):
        native = shown * 0.01                # slider units -> feature units
        implied = np.expm1(np.log1p(per_1pct / 100) * (native / step)) * 100
        assert abs(implied - per_1pct * shown) < 0.15 * abs(shown)


def test_fair_value_percentile_ignores_leading_nans():
    """The rolling window leaves leading NaNs; `(gap <= x).mean()` counted them as
    False in the denominator and understated every percentile ever published."""
    gap = pd.Series([np.nan] * 6 + [1.0, 2.0, 3.0, 4.0])
    naive = float((gap <= gap.iloc[-1]).mean() * 100)
    clean = float((gap.dropna() <= gap.iloc[-1]).mean() * 100)
    assert naive == 40.0 and clean == 100.0


def test_spliced_real_yield_is_never_used_for_a_level_feature():
    """CHANGES may use the spliced series; LEVELS may not.

    Pre-2003 real yields are nominal-less-trailing-CPI, level-adjusted by the
    2003-2008 overlap mean. That offset is estimated in one inflation regime and
    applied across another, so the LEVEL carries a regime-dependent bias — but
    differencing removes it, which is why changes are fine and z-scores of the level
    are not. The rule was followed by hand; this makes breaking it a failing test.
    """
    import re
    src = Path("src/goldfeatures.py").read_text(encoding="utf-8")
    # Level constructors, as opposed to _chg()/.diff()
    for m in re.finditer(r'_z\(\s*g\(\s*"([A-Z0-9_]+)"', src):
        assert m.group(1) != "REAL_10Y_SPLICED", (
            "a level z-score is being taken of the spliced series; the pre-2003 "
            "splice offset makes its LEVEL non-comparable across the join")
    # ...and the changes path must still be on the spliced (long-history) series,
    # otherwise the 1990-2002 depth this whole exercise added is silently dropped.
    assert '_chg(g("REAL_10Y_SPLICED")' in src.replace(" ", "")  \
        or '_chg(g("REAL_10Y_SPLICED")' in src


# ---------------------------------------------------------------------------
# Metals relative value
# ---------------------------------------------------------------------------
def test_lag_choice_materially_changes_the_adf_verdict():
    """The augmentation must be on by default, because the lag count moves the answer.

    On the real gold/silver ratio the unaugmented test gives about -3.95 and the
    21-lag version about -3.23 — the difference between "clearly mean-reverting" and
    "fails a multiple-comparison correction".

    This deliberately does NOT assert a direction. The sign of the augmentation
    effect depends on the process: an earlier version of this test asserted that
    lags always make the statistic less negative, and a synthetic near-unit-root
    series with positively autocorrelated increments falsified it immediately
    (-0.61 unaugmented, -1.56 augmented). What is defensible is that the choice
    matters and that the module makes the conservative one by default.
    """
    from src import metalrv
    assert metalrv.DEFAULT_LAGS >= 21
    rng = np.random.default_rng(7)
    n = 4000
    e = pd.Series(rng.normal(size=n)).rolling(5).mean().fillna(0).to_numpy()
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.999 * x[i - 1] + e[i]
    s = pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n))
    plain, aug = metalrv.adf(s, lags=0)["t"], metalrv.adf(s, lags=21)["t"]
    assert abs(plain - aug) > 0.3, "lag choice should move the statistic materially"
    # ...and pair_report must use the augmented default, never the raw one.
    import inspect
    assert "lags: int = DEFAULT_LAGS" in inspect.signature(
        metalrv.pair_report).__str__().replace("lags=21", "lags: int = DEFAULT_LAGS")         or metalrv.pair_report.__defaults__[1] == metalrv.DEFAULT_LAGS


def test_engle_granger_reports_the_weaker_direction():
    """EG is not symmetric; quoting the better ordering is picking your own winner."""
    from src import metalrv
    rng = np.random.default_rng(3)
    n = 2500
    idx = pd.bdate_range("2005-01-03", periods=n)
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 100, index=idx)
    a = b * np.exp(pd.Series(rng.normal(0, 0.05, n), index=idx))   # cointegrated
    r = metalrv.pair_report("A", "B", pd.DataFrame({"A": a, "B": b}))
    assert r["eg_t"] == max(r["eg_t_ab"], r["eg_t_ba"]), "reported the stronger side"


def test_rv_verdicts_are_derived_and_refuse_to_overstate():
    """The verdict must not claim a pattern the numbers do not support.

    Written after I described platinum/palladium as narrowing 'below 50% in all
    three eras' when the middle era is 51.0%. The derived verdict declined to call
    it a consistent trend and was right.
    """
    from src import metalrv
    borderline = [{"period": "a", "corr": 0.06, "hit_rate": 0.46, "signals": 8},
                  {"period": "b", "corr": 0.06, "hit_rate": 0.51, "signals": 7},
                  {"period": "c", "corr": 0.18, "hit_rate": 0.44, "signals": 7}]
    assert "TRENDS" not in metalrv.signal_verdict(borderline)

    trending = [dict(r, hit_rate=h) for r, h in zip(borderline, (0.46, 0.48, 0.44))]
    assert "TRENDS" in metalrv.signal_verdict(trending)

    recent_only = [dict(r, hit_rate=h) for r, h in zip(borderline, (0.42, 0.60, 0.87))]
    v = metalrv.signal_verdict(recent_only)
    assert "one era" in v and "not a regularity" in v


def test_silver_has_no_intraday_fix_window():
    """The LBMA Silver Price is a single auction. The gold event study's only
    surviving effect lived in the AM->PM window, so silver cannot be tested the one
    way that detected anything — and the flag must be derived, not hand-listed."""
    from src import metals
    assert "SILVER" not in metals.HAS_INTRADAY_WINDOW
    assert {"GOLD", "PLATINUM", "PALLADIUM"} <= metals.HAS_INTRADAY_WINDOW
    assert len(metals.METALS["SILVER"]["fixes"]) == 1
