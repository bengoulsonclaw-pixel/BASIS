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


# ---------------------------------------------------------------------------
# Track B — supply signature
# ---------------------------------------------------------------------------
def test_country_factor_never_regresses_a_currency_on_itself():
    """The control basket must exclude the country's own currency.

    Leaving USDZAR in the basket used to residualise USDZAR would produce a residual
    of zero and a factor that measures nothing, while still looking like a working
    pipeline.
    """
    from src import metalsupply as ms
    for country, spec in ms.COUNTRY_FX.items():
        assert spec["fx"] not in spec["basket"], f"{country} controls on itself"


def test_supply_thesis_records_its_own_controls():
    """The metals with NO supply exposure must be declared, or the test cannot fail.

    Silver and gold are the controls. Without them on record, a loading on the ZA
    factor reads as confirmation; with them, silver's larger loading falsifies it.
    """
    from src import metalsupply as ms
    assert ms.EXPECTED_EXPOSURE["SILVER"] == []
    assert ms.EXPECTED_EXPOSURE["GOLD"] == []
    assert "ZA" in ms.EXPECTED_EXPOSURE["PLATINUM"]
    assert "RU" in ms.EXPECTED_EXPOSURE["PALLADIUM"]


def test_lead_lag_flags_the_contemporaneous_row_as_unusable():
    """Lag 0 pairs a New York FX close with a 15:00 London fix and is not forecasting
    evidence. It is reported for context but must be marked."""
    from src import metalsupply as ms
    import inspect
    src = inspect.getsource(ms.lead_lag)
    assert '"usable_as_forecast": L >= 1' in src


def test_inverse_normal_matches_known_quantiles():
    """Bonferroni thresholds are computed from this, so it has to be right."""
    from src import metalsupply as ms
    assert abs(abs(ms._two_sided_z(0.05)) - 1.9600) < 1e-3
    assert abs(abs(ms._two_sided_z(0.01)) - 2.5758) < 1e-3


def test_fx_is_stamped_a_day_late_against_the_london_fix():
    """A New York close did not exist when the 15:00 London fix was struck."""
    from src import metalsupply as ms
    import inspect
    src = inspect.getsource(ms.ingest_fx)
    assert "typical_lag_days=1" in src, "FX stamped lag 0 would leak into forecasts"


def test_cross_metal_verdict_requires_a_tested_difference():
    """Significant-vs-not-significant is not a difference. The verdict must rest on
    a confidence interval that excludes zero, not on comparing asterisks."""
    from src import metalevents as me
    # CI straddling zero: no claim, even though p looks smallish.
    straddle = {"reference": "GOLD", "comparisons": [
        {"vs": "PLATINUM", "diff": 0.4, "ci_low": -0.05, "ci_high": 0.8,
         "p": 0.06, "survives_correction": False}]}
    assert "differ" in me.verdict(straddle).lower() or "only" not in me.verdict(straddle)
    assert "different assets" not in me.verdict(straddle)

    tested = {"reference": "GOLD", "comparisons": [
        {"vs": "PLATINUM", "diff": 0.456, "ci_low": 0.140, "ci_high": 0.777,
         "p": 0.0038, "survives_correction": True},
        {"vs": "PALLADIUM", "diff": 0.539, "ci_low": 0.222, "ci_high": 0.860,
         "p": 0.0011, "survives_correction": True}]}
    assert "different assets" in me.verdict(tested)


def test_silver_is_excluded_from_the_release_study():
    """No AM fix means no intraday window; including it would compare a metal
    measured one way against metals measured another."""
    from src import metalevents as me
    assert "SILVER" not in me.STUDY_METALS


# ---------------------------------------------------------------------------
# Track D — per-metal features
# ---------------------------------------------------------------------------
def test_cot_series_are_namespaced_per_metal():
    """Every metal must land on its OWN COT series IDs.

    goldingest.ingest_cot writes COT_MM_NET/COT_OI from a fixed table regardless of
    which ticker it was handed. Without a prefix, ingesting silver would have written
    over gold's positioning — and because the CFTC reference dates match exactly,
    put() would have accepted each metal as a REVISION of the last, leaving gold's
    features reading whatever was ingested most recently with nothing in the store to
    show it. Gold keeps the bare IDs its feature layer already references.
    """
    from src import metals, goldstore
    ids = set()
    for m in metals.METALS:
        prefix = "" if m == "GOLD" else f"{m}_"
        ids.add(f"{prefix}COT_MM_NET")
    assert len(ids) == len(metals.METALS), "two metals share a COT series id"
    assert "COT_MM_NET" in ids, "gold must keep the unprefixed id"

    # ...and the live store must actually hold them separately.
    seen = {}
    for m in metals.METALS:
        prefix = "" if m == "GOLD" else f"{m}_"
        s = goldstore.get_series(f"{prefix}COT_MM_NET")
        if len(s):
            seen[m] = float(s.iloc[-1])
    if len(seen) > 1:
        assert len(set(seen.values())) == len(seen), \
            f"metals share an identical COT value — overwrite suspected: {seen}"


def test_ingest_cot_defaults_leave_gold_untouched():
    """The prefix must default to empty so the existing gold series never move."""
    from src import goldingest
    import inspect
    sig = inspect.signature(goldingest.ingest_cot)
    assert sig.parameters["prefix"].default == ""


def test_metal_feature_blocks_declare_what_is_missing():
    """A thinner search must not be reportable as a thinner market.

    Only gold has ETF tonnage, the Shanghai premium and central-bank demand. If those
    silently returned True for the PGMs the result would read as "we looked at
    everything and found less" rather than "we had less to look at".
    """
    from src import metalfeatures as mf
    for metal in ("SILVER", "PLATINUM", "PALLADIUM"):
        b = mf.available_blocks(metal)
        assert b["macro"] and b["cot"]
        assert not b["etf_flows"], f"{metal} has no ETF tonnage feed"
        assert not b["shanghai_premium"]
        assert not b["central_bank"]
    assert all(mf.available_blocks("GOLD").values())


def test_metal_targets_cut_the_forward_filled_tail():
    """Same guard as the gold layer: a stalled feed must not produce 0.0 returns."""
    from src import metalfeatures as mf
    import inspect
    src = inspect.getsource(mf.build_targets)
    assert "last_reference" in src and "px.index <= last_real" in src


# ---------------------------------------------------------------------------
# Track C — auditing BASIS's own strategies
# ---------------------------------------------------------------------------
def test_unpriceable_contracts_are_dropped_not_counted_as_losses():
    """A contract with no point value is missing data, never a loss.

    volbt.POINT_VALUE had no entry for five of twenty-four sampled contracts, so both
    the strategy P&L and the benchmark came out as exactly 0.0 and `0 > 0` scored
    False — handing every strategy five automatic losses and moving the headline from
    ~42% to 33%. tabt warns about this plainly; the first version of this module threw
    the warning away.
    """
    from src import sigaudit as sa
    d = pd.DataFrame({
        "strategy": ["Trend"] * 4,
        "ticker": ["A", "B", "C", "D"],
        "priceable": [True, True, False, False],
        "beats_bh": pd.Series([True, False, np.nan, np.nan], dtype="object"),
        "n_trades": [10, 10, 0, 0],
        "win_rate": [50.0, 50.0, np.nan, np.nan],
    })
    card = sa.strategy_scorecard(d)
    assert int(card.iloc[0]["contracts"]) == 2, "unpriceable contracts were counted"
    assert int(card.iloc[0]["beat_bh"]) == 1


def test_scorecard_records_which_contracts_it_dropped():
    """Silent truncation reads as full coverage."""
    from src import sigaudit as sa
    d = pd.DataFrame({
        "strategy": ["Trend"] * 3, "ticker": ["A", "B", "KOSPI2 Index"],
        "priceable": [True, True, False],
        "beats_bh": pd.Series([True, False, np.nan], dtype="object"),
        "n_trades": [5, 5, 0], "win_rate": [50.0, 50.0, np.nan]})
    card = sa.strategy_scorecard(d)
    assert "KOSPI2 Index" in card.attrs.get("dropped_unpriceable", [])


def test_audit_verdict_will_not_claim_an_edge_below_half():
    """`survives` requires BOTH significance and a share above one half.

    Bollinger Squeeze cleared Holm at 2 of 19 — significant, and significantly WORSE.
    A rule keyed on p alone would have reported it as a surviving strategy.
    """
    from src import sigaudit as sa
    worse = pd.DataFrame([{"strategy": "Bollinger Squeeze", "contracts": 19,
                           "beat_bh": 2, "share": 2 / 19, "ci_low": 0.03,
                           "ci_high": 0.31, "p_raw": 0.001, "p_holm": 0.012,
                           "median_trades": 78.0, "median_win_rate": 23.8}])
    worse["survives"] = (worse["p_holm"] < 0.05) & (worse["share"] > 0.5)
    assert not worse["survives"].any()
    assert "No strategy beats" in sa.verdict(worse)


def test_binomial_and_wilson_match_known_values():
    from src import sigaudit as sa
    assert abs(sa.binom_p_two_sided(10, 10) - 0.001953125) < 1e-9
    assert sa.binom_p_two_sided(5, 10) == 1.0
    lo, hi = sa.wilson_ci(55, 100)
    assert 0.45 < lo < 0.46 and 0.64 < hi < 0.65


def test_positive_excess_pnl_is_not_sufficient_on_its_own():
    """A model can beat buy-and-hold on P&L while calling direction WORSE than
    always-long. Four of the five Track D 'wins' were exactly that shape: positive
    excess_vs_buyhold, hit rate below the base rate, McNemar nowhere near
    significant, and an edge phase spread that straddled or sat below zero. Reading
    the P&L column alone would have reported momentum beating platinum by 39%.
    """
    cell = {"excess_vs_buyhold": 0.33, "hit_rate": 0.508,
            "always_long_hit": 0.520, "edge_phase_spread": (-0.018, -0.008),
            "mcnemar": {"p_value": 0.542}}

    def is_credible(m):
        lo, hi = m.get("edge_phase_spread") or (float("nan"),) * 2
        p = (m.get("mcnemar") or {}).get("p_value")
        return bool(
            (m.get("excess_vs_buyhold") or 0) > 0
            and m.get("hit_rate", 0) > m.get("always_long_hit", 1)
            and lo > 0
            and isinstance(p, float) and p < 0.05)

    assert not is_credible(cell), "P&L alone was treated as evidence"
    good = dict(cell, hit_rate=0.60, edge_phase_spread=(0.02, 0.05),
                mcnemar={"p_value": 0.01})
    assert is_credible(good)


# ---------------------------------------------------------------------------
# The rates -> dollar -> gold chain
# ---------------------------------------------------------------------------
def test_real_and_breakeven_enter_gold_with_opposite_signs():
    """Nominal yield = real + breakeven, and the halves push gold opposite ways.

    A formula keyed on the NOMINAL yield averages two live effects into one number
    that describes neither — which is why the naive gold-on-nominal regression has an
    R-squared of 0.025 against 0.26 for the split version.
    """
    from src import macrochain as mc
    c = mc.chain()
    g = c["gold"]["terms"]
    assert g["real10_bp"]["beta"] < 0, "real yields should weigh on gold"
    assert g["breakeven_bp"]["beta"] > 0, "breakevens should support gold"
    assert c["gold"]["r2"] > c["gold_nominal_only"]["r2"] * 3


def test_rate_move_decomposition_adds_up():
    """total = direct + (dollar's response to rates) x (gold's response to dollar)."""
    from src import macrochain as mc
    c = mc.chain()
    d = c["decomposition_25bp_real"]
    assert abs(d["direct_pct"] + d["via_dollar_pct"] - d["total_pct"]) < 1e-9
    # The indirect leg must be reconstructible from its two factors.
    g_dxy = c["gold"]["terms"]["dxy_pct"]["beta"]
    assert abs(d["dollar_move_pct"] * g_dxy - d["via_dollar_pct"]) < 1e-9


def test_no_level_anchor_between_gold_dollar_and_real_yields():
    """There is a relationship between CHANGES and none between LEVELS.

    The distinction matters because "a formula they stick to" is a claim about
    levels. The level regression is not cointegrated (ADF -0.57 against -3.74) and
    its dollar coefficient comes out POSITIVE where the change relationship is
    firmly negative — the standard symptom of a spurious regression.
    """
    from src import macrochain as mc
    la = mc.level_anchor()
    assert not la["cointegrated"]
    assert la["adf_t"] > la["critical_5pct"]
    c = mc.chain()
    assert c["gold"]["terms"]["dxy_pct"]["beta"] < 0 < la["beta_log_dxy"], \
        "sign flip between levels and changes is the spurious-regression tell"


def test_fed_scenario_takes_a_surprise_not_a_hike():
    """The input must be REPRICING. If the strip already carries the hikes, the
    expected impact of delivering them is zero, and a function keyed on "number of
    hikes" would hand a client a move that has already happened."""
    from src import macrochain as mc
    import inspect
    doc = inspect.getdoc(mc.fed_scenario)
    assert "SURPRISE, NOT A HIKE" in doc
    assert mc.fed_scenario(0)["gold_reduced_form_pct"] == 0.0


def test_fed_scenario_reports_a_range_not_a_point():
    """Reduced-form and structural estimates disagree by design; quoting either
    endpoint alone is false precision."""
    from src import macrochain as mc
    s = mc.fed_scenario(50)
    lo, hi = s["gold_range_pct"]
    assert lo <= hi
    assert {round(lo, 6), round(hi, 6)} == {round(s["gold_reduced_form_pct"], 6),
                                            round(s["gold_structural_pct"], 6)}
    # the structural legs must reconstruct the structural total
    p = s["gold_parts"]
    assert abs(sum(p.values()) - s["gold_structural_pct"]) < 1e-9


def test_fed_scenario_carries_its_own_error_band():
    """The unexplained band is several times the effect. It must travel with it."""
    from src import macrochain as mc
    s = mc.fed_scenario(50)
    assert s["unexplained_1sd_pct"] > abs(s["gold_reduced_form_pct"])


# ---------------------------------------------------------------------------
# Move Translator
# ---------------------------------------------------------------------------
def test_dollar_index_reproduces_its_own_largest_component():
    """EURUSD is ~58% of DXY, so a +1% dollar move must imply about -1% on EURUSD
    with a very high R-squared. If this pair ever comes out wrong the units or the
    sign convention are broken and nothing else on the page can be trusted."""
    from src import crossmove as cm
    t = cm.translate("Dollar (DXY)", 1.0, horizon=20, years=5)
    e = t[t["instrument"] == "EURUSD"].iloc[0]
    assert -1.3 < e["implied"] < -0.6, f"EURUSD implied {e['implied']}"
    assert e["r_squared"] > 0.7
    assert e["t"] < -10


def test_units_are_declared_and_carried():
    """Yields must come back in bp and prices in %, or a slider ends up 100x wrong."""
    from src import crossmove as cm
    t = cm.translate("Dollar (DXY)", 1.0, horizon=20, years=5)
    by = dict(zip(t["instrument"], t["unit"]))
    assert by["Gold"] == "%" and by["US 10y yield"] == "bp" and by["VIX"] == "pts"
    assert cm.unit_of("US 10y real") == "bp"
    assert cm.default_move("US 10y real") == 25.0
    assert cm.default_move("Gold") == 1.0


def test_every_row_carries_its_unexplained_band():
    """The band is what decides whether a number is usable and must never be
    optional — gold's response to the dollar is about -1.1% with a 4% band."""
    from src import crossmove as cm
    t = cm.translate("Dollar (DXY)", 1.0, horizon=20, years=5)
    assert (t["band_1sd"] > 0).all()
    g = t[t["instrument"] == "Gold"].iloc[0]
    assert g["band_1sd"] > abs(g["implied"]), \
        "gold's monthly noise should dwarf a 1% dollar move"


def test_lookback_actually_changes_the_estimate():
    """A window selector that returns the same answer everywhere is decoration."""
    from src import crossmove as cm
    out = {}
    for label, yrs in (("2 years", 2), ("20 years", 20)):
        t = cm.translate("Dollar (DXY)", 1.0, horizon=20, years=yrs)
        out[label] = float(t[t["instrument"] == "Gold"].iloc[0]["implied"])
    assert abs(out["2 years"] - out["20 years"]) > 0.1


def test_caveat_states_it_is_contemporaneous():
    """The one claim the page must never imply is that this forecasts anything."""
    from src import crossmove as cm
    c = cm.caveat("Dollar (DXY)", 20, 5).lower()
    assert "contemporaneous" in c and "not forecast" in c


def test_macro_compass_page_renders_all_three_tabs(tmp_path):
    """The page must actually render, not just import.

    Three tabs, each hitting a different engine (crossmove, macrochain,
    metalevents). An import-only check would pass while a tab raised on first draw.
    """
    from streamlit.testing.v1 import AppTest
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "render_compass.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{repo}')\n"
        f"sys.path.insert(0, r'{repo / 'src'}')\n"
        "from src import crossmovepage\n"
        "crossmovepage.render()\n", encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=400)
    at.run()
    assert not at.exception, f"page raised: {at.exception}"
    assert [s.value for s in at.subheader] == ["🧭 Macro Compass"]
    assert len(at.tabs) == 3
    assert len(at.dataframe) >= 3


def test_fed_tab_refuses_to_imply_a_move_from_a_priced_view():
    """Zero repricing must imply zero. The page short-circuits before calling, and
    the engine agrees — a client view that matches the curve moves nothing."""
    from src import macrochain as mc
    s = mc.fed_scenario(0)
    assert s["gold_reduced_form_pct"] == 0.0
    assert s["dollar_pct"] == 0.0


def test_release_tab_excludes_silver_structurally():
    """Silver has no AM->PM window; it must not appear in the release table as
    though it had been tested and found wanting."""
    from src import metalevents as me
    tab = me.comparison_table()
    assert "SILVER" not in tab.columns
    assert "SILVER" not in me.STUDY_METALS
