"""Locks on the STIR Paths engine (src/stirpaths.py): the central-bank meeting
calendars, contract-window construction, rule-generated expiry dates, the
implied-path inversion round-trip, probability scenarios and option-expiry
landings. Pure-date maths — no feed, no Streamlit."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src import stirpaths as sp
from src.fedpath import effective_date, price


ASOF = date(2026, 8, 10)


# ── calendars ────────────────────────────────────────────────────────────────
def test_meeting_calendars_sane():
    for bank in sp.BANKS.values():
        assert bank.meetings == sorted(bank.meetings)
        assert all(m.weekday() < 5 for m in bank.meetings)          # decisions on weekdays
        assert {m.year for m in bank.meetings} == {2026, 2027}
    # ECB and BoE both publish 8 decisions per calendar year
    for key in ("ECB", "BOE"):
        for yr in (2026, 2027):
            assert sum(m.year == yr for m in sp.BANKS[key].meetings) == 8


def test_known_decision_dates():
    """Spot-checks against the official 2026 calendars."""
    assert date(2026, 9, 16) in sp.BANKS["FED"].meetings            # FOMC Sep
    assert date(2026, 9, 10) in sp.BANKS["ECB"].meetings            # GC Sep (Berlin)
    assert date(2026, 9, 17) in sp.BANKS["BOE"].meetings            # MPC Sep
    assert date(2026, 11, 5) in sp.BANKS["BOE"].meetings            # MPC Nov (no Oct meeting)


# ── contracts + strips ───────────────────────────────────────────────────────
def test_quarterly_strip_windows_contiguous():
    prod = sp.PRODUCTS["SFRA Comdty"]
    s = sp.strip(prod, ASOF, 6)
    assert s[0].code == "SFRM6" and s[0].start == date(2026, 6, 17)  # accruing Jun IMM
    for a, b in zip(s, s[1:]):
        assert a.end == b.start                                     # no gaps, no overlap
    assert all(c.start.weekday() == 2 for c in s)                   # IMM = 3rd Wednesday


def test_monthly_strip_calendar_windows():
    prod = sp.PRODUCTS["FFA Comdty"]
    s = sp.strip(prod, ASOF, 3)
    assert s[0].code == "FFQ6"                                      # Aug-26 front
    assert s[0].start == date(2026, 8, 1) and s[0].end == date(2026, 9, 1)
    assert s[1].start == date(2026, 9, 1)


# ── expiry rules (vs the exchange-standard cycle) ────────────────────────────
def test_sr3_expiries_golden():
    rows = sp.expiry_rows(sp.PRODUCTS["SFRA Comdty"], ASOF, 3)
    futs = {r.month: r.expiry for r in rows if r.kind == "Future"}
    opts = {r.month: r.expiry for r in rows if r.kind == "Option"}
    assert futs["Sep-26"] == date(2026, 9, 15)      # bday before 3rd Wed (16 Sep)
    assert opts["Sep-26"] == date(2026, 9, 11)      # Friday before 3rd Wed
    assert "Aug-26" in opts and "Aug-26" not in futs  # serial option, no serial future


def test_fut_last_trade_conventions():
    """In-arrears quarterlies trade until their reference window ENDS (the big
    ED→SOFR change); Euribor fixes in advance and dies before its window starts;
    monthlies run to their own month-end."""
    sr3 = sp.PRODUCTS["SFRA Comdty"]
    jun = sp.quarterly_contract(sr3, 2026, 6)               # window 17 Jun → 16 Sep
    assert sp.fut_last_trade(sr3, jun) == date(2026, 9, 15)  # bd before window-end 3rd Wed
    son = sp.PRODUCTS["SFIA Comdty"]
    assert sp.fut_last_trade(son, sp.quarterly_contract(son, 2026, 9)) == date(2026, 12, 15)
    er = sp.PRODUCTS["ERA Comdty"]
    sep = sp.quarterly_contract(er, 2026, 9)                # fixes ~2bd before 16 Sep
    assert sp.fut_last_trade(er, sep) == date(2026, 9, 14)
    ff = sp.PRODUCTS["FFA Comdty"]
    assert sp.fut_last_trade(ff, sp.monthly_contract(ff, 2026, 8)) == date(2026, 8, 31)


def test_estr_has_no_options():
    rows = sp.expiry_rows(sp.PRODUCTS["TKYA Comdty"], ASOF, 12)
    assert rows and all(r.kind == "Future" for r in rows)


def test_serial_option_underlying_is_next_quarterly():
    prod = sp.PRODUCTS["SFRA Comdty"]
    assert sp.option_underlying(prod, 2026, 8).code == "SFRU6"      # Aug serial → Sep fut
    assert sp.option_underlying(prod, 2026, 9).code == "SFRU6"      # Sep quarterly → itself
    assert sp.option_underlying(prod, 2026, 10).code == "SFRZ6"     # Oct serial → Dec fut
    ff = sp.PRODUCTS["FFA Comdty"]
    assert sp.option_underlying(ff, 2026, 10).code == "FFV6"        # monthly → same month


# ── meetings vs windows ──────────────────────────────────────────────────────
def test_meetings_in_window_counts():
    bank = sp.BANKS["FED"]
    c = sp.quarterly_contract(sp.PRODUCTS["SFRA Comdty"], 2026, 9)  # 16 Sep → 16 Dec
    mtgs = sp.meetings_in_window(bank, c)
    # Sep FOMC decides 16 Sep, EFFECTIVE 17 Sep — inside the window, so it counts,
    # along with Oct and Dec. Three decisions shape this one contract's settlement.
    assert effective_date(date(2026, 9, 16)) == date(2026, 9, 17)
    assert mtgs == [date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]


# ── implied path round-trip ──────────────────────────────────────────────────
def test_implied_path_roundtrip():
    """Price a strip off a known meeting path (simple average, zero spread),
    invert it, and recover the cumulative path to <0.5bp."""
    bank = sp.BANKS["BOE"]
    prod = sp.PRODUCTS["SFIA Comdty"]
    contracts = sp.strip(prod, ASOF, 8)
    r0 = 4.0
    ups = [m for m in bank.meetings if m > ASOF]
    moves = [-25.0 if i % 2 == 0 else 0.0 for i in range(len(ups))]
    fn = sp.overnight_rate_fn(r0, ups, moves)
    prices = [price(c, fn, compound=False) for c in contracts]
    ip = sp.implied_path(bank, contracts, prices, ASOF, r0)
    assert max(abs(ip.residual_bp)) < 0.5
    # cumulative move at the LAST covered meeting matches the input path
    want = sum(m for m, mtg in zip(moves, ups) if mtg <= ip.meetings[-1])
    assert ip.cum_bp[-1] == pytest.approx(want, abs=0.5)


def test_implied_path_spread_shifts_prices_not_path():
    """A settlement-index spread (Euribor−€STR) moves fair PRICES, not the path."""
    bank = sp.BANKS["ECB"]
    prod = sp.PRODUCTS["ERA Comdty"]
    contracts = sp.strip(prod, ASOF, 6)
    r0 = 1.92
    fn = sp.overnight_rate_fn(r0, [], [])
    base = [price(c, fn, compound=False) for c in contracts]
    spread = 10.0
    shifted = [p - spread / 100.0 for p in base]                    # Euribor prints higher rate
    ip0 = sp.implied_path(bank, contracts, base, ASOF, r0)
    ip1 = sp.implied_path(bank, contracts, shifted, ASOF, r0, [spread] * len(contracts))
    assert max(abs(ip0.cum_bp - ip1.cum_bp)) < 1e-6


# ── probability scenarios + landings ─────────────────────────────────────────
def test_meeting_view_expected_move():
    v = sp.MeetingView(date(2026, 9, 16), p_hike=0.0, p_cut=0.6)
    assert v.expected_bp == pytest.approx(-15.0)
    v2 = sp.MeetingView(date(2026, 9, 16), p_hike=0.2, p_cut=0.1, hike_bp=50.0)
    assert v2.expected_bp == pytest.approx(50 * 0.2 - 25 * 0.1)


def test_landing_prices_move_with_the_scenario():
    bank = sp.BANKS["FED"]
    prod = sp.PRODUCTS["SFRA Comdty"]
    ups = [m for m in bank.meetings if m > ASOF]
    hold = [sp.MeetingView(m, 0.0, 0.0) for m in ups]
    cuts = [sp.MeetingView(m, 0.0, 1.0) for m in ups]
    r0 = 4.375
    l_hold = sp.landings(prod, bank, ASOF, hold, r0, 9)
    l_cuts = sp.landings(prod, bank, ASOF, cuts, r0, 9)
    assert len(l_hold) == len(l_cuts) > 0
    # certain cuts at every meeting → every landing price above the hold path
    assert all(c.fair > h.fair for h, c in zip(l_hold, l_cuts))
    # hold-everything → every contract prints ~100 − r0; ACT/360 daily compounding
    # adds ≈ r²·days/72000 ≈ 2.4bp on a flat 4.375% quarter, hence the tolerance
    assert all(abs(h.fair - (100 - r0)) < 0.04 for h in l_hold)


def test_landing_meeting_split():
    """Sep-26 SR3 option (exp 11 Sep 26): Sep FOMC is 5 days later — not yet
    decided, but inside the Sep contract's window → 'still open after'."""
    bank = sp.BANKS["FED"]
    prod = sp.PRODUCTS["SFRA Comdty"]
    views = [sp.MeetingView(m, 0.0, 0.0) for m in bank.meetings if m > ASOF]
    L = {x.opt_month: x for x in sp.landings(prod, bank, ASOF, views, 4.375, 6)}
    sep = L["Sep-26"]
    assert sep.expiry == date(2026, 9, 11) and sep.underlying.code == "SFRU6"
    assert date(2026, 9, 16) not in sep.meetings_decided
    assert date(2026, 9, 16) in sep.meetings_open
    assert date(2026, 10, 28) in sep.meetings_open                  # Oct FOMC also in window


def test_fair_price_spread_override():
    prod = sp.PRODUCTS["ERA Comdty"]
    c = sp.quarterly_contract(prod, 2026, 9)
    fn = sp.overnight_rate_fn(2.0, [], [])
    assert sp.fair_price(prod, c, fn) == pytest.approx(
        sp.fair_price(prod, c, fn, spread_bp=0.0) - prod.spread_bp / 100.0)


# ── calendar runway (time-dependent BY DESIGN) ───────────────────────────────
def test_meeting_calendars_not_running_out():
    """Hard backstop behind the Data-health 'CB calendars' warn (which fires at
    9 months of runway): this test goes RED at <3 months, i.e. the *_DECISIONS
    lists were left to run dry despite the nudge. Fix = append the banks' newly
    published decision dates in src/fedpath.py / src/stirpaths.py."""
    from datetime import timedelta
    horizon = date.today() + timedelta(days=90)
    for bank in sp.BANKS.values():
        assert max(bank.meetings) >= horizon, (
            f"{bank.key} meeting calendar ends {max(bank.meetings)} — extend the list "
            "from the bank's published schedule (see Data health → CB calendars)")


def test_meeting_calendar_runway_frame():
    from src import health
    cal = health.meeting_calendar_runway()
    assert set(cal["key"]) == {"FED", "ECB", "BOE", "BCB"}
    assert (cal["future_meetings"] > 0).all()
    assert (cal["months_left"] > 0).all()


# ── front-stub correctness (the post-move weeks) ─────────────────────────────
def test_front_stub_fixes_phantom_odds():
    """A 25bp cut landed mid-window BEFORE asof: without the stub the front
    contract carries ~phantom mispricing that pollutes the implied odds; with
    stub_rate = realized average, the inversion round-trips to a flat path."""
    bank = sp.BANKS["BOE"]
    prod = sp.PRODUCTS["SFIA Comdty"]
    contracts = sp.strip(prod, ASOF, 6)             # front window 17 Jun → 16 Sep
    old_r, new_r = 4.25, 4.00
    change = date(2026, 7, 10)                      # inside the front window, before ASOF

    def true_fn(d):
        return old_r if d < change else new_r
    prices = [price(c, true_fn, compound=False) for c in contracts]
    stub_days = list(sp._daterange(contracts[0].start, ASOF))
    stub_avg = sum(true_fn(d) for d in stub_days) / len(stub_days)

    naive = sp.implied_path(bank, contracts, prices, ASOF, new_r)
    fixed = sp.implied_path(bank, contracts, prices, ASOF, new_r, stub_rate=stub_avg)
    # the truth is a flat path from today (the cut already happened)
    assert max(abs(fixed.cum_bp)) < 0.75
    assert abs(fixed.residual_bp[0]) < 0.5
    # the naive version is materially wrong: with the smoothing penalty the phantom
    # surfaces as a large FRONT RESIDUAL (a bogus 'front is rich' RV signal) instead
    # of odds wiggle — either way, wrong until the stub is priced off realized fixings
    assert abs(naive.residual_bp[0]) > 4.0


def test_scenario_rate_fn_stub():
    fn = sp.scenario_rate_fn(4.0, [], asof=ASOF, stub_rate=4.25)
    assert fn(ASOF - timedelta(days=1)) == 4.25
    assert fn(ASOF) == 4.0


# ── scenario distribution ────────────────────────────────────────────────────
def test_pmf_from_view():
    v = sp.MeetingView(date(2026, 9, 17), 0.0, 1.5)          # 1.5 cuts
    assert sp._pmf_from_view(v) == {-25.0: 0.5, -50.0: 0.5}
    v2 = sp.MeetingView(date(2026, 9, 17), 0.6, 0.0)
    assert sp._pmf_from_view(v2) == {0.0: pytest.approx(0.4), 25.0: pytest.approx(0.6)}


def test_landing_distribution_mean_and_split():
    bank = sp.BANKS["FED"]
    prod = sp.PRODUCTS["SFRA Comdty"]
    c = sp.quarterly_contract(prod, 2026, 9)        # window 16 Sep → 16 Dec
    ups = [m for m in bank.meetings if m > ASOF]
    views = [sp.MeetingView(m, 0.0, 0.6 if m.year == 2026 else 0.0) for m in ups]
    r0 = 4.375
    dist = sp.landing_distribution(prod, bank, c, ASOF, views, r0)
    probs = [p for _, p in dist]
    assert sum(probs) == pytest.approx(1.0)
    fn = sp.scenario_rate_fn(r0, views, asof=ASOF)
    base = sp.fair_price(prod, c, fn)
    mean = sum(px * p for px, p in dist)
    assert mean == pytest.approx(base, abs=1e-6)    # centred: E[dist] = expected-path fair
    assert len(dist) > 2                            # genuinely multi-modal
    # option dying before ALL in-window meetings → fewer realised outcomes
    sep_opt = date(2026, 9, 11)
    dist_sep = sp.landing_distribution(prod, bank, c, ASOF, views, r0, upto=sep_opt)
    assert len(dist_sep) == 1                       # nothing decided by then → point mass
    assert dist_sep[0][0] == pytest.approx(base, abs=1e-9)


# ── midcurves ────────────────────────────────────────────────────────────────
def test_midcurve_expiry_and_underlying():
    prod = sp.PRODUCTS["SFRA Comdty"]
    rows = sp.midcurve_expiries(prod, ASOF, 6)
    sep = [r for r in rows if r.month == "Sep-26"][0]
    assert sep.expiry == date(2026, 9, 11)          # same Friday as the standard Sep option
    assert sp.option_underlying_mc(prod, 2026, 9).code == "SFRU7"   # exercises 12mo out
    assert sp.midcurve_expiries(sp.PRODUCTS["TKYA Comdty"], ASOF, 12) == []  # no options
    L = sp.landings(prod, sp.BANKS["FED"], ASOF, [], 4.375, 6, include_midcurves=True)
    assert any(x.series == "1Y MC" for x in L) and any(x.series == "Std" for x in L)
    mc = [x for x in L if x.series == "1Y MC"][0]
    assert mc.underlying.year >= 2027               # deferred underlying


# ── realized fixings averaging ───────────────────────────────────────────────
def test_realized_stub_avg_carry_convention():
    """Weekend/holiday days carry the prior fixing — Fri 4.0 covers Sat+Sun."""
    bank = sp.BANKS["FED"]
    fx = {}
    d = date(2026, 6, 15)
    while d < date(2026, 8, 12):
        if d.weekday() < 5:
            fx[d.isoformat()] = 4.25 if d < date(2026, 7, 10) else 4.00
        d += timedelta(days=1)
    avg = sp.realized_stub_avg(bank, date(2026, 6, 17), ASOF, fixings=fx)
    days = list(sp._daterange(date(2026, 6, 17), ASOF))
    want = sum(4.25 if x < date(2026, 7, 10) else 4.00 for x in days) / len(days)
    assert avg == pytest.approx(want, abs=1e-9)
    assert sp.realized_stub_avg(bank, date(2026, 6, 17), ASOF, fixings={}) is None
    thin = {date(2026, 8, 1).isoformat(): 4.0}      # doesn't reach back to the start
    assert sp.realized_stub_avg(bank, date(2026, 6, 17), ASOF, fixings=thin) is None


# ── decision-day helper (Home banner/popup times) ────────────────────────────
def test_decisions_today():
    assert sp.decisions_today(date(2026, 8, 11)) == []              # no meetings that day
    # 16 Sep 2026 is a DOUBLE decision day: FOMC + Copom (Copom shows 17:30 ET
    # = 18:30 São Paulo, after the B3 close)
    both = sp.decisions_today(date(2026, 9, 16))
    assert [d["bank"] for d in both] == ["FED", "BCB"]
    assert both[0]["t"] == "14:00" and both[1]["t"] == "17:30"
    ecb = sp.decisions_today(date(2026, 9, 10))
    assert [d["bank"] for d in ecb] == ["ECB"] and ecb[0]["t"] == "08:15"
    boe = sp.decisions_today(date(2026, 9, 17))
    assert [d["bank"] for d in boe] == ["BOE"] and boe[0]["t"] == "07:00"


def test_decision_alerts_registered():
    """The decision-day banner/popup ride the report-alert rail: each release name
    emitted by stirpaths.decisions_today must map to an Alert Settings toggle
    (unmapped names are fail-open, but then Ben couldn't switch them off)."""
    from src import alerts
    assert alerts.key_for_release("FOMC rate decision") == "fomc"
    assert alerts.key_for_release("ECB rate decision") == "ecb_decision"
    assert alerts.key_for_release("BoE rate decision") == "boe_decision"
    for key in ("fomc", "ecb_decision", "boe_decision"):
        assert alerts.ALERT_REPORTS[key]["group"] == "Central banks"
        assert alerts.alert_enabled(key, "banner") in (True, False)   # flags readable


def test_decision_time_dst_misalignment():
    """29 Oct 2026: Europe has left summer time (25 Oct) but the US hasn't (1 Nov),
    so the ECB's 14:15 CET lands 09:15 ET — an hour later than usual. 5 Nov 2026
    BoE: both sides on winter time again → the normal 07:00 ET."""
    ecb = sp.decisions_today(date(2026, 10, 29))
    assert [d["bank"] for d in ecb] == ["ECB"] and ecb[0]["t"] == "09:15"
    boe = sp.decisions_today(date(2026, 11, 5))
    assert [d["bank"] for d in boe] == ["BOE"] and boe[0]["t"] == "07:00"


# ── implied odds ─────────────────────────────────────────────────────────────
def test_implied_odds():
    assert sp.implied_odds(0.2) == ("hold", 0.0)
    d, p = sp.implied_odds(-18.0)
    assert d == "cut" and p == pytest.approx(0.72)
    d, p = sp.implied_odds(30.0)
    assert d == "hike" and p == 1.0                                 # capped at one full step
