"""RV Trade Tickets turn the Curve / RV book into backtested, sized tickets.

The numbers that must not drift are the ones a trader would act on:

  * a fade episode is counted ONCE per stretched stint, entered on a fresh crossing, and
    followed to a FROZEN target / stop — not re-entered every day |z| stays beyond the band;
  * a day that gaps through both levels is scored as a loss (stop first), never a free win;
  * sizing is honest — a reconciled point value gives a lot count, a single-market curve is
    DV01-weighted off the editable CTD table, and anything needing an FX or notional ratio we
    don't hold is returned "manual" with no fabricated contract count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import curvemon, futyield, rvtickets


# ── fade-rule backtest ────────────────────────────────────────────────────────
def _series(segments: list[float]) -> pd.Series:
    return pd.Series(segments, index=pd.bdate_range("2016-01-01", periods=len(segments)))


def _warm(window: int, reps: int = 2) -> list[float]:
    """±1 noise → rolling mean ≈ 0, std ≈ 1.2 once a spike enters the window, so a push to
    +3.0 lands at z ≈ 2.4 — inside the [threshold, INVAL_SIGMA) band where the stop still
    sits ABOVE the entry."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(window * reps)]


def test_a_clean_reversion_is_a_win_and_pays_positive_expectancy():
    """A spread that spikes to ~+2.4σ and falls back through its mean is the textbook fade:
    target (the mean) is reached before the +3σ stop, so it must score as a win with a
    positive per-trade expectancy."""
    window = 20
    seg = _warm(window)
    for _ in range(3):                                   # three identical, cleanly reverting spikes
        seg += [3.0] * 6 + [1.0, 0.0, 0.0] + _warm(window, 1)
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] >= 2
    assert bt["wins"] == bt["n"] and bt["losses"] == 0          # every spike reverted
    assert bt["win_rate"] == 1.0
    assert bt["expectancy_sigma"] > 0
    assert bt["median_days_win"] == bt["median_days_win"]       # finite


def test_a_runaway_stretch_is_a_loss_not_a_win():
    """Fading works until it doesn't: a spread that keeps going once stretched hits the stop.
    The rule must book that as a loss with negative expectancy, or the screen would flatter
    every trend."""
    window = 20
    seg = _warm(window) + [3.0, 4.5, 6.0, 8.0, 10.0]            # crosses +2σ then runs away
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] >= 1
    assert bt["losses"] >= 1 and bt["wins"] == 0
    assert bt["win_rate"] == 0.0
    assert bt["expectancy_sigma"] < 0


def test_one_long_stretched_stint_is_a_single_episode():
    """The whole point of entering on a FRESH crossing and holding one position: a spread that
    sits beyond the band for forty sessions is one trade, not forty. Counting every day would
    manufacture a sample out of a single event."""
    window = 20
    seg = _warm(window) + [3.0] * 40 + [1.0, 0.0, 0.0] + _warm(window, 1)
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=12.0)
    assert bt["n"] == 1


def test_a_stretch_that_never_resolves_times_out_rather_than_scoring():
    """A spread that stays wedged between target and stop past the hold cap is marked out at
    the market as a timeout — a non-win that is neither a free win nor a stop-out."""
    window = 20
    seg = _warm(window) + [3.0] * 60                           # held ~60 bars; cap at 5×10 = 50
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] == 1 and bt["timeouts"] == 1 and bt["wins"] == 0


def test_a_partial_take_profit_wins_a_stretch_the_full_mean_would_miss():
    """A spread that reverts only part of the way and stalls is a timeout against the full mean,
    but a win for a nearer take-profit — the whole point of the tp_z option (more, smaller wins)."""
    window = 20
    seg = _warm(window) + [3.0, 3.0] + [0.5] * 60              # reverts to ~0.5σ and holds, never 0
    s = _series(seg)
    full = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0, tp_z=0.0)
    partial = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0, tp_z=0.5)
    assert full["wins"] == 0 and full["timeouts"] == 1         # never reaches the mean
    assert partial["wins"] == 1                                # but clears the 0.5σ target
    assert partial["win_rate"] > full["win_rate"]


def test_an_unresolved_tail_is_dropped_not_counted():
    """A fresh stretch in the last few sessions can't be followed to target OR stop, so it
    must be excluded — counting it would bias the sample toward whatever the data happens to
    end on."""
    window = 20
    seg = _warm(window) + [3.0, 3.0]                           # crosses at the very end, no room
    bt = rvtickets.backtest_spread(_series(seg), window, threshold=2.0, half_life=10.0)
    assert bt["n"] == 0


# ── rolls: a contract-pair switch is not a market move ────────────────────────
def _roll_at(s: pd.Series, i: int) -> pd.Series:
    b = pd.Series(False, index=s.index)
    b.iloc[i] = True
    return b


def test_a_roll_that_resets_the_level_is_not_a_win():
    """The 2026-10-06 bug: HG 1−2 jumped 2–5.6¢ on every roll, so a fade held across one was
    'won' by the switch to a new contract pair. Roll-neutral, the position only earns the moves
    within a pair — a reset that happens AT the roll leaves it where it was (here: timeout)."""
    window = 20
    seg = _warm(window) + [3.0, 3.0, 3.0] + [0.0] * 60         # level resets on the roll bar
    s = _series(seg)
    roll = _roll_at(s, 2 * window + 3)
    naive = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0)
    fixed = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0, breaks=roll)
    assert naive["wins"] == 1                                  # the old, flattering read
    assert fixed["wins"] == 0 and fixed["n"] == 1 and fixed["timeouts"] == 1


def test_a_roll_jump_against_the_fade_is_not_a_stop():
    """The mirror image: a roll that jumps the level AWAY from the mean must not stop a fade
    out; the within-pair reversion that follows is what the position actually earns."""
    window = 20
    seg = _warm(window) + [3.0, 3.0] + [10.0, 0.0] + [0.0] * 10   # jump on the roll, then −10
    s = _series(seg)
    roll = _roll_at(s, 2 * window + 2)
    naive = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0)
    fixed = rvtickets.backtest_spread(s, window, threshold=2.0, half_life=10.0, breaks=roll)
    assert naive["losses"] == 1
    assert fixed["wins"] == 1 and fixed["losses"] == 0
    assert fixed["expectancy_sigma"] > 0


def test_no_breaks_is_the_plain_level_path():
    """Spreads with no rolling legs (yield curves) pass breaks=None or all-False — identical
    results either way, so the fix cannot move a curve spread's backtest."""
    window = 20
    seg = _warm(window)
    for _ in range(3):
        seg += [3.0] * 6 + [1.0, 0.0, 0.0] + _warm(window, 1)
    s = _series(seg)
    a = rvtickets._episodes(s, window, 2.0, 50)
    b = rvtickets._episodes(s, window, 2.0, 50, breaks=pd.Series(False, index=s.index))
    assert a == b and len(a) >= 2


# ── flat by first notice / last trade: never ride a front into delivery ───────
def _exit_at(s: pd.Series, i: int) -> pd.Series:
    """Every date's front must be flat by the close of bar i (one contract, one exit-by)."""
    return pd.Series(s.index[i], index=s.index)


def test_a_fade_is_force_exited_at_the_fronts_exit_by_as_a_notice():
    """The 2026-10-06 copper finding: the old backtest held calendars through the delivery
    window, where contango compresses. A stretch still open at the front's exit-by is closed at
    that close — outcome 'notice', marked at the market, a non-win — even though, left alone, it
    would have reverted to the mean a few sessions later."""
    window = 20
    seg = _warm(window) + [3.0, 3.0, 2.5, 2.5, 2.5] + [0.0] * 10
    s = _series(seg)
    entry_i = 2 * window
    free = rvtickets.backtest_spread(s, window, 2.0, half_life=10.0)
    flat = rvtickets.backtest_spread(s, window, 2.0, half_life=10.0,
                                     exit_by=_exit_at(s, entry_i + 3))
    assert free["wins"] == 1                                   # held on, it reverted
    assert flat["n"] == 1 and flat["notices"] == 1 and flat["wins"] == 0
    e = rvtickets._episodes(s, window, 2.0, 50, exit_by=_exit_at(s, entry_i + 3))[0]
    assert e["outcome"] == "notice" and e["bars"] == 3
    assert e["exit"] == pytest.approx(2.5) and e["pnl"] == pytest.approx(0.5)   # short 3.0 → 2.5


def test_no_entry_inside_the_exit_window():
    """A fresh crossing on or after the front's exit-by is not a trade a non-delivery desk can
    put on (it would be closed the same day) — skipped, not entered."""
    window = 20
    seg = _warm(window) + [3.0, 3.0] + [0.0] * 10
    s = _series(seg)
    entry_i = 2 * window
    for ex_i in (entry_i, entry_i - 5):                        # exit-by ON / BEFORE the crossing
        assert rvtickets._episodes(s, window, 2.0, 50, exit_by=_exit_at(s, ex_i)) == []


def test_an_exit_by_beyond_the_data_leaves_the_position_open_not_noticed():
    """Today's front has weeks to run: an unresolved fade is an OPEN ledger position (and an
    unscored backtest tail), never pre-emptively closed as a notice."""
    window = 20
    seg = _warm(window) + [3.0, 3.0, 3.0]
    s = _series(seg)
    later = pd.Series(s.index[-1] + pd.Timedelta(days=30), index=s.index)
    eps = rvtickets._episodes(s, window, 2.0, 50, keep_open=True, exit_by=later)
    assert [e["outcome"] for e in eps] == ["open"]
    assert rvtickets._episodes(s, window, 2.0, 50, exit_by=later) == []


def test_no_exit_rule_is_the_unconstrained_path():
    """Spreads with no flat rule (curves, ratios — exit_by None or all-NaT) are untouched."""
    window = 20
    seg = _warm(window)
    for _ in range(3):
        seg += [3.0] * 6 + [1.0, 0.0, 0.0] + _warm(window, 1)
    s = _series(seg)
    a = rvtickets._episodes(s, window, 2.0, 50)
    b = rvtickets._episodes(s, window, 2.0, 50,
                            exit_by=pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]"))
    assert a == b and len(a) >= 2


@pytest.mark.parametrize("tkr, month, flat_by, exit_by, kind", [
    # COMEX copper: FND = last bd of the prior month; out 3 bd before (Thanksgiving skipped)
    ("HGA Comdty", "2026-12-01", "2026-11-30", "2026-11-24", "first notice"),
    # NYMEX WTI: FND = bd after last trade (LTD 3 bd before the 25th of the prior month)
    ("CLA Comdty", "2026-11-01", "2026-10-22", "2026-10-19", "first notice"),
    # SOFR 3M is named for the START of its reference quarter: SFRU6 trades to mid-Dec
    ("SFRA Comdty", "2026-09-01", "2026-12-15", "2026-12-15", "last trade"),
    ("ERA Comdty", "2026-12-01", "2026-12-14", "2026-12-14", "last trade"),
])
def test_flat_dates_per_contract(tkr, month, flat_by, exit_by, kind):
    d = rvtickets._flat_dates(tkr, pd.Timestamp(month))
    assert d["flat_by"].isoformat() == flat_by
    assert d["exit_by"].isoformat() == exit_by
    assert d["kind"] == kind


def test_every_calendar_has_a_flat_rule_and_curves_do_not():
    cals = ["hg_cal", "cl_cal", "co_cal", "ng_cal", "ttf_cal", "sfr_cal", "er_cal", "sfi_cal"]
    for k in cals:
        assert rvtickets._flat_ticker(curvemon.SPREAD_BY_KEY[k]) is not None, k
    for k in ("us_2s10s", "wti_brent", "gc_si"):
        assert rvtickets._flat_ticker(curvemon.SPREAD_BY_KEY[k]) is None, k


def test_observed_rolls_anchor_the_last_trade_rules():
    """Observed anchor (the expiries.py discipline): every completed front run in the deep store
    must end ON the rule's last trade or one business day before it — so the exit-by derived
    from it is real, and a fade is always flat before the pair switches. Caught the SOFR/SONIA
    3M naming (reference-quarter START) on the way in; never rebaseline this to make it pass."""
    from src import expiries
    hist = curvemon.load_history()
    if hist is None or hist.empty:
        pytest.skip("no deep store on this machine")
    for key in ["hg_cal", "cl_cal", "co_cal", "ng_cal", "ttf_cal", "sfr_cal", "er_cal", "sfi_cal"]:
        spec = curvemon.SPREAD_BY_KEY[key]
        tkr = rvtickets._flat_ticker(spec)
        sched = rvtickets.flat_schedule(spec, hist)
        assert sched is not None and len(sched) > 5, key
        runs = curvemon._runs_for(hist, tkr)
        settle, _, shift = rvtickets._FLAT_RULES[tkr]
        hol = expiries._holidays_for(tkr, "")
        bad = []
        for r, (_, sc) in zip(runs.iloc[:-1].itertuples(index=False), sched.iloc[:-1].iterrows()):
            if r.month is None or pd.isna(r.month):
                continue
            y, m = expiries._shift_month(r.month.year, r.month.month, shift)
            ltd = expiries._eval(expiries.spec_for(tkr)["fut"], y, m, hol)
            end = pd.Timestamp(r.end).date()
            if not (end == ltd or end == expiries._prev_bday(ltd, hol)):
                bad.append((r.contract, end.isoformat(), ltd.isoformat()))
            assert pd.Timestamp(sc["exit_by"]) <= pd.Timestamp(r.end), (key, r.contract)
        assert len(bad) <= 0.05 * len(runs), (key, bad[:5])


# ── the ranking: forward EV scores each outcome at what it actually does ──────
def test_edge_charges_the_stop_only_to_trades_that_hit_it():
    """A notice exit or timeout that made money must not be charged the full stop. 10 trades:
    2 wins, 2 stops, 6 notice exits averaging +0.5σ, at a 2σ entry (reward 2σ, risk 1σ):
    EV = 0.2×2 − 0.2×1 + 0.6×0.5 = +0.5σ — the old 'every non-win is a stop' formula said
    0.2×2 − 0.8×1 = −0.4σ and sank exactly the calendars the notice rule produced."""
    bt = {"win_rate": 0.2, "loss_rate": 0.2, "other_rate": 0.6, "other_avg_sigma": 0.5}
    assert rvtickets.forward_ev(bt, 2.0, 1.0) == pytest.approx(0.5)


def test_edge_with_only_wins_and_stops_is_the_classic_formula():
    """No timeouts/notices (a curve spread that always resolves) → win×reward − loss×risk, so
    the change cannot move a spread whose trades all reach the target or the stop."""
    bt = {"win_rate": 0.3, "loss_rate": 0.7, "other_rate": 0.0, "other_avg_sigma": 0.0}
    assert rvtickets.forward_ev(bt, 2.5, 1.0) == pytest.approx(0.3 * 2.5 - 0.7)


def test_backtest_reports_the_outcome_mix_the_edge_needs():
    window = 20
    seg = _warm(window) + [3.0, 3.0, 2.5, 2.5, 2.5] + [0.0] * 10
    s = _series(seg)
    bt = rvtickets.backtest_spread(s, window, 2.0, half_life=10.0,
                                   exit_by=_exit_at(s, 2 * window + 3))
    assert bt["other_rate"] == 1.0 and bt["loss_rate"] == 0.0
    assert bt["other_avg_sigma"] > 0                           # the notice exit made money
    assert rvtickets.forward_ev(bt, 2.0, 1.0) == pytest.approx(bt["other_avg_sigma"])
    empty = rvtickets.backtest_spread(_series(_warm(window)), window, 2.0, half_life=10.0)
    assert rvtickets.forward_ev(empty, 2.0, 1.0) != rvtickets.forward_ev(empty, 2.0, 1.0)  # NaN


# ── sizing: the three honest regimes ──────────────────────────────────────────
def test_calendar_sizes_one_lot_per_leg_off_the_reconciled_point_value():
    """Same-product calendars reconcile to a single $/unit, so they get a real lot count —
    and a rich spread sells the front, buys the second."""
    spec = {"key": "x_cal", "group": "STIR Calendars", "unit": "bp", "dp": 1, "scale": 100.0,
            "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")]}
    row = {"level": 20.0, "invalidation": 25.0, "sigma": 5.0, "direction": -1,
           "dollar_sigma": 125.0, "dsig_sym": "$"}
    t = rvtickets.size_ticket(spec, row, risk_budget=2500.0)
    assert t["method"] == "calendar"
    assert t["lots"] == 20                                   # risk 2500 / (5 unit × $25/unit)
    assert t["risk_money"] == pytest.approx(2500.0)
    sides = {lg["label"]: lg["side"] for lg in t["structure_legs"]}
    assert sides == {"front": "Sell", "2nd": "Buy"}          # fade a rich calendar


def test_single_market_curve_is_dv01_weighted_into_two_futures():
    """A 2s10s ticket must read as two real futures in a DV01 ratio off the editable CTD
    table — buy the long end / sell the short end to fade a steep (rich) curve."""
    spec = curvemon.SPREAD_BY_KEY["us_2s10s"]
    assert rvtickets._is_dv01_sizeable(spec) is True
    row = {"level": 58.0, "invalidation": 78.0, "sigma": 10.0, "direction": -1,
           "dollar_sigma": None, "dsig_sym": "$"}
    raw_last = {"TUA Comdty": 104.0, "TYA Comdty": 110.0}
    t = rvtickets.size_ticket(spec, row, risk_budget=10_000.0,
                              raw_last=raw_last, ctd=futyield.load_ctd())
    assert t["method"] == "dv01"
    sides = {lg["label"]: lg["side"] for lg in t["structure_legs"]}
    assert sides.get("TY") == "Buy" and sides.get("TU") == "Sell"   # flattener
    assert all(lg["lots"] >= 1 for lg in t["structure_legs"])
    assert t["per_bp"] > 0 and t["risk_money"] > 0
    assert t["risk_sym"] == "$"                                     # both legs USD


def test_cross_currency_curve_is_fx_dv01_weighted_in_usd():
    """UST–Bund has a USD leg and a EUR leg; given an FX rate the sizer must put both DV01s on
    USD and emit a real two-future ticket (buy TY / sell RX to fade a rich spread), sized in $,
    rather than falling back to 'manual'."""
    spec = curvemon.SPREAD_BY_KEY["ust_bund"]
    assert rvtickets._is_dv01_sizeable(spec) is True
    row = {"level": 180.0, "invalidation": 200.0, "sigma": 10.0, "direction": -1,
           "dollar_sigma": None, "dsig_sym": "$"}
    raw_last = {"TYA Comdty": 110.0, "RXA Comdty": 133.0}
    t = rvtickets.size_ticket(spec, row, risk_budget=10_000.0, raw_last=raw_last,
                              ctd=futyield.load_ctd(), fx={"EUR": 1.08})
    assert t["method"] == "dv01" and t["risk_sym"] == "$"
    sides = {lg["label"]: lg["side"] for lg in t["structure_legs"]}
    assert sides.get("TY") == "Buy" and sides.get("RX") == "Sell"
    assert all(lg["lots"] >= 1 for lg in t["structure_legs"])
    assert t["per_bp"] > 0 and t["risk_money"] > 0
    # the SAME cross-currency spread with no FX rate on hand must not fake a count
    nofx = rvtickets.size_ticket(spec, row, 10_000.0, raw_last=raw_last, ctd=futyield.load_ctd())
    assert nofx["method"] == "manual"


def test_a_ratio_is_never_dv01_sized():
    """A metal ratio carries no CTD yield and needs a notional split we don't hold, so it stays
    'manual' however much price data is on hand — never a made-up lot count."""
    base = {"level": 85.0, "invalidation": 95.0, "sigma": 3.0, "dollar_sigma": None,
            "direction": -1, "dsig_sym": "$"}
    assert rvtickets._is_dv01_sizeable(curvemon.SPREAD_BY_KEY["gc_si"]) is False
    ratio = rvtickets.size_ticket(curvemon.SPREAD_BY_KEY["gc_si"], base, 10_000.0)
    assert ratio["method"] == "manual" and "ratio" in ratio["note"].lower()


def test_no_stop_distance_sizes_to_nothing():
    """If the invalidation sits on the entry there is no risk to size against — the sizer must
    say so, not divide by zero."""
    spec = {"key": "x_cal", "group": "STIR Calendars", "unit": "bp", "dp": 1, "scale": 100.0,
            "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")]}
    row = {"level": 25.0, "invalidation": 25.0, "sigma": 5.0, "direction": -1,
           "dollar_sigma": 125.0, "dsig_sym": "$"}
    assert rvtickets.size_ticket(spec, row, 2500.0)["method"] == "none"


# ── the assembled, ranked book (needs the deep store) ─────────────────────────
def test_tickets_rank_by_edge_and_carry_a_full_ticket():
    """End to end on whatever history the machine holds: the book must come back sorted best
    edge first, and every ticket must carry the levels and structure a desk would act on."""
    hist = curvemon.load_history()
    if hist is None or hist.empty:
        pytest.skip("no deep store on this machine")
    ts = rvtickets.tickets(window=252, threshold=1.0, history=hist)
    assert isinstance(ts, list)
    if not ts:
        pytest.skip("calm day — nothing stretched beyond 1σ")
    assert [t["score"] for t in ts] == sorted((t["score"] for t in ts), reverse=True)
    for t in ts:
        for k in ("name", "level", "objective", "invalidation", "structure", "action",
                  "rr", "ev_sigma", "bt_n", "conf", "risk_units"):
            assert k in t
        assert t["risk_units"] > 0
        assert t["structure"]
        assert t["size_method"] in {"calendar", "dv01", "manual", "none"}


def test_coffee_arb_is_sized_pound_for_pound():
    """KC−RC arb: 1 KC lot (37,500 lb) against 1.701 RC lots (10 t = 22,046 lb), so
    1¢/lb of spread is $375 per KC lot on both legs — never 1 lot per leg."""
    from src import curvemon
    spec = curvemon.SPREAD_BY_KEY["kc_rc"]
    assert spec["hedge_ratio"]["DFA Comdty"] == pytest.approx(1.701, abs=1e-3)
    # the RC leg's $ per ¢/lb at that ratio matches KC's $375
    assert 10.0 * 22.0462 * spec["hedge_ratio"]["DFA Comdty"] == pytest.approx(375.0, rel=1e-3)
    row = dict(level=133.8, invalidation=102.3, sigma=31.46, direction=1,
               dollar_sigma=31.46 * 375.0, dsig_sym="$")
    t = rvtickets.size_ticket(spec, row, 100_000)
    kc, rc = t["structure_legs"]
    assert (kc["side"], kc["lots"]) == ("Buy", 8)
    assert (rc["side"], rc["lots"]) == ("Sell", 14)
