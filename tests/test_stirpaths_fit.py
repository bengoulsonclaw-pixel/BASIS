"""Locks on the full-accuracy fit machinery (2026-08-15 build): serial-month
contracts, the display-decoupled fit universe, the market-solved front stub,
clean-month anchoring and the structural pinned/interpolated rule. Pure maths —
no feed, no Streamlit (the store is monkeypatched where needed)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src import stirpaths as sp
from src.fedpath import price

ASOF = date(2026, 8, 14)


# ── serial contracts ─────────────────────────────────────────────────────────
def test_serial_strip_skips_imm_months_and_dead_contracts():
    er = sp.PRODUCTS["ERA Comdty"]
    serials = sp.serial_strip(er, ASOF, months=10)
    assert serials, "no ER serials generated"
    assert all(c.month not in (3, 6, 9, 12) for c in serials)
    assert all(sp.fut_last_trade(er, c) >= ASOF for c in serials)
    # windows are the standard [3rd Wed, 3rd Wed +3M) construction
    for c in serials:
        assert c.start == sp.third_wednesday(c.year, c.month)


def test_estr_serial_outlives_named_month():
    # TKY (arrears) July serial trades to its window END in October — it must
    # still be in the August strip; advance-fixed ER July died before August.
    tky = sp.PRODUCTS["TKYA Comdty"]
    codes = {c.code for c in sp.serial_strip(tky, ASOF)}
    assert "TKYN6" in codes
    er = sp.PRODUCTS["ERA Comdty"]
    er_codes = {c.code for c in sp.serial_strip(er, ASOF)}
    assert "ERN6" not in er_codes


# ── effective-date conventions ───────────────────────────────────────────────
def test_bank_effective_dates():
    # BoE: same day. Fed/ECB: next business day (following-Wednesday for the
    # ECB was tried and rejected — the market prices next-day, see the code note).
    assert sp.bank_effective_date("BOE", date(2026, 9, 17)) == date(2026, 9, 17)
    assert sp.bank_effective_date("FED", date(2026, 9, 16)) == date(2026, 9, 17)
    assert sp.bank_effective_date("ECB", date(2026, 9, 10)) == date(2026, 9, 11)
    # Friday decision rolls over the weekend for next-bday banks
    assert sp.bank_effective_date("FED", date(2026, 10, 30)) == date(2026, 11, 2)


# ── solve_stub: recover a mid-window realized average from prices ────────────
def test_solve_stub_recovers_planted_realized_average():
    """Plant a world where the o/n rate CUT 25bp mid-front-window, price the
    contracts exactly, then check solve_stub recovers both the blended realized
    average and clean forward odds — without being told about the cut."""
    bank = sp.BANKS["BOE"]
    p = sp.PRODUCTS["SFIA Comdty"]
    contracts = sp.strip(p, ASOF, 6)
    r_now = 3.75
    r_old = 4.00                                   # before a cut on 10 Jul
    cut_day = date(2026, 7, 10)

    def world(d: date) -> float:                   # true realized+expected path
        return r_old if d < cut_day else r_now

    prices = [price(c, world, compound=False) for c in contracts]
    ip = sp.implied_path(bank, contracts, prices, ASOF, r_now,
                         solve_stub=True)
    # true realized average over the front window's elapsed days
    front = contracts[0]
    days = [d for d in sp._daterange(front.start, front.end) if d < ASOF]
    true_avg = float(np.mean([world(d) for d in days]))
    assert ip.stub == pytest.approx(true_avg, abs=0.005)
    # and the forward path is flat: no phantom odds at the next meetings
    assert max(abs(float(b)) for b in ip.per_meeting_bp) < 2.0


def test_solve_stub_none_when_nothing_elapsed():
    bank = sp.BANKS["BOE"]
    p = sp.PRODUCTS["SFIA Comdty"]
    contracts = [c for c in sp.strip(p, ASOF, 6) if c.start > ASOF]
    prices = [price(c, lambda d: 3.75, compound=False) for c in contracts]
    ip = sp.implied_path(bank, contracts, prices, ASOF, 3.75, solve_stub=True)
    assert ip.stub is None


# ── fit universe (store-patched) ─────────────────────────────────────────────
@pytest.fixture()
def fake_store(monkeypatch):
    """A store where only specific codes price — fit_instruments must include
    exactly those and never silently mock the rest."""
    def install(prices: dict):
        monkeypatch.setattr(sp, "_load_strip_store",
                            lambda: {"asof": ASOF.isoformat(),
                                     "prices": prices, "settles": {}})
    return install


def test_fit_instruments_store_filter_and_exclusions(fake_store):
    fake_store({"SFIU6": 96.155, "SFIZ6": 95.945, "SOOU6": 96.25})
    owners, contracts, spreads, prices = sp.fit_instruments("BOE", ASOF)
    codes = [c.code for c in contracts]
    assert set(codes) == {"SFIU6", "SFIZ6"}        # store-priced only...
    assert "SOOU6" not in codes                    # ...and SOO always excluded


def test_fit_instruments_drops_only_wholly_dead_contracts(fake_store):
    # With per-window-start stub groups an 80%-elapsed front HELPS (its own
    # group absorbs straddled history) — only >95% elapsed drops. SFIM6
    # (Jun17–Sep16): kept at 14 Aug (64%) and 1 Sep (84%), out on 14 Sep (98%).
    fake_store({"SFIM6": 96.2525, "SFIU6": 96.155})
    _, contracts, _, _ = sp.fit_instruments("BOE", date(2026, 9, 1))
    assert "SFIM6" in [c.code for c in contracts]
    _, contracts_late, _, _ = sp.fit_instruments("BOE", date(2026, 9, 14))
    assert "SFIM6" not in [c.code for c in contracts_late]
    _, contracts_aug, _, _ = sp.fit_instruments("BOE", ASOF)
    assert "SFIM6" in [c.code for c in contracts_aug]


def test_clean_month_anchor_reads_no_meeting_month(fake_store):
    # Aug-26 has no FOMC: the FF August contract is a pure read of the current
    # effective rate; with FED basis 0 the implied policy equals it.
    fake_store({"FFQ6": 96.3675})
    anchor = sp.clean_month_anchor("FED", ASOF)
    assert anchor is not None
    val, code = anchor
    assert code == "FFQ6"
    assert val == pytest.approx(3.6325, abs=1e-6)
    # September HAS a meeting — a September-only store must give no anchor
    fake_store({"FFU6": 96.33})
    assert sp.clean_month_anchor("FED", ASOF) is None


def _fed_world_prices(codes: list[str]) -> dict[str, float]:
    """Exact prices for a planted Fed world (eff 3.6325, +7.5bp on 17 Sep,
    +8bp more on 29 Oct, +6 on 10 Dec) — self-consistent to machine precision.
    The old hand-typed fixture hid a ~7bp FFV6-vs-FFX6 standoff on the post-
    October level; the robust refit now correctly refuses to arbitrate the
    only two witnesses disagreeing, so incoherent fixtures no longer pin."""
    steps = [(date(2026, 9, 17), 3.7075), (date(2026, 10, 29), 3.7875),
             (date(2026, 12, 10), 3.8475)]

    def world(d: date) -> float:
        r = 3.6325
        for b, v in steps:
            if d >= b:
                r = v
        return r

    out = {}
    for p in (sp.PRODUCTS["FFA Comdty"], sp.PRODUCTS["SFRA Comdty"]):
        for c in sp.strip(p, ASOF, 13 if not p.quarterly else 12):
            if c.code in codes:
                out[c.code] = round(price(c, world, compound=False), 6)
    assert set(out) == set(codes)
    return out


def test_bank_fit_pins_monthly_isolated_meetings(fake_store):
    # With FF monthlies through Dec, Sep/Oct/Dec FOMCs are each isolated by
    # their own month -> pinned; far meetings (quarterly-covered only) are not.
    fake_store(_fed_world_prices(["FFQ6", "FFU6", "FFV6", "FFX6", "FFZ6",
                                  "SFRU6", "SFRZ6", "SFRH7", "SFRM7"]))
    bf = sp.bank_fit("FED", ASOF)
    assert bf is not None
    by_date = dict(zip((m.isoformat() for m in bf.implied.meetings), bf.pinned))
    assert by_date["2026-09-16"] is True
    assert by_date["2026-10-28"] is True
    assert by_date["2027-03-17"] is False          # beyond the monthlies


def test_bank_fit_anchor_refines_r0(fake_store):
    # r0 defaults to the registry (3.625) but the clean month says 3.6325 —
    # the fit's seg0 must sit on the anchor, not the registry.
    fake_store({"FFQ6": 96.3675, "FFU6": 96.33, "SFRU6": 96.215, "SFRZ6": 96.065})
    bf = sp.bank_fit("FED", ASOF)
    assert bf.implied.seg_rates[0] == pytest.approx(3.6325, abs=1e-6)


# ── review-driven locks (2026-08-15 adversarial pass) ────────────────────────
def test_pull_universe_covers_fit_universe(monkeypatch):
    """The morning pull must fetch every contract any bank's fit can want —
    pull/fit divergence silently starves the fit after the next good pull."""
    pull_codes = {c.code for _, c in sp.pull_universe(ASOF)}
    monkeypatch.setattr(sp, "_load_strip_store", lambda: {})   # demo: full wish-list
    for bk in sp.BANKS:
        _, contracts, _, _ = sp.fit_instruments(bk, ASOF)
        missing = {c.code for c in contracts} - pull_codes
        assert not missing, f"{bk}: fit wants codes the pull never fetches: {missing}"


def test_solve_stub_groups_absorb_move_between_window_starts():
    """Review scenario: a 25bp cut lands BETWEEN different window starts (June
    quarterly vs July serial) — one shared stub scalar printed ~3.5bp of
    phantom odds; per-start groups must absorb it to under 1bp."""
    bank = sp.BANKS["ECB"]
    tky = sp.PRODUCTS["TKYA Comdty"]
    cand = sp.strip(tky, ASOF, 6) + sp.serial_strip(tky, ASOF)
    contracts = [c for c in cand
                 if sp.fut_last_trade(tky, c) >= ASOF
                 and (min(ASOF, c.end) - c.start).days
                 / max(1, (c.end - c.start).days) <= 0.70]
    r_now, r_old, cut = 2.17, 2.42, date(2026, 7, 24)
    prices = [price(c, lambda d: r_old if d < cut else r_now, compound=False)
              for c in contracts]
    ip = sp.implied_path(bank, contracts, prices, ASOF, r_now,
                         solve_stub=True, lam=2e-2)
    assert max(abs(float(b)) for b in ip.per_meeting_bp) < 1.0


def test_fit_instruments_rejects_implausible_prices(fake_store):
    """First live morning (2026-08-19): the pull returned ERJ7=201.0 for a
    barely-listed far serial — not a price — and the fit printed thousands of
    phantom bp. Implausible store values must never reach a fit."""
    fake_store({"SFIU6": 96.155, "SFIZ6": 95.945, "SFIH7": 201.0})
    _, contracts, _, _ = sp.fit_instruments("BOE", ASOF)
    assert "SFIH7" not in [c.code for c in contracts]
    assert {"SFIU6", "SFIZ6"} <= {c.code for c in contracts}


def test_clean_month_anchor_rejects_future_clean_months(fake_store):
    """Calendar-roll trap (bit for real on 8 Sep 2026): a FUTURE no-meeting
    month (Nov-26) sits after the Sep+Oct decisions and reads the POST-move
    rate — anchoring r0 there printed a phantom -30% September cut into a
    hiking market. Clean = no decision between ASOF and the month's end."""
    sep8 = date(2026, 9, 8)
    # November monthly priced, no earlier monthlies: must NOT anchor
    fake_store({"SERX6": 96.13})
    assert sp.clean_month_anchor("FED", sep8) is None
    # ...but the CURRENT month anchors when it is genuinely meeting-free
    fake_store({"FFQ6": 96.3675})
    assert sp.clean_month_anchor("FED", date(2026, 8, 14)) is not None


def test_dissenting_contract_rejected_not_accommodated():
    """The 15 Sep 2026 Euribor sawtooth, distilled: mark ONE contract (the
    front €STR quarterly) 11bp off an otherwise exactly-consistent world.
    Least squares ACCOMMODATES a dissenter — it bends the path (Oct +66 /
    Dec −15 / Feb +51 printed as market pricing) while the dissenter's own
    residual stays small, so plain residual-based downweighting misses it.
    The leverage-adjusted Tukey refit must instead REJECT the mark: path
    recovered near the planted one, the 11bp left sitting on the culprit."""
    bank = sp.BANKS["ECB"]
    tky = sp.PRODUCTS["TKYA Comdty"]
    er = sp.PRODUCTS["ERA Comdty"]
    # the REAL fit complex: both families — rejection needs a consensus to
    # outvote the dissenter (TKY alone was too thin to overrule it)
    cand = ([(er, c) for c in sp.strip(er, ASOF, 6)]
            + [(tky, c) for c in sp.strip(tky, ASOF, 6)]
            + [(tky, c) for c in sp.serial_strip(tky, ASOF)])
    cand = [(p, c) for p, c in cand
            if sp.fut_last_trade(p, c) >= ASOF
            and (min(ASOF, c.end) - c.start).days
            / max(1, (c.end - c.start).days) <= 0.95]
    contracts = [c for _, c in cand]
    spreads = [p.spread_bp for p, _ in cand]
    ups = [m for m in bank.meetings if m >= ASOF]
    steps = {sp.bank_effective_date(bank, ups[1]): 0.50,
             sp.bank_effective_date(bank, ups[3]): 0.25}

    def world(d: date) -> float:
        return 2.19 + sum(v for b, v in steps.items() if d >= b)

    prices = [price(c, world, compound=False) - s / 100.0
              for c, s in zip(contracts, spreads)]
    planted = [50.0 if m == ups[1] else (25.0 if m == ups[3] else 0.0)
               for m in ups]

    def fit(px):
        return sp.implied_path(bank, contracts, px, ASOF, 2.19,
                               spreads_bp=spreads, solve_stub=True, lam=2e-2)

    base = fit(prices)                 # ~2.3bp: the D2 ramp spreading a +50
    assert max(abs(float(b) - w) for b, w      # step — the smoothing's known
               in zip(base.per_meeting_bp, planted)) < 3.5   # cost, not noise
    # poison the front forward quarterly 11bp cheap (= demands a bigger move)
    culprit = next(i for i, c in enumerate(contracts)
                   if c.start > ASOF and c.month in (3, 6, 9, 12))
    poisoned = list(prices)
    poisoned[culprit] -= 0.11
    ip = fit(poisoned)
    errs = [abs(float(b) - w) for b, w in zip(ip.per_meeting_bp, planted)]
    assert max(errs) < 5.0, f"path bent toward the dissenter: {errs}"
    assert abs(float(ip.residual_bp[culprit])) > 8.0, \
        "the off mark was accommodated instead of rejected"
