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


def test_fit_instruments_drops_nearly_dead_contracts(fake_store):
    # SFIM6 (Jun-Sep window) is ~64% elapsed on 14 Aug — kept; give it a
    # late-August asof where it crosses 70% and it must drop out.
    fake_store({"SFIM6": 96.2525, "SFIU6": 96.155})
    late = date(2026, 9, 1)                        # ~84% of Jun17-Sep16 elapsed
    _, contracts, _, _ = sp.fit_instruments("BOE", late)
    assert "SFIM6" not in [c.code for c in contracts]
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


def test_bank_fit_pins_monthly_isolated_meetings(fake_store):
    # With FF monthlies through Dec, Sep/Oct/Dec FOMCs are each isolated by
    # their own month -> pinned; far meetings (quarterly-covered only) are not.
    fake_store({"FFQ6": 96.3675, "FFU6": 96.33, "FFV6": 96.28, "FFX6": 96.235,
                "FFZ6": 96.165, "SFRU6": 96.215, "SFRZ6": 96.065,
                "SFRH7": 95.98, "SFRM7": 95.945})
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
