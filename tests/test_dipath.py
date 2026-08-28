"""Locks on the Brazilian DI engine (2026-08-19 build): B3 DI1 zeros —
rate-quoted, business-days/252 compounded from asof to the 1st business day of
the named month — fitted in log(1+r) space where the Copom-step system is
linear. Pure maths; the store is monkeypatched where needed."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src import stirpaths as sp

ASOF = date(2026, 8, 19)
BANK = sp.BANKS["BCB"]
DI = sp.PRODUCTS["OD1 Comdty"]


# ── contract construction ────────────────────────────────────────────────────
def test_di_strip_maturities_are_first_bdays():
    cs = sp.di_strip(DI, ASOF, 6)
    assert [c.code for c in cs[:3]] == ["ODU6", "ODV6", "ODX6"]
    for c in cs:
        assert c.start == ASOF                     # a zero from TODAY
        assert c.end.day <= 3 and c.end.weekday() < 5
        assert c.end > ASOF
    # Nov 1 2026 is a Sunday — the Nov maturity must roll to Mon 2 Nov
    nov = next(c for c in cs if c.month == 11)
    assert nov.end == date(2026, 11, 2)


def test_di_excluded_from_morning_pull():
    # root unverified + fresh -4002 history: OD must NOT be in the pull set
    assert not any(c.code.startswith("OD") for _, c in sp.pull_universe(ASOF))


# ── the log-space fit ────────────────────────────────────────────────────────
def test_di_implied_path_recovers_planted_copom_cuts():
    """Price the DI strip exactly under a −50bp-at-each-of-3-meetings world,
    fit blind, recover the moves (±3bp smoothing smear at the cliff is the
    same order the price-space fits show)."""
    cs = sp.di_strip(DI, ASOF, 10)
    ups = [m for m in BANK.meetings if m > ASOF][:3]
    fn = sp.overnight_rate_fn(13.65, ups, [-50.0, -50.0, -50.0])
    quotes = [sp.di_fair_rate(c, fn, ASOF) for c in cs]
    ip = sp.di_implied_path(BANK, cs, quotes, ASOF, 13.65)
    got = {m.isoformat(): float(bp) for m, bp in zip(ip.meetings, ip.per_meeting_bp)}
    for m in ups:
        assert got[m.isoformat()] == pytest.approx(-50.0, abs=3.0)
    for k, v in got.items():
        if k not in {m.isoformat() for m in ups}:
            assert abs(v) < 3.0                     # nothing planted elsewhere
    assert max(abs(float(r)) for r in ip.residual_bp) < 1.0
    assert ip.stub is None                          # zeros have no stub


def test_di_tail_meeting_guard():
    # a meeting within ~a week of the LAST maturity has only a sliver of
    # coverage — it printed a −42bp smoothing artifact and must be excluded
    cs = sp.di_strip(DI, ASOF, 6)                   # last maturity 1 Feb 27
    quotes = [13.65] * len(cs)
    ip = sp.di_implied_path(BANK, cs, quotes, ASOF, 13.65)
    assert date(2027, 1, 27) not in ip.meetings     # eff 28 Jan, mat 1 Feb


def test_di_flat_world_prices_flat():
    cs = sp.di_strip(DI, ASOF, 8)
    quotes = [sp.di_fair_rate(c, lambda d: 13.65, ASOF) for c in cs]
    assert all(q == pytest.approx(13.65, abs=1e-9) for q in quotes)
    ip = sp.di_implied_path(BANK, cs, quotes, ASOF, 13.65)
    assert max(abs(float(b)) for b in ip.per_meeting_bp) < 0.5


# ── bank_fit dispatch ────────────────────────────────────────────────────────
def test_bcb_fit_demo_pins_every_meeting(monkeypatch):
    """Consecutive DI maturities difference out single Copom meetings — the
    structural pin should hold at (nearly) every covered meeting, the DI
    curve's built-in advantage over IMM quarterlies."""
    monkeypatch.setattr(sp, "_load_strip_store", lambda: {})   # demo mode
    bf = sp.bank_fit("BCB", ASOF)
    assert bf is not None and bf.n_instruments >= 12
    assert all(bf.pinned[:6])
    # front-DI anchor: no meeting before ODU6's maturity -> current-CDI read
    assert bf.anchor is not None and bf.anchor[1] == "ODU6"


def test_bcb_fit_live_store_without_di_returns_none(monkeypatch):
    # live store present but no DI codes: NEVER mock inside a real fit
    monkeypatch.setattr(sp, "_load_strip_store",
                        lambda: {"asof": ASOF.isoformat(),
                                 "prices": {"SFRU6": 96.215}, "settles": {}})
    assert sp.bank_fit("BCB", ASOF) is None


def test_bcb_fit_gate_rejects_implausible_rates(monkeypatch):
    cs = sp.di_strip(DI, ASOF, 4)
    good = {c.code: 13.6 for c in cs}
    good[cs[1].code] = 201.0                        # the ERJ7-class garbage
    monkeypatch.setattr(sp, "_load_strip_store",
                        lambda: {"asof": ASOF.isoformat(),
                                 "prices": good, "settles": {}})
    bf = sp.bank_fit("BCB", ASOF)
    assert bf is not None
    assert cs[1].code not in [c.code for c in bf.implied.contracts]


# ── calendar hygiene ─────────────────────────────────────────────────────────
def test_copom_calendar_sane():
    ms = sp.BCB_DECISIONS
    assert len([m for m in ms if m.year == 2026]) == 8
    assert len([m for m in ms if m.year == 2027]) == 8
    assert ms == sorted(ms)
    # decision = day two of a Tue-Wed (occasionally Mon-Tue) pair — the real
    # published calendar has Tuesday decisions in late 2027
    assert all(m.weekday() in (1, 2) for m in ms)
    assert date(2026, 9, 16) in ms                  # confirmed vs the BCB API
