"""The time machine: run every bank's full fit on EVERY future business day
through its calendar tail, in demo mode where the planted world is known, and
demand the fit recovers it. The class of bug this hunts — date logic that works
today and breaks when the calendar rolls — has bitten twice (the mv-window roll,
then the future-clean-month anchor that printed a phantom −30% September cut on
8 Sep 2026, ten days after shipping green). Both would have failed here on day
one. ~8s for the whole sweep.

Tolerances (calibrated 2026-09-08 against the mock noise floor):
  front 4 covered meetings that the fit itself marks PINNED  ≤ 10bp
  (worst observed 8.9bp: a singleton-pinned meeting carrying only 45%
  weight in its sole contract, against the mock's worst-case ±25 zigzag;
  the real 8-Sep anchor bug corrupted a pinned meeting by 24.6bp)
  any covered meeting                                        ≤ 22bp
The pin condition matters: BoE legitimately cannot split Sep-vs-Nov in the
last ~3 weeks of its front window (quarterlies only, no monthlies) — and the
cockpit marks that meeting ≈ interpolated, so the harness holds the tight bar
exactly where the product claims trustworthiness. Anchor/stub/r0-class bugs
corrupt PINNED meetings (the 8-Sep phantom cut was on a pinned one), so they
still fail loudly. G3 worst observed ~17bp at far ramped-smoothed meetings;
BCB worst 1.5bp (DI pair-pinning). Never widen a front breach — investigate."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src import stirpaths as sp

START = date(2026, 8, 28)
FRONT_TOL_BP = 10.0
ANY_TOL_BP = 22.0
# ECB is fit EURIBOR-LED with a €STR front-level anchor (stirpaths
# FRONT_ANCHOR_INSTRUMENTS / FRONT_ANCHOR_MEETINGS): past the near, well-
# separated meetings the €STR serials are dropped from the SHAPE — on real
# marks they inject a phantom Apr-2027 seesaw (the "dip") — so the far meetings
# identify off the 5 Euribor quarterlies alone. That lifts the FAR mock-noise
# floor ~1.6bp (measured worst 23.6bp at #7, 2027-07-22, across the sweep) while
# the PINNED FRONT stays exact (worst 0.3bp). Only the far, interpolated tail of
# the one bank whose far instrument set actually thinned gets the wider band;
# the front bar and every other bank are unchanged. Never widen the FRONT.
ANY_TOL_FAR = {"ECB": 25.0}


def _planted(bank, asof, meetings):
    step, every = sp._MOCK_STYLE[bank.key]
    ups = [m for m in bank.meetings if m > asof]
    mv = {m: (step if i % every == 0 else 0.0) for i, m in enumerate(ups)}
    return [mv.get(m, 0.0) for m in meetings]


@pytest.mark.parametrize("bk", ["FED", "ECB", "BOE", "BCB"])
def test_calendar_roll_sweep(bk, monkeypatch):
    monkeypatch.setattr(sp, "_load_strip_store", lambda: {})   # demo world
    bank = sp.BANKS[bk]
    end = max(bank.meetings) - timedelta(days=45)   # the tail-guard region has
    d = START                                       # its own (looser) physics
    while d <= end:
        if d.weekday() < 5:
            bf = sp.bank_fit(bk, d)                 # must never raise
            assert bf is not None, f"{bk} {d}: fit returned None"
            ip = bf.implied
            want = _planted(bank, d, ip.meetings)
            for i, (m, got, w, pin) in enumerate(zip(ip.meetings, ip.per_meeting_bp,
                                                     want, bf.pinned)):
                err = abs(float(got) - w)
                tol = (FRONT_TOL_BP if (i < 4 and pin)
                       else ANY_TOL_FAR.get(bk, ANY_TOL_BP))
                assert err <= tol, (f"{bk} asof {d} meeting {m} (#{i}, "
                                    f"{'PIN' if pin else '~'}): fit "
                                    f"{float(got):+.1f}bp vs planted {w:+.1f}bp "
                                    f"— {err:.1f}bp off (tol {tol})")
        d += timedelta(days=1)
