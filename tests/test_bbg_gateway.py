"""The Bloomberg-gateway guarantee (src/bbg.py, built 2026-08-18).

Two halves:
  1. STRUCTURAL — no module outside the gateway may import xbbg. This is what
     makes the duplicate-pull error class (the triple ATM pull, the twin
     volume/yields pulls, the COT/deep double store found in the 2026-08-18
     audit) impossible to reintroduce silently: every request must pass the
     chokepoint that counts and registers it.
  2. FUNCTIONAL — the gateway's accounting: hits = securities × fields; a
     (security, field) pair requested by two DIFFERENT call sites is a design
     duplicate; a re-request from the SAME site (serial fallback, forced cache
     refresh) is a retry and is ignored; documented ACCEPTED_OVERLAPS are
     reported separately, not as duplicates.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Manual, never-scheduled diagnostics may talk to xbbg directly — they are run by
# hand with the Terminal up and are not part of any pull pipeline.
_ADHOC = re.compile(r"^(diag_|probe_|pnl_|check_bloomberg|backfill_)")

# Raw blpapi (not xbbg) is allowed ONLY in the low-level diagnostics that exist
# to probe the connection itself (block detection / reachability): app.py's
# _blp_block_probe, surface_topup.py, and run_pull.py's pre-flight probe (the
# pull driver refuses to start on a -4002 block or a logged-out Terminal —
# same role, ~2s, zero data hits; added 2026-08-21).
_BLPAPI_OK = {"app.py", "surface_topup.py", "run_pull.py"}


def _repo_py_files():
    yield from (ROOT / "src").glob("*.py")
    yield from ROOT.glob("*.py")


def test_no_direct_xbbg_outside_gateway():
    bad = []
    for p in _repo_py_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel == "src/bbg.py" or _ADHOC.match(p.name):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(from xbbg\b|import xbbg\b)", txt, flags=re.M):
            bad.append(rel)
    assert not bad, (
        f"Direct xbbg import outside the src/bbg.py gateway: {bad}. Route the call "
        "through `from . import bbg as blp` so it is counted and duplicate-checked — "
        "this ban exists because three modules once pulled the same vol field every "
        "morning for weeks without anyone noticing (2026-08-18 audit).")


def test_no_raw_blpapi_outside_diagnostics():
    bad = []
    for p in _repo_py_files():
        rel = p.relative_to(ROOT).as_posix()
        if p.name in _BLPAPI_OK or _ADHOC.match(p.name):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*import blpapi\b", txt, flags=re.M):
            bad.append(rel)
    assert not bad, f"Raw blpapi use outside the connection diagnostics: {bad}"


class _FakeBlp:
    """Stands in for xbbg — records calls, returns nothing."""
    def __init__(self):
        self.calls = []

    def bdh(self, tickers=None, flds=None, **kw):
        self.calls.append(("bdh", tickers, flds))
        return None

    def bdp(self, tickers=None, flds=None, **kw):
        self.calls.append(("bdp", tickers, flds))
        return None

    def bds(self, tickers=None, flds=None, **kw):
        self.calls.append(("bds", tickers, flds))
        return None


def _gateway():
    from src import bbg, pullguard
    bbg._blp_mod = _FakeBlp()
    bbg.reset_session()
    pullguard.reset_hits()
    return bbg, pullguard


def teardown_function(_fn):
    from src import bbg
    bbg._blp_mod = None
    bbg.reset_session()


def test_hits_counted_per_security_field():
    bbg, pullguard = _gateway()
    bbg.bdh(tickers=["CLA Comdty", "COA Comdty"], flds="PX_SETTLE",
            start_date="2026-01-01", end_date="2026-01-31")          # 2×1
    bbg.bdp(["SFRU6 Comdty"], ["PX_LAST", "PX_SETTLE"])              # 1×2
    bbg.bds("CLA Comdty", "FUT_CHAIN")                               # 1×1 bulk
    assert pullguard.get_hits() == 5


def test_same_site_rerequest_is_retry_not_duplicate():
    bbg, _ = _gateway()

    def leg_a():
        bbg.bdh(tickers=["CLA Comdty"], flds="PX_SETTLE",
                start_date="2026-01-01", end_date="2026-01-31")

    leg_a()
    leg_a()                                    # serial-fallback style re-request
    assert bbg.duplicates() == {}


def test_cross_site_request_is_a_duplicate():
    bbg, _ = _gateway()

    def leg_a():
        bbg.bdh(tickers=["CLA Comdty"], flds="PX_SETTLE",
                start_date="2026-01-01", end_date="2026-01-31")

    def leg_b():
        bbg.bdh(tickers=["CLA Comdty"], flds=["PX_SETTLE"],
                start_date="2026-06-01", end_date="2026-06-30")

    leg_a()
    leg_b()                                    # different window, SAME data family
    dup = bbg.duplicates()
    assert ("CLA Comdty", "PX_SETTLE") in dup
    assert len(dup[("CLA Comdty", "PX_SETTLE")]) == 2
    assert "DUPLICATE" in bbg.report()


def test_accepted_overlap_is_reported_separately():
    from src import bbg
    _gateway()
    pair_sites = {"datafeed._bloomberg_yield", "deepstore._pull_field"}
    bbg._pairs[("USGG10YR Index", "PX_LAST")] = set(pair_sites)
    assert bbg.duplicates() == {}
    assert ("USGG10YR Index", "PX_LAST") in bbg.accepted_overlaps()
    assert "none" in bbg.report()


def test_reset_session_clears_registry():
    bbg, _ = _gateway()
    bbg.bdh(tickers=["CLA Comdty"], flds="PX_SETTLE",
            start_date="2026-01-01", end_date="2026-01-31")
    bbg.reset_session()
    assert bbg._pairs == {}
