"""THE Bloomberg gateway — every xbbg request in BASIS goes through this module.

WHY THIS EXISTS. The 2026-08-18 pull audit (docs/bloomberg_pull_budget.md, prompted
by Bloomberg's own DAPI best-practice note) found the same security×field being
pulled up to THREE times every morning by different modules — the 30d ATM vol field
by the implied-vol, skew and term legs; volume and yields by datafeed and the deep
store; 47 price histories by both cotdata and the deep store. Nobody could see it,
because each module talked to xbbg directly and no chokepoint existed. This module
is that chokepoint, and it makes the error class self-announcing:

  1. Every request is COUNTED into pullguard's runtime hit tally (hits =
     security × field, Bloomberg's capacity unit) — the usage ledger no longer
     depends on scattered per-module instrumentation.
  2. Every (security, field) pair is REGISTERED with its calling site. A pair
     requested by TWO DIFFERENT sites within one pull session is a DESIGN
     DUPLICATE: the morning snapshot prints them loudly and ledgers the count.
     Re-requests from the SAME site (serial fallbacks after a failed batch, a
     forced chain-cache refresh) are retries, not duplicates, and are ignored.
  3. tests/test_bbg_gateway.py BANS direct xbbg imports outside this file (the
     pre-push hook runs the suite), so no future code can quietly bypass the gate.
     Ad-hoc diagnostics (diag_*/probe_*/pnl_*) are exempt — manual tools, never
     scheduled. The two raw-blpapi uses (app._blp_block_probe, surface_topup's
     reachability check) are deliberate low-level diagnostics, also exempt.

OWNERSHIP RULE (the design-time half of the guard): every security×field family
has exactly ONE owning module — prices/vol surface/put-call = datafeed, deep
history = deepstore, option chains = owncurve, STIR ladders = stircurve, strips =
stirpaths. Anything else that needs the data reads the STORE, never Bloomberg.
Deliberate exceptions live in ACCEPTED_OVERLAPS below, each with its reason.

Import as `from . import bbg as blp` so call sites keep xbbg's call shapes.
Off-Terminal boxes never import xbbg: the import is lazy, inside the calls.
"""
from __future__ import annotations

import inspect
from collections import defaultdict

_blp_mod = None                # test hook — set to a stub to run without xbbg


def _blp():
    global _blp_mod
    if _blp_mod is None:
        from xbbg import blp as _x
        _blp_mod = _x
    return _blp_mod


# (ticker, FIELD) -> {call sites that requested it} — session-scoped, cleared by
# reset_session() at the start of each snapshot fetch phase.
_pairs: dict = defaultdict(set)

# Cross-site overlaps ACCEPTED by design: (fields, the exact site pair, reason).
# A flagged pair matching an entry is reported as "accepted", not as a duplicate.
ACCEPTED_OVERLAPS = [
    ({"PX_LAST"}, {"datafeed._bloomberg_yield", "deepstore._pull_field"},
     "11 bond yield sources — datafeed serves the 400d ffilled TA frame, the deep "
     "store keeps raw prints (holiday NaNs are load-bearing for Curve/RV): same "
     "request, deliberately different post-processing (2026-08-18 review)"),
    ({"PX_LAST", "PX_SETTLE"}, {"datafeed._bloomberg_history", "deepstore._pull_field"},
     "the 8 cash indices pass through unchained into the deep tail (~8 hits) — "
     "candidate to mirror from prices.parquet later"),
]


def _call_site() -> str:
    """'module.function' of the first frame outside this file — the requesting leg."""
    try:
        for fr in inspect.stack()[1:9]:
            mod = fr.frame.f_globals.get("__name__", "")
            if not mod.endswith("bbg"):
                return f"{mod.rsplit('.', 1)[-1]}.{fr.function}"
    except Exception:
        pass
    return "unknown"


def _register(tickers, flds) -> None:
    """Count the request into the usage ledger + the duplicate registry.
    Never raises — accounting must not break a pull."""
    try:
        from . import pullguard
        tk = [tickers] if isinstance(tickers, str) else list(tickers)
        fl = [flds] if isinstance(flds, str) else list(flds)
        pullguard.add_hits(len(tk) * len(fl))
        site = _call_site()
        for t in tk:
            for f in fl:
                _pairs[(str(t), str(f).upper())].add(site)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The xbbg surface (signature-compatible)
# ---------------------------------------------------------------------------
def bdh(tickers=None, flds=None, start_date=None, end_date=None, **kwargs):
    _register(tickers, flds)
    return _blp().bdh(tickers=tickers, flds=flds, start_date=start_date,
                      end_date=end_date, **kwargs)


def bdp(tickers=None, flds=None, **kwargs):
    _register(tickers, flds)
    return _blp().bdp(tickers, flds, **kwargs)


def bds(tickers=None, flds=None, **kwargs):
    _register(tickers, flds)                 # a bulk request: 1 security × 1 field
    return _blp().bds(tickers, flds, **kwargs)


# ---------------------------------------------------------------------------
# Session accounting
# ---------------------------------------------------------------------------
def reset_session() -> None:
    """Clear the duplicate registry (the fetch phase calls this alongside
    pullguard.reset_hits())."""
    _pairs.clear()


def _accepted_reason(field: str, sites: set):
    for fields, pair_sites, reason in ACCEPTED_OVERLAPS:
        if field in fields and sites <= pair_sites:
            return reason
    return None


def duplicates() -> dict:
    """{(ticker, field): sorted sites} for every pair requested by 2+ DIFFERENT
    sites this session, excluding the documented ACCEPTED_OVERLAPS."""
    return {p: sorted(s) for p, s in _pairs.items()
            if len(s) >= 2 and _accepted_reason(p[1], s) is None}


def accepted_overlaps() -> dict:
    """The pairs that matched an ACCEPTED_OVERLAPS entry this session."""
    return {p: sorted(s) for p, s in _pairs.items()
            if len(s) >= 2 and _accepted_reason(p[1], s) is not None}


def report() -> str:
    """One line for the morning log: clean, or naming the duplicated pulls."""
    dup, acc = duplicates(), accepted_overlaps()
    if not dup:
        base = "duplicate Bloomberg requests: none — every security×field pulled once"
        return base + (f" ({len(acc)} documented accepted overlaps)" if acc else "")
    by_sites: dict = defaultdict(list)
    for (t, f), sites in dup.items():
        by_sites[" + ".join(sites)].append(f"{t}/{f}")
    parts = [f"{len(v)} pair(s) by [{k}] e.g. {v[0]}" for k, v in by_sites.items()]
    return (f"DUPLICATE Bloomberg requests: {len(dup)} security×field pair(s) pulled "
            f"by more than one leg — {'; '.join(parts)}. Same data, twice the hits: "
            "give the pair ONE owning module (see src/bbg.py ownership rule).")
