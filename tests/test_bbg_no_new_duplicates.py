"""BBG duplicate-pull BRAKE (2026-09-07).

src/bbg.py already DETECTS when the same security x field is pulled by more than one
module in a daily pull and dumps the offenders to data/pull_duplicates.json. But detection
is advisory only — nothing stops a NEW module from re-pulling data an earlier leg already
fetched, and as the app grows that is exactly how wasted Bloomberg capacity creeps back.

This test turns the detector into a BRAKE: it FAILS the build if the last real pull produced
a duplicate whose owning-module SET is not one of the KNOWN, reviewed overlaps below. To
accept a genuinely new overlap you must add its site-group here on purpose — that deliberate
edit is the review gate the ownership rule was missing.

KNOWN_GROUPS — each a frozenset of the modules that pulled the offending pair:
  * {stircurve._bdh, stirpaths.refresh_strip_store} on STIR PX_SETTLE — NOT true waste:
    stircurve pulls a multi-day settle HISTORY WINDOW per contract stem (bdh), while
    stirpaths takes ONE batched bdp of ~72 specific-strip tickers for that morning's value.
    Different request shapes over a partially overlapping contract set, so neither can serve
    the other without coupling the two modules (2026-09-07 audit, re-checked 2026-10-06:
    15 pairs, all SFR/SFI/ER quarterlies). Left in place deliberately.
  * {owncurve._bdp_many, stircurve._bdh} on bond PX_SETTLE (first seen 2026-09-28) — the
    skew-backfill drip: a product's settle HISTORY for the Dec/Mar/Jun contracts goes
    through stircurve's bdh helper, while the own-curve marks fetch takes the same
    contracts' one-day settle. The detector keys on security x field, not dates, so a
    history backfill always collides with the live snapshot. TRANSIENT per product, but it
    recurs for each one the drip reaches: 2026-09-28 was TYA, 2026-10-06 is FVA (3 pairs,
    FV Z6/H7/M7). Both are now at MAX_SKEW_ATTEMPTS, but USA and WNA still sit on one
    attempt each, so expect it again on the mornings they drip. Drop this entry once the
    bond products are exhausted AND a later pull has regenerated the file without it.

Dropped 2026-10-06: {history, live_quote, deepstore} on the cash indices' PX_LAST. The
2026-09-07 live-quote fix (live_quote reads 'last' from the history frame already in hand)
retired it, and this entry was only held open until a post-fix pull regenerated the file.
Today's pull did: that three-way no longer appears in `duplicates` at all, and the residual
{history, deepstore} two-way sits in `accepted_overlaps` via bbg.ACCEPTED_OVERLAPS, which
this test does not police. Re-adding it would mean the fix regressed.
"""
import json
from pathlib import Path

import pytest

DUP_FILE = Path(__file__).resolve().parents[1] / "data" / "pull_duplicates.json"

KNOWN_GROUPS = {
    frozenset({"stircurve._bdh", "stirpaths.refresh_strip_store"}),
    frozenset({"owncurve._bdp_many", "stircurve._bdh"}),        # transient — see docstring
}


def test_no_new_bbg_duplicate_pulls():
    """Fail if the last pull re-pulled a security x field from a module set we have not
    reviewed. Skips cleanly when no pull has run yet (fresh checkout / CI without data)."""
    if not DUP_FILE.exists():
        pytest.skip("no pull_duplicates.json yet — needs one real Bloomberg pull first")
    try:
        blob = json.loads(DUP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        pytest.skip(f"pull_duplicates.json unreadable: {e}")

    offenders: dict = {}
    for pair, sites in (blob.get("duplicates") or {}).items():
        grp = frozenset(sites)
        if grp not in KNOWN_GROUPS:
            offenders.setdefault(tuple(sorted(grp)), []).append(pair)

    assert not offenders, (
        "NEW Bloomberg duplicate pull(s) detected in data/pull_duplicates.json — a module is "
        "re-pulling data another leg already fetched, wasting the day's Bloomberg allowance. "
        "De-duplicate it (read the shared snapshot the fetch already wrote), or if the overlap "
        "is genuinely justified add its site-group to KNOWN_GROUPS in this test WITH A REASON.\n"
        + "\n".join(f"  {' + '.join(g)}  ::  {len(ps)} pair(s), e.g. {ps[0]}"
                    for g, ps in sorted(offenders.items())))
