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
  * {history, live_quote, deepstore} on the cash indices' PX_LAST — being retired by the
    2026-09-07 live-quote fix (live_quote now reads 'last' from the history frame already in
    hand); the residual {history, deepstore} is whitelisted in bbg.ACCEPTED_OVERLAPS. Kept
    here so this test passes on the pre-fix pull_duplicates.json; safe to drop once a
    post-fix pull has regenerated the file.
  * {stircurve._bdh, stirpaths.refresh_strip_store} on STIR PX_SETTLE — NOT true waste:
    stircurve pulls a multi-day settle HISTORY WINDOW for a different contract set than the
    single-day snapshot stirpaths writes, so the two are not interchangeable (2026-09-07
    audit). Left in place deliberately.
"""
import json
from pathlib import Path

import pytest

DUP_FILE = Path(__file__).resolve().parents[1] / "data" / "pull_duplicates.json"

KNOWN_GROUPS = {
    frozenset({"datafeed._bloomberg_history",
               "datafeed._bloomberg_live_quote",
               "deepstore._pull_field"}),
    frozenset({"stircurve._bdh", "stirpaths.refresh_strip_store"}),
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
