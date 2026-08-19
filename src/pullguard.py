"""Pull guard — pre-flight review-risk assessment for Bloomberg pulls.

Bloomberg has put this account under -4002 WORKFLOW_REVIEW_NEEDED twice (Jul 2026 after
the ~70k-hit Russell fundamentals bursts; 8-9 Aug 2026 the weekend after a Friday-night
one-off backfill of ~176 never-before-pulled '1'/'2' generics). Both times the trigger
was the SHAPE of usage, not its volume: a sudden batch of new unique securities from an
automated-looking process, at an odd hour. This module encodes those lessons as a
pre-flight check so a risky pull WARNS before it spends — and keeps our own usage
ledger so any future review conversation starts from our records, not their inference.

What it flags (each an observed or strongly suspected review trigger):
  • NEW SECURITIES — tickers this account has never pulled through the guarded paths
    (seeded from the current established universe + its derived sources). A handful is
    a caution; a bulk batch is the highest-risk act we know of. Introduce new surfaces
    a few securities per day (the own-skew drip pattern), inside the morning window.
  • ODD HOURS / WEEKENDS — the established pattern is one morning pull; an evening or
    weekend burst reads as a new unattended system even when it's Ben at the keyboard.
  • SAME-DAY RE-PULLS — re-spends the day's allowance on near-identical data and makes
    the cadence look erratic.

The NEW-SECURITY assessment deliberately excludes option-chain contracts (own-curve
marks, the weekly OI job) — individual contracts churn by design, that pattern is
long-established, and neither review followed it. The guard watches the futures/
index/cash universe. Their REQUEST VOLUME does count, though: since 2026-08-18 the
runtime hit tally (add_hits / get_hits below) is fed by every variable-size pull
site, so the ledger's est_hits matches what Bloomberg actually saw — the old
len(tickers)*20 estimate under-counted the morning by ~4-5x.

data/pull_known.json  — every ticker ever pulled through guarded paths (auto-seeded).
data/pull_log.csv     — one line per fetch: when, label, tickers, est hits, new count.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import universe

DATA = Path(__file__).resolve().parents[1] / "data"
KNOWN_FILE = DATA / "pull_known.json"
LOG_FILE = DATA / "pull_log.csv"

ET = ZoneInfo("America/New_York")
MORNING_WINDOW = (6, 14)      # ET hours the established daily pull lives in
BULK_NEW = 15                 # new securities at/above this = the highest-risk flag


def _expected_universe() -> set:
    """Every security the ESTABLISHED daily pattern touches: the app universe, the
    deep store's '1'/'2' generic sources, and the bond benchmark-yield tickers."""
    out = set(universe.INSTRUMENTS)
    try:
        from . import deepstore
        for t in universe.INSTRUMENTS:
            out.add(deepstore._src(t, "1"))
            out.add(deepstore._src(t, "2"))
    except Exception:
        pass
    out.update(v for v in universe.BOND_YIELD_SOURCE.values() if v)
    return out


def known() -> set:
    """Tickers considered established — the seed universe plus everything recorded."""
    out = _expected_universe()
    try:
        out.update(json.loads(KNOWN_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return out


def assess(tickers=None, same_day: bool = False, now=None) -> list:
    """Review-risk warnings for a prospective pull of `tickers` (None = the standard
    snapshot universe). Returns [] when the pull looks like the established pattern —
    each entry is a display-ready markdown string, worst first."""
    now = now or datetime.now(ET)
    tickers = list(tickers if tickers is not None else universe.INSTRUMENTS)
    warns = []
    new = sorted(set(tickers) - known())
    if new:
        ex = ", ".join(f"`{t}`" for t in new[:4]) + (" …" if len(new) > 4 else "")
        if len(new) >= BULK_NEW:
            warns.append(f"🚨 **{len(new)} securities this account has never pulled** ({ex}). "
                         "A bulk batch of new names is the strongest known review trigger "
                         "(it's what preceded the Aug-9 block). Introduce new surfaces a few "
                         "per day inside the morning pull instead — or accept the risk "
                         "knowingly.")
        else:
            warns.append(f"⚠️ {len(new)} never-before-pulled securit"
                         f"{'y' if len(new) == 1 else 'ies'} ({ex}) — small additions are "
                         "usually fine, but they accumulate toward monthly unique-security "
                         "quotas.")
    if now.weekday() >= 5:
        warns.append("⚠️ **Weekend pull** — markets are closed, so this buys near-identical "
                     "data while looking like unattended automation (the Aug-9 review "
                     "surfaced on a Sunday-morning pull). Monday's normal pull gets the "
                     "same settles.")
    elif not (MORNING_WINDOW[0] <= now.hour < MORNING_WINDOW[1]):
        warns.append(f"⚠️ **Off-hours pull** ({now:%H:%M} ET) — the account's established "
                     "pattern is one morning pull; evening bursts read as a new automated "
                     "workload (the Friday-night backfill that preceded the Aug-9 review "
                     "ran at 20:40).")
    if same_day:
        warns.append("⚡ **Snapshot already pulled today** — a re-pull re-spends thousands of "
                     "hits on near-identical data and makes the cadence look erratic. Worth "
                     "it only if the first pull was bad or markets have moved a lot.")
    return warns


# ---------------------------------------------------------------------------
# Runtime hit tally — a per-pull security×field counter. The fetch phase resets
# it; modules whose request sizes are only known at runtime (option chains,
# strike ladders, deep-store backfill groups) call add_hits() at each request
# site. In-process and advisory: it can never raise, and a missed count just
# means the ledger under-reports slightly instead of by multiples.
# ---------------------------------------------------------------------------
_HITS = 0


def reset_hits() -> None:
    global _HITS
    _HITS = 0


def add_hits(n) -> None:
    global _HITS
    try:
        _HITS += max(int(n), 0)
    except Exception:
        pass


def get_hits() -> int:
    return _HITS


def record(label: str, tickers, est_hits: int | None = None) -> None:
    """Log a completed pull (our own usage ledger) and absorb its tickers into the
    known set. Never raises — a logging failure must not break a pull."""
    try:
        tickers = list(tickers)
        prev = known()
        n_new = len(set(tickers) - prev)
        DATA.mkdir(parents=True, exist_ok=True)
        try:
            stored = set(json.loads(KNOWN_FILE.read_text(encoding="utf-8")))
        except Exception:
            stored = set()
        stored.update(tickers)
        KNOWN_FILE.write_text(json.dumps(sorted(stored), indent=0), encoding="utf-8")
        line = (f"{datetime.now(ET):%Y-%m-%d %H:%M:%S%z},{label},{len(tickers)},"
                f"{'' if est_hits is None else int(est_hits)},{n_new}\n")
        if not LOG_FILE.exists():
            LOG_FILE.write_text("when_et,label,n_tickers,est_hits,n_new_securities\n",
                                encoding="utf-8")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
