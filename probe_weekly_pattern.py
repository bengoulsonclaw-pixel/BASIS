"""Probe 2: is the S&P weekly option root PATTERNED by week number?

    (PowerShell, Terminal open)
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_pattern.py --dry-run
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_pattern.py

WHAT WE ALREADY KNOW (probe_weekly_opts.py, 2026-09-01, 109 hits)
`1EU6C 7655 Index` resolves, and Bloomberg names it "S&P Emini 1st Wee Sep26C 7655" with
OPT_EXPIRE_DT 2026-09-04 — the first Friday of September. So weekly options are NOT reachable
through any option chain (OPT_CHAIN on the future returns only standard series; CHAIN_TICKERS
and OPT_CHAIN_FULL return nothing at all) but ARE reachable if you know the root, and the ES
week-1 root is `1E`.

THE QUESTION THIS ANSWERS
If `1E` is week 1, are `2E`-`5E` weeks 2-5? If yes the roots are a PATTERN, and every S&P
weekly — past, present and future — can be generated arithmetically with no further Bloomberg
request, ever. If no, each series has to be discovered individually and the whole feature
becomes a recurring pull rather than a one-off. That is the difference between a cheap feature
and an expensive one, and it costs a few dozen requests to settle.

METHOD. For each candidate week root, build an at-the-money call on the front contract and ask
for its NAME and expiry. A resolve is proof; a failure is NOT proof of absence (the strike may
simply not be listed on that series), so several strikes are tried before giving up and the
output says so. Roots that resolve get a second, cheap pass for OPT_UNDL_TICKER — which future
the weekly actually settles against — and a check that the same root works on the NEXT contract
month, since a root that only works for September would not be a pattern at all.

VERIFY THE EXPIRIES, don't just collect them. A real week-N option must expire on the Nth
Friday of its month. The script checks that itself and flags any that doesn't, because a root
that resolves but lands on the wrong date means the week numbering is not what we assumed.

Same rails as probe 1: --dry-run spends nothing, raw-blpapi block check first, hard MAX_HITS
abort, weekday-morning window unless --force, every request through src/bbg.py so pullguard
meters it and the run lands in data/pull_log.csv.
"""
from __future__ import annotations

import argparse
import calendar
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from probe_weekly_opts import _active_contract, _atm_strikes, _blocked, _rows, ET, MORNING
from src import bbg as blp, pullguard

MAX_HITS = 150

GENERIC, ROOT, KEY = "ESA Index", "ES", "Index"
WEEK_ROOTS = ["1E", "2E", "3E", "4E", "5E"]
FIELDS = ["NAME", "OPT_EXPIRE_DT"]
UNDL_FIELD = "OPT_UNDL_TICKER"
STRIKE_TRIES = 3                # weeklies list far fewer strikes than the monthly board
MONTH_CODE = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
              7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_MONTH = {v: k for k, v in MONTH_CODE.items()}


def _nth_friday(y: int, m: int, n: int):
    """The n-th Friday of a month — what a genuine 'week n' option must expire on."""
    first = date(y, m, 1)
    d = first + timedelta(days=(calendar.FRIDAY - first.weekday()) % 7 + 7 * (n - 1))
    return d if d.month == m else None


def _strikes(level, n: int) -> list:
    """At-the-money spellings, widened outward — a weekly board is thin, so the exact ATM
    strike may not be listed even though the series exists."""
    out = list(_atm_strikes(ROOT, level))
    if level is not None:
        for off in (5, -5, 10, -10, 25, -25):     # ES lists on a 5-point grid
            s = f"{round((float(level) + off) / 5) * 5:g}"
            if s not in out:
                out.append(s)
    return out[:n]


def _lookup(ticker: str, fields) -> dict:
    from src.datafeed import _bdp_rows
    try:
        got = _bdp_rows(_rows(blp.bdp(ticker, fields)))
    except Exception:
        return {}
    for _sec, flds in got.items():
        clean = {f: v for f, v in flds.items()
                 if v is not None and str(v).lower() not in ("nan", "none", "")}
        if clean:
            return clean
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-hits", type=int, default=MAX_HITS)
    ap.add_argument("--stem", default=None,
                    help="override the contract stem to test, e.g. Z6 — stage B showed the "
                         "week-3/4 underlyings rolling to Z6, so the month a weekly is "
                         "coded under may follow the QUARTERLY cycle, not the calendar month")
    args = ap.parse_args()

    worst = (len(WEEK_ROOTS) * STRIKE_TRIES * len(FIELDS)     # stage A: week roots
             + len(WEEK_ROOTS)                                # stage B: underlying
             + len(WEEK_ROOTS) * STRIKE_TRIES * len(FIELDS))  # stage C: next month
    print("=" * 78)
    print("S&P WEEKLY ROOT PATTERN PROBE")
    print("=" * 78)
    print(f"  roots tested : {WEEK_ROOTS}")
    print(f"  fields       : {FIELDS} (+ {UNDL_FIELD} on roots that resolve)")
    print(f"  strikes      : up to {STRIKE_TRIES} per root, stopping at the first resolve")
    print(f"  WORST CASE   : {worst} security×field hits (cap {args.max_hits}).")
    print("  Typical is far less — every root that resolves on its first strike saves "
          f"{(STRIKE_TRIES - 1) * len(FIELDS)} hits.\n")

    if args.dry_run:
        print("DRY RUN — no Bloomberg request was issued.")
        print(f"PROBE_DONE dry_run=1 worst_case_hits={worst}")
        return 0
    if worst > args.max_hits:
        print(f"ABORT: worst case {worst} exceeds cap {args.max_hits}.")
        print("PROBE_DONE aborted=cap")
        return 2

    now = datetime.now(ET)
    if not args.force and (now.weekday() >= 5 or not MORNING[0] <= now.hour < MORNING[1]):
        print(f"ABORT: it is {now:%a %H:%M} ET, outside the weekday morning window. "
              "Re-run in the morning, or pass --force.")
        print("PROBE_DONE aborted=window")
        return 2

    nid = _blocked()
    if nid:
        print(f"ABORT: -4002 WORKFLOW_REVIEW_NEEDED nid:{nid}. Nothing spent.")
        print(f"PROBE_DONE aborted=blocked nid={nid}")
        return 2
    if nid is None:
        print("ABORT: Bloomberg API not answering. Nothing spent.")
        print("PROBE_DONE aborted=unreachable")
        return 2
    print("Block check clear — data is flowing.\n")

    pullguard.reset_hits()
    active = _active_contract(GENERIC, KEY)
    level = None
    got = _lookup(active or GENERIC, "PX_LAST")
    if got.get("PX_LAST") is not None:
        try:
            level = float(got["PX_LAST"])
        except (TypeError, ValueError):
            pass
    stem = args.stem or (active or "").rsplit(" ", 1)[0][len(ROOT):]   # e.g. 'U6'
    print(f"front contract {active!r}   level {level!r}   month/year stem {stem!r}\n")
    if not stem:
        print("PROBE_DONE aborted=no_contract")
        return 2

    mcode, yr = stem[0], stem[1:]
    year = 2020 + int(yr) if len(yr) == 1 else 2000 + int(yr)
    year = year if year >= now.year - 1 else year + 10
    month = CODE_MONTH.get(mcode)
    tries = _strikes(level, STRIKE_TRIES)
    print(f"probing strikes {tries}\n")

    resolved: dict = {}
    print("── stage A: does each week root resolve on the front contract? " + "─" * 14)
    for i, wr in enumerate(WEEK_ROOTS, start=1):
        for ks in tries:
            tk = f"{wr}{stem}C {ks} {KEY}"
            info = _lookup(tk, FIELDS)
            if info:
                exp = pd.to_datetime(info.get("OPT_EXPIRE_DT"), errors="coerce")
                want = _nth_friday(year, month, i) if month else None
                ok = (exp is not None and want is not None
                      and exp.date() == want)
                mark = "OK " if ok else "?? "
                print(f"  {mark}{tk}")
                print(f"       NAME   {info.get('NAME')}")
                print(f"       EXPIRY {info.get('OPT_EXPIRE_DT')}"
                      + (f"   (week {i} = {want:%a %d %b} {'✓ matches' if ok else '✗ MISMATCH'})"
                         if want else ""))
                resolved[wr] = {"ticker": tk, "strike": ks, "week": i,
                                "expiry": info.get("OPT_EXPIRE_DT"),
                                "name": info.get("NAME"), "week_ok": ok}
                break
        else:
            print(f"  -- {wr}{stem}C  no data at strikes {tries} "
                  "(not proof the root is absent — the board may be thin here)")

    print("\n── stage B: which future does each weekly settle against? " + "─" * 18)
    for wr, rec in resolved.items():
        info = _lookup(rec["ticker"], UNDL_FIELD)
        rec["undl"] = info.get(UNDL_FIELD)
        print(f"  {wr}{stem} -> {rec['undl']!r}")

    print("\n── stage C: does the same root work on the NEXT contract month? " + "─" * 12)
    nxt = None
    if month:
        ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
        nxt = f"{MONTH_CODE[nm]}{ny % 10}"
        for wr in list(resolved):
            hit = False
            for ks in tries:
                tk = f"{wr}{nxt}C {ks} {KEY}"
                info = _lookup(tk, FIELDS)
                if info:
                    exp = pd.to_datetime(info.get("OPT_EXPIRE_DT"), errors="coerce")
                    want = _nth_friday(ny, nm, resolved[wr]["week"])
                    ok = exp is not None and want is not None and exp.date() == want
                    print(f"  {'OK ' if ok else '?? '}{tk}  expiry {info.get('OPT_EXPIRE_DT')}"
                          + (f"  (week {resolved[wr]['week']} = {want:%a %d %b} "
                             f"{'✓' if ok else '✗ MISMATCH'})" if want else ""))
                    resolved[wr]["next_month_ok"] = ok
                    hit = True
                    break
            if not hit:
                print(f"  -- {wr}{nxt}C  no data at strikes {tries}")
                resolved[wr]["next_month_ok"] = False

    spent = pullguard.get_hits()
    print("\n" + "=" * 78)
    print("VERDICT")
    n_ok = sum(1 for r in resolved.values() if r.get("week_ok"))
    n_next = sum(1 for r in resolved.values() if r.get("next_month_ok"))
    if len(resolved) >= 4 and n_ok == len(resolved) and n_next >= len(resolved) - 1:
        print(f"  PATTERNED. {len(resolved)}/{len(WEEK_ROOTS)} week roots resolved, every "
              "one expiring on the correct n-th Friday, and the same roots work on the "
              "next contract month.")
        print("  => S&P weeklies can be GENERATED arithmetically: <week><E><month><year>.")
        print("     One-off knowledge, no recurring Bloomberg pull. Build it into "
              "src/bbgcodes.py and the page covers weeklies for free from here on.")
    elif resolved:
        print(f"  PARTIAL. {len(resolved)}/{len(WEEK_ROOTS)} roots resolved; {n_ok} landed "
              f"on the expected n-th Friday; {n_next} carried to the next month.")
        print("  Read the rows above before generalising — a root that resolves on the "
              "wrong date means the week numbering is not what we assumed.")
    else:
        print("  INCONCLUSIVE — nothing resolved. Do NOT read this as 'weeklies don't "
              "exist': 1EU6C resolved in probe 1, so the roots are real and the strikes "
              "tried here were simply not listed. Widen the strikes and re-run.")
    print(f"\nSpent: {spent} security×field hits (worst case {worst}).")
    try:
        pullguard.record("S&P weekly root pattern probe", [GENERIC], est_hits=spent)
        print("Logged to data/pull_log.csv.")
    except Exception as exc:
        print(f"ledger write failed (harmless): {exc!r}")
    print(f"PROBE_DONE ok=1 hits={spent} resolved={len(resolved)} week_ok={n_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
