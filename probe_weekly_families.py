"""Probe 3: confirm the weekly roots Ben's BBG_Code_List spreadsheet gives us.

    (PowerShell, Terminal open)
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_families.py --dry-run
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_families.py

WHERE THESE CANDIDATES COME FROM
Ben supplied a CME "BBG Code List" workbook whose 'ALL Exch' sheet maps every CME product
name to its Bloomberg ticker. Filtering it to rows named "... Week N" whose Bloomberg root
STARTS WITH the week number yields a clean family per product — the same <week><letter>
shape probe 2 confirmed live for the S&P (1E..4E, Nth Friday, 10/10).

That is a document, not the Terminal, so it is a CANDIDATE LIST — good enough to stop
guessing, not good enough to ship. Two things it cannot settle:

  1. THE TRAILING 'A'. Some roots are written '1XA', others '1M'. 'A' is Bloomberg's
     front-generic marker (CLA, ESA), so '1XA' may well mean the root is '1X' and the
     sheet quoted its generic. Both forms are tried, and whichever resolves wins.
  2. WHETHER WEEK N REALLY IS THE Nth FRIDAY for these products. The S&P follows that
     rule; nothing says corn must. So week 2 is checked against the 2nd Friday, and a
     family whose dates disagree is reported rather than trusted.

A wrong candidate returns no data instead of raising, so a miss costs a hit or two and a
resolve is proof. As always: a miss is NOT proof of absence — the strike may not be listed.

Note what the sheet already fixed for free: probe 1 guessed '1C' for the 10-year and it
failed. '1C' is the BOND; the 10-year is '1M'. Guessing was never going to get there.

Same rails as probes 1 and 2 — dry-run spends nothing, raw-blpapi -4002 check first, hard
MAX_HITS abort, weekday-morning window unless --force, everything through src/bbg.py so
pullguard meters it and the run lands in data/pull_log.csv.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from probe_weekly_opts import _active_contract, _atm_strikes, _blocked, _rows, ET, MORNING
from src import bbg as blp, bbgcodes, pullguard

MAX_HITS = 250
STRIKE_TRIES = 2

# universe ticker -> root forms to try for week N, from the spreadsheet. Where the sheet
# wrote a trailing 'A' both forms are listed, longest-shot last.
FAMILIES = [
    ("TYA Comdty", ["{n}M"]),                    # 10-Year T-Note
    ("FVA Comdty", ["{n}I"]),                    # 5-Year T-Note
    ("TUA Comdty", ["{n}W"]),                    # 2-Year T-Note
    ("USA Comdty", ["{n}C", "{n}CA"]),           # U.S. Treasury Bond
    ("WNA Comdty", ["{n}J"]),                    # Ultra Bond
    ("NQA Index",  ["{n}O"]),                    # E-mini Nasdaq-100
    ("C A Comdty", ["{n}X", "{n}XA"]),           # Corn
    ("W A Comdty", ["{n}Z", "{n}ZA"]),           # Chicago SRW Wheat
    ("S A Comdty", ["{n}S", "{n}SA"]),           # Soybean
    ("SMA Comdty", ["{n}D", "{n}DA"]),           # Soybean Meal
    ("BOA Comdty", ["{n}A", "{n}AA"]),           # Soybean Oil
]
CHECK_WEEKS = (1, 2)          # week 1 finds the root; week 2 proves the number increments


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
    ap.add_argument("--only", default="",
                    help="comma-separated universe tickers to re-check, e.g. 'C A Comdty'")
    ap.add_argument("--wide", type=int, default=0,
                    help="extra strikes each side of the money for the week-2 check — a "
                         "weekly board is thin, and a single strike produces FALSE "
                         "negatives that look like a broken rule")
    args = ap.parse_args()
    families = FAMILIES
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        families = [f for f in FAMILIES if f[0] in want]
        if not families:
            print(f"no families match {sorted(want)}")
            return 2

    n_variants = sum(len(v) for _t, v in families)
    worst = (len(families)                                    # PX_LAST per product
             + n_variants * STRIKE_TRIES                      # week-1 discovery
             + len(families)                                  # NAME confirm
             + len(families) * (STRIKE_TRIES + 2 * args.wide))  # week-2 check
    print("=" * 78)
    print("WEEKLY FAMILY CONFIRMATION  (candidates from Ben's BBG_Code_List workbook)")
    print("=" * 78)
    for tk, variants in families:
        print(f"  {tk:12} week-1 candidates: "
              + ", ".join(v.format(n=1) for v in variants))
    print(f"\n  WORST CASE: {worst} security×field hits (cap {args.max_hits}); "
          "a normal morning is 6,000-9,000.")
    print("  Typical is far less — each family stops at its first resolving form.\n")

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
        print(f"ABORT: it is {now:%a %H:%M} ET, outside the weekday morning window.")
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
    confirmed: dict = {}

    for tk, variants in families:
        key = tk.rsplit(" ", 1)[1]
        root = bbgcodes._root_from_generic(tk) or ""
        active = _active_contract(tk, key)
        if not active:
            print(f"{tk:12} no active contract on disk — skipped")
            continue
        stem = active.rsplit(" ", 1)[0][len(root):]
        level = None
        got = _lookup(active, "PX_LAST")
        try:
            level = float(got.get("PX_LAST"))
        except (TypeError, ValueError):
            pass
        strikes = _atm_strikes(root, level)[:STRIKE_TRIES]
        print(f"── {tk:12} front {active:14} level {level!r:>10} "
              f"stem {stem!r} strikes {strikes}")
        if not strikes:
            print("     no strike grid on disk for this product — skipped")
            continue

        hit = None
        for v in variants:
            wr = v.format(n=1)
            for ks in strikes:
                probe = f"{wr}{stem}C {ks} {key}"
                info = _lookup(probe, "OPT_EXPIRE_DT")
                if info.get("OPT_EXPIRE_DT"):
                    hit = (wr, v, ks, info["OPT_EXPIRE_DT"], probe)
                    break
            if hit:
                break
        if not hit:
            print(f"     -- none of {[v.format(n=1) for v in variants]} resolved at "
                  f"{strikes} (not proof of absence — the board may be thin)")
            continue

        wr, form, ks, exp, probe = hit
        nm = _lookup(probe, "NAME").get("NAME", "")
        exp_d = pd.to_datetime(exp, errors="coerce")
        want = bbgcodes.nth_weekday_of(
            exp_d.year, exp_d.month, 1, bbgcodes.FRI) if exp_d is not None else None
        ok1 = want is not None and exp_d.date() == want
        print(f"     ** {probe}")
        print(f"        NAME {nm}")
        print(f"        EXPIRY {exp}   week1 = 1st Friday {want}  "
              f"{'MATCHES' if ok1 else 'DIFFERS — read before trusting'}")

        # week 2 must move to the following Friday
        wr2 = form.format(n=2)
        ok2 = False
        wide = list(strikes)
        if args.wide and level is not None:
            hint = bbgcodes.strike_hint(root)
            step = hint.get("step") or 1
            for i in range(1, args.wide + 1):
                for sgn in (1, -1):
                    cand = f"{round((float(level) + sgn * i * step) / step) * step:g}"
                    if cand not in wide:
                        wide.append(cand)
        for ks2 in wide:
            info2 = _lookup(f"{wr2}{stem}C {ks2} {key}", "OPT_EXPIRE_DT")
            if info2.get("OPT_EXPIRE_DT"):
                e2 = pd.to_datetime(info2["OPT_EXPIRE_DT"], errors="coerce")
                w2 = bbgcodes.nth_weekday_of(e2.year, e2.month, 2, bbgcodes.FRI)
                ok2 = w2 is not None and e2.date() == w2
                print(f"        week2 {wr2}{stem} -> {info2['OPT_EXPIRE_DT']}  "
                      f"(2nd Friday {w2}) {'MATCHES' if ok2 else 'DIFFERS'}")
                break
        else:
            print(f"        week2 {wr2}{stem} did not resolve at {wide} — a thin "
                  "board, not necessarily a different rule; re-run with --wide")
        confirmed[tk] = {"form": form, "week1": wr, "name": nm, "expiry": str(exp),
                         "nth_friday_ok": bool(ok1 and ok2)}

    spent = pullguard.get_hits()
    print("\n" + "=" * 78)
    print("CONFIRMED FAMILIES")
    good = {t: c for t, c in confirmed.items() if c["nth_friday_ok"]}
    for tk, c in confirmed.items():
        mark = "OK  " if c["nth_friday_ok"] else "??  "
        print(f"  {mark}{tk:12} root form {c['form']:6} e.g. {c['week1']:4} "
              f"| {c['name'][:44]}")
    print(f"\n  {len(good)}/{len(FAMILIES)} families confirmed on the Nth-Friday rule "
          "and ready to add to bbgcodes.WEEKLY_SPECS.")
    if len(confirmed) > len(good):
        print("  Families marked ?? resolved but their dates did not match the Nth-Friday "
              "rule — do NOT add those until the actual rule is understood.")
    print(f"\nSpent: {spent} security×field hits (worst case {worst}).")
    try:
        pullguard.record("weekly family confirmation probe",
                         [t for t, _v in families], est_hits=spent)
        print("Logged to data/pull_log.csv.")
    except Exception as exc:
        print(f"ledger write failed (harmless): {exc!r}")
    print(f"PROBE_DONE ok=1 hits={spent} confirmed={len(good)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
