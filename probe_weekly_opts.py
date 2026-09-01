"""Probe: how do WEEKLY and DAILY option series surface on the Bloomberg API?

    (PowerShell, Terminal open)
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_opts.py --dry-run
      $env:DATAFEED_MODE="bloomberg"; .venv\\Scripts\\python.exe probe_weekly_opts.py

WHY THIS EXISTS
The "🔤 BBG Codes" page (src/bbgcodes.py) covers quarterlies and serial monthlies and says so
on its face. The gap is weeklies/dailies/midcurves — which is where the desk actually needs
it. We know one hard fact about that gap, established 2026-08-31 from data already on disk:

    `bds(<dated future>, "OPT_CHAIN")` DOES NOT RETURN WEEKLY SERIES.

That is evidence, not inference. owncurve._parse_chain applies NO root filter — it keys on
whatever root comes back — and across 219 cached contracts / 290 series stems, captured over
weeks of morning pulls, not one weekly root has ever appeared. So the weeklies are not being
fetched-and-discarded; the call simply doesn't carry them.

This probe answers the only remaining question: WHICH CALL DOES? It tries four routes on
three products and prints exactly what each returns. Nothing is cached, written, or acted on.

COST. Every request goes through src/bbg.py, so pullguard meters it and the run is written to
data/pull_log.csv — if Bloomberg ever asks, our own ledger has the answer. The full run is
**134 security×field hits at most**, and less in practice (stage 4 stops at the first strike
spelling that resolves). For scale, the morning snapshot is 6,000–9,000 EVERY DAY and the
weekly OI capture is 10,000–80,000. This whole probe is about two percent of one morning.

SAFETY, because this account has been under -4002 WORKFLOW_REVIEW_NEEDED twice:
  • --dry-run prints every ticker and field it WOULD request and the exact hit count, and
    issues ZERO requests. Read that first.
  • A raw-blpapi block check runs FIRST (the playbook's rule) — if -4002 is active the probe
    stops before spending anything, and prints the nid for the Help Desk.
  • MAX_HITS is a hard abort, not a warning.
  • It refuses to run outside the weekday morning window without --force. The established
    pattern is one morning pull; an evening or weekend burst is the shape that got us
    flagged, and this is a manual tool that must never look like unattended automation.
  • Stage 4 guesses candidate roots. A wrong guess returns no data rather than raising, so a
    miss costs at most 4 hits and a resolve is a CONFIRMED root. Note the asymmetry: a
    resolve proves the root exists, but "no data" does NOT prove it doesn't — the strike
    spelling could be wrong. The output says so on every miss rather than implying absence.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src import bbg as blp, pullguard

ROOT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
MAX_HITS = 200                 # hard ceiling for the whole run
MORNING = (6, 14)              # ET hours the established daily pull lives in

# Three products, one per asset class we care about, all heavily optioned.
PRODUCTS = [
    ("ESA Index", "ES", "Index"),      # equity index — the deepest weekly/daily board
    ("TYA Comdty", "TY", "Comdty"),    # US 10Y — Wednesday and Friday weeklies
    ("CLA Comdty", "CL", "Comdty"),    # WTI — weeklies plus the standard monthly
]

# ── Stage 3: chain fields other than the one we already use ───────────────────────────
# OPT_CHAIN is what BASIS calls today and it yields only standard series. These are the
# documented alternatives worth testing. An unknown field name costs one hit and returns
# empty; it does not raise.
CHAIN_FIELDS = ["OPT_CHAIN", "CHAIN_TICKERS", "OPT_CHAIN_FULL"]

# Overrides that make CHAIN_TICKERS enumerate a specific expiry rather than the default
# front series — the usual route for equity option chains, untested here on futures.
CHAIN_OVERRIDES = {"CHAIN_PUT_CALL_TYPE_OVRD": "C", "CHAIN_POINTS_OVRD": "20"}

# ── Stage 4: candidate weekly/daily roots ─────────────────────────────────────────────
# GUESSES, to be confirmed or killed. The exchange product codes are public (CME lists ES
# Friday weeklies as EW1-EW4 and end-of-month as EW; Treasury weeklies run as separate
# weekly products), but the BLOOMBERG root for them is not derivable from the futures root
# and is exactly what we're trying to learn. Each entry is a root PREFIX that will be
# combined with the live contract's month+year and a C.
CANDIDATE_ROOTS = {
    "ES": ["EW1", "EW2", "EW3", "EW4", "EW", "1A", "2A", "3A", "4A", "1E", "2E"],
    "TY": ["TYW", "1C", "2C", "3C", "4C", "5C", "TY1", "TY2", "WY1", "VY1"],
    "CL": ["LO1", "LO2", "LO3", "LO4", "CLW", "1CL", "ML1", "WT1"],
}
PROBE_FIELDS = ["NAME", "OPT_EXPIRE_DT"]

# A option ticker without a strike is not a security, so a candidate root can only be
# tested at a real strike — and if that strike is wrong, a REAL root reports "no data" and
# we wrongly kill it. So each candidate is tried at up to this many strike SPELLINGS
# (owncurve's lesson: WTI wants '84.5', corn '460', Henry Hub '.25' with the zero dropped),
# stopping at the first that resolves. Hits therefore vary; the plan prints the maximum.
STRIKE_TRIES = 2


# ══════════════════════════════════════════════════════════════════════════════════════
def _blocked() -> str | None:
    """Raw blpapi refdata request — the ONLY reliable way to tell 'not logged in' from the
    -4002 block, which otherwise looks identical to a pull that returns nothing.
    Returns the nid if blocked, '' if data flows, None if the API isn't answering."""
    try:
        import blpapi
        opts = blpapi.SessionOptions()
        opts.setServerHost("localhost")
        opts.setServerPort(8194)
        s = blpapi.Session(opts)
        if not s.start():
            return None
        try:
            if not s.openService("//blp/refdata"):
                return None
            svc = s.getService("//blp/refdata")
            req = svc.createRequest("ReferenceDataRequest")
            req.getElement("securities").appendValue("CLA Comdty")
            req.getElement("fields").appendValue("PX_LAST")
            s.sendRequest(req)
            while True:
                ev = s.nextEvent(8000)
                if ev.eventType() == blpapi.Event.TIMEOUT:
                    return None
                for msg in ev:
                    txt = str(msg)
                    if "WORKFLOW_REVIEW_NEEDED" in txt:
                        m = re.search(r"nid:(\d+)", txt)
                        return m.group(1) if m else "unknown"
                if ev.eventType() == blpapi.Event.RESPONSE:
                    return ""
        finally:
            s.stop()
    except Exception:
        return None


def _rows(obj):
    """Normalise an xbbg result to pandas via the datafeed adapter.

    xbbg returns narwhals LONG frames now, and hand-rolling this is how the first run of
    this probe broke: scraping `.values.ravel()` hit a float where a ticker was expected
    and stage 0 died before a single route was tested. Always use the house adapters."""
    from src.datafeed import _coerce_pd
    try:
        df = _coerce_pd(obj)
    except Exception:
        df = obj
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _active_contract(generic: str, key: str) -> str | None:
    """The dated contract behind a generic, read from the DEEP STORE, not Bloomberg.

    data/price_store/deep_contract.parquet already holds FUT_CUR_GEN_TICKER for every
    product on every day back to 2016 (it is what the Roll Board runs on), so the active
    contract costs nothing. Saves the request AND removes the field-exception that broke
    the first run."""
    try:
        df = pd.read_parquet(ROOT_DIR / "data" / "price_store" / "deep_contract.parquet")
        col = df[generic].dropna()
        if col.empty:
            return None
        sym = str(col.iloc[-1]).strip()
        return sym if sym.split()[-1] in ("Comdty", "Index") else f"{sym} {key}"
    except Exception:
        return None


def _show(label: str, obj, max_rows: int = 12) -> int:
    df = _rows(obj)
    n = len(df)
    print(f"    {label}: rows={n} cols={list(df.columns)[:6]}")
    if n:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("      " + df.head(max_rows).to_string().replace("\n", "\n      "))
    return n


def _atm_strikes(root: str, level) -> list:
    """At-the-money strike SPELLINGS to test a candidate root at.

    The grid step and the exact string form are learned from this product's own observed
    chain (src/bbgcodes.strike_hint) rather than assumed — Henry Hub writes 0.25 as `.25`
    and corn writes a bare integer, and a wrongly-spelled strike is indistinguishable from
    a root that doesn't exist. Falls back to the middle of the observed ladder when the
    live level is unavailable."""
    from src import bbgcodes
    hint = bbgcodes.strike_hint(root, n=9)
    if level is None:
        return hint["examples"][:STRIKE_TRIES]
    step = hint["step"]
    if not step:
        return hint["examples"][:STRIKE_TRIES]
    k = round(float(level) / step) * step
    # SPELLINGS of the SAME strike, not different strikes: the thing being tested is the
    # root, so moving the strike between tries would confound the two. Decimal places come
    # from what this product's own chain actually uses.
    out = [f"{k:g}"]
    if out[0].startswith("0."):                  # Henry Hub drops the leading zero
        out.append(out[0][1:])
    for dp in sorted({len(s.split(".")[1]) for s in hint["examples"] if "." in s},
                     reverse=True):
        s = f"{k:.{dp}f}"
        if s not in out:
            out.append(s)
    return out[:STRIKE_TRIES]


def _stems(obj) -> set:
    """Distinct option series roots in a chain result — the thing we actually care about."""
    out = set()
    for val in _rows(obj).astype(str).values.ravel():
        body = re.sub(r"\s+", " ", val).strip()
        for key in (" Comdty", " Index", " Curncy"):
            if body.endswith(key):
                body = body[: -len(key)]
        parts = body.split()
        if len(parts) >= 2 and parts[0][-1:] in ("C", "P"):
            out.add(parts[0][:-1])
    return out


# ══════════════════════════════════════════════════════════════════════════════════════
def plan() -> list:
    """Every (label, securities, fields) this probe would request. Drives both the
    dry-run print and the hit count — one source of truth, so the estimate cannot drift
    from what actually gets sent."""
    steps = [("stage 0  level, to set the probe strike (active contract read from the "
               "deep store — no request)", [g for g, _r, _k in PRODUCTS], ["PX_LAST"])]
    for generic, root, key in PRODUCTS:
        steps.append((f"stage 1  control OPT_CHAIN on the dated {root} future",
                      [f"<{root} active contract>"], ["OPT_CHAIN"]))
        steps.append((f"stage 2  OPT_CHAIN on the {root} generic", [generic],
                      ["OPT_CHAIN"]))
        for fld in CHAIN_FIELDS[1:]:
            steps.append((f"stage 3  {fld} on the dated {root} future",
                          [f"<{root} active contract>"], [fld]))
        cands = [f"{p}<mth><yr>C <atm strike> {key}"
                 for p in CANDIDATE_ROOTS.get(root, [])
                 for _ in range(STRIKE_TRIES)]
        steps.append((f"stage 4  {len(CANDIDATE_ROOTS.get(root, []))} candidate weekly "
                      f"roots for {root}, up to {STRIKE_TRIES} strike spellings each (max)",
                      cands, PROBE_FIELDS))
    return steps


def estimate() -> int:
    return sum(len(secs) * len(flds) for _l, secs, flds in plan())


# ══════════════════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact request plan and hit count; issue NOTHING")
    ap.add_argument("--force", action="store_true",
                    help="run outside the weekday morning window (discouraged)")
    ap.add_argument("--max-hits", type=int, default=MAX_HITS)
    args = ap.parse_args()

    est = estimate()
    print("=" * 78)
    print("WEEKLY / DAILY OPTION ROOT PROBE")
    print("=" * 78)
    print(f"Planned spend: {est} security×field hits "
          f"(cap {args.max_hits}; a normal morning is 6,000-9,000).\n")
    for label, secs, flds in plan():
        print(f"  {label}")
        print(f"      fields:     {flds}")
        print(f"      securities: {secs if len(secs) <= 6 else secs[:6] + ['…']}")
        print(f"      hits:       {len(secs) * len(flds)}")
    print()

    if args.dry_run:
        print("DRY RUN — no Bloomberg request was issued.")
        print(f"PROBE_DONE dry_run=1 planned_hits={est}")
        return 0

    if est > args.max_hits:
        print(f"ABORT: plan is {est} hits, above the {args.max_hits} cap.")
        print("PROBE_DONE aborted=cap")
        return 2

    now = datetime.now(ET)
    if not args.force and (now.weekday() >= 5 or not MORNING[0] <= now.hour < MORNING[1]):
        print(f"ABORT: it is {now:%a %H:%M} ET. The established pattern is one WEEKDAY "
              "MORNING pull, and an off-cycle burst is the shape that has twice put this "
              "account under review. Re-run in the morning window, or pass --force if you "
              "have a reason.")
        print("PROBE_DONE aborted=window")
        return 2

    for w in pullguard.assess([g for g, _r, _k in PRODUCTS]):
        print(f"  pullguard: {w}")

    nid = _blocked()
    if nid:
        print(f"ABORT: Bloomberg is refusing data — -4002 WORKFLOW_REVIEW_NEEDED nid:{nid}. "
              "Quote that to the Help Desk (F1 F1) with the timestamp. Nothing was spent.")
        print(f"PROBE_DONE aborted=blocked nid={nid}")
        return 2
    if nid is None:
        print("ABORT: the Bloomberg API isn't answering — Terminal closed, logged out, or "
              "the login is live on another PC. Nothing was spent.")
        print("PROBE_DONE aborted=unreachable")
        return 2
    print("Block check clear — data is flowing.\n")

    pullguard.reset_hits()
    findings: dict = {}
    tested = 0          # routes that actually completed — a probe that never ran must
                        # NEVER report 'nothing found' (the first run did exactly that)

    for generic, root, key in PRODUCTS:
        print("=" * 78)
        print(f"{generic}  (root {root!r})")

        active = _active_contract(generic, key)      # from disk — no Bloomberg request
        level = None
        try:
            from src.datafeed import _bdp_rows
            got = _bdp_rows(_rows(blp.bdp(active or generic, "PX_LAST")))
            for _sec, flds in got.items():
                for fld, val in flds.items():
                    if fld == "PX_LAST":
                        try:
                            level = float(val)
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:
            print(f"  level lookup failed: {exc!r}")
        print(f"  stage 0  active contract: {active!r} (from deep store)   "
              f"level: {level!r}")
        if not active:
            print("      no active contract on disk — skipped")
            continue

        base: set = set()
        if True:
            print("  stage 1  CONTROL — OPT_CHAIN on the dated future "
                  "(expected: standard series only)")
            try:
                res = blp.bds(active, "OPT_CHAIN")
                _show("OPT_CHAIN", res, max_rows=6)
                base = _stems(res)
                tested += 1
                print(f"      distinct series roots: {sorted(base)}")
            except Exception as exc:
                print(f"      raised: {exc!r}")

        print("  stage 2  OPT_CHAIN on the GENERIC")
        try:
            res = blp.bds(generic, "OPT_CHAIN")
            got = _stems(res)
            tested += 1
            _show("OPT_CHAIN(generic)", res, max_rows=6)
            print(f"      distinct series roots: {sorted(got)}")
            if got - base:
                findings.setdefault(root, set()).update(got - base)
                print(f"      *** NEW ROOTS vs control: {sorted(got - base)}")
        except Exception as exc:
            print(f"      raised: {exc!r}")

        for fld in CHAIN_FIELDS[1:]:
            print(f"  stage 3  {fld} on the dated future")
            try:
                res = blp.bds(active, fld, **CHAIN_OVERRIDES)
                got = _stems(res)
                tested += 1
                _show(fld, res, max_rows=6)
                if got - base:
                    findings.setdefault(root, set()).update(got - base)
                    print(f"      *** NEW ROOTS vs control: {sorted(got - base)}")
            except Exception as exc:
                print(f"      raised: {exc!r}")

        print("  stage 4  candidate weekly roots (a miss returns nothing, not an error)")
        m = re.match(r"^([A-Z0-9 ]+?)([FGHJKMNQUVXZ])(\d{1,2})$", active.rsplit(" ", 1)[0])
        if not m:
            print(f"      couldn't parse a month/year off {active!r} — skipped")
            continue
        mth, yr = m.group(2), m.group(3)
        strikes = _atm_strikes(root, level)
        if not strikes:
            print(f"      no level or strike grid for {root} — cannot build a testable "
                  "option ticker; skipped rather than guessing")
            continue
        print(f"      probing at strike(s) {strikes}")
        for prefix in CANDIDATE_ROOTS.get(root, []):
            hit = False
            for ks in strikes:
                tk = f"{prefix}{mth}{yr}C {ks} {key}"
                try:
                    from src.datafeed import _bdp_rows
                    got = _bdp_rows(_rows(blp.bdp(tk, PROBE_FIELDS)))
                    vals = [f"{f}={v}" for flds in got.values() for f, v in flds.items()
                            if v is not None and str(v).lower() not in ("nan", "none", "")]
                except Exception as exc:
                    print(f"          {tk}  raised: {exc!r}")
                    continue
                tested += 1
                if vals:
                    findings.setdefault(root, set()).add(f"{prefix}{mth}{yr}")
                    print(f"      *** {tk}  RESOLVES -> {vals[:3]}")
                    hit = True
                    break
            if not hit:
                print(f"          {prefix}{mth}{yr}C  no data at any tried strike "
                      "(root may still exist at another strike — not proof of absence)")

    spent = pullguard.get_hits()
    print("\n" + "=" * 78)
    print("FINDINGS")
    if findings:
        for root, roots in sorted(findings.items()):
            print(f"  {root}: candidate/extra series roots that resolved -> "
                  f"{sorted(roots)}")
        print("\n  Next: confirm these on the Terminal UI, then decide whether the roots "
              "are PATTERNED (week number increments -> generate arithmetically, one-off "
              "harvest) or ARBITRARY (a per-series pull, and a recurring one).")
    elif tested == 0:
        print("  INCONCLUSIVE — no route actually completed, so this run proves NOTHING "
              "about weeklies. Fix the errors above and re-run; do not read this as an "
              "answer.")
    else:
        print(f"  Nothing beyond the standard series resolved, across {tested} completed "
              "route(s).")
        print("  Take: weeklies are not reachable from the futures' chain, and the root "
              "must come from elsewhere — the Terminal UI (CTM / OMON), or the exchange's "
              "own product codes mapped by hand once.")
    print(f"\nSpent: {spent} security×field hits (planned {est}).")
    try:
        pullguard.record("weekly-option root probe",
                         [g for g, _r, _k in PRODUCTS], est_hits=spent)
        print("Logged to data/pull_log.csv.")
    except Exception as exc:
        print(f"ledger write failed (harmless): {exc!r}")
    print(f"PROBE_DONE ok=1 hits={spent} routes_tested={tested} "
          f"findings={sum(len(v) for v in findings.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
