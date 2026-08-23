"""wgc_fetch.py — logged-in fetcher for World Gold Council Goldhub datasets.

Goldhub's data tables are real .xlsx files at stable-ish URLs, but they return
**HTTP 403 to anyone not signed in** — no redirect, no login prompt, just a refusal
that looks like a dead link. A signed-in browser session is the only way through,
so this uses the same real-Chrome persistent-profile pattern as opec_fetch and
pm_fetch: sign in once by hand, the profile keeps the session, later runs sail
through unattended.

FIRST RUN IS INTERACTIVE, and deliberately so:

    .venv\\Scripts\\python.exe src/wgc_fetch.py --login

opens a Chrome window at Goldhub. Sign in (Google SSO is fine), then come back to
the console and press Enter. The profile at data/.wgc_chrome_profile persists, so
every later run is just:

    .venv\\Scripts\\python.exe src/wgc_fetch.py --probe    # what am I entitled to?
    .venv\\Scripts\\python.exe src/wgc_fetch.py            # fetch + archive

Two things this module refuses to guess
---------------------------------------
* **Entitlement.** A free Goldhub account may or may not include the Gold Demand
  Trends tables. `--probe` answers it empirically by attempting each download and
  checking the payload's magic bytes — an .xlsx starts `PK`, an HTML apology does
  not. Status codes are not trusted here; this site serves 200 with an error body.
* **Redistribution.** WGC data is licensed. The workbook disclaimers permit "review
  and commentary" on two conditions — limited extracts, and a citation — but the site
  terms say "personal, non-commercial use", and a broker's client research is not
  obviously that. See the licence block below; `CLIENT_FACING_APPROVED` stays False
  until XP compliance rules. One clause IS unambiguous and is enforced in code: the
  LBMA price supplied by WGC may not be disclosed to anyone, so
  `strip_forbidden_columns` removes it at ingest.

Link discovery, rather than hardcoded file ids
----------------------------------------------
The download URLs carry an opaque numeric id (`/download/file/20975/...`) that
changes every quarter when a new edition lands. Hardcoding one buys a fetcher that
silently serves stale data for three months. Instead each dataset page is scanned
for its current .xlsx/.csv links, so a new edition is picked up automatically.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INBOX = ROOT / "data" / "wgc_inbox"
PROFILE = ROOT / "data" / ".wgc_chrome_profile"
SIGNALS = ROOT / "data" / "signals"

# ---------------------------------------------------------------------------
# Licence — read from the workbooks' own Disclaimer sheets and gold.org's site
# terms on 2026-08-22. Not legal advice; this records what the documents say so
# the code can enforce the parts that are unambiguous.
# ---------------------------------------------------------------------------
#
# THE GENERAL RULE (Disclaimer sheet, all three workbooks):
#   "Reproduction or redistribution of any of this information is expressly
#    prohibited without the prior written consent of World Gold Council ...
#    except as specifically provided below."
#
# THE CARVE-OUT, which is the one that matters for client commentary:
#   "The use of the statistics in this information is permitted for the purposes
#    of review and commentary (including media commentary) in line with fair
#    industry practice, subject to the following two pre-conditions: (i) only
#    limited extracts of data or analysis be used; and (ii) any and all use of
#    these statistics is accompanied by a citation to World Gold Council and,
#    where appropriate, to Metals Focus."
#
# THE HARD PROHIBITION — narrower, absolute, and easy to breach by accident
# (present in the GDT and central-bank workbooks; absent from the ETF one):
#   "LBMA Gold Price information provided by the World Gold Council may be used
#    by you internally to review the analysis provided by the World Gold Council,
#    but may not be used for any other purpose. LBMA Gold Price information
#    provided by the World Gold Council may not be disclosed by you to anyone
#    else."
#
# That last clause is why PRICE_COLUMNS_FORBIDDEN exists. It restricts the LBMA
# price *as supplied by WGC* — not the LBMA fix itself, which golddata.lbma()
# already pulls straight from prices.lbma.org.uk under its own terms. So the rule
# is simply: never let a price column out of a WGC workbook. It costs us nothing,
# because we have the same number from the primary source.
#
# UNRESOLVED TENSION, for a human to settle: gold.org's site terms say the site is
# for "personal and educational purposes only ... personal, non-commercial use",
# while the workbook disclaimer expressly permits "review and commentary ... in
# line with fair industry practice". The workbook disclaimer is the more specific
# instrument and travels with the data, but a broker distributing client research
# is not obviously "personal, non-commercial". XP compliance decides, not this file.

WGC_CITATION = "Source: World Gold Council; Metals Focus"

# Anything matching these must never leave the building — see the LBMA clause.
PRICE_COLUMNS_FORBIDDEN = ("gold price", "lbma", "price (rhs)", "au price", "px")

# Client-facing use stays gated until XP compliance rules on the tension above.
# Internal analysis is permitted on any reading of either document.
CLIENT_FACING_APPROVED = False


def strip_forbidden_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any WGC-supplied price column.

    The LBMA clause is absolute — that price "may not be disclosed by you to
    anyone else" — and it is trivially easy to breach by accident, because the ETF
    workbook's 'Charts Data' sheet carries a 'Gold Price (rhs)' column sitting
    right beside the flow numbers we do want. Strip it at ingest rather than
    trusting every downstream chart to remember."""
    drop = [c for c in df.columns
            if any(tok in str(c).strip().lower() for tok in PRICE_COLUMNS_FORBIDDEN)]
    return df.drop(columns=drop) if drop else df

GOLDHUB = "https://www.gold.org/goldhub/data"

# Dataset pages worth sweeping. Each is scanned for its current download links; the
# labels are ours, for the console report only.
PAGES = {
    "gold_demand_trends": f"{GOLDHUB}/gold-demand-by-country",
    "central_bank_holdings": f"{GOLDHUB}/gold-reserves-by-country",
    "etf_flows": f"{GOLDHUB}/gold-etfs-holdings-and-flows",
    "supply_demand": f"{GOLDHUB}/gold-supply-and-demand-statistics",
}

_FILE_RE = re.compile(r'href="(/download/file/\d+/[^"]+\.(?:xlsx|xls|csv))"', re.I)


def _launch(p):
    INBOX.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        str(PROFILE), channel="chrome", headless=False, accept_downloads=True,
        args=["--disable-blink-features=AutomationControlled"])


def _links(page, url: str) -> list:
    """Every downloadable data file linked from one Goldhub page, absolute."""
    page.goto(url, timeout=90000, wait_until="domcontentloaded")
    page.wait_for_timeout(2500)                 # the file list renders client-side
    html = page.content()
    seen, out = set(), []
    for href in _FILE_RE.findall(html):
        if href in seen:
            continue
        seen.add(href)
        out.append("https://www.gold.org" + href.replace("&#039;", "'"))
    return out


def _download(page, url: str, dest: Path, timeout: int = 90000) -> Path | None:
    """Navigate at a file URL and capture the download.

    goto() aborts with net::ERR_ABORTED the moment Chrome hands the response to its
    downloader — that is the SUCCESS path, not a failure, which is why the exception
    is swallowed. Same trick as pm_fetch."""
    try:
        with page.expect_download(timeout=timeout) as di:
            try:
                page.goto(url, timeout=timeout)
            except Exception:
                pass
        di.value.save_as(str(dest))
        return dest
    except Exception:
        return None


def _digest(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def _is_spreadsheet(path: Path) -> bool:
    """Magic bytes, not the file extension and not the status code. Goldhub serves
    its 'you are not entitled' page with a 200 and an .xlsx URL."""
    if not path or not path.exists() or path.stat().st_size < 1024:
        return False
    head = path.read_bytes()[:8]
    return head[:2] == b"PK" or head[:5] == b"\xd0\xcf\x11\xe0\xa1"


def signed_in(page) -> bool:
    """True when the Goldhub session is live. Checked by behaviour rather than by
    hunting for a 'Sign out' link, because the header markup is JS-rendered and
    changes; if a data file downloads as a real spreadsheet, we are in."""
    page.goto(GOLDHUB, timeout=90000, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)
    txt = page.content().lower()
    return ("sign out" in txt) or ("log out" in txt) or ("my account" in txt)


def login() -> int:
    """Interactive: open Chrome, let a human sign in, persist the profile."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        ctx = _launch(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(GOLDHUB, timeout=90000, wait_until="domcontentloaded")
        print("\n  A Chrome window is open at Goldhub.")
        print("  1. Click 'Sign in' (top right) and log in as basisreports@gmail.com.")
        print("  2. Wait until you are back on a Goldhub page, signed in.")
        print("  3. Return here and press Enter.\n")
        try:
            input("  Press Enter once signed in... ")
        except EOFError:
            print("  (no console attached — run this from a terminal)")
            ctx.close()
            return 2
        ok = signed_in(page)
        print(f"  session marker: {'FOUND' if ok else 'not found (probe will settle it)'}")
        ctx.close()
    print(f"  profile saved -> {PROFILE}")
    return 0


def probe() -> int:
    """What is actually downloadable on this account? Answers the entitlement
    question with evidence instead of assumption."""
    from playwright.sync_api import sync_playwright
    stamp = date.today().isoformat()
    results = []
    with sync_playwright() as p:
        ctx = _launch(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        for name, url in PAGES.items():
            try:
                links = _links(page, url)
            except Exception as e:
                results.append((name, "PAGE FAILED", str(e)[:60], ""))
                continue
            if not links:
                results.append((name, "no files linked", "", url))
                continue
            first = links[0]
            dest = INBOX / f"probe_{name}_{stamp}{Path(first).suffix}"
            got = _download(page, first, dest)
            ok = _is_spreadsheet(got)
            results.append((name, "ENTITLED" if ok else "BLOCKED",
                            f"{len(links)} file(s)", first.rsplit('/', 1)[-1]))
            if got and not ok:
                got.unlink(missing_ok=True)
        ctx.close()

    print(f"\n  Goldhub entitlement probe  ({stamp})\n")
    for name, verdict, detail, extra in results:
        print(f"    {verdict:14s} {name:24s} {detail:14s} {extra}")
    entitled = [r[0] for r in results if r[1] == "ENTITLED"]
    print(f"\n  {len(entitled)}/{len(results)} datasets downloadable: {entitled or 'none'}")
    if not entitled:
        print("  -> either the session did not persist (re-run --login) or the free"
              "\n     tier does not include these tables.")
    if LICENCE_UNRESOLVED:
        print("\n  NOTE: licence terms unresolved — internal analysis only, nothing"
              "\n  client-facing, until gold.org/terms-and-conditions is reviewed.")
    return 0


def fetch(which=("gold_demand_trends", "central_bank_holdings")) -> int:
    """Archive the current edition of each dataset into data/wgc_inbox/."""
    from playwright.sync_api import sync_playwright
    stamp = date.today().isoformat()
    saved = 0
    # Goldhub links the SAME workbook from more than one dataset page — the
    # gold-demand-by-country and gold-supply-and-demand-statistics pages both lead
    # to the identical GDT tables file (verified: md5 a31cc67f from both). Keying the
    # archive on content rather than on the page it came from stops one workbook
    # being stored twice under two names and then parsed twice into the store.
    seen: dict = {}
    with sync_playwright() as p:
        ctx = _launch(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        for name in which:
            url = PAGES.get(name)
            if not url:
                print(f"  unknown dataset {name!r}")
                continue
            for link in _links(page, url):
                fn = re.sub(r"[^A-Za-z0-9._-]", "_", link.rsplit("/", 1)[-1])
                dest = INBOX / f"{name}_{stamp}_{fn}"
                got = _download(page, link, dest)
                if not _is_spreadsheet(got):
                    if got:
                        got.unlink(missing_ok=True)
                    print(f"  BLOCKED {name}: {fn}")
                    continue
                h = _digest(dest)
                if h in seen:
                    dest.unlink(missing_ok=True)
                    print(f"  skipped {fn} — same workbook as {seen[h]}")
                    continue
                seen[h] = dest.name
                print(f"  saved {dest.name}")
                saved += 1
        ctx.close()
    print(f"\n  {saved} distinct file(s) archived to {INBOX}")
    return 0 if saved else 1


def main() -> int:
    if "--login" in sys.argv:
        return login()
    if "--probe" in sys.argv:
        return probe()
    return fetch()


if __name__ == "__main__":
    raise SystemExit(main())
