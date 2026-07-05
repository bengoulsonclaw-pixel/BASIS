"""Unattended fetch of the latest OPEC MOMR PDF + Excel appendix.

OPEC's download host (momr.opec.org) sits behind Cloudflare, which hard-blocks
bundled-Chromium/headless automation — but it lets a *real Chrome* with a
persistent profile through, exactly as a normal user's browser is trusted. The
first visit shows a one-time "Accept OPEC Data Policy" cookie banner; once
accepted, the consent cookie is stored in the persistent profile and every later
month's download fires automatically.

So this module drives **channel="chrome"** with a persistent profile kept in
opec/.chrome_profile. It:
  1. opens the MOMR landing page (opec.org — not gated),
  2. reads the latest edition label (e.g. "MOMR June 2026"),
  3. clicks "Download latest MOMR" → accepts the policy if shown → saves the PDF,
  4. clicks "MOMR Appendix <month>" → saves the Excel.

Headless is set False because Cloudflare trusts headed real-Chrome far more; a
scheduled run briefly opens a window, which is fine for an unattended job.

CLI:
  python src/opec_fetch.py                 # fetch latest into opec/inbox, print JSON result
  python src/opec_fetch.py --detect-only   # just report the latest edition label (no download)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT.parent
INBOX = AI_ROOT / "opec" / "inbox"
PROFILE = AI_ROOT / "opec" / ".chrome_profile"
LANDING = "https://www.opec.org/monthly-oil-market-report.html"

_MONTHS = ("January February March April May June July August September October "
           "November December").split()


def _edition_from_text(txt: str):
    """Return (label, MonthName, Year) for the newest 'MOMR <Month> <Year>' on the page."""
    m = re.search(r"MOMR\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})", txt)
    if not m:
        return None
    return f"MOMR {m.group(1)} {m.group(2)}", m.group(1), m.group(2)


def _accept_and_download(page, ctx, link_text, dest: Path, timeout=45000):
    """Click a download link that opens a momr.opec.org popup, accept the data-policy
    cookie banner if present, then capture the delivered file."""
    with ctx.expect_page(timeout=25000) as pi:
        page.get_by_text(link_text, exact=False).first.click()
    pop = pi.value
    time.sleep(4)
    try:                                   # one-time consent banner (cookie persists after)
        acc = pop.locator("a.cc-dismiss, a.cc-btn").first
        if acc.is_visible(timeout=4000):
            acc.click()
            time.sleep(2)
    except Exception:
        pass
    try:
        with pop.expect_download(timeout=timeout) as di:
            pop.get_by_text("click here", exact=False).first.click()
        di.value.save_as(str(dest))
    except Exception:                      # some links auto-download on (re)load
        with pop.expect_download(timeout=timeout) as di:
            pop.reload()
        di.value.save_as(str(dest))
    try:
        pop.close()
    except Exception:
        pass
    return dest


def fetch(detect_only: bool = False) -> dict:
    from playwright.sync_api import sync_playwright
    INBOX.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    out = {"ok": False}
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE), channel="chrome", headless=False, accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled"])
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(LANDING, wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)
            ed = _edition_from_text(page.inner_text("body"))
            if not ed:
                raise RuntimeError("could not read the latest MOMR edition label from the page")
            label, month, year = ed
            out.update(edition=label, month=month, year=year)
            if detect_only:
                out["ok"] = True
                return out
            tag = f"{month}{year}"
            pdf = _accept_and_download(page, ctx, "Download latest MOMR", INBOX / f"MOMR_{tag}.pdf")
            out["pdf"] = str(pdf)
            try:
                xlsx = _accept_and_download(page, ctx, "MOMR Appendix", INBOX / f"MOMR_Appendix_{tag}.xlsx")
                out["xlsx"] = str(xlsx)
            except Exception as e:           # appendix is optional — PDF carries the same tables
                out["xlsx_error"] = str(e)[:160]
            ok_pdf = Path(pdf).exists() and Path(pdf).read_bytes()[:4] == b"%PDF"
            out["ok"] = ok_pdf
            return out
        finally:
            ctx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()
    res = fetch(detect_only=args.detect_only)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
