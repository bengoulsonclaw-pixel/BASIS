"""Unattended fetch of the latest OPEC MOMR PDF.

OPEC's download host sits behind Cloudflare, which hard-blocks bundled-Chromium/
headless automation — but it lets a *real Chrome* with a persistent profile
through, exactly as a normal user's browser is trusted. So this module drives
**channel="chrome"** with a persistent profile kept in opec/.chrome_profile.

Since ~Aug 2026 OPEC gates the download behind a **registration form** (title,
first/last name, email, company, position, sector + a consent checkbox → "Download
PDF"): the old auto-download page is gone. This module fills that form with
FORM_DETAILS below and submits it to capture the delivered PDF. Chrome is also
told to download PDFs rather than render them inline (else the delivery navigates
into the built-in viewer and no download event fires).

The report builds fully from the PDF alone (opec_parse.parse_balance_pdf), so the
separate Excel appendix is no longer fetched — one less thing to break.

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

# Details submitted to OPEC's download form each month. OPEC uses the email for
# occasional MOMR surveys — a personal Gmail keeps that off the work address.
# Edit here to change what's submitted.
FORM_DETAILS = {
    "title": "Mr.", "firstname": "Ben", "lastname": "Goulson",
    "email": "bengoulson@gmail.com", "company": "XP Inc.",
    "position": "Futures Broker", "sector": "energy",   # energy|media|governmental|education|other
}

_MONTHS = ("January February March April May June July August September October "
           "November December").split()


def _edition_from_text(txt: str):
    """Return (label, MonthName, Year) for the newest 'MOMR <Month> <Year>' on the page."""
    m = re.search(r"MOMR\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})", txt)
    if not m:
        return None
    return f"MOMR {m.group(1)} {m.group(2)}", m.group(1), m.group(2)


def _set_pdf_prefs(profile: Path) -> None:
    """Make Chrome DOWNLOAD PDFs instead of opening them inline, so the form's
    'Download PDF' fires a real download event we can capture."""
    try:
        prefs = profile / "Default" / "Preferences"
        data = json.loads(prefs.read_text(encoding="utf-8")) if prefs.exists() else {}
        data.setdefault("plugins", {})["always_open_pdf_externally"] = True
        data.setdefault("download", {})["prompt_for_download"] = False
        prefs.parent.mkdir(parents=True, exist_ok=True)
        prefs.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _dismiss_cookie(pg) -> None:
    try:
        pg.evaluate("document.querySelectorAll('.cc-dismiss,.cc-btn').forEach(b=>b.click())")
    except Exception:
        pass


def _download_via_form(page, ctx, link_text, dest: Path, timeout=60000) -> Path:
    """Click the download link, fill OPEC's registration form on the popup, submit,
    and save the delivered PDF."""
    with ctx.expect_page(timeout=25000) as pi:
        page.get_by_text(link_text, exact=False).first.click()
    pop = pi.value
    pop.wait_for_load_state("domcontentloaded")
    time.sleep(2)
    _dismiss_cookie(pop)
    d = FORM_DETAILS
    pop.select_option("select[name=title]", d["title"])
    pop.fill("input[name=firstname]", d["firstname"])
    pop.fill("input[name=lastname]", d["lastname"])
    pop.fill("input[name=email]", d["email"])
    pop.fill("input[name=company]", d["company"])
    pop.fill("input[name=position]", d["position"])
    pop.select_option("select[name=sector]", d["sector"])
    pop.check("#checkBoxConsent", force=True)
    got = {}
    pop.on("download", lambda dl: got.setdefault("dl", dl))
    pop.click("button[type=submit]")
    deadline = time.time() + timeout / 1000
    while "dl" not in got and time.time() < deadline:
        time.sleep(0.5)
    if "dl" not in got:
        raise RuntimeError("form submitted but no PDF download was delivered")
    got["dl"].save_as(str(dest))
    try:
        pop.close()
    except Exception:
        pass
    return dest


def fetch(detect_only: bool = False) -> dict:
    from playwright.sync_api import sync_playwright
    INBOX.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    _set_pdf_prefs(PROFILE)
    out = {"ok": False}
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE), channel="chrome", headless=False, accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled"])
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(LANDING, wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)
            _dismiss_cookie(page)
            ed = _edition_from_text(page.inner_text("body"))
            if not ed:
                raise RuntimeError("could not read the latest MOMR edition label from the page")
            label, month, year = ed
            out.update(edition=label, month=month, year=year)
            if detect_only:
                out["ok"] = True
                return out
            tag = f"{month}{year}"
            pdf = _download_via_form(page, ctx, "Download latest MOMR", INBOX / f"MOMR_{tag}.pdf")
            out["pdf"] = str(pdf)
            out["ok"] = Path(pdf).exists() and Path(pdf).read_bytes()[:4] == b"%PDF"
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
