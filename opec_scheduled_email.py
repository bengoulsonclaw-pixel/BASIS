"""Scheduled OPEC MOMR emailer — when a new Monthly Oil Market Report is published,
fetch it, build the branded synopsis + chart-deck PDF, and email it to the desk list.

Fully unattended: the fetch drives a persistent real-Chrome profile (the only thing
OPEC's Cloudflare trusts — see src/opec_fetch.py), the figures/prose are parsed
deterministically (src/opec_parse.py), the PDF is built by src/opecreport.py, and the
mail reuses the COT emailer's Gmail credentials + the dashboard's managed recipient list
(report key "opec").

Idempotent + safe on a generous schedule: it reads the latest edition label off the OPEC
site and only sends when that edition is NEWER than the one already emailed (recorded in a
marker file). Run it daily across the release window; it no-ops until the new report appears,
sends once, then no-ops for the rest of the window.

Usage:
  python opec_scheduled_email.py                # scheduled run (send only if a new edition)
  python opec_scheduled_email.py --dry-run      # fetch + build, but do NOT send
  python opec_scheduled_email.py --force-send    # rebuild + send the current latest now
  python opec_scheduled_email.py --seed          # record current latest as already sent (no email)
  python opec_scheduled_email.py --detect-only   # just print the latest edition label
  python opec_scheduled_email.py --from-inbox    # skip fetch; use the newest files already in opec/inbox
  python opec_scheduled_email.py --to a@b.com    # override recipients (e.g. a test to yourself)
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import datetime
from pathlib import Path

# Release-window days of the month (OPEC publishes ~9th–14th; pad to 8–18 for safety).
WINDOW_DAYS = range(8, 19)

ROOT = Path(__file__).resolve().parent
AI_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

INBOX = AI_ROOT / "opec" / "inbox"
OUTDIR = AI_ROOT / "opec" / "out"
MARKER = ROOT / "data" / "signals" / "opec_emailed.txt"


def read_marker() -> str:
    return MARKER.read_text(encoding="utf-8").strip() if MARKER.exists() else ""


def write_marker(label: str) -> None:
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(label, encoding="utf-8")


def _newest_inbox():
    """(pdf, xlsx, month, year) for the most recent MOMR already in opec/inbox."""
    pdfs = sorted(INBOX.glob("MOMR_*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return None
    pdf = pdfs[0]
    import re
    m = re.search(r"MOMR_([A-Za-z]+)(\d{4})\.pdf", pdf.name)
    if not m:
        return None
    month, year = m.group(1), m.group(2)
    xlsx = INBOX / f"MOMR_Appendix_{month}{year}.xlsx"
    return pdf, (xlsx if xlsx.exists() else None), month, year


def build_and_send(pdf_path, xlsx_path, month, year, dry_run, to_override):
    import opec_parse as P
    from opecreport import build_pdf
    from cot_scheduled_email import send_report_email

    # Balance/revision data: prefer the Excel appendix (2-dp), but fall back to the same
    # tables in the PDF when the Excel didn't download or is incomplete — so a flaky/absent
    # appendix can never again strip the report down to a stub.
    ap = P.parse_appendix(Path(xlsx_path)) if xlsx_path else {}
    if not ap.get("demand_by_year"):
        ap = {**P.parse_balance_pdf(Path(pdf_path)), **ap}
    pf = P.parse_pdf(Path(pdf_path))
    d = P.build_synopsis(ap, pf, month, year)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTDIR / f"OPEC_MOMR_Synopsis_{month}{year}.pdf"
    build_pdf(d, out_pdf)
    # save the synopsis JSON alongside (auditable + hand-editable for a re-send)
    (AI_ROOT / "opec" / f"{month}{year}_synopsis.json").write_text(
        json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    subject = f"OPEC Oil Market Report — {month} {year} synopsis"
    intro = (f'<p>Please find our one-page synopsis of OPEC’s Monthly Oil Market Report '
             f'({month} {year}), with the demand/supply balance and a short chart deck.</p>')
    recips = send_report_email(out_pdf, subject, intro,
                               attachment_name=f"OPEC_MOMR_Synopsis_{month}{year}.pdf",
                               report_key="opec", to_override=to_override, dry_run=dry_run)
    return out_pdf, recips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build from the newest inbox PDF but do NOT send")
    ap.add_argument("--force-send", action="store_true", help="rebuild + send the newest inbox edition now, bypassing the 'already sent' gate")
    ap.add_argument("--seed", action="store_true", help="record the newest inbox edition as already sent (no email)")
    ap.add_argument("--from-inbox", action="store_true", help="(default) build from the newest PDF in opec/inbox")
    ap.add_argument("--fetch", action="store_true", help="attempt the OPEC browser download first — interactive/unreliable: OPEC gates it behind a registration form + a Cloudflare bot-check")
    ap.add_argument("--detect-only", action="store_true", help="browser-detect the latest edition label and exit")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients (e.g. a test to yourself)")
    args = ap.parse_args()

    # The scheduled auto-send is the plain no-flag run — honour the on/off toggle.
    scheduled = not (args.force_send or args.dry_run or args.seed or args.detect_only or args.fetch or args.from_inbox)
    if scheduled:
        from src import automation
        if not automation.report_enabled("opec"):
            print("OPEC automatic sending is OFF (Recipients -> Scheduled reports). Nothing to do.")
            return

    marker = read_marker()

    # Optional interactive browser fetch. OPEC now requires filling a registration form AND
    # passes the download through a Cloudflare bot challenge that blocks automation (and
    # bypassing a bot-check is off-limits) — so the reliable path is a manual monthly download
    # into opec/inbox, which the default run below picks up. --fetch is a best-effort attempt.
    if args.detect_only or args.fetch:
        from opec_fetch import fetch
        res = fetch(detect_only=args.detect_only)
        print(json.dumps(res, indent=2))
        if args.detect_only:
            return
        if not res.get("ok"):
            print("Browser fetch failed (OPEC form / Cloudflare) — download the PDF manually into "
                  "opec/inbox and re-run.")
            sys.exit(1)

    # Default path: build + email from the newest MOMR PDF already in opec/inbox.
    got = _newest_inbox()
    if not got:
        print("No MOMR PDF in opec/inbox yet — waiting for this month's manual download. Nothing to do.")
        return
    pdf, xlsx, month, year = got
    label = f"MOMR {month} {year}"

    if args.seed:
        write_marker(label)
        print(f"Seeded marker to '{label}' (no email).")
        return
    if not args.force_send and label == marker:
        print(f"Edition '{label}' already emailed; nothing to do.")
        return

    print(f"Building + {'(dry-run) ' if args.dry_run else ''}sending {label}…")
    out_pdf, recips = build_and_send(pdf, xlsx, month, year, args.dry_run, args.to)
    print(f"{'[dry-run] would send' if args.dry_run else 'Sent'} {out_pdf.name} to: {', '.join(recips)}")
    if not args.dry_run:
        write_marker(label)
        print(f"Marker updated to '{label}'.")


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("OPEC MOMR synopsis", main)
