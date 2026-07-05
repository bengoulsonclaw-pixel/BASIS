"""Scheduled COT emailer — after the weekly CFTC Commitments of Traders release, build
the email-size COT report and send it to the desk (Ben + Said).

Self-checking and idempotent, so it is safe to run on a generous schedule: it refreshes
from the free CFTC API, reads the latest report's as-of date, and only sends when that
date is NEWER than the last one already emailed (recorded in a marker file). Before that
release it simply finds last week's data and does nothing; after sending once it no-ops
for the rest of the window. This is what makes "immediately after release" robust to
timezone drift and to holiday weeks (when the release shifts from Friday to Monday).

Email credentials + the desk recipient list are read from the Morning Coffee project at
runtime (single source of truth — no duplicated password here).

Usage:
  python cot_scheduled_email.py                 # normal scheduled run (send only if new)
  python cot_scheduled_email.py --dry-run       # do everything EXCEPT actually send
  python cot_scheduled_email.py --force-send     # rebuild + send for the current latest now
  python cot_scheduled_email.py --seed          # record the current latest as 'already sent' (no email)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import smtplib
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Price overlay uses the cached morning snapshot if the Terminal isn't open at run time
# (the COT positioning data itself is always live from the CFTC API regardless of mode).
os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import pandas as pd            # noqa: E402
from src import cotdata        # noqa: E402

MARKER = ROOT / "data" / "signals" / "cot_emailed.txt"            # last as-of date emailed (release dedup)
COTREPORT_CLI = ROOT / "src" / "cotreport.py"
NY_TZ = ZoneInfo("America/New_York")
OUT_PDF = ROOT / "data" / "COT_Positioning_Report_email.pdf"
MC_MAIN = Path(os.getenv("BASIS_MC_DIR", r"C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements")) / "main.py"
MAX_STALE_DAYS = 10            # don't email a report older than this (guards against stale cache)


def load_email_cfg():
    """Read (_EMAIL_FROM, _EMAIL_APP_PW, _EMAIL_TO) from the Morning Coffee project
    without importing it (it pulls heavy deps the dashboard venv lacks)."""
    tree = ast.parse(MC_MAIN.read_text(encoding="utf-8", errors="ignore"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("_EMAIL_FROM", "_EMAIL_APP_PW", "_EMAIL_TO"):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out["_EMAIL_FROM"], out["_EMAIL_APP_PW"], out["_EMAIL_TO"]


def _managed_recipients(report, fallback):
    """Recipients set on the dashboard's Recipients page (data/email_recipients.json),
    if any; otherwise the given fallback (the Morning Coffee _EMAIL_TO)."""
    try:
        p = ROOT / "data" / "email_recipients.json"
        if p.exists():
            lst = [e for e in (json.loads(p.read_text(encoding="utf-8")).get(report) or [])
                   if e and str(e).strip()]
            if lst:
                return lst
    except Exception:
        pass
    return fallback


def latest_release_date():
    """Cheap one-row API call for the most recent report date (so no-op runs skip the
    full 48-market refresh)."""
    params = {"$select": "report_date_as_yyyy_mm_dd", "$order": "report_date_as_yyyy_mm_dd DESC",
              "$limit": "1", "cftc_contract_market_code": "067651"}   # WTI
    url = cotdata.BASE + cotdata.DATASETS["disagg"] + ".json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "cot-scheduler"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.load(r)
    return pd.to_datetime(rows[0]["report_date_as_yyyy_mm_dd"]).date() if rows else None


def read_marker() -> str:
    return MARKER.read_text().strip() if MARKER.exists() else ""


def _build_email_image():
    """Render the continuous email-overview image via cotreport (one full-height sidebar,
    content flowing with no page gaps). Returns PNG bytes, or None on failure."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cot_overview.png"
            r = subprocess.run(
                [sys.executable, str(COTREPORT_CLI), str(cotdata.COT_DETAIL_FILE),
                 str(Path(td) / "_unused.pdf"), "--asof", "", "--threshold", "80",
                 "--email-image", str(out)],
                capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                return out.read_bytes()
            print(f"  (email overview image failed: {(r.stderr or r.stdout)[-300:]})")
    except Exception as e:
        print(f"  (email overview image skipped: {e})")
    return None


def send_email(pdf_path: Path, asof, dry_run: bool = False, to_override=None, subject_note=""):
    sender, app_pw, recipients = load_email_cfg()
    recipients = to_override or _managed_recipients("cot", recipients)
    asof_s = pd.Timestamp(asof).strftime("%d %b %Y")
    subject = f"COT Positioning Report{subject_note} — data as of {asof_s}"
    mb = pdf_path.stat().st_size / 1e6
    overview = _build_email_image()        # continuous one-sidebar overview (None on failure)

    # Outlook's compose/forward engine (Word) IGNORES max-width CSS and renders images at their
    # natural pixel size — that's why the text ballooned the instant you hit Forward. Explicit
    # width/height HTML *attributes* ARE honoured by Outlook (and stop its high-DPI upscaling), so
    # they pin the display size whether the mail is read, replied, or forwarded.
    if overview:
        import struct
        w_px, h_px = struct.unpack(">II", overview[16:24])     # PNG IHDR carries width/height
        disp_w, disp_h = max(1, w_px // 2), max(1, h_px // 2)  # rendered at device_scale 2 → show at logical px size
        body_img = (f'<img src="cid:cotoverview" width="{disp_w}" height="{disp_h}" '
                    f'style="display:block;width:{disp_w}px;max-width:100%;height:auto;border:0;'
                    f'margin:0 0 12px;">')
    else:
        body_img = ''
    intro = (
        f'<p>Please find the weekly CFTC Commitments of Traders positioning report, data as of '
        f'Tuesday {asof_s}.</p>')
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#222;font-size:14px;">'
        f'{intro}{body_img}'
        '<p>Individual product details can be found within the attached PDF.</p>'
        '<p style="color:#888;font-size:12px;"><b>THIS IS A SALES COMMENTARY AND SHOULD BE READ AS SUCH</b>'
        '</p></div>')
    text = (f"The weekly CFTC Commitments of Traders positioning report, data as of Tuesday {asof_s}. "
            "Individual product details can be found within the attached PDF.\n\n"
            "THIS IS A SALES COMMENTARY AND SHOULD BE READ AS SUCH.")

    msg = MIMEMultipart("mixed")
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    related = MIMEMultipart("related")
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain"))
    alt.attach(MIMEText(html, "html"))
    related.attach(alt)
    if overview:
        img = MIMEImage(overview, _subtype="png")
        img.add_header("Content-ID", "<cotoverview>")
        img.add_header("Content-Disposition", "inline", filename="cot_overview.png")
        related.attach(img)
    msg.attach(related)

    with open(pdf_path, "rb") as f:
        ap = MIMEApplication(f.read(), _subtype="pdf")
    ap.add_header("Content-Disposition", "attachment", filename="COT_Positioning_Report.pdf")
    msg.attach(ap)

    note = "overview image" if overview else "no inline image"
    if dry_run:
        print(f"[dry-run] would send '{subject}' to {recipients} — {note} + a {mb:.1f} MB attachment")
        return
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, app_pw)
        s.sendmail(sender, recipients, msg.as_string())
    print(f"Emailed '{subject}' ({note} + {mb:.1f} MB attachment) to {', '.join(recipients)}")


def send_report_email(pdf_path, subject, intro_html, attachment_name,
                      report_key="", to_override=None, dry_run=False):
    """Generic desk emailer for the visual client reports (Volatility / Skew / Term / …):
    a short branded body with the PDF attached. Reuses this module's Gmail credentials and
    recipient list (data/email_recipients.json keyed by report_key, else the Morning Coffee
    desk list). Returns the recipient list it sent to."""
    sender, app_pw, recipients = load_email_cfg()
    recipients = to_override or _managed_recipients(report_key, recipients)
    pdf_path = Path(pdf_path)
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#222;font-size:14px;">'
        f'{intro_html}'
        '<p>Full detail is in the attached PDF.</p>'
        '<p style="color:#888;font-size:12px;">'
        '<b>THIS IS A SALES COMMENTARY AND SHOULD BE READ AS SUCH</b></p></div>')
    text = ("Full detail is in the attached PDF.\n\n"
            "THIS IS A SALES COMMENTARY AND SHOULD BE READ AS SUCH.")
    msg = MIMEMultipart("mixed")
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain"))
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)
    with open(pdf_path, "rb") as f:
        ap = MIMEApplication(f.read(), _subtype="pdf")
    ap.add_header("Content-Disposition", "attachment", filename=attachment_name)
    msg.attach(ap)
    if dry_run:
        print(f"[dry-run] would send '{subject}' to {recipients}")
        return recipients
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, app_pw)
        s.sendmail(sender, recipients, msg.as_string())
    return recipients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="do everything except actually send")
    ap.add_argument("--force-send", action="store_true", help="rebuild + send the current latest now, bypassing all gates")
    ap.add_argument("--seed", action="store_true", help="record current latest as already sent (no email)")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients (e.g. a test to Ben only)")
    args = ap.parse_args()

    if not (args.force_send or args.dry_run or args.seed):   # scheduled auto-send path
        from src import automation
        if not automation.report_enabled("cot"):
            print("COT automatic sending is OFF (turn it on: dashboard -> Recipients -> Scheduled reports). Skipping.")
            return

    marker = read_marker()
    ny = datetime.now(NY_TZ)

    # Release window gate (DST-aware): the CFTC posts the COT at ~15:30 New York. Begin
    # polling at 15:20 NY so we never act before the release; the 15-minute trigger then
    # catches the new data on the next tick (~15:35 NY) and the marker stops further sends.
    # The machine clock is fixed UTC-5 with NO daylight saving, so this in-script NY check —
    # not the OS trigger time — is what keeps the 3:20pm-ET start correct year-round.
    # Bypassed by --force-send; not relevant to --seed.
    if not (args.force_send or args.seed):
        if (ny.hour, ny.minute) < (15, 20):
            print(f"Release: before the 15:20 NY release window (NY now {ny:%a %H:%M}); nothing to do.")
            return

    # Cheap pre-check: skip the heavy refresh when there's nothing new (release mode only;
    # force always proceeds to a full refresh + send).
    if not (args.force_send or args.seed):
        try:
            cheap = latest_release_date()
            if cheap is not None and str(cheap) == marker:
                print(f"No new COT report (latest {cheap} already emailed on record). Nothing to do.")
                return
        except Exception as e:
            print(f"Pre-check failed ({e}); continuing to full refresh.")

    detail, _ = cotdata.compute(force=True)
    if detail is None or detail.empty:
        print("No COT data available (network?) — skipping this run.")
        return
    latest = pd.to_datetime(detail["date"]).max().date()

    # Sanity checks: the as-of date must be a Tuesday and reasonably fresh.
    if latest.weekday() != 1:
        print(f"Latest as-of {latest} is not a Tuesday — unexpected; skipping out of caution.")
        return
    stale_days = (pd.Timestamp.now().date() - latest).days

    if args.seed:
        MARKER.write_text(str(latest))
        print(f"Seeded marker to {latest} (no email sent).")
        return
    # Staleness guard (bypassed only by --force-send): don't email a non-fresh report.
    if not args.force_send and stale_days > MAX_STALE_DAYS:
        print(f"Latest report {latest} is {stale_days}d old (> {MAX_STALE_DAYS}) — not a fresh release; skipping.")
        return
    # Release dedup: don't re-send a report already emailed.
    if not args.force_send and str(latest) == marker:
        print(f"Report as-of {latest} already emailed. Nothing to do.")
        return

    print(f"Sending COT report as of {latest}…")
    r = subprocess.run(
        [sys.executable, str(COTREPORT_CLI), str(cotdata.COT_DETAIL_FILE), str(OUT_PDF),
         "--asof", str(latest), "--threshold", "80", "--quality", "email"],
        capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    send_email(OUT_PDF, latest, dry_run=args.dry_run, to_override=args.to)
    if not args.dry_run:
        MARKER.write_text(str(latest))                 # release marker
        print(f"Sent. Marker updated (release={latest}).")


if __name__ == "__main__":
    main()
