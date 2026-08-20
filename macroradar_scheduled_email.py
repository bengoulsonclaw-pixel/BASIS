"""Scheduled Macro Rate Radar emailer — Monday mornings, the week's policy-rule read.

Builds the branded Macro Rate Radar PDF for each bank (Fed / ECB / BoE) off free public
data — no Bloomberg anywhere in this path, so it works on a morning the Terminal is shut —
and emails the three as one message. Idempotent per ISO week (marker file), gated on the
Recipients-panel toggle exactly like every other scheduled report, and the Windows task
ships DISABLED until Ben switches it on in the app.

Usage:
  python macroradar_scheduled_email.py              # scheduled run (send once per ISO week)
  python macroradar_scheduled_email.py --dry-run    # build everything, send nothing
  python macroradar_scheduled_email.py --force-send # rebuild + send now regardless of marker
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from datetime import date, datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATAFEED_MODE", "snapshot")   # strip prices come from the morning store

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MARKER = ROOT / "data" / "signals" / "macroradar_emailed.txt"    # ISO week last sent
MC_MAIN = Path(os.getenv("BASIS_MC_DIR",
                         r"C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements")) / "main.py"
BANKS = ["FED", "ECB", "BOE"]
NY_TZ = ZoneInfo("America/New_York")


def load_email_cfg():
    """(_EMAIL_FROM, _EMAIL_APP_PW, _EMAIL_TO) parsed out of the Morning Coffee project
    without importing it — single source of truth for desk email settings."""
    tree = ast.parse(MC_MAIN.read_text(encoding="utf-8", errors="ignore"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("_EMAIL_FROM", "_EMAIL_APP_PW",
                                                        "_EMAIL_TO"):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out["_EMAIL_FROM"], out.get("_EMAIL_APP_PW"), out["_EMAIL_TO"]


def _managed_recipients(report, fallback):
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


# ── Gmail API (OAuth) — same token reuse as the other schedulers ─────────────────────
_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send",
                 "https://www.googleapis.com/auth/gmail.modify"]
_GMAIL_DIR = MC_MAIN.parent
_GMAIL_SVC = None


def _gmail_service():
    global _GMAIL_SVC
    if _GMAIL_SVC is not None:
        return _GMAIL_SVC
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    tok = _GMAIL_DIR / "token.json"
    creds = Credentials.from_authorized_user_file(str(tok), _GMAIL_SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            tok.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise RuntimeError(f"Gmail token invalid — re-run gmail_auth.py in {_GMAIL_DIR}")
    _GMAIL_SVC = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _GMAIL_SVC


def _gmail_send(msg) -> None:
    import base64
    import time
    global _GMAIL_SVC
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    last = None
    for attempt in range(4):
        try:
            _gmail_service().users().messages().send(userId="me", body={"raw": raw}).execute()
            return
        except Exception as e:
            last = e
            print(f"  (send attempt {attempt + 1} failed: {str(e)[:80]} — retrying)")
            _GMAIL_SVC = None
            time.sleep(4 * (attempt + 1))
    raise last


def build_pdfs() -> list[tuple[str, Path]]:
    """One branded PDF per bank. A failed bank is reported and skipped, not fatal —
    a dead ONS endpoint should not silence the Fed and ECB reads."""
    from src import macroradarreport
    out = []
    for bank in BANKS:
        try:
            p = macroradarreport.build(bank=bank)
            out.append((bank, p))
            print(f"  built {bank}: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ({bank} report FAILED: {str(e)[:160]})")
    return out


def headline_lines() -> list[str]:
    """One observational line per bank for the email body (client-safe voice)."""
    from src import macroradar
    lines = []
    for bank in BANKS:
        try:
            r = macroradar.compare(bank)
            if r.summary and r.summary.median_gap_bp is not None:
                g = r.summary.median_gap_bp
                side = "above" if g > 0 else "below"
                vs = ""
                if r.headline_bp is not None:
                    vs = (f"; versus the strip the rule path sits "
                          f"{abs(r.headline_bp):.0f}bp "
                          f"{'higher' if r.headline_bp > 0 else 'lower'} at the far meeting")
                lines.append(f"<b>{bank}</b>: median rule {r.summary.median:.2f}% — "
                             f"{abs(g):.0f}bp {side} the current setting{vs}.")
        except Exception:
            continue
    return lines


def send_email(pdfs: list[tuple[str, Path]], dry_run: bool = False) -> None:
    sender, _pw, fallback = load_email_cfg()
    recipients = _managed_recipients("macroradar", fallback)
    today = datetime.now(NY_TZ).strftime("%d %b %Y")
    subject = f"Macro Rate Radar — policy rules vs market pricing, week of {today}"

    bullets = "".join(f"<li style='margin:2px 0'>{l}</li>" for l in headline_lines())
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#222;font-size:14px;">'
        "<p>Please find this week's Macro Rate Radar — simple monetary-policy rules "
        "(the Taylor-rule family) evaluated on current public data for the Fed, ECB and "
        "Bank of England, set against the meeting path each futures strip currently "
        "prices.</p>"
        + (f"<ul style='margin:6px 0 10px'>{bullets}</ul>" if bullets else "")
        + "<p>Rule prescriptions are illustrative benchmarks rather than forecasts; the "
          "attached reports set out the inputs, sources and workings in full.</p>"
        "<p>Best regards,<br>Ben</p></div>")

    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    msg["Date"] = datetime.now(NY_TZ).strftime("%a, %d %b %Y %H:%M:%S %z")
    msg.attach(MIMEText(html, "html"))
    for bank, p in pdfs:
        with open(p, "rb") as fh:
            part = MIMEApplication(fh.read(), _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=p.name)
        msg.attach(part)

    if dry_run:
        print(f"  DRY RUN — would send '{subject}' to {recipients} "
              f"with {len(pdfs)} PDFs attached")
        return
    _gmail_send(msg)
    print(f"  sent to {', '.join(recipients)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-send", action="store_true")
    a = ap.parse_args()

    # Second safety layer on top of the disabled task: refuse when the Recipients-panel
    # toggle is off (data/automation.json).
    if not (a.dry_run or a.force_send):
        from src import automation
        if not automation.report_enabled("macroradar"):
            print("macroradar auto-send is OFF in the Recipients panel — not sending.")
            return

    week = date.today().isocalendar()
    week_key = f"{week[0]}-W{week[1]:02d}"
    if not a.force_send and not a.dry_run:
        last = MARKER.read_text().strip() if MARKER.exists() else ""
        if last == week_key:
            print(f"already sent for {week_key} — nothing to do.")
            return

    print("building Macro Rate Radar PDFs…")
    pdfs = build_pdfs()
    if not pdfs:
        print("no PDF could be built — not sending.")
        return
    # build_pdfs let the macrodata fetchers refresh any stale data/macro_cache series,
    # and the Hot Sheet's MACRO lines read exactly those caches — re-persist the sheet
    # (cache only, no history re-stamp).
    try:
        from src import hotsheet
        hotsheet.refresh_collection()
        print("Hot Sheet re-collected off the fresh macro caches.")
    except Exception as e:
        print(f"(Hot Sheet refresh skipped: {e})")
    send_email(pdfs, dry_run=a.dry_run)
    if not a.dry_run:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(week_key, encoding="utf-8")


if __name__ == "__main__":
    main()
