"""Scheduled BASIS Hot Sheet emailer.

Builds the client-facing daily Hot Sheet PDF (src/hotsheetreport.py — the print cut of
the 🔥 Hot Sheet page, off the SAME persisted sheet the page reads) and emails it to the
managed recipients. Defaults to OFF — the scheduled path sends only when the report has
been switched on in the dashboard (Recipients -> Alert settings), the same two-layer
gate as every other auto-report (Windows task enabled AND the data/automation.json flag).

Two further guards sit on top of that gate:
  * FRESHNESS — the desk pulls by hand, so a morning without a pull must never email
    yesterday's sheet: unless data/signals/hotsheet_cache.json carries TODAY's collected
    stamp (machine-local date) the run skips cleanly — BEFORE the report engine runs, so
    a stale cache can never trigger the engine's live re-collect either. This guard is
    absolute: even --force-send won't email a sheet that isn't today's (refresh it from
    the app's Pull button / the Hot Sheet page first).
  * MARKER — data/signals/hotsheet_emailed.txt records the last date emailed, so a
    re-run the same day no-ops (the sectorcorr convention).

Suggested task (created disabled; the Recipients page toggle enables it):
  schtasks /create /tn "BASIS Hot Sheet (daily)" /sc weekly /d MON,TUE,WED,THU,FRI ^
    /st 08:15 /tr "<venv python> <this script>"

Usage:
  python hotsheet_scheduled_email.py               # scheduled run (send only if switched on + fresh)
  python hotsheet_scheduled_email.py --dry-run     # build + show, do NOT send
  python hotsheet_scheduled_email.py --force-send  # bypass the on/off gate + today's sent-marker
                                                   #   (the freshness guard still applies)
  python hotsheet_scheduled_email.py --to a@b.com  # override recipients (e.g. a test to yourself)
  python hotsheet_scheduled_email.py --no-ai       # deterministic template intro (skip AI polish)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

REPORT_CLI = ROOT / "src" / "hotsheetreport.py"
CACHE_FILE = ROOT / "data" / "signals" / "hotsheet_cache.json"   # the morning pull writes this
MARKER = ROOT / "data" / "signals" / "hotsheet_emailed.txt"      # last date emailed (daily dedup)
OUT_PDF = ROOT / "data" / "Hot_Sheet_email.pdf"


def sheet_collected_date():
    """Machine-local date of the persisted sheet's `collected` stamp, or None when the
    cache is missing/unreadable (both mean: no pull today — do not email)."""
    try:
        c = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return datetime.fromtimestamp(float(c["collected"])).date()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build the report but do NOT send")
    ap.add_argument("--force-send", action="store_true",
                    help="bypass the on/off gate and the sent-marker (NOT the freshness guard)")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the AI polish of the intro (deterministic template only)")
    args = ap.parse_args()

    # Auto-send gate: only email when the user has switched this report ON in the app.
    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("hotsheet"):
            print("Hot Sheet automatic email is OFF "
                  "(dashboard -> Recipients -> Alert settings). Skipping.")
            return

    today = date.today()
    marker = MARKER.read_text().strip() if MARKER.exists() else ""
    if not (args.force_send or args.dry_run) and marker == str(today):
        print(f"Already emailed today ({today}). Nothing to do.")
        return

    # Freshness guard — absolute, and BEFORE the engine runs: the desk pulls by hand, so a
    # morning without a pull must never email yesterday's sheet (and a stale cache must
    # never trigger the engine's live re-collect from a scheduled run either).
    stamp = sheet_collected_date()
    if stamp != today:
        print(f"Hot Sheet cache is not from today (collected "
              f"{stamp or 'missing/unreadable'}, today {today}) — the morning pull hasn't "
              "run yet. Skipping without sending; pull first, then re-run.")
        return

    print(f"Building the Hot Sheet PDF ({today})…")
    cmd = [sys.executable, str(REPORT_CLI), str(OUT_PDF)] + (["--no-ai"] if args.no_ai else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    from src.reportkit import pretty_date
    asof = pretty_date(today)
    intro = (f"<p>Please find today's <b>Hot Sheet</b> ({asof}) — the desk's cross-asset "
             "daily highlights: each line is something one of the desk's own screens is "
             "flagging today, ranked on a common heat scale. Observations of the screens, "
             "not recommendations.</p>")
    from cot_scheduled_email import send_report_email
    to = send_report_email(OUT_PDF, f"BASIS Hot Sheet — {asof}", intro,
                           "Hot_Sheet.pdf", report_key="hotsheet",
                           to_override=args.to, dry_run=args.dry_run)
    if args.dry_run:
        print("[dry-run] would email to " + ", ".join(to or []))
    else:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(str(today))
        print(f"Emailed to {', '.join(to or [])}. Marker updated ({today}).")


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("Hot Sheet daily email", main)
