"""Scheduled Weekly Signal Scorecard emailer.

Builds the Weekly Signal Scorecard (src/sigscore.py) off the precomputed signal ledger
and emails it to the managed recipients. Defaults to OFF — the scheduled path sends only
when the report has been switched on in the dashboard (Recipients -> Alert settings),
the same two-layer gate as every other auto-report (Windows task enabled AND the
data/automation.json flag).

Runs entirely off data/signal_cache/ledger_outcomes.parquet (rebuilt by each morning
snapshot), so a Monday-morning run after the snapshot covers the full prior trading
week. No Bloomberg needed at run time.

Suggested task (created disabled; the Recipients page toggle enables it):
  schtasks /create /tn "BASIS Weekly Signal Scorecard" /sc weekly /d MON /st 07:45 ^
    /tr "<python> <this script>"

Usage:
  python sigscore_scheduled_email.py               # scheduled run (send only if switched on)
  python sigscore_scheduled_email.py --dry-run     # build + show, do NOT send
  python sigscore_scheduled_email.py --force-send  # build + send now, bypassing the on/off gate
  python sigscore_scheduled_email.py --to a@b.com  # override recipients (e.g. a test to yourself)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SIGSCORE_CLI = ROOT / "src" / "sigscore.py"
LEDGER_FILE = ROOT / "data" / "signal_cache" / "ledger_outcomes.parquet"
NY_TZ = ZoneInfo("America/New_York")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build the report but do NOT send")
    ap.add_argument("--force-send", action="store_true",
                    help="build + send now, bypassing the on/off gate")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    args = ap.parse_args()

    # Auto-send gate: only email when the user has switched this report ON in the app.
    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("sigscore"):
            print("Weekly Signal Scorecard automatic email is OFF "
                  "(dashboard -> Recipients -> Alert settings). Skipping.")
            return

    if not LEDGER_FILE.exists():
        print("No signal ledger on disk yet — run backfill_signals.py / a snapshot first. Skipping.")
        return

    asof = datetime.now(NY_TZ).strftime("%d %b %Y")
    out_pdf = ROOT / "data" / "Weekly_Signal_Scorecard_email.pdf"
    print(f"Building the Weekly Signal Scorecard ({asof})…")
    r = subprocess.run([sys.executable, str(SIGSCORE_CLI), str(out_pdf), "--asof", asof],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out_pdf.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    from cot_scheduled_email import send_report_email
    intro = (f"<p>Please find this week's <b>Signal Scorecard</b> ({asof}) — what the desk's "
             "technical-signal book flagged this week, how the prior weeks' calls resolved at "
             "5, 10 and 21 sessions, the book's accuracy trend, and which signal families are "
             "carrying the current regime. A historical read of the signals' accuracy — "
             "not P&amp;L and not a recommendation.</p>")
    to = send_report_email(out_pdf, f"Weekly Signal Scorecard — {asof}", intro,
                           "Weekly_Signal_Scorecard.pdf", report_key="sigscore",
                           to_override=args.to, dry_run=args.dry_run)
    print(("[dry-run] would email to " if args.dry_run else "Emailed to ") + ", ".join(to or []))


if __name__ == "__main__":
    main()
