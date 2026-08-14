"""Scheduled BASIS Weekly Review emailer.

Builds the cross-module Weekly Review (src/weekreview.py) — the exception-based Monday
wrap of what every module's own thresholds flagged this week — and emails it to the
managed recipients. Defaults to OFF — the scheduled path sends only when the report has
been switched on in the dashboard (Recipients -> Alert settings), the same two-layer
gate as every other auto-report (Windows task enabled AND the data/automation.json flag).

Runs entirely off the modules' cached parquet/json stores (each rebuilt by its own daily
job / the morning snapshot), so a Monday-morning run after the snapshot frames the week
ahead and catches Friday's COT. No Bloomberg needed at run time.

Suggested task (created disabled; the Recipients page toggle enables it):
  schtasks /create /tn "BASIS Weekly Review" /sc weekly /d MON /st 07:55 ^
    /tr "<python> <this script>"

Usage:
  python weekreview_scheduled_email.py               # scheduled run (send only if switched on)
  python weekreview_scheduled_email.py --dry-run     # build + show, do NOT send
  python weekreview_scheduled_email.py --force-send  # build + send now, bypassing the on/off gate
  python weekreview_scheduled_email.py --to a@b.com  # override recipients (e.g. a test to yourself)
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

WEEKREVIEW_CLI = ROOT / "src" / "weekreview.py"
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
        if not automation.report_enabled("weekreview"):
            print("Weekly Review automatic email is OFF "
                  "(dashboard -> Recipients -> Alert settings). Skipping.")
            return

    asof = datetime.now(NY_TZ).strftime("%d %b %Y")
    out_pdf = ROOT / "data" / "Weekly_Review_email.pdf"
    print(f"Building the Weekly Review ({asof})…")
    # --baseline: only the scheduled weekly send rolls the previous-edition store, so
    # ad-hoc previews in the app never eat the week's deltas (the convreport convention).
    r = subprocess.run([sys.executable, str(WEEKREVIEW_CLI), str(out_pdf), "--asof", asof,
                        "--baseline"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out_pdf.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    from cot_scheduled_email import send_report_email
    intro = (f"<p>Please find this week's <b>Weekly Review</b> ({asof}) — the cross-module "
             "wrap of what the desk's screens are flagging going into the week: volatility "
             "dislocations, curve extremes, positioning reads, technical setups past the "
             "quality bar, correlation breaks, opening seasonal windows and metals flows — "
             "with the technical scorecard folded in (how the signal book's earlier calls "
             "actually resolved), plus the week's releases and the calendar ahead. Each line "
             "is that module's "
             "own threshold speaking — observations of the screens, not recommendations.</p>")
    to = send_report_email(out_pdf, f"BASIS Weekly Review — {asof}", intro,
                           "Weekly_Review.pdf", report_key="weekreview",
                           to_override=args.to, dry_run=args.dry_run)
    print(("[dry-run] would email to " if args.dry_run else "Emailed to ") + ", ".join(to or []))


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("Weekly Review", main)
