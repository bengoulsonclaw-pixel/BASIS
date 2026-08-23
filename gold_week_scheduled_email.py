"""Scheduled Gold — Week Ahead emailer.

Builds the weekly gold report (src/goldreport.py) and emails it to the managed list
(seeded to the desk only — Ben proofreads, then forwards).

The Recipients panel has offered a "Gold — Week Ahead" subscription and
data/automation.json has carried a `gold_week` flag since the report was built, but
nothing read either: there was no sender, no automation-registry entry and no task.
Adding recipients did nothing at all, silently. This closes that.

Defaults to OFF. The scheduled path sends only when switched on in the dashboard
(Recipients -> Scheduled reports), on top of the Windows task toggle — Ben owns both
switches and neither is flipped from here.

Schedule (register once; the panel then toggles it):
  schtasks /create /tn "Gold Week Ahead (Mondays)" /sc weekly /d MON /st 07:15 ^
    /tr "\"%CD%\run_gold_week.bat\""

Usage:
  python gold_week_scheduled_email.py               # scheduled: send if enabled
  python gold_week_scheduled_email.py --dry-run     # build the PDF, no send
  python gold_week_scheduled_email.py --force-send  # send now, ignoring the toggle
  python gold_week_scheduled_email.py --to a@b.com  # override recipients
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports"
PDF_NAME = "Gold_Week_Ahead_{d}.pdf"
SUBJECT = "Gold — Week Ahead, {d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build but do NOT send")
    ap.add_argument("--force-send", action="store_true",
                    help="send now, ignoring the dashboard toggle")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--asof", default=date.today().isoformat())
    args = ap.parse_args()

    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("gold_week"):
            print("Gold — Week Ahead auto-email is OFF "
                  "(dashboard -> Recipients -> Scheduled reports). Skipping.")
            return 0

    from src import goldreport
    asof = date.fromisoformat(args.asof)
    payload = goldreport.build_payload(asof)

    # The report is explicitly allowed to run with a dead calendar — it says so on the
    # page rather than printing an empty week — so this is a note, not a failure.
    if not payload["calendar_ok"]:
        print("  note: calendar feed unreachable; the report states that in section 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / PDF_NAME.format(d=asof.isoformat())
    goldreport.build_pdf(payload, out_pdf)
    print(f"built {out_pdf.name} ({out_pdf.stat().st_size:,} bytes), "
          f"{len(payload['events'])} high-impact events")

    if args.dry_run:
        print("[dry-run] not sending")
        return 0

    from cot_scheduled_email import send_report_email
    intro = (
        "<p>Please find this week's <b>Gold — Week Ahead</b>: what is scheduled, how "
        "gold has behaved around those releases historically, and what a given move "
        "in rates or the dollar has been worth.</p>"
        "<p><b>This report contains no forecast of the gold price.</b> It describes "
        "measured relationships and historical behaviour only — see the methodology "
        "note on the final page.</p>")
    to = send_report_email(out_pdf, SUBJECT.format(d=asof.strftime("%d %B %Y")), intro,
                           out_pdf.name, report_key="gold_week",
                           to_override=args.to, dry_run=False)
    print("Emailed to " + ", ".join(to or []))
    return 0


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash (one alert/day)
    guard("Gold — Week Ahead", main)
