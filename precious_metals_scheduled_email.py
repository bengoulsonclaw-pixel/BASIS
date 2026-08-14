"""Scheduled MONTHLY Precious Metals Fundamentals monitor emailer.

Builds the 4-page client monitor (src/pmdata.py -> src/pmreport.py) and emails it
to the managed recipient list — seeded to the desk only, so the report lands in
Ben's inbox for proofreading and he forwards the checked copy to clients.
Defaults to OFF — the scheduled path sends only when the report is switched on in
the dashboard (Recipients -> Alert settings -> auto-email), a second safety layer
on top of the Windows task being enabled.

Schedule (register once; the panel then toggles it):
  schtasks /create /tn "Precious Metals Monitor (monthly)" /sc monthly /d 5 /st 07:00 ^
    /tr "\"%CD%\run_precious_metals.bat\""

Usage:
  python precious_metals_scheduled_email.py               # scheduled run (send only if switched on)
  python precious_metals_scheduled_email.py --dry-run     # rebuild data + PDF, do NOT send
  python precious_metals_scheduled_email.py --force-send  # build + send now, bypassing gate + month marker
  python precious_metals_scheduled_email.py --to a@b.com  # override recipients (e.g. a test to yourself)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "data" / "pm_monitor.json"
OUT_PDF = ROOT / "data" / "Precious_Metals_Report.pdf"
MARKER = ROOT / "data" / "signals" / "pm_emailed.txt"
NY_TZ = ZoneInfo("America/New_York")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build the report but do NOT send")
    ap.add_argument("--force-send", action="store_true",
                    help="build + send now, bypassing the on/off gate and the month marker")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--force", action="store_true", help="bypass the data-pull freshness caches")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the real-Chrome CME/US Mint refresh (uses the existing archive)")
    args = ap.parse_args()

    month = datetime.now(NY_TZ).strftime("%B %Y")

    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("precious_metals"):
            print("Precious Metals automatic email is OFF "
                  "(dashboard -> Recipients -> Alert settings -> auto-email). Skipping.")
            return
        if MARKER.exists() and MARKER.read_text(encoding="utf-8").strip() == month:
            print(f"{month} edition already emailed — skipping (marker: {MARKER}).")
            return

    if not args.no_fetch:
        # Refresh the bot-protected pulls (CME stocks + US Mint) via the real-Chrome
        # fetcher — best effort: a Chrome hiccup must not kill the monthly send,
        # pmdata just falls back to the existing archive (or mock, DRAFT-flagged).
        import subprocess
        print("Refreshing CME / US Mint / Swiss pulls (brief Chrome window + ~680 MB Swiss file)…")
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pm_fetch.py")],
                           capture_output=True, text=True, timeout=2400)
        print(r.stdout or "", end="")
        if r.returncode != 0:
            print(f"(pm_fetch failed — continuing on the existing archive)\n{r.stderr[-800:]}")
        # Bloomberg pulls (ETF holdings, real yield, dollar) — only works with a
        # terminal session; on other machines it fails fast and the cached
        # parquets from the last terminal run carry the report.
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pm_bbg.py")],
                           capture_output=True, text=True, timeout=600)
        print(r.stdout or "", end="")
        if r.returncode != 0:
            print("(pm_bbg failed — no terminal? continuing on the cached pulls)")

    print(f"Building the Precious Metals Fundamentals monitor ({month})…")
    from src import pmdata, pmreport
    d = pmdata.build(force=args.force)
    OUT_JSON.write_text(json.dumps(d, indent=1), encoding="utf-8")
    pmreport.build_pdf(d, OUT_PDF)
    print(f"Built {OUT_PDF} (mock blocks: {', '.join(d['mock_blocks']) or 'none'})")

    intro = (f"<p>Please find the <b>Precious Metals Fundamentals</b> monitor for <b>{month}</b> — "
             "macro & positioning across gold, silver, platinum and palladium, then physical flows, "
             "official-sector activity, holdings and the published market balances per metal.</p>")
    if d["mock_blocks"]:
        intro += ("<p><b>DRAFT — placeholder data in: " + ", ".join(d["mock_blocks"]) +
                  ". Not for distribution.</b></p>")

    from cot_scheduled_email import send_report_email
    to = send_report_email(OUT_PDF, f"Precious Metals Fundamentals — {month}", intro,
                           "Precious_Metals_Fundamentals.pdf", report_key="precious_metals",
                           to_override=args.to, dry_run=args.dry_run)
    if not args.dry_run:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(month, encoding="utf-8")
    print(("[dry-run] would email to " if args.dry_run else "Emailed to ") + ", ".join(to or []))


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("Precious Metals monitor", main)
