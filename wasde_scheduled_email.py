"""Scheduled WASDE emailer — on each USDA WASDE release (monthly), build the dedicated WASDE
report (the US + world supply & demand / stocks-to-use balance sheet) and email it to the
dashboard's WASDE recipient list.

Idempotent + safe on a generous schedule (mirrors cot / opec / usda_reaction emailers): it reads
the latest past WASDE date from the USDA release calendar (agdata) and only sends when that date
is NEWER than the last one emailed (marker) AND fresh (a staleness guard rejects an old WASDE, so
an empty marker never dumps a months-old one). The dedicated WASDE report self-fetches the current
USDA PS&D balance sheet each run, so it always reflects whatever USDA is serving — the afternoon
task window lets PS&D update after the noon-ET release before the note goes out. **Automatic
sending is OFF unless switched on in the dashboard Recipients page** (the automation flag). Email
creds + managed recipients reuse the COT emailer.

Usage:
  python wasde_scheduled_email.py                # scheduled run (send only if a new WASDE posted)
  python wasde_scheduled_email.py --dry-run      # build but do NOT send
  python wasde_scheduled_email.py --force-send   # rebuild + send the latest now, bypass gates
  python wasde_scheduled_email.py --seed         # record the latest as already sent (no email)
  python wasde_scheduled_email.py --asof 2026-06-11 --dry-run   # test a specific WASDE date
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
try:                                    # keep prints safe under the scheduled task's cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import pandas as pd                                                              # noqa: E402
from src import agdata                                                           # noqa: E402
from cot_scheduled_email import load_email_cfg, _managed_recipients, send_report_email  # noqa: E402

MARKER = ROOT / "data" / "signals" / "wasde_emailed.txt"     # last WASDE date emailed (YYYY-MM-DD)
WASDEREPORT_CLI = ROOT / "src" / "wasdereport.py"
OUT_PDF = ROOT / "data" / "WASDE_Report.pdf"
MAX_STALE_DAYS = 12          # don't email a WASDE older than this (rejects a months-old prior release)


def read_marker() -> str:
    return MARKER.read_text().strip() if MARKER.exists() else ""


def latest_wasde():
    """The most recent WASDE release date on/before today, from the USDA release calendar."""
    cal = agdata.report_calendar()
    past = cal[(cal["report"] == "WASDE") & (cal["date"] <= pd.Timestamp.now().normalize())]
    return past["date"].max() if not past.empty else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build but do NOT send")
    ap.add_argument("--force-send", action="store_true", help="rebuild + send the latest now, bypass gates")
    ap.add_argument("--seed", action="store_true", help="record the latest as already sent (no email)")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--asof", default=None, help="force a WASDE date YYYY-MM-DD (testing)")
    args = ap.parse_args()

    if not (args.force_send or args.dry_run or args.seed):   # scheduled auto-send path
        from src import automation
        if not automation.report_enabled("wasde"):
            print("WASDE automatic sending is OFF (turn it on: dashboard -> Recipients -> Scheduled reports). Skipping.")
            return

    marker = read_marker()
    wdate = pd.Timestamp(args.asof) if args.asof else latest_wasde()
    if wdate is None:
        print("No WASDE release found in the calendar. Nothing to do.")
        return
    asof_str = wdate.strftime("%Y-%m-%d")

    if args.seed:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(asof_str)
        print(f"Seeded marker to {asof_str}; no email.")
        return

    if not args.force_send:
        if asof_str == marker:
            print(f"Latest WASDE ({asof_str}) already emailed. Nothing to do.")
            return
        stale = (pd.Timestamp.now().normalize() - wdate).days
        if stale > MAX_STALE_DAYS:
            print(f"Latest WASDE ({asof_str}) is {stale}d old (> {MAX_STALE_DAYS}) - not a fresh release; skipping.")
            return
        # Fire the moment PS&D reflects the new WASDE, NOT on a fixed clock time: before the noon-ET
        # release (or while the FAS feed lags) PS&D still shows last month's stocks, so skip and let the
        # scheduled repeat retry; the first run after the data posts sends. This also sidesteps the
        # machine's fixed UTC-5 / no-DST clock drifting vs ET across the year. (Bypass with --force-send.)
        chk = subprocess.run([sys.executable, str(WASDEREPORT_CLI), "--check-fresh"],
                             capture_output=True, text=True)
        if "FRESH" not in (chk.stdout or ""):
            print("USDA PS&D hasn't refreshed with the new WASDE yet (still last month's stocks); "
                  "skipping — the scheduled repeat will retry the moment it posts.")
            return

    # The dedicated WASDE report self-fetches the current PS&D balance sheet each run (a fresh HTTP pull).
    print(f"Building WASDE report for {asof_str} (from USDA PS&D) ...")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(WASDEREPORT_CLI), str(OUT_PDF),
                        "--asof", wdate.strftime("%d %b %Y") + " WASDE"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    sender, app_pw, desk = load_email_cfg()
    to = args.to or _managed_recipients("wasde", desk)
    subject = f"USDA WASDE — Supply & Demand ({wdate:%B %Y})"
    intro = (f"<p>The monthly USDA <b>WASDE</b> for <b>{wdate:%B %Y}</b> is out — the US balance sheet "
             "(production, use, exports, ending stocks), world ending stocks and stocks-to-use are in the "
             "attached note.</p>")
    sent = send_report_email(OUT_PDF, subject=subject, intro_html=intro,
                             attachment_name="WASDE_Report.pdf",
                             report_key="wasde", to_override=to, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] would email '{subject}' to {', '.join(sent)}.")
    else:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(asof_str)
        print(f"Emailed '{subject}' to {', '.join(sent)}. Marker -> {asof_str}.")


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("USDA WASDE report", main)
