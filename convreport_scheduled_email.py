"""Scheduled WEEKLY Technical Analysis Report emailer.

Builds the merged Technical Analysis report (src/convreport.py) from the current
signals cache and emails it to the desk / managed recipients. Defaults to OFF — the
scheduled path sends only when the report has been switched on in the dashboard
(Recipients -> Alert settings -> auto-email), a second safety layer on top of the
Windows task being enabled.

Runs off the cached snapshot (no Bloomberg needed at run time). For a genuinely fresh
weekly read, schedule a snapshot pull (or 'Re-run signals') to land before this job, or
pass --refresh to recompute the signals offline from the last snapshot first.

Usage:
  python convreport_scheduled_email.py                 # scheduled run (send only if switched on)
  python convreport_scheduled_email.py --dry-run       # build + show, do NOT send
  python convreport_scheduled_email.py --force-send    # build + send now, bypassing the on/off gate
  python convreport_scheduled_email.py --to a@b.com    # override recipients (e.g. a test to yourself)
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

CONVREPORT_CLI = ROOT / "src" / "convreport.py"
SIGNALS_FILE = ROOT / "data" / "signals" / "opportunities.parquet"
OUT_PDF = ROOT / "data" / "Technical_Analysis_Report_email.pdf"
NY_TZ = ZoneInfo("America/New_York")


def _qualifying(df, mode, top, min_conv, min_score) -> int | None:
    """How many setups the report WOULD write up, without building it — so a week where
    nothing clears the quality bar costs nothing and sends nothing. Returns None if the
    check can't run (caller then builds anyway: a missing weekly report is worse than a
    thin one). Uses convreport.select itself, so it can never drift from the report."""
    try:
        sys.path.insert(0, str(ROOT / "src"))       # convreport resolves reportkit via src/
        from src import convreport
        sel = convreport.select(df, top_n=top, mode=mode,
                                min_conviction=min_conv, min_score=min_score)
        return len(sel["constructive"]) + len(sel["cautious"])
    except Exception as e:
        print(f"(pre-check unavailable: {e}) — building anyway")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build the report but do NOT send")
    ap.add_argument("--force-send", action="store_true", help="build + send now, bypassing the on/off gate")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--top", type=int, default=None, help="picks in total (strongest-overall mode) "
                                                          "or per side (balanced mode)")
    ap.add_argument("--mode", choices=["per_side", "overall", "threshold"], default=None,
                    help="selection mode; defaults to the app's saved default. In 'threshold' mode "
                         "only setups clearing the desk's quality bar go out, and if NOTHING clears "
                         "it no email is sent at all.")
    ap.add_argument("--min-conviction", type=float, default=None,
                    help="threshold mode floor; defaults to the value saved in the app")
    ap.add_argument("--min-score", type=float, default=None,
                    help="threshold mode floor; defaults to the value saved in the app")
    ap.add_argument("--send-empty", action="store_true",
                    help="send even when nothing clears the bar (default: stay silent)")
    ap.add_argument("--refresh", action="store_true", help="recompute signals offline from the last snapshot first")
    args = ap.parse_args()

    # Everything defaults to what the desk saved in the app (Technical Analysis → report →
    # 📌 Set as default), so the scheduled send runs on exactly the configuration they see.
    from src.specs import ta_report_defaults
    _rd = ta_report_defaults()
    mode = args.mode or _rd["mode"]
    top = int(args.top if args.top is not None else _rd["top_n"])
    min_conv = float(args.min_conviction if args.min_conviction is not None else _rd["min_conviction"])
    min_score = float(args.min_score if args.min_score is not None else _rd["min_score"])
    print(f"Report settings: mode={mode}, top={top}"
          + (f", bar conviction ≥ {min_conv:g} / |score| ≥ {min_score:g}" if mode == "threshold" else ""))

    # Auto-send gate: only email when the user has switched this report ON in the app.
    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("convreport"):
            print("Technical Analysis Report automatic email is OFF "
                  "(dashboard -> Recipients -> Alert settings -> auto-email). Skipping.")
            return

    if args.refresh:
        try:
            import run_daily
            run_daily.run()
            print("Recomputed signals from the cached snapshot.")
        except Exception as e:
            print(f"(refresh skipped: {e})")

    if not SIGNALS_FILE.exists():
        print("No signals cache yet — run 'Re-run signals' / pull a snapshot first. Skipping.")
        return

    # Quality bar: if nothing clears it there is nothing worth a client's attention this week, so
    # send NOTHING rather than a padded (or empty) report. Checked BEFORE building, so a quiet week
    # costs no render time either. --force-send / --send-empty override; a failed check builds anyway.
    if mode == "threshold" and not (args.force_send or args.send_empty):
        try:
            import pandas as pd
            n_q = _qualifying(pd.read_parquet(SIGNALS_FILE), mode, top, min_conv, min_score)
        except Exception as e:
            print(f"(pre-check unavailable: {e}) — building anyway")
            n_q = None
        if n_q == 0:
            print(f"Nothing clears the quality bar (conviction ≥ {min_conv:g}, "
                  f"|score| ≥ {min_score:g}) — no report sent this week.")
            return

    asof = datetime.now(NY_TZ).strftime("%d %b %Y")
    print(f"Building the Technical Analysis Report ({asof})…")
    cmd = [sys.executable, str(CONVREPORT_CLI), str(SIGNALS_FILE), str(OUT_PDF),
           "--asof", asof, "--top", str(top), "--mode", mode, "--baseline"]
    if _rd.get("ai_polish", True):
        cmd.append("--ai-polish")
    if mode == "threshold":
        cmd += ["--min-conviction", str(min_conv), "--min-score", str(min_score)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    from cot_scheduled_email import send_report_email
    intro = (f"<p>Please find this week's <b>Technical Analysis Report</b> ({asof}) — the conviction "
             "leaderboard across the strategy book, then the strongest constructive and cautious "
             "setups, each with a multi-indicator chart and a short read, plus a watchlist of names "
             "where the strategies disagree.</p>")
    to = send_report_email(OUT_PDF, f"Technical Analysis Report — {asof}", intro,
                           "Technical_Analysis_Report.pdf", report_key="convreport",
                           to_override=args.to, dry_run=args.dry_run)
    print(("[dry-run] would email to " if args.dry_run else "Emailed to ") + ", ".join(to or []))


if __name__ == "__main__":
    main()
