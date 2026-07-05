"""Scheduled USDA Reaction emailer — when a new quarterly Grain Stocks report posts
(Jan→Dec-1, Mar→Mar-1, Jun→Jun-1, Sep→Sep-1; June also brings Acreage), build the
reaction note and email it to the dashboard's USDA Reaction recipient list.

Idempotent + safe on a generous schedule (mirrors cot_scheduled_email.py / opec_scheduled_email.py):
it asks NASS QuickStats for the most recent quarterly stocks reading and only sends when that
reading is NEWER than the last one emailed (marker stores its as-of date) AND fresh (a staleness
guard rejects a months-old prior quarter, so an empty marker never dumps a stale note). Needs
NASS_API_KEY in the environment. Email credentials + the managed recipient list ("usda_reaction")
are reused from the COT emailer.

Usage:
  python usda_reaction_scheduled_email.py                # scheduled run (send only if a new quarter posted)
  python usda_reaction_scheduled_email.py --dry-run      # build but do NOT send
  python usda_reaction_scheduled_email.py --force-send   # rebuild + send the latest now, bypass gates
  python usda_reaction_scheduled_email.py --seed         # record the latest as already sent (no email)
  python usda_reaction_scheduled_email.py --period SEP --year 2026 --dry-run   # test a specific quarter
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import pandas as pd                                                              # noqa: E402
from src import agdata                                                           # noqa: E402
from cot_scheduled_email import load_email_cfg, _managed_recipients, send_report_email  # noqa: E402

MARKER = ROOT / "data" / "signals" / "usda_reaction_emailed.txt"     # last stocks as-of date emailed (YYYY-MM-DD)
USDAREACTION_CLI = ROOT / "src" / "usdareaction.py"
OUT_PDF = ROOT / "data" / "USDA_Reaction.pdf"
MAX_STALE_DAYS = 60        # don't email a quarter whose as-of is older than this (rejects the prior quarter)
PERIOD_MMDD = {"MAR": (3, 1), "JUN": (6, 1), "SEP": (9, 1), "DEC": (12, 1)}
PERIOD_LABEL = {"MAR": "March 1", "JUN": "June 1", "SEP": "September 1", "DEC": "December 1"}


def read_marker() -> str:
    return MARKER.read_text().strip() if MARKER.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build but do NOT send")
    ap.add_argument("--force-send", action="store_true", help="rebuild + send the latest now, bypass gates")
    ap.add_argument("--seed", action="store_true", help="record the latest as already sent (no email)")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    ap.add_argument("--period", choices=list(PERIOD_LABEL), default=None, help="force a quarter (testing)")
    ap.add_argument("--year", type=int, default=None, help="force a stocks year (testing)")
    args = ap.parse_args()

    if not (args.force_send or args.dry_run or args.seed):   # scheduled auto-send path
        from src import automation
        if not automation.report_enabled("usda_reaction"):
            print("USDA automatic sending is OFF (turn it on: dashboard -> Recipients -> Scheduled reports). Skipping.")
            return

    marker = read_marker()

    if args.period and args.year:
        mm, dd = PERIOD_MMDD[args.period]
        latest = {"key": args.period, "year": args.year, "asof": pd.Timestamp(year=args.year, month=mm, day=dd)}
    else:
        if not agdata.NASS_KEY:
            print("NASS_API_KEY not set — cannot check the release. Nothing to do.")
            return
        latest = agdata.latest_stocks_period()
        if latest is None:
            print("Could not read the latest stocks period from NASS. Nothing to do.")
            return

    period, year = latest["key"], latest["year"]
    asof_str = latest["asof"].strftime("%Y-%m-%d")
    label = PERIOD_LABEL[period]

    if args.seed:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(asof_str)
        print(f"Seeded marker to {asof_str} ({label} {year}); no email.")
        return

    if not args.force_send:
        if asof_str == marker:
            print(f"Latest stocks ({label} {year}, as-of {asof_str}) already emailed. Nothing to do.")
            return
        stale = (pd.Timestamp.now().normalize() - latest["asof"]).days
        if stale > MAX_STALE_DAYS:
            print(f"Latest stocks ({label} {year}, as-of {asof_str}) is {stale}d past its as-of "
                  f"(> {MAX_STALE_DAYS}) — not a fresh release; skipping.")
            return

    print(f"Building USDA Reaction note — {label} {year} ({period})…")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(USDAREACTION_CLI), str(OUT_PDF), "--period", period, "--year", str(year)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    sender, app_pw, desk = load_email_cfg()
    to = args.to or _managed_recipients("usda_reaction", desk)
    if period == "JUN":
        subject = f"USDA Acreage & Grain Stocks — Reaction (June {year})"
        intro = (f"<p>The USDA <b>Acreage</b> and <b>Grain Stocks</b> reports for {year} are out — planted area "
                 "vs the March intentions, wheat by class, June-1 stocks (on-farm vs off-farm) and implied use "
                 "are in the attached note.</p>")
    else:
        subject = f"USDA Grain Stocks — Reaction ({label} {year})"
        intro = (f"<p>USDA <b>Grain Stocks</b> as of <b>{label} {year}</b> are out — total vs year-ago, the "
                 "on-farm vs off-farm split and implied quarterly use are in the attached note.</p>")

    sent = send_report_email(OUT_PDF, subject=subject, intro_html=intro, attachment_name="USDA_Reaction.pdf",
                             report_key="usda_reaction", to_override=to, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] would email '{subject}' to {', '.join(sent)}.")
    else:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(asof_str)
        print(f"Emailed '{subject}' to {', '.join(sent)}. Marker → {asof_str}.")


if __name__ == "__main__":
    main()
