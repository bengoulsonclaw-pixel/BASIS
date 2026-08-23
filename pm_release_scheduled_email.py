"""Scheduled PM release-synopsis emailer — the OPEC model for metals.

Checks daily whether WGC Gold Demand Trends or the WPIC Platinum Quarterly has
a new edition; on a new release it fetches + parses the publication, builds the
one-page branded synopsis (src/pmrel.py -> src/pmrelreport.py) and emails it to
the managed list (seeded to the desk only — Ben proofreads, then forwards).
Defaults to OFF: the scheduled path sends only when switched on in the
dashboard (Recipients -> Alert settings), on top of the Windows task toggle.

Schedule (register once; the panel then toggles it):
  schtasks /create /tn "PM Release Synopses (daily)" /sc daily /st 08:30 ^
    /tr "\"%CD%\run_pm_releases.bat\""

Usage:
  python pm_release_scheduled_email.py                # scheduled: send new editions only
  python pm_release_scheduled_email.py --dry-run      # build whatever is latest, no send
  python pm_release_scheduled_email.py --force-send   # (re)send the latest editions now
  python pm_release_scheduled_email.py --pub wgc      # limit to one publication
  python pm_release_scheduled_email.py --to a@b.com   # override recipients
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MARKER = ROOT / "data" / "signals" / "pm_releases_emailed.json"
REL_DIR = ROOT / "data" / "pm_releases"
PDF_NAME = {"wgc": "WGC_GDT_{ed}_Synopsis.pdf", "wpic": "WPIC_PQ_{ed}_Synopsis.pdf"}
SUBJECT = {"wgc": "Gold Demand Trends {ed} — Synopsis",
           "wpic": "WPIC Platinum Quarterly {ed} — Synopsis"}


def _marker() -> dict:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build but do NOT send")
    ap.add_argument("--force-send", action="store_true",
                    help="send the latest editions now, ignoring the sent-marker")
    ap.add_argument("--pub", choices=["wgc", "wpic"], default=None, help="limit to one publication")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients")
    args = ap.parse_args()

    if not (args.force_send or args.dry_run):
        from src import automation
        if not automation.report_enabled("pm_releases"):
            print("PM release synopses auto-email is OFF "
                  "(dashboard -> Recipients -> Alert settings -> auto-email). Skipping.")
            return

    from src import pmrel, pmrelreport
    sent_marker = _marker()
    pubs = [args.pub] if args.pub else ["wgc", "wpic"]
    for pub in pubs:
        try:
            info = pmrel.detect(pub)
        except Exception as e:
            print(f"{pub}: detection failed ({e}) — will retry next run.")
            continue
        if not info:
            print(f"{pub}: no edition found on the page — will retry next run.")
            continue
        if not (args.force_send or args.dry_run) and sent_marker.get(pub) == info["edition"]:
            print(f"{pub}: {info['edition']} already emailed — skipping.")
            continue

        # Source licence gate. Refusing here, BEFORE the build, means an unapproved
        # source costs a log line rather than a crash mail — and the marker is left
        # untouched, so the edition sends itself the moment Ben approves the source
        # on the Compliance page.
        blockers = pmrelreport.licence_blockers(pub)
        if blockers and not args.dry_run:
            print(f"{pub}: NOT SENT — {'; '.join(blockers)}")
            print(f"{pub}: approve this source on the BASIS Compliance page "
                  f"(SYSTEM -> 🛡️ Compliance -> Third-party data) to release it.")
            continue

        print(f"{pub}: building the {info['edition']} synopsis…")
        d = pmrel.build(pub, info)
        ed_tag = d["edition"].replace(" ", "_")
        out_pdf = REL_DIR / PDF_NAME[pub].format(ed=ed_tag)
        pmrelreport.build_pdf(d, out_pdf, client_facing=not args.dry_run)

        from cot_scheduled_email import send_report_email
        intro = (f"<p>Please find a one-page synopsis of <b>{d['label']}</b> "
                 f"({d['edition']} edition), published by {d['publisher']} — key figures, "
                 f"the publisher's own highlights and a context chart.</p>"
                 f"<p>{d['headline']}</p>")
        to = send_report_email(out_pdf, SUBJECT[pub].format(ed=d["edition"]), intro,
                               out_pdf.name, report_key="pm_releases",
                               to_override=args.to, dry_run=args.dry_run)
        print(("[dry-run] would email to " if args.dry_run else "Emailed to ")
              + ", ".join(to or []))
        if not args.dry_run:
            sent_marker[pub] = d["edition"]
            MARKER.parent.mkdir(parents=True, exist_ok=True)
            MARKER.write_text(json.dumps(sent_marker, indent=1), encoding="utf-8")


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("PM release synopsis (WGC/WPIC)", main)
