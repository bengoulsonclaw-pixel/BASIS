"""Scheduled Product Correlations alert — daily after the close, check whether any
product pair's 1-month correlation sits at an extreme (≤5th / ≥95th percentile) of
its own rolling 1-year range with a material move (|1M − 1Y| ≥ 0.30). If nothing is
extreme it exits silently; when something is, it builds the branded Product
Correlations PDF for the sectors involved and emails it with the pairs listed.

Idempotent: a marker file records the last date emailed, so re-runs the same day
no-op. Gated (like every auto-report) by the Recipients page toggle — the flag in
data/automation.json — plus the Windows task state itself.

Register the Windows task (the panel then shows/toggles it):
  schtasks /create /tn "Product Correlations Alert (daily)" /sc daily /st 18:30 ^
    /tr "\"<venv python>\" \"<this file>\""

Usage:
  python sectorcorr_scheduled_email.py             # scheduled run (send only if extremes + not yet sent today)
  python sectorcorr_scheduled_email.py --dry-run   # everything except the actual send
  python sectorcorr_scheduled_email.py --force-send
  python sectorcorr_scheduled_email.py --to a@b.c  # override recipients (test send)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")   # evening run: the morning snapshot is the day's data

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import pandas as pd                    # noqa: E402
from src import sectorcorr, universe   # noqa: E402

MARKER = ROOT / "data" / "signals" / "sectorcorr_emailed.txt"   # last date emailed (daily dedup)
REPORT_CLI = ROOT / "src" / "sectorcorrreport.py"
OUT_PDF = ROOT / "data" / "Product_Correlations_email.pdf"


def _mat(m: pd.DataFrame) -> list:
    return [[None if pd.isna(v) else float(v) for v in row] for row in m.values]


MAX_PDF_SECTORS = 2      # matrices stay readable; further extremes live in the breaks table


def build_payload(asof: date, extremes: pd.DataFrame) -> dict:
    """The PDF payload for the sectors of the TOP extreme pairs (extremes arrive
    ranked by |1M − 1Y|), plus the whole-book breaks table and the diversification
    index — same shape the app page writes. On a noisy day extremes can span the
    whole book; the sector cap keeps the matrices legible."""
    sectors: list = []
    for sa, sb in zip(extremes["sector_a"], extremes["sector_b"]):
        need = [s for s in (sa, sb) if s not in sectors]
        if len(sectors) + len(need) <= MAX_PDF_SECTORS:
            sectors.extend(need)
    sectors = [s for s in sectorcorr.SECTOR_ORDER if s in sectors]
    ci = sectorcorr.instrument_corr("realized", asof, sectors)
    names = {}
    for t in ci.labels:
        nm = universe.name(t)
        names[t] = f"{nm} ({t.split()[0]})" if any(
            universe.name(o) == nm for o in ci.labels if o != t) else nm
    D = ci.diff.rename(index=names, columns=names)
    span = max(0.2, float((D.abs().max().max() * 10 // 1 + 1) / 10))
    bt = sectorcorr.top_breaks("realized", asof, n=15)
    payload = {
        "asof": asof.isoformat(), "sectors": sectors, "mode": os.environ["DATAFEED_MODE"],
        "metric_label": "1M realized-vol changes",
        "labels": [names[t] for t in ci.labels],
        "m1y": _mat(ci.long_.rename(index=names, columns=names)),
        "m1m": _mat(ci.short_.rename(index=names, columns=names)),
        "diff": _mat(D), "diff_span": span,
        "breaks": [
            {"pair": f"{universe.name(a)} ↔ {universe.name(b)}",
             "sectors": sa if sa == sb else f"{sa} / {sb}",
             "c1y": float(y), "c1m": float(m), "d": float(d),
             "pctl": None if pd.isna(p) else float(p)}
            for a, b, sa, sb, y, m, d, p in zip(
                bt["a"], bt["b"], bt["sector_a"], bt["sector_b"],
                bt["corr_1y"], bt["corr_1m"], bt["diff"], bt["pctl"])],
    }
    di = sectorcorr.diversification_index("realized", asof)
    if di is not None:
        payload.update(div_dates=[x.isoformat() for x in di.index.date],
                       div_avg=[float(v) for v in di["avg"]],
                       div_abs=[float(v) for v in di["avg_abs"]])
    return payload


MAX_INTRO_PAIRS = 10     # a noisy day shouldn't produce a 40-bullet email body


def intro_html(asof: date, extremes: pd.DataFrame) -> str:
    shown = extremes.head(MAX_INTRO_PAIRS)
    items = "".join(
        f"<li><b>{universe.name(a)} &harr; {universe.name(b)}</b> — 1M correlation "
        f"{m:+.2f} vs {y:+.2f} over the year ({p:.0f}th percentile of its own range, "
        f"{'co-movement loosened' if k == 'breakdown' else 'unusual lockstep'})</li>"
        for a, b, y, m, p, k in zip(shown["a"], shown["b"], shown["corr_1y"],
                                    shown["corr_1m"], shown["pctl"], shown["kind"]))
    more = (f"<p>&hellip; and {len(extremes) - len(shown)} further pairs in the attached "
            f"report's breaks table.</p>" if len(extremes) > len(shown) else "")
    return (f"<p>As of {asof:%d %b %Y}, the following product pairs are trading at an extreme "
            f"of their own 1-year correlation range and may be worth a closer look:</p>"
            f"<ul>{items}</ul>{more}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="do everything except actually send")
    ap.add_argument("--force-send", action="store_true", help="send now, bypassing the gates")
    ap.add_argument("--to", nargs="+", default=None, help="override recipients (e.g. a test to Ben only)")
    args = ap.parse_args()

    if not (args.force_send or args.dry_run):        # scheduled auto-send path
        from src import automation
        if not automation.report_enabled("sectorcorr"):
            print("Product Correlations alert is OFF (dashboard -> Alert Settings). Skipping.")
            return

    today = date.today()
    marker = MARKER.read_text().strip() if MARKER.exists() else ""
    if not args.force_send and marker == str(today):
        print(f"Already emailed today ({today}). Nothing to do.")
        return

    extremes = sectorcorr.percentile_extremes("realized", today)
    if extremes.empty:
        print(f"No correlation extremes as of {today} — nothing to alert. "
              "(Pairs need |1M − 1Y| ≥ 0.30 at the ≤5th / ≥95th percentile.)")
        return

    print(f"{len(extremes)} extreme pair(s) as of {today} — building the report…")
    payload = build_payload(today, extremes)
    with tempfile.TemporaryDirectory() as td:
        pj = Path(td) / "payload.json"
        pj.write_text(json.dumps(payload), encoding="utf-8")
        r = subprocess.run([sys.executable, str(REPORT_CLI), str(pj), str(OUT_PDF)],
                           capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Report generation FAILED:\n" + (r.stderr or r.stdout or "no output"))
        sys.exit(1)

    import cot_scheduled_email as mail   # the shared desk emailer (Gmail creds + recipients)
    sent = mail.send_report_email(
        OUT_PDF, subject=f"BASIS — Correlation Break Alert ({today:%d %b %Y})",
        intro_html=intro_html(today, extremes),
        attachment_name="Product_Correlations.pdf",
        report_key="sectorcorr", to_override=args.to, dry_run=args.dry_run)
    if not args.dry_run:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(str(today))
        print(f"Sent to {', '.join(sent)}. Marker updated ({today}).")


if __name__ == "__main__":
    main()
