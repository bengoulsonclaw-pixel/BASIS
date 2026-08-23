"""Email the Brazil Production report, and/or the hedge-sizing table on its own.

Two sends, deliberately separate:

  --report   the full branded PDF (front-page ranking, a page per commodity, the
             hedge-sizing appendix) as an attachment.
  --matrix   the company x product hedge table ONLY, rendered inline as HTML so it
             reads on a phone without opening anything, with the CSV attached for
             working up a call list.

Both reuse the desk's Gmail OAuth path from cot_scheduled_email.py rather than
minting a second credential, and both carry the XP compliance disclaimer.

    python brazil_scheduled_email.py --report --matrix [--to a@b.com] [--dry-run]

--dry-run builds everything and prints the recipients WITHOUT sending.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cot_scheduled_email import DISCLAIMER_HTML, DISCLAIMER_TEXT, _gmail_send, load_email_cfg
from src import brazilprod, recipients as recips

PDF = ROOT / "reports" / "Brazil_Production.pdf"
FONT = "font-family:Arial,Helvetica,sans-serif;color:#222;"
GOLD, RULE, DIM = "#C8901A", "#D8D8D2", "#9AA0A6"


def _to_list(to_override: str | None) -> list:
    if to_override:
        return [a.strip() for a in to_override.split(",") if a.strip()]
    return recips.get("brazil", None) or ["benjamin.goulson@xpi.com.br"]


# ── the hedge table as inline HTML ───────────────────────────────────────────
def matrix_html(mat: pd.DataFrame) -> str:
    """The company x product table as an email-safe HTML table: inline styles only,
    no flexbox, no external CSS — Outlook's Word renderer ignores most of it."""
    ratios, periods = brazilprod.HEDGE_RATIOS, [s for s, _d in brazilprod.HEDGE_PERIODS]
    th = f"padding:3px 5px;border-bottom:1px solid {RULE};font-size:11px;text-align:right;"
    thl = th.replace("text-align:right", "text-align:left")
    td = "padding:3px 5px;font-size:11px;text-align:right;white-space:nowrap;"
    tdl = td.replace("text-align:right", "text-align:left")

    head1 = (f'<th style="{thl}" rowspan="2">Company</th>'
             f'<th style="{thl}" rowspan="2">Product</th>'
             f'<th style="{th}" rowspan="2">Annual production</th>'
             f'<th style="{thl}" rowspan="2">Contract</th>')
    for p in ratios:
        head1 += (f'<th style="{th}text-align:center;border-left:1px solid {RULE};'
                  f'color:{GOLD};font-weight:700;" colspan="3">{p}%</th>')
    # one hue per period, shared with the page and the PDF via brazilprod
    PC = brazilprod.PERIOD_COLOUR
    head2 = "".join(
        f'<th style="{th}font-weight:700;color:{PC[s]};'
        f'{f"border-left:1px solid {RULE};" if i == 0 else ""}">{s}</th>'
        for _p in ratios for i, s in enumerate(periods))

    body, prev = "", None
    for _i, r in mat.iterrows():
        dim = f"color:{DIM};" if not r["_avail"] else ""
        top = f"border-top:1px solid {RULE};" if r["Company"] != prev else ""
        body += f'<tr><td style="{tdl}{top}font-weight:700;">' \
                f'{"" if r["Company"] == prev else r["Company"]}</td>'
        body += f'<td style="{tdl}{top}{dim}">{r["Product"]}</td>'
        body += f'<td style="{td}{top}">{r["Annual production"]:,.2f} {r["Unit"]}</td>'
        body += f'<td style="{tdl}{top}{dim}">{r["Contract"]}</td>'
        for p in ratios:
            for i, s in enumerate(periods):
                v = r[f"{p}% {s}"]
                edge = f"border-left:1px solid {RULE};" if i == 0 else ""
                tint = dim or f"color:{PC[s]};"
                body += (f'<td style="{td}{top}{edge}{tint}">'
                         f'{"—" if pd.isna(v) else f"{int(v):,}"}</td>')
        body += "</tr>"
        prev = r["Company"]

    # TOTAL at the foot — the whole addressable book on one line.
    tot = brazilprod.hedge_totals(mat)
    tf = "padding:5px;border-top:2px solid #1a1a1a;font-size:11px;font-weight:700;"
    foot = (f'<tr><td style="{tf}text-align:left;">TOTAL</td>'
            f'<td style="{tf}text-align:left;" colspan="3">'
            f'{tot["_n_hedgeable"]} hedgeable lines</td>')
    for p_ in ratios:
        for i, s_ in enumerate(periods):
            edge = f"border-left:1px solid {RULE};" if i == 0 else ""
            foot += (f'<td style="{tf}text-align:right;white-space:nowrap;{edge}'
                     f'color:{PC[s_]};">{tot.get(f"{p_}% {s_}", 0):,}</td>')
    foot += "</tr>"

    return (f'<table cellspacing="0" cellpadding="0" style="border-collapse:collapse;{FONT}">'
            f'<thead><tr>{head1}</tr><tr>{head2}</tr></thead><tbody>{body}</tbody>'
            f'<tfoot>{foot}</tfoot></table>')


def send_matrix(to: list, dry_run: bool = False) -> None:
    sender, _pw, _desk = load_email_cfg()
    mat = brazilprod.hedge_matrix(include_unhedgeable=True)
    if mat.empty:
        raise SystemExit("no hedge matrix — run src/brazilprod.py first")
    hedgeable = mat[mat["_avail"]]
    n_co, n_rows = mat["Company"].nunique(), len(mat)
    total = int(hedgeable["100% yr"].sum())
    asof = datetime.now().strftime("%d %b %Y")

    intro = (
        f'<p style="{FONT}font-size:14px;">Brazil producer hedge sizing, {asof}.</p>'
        f'<p style="{FONT}font-size:14px;">Every named Brazilian producer split by product, with '
        f'the number of futures lots a hedge of that year\'s output would require at four hedge '
        f'ratios — each shown per year, per month and per trading day (252 days). '
        f'<b>{n_co} companies, {n_rows} company-product lines, '
        f'{total:,} lots</b> at a full hedge of a full year.</p>'
        f'<p style="{FONT}font-size:13px;color:#555;">Lots are each producer\'s share of Brazil '
        f'applied to Brazil\'s national output, converted at each contract\'s size. A blank row '
        f'means the product has <b>no listed future</b> (pulp, niobium, nickel, manganese, '
        f'bauxite) — not that the producer was missed. Bracketed qualifiers say what the share '
        f'measures: <i>exports</i> for the trade houses, <i>processing</i> for the meat packers, '
        f'<i>equity</i> for the oil partners. Poultry has no meat contract, so the integrators '
        f'are sized on their feed hedge (corn + soybean meal). These are the lots a full hedge '
        f'would require, not a forecast of what any producer will trade.</p>'
        f'<p style="{FONT}font-size:13px;">Colour marks the period, not the value: '
        f'<b style="color:{PC["yr"]}">per year</b> &middot; '
        f'<b style="color:{PC["mth"]}">per month</b> &middot; '
        f'<b style="color:{PC["day"]}">per trading day</b>.</p>')
    html = (f'<div style="{FONT}">{intro}{matrix_html(mat)}'
            f'<p style="{FONT}font-size:12px;color:#555;margin-top:14px;">CSV attached.</p>'
            f'{DISCLAIMER_HTML}</div>')
    text = (f"Brazil producer hedge sizing, {asof}. {n_co} companies, {n_rows} company-product "
            f"lines, {total:,} lots at a full hedge of a full year. See the attached CSV.\n\n"
            f"{DISCLAIMER_TEXT}")

    msg = MIMEMultipart("mixed")
    msg["From"], msg["To"] = sender, ", ".join(to)
    msg["Subject"] = f"Brazil producers — futures hedge sizing by company and product ({asof})"
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain"))
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)
    csv = mat.drop(columns=["_lots", "_avail"]).to_csv(index=False).encode("utf-8")
    att = MIMEApplication(csv, _subtype="csv")
    att.add_header("Content-Disposition", "attachment",
                   filename="Brazil_hedge_by_company_product.csv")
    msg.attach(att)

    if dry_run:
        print(f"  [dry-run] MATRIX email ready for {', '.join(to)} "
              f"({n_rows} rows, {len(csv) / 1024:.0f}KB CSV)")
        return
    _gmail_send(msg)
    print(f"  Sent the hedge table to {', '.join(to)}")


def send_report(to: list, dry_run: bool = False) -> None:
    sender, _pw, _desk = load_email_cfg()
    if not PDF.exists():
        raise SystemExit(f"no report at {PDF} — run src/brazilreport.py first")
    store = brazilprod.load() or {}
    coms = store.get("commodities") or {}
    n_co = sum(1 for c in coms.values() if c.get("companies"))
    asof = datetime.now().strftime("%d %b %Y")

    html = (
        f'<div style="{FONT}">'
        f'<p style="font-size:14px;">Please find the Brazil Production report, {asof}.</p>'
        f'<p style="font-size:14px;">Where Brazil sits in the global supply of {len(coms)} '
        f'commodities, then a page per commodity showing what share of world output Brazil '
        f'produces and which companies produce Brazil\'s share ({n_co} of them carry a '
        f'company-level breakdown). A closing appendix sizes the futures hedge for every '
        f'named producer, by product.</p>'
        f'<p style="font-size:13px;color:#555;">Country data is USDA FAS PS&amp;D and the EIA; '
        f'metals, pulp and ethanol come from the USGS Mineral Commodity Summaries and the '
        f'industry associations. Each company table states what it measures and how good the '
        f'data is — several are desk estimates and are marked as such.</p>'
        f'{DISCLAIMER_HTML}</div>')
    text = (f"The Brazil Production report, {asof}. Brazil's share of world supply across "
            f"{len(coms)} commodities, a page per commodity, and a hedge-sizing appendix. "
            f"See the attached PDF.\n\n{DISCLAIMER_TEXT}")

    msg = MIMEMultipart("mixed")
    msg["From"], msg["To"] = sender, ", ".join(to)
    msg["Subject"] = f"Brazil Production report — {asof}"
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain"))
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)
    att = MIMEApplication(PDF.read_bytes(), _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename=PDF.name)
    msg.attach(att)

    if dry_run:
        print(f"  [dry-run] REPORT email ready for {', '.join(to)} "
              f"({PDF.stat().st_size / 1e6:.1f}MB PDF)")
        return
    _gmail_send(msg)
    print(f"  Sent the report to {', '.join(to)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="email the full PDF")
    ap.add_argument("--matrix", action="store_true", help="email the hedge table on its own")
    ap.add_argument("--to", default=None, help="comma-separated override recipients")
    ap.add_argument("--dry-run", action="store_true", help="build but do not send")
    args = ap.parse_args()
    if not (args.report or args.matrix):
        ap.error("nothing to send — pass --report and/or --matrix")
    to = _to_list(args.to)
    if args.report:
        send_report(to, args.dry_run)
    if args.matrix:
        send_matrix(to, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
