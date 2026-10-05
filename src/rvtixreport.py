"""RV Trade Tickets — desk sheet PDF.

A branded, one-piece desk sheet of the RV Trade Tickets book: each stretched Curve / RV spread
as a mechanical fade ticket (structure, entry / target / stop, size and its 10-year backtest),
ranked by forward edge, with the forward track record summarised at the head.

These are prescriptive tickets (explicit structures and sizes), so this is an INTERNAL DESK tool,
not a client send — the title and summary say so, and the standard compliance disclaimer carries.
Mechanical observations against each spread's own history, never a recommendation.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on screen):
    python src/rvtixreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from reportkit import data_uri, pretty_date, render_pdf

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    ov = d.get("overall") or {}
    return env.get_template("rvtixreport.html").render(
        asof=_fmt_asof(d.get("asof", "")),
        window=d.get("window"), threshold=f"{d.get('threshold', 2.0):g}",
        tp_label=d.get("tp_label", "the mean"), risk=d.get("risk"),
        n_tickets=d.get("n_tickets", len(d.get("tickets", []))),
        n_positive=d.get("n_positive", 0),
        tickets=d.get("tickets", []), overall=ov, has_overall=bool(ov),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_path) -> str:
    return render_pdf(render_html(d), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload_json")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    d = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))
    build_pdf(d, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
