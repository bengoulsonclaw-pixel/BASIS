"""Global Macro Trade Idea — the desk's BLANK house template.

Every other report in this repo renders a computed view; this one deliberately renders
nothing but the XP layout (inset black sidebar, yellow stripe, title band, banners,
compliance disclaimer) under a fixed `Global Macro Trade Idea` headline, so a colleague
can drop their own idea into it and send it on to clients in house style.

Two artefacts off the same template (templates/tradeidea.html):
  • build_pdf()  — the blank PDF: what the finished page looks like.
  • build_html() — a self-contained, click-to-type copy the recipient opens in any
                   browser and prints back to PDF (Ctrl+P → Save as PDF). The toolbar,
                   dashed guides and grey prompts are screen-only, so what they print is
                   the same house page. The logo is inlined as a data URI, so the file
                   works as an email attachment with no network and no other files.

Standalone (the app calls this as a subprocess, like every other report script):
    python src/tradeidea.py out.pdf --html out.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Dual import: `python src/tradeidea.py …` (how the app runs it) AND `from src import tradeidea`
# (how the app reads the file names below). Put src/ on the path so `reportkit` resolves either way.
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jinja2 import Environment, FileSystemLoader                  # noqa: E402
from reportkit import pretty_date, data_uri, render_pdf           # noqa: E402

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"

PDF_NAME = "XP_Global_Macro_Trade_Idea_TEMPLATE.pdf"
HTML_NAME = "XP_Global_Macro_Trade_Idea_TEMPLATE.html"


def render_html(editable: bool = False, asof=None) -> str:
    """The template's HTML. `editable=True` gives the fillable browser copy."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("tradeidea.html").render(
        editable=editable,
        asof=pretty_date(asof or date.today()),
        logo=data_uri(ASSETS / "logo.png"),
    )


def build_pdf(out_path, asof=None) -> str:
    return render_pdf(render_html(editable=False, asof=asof), out_path)


def build_html(out_path, asof=None) -> str:
    out_path = Path(out_path)
    out_path.write_text(render_html(editable=True, asof=asof), encoding="utf-8")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Build the blank Global Macro Trade Idea template.")
    ap.add_argument("out_pdf")
    ap.add_argument("--html", default=None, help="also write the fillable browser copy here")
    args = ap.parse_args()
    build_pdf(args.out_pdf)
    print(f"Wrote {args.out_pdf}")
    if args.html:
        build_html(args.html)
        print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
