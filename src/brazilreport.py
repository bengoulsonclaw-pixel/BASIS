"""Brazil Production report — a branded client PDF of the 🇧🇷 Brazil Production page:
Brazil's place in world supply across the whole book, then, for the commodities that
carry one, the company breakdown of Brazil's own share.

The prose is deliberately neutral-observational (compliance: no buy/sell language),
and every company block prints its BASIS and its DATA QUALITY, so an export share or
a desk estimate can never be read as reported production.

Run standalone (the app calls it as a subprocess):
    python src/brazilreport.py out.pdf [--keys iron_ore,crude_oil,pulp]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from jinja2 import Environment, FileSystemLoader      # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from reportkit import pretty_date, data_uri, png, render_pdf   # noqa: E402
from src import brazilprod                                     # noqa: E402

TEMPLATES = _ROOT / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"
BLUE = "#3C6E9F"
GREY = "#B8BEC6"
INK = "#1a1a1a"

# Print at most this many company blocks — the PDF is a client leave-behind, not the
# whole database. Anything dropped is named in the footnote, never silently cut.
MAX_BLOCKS = 6


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#CCCCCC")
    ax.tick_params(length=0, labelsize=6.4, colors="#333333")
    ax.grid(axis="x", color="#EEEEEE", lw=0.8)
    ax.set_axisbelow(True)


def overview_png(coms: list) -> str:
    """Every commodity ranked by Brazil's share of world production."""
    d = sorted(coms, key=lambda c: c["share"])
    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.26 * len(d) + 0.7)))
    ax.barh([c["label"] for c in d], [c["share"] for c in d], color=GOLD, height=0.68)
    for c, y in zip(d, range(len(d))):
        ax.text(c["share"] + 0.9, y, f"{c['share']:.1f}%", va="center", fontsize=6.2, color=INK)
    ax.set_xlabel("Brazil's share of world production (%)", fontsize=6.6, color="#333333")
    ax.set_xlim(0, max(c["share"] for c in d) * 1.16)
    _style(ax)
    fig.tight_layout()
    return png(fig)


def country_png(com: dict) -> str:
    """Producing countries for one commodity, Brazil picked out in gold."""
    # barh puts index 0 at the BOTTOM, so build bottom-up: the "Other" bucket first,
    # then the named producers ascending — reading down the chart gives biggest to
    # smallest with Other pinned at the foot.
    named = sorted([r for r in com["countries"] if not r.get("is_other")],
                   key=lambda r: -r["value"])[:10]
    other = [r for r in com["countries"] if r.get("is_other")][:1]
    rows = other + named[::-1]
    fig, ax = plt.subplots(figsize=(3.0, max(1.7, 0.22 * len(rows) + 0.6)))
    colours = [GOLD if r["is_brazil"] else (GREY if r.get("is_other") else BLUE) for r in rows]
    ax.barh([r["country"] for r in rows], [r["value"] for r in rows], color=colours, height=0.66)
    ax.set_xlabel(f"{com['year_label']} ({com['unit']})", fontsize=6.2, color="#333333")
    _style(ax)
    fig.tight_layout()
    return png(fig)


def company_png(com: dict) -> str:
    """Who inside Brazil produces it — the 'Other' bucket muted so the named
    producers carry the eye."""
    blk = com["companies"]
    rows = blk["rows"][::-1]
    fig, ax = plt.subplots(figsize=(3.0, max(1.7, 0.24 * len(rows) + 0.6)))
    colours = [GREY if r["is_other"] else GOLD for r in rows]
    ax.barh([r["company"][:34] for r in rows], [r["share_brazil"] for r in rows],
            color=colours, height=0.66)
    for r, y in zip(rows, range(len(rows))):
        ax.text(r["share_brazil"] + 0.7, y, f"{r['share_brazil']:.0f}%", va="center",
                fontsize=6.0, color=INK)
    ax.set_xlabel(blk.get("axis_label") or "share of Brazil (%)", fontsize=6.2, color="#333333")
    ax.set_xlim(0, max(r["share_brazil"] for r in rows) * 1.2)
    _style(ax)
    fig.tight_layout()
    return png(fig)


def _block(com: dict) -> dict:
    blk = com["companies"]
    return {
        "label": com["label"], "icon": com["icon"],
        "brazil": f"{com['brazil']:,.2f}", "world": f"{com['world']:,.2f}",
        "unit": com["unit"], "share": f"{com['share']:.1f}", "rank": com.get("rank"),
        "year_label": com["year_label"], "source": com["source_label"],
        "basis_label": blk["basis_label"], "basis_note": blk["basis_note"],
        "confidence_label": blk["confidence_label"], "is_estimate": blk["confidence"] == "estimate",
        "is_export": blk["basis"] == "export",
        "co_year": blk.get("year"), "co_source": blk.get("source"), "note": blk.get("note"),
        "coverage": (None if blk.get("coverage_pct") is None else f"{blk['coverage_pct']:.0f}"),
        "country_chart": country_png(com), "company_chart": company_png(com),
        "rows": [{"company": r["company"],
                  "volume": ("" if blk.get("unit_is_pct") else f"{r['volume']:,.2f}"),
                  "share_brazil": f"{r['share_brazil']:.1f}",
                  "share_world": f"{r['share_world']:.2f}",
                  "ticker": r.get("ticker") or "—",
                  "is_other": r["is_other"]} for r in blk["rows"]],
        "unit_is_pct": bool(blk.get("unit_is_pct")),
        "co_unit": blk.get("unit"),
    }


def render_html(store: dict, keys: list | None = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    coms = list((store.get("commodities") or {}).values())
    if not coms:
        raise SystemExit("no Brazil production store — run src/brazilprod.py first")

    head = sorted(coms, key=lambda c: -c["share"])
    withco = [c for c in head if c.get("companies")]
    if keys:
        withco = [c for c in withco if c["key"] in keys]
    shown, dropped = withco[:MAX_BLOCKS], withco[MAX_BLOCKS:]

    top = head[0]
    lead = (f"Brazil is the world's largest producer of {top['label'].lower()} at "
            f"{top['share']:.0f}% of global output" if top.get("rank") == 1 else
            f"Brazil's largest share of any market here is {top['label'].lower()}, at "
            f"{top['share']:.0f}% of world output")

    return env.get_template("brazilreport.html").render(
        asof=pretty_date(store.get("built", "")[:10]),
        n_com=len(coms), n_blocks=len(shown), lead=lead,
        curated_as_of=store.get("curated_as_of") or "—",
        overview=overview_png(head),
        rows=[{"label": c["label"], "icon": c["icon"], "group": c["group"],
               "year_label": c["year_label"], "brazil": f"{c['brazil']:,.2f}",
               "world": f"{c['world']:,.2f}", "unit": c["unit"],
               "share": f"{c['share']:.1f}", "rank": c.get("rank"),
               "has_co": bool(c.get("companies"))} for c in head],
        blocks=[_block(c) for c in shown],
        dropped=", ".join(c["label"] for c in dropped),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(out_path, keys: list | None = None, store: dict | None = None) -> str:
    store = store or brazilprod.load_or_build()
    return render_pdf(render_html(store, keys), out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_pdf")
    ap.add_argument("--keys", default="", help="comma-separated commodity keys to detail")
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None
    build_pdf(args.out_pdf, keys)
    print(f"Wrote {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
