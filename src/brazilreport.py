"""Brazil Production report — a branded client PDF of the 🇧🇷 Brazil Production page.

Front page: where Brazil sits in world supply across the whole book, as the ranked
share chart plus the numbers behind it. Then ONE PAGE PER COMMODITY, ordered by
Brazil's share of world production, each showing what share of world output Brazil
produces and — where the industry is concentrated enough for the question to have an
answer — which companies produce what share of Brazil's own output.

The prose is deliberately neutral-observational (compliance: no buy/sell language),
and every company block prints its BASIS and its DATA QUALITY, so an export share or
a desk estimate can never be read as reported production. Commodities with no company
table say so on their page rather than being quietly dropped from the report.

Run standalone (the app calls it as a subprocess):
    python src/brazilreport.py out.pdf [--keys iron_ore,crude_oil,pulp]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd
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

# Every commodity gets its own page. --keys narrows it; whatever that leaves out is
# named in the closing footnote rather than silently cut.


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
    # Tight per-row height: this chart and the full ranking table share page one.
    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.185 * len(d) + 0.6)))
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
    # Blue marks a producer that is NOT a company (Brazil's garimpo gold), so it reads
    # as a different kind of thing rather than another miner.
    colours = [GREY if r["is_other"] else (BLUE if r.get("is_artisanal") else GOLD)
               for r in rows]
    ax.barh([r["company"][:42] for r in rows], [r["share_brazil"] for r in rows],
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
    """One commodity's page. `has_co` is False for the commodities whose Brazilian
    output has no published company-level split — those pages still run, and say so,
    rather than being dropped from the report."""
    out = {
        "label": com["label"], "icon": com["icon"],
        "brazil": f"{com['brazil']:,.2f}", "world": f"{com['world']:,.2f}",
        "unit": com["unit"], "share": f"{com['share']:.1f}", "rank": com.get("rank"),
        "year_label": com["year_label"], "source": com["source_label"],
        "n_producers": com.get("n_producers"), "com_note": com.get("note"),
        "country_chart": country_png(com), "has_co": False,
    }
    h = com.get("hedge") or {}
    out["hedge"] = {
        "available": bool(h.get("available")),
        "reason": h.get("reason", ""),
        "ticker": h.get("ticker", ""), "name": h.get("name", ""),
        "size": f"{h.get('size', 0):,}", "size_unit": h.get("size_unit", ""),
        "proxy": bool(h.get("proxy")), "note": h.get("note", ""),
        "basis_word": "exports" if h.get("qty_basis") == "exports" else "production",
        "national_qty": f"{h.get('national_qty', 0):,.2f}",
        "national_unit": h.get("national_unit", ""),
        "lots": f"{h.get('national_lots', 0):,}",
        "per_month": f"{h.get('national_lots_per_month', 0):,}",
        "by_company": {r["company"]: r for r in h.get("rows", [])},
    }
    blk = com.get("companies")
    if not blk:
        return out
    # Producers known, volumes not: name them, print no numbers, say why.
    if blk.get("unsourced"):
        out.update({"has_co": False, "unsourced": True,
                    "unsourced_reason": blk.get("reason", ""),
                    "producer_names": ", ".join(blk.get("names", [])),
                    "entity_label": blk.get("entity_label", "Producer")})
        return out
    out.update({
        "has_co": True,
        "basis_label": blk["basis_label"], "basis_note": blk["basis_note"],
        "confidence_label": blk["confidence_label"], "is_estimate": blk["confidence"] == "estimate",
        "is_export": blk["basis"] == "export",
        "co_year": blk.get("year"), "co_source": blk.get("source"), "note": blk.get("note"),
        "coverage": (None if blk.get("coverage_pct") is None else f"{blk['coverage_pct']:.0f}"),
        "company_chart": company_png(com),
        "rows": [{"company": r["company"],
                  "volume": ("" if blk.get("unit_is_pct") else f"{r['volume']:,.2f}"),
                  "share_brazil": f"{r['share_brazil']:.1f}",
                  "share_world": f"{r['share_world']:.2f}",
                  "ticker": r.get("ticker") or "—",
                  "lots": (f"{out['hedge']['by_company'][r['company']]['lots']:,}"
                           if r["company"] in out["hedge"]["by_company"] else ""),
                  "is_other": r["is_other"],
                  "is_artisanal": bool(r.get("is_artisanal"))} for r in blk["rows"]],
        "unit_is_pct": bool(blk.get("unit_is_pct")),
        "co_unit": blk.get("unit"),
        "entity_label": blk.get("entity_label") or "Company",
        "has_artisanal": bool(blk.get("has_artisanal")),
        "artisanal_share": f"{blk.get('artisanal_share', 0):.0f}",
        "top": blk["rows"][0]["company"] if blk["rows"] else "",
        "top_share": f"{blk['rows'][0]['share_brazil']:.0f}" if blk["rows"] else "",
    })
    return out


def render_html(store: dict, keys: list | None = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    coms = list((store.get("commodities") or {}).values())
    if not coms:
        raise SystemExit("no Brazil production store — run src/brazilprod.py first")

    head = sorted(coms, key=lambda c: -c["share"])
    # One page per commodity, in the same order as the front-page ranking.
    shown = [c for c in head if c["key"] in keys] if keys else list(head)
    dropped = [c for c in head if c not in shown]
    n_co = sum(1 for c in shown if c.get("companies"))

    top = head[0]
    lead = (f"Brazil is the world's largest producer of {top['label'].lower()} at "
            f"{top['share']:.0f}% of global output" if top.get("rank") == 1 else
            f"Brazil's largest share of any market here is {top['label'].lower()}, at "
            f"{top['share']:.0f}% of world output")

    # The hedge matrix: company x product, lots at each ratio broken into
    # year / month / trading day. Internal-facing brokerage sizing, so it prints as a
    # clearly-marked appendix rather than inside the client-facing commodity pages.
    mat = brazilprod.hedge_matrix(store, include_unhedgeable=True)
    matrix_rows, ratio_heads = [], []
    if not mat.empty:
        ratio_heads = [{"pct": p, "periods": [s for s, _d in brazilprod.HEDGE_PERIODS]}
                       for p in brazilprod.HEDGE_RATIOS]
        prev = None
        for _i, r in mat.iterrows():
            cells = []
            for p in brazilprod.HEDGE_RATIOS:
                for s, _d in brazilprod.HEDGE_PERIODS:
                    v = r[f"{p}% {s}"]
                    cells.append("—" if pd.isna(v) else f"{int(v):,}")
            matrix_rows.append({
                # blank the repeated company name so each client reads as one block
                "company": "" if r["Company"] == prev else r["Company"],
                "first": r["Company"] != prev,
                # the page can spell the qualifier out; at 5.6pt it has to be short
                "product": (r["Product"].replace("(share by processing)", "(processing)")
                            .replace("(share by cane crush)", "(cane crush)")
                            .replace("(equity share)", "(equity)")
                            .replace("(feed hedge)", "(feed)")),
                "volume": f"{r['Annual production']:,.2f} {r['Unit']}",
                "contract": r["Contract"], "cells": cells, "nohedge": not r["_avail"]})
            prev = r["Company"]

    tot = brazilprod.hedge_totals(mat) if not mat.empty else {}
    total_cells = [f"{tot.get(f'{p}% {s}', 0):,}"
                   for p in brazilprod.HEDGE_RATIOS
                   for s, _d in brazilprod.HEDGE_PERIODS] if tot else []

    return env.get_template("brazilreport.html").render(
        total_cells=total_cells, total_hedgeable=tot.get("_n_hedgeable", 0),
        periods=[s for s, _d in brazilprod.HEDGE_PERIODS],
        col_yr=brazilprod.PERIOD_COLOUR["yr"],
        col_mth=brazilprod.PERIOD_COLOUR["mth"],
        col_day=brazilprod.PERIOD_COLOUR["day"],
        hcol_yr=brazilprod.PERIOD_COLOUR_DARK["yr"],
        hcol_mth=brazilprod.PERIOD_COLOUR_DARK["mth"],
        hcol_day=brazilprod.PERIOD_COLOUR_DARK["day"],
        matrix_rows=matrix_rows, ratio_heads=ratio_heads,
        n_matrix_co=int(mat["Company"].nunique()) if not mat.empty else 0,
        trading_days=brazilprod.TRADING_DAYS,
        asof=pretty_date(store.get("built", "")[:10]),
        n_com=len(coms), n_blocks=len(shown), n_co=n_co, lead=lead,
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
