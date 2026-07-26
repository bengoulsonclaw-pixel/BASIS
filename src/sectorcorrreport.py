"""Product Correlations report — a branded PDF of one Product Correlations view:
the 1-year and 1-month correlation matrices for the chosen sectors' products,
the 1M − 1Y regime-shift map, the cross-sector diversification index and the
biggest correlation breaks across the whole book.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on
screen). Run standalone (the app calls it as a subprocess):
    python src/sectorcorrreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, NEUTRAL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"
ANNOTATE_MAX = 14        # past this many products the cell numbers become confetti


def _arr(m: list) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in m], dtype=float)


def heat_png(labels: list, m: list, span: float = 1.0) -> str:
    """One correlation matrix as a red/blue heatmap (blue −, red +, house-white
    for empty cells), annotated when it stays readable."""
    n = len(labels)
    arr = _arr(m)
    # height scales with the matrix but caps below a page, so a whole-book
    # selection squeezes rather than overflowing the print layout
    fig, ax = plt.subplots(figsize=(6.1, min(8.4, max(2.1, 0.7 + 0.24 * n))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(arr, cmap=cmap, vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=1.1)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if n <= ANNOTATE_MAX:
        for i in range(n):
            for j in range(n):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.6,
                            color="#FFFFFF" if abs(v) > 0.55 * span else "#333333")
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cb.ax.tick_params(labelsize=5.6, length=0)
    cb.outline.set_visible(False)
    return png(fig)


def div_png(d: dict) -> str:
    """The diversification index — average rolling 1M correlation across every
    cross-sector composite pair, signed and absolute, with the year's mean."""
    dates = [date.fromisoformat(x) for x in d["div_dates"]]
    fig, ax = plt.subplots(figsize=(6.1, 2.0))
    ax.plot(dates, d["div_avg"], color=BLACK, lw=1.6, label="Average (signed)")
    ax.plot(dates, d["div_abs"], color=GOLD, lw=1.8, label="Average |corr|")
    ax.axhline(float(np.mean(d["div_avg"])), color=NEUTRAL, lw=1.0, ls=(0, (5, 3)),
               label="Signed 1Y mean")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_ylabel("avg pairwise correlation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    rows = [{
        "pair": b["pair"], "sectors": b["sectors"],
        "c1y": f"{b['c1y']:+.2f}", "c1m": f"{b['c1m']:+.2f}", "d": f"{b['d']:+.2f}",
        "pctl": "—" if b.get("pctl") is None else f"{b['pctl']:.0f}%",
        "extreme": b.get("pctl") is not None and (b["pctl"] <= 5 or b["pctl"] >= 95),
    } for b in d.get("breaks", [])]
    span = d.get("diff_span", 0.5)
    return env.get_template("sectorcorrreport.html").render(
        asof=pretty_date(d["asof"]),
        sectors=" · ".join(d["sectors"]), metric=d["metric_label"], mode=d["mode"],
        n_products=len(d["labels"]),
        heat_1y=heat_png(d["labels"], d["m1y"]),
        heat_1m=heat_png(d["labels"], d["m1m"]),
        heat_diff=heat_png(d["labels"], d["diff"], span=span),
        div_chart=div_png(d) if d.get("div_dates") else "",
        rows=rows,
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_path) -> str:
    return render_pdf(render_html(d), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload_json")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    d = json.loads(Path(args.payload_json).read_text())
    build_pdf(d, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
