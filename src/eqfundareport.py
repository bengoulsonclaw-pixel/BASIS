"""Company Fundamentals tearsheet — a branded PDF of one company's fundamentals exactly as
shown on the Company Fundamentals page: the four metric groups (valuation / profitability /
leverage / growth / income) with GICS-sector medians and percentiles, plus a percentile chart.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on screen).
Run standalone (the app calls it as a subprocess):
    python src/eqfundareport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import data_uri, png, render_pdf, RICH, CHEAP, NEUTRAL, BLACK
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"


def pctl_png(groups: list) -> str:
    """Every metric's sector percentile as one horizontal bar chart, in group order — green
    where the stock sits at the good end of its sector, red at the poor end, grey mid-pack
    or direction-less. The gold line is the sector median (50th)."""
    rows = [(r["label"], r["pctl"], r["good"]) for g in groups for r in g["rows"]
            if r.get("pctl") is not None]
    if not rows:
        return ""
    labels = [r[0] for r in rows][::-1]           # barh draws bottom-up; keep page order top-down
    vals = [r[1] for r in rows][::-1]
    cols = [CHEAP if r[2] > 0 else RICH if r[2] < 0 else NEUTRAL for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(6.1, max(2.6, 0.21 * len(rows))))
    ax.barh(range(len(rows)), vals, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(50, color=GOLD, lw=1.2, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 50, 80, 100])
    ax.set_xlabel("percentile within GICS sector")
    ax.set_title("Sector percentiles — all metrics")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    groups = [{
        "name": g["name"],
        "rows": [{
            "label": r["label"], "value": r["value"], "median": r["median"],
            "pctl": r["pctl_txt"],
            "good": r["good"] > 0, "poor": r["good"] < 0,
        } for r in g["rows"]],
    } for g in d.get("groups", [])]
    return env.get_template("eqfundareport.html").render(
        asof=date.fromisoformat(d["asof"]).strftime("%d %b %Y"),
        name=d["name"], ticker=d["ticker"], sector=d["sector"], region=d.get("region", ""),
        indices=d.get("indices", ""), mktcap=d.get("mktcap", "—"), crncy=d.get("crncy", ""),
        next_report=d.get("next_report", "—"), n_peers=d.get("n_peers", 0), mode=d.get("mode", ""),
        groups=groups, pctl_chart=pctl_png(d.get("groups", [])),
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
