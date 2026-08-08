"""Curve / RV Monitor report — a branded client PDF of the spread book from the
BASIS Curve / RV Monitor: every curve and relative-value spread with its rolling
z-score and ~10-year percentile, plus a chart panel for each stretched spread.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on
screen). Run standalone (the app calls it as a subprocess):
    python src/curvereport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, RICH, CHEAP
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
BLUE = "#1F5FA8"
GOLD = "#C8901A"


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def _fmt(v, dp: int) -> str:
    return "—" if v is None else f"{v:,.{dp}f}"


def spread_png(c: dict) -> str:
    """One spread panel: level (blue) over its rolling mean (gold dashes) and the
    ±threshold σ band, full stored depth."""
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in c["dates"]]
    fig, ax = plt.subplots(figsize=(6.1, 2.5))
    lo = [v for v in c["lower"]]
    hi = [v for v in c["upper"]]
    band_x = [d for d, a, b in zip(dates, lo, hi) if a is not None and b is not None]
    band_lo = [a for a, b in zip(lo, hi) if a is not None and b is not None]
    band_hi = [b for a, b in zip(lo, hi) if a is not None and b is not None]
    if band_x:
        ax.fill_between(band_x, band_lo, band_hi, color=GOLD, alpha=0.14, lw=0, zorder=1)
    ax.plot(dates, c["mean"], color=GOLD, lw=1.2, ls=(0, (5, 3)), zorder=2, label="rolling mean")
    ax.plot(dates, c["spread"], color=BLUE, lw=1.4, zorder=3, label="spread")
    last = next((v for v in reversed(c["spread"]) if v is not None), None)
    if last is not None:
        ax.axhline(last, color=BLACK, lw=0.6, alpha=0.35)
    ax.set_ylabel(c["unit"], fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return png(fig)


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    groups = []
    for g in d["groups"]:
        rows = []
        for r in [r for r in d["rows"] if r["group"] == g]:
            dp = int(r["dp"])
            rows.append({
                "name": r["name"],
                "level": f"{_fmt(r['level'], dp)} {r['unit']}",
                "chg": "—" if r["chg1d"] is None else f"{r['chg1d']:+,.{dp}f}",
                "z": f"{r['z']:+.2f}",
                "pctl": f"{r['pctl']:.0f}",
                "hl": "—" if r["half_life"] is None else f"{r['half_life']:.0f}d",
                "signal": r["signal"],
                "dir": r["direction"],
            })
        if rows:
            groups.append({"name": g, "rows": rows})

    charts = []
    for c in d["charts"]:
        info, dp = c["info"], int(c["dp"])
        dsig = ""
        if info.get("dollar_sigma"):
            dsig = f" (≈${info['dollar_sigma']:,.0f} per 1-lot spread)"
        charts.append({
            "name": c["name"], "desc": c["desc"], "img": spread_png(c),
            "stats": (f"{_fmt(info.get('level'), dp)} {c['unit']} · z {info.get('z', 0):+.2f} · "
                      f"{info.get('pctl', 0):.0f}th percentile of the stored decade · "
                      f"mean {_fmt(info.get('mean'), dp)} · "
                      f"±3σ reference {_fmt(info.get('invalidation'), dp)} · "
                      f"1σ = {_fmt(info.get('sigma'), dp)} {c['unit']}{dsig}"),
        })

    n = len(d["rows"])
    stretched = [r for r in d["rows"] if r["signal"] != "—"]
    first = min((r["first"] for r in d["rows"]), default="")
    return env.get_template("curvereport.html").render(
        asof=_fmt_asof(d["asof"]), window=d["window"], threshold=f"{d['threshold']:g}",
        n_spreads=n, n_stretched=len(stretched),
        stretched_names=", ".join(r["name"] for r in stretched) or "none at the threshold",
        since=first[:4], groups=groups, charts=charts,
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
