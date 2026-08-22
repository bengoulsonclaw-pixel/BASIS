"""Seasonality Monitor report — a branded client PDF of the BASIS Seasonality page:
the month screener across the book (how the screened month behaved over the
stored decade), a detail panel for the most seasonal names (year × month heatmap,
average-year path with the current year overlaid, the calendar windows the decade
rewarded most consistently) and the front-calendar-spread screener.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on
screen). Run standalone (the app calls it as a subprocess):
    python src/seasreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, RICH, CHEAP
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
BLUE = "#1F5FA8"
GOLD = "#C8901A"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def _sf(v, unit: str) -> str:
    """Signed seasonality number: bp whole, % one decimal; em-dash for missing."""
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+,.0f}" if unit == "bp" else f"{v:+,.1f}"


def heatmap_png(p: dict) -> str:
    """Year × month heatmap of the product's monthly changes — red/green diverging,
    clamped at the 90th percentile of |moves| (one outlier month can't wash it out),
    the in-progress month starred."""
    years = p["years"]
    mat = np.array([[np.nan if v is None else float(v) for v in row] for row in p["matrix"]],
                   dtype=float)
    vals = np.abs(mat[~np.isnan(mat)])
    vmax = float(np.percentile(vals, 90)) if vals.size else 1.0
    vmax = vmax or 1.0
    cmap = LinearSegmentedColormap.from_list("seas", [RICH, "#F4F4F4", CHEAP])
    h = max(1.6, 0.22 * len(years) + 0.6)
    fig, ax = plt.subplots(figsize=(6.3, h))
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax),
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS, fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(years))); ax.set_yticklabels([str(y) for y in years], fontsize=7)
    partial = p.get("partial")                      # [year, month] or None
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            if np.isnan(v):
                continue
            star = "*" if (partial and years[i] == partial[0] and j + 1 == partial[1]) else ""
            ax.text(j, i, _sf(v, p["unit"]) + star, ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > vmax * 0.55 else BLACK, family="monospace")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return png(fig)


def path_png(p: dict) -> str:
    """Average-year cumulative path: median of the stored years (blue) inside the
    25–75% band, the current year in gold."""
    sp = p["path"]
    wd = [datetime.fromordinal(datetime(2001, 1, 1).toordinal() + 7 * (int(w) - 1))
          for w in sp["woy"]]
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    p25 = [np.nan if v is None else v for v in sp["p25"]]
    p75 = [np.nan if v is None else v for v in sp["p75"]]
    ax.fill_between(wd, p25, p75, color="#9E9E9E", alpha=0.25, lw=0, label="25–75% of years")
    ax.plot(wd, [np.nan if v is None else v for v in sp["med"]], color=BLUE, lw=1.6,
            label=f"median of {p['path_years']} years")
    ax.plot(wd, [np.nan if v is None else v for v in sp["current"]], color=GOLD, lw=2.0,
            label=str(p.get("cur_year") or "current year"))
    ax.axhline(0, color="#999", lw=0.7, ls=(0, (4, 3)))
    ax.set_ylabel(f"cumulative {p['unit']}", fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
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
    month = d["month_label"]
    hit_strong = float(d.get("hit_strong", 0.7))

    # month screener, grouped by sector (already in the page's order)
    sectors = []
    for sec in d["sectors"]:
        rows = []
        for r in [r for r in d["screener"] if r["asset"] == sec]:
            u = r["unit"]
            rows.append({
                "name": r["name"], "med": _sf(r["med"], u), "mean": _sf(r["mean"], u),
                "hit": f"{round(r['hit'] * r['n'])}/{r['n']}", "bias": r["bias"],
                "best": _sf(r["best"], u), "worst": _sf(r["worst"], u),
                "this": _sf(r.get("this_year"), u) + ("*" if r.get("this_partial") else ""),
                "dir": 1 if r["bias"] == "↑" else (-1 if r["bias"] == "↓" else 0),
            })
        if rows:
            sectors.append({"name": sec, "unit": rows and [r for r in d["screener"]
                                                            if r["asset"] == sec][0]["unit"],
                            "rows": rows})

    strong = [r for r in d["screener"] if r["bias"] != "—"]
    strong.sort(key=lambda r: abs(r["hit"] - 0.5), reverse=True)
    bias_line = " · ".join(f"{r['name']} {r['bias']} {round(r['hit'] * r['n'])}/{r['n']}"
                           for r in strong[:8]) or "no product clears the agreement bar this month"

    # detail panels
    panels = []
    for p in d["products"]:
        u = p["unit"]
        st = p["stats"]                                  # {month: {med, hit, n, best, worst}}
        srow = st.get(str(d["month"])) or st.get(d["month"]) or {}
        comp = {int(k): v for k, v in st.items() if v.get("med") is not None}
        best_m = max(comp, key=lambda k: comp[k]["med"]) if comp else None
        worst_m = min(comp, key=lambda k: comp[k]["med"]) if comp else None
        wins = {"Higher": [], "Lower": []}
        for w in p.get("windows", []):
            wins.setdefault(w["dir"], []).append({
                "label": w["label"], "weeks": f"{int(w['weeks'])}w",
                "hit": f"{int(w['wins'])}/{int(w['n'])}", "med": _sf(w["med"], u),
                "worst": _sf(w["worst"], u)})
        panels.append({
            "name": p["name"], "ticker": p["ticker"], "unit": u,
            "heat": heatmap_png(p), "path": path_png(p) if p.get("path") else "",
            "kpi_month": (f"{_sf(srow.get('med'), u)} median · higher in "
                          f"{round(srow['hit'] * srow['n'])} of {int(srow['n'])} years"
                          if srow.get("n") else "—"),
            "kpi_best": (f"{MONTHS[best_m - 1]} {_sf(comp[best_m]['med'], u)}" if best_m else "—"),
            "kpi_worst": (f"{MONTHS[worst_m - 1]} {_sf(comp[worst_m]['med'], u)}" if worst_m else "—"),
            "years": len(p["years"]),               # years on the store (the heatmap's rows)
            "stats_row": [{"lab": MONTHS[m - 1],
                           "med": _sf((st.get(str(m)) or {}).get("med"), u),
                           "hit": (f"{round(v['hit'] * v['n'])}/{int(v['n'])}"
                                   if (v := (st.get(str(m)) or {})).get("n") else "—")}
                          for m in range(1, 13)],
            "higher": wins["Higher"], "lower": wins["Lower"],
        })

    # spread screener — the page shows the whole book; on paper keep the flagged
    # names (up to 20) or, on a quiet day, the 12 most stretched (payload is |z|-sorted)
    _all = list(d.get("spreads", []))
    _flag = [r for r in _all if str(r.get("signal", "—")) != "—"]
    spreads = []
    for r in (_flag[:20] if _flag else _all[:12]):
        spreads.append({
            "name": r["name"], "legs": r["legs"], "unit": r["unit"],
            "now": f"{r['now']:,.2f}", "norm": f"{r['norm']:,.2f}", "z": f"{r['z']:+.1f}",
            "strength": f"{r['strength']:.0f}%", "peak": r["peak"], "trough": r["trough"],
            "signal": ("Rich" if str(r["signal"]).startswith("Rich") else
                       "Cheap" if str(r["signal"]).startswith("Cheap") else "—"),
            "dir": 1 if r["signal"].startswith("Rich") else (-1 if r["signal"].startswith("Cheap") else 0),
        })

    return env.get_template("seasreport.html").render(
        asof=_fmt_asof(d["asof"]), month=month, hit_pct=f"{hit_strong:.0%}",
        n_products=len(d["screener"]), n_bias=len(strong), years=d.get("years", "~10"),
        n_sectors=len(sectors), bias_line=bias_line, sectors=sectors, panels=panels,
        spreads=spreads, n_spreads=len(spreads), n_spreads_all=len(_all),
        n_spread_flags=sum(1 for r in _all if str(r.get("signal", "—")) != "—"),
        seas_z=f"{float(d.get('seas_z_flag', 1.5)):g}",
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
