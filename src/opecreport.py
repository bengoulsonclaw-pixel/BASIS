"""OPEC Monthly Oil Market Report — synopsis + chart deck (branded BASIS PDF).

Page 1 is a one-page synopsis of the latest MOMR (demand/supply balance lead +
neutral price/futures observations); the following pages are a chart deck built
from the same JSON. Mirrors the house style of cotreport — same inset
black sidebar, banners, compliance disclaimer (_disclaimer.html) — via _report_style.html.

The monthly content is supplied as a JSON file (filled from the primary MOMR PDF
each release); every chart guards on its data, so a sparse month still renders.

Run standalone (so Playwright stays off Streamlit's event loop):
    python src/opecreport.py synopsis.json out.pdf
where synopsis.json carries: month, asof, headline, balance_rows, implications,
changes, demand_by_year, supply_by_year, supply_contributors, doc_production,
balance_quarters, basket_history.  See opec/sample_synopsis.json for the shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportkit import data_uri, png            # noqa: E402  (sets matplotlib Agg backend)
import matplotlib.pyplot as plt                # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

# Chart palette — gold reads as a filled bar on white but not as a thin line, so
# lines use ink/navy and gold is reserved for fills, accents and the "latest" marker.
INK, GOLD, GOLDEDGE = "#1A1A1A", "#F5C518", "#B8860B"
GREY, NAVY = "#9AA0A8", "#0B3D91"
GREEN, RED = "#2E7D32", "#C62828"

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold"})


def _despine(ax, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.grid(True, axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --- charts -----------------------------------------------------------------
def chart_balance(demand, supply) -> str | None:
    """Hero chart: world oil demand growth vs non-DoC liquids supply growth, by
    year. The gap between the two bars is the year's tightening/loosening — when
    demand growth out-runs non-DoC supply growth, the call on DoC crude rises."""
    if not demand:
        return None
    years = [d.get("year", "") for d in demand]
    dem = [_num(d.get("total")) for d in demand]
    sup_by = {s.get("year"): _num(s.get("nondoc")) for s in (supply or [])}
    sup = [sup_by.get(y) for y in years]
    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    x = range(len(years))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [d or 0 for d in dem], width=w, color=GOLD,
           edgecolor=GOLDEDGE, linewidth=0.6, label="World oil demand growth", zorder=3)
    ax.bar([i + w / 2 for i in x], [s or 0 for s in sup], width=w, color=NAVY,
           edgecolor="white", linewidth=0.6, label="Non-DoC liquids supply growth", zorder=3)
    for i, (d, s) in enumerate(zip(dem, sup)):
        if d is not None:
            ax.text(i - w / 2, d + 0.02, f"{d:+.2f}", ha="center", va="bottom", fontsize=6.4, color="#333")
        if s is not None:
            ax.text(i + w / 2, s + 0.02, f"{s:+.2f}", ha="center", va="bottom", fontsize=6.4, color="#333")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_ylabel("y/y change (mb/d)")
    ax.set_title("Demand vs non-DoC supply growth — the call on DoC crude")
    ax.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")
    _despine(ax)
    return png(fig)


def chart_demand_split(demand) -> str | None:
    """Stacked OECD / non-OECD contribution to world demand growth by year."""
    rows = [d for d in demand or [] if _num(d.get("oecd")) is not None or _num(d.get("nonoecd")) is not None]
    if not rows:
        return None
    years = [d.get("year", "") for d in rows]
    oecd = [_num(d.get("oecd")) or 0 for d in rows]
    nono = [_num(d.get("nonoecd")) or 0 for d in rows]
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    x = range(len(years))
    ax.bar(x, oecd, color=GREY, edgecolor="white", linewidth=0.6, label="OECD", zorder=3)
    ax.bar(x, nono, bottom=oecd, color=GOLD, edgecolor=GOLDEDGE, linewidth=0.6, label="Non-OECD", zorder=3)
    for i, (a, b) in enumerate(zip(oecd, nono)):
        ax.text(i, a + b + 0.02, f"{a + b:+.2f}", ha="center", va="bottom", fontsize=6.6, color="#333")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_ylabel("y/y change (mb/d)")
    ax.set_title("Where the demand growth comes from — OECD vs non-OECD")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")
    _despine(ax)
    return png(fig)


def chart_supply_contrib(contrib) -> str | None:
    """Horizontal bars: non-DoC liquids supply growth by contributing country."""
    rows = [c for c in contrib or [] if _num(c.get("value")) is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda c: _num(c.get("value")))
    names = [c.get("name", "") for c in rows]
    vals = [_num(c.get("value")) for c in rows]
    fig, ax = plt.subplots(figsize=(6.9, max(1.5, 0.3 * len(rows) + 0.55)))
    y = range(len(rows))
    ax.barh(list(y), vals, color=[GREEN if v and v >= 0 else RED for v in vals],
            edgecolor="white", linewidth=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.6, color="#333")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("y/y change (mb/d)")
    ax.set_title("Non-DoC supply growth — leading contributors")
    span = max((abs(v) for v in vals if v is not None), default=1.0) or 1.0
    ax.set_xlim(-span * 1.3, span * 1.3)
    ax.grid(True, axis="x", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def chart_doc_production(prod) -> str | None:
    """OPEC+/DoC crude production by country (secondary sources), latest month,
    with the month-over-month change annotated (red down / green up)."""
    rows = [p for p in prod or [] if _num(p.get("value")) is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda p: _num(p.get("value")))
    names = [p.get("country", "") for p in rows]
    vals = [_num(p.get("value")) for p in rows]
    moms = [_num(p.get("mom")) for p in rows]
    fig, ax = plt.subplots(figsize=(6.9, max(1.4, 0.2 * len(rows) + 0.45)))
    y = range(len(rows))
    ax.barh(list(y), vals, color=GOLD, edgecolor=GOLDEDGE, linewidth=0.5, zorder=3)
    for i, (v, m) in enumerate(zip(vals, moms)):
        lbl = f"{v:.2f}"
        if m is not None:
            lbl += f"  ({m:+.2f})"
        ax.text(v + 0.05, i, lbl, va="center", ha="left", fontsize=6.4,
                color=(GREEN if (m or 0) > 0 else RED if (m or 0) < 0 else "#333"))
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("crude production (mb/d) — value, and (m/m change)")
    ax.set_title("DoC crude production by country — secondary sources")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.grid(True, axis="x", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def chart_balance_quarters(rows, prod_ref=None, prod_ref_label="") -> str | None:
    """Required DoC crude (the "call") by period, as bars, with an optional dashed
    reference line at current DoC crude output. The gap between each bar and the line
    is the implied stock draw the market needs DoC producers to cover (call > output)
    — or a build if output sits above the call. Each row is {q, call}; rows may also
    carry an explicit {prod} to draw an actual-production bar alongside the call."""
    rows = [r for r in rows or [] if _num(r.get("call")) is not None]
    if not rows:
        return None
    labels = [r.get("q", "") for r in rows]
    call = [_num(r.get("call")) for r in rows]
    prod = [_num(r.get("prod")) for r in rows]
    ref = _num(prod_ref)
    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    x = range(len(rows))
    has_prod = any(p is not None for p in prod)
    w = 0.40 if has_prod else 0.62
    ax.bar([i - (w / 2 if has_prod else 0) for i in x], call, width=w, color=NAVY,
           edgecolor="white", linewidth=0.6, label="Required DoC crude (call)", zorder=3)
    for i, c in enumerate(call):
        ax.text(i - (w / 2 if has_prod else 0), c + 0.1, f"{c:.1f}", ha="center",
                va="bottom", fontsize=6.2, color="#333")
    if has_prod:
        px = [i + w / 2 for i in x]
        ax.bar([px[i] for i, p in enumerate(prod) if p is not None],
               [p for p in prod if p is not None], width=w, color=GOLD,
               edgecolor=GOLDEDGE, linewidth=0.6, label="DoC crude production", zorder=3)
        for i, (c, p) in enumerate(zip(call, prod)):
            if p is not None:
                ax.annotate(f"draw {c - p:.1f}", (i, max(c, p)), textcoords="offset points",
                            xytext=(8, 4), ha="center", fontsize=6, color=GREEN)
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1.2, color=RED, zorder=4)
        ax.text(len(rows) - 0.5, ref, f" {prod_ref_label or 'current output'} {ref:.1f}",
                va="bottom", ha="right", fontsize=6.4, color=RED, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mb/d")
    ax.set_title("Required DoC crude vs current output — the gap is the implied draw")
    ax.legend(fontsize=6.6, loc="lower left", frameon=True, framealpha=0.9, edgecolor="#ccc")
    base = [v for v in call + prod + [ref] if v is not None]
    ax.set_ylim(min(base) - 1.5, max(call) + 1.6)
    _despine(ax)
    return png(fig)


def _mlabel(ym: str) -> str:
    try:
        y, m = ym.split("-")
        return f"{['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(m)]}-{y[2:]}"
    except Exception:
        return ym


def chart_eia_prices(prices, split=None) -> str | None:
    """EIA Brent & WTI monthly spot ($/b): solid up to the report month, dashed beyond
    (STEO forecast). Shows the independent price level and the forecaster's forward path."""
    rows = [p for p in prices or [] if _num(p.get("brent")) is not None]
    if len(rows) < 3:
        return None
    months = [p["m"] for p in rows]
    brent = [_num(p.get("brent")) for p in rows]
    wti = [_num(p.get("wti")) for p in rows]
    cut = next((i for i, m in enumerate(months) if split and m >= split), len(months) - 1)
    fig, ax = plt.subplots(figsize=(6.9, 2.0))
    x = list(range(len(rows)))
    for series, col, lbl in ((brent, NAVY, "Brent"), (wti, "#E67E22", "WTI")):
        ax.plot(x[:cut + 1], series[:cut + 1], lw=1.6, color=col, label=lbl, zorder=3)
        ax.plot(x[cut:], series[cut:], lw=1.5, color=col, ls="--", zorder=3)
    if 0 < cut < len(rows) - 1:
        ax.axvspan(cut, len(rows) - 1, color="#F5C518", alpha=0.08, zorder=0)
        ax.axvline(cut, color="#999", lw=0.8, ls=":")
        ax.text(cut + 0.1, ax.get_ylim()[1], " STEO forecast", va="top", ha="left",
                fontsize=6.2, color="#777")
    step = max(1, len(rows) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([_mlabel(months[i]) for i in x[::step]], fontsize=6.3, rotation=0)
    ax.set_ylabel("$/bbl")
    ax.set_title("EIA Brent & WTI — monthly spot (dashed = STEO forecast)")
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#ccc")
    _despine(ax)
    return png(fig)


def chart_basket(hist) -> str | None:
    """OPEC Reference Basket price ($/b) as bars — handles a mix of annual averages
    and recent months (each {m, v}); the latest bar is highlighted in gold."""
    rows = [h for h in hist or [] if _num(h.get("v")) is not None]
    if len(rows) < 2:
        return None
    months = [h.get("m", "") for h in rows]
    vals = [_num(h.get("v")) for h in rows]
    colors = [NAVY] * (len(rows) - 1) + [GOLD]
    edges = ["white"] * (len(rows) - 1) + [GOLDEDGE]
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    x = range(len(rows))
    ax.bar(list(x), vals, color=colors, edgecolor=edges, linewidth=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.012, f"${v:.2f}", ha="center", va="bottom",
                fontsize=7, fontweight=("bold" if i == len(rows) - 1 else "normal"), color="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, fontsize=7)
    ax.set_ylabel("$/bbl")
    ax.set_ylim(0, max(vals) * 1.14)
    ax.set_title("OPEC Reference Basket — annual averages and latest months")
    _despine(ax)
    return png(fig)


# --- render -----------------------------------------------------------------
def render_html(d: dict) -> str:
    charts = {
        "balance": chart_balance(d.get("demand_by_year"), d.get("supply_by_year")),
        "demand_split": chart_demand_split(d.get("demand_by_year")),
        "supply_contrib": chart_supply_contrib(d.get("supply_contributors")),
        "doc_production": chart_doc_production(d.get("doc_production")),
        "balance_quarters": chart_balance_quarters(
            d.get("balance_quarters"), d.get("doc_prod_current"),
            d.get("doc_prod_current_label", "DoC output (May)")),
        "basket": chart_basket(d.get("basket_history")),
        "eia_prices": chart_eia_prices(d.get("eia_prices"), d.get("eia_price_split")),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("opecreport.html").render(
        d=d, charts=charts,
        logo=data_uri(ASSETS / "logo.png"),
        watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_pdf) -> str:
    from reportkit import render_pdf
    return render_pdf(render_html(d), out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synopsis_json")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    d = json.loads(Path(args.synopsis_json).read_text(encoding="utf-8"))
    build_pdf(d, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
