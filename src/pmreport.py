"""pmreport.py — Precious Metals Fundamentals Monitor PDF (house style).

Consumes the JSON built by pmdata.build() and renders the 4-page monthly
client PDF: macro & positioning overview, then gold / silver / PGM pages.
Every chart guards on its data (returns None) so a sparse month still renders.

CLI:  python src/pmreport.py data/pm_monitor.json data/Precious_Metals_Report.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reportkit import data_uri, png, render_pdf  # noqa: E402  (sets Agg first)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

# Same palette discipline as opecreport: gold for fills/accents, lines in ink/navy.
INK, GOLD, GOLDEDGE = "#1A1A1A", "#F5C518", "#B8860B"
GREY, NAVY = "#9AA0A8", "#0B3D91"
GREEN, RED = "#2E7D32", "#C62828"
STACK = [GOLD, NAVY, GREY, GREEN, RED, "#D5D8DC"]

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold"})

METAL_NAMES = {"gold": "Gold", "silver": "Silver", "platinum": "Platinum", "palladium": "Palladium"}


def _despine(ax, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.grid(True, axis="y", color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)


def _s(blob) -> pd.Series:
    """{d, v} JSON blob → date-indexed series (empty-safe)."""
    if not blob or not blob.get("d"):
        return pd.Series(dtype=float)
    return pd.Series(blob["v"], index=pd.to_datetime(blob["d"]))


# --- charts -----------------------------------------------------------------
def chart_cot_panel(cot: dict) -> str | None:
    """2×2 small multiples: COT index per metal with crowded-long/short bands."""
    keys = [k for k in METAL_NAMES if not _s(cot.get(k, {}).get("idx")).empty]
    if not keys:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 2.9), sharey=True)
    for ax, k in zip(axes.flat, keys):
        s = _s(cot[k]["idx"])
        ax.axhspan(80, 100, color="#FDECEA", zorder=1)
        ax.axhspan(0, 20, color="#EAF4EB", zorder=1)
        ax.plot(s.index, s.values, color=INK, lw=1.0, zorder=3)
        ax.plot(s.index[-1], s.values[-1], "o", ms=4, color=GOLD,
                mec=GOLDEDGE, mew=0.7, zorder=4)
        ax.set_title(METAL_NAMES[k], fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=6.5)
        _despine(ax)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.text(0.005, 0.5, "COT index (0–100)", rotation=90, va="center", fontsize=7)
    fig.tight_layout(pad=0.8)
    return png(fig)


def chart_real_yield(gold: dict, ry: dict) -> str | None:
    g, r = _s(gold), _s(ry)
    if g.empty or r.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    ax.plot(g.index, g.values, color=NAVY, lw=1.2, label="Gold ($/oz, lhs)")
    ax2 = ax.twinx()
    ax2.plot(r.index, r.values, color=INK, lw=1.0, ls="--", label="10y real yield (%, rhs, inverted)")
    ax2.invert_yaxis()
    ax.set_title("Gold vs US 10y TIPS real yield")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6.8, loc="upper left",
              frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    ax2.tick_params(labelsize=6.8)
    ax2.grid(False)
    _despine(ax)
    ax2.spines["top"].set_visible(False)
    return png(fig)


def chart_swiss(sw: dict) -> str | None:
    if not sw or not sw.get("d"):
        return None
    idx = pd.to_datetime(sw["d"])
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    bottom = [0.0] * len(idx)
    for dest, row, col in zip(sw["dests"], sw["rows"], STACK):
        ax.bar(idx, row, bottom=bottom, width=20, color=col,
               edgecolor="white", linewidth=0.4, label=dest, zorder=3)
        bottom = [b + v for b, v in zip(bottom, row)]
    ax.set_ylabel("tonnes / month")
    ax.set_title("Swiss gold exports by destination")
    ax.legend(fontsize=6.4, ncol=6, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_cb(cb: dict) -> str | None:
    s = _s(cb)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    ax.bar(s.index, s.values, width=20, color=GOLD, edgecolor=GOLDEDGE,
           linewidth=0.5, zorder=3, label="Monthly net purchases")
    ax.plot(s.index, s.rolling(12, min_periods=6).mean().values, color=INK,
            lw=1.1, zorder=4, label="12m average")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("tonnes")
    ax.set_title("Central-bank net gold purchases (reported)")
    ax.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_premiums(sge: dict, india: dict) -> str | None:
    a, b = _s(sge), _s(india)
    if a.empty and b.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    if not a.empty:
        ax.plot(a.index, a.values, color=NAVY, lw=1.1, label="Shanghai (SGE vs London)")
    if not b.empty:
        ax.plot(b.index, b.values, color=GREY, lw=1.1, label="India (local vs landed)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("$/oz")
    ax.set_title("Physical gold premium / discount vs London")
    ax.legend(fontsize=6.8, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_etf(etf: dict, key: str) -> str | None:
    blk = (etf or {}).get(key)
    if not blk:
        return None
    s = _s(blk["s"])
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    ax.fill_between(s.index, s.values, color="#FDF3CF", zorder=2)
    ax.plot(s.index, s.values, color=NAVY, lw=1.2, zorder=3)
    ax.set_ylabel(blk["unit"])
    ax.set_title(f"{METAL_NAMES[key]} ETF holdings — total known")
    ax.set_ylim(s.min() * 0.97, s.max() * 1.03)
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_ratio(ratio: dict) -> str | None:
    s = _s(ratio)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    ax.plot(s.index, s.values, color=INK, lw=1.1, label="Gold / silver ratio")
    ax.axhline(s.mean(), color=GREY, lw=1.0, ls="--", label=f"Period mean {s.mean():.0f}")
    ax.plot(s.index[-1], s.values[-1], "o", ms=4, color=GOLD, mec=GOLDEDGE, mew=0.7, zorder=4)
    ax.set_title("Gold / silver ratio")
    ax.legend(fontsize=6.8, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_mint(gold_oz: dict, silver_oz: dict) -> str | None:
    g, s = _s(gold_oz), _s(silver_oz)
    if g.empty and s.empty:
        return None
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.9))
    for ax, series, label, scale in ((axes[0], s, "Silver eagles (Moz)", 1e6),
                                     (axes[1], g, "Gold coins (koz)", 1e3)):
        if series.empty:
            ax.axis("off")
            continue
        ax.bar(series.index, series.values / scale, width=20, color=GOLD,
               edgecolor=GOLDEDGE, linewidth=0.5, zorder=3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax.set_title(label, fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        _despine(ax)
    fig.suptitle("US Mint bullion sales — monthly", fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)
    return png(fig)


def chart_pgm_balance(pt: dict, pd_: dict) -> str | None:
    if not pt and not pd_:
        return None
    years = sorted(set(pt or {}) | set(pd_ or {}))
    fig, ax = plt.subplots(figsize=(6.9, 1.7))
    x = range(len(years))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [(pt or {}).get(y, 0) for y in years], width=w,
           color=GOLD, edgecolor=GOLDEDGE, linewidth=0.6, label="Platinum", zorder=3)
    ax.bar([i + w / 2 for i in x], [(pd_ or {}).get(y, 0) for y in years], width=w,
           color=NAVY, edgecolor="white", linewidth=0.6, label="Palladium", zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_ylabel("koz (deficit < 0)")
    ax.set_title("Published market balances by year")
    ax.legend(fontsize=6.8, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_autos(autos: dict) -> str | None:
    s = _s(autos)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 1.7))
    ax.bar(s.index, s.values, width=55, color=GREY, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_ylabel("Mn units / qtr")
    ax.set_title("World light-vehicle production")
    ax.set_ylim(0, s.max() * 1.15)
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


def chart_pgm_stocks(comex: dict) -> str | None:
    pt = _s((comex or {}).get("platinum", {}).get("reg"))
    pa = _s((comex or {}).get("palladium", {}).get("reg"))
    # the live CME archive builds one snapshot per fetch — a 1-2 point "line"
    # reads as broken, so hold the chart back until there is a little history
    if len(pt) < 3 and len(pa) < 3:
        return None
    unit = (comex or {}).get("platinum", {}).get("unit", "koz")
    fig, ax = plt.subplots(figsize=(6.9, 1.7))
    if not pt.empty:
        ax.plot(pt.index, pt.values, color=NAVY, lw=1.2, label="Platinum")
    if not pa.empty:
        ax.plot(pa.index, pa.values, color=GREY, lw=1.2, label="Palladium")
    ax.set_ylabel(f"{unit} registered")
    ax.set_title("NYMEX registered stocks")
    ax.legend(fontsize=6.8, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#ccc")
    ax.tick_params(labelsize=6.8)
    _despine(ax)
    return png(fig)


# --- render ------------------------------------------------------------------
def render_html(d: dict) -> str:
    ser = d.get("series", {})
    charts = {
        "cot_panel": chart_cot_panel(ser.get("cot", {})),
        "real_yield": chart_real_yield(ser.get("gold"), ser.get("real_yield")),
        "swiss": chart_swiss(ser.get("swiss")),
        "cb": chart_cb(ser.get("cb")),
        "premiums": chart_premiums(ser.get("sge"), ser.get("india")),
        "etf_silver": chart_etf(ser.get("etf"), "silver"),
        "ratio": chart_ratio(ser.get("ratio")),
        "mint": chart_mint(ser.get("mint_gold"), ser.get("mint_silver")),
        "pgm_balance": chart_pgm_balance(ser.get("pt_balance"), ser.get("pd_balance")),
        "autos": chart_autos(ser.get("autos")),
        "pgm_stocks": chart_pgm_stocks(ser.get("comex")),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    return env.get_template("pmreport.html").render(
        d=d, charts=charts,
        logo=data_uri(ASSETS / "logo.png"),
        watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_pdf: Path) -> Path:
    render_pdf(render_html(d), out_pdf)
    return out_pdf


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: pmreport.py <pm_monitor.json> <out.pdf>")
        return 2
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = build_pdf(d, Path(sys.argv[2]))
    print(f"pmreport: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
