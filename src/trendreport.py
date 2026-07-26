"""Trend Report — a branded client PDF that, for each selected product, draws its price
with the MA20 / MA100 and the 3-month-return leg (the on-page Trend chart, for print) plus
the headline stats. Mirrors the Mean Reversion / vol / skew / term report style and shares
the Morning-Coffee house stylesheet via templates/_report_style.html.

Run standalone (the app calls it as a subprocess):
    python src/trendreport.py tickers.json out.pdf --asof "2026-06-18" --threshold 0
where tickers.json is a JSON list of Bloomberg tickers (universe instruments).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the `src` package importable (trend / datafeed use relative imports).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportkit import pretty_date, data_uri, png, RICH, CHEAP   # noqa: E402  (sets matplotlib Agg backend)
import matplotlib.pyplot as plt                     # noqa: E402
from jinja2 import Environment, FileSystemLoader    # noqa: E402

from src.strategies.trend import trend_chart_data   # noqa: E402
from src.datafeed import get_history                 # noqa: E402

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

PRICE = "#222222"   # the price line (primary)
MA_FAST = "#0B3D91"  # MA20 — navy
MA_SLOW = "#E67E22"  # MA100 — orange (the brand gold doesn't read on white)


def _price_png(cdata, info) -> str:
    """Price + MA20 + MA100, with the dashed 3-month-return leg coloured green (up) / red (down)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(cdata["date"], cdata["price"], lw=1.6, color=PRICE, label="Price")
    ax.plot(cdata["date"], cdata["fast"], lw=1.1, color=MA_FAST, label=f"MA{info['fast_w']}")
    ax.plot(cdata["date"], cdata["slow"], lw=1.1, color=MA_SLOW, label=f"MA{info['slow_w']}")
    dir_col = CHEAP if info["direction"] > 0 else RICH
    ax.plot([info["mom_date"], cdata["date"].iloc[-1]], [info["mom_price"], info["last"]],
            ls="--", lw=2.2, color=dir_col, label=f"3-month return {info['mom'] * 100:+.1f}%")
    ax.scatter([info["mom_date"], cdata["date"].iloc[-1]], [info["mom_price"], info["last"]],
               color=dir_col, s=32, zorder=5)
    ax.set_ylabel("Price")
    ax.set_title("Price with MA20 / MA100 — dashed leg = the 3-month return window")
    ax.grid(True, color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.8, ncol=2, loc="best", frameon=True, framealpha=0.9, edgecolor="#ccc")
    return png(fig)


def _signal(mom_pct: float, thr: float):
    if mom_pct != mom_pct:                       # NaN
        return "—", 0
    if mom_pct >= thr:
        return "Long", 1
    if mom_pct <= -thr:
        return "Short", -1
    return "—", 0


def render_html(tickers, asof, threshold) -> str:
    hist = get_history(list(tickers)) if tickers else None
    sections = []
    for tk in tickers:
        cdata, info = trend_chart_data(tk, hist)
        if cdata is None or cdata.empty:
            continue
        mom_pct = info["mom"] * 100.0
        sig, _ = _signal(mom_pct, threshold)
        sections.append({
            "name": info["name"], "signal": sig, "direction": info["direction"],
            "mom": f"{mom_pct:+.1f}", "ma_gap": f"{info['ma_gap']:+.1f}",
            "last": f"{info['last']:g}", "mixed": info["mixed"],
            "fast_w": info["fast_w"], "slow_w": info["slow_w"],
            "img": _price_png(cdata, info),
        })
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("trendreport.html").render(
        asof=pretty_date(asof), thr=f"{threshold:g}", sections=sections, n=len(sections),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(tickers, asof, threshold, out_pdf):
    from reportkit import render_pdf   # self-healing headless-Chromium renderer
    return render_pdf(render_html(tickers, asof, threshold), out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers_json")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()
    tickers = json.loads(Path(args.tickers_json).read_text(encoding="utf-8"))
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            tickers = [tk for tk in tickers if tk in _e]
    except Exception:
        pass
    build_pdf(tickers, args.asof, args.threshold, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
