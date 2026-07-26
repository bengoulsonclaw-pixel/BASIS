"""Mean Reversion Report — a branded client PDF that, for each selected pair, shows
the spread vs its 90-day mean ±trigger·σ band (the "how stretched is it" chart) and the
two legs rebased to 100, plus the key stats. Mirrors the vol/skew/term report style.

Run standalone (the app calls it as a subprocess):
    python src/mrreport.py pairs.json out.pdf --asof "2026-06-05" --threshold 1.5
where pairs.json is a JSON list of pair names (matching universe.PAIRS[*]["name"]).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the `src` package importable (mean_reversion / datafeed use relative imports).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportkit import pretty_date, data_uri, png, RICH   # noqa: E402  (sets matplotlib Agg backend)
import matplotlib.pyplot as plt              # noqa: E402
import pandas as pd                          # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from src.strategies.mean_reversion import pair_chart_data  # noqa: E402
from src.universe import PAIRS                              # noqa: E402

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"
_BY_NAME = {p["name"]: p for p in PAIRS}

NAVY = "#0B3D91"
LEGB = "#E67E22"   # second leg — orange reads on white (the yellow brand colour doesn't)


def _spread_png(cdata: pd.DataFrame, info: dict) -> str:
    d = cdata.dropna(subset=["spread"])
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.fill_between(d["date"], d["lower"], d["upper"], color=NAVY, alpha=0.13, linewidth=0,
                    label=f"90-day mean ± {info['threshold']:g}σ")
    ax.plot(d["date"], d["mean"], ls="--", lw=1.0, color="#8A8A8A", label="90-day mean")
    ax.plot(d["date"], d["spread"], lw=1.4, color=NAVY, label="Spread")
    last = d.iloc[-1]
    ax.scatter([last["date"]], [last["spread"]], color=RICH, s=45, zorder=5, label="Today")
    ax.set_ylabel("ratio (A ÷ B)" if info["kind"] == "ratio" else "spread (A − B)")
    ax.set_title("Spread vs its 90-day mean band — outside the band = stretched")
    ax.grid(True, color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, ncol=2, loc="best", frameon=True, framealpha=0.9, edgecolor="#ccc")
    return png(fig)


def _legs_png(cdata: pd.DataFrame, info: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    ax.plot(cdata["date"], cdata["a_idx"], lw=1.3, color=NAVY, label=info["a_name"])
    ax.plot(cdata["date"], cdata["b_idx"], lw=1.3, color=LEGB, label=info["b_name"])
    ax.set_ylabel("Rebased to 100")
    ax.set_title("The two legs, rebased to 100")
    ax.grid(True, color="#ECECEC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9, edgecolor="#ccc")
    return png(fig)


def _signal(z: float, thr: float):
    if z != z:                       # NaN
        return "—", 0
    if z >= thr:
        return "Short spread", -1
    if z <= -thr:
        return "Long spread", 1
    return "—", 0


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if v == v else "—"     # v==v is False for NaN


def render_html(pairs, asof, threshold) -> str:
    sections = []
    for nm in pairs:
        p = _BY_NAME.get(nm)
        if not p:
            continue
        cdata, info = pair_chart_data(p, threshold)
        if cdata is None or cdata.empty:
            continue
        sig, direction = _signal(info["z"], info["threshold"])
        sections.append({
            "name": nm, "signal": sig, "direction": direction,
            "z": f"{info['z']:+.2f}" if info["z"] == info["z"] else "—",
            "pctl": _fmt(info["pctl"], 0),
            "half_life": (f"{info['half_life']:.0f}d" if info["half_life"] == info["half_life"] else "—"),
            "spread": _fmt(info["spread"], 3), "mean": _fmt(info["mean"], 3),
            "kind": "ratio (A ÷ B)" if info["kind"] == "ratio" else "differential (A − B)",
            "spread_img": _spread_png(cdata, info),
            "legs_img": _legs_png(cdata, info),
        })
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("mrreport.html").render(
        asof=pretty_date(asof), zflag=f"{threshold:g}", sections=sections, n=len(sections),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(pairs, asof, threshold, out_pdf):
    html = render_html(pairs, asof, threshold)
    from playwright.sync_api import sync_playwright
    from reportkit import launch_chromium
    with sync_playwright() as pw:
        browser = launch_chromium(pw)               # bundled Chromium, else the machine's Edge/Chrome
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(path=str(out_pdf), format="A4", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()
    return out_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_json")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()
    pairs = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            pairs = [nm for nm in pairs
                     if (_BY_NAME.get(nm) or {}).get("a") in _e
                     and (_BY_NAME.get(nm) or {}).get("b") in _e]
    except Exception:
        pass
    build_pdf(pairs, args.asof, args.threshold, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
