"""MA Crossover — per-product scorecard PDF for the desk.

Renders the risk-adjusted product comparison (from pnl_sim.py's CSV) on the
Morning-Coffee house style: a ranked Sharpe bar for every product, a by-asset
summary, best/worst tables, and the methodology + the key 'diversification, not
per-product skill' finding.

    python src/scorecard_report.py --csv data/signals/ma_product_scorecard.csv \
        --out data/MA_Product_Scorecard.pdf --variant "MA Swing (20/50)" \
        --asof "2016-10-17 -> 2026-06-12" --years 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from reportkit import BLACK, CHEAP, RICH, data_uri, png, render_pdf
import matplotlib.pyplot as plt

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def _dir(v) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def ranked_bar_png(res: pd.DataFrame) -> str:
    d = res.sort_values("sharpe")                       # barh draws bottom-up → best on top
    fig, ax = plt.subplots(figsize=(6.0, max(4.0, 0.165 * len(d))))
    y = range(len(d))
    ax.barh(list(y), d["sharpe"], color=[CHEAP if v >= 0 else RICH for v in d["sharpe"]],
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(y)); ax.set_yticklabels(d["market"], fontsize=5.4)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Sharpe (annualised, per product)")
    ax.set_title("Risk-adjusted performance by product")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def asset_bar_png(res: pd.DataFrame) -> str:
    a = res.groupby("asset")["sharpe"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.barh(range(len(a)), a.to_numpy(), color=[CHEAP if v >= 0 else RICH for v in a],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a.index, fontsize=8)
    for i, v in enumerate(a):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.5, color="#333")
    ax.set_xlabel("mean Sharpe"); ax.set_title("By asset class (mean per-product Sharpe)")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def _rows(df: pd.DataFrame):
    out = []
    for r in df.itertuples(index=False):
        pf = "∞" if r.pf == float("inf") else f"{r.pf:.1f}"
        out.append({"market": r.market, "asset": r.asset, "sharpe": f"{r.sharpe:+.2f}", "pf": pf,
                    "hit": f"{r.hit:.0f}", "trades": int(r.trades), "vt": f"{r.vt_usd:+,.0f}",
                    "direction": _dir(r.sharpe)})
    return out


def render_html(res: pd.DataFrame, meta: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    res = res.sort_values("sharpe", ascending=False)
    return env.get_template("scorecard_report.html").render(
        variant=meta["variant"], asof=meta["asof"], years=meta["years"],
        n=len(res), n_pos=int((res["sharpe"] > 0).sum()),
        vt_total=f"{res['vt_usd'].sum():+,.0f}",
        best=_rows(res.head(12)), worst=_rows(res.tail(8)),
        ranked=ranked_bar_png(res), assets=asset_bar_png(res),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(res: pd.DataFrame, meta: dict, out_path) -> str:
    return render_pdf(render_html(res, meta), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="MA Swing (20/50)")
    ap.add_argument("--asof", default="")
    ap.add_argument("--years", default="10")
    args = ap.parse_args()
    res = pd.read_csv(args.csv)
    build_pdf(res, {"variant": args.variant, "asof": args.asof, "years": args.years}, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
