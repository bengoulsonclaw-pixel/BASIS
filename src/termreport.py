"""Vol Term Structure Report — a branded client PDF for the ATM implied-vol curve
across the book (1M/3M/6M/12M):
  1. Scatter — 1M (x) vs 3M (y) with the 45° flat line; above = contango, below =
     backwardation.
  2. Diverging bar — most stretched curves ranked by the 3M−1M slope z-score.
  3. Curve small-multiples — the 1M→12M curve for the most inverted / most contango.
  4. Vol-risk premium by tenor — where on the curve options are richest vs realized.
Plus a flagged-opportunities table.

Run standalone (the app calls it as a subprocess):
    python src/termreport.py data/signals/termstructure.parquet out.pdf --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportkit import data_uri, png, color, legend, bar_png, flagged, reflag, render_pdf, BLACK, RICH, CHEAP, NEUTRAL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"

TENORS = ["1M", "3M", "6M", "12M"]
TENOR_MONTHS = [1, 3, 6, 12]
IVCOLS = [f"iv_{t.lower()}" for t in TENORS]
RVCOLS = [f"rv_{t.lower()}" for t in TENORS]
VRPCOLS = [f"vrp_{t.lower()}" for t in TENORS]
Z_FLAG = 1.5   # signal threshold — keep in sync with strategies/volatility.Z_FLAG
               # (this script runs standalone and can't import the strategy package)


def ts_scatter_png(df: pd.DataFrame) -> str:
    d = df.dropna(subset=["iv_1m", "iv_3m"])
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    hi = float(max(d["iv_1m"].max(), d["iv_3m"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "1M = 3M (flat)", rotation=45, ha="right", va="bottom",
            fontsize=7, color="#555", rotation_mode="anchor")
    # Whole book plotted; only FLAGGED markets highlighted (coloured + named), rest muted grey.
    neu, fl = d[d["direction"] == 0], d[d["direction"] != 0]
    ax.scatter(neu["iv_1m"], neu["iv_3m"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.scatter(fl["iv_1m"], fl["iv_3m"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.iv_1m, r.iv_3m), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Inverted — front rich", "Steep — front cheap")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("1M ATM vol  (%)")
    ax.set_ylabel("3M ATM vol  (%)")
    ax.set_title("Above the line → contango (3M > 1M);  below → backwardation")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return png(fig)


def curves_png(df: pd.DataFrame, n: int = 4) -> str:
    """Small-multiples for the n most inverted + n most contango: the implied curve
    (solid, signal-coloured) with the matched REALIZED curve overlaid (dashed grey)
    — the gap at each tenor is that tenor's vol-risk premium."""
    d = df.dropna(subset=["z"])
    pick = pd.concat([d.sort_values("z").head(n),
                      d.sort_values("z", ascending=False).head(n)]).drop_duplicates("ticker")
    ncol = 4
    nrow = max(1, int(np.ceil(len(pick) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.1, 1.85 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(pick.itertuples(index=False)):
        ax = axes.flat[i]
        ax.axis("on")
        iv = [getattr(r, c) for c in IVCOLS]
        rv = [getattr(r, c) for c in RVCOLS]
        ax.plot(TENOR_MONTHS, iv, marker="o", ms=3, lw=1.5, color=color(r.direction))
        ax.plot(TENOR_MONTHS, rv, marker="s", ms=2.5, lw=1.1, ls="--", color="#8A8A8A")
        ax.set_title(r.market, fontsize=6.4)
        ax.set_xticks(TENOR_MONTHS)
        ax.set_xticklabels(TENORS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(True, color="#EEE", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#444", lw=1.5, marker="o", ms=3, label="Implied"),
               Line2D([0], [0], color="#8A8A8A", lw=1.1, ls="--", marker="s", ms=2.5,
                      label="Realized (matched window)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.975))
    return png(fig)


def vrp_png(df: pd.DataFrame) -> str:
    means = [float(df[c].mean()) for c in VRPCOLS]
    fig, ax = plt.subplots(figsize=(6.1, 2.3))
    ax.bar(range(len(TENORS)), means, width=0.6, zorder=3, edgecolor="white",
           color=[CHEAP if m >= 0 else RICH for m in means])
    ax.axhline(0, color=BLACK, lw=0.8)
    pad = (max(abs(m) for m in means) or 1.0) * 0.04
    for i, m in enumerate(means):
        ax.text(i, m + (pad if m >= 0 else -pad), f"{m:+.1f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=7, color="#333")
    ax.set_xticks(range(len(TENORS)))
    ax.set_xticklabels(TENORS)
    ax.set_ylabel("mean IV − RV (vol pts)")
    ax.set_title("Vol-risk premium by tenor (book average)")
    ax.grid(True, axis="y", color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def _sd_str(v, dec) -> str:
    """The 1-day 1σ price move formatted to the contract's native decimals, '' if n/a."""
    try:
        if v is None or not pd.notna(v):
            return ""
        return f"{float(v):.{int(dec)}f}"
    except Exception:
        return ""


def render_html(df: pd.DataFrame, asof: str, threshold: float = Z_FLAG) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Steep — front cheap", 1), ("Inverted — front rich", -1))
    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "iv_1m": f"{r.iv_1m:.1f}", "iv_3m": f"{r.iv_3m:.1f}",
        "iv_6m": f"{r.iv_6m:.1f}", "iv_12m": f"{r.iv_12m:.1f}",
        "iv_1m_sd": _sd_str(getattr(r, "iv_sd_1m", None), getattr(r, "px_dec", 1)),
        "iv_3m_sd": _sd_str(getattr(r, "iv_sd_3m", None), getattr(r, "px_dec", 1)),
        "iv_6m_sd": _sd_str(getattr(r, "iv_sd_6m", None), getattr(r, "px_dec", 1)),
        "iv_12m_sd": _sd_str(getattr(r, "iv_sd_12m", None), getattr(r, "px_dec", 1)),
        "slope": f"{r.slope:+.1f}", "ratio": f"{r.ratio:.2f}" if pd.notna(r.ratio) else "—",
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in flagged(df).itertuples(index=False)]
    return env.get_template("termreport.html").render(
        asof=asof,
        scatter=ts_scatter_png(df),
        bars=bar_png(df, value_col="slope", value_fmt="{:+.1f}", n=None,
                     xlabel="3M − 1M slope,  z-score vs 1-yr   (label = slope, vol pts)"),
        curves=curves_png(df), vrp=vrp_png(df), rows=rows,
        n_markets=len(df), zflag=f"{threshold:g}",
        n_steep=int((df["direction"] > 0).sum()), n_inv=int((df["direction"] < 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = Z_FLAG) -> str:
    return render_pdf(render_html(df, asof, threshold), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=Z_FLAG)
    args = ap.parse_args()
    _d = pd.read_parquet(args.detail_parquet)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa() and "ticker" in _d:
            _d = _d[_d["ticker"].isin(_en())]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf, args.threshold)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
