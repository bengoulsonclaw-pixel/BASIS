"""Skew Volatility Report — a branded, client-style PDF for option skew across the
book: the normalized skew (90% put − 110% call)/ATM, z-scored over a trailing year
(positive = puts richer than calls).
  1. Scatter — 90% put vol (y) vs 110% call vol (x) with the 45° no-skew line; a
     dot's distance from the line is its put−call skew.
  2. Diverging bar — most stretched skews ranked by the 1-yr z-score.
  3. Per-flagged-product dumbbell (110% call → 90% put wing) + 1-year skew-vs-price
     small-multiples — the same per-product view the Volatility Report uses.
Plus a flagged-opportunities table.

Listed markets use the fixed 90/110% moneyness wings of the option surface; FX uses
the native OTC 25Δ risk reversal (it has no moneyness surface). Bonds and STIRs are
excluded (low price vol pushes the 90/110% strikes far OTM into extrapolation).

Run standalone (the app calls it as a subprocess):
    python src/skewreport.py data/signals/skew.parquet out.pdf --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportkit import (data_uri, png, color, legend, bar_png, flagged, reflag, render_pdf,
                       BLACK, RICH, CHEAP, NEUTRAL)
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def scatter_png(df: pd.DataFrame) -> str:
    """90% put vol (y) vs 110% call vol (x), with the 45° no-skew line; distance from
    the line is the put−call skew (above = puts richer = downside skew)."""
    d = df.dropna(subset=["put", "call"])
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    hi = float(max(d["put"].max(), d["call"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "put = call (no skew)", rotation=45, ha="right",
            va="bottom", fontsize=7, color="#555", rotation_mode="anchor")
    # Whole book plotted; only FLAGGED markets highlighted (coloured + named), rest muted grey.
    neu, fl = d[d["direction"] == 0], d[d["direction"] != 0]
    ax.scatter(neu["call"], neu["put"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.scatter(fl["call"], fl["put"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.call, r.put), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Rich — sell skew", "Cheap — buy skew")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("110% moneyness call vol  (%)")
    ax.set_ylabel("90% moneyness put vol  (%)")
    ax.set_title("Above the line → puts richer than calls (downside skew)")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return png(fig)


def wings_png(df: pd.DataFrame, n: int = 18) -> str:
    """Dumbbell of the FLAGGED markets: the 110% call wing (grey) and the 90% put
    wing (signal-coloured) joined by a line whose length IS the put−call gap — so the
    reader sees each flagged product's skew at a glance. (FX wings are the 25Δ RR.)"""
    d = flagged(df)
    if d.empty:
        return ""
    if len(d) > n:                                   # keep it readable — most stretched
        d = d.reindex(d["z"].abs().sort_values(ascending=False).index).head(n)
    d = d.sort_values("skew")                        # call-skew (neg) bottom → put-skew (pos) top
    y = list(range(len(d)))
    put, call, dirn = d["put"].to_numpy(), d["call"].to_numpy(), d["direction"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.1, max(1.7, 0.36 * len(d) + 0.6)))
    for i in y:
        ax.plot([call[i], put[i]], [i, i], color="#CFCFCF", lw=1.6, zorder=1)
    ax.scatter(call, y, s=46, color=NEUTRAL, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(put, y, s=46, color=[color(x) for x in dirn],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["market"], fontsize=6.8)
    ax.set_ylim(-0.6, len(d) - 0.4)        # uniform half-row padding → no sparse stretch
    ax.set_xlabel("wing implied vol (%)")
    ax.set_title("Flagged markets — 110% call → 90% put (the line is the skew)")
    ax.grid(True, axis="x", color="#EEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=NEUTRAL, mec="white", ms=8, label="110% call wing"),
               Line2D([0], [0], marker="o", ls="", mfc=RICH, mec="white", ms=8, label="90% put — rich (sell skew)"),
               Line2D([0], [0], marker="o", ls="", mfc=CHEAP, mec="white", ms=8, label="90% put — cheap (buy skew)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=6.3, frameon=False)
    return png(fig)


def _history_chart(rows, hist) -> str:
    """One 2-column grid of dual-axis panels, one per market in `rows`: the skew
    (left, shaded, signal-coloured) over the underlying price (right, grey)."""
    import matplotlib.dates as mdates
    ncol = 2
    nrow = max(1, (len(rows) + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.1, 1.95 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(rows):
        ax = axes.flat[i]
        ax.axis("on")
        h = hist[hist["ticker"] == r.ticker].sort_values("date")
        dates = pd.to_datetime(h["date"])
        c = color(r.direction)
        ax.axhline(0, color="#BBB", lw=0.6, zorder=1)
        ax.fill_between(dates, h["skew"], 0, color=c, alpha=0.18, zorder=2)
        ax.plot(dates, h["skew"], color=c, lw=1.1, zorder=3)
        ax.set_title(r.market, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=5, colors=c, length=2)
        ax.tick_params(axis="x", labelsize=5, length=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax2 = ax.twinx()
        ax2.plot(dates, h["price"], color="#444", lw=1.0, zorder=4)
        ax2.tick_params(axis="y", labelsize=5, colors="#444", length=2)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    fig.suptitle("left / shaded: (90% put − 110% call) / ATM skew      right / grey: underlying price",
                 fontsize=6.6, y=1.0)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.96))
    return png(fig)


def history_pngs(df: pd.DataFrame, hist: pd.DataFrame, per_chart: int = 6) -> list:
    """ALL flagged markets as 1-year dual-axis panels — the skew (left, shaded,
    signal-coloured) vs the underlying price (right, grey). Returns a list of images,
    ≤per_chart panels each, so every chunk fits on a page."""
    if hist is None or len(hist) == 0:
        return []
    fl = flagged(df)
    if fl.empty:
        return []
    have = set(hist["ticker"].unique())
    picks = fl.reindex(fl["z"].abs().sort_values(ascending=False).index)
    rows = [r for r in picks.itertuples(index=False) if r.ticker in have]
    return [_history_chart(rows[i:i + per_chart], hist) for i in range(0, len(rows), per_chart)]


def render_html(df: pd.DataFrame, asof: str, threshold: float = 1.5, hist: pd.DataFrame = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Rich — sell skew", -1), ("Cheap — buy skew", 1))
    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "put": f"{r.put:.1f}", "call": f"{r.call:.1f}", "atm": f"{r.atm:.1f}",
        "skew": f"{r.skew:+.2f}",
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in flagged(df).itertuples(index=False)]
    return env.get_template("skewreport.html").render(
        asof=asof,
        scatter=scatter_png(df),
        bars=bar_png(df, value_col="skew", value_fmt="{:+.2f}", n=None,
                     xlabel="(90% put − 110% call) / ATM skew,  z-score vs 1-yr   (label = skew)"),
        wings=wings_png(df),
        history=history_pngs(df, hist),
        rows=rows, n_markets=len(df), zflag=f"{threshold:g}",
        n_rich=int((df["direction"] < 0).sum()), n_cheap=int((df["direction"] > 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = 1.5, hist=None) -> str:
    return render_pdf(render_html(df, asof, threshold, hist), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "skew_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    _d = pd.read_parquet(detail)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            if "ticker" in _d:
                _d = _d[_d["ticker"].isin(_e)]
            if hist is not None and "ticker" in getattr(hist, "columns", []):
                hist = hist[hist["ticker"].isin(_e)]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf, args.threshold, hist)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
