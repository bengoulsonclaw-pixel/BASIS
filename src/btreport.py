"""MA Crossover trigger-backtest — branded PDF for the desk.

backtest_ma.py does the compute and hands over the result tables; this renders them
on the same Morning-Coffee house style as the other reports (reportkit +
_report_style.html). Two visuals: a Sharpe-by-pair bar (filtered vs raw) and a
Sharpe heatmap by asset class × pair, plus the ranked book table and a best-pair-
per-asset-class table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .reportkit import BLACK, CHEAP, RICH, data_uri, png, render_pdf
import matplotlib.pyplot as plt

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def _dir(v) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def sharpe_bars_png(book: pd.DataFrame) -> str:
    pairs = book["Pair"].tolist()
    x = np.arange(len(pairs)); w = 0.38
    filt, raw = book["sharpe_filt"].to_numpy(), book["sharpe_raw"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.1, 3.0))
    ax.bar(x - w / 2, filt, width=w, color=[CHEAP if v >= 0 else RICH for v in filt],
           edgecolor="white", linewidth=0.5, zorder=3, label="Filtered (cross + momentum)")
    ax.bar(x + w / 2, raw, width=w, color="#C9C9C9", edgecolor="white", linewidth=0.5,
           zorder=3, label="Raw cross")
    ax.axhline(0, color=BLACK, lw=0.9, zorder=4)
    for xi, v in zip(x - w / 2, filt):
        if v == v:
            ax.text(xi, v + (0.012 if v >= 0 else -0.012), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=5.8, color="#222")
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=7)
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Risk-adjusted return by MA pair")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(True, axis="y", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def heatmap_png(mat: pd.DataFrame) -> str:
    M = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.1, 0.42 * len(mat) + 1.3))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontsize=6.6, rotation=18, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                        color="#111" if abs(v) < 0.45 else "white")
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Filtered Sharpe by asset class × MA pair  (green = better)")
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    return png(fig)


def render_html(book: pd.DataFrame, mat: pd.DataFrame, meta: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    book_rows = [{
        "pair": r.Pair, "sharpe_f": f"{r.sharpe_filt:+.2f}", "sharpe_r": f"{r.sharpe_raw:+.2f}",
        "ann": f"{r.ann_ret:+.1f}", "dd": f"{r.maxdd:.1f}", "trades": f"{r.trades_yr:.1f}",
        "hit": f"{r.hit:.0f}", "direction": _dir(r.sharpe_filt),
    } for r in book.itertuples(index=False)]

    best_rows = []
    for a in mat.index:
        row = mat.loc[a].dropna()
        if row.empty:
            continue
        best_rows.append({"asset": a, "n": meta["n_by_asset"].get(a, 0), "pair": row.idxmax(),
                          "sharpe": f"{row.max():+.2f}", "direction": _dir(row.max())})

    return env.get_template("btreport.html").render(
        start=meta["start"], end=meta["end"], years=f"{meta['years']:.1f}",
        n_markets=meta["n_markets"], cost=f"{meta['cost_bps']:g}", bh=f"{meta['bh_sharpe']:+.2f}",
        best_pair=book.iloc[0]["Pair"], bars=sharpe_bars_png(book), heatmap=heatmap_png(mat),
        book_rows=book_rows, best_rows=best_rows,
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(book: pd.DataFrame, mat: pd.DataFrame, meta: dict, out_path) -> str:
    return render_pdf(render_html(book, mat, meta), out_path)
