"""Daily Technical Analysis Report — a branded PDF of the cross-strategy opportunity board:
the **conviction leaderboard** (products ranked by Σ signed strength across the technical
strategies), the **stacked-signals** table (products flagged by 2+), the per-strategy count
bar, and the full flagged list. Reuses src/tascore.py, so it scores identically to the
dashboard hub.

Run standalone (the app calls it as a subprocess):
    python src/tareport.py data/signals/opportunities.parquet out.pdf --asof "2026-06-21"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportkit import data_uri, png, render_pdf, BLACK
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project root, for `src` imports
from src import tascore
from src.universe import INSTRUMENTS

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
LONG, SHORT = "#2E7D32", "#C62828"
ROWS_CAP = 60       # cap the full table in the PDF (the dashboard hub keeps the full list)

# Compact strategy tags for the report (kept here so the report is self-contained).
_SHORT = {"Mean Reversion": "MeanRev", "Trend": "Trend", "MA Crossover": "MA×", "MA Swing": "MA∿",
          "Flag Breakout": "Flag", "Support & Resistance": "S/R", "Fibonacci Retracement": "Fib",
          "Breakout & Retest": "Retest", "Momentum (RSI/MACD)": "Mom", "Bollinger Squeeze": "BBands"}


def _sector(k) -> str:
    return "pair" if " / " in str(k) else (INSTRUMENTS.get(k, (k, 0.0, "", ""))[2] or "—")


def _arrow(d) -> str:
    return "▲" if d > 0 else "▼" if d < 0 else ""


def leaderboard_png(prod: pd.DataFrame, n: int = 20) -> str:
    """Diverging bar of the top-n products by signed conviction score (long green / short red),
    each labelled with how many strategies stack on it."""
    d = prod.head(n).iloc[::-1]              # barh draws bottom-up → strongest at top
    if d.empty:
        return ""
    fig, ax = plt.subplots(figsize=(6.1, max(2.4, 0.32 * len(d))))
    y = list(range(len(d)))
    ax.barh(y, d["score"], color=[LONG if s > 0 else SHORT for s in d["score"]],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(d["market"], fontsize=6.8)
    span = float(d["score"].abs().max()) or 1.0
    for i, (sc, nn) in enumerate(zip(d["score"], d["n"])):
        ax.text(sc + (span * 0.02 if sc >= 0 else -span * 0.02), i, f"{int(nn)}×",
                va="center", ha="left" if sc >= 0 else "right", fontsize=6, color="#333")
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.set_xlabel("← short        cross-strategy conviction (Σ signed strength)        long →   (label = # strategies)")
    ax.set_title("Conviction leaderboard")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


def per_strategy_png(flagged: pd.DataFrame) -> str:
    """Long (green, right) vs short (red, left) flagged counts per technical strategy."""
    cnt = [(s, int(((flagged["strategy"] == s) & (flagged["direction"] > 0)).sum()),
            int(((flagged["strategy"] == s) & (flagged["direction"] < 0)).sum()))
           for s in tascore.TA_STRATEGIES]
    cnt = [c for c in cnt if c[1] + c[2] > 0][::-1]
    if not cnt:
        return ""
    fig, ax = plt.subplots(figsize=(6.1, max(1.8, 0.42 * len(cnt))))
    y = list(range(len(cnt)))
    ax.barh(y, [c[1] for c in cnt], color=LONG, edgecolor="white", linewidth=0.4, label="Long", zorder=3)
    ax.barh(y, [-c[2] for c in cnt], color=SHORT, edgecolor="white", linewidth=0.4, label="Short", zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([_SHORT.get(c[0], c[0]) for c in cnt], fontsize=7)
    ax.set_xlabel("← short            flagged signals            long →")
    ax.set_title("Flagged by strategy")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    return png(fig)


def render_html(df: pd.DataFrame, asof: str) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    flagged = tascore.ta_flagged(df)
    common = dict(logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"), asof=asof)
    if flagged is None or flagged.empty:
        return env.get_template("tareport.html").render(
            n_sig=0, n_long=0, n_short=0, n_prod=0, n_multi=0,
            leaderboard="", per_strategy="", stacked=[], rows=[], **common)

    flagged = flagged.copy()
    flagged["dir"] = pd.to_numeric(flagged["direction"], errors="coerce").fillna(0).astype(int)
    flagged["strength"] = [tascore.strength(s, m) for s, m in zip(flagged["strategy"], flagged["metric"])]
    prod = tascore.score_products(flagged)
    score_map = dict(zip(prod["instruments"], prod["score"]))

    stacked = []
    for r in prod[prod["n"] >= 2].itertuples(index=False):
        tags = ", ".join(_SHORT.get(s, s) + _arrow(d) for s, d, _st in r.tags)
        net = "mixed" if r.conflict else ("long" if r.net_dir > 0 else "short" if r.net_dir < 0 else "—")
        stacked.append({"market": r.market, "sector": _sector(r.instruments), "n": int(r.n),
                        "net": net, "dir": int(r.net_dir), "conviction": f"{r.conviction:.0f}",
                        "score": f"{abs(r.score):.0f}", "tags": tags})

    v = flagged.assign(
        score_abs=flagged["instruments"].map(lambda k: abs(score_map.get(k, 0.0))),
        mkt=flagged["market"].map(lambda m: str(m).split(" · ")[0].strip()),
    ).sort_values(["score_abs", "mkt", "strategy"], ascending=[False, True, True])
    rows = [{"market": r.mkt, "sector": _sector(r.instruments), "strategy": _SHORT.get(r.strategy, r.strategy),
             "signal": r.signal, "dir": int(r.dir), "conviction": f"{r.strength:.0f}",
             "score": f"{r.score_abs:.0f}", "notes": r.context} for r in v.itertuples(index=False)]
    rows_total = len(rows)
    rows = rows[:ROWS_CAP]

    return env.get_template("tareport.html").render(
        n_sig=len(flagged), n_long=int((flagged["dir"] > 0).sum()), n_short=int((flagged["dir"] < 0).sum()),
        n_prod=int(prod.shape[0]), n_multi=int((prod["n"] >= 2).sum()),
        leaderboard=leaderboard_png(prod), per_strategy=per_strategy_png(flagged),
        stacked=stacked, rows=rows, rows_total=rows_total, rows_shown=len(rows), **common)


def build_pdf(df: pd.DataFrame, asof: str, out_path) -> str:
    return render_pdf(render_html(df, asof), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    args = ap.parse_args()
    _d = pd.read_parquet(args.signals_parquet)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa() and "instruments" in _d:
            _e = _en()
            _d = _d[_d["instruments"].map(lambda i: all(p.strip() in _e for p in str(i).split("/")))]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
