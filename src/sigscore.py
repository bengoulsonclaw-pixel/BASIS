"""Weekly Signal Scorecard — a branded one-pager of the Signal Ledger's week.

The accountability wrap for the desk: what the TA book flagged this week, which of the
prior weeks' calls just got their verdicts (5 / 10 / 21 sessions on), how the book's
accuracy is trending, and which strategies are carrying this era — all straight off the
precomputed ledger (src/sigledger.py, data/signal_cache/ledger_outcomes.parquet), so the
numbers are exactly the Signal Ledger page's numbers.

Judged in SIGNAL SPACE like the ledger itself (yields for FI, pair spreads for Mean
Reversion, prices elsewhere); hit = the call's direction was right, σ-move = the size of
the move in the product's own trailing-vol units. Historical accuracy, not P&L and not
advice — the fine print says so.

Run standalone (the app and the weekly emailer call it as a subprocess):
    python src/sigscore.py data/Weekly_Signal_Scorecard.pdf --asof "2026-08-07"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, CHEAP, RICH
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src import sigledger, tascore

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

WEEK_SESSIONS = 5        # "this week" = the ledger's last 5 sessions
TREND_WEEKS = 12         # weekly hit-rate bars on the trend chart
LEAGUE_MIN_N = 25        # same thin-sample bar as the page default
BAR_MIN_N_1Y = 60        # 1y strategy bar: enough signals in the window to mean something
TOP_CALLS = 6            # resolved best/worst confluence calls listed
FRESH_CALLS = 12         # this week's strongest fresh composite calls listed


# ---------------------------------------------------------------------------
# cohort slicing on the ledger's own session grid
# ---------------------------------------------------------------------------
def _sessions(out: pd.DataFrame) -> list:
    return sorted(out["date"].unique())


def week_slice(out: pd.DataFrame) -> pd.DataFrame:
    """Rows from the last WEEK_SESSIONS sessions in the ledger."""
    s = _sessions(out)
    return out[out["date"].isin(s[-WEEK_SESSIONS:])]


def resolved_cohort(out: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The signals whose `horizon`-session verdict landed during this week: flagged in the
    WEEK_SESSIONS sessions ending `horizon` sessions before the ledger's last date."""
    s = _sessions(out)
    if len(s) < horizon + WEEK_SESSIONS:
        return out.iloc[0:0]
    cohort = s[-(horizon + WEEK_SESSIONS):-horizon]
    sub = out[out["date"].isin(cohort)]
    return sub[sub[f"hit{horizon}"].notna()]


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def trend_png(out: pd.DataFrame) -> str:
    """Weekly book hit rate (5-session verdicts, core strategies) over the last
    TREND_WEEKS full weeks — is the book's accuracy warming or cooling?"""
    core = out[(out["strategy"] != sigledger.CONFLUENCE) & out["hit5"].notna()].copy()
    core["week"] = core["date"].dt.to_period("W").dt.start_time
    g = core.groupby("week")["hit5"].agg(["mean", "count"])
    g = g[g["count"] >= 20].tail(TREND_WEEKS)
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    vals = g["mean"].to_numpy() * 100.0
    ax.bar(range(len(g)), vals, color=[CHEAP if v >= 50 else RICH for v in vals],
           edgecolor="white", linewidth=0.5, zorder=3, width=0.72)
    ax.axhline(50, color=BLACK, lw=0.9, ls="--", zorder=4)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([d.strftime("%d %b") for d in g.index], fontsize=6, rotation=0)
    ax.set_ylabel("hit % (5d)", fontsize=7)
    ax.set_ylim(max(0, vals.min() - 8), min(100, vals.max() + 8))
    for i, (v, n) in enumerate(zip(vals, g["count"])):
        ax.text(i, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8, color="#333")
    ax.set_title("Book hit rate by week — all core strategies, 5-session verdicts")
    ax.grid(True, axis="y", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return png(fig)


def strat_bar_png(out: pd.DataFrame, horizon: int = 21) -> str:
    """Per-strategy hit rate over the trailing year at `horizon`, min-n gated —
    'what is carrying the book right now', ranked."""
    hi = out["date"].max()
    sub = out[(out["date"] >= hi - pd.DateOffset(years=1)) & out[f"hit{horizon}"].notna()]
    g = sub.groupby("strategy")[f"hit{horizon}"].agg(["mean", "count"])
    g = g[g["count"] >= BAR_MIN_N_1Y].sort_values("mean")
    vals = g["mean"].to_numpy() * 100.0
    fig, ax = plt.subplots(figsize=(6.4, max(1.8, 0.26 * len(g))))
    ax.barh(range(len(g)), vals - 50.0, left=50.0,
            color=[CHEAP if v >= 50 else RICH for v in vals],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(50, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{s}  (n={int(n):,})" for s, n in zip(g.index, g["count"])],
                       fontsize=6.4)
    for i, v in enumerate(vals):
        ax.text(v + (0.3 if v >= 50 else -0.3), i, f"{v:.0f}%", va="center",
                ha="left" if v >= 50 else "right", fontsize=6, color="#333")
    ax.set_xlabel(f"hit % at {horizon} sessions — trailing 1y", fontsize=7)
    ax.set_title("Strategy hit rates over the last year")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return png(fig)


# ---------------------------------------------------------------------------
# table rows
# ---------------------------------------------------------------------------
def _cell_bg(v, centre: float, span: float) -> str:
    """The Signal Ledger page's diverging cell shade, as an inline style for the PDF."""
    if v is None or pd.isna(v):
        return ""
    x = max(-1.0, min(1.0, (float(v) - centre) / span))
    r, g, b = (30, 132, 73) if x > 0 else (192, 57, 43)
    return f"background-color: rgba({r},{g},{b},{0.10 + 0.35 * abs(x):.2f})"


def verdict_rows(out: pd.DataFrame) -> list:
    """One row per horizon: the cohort whose verdict landed this week."""
    rows = []
    for h in sigledger.HORIZONS:
        c = resolved_cohort(out, h)
        core = c[c["strategy"] != sigledger.CONFLUENCE]
        conf = c[c["strategy"] == sigledger.CONFLUENCE]
        if core.empty:
            continue
        hit = core[f"hit{h}"].mean() * 100.0
        rows.append({
            "h": h, "n": f"{len(core):,}",
            "hit": f"{hit:.1f}%", "hit_bg": _cell_bg(hit, 50.0, 8.0),
            "sig": f"{core[f'sig{h}'].mean():+.2f}σ",
            "conf_hit": f"{conf[f'hit{h}'].mean() * 100.0:.1f}%" if len(conf) else "—",
            "conf_bg": _cell_bg(conf[f"hit{h}"].mean() * 100.0, 50.0, 8.0) if len(conf) else "",
        })
    return rows


def call_rows(out: pd.DataFrame, horizon: int = 5) -> tuple[list, list]:
    """This week's resolved composite calls, best and worst by σ-move at `horizon`."""
    c = resolved_cohort(out, horizon)
    c = c[(c["strategy"] == sigledger.CONFLUENCE) & c[f"sig{horizon}"].notna()]
    scol, hcol = f"sig{horizon}", f"hit{horizon}"

    def _fmt(sub):
        rows = []
        for r in sub.itertuples(index=False):
            s, hitv = getattr(r, scol), getattr(r, hcol)
            rows.append({"date": pd.Timestamp(r.date).strftime("%d %b"),
                         "market": r.market, "signal": r.signal,
                         "ok": bool(hitv), "sig": f"{s:+.1f}σ"})
        return rows
    return (_fmt(c.sort_values(scol, ascending=False).head(TOP_CALLS)),
            _fmt(c.sort_values(scol).head(TOP_CALLS)))


def fresh_rows(out: pd.DataFrame) -> list:
    """This week's strongest fresh composite calls (all still pending their verdicts)."""
    wk = week_slice(out)
    conf = wk[wk["strategy"] == sigledger.CONFLUENCE].copy()
    conf["absm"] = conf["metric"].abs()
    conf = conf.sort_values("absm", ascending=False).head(FRESH_CALLS)
    return [{"date": pd.Timestamp(r.date).strftime("%d %b"), "market": r.market,
             "signal": r.signal, "score": f"{r.metric:+.1f}",
             "level": f"{r.entry_level:,.3f}" if pd.notna(r.entry_level) else "—"}
            for r in conf.itertuples(index=False)]


def league_rows(out: pd.DataFrame) -> tuple[list, list]:
    """The era league (windows_league) as template rows + its column labels."""
    wl = sigledger.windows_league(out, horizon=21, min_n=LEAGUE_MIN_N)
    if wl.empty:
        return [], []
    yrs = out["date"].max().year - out["date"].min().year
    labels = [f"Full {yrs}y" if lab == "Full" else lab for _, lab in sigledger.WINDOWS]
    rows = []
    for _, r in wl.iterrows():
        cells = [{"txt": f"{r[lab]:.1f}%" if pd.notna(r[lab]) else "—",
                  "bg": _cell_bg(r[lab], 50.0, 5.0)} for _, lab in sigledger.WINDOWS]
        rows.append({"strategy": r["strategy"],
                     "category": tascore.axis_tag(r["strategy"]),
                     "n": f"{int(r['n Full']):,}", "cells": cells,
                     "delta": f"{r['delta']:+.1f}pp",
                     "delta_bg": _cell_bg(r["delta"], 0.0, 5.0)})
    return rows, labels


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def render_html(out: pd.DataFrame, asof: str) -> str:
    wk = week_slice(out)
    core_wk = wk[wk["strategy"] != sigledger.CONFLUENCE]
    conf_wk = wk[wk["strategy"] == sigledger.CONFLUENCE]
    s = _sessions(out)
    week_lo, week_hi = pd.Timestamp(s[-WEEK_SESSIONS]), pd.Timestamp(s[-1])

    n_resolved = sum(len(resolved_cohort(out, h)) for h in sigledger.HORIZONS)
    r5 = resolved_cohort(out, 5)
    r5 = r5[r5["strategy"] != sigledger.CONFLUENCE]
    hit5_week = r5["hit5"].mean() * 100.0 if len(r5) else None

    hi = out["date"].max()
    y1 = out[(out["date"] >= hi - pd.DateOffset(years=1))
             & (out["strategy"] != sigledger.CONFLUENCE)]
    hit21_1y = y1["hit21"].mean() * 100.0 if y1["hit21"].notna().any() else None

    longs = int((core_wk["direction"] > 0).sum())
    shorts = int((core_wk["direction"] < 0).sum())
    best_calls, worst_calls = call_rows(out)
    lg_rows, lg_labels = league_rows(out)
    rr = sigledger.regime_read(out)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("sigscore_report.html").render(
        asof=pretty_date(asof or str(week_hi.date())),
        week_span=f"{week_lo.strftime('%d %b')} – {week_hi.strftime('%d %b %Y')}",
        n_week=f"{len(core_wk):,}", n_conf=f"{len(conf_wk):,}",
        longs=f"{longs:,}", shorts=f"{shorts:,}",
        n_resolved=f"{n_resolved:,}",
        hit5_week=f"{hit5_week:.1f}%" if hit5_week is not None else "—",
        hit5_week_bg=_cell_bg(hit5_week, 50.0, 8.0) if hit5_week is not None else "",
        hit21_1y=f"{hit21_1y:.1f}%" if hit21_1y is not None else "—",
        verdicts=verdict_rows(out),
        best_calls=best_calls, worst_calls=worst_calls,
        fresh=fresh_rows(out),
        league=lg_rows, league_labels=lg_labels,
        regime=(rr or {}).get("text", "").replace("**", ""),
        trend=trend_png(out), strat_bar=strat_bar_png(out),
        ledger_span=f"{out['date'].min().date()} → {out['date'].max().date()}",
        n_ledger=f"{len(out):,}",
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(out_path, asof: str = "", ledger: pd.DataFrame | None = None) -> str:
    out = ledger if ledger is not None else sigledger.load()
    if out is None or out.empty:
        raise SystemExit("No signal ledger on disk — run backfill_signals.py first.")
    return render_pdf(render_html(out, asof), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--asof", default="")
    ap.add_argument("--ledger", default=None, help="override the outcomes parquet path")
    args = ap.parse_args()
    ledger = None
    if args.ledger:
        ledger = pd.read_parquet(args.ledger)
        ledger["date"] = pd.to_datetime(ledger["date"])
    build_pdf(args.out, args.asof, ledger)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
