"""Weekly Signal Scorecard — a branded weekly PDF of the Signal Ledger's track record.

The client-facing accountability wrap, read in ninety seconds: how the desk's technical
signals are actually performing, where the edge lives, and what changed this week. The
honesty is the point — most strategies sit near 50%, a few carry a real edge, and the
report says so with sample sizes attached.

Page 1 — the scorecard: KPI strip, the ledger's auto-written regime read, the era league
(full / 5y / 3y / 1y hit rates side by side, delta-annotated against the previous
edition), and the calls that just became old enough to judge at 21 sessions — hits AND
misses, because showing the misses is what makes the hits believable.
Page 2 — where the edge lives: the strategy × product hit-rate heatmap with programmatic
callouts, and a Watch list of products where the confluence composite is currently
flagging AND the flagging strategies have a real track record on that product.

Everything reads data/signal_cache/ledger_outcomes.parquet only (src/sigledger.py,
rebuilt each morning snapshot) — no recompute, builds in seconds, works offline. Every
cell, callout and watch item is min-sample gated so a 4-signal 100% can never print.
The intro paragraph is AI-polished into the desk voice by default (ai_polish.py chain,
deterministic template fallback — the report never depends on the model being reachable).

Run standalone (the app and the weekly emailer call it as a subprocess):
    python src/sigscore.py data/Weekly_Signal_Scorecard.pdf --asof "2026-08-07" --baseline
(--baseline rolls the previous-edition store data/scorecard_last.json — the scheduled
weekly run passes it; ad-hoc previews don't, so kicking the tyres never eats the deltas.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, CHEAP, RICH
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src import sigledger, tascore

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"
LAST_FILE = ROOT / "data" / "scorecard_last.json"

WEEK_SESSIONS = 5        # "this week" = the ledger's last 5 sessions
TREND_WEEKS = 12         # weekly hit-rate bars on the trend chart
ROLL_SESSIONS = 252      # rolling window (sessions ≈ 1y) for the regime-rotation chart
ROLL_MIN_N = 200         # rolling points on fewer evaluable signals than this are masked
ROLL_TOP = 3             # strategies coloured on the rotation chart (rest grey context)
LEAGUE_MIN_N = 25        # same thin-sample bar as the Signal Ledger page default
CELL_MIN_N = 25          # heatmap / watch: min signals in a strategy × product cell
CALLOUT_MIN_N = 40       # callouts quote a single cell — hold them to a higher bar
TOP_CALLS = 6            # resolved best/worst confluence calls listed per side
WATCH_MAX = 8            # watch-list rows
WATCH_MIN_STRATS = 2     # a watch item needs ≥ this many above-median flagging strategies


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
# previous-edition store — what makes week 40 worth opening
# ---------------------------------------------------------------------------
def load_last() -> dict:
    try:
        return json.loads(LAST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_last(asof, league_1y: dict, regime_text: str, heat_cells: dict) -> None:
    LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_FILE.write_text(json.dumps(
        {"asof": str(asof), "league_1y": league_1y, "regime": regime_text,
         "heat": heat_cells}, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# gated strategy × product cells — shared by heatmap, callouts and watch list
# ---------------------------------------------------------------------------
def gated_cells(out: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """sigledger.heat() with the min-sample gate applied — the ONLY cell table the report
    uses, so no surface can quote a thin cell. Core strategies only."""
    hm = sigledger.heat(out[out["strategy"] != sigledger.CONFLUENCE], horizon)
    if hm.empty:
        return hm
    return hm[hm["n"] >= CELL_MIN_N].reset_index(drop=True)


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
    fig, ax = plt.subplots(figsize=(6.4, 1.8))
    vals = g["mean"].to_numpy() * 100.0
    ax.bar(range(len(g)), vals, color=[CHEAP if v >= 50 else RICH for v in vals],
           edgecolor="white", linewidth=0.5, zorder=3, width=0.72)
    ax.axhline(50, color=BLACK, lw=0.9, ls="--", zorder=4)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([d.strftime("%d %b") for d in g.index], fontsize=6)
    ax.set_ylabel("hit % (5d)", fontsize=7)
    ax.set_ylim(max(0, vals.min() - 8), min(100, vals.max() + 8))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8, color="#333")
    ax.set_title("Book hit rate by week — all core strategies, 5-session verdicts")
    ax.grid(True, axis="y", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return png(fig)


def rolling_png(out: pd.DataFrame) -> str:
    """The regime rotation in one picture: each core strategy's TRAILING-YEAR hit rate
    (21-session verdicts, rolling ROLL_SESSIONS sessions) through the ledger's whole
    history. Emphasis styling — the ROLL_TOP strategies leading the CURRENT trailing
    year are coloured and end-labelled, everything else is grey context; where the lines
    cross is where signal leadership rotated."""
    core = out[(out["strategy"] != sigledger.CONFLUENCE) & out["hit21"].notna()]
    daily = (core.groupby(["strategy", "date"])["hit21"].agg(["sum", "count"])
             .reset_index())
    sessions = pd.DatetimeIndex(sorted(core["date"].unique()))
    curves, currents = {}, {}
    for s, g in daily.groupby("strategy"):
        g = g.set_index("date").reindex(sessions).fillna(0.0)
        hits = g["sum"].rolling(ROLL_SESSIONS).sum()
        n = g["count"].rolling(ROLL_SESSIONS).sum()
        curve = (hits / n * 100.0).where(n >= ROLL_MIN_N)
        if curve.notna().any():
            curves[s] = curve
            currents[s] = curve.dropna().iloc[-1]
    if not curves:
        return ""
    top = [s for s, _ in sorted(currents.items(), key=lambda kv: -kv[1])[:ROLL_TOP]]
    colours = ["#1e8449", "#C8901A", "#1F5FA8"]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for s, curve in curves.items():
        if s not in top:
            ax.plot(sessions, curve, color="#D2D2D2", lw=0.7, zorder=2)
    # End labels: the current leaders sit within a point of one another, so labels are
    # spread to a minimum vertical gap (label y only — the lines stay where they are).
    ends = []
    for i, s in enumerate(top):
        curve = curves[s]
        ax.plot(sessions, curve, color=colours[i], lw=1.5, zorder=4)
        last = curve.dropna()
        ends.append([last.index[-1], float(last.iloc[-1]), s, colours[i]])
    gap = 3.2
    for j, e in enumerate(sorted(ends, key=lambda e: e[1])):
        if j and e[1] - prev_y < gap:                      # noqa: F821 (set below)
            e[1] = prev_y + gap
        prev_y = e[1]
    for x, y, s, col in ends:
        ax.annotate(f" {s} {curves[s].dropna().iloc[-1]:.0f}%", (x, y),
                    fontsize=6.2, color=col, va="center", fontweight="bold")
    ax.axhline(50, color=BLACK, lw=0.9, ls="--", zorder=3)
    ax.set_ylabel("trailing-1y hit % (21d)", fontsize=7)
    ax.margins(x=0.01)
    ax.set_xlim(right=sessions[-1] + (sessions[-1] - sessions[0]) * 0.14)
    ax.tick_params(labelsize=6)
    ax.set_title("Trailing-year hit rate through time — where the lines cross, "
                 "leadership rotated")
    ax.grid(True, axis="y", color="#ECECEC", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return png(fig, dpi=200)


def axis_rows(out: pd.DataFrame) -> list:
    """The de-duplicated view: one vote per (day, product, axis) — the axis's net
    majority call — full-history vs trailing-year hit rates per axis. The same counting
    the Signal Ledger page now defaults to: five trend methods echoing one call is one
    vote here, so no family can flatter the book by shouting."""
    v = sigledger.axis_votes(out)
    if v.empty:
        return []
    hi = v["date"].max()
    y1 = v[v["date"] >= hi - pd.DateOffset(years=1)]
    rows = []
    for ax_tag, g in v.groupby("strategy"):
        full_n = int(g["hit21"].notna().sum())
        if full_n < LEAGUE_MIN_N:
            continue
        full_hit = g["hit21"].mean() * 100.0
        g1 = y1[y1["strategy"] == ax_tag]
        n1 = int(g1["hit21"].notna().sum())
        hit1 = g1["hit21"].mean() * 100.0 if n1 >= LEAGUE_MIN_N else None
        rows.append({"axis": ax_tag, "n": f"{full_n:,}",
                     "full": f"{full_hit:.1f}%", "full_bg": _cell_bg(full_hit, 50.0, 5.0),
                     "y1": f"{hit1:.1f}%" if hit1 is not None else "—",
                     "y1_bg": _cell_bg(hit1, 50.0, 5.0) if hit1 is not None else "",
                     "delta": f"{hit1 - full_hit:+.1f}pp" if hit1 is not None else "—",
                     "delta_bg": _cell_bg(hit1 - full_hit, 0.0, 5.0)
                                 if hit1 is not None else "",
                     "_sort": hit1 if hit1 is not None else -1})
    rows.sort(key=lambda r: -r["_sort"])
    for r in rows:
        r.pop("_sort")
    return rows


def heat_png(cells: pd.DataFrame) -> str:
    """Strategy × product hit-rate grid at 21 sessions, full history, report styling —
    green = the family has historically been right on that product, red = wrong more often
    than right. Gated cells only; ungated cells paint grey ('too few signals to read')."""
    pv = cells.pivot(index="strategy", columns="market", values="hit")
    pv = pv.reindex(index=sorted(pv.index), columns=sorted(pv.columns))
    cmap = LinearSegmentedColormap.from_list("hit", [RICH, "#FFFFFF", CHEAP])
    cmap.set_bad("#EFEFEF")
    fig, ax = plt.subplots(figsize=(6.6, 0.21 * len(pv.index) + 1.15))
    ax.imshow(np.ma.masked_invalid(pv.to_numpy()), cmap=cmap, vmin=35, vmax=65,
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=90, fontsize=4.6)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=5.8)
    ax.set_title("Hit rate by strategy × product — 21 sessions, full history "
                 f"(cells with < {CELL_MIN_N} signals in grey)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return png(fig, dpi=200)


# ---------------------------------------------------------------------------
# programmatic callouts + watch list
# ---------------------------------------------------------------------------
def callouts(cells: pd.DataFrame, prev_heat: dict) -> list:
    """2-3 deterministic reads off the gated cell table: the strongest and weakest cells
    with a real sample, and the biggest mover since the previous edition."""
    outp = []
    big = cells[cells["n"] >= CALLOUT_MIN_N]
    if len(big):
        b = big.loc[big["hit"].idxmax()]
        outp.append(f"The book's most reliable pairing has been <b>{b['strategy']} on "
                    f"{b['market']}</b> — right {b['hit']:.0f}% of the time over "
                    f"{int(b['n']):,} signals.")
        w = big.loc[big["hit"].idxmin()]
        outp.append(f"At the other end, <b>{w['strategy']} on {w['market']}</b> has been "
                    f"wrong more often than right ({w['hit']:.0f}% over {int(w['n']):,} "
                    f"signals) — historically a read to treat with caution.")
    if prev_heat:
        cur = {f"{r.strategy}|{r.market}": r.hit for r in cells.itertuples(index=False)}
        moves = [(k, cur[k] - prev_heat[k]) for k in cur.keys() & prev_heat.keys()]
        if moves:
            k, d = max(moves, key=lambda kv: abs(kv[1]))
            if abs(d) >= 0.5:
                s, m = k.split("|", 1)
                outp.append(f"Biggest mover since the last edition: <b>{s} on {m}</b>, "
                            f"{d:+.1f}pp.")
    return outp


def watch_rows(out: pd.DataFrame, cells: pd.DataFrame) -> list:
    """Products where the confluence composite is CURRENTLY flagging AND the flagging
    strategies have above-median era hit rates on that product — 'signal now' ∩ 'track
    record here'. The median is taken across the gated cell table, so the bar itself is
    sample-safe. Client phrasing only — observations, never a recommendation."""
    if cells.empty:
        return []
    last = out["date"].max()
    conf = out[(out["strategy"] == sigledger.CONFLUENCE) & (out["date"] == last)]
    core = out[(out["strategy"] != sigledger.CONFLUENCE) & (out["date"] == last)]
    median_hit = cells["hit"].median()
    cell_hit = {(r.strategy, r.market): (r.hit, int(r.n)) for r in cells.itertuples(index=False)}
    rows = []
    for c in conf.itertuples(index=False):
        flg = core[(core["instruments"] == c.instruments) & (core["direction"] == c.direction)]
        quals = []
        for s in flg["strategy"].unique():
            hit_n = cell_hit.get((s, c.market))
            if hit_n and hit_n[0] > median_hit:
                quals.append((s, hit_n[0], hit_n[1]))
        if len(quals) >= WATCH_MIN_STRATS:
            quals.sort(key=lambda q: -q[1])
            rows.append({"market": c.market, "signal": c.signal,
                         "n_flag": int(flg["strategy"].nunique()),
                         "avg_hit": float(np.mean([q[1] for q in quals])),
                         "who": ", ".join(f"{s} ({h:.0f}%, n={n:,})"
                                          for s, h, n in quals[:3])})
    rows.sort(key=lambda r: -r["avg_hit"])
    for r in rows:
        r["avg_hit"] = f"{r['avg_hit']:.0f}%"
    return rows[:WATCH_MAX]


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


def call_rows(out: pd.DataFrame, horizon: int = 21) -> tuple[list, list]:
    """The composite calls that just became old enough to judge at `horizon` sessions —
    the sharpest hits AND the clearest misses, ranked by the σ-size of the outcome."""
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


def league_rows(out: pd.DataFrame, prev_league: dict) -> tuple[list, list]:
    """The era league (windows_league) as template rows + its column labels, each row
    delta-annotated against the previous edition's 1y hit rate."""
    wl = sigledger.windows_league(out, horizon=21, min_n=LEAGUE_MIN_N)
    if wl.empty:
        return [], []
    yrs = out["date"].max().year - out["date"].min().year
    labels = [f"Full {yrs}y" if lab == "Full" else lab for _, lab in sigledger.WINDOWS]
    rows = []
    for _, r in wl.iterrows():
        cells = [{"txt": f"{r[lab]:.1f}%" if pd.notna(r[lab]) else "—",
                  "bg": _cell_bg(r[lab], 50.0, 5.0)} for _, lab in sigledger.WINDOWS]
        prev = prev_league.get(r["strategy"])
        wk = (r["1y"] - prev) if (prev is not None and pd.notna(r["1y"])) else None
        rows.append({"strategy": r["strategy"],
                     "category": tascore.axis_tag(r["strategy"]),
                     "n": f"{int(r['n Full']):,}", "cells": cells,
                     "delta": f"{r['delta']:+.1f}pp",
                     "delta_bg": _cell_bg(r["delta"], 0.0, 5.0),
                     "wk": (f"{wk:+.1f}pp" if wk is not None and abs(wk) >= 0.05 else "·"),
                     "wk_bg": _cell_bg(wk, 0.0, 2.0) if wk is not None else ""})
    return rows, labels


# ---------------------------------------------------------------------------
# intro — deterministic template, AI-polished into the desk voice by default
# ---------------------------------------------------------------------------
INTRO_SYSTEM = (
    "You are a senior futures strategist writing the opening paragraph of the desk's "
    "weekly Signal Scorecard for professional clients. Rewrite the terse note you are "
    "given so it reads like a real person opening a weekly letter — flowing, plain-"
    "English, warm but professional, three to four sentences.\n"
    "HARD RULES: keep EVERY number and percentage EXACTLY as given, wrapped in the same "
    "**bold** markers; keep every strategy and product name; stay neutral and "
    "observational — this is a track record being reported, never advice: no buy/sell/"
    "recommend language and nothing that implies the reader should act.\n"
    "Return ONLY a JSON array with the single rewritten string.")


def _mc_python() -> str:
    """The Morning Coffee interpreter (has anthropic + the API key); '' if not found."""
    base = Path(r"C:\Users\Ben\AppData\Local\Python")
    for exe in sorted(base.glob("pythoncore-*-64/python.exe"), reverse=True):
        if exe.exists():
            return str(exe)
    import shutil
    return shutil.which("python") or ""


def _ai_polish(texts: list, system: str) -> list:
    """ai_polish.py under the Morning Coffee interpreter; originals unchanged on ANY
    failure (offline, no key, timeout, bad output) — same contract as convreport."""
    if not texts:
        return texts
    try:
        mc_py = _mc_python()
        if not mc_py:
            return texts
        with tempfile.TemporaryDirectory() as td:
            inp, outp = Path(td) / "in.json", Path(td) / "out.json"
            sysf = Path(td) / "system.txt"
            inp.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
            sysf.write_text(system, encoding="utf-8")
            r = subprocess.run([mc_py, str(ROOT / "ai_polish.py"), str(inp), str(outp),
                                str(sysf)], capture_output=True, text=True, timeout=240)
            if r.returncode == 0 and outp.exists():
                got = json.loads(outp.read_text(encoding="utf-8"))
                if isinstance(got, list) and len(got) == len(texts):
                    return [g if (isinstance(g, str) and g.strip()) else t
                            for g, t in zip(got, texts)]
    except Exception:
        pass
    return texts


def _md2html(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s or "")


def intro_text(out: pd.DataFrame, n_week: int, n_resolved: int, hit21_wk,
               lg_rows: list, prev: dict, regime_changed: bool) -> str:
    """The deterministic intro the AI pass rewrites (and the fallback if it can't)."""
    wk = week_slice(out)
    n_prod = wk["market"].nunique()
    parts = [f"This week the book flagged **{n_week:,}** signals across **{n_prod}** "
             f"products, and **{n_resolved:,}** earlier calls received their verdicts"
             + (f" — the cohort judged at 21 sessions came in at **{hit21_wk}**."
                if hit21_wk else ".")]
    lead = [r for r in lg_rows if r["cells"][-1]["txt"] != "—"][:2]
    if lead:
        parts.append("Over the trailing year the book's strongest reads have been "
                     + " and ".join(f"**{r['strategy']}** (**{r['cells'][-1]['txt']}** "
                                    f"at 21 sessions)" for r in lead) + ".")
    if regime_changed:
        parts.append("The regime read itself rotated this week — the era comparison "
                     "below is this edition's story.")
    elif prev:
        movers = [(r["strategy"], r["wk"]) for r in lg_rows if r["wk"] != "·"]
        if movers:
            s, d = max(movers, key=lambda kv: abs(float(kv[1].replace("pp", ""))))
            parts.append(f"Sharpest move since the last edition: **{s}**, **{d}** "
                         f"on its trailing-year hit rate.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def render_html(out: pd.DataFrame, asof: str, ai_polish: bool = True,
                update_baseline: bool = False) -> str:
    wk = week_slice(out)
    core_wk = wk[wk["strategy"] != sigledger.CONFLUENCE]
    conf_wk = wk[wk["strategy"] == sigledger.CONFLUENCE]
    s = _sessions(out)
    week_lo, week_hi = pd.Timestamp(s[-WEEK_SESSIONS]), pd.Timestamp(s[-1])

    n_resolved = sum(len(resolved_cohort(out, h)) for h in sigledger.HORIZONS)
    r21 = resolved_cohort(out, 21)
    r21 = r21[r21["strategy"] != sigledger.CONFLUENCE]
    hit21_wk = f"{r21['hit21'].mean() * 100.0:.1f}%" if len(r21) else None

    hi = out["date"].max()
    y1 = out[(out["date"] >= hi - pd.DateOffset(years=1))
             & (out["strategy"] != sigledger.CONFLUENCE)]
    hit21_1y = y1["hit21"].mean() * 100.0 if y1["hit21"].notna().any() else None

    prev = load_last()
    rr = sigledger.regime_read(out)
    regime_text = (rr or {}).get("text", "")
    regime_changed = bool(prev.get("regime")) and prev.get("regime") != regime_text

    cells = gated_cells(out)
    lg_rows, lg_labels = league_rows(out, prev.get("league_1y", {}))
    best_calls, worst_calls = call_rows(out)
    watch = watch_rows(out, cells)

    intro = intro_text(out, len(core_wk), n_resolved, hit21_wk, lg_rows, prev,
                       regime_changed)
    if ai_polish:
        intro = _ai_polish([intro], INTRO_SYSTEM)[0]

    if update_baseline or not LAST_FILE.exists():
        save_last(week_hi.date(),
                  {r["strategy"]: float(r["cells"][-1]["txt"].rstrip("%"))
                   for r in lg_rows if r["cells"][-1]["txt"] != "—"},
                  regime_text,
                  {f"{r.strategy}|{r.market}": float(r.hit)
                   for r in cells.itertuples(index=False)})

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("sigscore_report.html").render(
        asof=pretty_date(asof or str(week_hi.date())),
        week_span=f"{week_lo.strftime('%d %b')} – {week_hi.strftime('%d %b %Y')}",
        intro=_md2html(intro),
        n_week=f"{len(core_wk):,}", n_conf=f"{len(conf_wk):,}",
        longs=f"{int((core_wk['direction'] > 0).sum()):,}",
        shorts=f"{int((core_wk['direction'] < 0).sum()):,}",
        n_resolved=f"{n_resolved:,}",
        hit21_wk=hit21_wk or "—",
        hit21_wk_bg=_cell_bg(r21["hit21"].mean() * 100.0, 50.0, 8.0) if len(r21) else "",
        hit21_1y=f"{hit21_1y:.1f}%" if hit21_1y is not None else "—",
        verdicts=verdict_rows(out),
        best_calls=best_calls, worst_calls=worst_calls,
        league=lg_rows, league_labels=lg_labels, has_prev=bool(prev),
        regime=regime_text.replace("**", ""), regime_changed=regime_changed,
        rotation=rolling_png(out), axes=axis_rows(out),
        heatmap=heat_png(cells) if not cells.empty else "",
        callouts=callouts(cells, prev.get("heat", {})),
        watch=watch, median_note=f"{cells['hit'].median():.0f}%" if len(cells) else "—",
        trend=trend_png(out),
        ledger_span=f"{out['date'].min().date()} → {out['date'].max().date()}",
        n_ledger=f"{len(out):,}", cell_min_n=CELL_MIN_N, league_min_n=LEAGUE_MIN_N,
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(out_path, asof: str = "", ledger: pd.DataFrame | None = None,
              ai_polish: bool = True, update_baseline: bool = False) -> str:
    out = ledger if ledger is not None else sigledger.load()
    if out is None or out.empty:
        raise SystemExit("No signal ledger on disk — run backfill_signals.py first.")
    return render_pdf(render_html(out, asof, ai_polish, update_baseline), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--asof", default="")
    ap.add_argument("--ledger", default=None, help="override the outcomes parquet path")
    ap.add_argument("--baseline", action="store_true",
                    help="roll the previous-edition store (the scheduled weekly run "
                         "passes this; ad-hoc previews don't)")
    ap.add_argument("--no-ai-polish", action="store_true",
                    help="keep the deterministic intro (polish is ON by default)")
    args = ap.parse_args()
    ledger = None
    if args.ledger:
        ledger = pd.read_parquet(args.ledger)
        ledger["date"] = pd.to_datetime(ledger["date"])
    build_pdf(args.out, args.asof, ledger, ai_polish=not args.no_ai_polish,
              update_baseline=args.baseline)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
