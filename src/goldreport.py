"""goldreport.py — the weekly Gold Week Ahead report.

What this report is, and the line it does not cross
---------------------------------------------------
Seven milestones of modelling established two things about gold, and this report is
built entirely on the second:

  1. Gold's DIRECTION is not forecastable from public drivers at any horizon we can
     test — pooled or within volatility/trend regimes, across 21 years. Nothing here
     forecasts direction, and the methodology page says so in as many words.
  2. The drivers EXPLAIN gold's moves well, and event days have a measurable
     volatility signature. Both are historical facts, and both are useful.

So the report answers "what is coming, how has gold behaved around it before, and
what would a given move in rates or the dollar be worth" — never "gold will rise".
That is also the right side of the compliance line: neutral observation, no
recommendation, no implied advice.

Sections
--------
  1  Week ahead          high-impact US releases, from the free calendar feed
  2  Event-day behaviour measured over 1990-2026: only the employment report shows
                         a volatility premium (1.43x); CPI and PCE do not
  3  Driver sensitivities gold % per unit move, Newey-West t, point-in-time data
  4  Scenario grid       three stated driver views run through those sensitivities
  5  Where gold sits     fair-value gap and an attribution of the month just gone
  6  Methodology         what was measured, and explicitly what is NOT claimed

CLI:  python src/goldreport.py out.pdf [--asof 2026-08-24]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import reportkit                                    # sets Agg + rcParams
import matplotlib.pyplot as plt                              # noqa: E402
from src import econ, goldsens                               # noqa: E402
from src.reportkit import data_uri, png, render_pdf          # noqa: E402

TEMPLATES = _ROOT / "templates"
ASSETS = TEMPLATES / "assets"
STORE_DIR = _ROOT / "data" / "gold_store"
EVENT_STUDY = STORE_DIR / "event_study.json"
OUT_DEFAULT = _ROOT / "reports" / "Gold_Week_Ahead.pdf"

# pm/opec fundamentals palette. Gold is for FILLS and accents only — it does not read
# as a thin line on white; lines are ink and navy.
INK, GOLD, GOLDEDGE = "#1A1A1A", "#F5C518", "#B8860B"
GREY, NAVY, GREEN, RED = "#9AA0A8", "#0B3D91", "#2E7D32", "#C62828"

# Releases worth flagging, mapped onto the event study's measured behaviour.
#
# COUNTRY-GATED. The event study measured US releases only, so the behaviour it
# reports may only ever be attached to a US release. Keying on the title alone let an
# Australian "CPI m/m" inherit the US CPI statistics — attributing a measurement to a
# release it was never made on, in a client document.
US_ONLY = "USD"
WATCHED = {
    "Non-Farm Employment Change": "employment",
    "Unemployment Rate": "employment",
    "Average Hourly Earnings m/m": "employment",
    "CPI m/m": "cpi", "CPI y/y": "cpi", "Core CPI m/m": "cpi",
    "Core PCE Price Index m/m": "pce", "PCE Price Index m/m": "pce",
    "PPI m/m": "ppi", "Core PPI m/m": "ppi",
    "Retail Sales m/m": "retail", "Core Retail Sales m/m": "retail",
    "Federal Funds Rate": "fomc", "FOMC Statement": "fomc",
    "FOMC Economic Projections": "fomc", "FOMC Press Conference": "fomc",
    "FOMC Meeting Minutes": "fomc",
}

# Which measured results are SAFE to print. The adversarial review of the event study
# was explicit: the FOMC block rests on five dates all inside 2026 and its apparent
# 2.67x collapses to 1.29x once compared with a 2026 baseline — publish neither. The
# 24-hour (fix-to-fix) employment ratio has a bootstrap CI that nearly touches 1.0, so
# it is not a headline. The release-window figures are the ones that survived.
PUBLISHABLE = {"employment", "cpi", "pce", "ppi", "retail"}

# A stability claim needs enough buckets to be a claim. Four 5-year buckets is 20
# years of history; below that, "consistent across every bucket" is not evidence.
MIN_BUCKETS_FOR_CLAIM = 4

# Stability across ANCHORED, EQUAL-WIDTH, DISJOINT buckets is what the report leads
# with, because it is a diagnostic a reader can apply without trusting a p-value: a
# real effect barely moves when the slicing changes; one that wanders was never
# established, however good a single cut looked.
#
# This replaced a single full-sample ratio, which was actively misleading. Quoted
# alone, CPI reads 1.07x ("no effect"); cut to the recent decade it reads 1.20x
# ("a real premium"); and neither is the answer. The answer is that CPI runs
# 0.90-1.30 with no pattern and every interval containing 1.0, while the employment
# report sits above 1.0 in all seven buckets. Only the second is a finding.


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=7.5)
    ax.grid(axis="y", color="#E8E8E8", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def week_ahead(asof: date, days: int = 7) -> list:
    """High-impact events from the free calendar feed, asof..asof+days.

    The feed carries the CURRENT WEEK ONLY. Run on a Monday it gives the week; run on
    a Friday it gives what is left of it. That is a property of the source, not a bug
    to code around, and the report states the window it actually covers."""
    # Distinguish "nothing scheduled" from "could not find out". An empty table in a
    # client report otherwise reads as a quiet week, which is a different and wrong
    # message. The feed rate-limits under load, so this is not hypothetical.
    try:
        econ._feed()
        reachable = True
    except econ.FeedUnavailable:
        reachable = False
    if not reachable:
        return None
    rows = []
    for i in range(days + 1):
        d = asof + timedelta(days=i)
        for e in econ.fetch_day(d, impacts=("High",)):
            title = (e.get("title") or "").strip()
            rows.append({
                "date": d, "day": d.strftime("%a %d %b"), "time": e.get("time_et", ""),
                "country": e.get("country", ""), "title": title,
                "forecast": e.get("forecast", ""), "previous": e.get("previous", ""),
                "kind": (WATCHED.get(title, "")
                         if (e.get("country") or "").upper() == US_ONLY else ""),
            })
    return rows


def event_behaviour() -> list:
    """The measured release-window behaviour, filtered to what survived review."""
    if not EVENT_STUDY.exists():
        return []
    d = json.loads(EVENT_STUDY.read_text(encoding="utf-8"))
    blocks = (d.get("results") or {}).get("am_to_pm_intraday") or {}
    key = {"employment": "Employment report", "cpi": "CPI", "pce": "PCE",
           "ppi": "PPI", "retail": "Retail sales"}
    out = []
    for name, b in blocks.items():
        kind = ("employment" if "Employment" in name else
                "cpi" if name.startswith("CPI") else
                "pce" if name.startswith("PCE") else
                "ppi" if name.startswith("PPI") else
                "retail" if name.startswith("Retail") else "")
        if kind not in PUBLISHABLE:
            continue
        ev = b.get("mean_abs_event_pct")
        base = b.get("mean_abs_baseline_matched_pct") or b.get("mean_abs_baseline_pct")
        if not ev or not base:
            continue
        fine = b.get("by_era_fine") or {}
        buckets = [{"label": k, "ratio": v.get("ratio"), "n": v.get("n")}
                   for k, v in fine.items()]
        vals = [x["ratio"] for x in buckets if x["ratio"]]
        recent = (b.get("recent") or {}).get("ratio_matched")
        out.append({"kind": kind, "label": key.get(kind, name),
                    "n": int(b.get("n_events", 0)),
                    "event_pct": float(ev), "base_pct": float(base),
                    "ratio": float(ev) / float(base),
                    "recent": recent,
                    "buckets": buckets,
                    "min_bucket": min(vals) if vals else None,
                    "max_bucket": max(vals) if vals else None,
                    "n_buckets": len(vals),
                    # The headline test: above 1 in EVERY bucket — but only meaningful
                    # with enough buckets to be a test at all. PPI has ALFRED vintages
                    # only from 2011, so it fills two buckets; "elevated in both" is
                    # a coin flip dressed as consistency, and shown beside the
                    # employment report's seven-for-seven it would read as equivalent
                    # evidence. It is not.
                    "always_elevated": bool(len(vals) >= MIN_BUCKETS_FOR_CLAIM
                                            and min(vals) > 1.0),
                    "too_short_to_judge": len(vals) < MIN_BUCKETS_FOR_CLAIM,
                    "significant": bool((b.get("recent") or {}).get("significant")),
                    "window": b.get("window", "")})
    return sorted(out, key=lambda r: -(r["recent"] or r["ratio"]))


def chart_event_ratio(rows: list) -> str | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.1, 1.9))
    labels = [r["label"] for r in rows]
    # The LAST DECADE, to match the table's headline column. The chart previously
    # plotted the full-sample ratio while the table led with the recent one, so the
    # same section showed two different numbers for the same release — 1.43x in the
    # bar and 1.67x in the row.
    ratios = [(r.get("recent") or r["ratio"]) for r in rows]
    # Gold fill only where the premium is consistent across every period; everything
    # else is grey, so colour carries the verdict rather than just the magnitude.
    cols = [GOLD if r.get("always_elevated") else GREY for r in rows]
    ax.barh(labels, ratios, color=cols, edgecolor=GOLDEDGE, height=0.55)
    ax.axvline(1.0, color=INK, lw=1.0)
    ax.set_xlim(0, max(1.6, max(ratios) * 1.15))
    ax.set_xlabel("release-window move ÷ an ordinary day, last 10 years", fontsize=7.5)
    for i, r in enumerate(ratios):
        ax.text(r + 0.02, i, f"{r:.2f}x", va="center", fontsize=7.5, color=INK)
    _despine(ax)
    return png(fig)


def chart_sensitivities(sens: list) -> str | None:
    if not sens:
        return None
    s = [x for x in sens][::-1]
    fig, ax = plt.subplots(figsize=(6.1, 2.2))
    labels = [f"{x['driver']}  {x['move']}" for x in s]
    vals = [x["gold_pct"] for x in s]
    cols = [GREEN if v > 0 else RED for v in vals]
    alpha = [1.0 if abs(x["t_stat"]) >= 2 else 0.35 for x in s]
    bars = ax.barh(labels, vals, color=cols, height=0.55)
    for b, a in zip(bars, alpha):
        b.set_alpha(a)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("implied move in gold (%)", fontsize=7.5)
    # Pad the axis so a value label never runs into the y-tick text. Without this the
    # largest negative bar's label collided with its own axis label in the PDF.
    span = max(abs(v) for v in vals) or 1.0
    ax.set_xlim(-span * 1.45, span * 1.45)
    for i, x in enumerate(s):
        v = x["gold_pct"]
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), i,
                f"{v:+.2f}%", va="center", fontsize=7.5, color=INK,
                ha="left" if v >= 0 else "right")
    _despine(ax)
    return png(fig)


SCENARIOS = [
    ("Dovish", "Real yields −20bp, dollar −2%, breakevens +10bp",
     {"real_yield_10y_chg_20d": -0.20, "dxy_chg_20d": -0.02,
      "breakeven_10y_chg_20d": 0.10}),
    ("Hawkish", "Real yields +25bp, dollar +2%",
     {"real_yield_10y_chg_20d": 0.25, "dxy_chg_20d": 0.02}),
    ("Risk-off", "VIX +1sd, credit +50bp, real yields −15bp, dollar +1%",
     {"vix_z_1y": 1.0, "hy_spread_chg_20d": 0.50,
      "real_yield_10y_chg_20d": -0.15, "dxy_chg_20d": 0.01}),
]


def build_payload(asof: date) -> dict:
    from src import goldfeatures
    feats, targs = goldfeatures.load()
    f = goldsens.fit(feats, targs, goldsens.DEFAULT_H)
    sens = json.loads(goldsens.sensitivities(f).to_json(orient="records")) if f else []
    attrib = goldsens.attribution(f, feats) if f else {}
    gap = goldsens.fair_value(feats, goldsens.DEFAULT_H)

    scen = []
    for name, desc, moves in SCENARIOS:
        s = goldsens.scenario(f, moves) if f else {}
        scen.append({"name": name, "desc": desc,
                     "gold_pct": s.get("gold_pct"), "band": s.get("band_1sd_pct")})

    events = week_ahead(asof)
    calendar_ok = events is not None
    events = events or []
    behaviour = event_behaviour()
    by_kind = {b["kind"]: b for b in behaviour}
    for e in events:
        b = by_kind.get(e["kind"])
        e["ratio"] = b["ratio"] if b else None

    stamp = reportkit.snapshot_stamp()
    return {
        "asof": asof.isoformat(),
        "asof_pretty": asof.strftime("%d %B %Y"),
        "window_to": (asof + timedelta(days=7)).strftime("%d %B"),
        "events": events,
        "calendar_ok": calendar_ok,
        "behaviour": behaviour,
        "sens": sens,
        "r2": f.get("r2") if f else None,
        "n_obs": f.get("n") if f else None,
        "resid_sd": (f.get("resid_sd") or 0) * 100 if f else None,
        "scenarios": scen,
        "attrib": attrib,
        "fv_gap": float(gap.iloc[-1]) if len(gap) else None,
        "fv_pctile": float((gap <= gap.iloc[-1]).mean() * 100) if len(gap) else None,
        "charts": {"events": chart_event_ratio(behaviour),
                   "sens": chart_sensitivities(sens)},
        "stamp": stamp,
        "logo": data_uri(ASSETS / "logo.png") if (ASSETS / "logo.png").exists() else "",
        "watermark": (data_uri(ASSETS / "building.jpg")
                      if (ASSETS / "building.jpg").exists() else ""),
    }


def render_html(payload: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda v, d=2: ("—" if v is None else f"{v:+.{d}f}%")
    env.filters["num"] = lambda v, d=2: ("—" if v is None else f"{v:.{d}f}")
    return env.get_template("goldreport.html").render(**payload)


def build_pdf(payload: dict, out_path) -> str:
    return render_pdf(render_html(payload), out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(OUT_DEFAULT))
    ap.add_argument("--asof", default=date.today().isoformat())
    a = ap.parse_args()
    payload = build_payload(date.fromisoformat(a.asof))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(payload, out)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    if not payload["calendar_ok"]:
        print("  WARNING: calendar feed unreachable — the report says so rather than "
              "showing an empty week")
    print(f"  {len(payload['events'])} high-impact events in the window")
    print(f"  {len(payload['sens'])} driver sensitivities, R2 {payload['r2']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
