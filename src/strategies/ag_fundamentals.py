"""Agriculture fundamentals strategy — USDA supply/demand (the real S&D layer) plus
report-calendar event risk; the fundamentals complement to COT (positioning).

Signals, all mapped onto the shared schema (base.COLUMNS):
  • Stocks-to-use (USDA PS&D — free, no key): each crop's marketing-year
    stocks-to-use vs its last 20 years — ≤ 25th pctile = Tight S&D (bullish),
    ≥ 75th = Ample (bearish). The headline fundamental.
  • Report event risk: an upcoming USDA release within 14 days (USDA calendar).
  • NASS grain stocks (free key): supplementary US stocks-tightness percentile.

Thin wrapper over agdata (fetch + cache), same shape as cot.py.
"""
from __future__ import annotations

import pandas as pd

from .. import agdata
from ..universe import name, asset, INSTRUMENTS
from .base import frame

STRATEGY = "AG Fundamentals"
EVENT_WINDOW_DAYS = 14       # flag USDA reports landing within this horizon (~2wk)
PCTL_WINDOW = 20             # years of history for the percentile rankings
TIGHT_PCTL = 25.0
AMPLE_PCTL = 75.0


def _psd_rows() -> list:
    try:
        df = pd.read_parquet(agdata.PSD_FILE)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    scored = []
    for tkr, g in df.groupby("ticker"):
        if tkr not in INSTRUMENTS:
            continue
        g = g.sort_values("market_year").tail(PCTL_WINDOW)
        if len(g) < 5:
            continue
        last = g.iloc[-1]
        pctl = float(g["stu"].rank(pct=True).iloc[-1] * 100.0)
        scope = str(last["scope"])
        if pctl <= TIGHT_PCTL:
            sig, dirn = "Tight S&D", 1
        elif pctl >= AMPLE_PCTL:
            sig, dirn = "Ample S&D", -1
        else:
            sig, dirn = "—", 0
        scored.append((abs(pctl - 50.0), {
            "strategy": STRATEGY,
            "market": f"{name(tkr)} · {asset(tkr)}",
            "instruments": tkr,
            "signal": sig,
            "direction": dirn,
            "metric": round(float(last["stu"]), 1),
            "metric_label": f"{scope} stocks-to-use %",
            "level": float(last["ending_stocks"]),
            "context": (f"{scope} stocks-to-use {last['stu']:.1f}% ({int(last['market_year'])}) · "
                        f"{pctl:.0f}th pctile vs {int(g['market_year'].min())}–{int(g['market_year'].max())}"),
        }))
    scored.sort(key=lambda x: x[0], reverse=True)   # most extreme first
    return [r for _, r in scored]


def _nass_rows() -> list:
    try:
        df = pd.read_parquet(agdata.NASS_FILE)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    rows = []
    for tkr, g in df.groupby("ticker"):
        if tkr not in INSTRUMENTS or len(g) < 2:
            continue
        g = g.sort_values("year")
        last, prev = g.iloc[-1], g.iloc[-2]
        if prev["value"] <= 0:
            continue
        yoy = (last["value"] / prev["value"] - 1.0) * 100.0   # YoY, not a trend-biased level percentile
        if yoy >= 5.0:
            sig, dirn = "Stocks building", -1
        elif yoy <= -5.0:
            sig, dirn = "Stocks drawing", 1
        else:
            sig, dirn = "—", 0
        rows.append({
            "strategy": STRATEGY,
            "market": f"{name(tkr)} · {asset(tkr)}",
            "instruments": tkr,
            "signal": sig,
            "direction": dirn,
            "metric": round(yoy, 1),
            "metric_label": "US stocks YoY % (NASS)",
            "level": float(last["value"]),
            "context": (f"NASS {str(last['commodity']).title()} Dec-1 stocks {last['value']:,.0f} bu "
                        f"({int(last['year'])}) · {yoy:+.1f}% vs {int(prev['year'])}"),
        })
    return rows


def _esr_rows() -> list:
    try:
        df = pd.read_parquet(agdata.ESR_FILE)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    rows = []
    for r in df.itertuples(index=False):
        if r.ticker not in INSTRUMENTS or r.yoy != r.yoy:   # skip NaN
            continue
        if r.yoy >= 10.0:
            sig, dirn = "Demand ahead", 1
        elif r.yoy <= -10.0:
            sig, dirn = "Demand behind", -1
        else:
            sig, dirn = "—", 0
        wk = pd.Timestamp(r.week)
        rows.append({
            "strategy": STRATEGY,
            "market": f"{name(r.ticker)} · {asset(r.ticker)}",
            "instruments": r.ticker,
            "signal": sig,
            "direction": dirn,
            "metric": round(float(r.yoy), 1),
            "metric_label": "exports YoY % (FAS)",
            "level": float(r.commit),
            "context": (f"MY{int(r.market_year)} export commitments {r.commit / 1e6:.1f} MMT · "
                        f"{r.yoy:+.1f}% vs year-ago (wk {wk:%d %b})"),
        })
    return rows


def _event_rows(today: pd.Timestamp) -> list:
    try:
        cal = agdata.report_calendar()
    except Exception:
        return []
    horizon = today + pd.Timedelta(days=EVENT_WINDOW_DAYS)
    upcoming = cal[(cal["date"] >= today) & (cal["date"] <= horizon)]
    if upcoming.empty:
        return []
    rows = []
    for tkr, scopes in agdata.SCOPES.items():
        if tkr not in INSTRUMENTS:
            continue
        hits = upcoming[upcoming["scope"].isin(scopes)]
        if hits.empty:
            continue
        nxt = hits.iloc[0]
        days = int((nxt["date"].normalize() - today).days)
        rows.append({
            "strategy": STRATEGY,
            "market": f"{name(tkr)} · {asset(tkr)}",
            "instruments": tkr,
            "signal": "Event risk",
            "direction": 0,
            "metric": days,
            "metric_label": "days to USDA report",
            "level": float("nan"),
            "context": f"{nxt['report']} on {nxt['date']:%d %b %Y}",
        })
    return rows


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        agdata.compute(force=False)        # PS&D + calendar always; NASS if key set
    except Exception:
        pass
    today = pd.Timestamp.now().normalize()
    rows = _psd_rows() + _esr_rows() + _nass_rows() + _event_rows(today)
    return frame(rows)
