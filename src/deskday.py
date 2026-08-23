"""Desk day export — the day's FICC events (fundamental reports, central-bank
decisions, the majors on the economic calendar, futures/options expiries) as a
tiny JSON for OTHER report pipelines: the Morning Coffee PDF prints it under the
Hot Sheet (Ben, 2026-08-22). Same assembly as the BASIS landing board's FICC
panel (app.py render_landing), without Streamlit. Written by the morning compute.

    data/signals/desk_calendar.json = {"date": "2026-08-24", "label": "Mon 24 Aug",
                                       "rows": [{"time", "label", "sub", "kind"}, ...]}
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from . import repcal, universe

SIG_DIR = Path(__file__).resolve().parents[1] / "data" / "signals"
FILE = SIG_DIR / "desk_calendar.json"


def trading_day(d: date | None = None) -> date:
    """Today, or on a weekend the next weekday (the landing board's rule)."""
    d = d or date.today()
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _macro(day: date) -> list:
    """High-impact economic prints for `day` (FairEconomy feed; current week only)."""
    from . import econ
    out = []
    try:
        rows = econ.fetch_day(day)
    except Exception:
        return out
    for r in rows:
        lbl = f"{r['country']} {r['title']}"
        if r.get("actual"):
            lbl += f" · {r['actual']}" + (f" (vs {r['forecast']}e)" if r.get("forecast") else "")
        tip = f"{r['title']} ({r['country']})"
        if r.get("forecast"):
            tip += f" · fcst {r['forecast']}"
        if r.get("previous"):
            tip += f" · prev {r['previous']}"
        out.append({"date": day, "icon": "", "label": lbl, "auto": False,
                    "tip": tip, "time": r.get("time_et")})
    return out


def _expiries(day: date) -> list:
    """Products whose next futures/options expiry lands on `day` — one row per
    product+kind (indices live in the universe twice: futures generic + cash), big
    shared expiries grouped per sector."""
    from . import expiries as _exp
    key = f"{day:%a %d %b %Y}"
    bucket, seen = {}, set()
    for t, meta in universe.INSTRUMENTS.items():
        name, asset = meta[0], meta[2]
        try:
            ex = _exp.describe(t, asset, day)
        except Exception:
            continue
        for kind, dkey, tkey in (("future", "fut", "fut_time"), ("options", "opt", "opt_time")):
            if ex.get(dkey) == key and (name, kind) not in seen:
                seen.add((name, kind))
                bucket.setdefault((asset, kind, ex.get(tkey) or ""), []).append(name)
    ev = []
    for (asset, kind, tm), names in bucket.items():
        base = {"date": day, "icon": "", "auto": False, "time": tm or None}
        if len(names) <= 2:
            for n in names:
                ev.append({**base, "label": f"{n} — {kind} expiry",
                           "tip": f"{n}: next {kind} expiry"})
        else:
            ev.append({**base, "label": f"{asset} — {kind} expiries (×{len(names)})",
                       "tip": ", ".join(sorted(names))})
    return ev


def events_for(day: date) -> list:
    evs = [e for e in repcal.calendar_events() if e.get("date") == day]
    evs += _macro(day)
    evs += _expiries(day)
    return sorted(evs, key=lambda e: repcal._time_key(e, day))


def export(day: date | None = None) -> int:
    """Write FILE for `day` (default: today / next trading day). Returns row count."""
    day = trading_day(day)
    rows = []
    for e in events_for(day):
        kind = "expiry" if "expir" in str(e.get("label", "")).lower() else (
            "macro" if e.get("color") == "#546E7A" or e.get("tip", "").startswith(("CPI", "PPI"))
            else "report")
        label = str(e.get("label", "")).strip()
        sub = str(e.get("sub") or e.get("tip") or "").strip()
        if kind == "expiry" and "," not in sub:      # single-product expiry: tip just repeats the label
            sub = ""
        if sub == label:
            sub = ""
        rows.append({"time": e.get("time") or "", "label": label, "sub": sub,
                     "kind": kind, "emails_desk": bool(e.get("auto"))})
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps({"date": day.isoformat(), "label": f"{day:%a %d %b}",
                                "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(rows)
