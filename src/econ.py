"""Today's major (high-impact) economic releases.

Pulls the free FairEconomy / ForexFactory weekly calendar feed — the same major
figures the Bloomberg ECO page tracks — and returns today's high-impact events in
US-Eastern time. Stdlib only (no Bloomberg, no deps); returns [] on any problem
(weekends / holidays legitimately have nothing). Mirrors the Morning Coffee
project's fetch_econ_calendar, plus the live `actual` once a figure has printed.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch_today(impacts=("High",)) -> list:
    """[{time, country, title, actual, forecast, previous}, ...] for today, sorted
    by release time. `impacts` filters the feed's impact level."""
    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    except Exception:
        return []
    et = ZoneInfo("America/New_York")
    today = datetime.now(et).date()
    rows = []
    for e in data:
        if e.get("impact") not in impacts:
            continue
        try:
            dt = datetime.fromisoformat(e["date"]).astimezone(et)
        except Exception:
            continue
        if dt.date() != today:
            continue
        rows.append({"_dt": dt,
                     "time": dt.strftime("%I:%M %p").lstrip("0"),     # '8:30 AM'
                     "country": e.get("country", ""),
                     "title": (e.get("title") or "").strip(),
                     "actual": (e.get("actual") or "").strip(),
                     "forecast": (e.get("forecast") or "").strip(),
                     "previous": (e.get("previous") or "").strip()})
    rows.sort(key=lambda r: r["_dt"])
    for r in rows:
        r.pop("_dt", None)
    return rows
