"""Today's major (high-impact) economic releases.

Pulls the free FairEconomy / ForexFactory weekly calendar feed — the same major
figures the Bloomberg ECO page tracks — and returns today's high-impact events in
US-Eastern time. Stdlib only (no Bloomberg, no deps); returns [] on any problem
(weekends / holidays legitimately have nothing). Mirrors the Morning Coffee
project's fetch_econ_calendar, plus the live `actual` once a figure has printed.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# The feed rate-limits (HTTP 429) under repeated calls, and every consumer here —
# the Hot Sheet, Morning Coffee, the surprise accrual, the gold report — hits the
# same URL. A short disk cache makes them share one fetch instead of competing, and
# keeps a usable copy when the host says no.
_CACHE = Path(__file__).resolve().parents[1] / "data" / "econ_calendar_cache.json"
_TTL_SECONDS = 1800


class FeedUnavailable(RuntimeError):
    """The calendar could not be reached AND no cache was usable.

    Raised rather than returning [] so a caller can tell "nothing is scheduled" from
    "we could not find out" — a distinction that matters in a client report, where an
    empty table otherwise reads as a quiet week."""


def _feed(force: bool = False) -> list:
    """The raw weekly feed, disk-cached. Falls back to a stale cache before failing."""
    if not force and _CACHE.exists():
        age = time.time() - _CACHE.stat().st_mtime
        if age <= _TTL_SECONDS:
            try:
                return json.loads(_CACHE.read_text(encoding="utf-8"))
            except Exception:
                pass
    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=20).read())
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if _CACHE.exists():                    # stale beats nothing
            try:
                return json.loads(_CACHE.read_text(encoding="utf-8"))
            except Exception:
                pass
        raise FeedUnavailable("economic calendar feed unreachable and no cache")


def fetch_day(day, impacts=("High",)) -> list:
    """[{time_et, country, title, actual, forecast, previous}, ...] for `day`
    (a date), sorted by release time. The feed only covers the CURRENT week —
    days outside it legitimately return []."""
    try:
        data = _feed()
    except FeedUnavailable:
        return []
    et = ZoneInfo("America/New_York")
    rows = []
    for e in data:
        if e.get("impact") not in impacts:
            continue
        try:
            dt = datetime.fromisoformat(e["date"]).astimezone(et)
        except Exception:
            continue
        if dt.date() != day:
            continue
        rows.append({"_dt": dt,
                     "time_et": dt.strftime("%H:%M ET"),
                     "country": e.get("country", ""),
                     "title": (e.get("title") or "").strip(),
                     "actual": (e.get("actual") or "").strip(),
                     "forecast": (e.get("forecast") or "").strip(),
                     "previous": (e.get("previous") or "").strip()})
    rows.sort(key=lambda r: r["_dt"])
    for r in rows:
        r.pop("_dt", None)
    return rows


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
