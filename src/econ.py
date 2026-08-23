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
# HOW STALE IS TOO STALE. The feed carries the CURRENT WEEK only, so a cache older
# than a few days is not a degraded copy of this week's calendar — it is last week's,
# and serving it silently is worse than admitting the feed is down. Without a ceiling
# the stale fallback defeated the very guard it was built for: goldreport.week_ahead
# probes _feed() to decide whether to print "no high-impact releases scheduled" or
# "the calendar was unreachable", and an ancient cache made a dead feed look alive.
_MAX_STALE_SECONDS = 3 * 24 * 3600
# Today's board refreshes faster than the week-ahead schedule does.
_LIVE_TTL_SECONDS = 240


class FeedUnavailable(RuntimeError):
    """The calendar could not be reached AND no cache was usable.

    Raised rather than returning [] so a caller can tell "nothing is scheduled" from
    "we could not find out" — a distinction that matters in a client report, where an
    empty table otherwise reads as a quiet week."""


def _looks_like_feed(data) -> bool:
    """A list of event dicts carrying at least a date. Anything else is not the feed.

    The cache used to accept whatever parsed as JSON, so an error page that happened
    to be valid JSON became a sticky cached "calendar" and every later call raised an
    AttributeError from deep inside the parse loop. A test fixture carrying a single
    event titled "X" also reached this file once and would have printed a fabricated
    release into a client PDF."""
    if not isinstance(data, list) or not data:
        return False
    head = data[0]
    return isinstance(head, dict) and "date" in head


def _read_cache():
    try:
        data = json.loads(_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if _looks_like_feed(data) else None


def _feed(force: bool = False, max_age: float | None = None) -> list:
    """The raw weekly feed, disk-cached.

    Falls back to a stale cache before failing, but only up to _MAX_STALE_SECONDS —
    past that the feed is reported as unavailable rather than quietly served."""
    ttl = _TTL_SECONDS if max_age is None else float(max_age)
    if not force and _CACHE.exists():
        age = time.time() - _CACHE.stat().st_mtime
        if age <= ttl:
            cached = _read_cache()
            if cached is not None:
                return cached
    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=20).read())
        if not _looks_like_feed(data):
            raise ValueError("calendar feed returned an unrecognised payload")
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if _CACHE.exists():
            age = time.time() - _CACHE.stat().st_mtime
            cached = _read_cache()
            if cached is not None and age <= _MAX_STALE_SECONDS:
                return cached
            if cached is not None:
                raise FeedUnavailable(
                    f"economic calendar feed unreachable; cache is "
                    f"{age / 86400:.1f} days old and covers a week that has passed")
        raise FeedUnavailable("economic calendar feed unreachable and no usable cache")


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
    # Via _feed(), so this shares the disk cache, the payload validation and the
    # staleness ceiling. It used to call the network directly — which meant the
    # landing board went silently empty whenever the feed rate-limited, and the
    # caching work missed its single biggest consumer.
    # _LIVE_TTL, not the 30-minute default: this feeds the landing board, where the
    # `actual` column is the whole point. A release that printed at 08:30 read as
    # still-pending until 09:00 on the shared cache.
    try:
        data = _feed(max_age=_LIVE_TTL_SECONDS)
    except FeedUnavailable:
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
