"""Economic surprise index (BASIS · Macro Rate Radar).

What moves the front end intraday is not the LEVEL of inflation or growth — it is the
difference between the print and what was expected. This module turns the free
FairEconomy calendar feed that src/econ.py already pulls (it carries `forecast` and
`actual` side by side) into a Citi-ESI-style standardised surprise index per currency
bloc, split into inflation and growth/labour sub-indices.

Read this before relying on it
------------------------------
THE INDEX CANNOT BE BACKFILLED. The feed publishes the CURRENT WEEK only — the
`lastweek` and `nextweek` endpoints do not exist — and no free source carries a history
of consensus forecasts. So this store starts empty and accrues forward, one week at a
time. Practical consequences:

  * It needs roughly 3 months of accrual before the per-event standardisation has enough
    observations to mean anything. `readiness()` reports exactly how far along it is, and
    `index()` refuses to return a scaled number before then rather than emitting a
    confident-looking figure built on four data points.
  * The store is APPEND-ONLY and keyed by event identity. A re-run that sees the same
    event again must not double-count it, and a settled surprise must never be silently
    rewritten — the same discipline the signal ledger learned the hard way after a wedged
    morning re-marked ten years of track record.
  * Because it only records what it saw, a week where the machine was off is a permanent
    hole. `gaps()` surfaces those rather than letting the index quietly understate.

If a backfillable alternative is ever needed, the honest one is a vintage-momentum index
off ALFRED (is incoming data beating its own prior vintage?) — different thing, not
consensus surprise, but reconstructible to the 1990s.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "data" / "macro_surprise"
_EVENTS = _STORE / "events.json"
_RUNS = _STORE / "runs.json"

_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Currency -> the bank whose curve the surprise actually trades.
BLOC = {"USD": "FED", "EUR": "ECB", "GBP": "BOE"}

# Minimum observations of a given event type before its surprises can be standardised.
MIN_OBS = 6
# Observations across the whole bloc before the index is considered meaningful.
MIN_BLOC_OBS = 40
# Half-life of an event's weight in the rolling index, in days (Citi's ESI uses ~3m).
HALF_LIFE_D = 45.0
WINDOW_D = 120


# ── event classification ────────────────────────────────────────────────────────────
_INFLATION = ("cpi", "ppi", "inflation", "price index", "hicp", "rpi", "deflator",
              "prices", "wage", "earnings", "unit labor")
_LABOUR = ("payroll", "employment", "unemployment", "jobless", "claims", "jobs",
           "nfp", "labor", "labour")
_GROWTH = ("gdp", "pmi", "ism", "retail sales", "industrial production", "durable",
           "confidence", "sentiment", "orders", "trade balance", "housing", "starts",
           "spending", "output", "production", "business climate", "ifo", "zew")


def classify(title: str) -> str:
    """inflation | labour | growth | other. Order matters: 'Average Hourly Earnings' is an
    inflation signal that lives inside the payrolls release, and the desk trades it that
    way, so the inflation keywords are tested first."""
    t = (title or "").lower()
    for kw in _INFLATION:
        if kw in t:
            return "inflation"
    for kw in _LABOUR:
        if kw in t:
            return "labour"
    for kw in _GROWTH:
        if kw in t:
            return "growth"
    return "other"


# Events where a HIGHER print means LOWER growth/looser policy — the sign has to be
# flipped before it goes into an index where positive = economy beating expectations.
# 'claims' on its own covers the feed's actual titles — it publishes "Unemployment Claims",
# not "Jobless Claims", so a more specific keyword list silently missed the sign flip and
# let a bad claims number read as good news.
_INVERTED = ("unemployment rate", "claims", "claimant count")


def is_inverted(title: str) -> bool:
    t = (title or "").lower()
    return any(kw in t for kw in _INVERTED)


# ── number parsing ──────────────────────────────────────────────────────────────────
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_value(raw) -> float | None:
    """'3.2%' -> 3.2, '250K' -> 250000, '-1.5' -> -1.5, '<0.1%' -> 0.1, '' -> None.

    Deliberately keeps percent values in percent (3.2, not 0.032): every consumer here is
    comparing like with like, and rescaling would only invite a units bug."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in ("N/A", "NA", "-", "--"):
        return None
    s = s.replace(",", "").replace("%", "").replace("$", "").replace("<", "").replace(">", "")
    m = re.match(r"^(-?\d*\.?\d+)\s*([KkMmBbTt])?$", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if m.group(2):
        v *= _SUFFIX[m.group(2).lower()]
    return v


# ── store ───────────────────────────────────────────────────────────────────────────
def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _save(path: Path, blob) -> None:
    _STORE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)                       # atomic-ish: never leave a half-written store


def event_key(country: str, title: str, when: str) -> str:
    """Identity of one release. Country + title + calendar date: the feed's own ids are
    not stable across pulls, and (country, title, date) is unique in practice because a
    given indicator prints once per country per day."""
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    return f"{(country or '').upper()}|{t}|{when}"


@dataclass
class Surprise:
    key: str
    country: str
    title: str
    when: str                # ISO date of the release
    actual: float
    forecast: float
    raw_surprise: float      # actual − forecast, sign-corrected so +ve = beat
    category: str
    bloc: str


def fetch_week() -> list[dict]:
    """This week's calendar, raw. Returns [] on any problem — a dead feed must not raise
    into a scheduled refresh."""
    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
        return json.loads(urllib.request.urlopen(req, timeout=25).read())
    except Exception:
        return []


def refresh(impacts=("High", "Medium")) -> dict:
    """Pull the current week and append any newly-settled surprises to the store.

    Append-only by construction: an event already recorded is left ALONE, even if the
    feed later revises its `actual`. A settled surprise is a historical fact about what
    the market learned that morning; letting a revision rewrite it would corrupt the
    standardisation that everything downstream depends on."""
    events = _load(_EVENTS, {})
    runs = _load(_RUNS, [])
    raw = fetch_week()
    added, skipped, unusable = 0, 0, 0
    et = ZoneInfo("America/New_York")

    for e in raw:
        if e.get("impact") not in impacts:
            continue
        ccy = (e.get("country") or "").upper()
        if ccy not in BLOC:
            continue
        actual = parse_value(e.get("actual"))
        forecast = parse_value(e.get("forecast"))
        if actual is None or forecast is None:
            unusable += 1                    # not printed yet, or no consensus published
            continue
        try:
            when = datetime.fromisoformat(e["date"]).astimezone(et).date().isoformat()
        except Exception:
            continue
        title = (e.get("title") or "").strip()
        k = event_key(ccy, title, when)
        if k in events:
            skipped += 1
            continue
        diff = actual - forecast
        if is_inverted(title):
            diff = -diff
        events[k] = {"country": ccy, "title": title, "when": when, "actual": actual,
                     "forecast": forecast, "raw_surprise": diff,
                     "category": classify(title), "bloc": BLOC[ccy],
                     "recorded": date.today().isoformat()}
        added += 1

    _save(_EVENTS, events)
    runs.append({"run": date.today().isoformat(), "added": added, "seen": len(raw)})
    _save(_RUNS, runs[-500:])
    return {"added": added, "already_had": skipped, "not_yet_printed": unusable,
            "total_stored": len(events)}


def all_events() -> list[dict]:
    return sorted(_load(_EVENTS, {}).values(), key=lambda r: r.get("when", ""))


# ── standardisation + index ─────────────────────────────────────────────────────────
def _event_type(rec: dict) -> str:
    """Grouping key for standardisation. Must normalise whitespace the same way
    event_key does, or the same indicator splits into two buckets and neither ever
    reaches MIN_OBS."""
    title = re.sub(r"\s+", " ", rec["title"].strip().lower())
    return f"{rec['country']}|{title}"


def _scales(events: list[dict]) -> dict:
    """Per-event-type standard deviation of past surprises. This is what makes a 0.2pp CPI
    miss and a 60k payrolls miss comparable — each is expressed in units of its own
    historical noise."""
    buckets: dict[str, list[float]] = {}
    for r in events:
        buckets.setdefault(_event_type(r), []).append(r["raw_surprise"])
    out = {}
    for k, vals in buckets.items():
        if len(vals) < MIN_OBS:
            continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            out[k] = sd
    return out


def readiness() -> dict:
    """How usable the index is yet. The page shows this instead of an index while the
    store is still filling — an unscaled 'surprise index' built on a handful of events is
    worse than no index, because it looks authoritative."""
    events = all_events()
    scales = _scales(events)
    by_bloc: dict[str, int] = {}
    for r in events:
        by_bloc[r["bloc"]] = by_bloc.get(r["bloc"], 0) + 1
    first = events[0]["when"] if events else None
    weeks = 0
    if first:
        weeks = max(0, (date.today() - date.fromisoformat(first)).days // 7)
    return {"total": len(events), "by_bloc": by_bloc,
            "types_standardisable": len(scales),
            "first_event": first, "weeks_accrued": weeks,
            "ready": {b: by_bloc.get(b, 0) >= MIN_BLOC_OBS for b in BLOC.values()},
            "note": ("Accrues forward only — the free calendar feed carries the current "
                     "week alone, and no free source has consensus-forecast history.")}


def index(bloc: str = "FED", category: str | None = None,
          asof: date | None = None) -> dict:
    """Rolling standardised surprise index for one bloc.

    Exponentially decayed sum of standardised surprises over the trailing window, so a
    hot payrolls print dominates for a few weeks and then fades — the same construction
    as the Citi ESI. Positive = data is beating consensus = hawkish pressure on the curve.

    Returns {'ok', 'value', 'n', 'reason'} and REFUSES to produce a value until the store
    has enough history to standardise honestly."""
    asof = asof or date.today()
    events = all_events()
    scales = _scales(events)
    rel = [r for r in events if r["bloc"] == bloc
           and (category is None or r["category"] == category)]
    in_window = [r for r in rel
                 if 0 <= (asof - date.fromisoformat(r["when"])).days <= WINDOW_D]

    usable = [r for r in in_window if _event_type(r) in scales]
    bloc_total = sum(1 for r in events if r["bloc"] == bloc)
    if bloc_total < MIN_BLOC_OBS:
        return {"ok": False, "value": None, "n": len(in_window),
                "reason": (f"only {bloc_total} {bloc} events stored; the index needs "
                           f"{MIN_BLOC_OBS} before its standardisation means anything")}
    if not usable:
        return {"ok": False, "value": None, "n": 0,
                "reason": "no events in the window have enough history to standardise"}

    num = 0.0
    for r in usable:
        age = (asof - date.fromisoformat(r["when"])).days
        w = 0.5 ** (age / HALF_LIFE_D)
        num += w * (r["raw_surprise"] / scales[_event_type(r)])
    return {"ok": True, "value": round(num, 3), "n": len(usable), "reason": ""}


def gaps(min_days: int = 10) -> list[dict]:
    """Stretches where nothing was recorded. Because the store only knows what it saw, a
    machine that was off for a fortnight leaves a hole that would otherwise silently
    flatten the index."""
    runs = _load(_RUNS, [])
    days = sorted({r["run"] for r in runs})
    out = []
    for a, b in zip(days, days[1:]):
        da, db = date.fromisoformat(a), date.fromisoformat(b)
        if (db - da).days >= min_days:
            out.append({"from": a, "to": b, "days": (db - da).days})
    if days:
        last = date.fromisoformat(days[-1])
        if (date.today() - last).days >= min_days:
            out.append({"from": days[-1], "to": date.today().isoformat(),
                        "days": (date.today() - last).days})
    return out


def recent(bloc: str | None = None, n: int = 25) -> list[dict]:
    """Latest recorded surprises, most recent first — the table under the index chart."""
    evs = [r for r in all_events() if bloc is None or r["bloc"] == bloc]
    scales = _scales(all_events())
    out = []
    for r in reversed(evs[-n:]):
        sd = scales.get(_event_type(r))
        out.append({**r, "z": (round(r["raw_surprise"] / sd, 2) if sd else None)})
    return out
