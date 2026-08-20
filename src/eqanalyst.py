"""Analyst view — the sell-side consensus layer for the BASIS Equities side.

What the Street currently publishes on a name, gathered FREE from Yahoo Finance through
the yfin adapter (Bloomberg-style tickers at the seams, zero Terminal hits):

  consensus   recommendationMean + the analyst count, and the Street's own label
              ("Buy", "Hold", …). Yahoo's mean runs 1 = most positive … 5 = least;
              eqfunda's BEST_ANALYST_RATING is the SAME number inverted (6 − mean, 5 best,
              the Bloomberg convention), so both are exposed here and every caller must
              say which scale it is showing.
  counts      the strongBuy / buy / hold / sell / strongSell split for the last four
              monthly vintages (0m / -1m / -2m / -3m) — the *trend* in the split matters
              more than its level.
  targets     mean / high / low / median price target, the spot they were pulled against
              and the implied move to the mean target, in the LISTING currency (a .L name
              quotes GBp and so do its targets, so the ratio is unit-safe).
  actions     the upgrades/downgrades feed — firm, from-grade, to-grade, action and the
              new/prior price target. NOTE: Yahoo only carries this grade history for
              US-listed names; European/Asian lines return consensus and targets but an
              EMPTY actions feed. That is a source limit, not a bug — the UI says so.

Cached to data/equities/analyst.json per ticker behind a staleness guard (same pattern as
eqetf.fund_info): a page opens instantly off disk, only stale names re-pull, and a dead
pull always keeps the cached record rather than wiping it.

Everything here is descriptive — it reports what third-party analysts have published. No
Streamlit; app.py owns the pages.
"""
from __future__ import annotations

import json
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd

from src import yfin

_CACHE_FILE = Path(__file__).resolve().parents[1] / "data" / "equities" / "analyst.json"
MAX_AGE_DAYS = 1          # consensus/targets move daily — a name viewed today re-pulls once
_MAX_ACTIONS = 30         # rating actions stored per name (newest first) — keeps the file small
_ACTION_MAX_DAYS = 730    # …and drops anything older than ~2y

# Yahoo's `Action` code -> plain-English label + direction (+1 more positive / -1 less / 0 flat)
_ACTIONS = {
    "up":   ("Upgrade", 1),
    "down": ("Downgrade", -1),
    "init": ("Initiated", 0),
    "main": ("Maintained", 0),
    "reit": ("Reiterated", 0),
}

_TREND_PERIODS = ["0m", "-1m", "-2m", "-3m"]
_COUNT_KEYS = ["strongBuy", "buy", "hold", "sell", "strongSell"]


# ── labels / formatting ───────────────────────────────────────────────────────
def consensus_label(mean) -> str:
    """Yahoo's 1–5 consensus mean -> the Street's own summary word. 1 = most positive."""
    try:
        m = float(mean)
    except (TypeError, ValueError):
        return "—"
    if m != m:
        return "—"
    if m < 1.5:
        return "Strong buy"
    if m < 2.5:
        return "Buy"
    if m < 3.5:
        return "Hold"
    if m < 4.5:
        return "Underperform"
    return "Sell"


def action_label(code: str) -> str:
    return _ACTIONS.get(str(code or "").lower(), (str(code or "—").title(), 0))[0]


def action_direction(code: str) -> int:
    """+1 = a more positive grade, -1 = less positive, 0 = initiation / no change."""
    return _ACTIONS.get(str(code or "").lower(), ("", 0))[1]


def is_us_line(ticker: str) -> bool:
    """True for a US-listed line — the only ones Yahoo carries a grade history for."""
    sym = yfin.to_yahoo(ticker) or ""
    return bool(sym) and "." not in sym


def fmt_px(v, ccy: str = "") -> str:
    """Price / target in the listing currency ('305.93 USD', '522.90 GBp')."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return f"{f:,.2f}" + (f" {ccy}" if ccy else "")


def fmt_pct(v, signed: bool = True) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return f"{f:+,.1f}%" if signed else f"{f:,.1f}%"


# ── one name's pull ───────────────────────────────────────────────────────────
def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if f == f else np.nan


def _clean(v):
    """NaN -> None so the record round-trips through JSON."""
    f = _num(v)
    return None if f != f else float(f)


def _trend_rows(rec) -> list:
    """yfinance .recommendations frame -> [{period, strongBuy…, total, pos_pct}] newest first.
    pos_pct = (strongBuy + buy) share of the analysts covering that vintage."""
    if rec is None or getattr(rec, "empty", True) or "period" not in getattr(rec, "columns", []):
        return []
    out = []
    for p in _TREND_PERIODS:
        sub = rec[rec["period"].astype(str) == p]
        if sub.empty:
            continue
        r = sub.iloc[0]
        counts = {k: int(_num(r.get(k)) if _num(r.get(k)) == _num(r.get(k)) else 0)
                  for k in _COUNT_KEYS}
        tot = sum(counts.values())
        out.append({"period": p, **counts, "total": tot,
                    "pos_pct": (100.0 * (counts["strongBuy"] + counts["buy"]) / tot) if tot else None})
    return out


def _action_rows(ud) -> list:
    """yfinance .upgrades_downgrades frame -> [{date, firm, from, to, action, target, prior}],
    newest first, trimmed to _MAX_ACTIONS within _ACTION_MAX_DAYS."""
    if ud is None or getattr(ud, "empty", True):
        return []
    d = ud.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()].sort_index(ascending=False)
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=_ACTION_MAX_DAYS)
    d = d[d.index >= cutoff].head(_MAX_ACTIONS)
    rows = []
    for ts, r in d.iterrows():
        tgt, prior = _clean(r.get("currentPriceTarget")), _clean(r.get("priorPriceTarget"))
        rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "firm": str(r.get("Firm") or "—").strip(),
            "from": str(r.get("FromGrade") or "").strip(),
            "to": str(r.get("ToGrade") or "").strip(),
            "action": str(r.get("Action") or "").strip().lower(),
            "target": tgt or None,          # Yahoo writes 0.0 for "no target" — treat as absent
            "prior": prior or None,
        })
    return rows


def _fetch_one(ticker: str) -> dict | None:
    """One name's full analyst record, or None when unmapped / offline / Yahoo has nothing.
    Three Ticker requests (~0.5s); a failing leg leaves that section empty rather than
    killing the record."""
    sym = yfin.to_yahoo(ticker)
    if not sym or yfin.OFFLINE:
        return None
    import yfinance as yf
    try:
        t = yf.Ticker(sym)
    except Exception:
        return None
    try:
        info = t.info or {}
    except Exception:
        info = {}
    mean = _num(info.get("recommendationMean"))
    spot = _num(info.get("currentPrice"))
    if spot != spot:
        spot = _num(info.get("regularMarketPrice"))
    if spot != spot:
        spot = _num(info.get("previousClose"))
    tgt_mean = _num(info.get("targetMeanPrice"))
    upside = (tgt_mean / spot - 1.0) * 100.0 if (spot == spot and spot > 0
                                                 and tgt_mean == tgt_mean) else np.nan
    try:
        trend = _trend_rows(t.recommendations)
    except Exception:
        trend = []
    try:
        actions = _action_rows(t.upgrades_downgrades)
    except Exception:
        actions = []
    rec = {
        "ticker": ticker,
        "symbol": sym,
        "pulled": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "name": str(info.get("shortName") or info.get("longName") or "").strip() or None,
        "consensus": {
            "mean": _clean(mean),                              # 1 = most positive … 5 = least
            "score5": _clean(6.0 - mean) if mean == mean else None,   # BDP convention, 5 best
            "label": consensus_label(mean),
            "n": int(_num(info.get("numberOfAnalystOpinions")))
                 if _num(info.get("numberOfAnalystOpinions")) == _num(info.get("numberOfAnalystOpinions"))
                 else None,
        },
        "targets": {
            "ccy": str(info.get("currency") or "").strip() or None,
            "spot": _clean(spot), "mean": _clean(tgt_mean),
            "high": _clean(info.get("targetHighPrice")), "low": _clean(info.get("targetLowPrice")),
            "median": _clean(info.get("targetMedianPrice")), "upside_pct": _clean(upside),
        },
        "trend": trend,
        "actions": actions,
    }
    if rec["consensus"]["mean"] is None and not trend and not actions \
            and rec["targets"]["mean"] is None:
        return None                                            # nothing published on this name
    return rec


# ── disk cache ────────────────────────────────────────────────────────────────
def _read_cache() -> dict:
    try:
        d = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _write_cache(cache: dict) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass                                                   # a read-only disk must not break a page


def age_days(rec: dict | None) -> int | None:
    """Age of a cached record in days; None when it carries no stamp."""
    try:
        return int((pd.Timestamp.today().normalize()
                    - pd.Timestamp(str((rec or {}).get("pulled", "")))).days)
    except (TypeError, ValueError):
        return None


def _is_fresh(rec: dict | None, max_age_days: int) -> bool:
    a = age_days(rec)
    return a is not None and a < max_age_days


def _blank(ticker: str) -> dict:
    """Negative-cache stub for a name Yahoo publishes nothing on — without it every uncovered
    name (small caps, most non-US lines) would be re-requested on every single call."""
    return {"ticker": ticker, "pulled": pd.Timestamp.today().strftime("%Y-%m-%d"), "empty": True}


def has_data(rec: dict | None) -> bool:
    return bool(rec) and not rec.get("empty")


def analyst(ticker: str, force: bool = False, max_age_days: int = MAX_AGE_DAYS) -> dict | None:
    """One name's analyst record — cached on disk, re-pulled only when stale (or `force`).
    A dead pull keeps whatever was cached; None when Yahoo has nothing on the name."""
    cache = _read_cache()
    rec = cache.get(ticker)
    if not force and _is_fresh(rec, max_age_days):
        return rec if has_data(rec) else None
    fresh = _fetch_one(ticker)
    if fresh is None:
        if has_data(rec):
            return rec                                         # keep the last good pull, never wipe
        cache[ticker] = _blank(ticker)                          # remember the miss for a day
        _write_cache(cache)
        return None
    cache[ticker] = fresh
    _write_cache(cache)
    return fresh


def bulk(tickers, force: bool = False, max_age_days: int = MAX_AGE_DAYS,
         max_fetch: int = 40) -> dict:
    """{ticker: record} for a list of names, threaded. Serves cached records instantly and
    pulls at most `max_fetch` stale ones per call, so a big selection degrades into "the
    freshest names we have" rather than a minutes-long stall. One cache write per call."""
    tickers = list(dict.fromkeys(tickers))
    cache = _read_cache()
    stale = [t for t in tickers if force or not _is_fresh(cache.get(t), max_age_days)]
    stale = [t for t in stale if yfin.to_yahoo(t)][:max_fetch]
    if stale and not yfin.OFFLINE:
        got = {}
        try:
            with futures.ThreadPoolExecutor(max_workers=yfin._INFO_WORKERS) as ex:
                futs = {ex.submit(_fetch_one, t): t for t in stale}
                for f in futures.as_completed(futs):
                    try:
                        r = f.result()
                    except Exception:
                        r = None
                    t = futs[f]
                    if r is not None:
                        got[t] = r
                    elif not has_data(cache.get(t)):
                        got[t] = _blank(t)                     # remember the miss, keep good records
        except Exception:
            got = {}
        if got:
            cache.update(got)
            _write_cache(cache)
    return {t: cache[t] for t in tickers if has_data(cache.get(t))}


def refresh_stale(max_age_days: int = MAX_AGE_DAYS, limit: int = 250) -> dict:
    """Re-pull the names ALREADY in the cache that have gone stale — the snapshot's leg.
    Deliberately not a universe-wide pull: the cache holds the names the desk has actually
    looked at, so this keeps those warm for a few seconds of Yahoo traffic and nothing more."""
    cache = _read_cache()
    if not cache:
        return {"ok": True, "n": 0, "reason": "nothing cached yet"}
    # Stubs (names Yahoo publishes nothing on) are left alone — re-asking daily buys nothing.
    stale = [t for t, r in cache.items() if has_data(r) and not _is_fresh(r, max_age_days)]
    if not stale:
        return {"ok": True, "n": 0, "reason": "all cached names fresh"}
    before = len(stale)
    got = bulk(sorted(stale), max_age_days=max_age_days, max_fetch=limit)
    n = sum(1 for t in stale if _is_fresh(got.get(t), max_age_days))
    return {"ok": True, "n": n, "stale": before, "cached": len(cache)}


# ── read seams for the pages ──────────────────────────────────────────────────
def implied_upside(rec: dict | None, spot=None):
    """Implied move from spot to the MEAN target, in percent. Pass a live `spot` to
    re-base a cached record against today's price; otherwise the pull-time spot is used."""
    t = (rec or {}).get("targets") or {}
    mean = _num(t.get("mean"))
    px = _num(spot) if spot is not None else _num(t.get("spot"))
    if mean != mean or px != px or px <= 0:
        return np.nan
    return (mean / px - 1.0) * 100.0


def target_range_pos(rec: dict | None, spot=None):
    """Where spot sits inside the low–high target range, 0–100 (None when the range is
    degenerate). Reads as 'the Street's range brackets the price here'."""
    t = (rec or {}).get("targets") or {}
    lo, hi = _num(t.get("low")), _num(t.get("high"))
    px = _num(spot) if spot is not None else _num(t.get("spot"))
    if lo != lo or hi != hi or px != px or hi <= lo:
        return None
    return float(max(0.0, min(100.0, (px - lo) / (hi - lo) * 100.0)))


def trend_frame(rec: dict | None) -> pd.DataFrame:
    """The four monthly count vintages as a frame (newest first), for the tearsheet table."""
    rows = (rec or {}).get("trend") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def trend_shift(rec: dict | None):
    """Change in the positive share (strongBuy+buy) between the oldest and newest stored
    vintage, in percentage points — the 'is the Street warming or cooling' number."""
    rows = (rec or {}).get("trend") or []
    have = [r for r in rows if r.get("pos_pct") is not None]
    if len(have) < 2:
        return np.nan
    return float(have[0]["pos_pct"] - have[-1]["pos_pct"])      # rows are newest-first


def actions_frame(rec: dict | None, days: int | None = None, limit: int | None = None) -> pd.DataFrame:
    """One name's rating actions as a frame: date · firm · from · to · action · target · prior.
    Empty for non-US lines — Yahoo's grade history is US-listed only."""
    rows = (rec or {}).get("actions") or []
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].sort_values("date", ascending=False)
    if days is not None:
        d = d[d["date"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=int(days))]
    return d.head(int(limit)) if limit else d


GRADE_CHANGES = ("up", "down", "init")     # actual rating moves, vs 'main'/'reit' target tweaks


def recent_actions(tickers, names: dict | None = None, days: int = 45, limit: int = 12,
                   fetch: bool = True, max_fetch: int = 25, kinds=GRADE_CHANGES) -> pd.DataFrame:
    """The cross-name feed for the Equities home: the newest rating actions across `tickers`,
    one row per action — date · stock · firm · from → to · action · new target.

    `kinds` filters the action codes: the default keeps genuine grade changes (upgrade /
    downgrade / initiation) and drops the far more numerous 'maintained' target tweaks;
    pass None for the unfiltered feed. `names` optionally maps ticker -> display name (the
    home's company names). With fetch=False it renders purely from cache (no network)."""
    tickers = list(dict.fromkeys(tickers))
    recs = (bulk(tickers, max_fetch=max_fetch) if fetch
            else {t: r for t, r in _read_cache().items()
                  if t in set(tickers) and has_data(r)})
    keep = set(kinds) if kinds else None
    rows = []
    for t, rec in recs.items():
        d = actions_frame(rec, days=days)
        if keep is not None and not d.empty:
            d = d[d["action"].isin(keep)]
        disp = (names or {}).get(t) or (rec.get("name") or t)
        for _, r in d.iterrows():
            rows.append({"date": r["date"], "ticker": t, "stock": disp, "firm": r["firm"],
                         "from": r["from"], "to": r["to"], "action": r["action"],
                         "dir": action_direction(r["action"]), "target": r.get("target"),
                         "prior": r.get("prior"),
                         "ccy": ((rec.get("targets") or {}).get("ccy") or "")})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("date", ascending=False)
    return out.head(int(limit)) if limit else out


def coverage(tickers) -> dict:
    """How much of a list the cache can answer for, for the UI's honesty note."""
    cache = _read_cache()
    tickers = list(dict.fromkeys(tickers))
    have = [t for t in tickers if has_data(cache.get(t))]
    return {"asked": len(tickers), "cached": len(have),
            "with_actions": sum(1 for t in have if cache[t].get("actions")),
            "mappable": sum(1 for t in tickers if yfin.to_yahoo(t))}


def data_status() -> str:
    """Short 'where the analyst data comes from' label for page captions."""
    cache = {t: r for t, r in _read_cache().items() if has_data(r)}
    if not cache:
        return "no analyst pull yet — opens a name to fetch it"
    ages = [a for a in (age_days(r) for r in cache.values()) if a is not None]
    freshest = min(ages) if ages else None
    return (f"Yahoo Finance · {len(cache)} names cached"
            + (f" · newest {freshest}d old" if freshest is not None else ""))


# ── Hot Sheet provider ────────────────────────────────────────────────────────
RADAR_DAYS = 7      # the sheet is a daily screen — only this week's actions qualify
RADAR_MAX = 5       # editorial cap, matched to the sheet's per-provider cap
RADAR_HEAT = 45.0   # flat: rating actions are newsy dated events, not rankable extremes —
                    # there's no z / percentile to grade one against another, so they all
                    # sit mid-sheet and the date does the sorting


def radar_items() -> list:
    """Hot Sheet provider — this week's published rating actions (upgrades / downgrades /
    initiations) across the names already in the analyst cache, purely from disk. The cache
    holds exactly the names the Equities pages have looked at (the home's movers feed and the
    snapshot's refresh_stale keep it warm), so this is the page's own universe; fetch=False
    keeps recent_actions on its cache-only path — nothing here can pull."""
    from src import hotsheet
    cache = _read_cache()
    tickers = [t for t, r in cache.items() if has_data(r)]
    if not tickers:                                # no analyst pull yet — quiet, not broken
        return []
    feed = recent_actions(tickers, days=RADAR_DAYS, limit=RADAR_MAX, fetch=False)
    out, seen = [], set()
    for _, r in feed.iterrows():
        # dual-class listings (FOX/FOXA) carry the SAME published action once per
        # ticker — dedupe on the event itself, keep the first (highest-ranked) class
        ev = (r["stock"], r["firm"], f"{r['date']:%Y-%m-%d}", r["from"], r["to"])
        if ev in seen:
            continue
        seen.add(ev)
        if r["from"] and r["to"] and r["from"] != r["to"]:
            what = f"moved to **{r['to']}** from {r['from']}"
        elif r["action"] == "init":
            what = (f"initiated coverage at **{r['to']}**" if r["to"]
                    else "initiated coverage")
        else:
            what = f"moved to **{r['to'] or r['from']}**"
        out.append(hotsheet.item(
            tag="EQ-STREET",
            key=f"{r['ticker']}:{r['date']:%Y-%m-%d}:{r['firm']}",  # an action is a dated event — stable id
            section="Equities · Street",
            text=f"**{r['stock']}**: {r['firm']} {what} ({r['date']:%d %b}).",
            metric=action_label(r["action"]), sub="published grade change",
            heat=RADAR_HEAT, value=None,
            ticker=r["ticker"], page="eq:Fundamentals", book="equities"))
    return out
