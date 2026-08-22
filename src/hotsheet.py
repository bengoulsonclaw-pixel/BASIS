"""BASIS Hot Sheet — the daily cross-module highlights engine.

One page, checked every morning: everything the desk's modules are ALREADY shouting
about, ranked on a common 0-100 heat scale, each line one click from the module that
owns it. The sheet never invents analysis — every item earns its place by clearing
its home module's existing bar (the vol book's own z-flags, the curve monitor's ±2σ,
COT's crowding cutoffs, the TA report's saved thresholds…). Quiet modules simply
don't appear.

PROVIDER CONTRACT — how a module gets onto the sheet
----------------------------------------------------
Any module in src/ that defines a top-level ``radar_items()`` function is discovered
automatically (no registry to edit — build a new module, give it radar_items(), it
appears on the page and, via the daily history stamp, in the Weekly Review).

    def radar_items() -> list[dict]:
        from src import hotsheet
        return [hotsheet.item(
            tag="CURVE",                 # short chip, UPPER, stable
            key="wti_brent:cheap",       # stable story id (same story ⇒ same key
                                         # across days — the NEW/streak badges and
                                         # the weekly aggregation key off it)
            section="Curve / RV",        # display group on the page
            text="**WTI–Brent** screens **cheap** at -4.10 $/bbl.",
            metric="z -2.3", sub="vs its 1y band",
            heat=hotsheet.heat_from_z(-2.3),
            value=-4.10,                 # the underlying number → generic Δ-on-week
            ticker="CLA Comdty",         # for the sector filter (optional)
            page="Curve Monitor",        # app nav key the jump button targets
            book="ficc",                 # "ficc" | "equities" | "meta"
        )]

Rules for providers:
- READ ONLY what is already on disk — never trigger a Bloomberg/Yahoo pull.
- Only emit what the module itself already flags; use the module's own thresholds.
- Prose is CLIENT-SAFE neutral observation (it flows into the Weekly Review PDF):
  no buy/sell/recommend language. Mark desk-only lines ``internal_only=True``.
- Trim to your own editorial cap; the engine hard-caps each provider at
  ``PROVIDER_CAP`` by heat regardless.
- A provider that raises drops off the sheet (logged in the page footer) — it can
  never take the page down. Keep heavy imports inside radar_items().

The daily history store (data/signals/hotsheet_history.parquet) is stamped once per
snapshot compute run: same-day re-stamps replace that day, past days are frozen
(the signal-ledger lesson — derived stores never rewrite written history). Badges
("NEW", "4th day") and the Weekly Review's week_view() aggregation derive from it.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                # standalone runs (python src/hotsheet.py,
    sys.path.insert(0, str(ROOT))            # the snapshot stamp) need src.* importable
SIG_DIR = ROOT / "data" / "signals"
HISTORY_FILE = SIG_DIR / "hotsheet_history.parquet"
META_FILE = SIG_DIR / "hotsheet_meta.json"

PROVIDER_CAP = 5            # hard per-provider cap — the sheet is a screen, not an inventory
BOOKS = ("ficc", "equities", "meta")
SPARK_MAX = 120             # sparkline points kept per item (longer series are strided down)
SPARK_MIN = 8               # fewer points than this isn't a shape — the spark is dropped

# columns persisted per item per day — the history store's schema
COLUMNS = ["date", "key", "tag", "section", "book", "text", "metric", "sub",
           "heat", "value", "ticker", "page", "weekly", "internal_only", "provider"]


# ---------------------------------------------------------------------------
# item factory + heat helpers
# ---------------------------------------------------------------------------
def item(tag: str, key: str, section: str, text: str, heat: float,
         metric: str = "", sub: str = "", value: float | None = None,
         ticker: str = "", page: str = "", book: str = "ficc",
         weekly: bool = True, internal_only: bool = False,
         spark: list | None = None) -> dict:
    """Build one validated Hot Sheet item. ``key`` gets the tag prefixed
    (weekreview's convention) so story ids can't collide across modules.

    ``spark`` (optional) is the story's own recent daily series, oldest→newest —
    the page draws it as an inline sparkline so a slow grind and a sudden snap
    read differently at a glance. Pass the series in the SAME space the prose
    speaks (FI stories in yield space, vol stories in vol points). Page-render
    only: the daily history stamp drops it (COLUMNS is the store's schema)."""
    if not tag or not key or not section or not text:
        raise ValueError("hotsheet.item: tag, key, section and text are required")
    if book not in BOOKS:
        raise ValueError(f"hotsheet.item: book must be one of {BOOKS}, got {book!r}")
    sp = None
    if spark is not None:
        import math
        vals = [float(v) for v in list(spark)
                if v is not None and math.isfinite(float(v))]
        if len(vals) >= SPARK_MIN:
            if len(vals) > SPARK_MAX:                # stride down, always keep the last point
                step = (len(vals) - 1) / (SPARK_MAX - 1)
                vals = [vals[round(i * step)] for i in range(SPARK_MAX)]
            sp = vals
    return {"key": f"{tag}:{key}", "tag": str(tag).upper(), "section": section,
            "text": text, "metric": metric, "sub": sub,
            "heat": float(max(0.0, min(100.0, float(heat)))),
            "value": (float(value) if value is not None else None),
            "ticker": ticker or "", "page": page or "",
            "book": book, "weekly": bool(weekly), "internal_only": bool(internal_only),
            "spark": sp}


def heat_from_z(z: float, full: float = 4.0) -> float:
    """|z| onto 0-100, saturating at ``full`` sigma (the weekreview gauge convention)."""
    return min(100.0, abs(float(z)) / full * 100.0)


def heat_from_pctl(pctl: float) -> float:
    """Distance from the 50th percentile onto 0-100 (0th and 100th are both extremes)."""
    return min(100.0, abs(float(pctl) - 50.0) * 2.0)


# ---------------------------------------------------------------------------
# provider discovery — scan src/ for `def radar_items`, import lazily
# ---------------------------------------------------------------------------
_SKIP = {"hotsheet", "__init__"}
_DEF_RX = re.compile(r"^def radar_items\s*\(", re.M)


def discover() -> list[str]:
    """Module names in src/ whose SOURCE defines a top-level radar_items().
    Text-scan first so non-providers are never imported (some src modules pull
    heavy deps at import time)."""
    names = []
    for p in sorted((ROOT / "src").glob("*.py")):
        if p.stem in _SKIP:
            continue
        try:
            if _DEF_RX.search(p.read_text(encoding="utf-8", errors="replace")):
                names.append(p.stem)
        except Exception:
            continue
    return names


def collect(book: str | None = None) -> tuple[list[dict], dict]:
    """Run every discovered provider; one that raises drops out instead of killing
    the sheet. Returns (items, report) — report is {provider: {status, n, ms, err}}
    for the page footer, so a newly built module visibly plugs itself in and a
    broken one visibly falls off."""
    items, report = [], {}
    for name in discover():
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f"src.{name}")
            got = list(mod.radar_items() or [])
            for it in got:                       # validate/normalize via the factory
                missing = {"tag", "key", "section", "text", "heat"} - set(it)
                if missing:
                    raise ValueError(f"item missing {sorted(missing)}: {it}")
            got.sort(key=lambda it: -it["heat"])
            dropped = len(got) - PROVIDER_CAP
            got = got[:PROVIDER_CAP]
            for it in got:
                it["provider"] = name
            items.extend(got)
            report[name] = {"status": "ok" if got else "quiet", "n": len(got),
                            "ms": int((time.perf_counter() - t0) * 1000),
                            "err": "", "over_cap": max(0, dropped)}
        except Exception as e:
            print(f"[hotsheet] provider {name} failed:", file=sys.stderr)
            traceback.print_exc()
            report[name] = {"status": "failed", "n": 0,
                            "ms": int((time.perf_counter() - t0) * 1000),
                            "err": f"{type(e).__name__}: {e}", "over_cap": 0}
    if book:
        items = [it for it in items if it["book"] == book]
    items.sort(key=lambda it: -it["heat"])
    return items, report


# ---------------------------------------------------------------------------
# daily history store — stamped once per snapshot compute run
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    try:
        h = pd.read_parquet(HISTORY_FILE)
        h["date"] = pd.to_datetime(h["date"])
        return h
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def stamp(items: list[dict], report: dict, asof: date | None = None,
          log=print) -> int:
    """Write today's items into the history store. Same-day re-stamps replace that
    day; past days are never touched. Refuses an all-failed empty stamp so a wedged
    morning can't write a bogus quiet day over the record."""
    asof = asof or date.today()
    failed = [k for k, v in report.items() if v["status"] == "failed"]
    if not items and failed and len(failed) == len(report):
        log(f"[hotsheet] stamp refused — all {len(failed)} providers failed, "
            "not writing an empty day")
        return 0
    day = pd.Timestamp(asof)
    rows = pd.DataFrame([{**it, "date": day} for it in items], columns=COLUMNS)
    hist = load_history()
    hist = hist[hist["date"] != day]             # replace today only; the past is frozen
    out = pd.concat([hist, rows], ignore_index=True) if len(rows) else hist
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(HISTORY_FILE, index=False)
    META_FILE.write_text(json.dumps(
        {"stamped": asof.isoformat(), "n_items": len(items),
         "providers": {k: {kk: vv for kk, vv in v.items() if kk != "ms"}
                       for k, v in report.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return len(items)


def stamp_today(log=print) -> int:
    """The snapshot compute phase's one-liner: collect everything, stamp the history,
    and persist the finished sheet so every page open until the next stamp is a
    millisecond file read instead of a live provider run."""
    items, report = collect()
    n = stamp(items, report, log=log)
    try:        # the top-10 export other report pipelines read (Morning Coffee PDF)
        n_exp = write_top10_export(items)
        if n_exp < 3:           # thin live sheet (off-hours / weekend stamp): serve the
            n_exp = export_from_history()      # latest stamped morning instead
            log(f"  Hot Sheet top-10 export: live sheet thin — {n_exp} rows from history")
        else:
            log(f"  Hot Sheet top-10 export: {n_exp} rows")
    except Exception as e:
        log(f"  (Hot Sheet top-10 export skipped: {e})")
    bad = [k for k, v in report.items() if v["status"] == "failed"]
    if items or len(bad) < len(report):         # the stamp's wedged-morning guard, mirrored
        _persist_collection(items, report)
    log(f"  Hot Sheet stamped: {n} items from "
        f"{sum(1 for v in report.values() if v['n'])} providers"
        + (f" ({len(bad)} failed: {', '.join(bad)})" if bad else ""))
    return n


# ---------------------------------------------------------------------------
# the persisted sheet — the seasonality-cache pattern applied whole-sheet, so the
# page never pays a provider run on an ordinary click
# ---------------------------------------------------------------------------
CACHE_FILE = SIG_DIR / "hotsheet_cache.json"
CACHE_STALE_H = 20.0        # beyond this the file is a missed morning, not a cache

# The desk-home card's TOP STRIP as a tiny export for OTHER report pipelines (the
# Morning Coffee PDF carries it as "BASIS Hot Sheet — top 10", Ben 2026-08-22):
# FICC book, NEW badges, 2 per module, the first 10 by heat — the exact rows the
# home card and module 02 show. Written with every stamp / persist so it's never
# older than the sheet itself.
TOP10_FILE = SIG_DIR / "hotsheet_top10.json"


def _top10_file() -> Path:
    """Resolved at CALL time from SIG_DIR, so a test that redirects the store
    directory can never write the export into the live data/ tree (it did,
    2026-08-22: the suite's TST fixture row replaced the morning's ten)."""
    return SIG_DIR / TOP10_FILE.name


def export_from_history(min_rows: int = 5) -> int:
    """Rebuild the top-10 export from the STAMPED history when the live sheet is thin
    (weekend / off-hours re-collects): the latest stamped day with >= min_rows FICC
    rows, heat-sorted, through the same top-strip rule. Returns rows written."""
    try:
        h = load_history()
        if h.empty:
            return 0
        h = h[h["book"] == "ficc"]
        for day, grp in sorted(h.groupby("date"), key=lambda kv: kv[0], reverse=True):
            if len(grp) < min_rows:
                continue
            items = grp.sort_values("heat", ascending=False).to_dict("records")
            for it in items:
                it["badge"] = ""
            top = top_strip(items)
            payload = {
                "collected": float(time.time()),      # written NOW: the thin-guard's 36h
                "collected_et": f"{pd.Timestamp(day):%Y-%m-%d} (morning stamp)",   # window protects it
                "rows": [{"tag": it.get("tag", ""), "text": it.get("text", ""),
                          "metric": it.get("metric", ""), "sub": it.get("sub", ""),
                          "heat": float(it.get("heat", 0) or 0), "badge": "",
                          "page": it.get("page", ""), "provider": it.get("provider", "")}
                         for it in top],
            }
            SIG_DIR.mkdir(parents=True, exist_ok=True)
            _top10_file().write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
            return len(top)
    except Exception:
        pass
    return 0


def top_strip(items: list, book: str = "ficc", per_tag: int = 2, n: int = 10) -> list:
    """The editorial top strip: `book` only, at most `per_tag` rows per module, first
    `n` by heat (items arrive heat-sorted). Pure selection — no I/O."""
    top, seen = [], {}
    for it in items:
        if it.get("book") != book:
            continue
        if seen.get(it["tag"], 0) >= per_tag:
            continue
        seen[it["tag"]] = seen.get(it["tag"], 0) + 1
        top.append(it)
        if len(top) >= n:
            break
    return top


def write_top10_export(items: list, collected: float | None = None) -> int:
    """Persist the FICC top strip to TOP10_FILE; returns the row count. Badges are
    applied on a copy so the caller's items stay pristine."""
    try:
        rows = [dict(it) for it in items]
        try:
            apply_badges(rows)
        except Exception:
            pass
        top = top_strip(rows)
        # Never let a thin re-collect clobber a rich export: an off-hours / weekend
        # stamp (providers quiet) produced 0-1 rows and overwrote the morning's ten
        # (2026-08-22). Keep the existing file when it is materially fuller and
        # less than 36h old — the next full morning stamp replaces it anyway.
        try:
            old = json.loads(_top10_file().read_text(encoding="utf-8"))
            old_rows = old.get("rows") or []
            old_age_h = (time.time() - float(old.get("collected", 0))) / 3600.0
            if old_rows and len(top) < max(3, len(old_rows) // 2) and old_age_h < 36:
                return 0
        except Exception:
            pass
        payload = {
            "collected": float(collected or time.time()),
            "collected_et": pd.Timestamp(collected or time.time(), unit="s", tz="UTC")
                              .tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET"),
            "rows": [{"tag": it.get("tag", ""), "text": it.get("text", ""),
                      "metric": it.get("metric", ""), "sub": it.get("sub", ""),
                      "heat": float(it.get("heat", 0) or 0), "badge": it.get("badge", ""),
                      "page": it.get("page", ""), "provider": it.get("provider", "")}
                     for it in top],
        }
        SIG_DIR.mkdir(parents=True, exist_ok=True)
        _top10_file().write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(top)
    except Exception:
        return 0


def _persist_collection(items: list, report: dict) -> None:
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(
        {"collected": time.time(), "items": items, "report": report},
        ensure_ascii=False), encoding="utf-8")


def cached_collection() -> tuple[list, dict, float, bool]:
    """The page's entry point: (items, report, collected_epoch, from_cache).
    Serves the persisted sheet while it's fresh (the morning stamp writes it);
    missing/stale falls through to a live collect that re-fills the file."""
    try:
        c = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        age_h = (time.time() - float(c["collected"])) / 3600.0
        if 0.0 <= age_h < CACHE_STALE_H:
            return list(c["items"]), dict(c["report"]), float(c["collected"]), True
    except Exception:
        pass
    return refresh_collection()


def refresh_collection() -> tuple[list, dict, float, bool]:
    """Live provider run + persist — the page's ↻ and the stale-cache path. An
    all-failed collect never overwrites a usable sheet with an empty one; it serves
    the old file (however stale) so a wedged run can't blank the page."""
    items, report = collect()
    failed = [k for k, v in report.items() if v["status"] == "failed"]
    if not items and failed and len(failed) == len(report):
        try:
            c = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            return list(c["items"]), dict(c["report"]), float(c["collected"]), True
        except Exception:
            pass
        return items, report, time.time(), False
    _persist_collection(items, report)
    return items, report, time.time(), False


# ---------------------------------------------------------------------------
# badges — NEW / nth day, derived from the stamped history
# ---------------------------------------------------------------------------
def apply_badges(items: list[dict], asof: date | None = None) -> None:
    """Annotate items in place. NEW = never stamped before today; a streak is
    consecutive STAMPED days (the store's own day axis — weekends/holidays don't
    break it) ending at the most recent stamp. Quiet by default: persistence is
    only called out once it's a story (3rd day on), the weekreview convention."""
    hist = load_history()
    asof = pd.Timestamp(asof or date.today())
    past = hist[hist["date"] < asof]
    if past.empty:                               # day one — badges would all be NEW noise
        for it in items:
            it["badge"] = ""
        return
    days = sorted(set(past["date"]))             # iterate the column, not .unique() — keeps
    by_key = past.groupby("key")["date"].agg(set)   # everything pd.Timestamp for the set probes
    for it in items:
        seen = by_key.get(it["key"], set())
        if not seen:
            it["badge"] = "NEW"
            continue
        streak = 1                               # today counts; walk back over stamped days
        for d in reversed(days):
            if d in seen:
                streak += 1
            else:
                break
        it["badge"] = f"day {streak}" if streak >= 3 else ""


# ---------------------------------------------------------------------------
# weekly aggregation — what the Weekly Review's front page will read
# ---------------------------------------------------------------------------
def week_view(days: int = 7, book: str = "ficc", asof: date | None = None) -> pd.DataFrame:
    """The week's stories, one row per key: the most extreme reading kept, ranked by
    persistence (days on the sheet) then peak heat, with a generic value Δ over the
    window. Excludes internal-only and weekly=False items — this is the client cut."""
    hist = load_history()
    if hist.empty:
        return hist
    asof = pd.Timestamp(asof or date.today())
    wk = hist[(hist["date"] > asof - pd.Timedelta(days=days)) & (hist["date"] <= asof)]
    wk = wk[(wk["book"] == book) & wk["weekly"] & ~wk["internal_only"]]
    if wk.empty:
        return wk.iloc[0:0]
    rows = []
    for key, g in wk.groupby("key"):
        g = g.sort_values("date")
        peak = g.loc[g["heat"].idxmax()]
        vals = g["value"].dropna()
        rows.append({**peak.to_dict(), "days_on": int(g["date"].nunique()),
                     "first_seen": g["date"].min(), "last_seen": g["date"].max(),
                     "wk_delta": (float(vals.iloc[-1] - vals.iloc[0])
                                  if len(vals) >= 2 else None)})
    out = pd.DataFrame(rows).sort_values(["days_on", "heat"], ascending=[False, False])
    return out.reset_index(drop=True)


if __name__ == "__main__":                       # quick eyeball: python src/hotsheet.py
    its, rep = collect()
    for k, v in rep.items():
        print(f"{k:>14}: {v['status']:6} n={v['n']:2d} {v['ms']:5d}ms  {v['err']}")
    print(f"\n{len(its)} items")
    for it in its[:20]:
        print(f"  [{it['tag']:6}] {it['heat']:5.1f}  {re.sub(r'[*]+', '', it['text'])[:100]}")
