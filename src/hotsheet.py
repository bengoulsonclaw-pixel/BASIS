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

# columns persisted per item per day — the history store's schema
COLUMNS = ["date", "key", "tag", "section", "book", "text", "metric", "sub",
           "heat", "value", "ticker", "page", "weekly", "internal_only", "provider"]


# ---------------------------------------------------------------------------
# item factory + heat helpers
# ---------------------------------------------------------------------------
def item(tag: str, key: str, section: str, text: str, heat: float,
         metric: str = "", sub: str = "", value: float | None = None,
         ticker: str = "", page: str = "", book: str = "ficc",
         weekly: bool = True, internal_only: bool = False) -> dict:
    """Build one validated Hot Sheet item. ``key`` gets the tag prefixed
    (weekreview's convention) so story ids can't collide across modules."""
    if not tag or not key or not section or not text:
        raise ValueError("hotsheet.item: tag, key, section and text are required")
    if book not in BOOKS:
        raise ValueError(f"hotsheet.item: book must be one of {BOOKS}, got {book!r}")
    return {"key": f"{tag}:{key}", "tag": str(tag).upper(), "section": section,
            "text": text, "metric": metric, "sub": sub,
            "heat": float(max(0.0, min(100.0, float(heat)))),
            "value": (float(value) if value is not None else None),
            "ticker": ticker or "", "page": page or "",
            "book": book, "weekly": bool(weekly), "internal_only": bool(internal_only)}


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
    """The snapshot compute phase's one-liner: collect everything and stamp it."""
    items, report = collect()
    n = stamp(items, report, log=log)
    bad = [k for k, v in report.items() if v["status"] == "failed"]
    log(f"  Hot Sheet stamped: {n} items from "
        f"{sum(1 for v in report.values() if v['n'])} providers"
        + (f" ({len(bad)} failed: {', '.join(bad)})" if bad else ""))
    return n


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
