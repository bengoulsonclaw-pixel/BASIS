"""Data-health engine — everything the 🩺 Data health page reports, with no Streamlit.

The app is ~30 modules riding a handful of on-disk stores (the morning snapshot, the
deep '1'-generic price store, the signal cache, the own-vol-curve histories). Each
store already guards itself locally; this module is the one place that LOOKS AT ALL
OF THEM and says what's fresh, what's truncated and what's quietly stale — so a bad
morning shows up as a red line on one page instead of as a mysteriously thin report
three modules downstream.

Pure readers: every function tolerates missing files/stores (VPS, fresh clone, demo
mode) by returning an empty frame / harmless default — the page must always render.
Timestamps are returned tz-aware UTC (file mtimes are UTC epochs on this box); the
PAGE formats them for display (ET), never this module.

The regression suite (tests/, run_tests.py) writes logs/last_test_run.json; this
module surfaces it so the safety net itself is visible on the health board.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import universe

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot"
STORE = ROOT / "data" / "price_store"
TEST_LOG = ROOT / "logs" / "last_test_run.json"

# The frames one FICC pull writes together (snapshot.py's _fetch_phase). A frame whose
# file lags live.parquet by more than a slack window means a PARTIAL pull — some legs
# died mid-fetch and the app is mixing this morning's prices with yesterday's vols.
CORE_FRAMES = ("prices", "yields", "volume", "implied_vol", "skew_put", "skew_call",
               "skew_atm", "live")
PARTIAL_PULL_SLACK_H = 2.0     # frames of one pull are written minutes apart, not hours

# Long-format stores (a `ticker` column, not date-indexed wide) — "markets" is counted
# as distinct tickers; everything else counts data columns.
_LONG_STORES = {"oi_chain", "own30_curve", "own30_history", "own_term_history",
                "own_skew_history", "stir_curve_history"}

SNAPSHOT_OLD_H = 24            # snapshot older than this -> warn (matches the page's old rule)
OI_OLD_DAYS = 9                # the weekly Monday OI capture has been missed
CACHE_LAG_DAYS = 5             # a daily-fed store this far behind the snapshot settle -> warn
CB_CAL_MIN_MONTHS = 9          # STIR Paths meeting-calendar runway below this -> warn (the
                               # *_DECISIONS lists are hardcoded and need a yearly extension)


def _utc(ts: float) -> pd.Timestamp:
    return pd.Timestamp(ts, unit="s", tz="UTC")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _age_h(ts: pd.Timestamp | None) -> float:
    return float("nan") if ts is None else (_now() - ts).total_seconds() / 3600.0


def load_manifest() -> dict:
    try:
        return json.loads((SNAP / "manifest.json").read_text())
    except Exception:
        return {}


def parse_stamp(raw) -> pd.Timestamp | None:
    """A manifest stamp -> tz-aware UTC. Offset-tagged stamps ('…+00:00') are honored;
    legacy naive stamps are UTC too (they were file mtimes) — same rule the app uses."""
    s = str(raw or "").strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        ts = pd.Timestamp(s)
    except Exception:
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Snapshot frames — one row per parquet under data/snapshot/
# ---------------------------------------------------------------------------
def _frame_meta(p: Path) -> dict:
    """Row counts + column names from parquet METADATA (no data read), then the two
    cheap column reads (date span, distinct tickers) only where they apply."""
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(p)
    cols = [c for c in meta.schema_arrow.names]
    n_rows = int(meta.metadata.num_rows)
    name = p.stem
    last_date = None
    if "date" in cols and n_rows:
        try:
            d = pd.read_parquet(p, columns=["date"])["date"]
            last_date = pd.to_datetime(d).max()
        except Exception:
            pass
    if name in _LONG_STORES:
        markets = 0
        if "ticker" in cols and n_rows:
            try:
                markets = int(pd.read_parquet(p, columns=["ticker"])["ticker"].nunique())
            except Exception:
                pass
    elif name == "live":
        markets = n_rows                 # ticker-indexed: one ROW per market, not one column
    else:
        markets = max(len(cols) - (1 if "date" in cols else 0), 0)
    mtime = _utc(p.stat().st_mtime)
    return {"frame": name, "rows": n_rows, "markets": markets,
            "last_date": (last_date.date() if last_date is not None else None),
            "written": mtime, "age_h": round(_age_h(mtime), 1),
            "size_mb": round(p.stat().st_size / 1e6, 2)}


def snapshot_frames() -> pd.DataFrame:
    """Inventory of every parquet in data/snapshot/ — the 'snapshot age per frame' table.
    Core pull frames first (pull order), then the accruing history stores."""
    rows = []
    for p in sorted(SNAP.glob("*.parquet")):
        try:
            rows.append(_frame_meta(p))
        except Exception:
            rows.append({"frame": p.stem, "rows": None, "markets": None, "last_date": None,
                         "written": None, "age_h": float("nan"), "size_mb": None})
    if not rows:
        return pd.DataFrame(columns=["frame", "rows", "markets", "last_date",
                                     "written", "age_h", "size_mb"])
    df = pd.DataFrame(rows)
    order = {n: i for i, n in enumerate(CORE_FRAMES)}
    df["_o"] = df["frame"].map(lambda n: order.get(n, 100 + int(n in _LONG_STORES) * 100))
    return df.sort_values(["_o", "frame"]).drop(columns="_o").reset_index(drop=True)


def missing_core_frames(frames: pd.DataFrame | None = None) -> list:
    frames = snapshot_frames() if frames is None else frames
    have = set(frames["frame"]) if not frames.empty else set()
    return [f for f in CORE_FRAMES if f not in have]


def partial_pull_frames(frames: pd.DataFrame | None = None) -> list:
    """Core frames written more than PARTIAL_PULL_SLACK_H before live.parquet — the
    signature of a pull where some legs failed and stale files were left in place."""
    frames = snapshot_frames() if frames is None else frames
    if frames.empty:
        return []
    core = frames[frames["frame"].isin(CORE_FRAMES)].dropna(subset=["written"])
    if core.empty or "live" not in set(core["frame"]):
        return []
    anchor = core.loc[core["frame"] == "live", "written"].iloc[0]
    lag = (anchor - core["written"]).dt.total_seconds() / 3600.0
    return sorted(core.loc[lag > PARTIAL_PULL_SLACK_H, "frame"])


def compute_lag_h(frames: pd.DataFrame | None = None) -> float:
    """Hours the manifest trails the on-disk fetch. The manifest is written by the
    COMPUTE phase (its `created` = live.parquet's mtime at that moment), so fresh
    frames + an old manifest means `snapshot.py --fetch` ran but `--compute` never
    followed — signals, own-curve fits and the signal cache are all stale even
    though prices look current. NaN when either side is unknown."""
    frames = snapshot_frames() if frames is None else frames
    created = parse_stamp(load_manifest().get("created"))
    if created is None or frames.empty:
        return float("nan")
    live = frames.loc[frames["frame"] == "live", "written"]
    if live.empty or pd.isna(live.iloc[0]):
        return float("nan")
    return float((live.iloc[0] - created).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# Deep price store — coverage, truncation, freshness
# ---------------------------------------------------------------------------
def deep_health() -> dict:
    """The deep-store panel's numbers: per-ticker coverage (+flags), truncated /
    missing / laggard ticker lists, and how far the store trails the snapshot settle.

      coverage      DataFrame: ticker, market, first, last, days, rolls, flag
      truncated     tickers held below deepstore.HEAL_MIN_DAYS (will re-backfill)
      missing       universe tickers not in the store at all
      laggards      tickers whose last settle trails the store-wide max by more than
                    deepstore.FRESH_TOLERANCE_DAYS (overlay() silently skips these)
      bonds_no_yield bond futures with no deep yield series — get_ta DROPS them, so
                    FI technicals lose deep history on those names
      store_last / lag_vs_snap_days  the store's newest settle vs the snapshot's
    """
    from . import deepstore
    out = {"coverage": pd.DataFrame(), "truncated": [], "missing": [], "laggards": [],
           "bonds_no_yield": [], "store_last": None, "lag_vs_snap_days": None,
           "min_days": deepstore.HEAL_MIN_DAYS, "fresh_tol_days": deepstore.FRESH_TOLERANCE_DAYS}
    try:
        cov = deepstore.coverage()
    except Exception:
        return out
    tickers = list(universe.INSTRUMENTS)
    if cov is None or cov.empty:
        out["missing"] = tickers
        return out
    cov = cov.copy()
    held = set(cov["ticker"])
    out["missing"] = [t for t in tickers if t not in held]
    out["truncated"] = sorted(cov.loc[cov["days"] < deepstore.HEAL_MIN_DAYS, "ticker"])
    last_all = pd.to_datetime(cov["last"]).max()
    out["store_last"] = last_all
    lag = (last_all - pd.to_datetime(cov["last"])).dt.days
    out["laggards"] = sorted(cov.loc[lag > deepstore.FRESH_TOLERANCE_DAYS, "ticker"])
    try:
        ylds = deepstore.get_yields(tickers)
        ycols = set(ylds.columns) if not ylds.empty else set()
        out["bonds_no_yield"] = sorted(t for t in tickers
                                       if universe.is_bond(t) and t in held and t not in ycols)
    except Exception:
        pass
    snap_asof = pd.Timestamp(load_manifest().get("as_of")) if load_manifest().get("as_of") else None
    if snap_asof is not None and last_all is not None:
        out["lag_vs_snap_days"] = int((snap_asof - last_all).days)

    def _flag(r):
        t = r["ticker"]
        if t in set(out["truncated"]):
            return "truncated"
        if t in set(out["laggards"]):
            return "stale"
        return ""
    cov["market"] = cov["ticker"].map(universe.name)
    cov["flag"] = cov.apply(_flag, axis=1)
    out["coverage"] = cov[["ticker", "market", "first", "last", "days", "rolls", "flag"]] \
        .sort_values(["flag", "days"], ascending=[False, True]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Vol-surface staleness — the same guard the vol/skew/term pages apply
# ---------------------------------------------------------------------------
def stale_surfaces() -> pd.DataFrame:
    """One row per (ticker, surface) whose implied series looks dead — frozen at one
    value or stopped publishing — via datafeed.stale_iv_reasons, over the 1M ATM
    surface, the skew wings and every term pillar. These markets are already being
    left unscored by the strategies; this table says WHICH and WHY."""
    from .datafeed import (get_implied_vol_history, get_skew_components,
                           get_term_structure, stale_iv_reasons, TENOR_LABELS)
    tickers = list(universe.INSTRUMENTS)
    rows = []

    def _collect(frame, surface):
        try:
            for tk, why in stale_iv_reasons(frame).items():
                rows.append({"ticker": tk, "market": universe.name(tk),
                             "surface": surface, "reason": why})
        except Exception:
            pass

    try:
        _collect(get_implied_vol_history(tickers), "1M ATM")
    except Exception:
        pass
    try:
        skew = get_skew_components(tickers)
        for key, lab in (("put", "skew 90% put"), ("call", "skew 110% call")):
            _collect(skew.get(key), lab)
    except Exception:
        pass
    try:
        term = get_term_structure(tickers)
        for lab in TENOR_LABELS:
            _collect(term.get(lab), f"term {lab}")
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(columns=["ticker", "market", "surface", "reason"])
    return (pd.DataFrame(rows).sort_values(["ticker", "surface"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Accruing caches / history stores fed by the daily pull
# ---------------------------------------------------------------------------
def _long_store_row(name: str, path: Path, date_col: str = "date",
                    ticker_col: str = "ticker") -> dict:
    row = {"store": name, "last_date": None, "items": None, "age_h": float("nan")}
    try:
        if not path.exists():
            return row
        d = pd.read_parquet(path, columns=[c for c in (date_col, ticker_col) if c])
        if date_col in d.columns and len(d):
            row["last_date"] = pd.to_datetime(d[date_col]).max().date()
        if ticker_col and ticker_col in d.columns and len(d):
            row["items"] = int(d[ticker_col].nunique())
        row["age_h"] = round(_age_h(_utc(path.stat().st_mtime)), 1)
    except Exception:
        pass
    return row


def cache_health() -> pd.DataFrame:
    """Freshness of every accruing store the daily pull feeds — last data date, how
    many products it holds, file age. `last_date` trailing the snapshot settle means
    that leg of the pull has been failing quietly."""
    rows = []
    try:
        from . import sigcache
        cov = sigcache.coverage()
        rows.append({"store": "signal cache (sigcache)",
                     "last_date": (pd.to_datetime(cov["date"]).max().date()
                                   if cov is not None and not cov.empty else None),
                     "items": (int(cov["strategy"].nunique())
                               if cov is not None and not cov.empty else None),
                     "age_h": float("nan")})
    except Exception:
        rows.append({"store": "signal cache (sigcache)", "last_date": None,
                     "items": None, "age_h": float("nan")})
    # Duplicate-request check from the bbg gateway (src/bbg.py): the last pull's
    # count of security×field pairs requested by more than one leg. Anything but 0
    # means two modules are pulling the same data — the 2026-08-18 error class.
    try:
        import json as _json
        _dup = _json.loads((ROOT / "data" / "pull_duplicates.json").read_text(encoding="utf-8"))
        rows.append({"store": "duplicate Bloomberg requests (last pull)",
                     "last_date": pd.to_datetime(_dup.get("asof")).date(),
                     "items": int(_dup.get("n_duplicate_pairs", 0)),
                     "age_h": float("nan")})
    except Exception:
        pass                                   # no gateway report yet — nothing to show
    rows.append(_long_store_row("own vol curve (30/90d)", SNAP / "own30_history.parquet"))
    rows.append(_long_store_row("own term structure", SNAP / "own_term_history.parquet"))
    rows.append(_long_store_row("own skew (recording)", SNAP / "own_skew_history.parquet"))
    rows.append(_long_store_row("STIR 90d curve", SNAP / "stir_curve_history.parquet"))
    # wide date-indexed stores — items = market columns
    for name, p in (("COT price DB", STORE / "cot_prices.parquet"),
                    ("fundamentals DB (equities)", ROOT / "data" / "equities" / "fundamentals.parquet")):
        row = {"store": name, "last_date": None, "items": None, "age_h": float("nan")}
        try:
            if p.exists():
                import pyarrow.parquet as pq
                meta = pq.ParquetFile(p)
                cols = [c for c in meta.schema_arrow.names]
                dcol = next((c for c in ("date", "pulled", "as_of", "asof") if c in cols), None)
                if dcol:
                    d = pd.read_parquet(p, columns=[dcol])[dcol]
                    row["last_date"] = pd.to_datetime(d).max().date()
                if "ticker" in cols:                       # long DB (fundamentals): count names
                    row["items"] = int(pd.read_parquet(p, columns=["ticker"])["ticker"].nunique())
                elif dcol:
                    row["items"] = len(cols) - 1           # wide DB: market columns
                row["age_h"] = round(_age_h(_utc(p.stat().st_mtime)), 1)
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regression suite — surface the last run so the safety net is itself visible
# ---------------------------------------------------------------------------
def meeting_calendar_runway() -> pd.DataFrame:
    """Per central bank: the last listed decision date and the runway (months) left
    on the hardcoded STIR Paths meeting calendars (fedpath.FOMC_DECISIONS +
    stirpaths.ECB/BOE_DECISIONS). The lists don't self-update — when a bank
    publishes its next calendar year the dates are appended by hand; past the last
    listed meeting the path model quietly sees NO meetings, so the runway needs to
    stay comfortably ahead of the strip."""
    from . import stirpaths                       # lazy: keep health importable standalone
    today = pd.Timestamp.today().normalize()
    rows = []
    for bank in stirpaths.BANKS.values():
        last = max(bank.meetings)
        rows.append({"bank": bank.name, "key": bank.key, "last_meeting": last,
                     "months_left": round((pd.Timestamp(last) - today).days / 30.44, 1),
                     "future_meetings": sum(1 for m in bank.meetings if m > today.date())})
    return pd.DataFrame(rows)


def copom_calendar_drift(ttl: int = 24 * 3600) -> dict:
    """Hardcoded BCB_DECISIONS vs the calendar the BCB actually publishes.

    Brazil is the one bank whose meeting calendar we can check against its source: the
    BCB serves it as an API (macrodata.copom_decisions), while stirpaths keeps a
    hardcoded list like every other bank. That makes two failure modes visible that the
    runway check alone cannot see:

      * a date MOVED or was added — the runway can look healthy while the model prices a
        meeting on a day the Copom is not sitting;
      * the BCB EXTENDED its calendar — the published tail runs past the hardcoded one,
        which is the cue to append the new year rather than wait for the runway to thin.

    Read-only and cache-friendly: it never writes the hardcoded list, because a live feed
    silently rewriting a traded calendar is exactly the kind of magic this codebase avoids.
    Returns {} when the feed is unreachable, so a dead source degrades to "no opinion"
    rather than to a false alarm.
    """
    try:
        from . import macrodata, stirpaths
    except Exception:
        return {}
    try:
        live = [d for d in macrodata.copom_decisions(ttl=ttl)]
        hard = list(getattr(stirpaths, "BCB_DECISIONS", []) or [])
    except Exception:
        return {}
    if not live or not hard:
        return {}
    # Compare the WHOLE overlap, not just the forward window. Clamping at today hid a
    # wrong 2026 date on the first run: past decisions still matter to anything that
    # replays history, and a list wrong behind you is usually wrong ahead of you too.
    lo = max(min(live), min(hard))
    live_f = sorted(d for d in live if d >= lo)
    hard_f = sorted(d for d in hard if d >= lo)
    common_end = min(max(live_f), max(hard_f)) if live_f and hard_f else None
    if common_end is None:
        return {}
    return {
        "live_last": max(live_f), "hard_last": max(hard_f),
        # mismatches WITHIN the overlap — a moved or missing decision date
        "missing": [d for d in live_f if d <= common_end and d not in hard_f],
        "extra": [d for d in hard_f if d <= common_end and d not in live_f],
        # the BCB has published further than we have hardcoded
        "extends_by": [d for d in live_f if d > max(hard_f)],
    }


def last_test_run() -> dict:
    """logs/last_test_run.json as written by run_tests.py (also invoked by the
    pre-push hook). {} when the suite has never run on this box."""
    try:
        return json.loads(TEST_LOG.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# The roll-up status board
# ---------------------------------------------------------------------------
def checks(*, frames: pd.DataFrame | None = None, deep: dict | None = None,
           stale: pd.DataFrame | None = None, caches: pd.DataFrame | None = None) -> list:
    """[{level, area, message}] — level 'ok' | 'warn' | 'bad'. The page renders these
    as the status board; the order here is the display order (worst areas first is
    the page's job — it re-sorts by severity). Pass the pieces in when the caller has
    already gathered them for its own sections (the page does) — each falls back to a
    fresh read so `checks()` alone still works headless."""
    out = []

    def add(level, area, message):
        out.append({"level": level, "area": area, "message": message})

    man = load_manifest()
    frames = snapshot_frames() if frames is None else frames
    if not man:
        add("bad", "Snapshot", "No snapshot manifest — the app is on demo/mock data.")
    else:
        src = man.get("source", "?")
        if src != "bloomberg":
            add("bad", "Snapshot", f"Snapshot source is '{src}' (demo), not a live Bloomberg pull.")
        age = _age_h(parse_stamp(man.get("created")))
        if np.isfinite(age) and age > SNAPSHOT_OLD_H:
            add("warn", "Snapshot", f"Snapshot is ~{age / 24:.1f} days old — pull a fresh one on Home.")
        elif np.isfinite(age):
            add("ok", "Snapshot", f"Snapshot pulled {age:.0f}h ago (settle {man.get('as_of', '?')}).")
        if man.get("last_skipped_pull"):
            add("warn", "Snapshot", "A recent pull returned no data and was skipped "
                f"({man['last_skipped_pull']}) — the Terminal was probably closed.")
        missing = missing_core_frames(frames)
        if missing:
            add("bad", "Snapshot", "Core frames missing from data/snapshot/: " + ", ".join(missing))
        partial = partial_pull_frames(frames)
        if partial:
            add("warn", "Snapshot", "Frames older than the rest of the pull (partial fetch?): "
                + ", ".join(partial))
        clag = compute_lag_h(frames)
        if np.isfinite(clag) and clag > PARTIAL_PULL_SLACK_H:
            add("warn", "Snapshot", f"Fetched data is {clag:.0f}h newer than the manifest — "
                "the fetch ran but the COMPUTE phase didn't (signals, own-curve fits and the "
                "signal cache are stale). Run:  python snapshot.py --compute")
        oi_age = _age_h(parse_stamp(man.get("oi_as_of")))
        if np.isfinite(oi_age) and oi_age > OI_OLD_DAYS * 24:
            add("warn", "Snapshot", f"Weekly OI capture is {oi_age / 24:.0f} days old — run "
                "snapshot.py --oi on a Monday with the Terminal up.")

    dh = deep_health() if deep is None else deep
    n_uni = len(list(universe.INSTRUMENTS))
    if dh["coverage"].empty:
        add("warn", "Deep store", "Deep price store is empty — backtests are capped at the "
            "live feed's ~400 sessions (it self-backfills on the next Bloomberg pull).")
    else:
        n_bad = len(dh["truncated"]) + len(dh["missing"])
        if n_bad:
            add("warn", "Deep store",
                f"{len(dh['missing'])} universe tickers missing / {len(dh['truncated'])} truncated "
                f"(<{dh['min_days']}d held) — they re-backfill on the next Bloomberg pull.")
        else:
            add("ok", "Deep store", f"All {len(dh['coverage'])}/{n_uni} tickers held deep.")
        if dh["laggards"]:
            add("warn", "Deep store", f"{len(dh['laggards'])} tickers lag the store by "
                f">{dh['fresh_tol_days']}d (overlay() is silently skipping them): "
                + ", ".join(dh["laggards"][:8]) + ("…" if len(dh["laggards"]) > 8 else ""))
        if dh["lag_vs_snap_days"] is not None and dh["lag_vs_snap_days"] > dh["fresh_tol_days"]:
            add("warn", "Deep store", f"Store's last settle trails the snapshot by "
                f"{dh['lag_vs_snap_days']}d — beyond the {dh['fresh_tol_days']}d overlay tolerance, "
                "so backtests are quietly falling back to the short live feed.")
        if dh["bonds_no_yield"]:
            add("warn", "Deep store", "Bond futures with no deep yield series (FI technicals lose "
                "deep history): " + ", ".join(dh["bonds_no_yield"]))

    stale = stale_surfaces() if stale is None else stale
    if stale.empty:
        add("ok", "Vol surfaces", "No frozen or dead implied-vol surfaces detected.")
    else:
        n = stale["ticker"].nunique()
        add("warn", "Vol surfaces", f"{n} market(s) with a stale surface ({len(stale)} "
            "surface rows) — already left unscored by the vol/skew/term strategies.")

    caches = cache_health() if caches is None else caches
    asof = pd.Timestamp(man.get("as_of")) if man.get("as_of") else None
    if asof is not None and not caches.empty:
        for r in caches.itertuples(index=False):
            if r.last_date is None:
                continue
            lag = (asof - pd.Timestamp(r.last_date)).days
            if lag > CACHE_LAG_DAYS:
                add("warn", "Caches", f"{r.store} last has {r.last_date} — {lag}d behind the "
                    "snapshot settle; that leg of the daily pull is failing quietly.")

    if not universe.enabled_tickers():
        add("bad", "Filter", "The Sectors & products filter has EVERYTHING switched off — "
            "the TA overview and reports will be blank until it's restored on Home.")

    try:
        cal = meeting_calendar_runway()
        thin = cal[cal["months_left"] < CB_CAL_MIN_MONTHS]
        if thin.empty:
            nxt = cal.loc[cal["months_left"].idxmin()]
            add("ok", "CB calendars", f"STIR meeting calendars run ≥{CB_CAL_MIN_MONTHS} months out "
                f"(thinnest: {nxt['key']} to {nxt['last_meeting']:%b %Y}).")
        else:
            for r in thin.itertuples(index=False):
                add("warn", "CB calendars", f"{r.bank} meeting calendar runs out {r.last_meeting:%b %Y} "
                    f"(~{r.months_left:.0f} months) — append the new published dates to the "
                    "*_DECISIONS list (fedpath/stirpaths) or the STIR Paths pages go blind past it.")
    except Exception:
        add("warn", "CB calendars", "Could not read the STIR meeting calendars (stirpaths import "
            "failed) — the STIR Paths module may be broken.")

    # Brazil only: the BCB publishes its Copom calendar as an API, so unlike every other
    # bank the hardcoded list can be checked against its own source.
    try:
        drift = copom_calendar_drift()
        if drift:
            _fmt = lambda ds: ", ".join(f"{d:%d %b %Y}" for d in ds[:6])
            if drift["missing"] or drift["extra"]:
                add("bad", "Copom calendar",
                    "stirpaths.BCB_DECISIONS disagrees with the calendar the BCB publishes"
                    + (f" — missing {_fmt(drift['missing'])}" if drift["missing"] else "")
                    + (f" — has {_fmt(drift['extra'])} which the BCB does not list"
                       if drift["extra"] else "")
                    + ". The DI path would price a decision on the wrong day; fix the "
                      "hardcoded list.")
            elif drift["extends_by"]:
                add("warn", "Copom calendar",
                    f"The BCB has published {len(drift['extends_by'])} further Copom "
                    f"date(s) to {drift['live_last']:%b %Y} — append them to "
                    f"stirpaths.BCB_DECISIONS (currently ends {drift['hard_last']:%b %Y}).")
            else:
                add("ok", "Copom calendar", "stirpaths.BCB_DECISIONS matches the BCB's "
                    f"published calendar through {drift['hard_last']:%b %Y}.")
    except Exception:
        pass                      # the BCB feed being down is not a data-health failure

    try:
        from src import stirpaths as _sp
        _err = _sp.strip_error()
        if _err:
            add("warn", "STIR strip store", f"The morning STIR strip pull is FAILING "
                f"(last try {_err.get('when', '?')}, stage: {_err.get('stage', '?')}): "
                f"{_err.get('detail', 'no detail recorded')} The cockpits run on the last "
                "good store (or synthetic demo) until it succeeds.")
        elif not _sp.STRIP_STORE.exists():
            add("warn", "STIR strip store", "Never filled — the STIR cockpits run on the "
                "synthetic demo strip until the morning leg succeeds (or prices are "
                "entered manually).")
        else:
            _st = json.loads(_sp.STRIP_STORE.read_text(encoding="utf-8"))
            _age = (pd.Timestamp.now().normalize()
                    - pd.Timestamp(_st.get("asof", "1970-01-01"))).days
            _rej = _st.get("rejected", {})
            if _age > 5:
                add("warn", "STIR strip store", f"Store is {_age} days old (asof "
                    f"{_st.get('asof')}) — the morning leg hasn't refreshed it; the "
                    "cockpit odds are priced off stale strips.")
            elif _rej:
                add("warn", "STIR strip store", f"Strip store fresh (asof {_st.get('asof')}) "
                    f"but the pull returned IMPLAUSIBLE prices for {len(_rej)} contract(s), "
                    "rejected by the 90–100.5 gate: "
                    + ", ".join(f"{k}={v}" for k, v in list(_rej.items())[:6])
                    + ". Those contracts are absent from the fits (usually fine — dead "
                    "far serials); if a LIQUID contract appears here, investigate the pull.")
            else:
                add("ok", "STIR strip store", f"Strip store fresh (asof {_st.get('asof')}, "
                    f"{len(_st.get('prices', {}) or _st.get('settles', {}))} contracts).")
    except Exception:
        add("warn", "STIR strip store", "Could not read the STIR strip store state.")

    tr = last_test_run()
    if not tr:
        add("warn", "Regression suite", "Never run on this box — run run_tests.bat (or push "
            "code; the pre-push hook runs it).")
    elif tr.get("skipped"):
        add("warn", "Regression suite", f"Last run was skipped: {tr['skipped']}.")
    elif tr.get("ok"):
        add("ok", "Regression suite", f"Green — {tr.get('summary', 'passed')} "
            f"({str(tr.get('when', ''))[:16]} UTC).")
    else:
        add("bad", "Regression suite", f"RED — {tr.get('summary', 'failures')} "
            f"({str(tr.get('when', ''))[:16]} UTC). Fix before trusting engine output.")

    return out


# ---------------------------------------------------------------------------
# Hot Sheet provider (src/hotsheet.py) — trust-affecting problems only
# ---------------------------------------------------------------------------
RADAR_COMPUTE_LAG_H = 30.0   # a day+ behind: checks() nags at 2h, the sheet only at
                             # "everything derived is yesterday's" scale
RADAR_HEAT = 20.0            # health lines render as a flat caveat strip, not ranked stories


def radar_items() -> list:
    """Hot Sheet provider — the health board's trust-affecting exceptions (max 3):
    missing core snapshot frames, a compute phase far behind the fetch, and stale
    implied-vol surfaces. Desk-only (internal_only), flat low heat — the sheet shows
    these as caveats on everything else, not as stories. Disk reads only: the
    surface census reads the snapshot parquets straight off disk via
    datafeed._read_snapshot (stale_surfaces() routes through the mode-dependent
    getters, which PULL when the process runs with DATAFEED_MODE=bloomberg)."""
    from src import hotsheet
    from .datafeed import _read_snapshot, stale_iv_reasons, TENOR_LABELS

    common = dict(tag="DATA", section="Data health", page="Data health",
                  book="meta", weekly=False, internal_only=True)
    items = []
    frames = snapshot_frames()

    missing = missing_core_frames(frames)
    if missing:
        items.append(hotsheet.item(
            key="core:missing", heat=RADAR_HEAT, metric=f"{len(missing)} missing",
            text=("Core snapshot frames are **missing** from data/snapshot/ ("
                  + ", ".join(missing) + ") — the modules riding them are on "
                  "fallback/demo data."),
            sub="one pull writes all core frames together",
            value=float(len(missing)), **common))

    lag = compute_lag_h(frames)
    if np.isfinite(lag) and lag > RADAR_COMPUTE_LAG_H:
        items.append(hotsheet.item(
            key="compute:lag", heat=RADAR_HEAT, metric=f"{lag:.0f}h behind",
            text=(f"The morning compute is **{lag:.0f} hours behind** the fetched data — "
                  "signals, own-curve fits and the signal cache are stale. "
                  "Run:  python snapshot.py --compute"),
            sub="fetch ran, compute didn't",
            value=float(round(lag, 1)), **common))

    # Stale-surface census on the SNAPSHOT files — the same stale_iv_reasons guard the
    # vol/skew/term pages apply, run over the frames one pull writes to disk.
    surfaces = [("implied_vol.parquet", "1M ATM"),
                ("skew_put.parquet", "skew 90% put"),
                ("skew_call.parquet", "skew 110% call")]
    surfaces += [(f"term_{lab.lower()}.parquet", f"term {lab}") for lab in TENOR_LABELS]
    tickers = list(universe.INSTRUMENTS)
    stale_markets, n_rows = set(), 0
    for fname, _surface in surfaces:
        try:
            f = _read_snapshot(fname)
            reasons = stale_iv_reasons(f.reindex(columns=tickers)) if f is not None else {}
        except Exception:
            reasons = {}
        stale_markets.update(reasons)
        n_rows += len(reasons)
    if stale_markets:
        names = sorted(universe.name(t) for t in stale_markets)
        items.append(hotsheet.item(
            key="surfaces:stale", heat=RADAR_HEAT, metric=f"{len(stale_markets)} stale",
            text=(f"**{len(stale_markets)} market(s)** show a frozen or dead implied-vol "
                  f"surface ({n_rows} surface rows) — already left unscored by the "
                  "vol/skew/term strategies."),
            sub=", ".join(names[:4]) + ("…" if len(names) > 4 else ""),
            value=float(len(stale_markets)), **common))
    return items
