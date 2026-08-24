"""Signal cache — precomputed per-(day, strategy) raw TA signal rows, whole FICC book.

WHY THIS EXISTS. The TA Backtester walks forward day by day calling the SAME
`find_opportunities()` functions the live pages use (~90ms+ per call — see tabt.py's
perf note), so a 10-year confluence backtest is tens of thousands of calls and hours of
wall clock now that the deep store serves 10y of history. But one `find_opportunities()`
call scores EVERY product in the frame at once — so computing each (day, strategy) ONCE
for the whole book and caching the raw rows turns any later backtest into a parquet
slice, and gives the Signal Ledger (hit-rate analytics) its raw material for free. The
full 10-year × 16-strategy backfill is pure local compute off the deep store — no
Bloomberg, no Terminal.

THE WINDOW CONVENTION (the correctness decision). Each day's rows are computed on the
trailing LOOKBACK_DAYS (400) sessions ending that day — the same window the LIVE hub
sees — so a cached day-d signal is "what the dashboard would have said on day d",
canonical and independent of any backtest run's own start date. (The backtester's live
fallback path slices from its run-anchored warm-up buffer instead; every shipped
strategy reads a bounded trailing window well inside 400 sessions, so the two agree —
parity-verified. A future full-history-scanning strategy would make them diverge, and
the cache's fixed window is the defensible one.)

STORE  data/signal_cache/
  ficc_YYYY.parquet   long rows: date, strategy, instruments, market, signal, direction,
                      metric, level — the columns tascore.ta_flagged / score_products /
                      tabt read. context/metric_label are dropped (nothing downstream
                      uses them; they're the bulk of the string weight).
  ficc_days.parquet   coverage log: every (date, strategy) pair actually COMPUTED. An
                      event-driven strategy legitimately emits zero rows most days, so
                      "no rows" cannot mean "cache miss" — presence lives here.

Mean Reversion rows are cached exactly as emitted (instruments = "A / B" pair strings);
the backtester translates them onto a leg at read time, same as its live path.

PANAMA DRIFT (accepted, documented). Rows are computed on the deep store's CURRENT
panama-adjusted series. After a later roll shifts pre-roll history by a constant,
diff-based indicators are invariant but %-based metrics recomputed today would differ
slightly from the cached values. This is inherent to continuation series, the effect is
a few bps on old rows, and the cache is "what the desk saw at the time" — which is the
honest object for hit-rate accounting. `python backfill_signals.py --rebuild` recomputes
from scratch if wanted.

No Streamlit here — tabt.py / sigledger.py / snapshot.py drive this module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import deepstore, universe
from .datafeed import (LOOKBACK_DAYS, get_history, get_history_ta,
                       get_volume_history)

DATA = Path(__file__).resolve().parents[1] / "data"
STORE_DIR = DATA / "signal_cache"

KEEP_COLS = ["strategy", "market", "instruments", "signal", "direction", "metric", "level"]

# The backtestable strategy modules, resolved lazily so this module imports fast and a
# broken strategy import can't take the cache down with it. Mean Reversion sits OUTSIDE
# tabt.STRATEGY_MODULES (pair-based, special-cased there) but caches like any other:
# its find_opportunities ignores the per-ticker seam and emits "A / B" pair rows.
def _modules() -> dict:
    from . import tabt
    from .strategies import mean_reversion
    return {**tabt.STRATEGY_MODULES, "Mean Reversion": mean_reversion}


def _volume_strategies() -> set:
    from . import tabt
    return set(tabt.VOLUME_STRATEGIES)


def cacheable_strategies(scope: str = "ficc") -> list:
    """Every strategy the cache covers for `scope`, in display order: the backtestable
    modules plus pair-based Mean Reversion (FICC only — no single-name price line to
    pair on the equities side, same exclusion as the Equities TA page)."""
    from . import tascore
    names = set(_modules()) | {"Mean Reversion"}
    if scope == "equities":
        names -= {"Mean Reversion"}
    return [s for s in tascore.TA_STRATEGIES if s in names]


# ---------------------------------------------------------------------------
# store I/O — yearly partitions so the daily append only rewrites one small file
# ---------------------------------------------------------------------------
def _rows_path(year: int, scope: str = "ficc") -> Path:
    return STORE_DIR / f"{'eq' if scope == 'equities' else 'ficc'}_{year}.parquet"


def _days_path(scope: str = "ficc") -> Path:
    return STORE_DIR / f"{'eq' if scope == 'equities' else 'ficc'}_days.parquet"


def _read_years(years, scope: str = "ficc") -> pd.DataFrame:
    frames = []
    for y in sorted(set(years)):
        p = _rows_path(y, scope)
        if p.exists():
            try:
                frames.append(pd.read_parquet(p))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["date", *KEEP_COLS])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def coverage(scope: str = "ficc") -> pd.DataFrame:
    """The (date, strategy) pairs the cache has actually computed."""
    p = _days_path(scope)
    if not p.exists():
        return pd.DataFrame(columns=["date", "strategy"])
    try:
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        return d
    except Exception:
        return pd.DataFrame(columns=["date", "strategy"])


def rows_for(start, end, strategies=None, scope: str = "ficc") -> pd.DataFrame:
    """Cached raw rows with date ∈ [start, end] (and strategy ∈ `strategies` if given),
    long frame with a `date` column. Pure disk — no compute."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = _read_years(range(start.year, end.year + 1), scope)
    if out.empty:
        return out
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    if strategies is not None:
        out = out[out["strategy"].isin(set(strategies))]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# frame building — the tabt-parity data path, computed once per process
# ---------------------------------------------------------------------------
_FRAMES_CACHE: dict = {}


def book_frames(refresh: bool = False, scope: str = "ficc"):
    """(signal_hist, vol_hist) for the WHOLE `scope` universe at full cached depth.
    FICC: built exactly the way tabt._load_history builds its frames (get_history_ta /
    volume feed + deepstore.overlay), so cached rows can never drift from what the
    backtester's live path would compute on the same data. Equities: the eqta parquet
    cache (split+dividend-adjusted Yahoo closes + volume — the same frames the Equities
    TA page scores)."""
    if scope in _FRAMES_CACHE and not refresh:
        return _FRAMES_CACHE[scope]
    if scope == "equities":
        from . import eqta
        sig, vol = eqta.load_history()
    else:
        uni = sorted(universe.INSTRUMENTS)
        today = pd.Timestamp.today().normalize()
        # These frames are the Signal Ledger's MEASURING STICK, so they must be a function
        # of the deep store alone — identical whether the process runs in snapshot or
        # bloomberg mode. Two things enforce that (both added 2026-08-24 after the ledger
        # sat frozen for 10 days): never ask for history the store cannot serve, and require
        # the panama upgrade rather than letting a deeper live feed veto it. Asking for
        # `today − 10y − 30d` against a store floored at 2016-08-08 tripped deepstore's
        # depth heuristic in bloomberg mode only, leaving 14 products on their RAW
        # roll-gapped series whose roll gaps then re-marked ~6% of settled outcomes.
        start = today - pd.DateOffset(years=deepstore.STORE_YEARS, days=30)
        floor = deepstore.first_date()
        if floor is not None:
            start = max(start, floor)
        sig = get_history_ta(uni, start=start, end=today)
        pnl = get_history(uni, start=start, end=today)
        vol = get_volume_history(uni, start=start, end=today)
        _, _, sig, vol = deepstore.overlay(uni, start, today, pnl, sig, vol, require_deep=True)
    _FRAMES_CACHE[scope] = (sig, vol)
    return sig, vol


def compute_day(day, strategies, sig: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Raw find_opportunities() rows for every product, one `day`, the given strategies —
    computed on the trailing LOOKBACK_DAYS sessions ending `day` (the live hub's window)."""
    day = pd.Timestamp(day)
    hs = sig.loc[:day].iloc[-LOOKBACK_DAYS:]
    vs = vol.loc[:day].iloc[-LOOKBACK_DAYS:] if vol is not None else None
    mods, vol_strats = _modules(), _volume_strategies()
    frames = []
    for s in strategies:
        mod = mods.get(s)
        if mod is None:
            continue
        kwargs = {"history": hs}
        if s in vol_strats:
            kwargs["volume"] = vs
        try:
            raw = mod.find_opportunities(**kwargs)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        keep = [c for c in KEEP_COLS if c in raw.columns]
        frames.append(raw[keep].assign(date=day))
    if not frames:
        return pd.DataFrame(columns=["date", *KEEP_COLS])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# building / extending the cache
# ---------------------------------------------------------------------------
def missing_pairs(days, strategies, scope: str = "ficc") -> list:
    """(day, [strategies-not-yet-computed]) for each day that has gaps."""
    cov = coverage(scope)
    have = set(zip(cov["date"], cov["strategy"])) if not cov.empty else set()
    out = []
    for d in days:
        d = pd.Timestamp(d)
        todo = [s for s in strategies if (d, s) not in have]
        if todo:
            out.append((d, todo))
    return out


def _persist(new_rows: pd.DataFrame, done_pairs: list, scope: str = "ficc") -> None:
    """Merge `new_rows` into the yearly partitions (replacing any existing rows for the
    same (date, strategy)) and log `done_pairs` into the coverage file."""
    if not new_rows.empty:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        for y, chunk in new_rows.groupby(new_rows["date"].dt.year):
            old = _read_years([y], scope)
            if not old.empty:
                repl = set(zip(chunk["date"], chunk["strategy"]))
                old = old[~pd.Series(list(zip(old["date"], old["strategy"])),
                                     index=old.index).isin(repl)]
                chunk = pd.concat([old, chunk], ignore_index=True)
            chunk.sort_values(["date", "strategy", "instruments"]).to_parquet(
                _rows_path(int(y), scope), index=False)
    if done_pairs:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        cov = coverage(scope)
        add = pd.DataFrame(done_pairs, columns=["date", "strategy"])
        cov = (pd.concat([cov, add], ignore_index=True)
               .drop_duplicates(["date", "strategy"]).sort_values(["date", "strategy"]))
        cov.to_parquet(_days_path(scope), index=False)


def extend(days=None, strategies=None, log=print, flush_every: int = 25,
           scope: str = "ficc", max_days: int | None = None) -> int:
    """Compute + persist every missing (day, strategy) pair. `days=None` = every session
    the scope's signal frame has; `max_days` limits to the LAST n sessions of the frame
    (the routine-refresh guard — an incomplete equities backfill must not hijack the
    daily pull). Resume-safe (already-computed pairs are skipped) and incremental
    (persists every `flush_every` days so an interrupted run keeps its work). Returns
    the number of (day, strategy) pairs computed."""
    import time
    strategies = list(strategies or cacheable_strategies(scope))
    sig, vol = book_frames(scope=scope)
    if sig is None or sig.empty:
        log(f"  signal cache[{scope}]: no signal history available — nothing done")
        return 0
    all_days = sig.dropna(how="all").index
    if days is not None:
        want = {pd.Timestamp(d) for d in days}
        all_days = all_days[all_days.isin(want)]
    if max_days is not None:
        all_days = all_days[-max_days:]
    todo = missing_pairs(all_days, strategies, scope)
    if not todo:
        return 0
    n_pairs = sum(len(s) for _, s in todo)
    log(f"  signal cache[{scope}]: {len(todo)} days / {n_pairs} (day×strategy) pairs to compute")
    t0, done, rows_buf, pairs_buf = time.time(), 0, [], []
    for i, (d, strats) in enumerate(todo):
        rows = compute_day(d, strats, sig, vol)
        if not rows.empty:
            rows_buf.append(rows)
        pairs_buf.extend((d, s) for s in strats)
        done += len(strats)
        if (i + 1) % flush_every == 0 or i == len(todo) - 1:
            _persist(pd.concat(rows_buf, ignore_index=True) if rows_buf
                     else pd.DataFrame(columns=["date", *KEEP_COLS]), pairs_buf, scope)
            rows_buf, pairs_buf = [], []
            rate = done / max(time.time() - t0, 1e-9)
            eta_min = (n_pairs - done) / max(rate, 1e-9) / 60
            log(f"    {done}/{n_pairs} pairs ({rate:.1f}/s, ~{eta_min:.0f} min left)")
    return done


def daily_update(log=print, scope: str = "ficc", max_days: int | None = None) -> int:
    """The compute-phase entry point: cache any sessions not yet covered (normally just
    today; self-heals longer gaps after a skipped day). Cheap once backfilled. Equities
    callers pass `max_days` so a routine pull only tops up recent sessions — deep
    backfill stays with backfill_signals.py."""
    return extend(log=log, scope=scope, max_days=max_days)
