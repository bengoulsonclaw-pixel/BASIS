"""Deep futures price store — ~10y of back-history for the TA Backtester.

WHY THIS EXISTS. The app's whole FICC book runs on Bloomberg 'A' active-contract
generics ("SERA Comdty", "TYA Comdty", …), and an 'A' ticker only serves history for
as long as the CURRENT active contract has been listed — Treasuries ~6 months, 1M SOFR
~15 months (see cotdata._hist_source, verified live 2026-06-15). On top of that the
daily snapshot caps every pull at datafeed.LOOKBACK_DAYS (400 business days). Both
limits made backtests before ~2025 impossible and starve the longest-window live
strategies (200-day MAs) on the truncated products.

WHAT IT DOES. Keeps a persistent, incrementally-extended store (cotdata price-DB
pattern) of the '1' FIRST-generic chain for every product in the universe:

  deep_prices.parquet   raw spliced '1' series (PX_SETTLE), columns = the app's own
                        'A' tickers so everything downstream stays aligned
  deep_front2.parquet   the '2' second-generic — used to price each roll gap
  deep_contract.parquet FUT_CUR_GEN_TICKER strings per date — the exact roll dates
  deep_volume.parquet   FUT_AGGTE_VOL (aggregate across all contracts, same value on
                        any generic of the chain)
  deep_yields.parquet   benchmark yields for BOND futures (universe.BOND_YIELD_SOURCE),
                        deep-pulled — these are spliced benchmark series with no rolls

ROLL ADJUSTMENT (difference / "panama", computed on read, never stored). The raw '1'
series jumps at every roll (new contract ≠ old contract). On the last day the old
contract is front, the '2' generic already holds the incoming one, so the roll's
artificial gap is exactly (px2 − px1) on that day. Anchored to TODAY's prices, every
date at or before each roll is shifted by that gap, cumulatively — point moves (and
therefore point-value × move P&L) are preserved exactly; only absolute LEVELS deep in
the past drift from what the screen showed at the time (continuation levels). When the
'2' quote is missing on a roll date the whole observed jump is attributed to the roll
(flagged approximate — better than leaving a fake gap for the TA to trade).

Adjustment is always computed on the FULL stored series and only then sliced — offsets
depend on every roll after the window, so adjusting a slice would re-anchor it.

Cash indices in the universe (e.g. "KOSPI2 Index") have no generic chain: their own
ticker serves deep history natively, so they pass through with no '2'/contract pull
and no adjustment.

Bloomberg cost: hits = securities × fields, INDEPENDENT of date depth — the full 10y
backfill is ~350 hits (88 × ('1','2',contract,volume) minus cash/dup sources), about
a fifth of one normal morning snapshot. The daily incremental append is the same
security×field count on a 10-day tail. Off-Bloomberg (snapshot / mock) every writer is
a no-op and every reader just serves what's on disk.

No Streamlit here — tabt.py (and anything else) drives this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import universe
from .datafeed import (DEFAULT_FIELD, MODE, PRICE_FIELD_OVERRIDE, VOLUME_FIELD,
                       _bdh_to_wide)

DATA = Path(__file__).resolve().parents[1] / "data"
STORE_DIR = DATA / "price_store"

STORE_YEARS = 10
STORE_BUFFER_DAYS = 10     # re-pull this many trailing days each run (revised settles)
STORE_CHUNK = 8            # backfill in small ticker groups — avoids single-request data caps
HEAL_MIN_DAYS = 750        # < ~3y of daily price stored -> treated as truncated -> re-backfill
FRESH_TOLERANCE_DAYS = 7   # store may lag the live feed this much before overlay() refuses it

CONTRACT_FIELD = "FUT_CUR_GEN_TICKER"   # the actual contract behind a generic, per date

_FRAMES = ("prices", "front2", "contract", "volume", "yields")


# ---------------------------------------------------------------------------
# 'A' active-generic  ->  '1' / '2' chained-generic mapping
# (same rule cotdata._hist_source verified live; non-'A' names — cash indices,
#  yield sources — pass through unchanged and never get a '2'.)
# ---------------------------------------------------------------------------
def _src(tkr: str, gen_n: str) -> str:
    gen, key = tkr.rsplit(" ", 1)
    if gen.endswith("A"):
        gen = gen[:-1] + gen_n
    return f"{gen} {key}"


def _has_chain(tkr: str) -> bool:
    """True when `tkr` is an 'A' generic (rolls; needs '2' + contract strings)."""
    return _src(tkr, "1") != tkr


# ---------------------------------------------------------------------------
# store I/O  (wide date-indexed parquets, cotdata merge semantics)
# ---------------------------------------------------------------------------
def _path(name: str) -> Path:
    return STORE_DIR / f"deep_{name}.parquet"


def _read(name: str) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _write(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        return
    try:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        df.rename_axis("date").reset_index().to_parquet(_path(name), index=False)
    except Exception:
        pass


def _merge(store: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of dates/columns; `new` wins on overlap (revised settles update)."""
    if store is None or store.empty:
        return new.sort_index() if new is not None and not new.empty else pd.DataFrame()
    if new is None or new.empty:
        return store.sort_index()
    return new.combine_first(store).sort_index()


# ---------------------------------------------------------------------------
# Bloomberg pulls (fetch phase only — every function below is a no-op off-Bloomberg)
# ---------------------------------------------------------------------------
def _pull_field(src_map: dict, field_for: dict, start, end) -> pd.DataFrame | None:
    """One batched bdh per field over `src_map` (source ticker -> app ticker), returned
    keyed by the APP ticker. `field_for` maps app ticker -> Bloomberg field."""
    if not src_map:
        return None
    from . import bbg as blp
    by_field: dict = {}
    for s, t in src_map.items():
        by_field.setdefault(field_for[t], []).append(s)
    cols = {}
    for fld, grp in by_field.items():
        try:
            wide = _bdh_to_wide(blp.bdh(tickers=grp, flds=fld, start_date=start, end_date=end))
        except Exception:
            wide = None
        if wide is None:
            continue
        for s in grp:
            if s in wide.columns:
                cols[src_map[s]] = wide[s]
    return pd.DataFrame(cols) if cols else None


def _pull_group(tickers: list, start, end, names=None) -> dict:
    """Deep frames for `tickers` over [start, end], keyed by app ticker. `names`
    restricts which frames are PULLED (default: all five) — the daily tail sources
    volume/yields from the snapshot parquets instead (see _tails_from_snapshot)."""
    names = set(_FRAMES if names is None else names)
    px_field = {t: PRICE_FIELD_OVERRIDE.get(t, DEFAULT_FIELD) for t in tickers}
    chained = [t for t in tickers if _has_chain(t)]
    out = {}
    if "prices" in names:
        out["prices"] = _pull_field({_src(t, "1"): t for t in tickers}, px_field, start, end)
    if "front2" in names:
        out["front2"] = _pull_field({_src(t, "2"): t for t in chained}, px_field, start, end)
    if "contract" in names:
        out["contract"] = _pull_field({_src(t, "1"): t for t in chained},
                                      {t: CONTRACT_FIELD for t in chained}, start, end)
    if "volume" in names:
        out["volume"] = _pull_field({_src(t, "1"): t for t in tickers},
                                    {t: VOLUME_FIELD for t in tickers}, start, end)
    if "yields" in names:
        bonds = [t for t in tickers if universe.yield_source(t)]
        ysrcs: dict = {}
        for t in bonds:
            ysrcs.setdefault(universe.yield_source(t), []).append(t)
        yw = _pull_field({s: s for s in ysrcs}, {s: "PX_LAST" for s in ysrcs}, start, end)
        if yw is not None:
            out["yields"] = pd.DataFrame({t: yw[s] for s, futs in ysrcs.items()
                                          for t in futs if s in yw.columns})
        else:
            out["yields"] = None
    return out


def _tails_from_snapshot(tickers: list, start) -> dict:
    """The VOLUME daily tail read from the snapshot parquet the morning pull wrote
    MOMENTS earlier in the same fetch phase — identical data to what this module
    re-pulled until 2026-08-18 (FUT_AGGTE_VOL returns the chain-wide aggregate on
    ANY generic, and the raw=True snapshot never ffills volume). ~89 hits/day
    saved. YIELDS deliberately stay a Bloomberg pull (review 2026-08-18):
    yields.parquet is the get_yield_history frame, which is .ffill()ed — holiday
    cells filled, a dead source frozen forward daily — and this store's raw-print
    semantics are load-bearing (curvemon drops cross-calendar holiday rows via
    dropna; an ffilled cell would fake a spread observation into 10y z history).
    Returns {} unless the snapshot is ACTUALLY fresh — covering the tail window
    AND written within ~4 days — so a standalone bloomberg-mode run days later
    (backfill_deep.py) falls back to Bloomberg for everything instead of silently
    desynchronizing the frames."""
    out = {}
    for name, fname in (("volume", "volume.parquet"),):
        try:
            df = pd.read_parquet(DATA / "snapshot" / fname)
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception:
            return {}
        if df.empty or df.index.max() < pd.Timestamp(start).normalize() \
                or df.index.max() < pd.Timestamp.now().normalize() - pd.Timedelta(days=4):
            return {}
        # the snapshot frames are column-reindexed to the WHOLE book (all-NaN columns
        # for products without the series) — drop those so the deep store only ever
        # holds columns that carry data, same as a real pull returns
        tail = df.loc[df.index >= pd.Timestamp(start).normalize()].dropna(axis=1, how="all")
        out[name] = tail[[t for t in tickers if t in tail.columns]]
    return out


def update(tickers=None, years: int = STORE_YEARS,
           buffer_days: int = STORE_BUFFER_DAYS, log=print) -> pd.DataFrame:
    """Extend the store when on Bloomberg: full-`years` backfill for any ticker not yet
    held deep (or truncated below HEAL_MIN_DAYS — self-healing), a `buffer_days` tail
    pull for the rest. Returns the raw price store either way; read-only off-Bloomberg."""
    stores = {n: _read(n) for n in _FRAMES}
    if MODE != "bloomberg":
        return stores["prices"]
    tickers = list(tickers or universe.INSTRUMENTS)
    today = pd.Timestamp.now().normalize()
    full_start = today - pd.DateOffset(years=years)
    px = stores["prices"]

    def _held(t):
        return int(px[t].notna().sum()) if (not px.empty and t in px.columns) else 0

    have = [t for t in tickers if _held(t) >= HEAL_MIN_DAYS]
    need_full = [t for t in tickers if _held(t) < HEAL_MIN_DAYS]
    groups: list = []
    if have:
        inc_start = max(px.index.max() - pd.Timedelta(days=buffer_days), full_start)
        groups.append((have, inc_start, True))         # tail: reuse snapshot frames
    groups += [(need_full[i:i + STORE_CHUNK], full_start, False)
               for i in range(0, len(need_full), STORE_CHUNK)]
    if need_full:
        log(f"  Deep store: backfilling {len(need_full)} tickers ~{years}y "
            f"(~{len(need_full) * 4} hits), tailing {len(have)}")
    pulled = {n: [] for n in _FRAMES}
    for grp, start, tail in groups:
        snap_frames = _tails_from_snapshot(grp, start) if tail else {}
        frames = _pull_group(grp, start, today,
                             names=[n for n in _FRAMES if n not in snap_frames])
        frames.update(snap_frames)
        for n in _FRAMES:
            if frames.get(n) is not None:
                pulled[n].append(frames[n])
    for n in _FRAMES:
        if pulled[n]:
            new = pd.concat(pulled[n], axis=1)
            new = new.loc[:, ~new.columns.duplicated()]
            stores[n] = _merge(stores[n], new)
            _write(stores[n], n)
    return stores["prices"]


# ---------------------------------------------------------------------------
# Roll detection + difference ("panama") back-adjustment
# ---------------------------------------------------------------------------
def _offsets(s: pd.Series, px2: pd.Series | None, ct: pd.Series | None) -> list:
    """[(last_day_of_old_front, roll_gap), …] for one raw '1' series `s`. Roll dates
    come from FUT_CUR_GEN_TICKER changes; each gap is ('2' − '1') on the day before —
    falling back to the observed jump itself when the '2' quote is missing."""
    if ct is None:
        return []
    c = ct.reindex(s.index).ffill()
    prev = c.shift()
    roll_days = s.index[(c != prev) & c.notna() & prev.notna()]
    offs = []
    for d in roll_days:
        i = s.index.get_loc(d)
        if i == 0:
            continue
        dp = s.index[i - 1]
        gap = np.nan
        if px2 is not None and dp in px2.index and pd.notna(px2.loc[dp]):
            gap = float(px2.loc[dp]) - float(s.loc[dp])
        if pd.isna(gap):
            gap = float(s.loc[d]) - float(s.loc[dp])
        offs.append((dp, gap))
    return offs


def _adjust(s: pd.Series, offs: list) -> pd.Series:
    adj = s.astype(float).copy()
    for dp, gap in offs:
        adj.loc[:dp] += gap
    return adj


# ---------------------------------------------------------------------------
# Readers (any mode — pure disk)
# ---------------------------------------------------------------------------
def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def get_raw(tickers, start=None, end=None) -> pd.DataFrame:
    """Raw spliced '1' prices (ACTUAL traded levels — the right series for point-in-time
    lookups like FX conversion, where an adjusted level would be fiction)."""
    px = _read("prices")
    cols = [t for t in tickers if t in px.columns]
    return _slice(px[cols].dropna(how="all"), start, end) if cols else pd.DataFrame()


def get_adjusted(tickers, start=None, end=None) -> pd.DataFrame:
    """Difference-back-adjusted continuation prices, anchored to the latest stored
    settle. Adjustment is computed on each FULL series, then sliced."""
    px, p2, ct = _read("prices"), _read("front2"), _read("contract")
    out = {}
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if s.empty:
            continue
        offs = _offsets(s, p2[t].dropna() if t in p2.columns else None,
                        ct[t] if t in ct.columns else None)
        out[t] = _adjust(s, offs)
    return _slice(pd.DataFrame(out), start, end) if out else pd.DataFrame()


def _front_price_anchor():
    """The price series the REST of BASIS calls a product's price — the morning snapshot's
    'A' (most-active) generic. Used only to re-anchor the STIR rate series' level.

    Deliberately NOT this store's own raw column, which is the '1' (first-expiry) generic:
    for SOFR the two are different contracts (96.04 vs 96.355 on 2026-08-26 — 31.5bp), and
    a chart whose level disagrees with the price shown on every other page is its own bug.
    Read as a plain file so this module keeps no import on datafeed (which imports it), and
    located RELATIVE TO STORE_DIR so a relocated store (the test fixture's tmp dir) doesn't
    silently reach back into the live data/ — which is exactly what the first cut did."""
    p = STORE_DIR.parent / "snapshot" / "prices.parquet"
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_ta(tickers, start=None, end=None) -> pd.DataFrame:
    """Deep signal-space history — datafeed.get_history_ta semantics on the deep store:
    STIRs as 100 − adjusted price, bond futures as their deep benchmark yield, everything
    else the adjusted price. A bond with no deep yield series is DROPPED (never served as
    a price — the FI strategies and sign convention assume yield space). Non-FI columns
    are TRIMMED to their positive tail: difference-adjustment drags heavy-contango
    products (NGA) through zero deep in the past, and %-based signal maths (momentum,
    MA gaps, logs) is meaningless across a zero crossing — P&L space (get_adjusted) keeps
    the full depth, where point moves stay exact regardless of level."""
    out = get_adjusted(tickers, start, end)
    if out.empty:
        return out
    ylds = _read("yields")
    raw = _front_price_anchor()
    for t in list(out.columns):
        if universe.is_stir(t):
            out[t] = 100.0 - out[t]
            # RE-ANCHOR to the front contract's true implied rate. 100 − adjusted is a rate
            # path plus every roll gap difference-adjustment has removed since the store was
            # built, so the LEVEL drifts away from the market: on 2026-08-26 this read SOFR
            # at 3.6450 while the front contract implied 3.9600 — 31.5bp — under an axis
            # labelled "Yield (%)", with those levels quoted to clients as objectives and
            # invalidations. A constant shift leaves every difference (and therefore every
            # signal, hit and backtest P&L) untouched, so this corrects what is read as a
            # number without disturbing what is measured as a move. (2026-08-26)
            try:
                r = (100.0 - raw[t]).dropna() if (raw is not None and t in raw.columns) else None
                a = out[t].dropna()
                if r is not None and len(r) and len(a):
                    common = a.index.intersection(r.index)
                    if len(common):
                        d = common.max()
                        out[t] = out[t] + (float(r.loc[d]) - float(a.loc[d]))
            except Exception:
                pass
        elif universe.is_bond(t):
            if t in ylds.columns:
                out[t] = ylds[t].reindex(out.index).ffill()
            else:
                out = out.drop(columns=[t])
        else:
            s = out[t].dropna()
            bad = s.index[s.le(0)]
            if len(bad):
                out.loc[out.index <= bad.max(), t] = np.nan
    return out


def get_volume(tickers, start=None, end=None) -> pd.DataFrame:
    vol = _read("volume")
    cols = [t for t in tickers if t in vol.columns]
    return _slice(vol[cols], start, end) if cols else pd.DataFrame()


def get_front2(tickers, start=None, end=None) -> pd.DataFrame:
    """Raw '2' second-generic prices (ACTUAL levels, like get_raw). Front1 − front2 is
    the live 1st/2nd calendar spread — the curve-monitor's time-spread series."""
    p2 = _read("front2")
    cols = [t for t in tickers if t in p2.columns]
    return _slice(p2[cols].dropna(how="all"), start, end) if cols else pd.DataFrame()


def get_yields(tickers, start=None, end=None) -> pd.DataFrame:
    """Deep benchmark yields (%) for BOND futures, keyed by the future's own ticker
    (universe.BOND_YIELD_SOURCE series) — the yield-space legs for curve spreads."""
    ylds = _read("yields")
    cols = [t for t in tickers if t in ylds.columns]
    return _slice(ylds[cols].dropna(how="all"), start, end) if cols else pd.DataFrame()


def first_date():
    """The store's earliest date, or None when there is no store. Callers that MEASURE off
    the store (rather than backtest on it) clamp their request to this — asking for history
    the store cannot serve is what tripped overlay's depth heuristic and froze the Signal
    Ledger for 10 days in Aug 2026."""
    px = _read("prices")
    return None if px.empty else px.index.min()


def coverage() -> pd.DataFrame:
    """Per-ticker diagnostics: first/last date, days held, rolls detected."""
    px, p2, ct = _read("prices"), _read("front2"), _read("contract")
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if s.empty:
            continue
        n_rolls = len(_offsets(s, p2[t].dropna() if t in p2.columns else None,
                               ct[t] if t in ct.columns else None))
        rows.append({"ticker": t, "first": s.index.min().date(), "last": s.index.max().date(),
                     "days": len(s), "rolls": n_rolls})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# tabt seam — column-wise upgrade of the live-feed frames to deep history
# ---------------------------------------------------------------------------
def overlay(tickers, start, end, pnl_hist, signal_hist, vol_hist, require_deep: bool = False):
    """Swap each ticker's feed columns for the deep-store versions where the store
    genuinely helps: it must reach at least as far back as the feed AND be no more than
    FRESH_TOLERANCE_DAYS staler at the recent end (a lapsed store must never silently
    truncate a backtest's right edge — fall back to the live feed instead). Columns are
    swapped whole, never mixed: feed 'A' prices and adjusted '1' prices are different
    series and splicing them would re-introduce exactly the gaps this store removes.

    `require_deep=True` skips the DEPTH test only (never the freshness test): the caller
    is measuring rather than backtesting, and for a measurement a raw roll-gapped series
    is never the right answer however far back the feed reaches. This exists because that
    heuristic silently froze the Signal Ledger for 10 days (2026-08-14 → 08-24): the
    ledger asked for `today − 10y − 30d` while the store's floor is fixed at 2016-08-08,
    so in bloomberg mode the live 'A' generic reached back PAST the store, the depth test
    refused the panama upgrade for 14 products, and their raw roll gaps re-marked ~6% of
    settled outcomes every morning. See [[project-signal-ledger]].

    Returns (deep_used, pnl_hist, signal_hist, vol_hist); no-ops to the inputs when the
    store is absent (VPS / demo / pre-backfill)."""
    adj = get_adjusted(tickers, start, end)
    if adj.empty:
        return [], pnl_hist, signal_hist, vol_hist
    ta = get_ta(tickers, start, end)
    vol = get_volume(tickers, start, end) if vol_hist is not None else pd.DataFrame()
    deep_used = []
    for t in tickers:
        if t not in adj.columns or t not in pnl_hist.columns:
            continue
        feed, deep = pnl_hist[t].dropna(), adj[t].dropna()
        if feed.empty or deep.empty:
            continue
        if t not in ta.columns:
            continue                                        # no deep SIGNAL series (bond w/o yields)
        if deep.index.min() > feed.index.min() and not require_deep:
            continue                                        # store adds no depth here
        if (feed.index.max() - deep.index.max()).days > FRESH_TOLERANCE_DAYS:
            continue                                        # store gone stale — keep the feed
        deep_used.append(t)
    if not deep_used:
        return [], pnl_hist, signal_hist, vol_hist
    idx = pnl_hist.index.union(adj.index).sort_values()
    pnl_hist = pnl_hist.reindex(idx)
    signal_hist = signal_hist.reindex(idx)

    def _asof(series: pd.Series, feed_col: pd.Series) -> pd.Series:
        """Deep column onto the union index AS-OF ffilled (a raw reindex would re-punch
        exchange-holiday holes — same lesson as datafeed._deep_upgrade: a hole changes
        window-scanning strategies' row counts). The tail beyond the deep store's last
        settle keeps the FEED's values so "today" never goes missing on a live run."""
        d = series.dropna()
        merged = d.reindex(idx, method="ffill")
        tail = idx[idx > d.index.max()]
        if len(tail) and feed_col is not None:
            merged.loc[tail] = feed_col.reindex(tail)
        return merged

    for t in deep_used:
        pnl_hist[t] = _asof(adj[t], pnl_hist[t])
        if t in ta.columns:
            signal_hist[t] = _asof(ta[t], signal_hist[t])
    if vol_hist is not None and not vol.empty:
        vol_hist = vol_hist.reindex(idx)
        for t in deep_used:
            if t in vol.columns:
                vol_hist[t] = vol[t].reindex(idx)      # volume is a FLOW — never ffilled
    return deep_used, pnl_hist, signal_hist, vol_hist
