"""Data adapter — the single seam between the app and the price source.

Every strategy calls `get_history(...)`. Today it returns SYNTHETIC data so
the whole app runs without Bloomberg. When your Bloomberg API is live, set
MODE = "bloomberg" below and real settlement data flows through the exact
same function — nothing else in the app changes.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import (INSTRUMENTS, IMPLIED_VOL_OVERRIDE, PRICE_FIELD_OVERRIDE,
                       is_stir, is_bond, yield_source, asset)


def _fx_override_tickers(tickers):
    """The FX tickers whose vol comes from the OTC pair source. Classified by ASSET
    CLASS, not by mere override membership — a non-FX market with a vol override
    (e.g. Euribor's listed call/put mid) must NOT fall into the OTC …V1M/25R ticker
    substitution the FX branches perform."""
    return [t for t in tickers if asset(t) == "FX" and t in IMPLIED_VOL_OVERRIDE]

# "mock"      -> synthetic data (no Bloomberg needed)
# "bloomberg" -> live Bloomberg via xbbg/blpapi (Terminal must be running)
# Default is mock; set env var DATAFEED_MODE=bloomberg to go live (no code edit needed).
MODE = os.getenv("DATAFEED_MODE", "mock")

DEFAULT_FIELD = "PX_SETTLE"   # Bloomberg futures settlement price
LOOKBACK_DAYS = 400           # business days of history to generate / pull

# Daily contract volume — the flag page's breakout-confirmation input (volume dries up
# through a flag and picks up on the break). VERIFIED LIVE 2026-06-20 (diag_volume.py):
# FUT_AGGTE_VOL (volume aggregated across ALL listed contracts of the future) is fully
# populated across every asset class with no zero gaps. The active-contract PX_VOLUME on
# the 'A' generic returns 0 on the latest session for several markets right after a roll
# (Fed Funds, the long bonds, KC/CC) — which would land exactly on the breakout bar — so
# we use the aggregate. A market with no volume series still comes back NaN, which the
# flag code treats as "no volume confirmation".
VOLUME_FIELD = "FUT_AGGTE_VOL"

# Constant-maturity 1-month ATM implied vol off the listed-option surface
# (100% moneyness = at-the-money-forward; 30 calendar days ≈ 1-month tenor),
# returned in vol points. Confirm the exact field per asset class on the
# Terminal (FLDS<GO> / OVDV<GO>) — it varies, and FX/index are often better
# taken from a dedicated source (see IMPLIED_VOL_OVERRIDE in universe.py).
IMPLIED_VOL_FIELD = "30DAY_IMPVOL_100.0%MNY_DF"

# Morning-snapshot cache: `snapshot.py` (run once with DATAFEED_MODE=bloomberg)
# writes every raw input here as parquet; with DATAFEED_MODE=snapshot the get_*
# seams read these files instead of calling Bloomberg, so the app + reports run
# all day with the Terminal closed. A missing file falls back to mock (demo).
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshot"


def _read_snapshot(name: str):
    """Read one snapshot frame (date-indexed), or None if it isn't there yet."""
    p = SNAPSHOT_DIR / name
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_history(tickers, start=None, end=None, field: str = DEFAULT_FIELD,
                raw: bool = False) -> pd.DataFrame:
    """Return a DataFrame indexed by date, one column per ticker.

    Since 2026-08-07 (Ben's call: the WHOLE app reads the deep data) the returned frame
    is upgraded column-wise to the deep price store's panama-adjusted series wherever the
    store genuinely helps — see _deep_upgrade. The 36 'A'-truncated products (Treasuries,
    Bunds, SOFR…) get full history inside the window instead of a stub, and every series
    is roll-gap-free. The adjustment anchors to the latest stored settle, so recent values
    match the feed except during 'A'-vs-'1' roll-divergence windows (the 'A' generic rolls
    on volume days before the '1' front expires — during those windows this serves the
    front contract's settle, the standard continuation-chart convention). Mock mode stays
    synthetic. `raw=True` skips the upgrade — for consumers that must show the ACTUAL
    traded/published levels of the past (client PDFs, snapshot persistence, point-in-time
    FX), where a back-adjusted level is continuation fiction."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        out = _bloomberg_history(tickers, start, end, field)
        if raw or field != DEFAULT_FIELD:
            return out
        return _deep_upgrade(out, tickers, start=start, end=end)
    if MODE == "snapshot":
        snap = _read_snapshot("prices.parquet")
        if snap is not None:
            out = snap.reindex(columns=tickers)
            if raw or field != DEFAULT_FIELD:
                return out
            return _deep_upgrade(out, tickers, start=start, end=end)
    return _mock_history(tickers, start, end, field)


# ---------------------------------------------------------------------------
# DEEP-STORE UPGRADE (2026-08-07). The deep '1'-generic store (src/deepstore.py) was
# built for the TA Backtester; Ben then asked for the WHOLE app to read it — the live
# 'A' feed only serves history for as long as the current active contract has been
# listed, which starved 36 products' long-window indicators (200d MAs unable to
# compute on any Treasury/Bund). These helpers swap feed columns for the deep store's
# panama-adjusted series INSIDE the feed's own date window, so frame shapes and every
# caller's contract are unchanged — the values just stop being truncated stubs with
# roll gaps. Whole columns only (feed 'A' and adjusted '1' are different series;
# splicing would re-introduce the gaps), except the live TAIL: dates beyond the
# store's last settle (bounded by the freshness guard, normally 0-1 days) keep the
# feed's values so "today" never goes missing on a live page.
# ---------------------------------------------------------------------------
_DEEP_CACHE: dict = {"sig": None, "adj": None, "vol": None}


def _deep_frames():
    """(adjusted_prices, volume) from the deep store, cached against ALL four chain
    files' (mtime, size) — the adjustment reads front2/contract too, so a sig on
    prices alone could serve stale rolls after a partial store write."""
    try:
        from . import deepstore
        sig = []
        for n in ("prices", "front2", "contract", "volume"):
            p = deepstore._path(n)
            if p.exists():
                st = p.stat()
                sig.append((n, st.st_mtime_ns, st.st_size))
        if not sig:
            return None, None
        sig = tuple(sig)
        if _DEEP_CACHE["sig"] != sig:
            uni = list(INSTRUMENTS)
            _DEEP_CACHE["adj"] = deepstore.get_adjusted(uni)
            _DEEP_CACHE["vol"] = deepstore.get_volume(uni)
            _DEEP_CACHE["sig"] = sig
        return _DEEP_CACHE["adj"], _DEEP_CACHE["vol"]
    except Exception:
        return None, None


def _deep_upgrade(out, tickers, kind: str = "price", start=None, end=None):
    """Column-wise upgrade of a feed frame to the deep store. Per-ticker guards mirror
    deepstore.overlay(): the store must reach at least as far back as the feed AND be no
    staler than FRESH_TOLERANCE_DAYS at the recent end — otherwise that column keeps the
    feed. The frame's index is EXTENDED to what the store can serve inside the requested
    [start, end] span (a single starved ticker's live-Bloomberg frame is otherwise stuck
    at the 'A' contract's own listing window even though the store holds the full span);
    with no explicit span the feed's own window is kept, so default-call shapes never
    change. Price columns are as-of ffilled onto the index — the feed is ffilled too, and
    a raw reindex would re-punch exchange-holiday holes that kill rolling realized-vol
    windows. Volume is a flow and is never ffilled. Whole columns only; the live TAIL
    beyond the store's last settle (bounded by the freshness guard) keeps feed values.
    No-op in mock mode / no store on disk."""
    if MODE == "mock" or out is None or getattr(out, "empty", True):
        return out
    adj, vol = _deep_frames()
    deep = vol if kind == "volume" else adj
    if deep is None or deep.empty:
        return out
    try:
        from .deepstore import FRESH_TOLERANCE_DAYS as _tol
    except Exception:
        _tol = 7
    feed = {t: out[t].dropna() for t in tickers if t in out.columns}
    ups = []
    for t in tickers:
        if t not in deep.columns or t not in feed:
            continue
        f, d = feed[t], deep[t].dropna()
        if f.empty or d.empty:
            continue
        if d.index.min() > f.index.min():
            continue                                  # store adds no depth here
        if (f.index.max() - d.index.max()).days > _tol:
            continue                                  # store gone stale — keep the feed
        ups.append(t)
    if not ups:
        return out
    lo = pd.Timestamp(start) if start is not None else out.index.min()
    hi = pd.Timestamp(end) if end is not None else out.index.max()
    if kind == "price":
        # difference-adjustment can drag a heavy-contango product's old levels through
        # ZERO (NGA: monthly rolls, min −0.9 even inside a 400d window) — %-returns,
        # logs and MA-gap maths all break on that. Keep the feed for such columns.
        ups = [t for t in ups
               if not deep[t].loc[(deep.index >= lo) & (deep.index <= hi)].dropna().le(0).any()]
        if not ups:
            return out
    ext = deep.index[(deep.index >= lo) & (deep.index <= hi)]
    idx = out.index.union(ext).sort_values()
    if len(idx) != len(out.index):
        out = out.reindex(idx)
    for t in ups:
        d = deep[t].dropna()
        if kind == "volume":
            merged = d.reindex(out.index)                     # flows keep their gaps
        else:
            merged = d.reindex(out.index, method="ffill")     # as-of, like the feed
        tail = out.index[out.index > d.index.max()]           # live days the store hasn't seen
        if len(tail):
            merged.loc[tail] = feed[t].reindex(tail)
        out[t] = merged
    return out


# ---------------------------------------------------------------------------
# MOCK feed  (deterministic so the demo is stable across runs/processes)
# ---------------------------------------------------------------------------
def _seed(ticker: str) -> int:
    return int.from_bytes(hashlib.md5(ticker.encode()).digest()[:4], "little")


def _mock_series(ticker: str, base: float, n: int, vol_frac: float = 0.012) -> np.ndarray:
    rng = np.random.default_rng(_seed(ticker))
    vol = base * vol_frac       # ~1.2% daily move (STIRs get a tiny frac — see _mock_history)
    kappa = 0.015               # gentle pull back toward the base level
    prices = np.empty(n)
    prices[0] = base
    for i in range(1, n):
        pull = kappa * (base - prices[i - 1])
        prices[i] = max(0.01, prices[i - 1] + pull + rng.normal(0, vol))
    return prices


def _mock_history(tickers, start, end, field) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    dates = pd.bdate_range(end=end, periods=LOOKBACK_DAYS)
    # STIR futures sit near 100 and barely move, so they get a tiny vol fraction —
    # otherwise the demo's implied rate (100 − price) would swing wildly.
    data = {t: _mock_series(t, INSTRUMENTS.get(t, (t, 100.0, "", ""))[1], len(dates),
                            0.0005 if is_stir(t) else 0.012)
            for t in tickers}
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# FIXED-INCOME YIELDS  (the price-based TECHNICAL strategies run on yields, not
# futures prices — rates desks read yield charts, and the price↔yield inversion
# makes futures-price TA misleading). get_history_ta is aliased over get_history
# inside those strategy modules; vol/positioning keep true prices via get_history.
#   • STIRs -> implied rate = 100 − price (exact, from the price series).
#   • Bonds -> the benchmark generic yield each future tracks (universe.BOND_YIELD_SOURCE).
# ---------------------------------------------------------------------------
def get_history_ta(tickers, start=None, end=None, field: str = DEFAULT_FIELD) -> pd.DataFrame:
    """Like get_history, but fixed income comes back as YIELDS (%) instead of futures
    prices: STIRs as 100 − price, bond futures as their mapped benchmark yield. Non-FI
    tickers are unchanged (prices). This is what the technical strategies compute on."""
    tickers = list(tickers)
    out = get_history(tickers, start, end, field).copy()
    bonds = [t for t in tickers if is_bond(t) and yield_source(t)]
    ylds = get_yield_history(bonds, start, end) if bonds else None
    ycols = list(getattr(ylds, "columns", []))
    for t in tickers:
        if is_stir(t) and t in out.columns:
            out[t] = 100.0 - out[t]
        elif t in ycols:
            out[t] = ylds[t].reindex(out.index).ffill()
    return out


def get_yield_history(tickers, start=None, end=None) -> pd.DataFrame:
    """Benchmark yields (%) for BOND futures — one column per bond future ticker, valued
    by its mapped generic-yield series (universe.BOND_YIELD_SOURCE). Modes mirror
    get_history: bloomberg pulls the yield sources, snapshot reads yields.parquet, mock
    synthesises a plausible mean-reverting series. Non-bond tickers are dropped."""
    tickers = [t for t in tickers if yield_source(t)]
    if not tickers:
        return pd.DataFrame()
    if MODE == "bloomberg":
        return _bloomberg_yield(tickers, start, end)
    if MODE == "snapshot":
        snap = _read_snapshot("yields.parquet")
        if snap is not None:
            return snap.reindex(columns=tickers)
    return _mock_yield(tickers, start, end)


# Rough current-ish yield levels (%) per bond future, so mock/offline FI TA is realistic.
_MOCK_YIELD_BASE = {
    "TUA Comdty": 4.20, "FVA Comdty": 4.10, "TYA Comdty": 4.30, "UXYA Comdty": 4.35,
    "USA Comdty": 4.60, "WNA Comdty": 4.60, "DUA Comdty": 2.40, "OEA Comdty": 2.50,
    "RXA Comdty": 2.60, "UBA Comdty": 2.90, "OATA Comdty": 3.20, "G A Comdty": 4.40,
}


def _mock_yield(tickers, start, end) -> pd.DataFrame:
    """Synthetic yields (%): mean-reverting around a plausible level per tenor, ~3bp/day.
    Deterministic per ticker so the demo is stable."""
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    dates = pd.bdate_range(end=end, periods=LOOKBACK_DAYS)
    data = {}
    for t in tickers:
        b = _MOCK_YIELD_BASE.get(t, 4.0)
        rng = np.random.default_rng(_seed(t) ^ 0x51E1D)
        y = np.empty(len(dates))
        y[0] = b
        for i in range(1, len(dates)):
            y[i] = max(0.05, y[i - 1] + 0.02 * (b - y[i - 1]) + rng.normal(0, 0.03))
        data[t] = y
    return pd.DataFrame(data, index=dates)


def _bloomberg_yield(tickers, start, end) -> pd.DataFrame:
    """LIVE benchmark yields — one batched bdh (PX_LAST) on the distinct yield-source
    tickers, mapped back onto each bond future ticker."""
    from . import bbg as blp
    start, end = _span(start, end)
    srcs: dict = {}
    for t in tickers:
        srcs.setdefault(yield_source(t), []).append(t)
    wide = _bdh_to_wide(blp.bdh(tickers=list(srcs), flds="PX_LAST", start_date=start, end_date=end))
    cols = {}
    if wide is not None:
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        for src, futs in srcs.items():
            if src in wide.columns:
                for f in futs:
                    cols[f] = wide[src]
    if not cols:
        return pd.DataFrame(columns=list(tickers))
    return pd.DataFrame(cols).ffill().reindex(columns=list(tickers))


# ---------------------------------------------------------------------------
# VOLUME  (daily contract volume — the flag page's breakout-confirmation input)
# Mode-agnostic seam like get_history: bloomberg pulls PX_VOLUME, snapshot reads the
# cached frame, mock synthesises a move-size-driven series. Volume is a daily FLOW, so
# gaps are left NaN (never ffilled) and the flag code treats a missing series as simply
# "no volume confirmation available" rather than failing.
# ---------------------------------------------------------------------------
def get_volume_history(tickers, start=None, end=None, raw: bool = False) -> pd.DataFrame:
    """Daily contract volume, date-indexed, one column per ticker. Upgraded to the deep
    store's '1'-chain volume under the same guards as get_history (same aggregate field,
    just served deep instead of only as far back as the active contract's listing);
    `raw=True` skips the upgrade (snapshot persistence keeps the pristine feed)."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        out = _bloomberg_volume(tickers, start, end)
        return out if raw else _deep_upgrade(out, tickers, kind="volume", start=start, end=end)
    if MODE == "snapshot":
        snap = _read_snapshot("volume.parquet")
        if snap is not None:
            out = snap.reindex(columns=tickers)
            return out if raw else _deep_upgrade(out, tickers, kind="volume", start=start, end=end)
    return _mock_volume(tickers, start, end)


def _mock_volume(tickers, start, end) -> pd.DataFrame:
    """Synthetic daily volume so the flag page's volume confirmation works offline.
    Volume tracks the SIZE of each day's move — big (pole) days trade heavy, quiet
    (consolidation) days trade light — so a genuine flag shows the textbook volume
    dry-up through the flag and a pickup on the breakout. Deterministic per ticker."""
    px = get_history(list(tickers), start, end)
    out = {}
    for t in tickers:
        if t not in px.columns:
            continue
        ret = px[t].pct_change().abs()
        typ = float(ret.tail(LOOKBACK_DAYS).median())
        if not np.isfinite(typ) or typ <= 0:
            typ = 0.01
        rng = np.random.default_rng(_seed(t) ^ 0x0B0FA17E)
        base = float(_seed(t) % 80 + 20) * 1000.0                  # per-product baseline ADV
        drive = (0.5 + 2.5 * (ret / typ).fillna(0.0)).to_numpy()   # heavier on big-move days
        noise = rng.lognormal(0.0, 0.25, len(px))
        out[t] = np.round(np.maximum(1.0, base * drive * noise))
    return pd.DataFrame(out, index=px.index)


def _bloomberg_volume(tickers, start, end) -> pd.DataFrame:
    """LIVE daily volume — one batched bdh on VOLUME_FIELD. Left un-ffilled (a flow);
    markets with no volume series simply come back NaN (e.g. cash indices)."""
    from . import bbg as blp
    start, end = _span(start, end)
    wide = _bdh_to_wide(blp.bdh(tickers=tickers, flds=VOLUME_FIELD, start_date=start, end_date=end))
    if wide is None:
        return pd.DataFrame(columns=list(tickers))
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index().reindex(columns=list(tickers))


# ---------------------------------------------------------------------------
# BLOOMBERG feed  (wired in once your API access is live)
# ---------------------------------------------------------------------------
def _bloomberg_history(tickers, start, end, field) -> pd.DataFrame:
    # Requires the Terminal running + logged in, and `pip install xbbg blpapi`.
    from . import bbg as blp
    start, end = _span(start, end)

    # Most tickers use `field` (PX_SETTLE); cash indices have no settle, so they
    # override to PX_LAST (PRICE_FIELD_OVERRIDE). Pull one batched bdh per field.
    by_field = {}
    for t in tickers:
        by_field.setdefault(PRICE_FIELD_OVERRIDE.get(t, field), []).append(t)
    cols = {}
    for fld, grp in by_field.items():
        wide = _bdh_to_wide(blp.bdh(tickers=grp, flds=fld, start_date=start, end_date=end))
        if wide is not None:
            for t in grp:
                if t in wide.columns:
                    cols[t] = wide[t]
    if not cols:
        return pd.DataFrame(columns=list(tickers))
    wide = pd.DataFrame(cols)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().reindex(columns=list(tickers))

    # Use only fully-settled sessions: drop a trailing, still-settling day (e.g. today
    # before all markets have settled), so every spread leg comes from the same date.
    # Ignore all-NaN columns (invalid/unsupported tickers) so they don't drag the
    # completeness measure down and suppress the trim.
    valid_cols = [c for c in wide.columns if wide[c].notna().any()]
    if valid_cols:
        completeness = wide[valid_cols].notna().mean(axis=1)
        settled = completeness.index[completeness >= 0.95]
        if len(settled):
            wide = wide.loc[:settled.max()]
    return wide.ffill()


# ---------------------------------------------------------------------------
# VOLATILITY  (1-month realized + 1-month ATM implied)
# Realized is computed from settlement prices (works in mock AND live via the
# same get_history seam — no extra Bloomberg field/entitlement needed). Implied
# has no price to derive from, so live mode reads the option surface (with
# per-market overrides) and mock mode synthesizes it from realized. Both come
# back as ~1y histories so the strategy can z-score / percentile-rank the spread.
# ---------------------------------------------------------------------------
REALIZED_WINDOW = 21   # trading days ≈ 1 month, matched to the 1M implied tenor


def get_realized_vol_history(tickers, start=None, end=None,
                             window: int = REALIZED_WINDOW) -> pd.DataFrame:
    """Annualised ~1-month close-to-close realized vol (vol points), date-indexed,
    one column per ticker. Mode-agnostic: computed from get_history settlement
    prices, so it needs no Bloomberg vol entitlement and behaves the same in
    mock and live."""
    px = get_history(list(tickers), start, end)
    return np.log(px).diff().rolling(window).std() * np.sqrt(252) * 100.0


def get_implied_vol_history(tickers, start=None, end=None) -> pd.DataFrame:
    """1-month ATM implied vol (vol points), date-indexed, one column per ticker.

    LIVE: the listed-option surface (IMPLIED_VOL_FIELD) on each ticker, except the
    IMPLIED_VOL_OVERRIDE markets (FX -> OTC pair vol, etc.); pulls are batched by
    field and any market whose source returns nothing is left out (RV-only).
    MOCK: the realized series marked up by a per-ticker vol-risk premium.
    """
    tickers = list(tickers)
    if MODE == "bloomberg":
        return _bloomberg_implied_vol(tickers, start, end)
    if MODE == "snapshot":
        snap = _read_snapshot("implied_vol.parquet")
        if snap is not None:
            return snap.reindex(columns=tickers)
    return _mock_implied_vol(tickers, start, end)


def _mock_implied_vol(tickers, start, end) -> pd.DataFrame:
    """Synthetic implied = the mock realized series marked up by a per-ticker
    vol-risk premium plus daily noise, so the implied−realized spread varies
    across the book (mostly positive, occasionally inverted) in demo mode."""
    rv = get_realized_vol_history(tickers, start, end)
    out = {}
    for t in tickers:
        if t not in rv.columns:
            continue
        base = rv[t].bfill()
        rng = np.random.default_rng(_seed(t) ^ 0x5A5A5A5A)
        premium = 1.0 + rng.uniform(-0.05, 0.20)          # per-ticker rich/cheap bias
        noise = pd.Series(rng.normal(0.0, 0.8, len(base)), index=base.index)
        out[t] = (base * premium + noise).clip(lower=1.0)
    return pd.DataFrame(out)


# ----- Bloomberg implied-vol pull ------------------------------------------
def _span(start, end):
    if start is None:
        start = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(LOOKBACK_DAYS)
    if end is None:
        end = pd.Timestamp.today().normalize()
    return start, end


def _coerce_pd(raw):
    """xbbg may return a narwhals/pyarrow frame; normalise to pandas."""
    if hasattr(raw, "to_pandas"):
        return raw.to_pandas()
    if hasattr(raw, "to_native"):
        native = raw.to_native()
        return native.to_pandas() if hasattr(native, "to_pandas") else native
    return raw


def _bdh_to_wide(raw):
    """Normalise a (multi-ticker) bdh result to wide: index=date, columns=ticker."""
    pdf = _coerce_pd(raw)
    if pdf is None or len(pdf) == 0:
        return None
    lower = {str(c).lower(): c for c in pdf.columns}
    if {"ticker", "date", "value"}.issubset(lower):                  # long / tidy
        wide = pdf.pivot_table(index=lower["date"], columns=lower["ticker"],
                               values=lower["value"], aggfunc="last")
    else:                                                            # legacy wide
        if isinstance(pdf.columns, pd.MultiIndex):
            pdf = pdf.copy()
            pdf.columns = pdf.columns.get_level_values(0)
        wide = pdf
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Surface freshness. Bloomberg's smoothed *_DF implied-vol surface lags by a
# session or two on early pulls, and some contracts' surfaces go dead for weeks.
# Two guards keep a stale print from masquerading as a live quote:
#   • _ffill_internal — fill only INTERNAL gaps, never past a column's last real
#     observation, so a not-yet-computed recent surface stays a visible NaN
#     instead of silently repeating its last value onto the latest date.
#   • stale_iv_reasons — name the markets whose surface isn't live (frozen at one
#     value, or stopped publishing) so a strategy can leave them unscored rather
#     than flag a stale value as a rich/cheap opportunity.
# ---------------------------------------------------------------------------
def _ffill_internal(wide: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill internal gaps only — never past each column's last real
    observation, so a surface that returned nothing for the most recent session(s)
    keeps a trailing gap rather than repeating its stale last value."""
    if wide is None or wide.empty:
        return wide
    filled = wide.ffill()
    for c in wide.columns:
        lv = wide[c].last_valid_index()
        if lv is None:
            filled[c] = wide[c]                      # all-NaN column: leave untouched
        else:
            filled.loc[filled.index > lv, c] = np.nan
    return filled


IV_FROZEN_SESSIONS = 5    # last N real obs bit-identical -> a frozen/pinned surface
IV_STALE_DAYS = 8         # last real obs older than this (calendar days) -> dead surface


def stale_iv_reasons(iv: pd.DataFrame, frozen_sessions: int = IV_FROZEN_SESSIONS,
                     max_age_days: int = IV_STALE_DAYS) -> dict:
    """{ticker: reason} for markets whose implied-vol surface looks STALE — either
    FROZEN (the last `frozen_sessions` real observations are bit-identical, e.g. a
    dead surface pinned at one value) or OLD (the last real observation is more than
    `max_age_days` calendar days behind the frame's latest date, e.g. a surface that
    stopped publishing). Accepts any date-indexed vol frame (1M ATM, a skew wing, a
    term tenor). Such a surface is not a live quote, so its market should be left
    unscored rather than flagged on a value left over from an earlier session."""
    out: dict = {}
    if iv is None or getattr(iv, "empty", True):
        return out
    try:
        latest = pd.to_datetime(iv.index).max()
    except Exception:
        latest = None
    for c in iv.columns:
        s = iv[c].dropna()
        if s.empty:
            continue
        if latest is not None:
            try:
                age = int((latest - s.index[-1]).days)
            except Exception:
                age = 0
            if age > max_age_days:
                out[c] = f"{age}d stale"
                continue
        if len(s) >= frozen_sessions and bool((s.iloc[-frozen_sessions:] == s.iloc[-1]).all()):
            out[c] = f"frozen {frozen_sessions}+ sessions"
    return out


_STALE_WARNED: set = set()


def warn_stale(tag: str, items: list[str]) -> None:
    """Print once per process per unique stale-set which markets a strategy dropped
    for a stale vol surface — so the drop is visible in the run log, never silent."""
    if not items:
        return
    key = (tag, tuple(sorted(items)))
    if key in _STALE_WARNED:
        return
    _STALE_WARNED.add(key)
    print(f"[{tag}] skipped {len(items)} market(s) on a stale vol surface: " + ", ".join(items))


def price_decimals(prices, asset: str = "") -> int:
    """Native display precision of a futures settlement series — how many decimals the
    contract actually trades in — inferred from the finest granularity seen in recent
    prints (capped at 5). US Treasury-style contracts quote in 32nds/64ths, which
    Bloomberg decimalises into 6-8 ugly places; those (asset 'Bonds' inferring > 3)
    collapse to a readable 3. Used to render the 1σ price move next to each vol."""
    s = prices.dropna() if hasattr(prices, "dropna") else prices
    if s is None or len(s) == 0:
        return 1

    def _nd(v):
        txt = f"{round(float(v), 8):.8f}".rstrip("0")
        frac = txt.split(".")[1] if "." in txt else ""
        return len(frac)

    d = max((_nd(v) for v in s.tail(60)), default=0)
    if asset == "Bonds" and d > 3:        # US Treasuries in 32nds/64ths → readable
        return 3
    return min(d, 5)


def _bloomberg_implied_vol(tickers, start, end) -> pd.DataFrame:
    # Group markets by their implied-vol field so each field is ONE batched bdh
    # call (≈2-3 calls total) instead of ~80 per-ticker round trips.
    from . import bbg as blp
    start, end = _span(start, end)
    plans = {t: IMPLIED_VOL_OVERRIDE.get(t, (t, IMPLIED_VOL_FIELD)) for t in tickers}
    by_field = {}
    for fut, (src, fld) in plans.items():
        by_field.setdefault(fld, []).append((fut, src))

    cols = {}
    for fld, pairs in by_field.items():
        srcs = sorted({src for _, src in pairs})
        # A composite field "A+B" pulls each leg and takes the element-wise mean — e.g.
        # Euribor's "HIST_CALL_IMP_VOL+HIST_PUT_IMP_VOL" (the near-ATM call/put mid; no
        # smoothed surface exists for it). A date where only one leg prints keeps that leg.
        parts = [p.strip() for p in fld.split("+") if p.strip()] if "+" in fld else [fld]
        wides = []
        for part in parts:
            try:
                w = _bdh_to_wide(blp.bdh(tickers=srcs, flds=part, start_date=start, end_date=end))
            except Exception:
                w = None
            if w is not None:
                wides.append(w)
        if not wides:
            continue
        wide = wides[0] if len(wides) == 1 else pd.concat(wides).groupby(level=0).mean()
        for fut, src in pairs:
            if src in wide.columns and wide[src].notna().any():
                cols[fut] = wide[src]
    if not cols:
        return pd.DataFrame(columns=list(tickers))
    return _ffill_internal(pd.DataFrame(cols).sort_index().reindex(columns=list(tickers)))


# ---------------------------------------------------------------------------
# SKEW  (fixed 90/110% moneyness put − call wings, normalized by ATM)
#
# The skew metric is (90% put − 110% call) / 100% ATM. LISTED products read the
# three constant-maturity moneyness pillars straight off the surface: 90%
# (downside / put wing), 110% (upside / call wing) and 100% (ATM). FX has no
# moneyness surface — its vol is delta-quoted OTC — so it keeps its native OTC
# 25Δ risk reversal as the put/call wings (a different wing width, but FX skew is
# inherently a delta measure; user choice 2026-06-04). Each product is z-scored
# against its OWN history, so the differing wing definitions stay comparable.
#
# Low-price-vol products (bonds, STIRs) are EXCLUDED upstream
# (volatility.SKEW_EXCLUDE_EXTRA / EXCLUDE_ASSETS): a fixed 90/110% strike sits
# tens of σ OTM for them, in extrapolated surface (the US 2Y "6.2" blow-up).
# ---------------------------------------------------------------------------
def _mny_field(moneyness: float) -> str:
    """Bloomberg constant-maturity vol-by-moneyness field, e.g. 90.0 ->
    '30DAY_IMPVOL_90.0%MNY_DF'. The 100.0 pillar IS datafeed.IMPLIED_VOL_FIELD."""
    return f"30DAY_IMPVOL_{moneyness:.1f}%MNY_DF"


SKEW_PUT_FIELD = _mny_field(90.0)      # 90% moneyness  — downside / put wing
SKEW_CALL_FIELD = _mny_field(110.0)    # 110% moneyness — upside / call wing
SKEW_ATM_FIELD = _mny_field(100.0)     # 100% moneyness — ATM (== IMPLIED_VOL_FIELD)


def _frame3(put: dict, call: dict, atm: dict, tickers) -> dict:
    def _df(d):
        return (_ffill_internal(pd.DataFrame(d).sort_index().reindex(columns=list(tickers)))
                if d else pd.DataFrame(columns=list(tickers)))
    return {"put": _df(put), "call": _df(call), "atm": _df(atm)}


def get_skew_components(tickers, start=None, end=None, atm=None) -> dict:
    """Per-ticker {'put','call','atm'} 1M implied-vol histories (vol points,
    date-indexed); the skew metric is (90% put − 110% call)/atm. Listed markets
    read the 90/110/100% moneyness pillars off the surface; FX uses the native OTC
    25Δ risk reversal (it has no moneyness surface). Mode-agnostic seam.

    atm: an already-pulled get_implied_vol_history frame. The 100% pillar IS the
    implied-vol field (and FX's V1M), so passing the morning's frame saves the
    third identical bdh per surface pull (2026-08-18 dedupe); None pulls it as
    before, so ad-hoc callers stay correct."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        return _bloomberg_skew(tickers, start, end, atm_frame=atm)
    if MODE == "snapshot":
        put = _read_snapshot("skew_put.parquet")
        call = _read_snapshot("skew_call.parquet")
        atm = _read_snapshot("skew_atm.parquet")
        if put is not None and call is not None and atm is not None:
            return {"put": put.reindex(columns=tickers),
                    "call": call.reindex(columns=tickers),
                    "atm": atm.reindex(columns=tickers)}
    return _mock_skew(tickers, start, end)


def _mock_skew(tickers, start, end) -> dict:
    """Synthetic 90/110/100 wings: mark the mock ATM up/down by a per-ticker skew
    bias so the (put − call)/atm signal varies across the book in demo mode."""
    atm = get_implied_vol_history(tickers, start, end)
    put, call = {}, {}
    for t in tickers:
        if t not in atm.columns:
            continue
        a = atm[t]
        rng = np.random.default_rng(_seed(t) ^ 0x33CC33CC)
        base = float(rng.uniform(-0.15, 0.55))                 # per-ticker skew bias
        noise = pd.Series(rng.normal(0.0, 0.05, len(a)), index=a.index)
        skew = base + noise                                    # = (put − call)/atm
        put[t] = a * (1 + skew / 2.0)
        call[t] = a * (1 - skew / 2.0)
    return _frame3(put, call, {c: atm[c] for c in atm.columns}, tickers)


def _bloomberg_skew(tickers, start, end, atm_frame=None) -> dict:
    from . import bbg as blp
    start, end = _span(start, end)
    fx = _fx_override_tickers(tickers)                  # OTC 25Δ risk reversal
    listed = [t for t in tickers if t not in set(fx)]   # 90/110% moneyness wings
    put, call, atm = {}, {}, {}

    def _shared(t):
        """The ticker's column from the shared implied-vol frame — only where that
        frame's source is the same 100%-mny surface pillar / FX V1M this function
        would pull itself (Euribor's composite field ≠ the surface: stays out,
        exactly as its empty surface pull kept it out before)."""
        if atm_frame is None or t not in getattr(atm_frame, "columns", []):
            return None
        if t not in fx and IMPLIED_VOL_OVERRIDE.get(t, (t, IMPLIED_VOL_FIELD))[1] != IMPLIED_VOL_FIELD:
            return None
        s = atm_frame[t]
        return s if s.notna().any() else None

    if listed:
        # 90/110% moneyness wings — one batched bdh per field; the 100% ATM pillar
        # comes from the shared implied-vol frame when given (it is the SAME field),
        # else its own bdh as before.
        w_put = _bdh_to_wide(blp.bdh(tickers=listed, flds=SKEW_PUT_FIELD, start_date=start, end_date=end))
        w_call = _bdh_to_wide(blp.bdh(tickers=listed, flds=SKEW_CALL_FIELD, start_date=start, end_date=end))
        w_atm = (None if atm_frame is not None else
                 _bdh_to_wide(blp.bdh(tickers=listed, flds=SKEW_ATM_FIELD, start_date=start, end_date=end)))
        for t in listed:
            a = _shared(t) if atm_frame is not None else (
                w_atm[t] if w_atm is not None and t in w_atm.columns and w_atm[t].notna().any() else None)
            if a is not None and all(w is not None and t in w.columns and w[t].notna().any()
                                     for w in (w_put, w_call)):
                put[t], call[t], atm[t] = w_put[t], w_call[t], a

    if fx:
        atm_src = {t: IMPLIED_VOL_OVERRIDE[t][0] for t in fx}               # xxxV1M Curncy
        rr_src = {t: atm_src[t].replace("V1M", "25R1M") for t in fx}        # 25Δ risk reversal
        bf_src = {t: atm_src[t].replace("V1M", "25B1M") for t in fx}        # 25Δ butterfly
        w_atm = (None if atm_frame is not None else
                 _bdh_to_wide(blp.bdh(tickers=sorted(set(atm_src.values())), flds="PX_LAST", start_date=start, end_date=end)))
        w_rr = _bdh_to_wide(blp.bdh(tickers=sorted(set(rr_src.values())), flds="PX_LAST", start_date=start, end_date=end))
        w_bf = _bdh_to_wide(blp.bdh(tickers=sorted(set(bf_src.values())), flds="PX_LAST", start_date=start, end_date=end))
        for t in fx:
            a, r, b = atm_src[t], rr_src[t], bf_src[t]
            if atm_frame is not None:
                # KNOWN, ACCEPTED divergence from the old triple-pull (review
                # 2026-08-18): the shared frame is _ffill_internal'ed, so on a
                # vendor-gap date where V1M is missing but the RR still prints, the
                # wing = carried ATM + TODAY's RR/BF (old path carried the WHOLE
                # previous wing). More truthful on those rare dates; identical on
                # every date where V1M prints.
                A = _shared(t)
            else:
                A = (w_atm[a] if w_atm is not None and a in w_atm.columns
                     and w_atm[a].notna().any() else None)
            if A is None:
                continue
            if w_rr is None or r not in w_rr.columns or not w_rr[r].notna().any():
                continue
            RR = w_rr[r]
            BF = w_bf[b] if (w_bf is not None and b in w_bf.columns) else 0.0
            # 25Δ wings (RR = call − put): call = ATM + BF + RR/2; put = ATM + BF − RR/2.
            # (put − call)/ATM = −RR/ATM is exact regardless of the butterfly.
            call[t] = A + BF + RR / 2.0
            put[t] = A + BF - RR / 2.0
            atm[t] = A

    return _frame3(put, call, atm, tickers)


# ---------------------------------------------------------------------------
# TERM STRUCTURE  (ATM implied vol at 1M / 3M / 6M / 12M)
# Listed markets read the surface ATM (100% moneyness) at each maturity; FX reads
# the OTC ATM tenor tickers (…V1M/V3M/V6M/V1Y). The slope (e.g. 3M − 1M) says
# whether the curve is in contango (normal) or backwardation (front bid).
# Field tokens VERIFIED LIVE 2026-06-04: the surface publishes the FRONT in days
# (30/60DAY) but ≥3M in MONTHS (3MTH/6MTH/12MTH/18/24) — the 90/180/360DAY day
# tokens return nothing. FX V1M/V2M/V3M/V6M/V1Y all confirmed. These serve via bdh.
# ---------------------------------------------------------------------------
# (label, listed ATM surface field, FX tenor token, matched realized window [trading days])
TENORS = [
    ("1M",  "30DAY_IMPVOL_100.0%MNY_DF",  "V1M", 21),
    ("3M",  "3MTH_IMPVOL_100.0%MNY_DF",   "V3M", 63),
    ("6M",  "6MTH_IMPVOL_100.0%MNY_DF",   "V6M", 126),
    ("12M", "12MTH_IMPVOL_100.0%MNY_DF",  "V1Y", 252),
]
TENOR_LABELS = [lab for lab, *_ in TENORS]


def get_term_structure(tickers, start=None, end=None, atm=None) -> dict:
    """Per-tenor ATM implied-vol histories {'1M','3M','6M','12M'} (vol points,
    date-indexed, one column per ticker). Mode-agnostic seam like the others.

    atm: an already-pulled get_implied_vol_history frame — the 1M tenor IS that
    field (and FX's V1M), so passing the morning's frame skips the identical bdh
    (2026-08-18 dedupe); None pulls it as before."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        return _bloomberg_term_structure(tickers, start, end, atm_frame=atm)
    if MODE == "snapshot":
        frames = {lab: _read_snapshot(f"term_{lab.lower()}.parquet") for lab in TENOR_LABELS}
        if all(f is not None for f in frames.values()):
            return {lab: frames[lab].reindex(columns=tickers) for lab in TENOR_LABELS}
    return _mock_term_structure(tickers, start, end)


def _mock_term_structure(tickers, start, end) -> dict:
    """Synthetic curve: the mock 1M ATM scaled by a per-ticker term premium that
    grows with tenor (mostly upward-sloping / contango, occasionally inverted),
    plus per-tenor noise — enough to exercise the slope z-score in demo mode."""
    base = get_implied_vol_history(tickers, start, end)
    weight = {"1M": 0.0, "3M": 0.35, "6M": 0.65, "12M": 1.0}
    slope = {t: float(np.random.default_rng(_seed(t) ^ 0x7E57AA).uniform(-0.10, 0.30))
             for t in tickers}
    out = {}
    for i, (lab, *_rest) in enumerate(TENORS):
        cols = {}
        for t in tickers:
            if t not in base.columns:
                continue
            b = base[t]
            noise = pd.Series(
                np.random.default_rng(_seed(t) ^ (0xABC0 + i)).normal(0.0, 0.4, len(b)),
                index=b.index)
            cols[t] = (b * (1.0 + slope[t] * weight[lab]) + noise).clip(lower=1.0)
        out[lab] = pd.DataFrame(cols)
    return out


def _bloomberg_term_structure(tickers, start, end, atm_frame=None) -> dict:
    from . import bbg as blp
    start, end = _span(start, end)
    fx = _fx_override_tickers(tickers)
    listed = [t for t in tickers if t not in set(fx)]
    out = {}
    for lab, fld, tok, _win in TENORS:
        cols = {}
        # The 1M tenor is the implied-vol field itself (listed: the 30DAY 100%-mny
        # pillar; FX: V1M) — reuse the shared frame instead of re-pulling it. Only
        # products whose implied-vol source IS that pillar qualify (Euribor's
        # composite stays out, matching its empty surface pull before).
        if lab == "1M" and atm_frame is not None:
            for t in tickers:
                if t not in getattr(atm_frame, "columns", []):
                    continue
                if t not in set(fx) and \
                        IMPLIED_VOL_OVERRIDE.get(t, (t, IMPLIED_VOL_FIELD))[1] != IMPLIED_VOL_FIELD:
                    continue
                if atm_frame[t].notna().any():
                    cols[t] = atm_frame[t]
            out[lab] = (_ffill_internal(pd.DataFrame(cols).sort_index().reindex(columns=list(tickers)))
                        if cols else pd.DataFrame(columns=list(tickers)))
            continue
        if listed:                                  # surface ATM at this maturity
            w = _bdh_to_wide(blp.bdh(tickers=listed, flds=fld, start_date=start, end_date=end))
            if w is not None:
                for t in listed:
                    if t in w.columns and w[t].notna().any():
                        cols[t] = w[t]
        if fx:                                       # OTC ATM tenor ticker (…V3M etc.)
            src = {t: IMPLIED_VOL_OVERRIDE[t][0].replace("V1M", tok) for t in fx}
            w = _bdh_to_wide(blp.bdh(tickers=sorted(set(src.values())), flds="PX_LAST",
                                     start_date=start, end_date=end))
            if w is not None:
                for t in fx:
                    s = src[t]
                    if s in w.columns and w[s].notna().any():
                        cols[t] = w[s]
        out[lab] = (_ffill_internal(pd.DataFrame(cols).sort_index().reindex(columns=list(tickers)))
                    if cols else pd.DataFrame(columns=list(tickers)))
    return out


# ---------------------------------------------------------------------------
# PUT / CALL  (options open interest + volume, split puts vs calls)
#
# The put/call ratio = puts ÷ calls — a sentiment / positioning gauge carried on two
# bases: OPEN INTEREST (standing positioning, the headline) and VOLUME (today's traded
# flow, used for the day-over-day spike + the flow-vs-OI divergence). A raw level isn't
# comparable across products (index options sit structurally put-heavy from hedging;
# others sit nearer 1), so the strategy normalizes each product against its OWN history.
# LIVE pulls the aggregated option-chain totals off Bloomberg; MOCK synthesizes a
# realistic put-heavy/neutral book so the whole page works offline.
# ---------------------------------------------------------------------------
# Aggregated option-chain fields, one per returned frame. PROVISIONAL — confirm each
# per asset class on the Terminal (FLDS<GO> / OMON<GO>); coverage on futures options
# varies, and FX listed options are thin (many pairs will be absent, by design).
PUTCALL_FIELDS = {
    "put_oi":   "OPEN_INT_TOTAL_PUT",
    "call_oi":  "OPEN_INT_TOTAL_CALL",
    "put_vol":  "VOLUME_TOTAL_PUT",
    "call_vol": "VOLUME_TOTAL_CALL",
}
_PUTCALL_KEYS = list(PUTCALL_FIELDS)   # put_oi, call_oi, put_vol, call_vol

# A few products carry their listed put/call on a DIFFERENT security's option chain
# than their own ticker — pull those from the override source. European/Japanese equity
# index options trade on the CASH index (OMON), so the futures generic returns no
# aggregated put/call; map those futures tickers to the cash-index source. (Euro Stoxx
# 50 / Banks / DAX / FTSE are already in the universe AS their cash tickers
# SX5E/SX7E/DAX Index/UKX Index, which carry the put/call directly, so they need no
# mapping — only CAC and Nikkei are present solely as futures.) Sources per OMON,
# user-supplied 2026-06-19.
# EMPTY as of 2026-06-20: was {CAA Index→CAC Index, NKA Index→NKY Index}, but the CAC/NKY cash
# indices are now in the universe directly (self-sourcing their put/call like SX5E/DAX/UKX), so
# routing the FUTURES generics CAA/NKA to the same cash chain duplicated the CAC 40 / Nikkei rows.
# The futures generics have no listed put/call of their own (gen1 CA1/NK1 return nothing), so with
# no override they fall out cleanly and only the cash tickers remain.
PUTCALL_SOURCE = {}
# Default for everything else: pull put/call from the future's 1st-GENERIC (xx1), which serves
# far deeper option history than the active 'A' generic (verified live 2026-06-20 via
# diag_putcall.py — the same finding that fixed the COT price DB). PUTCALL_KEEP_ACTIVE = the
# markets where the active generic is DEEPER or the 1st-generic returns nothing, so they keep
# their own ticker.
PUTCALL_KEEP_ACTIVE = frozenset({
    "SFRA Comdty", "SFIA Comdty",                            # 3M SOFR, 3M SONIA — 1st-generic empty
    "FJSA Comdty",                                           # TTF Nat Gas — active deeper
    "C A Comdty", "S A Comdty", "SBA Comdty", "CCA Comdty",  # Corn, Soybeans, Sugar, Cocoa — active ≥ gen1
    "SFA Curncy", "JYA Curncy",                              # CHF, JPY — active deeper
})


def _putcall_first_generic(t: str) -> str:
    """1st-generic form: 'TYA Comdty' -> 'TY1 Comdty', 'C A Comdty' -> 'C 1 Comdty'. Tickers
    not ending in 'A' (cash indices SX5E/DAX/UKX Index) are returned unchanged."""
    gen, key = t.rsplit(" ", 1)
    return f"{gen[:-1]}1 {key}" if gen.endswith("A") else t


def _putcall_src(t: str) -> str:
    """Security to pull put/call from for product `t` (override → keep-active → 1st-generic)."""
    if t in PUTCALL_SOURCE:
        return PUTCALL_SOURCE[t]
    if t in PUTCALL_KEEP_ACTIVE:
        return t
    return _putcall_first_generic(t)


def get_putcall(tickers, start=None, end=None) -> dict:
    """Per-ticker options open-interest & volume split puts vs calls, as four
    date-indexed wide frames {'put_oi','call_oi','put_vol','call_vol'} (one column per
    ticker). The put/call ratio is puts/calls on each basis. Mode-agnostic seam like
    the others (bloomberg / snapshot / mock)."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        return _bloomberg_putcall(tickers, start, end)
    if MODE == "snapshot":
        frames = {k: _read_snapshot(f"putcall_{k}.parquet") for k in _PUTCALL_KEYS}
        if all(f is not None for f in frames.values()):
            return {k: frames[k].reindex(columns=tickers) for k in _PUTCALL_KEYS}
    return _mock_putcall(tickers, start, end)


def _mock_putcall(tickers, start, end) -> dict:
    """Synthetic options OI & volume: each product gets a structural baseline put/call
    (indices put-heavy from hedging, others nearer 1), a slowly-wandering OI ratio and a
    noisier volume ratio that occasionally spikes — enough to exercise the percentile,
    the day-over-day spike and the flow-vs-OI divergence in demo mode."""
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    dates = pd.bdate_range(end=end, periods=LOOKBACK_DAYS)
    n = len(dates)
    cols = {k: {} for k in _PUTCALL_KEYS}
    for t in tickers:
        a = INSTRUMENTS.get(t, (t, 100.0, "", ""))[2]
        base_pc = 1.5 if a == "Indices" else 1.05 if a in ("Bonds", "STIRs") else 0.85
        rng = np.random.default_rng(_seed(t) ^ 0x9C5A1234)
        oi_pc = np.empty(n)
        oi_pc[0] = base_pc
        for i in range(1, n):                       # slow AR(1) wander around the baseline
            oi_pc[i] = 0.985 * oi_pc[i - 1] + 0.015 * base_pc + rng.normal(0.0, 0.02)
        oi_pc = np.maximum(0.2, oi_pc)
        spikes = (rng.random(n) < 0.04) * rng.normal(0.0, 0.6, n)        # occasional flow bursts
        vol_pc = np.maximum(0.1, oi_pc * rng.uniform(0.9, 1.1, n) + rng.normal(0.0, 0.12, n) + spikes)
        scale = float(_seed(t) % 50 + 5) * 1000.0                       # per-contract OI size
        call_oi = np.maximum(100.0, scale * (1 + 0.15 * np.sin(np.linspace(0, 6, n)))
                             + rng.normal(0.0, scale * 0.03, n))
        call_vol = np.maximum(10.0, call_oi * float(rng.uniform(0.15, 0.45))
                              + rng.normal(0.0, scale * 0.05, n))
        cols["call_oi"][t] = call_oi
        cols["put_oi"][t] = np.maximum(100.0, call_oi * oi_pc)
        cols["call_vol"][t] = call_vol
        cols["put_vol"][t] = np.maximum(10.0, call_vol * vol_pc)
    return {k: pd.DataFrame(cols[k], index=dates) for k in _PUTCALL_KEYS}


def _bloomberg_putcall(tickers, start, end) -> dict:
    """Aggregated option-chain OI & volume split puts vs calls — one batched bdh per
    field. Each product pulls from its PUTCALL_SOURCE override (e.g. CAC/Nikkei from the
    cash-index OMON) or its own ticker; the result is stored under the PRODUCT ticker.
    Any market whose source returns nothing is simply left out (it then shows no
    put/call — expected for thin FX listed options)."""
    from . import bbg as blp
    start, end = _span(start, end)
    src_of = {t: _putcall_src(t) for t in tickers}              # product ticker -> deepest put/call source
    sources = sorted(set(src_of.values()))
    cols = {k: {} for k in _PUTCALL_KEYS}
    for key, fld in PUTCALL_FIELDS.items():
        try:
            wide = _bdh_to_wide(blp.bdh(tickers=sources, flds=fld, start_date=start, end_date=end))
        except Exception:
            wide = None
        if wide is None:
            continue
        for t in tickers:
            s = src_of[t]
            if s in wide.columns and wide[s].notna().any():
                cols[key][t] = wide[s]

    def _df(d, ffill):
        if not d:
            return pd.DataFrame(columns=list(tickers))
        out = pd.DataFrame(d).sort_index().reindex(columns=list(tickers))
        return out.ffill() if ffill else out      # ffill OI (a level); leave VOLUME gaps NaN (a daily flow)
    return {k: _df(cols[k], k in ("put_oi", "call_oi")) for k in _PUTCALL_KEYS}


# ---------------------------------------------------------------------------
# OPTION OPEN-INTEREST CHAIN  (one product's strike × expiry-month grid)
#
# Unlike get_putcall (which carries each product's AGGREGATED put/call OI totals over
# time), this returns the full option chain for ONE underlying as a tidy grid: open
# interest at every (expiry, strike), split puts vs calls. The Open Interest report
# sums put+call per cell into a strike×expiry heatmap. Mode-agnostic seam like the
# others (bloomberg / snapshot / mock).
# ---------------------------------------------------------------------------
OI_CHAIN_EXPIRIES = 6     # forward expiry months to build
OI_CHAIN_STRIKES = 21     # strikes in the ladder (ATM ± 10)
_OI_COLS = ["expiry", "expiry_label", "strike", "call_oi", "put_oi"]

# Per-option fields for the LIVE chain pull. VERIFIED live 2026-06-22 (probe_oi_chain.py):
# OPT_CHAIN lists the options in a 'Security Description' column; bdp on each returns
# OPT_STRIKE_PX (float), OPT_EXPIRE_DT (the OPTION's expiry — NOT the contract month),
# OPT_PUT_CALL ('P'/'C') and OPEN_INT (int).
OI_CHAIN_FIELDS = ["OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPEN_INT"]

# Bounds on the LIVE pull — a full board runs 250–3000 deep, so we filter the OPT_CHAIN
# members (cheap: parsed from the description) BEFORE the per-option OPEN_INT bdp, to the
# nearest expiries and near-the-money strikes. Keeps the morning snapshot fast and well
# under reference-data limits; the report windows tighter still on top of this.
OI_LIVE_MAX_EXPIRIES = 24      # nearest option-expiry MONTHS to keep — deep enough for the whole
                               # rates strip (deferred quarterlies out several years carry real OI)
OI_LIVE_MONEYNESS = 0.20       # keep strikes within ±20% of spot (well wider than any rate strike)
OI_LIVE_STRIKES_PER_EXP = 80   # and at most this many (nearest to spot) per expiry
OI_LIVE_MAX_CONTRACTS = 28     # walk the option chain across the nearest N DATED contracts: the
                               # GENERIC future only exposes the FRONT expiries (and nothing for
                               # SONIA/ESTR) — the full quarterly strip lives on the dated contracts,
                               # so this must reach the deferred quarters (e.g. SFRH8 = Mar-28)

# The OI snapshot captures option chains for these 11 FIXED-INCOME products ONLY (the Fixed
# Income book) — pulling more, or more often, would burn reference-data limits, and this is a
# rates tool. MUST stay in sync with FI_OI_PAGES in app.py. Products outside this list serve
# ONLY what the weekly capture holds — on-demand live chain pulls were removed 2026-08-18
# (they were the app's one unbounded Bloomberg spend); get_oi_chain(live=True) is the weekly
# job's sanctioned pull.
OI_SNAPSHOT_TICKERS = (
    "ERA Comdty", "SFIA Comdty", "SFRA Comdty",                   # 3M Euribor · SONIA · SOFR
    "TUA Comdty", "FVA Comdty", "TYA Comdty", "USA Comdty",       # US 2Y · 5Y · 10Y · Long Bond
    "DUA Comdty", "OEA Comdty", "RXA Comdty", "UBA Comdty",       # Schatz · Bobl · Bund · Buxl
)

_MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def get_oi_chain(ticker: str, n_expiries: int = OI_CHAIN_EXPIRIES,
                 n_strikes: int | None = None, step: float | None = None,
                 width: float | None = None, live: bool = False) -> pd.DataFrame:
    """One product's listed-option OPEN INTEREST as a tidy strike×expiry grid:
    columns ['expiry','expiry_label','strike','call_oi','put_oi'] (one row per strike
    per expiry). The heatmap sums call+put per cell. Mode-agnostic seam: bloomberg
    pulls the live chain, snapshot reads the cached grid, mock synthesises a realistic
    ATM-peaked surface so the report works offline.

    `n_strikes` is a DISPLAY window: None (default) returns the WHOLE chain — every listed
    strike (live) / the full realistic ladder (mock); an int keeps only the N strikes
    nearest spot (a tighter view, used by the compact report panels). The window is applied
    in every mode, so it also trims a full live chain.

    `step` / `width` are MOCK-only hints (ignored live): the strike increment and the
    open-interest concentration half-width, both in PRICE units. Omitted → asset-class
    defaults (fixed income needs a far finer grid than equities — see _asset_chain_defaults).

    `live=True` is the ONLY path that pulls a chain from Bloomberg (the weekly --oi
    capture). Since 2026-08-18 bloomberg mode otherwise serves the cached weekly
    grid like snapshot mode: a chain pull is up to ~29 bds + a ~1,900-security bdp
    PER PRODUCT, and the on-demand page path was the one unbounded Bloomberg spend
    in the app. In bloomberg mode a product missing from the capture returns EMPTY
    (never mock — synthetic OI must not render as real data)."""
    if MODE == "bloomberg" and live:
        chain = _bloomberg_oi_chain(ticker)
    elif MODE == "bloomberg":
        chain = _read_oi_snapshot(ticker)              # cached weekly capture only
    elif MODE == "snapshot":
        chain = _read_oi_snapshot(ticker)
        if chain is None:
            chain = _mock_oi_chain(ticker, n_expiries, n_strikes, step, width)
    else:
        chain = _mock_oi_chain(ticker, n_expiries, n_strikes, step, width)
    if chain is None or chain.empty:
        return pd.DataFrame(columns=_OI_COLS)
    if n_expiries is not None:
        chain = _window_expiries(chain, int(n_expiries))
    if n_strikes is not None:
        chain = _window_strikes(chain, ticker, int(n_strikes))
    return chain


def _window_expiries(chain: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the `n` nearest-dated expiries. A no-op when the chain already has ≤ n (the mock
    builds exactly n); for a full live / snapshot chain it trims to the months the report
    shows, so the expiry control works in every mode — not just mock."""
    exps = sorted(pd.to_datetime(chain["expiry"]).dropna().unique())
    if len(exps) <= n:
        return chain
    keep = set(exps[:n])
    return chain[pd.to_datetime(chain["expiry"]).isin(keep)].reset_index(drop=True)


def _window_strikes(chain: pd.DataFrame, ticker: str, n: int) -> pd.DataFrame:
    """Keep the `n` strikes nearest spot (centred on the ATM), dropping the far wings.
    A no-op when the chain already has ≤ n strikes."""
    ks = [float(k) for k in pd.unique(chain["strike"]) if pd.notna(k)]
    if len(ks) <= n:
        return chain
    s = get_history([ticker])
    spot = float(s[ticker].dropna().iloc[-1]) if (ticker in s and s[ticker].notna().any()) \
        else float(INSTRUMENTS.get(ticker, (ticker, 100.0, "", ""))[1])
    keep = (sorted(ks, key=lambda k: abs(k - spot)) if spot > 0 else sorted(ks))[:n]
    return chain[chain["strike"].isin(keep)].reset_index(drop=True)


def _read_oi_snapshot(ticker: str):
    """Read this product's chain out of a combined snapshot grid (tidy long with a 'ticker'
    column). Returns None — so the caller falls back to mock — when there's no usable OI
    capture at all: the file is missing, OR it exists but is EMPTY (the whole chain pull
    failed, e.g. the live fields returned nothing). When the capture DID populate but this
    particular product isn't in it, returns an EMPTY frame: that product genuinely has no
    listed-option chain, so the report shows nothing rather than inventing a synthetic one."""
    p = SNAPSHOT_DIR / "oi_chain.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty or "ticker" not in df.columns or df["ticker"].nunique() == 0:
        return None                                   # capture failed wholesale → mock fallback
    return df[df["ticker"] == ticker].reindex(columns=_OI_COLS).copy()


def _nice(x: float) -> float:
    """Round x UP to a 1 / 2 / 2.5 / 5 × 10^k value (a 'nice' strike increment)."""
    if x <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def _nice_step(spot: float) -> float:
    """A round strike increment near ~2.5% of spot (1 / 2 / 2.5 / 5 × 10^k)."""
    return _nice(spot * 0.025) if spot > 0 else 1.0


def _asset_chain_defaults(asset: str, spot: float) -> tuple:
    """(strike increment, OI half-width) in PRICE units, by asset class. Fixed income
    is the special case: price sits near par (~96–135) but options strike on a far finer
    grid than the ~2.5%-of-spot ladder that suits equities / commodities, and the open
    interest clusters within a point or two of ATM. Tenor isn't known here, so these are
    moderate rate grids — the Fixed Income report passes exact per-tenor step/width."""
    if asset == "STIRs":
        return 0.125, 0.6
    if asset == "Bonds":
        return (0.5 if spot < 125 else 1.0), max(2.0, spot * 0.02)
    return _nice_step(spot), spot * 0.07


def _strike_ladder(spot: float, n: int, step: float | None = None) -> list:
    step = step or _nice_step(spot)
    atm = round(spot / step) * step
    half = n // 2
    ks = [atm + (i - half) * step for i in range(n)]
    return [round(k, 10) for k in ks if k > 0]


def _forward_expiries(n: int) -> list:
    """The next `n` monthly option expiries (third Friday) from today forward."""
    today = pd.Timestamp.today().normalize()
    months = pd.date_range(today.replace(day=1), periods=n + 3, freq="MS")
    out = []
    for m in months:
        third_fri = m + pd.Timedelta(days=(4 - m.weekday()) % 7 + 14)   # 1st Fri + 2 weeks
        if third_fri >= today:
            out.append(third_fri)
        if len(out) == n:
            break
    return out


def _mock_oi_chain(ticker: str, n_expiries: int, n_strikes: int | None = None,
                   step: float | None = None, width: float | None = None) -> pd.DataFrame:
    """Synthetic option chain: an ATM-peaked open-interest surface, front-loaded with a
    quarterly (Mar/Jun/Sep/Dec) bump, calls building OTM-upside / puts OTM-downside and
    a structural put tilt for index options (hedging) — enough to exercise the heatmap
    in demo mode. Deterministic per ticker. Strikes are anchored on the same mock spot
    (last get_history settle) the report marks, so the ATM line sits centred. The strike
    increment (`step`) and OI concentration half-width (`width`) are in PRICE units;
    omitted → asset-class defaults so fixed income gets a fine rate grid, not a 2.5% one."""
    s = get_history([ticker])
    spot = float(s[ticker].dropna().iloc[-1]) if (ticker in s and s[ticker].notna().any()) \
        else float(INSTRUMENTS.get(ticker, (ticker, 100.0, "", ""))[1])
    if spot <= 0:
        return pd.DataFrame(columns=_OI_COLS)
    asset = INSTRUMENTS.get(ticker, (ticker, 0.0, "", ""))[2]
    d_step, d_width = _asset_chain_defaults(asset, spot)
    width = width or d_width
    if n_strikes is None:
        # Full chain: a realistic listed range at a realistic increment (~width/7), spanning
        # ~±3.6σ so it covers the whole board the way an exchange lists it — a hot ATM core
        # fading to light wings at the edges (a real chain carries far-OTM strikes too).
        inc = step or _nice(width / 7.0)
        n_each = max(8, min(44, int(round(3.6 * width / inc))))
        strikes = _strike_ladder(spot, 2 * n_each + 1, inc)
    else:
        strikes = _strike_ladder(spot, n_strikes, step or d_step)
    exps = _forward_expiries(n_expiries)
    if not strikes or not exps:
        return pd.DataFrame(columns=_OI_COLS)

    rng = np.random.default_rng(_seed(ticker) ^ 0x0117EEEE)
    scale = float(_seed(ticker) % 60 + 8) * 800.0                  # per-product OI size
    pc_tilt = 1.35 if asset == "Indices" else 1.08 if asset in ("Bonds", "STIRs") else 0.95

    rows = []
    for j, e in enumerate(exps):
        exp_w = float(np.exp(-0.30 * j)) * (1.5 if e.month in (3, 6, 9, 12) else 1.0)
        for k in strikes:
            d = k - spot                                           # distance from ATM (price units)
            m = d / spot                                           # moneyness (for the call/put skew)
            atm_peak = np.exp(-0.5 * (d / width) ** 2)            # OI concentrated near ATM
            wing = 0.22 * np.exp(-0.5 * ((abs(d) - 1.7 * width) / (0.8 * width)) ** 2)   # OTM wing bump
            shape = atm_peak + wing
            base = scale * exp_w * shape
            call = base * (1.0 + 2.5 * max(m, 0.0)) * (1.0 + rng.normal(0, 0.18))
            put = base * (1.0 + 2.5 * max(-m, 0.0)) * pc_tilt * (1.0 + rng.normal(0, 0.18))
            rows.append((e, e.strftime("%b %y"), float(k),
                         float(max(0.0, round(call))), float(max(0.0, round(put)))))
    return pd.DataFrame(rows, columns=_OI_COLS)


def _parse_opt_desc(desc: str):
    """Parse an OPT_CHAIN security description like 'TYN6P 84.25 Comdty' into
    (security, strike, (year, month), put_call) — used only to FILTER the chain cheaply
    before the OPEN_INT pull (the authoritative strike/expiry/PC come from bdp). The prefix
    is ROOT+MONTHCODE+YEAR+P/C; parsed right-to-left so the variable-length root is fine."""
    norm = re.sub(r"\s+", " ", str(desc)).strip()
    parts = norm.split(" ")
    if len(parts) < 2:
        return None
    prefix = parts[0]
    strike = None
    for tok in parts[1:]:                       # first numeric token after the prefix = strike
        try:
            strike = float(tok)
            break
        except ValueError:
            continue
    if strike is None or len(prefix) < 3:
        return None
    pc = prefix[-1].upper()
    if pc not in ("P", "C"):
        return None
    body = prefix[:-1]                           # root + month code + year
    i = len(body)
    while i > 0 and body[i - 1].isdigit():       # trailing digits = year
        i -= 1
    yr, mon_code = body[i:], (body[i - 1].upper() if i >= 1 else "")
    mon = _MONTH_CODE.get(mon_code)
    if not yr or mon is None:
        return None
    if len(yr) >= 2:
        year = 2000 + int(yr[-2:])
    else:                                        # single-digit year → roll within this decade
        this_year = pd.Timestamp.today().year
        year = (this_year // 10) * 10 + int(yr)
        if year < this_year - 1:
            year += 10
    return (norm, strike, (year, mon), pc)


def _bdp_rows(info) -> dict:
    """Normalise an xbbg bdp result (long ticker/field/value OR wide index=ticker) to
    {security: {FIELD: value}}."""
    if info is None or len(info) == 0:
        return {}
    low = {str(c).lower(): c for c in info.columns}
    out: dict = {}
    if {"ticker", "field", "value"}.issubset(low):                 # long / tidy
        tc, fc, vc = low["ticker"], low["field"], low["value"]
        for _, r in info.iterrows():
            out.setdefault(str(r[tc]), {})[str(r[fc]).upper()] = r[vc]
    else:                                                          # wide: index = ticker
        for tk, r in info.iterrows():
            out[str(tk)] = {str(c).upper(): r[c] for c in info.columns}
    return out


def _option_securities(ticker: str) -> list:
    """Every option security description for a future, across the EXPIRY STRIP. The generic
    future's OPT_CHAIN only exposes the FRONT expiries (and is empty for some, e.g. SONIA/ESTR),
    so we ALSO walk the dated contracts (FUT_CHAIN → OPT_CHAIN on the nearest OI_LIVE_MAX_CONTRACTS)
    and union everything — that's what surfaces the full strip of quarterlies. Members live in the
    bulk 'Security Description' column, not the 'ticker'/'field' metadata columns xbbg prepends."""
    from . import bbg as blp

    def _chain(sec):
        try:
            ch = _coerce_pd(blp.bds(sec, "OPT_CHAIN"))
        except Exception:
            return []
        if ch is None or len(ch) == 0:
            return []
        vc = next((c for c in ch.columns if str(c).lower() not in ("ticker", "field")), None)
        return [str(x) for x in ch[vc].dropna().tolist()] if vc else []

    secs = list(_chain(ticker))                        # generic chain (front expiries; may be empty)
    try:
        fc = _coerce_pd(blp.bds(ticker, "FUT_CHAIN"))   # the dated contracts → the rest of the strip
    except Exception:
        fc = None
    if fc is not None and len(fc):
        vc = next((c for c in fc.columns if str(c).lower() not in ("ticker", "field")), None)
        if vc is not None:
            for d in [str(x).strip() for x in fc[vc].dropna().tolist() if str(x).strip()][:OI_LIVE_MAX_CONTRACTS]:
                secs.extend(_chain(d))
    return list(dict.fromkeys(secs))                   # dedupe, preserve order                    # dedupe, preserve order


def _bloomberg_oi_chain(ticker: str) -> pd.DataFrame:
    """LIVE option chain → tidy (expiry, strike, call_oi, put_oi). Lists the chain members
    (_option_securities — generic OPT_CHAIN, or the dated-contract fallback), filters to the
    nearest expiries and near-the-money strikes (parsed from the description — see the OI_LIVE_*
    bounds), then pulls per-option strike / expiry / put-call / OPEN_INT in one batched bdp and
    pivots puts vs calls. Grouping is by the OPTION's expiry (OPT_EXPIRE_DT), which differs from
    the underlying contract month. A thin / absent chain (much of FX) returns empty."""
    from . import bbg as blp
    secs_raw = _option_securities(ticker)
    if not secs_raw:
        return pd.DataFrame(columns=_OI_COLS)
    parsed = [p for p in (_parse_opt_desc(s) for s in secs_raw) if p]
    if not parsed:
        return pd.DataFrame(columns=_OI_COLS)

    # Filter cheaply BEFORE the OPEN_INT pull: nearest expiry contracts, ±moneyness of spot,
    # capped strikes per expiry. (The contract month is only a proxy for ordering expiries.)
    px = get_history([ticker])
    spot = float(px[ticker].dropna().iloc[-1]) if (ticker in px and px[ticker].notna().any()) else 0.0
    near_ym = set(sorted({p[2] for p in parsed})[:OI_LIVE_MAX_EXPIRIES])
    cand = [p for p in parsed if p[2] in near_ym]
    if spot > 0:
        cand = [p for p in cand if abs(p[1] / spot - 1.0) <= OI_LIVE_MONEYNESS]
        by_ym: dict = {}
        for p in cand:
            by_ym.setdefault(p[2], []).append(p)
        cand = []
        for ps in by_ym.values():
            ps.sort(key=lambda p: abs(p[1] - spot))
            cand.extend(ps[:OI_LIVE_STRIKES_PER_EXP])
    secs = [p[0] for p in cand]
    if not secs:
        return pd.DataFrame(columns=_OI_COLS)

    try:
        info = _bdp_rows(_coerce_pd(blp.bdp(tickers=secs, flds=OI_CHAIN_FIELDS)))
    except Exception:
        return pd.DataFrame(columns=_OI_COLS)

    rec: dict = {}
    for _sec, d in info.items():
        strike = _num(d.get("OPT_STRIKE_PX"))
        oi = _num(d.get("OPEN_INT"))
        exp = pd.to_datetime(d.get("OPT_EXPIRE_DT"), errors="coerce")
        pc = str(d.get("OPT_PUT_CALL", "")).strip().upper()[:1]    # 'P' / 'C'
        if not np.isfinite(strike) or pd.isna(exp) or pc not in ("P", "C"):
            continue
        slot = rec.setdefault((exp.normalize(), float(strike)), {"call_oi": 0.0, "put_oi": 0.0})
        slot["call_oi" if pc == "C" else "put_oi"] += (oi if np.isfinite(oi) else 0.0)

    if not rec:
        return pd.DataFrame(columns=_OI_COLS)
    rows = [(e, e.strftime("%b %y"), k, v["call_oi"], v["put_oi"]) for (e, k), v in rec.items()]
    return (pd.DataFrame(rows, columns=_OI_COLS)
            .sort_values(["expiry", "strike"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
# LIVE OVERNIGHT QUOTE  (captured at snapshot time)
# Current price vs the PRIOR SETTLEMENT — Bloomberg CHG_PCT_1D / CHG_NET_1D — i.e.
# the move from the previous trading day's close to the instant the quote is pulled.
# This is the "net change on the day" a futures desk quotes, and it auto-handles
# weekends/holidays (Monday references Friday). get_history (settlement close-to-
# close) can only show the last COMPLETED session, so the STRATEGY MODELS keep
# using it untouched; ONLY the displayed "day move" uses this live quote.
# snapshot.py captures it into live.parquet at the morning pull (Terminal up);
# the dashboard reads that file in snapshot mode.
# ---------------------------------------------------------------------------
LIVE_FIELDS = ["PX_LAST", "CHG_NET_1D", "CHG_PCT_1D"]
_LIVE_COLS = ["last", "net", "pct"]


def _num(v):
    try:
        x = float(v)
        return x if x == x else np.nan       # reject NaN
    except (TypeError, ValueError):
        return np.nan


def _read_live():
    """Read the live-quote snapshot (ticker-indexed), or None if not present."""
    p = SNAPSHOT_DIR / "live.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    return df


def get_live_quote(tickers) -> pd.DataFrame:
    """Per-ticker live quote — DataFrame indexed by ticker with columns
    ['last','net','pct']: current price, plus net and % change vs the prior
    settlement (previous trading day's close -> now). Empty frame if unavailable.
    Mode-agnostic seam like get_history: bloomberg pulls live, snapshot reads
    live.parquet, mock synthesises a small overnight move."""
    tickers = list(tickers)
    if MODE == "bloomberg":
        return _bloomberg_live_quote(tickers)
    if MODE == "snapshot":
        live = _read_live()
        if live is not None:
            return live.reindex(tickers)[_LIVE_COLS]
    return _mock_live_quote(tickers)


def _bloomberg_live_quote(tickers) -> pd.DataFrame:
    from . import bbg as blp
    pdf = _coerce_pd(blp.bdp(tickers=list(tickers), flds=LIVE_FIELDS))
    if pdf is None or len(pdf) == 0:
        return pd.DataFrame(columns=_LIVE_COLS)
    lower = {str(c).lower(): c for c in pdf.columns}
    rows = {}
    if {"ticker", "field", "value"}.issubset(lower):              # long / tidy
        tcol, fcol, vcol = lower["ticker"], lower["field"], lower["value"]
        for _, r in pdf.iterrows():
            rows.setdefault(r[tcol], {})[str(r[fcol]).upper()] = r[vcol]
    else:                                                         # wide: index=ticker
        for tk, r in pdf.iterrows():
            rows[tk] = {str(c).upper(): r[c] for c in pdf.columns}
    out = {tk: {"last": _num(d.get("PX_LAST")),
                "net":  _num(d.get("CHG_NET_1D")),
                "pct":  _num(d.get("CHG_PCT_1D"))}
           for tk, d in rows.items()}
    df = pd.DataFrame.from_dict(out, orient="index")
    return df.reindex(list(tickers))[_LIVE_COLS] if len(df) else pd.DataFrame(columns=_LIVE_COLS)


def _mock_live_quote(tickers) -> pd.DataFrame:
    """Synthetic live quote so the dashboard's day-move works offline / in demo:
    a deterministic small overnight % move off the last mock settle."""
    px = get_history(list(tickers))
    out = {}
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if s.empty:
            continue
        settle = float(s.iloc[-1])
        pct = float(np.random.default_rng(_seed(t) ^ 0x10FE10FE).normal(0.0, 0.9))
        net = settle * pct / 100.0
        out[t] = {"last": round(settle + net, 4), "net": round(net, 4), "pct": round(pct, 2)}
    return (pd.DataFrame.from_dict(out, orient="index").reindex(list(tickers))[_LIVE_COLS]
            if out else pd.DataFrame(columns=_LIVE_COLS))
