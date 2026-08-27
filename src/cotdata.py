"""CFTC Commitments of Traders (COT) data layer — the seam between the app and the
free CFTC public-reporting API (Socrata). No Bloomberg / Terminal needed.

For each universe product we pull weekly speculative positioning:
  • COMMODITIES → the **Disaggregated** report, **Managed Money** (the hedge-fund /
    CTA bucket — the true "managed money" the user asked for).
  • FINANCIALS (equity indices, rates, FX) → the **Traders in Financial Futures**
    (TFF) report, **Leveraged Funds** (managed money has no analogue in TFF; this is
    the speculative hedge-fund bucket for financials).

Both are the Futures-and-Options **combined** datasets, so the long/short figures
match the Bloomberg `…MM…`/`…LF…` F&O tickers (CL = CFCDQMML/S/N, etc.).

The headline read is the **COT Index** — a 0–100 stochastic of the net position
over a rolling 3-year window (0 = most net-short in 3yrs, 100 = most net-long), the
trader-standard way to spot positioning extremes. `compute()` caches the full weekly
history (for the charts) + a latest cross-section (for the table / ranked bar), the
same compute-once-daily pattern as the other strategies. ICE-Europe products (Brent,
Gasoil, Robusta), Eurex, and non-US markets have NO CFTC COT and are simply absent.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import name, asset
from .datafeed import get_history, MODE

# ---------------------------------------------------------------------------
# CFTC Socrata API
# ---------------------------------------------------------------------------
BASE = "https://publicreporting.cftc.gov/resource/"
DATASETS = {
    "disagg": "kh3c-gbw2",   # Disaggregated — Futures-and-Options COMBINED (Managed Money)
    "tff": "yw9f-hn96",      # TFF — Futures-and-Options COMBINED (Leveraged Funds); includes options.
    # NB: gpe5-46if is "TFF - Futures Only" (excludes options) — do NOT use it here.
}
# (long field, short field) per report
FIELDS = {
    "disagg": ("m_money_positions_long_all", "m_money_positions_short_all"),
    "tff": ("lev_money_positions_long", "lev_money_positions_short"),
}
CATEGORY = {"disagg": "Managed Money", "tff": "Leveraged Funds"}

# universe Bloomberg ticker -> (CFTC contract_market_code, report). Validated live
# 2026-06-12 against the CFTC API (each is the primary contract by open interest);
# the WTI codes reproduce the user's known-good CFCDQMM* tickers exactly.
COT_MAP = {
    # ---- Commodities → Disaggregated / Managed Money ----
    "CLA Comdty": ("067651", "disagg"),   # WTI-PHYSICAL
    "XBA Comdty": ("111659", "disagg"),   # GASOLINE RBOB
    "NGA Comdty": ("023651", "disagg"),   # NAT GAS NYME (Henry Hub)
    "HOA Comdty": ("022651", "disagg"),   # NY HARBOR ULSD (heating oil)
    "GCA Comdty": ("088691", "disagg"),   # GOLD
    "SIA Comdty": ("084691", "disagg"),   # SILVER
    "PLA Comdty": ("076651", "disagg"),   # PLATINUM
    "PAA Comdty": ("075651", "disagg"),   # PALLADIUM
    "HGA Comdty": ("085692", "disagg"),   # COPPER- #1 (high-grade)
    "C A Comdty": ("002602", "disagg"),   # CORN
    "S A Comdty": ("005602", "disagg"),   # SOYBEANS
    "W A Comdty": ("001602", "disagg"),   # WHEAT-SRW (Chicago)
    "KWA Comdty": ("001612", "disagg"),   # WHEAT-HRW (KC)
    "SMA Comdty": ("026603", "disagg"),   # SOYBEAN MEAL
    "BOA Comdty": ("007601", "disagg"),   # SOYBEAN OIL
    "RRA Comdty": ("039601", "disagg"),   # ROUGH RICE
    "KCA Comdty": ("083731", "disagg"),   # COFFEE C
    "CCA Comdty": ("073732", "disagg"),   # COCOA
    "SBA Comdty": ("080732", "disagg"),   # SUGAR NO. 11
    "CTA Comdty": ("033661", "disagg"),   # COTTON NO. 2
    "JOA Comdty": ("040701", "disagg"),   # FRZN CONCENTRATED ORANGE JUICE
    "LHA Comdty": ("054642", "disagg"),   # LEAN HOGS
    "LCA Comdty": ("057642", "disagg"),   # LIVE CATTLE
    "FCA Comdty": ("061641", "disagg"),   # FEEDER CATTLE
    # ---- Financials → TFF / Leveraged Funds ----
    "ESA Index": ("13874A", "tff"),       # E-MINI S&P 500
    "NQA Index": ("209742", "tff"),       # NASDAQ MINI
    "RTYA Index": ("239742", "tff"),      # RUSSELL E-MINI
    "DMA Index": ("124603", "tff"),       # DJIA x $5 (e-mini)
    # NKA Index (Nikkei) intentionally excluded: the only CFTC-listed Nikkei is the thin
    # CME contract; the real positioning sits on Osaka/SGX, which CFTC COT doesn't cover.
    "FFA Comdty": ("045601", "tff"),      # FED FUNDS (30-day)
    "SFRA Comdty": ("134741", "tff"),     # SOFR-3M
    "SERA Comdty": ("134742", "tff"),     # SOFR-1M
    "USA Comdty": ("020601", "tff"),      # UST BOND
    "WNA Comdty": ("020604", "tff"),      # ULTRA US T BOND
    "UXYA Comdty": ("043607", "tff"),     # ULTRA UST 10Y
    "TYA Comdty": ("043602", "tff"),      # UST 10Y NOTE
    "FVA Comdty": ("044601", "tff"),      # UST 5Y NOTE
    "TUA Comdty": ("042601", "tff"),      # UST 2Y NOTE
    "ECA Curncy": ("099741", "tff"),      # EURO FX
    "BPA Curncy": ("096742", "tff"),      # BRITISH POUND
    "JYA Curncy": ("097741", "tff"),      # JAPANESE YEN
    "SFA Curncy": ("092741", "tff"),      # SWISS FRANC
    "CDA Curncy": ("090741", "tff"),      # CANADIAN DOLLAR
    "ADA Curncy": ("232741", "tff"),      # AUSTRALIAN DOLLAR
    "NVA Curncy": ("112741", "tff"),      # NZ DOLLAR
    "PEA Curncy": ("095741", "tff"),      # MEXICAN PESO
    "BRA Curncy": ("102741", "tff"),      # BRAZILIAN REAL
    "RAA Curncy": ("122741", "tff"),      # SO AFRICAN RAND
}

INDEX_WINDOW = 156      # weeks (~3y) for the COT Index stochastic
INDEX_MINP = 26         # need ≥ half a year of weeks before an index is meaningful
STORE_WEEKS = 1100      # keep full CFTC history (~21y) per market — deep enough for seasonality
EXTREME_HI = 80.0       # COT Index ≥ this → crowded long
EXTREME_LO = 20.0       # COT Index ≤ this → crowded short

DATA = Path(__file__).resolve().parents[1] / "data"
SIGNALS = DATA / "signals"
COT_HISTORY_FILE = SIGNALS / "cot_history.parquet"
COT_DETAIL_FILE = SIGNALS / "cot.parquet"

HISTORY_COLUMNS = ["date", "ticker", "market", "asset", "category",
                   "long", "short", "net", "oi", "net_pct_oi", "npo_p10", "npo_med",
                   "npo_p90", "cot_index", "price"]
DETAIL_COLUMNS = ["market", "ticker", "asset", "category", "date", "long", "short",
                  "net", "oi", "net_pct_oi", "cot_index", "chg_net", "signal", "direction"]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _get(dataset: str, params: dict):
    url = BASE + DATASETS[dataset] + ".json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "strategy-monitor-cot"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:   # transient network / rate-limit — retry a couple times
            last = e
    raise last


def fetch_cot(ticker: str) -> pd.DataFrame:
    """Full weekly history for one product: date, long, short, net, oi (ascending)."""
    code, ds = COT_MAP[ticker]
    lf, sf = FIELDS[ds]
    rows = _get(ds, {
        "cftc_contract_market_code": code,
        "$select": f"report_date_as_yyyy_mm_dd,{lf},{sf},open_interest_all",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": "20000",
    })
    if not rows:
        return pd.DataFrame(columns=["date", "long", "short", "net", "oi"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["long"] = pd.to_numeric(df[lf], errors="coerce")
    df["short"] = pd.to_numeric(df[sf], errors="coerce")
    df["oi"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
    df = df.dropna(subset=["long", "short"]).sort_values("date")
    df["net"] = df["long"] - df["short"]
    return df[["date", "long", "short", "net", "oi"]].reset_index(drop=True)


def cot_index(net: pd.Series, window: int = INDEX_WINDOW, min_periods: int = INDEX_MINP) -> pd.Series:
    """COT Index 0–100: the net position's position within its rolling min/max range
    (Williams %R style). 0 = most net-short in the window, 100 = most net-long."""
    lo = net.rolling(window, min_periods=min_periods).min()
    hi = net.rolling(window, min_periods=min_periods).max()
    rng = (hi - lo).where(lambda s: s > 0)
    return 100.0 * (net - lo) / rng


# ---------------------------------------------------------------------------
# Compute (cache-aware) + cross-section
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Persistent price database — backfill once, then extend incrementally
# ---------------------------------------------------------------------------
# The forward-return study needs years of underlying price aligned to the weekly COT
# dates. Rather than re-pull ~10y from Bloomberg every run, keep a local DB: the first
# Bloomberg run backfills `years`; each later Bloomberg run pulls ONLY the dates since
# the last stored row (plus a small overlap to catch revised settles) and appends them.
# Off-Bloomberg (snapshot / mock) it is read-only.
PRICE_STORE = DATA / "price_store" / "cot_prices.parquet"
STORE_YEARS = 10
STORE_BUFFER_DAYS = 10      # re-pull this many trailing days each run (revised settles)
STORE_CHUNK = 8            # backfill in small ticker groups — avoids single-request data caps
HEAL_MIN_DAYS = 750        # a ticker holding < ~3y of daily price is treated as truncated → re-backfill


def read_price_store() -> pd.DataFrame:
    """The persistent price DB as a wide, date-indexed frame (empty if not built yet)."""
    if not PRICE_STORE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PRICE_STORE)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _merge_price_store(store: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of dates/tickers; `new` wins on any overlap (so revised settles update)."""
    if store is None or store.empty:
        return new.sort_index() if new is not None and not new.empty else pd.DataFrame()
    if new is None or new.empty:
        return store.sort_index()
    return new.combine_first(store).sort_index()


def _hist_source(tkr: str) -> str:
    """The Bloomberg ticker to pull deep history from — the 1st-generic, e.g.
    'GCA Comdty' -> 'GC1 Comdty', 'C A Comdty' -> 'C 1 Comdty', 'ECA Curncy' -> 'EC1 Curncy'.
    The 'A' active-generic serves truncated history for many contracts (Treasuries ~6mo,
    gold ~1.7y…) whereas the '1' generic returns the full ~10y — verified live 2026-06-15."""
    gen, key = tkr.rsplit(" ", 1)
    if gen.endswith("A"):
        gen = gen[:-1] + "1"
    return f"{gen} {key}"


def _pull_history(cot_tickers, start, end):
    """Pull history from each ticker's 1st-generic source, returned keyed by the original
    (COT-map) ticker so the store columns stay aligned to the rest of the app."""
    if not cot_tickers:
        return None
    rev = {_hist_source(t): t for t in cot_tickers}        # source ticker -> COT ticker
    wide = get_history(list(rev), start=start, end=end)
    if wide is None or getattr(wide, "empty", True):
        return None
    wide = wide.rename(columns=rev)
    keep = [t for t in cot_tickers if t in wide.columns]
    return wide[keep] if keep else None


def update_price_store(tickers, years: int = STORE_YEARS, buffer_days: int = STORE_BUFFER_DAYS) -> pd.DataFrame:
    """Return the price DB, extending it first when on Bloomberg. Since 2026-08-18
    this NO LONGER pulls Bloomberg itself: the deep store (src/deepstore.py) holds
    the identical raw '1'-generic settle series for every COT market (same source
    generics via the same 'A'→'1' rule, same price fields, same 10y depth), so the
    Bloomberg-connected moment just MIRRORS the deep store's raw frame into this
    DB — deepstore.update must run first in the fetch phase. Any COT ticker the
    deep store doesn't carry falls back to the legacy direct pull (none today —
    the 47 COT markets are a subset of the book). A no-op read off-Bloomberg."""
    store = read_price_store()
    if MODE != "bloomberg":
        return store
    tickers = list(tickers)
    today = pd.Timestamp.now().normalize()
    full_start = today - pd.DateOffset(years=years)
    frames = []
    try:
        from . import deepstore
        deep = deepstore.get_raw(tickers, start=full_start)
        if deep is not None and not deep.empty:
            frames.append(deep)                        # full depth; merge handles revisions
    except Exception:
        deep = pd.DataFrame()
    missing = [t for t in tickers if not len(frames) or t not in frames[0].columns]
    for i in range(0, len(missing), STORE_CHUNK):      # legacy pull, only off-book tickers
        f = _pull_history(missing[i:i + STORE_CHUNK], full_start, today)
        if f is not None:
            frames.append(f)
    new = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    store = _merge_price_store(store, new)
    if not store.empty:
        try:
            PRICE_STORE.parent.mkdir(parents=True, exist_ok=True)
            store.rename_axis("date").reset_index().to_parquet(PRICE_STORE, index=False)
        except Exception:
            pass
    return store


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_h = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600.0
    return age_h <= max_age_hours


def _cross_section(hist: pd.DataFrame) -> pd.DataFrame:
    """Latest row per market + the crowded-long/short flag, most-extreme first."""
    recs = []
    for tkr, g in hist.groupby("ticker", sort=False):
        g = g.sort_values("date")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else last
        idx = float(last["cot_index"]) if pd.notna(last["cot_index"]) else float("nan")
        if np.isfinite(idx) and idx >= EXTREME_HI:
            signal, direction = "Crowded long", 1
        elif np.isfinite(idx) and idx <= EXTREME_LO:
            signal, direction = "Crowded short", -1
        else:
            signal, direction = "—", 0
        recs.append({
            "market": last["market"], "ticker": tkr, "asset": last["asset"],
            "category": last["category"], "date": last["date"],
            "long": float(last["long"]), "short": float(last["short"]),
            "net": float(last["net"]), "oi": float(last["oi"]),
            "net_pct_oi": float(last["net_pct_oi"]) if pd.notna(last["net_pct_oi"]) else np.nan,
            "cot_index": round(idx, 1) if np.isfinite(idx) else np.nan,
            "chg_net": float(last["net"] - prev["net"]),
            "signal": signal, "direction": direction,
        })
    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = (df["cot_index"].fillna(50) - 50).abs().sort_values(ascending=False).index
        df = df.reindex(order).reset_index(drop=True)
    return df


def compute(max_age_hours: float = 20.0, force: bool = False):
    """Return (cross_section_df, history_df), fetching from the CFTC API and caching
    to parquet. Reuses a cache younger than `max_age_hours` so the offline dev loop
    (Re-run signals) stays instant; the daily scheduled run refetches. On a total
    network failure it falls back to the last cache if present."""
    SIGNALS.mkdir(parents=True, exist_ok=True)

    if not force and _is_fresh(COT_HISTORY_FILE, max_age_hours):
        try:
            hist = pd.read_parquet(COT_HISTORY_FILE)
            return _cross_section(hist), hist
        except Exception:
            pass

    # The underlying price for the chart's overlay line — pulled once from the same
    # datafeed seam as every other report (mock / snapshot / bloomberg) and aligned
    # onto each weekly COT date (price as of that Tuesday). Baked into the history
    # cache so the standalone report script never has to call datafeed itself.
    try:
        px = update_price_store(list(COT_MAP))   # incremental Bloomberg DB; read-only off-Bloomberg
    except Exception:
        px = read_price_store()
    if px is None or px.empty:                   # no DB yet → one-off pull (mock = demo-thin)
        try:
            px = get_history(list(COT_MAP), raw=True,   # price CHART overlay — real settles
                             start=pd.Timestamp.now().normalize() - pd.DateOffset(years=10))
        except Exception:
            px = pd.DataFrame()

    frames = []
    for tkr, (code, ds) in COT_MAP.items():
        try:
            h = fetch_cot(tkr)
        except Exception:
            continue
        if h.empty:
            continue
        h = h.sort_values("date")
        h["cot_index"] = cot_index(h["net"])
        h["net_pct_oi"] = 100.0 * h["net"] / h["oi"].where(h["oi"] > 0)
        _roll = h["net_pct_oi"].rolling(INDEX_WINDOW, min_periods=INDEX_MINP)
        h["npo_p10"] = _roll.quantile(0.10)     # rolling 3y 10th/median/90th percentile
        h["npo_med"] = _roll.quantile(0.50)     # band → "stretch vs the market's own normal"
        h["npo_p90"] = _roll.quantile(0.90)
        h["ticker"] = tkr
        h["market"] = name(tkr)
        h["asset"] = asset(tkr)
        h["category"] = CATEGORY[ds]
        if tkr in getattr(px, "columns", []) and px[tkr].notna().any():
            p = px[tkr].dropna()
            p = pd.DataFrame({"date": pd.to_datetime(p.index).astype("datetime64[ns]"),
                              "price": p.values}).sort_values("date")
            # Normalise BOTH merge keys to the same datetime unit: snapshot/Bloomberg
            # prices can return ms while the CFTC dates parse to us, and merge_asof
            # requires identical key dtypes (else "incompatible merge keys").
            h["date"] = pd.to_datetime(h["date"]).astype("datetime64[ns]")
            h = pd.merge_asof(h.sort_values("date"), p, on="date", direction="backward")
        else:
            h["price"] = np.nan
        frames.append(h.tail(STORE_WEEKS)[HISTORY_COLUMNS])

    if not frames:
        if COT_HISTORY_FILE.exists():
            hist = pd.read_parquet(COT_HISTORY_FILE)
            return _cross_section(hist), hist
        empty = pd.DataFrame(columns=HISTORY_COLUMNS)
        return pd.DataFrame(columns=DETAIL_COLUMNS), empty

    hist = pd.concat(frames, ignore_index=True)
    detail = _cross_section(hist)
    try:
        hist.to_parquet(COT_HISTORY_FILE, index=False)
        detail.to_parquet(COT_DETAIL_FILE, index=False)
    except Exception:
        pass
    return detail, hist


# ---------------------------------------------------------------------------
# Hot Sheet provider
# ---------------------------------------------------------------------------
RADAR_EXTREME_MAX = 3      # crowded-positioning extremes shown (cross-section is already |idx−50| sorted)
RADAR_SHIFT_MAX = 2        # the week's largest size-normalised net shifts
RADAR_SHIFT_Z_FULL = 5.0   # σ that pins a shift's heat gauge at 100 (the weekreview convention)
RADAR_SPARK_WEEKS = 104    # sparkline lookback — ~2y of weekly CFTC prints


def _radar_spark(hist: pd.DataFrame, ticker: str, col: str) -> list:
    """Trailing ~2y of one market's weekly `col`, oldest→newest — the sparkline
    series for radar_items (cot_index for the crowding stories, net for shifts)."""
    g = hist.loc[hist["ticker"] == ticker].sort_values("date")
    return pd.to_numeric(g[col], errors="coerce").tail(RADAR_SPARK_WEEKS).tolist()


def radar_items() -> list:
    """Hot Sheet provider: Friday's positioning extremes (the crowded 80/20
    cutoff) plus the week's largest size-normalised net shifts — the same
    selection and prose as weekreview.collect_cot. Cache only: compute() runs
    with an effectively infinite max-age so the parquet on disk is reused,
    never refetched; no cache at all means quiet, not a pull."""
    import sys as _sys

    from src import hotsheet

    if not COT_HISTORY_FILE.exists():        # no cache yet — quiet, never fetch here
        return []
    try:                                     # probe the cache readable — a corrupt parquet
        pd.read_parquet(COT_HISTORY_FILE, columns=["date"])
    except Exception:                        # would fall through compute() to a live fetch
        return []
    detail, hist = compute(max_age_hours=1e9)
    out = []
    ext = detail[detail["direction"] != 0].head(RADAR_EXTREME_MAX)   # already |idx−50| sorted
    for r in ext.itertuples(index=False):
        side = "long" if r.direction > 0 else "short"
        # `direction` is the COT INDEX's read (a 3y min-max rank), while net_pct_oi is the
        # ABSOLUTE net. For a structurally one-sided book those disagree: leveraged funds are
        # permanently short the 2Y as the cash-futures basis counterparty, so index 100 means
        # "least short in three years", NOT net long. Asserting the side then quoting a net
        # that contradicts it published "crowded long — net -27% of open interest" as slot 1
        # of a client-facing sheet (2026-08-26). When the two disagree, describe the RANK,
        # which is what the index actually measures.
        _agrees = (r.net_pct_oi > 0) == (r.direction > 0)
        if _agrees:
            _claim = f"**crowded {side}** — net {r.net_pct_oi:+.0f}% of open interest"
        else:
            _pos = "short" if r.net_pct_oi < 0 else "long"
            _claim = (f"the **least {_pos} in three years** — still net "
                      f"{r.net_pct_oi:+.0f}% of open interest")
        out.append(hotsheet.item(
            tag="COT", key=f"{r.ticker}:{side}", section="Positioning",
            text=f"**{r.market}** {r.category.lower()} positioning is {_claim} "
                 f"({r.date:%d %b}).",
            heat=hotsheet.heat_from_pctl(r.cot_index),
            metric=f"idx {r.cot_index:.0f}", sub="COT index, 0–100",
            value=float(r.net_pct_oi),       # the crowding level — net %OI, Δ-able on the week
            ticker=r.ticker, page="COT Reports", book="ficc",
            # the metric quotes the COT index, so the spark draws its path
            spark=_radar_spark(hist, r.ticker, "cot_index"),
        ))
    try:
        # cotreport imports bare-name siblings (reportkit, cotstudy…) — its
        # standalone-script convention — so src/ must sit on sys.path first.
        _src = str(Path(__file__).resolve().parent)
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from src import cotreport
        tick = dict(zip(detail["market"], detail["ticker"]))
        for s in cotreport._weekly_shifts(detail, hist)[:RADAR_SHIFT_MAX]:
            _tkr = tick.get(s["market"], "")
            out.append(hotsheet.item(
                tag="COT", key=f"shift:{s['market']}", section="Positioning",
                text=f"**{s['market']}**: the week's net positioning move was "
                     f"**{s['chg']:+,.0f}** contracts ({s['pct']:+.1f}% of open "
                     f"interest).",
                heat=hotsheet.heat_from_z(s["z"], full=RADAR_SHIFT_Z_FULL),
                metric=f"{s['z']:+.1f}σ", sub="vs 3y of weekly moves",
                value=float(s["chg"]),       # contracts moved on the week
                ticker=_tkr, page="COT Reports", book="ficc",
                # net path in contracts — the last step IS the week's shift
                spark=_radar_spark(hist, _tkr, "net") if _tkr else None,
            ))
    except Exception:                        # shifts are additive — the extremes still stand
        pass
    return out
