"""Index Dispersion — engine.

Backs the index option-implied CORRELATION out of the vol market and compares it
with what the constituents have actually been doing. With w the basket weights,
sigma_i the single-name vols and sigma_I the index vol:

    rho = (sigma_I^2 - sum w_i^2 sigma_i^2) / ((sum w_i sigma_i)^2 - sum w_i^2 sigma_i^2)

Run once with 1M ATM IMPLIED vols (index + members) it gives the correlation the
options market is PRICING; run again with trailing 21-day REALIZED vols it gives
the correlation the stocks are DELIVERING. The spread between the two is the
dispersion monitor's flag: implied correlation at a high percentile of its own
history is the classic backdrop for short-index-vol / long-single-name-vol
structures, a low percentile for the reverse — either extreme may be worth a
closer look.

Basket: the top `top_n` members by index weight, weights renormalised (the
standard proxy — the S&P's tail adds noise, not signal). Weights are the CURRENT
membership applied through history, so the correlation series is a today's-basket
lens, not a point-in-time reconstruction.

Mode-branched like the rest of the Equities side: live Bloomberg (INDX_MWEIGHT
weights + the same 30-day ATM implied-vol surface field the futures vol pages
use, both verified live 2026-07-10) topping up on-disk caches that snapshot mode
reads; offline, member IVs are marked up from the factor-model realized vols and
the index IV is backed out of a seeded implied-correlation path, so the demo
percentiles genuinely move. Prices ride eqcorr.price_history (same cache).

No Streamlit here — the page drives this module.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import datafeed, eqcorr, equities

TENOR_SESSIONS = 21          # trailing realized window, matched to the 30-day IV tenor
HISTORY = {"6 months": 126, "1 year": 252, "2 years": 504}
HISTORY_ORDER = list(HISTORY)
DEFAULT_HISTORY = "1 year"
DEFAULT_TOP_N = 40           # basket cap — indices at/under it use full membership
RICH_PCTL, CHEAP_PCTL = 80.0, 20.0   # spread-percentile flags

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "equities"
_IV_FILE = _DATA_DIR / "disp_iv.parquet"          # date x ticker 1M ATM IV (members + indices)
_WEIGHTS_FILE = _DATA_DIR / "disp_weights.json"   # {index_key: {ticker: weight_pct}}


@dataclass
class DispersionSet:
    """One index's dispersion picture: the implied- and realized-correlation
    histories, their spread + current percentile, and the basket behind them."""
    asof: pd.Timestamp
    index_key: str
    source: str                   # prices + IVs provenance (for the caption)
    weight_source: str
    n_full: int                   # full membership the basket was cut from
    coverage: float               # kept basket weight / full index weight (0..1)
    basket: pd.DataFrame          # ticker, name, sector, weight, iv, rv, prem — by weight desc
    imp: pd.Series                # implied correlation, date-indexed
    real: pd.Series               # realized correlation (trailing 21d), date-indexed
    spread: pd.Series             # imp − real on common dates
    idx_iv: pd.Series             # index 1M ATM implied vol (%)
    idx_rv: pd.Series             # index 21d realized vol (ann. %)
    idx_spot: float               # latest index level (demo: reconstructed) — builder input
    pctl: float                   # current spread percentile within `spread`
    dropped: list                 # basket tickers excluded (no IV / no prices)


def flag(pctl: float) -> str:
    """Neutral-language signal label off the spread percentile."""
    if pctl != pctl:
        return "—"
    if pctl >= RICH_PCTL:
        return f"Implied corr rich · {pctl:.0f}th pctl"
    if pctl <= CHEAP_PCTL:
        return f"Implied corr cheap · {pctl:.0f}th pctl"
    return "Neutral"


# ── weights (mode-branched) ───────────────────────────────────────────────────
def _fmt_weights(bds_df) -> dict:
    """INDX_MWEIGHT → {'AAPL UQ Equity': pct}. Member codes live in the
    'Member Ticker and Exchange Code' bulk column, weights in 'Percentage Weight'
    (verified live 2026-07-10) — matched loosely in case the headers drift."""
    if bds_df is None or getattr(bds_df, "empty", True):
        return {}
    mcol = wcol = None
    for c in bds_df.columns:
        lc = str(c).lower()
        if "member" in lc or "exchange" in lc:
            mcol = mcol or c
        elif "weight" in lc:
            wcol = wcol or c
    if mcol is None or wcol is None:
        return {}
    out = {}
    for raw, w in zip(bds_df[mcol].astype(str), bds_df[wcol]):
        raw = raw.strip()
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if not raw or raw.lower() == "nan" or w != w:
            continue
        out[raw if raw.endswith("Equity") else f"{raw} Equity"] = w
    return out


def _cap_proxy_weights(members: list) -> dict:
    """Free-float-agnostic cap-weight proxy: CRNCY_ADJ_MKT_CAP (millions, local
    currency — every index this runs for is single-currency) normalised to 100.
    The fallback when INDX_MWEIGHT is zeroed out (see index_weights)."""
    from xbbg import blp
    rows = datafeed._bdp_rows(datafeed._coerce_pd(blp.bdp(members, ["crncy_adj_mkt_cap"])))
    caps = {}
    for t in members:
        try:
            v = float(rows.get(t, {}).get("CRNCY_ADJ_MKT_CAP"))
        except (TypeError, ValueError):
            continue
        if v == v and v > 0:
            caps[t] = v
    # dual share classes (GOOGL/GOOG, FOX/FOXA...) each report the COMPANY cap —
    # identical values to the cent — so split the cap across the classes rather
    # than double-counting the issuer
    if caps:
        s = pd.Series(caps).round(2)
        nclass = s.groupby(s).transform("size")
        caps = {t: v / float(nclass[t]) for t, v in caps.items()}
    tot = sum(caps.values())
    return {t: 100.0 * v / tot for t, v in caps.items()} if tot > 0 else {}


def _mock_weights(tickers: list) -> dict:
    """Deterministic cap-weight-shaped weights: lognormal by ticker seed, so a
    few names dominate each index the way real cap weights do."""
    raw = {t: float(np.exp(1.1 * np.random.default_rng(equities._seed("EQD-" + t))
                           .standard_normal())) for t in tickers}
    tot = sum(raw.values())
    return {t: 100.0 * v / tot for t, v in raw.items()}


def index_weights(index_key: str, fallback_tickers: list) -> tuple:
    """({ticker: weight_pct}, source label). bloomberg -> INDX_MWEIGHT, falling
    back to cap-proxy weights where the weight values come back ZEROED (~-2e-14:
    S&P/NASDAQ/DAX/STOXX/CAC weights need the index provider's licence — only UKX
    and INDU are entitled here, verified live 2026-07-10; the member CODES still
    populate). Caches to disp_weights.json; snapshot -> the cache; otherwise
    seeded mock weights."""
    if datafeed.MODE == "bloomberg":
        try:
            from xbbg import blp
            idx = equities.INDICES[index_key][1]
            w = _fmt_weights(datafeed._coerce_pd(blp.bds(idx, "INDX_MWEIGHT")))
            label = "live Bloomberg weights"
            if w and sum(abs(v) for v in w.values()) < 1.0:   # entitled pulls sum to ~100
                w = _cap_proxy_weights(list(w))
                label = "cap-weight proxy (index weights not licensed)"
            if w:
                try:
                    cached = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8")) \
                        if _WEIGHTS_FILE.exists() else {}
                except Exception:
                    cached = {}
                cached[index_key] = w
                # remember whether these were real or proxy weights, so snapshot
                # mode keeps showing the licence note on proxy-weighted indices
                cached.setdefault("_sources", {})[index_key] = label
                _DATA_DIR.mkdir(parents=True, exist_ok=True)
                _WEIGHTS_FILE.write_text(json.dumps(cached, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
                return w, label
        except Exception:
            pass
    if datafeed.MODE == "snapshot":
        try:
            cached = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
            if cached.get(index_key):
                label = cached.get("_sources", {}).get(index_key, "Bloomberg weights")
                return cached[index_key], f"cached {label}"
        except Exception:
            pass
    return _mock_weights(fallback_tickers), "demo weights"


# ── implied-vol history (mode-branched; mock is synthesized in compute) ───────
def _merge_iv_cache(new: pd.DataFrame) -> None:
    """Fold a live IV pull into disp_iv.parquet — union of dates/tickers, newest
    wins, never truncating longer history (same contract as eqcorr._merge_cache)."""
    try:
        old = pd.read_parquet(_IV_FILE) if _IV_FILE.exists() else None
    except Exception:
        old = None
    try:
        merged = new if old is None else new.combine_first(old)
        merged = merged.sort_index().iloc[-(max(HISTORY.values()) + 60):]
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged.rename_axis("date").to_parquet(_IV_FILE)
    except Exception:
        pass


def iv_history(tickers: list, sessions: int) -> tuple:
    """(1M ATM IV date x ticker | None, source label) for members + index tickers.
    None means 'no implied source here' — compute() then synthesizes demo IVs."""
    if datafeed.MODE == "bloomberg":
        try:
            from xbbg import blp
            end = pd.Timestamp.today().normalize()
            start = end - pd.Timedelta(days=int(sessions * 7 / 5) + 40)
            wide = datafeed._bdh_to_wide(blp.bdh(
                tickers=list(tickers), flds=datafeed.IMPLIED_VOL_FIELD,
                start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")))
            if wide is not None and not wide.empty:
                _merge_iv_cache(wide)
                return wide, "live Bloomberg"
        except Exception:
            pass
    if datafeed.MODE == "snapshot":
        try:
            cached = pd.read_parquet(_IV_FILE) if _IV_FILE.exists() else None
        except Exception:
            cached = None
        if cached is not None and not cached.empty:
            have = [t for t in tickers if t in cached.columns]
            if len(have) >= max(2, len(tickers) // 2):
                px = cached[have].sort_index()
                return px, f"cached Bloomberg IVs · to {px.index.max():%Y-%m-%d}"
    return None, "demo (synthetic) data"


# ── the correlation ratio ─────────────────────────────────────────────────────
def _corr_series(vols: pd.DataFrame, idx_vol: pd.Series, w: pd.Series,
                 min_weight: float = 0.6) -> pd.Series:
    """The rho ratio per date, weights renormalised over the members that have a
    vol that day; dates covering under `min_weight` of the basket come back NaN."""
    V = vols.reindex(columns=w.index)
    mask = V.notna().to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    wsum = mask @ wv
    with np.errstate(divide="ignore", invalid="ignore"):
        Wn = (mask * wv) / wsum[:, None]
        Vf = np.nan_to_num(V.to_numpy(dtype=float))
        s1 = (Wn * Vf).sum(axis=1)                      # sum w_i sigma_i
        s2 = (Wn ** 2 * Vf ** 2).sum(axis=1)            # sum w_i^2 sigma_i^2
        sI = idx_vol.reindex(V.index).to_numpy(dtype=float)
        rho = (sI ** 2 - s2) / (s1 ** 2 - s2)
    bad = (wsum < min_weight) | ~np.isfinite(rho) | ((s1 ** 2 - s2) <= 1e-12)
    rho = np.where(bad, np.nan, rho)
    return pd.Series(rho, index=V.index)


# ── demo IVs (offline only) ───────────────────────────────────────────────────
def _mock_ivs(rv: pd.DataFrame, real: pd.Series, w: pd.Series,
              index_key: str) -> tuple:
    """(member IVs, index IV) marked up from the realized series. Member IVs carry
    a seeded per-name vol premium; the index IV is backed OUT of a seeded
    implied-correlation path around the realized one, so implied sits rich most of
    the time but the spread percentile genuinely cycles."""
    ivs = {}
    for t in rv.columns:
        rng = np.random.default_rng(equities._seed("EQD-IV-" + t))
        prem = 0.06 + 0.22 * rng.random()
        noise = rng.normal(0.0, 0.02, len(rv))
        ivs[t] = rv[t] * (1.0 + prem + noise)
    ivm = pd.DataFrame(ivs, index=rv.index).clip(lower=5.0)

    rng = np.random.default_rng(equities._seed("EQD-RHO-" + index_key))
    t = np.arange(len(rv))
    base = real.ewm(span=20, min_periods=1).mean().fillna(0.40)
    rho = (0.55 * base + 0.16 + 0.10 * np.sin(2 * np.pi * t / 126 + rng.uniform(0, 6.3))
           + rng.normal(0.0, 0.03, len(rv))).clip(0.05, 0.95)

    V = ivm.reindex(columns=w.index).fillna(0.0).to_numpy()
    wv = w.to_numpy()
    s1 = V @ wv
    s2 = (V ** 2) @ (wv ** 2)
    idx_iv = pd.Series(np.sqrt(np.maximum(s2 + rho.to_numpy() * (s1 ** 2 - s2), 1.0)),
                       index=rv.index)
    return ivm, idx_iv


# ── shared data assembly ──────────────────────────────────────────────────────
def _inputs(universe: dict, index_key: str, hist_label: str, top_n: int) -> dict | None:
    """Everything compute() and backtest() share: the weighted basket, price /
    vol histories and the two UNTRIMMED correlation series. None when the index
    has no members or too little data to correlate."""
    n_hist = HISTORY[hist_label]
    members = universe.get(index_key, [])
    if not members:
        return None
    meta = {c["ticker"]: c for c in members}
    idx_ticker = equities.INDICES[index_key][1]

    w_all, weight_source = index_weights(index_key, list(meta))
    w_all = pd.Series(w_all, dtype=float).sort_values(ascending=False)
    w_all = w_all[w_all > 0]
    if len(w_all) < 5:
        return None
    kept = w_all.head(top_n)
    coverage = float(kept.sum() / w_all.sum())
    w = kept / kept.sum()
    tickers = list(w.index)
    sectors = {t: meta.get(t, {}).get("sector", "—") for t in tickers}

    # prices for members + the cash index (the index column is only synthesized
    # offline — the mock generator would hand it an unrelated random walk)
    sessions = n_hist + TENOR_SESSIONS + 30
    px, px_source = eqcorr.price_history(tickers + [idx_ticker], sectors, sessions)
    have = [t for t in tickers if t in px.columns]
    if len(have) < 5:
        return None
    rets = np.log(px[have].where(px[have] > 0)).diff()
    ann = np.sqrt(252.0) * 100.0
    rv = rets.rolling(TENOR_SESSIONS, min_periods=TENOR_SESSIONS - 4).std() * ann

    demo = px_source.startswith("demo")
    if not demo and idx_ticker in px.columns and px[idx_ticker].notna().sum() > TENOR_SESSIONS:
        idx_px = px[idx_ticker]
        idx_rets = np.log(idx_px.where(idx_px > 0)).diff()
    else:   # offline (or index prices missing): the weighted basket stands in
        wn = w.reindex(have).fillna(0.0)
        idx_rets = (rets.fillna(0.0) @ (wn / max(wn.sum(), 1e-9)))
        idx_px = 1000.0 * np.exp(idx_rets.fillna(0.0).cumsum())   # demo level (builder)
    idx_rv = idx_rets.rolling(TENOR_SESSIONS, min_periods=TENOR_SESSIONS - 4).std() * ann

    wk = w.reindex(have)
    wk = wk / wk.sum()
    real = _corr_series(rv, idx_rv, wk)

    ivm, iv_source = iv_history(tickers + [idx_ticker], sessions)
    if ivm is not None and idx_ticker in ivm.columns:
        # the smoothed *_DF surface lags a day or two on some names (same story
        # as the futures side's staleness guard) — a short bounded ffill keeps a
        # slightly-stale mark in the basket instead of dropping the name
        ivm = ivm.sort_index().ffill(limit=3)
        idx_iv = ivm[idx_ticker]
        ivm = ivm.drop(columns=[idx_ticker])
    else:                                   # demo IVs off the realized series
        ivm, idx_iv = _mock_ivs(rv, real, wk, index_key)
        iv_source = "demo (synthetic) data"
    imp = _corr_series(ivm, idx_iv, wk)

    source = px_source if px_source == iv_source else f"prices: {px_source} · IVs: {iv_source}"
    return dict(n_hist=n_hist, meta=meta, idx_ticker=idx_ticker, n_full=len(w_all),
                coverage=coverage, wk=wk, have=have, sectors=sectors, px=px, idx_px=idx_px,
                rets=rets, rv=rv, idx_rets=idx_rets, idx_rv=idx_rv, ivm=ivm, idx_iv=idx_iv,
                imp=imp, real=real, tickers=tickers, source=source, weight_source=weight_source)


# ── the monitor ───────────────────────────────────────────────────────────────
def compute(universe: dict, index_key: str, hist_label: str = DEFAULT_HISTORY,
            top_n: int = DEFAULT_TOP_N) -> DispersionSet | None:
    """DispersionSet for one index: cut the top-`top_n` basket by weight, run the
    correlation ratio on implied and on realized vols through `hist_label` of
    history, and flag where today's spread sits in that history."""
    d = _inputs(universe, index_key, hist_label, top_n)
    if d is None:
        return None
    n_hist, have, wk, meta = d["n_hist"], d["have"], d["wk"], d["meta"]
    ivm, rv = d["ivm"], d["rv"]

    # trim to the display window, align the spread on common dates
    real, imp = d["real"].iloc[-n_hist:].dropna(), d["imp"].iloc[-n_hist:].dropna()
    idx_rv = d["idx_rv"].iloc[-n_hist:].dropna()
    idx_iv = d["idx_iv"].iloc[-n_hist:].dropna()
    common = imp.index.intersection(real.index)
    if len(common) < TENOR_SESSIONS:
        return None
    spread = (imp - real).loc[common].dropna()
    last = float(spread.iloc[-1])
    pctl = float((spread < last).mean() * 100.0)

    iv_now = ivm.reindex(columns=have).iloc[-1] if len(ivm) else pd.Series(dtype=float)
    rv_now = rv.iloc[-1]
    spot_now = d["px"][have].ffill().iloc[-1]
    basket = pd.DataFrame({
        "ticker": have,
        "name": [meta.get(t, {}).get("name", t.split()[0]) for t in have],
        "sector": [d["sectors"].get(t, "—") for t in have],
        "weight": [float(wk.get(t, np.nan)) for t in have],
        "iv": [float(iv_now.get(t, np.nan)) for t in have],
        "rv": [float(rv_now.get(t, np.nan)) for t in have],
        "spot": [float(spot_now.get(t, np.nan)) for t in have],
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    basket["prem"] = basket["iv"] - basket["rv"]
    dropped = sorted((set(d["tickers"]) - set(have))
                     | {t for t in have if iv_now.get(t, np.nan) != iv_now.get(t, np.nan)})

    return DispersionSet(asof=pd.Timestamp(common.max()), index_key=index_key,
                         source=d["source"], weight_source=d["weight_source"],
                         n_full=d["n_full"], coverage=d["coverage"], basket=basket,
                         imp=imp, real=real, spread=spread, idx_iv=idx_iv, idx_rv=idx_rv,
                         idx_spot=float(d["idx_px"].ffill().iloc[-1]), pctl=pctl,
                         dropped=dropped)


# ── Phase 2: basket builder ───────────────────────────────────────────────────
# ATM straddles at the monitor's own 1M tenor, r=0 Black-Scholes. Contracts use
# the standard listed multipliers: single-name options deliver 100 shares; index
# options per the exchange (Dow exposure trades as DJX = 1/100th of INDU, so its
# effective multiplier per INDU point is 1).
TENOR_YEARS = TENOR_SESSIONS / 252.0
EQ_OPT_MULT = 100.0
INDEX_OPT_MULT = {"SPX Index": 100.0, "NDX Index": 100.0, "INDU Index": 1.0,
                  "SX5E Index": 10.0, "DAX Index": 5.0, "UKX Index": 10.0,
                  "CAC Index": 10.0}
DIRECTIONS = {"classic": "Short index vol / long single-name vol",
              "reverse": "Long index vol / short single-name vol"}
VEGA_SCHEMES = {"cap": "Cap-weighted vega", "equal": "Equal vega per name"}
EARNINGS_WINDOW_DAYS = 30      # calendar days ≈ the option tenor


@dataclass
class DispersionBasket:
    """A sized dispersion structure: one row per single-name straddle plus the
    offsetting index leg, vega-neutral by construction."""
    direction: str                # DIRECTIONS key
    weighting: str                # VEGA_SCHEMES key
    vega: float                   # per leg, currency per vol point
    legs: pd.DataFrame            # name rows: vega/contracts/premium/theta/earnings
    index_leg: dict               # same fields for the index side
    net_theta: float              # currency per trading day (negative = paying decay)
    net_premium: float            # currency (negative = net outlay)
    n_earnings: int               # basket names reporting inside the tenor


def _straddle(spot: float, iv_pts: float, T: float = TENOR_YEARS) -> tuple:
    """(vega per vol point, premium, theta per trading day) for ONE ATM straddle
    on one share / one index unit — Black-Scholes, r=0, K=spot."""
    sig = max(float(iv_pts), 1e-6) / 100.0
    d1 = sig * math.sqrt(T) / 2.0
    phi = math.exp(-d1 * d1 / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    vega = 2.0 * spot * phi * math.sqrt(T) / 100.0        # per vol POINT
    prem = 2.0 * spot * (2.0 * cdf - 1.0)
    theta = -(spot * phi * sig / math.sqrt(T)) / 252.0    # both options, per session
    return float(vega), float(prem), float(theta)


def _earnings_dates(tickers: list) -> pd.Series:
    """ticker -> next expected report date (NaT when unknown), off the Company
    Fundamentals DB — no extra Bloomberg pull."""
    try:
        from src import eqfunda
        wide, _asof, _src = eqfunda.latest_wide(tickers)
        return pd.to_datetime(wide.get("EXPECTED_REPORT_DT"), errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=list(tickers))


def build_basket(ds: DispersionSet, vega_per_leg: float = 1000.0,
                 weighting: str = "cap", direction: str = "classic") -> DispersionBasket | None:
    """Size the structure off the monitor's current marks: split `vega_per_leg`
    across the single names (per `weighting`), offset it 1:1 with index vega, and
    express each leg as ATM straddle contracts with premium / theta / earnings
    flags. None when the basket has no usable marks."""
    bt = ds.basket.dropna(subset=["iv", "spot"])
    bt = bt[bt["weight"] > 0]
    if bt.empty or not (ds.idx_spot == ds.idx_spot) or not len(ds.idx_iv):
        return None
    sgn_names = 1.0 if direction == "classic" else -1.0
    shares = (bt["weight"] / bt["weight"].sum()).to_numpy() if weighting == "cap" \
        else np.full(len(bt), 1.0 / len(bt))

    edates = _earnings_dates(list(bt["ticker"]))
    horizon = ds.asof + pd.Timedelta(days=EARNINGS_WINDOW_DAYS)
    rows = []
    for (_, r), share in zip(bt.iterrows(), shares):
        vega_ct, prem_ct, theta_ct = _straddle(r["spot"], r["iv"])
        vega_ct, prem_ct, theta_ct = (x * EQ_OPT_MULT for x in (vega_ct, prem_ct, theta_ct))
        v = vega_per_leg * share
        n = v / vega_ct if vega_ct > 0 else np.nan
        edt = edates.get(r["ticker"], pd.NaT)
        rows.append({
            "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
            "share": share, "iv": r["iv"], "rv": r["rv"], "spot": r["spot"],
            "vega": sgn_names * v, "contracts": sgn_names * n,
            "premium": sgn_names * n * prem_ct, "theta": sgn_names * n * theta_ct,
            "earnings": edt.strftime("%Y-%m-%d")
                        if pd.notna(edt) and ds.asof < edt <= horizon else "",
        })
    legs = pd.DataFrame(rows)

    idx_iv_now = float(ds.idx_iv.iloc[-1])
    mult = INDEX_OPT_MULT.get(equities.INDICES[ds.index_key][1], 100.0)
    vega_ct, prem_ct, theta_ct = _straddle(ds.idx_spot, idx_iv_now)
    vega_ct, prem_ct, theta_ct = (x * mult for x in (vega_ct, prem_ct, theta_ct))
    n_idx = vega_per_leg / vega_ct if vega_ct > 0 else np.nan
    index_leg = {"name": ds.index_key, "iv": idx_iv_now, "spot": ds.idx_spot, "mult": mult,
                 "vega": -sgn_names * vega_per_leg, "contracts": -sgn_names * n_idx,
                 "premium": -sgn_names * n_idx * prem_ct, "theta": -sgn_names * n_idx * theta_ct}

    return DispersionBasket(
        direction=direction, weighting=weighting, vega=float(vega_per_leg), legs=legs,
        index_leg=index_leg,
        net_theta=float(legs["theta"].sum() + index_leg["theta"]),
        net_premium=float(legs["premium"].sum() + index_leg["premium"]),
        n_earnings=int((legs["earnings"] != "").sum()))


# ── Phase 3: vega-carry backtest ──────────────────────────────────────────────
BT_ROLL = TENOR_SESSIONS       # hold ≈ the option tenor, rolled back-to-back


@dataclass
class DispersionBT:
    """Rolled dispersion carry: each trade sells (buys) the index straddle and
    buys (sells) the vega-matched single-name basket, both booked with the
    vega-carry approximation P&L ≈ vega × (realized − implied at entry)."""
    index_key: str
    weighting: str
    roll: int
    trades: pd.DataFrame     # entry/exit, corr marks, per-leg + net P&L (vol pts per unit vega)
    curve: pd.Series         # cumulative net P&L by exit date
    stats: dict
    source: str


def backtest(universe: dict, index_key: str, hist_label: str = "2 years",
             top_n: int = DEFAULT_TOP_N, weighting: str = "cap",
             roll: int = BT_ROLL) -> DispersionBT | None:
    """Roll the CLASSIC structure (long single-name vol / short index vol)
    through `hist_label` of history: enter every `roll` sessions, hold to the
    next roll, book each leg as vega × (forward realized − entry implied). P&L is
    in vol points per unit of vega per leg (multiply by the leg vega for
    currency); the reverse structure is the exact mirror."""
    d = _inputs(universe, index_key, hist_label, top_n)
    if d is None:
        return None
    rets, idx_rets, wk = d["rets"], d["idx_rets"], d["wk"]
    dates = rets.index
    ivm = d["ivm"].reindex(dates).ffill(limit=3)
    idx_iv = d["idx_iv"].reindex(dates).ffill(limit=3)
    imp_al = d["imp"].reindex(dates)
    spread_al = (d["imp"] - d["real"]).reindex(dates)
    ann = np.sqrt(252.0) * 100.0

    trades = []
    t = TENOR_SESSIONS + 5                    # rv warm-up before the first entry
    while t + roll < len(dates):
        iv_e = ivm.iloc[t]
        iv_idx = float(idx_iv.iloc[t]) if idx_iv.iloc[t] == idx_iv.iloc[t] else np.nan
        fwd = rets.iloc[t + 1:t + roll + 1]
        enough = fwd.notna().sum() >= max(5, roll - 4)
        valid = [c for c in rets.columns
                 if iv_e.get(c, np.nan) == iv_e.get(c, np.nan) and bool(enough.get(c, False))
                 and wk.get(c, 0) > 0]
        if len(valid) >= 5 and iv_idx == iv_idx:
            shares = (wk[valid] / wk[valid].sum()).to_numpy() if weighting == "cap" \
                else np.full(len(valid), 1.0 / len(valid))
            rvf = (fwd[valid].std() * ann).to_numpy()
            ive = iv_e[valid].to_numpy(dtype=float)
            rvf_idx = float(idx_rets.iloc[t + 1:t + roll + 1].std() * ann)
            pnl_names = float((shares * (rvf - ive)).sum())
            pnl_index = float(iv_idx - rvf_idx)
            # forward realized correlation over the hold, same ratio as the monitor
            s1, s2 = float((shares * rvf).sum()), float((shares ** 2 * rvf ** 2).sum())
            rho_f = (rvf_idx ** 2 - s2) / (s1 ** 2 - s2) if (s1 ** 2 - s2) > 1e-12 else np.nan
            prior = spread_al.iloc[:t].dropna()
            se = spread_al.iloc[t]
            pctl = float((prior < se).mean() * 100.0) \
                if len(prior) >= 63 and se == se else np.nan
            trades.append({
                "entry": dates[t], "exit": dates[t + roll], "n_names": len(valid),
                "imp_corr": float(imp_al.iloc[t]) if imp_al.iloc[t] == imp_al.iloc[t] else np.nan,
                "fwd_real_corr": rho_f, "spread": float(se) if se == se else np.nan,
                "pctl": pctl, "pnl_names": pnl_names, "pnl_index": pnl_index,
                "pnl": pnl_names + pnl_index,
            })
        t += roll
    if not trades:
        return None
    tr = pd.DataFrame(trades)
    curve = tr.set_index("exit")["pnl"].cumsum()
    wins = int((tr["pnl"] > 0).sum())
    stats = {
        "n": len(tr), "total": float(tr["pnl"].sum()), "avg": float(tr["pnl"].mean()),
        "hit": 100.0 * wins / len(tr), "best": float(tr["pnl"].max()),
        "worst": float(tr["pnl"].min()),
        "pctl_ic": float(tr[["pctl", "pnl"]].dropna().corr().iloc[0, 1])
        if tr["pctl"].notna().sum() >= 8 else np.nan,
    }
    return DispersionBT(index_key=index_key, weighting=weighting, roll=roll,
                        trades=tr, curve=curve, stats=stats, source=d["source"])
