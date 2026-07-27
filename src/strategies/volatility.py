"""Implied-vs-realized volatility monitor — the Volatility Report.

For every market with listed options we compare 1-month at-the-money IMPLIED
vol against ~1-month (21 trading day) REALIZED vol. The headline number is the
**z-score of the implied−realized spread over the past year** — i.e. how far
today's vol-risk premium has stretched from its own norm. That is the "spread
has reached a high number" signal:

  • z >= +1.5  -> implied unusually rich vs realized -> SELL vol (sell premium)
  • z <= -1.5  -> implied unusually cheap vs realized -> BUY  vol

Both legs come from datafeed (mock today, Bloomberg once live): realized is
computed from settlement prices; implied is the constant-maturity 1M ATM surface
(FX from the OTC pair vol). `compute_table()` returns the full rich cross-section
(IV, RV, spread, z, percentile) and is also what the visual client report reads;
`find_opportunities()` maps it onto the shared strategy schema for the dashboard.

Note for fixed income: STIR/bond option vol may be quoted normal (bp) not
lognormal/price vol — confirm the convention so the implied leg matches this
price-return realized leg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datafeed import (get_realized_vol_history, get_implied_vol_history,
                        get_skew_components, get_history, stale_iv_reasons, warn_stale,
                        price_decimals)
from ..universe import INSTRUMENTS, name, asset, region
from .base import frame

STRATEGY = "Volatility"
STAT_WINDOW = 252       # ~1 year of obs for the z-score / percentile
Z_FLAG = 1.5            # |z| beyond this flags an opportunity (matches mean_reversion)
_SQRT252 = np.sqrt(252.0)   # de-annualiser for the daily 1σ move shown next to each vol

# STIR futures are pinned near 100, so their *price* realized vol is ~0 and can't
# be compared to the option-surface implied (which isn't price vol). Rate vol
# needs a dedicated normal/basis-point treatment — excluded here until then.
EXCLUDE_ASSETS = {"STIRs"}

# Non-US index FUTURES whose option surface returns nothing on the Terminal stay
# excluded; the CASH indices (SX5E/SX7E/DAX/UKX added 2026-06-04; CAC/SMI/NKY/KOSPI2
# added 2026-07-21) DO have surface vol, so they flow into the vol reports instead.
# XPA (ASX SPI) re-included 2026-07-21 — its FUTURES surface verified live (≈11.8 vol
# vs 8.0 realized on the Terminal), unlike the rest of this list.
EXCLUDE_INDEX = {"VGA Index", "CAA Index", "Z A Index", "GXA Index", "SMA Index",
                 "NKA Index", "KMA Index"}

# Skew uses fixed 90/110% moneyness wings for listed products (FX uses the OTC
# 25-delta risk reversal — it has no moneyness surface). When ATM price vol is low
# those strikes sit tens of σ OTM where the surface is extrapolated garbage, so
# BONDS are dropped from SKEW (their ATM vol page is fine — that's 100% moneyness)
# and STIRs too (via EXCLUDE_ASSETS). Any |skew| beyond the cap is treated as bad
# surface data.
SKEW_EXCLUDE_EXTRA = frozenset({"Bonds"})
SKEW_CAP = 2.5


def _excluded(t: str, extra_assets=frozenset()) -> bool:
    a = asset(t)
    if a in EXCLUDE_ASSETS or a in extra_assets:
        return True
    if a == "Indices" and t in EXCLUDE_INDEX:        # dead non-US index futures
        return True
    return False

# Rich per-market tables cached here for the visual client report (src/volreport.py).
DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "volatility.parquet"
SKEW_DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "skew.parquet"
HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "volatility_history.parquet"
SKEW_HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "skew_history.parquet"

DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "iv", "rv",
                  "spread", "z", "pctl", "signal", "direction",
                  "iv_sd", "rv_sd", "px_dec",
                  "src", "vs_bbg",   # implied source (own/bbg/otc) + own-vs-surface gap
                  "rv5", "rv_state"]  # 5d realized + decay/heat marker state
SKEW_DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "put", "call", "atm",
                       "skew", "z", "pctl", "signal", "direction"]


def _ord(n: float) -> str:
    """English ordinal: 1->1st, 2->2nd, 3->3rd, 11->11th, 92->92nd."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _z_and_pctl(series: pd.Series):
    """Current value, its z-score and percentile over the trailing STAT_WINDOW."""
    s = series.dropna()
    if s.empty:
        return float("nan"), float("nan"), float("nan")
    now = float(s.iloc[-1])
    win = s.iloc[-STAT_WINDOW:]
    sd = float(win.std())
    z = (now - float(win.mean())) / sd if sd > 1e-9 else float("nan")
    pctl = float((win <= now).mean() * 100.0) if len(win) >= 20 else float("nan")
    return now, z, pctl


def _last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _persist_history(iv, rv, px, window: int = STAT_WINDOW):
    """Cache ~1y of the implied−realized spread and the underlying price per market
    (long format) for the report's spread-vs-underlying time-series chart. The iv/rv
    legs are stored too so the report's chart captions can say which leg drove a move."""
    frames = []
    for t in INSTRUMENTS:
        if _excluded(t) or not all(t in d.columns for d in (iv, rv, px)):
            continue
        h = pd.DataFrame({"spread": iv[t] - rv[t], "price": px[t],
                          "iv": iv[t], "rv": rv[t]}).dropna().iloc[-window:]
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "spread", "price", "iv", "rv"]
        h["ticker"] = t
        frames.append(h[["date", "ticker", "spread", "price", "iv", "rv"]])
    if frames:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(HISTORY_FILE, index=False)
        except Exception:
            pass


def _persist_skew_history(put, call, atm, px, window: int = STAT_WINDOW):
    """Cache ~1y of the (90% put − 110% call)/ATM skew and the underlying price per
    market (long format) for the skew report's skew-vs-underlying time-series chart.
    The wing legs are stored too so the report's captions can say which wing moved."""
    frames = []
    for t in INSTRUMENTS:
        if _excluded(t, SKEW_EXCLUDE_EXTRA) or not all(t in d.columns for d in (put, call, atm, px)):
            continue
        skew = (put[t] - call[t]) / atm[t].replace(0, np.nan)
        h = pd.DataFrame({"skew": skew, "price": px[t],
                          "put": put[t], "call": call[t], "atm": atm[t]}).dropna().iloc[-window:]
        if h.empty:
            continue
        h = h.reset_index()
        h.columns = ["date", "skew", "price", "put", "call", "atm"]
        h["ticker"] = t
        frames.append(h[["date", "ticker", "skew", "price", "put", "call", "atm"]])
    if frames:
        try:
            SKEW_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(SKEW_HISTORY_FILE, index=False)
        except Exception:
            pass


OWN30_FILE = Path(__file__).resolve().parents[2] / "data" / "snapshot" / "own30_history.parquet"


def _own_overlay(iv: pd.DataFrame):
    """Run the book on the DESK'S OWN curve. Overlay our settle-dated constant-30-day
    values (src/owncurve.py — ATM option settlements inverted through Black-76, fitted
    curve per product) onto the vendor implied frame: wherever our history covers a
    date, that date runs on OUR number; vendor values remain only where we have none
    (FX pair vols, cash indices, the few names without invertible settles — and any
    day our own build missed, so a stalled chain degrades to the surface, not to a
    hole). Returns the blended frame + per-ticker {last own date, ours−surface gap}."""
    meta: dict = {}
    try:
        h = pd.read_parquet(OWN30_FILE)
    except Exception:
        return iv, meta
    if h.empty:
        return iv, meta
    h = h.assign(date=pd.to_datetime(h["date"]))
    piv = h.pivot_table(index="date", columns="ticker", values="ours30", aggfunc="last")
    out = iv.reindex(iv.index.union(piv.index)).sort_index()
    for t in piv.columns:
        s = piv[t].dropna()
        if s.empty:
            continue
        if t not in out.columns:
            out[t] = np.nan
        out.loc[s.index, t] = s
        sub = h[h["ticker"] == t].dropna(subset=["ours30", "bbg30"])
        vs = float(sub["ours30"].iloc[-1] - sub["bbg30"].iloc[-1]) if len(sub) else float("nan")
        meta[t] = {"last": s.index[-1], "vs_bbg": vs}
    return out, meta


def compute_table() -> pd.DataFrame:
    """Full cross-section: one row per market with IV, RV, spread, z and percentile,
    sorted by |z| (most stretched first). The single source of truth for both the
    dashboard rows and the visual report. The implied leg is our own curve wherever
    it exists (the whole listed-futures book); the vendor surface only backstops."""
    tickers = list(INSTRUMENTS)
    rv = get_realized_vol_history(tickers)
    rv5 = get_realized_vol_history(tickers, window=5)    # event-decay / heating diagnostic
    iv = get_implied_vol_history(tickers)
    px = get_history(tickers)                            # underlying settlement — SD + history
    iv, own_meta = _own_overlay(iv)                      # the book runs on OUR curve
    try:
        _persist_history(iv, rv, px)                     # 1y spread + underlying for the report
    except Exception:
        pass

    stale = stale_iv_reasons(iv)                     # surfaces that aren't a live quote
    recs, skipped_stale, own_stale = [], [], []
    for t in tickers:
        if _excluded(t):
            continue
        if t not in rv.columns or t not in iv.columns:
            continue
        if t in stale:                               # frozen/dead surface — don't score it
            skipped_stale.append(t)
            continue
        rv_s, iv_s = rv[t].dropna(), iv[t].dropna()
        spread_s = (iv[t] - rv[t]).dropna()
        if rv_s.empty or iv_s.empty or spread_s.empty:
            continue
        iv_now, rv_now = float(iv_s.iloc[-1]), float(rv_s.iloc[-1])
        if not (np.isfinite(iv_now) and np.isfinite(rv_now)):
            continue
        spread, z, pctl = _z_and_pctl(spread_s)

        # Directional flags require the z AND the spread's sign to agree (same rule as
        # the STIR section): a stretched z on a negative spread is a narrowing discount,
        # not rich vol — and vice versa. Sign-mismatched extremes stay neutral.
        if np.isfinite(z) and z >= Z_FLAG:
            if spread > 0:
                signal, direction = "Rich — sell vol", -1
            else:
                signal, direction = "Discount narrowing", 0
        elif np.isfinite(z) and z <= -Z_FLAG:
            if spread < 0:
                signal, direction = "Cheap — buy vol", 1
            else:
                signal, direction = "Premium compressing", 0
        else:
            signal, direction = "—", 0

        # Daily 1σ move in the contract's own price units — vol ÷ √252 × price — shown
        # in brackets next to each vol, to the underlying's native number of decimals.
        ps = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        S = float(ps.iloc[-1]) if not ps.empty else float("nan")
        dec = price_decimals(ps, asset(t)) if not ps.empty else 1
        iv_sd = round(S * iv_now / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan
        rv_sd = round(S * rv_now / 100.0 / _SQRT252, dec) if np.isfinite(S) else np.nan

        # 5-day realized companion — is the market still DELIVERING the 1-month
        # number, or is an old event rolling out of the window? Two consecutive
        # sessions beyond the band before a marker (a 5-obs vol estimate is noisy);
        # decay additionally requires the 1M leg elevated vs its own year, else the
        # ratio is just two small numbers arguing.
        rv5_s = rv5[t].dropna() if t in rv5.columns else pd.Series(dtype=float)
        rv5_now = float(rv5_s.iloc[-1]) if len(rv5_s) else float("nan")
        rv_state = ""
        pair = pd.concat([rv5_s, rv_s], axis=1, keys=["s", "l"]).dropna()
        if len(pair) >= 2:
            ratio = (pair["s"] / pair["l"]).iloc[-2:]
            if (ratio < 0.5).all() and rv_now > float(rv_s.iloc[-252:].median()):
                rv_state = "decay"
            elif (ratio > 2.0).all():
                rv_state = "heat"

        # which construction produced today's implied: our own curve, the vendor
        # surface (cash indices + own-curve gaps), or OTC pair vols (FX).
        # Freshness is SETTLE-to-SETTLE: our curve rows are stamped by settle date,
        # but a morning vendor surface can carry a partial row dated TODAY — comparing
        # against that flagged the whole book stale (vendor fallback) on any such
        # morning. A today-dated vendor row is ignored for the comparison.
        meta = own_meta.get(t)
        _iv_last_settle = iv_s.index[-1]
        if (len(iv_s.index) > 1
                and pd.Timestamp(_iv_last_settle).normalize()
                >= pd.Timestamp.now().normalize()):
            _iv_last_settle = iv_s.index[-2]
        if asset(t) == "FX":
            src = "otc"
        elif meta is not None and meta["last"] >= _iv_last_settle:
            src = "own"
        else:
            src = "bbg"
            if meta is not None:
                own_stale.append(name(t))
        vs_bbg = meta["vs_bbg"] if (src == "own" and meta is not None) else float("nan")

        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "iv": round(iv_now, 1), "rv": round(rv_now, 1), "spread": round(spread, 1),
            "z": round(z, 2) if np.isfinite(z) else np.nan,
            "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
            "signal": signal, "direction": direction,
            "iv_sd": iv_sd, "rv_sd": rv_sd, "px_dec": dec,
            "src": src, "vs_bbg": round(vs_bbg, 1) if np.isfinite(vs_bbg) else np.nan,
            "rv5": round(rv5_now, 1) if np.isfinite(rv5_now) else np.nan,
            "rv_state": rv_state,
        })

    warn_stale("Volatility", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
    warn_stale("Own vol curve", [f"{m} (fell back to vendor surface)" for m in own_stale])
    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


STIR_DETAIL_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "stirvol.parquet"
STIR_HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "signals" / "stirvol_history.parquet"
STIR_DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "rate", "iv", "rv",
                       "spread", "z", "pctl", "signal", "direction", "iv_bp", "rv_bp", "vs_bbg"]

# Contracts whose options barely trade (desk call 2026-07-21): the implied marks are
# illiquid quotes (~39 vol prints refreshed rarely), not vol anyone can deal in — so
# the section shows only the liquid three-month complex (SR3 / SONIA / Euribor).
# Delete a ticker from this set to bring it back if its options market picks up.
STIR_EXCLUDE = {"SERA Comdty", "FFA Comdty"}   # 1M SOFR, 30-Day Fed Funds


def compute_stir_table() -> pd.DataFrame:
    """Money-market rate-vol cross-section — OUR OWN constant-90-day curve.

    The implied leg is the desk's own construction (src/stircurve.py, 2026-07-22):
    listed quarterly option settlements inverted through Black-76 on the implied
    rate, ATM call/put mid per expiry, interpolated in total variance to a constant
    90 days — identical for all three products (Bloomberg's smoothed surface is
    absent for Euribor and opaque elsewhere; its 3M point is carried alongside as a
    cross-check where published). The realized leg is the MATCHING 63-trading-day
    lognormal vol of the implied rate (100−price); 1-day 1σ moves are in BASIS
    POINTS. Implied sits structurally above quiet day-to-day realized (options
    price the meetings ahead), so each market is judged by the z of its spread
    against ITS OWN trailing year, never against the price-vol book."""
    try:
        from ..stircurve import load_history
        hist = load_history()
    except Exception:
        hist = None
    if hist is None or hist.empty:
        return pd.DataFrame(columns=STIR_DETAIL_COLUMNS)
    tickers = [t for t in INSTRUMENTS if asset(t) == "STIRs" and t not in STIR_EXCLUDE
               and (hist["ticker"] == t).any()]
    if not tickers:
        return pd.DataFrame(columns=STIR_DETAIL_COLUMNS)
    px = get_history(tickers)
    recs, skipped, hist_frames = [], [], []
    for t in tickers:
        if t not in px.columns:
            continue
        g = hist[hist["ticker"] == t].set_index("date").sort_index()
        ours = g["ours"].dropna()
        if ours.empty:
            continue
        rate = (100.0 - px[t]).clip(lower=1e-6)
        if ours.index[-1] < rate.dropna().index[-1] - pd.Timedelta(days=8):
            skipped.append(t)                            # our curve has gone stale — don't score
            continue
        rv_s = (np.log(rate).diff().rolling(63).std() * np.sqrt(252) * 100.0).dropna()
        spread_s = (ours - rv_s).dropna()
        if rv_s.empty or spread_s.empty:
            continue
        iv_now, rv_now = float(ours.iloc[-1]), float(rv_s.iloc[-1])
        rate_now = float(rate.dropna().iloc[-1])
        spread, z, pctl = _z_and_pctl(spread_s)
        bbg_s = g["bbg"].dropna() if "bbg" in g.columns else pd.Series(dtype=float)
        vs_bbg = round(iv_now - float(bbg_s.iloc[-1]), 1) if len(bbg_s) else np.nan

        # A DIRECTIONAL flag needs the z and the spread's SIGN to agree: STIR spreads sit
        # structurally negative for long stretches, so a high z on a still-negative spread
        # means "the discount is narrowing", NOT that implied is rich — calling that a
        # sell would advise selling vol below what the market is delivering (user catch
        # 2026-07-22: SOFR z +1.54 / 98th %ile with implied 17.3 vs 19.7 realized).
        if np.isfinite(z) and z >= Z_FLAG:
            if spread > 0:
                signal, direction = "Rich — sell vol", -1
            else:
                signal, direction = "Discount narrowing", 0
        elif np.isfinite(z) and z <= -Z_FLAG:
            if spread < 0:
                signal, direction = "Cheap — buy vol", 1
            else:
                signal, direction = "Premium compressing", 0
        else:
            signal, direction = "—", 0

        bp = lambda v: rate_now * v / 100.0 / _SQRT252 * 100.0   # 1-day 1σ, basis points
        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "rate": round(rate_now, 2),
            "iv": round(iv_now, 1), "rv": round(rv_now, 1), "spread": round(spread, 1),
            "z": round(z, 2) if np.isfinite(z) else np.nan,
            "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
            "signal": signal, "direction": direction,
            "iv_bp": round(bp(iv_now), 1), "rv_bp": round(bp(rv_now), 1),
            "vs_bbg": vs_bbg,
        })
        # ~1y of the spread + the implied rate for the report's flagged-STIR panels.
        h = pd.DataFrame({"spread": spread_s, "rate": rate}).dropna().iloc[-STAT_WINDOW:]
        if not h.empty:
            h = h.reset_index()
            h.columns = ["date", "spread", "rate"]
            h["ticker"] = t
            hist_frames.append(h[["date", "ticker", "spread", "rate"]])
    warn_stale("Volatility (STIR)", [f"{name(t)} (own curve stale)" for t in skipped])
    if hist_frames:
        try:
            STIR_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(hist_frames, ignore_index=True).to_parquet(STIR_HISTORY_FILE, index=False)
        except Exception:
            pass
    df = pd.DataFrame(recs, columns=STIR_DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


def compute_skew_table() -> pd.DataFrame:
    """Cross-section of the normalized skew (90% put − 110% call)/ATM, z-scored over
    the same 252-day window (positive = puts richer than calls). Listed markets use
    the 90/110% moneyness wings off the surface, FX the OTC 25-delta risk reversal.
    Feeds the report's skew page. z >= +1.5 -> skew rich (sell); z <= -1.5 -> cheap
    (buy)."""
    tickers = list(INSTRUMENTS)
    comp = get_skew_components(tickers)
    put, call, atm = comp["put"], comp["call"], comp["atm"]
    try:
        _persist_skew_history(put, call, atm, get_history(tickers))   # 1y skew + underlying
    except Exception:
        pass

    stale = stale_iv_reasons(atm)                    # skew rides on the ATM surface
    recs, skipped_stale = [], []
    for t in tickers:
        if _excluded(t, SKEW_EXCLUDE_EXTRA):
            continue
        if not all(t in d.columns for d in (put, call, atm)):
            continue
        if t in stale:                               # frozen/dead surface — don't score it
            skipped_stale.append(t)
            continue
        skew_s = ((put[t] - call[t]) / atm[t].replace(0, np.nan)).dropna()
        if skew_s.empty:
            continue
        put_now, call_now, atm_now = _last(put[t]), _last(call[t]), _last(atm[t])
        if not np.isfinite(atm_now) or atm_now == 0:
            continue
        skew_now, z, pctl = _z_and_pctl(skew_s)
        if not np.isfinite(skew_now) or abs(skew_now) > SKEW_CAP:
            continue   # unreliable surface (e.g. low-vol wings far OTM)

        if np.isfinite(z) and z >= Z_FLAG:
            signal, direction = "Rich — sell skew", -1
        elif np.isfinite(z) and z <= -Z_FLAG:
            signal, direction = "Cheap — buy skew", 1
        else:
            signal, direction = "—", 0

        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "put": round(put_now, 1), "call": round(call_now, 1), "atm": round(atm_now, 1),
            "skew": round(skew_now, 3),
            "z": round(z, 2) if np.isfinite(z) else np.nan,
            "pctl": round(pctl) if np.isfinite(pctl) else np.nan,
            "signal": signal, "direction": direction,
        })

    warn_stale("Skew Volatility", [f"{name(t)} ({stale[t]})" for t in skipped_stale])
    df = pd.DataFrame(recs, columns=SKEW_DETAIL_COLUMNS)
    if not df.empty:
        order = df["z"].abs().sort_values(ascending=False, na_position="last").index
        df = df.reindex(order).reset_index(drop=True)
    return df


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    tbl = compute_table()

    # Cache the rich table for the visual report (compute-once-daily, like the rest).
    # (The skew table is computed + cached by the separate `skew` strategy.)
    try:
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(DETAIL_FILE, index=False)
    except Exception:
        pass

    # Realized-vol co-movement matrix for the page/report relative-value peer finder.
    try:
        from ..volcorr import build_and_cache
        build_and_cache(list(INSTRUMENTS))
    except Exception:
        pass

    # Money-market rate-vol cross-section (own convention, own section in the report).
    try:
        STIR_DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        compute_stir_table().to_parquet(STIR_DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        ctx = f"IV {r.iv:.1f} / RV {r.rv:.1f}"
        if pd.notna(r.pctl):
            ctx += f" · spread {_ord(r.pctl)} %ile (1y)"
        if r.rv:
            ctx += f" · IV/RV {r.iv / r.rv:.2f}"
        rows.append({
            "strategy": STRATEGY,
            "market": f"{r.market} · {r.asset}",
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.z) if pd.notna(r.z) else 0.0,
            "metric_label": "spread z (1y)",
            "level": float(r.spread),          # IV−RV spread, vol points
            "context": ctx,
        })
    return frame(rows)   # already sorted by |z| (most stretched first)
