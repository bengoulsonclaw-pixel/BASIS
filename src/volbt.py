"""Vol Swap Backtester — engine.

Backtests a two-legged listed-vol trade: BUY vol on one product, SELL vol on the
other, both via ATM straddles on a common option expiry, delta-hedged with futures
at every settlement. The daily option marks are reconstructed with Black-76 from
the same surfaces the vol/skew/term reports already pull (datafeed seams, so mock /
snapshot / live all work):

  • settlement prices        — get_history
  • ATM vol at the trade's actual days-to-expiry — get_term_structure pillars
    (1M/3M/6M/12M), interpolated linearly in total variance
  • fixed-strike smile tilt  — get_skew_components 90%/ATM/110% (1M), quadratic in
    log-moneyness, decayed ~1/sqrt(tenor) beyond 1M

Leg RATIO (sell straddles per buy straddle) at entry and at every re-strike:
  rn_gamma — risk-normalised gamma (Γ·F²·σ²·mult): each leg carries the same
             EXPECTED daily gamma earn at its own vol (default). Same-expiry
             identities: theta-flat (zero net carry) and vega·σ-flat (neutral
             to PROPORTIONAL vol moves rather than parallel ones).
  gamma    — equal dollar gamma (Γ·F²·mult): the realized-vol engines net out;
             the vol-swap-like weighting.
  vega     — equal dollar vega: P&L ≈ the implied spread mark-to-market.
  beta_vega— sell vega = β × buy vega, β from regressing daily 1M ATM IV changes
             (buy on sell) over the year before entry: vol-market-neutral.
  premium  — equal premium outlay (kept for comparison; exposures aren't clean).

RE-STRIKE modes (both legs together, as a package, re-ratioed on the day's greeks):
  never | daily | threshold (spot drifted ≥ X × the implied daily move on either leg)

No Streamlit here — the page and the PDF CLI both drive this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .datafeed import (get_history, get_implied_vol_history, get_skew_components,
                       get_term_structure, stale_iv_reasons, MODE)
from .universe import name as product_name

SQRT252 = math.sqrt(252.0)

# ── contract point values ($ per 1.0 price point per lot) ───────────────────
# Needed to turn greeks into dollars (and therefore into leg ratios). Seeded for
# the majors; the page shows the two values in use and lets the desk overwrite
# them, so an unknown ticker is a visible warning, never a silent bad ratio.
# Non-USD contracts are carried at 1 local-currency point = 1 "$" (FX-conversion
# free) — fine for ratios and for P&L read as local currency.
#
# UNITS: "1.0 price point" means one point OF THE FEED'S OWN QUOTE — and Bloomberg
# quotes several contracts in CENTS (HG/XB/HO in USd, most FX futures in cents per
# unit, JY in cents per 100 yen), where the value per point is contract-size ÷ 100
# (÷10,000 for JY), NOT the contract size. Verified against FUT_VAL_PT live on
# 2026-08-08 (every entry below matches Bloomberg's own value-of-1-point; the old
# table carried contract SIZES for the cent-quoted names — 100× off).
POINT_VALUE = {
    "ESA Index": 50.0, "NQA Index": 20.0, "RTYA Index": 50.0, "DMA Index": 5.0,
    "VGA Index": 10.0, "GXA Index": 25.0, "CAA Index": 50.0, "Z A Index": 10.0,
    "SMA Index": 10.0, "NKA Index": 1000.0, "XPA Index": 25.0, "KMA Index": 250000.0,
    "SX5E Index": 10.0, "DAX Index": 25.0, "UKX Index": 10.0, "SX7E Index": 50.0,
    "USA Comdty": 1000.0, "WNA Comdty": 1000.0, "UXYA Comdty": 1000.0,
    "TYA Comdty": 1000.0, "FVA Comdty": 1000.0, "TUA Comdty": 2000.0,
    "RXA Comdty": 1000.0, "UBA Comdty": 1000.0, "OEA Comdty": 1000.0,
    "DUA Comdty": 1000.0, "OATA Comdty": 1000.0, "G A Comdty": 1000.0,
    "SFRA Comdty": 2500.0, "SERA Comdty": 4167.0, "FFA Comdty": 4167.0,
    "ERA Comdty": 2500.0, "SFIA Comdty": 2500.0, "TKYA Comdty": 2500.0,
    "CLA Comdty": 1000.0, "COA Comdty": 1000.0, "XBA Comdty": 420.0,
    "HOA Comdty": 420.0, "NGA Comdty": 10000.0, "QSA Comdty": 100.0,
    "GCA Comdty": 100.0, "SIA Comdty": 5000.0, "HGA Comdty": 250.0,
    "PLA Comdty": 50.0, "PAA Comdty": 100.0,
    "C A Comdty": 50.0, "S A Comdty": 50.0, "W A Comdty": 50.0, "KWA Comdty": 50.0,
    "SMA Comdty": 100.0, "BOA Comdty": 600.0, "RRA Comdty": 2000.0,
    "KCA Comdty": 375.0, "SBA Comdty": 1120.0, "CTA Comdty": 500.0,
    "CCA Comdty": 10.0, "JOA Comdty": 150.0, "DFA Comdty": 10.0,
    "LCA Comdty": 400.0, "LHA Comdty": 400.0, "FCA Comdty": 500.0,
    "ECA Curncy": 125000.0, "BPA Curncy": 625.0, "JYA Curncy": 1250.0,
    "SFA Curncy": 1250.0, "CDA Curncy": 1000.0, "ADA Curncy": 1000.0,
    "NVA Curncy": 1000.0, "PEA Curncy": 5000.0,
}

# ── contract currencies (non-USD only; anything absent = USD) ───────────────
# A mixed-currency pair (e.g. buy ES / sell SX5E) needs the legs on one money
# before greeks can be ratioed or P&L netted. The engine converts non-USD point
# values to USD at the ENTRY-date rate, frozen for the run — the daily-margined
# / FX-swept convention — using the FX futures already in the universe.
CCY = {
    "VGA Index": "EUR", "SX5E Index": "EUR", "SX7E Index": "EUR", "DAX Index": "EUR",
    "GXA Index": "EUR", "CAA Index": "EUR", "CAC Index": "EUR",
    "SMA Index": "CHF", "SMI Index": "CHF",
    "Z A Index": "GBP", "UKX Index": "GBP",
    "NKA Index": "JPY", "NKY Index": "JPY",
    "KMA Index": "KRW", "KOSPI2 Index": "KRW", "XPA Index": "AUD",
    "RXA Comdty": "EUR", "UBA Comdty": "EUR", "OEA Comdty": "EUR",
    "DUA Comdty": "EUR", "OATA Comdty": "EUR", "ERA Comdty": "EUR",
    # TKYA = ICE 3M ESTR (Tokyo-looking Bloomberg root, but a EUR contract).
    "TKYA Comdty": "EUR",
    "G A Comdty": "GBP", "SFIA Comdty": "GBP",
}
# USD per 1 unit of ccy — CME FX futures generics (all quoted American terms).
FX_USD = {"EUR": "ECA Curncy", "GBP": "BPA Curncy", "JPY": "JYA Curncy",
          "CHF": "SFA Curncy", "AUD": "ADA Curncy", "CAD": "CDA Curncy"}
# Bloomberg quote convention of each generic: divide its price by this to get USD
# per 1 unit of the currency. ECA quotes dollars (1.1581); BPA/SFA/ADA/CDA quote
# CENTS (134.99 = $1.3499/GBP); JYA quotes cents per 100 yen (63.65 = $0.006365/¥).
# Verified via QUOTE_UNITS live 2026-08-08 — using the raw price as USD-per-unit
# made every GBP/JPY/CHF leg's USD conversion 100–10,000× too large.
FX_QUOTE_DIV = {"ECA Curncy": 1.0, "BPA Curncy": 100.0, "JYA Curncy": 10000.0,
                "SFA Curncy": 100.0, "ADA Curncy": 100.0, "CDA Curncy": 100.0}


def fx_usd_rate(src: str, px: float) -> float:
    """USD per 1 currency unit from an FX_USD generic's quoted price."""
    return float(px) / FX_QUOTE_DIV.get(src, 1.0)


def currency(ticker: str) -> str:
    return CCY.get(ticker, "USD")


WEIGHTINGS = ["rn_gamma", "gamma", "vega", "beta_vega", "premium"]
RESTRIKES = ["never", "daily", "threshold"]


def point_value(ticker: str) -> float:
    return float(POINT_VALUE.get(ticker, 0.0))


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def quarterly_expiries(after: date, n: int = 8) -> list[date]:
    """The next n quarterly (Mar/Jun/Sep/Dec) third Fridays strictly after `after` —
    the standard index/most-liquid option expiries offered as defaults."""
    out, y, m = [], after.year, 3 * ((after.month - 1) // 3 + 1)
    while len(out) < n:
        if m > 12:
            y, m = y + 1, 3
        tf = third_friday(y, m)
        if tf > after:
            out.append(tf)
        m += 3
    return out


# ── Black-76 straddle (r = 0: futures-style margining; discounting is noise
#    next to the vol spread this measures) ───────────────────────────────────
@dataclass
class Greeks:
    value: float   # straddle premium, price points per 1 straddle
    delta: float   # futures per straddle
    gamma: float   # d delta / d F
    vega: float    # price points per 1 VOL POINT
    theta: float   # price points per YEAR (negative)
    call: float = np.nan   # the straddle's call half, price points (blotter prices)
    put: float = np.nan    # ... and the put half; value = call + put


def straddle_greeks(F: float, K: float, vol_pts: float, tau_years: float) -> Greeks:
    sigma = max(float(vol_pts), 0.01) / 100.0
    tau = max(float(tau_years), 1.0 / 365.0 / 24.0)
    st = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    d2 = d1 - st
    N, phi = (lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))), \
             (lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))
    call = F * N(d1) - K * N(d2)
    put = call - (F - K)                              # parity (r=0); equal when ATM
    return Greeks(value=call + put,
                  delta=2.0 * N(d1) - 1.0,
                  gamma=2.0 * phi(d1) / (F * st),
                  vega=2.0 * F * phi(d1) * math.sqrt(tau) / 100.0,
                  theta=-F * phi(d1) * sigma / math.sqrt(tau),
                  call=call, put=put)


# ── per-ticker vol lookup: term-structure pillars + 1M smile ────────────────
PILLAR_DAYS = {"1M": 30.0, "3M": 91.0, "6M": 182.0, "12M": 365.0}
SMILE_X = (math.log(0.90), math.log(1.10))   # the two wing log-moneyness points


class VolSurface:
    """ATM vol at any days-to-expiry (variance-linear between the report's tenor
    pillars, flat beyond the ends) plus a fixed-strike smile tilt from the 1M
    90/ATM/110 wings (quadratic in ln(K/F), decayed by sqrt(30/tenor) past 1M,
    moneyness clamped to ±20%). Series are aligned to the settlement calendar
    and carried forward so a late surface never drops a backtest day."""

    def __init__(self, ticker: str, term: dict, skew: dict, index: pd.DatetimeIndex):
        self.ticker = ticker
        self.pillars = {}
        for lab, days in PILLAR_DAYS.items():
            fr = term.get(lab)
            if fr is not None and ticker in getattr(fr, "columns", []):
                s = fr[ticker].reindex(index).ffill()
                if s.notna().any():
                    self.pillars[days] = s
        def _wing(key):
            fr = skew.get(key)
            if fr is not None and ticker in getattr(fr, "columns", []):
                return fr[ticker].reindex(index).ffill()
            return None
        self.put, self.call, self.atm1m = _wing("put"), _wing("call"), _wing("atm")

    def ok(self) -> bool:
        return bool(self.pillars)

    def atm(self, day, tau_days: float) -> float:
        xs = sorted(self.pillars)
        vals = {}
        for d in xs:
            v = self.pillars[d].get(day, np.nan)
            if np.isfinite(v):
                vals[d] = float(v)
        if not vals:
            return np.nan
        xs = sorted(vals)
        t = float(max(tau_days, 1.0))
        if t <= xs[0]:
            return vals[xs[0]]
        if t >= xs[-1]:
            return vals[xs[-1]]
        for lo, hi in zip(xs, xs[1:]):
            if lo <= t <= hi:
                wlo, whi = vals[lo] ** 2 * lo, vals[hi] ** 2 * hi   # total variance
                w = wlo + (whi - wlo) * (t - lo) / (hi - lo)
                return math.sqrt(max(w, 1e-8) / t)
        return vals[xs[-1]]

    def vol(self, day, tau_days: float, F: float, K: float) -> float:
        """Fixed-strike vol: tenor ATM + the 1M smile shape at ln(K/F)."""
        base = self.atm(day, tau_days)
        if not np.isfinite(base) or any(s is None for s in (self.put, self.call, self.atm1m)):
            return base
        vp, vc, va = (s.get(day, np.nan) for s in (self.put, self.call, self.atm1m))
        if not all(np.isfinite(v) for v in (vp, vc, va)):
            return base
        xp, xc = SMILE_X
        # exact quadratic through (xp, vp-va), (0, 0), (xc, vc-va)
        det = xp * xc * (xp - xc)
        a = (xp * xp * (vc - va) - xc * xc * (vp - va)) / det
        b = (xc * (vp - va) - xp * (vc - va)) / det
        x = max(-0.20, min(0.20, math.log(K / F)))
        decay = min(1.0, math.sqrt(30.0 / max(tau_days, 30.0)))
        return max(base + (a * x + b * x * x) * decay, 1.0)


# ── the backtest ─────────────────────────────────────────────────────────────
@dataclass
class LegState:
    ticker: str
    mult: float
    n: float = 0.0            # straddles held (buy leg +, sell leg −)
    K: float = np.nan
    hedge: float = 0.0        # futures lots held against the leg
    g: Greeks | None = None   # greeks at last settlement
    F: float = np.nan
    vol: float = np.nan
    tau_d: float = np.nan


@dataclass
class Result:
    daily: pd.DataFrame       # per-day marks, P&L and attribution (see columns)
    events: pd.DataFrame      # entry / re-strikes / exit log
    summary: dict
    warnings: list = field(default_factory=list)
    trades: pd.DataFrame | None = None   # the blotter: every option/futures ticket


@dataclass
class PairCorr:
    """How the two products usually move together, as of a given date — the pair
    context behind a vol spread. Correlations are of DAILY CHANGES (log returns
    for price; vol-point diffs for the 1M ATM IV), so they read as co-movement,
    not shared trend."""
    asof: pd.Timestamp
    px_1y: float          # return corr, trailing 252 obs
    px_1m: float          # return corr, trailing 21 obs
    iv_1y: float          # 1M ATM IV daily-change corr, trailing 252 obs
    iv_1m: float          # ... trailing 21 obs
    rv_1y: float          # 1M realized-vol daily-change corr, trailing 252 obs
    rv_1m: float          # ... trailing 21 obs
    pctl: float           # where px_1m sits in the year of rolling 1M corr (0-100)
    rolling_px: pd.Series   # rolling 21d return corr, last ~252 obs
    rolling_iv: pd.Series   # rolling 21d IV-change corr, aligned window
    rolling_rv: pd.Series   # rolling 21d realized-vol-change corr, aligned window


def correlation_stats(buy: str, sell: str, asof, *, short: int = 21,
                      long: int = 252) -> PairCorr | None:
    """1-year vs 1-month correlation of the pair up to `asof` (what you'd have
    known at trade time), plus the rolling 1M series so a breakdown is visible
    as a picture, not just a number. None when there isn't enough history."""
    end = pd.Timestamp(asof)
    start = end - pd.Timedelta(days=580)          # 252+21 obs of rolling corr + slack
    tickers = [buy, sell]
    px = get_history(tickers, start=start, end=end).reindex(columns=tickers)
    iv = get_implied_vol_history(tickers, start=start, end=end).reindex(columns=tickers)
    rets = np.log(px).diff().dropna().loc[:end]
    if len(rets) < short + 5:
        return None
    dv = iv.diff().dropna().loc[:end]
    rvol = rets.rolling(short).std() * np.sqrt(252) * 100.0    # 1M realized vol (both legs)
    drv = rvol.diff().dropna().loc[:end]                       # its daily changes
    r1y = rets.iloc[-long:]
    r1m = rets.iloc[-short:]
    roll_px = rets[buy].rolling(short).corr(rets[sell]).dropna().iloc[-long:]
    roll_iv = (dv[buy].rolling(short).corr(dv[sell]).dropna().reindex(roll_px.index)
               if len(dv) >= short + 5 else pd.Series(index=roll_px.index, dtype=float))
    roll_rv = (drv[buy].rolling(short).corr(drv[sell]).dropna().reindex(roll_px.index)
               if len(drv) >= short + 5 else pd.Series(index=roll_px.index, dtype=float))
    px_1m = float(r1m[buy].corr(r1m[sell]))
    pctl = float((roll_px <= px_1m).mean() * 100.0) if len(roll_px) else np.nan
    def _c(d):
        return float(d[buy].corr(d[sell])) if len(d) >= 5 else np.nan
    return PairCorr(asof=end, px_1y=_c(r1y), px_1m=px_1m,
                    iv_1y=_c(dv.iloc[-long:]), iv_1m=_c(dv.iloc[-short:]),
                    rv_1y=_c(drv.iloc[-long:]), rv_1m=_c(drv.iloc[-short:]),
                    pctl=pctl, rolling_px=roll_px, rolling_iv=roll_iv, rolling_rv=roll_rv)


def vol_beta(iv: pd.DataFrame, buy: str, sell: str, entry, lookback: int = 252):
    """β of daily 1M ATM IV changes, buy regressed on sell, over the year before
    entry — the beta_vega weighting's scale. (n_obs, β); β=1 when too thin."""
    d = iv[[buy, sell]].dropna().loc[:pd.Timestamp(entry)].diff().dropna().iloc[-lookback:]
    if len(d) < 60:
        return len(d), 1.0
    var = float(d[sell].var())
    if var <= 0:
        return len(d), 1.0
    return len(d), float(np.clip(d[buy].cov(d[sell]) / var, 0.2, 5.0))


def _ratio(mode: str, gb: Greeks, gs: Greeks, Fb: float, Fs: float,
           mb: float, ms: float, beta: float,
           vb: float = np.nan, vs: float = np.nan) -> float:
    """Sell straddles per buy straddle so the chosen exposure nets to zero.
    `vb`/`vs` are the legs' marking vols (vol points) — rn_gamma only."""
    if mode == "gamma":
        return (gb.gamma * Fb * Fb * mb) / (gs.gamma * Fs * Fs * ms)
    if mode == "rn_gamma":
        return (gb.gamma * Fb * Fb * vb * vb * mb) / (gs.gamma * Fs * Fs * vs * vs * ms)
    if mode == "vega":
        return (gb.vega * mb) / (gs.vega * ms)
    if mode == "beta_vega":
        return beta * (gb.vega * mb) / (gs.vega * ms)
    if mode == "premium":
        return (gb.value * mb) / (gs.value * ms)
    raise ValueError(f"unknown weighting '{mode}'")


def run_backtest(buy: str | None, sell: str | None, entry: date, expiry: date,
                 weighting: str = "rn_gamma", restrike: str = "threshold",
                 restrike_mult: float = 1.0, buy_lots: float = 10.0,
                 opt_cost_lot: float = 0.0, fut_cost_lot: float = 0.0,
                 exit_bd_before_expiry: int = 5, end: date | None = None,
                 mult_override: dict | None = None) -> Result:
    """Run the delta-hedged straddle trade.

    Two products (buy + sell) -> the spread; ONE product (only `buy` = long vol,
    or only `sell` = short vol) -> a single leg, delta-hedged with its own
    futures, so the P&L is that product's implied vs its OWN realized. In
    single-leg mode `weighting` is ignored (size = `buy_lots` straddles).

    Costs are flat per-lot amounts in the instrument's own currency: option
    trades pay `opt_cost_lot` per option contract traded (a straddle = 1 call
    + 1 put = 2 lots), futures hedges pay `fut_cost_lot` per futures contract
    traded."""
    warnings: list[str] = []
    leg_defs = {k: t for k, t in (("buy", buy), ("sell", sell)) if t}
    if not leg_defs:
        raise ValueError("pick at least one product")
    single = len(leg_defs) == 1
    tickers = list(leg_defs.values())
    start_hist = pd.Timestamp(entry) - pd.Timedelta(days=420)   # β + realized stats
    end_ts = pd.Timestamp(min(end, expiry) if end else expiry)

    px = get_history(tickers, start=start_hist, end=end_ts)
    iv1m = get_implied_vol_history(tickers, start=start_hist, end=end_ts).reindex(columns=tickers)
    # Deliberately VENDOR term frames (not the own-curve overlay the Term strategy
    # uses): backtests need a deep, uniform tenor history and own_term_history only
    # accrues from 2026-07-27 — revisit once it's a year deep or backfilled.
    term = get_term_structure(tickers, start=start_hist, end=end_ts)
    skew = get_skew_components(tickers, start=start_hist, end=end_ts)
    for t in tickers:
        if t not in px.columns or px[t].dropna().empty:
            raise ValueError(f"no settlement history for {t}")

    stale = stale_iv_reasons(iv1m)
    for t in tickers:
        if t in stale:
            warnings.append(f"{product_name(t)}: implied-vol surface looks stale ({stale[t]}) — treat marks with care")

    idx = px.dropna(how="any").index
    days = idx[(idx >= pd.Timestamp(entry)) & (idx <= end_ts)]
    if len(days) < 2:
        raise ValueError(f"no settlement dates between {entry} and {end_ts.date()}")
    if days[0].date() != entry:
        warnings.append(f"entry moved to the first settlement on/after {entry}: {days[0].date()}")
    # exit: N business days before expiry (ATM gamma near the pin is not a backtest)
    cutoff = pd.Timestamp(expiry) - pd.tseries.offsets.BDay(exit_bd_before_expiry)
    days = days[days <= cutoff]
    if len(days) < 2:
        raise ValueError("entry is inside the pre-expiry exit buffer — pick an earlier entry or later expiry")

    surf = {t: VolSurface(t, term, skew, idx) for t in tickers}
    for t in tickers:
        if not surf[t].ok():
            raise ValueError(f"no term-structure vols for {t}")

    mults = dict(mult_override or {})
    for t in tickers:
        mults.setdefault(t, point_value(t))
        if mults[t] <= 0:
            raise ValueError(f"no point value for {t} — set one on the page")

    # ---- FX: put non-USD legs onto dollars at the entry-date rate (frozen) ------
    fx_used = {}
    for k, t in leg_defs.items():
        ccy = currency(t)
        if ccy == "USD":
            continue
        rate = np.nan
        src = FX_USD.get(ccy)
        if src:
            try:
                # raw=True: the entry-date FX RATE must be what actually traded that day —
                # a roll-adjusted continuation level is fiction for a point-in-time lookup.
                fxh = get_history([src], start=start_hist, end=end_ts, raw=True)
                if src in fxh.columns:
                    fxs = fxh[src].dropna()
                    fxs = fxs[fxs.index <= days[0]]
                    if len(fxs):
                        rate = fx_usd_rate(src, fxs.iloc[-1])
            except Exception:
                rate = np.nan
        if np.isfinite(rate) and rate > 0:
            mults[t] = mults[t] * rate
            fx_used[k] = {"ccy": ccy, "rate": rate}
            warnings.append(f"{product_name(t)} is {ccy}-denominated — its point value is "
                            f"converted at {ccy}USD {rate:.4f} (the entry-date rate, frozen "
                            f"for the run), so every figure reads as USD")
        else:
            warnings.append(f"{product_name(t)} is {ccy}-denominated but no {ccy}USD rate "
                            f"was available from the feed — treating 1 {ccy} = 1 USD")

    if single:
        n_beta, beta = 0, 1.0
    else:
        n_beta, beta = vol_beta(iv1m, buy, sell, entry)
        if weighting == "beta_vega" and n_beta < 60:
            warnings.append(f"vol-β lookback too thin ({n_beta} obs) — using β=1 (plain vega)")

    legs = {k: LegState(t, mults[t]) for k, t in leg_defs.items()}
    sides = {"buy": 1.0, "sell": -1.0}
    rows, events, trades = [], [], []
    cost_opt_total = cost_fut_total = 0.0

    def _trade(day, k: str, leg: LegState, instrument: str, action: str,
               lots: float, price: float, strike: float, reason: str) -> None:
        """One blotter ticket. Prices are the Black-76 model value at that day's
        settlement (mid) — the cost assumptions are charged separately, never
        baked into these prices. Cash: sells +, buys −, × point value."""
        trades.append({"date": day.date(), "leg": k, "product": product_name(leg.ticker),
                       "instrument": instrument, "action": action, "lots": float(lots),
                       "strike": float(strike) if np.isfinite(strike) else np.nan,
                       "price": float(price),
                       "cash": (1.0 if action == "sell" else -1.0) * lots * price * leg.mult,
                       "reason": reason})

    def _mark(leg: LegState, day) -> tuple[float, float, Greeks]:
        F = float(px.loc[day, leg.ticker])
        tau_d = max((expiry - day.date()).days, 1)
        vol = surf[leg.ticker].vol(day, tau_d, F, leg.K if np.isfinite(leg.K) else F)
        if not np.isfinite(vol):
            vol = leg.vol                                     # carry the last mark
        return F, vol, straddle_greeks(F, leg.K if np.isfinite(leg.K) else F, vol, tau_d / 365.0)

    def _open(day, tag: str) -> tuple[float, float]:
        """(Re)strike both legs ATM at today's settle, re-ratio, re-hedge.
        Returns (option cost, futures-hedge trade cost) for the new package."""
        F, g, vol = {}, {}, {}
        for k, leg in legs.items():
            F[k] = float(px.loc[day, leg.ticker])
            tau_d = max((expiry - day.date()).days, 1)
            vol[k] = surf[leg.ticker].vol(day, tau_d, F[k], F[k])
            if not np.isfinite(vol[k]):
                if not np.isfinite(leg.vol):
                    raise ValueError(f"no vol for {leg.ticker} on {day.date()}")
                vol[k] = leg.vol                      # stale surface: carry the mark
            g[k] = straddle_greeks(F[k], F[k], vol[k], tau_d / 365.0)
        if single:
            r = np.nan                              # no second leg to ratio against
            n = {k: buy_lots for k in legs}
        else:
            r = _ratio("vega" if (weighting == "beta_vega" and n_beta < 60) else weighting,
                       g["buy"], g["sell"], F["buy"], F["sell"],
                       legs["buy"].mult, legs["sell"].mult, beta,
                       vol["buy"], vol["sell"])
            n = {"buy": buy_lots, "sell": buy_lots * r}
        c_opt = c_fut = 0.0
        for k, leg in legs.items():
            c_opt += opt_cost_lot * 2.0 * n[k]          # a straddle = call + put = 2 lots
            new_hedge = -sides[k] * n[k] * g[k].delta
            c_fut += abs(new_hedge - leg.hedge) * fut_cost_lot
            act = "buy" if k == "buy" else "sell"
            reason = "Entry" if tag == "entry" else "Re-strike open"
            _trade(day, k, leg, "call", act, n[k], g[k].call, F[k], reason)
            _trade(day, k, leg, "put", act, n[k], g[k].put, F[k], reason)
            dq = new_hedge - leg.hedge
            if abs(dq) > 1e-9:
                _trade(day, k, leg, "future", "buy" if dq > 0 else "sell",
                       abs(dq), F[k], np.nan, "Delta hedge")
            leg.n, leg.K, leg.hedge = n[k], F[k], new_hedge
            leg.g, leg.F, leg.vol = g[k], F[k], vol[k]
            leg.tau_d = max((expiry - day.date()).days, 1)
        events.append({"date": day.date(), "event": tag,
                       "buy_K": F.get("buy", np.nan), "sell_K": F.get("sell", np.nan),
                       "buy_iv": vol.get("buy", np.nan), "sell_iv": vol.get("sell", np.nan),
                       "sell_per_buy": r, "buy_lots": n.get("buy", np.nan),
                       "sell_lots": n.get("sell", np.nan)})
        return c_opt, c_fut

    # ---- entry ----------------------------------------------------------------
    d0 = days[0]
    c_opt, c_fut = _open(d0, "entry")
    cost_opt_total += c_opt; cost_fut_total += c_fut
    entry_iv = {k: legs[k].vol for k in legs}
    row0 = {"date": d0, "net_opt": 0.0, "net_hedge": 0.0, "net_gamma": 0.0,
            "net_theta": 0.0, "net_vega": 0.0, "net_resid": 0.0,
            "cost": c_opt + c_fut, "net": -(c_opt + c_fut), "restrike": 1}
    for k, leg in legs.items():
        row0.update({f"{k}_F": leg.F, f"{k}_K": leg.K, f"{k}_iv": leg.vol,
                     f"{k}_gamma_usd": leg.g.gamma * leg.F ** 2 * leg.n * leg.mult,
                     f"{k}_vega_usd": leg.g.vega * leg.n * leg.mult,
                     f"{k}_pnl": 0.0})
    rows.append(row0)

    # ---- daily loop -------------------------------------------------------------
    for prev, day in zip(days, days[1:]):
        dt_years = max((day - prev).days, 1) / 365.0
        last = day == days[-1]
        row = {"date": day, "cost": 0.0, "restrike": 0}
        net = {k: 0.0 for k in ("opt", "hedge", "gamma", "theta", "vega", "resid")}
        for k, leg in legs.items():
            s = sides[k]
            F, vol, g = _mark(leg, day)
            dF = F - leg.F
            opt = s * leg.n * leg.mult * (g.value - leg.g.value)
            hedge = leg.hedge * dF * leg.mult
            gam = s * 0.5 * leg.g.gamma * dF * dF * leg.n * leg.mult
            the = s * leg.g.theta * dt_years * leg.n * leg.mult
            veg = s * leg.g.vega * (vol - leg.vol) * leg.n * leg.mult
            dlt = s * leg.g.delta * dF * leg.n * leg.mult
            net["opt"] += opt; net["hedge"] += hedge
            net["gamma"] += gam; net["theta"] += the; net["vega"] += veg
            net["resid"] += opt - dlt - gam - the - veg
            row.update({f"{k}_F": F, f"{k}_K": leg.K, f"{k}_iv": vol,
                        f"{k}_pnl": opt + hedge,
                        f"{k}_gamma_usd": g.gamma * F ** 2 * leg.n * leg.mult,
                        f"{k}_vega_usd": g.vega * leg.n * leg.mult})
            leg.F, leg.vol, leg.g = F, vol, g
            leg.tau_d = max((expiry - day.date()).days, 1)

        # ---- close / re-strike at today's settlement ----------------------------
        if last:
            for k, leg in legs.items():
                act = "sell" if k == "buy" else "buy"
                _trade(day, k, leg, "call", act, leg.n, leg.g.call, leg.K, "Exit close")
                _trade(day, k, leg, "put", act, leg.n, leg.g.put, leg.K, "Exit close")
                if abs(leg.hedge) > 1e-9:
                    _trade(day, k, leg, "future", "buy" if leg.hedge < 0 else "sell",
                           abs(leg.hedge), leg.F, np.nan, "Delta hedge (close)")
            c_opt = sum(opt_cost_lot * 2.0 * leg.n for leg in legs.values())
            c_fut = sum(abs(leg.hedge) * fut_cost_lot for leg in legs.values())
            cost_opt_total += c_opt; cost_fut_total += c_fut
            row["cost"] = c_opt + c_fut
            _a = lambda k, a: getattr(legs[k], a) if k in legs else np.nan
            events.append({"date": day.date(), "event": "exit",
                           "buy_K": _a("buy", "K"), "sell_K": _a("sell", "K"),
                           "buy_iv": _a("buy", "vol"), "sell_iv": _a("sell", "vol"),
                           "sell_per_buy": np.nan, "buy_lots": _a("buy", "n"),
                           "sell_lots": _a("sell", "n")})
        else:
            do_restrike = restrike == "daily"
            if restrike == "threshold":
                for leg in legs.values():
                    atm = surf[leg.ticker].atm(day, leg.tau_d)
                    move = (atm if np.isfinite(atm) else leg.vol) / 100.0 / SQRT252
                    if abs(leg.F / leg.K - 1.0) >= restrike_mult * move:
                        do_restrike = True
            if do_restrike:
                for k, leg in legs.items():
                    act = "sell" if k == "buy" else "buy"
                    _trade(day, k, leg, "call", act, leg.n, leg.g.call, leg.K, "Re-strike close")
                    _trade(day, k, leg, "put", act, leg.n, leg.g.put, leg.K, "Re-strike close")
                c_close = sum(opt_cost_lot * 2.0 * leg.n for leg in legs.values())
                c_opt, c_fut = _open(day, restrike)
                cost_opt_total += c_close + c_opt; cost_fut_total += c_fut
                row["cost"] = c_close + c_opt + c_fut
                row["restrike"] = 1
            else:                                     # plain re-hedge to flat delta
                c_fut = 0.0
                for k, leg in legs.items():
                    new_hedge = -sides[k] * leg.n * leg.g.delta
                    c_fut += abs(new_hedge - leg.hedge) * fut_cost_lot
                    dq = new_hedge - leg.hedge
                    if abs(dq) > 1e-9:
                        _trade(day, k, leg, "future", "buy" if dq > 0 else "sell",
                               abs(dq), leg.F, np.nan, "Delta hedge")
                    leg.hedge = new_hedge
                cost_fut_total += c_fut
                row["cost"] = c_fut

        for kk, v in net.items():
            row[f"net_{kk}"] = v
        row["net"] = net["opt"] + net["hedge"] - row["cost"]
        rows.append(row)

    daily = pd.DataFrame(rows).set_index("date")
    for k in ("buy", "sell"):                       # single-leg: absent side reads flat
        if k not in legs:
            for col, val in ((f"{k}_F", np.nan), (f"{k}_K", np.nan), (f"{k}_iv", np.nan),
                             (f"{k}_pnl", 0.0), (f"{k}_gamma_usd", np.nan),
                             (f"{k}_vega_usd", np.nan)):
                daily[col] = val
    daily["cum_net"] = daily["net"].cumsum()

    # ---- summary ---------------------------------------------------------------
    hold = px.loc[days[0]:days[-1]]
    rlz = (np.log(hold).diff().std() * SQRT252 * 100.0)
    summary = {
        "buy": buy, "sell": sell, "single": single,
        "buy_name": product_name(buy) if buy else "",
        "sell_name": product_name(sell) if sell else "",
        "mode": MODE, "weighting": weighting, "restrike": restrike,
        "restrike_mult": restrike_mult, "buy_lots": buy_lots,
        "opt_cost_lot": opt_cost_lot, "fut_cost_lot": fut_cost_lot,
        "entry": days[0].date(), "exit": days[-1].date(), "expiry": expiry,
        "n_days": len(days) - 1, "beta": beta, "beta_obs": n_beta,
        "mult_buy": mults.get(buy, np.nan), "mult_sell": mults.get(sell, np.nan),
        "fx": fx_used,
        "entry_iv_buy": entry_iv.get("buy", np.nan), "entry_iv_sell": entry_iv.get("sell", np.nan),
        "entry_iv_spread": (entry_iv["buy"] - entry_iv["sell"]) if not single else np.nan,
        "rlz_buy": float(rlz[buy]) if buy else np.nan,
        "rlz_sell": float(rlz[sell]) if sell else np.nan,
        "rlz_spread": float(rlz[buy] - rlz[sell]) if not single else np.nan,
        "total": float(daily["net"].sum()),
        "total_buy": float(daily["buy_pnl"].sum()),
        "total_sell": float(daily["sell_pnl"].sum()),
        "gamma_pnl": float(daily["net_gamma"].sum()),
        "theta_pnl": float(daily["net_theta"].sum()),
        "vega_pnl": float(daily["net_vega"].sum()),
        "resid_pnl": float(daily["net_resid"].sum()),
        "hedge_slip": float((daily["net_opt"] + daily["net_hedge"] - daily["net_gamma"]
                             - daily["net_theta"] - daily["net_vega"] - daily["net_resid"]).sum()),
        "costs": float(daily["cost"].sum()),
        "cost_opt": cost_opt_total, "cost_fut": cost_fut_total,
        "n_restrikes": int(daily["restrike"].sum()) - 1,
        "max_dd": float((daily["cum_net"].cummax() - daily["cum_net"]).max()),
        "sell_per_buy_entry": float(events[0]["sell_per_buy"]),
    }
    return Result(daily=daily, events=pd.DataFrame(events), summary=summary,
                  warnings=warnings, trades=pd.DataFrame(trades))
