"""Our OWN constant-maturity STIR ATM vol curve — built from listed option settlements.

The desk's construction (2026-07-22), identical for all three 3M money-market
contracts (SOFR / SONIA / Euribor) so the report's STIR section is methodologically
consistent — the thing Bloomberg's smoothed surface is not (absent for Euribor,
opaque elsewhere; their 3M point prints above the listed 142-day ATM, which they
are currently unable to explain):

  1. For each quarterly contract (H/M/U/Z), each day: the ATM strike is the listed
     strike nearest that day's future settlement.
  2. The ATM call and put settlement premiums are inverted through Black-76 on the
     IMPLIED RATE (R = 100 − F, lognormal, T in ACT/365, no discounting — futures-
     style premiums; the discount at these tenors moves ATM vol by well under a
     tenth of a point) and the two implied vols are MID-ed.
  3. Pillars are interpolated linearly in TOTAL VARIANCE to a constant 90 days —
     the natural tenor for quarterly contracts (always bracketed by two liquid
     expiries, and it spans the meetings the options actually price).

Because it runs on SETTLEMENT PRICES (which Bloomberg preserves even for expired
options), the same code path both BACKFILLS a year of history and appends the
daily point — one construction, one series. Bloomberg's 3MTH surface point is
carried alongside as a continuous cross-check where published (SOFR / SONIA).

History lives in data/snapshot/stir_curve_history.parquet
(columns: date, ticker, ours, bbg).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "data" / "snapshot" / "stir_curve_history.parquet"

TARGET_DAYS = 90.0
MIN_PREMIUM = 0.002          # below ~half a tick a settle carries no usable time value
QUARTER_CODES = "HMUZ"       # Mar / Jun / Sep / Dec

# generic -> (futures root, strike grid step). Grids verified on the live chains
# 2026-07-22 (SOFR/Euribor list sixteenths near ATM; SONIA 0.05s).
PRODUCTS = {
    "SFRA Comdty": ("SFR", 0.0625),
    "SFIA Comdty": ("SFI", 0.05),
    "ERA Comdty":  ("ER",  0.0625),
}


# ---------------------------------------------------------------------------
# Black-76 on the implied rate, and its inversion from a settlement premium.
# ---------------------------------------------------------------------------
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _price_on_rate(R: float, KR: float, T: float, sigma: float, call_on_rate: bool) -> float:
    if sigma <= 0 or T <= 0 or R <= 0 or KR <= 0:
        return max((R - KR) if call_on_rate else (KR - R), 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(R / KR) + 0.5 * st * st) / st
    d2 = d1 - st
    if call_on_rate:
        return R * _phi(d1) - KR * _phi(d2)
    return KR * _phi(-d2) - R * _phi(-d1)


def implied_vol(premium: float, F: float, K: float, dte: float, is_call_on_future: bool):
    """Invert a STIR option settlement into a lognormal RATE vol (%, annualised).
    A call on the FUTURE is a put on the RATE (and vice versa). None when the
    premium carries no invertible time value."""
    R, KR = 100.0 - F, 100.0 - K
    if R <= 0 or KR <= 0 or dte <= 0:
        return None
    call_on_rate = not is_call_on_future
    T = dte / 365.0
    intrinsic = max((R - KR) if call_on_rate else (KR - R), 0.0)
    if premium < max(MIN_PREMIUM, intrinsic + 5e-4):
        return None
    lo, hi = 1e-4, 4.0
    if _price_on_rate(R, KR, T, hi, call_on_rate) < premium:
        return None                                   # premium above any sane vol
    for _ in range(60):                               # bisection — price is monotone in vol
        mid = 0.5 * (lo + hi)
        if _price_on_rate(R, KR, T, mid, call_on_rate) < premium:
            lo = mid
        else:
            hi = mid
    return 100.0 * 0.5 * (lo + hi)


def const_maturity(pillars, target: float = TARGET_DAYS):
    """[(dte, vol%), ...] -> constant `target`-day vol, linear in TOTAL VARIANCE."""
    pts = sorted((float(d), float(v)) for d, v in pillars if v is not None and d > 0)
    if not pts:
        return None
    if len(pts) == 1 or target <= pts[0][0]:
        return pts[0][1]
    for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
        if t1 <= target <= t2:
            w1, w2 = v1 * v1 * t1, v2 * v2 * t2
            tv = w1 + (w2 - w1) * (target - t1) / (t2 - t1)
            return float(math.sqrt(tv / target))
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Bloomberg plumbing (lazy imports so snapshot-mode readers never need blpapi).
# ---------------------------------------------------------------------------
def _bdh(tickers, fld, start, end):
    from . import bbg as blp
    from .datafeed import _bdh_to_wide
    try:
        return _bdh_to_wide(blp.bdh(tickers=tickers, flds=fld, start_date=start, end_date=end))
    except Exception:
        return None


def _bdh_safe(tickers, fld, start, end, chunk: int = 20):
    """Chunked bdh with single-ticker fallback — one bad constructed ticker must not
    sink the batch. Returns a wide frame (possibly missing some columns)."""
    frames = []
    tickers = list(tickers)
    for i in range(0, len(tickers), chunk):
        grp = tickers[i:i + chunk]
        w = _bdh(grp, fld, start, end)
        if w is not None:
            frames.append(w)
            continue
        for t in grp:                                  # fall back one by one
            w1 = _bdh([t], fld, start, end)
            if w1 is not None:
                frames.append(w1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _bdp_one(ticker, fld):
    from . import bbg as blp
    from .datafeed import _coerce_pd
    try:
        pdf = _coerce_pd(blp.bdp(ticker, fld))
        if pdf is None or len(pdf) == 0:
            return None
        return pdf.iloc[0, -1]
    except Exception:
        return None


def _strike_str(k: float) -> str:
    """Bloomberg's strike string: 4dp, trailing zeros stripped to minimum 3dp
    ('96.1250'->'96.125', '96.0625' stays, '96.2000'->'96.200')."""
    s = f"{k:.4f}"
    if s.endswith("0"):
        s = s[:-1]
    return s


def _contracts(root: str, start: pd.Timestamp, end: pd.Timestamp):
    """Quarterly contract stems (e.g. 'SFRU5') whose OPTIONS could be alive in the
    window — from the quarter before `start` through ~a year past `end`."""
    out = []
    y, q = start.year, 0
    months = {"H": 3, "M": 6, "U": 9, "Z": 12}
    for year in range(start.year - 1, end.year + 2):
        for code in QUARTER_CODES:
            approx_exp = pd.Timestamp(year=year, month=months[code], day=15)
            if start - pd.Timedelta(days=100) <= approx_exp <= end + pd.Timedelta(days=380):
                out.append(f"{root}{code}{year % 10}")
    return out


def _build_product(generic: str, root: str, step: float,
                   start: pd.Timestamp, end: pd.Timestamp, log=print) -> pd.Series:
    """Daily constant-90d ATM vol series for one product over [start, end]."""
    s_iso, e_iso = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"
    pillars_by_date: dict = {}
    for stem in _contracts(root, start, end):
        fut = f"{stem} Comdty"
        opt_exp = _bdp_one(f"{stem}C {_strike_str(round(96 / step) * step)} Comdty", "OPT_EXPIRE_DT")
        # the 96-strike probe ticker may not exist for every contract — try a settle-based one below
        fpx = _bdh_safe([fut], "PX_SETTLE", s_iso, e_iso)
        if fut not in getattr(fpx, "columns", []):
            continue
        f_s = fpx[fut].dropna()
        if f_s.empty:
            continue
        if opt_exp is None:
            # second try: a strike near the contract's own median settle
            k0 = round(float(f_s.median()) / step) * step
            opt_exp = _bdp_one(f"{stem}C {_strike_str(k0)} Comdty", "OPT_EXPIRE_DT")
        if opt_exp is None:
            log(f"    {stem}: no option expiry date — skipped")
            continue
        opt_exp = pd.Timestamp(opt_exp)
        window = f_s[(f_s.index < opt_exp)]
        if window.empty:
            continue
        lo = math.floor((float(window.min()) - 2 * step) / step) * step
        hi = math.ceil((float(window.max()) + 2 * step) / step) * step
        strikes = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
        c_tks = {k: f"{stem}C {_strike_str(k)} Comdty" for k in strikes}
        p_tks = {k: f"{stem}P {_strike_str(k)} Comdty" for k in strikes}
        cw = _bdh_safe(list(c_tks.values()), "PX_LAST", s_iso, e_iso)
        pw = _bdh_safe(list(p_tks.values()), "PX_LAST", s_iso, e_iso)
        n_live = sum(1 for k in strikes if c_tks[k] in getattr(cw, "columns", []))
        log(f"    {stem}: opt-exp {opt_exp:%d %b %y}, {len(strikes)} strikes "
            f"({_strike_str(strikes[0])}–{_strike_str(strikes[-1])}), {n_live} call series")
        for dt, F in window.items():
            dte = (opt_exp - dt).days
            if dte <= 1:
                continue
            F = float(F)
            for k in sorted(strikes, key=lambda x: abs(x - F))[:3]:   # nearest, then neighbours
                cs = cw[c_tks[k]].get(dt) if c_tks[k] in getattr(cw, "columns", []) else None
                ps = pw[p_tks[k]].get(dt) if p_tks[k] in getattr(pw, "columns", []) else None
                ivs = [v for v in (
                    implied_vol(float(cs), F, k, dte, True) if pd.notna(cs) else None,
                    implied_vol(float(ps), F, k, dte, False) if pd.notna(ps) else None,
                ) if v is not None]
                if ivs:
                    pillars_by_date.setdefault(dt, []).append((dte, float(np.mean(ivs))))
                    break
    out = {dt: const_maturity(pl) for dt, pl in pillars_by_date.items()}
    ser = pd.Series(out, dtype=float).sort_index().dropna()
    log(f"    -> {len(ser)} daily 90d points" + (f", last {ser.iloc[-1]:.2f}" if len(ser) else ""))
    return ser


def build_history(start, end=None, log=print) -> pd.DataFrame:
    """Reconstruct the whole series for every product over [start, end] (long form)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    frames = []
    for generic, (root, step) in PRODUCTS.items():
        log(f"  {generic} ({root}*, step {step})")
        ser = _build_product(generic, root, step, start, end, log=log)
        if len(ser):
            frames.append(pd.DataFrame({"date": ser.index, "ticker": generic, "ours": ser.values}))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "ours", "bbg"])
    df = pd.concat(frames, ignore_index=True)
    df["bbg"] = np.nan
    try:                                              # Bloomberg's 3M point where published
        t3 = pd.read_parquet(ROOT / "data" / "snapshot" / "term_3m.parquet").set_index("date")
        t3.index = pd.to_datetime(t3.index)
        for tk in df["ticker"].unique():
            if tk in t3.columns and tk != "ERA Comdty":   # ERA's column is polluted pre-fix
                m = t3[tk]
                df.loc[df["ticker"] == tk, "bbg"] = df.loc[df["ticker"] == tk, "date"].map(m).values
    except Exception:
        pass
    return df


def save_history(df: pd.DataFrame) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["ticker", "date"]).to_parquet(HISTORY_FILE, index=False)


def load_history() -> "pd.DataFrame | None":
    if not HISTORY_FILE.exists():
        return None
    try:
        h = pd.read_parquet(HISTORY_FILE)
        h["date"] = pd.to_datetime(h["date"])
        return h
    except Exception:
        return None


def update_history(lookback_days: int = 5, log=print) -> None:
    """Daily top-up (called from the snapshot pull): recompute the recent sessions
    with the same construction and upsert — settles finalise, expiries roll, gaps
    heal. 12 → 5 days on 2026-08-18 (Bloomberg budget): exchange settle revisions
    land within a session or two, so a 5-day window catches them at ~40% of the
    strike-ladder request cost. Self-heal is PER PRODUCT (review 2026-08-18): the
    window stretches back to the stalest product's own last stored date (capped at
    45 days), and the upsert only replaces rows for products that actually produced
    fresh rows — so a single product stalling (e.g. its expiry probes failing for a
    week) neither gaps its history nor gets its stored rows deleted by the others'
    recompute window."""
    old = load_history()
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).normalize()
    if old is not None and not old.empty:              # per-product outage self-heal
        stalest = pd.Timestamp(old.groupby("ticker")["date"].max().min())
        start = max(min(start, stalest - pd.Timedelta(days=1)),
                    pd.Timestamp.today().normalize() - pd.Timedelta(days=45))
    fresh = build_history(start, log=log)
    if fresh.empty:
        return
    if old is None or old.empty:
        save_history(fresh)
        return
    cutoff = fresh["date"].min()
    refreshed = old["ticker"].isin(set(fresh["ticker"]))
    merged = pd.concat([old[(old["date"] < cutoff) | ~refreshed], fresh],
                       ignore_index=True)
    save_history(merged)
