"""Our OWN constant-maturity ATM vol for the LISTED-FUTURES book — the STIR-curve
construction (src/stircurve.py) generalised to every product with listed options.

Per product, per day:
  1. Walk the futures chain; for each front contract pull the live option chain and
     group it into option SERIES (quarterlies + serials each carry their own expiry).
  2. Per series: the ATM strike nearest the underlying future's settle, the ATM call
     and put settlement premiums inverted through Black-76 on the FUTURES PRICE
     (lognormal, ACT/365, no discounting — the same convention family as the STIR
     inversion, which works on the implied rate instead), the two vols MID-ed, and
     the pair's open interest kept as the pillar's weight.
  3. Across ~5 expiries: fit the 3-parameter decaying term structure
        sigma(T) = sigma_L + (sigma_S - sigma_L) * exp(-T / tau)
     by WEIGHTED least squares (tau on a log grid -> the rest is closed-form linear
     algebra; open interest weights let thin serial months inform the curve without
     steering it). Fewer than 4 usable pillars -> linear-in-total-variance
     interpolation (the STIR fallback). A fitted curve whose total variance is not
     increasing across the pillar span is rejected in favour of the fallback.
  4. Read BOTH constant-maturity points off the same curve: 30d (the Volatility
     Report's tenor) and 90d.

FX is excluded (its liquid vol IS the OTC pair market we already use) and so are
the CASH indices (options on spot need dividend/rate inputs — a different model).
Products without listed options simply produce no pillars and drop out.

Output: data/snapshot/own30_curve.parquet — one row per product with ours30/ours90,
Bloomberg's 30d surface value for validation, pillar count and the pillar detail.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "data" / "snapshot" / "own30_curve.parquet"

MIN_DTE = 7                  # a dying front option prints noise, not vol
MAX_FUTS = 6                 # futures contracts to walk per product
N_EXPIRIES = 5               # option expiries (pillars) to collect
MIN_PREMIUM = 1e-4           # relative to the future's price scale, see _min_prem

EXCLUDE_ASSETS = {"FX", "STIRs"}          # FX = OTC by design; STIRs have stircurve.py


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black76(F: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return max((F - K) if is_call else (K - F), 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    d2 = d1 - st
    if is_call:
        return F * _phi(d1) - K * _phi(d2)
    return K * _phi(-d2) - F * _phi(-d1)


def _min_prem(F: float) -> float:
    """A 'worthless' threshold that scales with the product's price level."""
    return max(1e-4, 2e-4 * F)


def implied_vol_price(premium: float, F: float, K: float, dte: float, is_call: bool):
    """Settlement premium -> lognormal PRICE vol (%, annualised); None if uninvertible."""
    if F <= 0 or K <= 0 or dte <= 0:
        return None
    T = dte / 365.0
    intrinsic = max((F - K) if is_call else (K - F), 0.0)
    if premium < max(_min_prem(F), intrinsic + _min_prem(F) * 0.5):
        return None
    lo, hi = 1e-4, 4.0
    if _black76(F, K, T, hi, is_call) < premium:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _black76(F, K, T, mid, is_call) < premium:
            lo = mid
        else:
            hi = mid
    return 100.0 * 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# The fitted term structure: sigma(T) = sL + (sS - sL) * exp(-T / tau).
# ---------------------------------------------------------------------------
def fit_curve(pillars):
    """pillars = [(dte, vol%, weight)]. Returns callable sigma(days) or None.
    tau is scanned on a log grid; for each tau the model is linear in (sS-sL, sL),
    solved by weighted least squares in closed form."""
    pts = [(float(d), float(v), max(float(w), 1.0)) for d, v, w in pillars if v is not None and d > 0]
    if len(pts) < 4:
        return None
    d = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    w = np.array([p[2] for p in pts])
    best = None
    for tau in np.geomspace(5.0, 500.0, 40):
        X = np.column_stack([np.exp(-d / tau), np.ones_like(d)])
        Wh = np.sqrt(w)
        try:
            beta, *_ = np.linalg.lstsq(X * Wh[:, None], v * Wh, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = v - X @ beta
        sse = float(np.sum(w * resid * resid))
        if best is None or sse < best[0]:
            best = (sse, tau, beta)
    if best is None:
        return None
    _, tau, (a, sL) = best                      # sigma(T) = sL + a * exp(-T/tau)

    def sigma(days: float) -> float:
        return float(sL + a * math.exp(-days / tau))

    # Sanity: positive vols at the tenors we read, and total variance increasing
    # across the pillar span (a decaying-vol fit can otherwise imply negative
    # forward variance) — otherwise reject and let the caller fall back.
    grid = np.linspace(min(d.min(), 25.0), d.max(), 24)
    tv = np.array([max(sigma(t), 0.0) ** 2 * t for t in grid])
    if sigma(30.0) <= 0 or sigma(90.0) <= 0 or np.any(np.diff(tv) < -1e-9):
        return None
    return sigma


def interp_variance(pillars, target: float):
    """Fallback: linear in total variance between the bracketing pillars."""
    pts = sorted((float(d), float(v)) for d, v, _ in pillars if v is not None and d > 0)
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
# Bloomberg plumbing (live chains — the daily point; lazy imports).
# ---------------------------------------------------------------------------
def _rows(raw):
    from .datafeed import _coerce_pd
    pdf = _coerce_pd(raw)
    if pdf is None or len(pdf) == 0:
        return []
    pdf.columns = [str(c).strip().lower().replace(" ", "_") for c in pdf.columns]
    return pdf.to_dict("records")


def _first(r):
    for k in ("security_description", "security", "value"):
        if k in r and r[k]:
            return str(r[k]).strip()
    return ""


def _bds(ticker, fld, **kw):
    from xbbg import blp
    try:
        return _rows(blp.bds(ticker, fld, **kw))
    except Exception:
        return []


def _bdp(ticker, fld):
    from xbbg import blp
    from .datafeed import _coerce_pd
    try:
        pdf = _coerce_pd(blp.bdp(ticker, fld))
        if pdf is None or len(pdf) == 0:
            return None
        return pdf.iloc[0, -1]
    except Exception:
        return None


def _parse_chain(fut: str):
    """Option chain of one futures contract -> {series_root: {strike: (call, put)}}.
    Tickers look like 'CLU6C 65.00 Comdty' / 'C U6C 460 Comdty' (single-letter ag
    roots carry a SPACE) — so split from the RIGHT: the last token is the strike,
    everything before it is the series head ending in C/P."""
    series: dict = {}
    for r in _bds(fut, "OPT_CHAIN"):
        tk = _first(r)
        parts = tk.replace(" Comdty", "").replace(" Index", "").split()
        if len(parts) < 2:
            continue
        head, strike_s = " ".join(parts[:-1]), parts[-1]
        cp = head[-1]
        if cp not in "CP":
            continue
        try:
            strike = float(strike_s)
        except ValueError:
            continue
        root = head[:-1]
        slot = series.setdefault(root, {}).setdefault(strike, [None, None])
        slot[0 if cp == "C" else 1] = tk
    return series


def _strike_scale(strikes, F: float) -> float:
    """Strike/price UNIT reconciliation (RBOB futures print 2.38 while its strikes
    list in cents, etc.): pick the decimal scaling that puts the strike grid's
    median on the future's price scale. 1.0 when units already agree."""
    med = float(np.median(list(strikes)))
    if med <= 0 or F <= 0:
        return 1.0
    return min((1.0, 0.1, 0.01, 10.0, 100.0), key=lambda s: abs(math.log((med * s) / F)))


def product_pillars(generic: str, log=print):
    """[(dte, atm_mid_vol%, oi_weight, series_root)] across the front option expiries."""
    futs = []
    for r in _bds(generic, "FUT_CHAIN"):
        tk = _first(r)
        if tk:
            futs.append(tk if tk.split()[-1] in ("Comdty", "Index") else tk + " Comdty")
        if len(futs) >= MAX_FUTS:
            break
    today = pd.Timestamp.today().normalize()
    pillars, seen_exp = [], set()
    for fut in futs:
        if len(pillars) >= N_EXPIRIES:
            break
        F = _bdp(fut, "PX_SETTLE")
        F = _bdp(fut, "PX_LAST") if F is None else F
        if F is None:
            continue
        F = float(F)
        for root, strikes in sorted(_parse_chain(fut).items()):
            if len(pillars) >= N_EXPIRIES:
                break
            both = {k: v for k, v in strikes.items() if v[0] and v[1]}
            if not both:
                continue
            scale = _strike_scale(both.keys(), F)      # strike + premium quote units
            atm = sorted(both, key=lambda k: abs(k * scale - F))[0]
            if abs(atm * scale - F) / F > 0.25:        # nothing near the money — skip series
                continue
            call_tk, put_tk = both[atm]
            exp = _bdp(call_tk, "OPT_EXPIRE_DT")
            if exp is None:
                continue
            dte = (pd.Timestamp(exp) - today).days
            if dte < MIN_DTE or dte in seen_exp:
                continue
            c_px, p_px = _bdp(call_tk, "PX_SETTLE"), _bdp(put_tk, "PX_SETTLE")
            ivs = [v for v in (
                implied_vol_price(float(c_px) * scale, F, atm * scale, dte, True)
                if c_px is not None and pd.notna(c_px) else None,
                implied_vol_price(float(p_px) * scale, F, atm * scale, dte, False)
                if p_px is not None and pd.notna(p_px) else None,
            ) if v is not None and v > 0.5]            # floor-vol prints are dead marks, not vol
            if not ivs:
                continue
            oi = sum(float(x) for x in (_bdp(call_tk, "OPEN_INT"), _bdp(put_tk, "OPEN_INT"))
                     if x is not None and pd.notna(x))
            seen_exp.add(dte)
            pillars.append((dte, float(np.mean(ivs)), oi, root))
    return sorted(pillars)


# Vendor-matching term tenors, evaluated off the SAME fitted curve as ours30/ours90.
# A tenor is only emitted when the longest pillar covers most of it — sig(365) read
# off 140-day pillars is extrapolation, not a curve value, and stays NaN.
OWN_TENORS = [("1M", 30), ("3M", 91), ("6M", 182), ("12M", 365)]
TENOR_COVERAGE = 0.75


def build_book(tickers=None, log=print) -> pd.DataFrame:
    """Our fitted 30d + 90d ATM vol for every listed-futures product; long frame."""
    from .universe import INSTRUMENTS, asset, name
    if tickers is None:
        tickers = [t for t in INSTRUMENTS
                   if asset(t) not in EXCLUDE_ASSETS and not t.endswith("Index")
                   or t in ("ESA Index", "NQA Index", "RTYA Index", "DMA Index", "XPA Index")]
    bbg30 = {}
    try:
        iv = pd.read_parquet(ROOT / "data" / "snapshot" / "implied_vol.parquet").set_index("date")
        for t in iv.columns:
            s = iv[t].dropna()
            if len(s):
                bbg30[t] = float(s.iloc[-1])
    except Exception:
        pass
    recs = []
    for t in tickers:
        try:
            pil = product_pillars(t, log=log)
        except Exception as e:
            log(f"  {name(t):24s} ERROR {type(e).__name__}: {e}")
            continue
        if len(pil) < 2:
            log(f"  {name(t):24s} no usable pillars (no listed options, or dead marks)")
            continue
        trip = [(d, v, w) for d, v, w, _ in pil]
        sig = fit_curve(trip)
        method = "fit" if sig else "interp"
        ours30 = sig(30.0) if sig else interp_variance(trip, 30.0)
        ours90 = sig(90.0) if sig else interp_variance(trip, 90.0)
        # term tenors off the same curve, gated on actual pillar coverage
        max_d = trip[-1][0]
        tenors = {}
        for lab, days in OWN_TENORS:
            v = None
            if max_d >= TENOR_COVERAGE * days:
                v = sig(float(days)) if sig else interp_variance(trip, float(days))
            tenors[lab] = round(v, 2) if v is not None else np.nan
        detail = "; ".join(
            f"{r}@{d}d {v:.1f} (oi {int(w) if math.isfinite(w) else 0})" for d, v, w, r in pil)
        tstr = " ".join(f"{lab} {tenors[lab]:.1f}" for lab, _ in OWN_TENORS
                        if pd.notna(tenors[lab]) and lab not in ("1M",))
        log(f"  {name(t):24s} {len(pil)} pillars [{method}]  30d {ours30:6.2f}  90d {ours90:6.2f}"
            + (f"  | {tstr}" if tstr else "")
            + (f"  | BBG30 {bbg30[t]:6.2f}  diff {ours30 - bbg30[t]:+.2f}" if t in bbg30 else ""))
        rec = {"ticker": t, "market": name(t), "n_pillars": len(pil), "method": method,
               "ours30": round(ours30, 2), "ours90": round(ours90, 2),
               "bbg30": round(bbg30[t], 2) if t in bbg30 else np.nan,
               "pillars": detail}
        rec.update({f"own_{lab.lower()}": tenors[lab] for lab, _ in OWN_TENORS})
        recs.append(rec)
    df = pd.DataFrame(recs)
    if len(df):
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_FILE, index=False)
    return df


HISTORY_FILE = ROOT / "data" / "snapshot" / "own30_history.parquet"

# ---------------------------------------------------------------------------
# BACKFILL — reconstruct the daily curve over the past year from settlement
# prices. Expired options lose their chains and IV marks on Bloomberg, but keep
# SETTLEMENTS — so historical pillars are rebuilt from CONSTRUCTED tickers whose
# conventions (series-root pattern, strike grid step + string format, futures
# cycle, serial months) are LEARNED from each product's live chain today.
# ---------------------------------------------------------------------------
_MONTH_CODE = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
_CODE_MONTH = {v: k for k, v in _MONTH_CODE.items()}


def _bdh_safe(tickers, fld, start, end, chunk: int = 20):
    from .stircurve import _bdh_safe as _f
    return _f(tickers, fld, start, end, chunk=chunk)


def _stem_parts(stem: str):
    """'CLU6' -> ('CL','U',6,1); 'NGQ26' -> ('NG','Q',26,2); 'C U6' -> ('C ','U',6,1).
    Henry Hub lists futures YEARS out, so its chain uses two-digit years (and a naive
    one-digit ticker silently resolves to the 2030s contract) — options stay one-digit."""
    if len(stem) >= 4 and stem[-2:].isdigit() and stem[-3] in _CODE_MONTH:
        return stem[:-3], stem[-3], int(stem[-2:]), 2
    if len(stem) >= 3 and stem[-1].isdigit() and stem[-2] in _CODE_MONTH:
        return stem[:-2], stem[-2], int(stem[-1]), 1
    return None


def learn_conventions(generic: str):
    """Study the LIVE chains: the series stems in use, the strike grid + exact strike
    string formats, the futures cycle months, and which serial months exist."""
    futs_all = []
    for r in _bds(generic, "FUT_CHAIN"):
        tk = _first(r)
        if tk:
            futs_all.append(tk if tk.split()[-1] in ("Comdty", "Index") else tk + " Comdty")
    conv = {"prefix": None, "suffix": "Comdty", "cycle": set(), "serial_months": set(),
            "strike_fmts": set(), "steps": [], "futs": futs_all[:MAX_FUTS],
            "fut_yd": 1, "opt_yd": 1}
    # the delivery CYCLE needs the FULL listing — the first six futures of a monthly
    # product only span half the year, and historical series exist for every month
    for fut in futs_all:
        parts = _stem_parts(fut.rsplit(" ", 1)[0])
        if parts is not None and (conv["prefix"] is None or parts[0] == conv["prefix"]):
            conv["prefix"] = parts[0] if conv["prefix"] is None else conv["prefix"]
            conv["cycle"].add(parts[1])
            conv["fut_yd"] = max(conv["fut_yd"], parts[3])
    for fut in conv["futs"]:
        stem = fut.rsplit(" ", 1)[0]
        conv["suffix"] = fut.rsplit(" ", 1)[1]
        parts = _stem_parts(stem)
        if parts is None:
            continue
        prefix, code = parts[0], parts[1]
        conv["prefix"] = prefix
        conv["cycle"].add(code)
        raw_strikes = {}
        for r in _bds(fut, "OPT_CHAIN"):
            tk = _first(r)
            p = tk.replace(" Comdty", "").replace(" Index", "").split()
            if len(p) < 2:
                continue
            head, strike_s = " ".join(p[:-1]), p[-1]
            if head[-1] not in "CP":
                continue
            sp = _stem_parts(head[:-1])
            if sp is None or sp[0] != prefix:
                continue
            conv["opt_yd"] = max(conv["opt_yd"], sp[3])
            if sp[1] != code:
                conv["serial_months"].add(sp[1])
            try:
                k = float(strike_s)
            except ValueError:
                continue
            raw_strikes.setdefault(round(k, 6), strike_s)
        ks = sorted(raw_strikes)
        if len(ks) >= 3:
            diffs = [round(b - a, 6) for a, b in zip(ks, ks[1:]) if b - a > 1e-9]
            if diffs:
                conv["steps"].append(min(diffs))
            for s in list(raw_strikes.values())[:200]:
                dp = len(s.split(".")[1]) if "." in s else 0
                conv["strike_fmts"].add(dp)
    conv["step"] = min(conv["steps"]) if conv["steps"] else None
    return conv


def _fmt_strike(k: float, dps) -> list:
    """Candidate strike strings, most-likely first: the minimal representation (CL 84.5,
    corn 460), its leading-zero-stripped twin (Henry Hub lists 0.25 as '.25'), then the
    learned fixed-decimal forms. A wrong candidate just returns no data — never an error."""
    outs = []
    g = f"{k:g}"
    for s in ([g, g[1:]] if g.startswith("0.") else [g]):
        if s not in outs:
            outs.append(s)
    for dp in sorted(dps, reverse=True):
        s = f"{k:.{dp}f}"
        if s not in outs:
            outs.append(s)
    return outs[:3]


def backfill_product(generic: str, start: pd.Timestamp, end: pd.Timestamp, log=print):
    """Daily ours30/ours90 series for one product over [start, end], reconstructed
    from settlement prices of CONSTRUCTED historical option tickers."""
    conv = learn_conventions(generic)
    prefix, suffix, step = conv["prefix"], conv["suffix"], conv["step"]
    if prefix is None or step is None or not conv["cycle"]:
        log("    conventions not learnable — skipped")
        return pd.DataFrame()
    cycle = sorted(_CODE_MONTH[c] for c in conv["cycle"])
    serials = sorted(_CODE_MONTH[c] for c in conv["serial_months"])
    dps = conv["strike_fmts"] or {1}

    # every month-coded series stem that could carry an expiry in the window
    stems = []
    m = pd.Timestamp(start).to_period("M")
    while m <= pd.Timestamp(end).to_period("M") + 14:
        mm, yy = m.month, m.year
        if mm in cycle or mm in serials:
            und_m = min((c for c in cycle if c >= mm), default=None)
            und_y = yy
            if und_m is None:
                und_m, und_y = min(cycle), yy + 1
            oy, fy = conv.get("opt_yd", 1), conv.get("fut_yd", 1)
            stems.append((f"{prefix}{_MONTH_CODE[mm]}{yy % 100 if oy == 2 else yy % 10}",
                          f"{prefix}{_MONTH_CODE[und_m]}"
                          f"{und_y % 100 if fy == 2 else und_y % 10} {suffix}"))
        m += 1
    # underlying settle histories (expired futures keep settlements)
    und_needed = sorted({u for _, u in stems})
    upx = _bdh_safe(und_needed, "PX_SETTLE", f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", chunk=15)

    pillars_by_date: dict = {}
    used = 0
    for stem, und in stems:
        if und not in getattr(upx, "columns", []):
            continue
        f_s = upx[und].dropna()
        if f_s.empty:
            continue
        # option expiry from a constructed near-the-middle strike ticker
        k0 = round(round(float(f_s.median()) / step) * step, 6)
        exp = None
        for ks in _fmt_strike(k0, dps):
            exp = _bdp(f"{stem}C {ks} {suffix}", "OPT_EXPIRE_DT")
            if exp is not None:
                break
        if exp is None:
            continue
        exp = pd.Timestamp(exp)
        win = f_s[(f_s.index < exp) & (f_s.index >= exp - pd.Timedelta(days=220))]
        if win.empty:
            continue
        lo = math.floor((float(win.min()) - 2 * step) / step) * step
        hi = math.ceil((float(win.max()) + 2 * step) / step) * step
        n = int(round((hi - lo) / step)) + 1
        if n > 80:                                    # wild price range: coarsen the grid
            step_use = step * math.ceil(n / 80)
            n = int(round((hi - lo) / step_use)) + 1
        else:
            step_use = step
        grid = [round(lo + i * step_use, 6) for i in range(n)]
        tks = {}
        for k in grid:
            for ks in _fmt_strike(k, dps):
                tks[(k, "C", ks)] = f"{stem}C {ks} {suffix}"
                tks[(k, "P", ks)] = f"{stem}P {ks} {suffix}"
        w = _bdh_safe(list(tks.values()), "PX_LAST",
                      f"{win.index[0]:%Y-%m-%d}", f"{min(win.index[-1], exp):%Y-%m-%d}", chunk=25)
        cols = set(getattr(w, "columns", []))
        got = False
        for dt, F in win.items():
            dte = (exp - dt).days
            if dte < MIN_DTE:
                continue
            F = float(F)
            for k in sorted(grid, key=lambda x: abs(x - F))[:3]:
                ivs = []
                for cp, is_call in (("C", True), ("P", False)):
                    for ks in _fmt_strike(k, dps):
                        tk = tks.get((k, cp, ks))
                        if tk in cols:
                            v = w[tk].get(dt)
                            if pd.notna(v):
                                iv = implied_vol_price(float(v), F, k, dte, is_call)
                                if iv is not None and iv > 0.5:
                                    ivs.append(iv)
                                break
                if ivs:
                    pillars_by_date.setdefault(dt, []).append((dte, float(np.mean(ivs))))
                    got = True
                    break
        used += 1 if got else 0
    out = {}
    for dt, pil in pillars_by_date.items():
        pil = sorted(pil)[:N_EXPIRIES]
        trip = [(d, v, 1.0) for d, v in pil]
        sig = fit_curve(trip)
        o30 = sig(30.0) if sig else interp_variance(trip, 30.0)
        o90 = sig(90.0) if sig else interp_variance(trip, 90.0)
        if o30 is not None:
            out[dt] = (o30, o90)
    ser = pd.DataFrame(out.values(), index=pd.DatetimeIndex(out.keys()),
                       columns=["ours30", "ours90"]).sort_index()
    log(f"    {used} series usable -> {len(ser)} daily points"
        + (f", last 30d {ser['ours30'].iloc[-1]:.2f}" if len(ser) else ""))
    return ser


def _settle_date() -> pd.Timestamp:
    """The settle date our curve's inputs belong to — the snapshot's last price date
    (the inversion runs on PX_SETTLE). Stamping rows with THIS date lets the values
    overlay the implied frame on the right day for the report's spread/z splice."""
    try:
        px = pd.read_parquet(ROOT / "data" / "snapshot" / "prices.parquet")
        return pd.to_datetime(px["date"]).max().normalize()
    except Exception:
        return pd.Timestamp.today().normalize()


TERM_HISTORY_FILE = ROOT / "data" / "snapshot" / "own_term_history.parquet"


def append_today(log=print) -> None:
    """Daily accumulation (called from the snapshot pull): rebuild the book and
    upsert the settle-dated ours30/ours90/bbg30 per product into the history — the
    report's implied source (spliced with the surface's history) + validation trail.
    The per-tenor term values accrue the same way into own_term_history (long:
    date/ticker/tenor/vol) — the term book's own leg. History starts 2026-07-27;
    a per-tenor settlement backfill is a separate Bloomberg-budget decision."""
    df = build_book(log=log)
    if df.empty:
        return
    stamp = _settle_date()
    rows = df[["ticker", "ours30", "ours90", "bbg30"]].assign(date=stamp)
    try:
        old = pd.read_parquet(HISTORY_FILE)
        old["date"] = pd.to_datetime(old["date"])
        rows = pd.concat([old[old["date"] != stamp], rows], ignore_index=True)
    except Exception:
        pass
    rows.sort_values(["ticker", "date"]).to_parquet(HISTORY_FILE, index=False)

    tcols = [c for c in df.columns if c.startswith("own_")]
    if not tcols:
        return
    long = df.melt(id_vars=["ticker"], value_vars=tcols,
                   var_name="tenor", value_name="vol").dropna(subset=["vol"])
    long["tenor"] = long["tenor"].str.replace("own_", "", regex=False).str.upper()
    long["date"] = stamp
    try:
        old = pd.read_parquet(TERM_HISTORY_FILE)
        old["date"] = pd.to_datetime(old["date"])
        long = pd.concat([old[old["date"] != stamp], long], ignore_index=True)
    except Exception:
        pass
    long.sort_values(["ticker", "tenor", "date"]).to_parquet(TERM_HISTORY_FILE, index=False)
