"""Our OWN constant-maturity ATM vol for the LISTED-FUTURES book — the STIR-curve
construction (src/stircurve.py) generalised to every product with listed options.

Per product, per day:
  1. Walk the futures chain; for each front contract pull the live option chain and
     group it into option SERIES (quarterlies + serials each carry their own expiry).
  2. Per series: the two listed options BRACKETING each smile marker (ATM plus
     80/90/110/120% moneyness — puts below the money, calls above), settlement
     premiums inverted through Black-76 on the FUTURES PRICE (lognormal, ACT/365,
     no discounting — the same convention family as the STIR inversion, which
     works on the implied rate instead), then interpolated IN STRIKE to the exact
     marker level, so every point is true constant-moneyness rather than the
     nearest listed strike. The ATM pair's open interest is the pillar's weight.
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
MAX_FUTS = 6                 # ceiling: futures contracts to walk per product
LEAN_FUTS = 5                # the normal walk since 2026-08-18. The 6th chain is NOT
                             # always waste: in front-expiry weeks the front series dies
                             # on MIN_DTE and the 6th future's series becomes the 5th
                             # pillar (review proof: CL/DF marks of 2026-08-13). So the
                             # walk is 6 only when yesterday's marks show a series
                             # expiring within MIN_DTE+3 days (or no prior marks exist),
                             # else 5 — the buffer exactly when it's load-bearing.
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
    from . import pullguard
    pullguard.add_hits(1)                              # usage ledger: 1 bulk request
    try:
        return _rows(blp.bds(ticker, fld, **kw))
    except Exception:
        return []


def _bdp(ticker, fld):
    from xbbg import blp
    from .datafeed import _coerce_pd
    from . import pullguard
    pullguard.add_hits(1)
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


CHAIN_CACHE_FILE = ROOT / "data" / "snapshot" / "opt_chain_cache.json"
CHAIN_CACHE_DAYS = 7           # strike ladders change on listing cycles, not daily


def _chain_for(fut: str, force: bool = False):
    """_parse_chain behind a 7-day disk cache — the bds OPT_CHAIN discovery was
    ~250 requests every morning for strike lists that barely change. Returns
    (series, from_cache); the caller force-refreshes when the future's settle has
    walked off the cached ladder. A failed live pull falls back to a stale cached
    copy rather than dropping the product for the day."""
    import json as _json
    try:
        cache = _json.loads(CHAIN_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    def _load(ent):
        return {r: {float(k): tuple(v) for k, v in s.items()}
                for r, s in ent["series"].items()}

    ent = cache.get(fut)
    if ent and not force:
        try:
            age = (pd.Timestamp.today().normalize() - pd.Timestamp(ent["asof"])).days
            series = _load(ent)
            if age <= CHAIN_CACHE_DAYS and series:
                return series, True
        except Exception:
            pass
    series = _parse_chain(fut)
    if series:
        cache[fut] = {"asof": str(pd.Timestamp.today().date()),
                      "series": {r: {str(k): list(v) for k, v in s.items()}
                                 for r, s in series.items()}}
        try:
            CHAIN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CHAIN_CACHE_FILE.write_text(_json.dumps(cache), encoding="utf-8")
        except Exception:
            pass
        return series, False
    if ent:
        try:
            return _load(ent), True
        except Exception:
            pass
    return {}, False


def _strike_scale(strikes, F: float) -> float:
    """Strike/price UNIT reconciliation (RBOB futures print 2.38 while its strikes
    list in cents, etc.): pick the decimal scaling that puts the strike grid's
    median on the future's price scale. 1.0 when units already agree."""
    med = float(np.median(list(strikes)))
    if med <= 0 or F <= 0:
        return 1.0
    return min((1.0, 0.1, 0.01, 10.0, 100.0), key=lambda s: abs(math.log((med * s) / F)))


SKEW_WINGS = (0.90, 1.10)      # vendor-matching moneyness: the headline skew pair
WING_TOL = 0.03                # near markers: a bracketing strike within 3 mny pts
FAR_TOL = 0.05                 # far wings: strike grids coarsen away from the money
ATM_TOL = 0.05                 # ATM bracketing gate before the straddle fallback

# The five-point smile (Ben's 2026-08-18 spec): each marker's vol comes from the
# TWO listed options bracketing the exact level — puts below the money, calls
# above — inverted separately and interpolated in strike, replacing the single
# nearest-strike pick.   key -> (moneyness, is_call, tolerance)
SMILE_MARKERS = {"p80": (0.80, False, FAR_TOL), "p90": (0.90, False, WING_TOL),
                 "c110": (1.10, True, WING_TOL), "c120": (1.20, True, FAR_TOL)}
SMILE_MNY = {"p80": 0.80, "p90": 0.90, "atm": 1.00, "c110": 1.10, "c120": 1.20}


def fit_smile(points: dict):
    """The curve of best fit through the five smile marks (Ben's 2026-08-18 spec,
    second half): a quadratic in LOG-moneyness — the standard smile shape, whose
    linear term is the tilt (risk-reversal) and curvature term the butterfly.

    `points` maps any of p80/p90/atm/c110/c120 -> vol (NaN/missing tolerated —
    far wings are sparse by construction). Needs >=3 marks including the ATM.
    Returns (iv_at(moneyness_ratio) callable floored at 0.5 vol, params dict)
    or (None, {}) when underdetermined."""
    xs, ys = [], []
    for key, m in SMILE_MNY.items():
        v = points.get(key)
        if v is not None and pd.notna(v) and float(v) > 0:
            xs.append(math.log(m))
            ys.append(float(v))
    if len(xs) < 3 or not any(abs(x) < 1e-9 for x in xs):
        return None, {}
    co = np.polyfit(xs, ys, min(2, len(xs) - 1))
    def iv_at(mny: float) -> float:
        return float(max(np.polyval(co, math.log(mny)), 0.5))
    return iv_at, {"coeffs": [round(float(c), 4) for c in co], "n_marks": len(xs)}


def _bdp_many(tickers, fields) -> dict:
    """ONE batched bdp over many securities -> {ticker: {FIELD: value}}. The whole
    reason the Terminal-connected window shrank: the serial one-security _bdp calls
    were thousands of round-trips (10-20 min); batches take seconds."""
    from . import bbg as blp
    from .datafeed import _bdp_rows, _coerce_pd
    if not tickers:
        return {}
    try:
        return _bdp_rows(_coerce_pd(blp.bdp(list(tickers), list(fields)))) or {}
    except Exception:
        return {}


def _field(rowmap: dict, ticker: str, field: str):
    v = (rowmap.get(ticker) or {}).get(field.upper())
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def fetch_product_marks(generic: str, log=print, max_futs: int | None = None) -> dict:
    """The BLOOMBERG HALF of the pillar build for one product: walk the chains
    (option-chain lists via the 7-day disk cache), choose the BRACKETING option
    pair at every smile marker locally, and pull their fields in TWO batched
    requests. Per series: ATM = the OTM put just below the future's settle + the
    OTM call just above (one straddle when the settle lands exactly on a strike);
    each smile marker (80/90/110/120% moneyness) = the two listed strikes around
    the exact level. Assembly interpolates the inverted vols in strike, so every
    mark is true constant-moneyness. Returns a JSON-able marks record —
    everything the (pure-math) assembly needs, so the Terminal can be closed
    before any curve is fitted."""
    n_futs = MAX_FUTS if max_futs is None else max(1, min(int(max_futs), MAX_FUTS))
    futs = []
    for r in _bds(generic, "FUT_CHAIN"):
        tk = _first(r)
        if tk:
            futs.append(tk if tk.split()[-1] in ("Comdty", "Index") else tk + " Comdty")
        if len(futs) >= n_futs:
            break
    fmap = _bdp_many(futs, ["PX_SETTLE", "PX_LAST"])

    def _series_recs(F: float, chain: dict) -> tuple:
        recs, bracketed = [], False
        for root, strikes in sorted(chain.items()):
            scale = _strike_scale(strikes.keys(), F)   # strike + premium quote units
            calls = {k: v[0] for k, v in strikes.items() if v[0]}
            puts = {k: v[1] for k, v in strikes.items() if v[1]}
            atm_pts = []
            k_lo = max((k for k in puts if k * scale <= F), default=None)
            k_hi = min((k for k in calls if k * scale >= F), default=None)
            if k_lo is not None and (F - k_lo * scale) / F <= ATM_TOL:
                atm_pts.append({"k": float(k_lo), "tk": puts[k_lo], "cp": "P"})
            if k_hi is not None and (k_hi * scale - F) / F <= ATM_TOL:
                atm_pts.append({"k": float(k_hi), "tk": calls[k_hi], "cp": "C"})
            if atm_pts:
                bracketed = True           # a real near-money bracket on this chain
            else:                          # coarse grid: legacy nearest-straddle fallback
                both = {k: v for k, v in strikes.items() if v[0] and v[1]}
                if not both:
                    continue
                k0 = sorted(both, key=lambda k: abs(k * scale - F))[0]
                if abs(k0 * scale - F) / F > 0.25:     # nothing near the money — skip
                    continue
                atm_pts = [{"k": float(k0), "tk": both[k0][1], "cp": "P"},
                           {"k": float(k0), "tk": both[k0][0], "cp": "C"}]
            smile = {}
            for key, (m, is_call, tol) in SMILE_MARKERS.items():
                cands = calls if is_call else puts
                target = m * F
                k_lo = max((k for k in cands if k * scale <= target), default=None)
                k_hi = min((k for k in cands if k * scale >= target), default=None)
                pts = [{"k": float(k), "tk": cands[k], "cp": "C" if is_call else "P"}
                       for k in dict.fromkeys(k for k in (k_lo, k_hi) if k is not None)
                       if abs(k * scale - target) / F <= tol]
                if pts:
                    smile[key] = pts
            recs.append({"root": root, "F": F, "scale": scale,
                         "atm": atm_pts, "smile": smile})
        return recs, bracketed

    series_recs = []
    for fut in futs:
        F = _field(fmap, fut, "PX_SETTLE")
        F = _field(fmap, fut, "PX_LAST") if F is None else F
        if F is None:
            continue
        F = float(F)
        chain, cached = _chain_for(fut)
        recs, ok = _series_recs(F, chain)
        # Force-refresh the cached ladder when the settle has walked off it — either
        # completely (no records) or into fallback territory (no series bracketed
        # within ATM_TOL: the 25%-gate straddle fallback keeping recs alive must not
        # mask newly listed near-money strikes for a week; review 2026-08-18).
        if cached and (not recs or not ok):
            chain, _ = _chain_for(fut, force=True)
            recs, ok = _series_recs(F, chain)
        series_recs.extend(recs)
    atm_tks = [p["tk"] for r in series_recs for p in r["atm"]]
    wing_tks = [p["tk"] for r in series_recs
                for pts in r["smile"].values() for p in pts]
    amap = _bdp_many(atm_tks, ["PX_SETTLE", "OPT_EXPIRE_DT", "OPEN_INT"])
    wmap = _bdp_many(wing_tks, ["PX_SETTLE"])
    if atm_tks and not amap:                           # batch failed -> serial fallback
        log(f"  ({generic}: batched bdp empty — falling back to serial field pulls)")
        amap = {t: {"PX_SETTLE": _bdp(t, "PX_SETTLE"), "OPT_EXPIRE_DT": _bdp(t, "OPT_EXPIRE_DT"),
                    "OPEN_INT": _bdp(t, "OPEN_INT")} for t in atm_tks}
        wmap = {t: {"PX_SETTLE": _bdp(t, "PX_SETTLE")} for t in wing_tks}
    for rec in series_recs:                            # resolve fields into pure data
        exp = next((_field(amap, p["tk"], "OPT_EXPIRE_DT") for p in rec["atm"]
                    if _field(amap, p["tk"], "OPT_EXPIRE_DT") is not None), None)
        rec["exp"] = str(pd.Timestamp(exp).date()) if exp is not None else None
        oi = 0.0
        for p in rec["atm"]:
            v = _field(amap, p["tk"], "PX_SETTLE")
            p["px"] = float(v) if v is not None else None
            o = _field(amap, p["tk"], "OPEN_INT")
            oi += float(o) if o is not None else 0.0
        rec["oi"] = oi
        for pts in rec["smile"].values():
            for p in pts:
                v = _field(wmap, p["tk"], "PX_SETTLE")
                p["px"] = float(v) if v is not None else None
    return {"asof": str(pd.Timestamp.today().date()), "series": series_recs}


def _interp_marker(pts, target_px: float, F: float, scale: float, dte: float):
    """Bracketing option points -> ONE vol at the exact target strike: each settle
    inverted separately (dead-mark floor 0.5 vol), then LINEAR IN STRIKE between
    the two. A single usable point returns its own vol (the old nearest-strike
    behaviour, kept as the fallback); a straddle at one strike returns the mid.
    Interpolation is clamped to the bracket — never extrapolated."""
    vals = []
    for p in pts or []:
        if p.get("px") is None:
            continue
        iv = implied_vol_price(p["px"] * scale, F, p["k"] * scale, dte, p["cp"] == "C")
        if iv is not None and iv > 0.5:
            vals.append((p["k"] * scale, iv))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0][1]
    (k1, v1), (k2, v2) = sorted(vals)[:2]
    if k2 - k1 < 1e-12:
        return 0.5 * (v1 + v2)
    t = min(max((target_px - k1) / (k2 - k1), 0.0), 1.0)
    return v1 + (v2 - v1) * t


def assemble_pillars(marks: dict):
    """The PURE-MATH half: marks record -> (atm_pillars, skew_pillars,
    smile_pillars), keeping the legacy gating (MIN_DTE, expiry de-dup, N_EXPIRIES
    cap, dead-mark floors, tolerances). No Bloomberg — runs Terminal-closed.

    atm_pillars: [(dte, atm_vol%, oi_weight, series_root)] — ATM interpolated to
    the future's exact level from the bracketing OTM put/call (straddle mid when
    the settle sits on a strike — the legacy convention's degenerate case).
    skew_pillars: [(dte, put90, call110, atm)] — the headline vendor-convention
    pair; metric downstream = (put−call)/atm.
    smile_pillars: [(dte, {p80/p90/atm/c110/c120: vol})] — the five-point smile;
    markers whose far-OTM settles are dead marks simply drop out of the dict.
    Legacy marks records (pre-smile cache format) still assemble via the old
    nearest-strike fields, so a cached marks file from before the changeover
    keeps working."""
    today = pd.Timestamp.today().normalize()
    pillars, skews, smiles, seen_exp = [], [], [], set()
    for rec in (marks or {}).get("series", []):
        if len(pillars) >= N_EXPIRIES:
            break
        if not rec.get("exp"):
            continue
        F, scale = rec["F"], rec["scale"]
        dte = (pd.Timestamp(rec["exp"]) - today).days
        if dte < MIN_DTE or dte in seen_exp:
            continue
        if "atm" in rec:                               # new bracketing-format record
            atm_v = _interp_marker(rec["atm"], F, F, scale, dte)
            if atm_v is None:
                continue
            seen_exp.add(dte)
            atm_v = float(atm_v)
            pillars.append((dte, atm_v, rec.get("oi", 0.0), rec["root"]))
            sm = {"atm": atm_v}
            for key, (m, _c, _t) in SMILE_MARKERS.items():
                v = _interp_marker(rec.get("smile", {}).get(key), m * F, F, scale, dte)
                if v is not None:
                    sm[key] = float(v)
            smiles.append((dte, sm))
            if "p90" in sm and "c110" in sm:
                skews.append((dte, sm["p90"], sm["c110"], atm_v))
            continue
        # legacy record: nearest-strike straddle + single-strike wings
        atm = rec["atm_k"]
        ivs = [v for v in (
            implied_vol_price(rec["c_px"] * scale, F, atm * scale, dte, True)
            if rec.get("c_px") is not None else None,
            implied_vol_price(rec["p_px"] * scale, F, atm * scale, dte, False)
            if rec.get("p_px") is not None else None,
        ) if v is not None and v > 0.5]                # floor-vol prints are dead marks, not vol
        if not ivs:
            continue
        seen_exp.add(dte)
        atm_mid = float(np.mean(ivs))
        pillars.append((dte, atm_mid, rec.get("oi", 0.0), rec["root"]))
        wings = {}
        for leg, kk, pk in (("put", "wp_k", "wp_px"), ("call", "wc_k", "wc_px")):
            if rec.get(kk) is not None and rec.get(pk) is not None:
                iv = implied_vol_price(rec[pk] * scale, F, rec[kk] * scale, dte, leg == "call")
                if iv is not None and iv > 0.5:
                    wings[leg] = iv
        sm = {"atm": atm_mid}
        if "put" in wings:
            sm["p90"] = wings["put"]
        if "call" in wings:
            sm["c110"] = wings["call"]
        smiles.append((dte, sm))
        if "put" in wings and "call" in wings:
            skews.append((dte, wings["put"], wings["call"], atm_mid))
    return sorted(pillars), sorted(skews), sorted(smiles, key=lambda s: s[0])


def product_pillars(generic: str, log=print):
    """Legacy one-call path (fetch + assemble in one go) — kept for the backfill and
    any caller that wants the old behaviour with a Terminal up."""
    return assemble_pillars(fetch_product_marks(generic, log=log))


# Vendor-matching term tenors, evaluated off the SAME fitted curve as ours30/ours90.
# A tenor is only emitted when the longest pillar covers most of it — sig(365) read
# off 140-day pillars is extrapolation, not a curve value, and stays NaN.
OWN_TENORS = [("1M", 30), ("3M", 91), ("6M", 182), ("12M", 365)]
TENOR_COVERAGE = 0.75


MARKS_FILE = ROOT / "data" / "snapshot" / "owncurve_marks.json"


def _book_tickers():
    from .universe import INSTRUMENTS, asset
    return [t for t in INSTRUMENTS
            if asset(t) not in EXCLUDE_ASSETS and not t.endswith("Index")
            or t in ("ESA Index", "NQA Index", "RTYA Index", "DMA Index", "XPA Index")]


def fetch_marks(tickers=None, log=print) -> dict:
    """The BLOOMBERG PHASE for the whole book: batched marks for every product,
    cached to owncurve_marks.json. Once this returns, the Terminal can be CLOSED —
    build_book(use_marks=True) / append_today(use_marks=True) fit the curves from
    the cache without any Bloomberg."""
    import json as _json
    from .universe import name

    def _walk_depth(prior: dict | None) -> int:
        """LEAN_FUTS normally; MAX_FUTS in front-expiry week (a prior series expires
        within MIN_DTE+3 days) or when there are no prior marks to judge by — the
        6th future's series is exactly the buffer pillar in those weeks."""
        try:
            today = pd.Timestamp.today().normalize()
            dtes = [(pd.Timestamp(s["exp"]) - today).days
                    for s in (prior or {}).get("series", []) if s.get("exp")]
            return LEAN_FUTS if dtes and min(dtes) >= MIN_DTE + 3 else MAX_FUTS
        except Exception:
            return MAX_FUTS
    try:                                   # yesterday's marks, ANY age — depth hint only
        prior_products = _json.loads(MARKS_FILE.read_text(encoding="utf-8"))["products"]
    except Exception:
        prior_products = {}

    out = {}
    for t in (tickers or _book_tickers()):
        try:
            m = fetch_product_marks(t, log=log, max_futs=_walk_depth(prior_products.get(t)))
            if m.get("series"):
                out[t] = m
                log(f"  {name(t):24s} marks: {len(m['series'])} series fetched")
            else:
                log(f"  {name(t):24s} no option series found")
        except Exception as e:
            log(f"  {name(t):24s} marks FAILED {type(e).__name__}: {e}")
    payload = {"asof": str(pd.Timestamp.today().date()), "products": out}
    MARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKS_FILE.write_text(_json.dumps(payload), encoding="utf-8")
    log(f"  marks cached for {len(out)} products -> {MARKS_FILE.name}")
    return payload


def load_marks(max_age_days: int = 1) -> dict | None:
    """The cached marks when fresh enough (default: today/yesterday), else None."""
    import json as _json
    try:
        payload = _json.loads(MARKS_FILE.read_text(encoding="utf-8"))
        age = (pd.Timestamp.today().normalize() - pd.Timestamp(payload["asof"])).days
        return payload if age <= max_age_days else None
    except Exception:
        return None


def build_book(tickers=None, log=print, use_marks: bool = False) -> pd.DataFrame:
    """Our fitted 30d + 90d ATM vol for every listed-futures product; long frame.
    use_marks=True fits from the cached marks (fetch_marks) — NO Bloomberg needed."""
    from .universe import name
    if tickers is None:
        tickers = _book_tickers()
    marks_products = None
    if use_marks:
        payload = load_marks()
        if payload is None:
            log("  no fresh marks cache (run the fetch phase first) — nothing to fit")
            return pd.DataFrame()
        marks_products = payload["products"]
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
            if marks_products is not None:
                m = marks_products.get(t)
                if not m:
                    log(f"  {name(t):24s} no marks in cache")
                    continue
                pil, skw, sml = assemble_pillars(m)
            else:
                pil, skw, sml = product_pillars(t, log=log)
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
        # own skew at 30d: each wing interpolated (total variance) across its own
        # expiry pillars; denominator = the book's fitted ours30 so the metric stays
        # coherent with the vol book. Needs both wings on >=2 expiries — bonds etc.
        # legitimately NaN out (their vendor wings are model extrapolation anyway).
        put30 = call30 = skew30 = np.nan
        if len(skw) >= 2 and ours30 is not None:
            ptrip = [(d, pv, 1.0) for d, pv, _, _ in skw]
            ctrip = [(d, cv, 1.0) for d, _, cv, _ in skw]
            p_v, c_v = interp_variance(ptrip, 30.0), interp_variance(ctrip, 30.0)
            if p_v is not None and c_v is not None and ours30 > 0:
                put30, call30 = round(p_v, 2), round(c_v, 2)
                skew30 = round((p_v - c_v) / ours30, 4)
        # far wings (80/120% mny) at constant 30d — sparse by nature: a marker only
        # exists on expiries whose far-OTM settles carry real premium, so quiet
        # products legitimately NaN out here while NG etc. populate.
        far = {}
        for key in ("p80", "c120"):
            trip = [(d, s[key], 1.0) for d, s in sml if key in s]
            v = interp_variance(trip, 30.0) if len(trip) >= 2 else None
            far[key] = round(v, 2) if v is not None else np.nan
        rec = {"ticker": t, "market": name(t), "n_pillars": len(pil), "method": method,
               "ours30": round(ours30, 2), "ours90": round(ours90, 2),
               "bbg30": round(bbg30[t], 2) if t in bbg30 else np.nan,
               "pillars": detail,
               "own_put30": put30, "own_call30": call30, "own_skew30": skew30,
               "own_put80": far["p80"], "own_call120": far["c120"],
               "n_skew_pillars": len(skw)}
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


def backfill_skew_product(generic: str, start: pd.Timestamp, end: pd.Timestamp, log=print):
    """Daily constant-30d wing vols (put90 / call110) for one product over [start, end],
    reconstructed from settlement prices of CONSTRUCTED wing-strike tickers. Cheaper
    than the ATM backfill per product: only the two moneyness BANDS are pulled (puts
    0.87–0.93×range, calls 1.07–1.13×range), and the ATM denominator comes free from
    own30_history. Returns (date, put, call, atm, skew) after the same warm-up trim
    and collapse filter as the ATM reconstruction."""
    conv = learn_conventions(generic)
    prefix, suffix, step = conv["prefix"], conv["suffix"], conv["step"]
    if prefix is None or step is None or not conv["cycle"]:
        log("    conventions not learnable — skipped")
        return pd.DataFrame()
    cycle = sorted(_CODE_MONTH[c] for c in conv["cycle"])
    serials = sorted(_CODE_MONTH[c] for c in conv["serial_months"])
    dps = conv["strike_fmts"] or {1}

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
    und_needed = sorted({u for _, u in stems})
    upx = _bdh_safe(und_needed, "PX_SETTLE", f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", chunk=15)

    def _band(lo_f, hi_f):
        lo = math.floor(lo_f / step) * step
        hi = math.ceil(hi_f / step) * step
        n = int(round((hi - lo) / step)) + 1
        s_use = step * math.ceil(n / 40) if n > 40 else step   # cap ~40 strikes per band
        n = int(round((hi - lo) / s_use)) + 1
        return [round(lo + i * s_use, 6) for i in range(n)]

    pillars_by_date: dict = {}
    for stem, und in stems:
        if und not in getattr(upx, "columns", []):
            continue
        f_s = upx[und].dropna()
        if f_s.empty:
            continue
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
        lo_w, hi_w = float(win.min()), float(win.max())
        legs = {"P": _band(SKEW_WINGS[0] * lo_w * 0.97, SKEW_WINGS[0] * hi_w * 1.03),
                "C": _band(SKEW_WINGS[1] * lo_w * 0.97, SKEW_WINGS[1] * hi_w * 1.03)}
        tks = {}
        for cp, grid in legs.items():
            for k in grid:
                for ks in _fmt_strike(k, dps):
                    tks[(cp, k, ks)] = f"{stem}{cp} {ks} {suffix}"
        w = _bdh_safe(list(tks.values()), "PX_LAST",
                      f"{win.index[0]:%Y-%m-%d}", f"{min(win.index[-1], exp):%Y-%m-%d}", chunk=25)
        cols = set(getattr(w, "columns", []))

        def _leg(dt, F, dte, cp, target):
            for k in sorted(legs[cp], key=lambda x: abs(x - target * F))[:3]:
                if abs(k / F - target) > WING_TOL:
                    break
                for ks in _fmt_strike(k, dps):
                    tk = tks.get((cp, k, ks))
                    if tk in cols:
                        v = w[tk].get(dt)
                        if pd.notna(v):
                            iv = implied_vol_price(float(v), F, k, dte, cp == "C")
                            return iv if iv is not None and iv > 0.5 else None
                        break
            return None

        for dt, F in win.items():
            dte = (exp - dt).days
            if dte < MIN_DTE:
                continue
            F = float(F)
            pw = _leg(dt, F, dte, "P", SKEW_WINGS[0])
            cw = _leg(dt, F, dte, "C", SKEW_WINGS[1])
            if pw is not None and cw is not None:
                pillars_by_date.setdefault(dt, []).append((dte, pw, cw))

    rows = {}
    for dt, pil in pillars_by_date.items():
        pil = sorted(pil)[:N_EXPIRIES]
        if len(pil) < 2:
            continue
        p30 = interp_variance([(d, pv, 1.0) for d, pv, _ in pil], 30.0)
        c30 = interp_variance([(d, cv, 1.0) for d, _, cv in pil], 30.0)
        if p30 is not None and c30 is not None:
            rows[dt] = (round(p30, 2), round(c30, 2))
    if not rows:
        log("    no wing pillars reconstructed")
        return pd.DataFrame()
    out = pd.DataFrame(rows.values(), index=pd.DatetimeIndex(rows.keys()),
                       columns=["put", "call"]).sort_index()

    # denominator: the already-backfilled ATM 30d, joined per settle date
    try:
        atm = pd.read_parquet(HISTORY_FILE)
        atm = atm[atm["ticker"] == generic].assign(date=lambda d: pd.to_datetime(d["date"]))
        out = out.join(atm.set_index("date")["ours30"].rename("atm"), how="inner")
    except Exception:
        return pd.DataFrame()
    out = out[out["atm"] > 0]
    out["skew"] = ((out["put"] - out["call"]) / out["atm"]).round(4)

    out = out.iloc[21:]                                   # strike-grid warm-up trim
    med = out["put"].rolling(21, center=True, min_periods=5).median()
    out = out[~(out["put"] < 0.35 * med)]                 # single-day fit collapses
    log(f"    {len(out)} daily skew points"
        + (f", last {out['skew'].iloc[-1]:+.3f}" if len(out) else ""))
    return out


SKEW_ATTEMPTS_FILE = ROOT / "data" / "snapshot" / "own_skew_backfill_attempts.json"
MAX_SKEW_ATTEMPTS = 2      # a product gets two full tries; quarterly/thin-wing books
                           # legitimately max out below a year — validation judges them


def _skew_attempts() -> dict:
    try:
        import json
        return json.loads(SKEW_ATTEMPTS_FILE.read_text())
    except Exception:
        return {}


def _wing_capable() -> list:
    try:
        book = pd.read_parquet(OUT_FILE)
        return (list(book[book["n_skew_pillars"] >= 2]["ticker"])
                if "n_skew_pillars" in book.columns else list(book["ticker"]))
    except Exception:
        return []


def skew_backfill_progress() -> tuple:
    """(products with a full year of own skew history, wing-capable products in the
    book) — shared by the app's progress note and the Home banner."""
    tickers = _wing_capable()
    if not tickers:
        return 0, 0
    try:
        h = pd.read_parquet(SKEW_HISTORY_FILE)
        counts = h.groupby("ticker")["date"].nunique()
    except Exception:
        counts = pd.Series(dtype=int)
    return sum(1 for t in tickers if int(counts.get(t, 0)) >= 120), len(tickers)


def skew_backfill_remaining() -> int:
    """Products the drip still intends to work on: under a year of history AND not
    yet retried out. 0 = the drip is finished (some products legitimately below a
    full year — quarterly expiries / sparse wings reconstruct all they can)."""
    tickers = _wing_capable()
    if not tickers:
        return 0
    try:
        h = pd.read_parquet(SKEW_HISTORY_FILE)
        counts = h.groupby("ticker")["date"].nunique()
    except Exception:
        counts = pd.Series(dtype=int)
    att = _skew_attempts()
    return sum(1 for t in tickers
               if int(counts.get(t, 0)) < 120 and int(att.get(t, 0)) < MAX_SKEW_ATTEMPTS)


def skew_backfill_drip(max_products: int = 2, min_days_done: int = 120,
                       start: str = "2025-06-16", log=print) -> int:
    """Backfill a FEW products' skew history per morning. Rate history: launched at
    8/day (2026-08-08); cut to 2/day (Ben, 2026-08-10) after daily WORKFLOW-review
    notices — ~2-3k requests keeps the anomaly under the review radar at the cost of
    ~16 mornings to finish. Runs in the snapshot FETCH phase; picks the next
    unfinished products and stops. Returns products done this run (0 = complete).
    Products whose LIVE build finds no wings (short-duration bonds — ±10%-moneyness
    strikes carry no real marks) are excluded rather than retried."""
    try:
        book = pd.read_parquet(OUT_FILE)
    except Exception:
        return 0
    tickers = list(book["ticker"])
    if "n_skew_pillars" in book.columns:                  # live-build evidence of wings
        tickers = list(book[book["n_skew_pillars"] >= 2]["ticker"])
    try:
        h = pd.read_parquet(SKEW_HISTORY_FILE)
        counts = h.groupby("ticker")["date"].nunique()
    except Exception:
        counts = pd.Series(dtype=int)
    att = _skew_attempts()
    todo = [t for t in tickers
            if int(counts.get(t, 0)) < min_days_done
            and int(att.get(t, 0)) < MAX_SKEW_ATTEMPTS][:max_products]
    if not todo:
        return 0
    end, done = _settle_date(), 0
    for t in todo:
        att[t] = int(att.get(t, 0)) + 1                # count the try BEFORE running:
        try:                                            # a crash must not retry forever
            import json
            SKEW_ATTEMPTS_FILE.write_text(json.dumps(att, indent=0))
        except Exception:
            pass
        try:
            df = backfill_skew_product(t, pd.Timestamp(start), end, log=log)
        except Exception as e:
            log(f"    skew drip {t}: FAILED {type(e).__name__}: {e}")
            continue
        if df is None or df.empty:
            continue
        rows = df.reset_index().rename(columns={"index": "date"})
        rows["ticker"] = t
        try:
            old = pd.read_parquet(SKEW_HISTORY_FILE)
            old["date"] = pd.to_datetime(old["date"])
            rows = pd.concat([rows, old], ignore_index=True)   # live rows win on collision
            rows = rows.drop_duplicates(subset=["ticker", "date"], keep="last")
        except Exception:
            pass
        rows.sort_values(["ticker", "date"]).to_parquet(SKEW_HISTORY_FILE, index=False)
        done += 1
    log(f"    skew drip: {done}/{len(todo)} products backfilled this run")
    return done


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
SKEW_HISTORY_FILE = ROOT / "data" / "snapshot" / "own_skew_history.parquet"


def append_today(log=print, use_marks: bool = False) -> None:
    """Daily accumulation (called from the snapshot pull): rebuild the book and
    upsert the settle-dated ours30/ours90/bbg30 per product into the history — the
    report's implied source (spliced with the surface's history) + validation trail.
    The per-tenor term values accrue the same way into own_term_history (long:
    date/ticker/tenor/vol) — the term book's own leg (history from 2026-07-27) —
    and the 30d wing/skew values into own_skew_history (recording-only from
    2026-07-28: the Skew page stays vendor until this is backfilled/deep enough).
    Any settlement backfill is a separate Bloomberg-budget decision.
    use_marks=True fits from the cached marks (fetch_marks) — Terminal-closed."""
    df = build_book(log=log, use_marks=use_marks)
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

    tcols = [f"own_{lab.lower()}" for lab, _ in OWN_TENORS if f"own_{lab.lower()}" in df.columns]
    if tcols:
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

    # own skew — RECORDING ONLY for now (the Skew page stays on the vendor surface
    # until this history is backfilled or deep enough to z-score; app carries a note)
    if "own_skew30" in df.columns:
        cols = ["ticker", "own_put30", "own_call30", "own_skew30"] + \
               [c for c in ("own_put80", "own_call120") if c in df.columns]
        sk = df[cols].dropna(subset=["own_skew30"])
        if len(sk):
            sk = sk.rename(columns={"own_put30": "put", "own_call30": "call",
                                    "own_skew30": "skew", "own_put80": "put80",
                                    "own_call120": "call120"}).assign(
                atm=df.set_index("ticker").loc[sk["ticker"], "ours30"].values, date=stamp)
            try:
                old = pd.read_parquet(SKEW_HISTORY_FILE)
                old["date"] = pd.to_datetime(old["date"])
                sk = pd.concat([old[old["date"] != stamp], sk], ignore_index=True)
            except Exception:
                pass
            sk.sort_values(["ticker", "date"]).to_parquet(SKEW_HISTORY_FILE, index=False)
