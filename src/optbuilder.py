"""Option strategy builder maths (the Strategy Builder page).

A strategy is a list of LEGS, each a plain dict:

    {"side": "Buy"|"Sell", "qty": int, "kind": "Call"|"Put"|"Future",
     "strike": float, "days": float, "vol": float,   # vol in %, days to expiry
     "premium": float|None}                          # None -> model price at entry

Pricing is **Black-76 on the futures price** — the same convention as the Vol
Backtester (r = 0 by default: futures-style margining makes discounting noise
next to everything else; a rate can still be passed for stock-style comparisons).
A "Future" leg is the underlying itself: `strike` holds the entry price, vol /
days are ignored, value = F - entry.

Everything below works in PRICE POINTS per 1 unit of each leg; the page scales
by contract point value x qty for currency P&L. Expiry payoff = intrinsic;
the "T + d days" scenario line re-prices each leg with its remaining life.
"""
from __future__ import annotations

import math

VOL_FLOOR = 0.05          # % — keeps d1 finite on degenerate inputs
_SQ2 = math.sqrt(2.0)


def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQ2))


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black76(F: float, K: float, tau: float, sigma_pct: float, kind: str,
            r: float = 0.0) -> dict:
    """Black-76 value + greeks per 1 unit. `tau` in years, `sigma_pct` in %,
    `kind` 'Call'|'Put'. Greeks: delta per 1.00 of F, gamma per 1.00², vega per
    1 vol POINT, theta per CALENDAR DAY (both premium-currency price points).
    At tau <= 0 collapses to intrinsic with a step delta."""
    call = kind == "Call"
    df = math.exp(-r * max(tau, 0.0))
    if tau <= 0.0 or F <= 0.0 or K <= 0.0:
        intr = max(F - K, 0.0) if call else max(K - F, 0.0)
        live = (F > K) if call else (F < K)
        return {"price": intr, "delta": (1.0 if call else -1.0) if live else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sigma = max(sigma_pct, VOL_FLOOR) / 100.0
    st_ = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * st_ * st_) / st_
    d2 = d1 - st_
    if call:
        price, delta = F * _N(d1) - K * _N(d2), df * _N(d1)
    else:
        price, delta = K * _N(-d2) - F * _N(-d1), -df * _N(-d1)
    theta_yr = -F * _phi(d1) * sigma / (2.0 * math.sqrt(tau)) * df \
        + r * df * price                       # d(value)/dt, futures-style
    return {"price": df * price, "delta": delta,
            "gamma": df * _phi(d1) / (F * st_),
            "vega": df * F * _phi(d1) * math.sqrt(tau) / 100.0,
            "theta": theta_yr / 365.0}


# ── legs ────────────────────────────────────────────────────────────────────
def _sgn(leg: dict) -> float:
    return 1.0 if leg["side"] == "Buy" else -1.0


def leg_model(leg: dict, F: float, days_from_now: float = 0.0,
              r: float = 0.0) -> dict:
    """Model value + greeks of ONE leg per 1 unit (unsigned, premium excluded),
    `days_from_now` days into the trade. Future legs: value F - entry, delta 1."""
    if leg["kind"] == "Future":
        return {"price": F - float(leg["strike"]), "delta": 1.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    tau = (float(leg["days"]) - days_from_now) / 365.0
    return black76(F, float(leg["strike"]), tau, float(leg["vol"]), leg["kind"], r)


def entry_premium(leg: dict, F0: float, r: float = 0.0) -> float:
    """The premium a leg trades at: the user's if given, else the Black-76
    value at entry (futures: 0 — the entry price sits in `strike`)."""
    p = leg.get("premium")
    if p is not None and not (isinstance(p, float) and p != p):
        return float(p)
    if leg["kind"] == "Future":
        return 0.0
    return leg_model(leg, F0, 0.0, r)["price"]


def front_days(legs: list[dict]) -> float:
    """Days to the NEAREST option expiry — the horizon the 'at expiry' line,
    breakevens and max-P&L are all drawn at (longer-dated legs stay alive there,
    the standard convention for calendars)."""
    ds = [float(l["days"]) for l in legs if l["kind"] != "Future"]
    return min(ds) if ds else 0.0


def strategy_pnl(legs: list[dict], F: float, F0: float,
                 days_from_now: float | None = None, r: float = 0.0) -> float:
    """P&L in price points at underlying F: model value minus entry premium,
    summed over signed legs. `days_from_now=None` -> at the front expiry."""
    d = front_days(legs) if days_from_now is None else days_from_now
    total = 0.0
    for leg in legs:
        v = leg_model(leg, F, d, r)["price"]
        total += _sgn(leg) * float(leg["qty"]) * (v - entry_premium(leg, F0, r))
    return total


def net_premium(legs: list[dict], F0: float, r: float = 0.0) -> float:
    """Net outlay in price points: >0 = debit paid, <0 = credit received."""
    return sum(_sgn(leg) * float(leg["qty"]) * entry_premium(leg, F0, r)
               for leg in legs)


def totals_greeks(legs: list[dict], F: float, days_from_now: float = 0.0,
                  r: float = 0.0) -> dict:
    out = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for leg in legs:
        g = leg_model(leg, F, days_from_now, r)
        w = _sgn(leg) * float(leg["qty"])
        for k in out:
            out[k] += w * g[k]
    return out


# ── expiry-payoff analytics ─────────────────────────────────────────────────
def grid_range(legs: list[dict], F0: float) -> tuple[float, float]:
    """Price axis for the payoff chart: spot ±35% widened to clear every strike
    by 15% — wings and breakevens stay on screen."""
    ks = [float(l["strike"]) for l in legs] or [F0]
    lo = min(F0, min(ks)) * 0.65
    hi = max(F0, max(ks)) * 1.35
    return max(lo, 0.0), hi


def breakevens(legs: list[dict], F0: float, r: float = 0.0,
               n: int = 4001) -> list[float]:
    """Zero crossings of the EXPIRY payoff, linearly interpolated on a fine grid."""
    lo, hi = grid_range(legs, F0)
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ys = [strategy_pnl(legs, x, F0, None, r) for x in xs]
    # snap float noise to a true zero — deep-ITM premium round-off otherwise
    # reads as an endless run of phantom crossings on flat payoff segments
    tol = 1e-9 * (1.0 + sum(abs(entry_premium(l, F0, r)) * float(l["qty"]) for l in legs))
    ys = [0.0 if abs(y) < tol else y for y in ys]
    out = []
    for i in range(1, n):
        if ys[i - 1] == 0.0 and ys[i] != 0.0:
            out.append(xs[i - 1])
        elif (ys[i - 1] < 0) != (ys[i] < 0) and ys[i - 1] != 0.0:
            out.append(xs[i - 1] - ys[i - 1] * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
    # collapse near-duplicates from flat segments grazing zero
    dedup: list[float] = []
    for b in out:
        if not dedup or abs(b - dedup[-1]) > (hi - lo) / 500:
            dedup.append(b)
    return dedup


def tail_slopes(legs: list[dict]) -> tuple[float, float]:
    """(dPnL/dF as F -> 0+, as F -> inf) of the expiry payoff — signed units.
    Downside is always bounded (F stops at 0); upside is unbounded iff slope != 0."""
    dn = up = 0.0
    for leg in legs:
        w = _sgn(leg) * float(leg["qty"])
        if leg["kind"] == "Future":
            dn += w; up += w
        elif leg["kind"] == "Call":
            up += w
        else:                                   # Put: ITM as F -> 0
            dn -= w
    return dn, up


def max_profit_loss(legs: list[dict], F0: float, r: float = 0.0,
                    n: int = 4001) -> tuple[tuple[float, bool], tuple[float, bool]]:
    """((max profit, unbounded?), (max loss, unbounded?)) of the expiry payoff.
    Scans a fine grid INCLUDING F = 0, then lets the upside tail slope override."""
    lo, hi = grid_range(legs, F0)
    xs = [0.0] + [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    xs += sorted({float(l["strike"]) for l in legs})       # kinks are the extremes
    ys = [strategy_pnl(legs, x, F0, None, r) for x in xs]
    _, up = tail_slopes(legs)
    best, worst = max(ys), min(ys)
    return ((best, up > 1e-12), (worst, up < -1e-12))


def pop(legs: list[dict], F0: float, ref_vol_pct: float, horizon_days: float,
        r: float = 0.0, n: int = 2001) -> float | None:
    """Probability the EXPIRY payoff ends positive, under a lognormal terminal
    distribution at `ref_vol_pct` over `horizon_days` (drift-free forward).
    A model estimate, not a market-implied probability."""
    tau = horizon_days / 365.0
    if tau <= 0 or F0 <= 0:
        return None
    sigma = max(ref_vol_pct, VOL_FLOOR) / 100.0
    s = sigma * math.sqrt(tau)
    # integrate in z: F = F0 * exp(-s²/2 + s·z), z ~ N(0,1) over ±5σ
    prob = 0.0
    dz = 10.0 / (n - 1)
    for i in range(n):
        z = -5.0 + i * dz
        F = F0 * math.exp(-0.5 * s * s + s * z)
        if strategy_pnl(legs, F, F0, None, r) > 0:
            w = 1.0 if 0 < i < n - 1 else 0.5
            prob += w * _phi(z) * dz
    return min(max(prob, 0.0), 1.0)


# ── ATM term-curve helper ───────────────────────────────────────────────────
def vol_at(curve: dict, days: float) -> float:
    """ATM vol (%) at `days` from a {calendar_days: vol%} term curve — linear
    interpolation between tenors, flat extrapolation beyond the ends. The page
    feeds this the latest live surface points (1M/3M/6M/12M) so preset and
    blank-vol legs pick up the market's own term structure."""
    pts = sorted((float(d), float(v)) for d, v in curve.items())
    if not pts:
        raise ValueError("empty vol curve")
    if days <= pts[0][0]:
        return pts[0][1]
    if days >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= days <= d1:
            return v0 + (v1 - v0) * (days - d0) / (d1 - d0)
    return pts[-1][1]


# ── preset strategies ───────────────────────────────────────────────────────
def _leg(side, qty, kind, strike, days, vol) -> dict:
    return {"side": side, "qty": int(qty), "kind": kind, "strike": strike,
            "days": float(days), "vol": float(vol), "premium": None}


def _steps(F: float) -> float:
    """A sensible strike step ≈ 1% of spot, rounded to 1/2.5/5 x 10^k."""
    raw = F * 0.01
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _K(F: float, n_steps: float) -> float:
    """Strike n steps away from spot, snapped to the step grid."""
    s = _steps(F)
    return round(round(F / s) * s + n_steps * s, 10)


def atm_strike(F: float, n_steps: float = 0.0) -> float:
    """Public grid helper for the page's add-leg buttons — the ATM strike (or a
    strike `n_steps` grid steps away) on the same 1/2.5/5 x 10^k step grid the
    presets use."""
    return _K(F, n_steps)


# name -> builder(F, vol %, days) -> legs. Order = the preset menu.
PRESETS = {
    "Long Call":            lambda F, v, d: [_leg("Buy", 1, "Call", _K(F, 0), d, v)],
    "Short Call":           lambda F, v, d: [_leg("Sell", 1, "Call", _K(F, 0), d, v)],
    "Long Put":             lambda F, v, d: [_leg("Buy", 1, "Put", _K(F, 0), d, v)],
    "Short Put":            lambda F, v, d: [_leg("Sell", 1, "Put", _K(F, 0), d, v)],
    "Bull Call Spread":     lambda F, v, d: [_leg("Buy", 1, "Call", _K(F, 0), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 2), d, v)],
    "Bear Call Spread":     lambda F, v, d: [_leg("Sell", 1, "Call", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 2), d, v)],
    "Bull Put Spread":      lambda F, v, d: [_leg("Sell", 1, "Put", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Put", _K(F, -2), d, v)],
    "Bear Put Spread":      lambda F, v, d: [_leg("Buy", 1, "Put", _K(F, 0), d, v),
                                             _leg("Sell", 1, "Put", _K(F, -2), d, v)],
    "Long Straddle":        lambda F, v, d: [_leg("Buy", 1, "Call", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Put", _K(F, 0), d, v)],
    "Short Straddle":       lambda F, v, d: [_leg("Sell", 1, "Call", _K(F, 0), d, v),
                                             _leg("Sell", 1, "Put", _K(F, 0), d, v)],
    "Long Strangle":        lambda F, v, d: [_leg("Buy", 1, "Put", _K(F, -2), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 2), d, v)],
    "Short Strangle":       lambda F, v, d: [_leg("Sell", 1, "Put", _K(F, -2), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 2), d, v)],
    "Long Call Butterfly":  lambda F, v, d: [_leg("Buy", 1, "Call", _K(F, -2), d, v),
                                             _leg("Sell", 2, "Call", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 2), d, v)],
    "Iron Butterfly":       lambda F, v, d: [_leg("Buy", 1, "Put", _K(F, -2), d, v),
                                             _leg("Sell", 1, "Put", _K(F, 0), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 2), d, v)],
    "Long Call Condor":     lambda F, v, d: [_leg("Buy", 1, "Call", _K(F, -3), d, v),
                                             _leg("Sell", 1, "Call", _K(F, -1), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 1), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 3), d, v)],
    "Iron Condor":          lambda F, v, d: [_leg("Buy", 1, "Put", _K(F, -3), d, v),
                                             _leg("Sell", 1, "Put", _K(F, -1), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 1), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 3), d, v)],
    "Risk Reversal":        lambda F, v, d: [_leg("Sell", 1, "Put", _K(F, -2), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 2), d, v)],
    "Covered Call":         lambda F, v, d: [_leg("Buy", 1, "Future", round(F, 10), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 2), d, v)],
    "Collar":               lambda F, v, d: [_leg("Buy", 1, "Future", round(F, 10), d, v),
                                             _leg("Buy", 1, "Put", _K(F, -2), d, v),
                                             _leg("Sell", 1, "Call", _K(F, 2), d, v)],
    "Call Calendar Spread": lambda F, v, d: [_leg("Sell", 1, "Call", _K(F, 0), d, v),
                                             _leg("Buy", 1, "Call", _K(F, 0), d * 2, v)],
}
