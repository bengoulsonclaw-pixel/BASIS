"""STIR Paths engine (BASIS · STIR Paths module).

Generalises the Fed Path machinery (src/fedpath.py) across the desk's whole STIR
book — three central banks, six products:

    FED   SR3 (3M SOFR, quarterly, compounded)   SR1 (1M SOFR, monthly, simple avg)
          FF  (30-Day Fed Funds, monthly, simple avg)
    ECB   ER  (3M Euribor, forward-looking term)  TKYA (3M €STR, quarterly, compounded,
          futures-only — no listed options)
    BOE   SFI (3M SONIA, quarterly, compounded)

Four jobs, all off ONE step-function policy-path model per bank:

  1.  EXPIRY TIMELINE  — every monthly futures + options expiry for each product
      (generated from the holiday-aware rules in src/expiries.py), overlaid with
      the bank's rate-decision dates. The point the module exists to make: how
      many decisions sit inside each contract's window / before each option expiry.
  2.  MARKET-IMPLIED   — invert each bank's strip into the meeting-step path it
      prices (fedpath's least-squares bootstrap, meetings parameterised per bank),
      plus FedWatch-style per-meeting odds (implied move ÷ step size).
  3.  YOUR SCENARIO    — per-meeting % chance of hike / cut (with step sizes) →
      probability-weighted expected path → fair value of every contract, and the
      landing price of each option's UNDERLYING at that option's expiry.
  4.  MEETING COUNTS   — decisions inside each contract window and each option's
      life, as data for the timeline + tables.

Modelling notes
---------------
* SR3 / SFI / €STR settle to the backward-COMPOUNDED overnight index over their
  IMM quarter — exactly fedpath's convention, reused directly. SR1 / FF settle to
  the simple calendar-month average.
* Euribor is a FORWARD-looking term rate fixed at the start of its window, so the
  fix embeds meeting expectations. Under a deterministic scenario path expected =
  realised, so ER prices as compounded-expected €STR over the same IMM window
  plus a Euribor–€STR basis (the `spread_bp` product field, user-tunable on the
  page). Good to ~a bp; footnoted in the UI.
* Under a deterministic path a future's fair price does not drift as time passes
  (no risk premium modelled), so "where the future lands at the option expiry" =
  the scenario fair value of the underlying — what the landing table shows.
* Probability-weighted landing uses the EXPECTED move per meeting
  (p_hike·hike − p_cut·cut). Futures are ~linear in the path (compounding
  convexity <~1bp/quarter), so pricing at the expected path ≈ expected price.

Pure (datetime + numpy + the expiries rule tables) → unit-testable; Bloomberg is
a thin adapter with the usual mock fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from . import expiries as _exp
from . import fedpath
from .fedpath import (Contract, _add_months, _daterange, contract_rate,  # noqa: F401
                      effective_date, price, overnight_rate_fn, third_wednesday)

# ── central-bank meeting calendars ────────────────────────────────────────────────────
# Decision (announcement) dates. FED: fedpath.FOMC_DECISIONS (federalreserve.gov).
# ECB: Governing Council MONETARY POLICY meetings, decision = second day, 14:15 CET
#      (ecb.europa.eu press calendar; 2026 full year + 2027 as published 2026-08).
# BOE: MPC announcement Thursdays, 12:00 London (bankofengland.co.uk; 2027 provisional).
# Extend these lists as the banks publish new years; everything downstream keys off them.
ECB_DECISIONS: list[date] = [
    date(2026, 2, 5), date(2026, 3, 19), date(2026, 4, 30), date(2026, 6, 11),
    date(2026, 7, 23), date(2026, 9, 10), date(2026, 10, 29), date(2026, 12, 17),
    date(2027, 2, 4), date(2027, 3, 18), date(2027, 4, 29), date(2027, 6, 10),
    date(2027, 7, 22), date(2027, 9, 9), date(2027, 10, 28), date(2027, 12, 16),
]
BOE_DECISIONS: list[date] = [
    date(2026, 2, 5), date(2026, 3, 19), date(2026, 4, 30), date(2026, 6, 18),
    date(2026, 7, 30), date(2026, 9, 17), date(2026, 11, 5), date(2026, 12, 17),
    date(2027, 2, 4), date(2027, 3, 18), date(2027, 4, 29), date(2027, 6, 17),
    date(2027, 7, 29), date(2027, 9, 16), date(2027, 11, 4), date(2027, 12, 16),
]

_MONTH_CODE = "FGHJKMNQUVXZ"                     # Jan..Dec futures month codes
_MONTHS = fedpath._MONTHS


@dataclass(frozen=True)
class Bank:
    key: str            # 'FED' / 'ECB' / 'BOE'
    name: str           # display
    meeting_name: str   # 'FOMC' / 'Governing Council' / 'MPC'
    rate_name: str      # what the policy lever is called
    meetings: list[date] = field(hash=False, default_factory=list)
    default_rate: float = 4.0        # seed for the policy-rate input (%)
    step_bp: float = 25.0            # conventional move size
    ccy: str = "$"


BANKS: dict[str, Bank] = {
    "FED": Bank("FED", "Federal Reserve", "FOMC", "Target band midpoint",
                fedpath.FOMC_DECISIONS, 4.375, 25.0, "$"),
    "ECB": Bank("ECB", "European Central Bank", "Governing Council", "Deposit facility rate",
                ECB_DECISIONS, 2.00, 25.0, "€"),
    "BOE": Bank("BOE", "Bank of England", "MPC", "Bank Rate",
                BOE_DECISIONS, 4.00, 25.0, "£"),
}


@dataclass(frozen=True)
class Product:
    ticker: str         # universe ticker ('SFRA Comdty')
    name: str           # display ('3M SOFR (SR3)')
    short: str          # tag for tables/legends ('SR3')
    bank: str           # BANKS key
    root: str           # Bloomberg contract root → specific tickers like 'SFRU6 Comdty'
                        # (all six roots verified on the Terminal 2026-08-10)
    family: str         # expiries.SPECS key (expiry rules)
    quarterly: bool     # IMM Mar/Jun/Sep/Dec strip (else monthly calendar contracts)
    compound: bool      # backward-compounded settlement (else simple monthly average)
    bp_value: float     # contract currency per bp (SR3 $25, SR1/FF $41.67, ER/€STR €25, SFI £25)
    color: str          # timeline series colour (dark-theme-safe)
    has_options: bool = True
    spread_bp: float = 0.0    # settlement-index spread vs the bank's overnight proxy
    in_strip: bool = True     # part of the bank's default path-fitting strip


PRODUCTS: dict[str, Product] = {
    "SFRA Comdty": Product("SFRA Comdty", "3M SOFR (SR3)", "SR3", "FED", "SFR",
                           "sofr3m", True, True, 25.0, "#F5C518"),
    "SERA Comdty": Product("SERA Comdty", "1M SOFR (SR1)", "SR1", "FED", "SER",
                           "sofr1m", False, False, 41.67, "#64B5F6", in_strip=False),
    "FFA Comdty":  Product("FFA Comdty", "30-Day Fed Funds", "FF", "FED", "FF",
                           "fedfunds", False, False, 41.67, "#81C784", in_strip=False),
    "ERA Comdty":  Product("ERA Comdty", "3M Euribor", "ER", "ECB", "ER",
                           "euribor", True, True, 25.0, "#BA68C8", spread_bp=10.0),
    "TKYA Comdty": Product("TKYA Comdty", "3M €STR", "€STR", "ECB", "TKY",
                           "estr", True, True, 25.0, "#4DD0E1", has_options=False),
    "SFIA Comdty": Product("SFIA Comdty", "3M SONIA", "SONIA", "BOE", "SFI",
                           "sonia", True, True, 25.0, "#FF8A65"),
}
# Overnight proxy vs the policy rate, in bp (page-tunable; these seed the input):
# SOFR ≈ target mid + 0 · €STR ≈ depo − 8 · SONIA ≈ Bank Rate − 5.
BANK_BASIS_SEED = {"FED": 0.0, "ECB": -8.0, "BOE": -5.0}


def bank_products(bank: str) -> list[Product]:
    return [p for p in PRODUCTS.values() if p.bank == bank]


# ── contract construction ────────────────────────────────────────────────────────────
def _code(root: str, y: int, m: int) -> str:
    return f"{root}{_MONTH_CODE[m - 1]}{y % 10}"


def quarterly_contract(prod: Product, year: int, month: int) -> Contract:
    """IMM quarterly: window = [3rd Wed of the IMM month, 3rd Wed +3M)."""
    ey, em = _add_months(year, month, 3)
    return Contract(_code(prod.root, year, month),
                    f"{_MONTHS[month - 1]}-{year % 100:02d}", year, month,
                    third_wednesday(year, month), third_wednesday(ey, em))


def monthly_contract(prod: Product, year: int, month: int) -> Contract:
    """Calendar-month contract (SR1 / FF): window = the calendar month."""
    ey, em = _add_months(year, month, 1)
    return Contract(_code(prod.root, year, month),
                    f"{_MONTHS[month - 1]}-{year % 100:02d}", year, month,
                    date(year, month, 1), date(ey, em, 1))


def strip(prod: Product, asof: date, n: int = 8) -> list[Contract]:
    """The front `n` contracts from `asof` — currently-accruing first."""
    if prod.quarterly:
        y, m = asof.year, ((asof.month - 1) // 3) * 3 + 3
        while third_wednesday(y, m) > asof:
            y, m = _add_months(y, m, -3)
        step = 3
    else:
        y, m = asof.year, asof.month
        step = 1
    out = []
    for _ in range(n):
        out.append((quarterly_contract if prod.quarterly else monthly_contract)(prod, y, m))
        y, m = _add_months(y, m, step)
    return out


# ── expiry timeline (rule-generated, holiday-aware — src/expiries.py) ────────────────
@dataclass(frozen=True)
class ExpiryRow:
    ticker: str
    kind: str           # 'Future' | 'Option'
    month: str          # contract-month label 'Sep-26'
    year: int
    mon: int
    expiry: date


def expiry_rows(prod: Product, start: date, months_ahead: int = 15) -> list[ExpiryRow]:
    """All futures + monthly-options expiries for `prod` falling in
    [start, start + months_ahead months), built from the expiries.py rule tables
    (indicative standard cycle, exchange-holiday aware)."""
    spec = _exp.SPECS[prod.family]
    hol = _exp._holidays_for(prod.ticker, "STIRs")
    end_y, end_m = _add_months(start.year, start.month, months_ahead)
    horizon_end = date(end_y, end_m, 1)
    out: list[ExpiryRow] = []
    # scan contract months a bit behind (rules can expire before the contract month)
    for delta in range(-3, months_ahead + 4):
        y, m = _add_months(start.year, start.month, delta)
        lbl = f"{_MONTHS[m - 1]}-{y % 100:02d}"
        if m in spec["cycle"]:
            d = _exp._eval(spec["fut"], y, m, hol)
            if start <= d < horizon_end:
                out.append(ExpiryRow(prod.ticker, "Future", lbl, y, m, d))
        if prod.has_options and "opt" in spec and m in spec.get("opt_cycle", []):
            d = _exp._eval(spec["opt"], y, m, hol)
            if start <= d < horizon_end:
                out.append(ExpiryRow(prod.ticker, "Option", lbl, y, m, d))
    return sorted(out, key=lambda r: (r.expiry, r.kind))


# Settle-in-arrears families: the future trades until its reference window ENDS (the big
# ED→SOFR structural change). NB distinct from Product.compound — ER also PRICES off a
# compounded expected path, but its Euribor fix is set in advance, so it dies at window start.
_ARREARS_FAMILIES = {"sofr3m", "sonia", "estr"}


def fut_last_trade(prod: Product, c: Contract) -> date:
    """The contract's actual last-trading day. In-arrears quarterlies (SR3 / SONIA /
    €STR) trade until the window ENDS — the family's expiry rule anchors on the
    window-end month, not the named month. Euribor fixes in advance (dies ~2bd
    before its window starts) and the monthlies run to their own month-end, so
    both anchor on the named month."""
    spec = _exp.SPECS[prod.family]
    hol = _exp._holidays_for(prod.ticker, "STIRs")
    if prod.family in _ARREARS_FAMILIES:               # in-arrears: rule at window end
        return _exp._eval(spec["fut"], c.end.year, c.end.month, hol)
    return _exp._eval(spec["fut"], c.year, c.month, hol)


def option_underlying(prod: Product, year: int, month: int) -> Contract:
    """The future a monthly option exercises into. Quarterly-strip products list
    serial options on the NEXT quarterly future (Mar option → Mar future, but a
    Jan/Feb serial also exercises into Mar). Monthly products: the same month."""
    if not prod.quarterly:
        return monthly_contract(prod, year, month)
    qm = ((month - 1) // 3) * 3 + 3              # next IMM month >= month
    qy = year
    if qm < month:                               # (never true: qm >= month by construction)
        qy, qm = _add_months(year, qm, 3)
    return quarterly_contract(prod, qy, qm)


# ── meetings vs windows (the module's core message) ──────────────────────────────────
def meetings_between(bank: Bank, a: date, b: date) -> list[date]:
    """Decision dates strictly after `a`, on/before `b`."""
    return [m for m in bank.meetings if a < m <= b]


def meetings_in_window(bank: Bank, c: Contract) -> list[date]:
    """Decisions whose EFFECTIVE date (next bday) lands inside the contract's
    reference window — the ones that move its settlement."""
    return [m for m in bank.meetings if c.start <= effective_date(m) < c.end]


# ── market-implied path + odds (per bank, meetings parameterised) ────────────────────
@dataclass
class BankImplied:
    contracts: list[Contract]
    meetings: list[date]        # decision dates covered
    seg_rates: np.ndarray       # overnight-proxy level per segment (%): [now, post-m1, …]
    per_meeting_bp: np.ndarray
    cum_bp: np.ndarray
    fair_price: np.ndarray
    residual_bp: np.ndarray


def implied_path(bank: Bank, contracts: list[Contract], prices: list[float],
                 asof: date, asof_rate: float,
                 spreads_bp: list[float] | None = None,
                 stub_rate: float | None = None) -> BankImplied:
    """fedpath.implied_path with the meeting calendar (and optional per-contract
    settlement-index spread, e.g. Euribor−€STR) parameterised. Linear (simple-avg)
    form, L[0] pinned to `asof_rate` — see fedpath for the derivation.

    `stub_rate` prices the ALREADY-ELAPSED days of any accruing window at the
    realized overnight average instead of today's rate. Without it, a policy move
    that landed inside the front window puts ~10bp of phantom mispricing on the
    front contract, which least squares then shoves into the next meeting's
    implied odds — wrong exactly in the weeks after a move. A single average is
    exact: the realized days enter the settlement linearly, so only their sum
    matters."""
    spreads = spreads_bp or [0.0] * len(contracts)
    decisions = [m for m in bank.meetings if asof < m < max(c.end for c in contracts)]
    bounds = [effective_date(m) for m in decisions]
    n_seg = len(bounds) + 1

    W = np.zeros((len(contracts), n_seg))
    const = np.zeros(len(contracts))               # realized-stub contribution (rate %·days)
    for ci, c in enumerate(contracts):
        days = list(_daterange(c.start, c.end))
        if not days:
            continue
        for d in days:
            if stub_rate is not None and d < asof:
                const[ci] += stub_rate
                continue
            s = 0
            for b in bounds:
                if d >= b:
                    s += 1
                else:
                    break
            W[ci, s] += 1.0
        W[ci, :] /= len(days)
        const[ci] /= len(days)

    y = np.array([100.0 - p - sp / 100.0 for p, sp in zip(prices, spreads)])
    rhs = y - const - W[:, 0] * asof_rate
    if n_seg > 1:
        # The system is usually under-determined (more meetings than contracts), and
        # min-norm lstsq wanders freely inside the null space. A tiny second-difference
        # penalty on the segment levels selects the SMOOTHEST fit consistent with the
        # prices — it leaves the cumulative path unbiased and residuals essentially
        # untouched, but kills the alternating per-meeting wiggle.
        lam = 5e-3
        if n_seg >= 3:
            D2 = np.zeros((n_seg - 2, n_seg))
            for i in range(n_seg - 2):
                D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
            A = np.vstack([W[:, 1:], lam * D2[:, 1:]])
            b = np.concatenate([rhs, -lam * D2[:, 0] * asof_rate])
        else:
            A, b = W[:, 1:], rhs
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        seg = np.concatenate([[asof_rate], sol])
    else:
        seg = np.array([asof_rate])

    fair = np.array([100.0 - const[ci] - float(W[ci] @ seg) - spreads[ci] / 100.0
                     for ci in range(len(contracts))])
    residual_bp = (np.array(prices) - fair) * 100.0
    return BankImplied(contracts, decisions, seg, np.diff(seg) * 100.0,
                       (seg[1:] - seg[0]) * 100.0, fair, residual_bp)


def implied_odds(per_meeting_bp: float, step_bp: float = 25.0) -> tuple[str, float]:
    """FedWatch-style read of one meeting's implied move: ('cut'|'hike'|'hold', p).
    p = |implied| ÷ step, capped at 1 (a >1-step implied means >100% of one step —
    the UI shows e.g. '−32bp ≈ 100% cut + 28% of a second')."""
    if abs(per_meeting_bp) < 0.5:
        return "hold", 0.0
    return ("cut" if per_meeting_bp < 0 else "hike",
            min(1.0, abs(per_meeting_bp) / step_bp))


# ── probability scenario → expected path → option-expiry landings ────────────────────
@dataclass
class MeetingView:
    decision: date
    p_hike: float       # 0..1
    p_cut: float        # 0..1
    hike_bp: float = 25.0
    cut_bp: float = 25.0

    @property
    def expected_bp(self) -> float:
        return self.p_hike * self.hike_bp - self.p_cut * self.cut_bp


def scenario_rate_fn(asof_rate: float, views: list[MeetingView],
                     asof: date | None = None, stub_rate: float | None = None):
    """Step path of the overnight proxy under the probability-weighted scenario.
    With `asof` + `stub_rate`, days before `asof` return the realized overnight
    average instead of today's rate (same stub logic as implied_path)."""
    base = overnight_rate_fn(asof_rate,
                             [v.decision for v in views],
                             [v.expected_bp for v in views])
    if asof is None or stub_rate is None:
        return base

    def rate(d: date) -> float:
        return stub_rate if d < asof else base(d)
    return rate


def fair_price(prod: Product, c: Contract, rate_fn, spread_bp: float | None = None) -> float:
    """Scenario fair value of one contract, honouring the product's settlement
    convention and index spread (ER: compounded-expected €STR + spread). Pass
    `spread_bp` to override the product default (the page's spread input)."""
    sp = prod.spread_bp if spread_bp is None else spread_bp
    return price(c, rate_fn, compound=prod.compound) - sp / 100.0


@dataclass
class Landing:
    """One option expiry: where its underlying future sits under the scenario."""
    prod: str               # product ticker
    opt_month: str          # option contract-month label
    expiry: date
    underlying: Contract
    fair: float             # scenario fair value of the underlying at expiry
    meetings_decided: list[date]      # decisions between now and the option expiry
    meetings_open: list[date]         # decisions still inside the underlying's window AFTER expiry
    series: str = "Std"               # 'Std' listed monthly · '1Y MC' midcurve


# ── 1Y midcurve options: short-dated premium on the quarterly 12 months out ──────────
# CME lists 1Y/2Y/3Y SOFR midcurves, ICE lists Euribor + SONIA midcurves; the 1Y
# quarterlies are where red-pack meeting trades actually clear, so they're the ones
# the timeline needs. Expiry rule = the family's standard option rule in the option
# month; underlying = the quarterly deferred by MIDCURVE_DEFER_MO.
MIDCURVE_DEFER_MO = 12


def midcurve_expiries(prod: Product, start: date, months_ahead: int = 15) -> list[ExpiryRow]:
    """Quarterly 1Y-midcurve option expiries in [start, start+months_ahead mo)."""
    if not (prod.has_options and prod.quarterly):
        return []
    spec = _exp.SPECS[prod.family]
    hol = _exp._holidays_for(prod.ticker, "STIRs")
    ey, em = _add_months(start.year, start.month, months_ahead)
    horizon_end = date(ey, em, 1)
    out = []
    for delta in range(-3, months_ahead + 4):
        y, m = _add_months(start.year, start.month, delta)
        if m not in (3, 6, 9, 12):
            continue
        d = _exp._eval(spec["opt"], y, m, hol)
        if start <= d < horizon_end:
            out.append(ExpiryRow(prod.ticker, "Midcurve", f"{_MONTHS[m - 1]}-{y % 100:02d}",
                                 y, m, d))
    return sorted(out, key=lambda r: r.expiry)


def option_underlying_mc(prod: Product, year: int, month: int) -> Contract:
    """The deferred quarterly a 1Y midcurve exercises into (option month + 12mo)."""
    yy, mm = _add_months(year, month, MIDCURVE_DEFER_MO)
    return quarterly_contract(prod, yy, mm)


def landings(prod: Product, bank: Bank, asof: date, views: list[MeetingView],
             asof_rate: float, months_ahead: int = 15,
             spread_bp: float | None = None, stub_rate: float | None = None,
             include_midcurves: bool = False) -> list[Landing]:
    """For every monthly option expiry inside the horizon (plus, optionally, the
    1Y quarterly midcurves): the scenario landing price of its underlying, plus
    the decided-by-then / still-open meeting split (the 'how many meetings does
    this option capture' answer)."""
    fn = scenario_rate_fn(asof_rate, views, asof=asof, stub_rate=stub_rate)
    rows = [(r, option_underlying(prod, r.year, r.mon), "Std")
            for r in expiry_rows(prod, asof, months_ahead) if r.kind == "Option"]
    if include_midcurves:
        rows += [(r, option_underlying_mc(prod, r.year, r.mon), "1Y MC")
                 for r in midcurve_expiries(prod, asof, months_ahead)]
    out = []
    for r, u, series in sorted(rows, key=lambda t: (t[0].expiry, t[2])):
        out.append(Landing(
            prod.ticker, r.month, r.expiry, u, fair_price(prod, u, fn, spread_bp),
            meetings_between(bank, asof, r.expiry),
            [m for m in meetings_in_window(bank, u) if m > r.expiry], series))
    return out


# ── scenario outcome DISTRIBUTION (not just the expected landing) ────────────────────
def _pmf_from_view(v: MeetingView) -> dict[float, float]:
    """One meeting's outcome pmf over the move in bp. The signed-odds convention
    decomposes beyond one step: p=1.5 cuts → 50/50 one-vs-two cuts. Hike and cut
    sides convolve independently (the UI only ever populates one)."""
    def side(p: float, size: float) -> dict[float, float]:
        k = int(p)
        frac = p - k
        if frac > 1e-9:
            return {k * size: 1.0 - frac, (k + 1) * size: frac}
        return {k * size: 1.0}
    out: dict[float, float] = {}
    for a, pa in side(v.p_hike, v.hike_bp).items():
        for b, pb in side(v.p_cut, -v.cut_bp).items():
            out[a + b] = out.get(a + b, 0.0) + pa * pb
    return out


def window_weight(c: Contract, decision: date) -> float:
    """Fraction of the contract's settlement a move at `decision` touches: the
    share of window days on/after the move's effective day."""
    days = list(_daterange(c.start, c.end))
    if not days:
        return 0.0
    eff = effective_date(decision)
    return sum(1 for d in days if d >= eff) / len(days)


def landing_distribution(prod: Product, bank: Bank, c: Contract, asof: date,
                         views: list[MeetingView], asof_rate: float,
                         upto: date | None = None, spread_bp: float | None = None,
                         stub_rate: float | None = None, tol_bp: float = 0.25,
                         max_nodes: int = 128) -> list[tuple[float, float]]:
    """The pmf of the contract's fair price under the scenario: meetings decided
    on/before `upto` (default: all in-window meetings) realise as trinomial
    outcomes; meetings AFTER `upto` but still inside the window enter at expected
    value — the market at the option's expiry still prices them as expectations.
    Centred so the mean equals the expected-path fair price. Returns
    [(price, prob)] sorted by price; exact convolution, no Monte Carlo."""
    fn = scenario_rate_fn(asof_rate, views, asof=asof, stub_rate=stub_rate)
    base = fair_price(prod, c, fn, spread_bp)
    dist = {0.0: 1.0}                              # shift vs expectation, in bp of rate
    for v in views:
        if upto is not None and v.decision > upto:
            continue
        w = window_weight(c, v.decision)
        if w <= 0:
            continue
        pmf = _pmf_from_view(v)
        if len(pmf) == 1:                          # deterministic meeting — no spread
            continue
        exp = v.expected_bp
        new: dict[float, float] = {}
        for s, p in dist.items():
            for x, px in pmf.items():
                k = round((s + (x - exp) * w) / tol_bp) * tol_bp
                new[k] = new.get(k, 0.0) + p * px
        if len(new) > max_nodes:                   # keep the mass, drop the dust
            keep = sorted(new.items(), key=lambda kv: -kv[1])[:max_nodes]
            tot = sum(p for _, p in keep)
            new = {k: p / tot for k, p in keep}
        dist = new
    # re-centre: grid rounding drifts the mean by a fraction of a bp — remove it
    # exactly so E[dist] equals the expected-path fair price.
    drift = sum(k * p for k, p in dist.items())
    return sorted(((base - (s - drift) / 100.0, p) for s, p in dist.items()),
                  key=lambda t: t[0])


# ── realized overnight fixings (the front-stub input) ────────────────────────────────
FIXING_TICKERS = {"FED": "SOFRRATE Index", "ECB": "ESTRON Index", "BOE": "SONIO/N Index"}
_FIX_STORE = Path(__file__).resolve().parents[1] / "data" / "stir_fixings.json"


def _load_fixings(bank_key: str) -> dict[str, float]:
    try:
        return json.loads(_FIX_STORE.read_text(encoding="utf-8")).get(bank_key, {})
    except Exception:
        return {}


def realized_stub_avg(bank: Bank, start: date, asof: date,
                      fixings: dict[str, float] | None = None) -> float | None:
    """Average realized overnight rate over [start, asof) — the number that fixes
    the front stub. Walks calendar days with weekends/holidays carrying the prior
    fixing (the published-index convention). Fixings come from the cache written
    by the live refresh (or are injected for tests); returns None when coverage
    is too thin to trust, and the page falls back to a manual input."""
    if start >= asof:
        return None
    fx = _load_fixings(bank.key) if fixings is None else fixings
    if not fx:
        return None
    dated = sorted((date.fromisoformat(k), v) for k, v in fx.items())
    if dated[0][0] > start + timedelta(days=4) or dated[-1][0] < asof - timedelta(days=5):
        return None                                # doesn't cover the stub span
    vals, last, i = [], None, 0
    for d in _daterange(start, asof):
        while i < len(dated) and dated[i][0] <= d:
            last = dated[i][1]
            i += 1
        if last is None:
            return None
        vals.append(last)
    return float(np.mean(vals))


def refresh_fixings(asof: date, lookback_days: int = 130) -> dict[str, int]:
    """Pull the three overnight indices into data/stir_fixings.json (live mode
    only; a blocked/failed pull leaves the cache untouched). Returns per-bank
    row counts — {} offline."""
    if MODE != "bloomberg":
        return {}
    out = {}
    try:
        from xbbg import blp
        try:
            store = json.loads(_FIX_STORE.read_text(encoding="utf-8"))
        except Exception:
            store = {}
        start = (asof - timedelta(days=lookback_days)).isoformat()
        for bk, tk in FIXING_TICKERS.items():
            df = blp.bdh(tk, "PX_LAST", start, asof.isoformat())
            if df is None or df.empty:
                continue
            ser = df.iloc[:, 0].dropna()
            cur = store.get(bk, {})
            cur.update({ts.date().isoformat(): float(v) for ts, v in ser.items()})
            store[bk] = cur
            out[bk] = len(cur)
        if out:
            _FIX_STORE.parent.mkdir(parents=True, exist_ok=True)
            _FIX_STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")
    except Exception:
        return {}
    return out


# ── decision-day helper (Home banner + release-time popup ride the report-alert rail) ─
_DECISION_LOCAL = {"FED": ("14:00", "America/New_York"),   # FOMC statement
                   "ECB": ("14:15", "Europe/Berlin"),      # GC press release (conf 14:45 CET)
                   "BOE": ("12:00", "Europe/London")}      # MPC announcement + minutes
_DECISION_LABEL = {"FED": ("FOMC rate decision", "🏛️"),
                   "ECB": ("ECB rate decision", "💶"),
                   "BOE": ("BoE rate decision", "💷")}


def decisions_today(today: date) -> list[dict]:
    """Central-bank decisions announced today: {bank, name, icon, t}, with `t` the
    announcement time converted to ET for THAT date. Converting per-date matters:
    in the weeks when Europe and the US disagree on summer time (late Oct / mid
    Mar) the fixed local times drift an hour in ET — e.g. the 29 Oct 2026 ECB is
    09:15 ET, not the usual 08:15."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    out = []
    for bk, bank in BANKS.items():
        if today in bank.meetings:
            t_local, tz = _DECISION_LOCAL[bk]
            hh, mm = map(int, t_local.split(":"))
            et = datetime(today.year, today.month, today.day, hh, mm,
                          tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo("America/New_York"))
            name, icon = _DECISION_LABEL[bk]
            out.append({"bank": bk, "name": name, "icon": icon, "t": f"{et:%H:%M}"})
    return out


# ── data feed (mock → bloomberg, mirrors fedpath) ────────────────────────────────────
MODE = fedpath.MODE

# mock path flavour per bank: (bp per move, move every k-th meeting) — mild easing FED,
# hold-with-late-cut ECB, steady easing BOE. Purely for offline demo realism.
_MOCK_STYLE = {"FED": (-25.0, 2), "ECB": (-25.0, 3), "BOE": (-25.0, 2)}


def strip_prices(prod: Product, bank: Bank, contracts: list[Contract], asof: date,
                 asof_rate: float) -> list[float]:
    """Live prices per contract (specific tickers off `prod.root`), mock fallback."""
    if MODE == "bloomberg":
        try:
            from xbbg import blp
            tickers = [f"{c.code} Comdty" for c in contracts]
            px = blp.bdp(tickers, "PX_LAST")
            return [float(px.loc[t, "px_last"]) for t in tickers]
        except Exception:
            pass
    step, every = _MOCK_STYLE.get(bank.key, (-25.0, 2))
    ups = [m for m in bank.meetings if m > asof]
    moves = [(step if i % every == 0 else 0.0) for i in range(len(ups))]
    fn = overnight_rate_fn(asof_rate, ups, moves)
    out = []
    for i, c in enumerate(contracts):
        p = fair_price(prod, c, fn)
        p += (0.5 - ((i * 7) % 5) / 4.0) * 0.004
        out.append(round(p, 4))
    return out
