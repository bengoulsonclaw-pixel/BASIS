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
# Copom decision (announcement) dates — the SECOND day of each two-day meeting,
# announced after the B3 close. CONFIRMED 2026-08-19 against the BCB's own
# calendar API (macrodata.copom_decisions(), cross-checked vs the minutes feed
# and SGS 432 by the Macro Radar session). Only Dec-2027 is still provisional —
# the BCB had published 2027 through October at the time.
BCB_DECISIONS: list[date] = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), date(2026, 6, 17),
    date(2026, 8, 5), date(2026, 9, 16), date(2026, 11, 4), date(2026, 12, 9),
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 16),
    date(2027, 8, 4), date(2027, 9, 21), date(2027, 10, 26), date(2027, 12, 8),
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
    # default_rate = the CURRENT policy setting and must track reality by hand:
    # it anchors the live fit (r0) AND generates the synthetic demo strip. The
    # Home cards and the Fed page's band picker derive from it — update HERE.
    # FED 3.625 = 3.50-3.75 target band, confirmed off WIRP 14 Aug 2026.
    "FED": Bank("FED", "Federal Reserve", "FOMC", "Target band midpoint",
                fedpath.FOMC_DECISIONS, 3.625, 25.0, "$"),
    # ECB 2.25 / BOE 3.75 derived 15 Aug 2026 from the realized front arrears
    # contracts Ben supplied (TKYM6 97.81 -> ESTR ~2.17 = depo-8bp; SFIM6
    # 96.2525 -> SONIA ~3.75) — pending Ben's on-screen confirmation.
    "ECB": Bank("ECB", "European Central Bank", "Governing Council", "Deposit facility rate",
                ECB_DECISIONS, 2.25, 25.0, "€"),
    "BOE": Bank("BOE", "Bank of England", "MPC", "Bank Rate",
                BOE_DECISIONS, 3.75, 25.0, "£"),
    # Selic target 14.00 per SGS 432 (target in force; confirmed by the Macro
    # Radar session 2026-08-19 — SGS 1178 effective 13.90 = CDI −10bp ✓).
    "BCB": Bank("BCB", "Banco Central do Brasil", "Copom", "Selic target",
                BCB_DECISIONS, 14.00, 25.0, "R$"),
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
    rate_quoted: bool = False # DI-style: quoted as an ANNUALIZED RATE (bd/252
                              # compounded, zero-from-today), not 100 − rate
    in_pull: bool = True      # fetched by the morning pull (False = store fed
                              # by paste/live-button only — e.g. unverified
                              # roots, kept OFF the pull to avoid tripping a
                              # fresh Bloomberg workflow review)


PRODUCTS: dict[str, Product] = {
    "SFRA Comdty": Product("SFRA Comdty", "3M SOFR (SR3)", "SR3", "FED", "SFR",
                           "sofr3m", True, True, 25.0, "#F5C518"),
    "SERA Comdty": Product("SERA Comdty", "1M SOFR (SR1)", "SR1", "FED", "SER",
                           "sofr1m", False, False, 41.67, "#64B5F6", in_strip=False),
    "FFA Comdty":  Product("FFA Comdty", "30-Day Fed Funds", "FF", "FED", "FF",
                           "fedfunds", False, False, 41.67, "#81C784", in_strip=False),
    # ER spread 15bp MEASURED off the 14-Aug-2026 real strips (ER vs ESTR-fair
    # implied +14.5/+14.9/+15.6bp on the liquid quarterlies); was 10, which made
    # the joint ECB fit dump phantom moves into late meetings. Page-tunable.
    "ERA Comdty":  Product("ERA Comdty", "3M Euribor", "ER", "ECB", "ER",
                           "euribor", True, True, 25.0, "#BA68C8", spread_bp=15.0),
    "TKYA Comdty": Product("TKYA Comdty", "3M €STR", "€STR", "ECB", "TKY",
                           "estr", True, True, 25.0, "#4DD0E1", has_options=False),
    "SFIA Comdty": Product("SFIA Comdty", "3M SONIA", "SONIA", "BOE", "SFI",
                           "sonia", True, True, 25.0, "#FF8A65"),
    # ICE 1M SONIA (root SOO, Ben confirmed 15 Aug 2026): the BoE's per-meeting
    # pinning instrument, like SR1/FF for the Fed. Screens show OI "N.A." — the
    # settles are exchange MARKS, fine for fitting, not dealable quotes. £3m
    # notional -> ~£25/bp. Simple-average settlement assumed (sub-bp vs
    # compounded over one month either way).
    "SOOA Comdty": Product("SOOA Comdty", "1M SONIA (SOO)", "SOO1", "BOE", "SOO",
                           "sonia1m", False, False, 25.0, "#FFCC80",
                           has_options=False, in_strip=False),
    # B3 one-day interbank deposit (DI1): the Brazilian curve. RATE-quoted
    # (annualized, business-days/252, compounded CDI from TODAY to the 1st bday
    # of the named month) — a zero, not a window, so consecutive maturities
    # difference out the Copom meetings between them and there is NO front
    # stub. bp_value varies with maturity (PU DV01) — left 0, unused by the
    # odds. root "OD" ⚠ UNVERIFIED on the Terminal; in_pull=False until Ben
    # confirms (a brand-new pull surface the day after a -4002 cleared is
    # asking for another review) — feed the store by paste / ⚡ button.
    "OD1 Comdty": Product("OD1 Comdty", "DI 1-day (B3)", "DI", "BCB", "OD",
                          "cdi", False, True, 0.0, "#7CE0B3",
                          has_options=False, rate_quoted=True, in_pull=False),
}
# Overnight proxy vs the policy rate, in bp (page-tunable; these seed the input):
# SOFR ≈ target mid + 0 · €STR ≈ depo − 8 · SONIA ≈ Bank Rate + 0 (the −5 seed
# left a −4.7bp phantom on the BoE front fit vs the realized SFIM6 window —
# SONIA is fixing AT Bank Rate on the real 14-Aug-2026 data).
BANK_BASIS_SEED = {"FED": 0.0, "ECB": -8.0, "BOE": 0.0,
                   "BCB": -10.0}   # CDI fixes ~10bp under the Selic target


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


def _first_bday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def di_contract(prod: Product, asof: date, year: int, month: int) -> Contract:
    """B3 DI1: a zero from TODAY to the 1st business day of the named month —
    the 'window' is [asof, maturity), rebuilt per asof, so meetings_in_window /
    pair-difference pinning work unchanged."""
    return Contract(_code(prod.root, year, month),
                    f"{_MONTHS[month - 1]}-{year % 100:02d}", year, month,
                    asof, _first_bday(year, month))


def di_strip(prod: Product, asof: date, n: int = 18) -> list[Contract]:
    """The next `n` monthly DI maturities still ahead of `asof`."""
    out = []
    y, m = _add_months(asof.year, asof.month, 1)
    for _ in range(n):
        c = di_contract(prod, asof, y, m)
        if c.end > asof:
            out.append(c)
        y, m = _add_months(y, m, 1)
    return out


def strip(prod: Product, asof: date, n: int = 8) -> list[Contract]:
    """The front `n` contracts from `asof` — currently-accruing first."""
    if prod.family == "cdi":                        # DI zeros, not windows
        return di_strip(prod, asof, n)
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


# Quarterly products whose SERIAL (non-IMM) months actually list and trade —
# confirmed on the Terminal 15 Aug 2026 (ERV6/ERX6…, TKYN6/TKYQ6/TKYV6…). Each
# serial has a full [3rd Wed, +3M) window one month offset from its quarterly
# neighbours, so consecutive serials DIFFERENCE out single months — the per-
# meeting pinning the quarterly strip alone cannot do.
SERIAL_FIT_PRODUCTS = {"ERA Comdty", "TKYA Comdty"}


def pull_universe(asof: date) -> list[tuple["Product", Contract]]:
    """EVERY contract the morning pull fetches and the store must carry — the
    SINGLE source of truth shared by refresh_strip_store and (as a superset)
    fit_instruments. Quarterlies 12 deep, monthlies 13 (the fit's monthly cap),
    serial months for SERIAL_FIT_PRODUCTS, ALL products including fit-excluded
    ones (SOO is stored for display even though the fit drops its marks).
    Divergence between pull and fit silently starved the fit of serials once —
    keep both sides on this function."""
    out: list[tuple[Product, Contract]] = []
    for p in PRODUCTS.values():
        if not p.in_pull:                           # unverified roots stay OFF the
            continue                                # morning pull (review-trip risk)
        for c in strip(p, asof, 12 if p.quarterly else 13):
            out.append((p, c))
        if p.quarterly and p.ticker in SERIAL_FIT_PRODUCTS:
            for c in serial_strip(p, asof):
                out.append((p, c))
    return out


def serial_strip(prod: Product, asof: date, months: int = 10) -> list[Contract]:
    """The serial (non-IMM-month) contracts of a quarterly product still trading
    at `asof` — same window construction as the quarterlies, off-cycle months.
    Looks a quarter back too: arrears serials outlive their named month."""
    out = []
    y, m = _add_months(asof.year, asof.month, -3)
    for _ in range(months + 3):
        if m not in (3, 6, 9, 12):
            c = quarterly_contract(prod, y, m)
            if fut_last_trade(prod, c) >= asof:
                out.append(c)
        y, m = _add_months(y, m, 1)
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


def bank_effective_date(bank: Bank | str, m: date) -> date:
    """When a decision actually hits the overnight fixing — per-bank convention:
    Fed and ECB next business day, BoE SAME day (announced noon, effective
    immediately). A following-Wednesday MRO rule for the ECB was tried and
    REJECTED by validation: hand-solving the €STR serial pair against the
    14-Aug-2026 market reproduces WIRP's front odds exactly under next-bday
    (91% vs 91.9), and degrades ~25pts under following-Wednesday — the market
    prices €STR as moving the day after the decision."""
    key = bank if isinstance(bank, str) else bank.key
    if key == "BOE":
        return m
    return effective_date(m)


def meetings_in_window(bank: Bank, c: Contract) -> list[date]:
    """Decisions whose EFFECTIVE date (bank convention, see bank_effective_date)
    lands inside the contract's reference window — the ones that move its
    settlement."""
    return [m for m in bank.meetings if c.start <= bank_effective_date(bank, m) < c.end]


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
    stub: float | None = None   # realized pre-asof o/n average: solved (solve_stub),
                                # passed (stub_rate) or None (nothing elapsed / legacy)


def implied_path(bank: Bank, contracts: list[Contract], prices: list[float],
                 asof: date, asof_rate: float,
                 spreads_bp: list[float] | None = None,
                 stub_rate: float | None = None,
                 solve_stub: bool = False, lam: float = 5e-3) -> BankImplied:
    """fedpath.implied_path with the meeting calendar (and optional per-contract
    settlement-index spread, e.g. Euribor−€STR) parameterised. Linear (simple-avg)
    form, L[0] pinned to `asof_rate` — see fedpath for the derivation.

    `stub_rate` prices the ALREADY-ELAPSED days of any accruing window at the
    realized overnight average instead of today's rate. Without it, a policy move
    that landed inside the front window puts ~10bp of phantom mispricing on the
    front contract, which least squares then shoves into the next meeting's
    implied odds — wrong exactly in the weeks after a move. A single average is
    exact: the realized days enter the settlement linearly, so only their sum
    matters.

    `solve_stub=True` estimates that realized average FROM the contracts instead
    (an extra unsmoothed unknown, lightly ridged toward `asof_rate`): the front
    arrears quarterly + front monthlies carry ~half their weight on elapsed days,
    which identifies it well — the market tells us what already fixed, no
    fixings feed needed. Ignored when `stub_rate` is given or nothing elapsed."""
    spreads = spreads_bp or [0.0] * len(contracts)
    # asof <= m: on DECISION DAY the meeting stays in the fit all day — the strip
    # prices the move until the announcement, and dropping it at midnight shoved
    # ~a full step of front-contract pricing into the wrong meeting's odds.
    decisions = [m for m in bank.meetings if asof <= m < max(c.end for c in contracts)]
    bounds = [bank_effective_date(bank, m) for m in decisions]
    n_seg = len(bounds) + 1
    solving = solve_stub and stub_rate is None

    W = np.zeros((len(contracts), n_seg))
    const = np.zeros(len(contracts))               # realized-stub contribution (rate %·days)
    # solve_stub: one realized-average unknown PER DISTINCT WINDOW START among
    # the elapsed spans — a single shared scalar cannot represent contracts
    # whose elapsed days cover different slices of history once a policy move
    # sits between their starts (front quarterly from the IMM date, serials a
    # month later, monthlies from month-start: measured 3-4bp of phantom odds
    # in the weeks after a move with one scalar).
    starts = sorted({c.start for c in contracts if c.start < asof}) if solving else []
    g_of = {s: gi for gi, s in enumerate(starts)}
    Wstub = np.zeros((len(contracts), max(1, len(starts))))
    for ci, c in enumerate(contracts):
        days = list(_daterange(c.start, c.end))
        if not days:
            continue
        for d in days:
            if d < asof and (solving or stub_rate is not None):
                if solving:
                    Wstub[ci, g_of[c.start]] += 1.0
                else:
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
        Wstub[ci, :] /= len(days)
        const[ci] /= len(days)

    n_stub = len(starts)
    if solving and (n_stub == 0 or Wstub.sum() <= 0):   # nothing elapsed
        solving = False
        n_stub = 0

    y = np.array([100.0 - p - sp / 100.0 for p, sp in zip(prices, spreads)])
    rhs = y - const - W[:, 0] * asof_rate
    stub_out = stub_rate
    stub_vec = None
    if n_seg > 1 or solving:
        # The system is usually under-determined (more meetings than contracts), and
        # min-norm lstsq wanders freely inside the null space. A tiny second-difference
        # penalty on the segment levels selects the SMOOTHEST fit consistent with the
        # prices — it leaves the cumulative path unbiased and residuals essentially
        # untouched, but kills the alternating per-meeting wiggle. The stub unknowns
        # (leading cols when solving) are kept OUT of the smoothing and instead get a
        # light ridge toward asof_rate so degenerate elapsed coverage can't blow up.
        n_unk = n_stub + (n_seg - 1)
        A_data = np.hstack([Wstub[:, :n_stub], W[:, 1:]]) if solving else W[:, 1:]
        rows = [A_data]
        rhss = [rhs]
        if n_seg >= 3:
            D2 = np.zeros((n_seg - 2, n_seg))
            for i in range(n_seg - 2):
                # Ramp: base weight over the front (where dedicated monthlies
                # genuinely pin meetings), rising with distance — far meetings
                # sit past the liquid monthlies, and sub-bp noise in far marks
                # otherwise prints as double-digit phantom odds out there.
                w = 1.0 + max(0, i - 3) * 0.75
                D2[i, i], D2[i, i + 1], D2[i, i + 2] = w, -2.0 * w, w
            D2u = lam * D2[:, 1:]
            if solving:
                D2u = np.hstack([np.zeros((D2u.shape[0], n_stub)), D2u])
            rows.append(D2u)
            rhss.append(-lam * D2[:, 0] * asof_rate)
        if solving:
            mu = 0.02
            ridge = np.zeros((n_stub, n_unk))
            for gi in range(n_stub):
                ridge[gi, gi] = mu
            rows.append(ridge)
            rhss.append(np.full(n_stub, mu * asof_rate))
        sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhss), rcond=None)
        if solving:
            stub_vec = sol[:n_stub]
            stub_out = float(stub_vec[0])           # front window's realized average
            seg = np.concatenate([[asof_rate], sol[n_stub:]])
        else:
            seg = np.concatenate([[asof_rate], sol])
    else:
        seg = np.array([asof_rate])

    stub_w = (Wstub[:, :n_stub] @ stub_vec) if (solving and stub_vec is not None) else const
    fair = np.array([100.0 - stub_w[ci] - float(W[ci] @ seg) - spreads[ci] / 100.0
                     for ci in range(len(contracts))])
    residual_bp = (np.array(prices) - fair) * 100.0
    return BankImplied(contracts, decisions, seg, np.diff(seg) * 100.0,
                       (seg[1:] - seg[0]) * 100.0, fair, residual_bp, stub_out)


def _bd(a: date, b: date) -> int:
    """Business days in [a, b) — weekends only. B3 holidays are approximated
    away: one holiday inside a segment shifts its weight by 1/n, sub-bp on the
    odds; swap in a B3 calendar table if the WIRP-BZ comparison demands it."""
    return int(np.busday_count(a, b))


def di_implied_path(bank: Bank, contracts: list[Contract], quotes: list[float],
                    asof: date, asof_rate: float, lam: float = 2e-2) -> BankImplied:
    """implied_path for RATE-QUOTED zeros (B3 DI1). Each quote is the
    annualized bd/252-compounded CDI from asof to its maturity, so in
    log(1+r) space the system is EXACTLY the linear meeting-step least
    squares: log(1+Q_i) = Σ_s w_is · log(1+r_s), w = business-day fractions.
    Same D2 ramp as the price-space fit (the data/penalty balance is
    scale-free); no stub — a zero from today has no elapsed days.

    Meetings with under ~a week of business days before the LAST maturity are
    excluded: their only information is a sliver of the final zero, and the
    fit printed a −42bp smoothing artifact on such a meeting in testing."""
    max_end = max(c.end for c in contracts)
    decisions = [m for m in bank.meetings
                 if asof <= m
                 and bank_effective_date(bank, m) <= max_end - timedelta(days=7)]
    bounds = [bank_effective_date(bank, m) for m in decisions]
    n_seg = len(bounds) + 1
    W = np.zeros((len(contracts), n_seg))
    for ci, c in enumerate(contracts):
        n_i = max(1, _bd(asof, c.end))
        lo = asof
        for s in range(n_seg):
            hi = bounds[s] if s < len(bounds) else c.end
            a_, b_ = lo, min(hi, c.end)
            if b_ > a_:
                W[ci, s] = _bd(a_, b_) / n_i
            lo = hi
            if lo >= c.end:
                break
    z0 = float(np.log1p(asof_rate / 100.0))
    y = np.array([float(np.log1p(q / 100.0)) for q in quotes])
    rhs = y - W[:, 0] * z0
    if n_seg > 1:
        rows = [W[:, 1:]]
        rhss = [rhs]
        if n_seg >= 3:
            D2 = np.zeros((n_seg - 2, n_seg))
            for i in range(n_seg - 2):
                w = 1.0 + max(0, i - 3) * 0.75
                D2[i, i], D2[i, i + 1], D2[i, i + 2] = w, -2.0 * w, w
            rows.append(lam * D2[:, 1:])
            rhss.append(-lam * D2[:, 0] * z0)
        sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhss), rcond=None)
        seg_z = np.concatenate([[z0], sol])
    else:
        seg_z = np.array([z0])
    seg = (np.expm1(seg_z)) * 100.0
    fair_q = np.expm1(W @ seg_z) * 100.0
    residual_bp = (np.array(quotes) - fair_q) * 100.0
    return BankImplied(contracts, decisions, seg, np.diff(seg) * 100.0,
                       (seg[1:] - seg[0]) * 100.0, fair_q, residual_bp, None)


def di_fair_rate(c: Contract, rate_fn, asof: date) -> float:
    """Scenario fair DI quote (annualized %, bd/252): the compounded scenario
    CDI from asof to maturity — exp(mean over business days of log(1+r_d))−1."""
    zs = [float(np.log1p(rate_fn(d) / 100.0))
          for d in _daterange(asof, c.end) if d.weekday() < 5]
    if not zs:
        return float(rate_fn(asof))
    return float(np.expm1(np.mean(zs)) * 100.0)


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
        from . import bbg as blp
        try:
            store = json.loads(_FIX_STORE.read_text(encoding="utf-8"))
        except Exception:
            store = {}
        start = (asof - timedelta(days=lookback_days)).isoformat()
        for bk, tk in FIXING_TICKERS.items():
            # xbbg's Rust binding returns narwhals LONG frames since ~2026-08 —
            # datafeed's adapter normalises to wide (index=date), old or new shape
            from . import datafeed as _dfeed
            w = _dfeed._bdh_to_wide(blp.bdh(tk, "PX_LAST", start, asof.isoformat()))
            if w is None or w.empty:
                continue
            ser = w.iloc[:, 0].dropna()
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
                   "BOE": ("12:00", "Europe/London"),      # MPC announcement + minutes
                   "BCB": ("18:30", "America/Sao_Paulo")}  # Copom, after the B3 close
_DECISION_LABEL = {"FED": ("FOMC rate decision", "🏛️"),
                   "ECB": ("ECB rate decision", "💶"),
                   "BOE": ("BoE rate decision", "💷"),
                   "BCB": ("Copom rate decision", "🇧🇷")}


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


# ── data feed ────────────────────────────────────────────────────────────────────────
# POLICY (Ben, 2026-08-11): the STIR pages NEVER pull Bloomberg on their own — prices
# come from the store the daily MORNING SNAPSHOT writes (refresh_strip_store, called
# from snapshot.py's fetch phase). The only other Bloomberg touch is the explicit
# per-page "⚡ Live pull" button (live_strip_prices), which requests EXACTLY the
# page's strip tickers and nothing else. Offline/demo falls back to the mock.
MODE = fedpath.MODE
STRIP_STORE = Path(__file__).resolve().parents[1] / "data" / "snapshot" / "stir_strips.json"

# mock path flavour per bank: (bp per move, move every k-th meeting) — mild easing FED,
# hold-with-late-cut ECB, steady easing BOE. Purely for offline demo realism.
_MOCK_STYLE = {"FED": (-25.0, 2), "ECB": (-25.0, 3), "BOE": (-25.0, 2),
               "BCB": (-50.0, 1)}                  # demo: a 50bp-per-meeting cutting cycle


def _load_strip_store() -> dict:
    try:
        return json.loads(STRIP_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def strip_store_asof() -> str | None:
    """The settle/pull date the store carries, or None when there is no store."""
    return _load_strip_store().get("asof") or None


STRIP_ERROR = STRIP_STORE.with_name("stir_strips_error.json")


def _strip_error_write(stage: str, detail: str) -> None:
    """Record WHY the morning strip leg failed — read by Data health + the page
    provenance captions. A silent `return 0` hid a -4002 block for a week."""
    try:
        from datetime import datetime
        STRIP_ERROR.parent.mkdir(parents=True, exist_ok=True)
        STRIP_ERROR.write_text(json.dumps(
            {"when": datetime.now().isoformat(timespec="seconds"),
             "stage": stage, "detail": detail}, indent=1), encoding="utf-8")
    except Exception:
        pass


def strip_error() -> dict | None:
    """The last recorded strip-pull failure, or None after a clean run."""
    try:
        return json.loads(STRIP_ERROR.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_strip_store(asof: date) -> int:
    """Pull ALL six products' specific strips (PX_LAST + PX_SETTLE, one batched bdp,
    ~72 tickers) into the snapshot store, and refresh the o/n fixings cache while
    the Terminal is connected. Called from snapshot.py's morning fetch phase ONLY —
    a failed pull leaves the previous store untouched (and records its reason to
    STRIP_ERROR — never fail silently). Returns priced-ticker count."""
    if MODE != "bloomberg":
        return 0
    try:
        from . import bbg as blp
        pairs = pull_universe(asof)
        contracts = [c for _, c in pairs]
        gate_of = {c.code: ((0.5, 40.0) if p.rate_quoted else (90.0, 100.5))
                   for p, c in pairs}
        tickers = [f"{c.code} Comdty" for c in contracts]
        # datafeed._bdp_rows handles both the binding's shapes (narwhals LONG
        # ticker/field/value since ~2026-08, legacy wide index=ticker before)
        from . import datafeed as _dfeed
        rows = _dfeed._bdp_rows(_dfeed._coerce_pd(blp.bdp(tickers, ["PX_LAST", "PX_SETTLE"])))
        if not rows:
            _strip_error_write("reference pull", f"empty response for {len(tickers)} tickers "
                               "with the Terminal expected up — this is Bloomberg's -4002 "
                               "WORKFLOW_REVIEW_NEEDED signature (a raw blpapi probe shows the "
                               "code + nid; ring the Help Desk with them).")
            return 0
        prices, settles, rejected = {}, {}, {}
        for c in contracts:
            r = rows.get(f"{c.code} Comdty") or {}
            for fld, out in (("PX_LAST", prices), ("PX_SETTLE", settles)):
                v = r.get(fld)
                if v is not None and v == v:
                    v = float(v)
                    # Plausibility gate, PER QUOTE TYPE: a price-quoted STIR
                    # future lives between ~90 and ~100.5; a rate-quoted DI
                    # between ~0.5 and 40%. The first real morning
                    # (2026-08-19) returned ERJ7=201.0 / ERK7=397.0 for
                    # barely-listed far serials — not prices at all — and the
                    # fit printed thousands of phantom bp honouring them.
                    # Reject and RECORD, never store.
                    glo, ghi = gate_of.get(c.code, (90.0, 100.5))
                    if glo < v < ghi:
                        out[c.code] = v
                    else:
                        rejected[c.code] = v
        if not prices and not settles:
            _strip_error_write("parse", "response carried no PX_LAST/PX_SETTLE values "
                               f"for any of {len(tickers)} tickers.")
            return 0
        STRIP_STORE.parent.mkdir(parents=True, exist_ok=True)
        STRIP_STORE.write_text(json.dumps(
            {"asof": asof.isoformat(), "prices": prices, "settles": settles,
             "rejected": rejected},
            indent=1), encoding="utf-8")
        try:
            STRIP_ERROR.unlink(missing_ok=True)     # clean run — clear the flag
        except Exception:
            pass
        refresh_fixings(asof)                       # 3 small bdh pulls, same session
        try:
            update_meeting_history(asof)            # pure math off the store just written
        except Exception:
            pass
        return len(set(prices) | set(settles))
    except Exception as e:
        _strip_error_write("exception", repr(e))
        return 0


def live_strip_prices(contracts: list[Contract]) -> dict[str, float] | None:
    """The '⚡ Live pull' button's worker: ONE bdp for exactly these contracts'
    tickers — nothing else. Returns {code: px} for what came back, or None when
    the pull failed entirely (the page must say so loudly, never fall to mock)."""
    if MODE != "bloomberg":
        return None
    try:
        from . import bbg as blp
        from . import datafeed as _dfeed
        tickers = [f"{c.code} Comdty" for c in contracts]
        rows = _dfeed._bdp_rows(_dfeed._coerce_pd(blp.bdp(tickers, "PX_LAST")))
        if not rows:
            return None
        out = {}
        for c in contracts:
            v = (rows.get(f"{c.code} Comdty") or {}).get("PX_LAST")
            if v is not None and v == v:
                out[c.code] = float(v)
        return out or None
    except Exception:
        return None


def _mock_prices(prod: Product, bank: Bank, contracts: list[Contract], asof: date,
                 asof_rate: float) -> dict[str, float]:
    step, every = _MOCK_STYLE.get(bank.key, (-25.0, 2))
    ups = [m for m in bank.meetings if m > asof]
    moves = [(step if i % every == 0 else 0.0) for i in range(len(ups))]
    fn = overnight_rate_fn(asof_rate, ups, moves)
    out = {}
    for i, c in enumerate(contracts):
        if prod.rate_quoted:
            q = di_fair_rate(c, fn, asof)
            out[c.code] = round(q + (0.5 - ((i * 7) % 5) / 4.0) * 0.01, 3)
        else:
            p = fair_price(prod, c, fn)
            out[c.code] = round(p + (0.5 - ((i * 7) % 5) / 4.0) * 0.004, 4)
    return out


def strip_prices(prod: Product, bank: Bank, contracts: list[Contract], asof: date,
                 asof_rate: float) -> list[float]:
    """Prices per contract, NO network: the morning-snapshot store first (PX_LAST,
    else PX_SETTLE), synthetic mock for anything the store doesn't carry. Use
    `strip_source()` for an honest label of what a strip is running on."""
    store = _load_strip_store()
    px, se = store.get("prices", {}), store.get("settles", {})
    mock = _mock_prices(prod, bank, contracts, asof, asof_rate)
    return [float(px.get(c.code, se.get(c.code, mock[c.code]))) for c in contracts]


# ── daily per-meeting repricing history (the "what moved" ledger) ────────────────────
# Written by the morning snapshot (after refresh_strip_store) — one entry per day of
# {bank: {meeting iso: implied bp}} computed off the store with the default strips and
# rates. The absolute level leans on those defaults; DAY-OVER-DAY DELTAS are the read.
MEETING_HISTORY = Path(__file__).resolve().parents[1] / "data" / "stir_meeting_history.json"
_HISTORY_KEEP_DAYS = 40


# Contracts whose settles are exchange MARKS with no open interest (dead 1M
# SONIA): including them DRAGS the fit off the liquid market — SOO's Sep-26 mark
# priced zero hike odds while the 3M strip and OIS both said ~+20%.
FIT_EXCLUDE = {"SOOA Comdty"}


def fit_instruments(bank_key: str, asof: date, r0: float | None = None,
                    override_prices: dict | None = None):
    """(owners, contracts, spreads_bp, prices): EVERY liquid instrument the
    bank's market odds should be fit from — quarterly strips, SR1/FF monthlies,
    ER/€STR serials — independent of what any page DISPLAYS. Store-aware: with
    a live store only store-priced (or page-overridden) contracts enter; a
    silent per-contract mock fallback would pollute a real fit. In demo mode
    everything enters (the mock generator is self-consistent)."""
    bank = BANKS[bank_key]
    if r0 is None:
        r0 = bank.default_rate + BANK_BASIS_SEED[bank_key] / 100.0
    store = _load_strip_store()
    have = set(store.get("prices", {})) | set(store.get("settles", {}))
    live = bool(have)
    ov = override_prices or {}
    owners, contracts, spreads, prices = [], [], [], []
    # The candidate set IS the pull universe (quarterlies + monthlies capped
    # ~13mo where SER-vs-FF marks stay coherent + serials) — same function the
    # morning pull fetches, so the store can never starve the fit again.
    for p, c in pull_universe(asof):
        if p.bank != bank_key or p.ticker in FIT_EXCLUDE:
            continue
        if fut_last_trade(p, c) < asof:
            continue
        if live and c.code not in have and c.code not in ov:
            continue
        # Drop nearly-dead contracts (>70% of the window already elapsed):
        # they are one number about HISTORY, and when a policy move sits in
        # that history the stub model cannot honour them — TKYK6 (96%
        # realized, straddling the June ECB hike) alone dragged the solved
        # stub 3bp low and printed +12pts of phantom front odds.
        total = max(1, (c.end - c.start).days)
        if (min(asof, c.end) - c.start).days / total > 0.70:
            continue
        px = ov.get(c.code)
        if px is None:
            px = strip_prices(p, bank, [c], asof, r0)[0]
        if not 90.0 < float(px) < 100.5:            # defense-in-depth vs a poisoned
            continue                                # store or a fat-fingered edit
        owners.append(p)
        contracts.append(c)
        spreads.append(p.spread_bp)
        prices.append(float(px))
    return owners, contracts, spreads, prices


def store_codes() -> set:
    """Codes the live store actually prices — the page's override filter uses
    this to keep mock-seeded display values out of a real fit."""
    store = _load_strip_store()
    return set(store.get("prices", {})) | set(store.get("settles", {}))


def clean_month_anchor(bank_key: str, asof: date,
                       override_prices: dict | None = None) -> tuple[float, str] | None:
    """The market's own read of the CURRENT policy rate: a monthly contract
    whose window contains no decision is a pure average of the prevailing
    overnight rate — no model, no fit. Returns (implied policy %, code) from
    the nearest such priced month, or None (needs monthlies — Fed only for
    now). This is the guard against a stale hand-maintained default_rate."""
    bank = BANKS[bank_key]
    store = _load_strip_store()
    ov = override_prices or {}
    px_of = {**store.get("settles", {}), **store.get("prices", {}), **ov}
    for p in bank_products(bank_key):
        if p.quarterly or p.ticker in FIT_EXCLUDE:
            continue
        for c in strip(p, asof, 3):
            if meetings_in_window(bank, c):
                continue
            px = px_of.get(c.code)
            if px is None:
                continue
            proxy = 100.0 - float(px) - p.spread_bp / 100.0
            return proxy - BANK_BASIS_SEED[bank_key] / 100.0, c.code
    return None


@dataclass
class BankFit:
    implied: BankImplied
    pinned: list[bool]          # per meeting: True = the data pins it, False =
                                # the smoothing chose it (indicative only)
    anchor: tuple[float, str] | None   # clean-month implied CURRENT policy rate
    n_instruments: int


def bank_fit(bank_key: str, asof: date, r0: float | None = None,
             stub_rate: float | None = None,
             override_prices: dict | None = None,
             override_spreads: dict | None = None) -> BankFit | None:
    """The one-stop full-accuracy market fit: all liquid instruments, the front
    stub solved from the market (unless a realized average is passed), and a
    per-meeting pinned/interpolated diagnostic (odds stable when the smoothing
    weight changes 8x = the data decided them; moving = the smoothing did).

    The starting rate is the CLEAN-MONTH ANCHOR when one prices (a no-meeting
    monthly is the market's own read of the current o/n level — e.g. it knows
    the effective rate is 3.63x when the hand-set band mid says 3.625, worth
    several points of front-meeting odds), sanity-clamped to ±40bp of the
    registry default so a corrupt store can't hijack the level. Pass `r0` to
    override both (the page's band/rate input does)."""
    bank = BANKS[bank_key]
    if bank_key == "BCB":
        return _bcb_fit(asof, r0=r0, override_prices=override_prices)
    anchor = clean_month_anchor(bank_key, asof, override_prices)
    if r0 is None:
        r0 = bank.default_rate + BANK_BASIS_SEED[bank_key] / 100.0
        if anchor is not None and abs(anchor[0] - bank.default_rate) <= 0.40:
            r0 = anchor[0] + BANK_BASIS_SEED[bank_key] / 100.0
    owners, contracts, spreads, prices = fit_instruments(
        bank_key, asof, r0=r0, override_prices=override_prices)
    if not contracts:
        return None
    if override_spreads:
        spreads = [override_spreads.get(c.code, s) for c, s in zip(contracts, spreads)]
    # lam 4x the legacy default: with dedicated pinning instruments in the fit
    # the data overwhelms the smoothing wherever it's real (Fed odds move <0.1pt
    # between 5e-3 and 2e-2), and the firmer floor stops serial-pair mark noise
    # printing implausible mid-path sawtooth (ECB Mar-27 was -8.9% mid-hiking-cycle).
    kw = dict(spreads_bp=spreads, stub_rate=stub_rate, solve_stub=stub_rate is None)
    ip = implied_path(bank, contracts, prices, asof, r0, **kw, lam=2e-2)
    # PINNED needs BOTH tests — the review measured each alone over-claiming:
    # (1) STRUCTURE: some fitted window contains the meeting alone, or two
    #     windows differ by exactly it (the serial-pair difference);
    # (2) STABILITY: the number barely moves when the smoothing weight rises 8x
    #     (structure can run through few-day slivers and conflicting marks that
    #     the smoothing still arbitrates).
    tight = implied_path(bank, contracts, prices, asof, r0, **kw, lam=2e-2 * 8)
    live_m = set(ip.meetings)
    msets = {frozenset(x for x in meetings_in_window(bank, c) if x in live_m)
             for c in contracts}
    singles = {next(iter(s)) for s in msets if len(s) == 1}
    for a in msets:
        for b in msets:
            d = a - b
            if len(d) == 1:
                singles.add(next(iter(d)))
    pins = [m in singles and abs(float(a) - float(b)) < 1.5
            for m, a, b in zip(ip.meetings, ip.per_meeting_bp, tight.per_meeting_bp)]
    return BankFit(ip, pins, anchor, len(contracts))


def _bcb_fit(asof: date, r0: float | None = None,
             override_prices: dict | None = None) -> BankFit | None:
    """bank_fit for Brazil: the DI strip, rate-quoted zeros, log-space fit.
    Store-fed only (root unverified → in_pull=False): live store or page
    overrides supply quotes; the synthetic demo covers everything else in
    demo mode, and in live mode unquoted maturities simply drop out. The
    ANCHOR is the front DI with no Copom before its maturity — a pure read
    of current CDI, trusted over the provisional registry Selic (wide 3%
    tolerance: the registry seed is a placeholder)."""
    bank = BANKS["BCB"]
    prod = PRODUCTS["OD1 Comdty"]
    store = _load_strip_store()
    have = {**store.get("settles", {}), **store.get("prices", {})}
    live = bool(set(store.get("prices", {})) | set(store.get("settles", {})))
    ov = override_prices or {}
    r_seed = (r0 if r0 is not None
              else bank.default_rate + BANK_BASIS_SEED["BCB"] / 100.0)
    cs, qs = [], []
    for c in di_strip(prod, asof):
        q = ov.get(c.code, have.get(c.code))
        if q is None:
            if live:                                # never mock inside a real fit
                continue
            q = _mock_prices(prod, bank, [c], asof, r_seed)[c.code]
        if not 0.5 < float(q) < 40.0:
            continue
        cs.append(c)
        qs.append(float(q))
    if not cs:
        return None
    # front-DI anchor: no meeting before its maturity = the market's own read
    # of current CDI, hence the Selic target (CDI ≈ Selic − 10bp)
    anchor = None
    for c, q in zip(cs, qs):
        if not meetings_in_window(bank, c):
            anchor = (q - BANK_BASIS_SEED["BCB"] / 100.0, c.code)
            break
    if r0 is None:
        if anchor is not None and abs(anchor[0] - bank.default_rate) <= 3.0:
            r_seed = anchor[0] + BANK_BASIS_SEED["BCB"] / 100.0
        r0 = r_seed
    ip = di_implied_path(bank, cs, qs, asof, r0)
    live_m = set(ip.meetings)
    msets = {frozenset(x for x in meetings_in_window(bank, c) if x in live_m)
             for c in cs}
    singles = {next(iter(s)) for s in msets if len(s) == 1}
    for a in msets:
        for b in msets:
            d = a - b
            if len(d) == 1:
                singles.add(next(iter(d)))
    tight = di_implied_path(bank, cs, qs, asof, r0, lam=2e-2 * 8)
    pins = [m in singles and abs(float(a) - float(b)) < 1.5
            for m, a, b in zip(ip.meetings, ip.per_meeting_bp, tight.per_meeting_bp)]
    return BankFit(ip, pins, anchor, len(cs))


def default_bank_fit(bank_key: str, asof: date) -> BankImplied | None:
    """The shared default fit (home page, ledger, weekly review): now the full
    liquid-instrument fit with the market-solved stub — no longer quarterlies-only."""
    bf = bank_fit(bank_key, asof)
    return bf.implied if bf is not None else None


def update_meeting_history(asof: date) -> None:
    """Append today's per-meeting implied map — ONLY when the strip store is real
    (never record mock fits as history)."""
    store = _load_strip_store()
    if not (store.get("prices") or store.get("settles")):
        return
    try:
        hist = json.loads(MEETING_HISTORY.read_text(encoding="utf-8"))
    except Exception:
        hist = {}
    entry = {}
    for bk in BANKS:
        ip = default_bank_fit(bk, asof)
        if ip is not None:
            entry[bk] = {m.isoformat(): float(bp)
                         for m, bp in zip(ip.meetings, ip.per_meeting_bp)}
    if not entry:
        return
    hist[asof.isoformat()] = entry
    keep = sorted(hist)[-_HISTORY_KEEP_DAYS:]
    MEETING_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    MEETING_HISTORY.write_text(json.dumps({k: hist[k] for k in keep}, indent=1),
                               encoding="utf-8")


def meeting_repricing(asof: date) -> dict[str, dict[str, tuple[float, float]]]:
    """{bank: {meeting iso: (bp_now, bp_change_vs_prior_entry)}} from the ledger's
    two most recent distinct days — {} until two mornings have recorded."""
    try:
        hist = json.loads(MEETING_HISTORY.read_text(encoding="utf-8"))
    except Exception:
        return {}
    days = sorted(hist)
    if len(days) < 2:
        return {}
    cur, prev = hist[days[-1]], hist[days[-2]]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for bk, mp in cur.items():
        pv = prev.get(bk, {})
        out[bk] = {iso: (bp, bp - pv[iso]) for iso, bp in mp.items() if iso in pv}
    return out


def strip_source(contracts: list[Contract]) -> tuple[str, str | None]:
    """('snapshot', asof) when the store covers most of these contracts, else
    ('mock', None) — the page's price-provenance label."""
    store = _load_strip_store()
    have = set(store.get("prices", {})) | set(store.get("settles", {}))
    codes = [c.code for c in contracts]
    if codes and sum(1 for c in codes if c in have) >= max(1, len(codes) // 2):
        return "snapshot", store.get("asof")
    return "mock", None


# ── Hot Sheet provider (src/hotsheet.py discovers radar_items by name) ────────────────
# A meeting must reprice this much vs the prior ledger day to make the Hot Sheet —
# small enough to catch a real repricing, big enough to ignore day-to-day fit noise.
STIR_DAY_MIN_BP = 3.0


def radar_items() -> list:
    """Daily Hot Sheet items: decision-week alerts (weekreview.collect_stir's
    selection, minus the options-expiry embellishment) plus the day's biggest
    per-meeting repricings off the ledger. Store-only by construction —
    default_bank_fit prices off the morning strip store and meeting_repricing
    reads the history ledger; nothing here can touch Bloomberg."""
    from src import hotsheet

    today = date.today()
    out = []
    # (a) meeting-week alerts: a central bank decides within 7 days.
    for bk, bank in BANKS.items():
        soon = [m for m in bank.meetings if today <= m <= today + timedelta(days=7)]
        if not soon:
            continue
        ip = default_bank_fit(bk, today)
        per = ({m.isoformat(): float(bp) for m, bp in zip(ip.meetings, ip.per_meeting_bp)}
               if ip is not None else {})
        for m in soon:
            bp = per.get(m.isoformat(), 0.0)
            d, p = implied_odds(bp, bank.step_bp)
            odds = "a hold" if d == "hold" else f"{p * 100:.0f}% of a {d}"
            # Heat floors at 50 so a live decision always surfaces, even when the
            # strip prices a clean hold — the event itself is the story, and the
            # priced move only scales it from there (one full step = 100).
            out.append(hotsheet.item(
                tag="STIR",
                key=f"{bk}:{m.isoformat()}",     # the date IS the story — stable
                section="STIR / Policy",
                text=(f"**{bank.name}** decides **{m:%A %d %b}** — the strip prices "
                      f"**{bp:+.0f}bp**, {odds}."),
                metric=f"{bp:+.0f} bp", sub=f"{bank.meeting_name} this week",
                heat=50.0 + 50.0 * min(1.0, abs(bp) / bank.step_bp),
                value=float(bp), page="STIR Timeline", book="ficc"))
    # (b) daily repricings: the ledger's per-meeting day-over-day deltas, top 3.
    moves = []
    for bk, per in meeting_repricing(today).items():
        for iso, (bp, chg) in per.items():
            if abs(chg) >= STIR_DAY_MIN_BP:
                moves.append((abs(chg), bk, iso, bp, chg))
    for _, bk, iso, bp, chg in sorted(moves, reverse=True)[:3]:
        bank = BANKS[bk]
        m = date.fromisoformat(iso)
        word = "added easing" if chg < 0 else "took easing out"
        out.append(hotsheet.item(
            tag="STIR", key=f"{bk}:reprice:{iso}", section="STIR / Policy",
            text=(f"**{bank.name} repricing** — the market {word} at the {m:%b %Y} "
                  f"{bank.meeting_name}: **{chg:+.0f}bp** on the day "
                  f"(now {bp:+.0f}bp priced)."),
            metric=f"{chg:+.0f} bp", sub="on the day",
            heat=min(100.0, abs(chg) / 10.0 * 100.0),
            value=float(bp), page="STIR Timeline", book="ficc"))
    return out
