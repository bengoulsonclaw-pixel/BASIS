"""Implied Fed-path engine for the SOFR strip (BASIS · Fed Path Calculator).

Two questions, one consistent pricing model:

  1.  MARKET-IMPLIED  — given the live 3M SOFR futures (SR3) strip, back out the
      step-function path of the overnight rate that only moves on FOMC dates, i.e.
      how many hikes/cuts the curve is pricing and where the target band lands at
      each meeting. (Same idea as CME FedWatch, but read straight off the strip
      the desk actually trades.)

  2.  YOUR SCENARIO   — given a hand-set move at each upcoming FOMC (±25bp steps,
      or hold), price the WHOLE strip off that path, so the trader sees where each
      contract *should* trade if the Fed does as they think — right next to the
      live market price. The gap is the mispricing / edge if they're right.

Both directions run through ONE pricing function (`contract_rate`), so the two
paths are directly comparable and a "match market" scenario reproduces the strip.

Rate conventions
----------------
* The overnight rate (SOFR) is modelled as a step function that changes only on
  the business day AFTER each FOMC decision (the target change is effective the
  next day). Between meetings it is flat.
* SOFR ≈ Fed target midpoint + a small `basis` (a few bp); the basis shifts every
  fair value in parallel but cancels out of the *cumulative* move, so the implied
  number of hikes/cuts is basis-independent.
* SR3 settles to the daily-COMPOUNDED SOFR over its IMM reference quarter
  (ACT/360). SR1 settles to the simple AVERAGE over the calendar month. Both are
  computed by walking calendar days (weekends carry the prior rate — exactly the
  published-index convention), so no separate holiday calendar is needed.

This module is pure (datetime + numpy only) so it can be unit-tested and reused
by both the Streamlit page and the PDF report. Live Bloomberg pricing is a thin
adapter (`strip_prices`) that mirrors the rest of the app's mock→bloomberg swap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

# --- FOMC calendar ---------------------------------------------------------
# Decision (announcement) dates — the SECOND day of each two-day meeting. Source:
# federalreserve.gov/monetarypolicy/fomccalendars.htm (2027 is tentative until
# confirmed at the preceding meeting). Extend this list as the Fed publishes new
# years; everything downstream keys off it.
FOMC_DECISIONS: list[date] = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 9),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 8),
]

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# CME/Bloomberg quarterly month codes for the IMM cycle (Mar/Jun/Sep/Dec).
_QUARTERLY_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}

# SR3 contract economics: $25 per basis point (0.01 price move = $25).
SR3_BP_VALUE = 25.0


def next_weekday(d: date) -> date:
    """Roll `d` forward to the next Mon–Fri (used for the rate's effective date —
    the business day after a decision). Holidays don't change the rate level, so
    weekend-only rolling is sufficient for a day-weighted average."""
    while d.weekday() >= 5:            # 5 = Sat, 6 = Sun
        d += timedelta(days=1)
    return d


def effective_date(decision: date) -> date:
    """The day a decision starts affecting the overnight rate: the next business
    day after the announcement."""
    return next_weekday(decision + timedelta(days=1))


def upcoming_meetings(asof: date) -> list[date]:
    """FOMC decision dates strictly after `asof`, in order — the meetings a trader
    can still express a view on."""
    return [d for d in FOMC_DECISIONS if d > asof]


def meeting_label(decision: date) -> str:
    """'Sep 2026' — a meeting's display label (keyed off the decision month)."""
    return f"{_MONTHS[decision.month - 1]} {decision.year}"


def third_wednesday(year: int, month: int) -> date:
    """IMM date — the third Wednesday of the month (SR3 reference-quarter bound)."""
    d = date(year, month, 1)
    # weekday(): Mon=0 … Wed=2. Days to the first Wednesday, then +14 for the third.
    first_wed = 1 + (2 - d.weekday()) % 7
    return date(year, month, first_wed + 14)


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    m0 = (year * 12 + (month - 1)) + n
    return m0 // 12, m0 % 12 + 1


@dataclass(frozen=True)
class Contract:
    """One SR3 quarterly future: its IMM reference window [start, end) and tickers."""
    code: str          # e.g. 'SFRU6' (Bloomberg generic quarterly)
    label: str         # e.g. 'Sep-26'
    year: int
    month: int         # IMM month (3/6/9/12)
    start: date        # 3rd Wednesday of the IMM month (accrual start)
    end: date          # 3rd Wednesday of IMM month + 3 (accrual end, exclusive)

    @property
    def bbg(self) -> str:
        return f"{self.code} Comdty"


def sr3_contract(year: int, month: int) -> Contract:
    """Build the SR3 quarterly contract for a given IMM (year, month)."""
    ey, em = _add_months(year, month, 3)
    start, end = third_wednesday(year, month), third_wednesday(ey, em)
    code = f"SFR{_QUARTERLY_CODE[month]}{year % 10}"
    return Contract(code, f"{_MONTHS[month - 1]}-{year % 100:02d}", year, month, start, end)


def sr3_strip(asof: date, n: int = 8) -> list[Contract]:
    """The front `n` SR3 quarterlies from `asof`: the currently-accruing contract
    (the IMM quarter containing today) plus the next n-1. This is the whites + reds
    the desk trades the Fed path in."""
    # Find the IMM quarter currently accruing: the latest IMM start on/before asof.
    y, m = asof.year, ((asof.month - 1) // 3) * 3 + 3   # this year's IMM month >= ...
    # step back until the IMM start is on/before asof
    while third_wednesday(y, m) > asof:
        y, m = _add_months(y, m, -3)
    out = []
    for _ in range(n):
        out.append(sr3_contract(y, m))
        y, m = _add_months(y, m, 3)
    return out


# --- the overnight-rate path ----------------------------------------------
def overnight_rate_fn(asof_rate: float, meetings: list[date], moves_bp: list[float]):
    """Return rate(d) — the step-function overnight rate (in %), starting at
    `asof_rate` and stepping by `moves_bp[i]` (basis points) on the business day
    after meeting `meetings[i]`. Days before the first move return `asof_rate`."""
    steps = sorted((effective_date(mtg), bp / 100.0) for mtg, bp in zip(meetings, moves_bp))

    def rate(d: date) -> float:
        r = asof_rate
        for eff, delta in steps:
            if d >= eff:
                r += delta
            else:
                break
        return r
    return rate


def _daterange(start: date, end: date):
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)


def contract_rate(contract: Contract, rate_fn, compound: bool = True) -> float:
    """The settlement rate (%) a contract prints under overnight path `rate_fn`.

    compound=True  → daily-compounded ACT/360 over the reference window (the true
                     SR3 convention). compound=False → simple calendar-day average
                     (the SR1 convention, and the linear form used for inversion).
    Walks calendar days; weekends inherit the prevailing rate, matching the
    published-index day-count."""
    days = list(_daterange(contract.start, contract.end))
    n = len(days)
    if n == 0:
        return float("nan")
    if not compound:
        return float(np.mean([rate_fn(d) for d in days]))
    factor = 1.0
    for d in days:
        factor *= 1.0 + (rate_fn(d) / 100.0) / 360.0
    return (factor - 1.0) * 360.0 / n * 100.0


def price(contract: Contract, rate_fn, compound: bool = True) -> float:
    """Futures price = 100 − settlement rate."""
    return 100.0 - contract_rate(contract, rate_fn, compound)


# --- market-implied path (invert the strip) -------------------------------
@dataclass
class ImpliedPath:
    """The step path the strip is pricing.

    meetings         upcoming FOMC decision dates covered
    seg_rates        overnight rate (%) in each segment: [current, post-mtg1, …]
    per_meeting_bp   the move priced at each meeting (bp) = seg_rates diffs ×100
    cum_bp           cumulative move from today at each meeting (bp)
    fair_price       model price of each input contract at the fitted path
    residual_bp      contract mispricing left after the fit (market − fair), in bp
    """
    contracts: list[Contract]
    meetings: list[date]
    seg_rates: np.ndarray
    per_meeting_bp: np.ndarray
    cum_bp: np.ndarray
    fair_price: np.ndarray
    residual_bp: np.ndarray


def implied_path(contracts: list[Contract], prices: list[float], asof: date,
                 asof_rate: float) -> ImpliedPath:
    """Bootstrap the meeting-step overnight path from SR3 prices by least squares.

    Model: contract c's rate ≈ Σ_s w[c,s]·L[s], where L[s] is the overnight level
    in inter-meeting segment s and w[c,s] is the fraction of c's reference window
    (calendar days) falling in segment s. L[0] (today→first meeting) is pinned to
    `asof_rate`; the rest are solved. Uses the simple-average form (compound off)
    so the system is linear — the compounding convexity is <~1bp over a quarter
    and washes out of the inferred path.
    """
    meetings = [effective_date(m) for m in FOMC_DECISIONS if asof < m < max(c.end for c in contracts)]
    decisions = [m for m in FOMC_DECISIONS if asof < m < max(c.end for c in contracts)]
    bounds = [asof] + meetings                       # segment s spans [bounds[s], bounds[s+1])
    n_seg = len(bounds)

    W = np.zeros((len(contracts), n_seg))
    for ci, c in enumerate(contracts):
        days = list(_daterange(c.start, c.end))
        if not days:
            continue
        for d in days:
            s = 0
            for b in meetings:
                if d >= b:
                    s += 1
                else:
                    break
            W[ci, s] += 1.0
        W[ci, :] /= len(days)

    y = np.array([100.0 - p for p in prices])        # implied avg rate per contract
    # Pin L[0] = asof_rate: move its contribution to the RHS, solve for L[1:].
    rhs = y - W[:, 0] * asof_rate
    if n_seg > 1:
        sol, *_ = np.linalg.lstsq(W[:, 1:], rhs, rcond=None)
        seg = np.concatenate([[asof_rate], sol])
    else:
        seg = np.array([asof_rate])

    fair = np.array([100.0 - float(W[ci] @ seg) for ci in range(len(contracts))])
    residual_bp = (np.array(prices) - fair) * 100.0
    per_meeting_bp = np.diff(seg) * 100.0
    cum_bp = (seg[1:] - seg[0]) * 100.0
    return ImpliedPath(contracts, decisions, seg, per_meeting_bp, cum_bp, fair, residual_bp)


# --- data feed (mock → bloomberg, mirrors src/datafeed.py) -----------------
MODE = os.getenv("DATAFEED_MODE", "mock")


def strip_prices(contracts: list[Contract], asof: date, asof_rate: float) -> list[float]:
    """Live SR3 prices for the strip. Bloomberg in live mode; a realistic synthetic
    easing curve in mock/demo mode so the tool is fully explorable offline."""
    if MODE == "bloomberg":
        try:
            return _bloomberg_strip(contracts)
        except Exception:
            pass                                     # fall through to mock on any feed error
    return _mock_strip(contracts, asof, asof_rate)


def _bloomberg_strip(contracts: list[Contract]) -> list[float]:
    """PX_LAST for each contract's generic quarterly ticker (e.g. 'SFRU6 Comdty')."""
    from . import bbg as blp                             # imported lazily; only on the Terminal
    from .datafeed import _bdp_rows, _coerce_pd      # handles the binding's narwhals LONG shape
    tickers = [c.bbg for c in contracts]
    rows = _bdp_rows(_coerce_pd(blp.bdp(tickers, "PX_LAST")))
    return [float(rows[t]["PX_LAST"]) for t in tickers]


def _mock_strip(contracts: list[Contract], asof: date, asof_rate: float) -> list[float]:
    """Synthetic strip: assume the market prices ~25bp of easing every OTHER meeting
    (a gentle cutting cycle), price the strip off that path, add a touch of
    deterministic curve noise. Recovers a sensible implied path when inverted."""
    ups = upcoming_meetings(asof)
    mock_moves = [(-25.0 if i % 2 == 0 else 0.0) for i in range(len(ups))]
    rate_fn = overnight_rate_fn(asof_rate, ups, mock_moves)
    out = []
    for i, c in enumerate(contracts):
        p = price(c, rate_fn, compound=True)
        p += (0.5 - ((i * 7) % 5) / 4.0) * 0.004     # ±~0.4bp deterministic wobble
        out.append(round(p, 4))
    return out
