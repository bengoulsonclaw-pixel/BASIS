"""Futures-price <-> yield conversion for the fixed-income book (the Fut / Yield page).

Two very different conversions live here:

  * STIRs (SOFR / Euribor / SONIA / ESTR / Fed Funds) — the IMM convention:
        implied rate = 100 - price.
    Exact by construction; the result is a money-market (add-on, ACT/360 or
    ACT/365 for sterling) forward rate for the contract's reference window.

  * BOND futures — a futures price has no yield of its own; the market reads it
    through the cheapest-to-deliver (CTD): forward CTD price ~= futures x
    conversion factor, and the "futures yield" is the CTD's yield at that price.
    That needs per-contract CTD assumptions (coupon, maturity, conversion
    factor). We seed indicative values and let the desk overwrite them from the
    delivery-basket screens (DLV) — edits persist to data/futyield.json, same
    per-field-merge pattern as blocksizes.py. The bond model is the standard
    whole-period fixed-coupon price/yield relation (clean price, no accrued /
    settlement-date day counts) — right to well under a bp of yield for a
    converter, but it is NOT a delivery-option model.

The yields the two sections produce are in DIFFERENT conventions (money-market
add-on vs bond semi/annual compounding) — comparable to their own curves, not
directly to each other.
"""
from __future__ import annotations

import json
from pathlib import Path

# ── STIR contracts ──────────────────────────────────────────────────────────
# ticker -> (underlying index, tenor, currency, day count). Price <-> rate is
# 100 - x both ways; the metadata is for display and the $/bp column.
STIRS = {
    "SERA Comdty": ("SOFR (compounded)", "1M", "USD", "ACT/360"),
    "SFRA Comdty": ("SOFR (compounded)", "3M", "USD", "ACT/360"),
    "FFA Comdty":  ("Fed Funds (avg)",   "1M", "USD", "ACT/360"),
    "ERA Comdty":  ("Euribor (fixing)",  "3M", "EUR", "ACT/360"),
    "TKYA Comdty": ("ESTR (compounded)", "3M", "EUR", "ACT/360"),
    "SFIA Comdty": ("SONIA (compounded)", "3M", "GBP", "ACT/365"),
}


def stir_rate(price: float) -> float:
    """Implied forward rate (%) from an IMM-quoted STIR futures price."""
    return 100.0 - price


def stir_price(rate: float) -> float:
    """IMM futures price from a forward rate (%)."""
    return 100.0 - rate


# ── bond price / yield maths ────────────────────────────────────────────────
def bond_price(coupon: float, ytm: float, years: float, freq: int = 2) -> float:
    """Clean price per 100 face of a fixed-coupon bond: `coupon` and `ytm` in %,
    `freq` coupons per year, whole coupon periods (n = round(years*freq))."""
    n = max(1, round(years * freq))
    c = coupon / freq
    y = ytm / 100.0 / freq
    if abs(y) < 1e-12:
        return c * n + 100.0
    d = (1.0 + y) ** -n
    return c * (1.0 - d) / y + 100.0 * d


def bond_yield(price: float, coupon: float, years: float, freq: int = 2) -> float:
    """Yield to maturity (%) from a clean price — bisection, robust for any input."""
    lo, hi = -5.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bond_price(coupon, mid, years, freq) > price:
            lo = mid          # price too high at this yield -> yield must be higher
        else:
            hi = mid
    return 0.5 * (lo + hi)


def conversion_factor(coupon: float, years: float, freq: int = 2,
                      notional_coupon: float = 6.0) -> float:
    """Indicative conversion factor: the CTD priced at the contract's notional
    coupon, per 1 face. (Exchanges round maturities to month/quarter grids —
    this is the textbook value, close enough to seed the editable table.)"""
    return bond_price(coupon, notional_coupon, years, freq) / 100.0


def ctd_yield(fut_price: float, cf: float, coupon: float, years: float,
              freq: int = 2) -> float:
    """Implied CTD yield (%) read through the futures: price x CF -> solve."""
    return bond_yield(fut_price * cf, coupon, years, freq)


def fut_price_from_yield(ytm: float, cf: float, coupon: float, years: float,
                         freq: int = 2) -> float:
    """Futures price consistent with a target CTD yield (%): price(y) / CF."""
    return bond_price(coupon, ytm, years, freq) / cf


def fut_dv01(fut_price: float, cf: float, coupon: float, years: float,
             freq: int = 2, point_value: float = 1000.0) -> float:
    """Futures DV01 per lot (contract currency per 1bp of CTD yield):
    dF/dy = (dP_ctd/dy) / CF, scaled by the contract's value per price point."""
    y = ctd_yield(fut_price, cf, coupon, years, freq)
    dp = (bond_price(coupon, y - 0.01, years, freq)
          - bond_price(coupon, y + 0.01, years, freq)) / 2.0   # CTD price pts per 1bp
    return dp / cf * point_value


# ── CTD assumptions (seed; editable + persisted) ────────────────────────────
# ticker -> {coupon %, years to maturity, coupons/yr, notional coupon %}.
# INDICATIVE mid-2026 placeholders — the whole point of the editable table is
# that the desk keeps these current from the delivery-basket (DLV) screens.
# CF is derived from these via conversion_factor() unless overridden.
_SEED = {
    # CME treasuries — 6% notional, semiannual CTD coupons
    "TUA Comdty":  {"coupon": 4.000, "years": 1.9,  "freq": 2, "notional": 6.0},
    "FVA Comdty":  {"coupon": 4.000, "years": 4.4,  "freq": 2, "notional": 6.0},
    "TYA Comdty":  {"coupon": 4.250, "years": 7.0,  "freq": 2, "notional": 6.0},
    "UXYA Comdty": {"coupon": 4.500, "years": 9.7,  "freq": 2, "notional": 6.0},
    "USA Comdty":  {"coupon": 4.500, "years": 16.5, "freq": 2, "notional": 6.0},
    "WNA Comdty":  {"coupon": 4.625, "years": 29.0, "freq": 2, "notional": 6.0},
    # Eurex — 6% notional, ANNUAL CTD coupons
    "DUA Comdty":  {"coupon": 2.200, "years": 2.0,  "freq": 1, "notional": 6.0},
    "OEA Comdty":  {"coupon": 2.400, "years": 5.0,  "freq": 1, "notional": 6.0},
    "RXA Comdty":  {"coupon": 2.600, "years": 9.0,  "freq": 1, "notional": 6.0},
    "UBA Comdty":  {"coupon": 2.900, "years": 28.0, "freq": 1, "notional": 6.0},
    "OATA Comdty": {"coupon": 3.000, "years": 9.0,  "freq": 1, "notional": 6.0},
    # ICE Long Gilt — 4% notional, semiannual coupons
    "G A Comdty":  {"coupon": 4.250, "years": 9.0,  "freq": 2, "notional": 4.0},
}

STORE = Path(__file__).resolve().parents[1] / "data" / "futyield.json"


def load_ctd() -> dict:
    """ticker -> {coupon, years, freq, notional, cf}. Stored fields win over the
    seed PER FIELD; cf falls back to conversion_factor() when not overridden."""
    if STORE.exists():
        try:
            stored = json.loads(STORE.read_text(encoding="utf-8")).get("ctd", {})
        except Exception:
            stored = {}
    else:
        stored = {}
    out = {}
    for tk, seed in _SEED.items():
        e = {**seed, **{k: v for k, v in stored.get(tk, {}).items() if v is not None}}
        if not e.get("cf"):
            e["cf"] = round(conversion_factor(e["coupon"], e["years"],
                                              int(e["freq"]), e["notional"]), 4)
        out[tk] = e
    return out


def save_ctd(m: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps({"ctd": m}, indent=2), encoding="utf-8")
