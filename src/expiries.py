"""Indicative "next expiry" calculator for the Market Hours tooltip.

For each contract FAMILY we encode: the listed delivery-month CYCLE, a FUTURES
last-trading-day rule and an OPTIONS expiry rule (each a small (kind, *args) tuple
evaluated per contract month), and an indicative last-trade TIME. `describe()` returns
the next futures and next options expiry (date + indicative time) for a product, from a
reference date. Business-day maths is holiday-aware via the Market Hours exchange
calendars, so a rule like "3 business days before the 25th" skips exchange holidays.

These are INDICATIVE, rule-based estimates of the STANDARD contract cycle — good enough
to see the next roll/expiry at a glance, not an exchange-official calendar. Easy to
correct in SPECS below; upgradeable to exact Bloomberg LAST_TRADEABLE_DT later.
"""
from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta

from . import markethours

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)


# ── business-day / weekday maths ──────────────────────────────────────────────────────
def _is_bday(d: date, hol) -> bool:
    return d.weekday() < 5 and d.isoformat() not in hol


def _prev_bday(d: date, hol) -> date:
    d -= timedelta(days=1)
    while not _is_bday(d, hol):
        d -= timedelta(days=1)
    return d


def _on_or_before_bday(d: date, hol) -> date:
    while not _is_bday(d, hol):
        d -= timedelta(days=1)
    return d


def _next_bday(d: date, hol) -> date:
    d += timedelta(days=1)
    while not _is_bday(d, hol):
        d += timedelta(days=1)
    return d


def _on_or_after_bday(d: date, hol) -> date:
    while not _is_bday(d, hol):
        d += timedelta(days=1)
    return d


def _bdays_before(d: date, n: int, hol) -> date:
    """The business day `n` business days before `d` (d itself not counted)."""
    for _ in range(n):
        d = _prev_bday(d, hol)
    return d


def _nth_bday(y: int, m: int, n: int, hol) -> date:
    """The n-th business day of month m (Lean Hogs)."""
    d = _on_or_after_bday(date(y, m, 1), hol)
    for _ in range(n - 1):
        d = _next_bday(d, hol)
    return d


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    """The n-th `wd` (0=Mon) of month m — e.g. 3rd Friday."""
    first = date(y, m, 1)
    return first + timedelta(days=(wd - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(y: int, m: int, wd: int) -> date:
    last = date(y, m, _cal.monthrange(y, m)[1])
    return last - timedelta(days=(last.weekday() - wd) % 7)


def _last_bday(y: int, m: int, hol) -> date:
    return _on_or_before_bday(date(y, m, _cal.monthrange(y, m)[1]), hol)


def _dom(y: int, m: int, day: int) -> date:
    return date(y, m, min(day, _cal.monthrange(y, m)[1]))


def _shift_month(y: int, m: int, delta: int) -> tuple[int, int]:
    idx = (y * 12 + (m - 1)) + delta
    return idx // 12, idx % 12 + 1


# ── rule evaluator: (kind, *args) → the last-trading/expiry date for one contract month ─
def _eval(rule, y: int, m: int, hol) -> date:
    kind = rule[0]
    if kind == "nth_wd":                       # (n-th, weekday) of the contract month
        _, n, wd = rule                         # → preceding exchange day if it's a holiday
        return _on_or_before_bday(_nth_weekday(y, m, wd, n), hol)
    if kind == "last_wd":                       # last <weekday> of the contract month
        return _on_or_before_bday(_last_weekday(y, m, rule[1]), hol)
    if kind == "bday_before_dom":               # business day before the <dom>th of the contract month
        return _prev_bday(_dom(y, m, rule[1]), hol)
    if kind == "bdays_before_dom_prior":        # n bdays before the <dom>th of the month `pm` before contract
        _, n, dm, pm = rule
        yy, mm = _shift_month(y, m, -pm)
        return _bdays_before(_dom(yy, mm, dm), n, hol)
    if kind == "nbd_before_last_bday":          # n-th business day preceding the last bday of delivery month
        return _bdays_before(_last_bday(y, m, hol), rule[1], hol)
    if kind == "bdays_before_nth_wd":           # n bdays before the k-th <weekday> of the contract month
        _, n, k, wd = rule
        return _bdays_before(_nth_weekday(y, m, wd, k), n, hol)
    if kind == "last_bday_prior":               # last business day of the month `pm` before the contract month
        yy, mm = _shift_month(y, m, -rule[1])
        return _last_bday(yy, mm, hol)
    if kind == "last_bday":                      # last business day of the contract month
        return _last_bday(y, m, hol)
    if kind == "last_wd_prior":                  # last <weekday> of the month `pm` before contract (grain options)
        _, wd, pm = rule
        yy, mm = _shift_month(y, m, -pm)
        return _on_or_before_bday(_last_weekday(yy, mm, wd), hol)
    if kind == "nth_bday":                       # n-th business day of the contract month (Lean Hogs)
        return _nth_bday(y, m, rule[1], hol)
    if kind == "bdays_before_dom_fwd":           # n bdays before the <dom>th (rolled FORWARD to a bday) — Eurex govvies
        _, n, dm = rule
        return _bdays_before(_on_or_after_bday(_dom(y, m, dm), hol), n, hol)
    if kind == "kth_wd_before_nth_wd":           # the k-th <wd> before the n-th <anchor_wd> (Fri before 3rd Wed …)
        _, wd, k, n, anchor_wd = rule
        anchor = _nth_weekday(y, m, anchor_wd, n)
        delta = (anchor.weekday() - wd) % 7 or 7
        return _on_or_before_bday(anchor - timedelta(days=delta + 7 * (k - 1)), hol)
    if kind == "minus_bdays":                    # evaluate inner rule, then step back n business days (option offset)
        _, n, inner = rule
        return _bdays_before(_eval(inner, y, m, hol), n, hol)
    raise ValueError(f"unknown expiry rule kind: {kind!r}")


# ── contract families: cycle + futures rule + options rule + indicative time ───────────
# Filled from exchange contract specs (see research). cycle = listed delivery months.
# `time` = (HH:MM local, tz short label). rule tuples evaluated by _eval above.
# NOTE: SPECS is populated below once the researched rules are confirmed.
def _q():                                       # quarterly cycle Mar/Jun/Sep/Dec
    return [3, 6, 9, 12]


def _all():                                     # every month (serial)
    return list(range(1, 13))


# `time` = (HH:MM in local exchange time, short label) — the time trading TERMINATES on the
# last trading day. Options modelled at the standard MONTHLY expiry (weeklies/dailies ignored
# — the useful number is the next monthly expiry). More families appended as researched.
SPECS: dict[str, dict] = {
    # ── Equity indices — all third-Friday-family, cash-settled off an SQ/auction ──────────
    "cme_eq": {"cycle": _q(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("09:30", "ET")},      # ES/NQ/RTY/YM SOQ, a.m.
    "estoxx": {"cycle": _q(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("12:00", "CET")},
    "dax":    {"cycle": _q(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("13:00", "CET")},     # Xetra intraday auction
    "smi":    {"cycle": _q(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("09:00", "CET")},     # SIX opening auction
    "cac":    {"cycle": _all(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("16:00", "CET")},     # serial monthly future
    "ftse":   {"cycle": _q(), "fut": ("nth_wd", 3, FRI), "opt": ("nth_wd", 3, FRI),
               "opt_cycle": _all(), "time": ("10:15", "London")},  # EDSP auction, a.m.
    "nikkei": {"cycle": _q(), "fut": ("bdays_before_nth_wd", 1, 2, FRI),   # day before SQ (2nd Fri)
               "opt": ("bdays_before_nth_wd", 1, 2, FRI), "opt_cycle": _all(),
               "time": ("15:15", "JST")},
    "kospi":  {"cycle": _q(), "fut": ("nth_wd", 2, THU), "opt": ("nth_wd", 2, THU),
               "opt_cycle": _all(), "time": ("15:20", "KST")},     # close-based settle, 2nd Thu
    "asx":    {"cycle": _all(), "fut": ("nth_wd", 3, THU), "opt": ("nth_wd", 3, THU),
               "opt_cycle": _all(), "time": ("12:00", "AEST")},    # serial months listed; SOQ noon
    # ── Softs (ICE) — verified vs ICE /specs (unusual business-day counts) ────────────────
    "cotton":  {"cycle": [3, 5, 7, 10, 12], "fut": ("nbd_before_last_bday", 17),
                "opt": ("nth_wd", 3, FRI), "opt_cycle": _all(), "time": ("14:20", "ET")},
    "oj":      {"cycle": [1, 3, 5, 7, 9, 11], "fut": ("nbd_before_last_bday", 14),
                "opt": ("nth_wd", 3, FRI), "opt_cycle": _all(), "time": ("14:00", "ET")},
    "robusta": {"cycle": [1, 3, 5, 7, 9, 11], "fut": ("nbd_before_last_bday", 4),
                "opt": ("nth_wd", 3, WED), "opt_cycle": _all(), "time": ("12:30", "London")},
    "coffee":  {"cycle": [3, 5, 7, 9, 12], "fut": ("nbd_before_last_bday", 8),   # LTD = day before 7th-bd notice day
                "opt": ("nth_wd", 2, FRI), "opt_cycle": _all(), "time": ("13:30", "ET")},
    "sugar":   {"cycle": [3, 5, 7, 10], "fut": ("last_bday_prior", 1),            # last bday of month before delivery
                "opt": ("nth_wd", 3, FRI), "opt_cycle": _all(), "time": ("13:00", "ET")},
    "cocoa":   {"cycle": [3, 5, 7, 9, 12], "fut": ("nbd_before_last_bday", 11),  # LTD = day before 10th-bd notice day
                "opt": ("nth_wd", 2, FRI), "opt_cycle": _all(), "time": ("13:30", "ET")},
    # ── Grains / oilseeds (CBOT) — futures: business day before the 15th; expiring contract closes noon CT ──
    "soymeal": {"cycle": [1, 3, 5, 7, 8, 9, 10, 12], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "soyoil":  {"cycle": [1, 3, 5, 7, 8, 9, 10, 12], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "rice":    {"cycle": [1, 3, 5, 7, 9, 11], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "corn":    {"cycle": [3, 5, 7, 9, 12], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "soybean": {"cycle": [1, 3, 5, 7, 8, 9, 11], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "wheat":   {"cycle": [3, 5, 7, 9, 12], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "kcwheat": {"cycle": [3, 5, 7, 9, 12], "fut": ("bday_before_dom", 15),
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:00", "CT")},
    # ── STIRs ────────────────────────────────────────────────────────────────────────────
    "sofr1m":  {"cycle": _all(), "fut": ("last_bday",), "opt": ("last_bday",),
                "opt_cycle": _all(), "time": ("14:00", "CT")},
    "sofr3m":  {"cycle": _q(), "fut": ("bdays_before_nth_wd", 1, 3, WED),   # bd before 3rd Wed
                "opt": ("kth_wd_before_nth_wd", FRI, 1, 3, WED), "opt_cycle": _all(), "time": ("14:00", "CT")},
    "fedfunds": {"cycle": _all(), "fut": ("last_bday",), "opt": ("last_bday",),
                 "opt_cycle": _all(), "time": ("14:00", "CT")},
    "euribor": {"cycle": _all(), "fut": ("bdays_before_nth_wd", 2, 3, WED),  # 2 bd before 3rd Wed (forward-looking)
                "opt": ("kth_wd_before_nth_wd", FRI, 1, 3, WED), "opt_cycle": _all(), "time": ("10:00", "London")},
    "estr":    {"cycle": _q(), "fut": ("bdays_before_nth_wd", 1, 3, WED),   # backward-looking, futures-only
                "time": ("18:00", "Brussels")},
    "sonia":   {"cycle": _q(), "fut": ("bdays_before_nth_wd", 1, 3, WED),   # backward-looking
                "opt": ("kth_wd_before_nth_wd", FRI, 1, 3, WED), "opt_cycle": _all(), "time": ("18:00", "London")},
    # ── Rates — US Treasuries (CBOT), Euro govvies (Eurex), Long Gilt (ICE) ───────────────
    "ust_short": {"cycle": _q(), "fut": ("last_bday",),                     # TU / FV: last bday of month
                  "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:01", "CT")},
    "ust_long": {"cycle": _q(), "fut": ("nbd_before_last_bday", 7),         # TY/UXY/US/WN: 7th bd before month-end
                 "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:01", "CT")},
    "eur_govt": {"cycle": _q(), "fut": ("bdays_before_dom_fwd", 2, 10),     # 2 exch days before the 10th (delivery day)
                 "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("12:30", "CET")},
    "gilt":    {"cycle": _q(), "fut": ("nbd_before_last_bday", 2),          # 2 bd before last bday of month
                "opt": ("last_wd_prior", FRI, 1), "opt_cycle": _all(), "time": ("11:00", "London")},
    # ── Energy — each with its own anchor; option offsets are NOT uniform ─────────────────
    "wti":     {"cycle": _all(), "fut": ("bdays_before_dom_prior", 3, 25, 1),
                "opt": ("minus_bdays", 3, ("bdays_before_dom_prior", 3, 25, 1)), "opt_cycle": _all(),
                "time": ("14:30", "ET")},
    "brent":   {"cycle": _all(), "fut": ("last_bday_prior", 2),
                "opt": ("minus_bdays", 3, ("last_bday_prior", 2)), "opt_cycle": _all(), "time": ("19:30", "London")},
    "gasoil":  {"cycle": _all(), "fut": ("bdays_before_dom_prior", 2, 14, 0),
                "opt": ("minus_bdays", 2, ("bdays_before_dom_prior", 2, 14, 0)), "opt_cycle": _all(),
                "time": ("12:00", "London")},
    "rbob":    {"cycle": _all(), "fut": ("last_bday_prior", 1),
                "opt": ("minus_bdays", 3, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("14:30", "ET")},
    "heatoil": {"cycle": _all(), "fut": ("last_bday_prior", 1),
                "opt": ("minus_bdays", 3, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("14:30", "ET")},
    "natgas":  {"cycle": _all(), "fut": ("bdays_before_dom_prior", 3, 1, 0),
                "opt": ("minus_bdays", 1, ("bdays_before_dom_prior", 3, 1, 0)), "opt_cycle": _all(),
                "time": ("14:30", "ET")},
    "ttf":     {"cycle": _all(), "fut": ("bdays_before_dom_prior", 2, 1, 0),
                "opt": ("minus_bdays", 3, ("bdays_before_dom_prior", 2, 1, 0)), "opt_cycle": _all(),
                "time": ("18:00", "CET")},
    "eua":     {"cycle": _all(), "fut": ("last_wd", MON),                   # last Monday of the contract month
                "opt": ("minus_bdays", 3, ("last_wd", MON)), "opt_cycle": _all(), "time": ("17:00", "London")},
    # ── Metals — all "third-last business day of the contract month"; times differ ────────
    "gold":    {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),
                "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("12:30", "CT")},
    "silver":  {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),
                "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("12:25", "CT")},
    "copper":  {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),
                "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "platinum": {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),
                 "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("13:05", "ET")},
    "palladium": {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),
                  "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("13:05", "ET")},
    "aluminium": {"cycle": _all(), "fut": ("nbd_before_last_bday", 2),      # COMEX metal rule (not separately researched)
                  "opt": ("minus_bdays", 4, ("last_bday_prior", 1)), "opt_cycle": _all(), "time": ("13:00", "ET")},
    "ironore": {"cycle": _all(), "fut": ("last_bday",), "opt": ("last_bday",),   # SGX FEF, cash-settled
                "opt_cycle": _all(), "time": ("18:30", "SGT")},
    # ── Livestock (CME) ──────────────────────────────────────────────────────────────────
    "livecattle": {"cycle": [2, 4, 6, 8, 10, 12], "fut": ("last_bday",),
                   "opt": ("nth_wd", 1, FRI), "opt_cycle": _all(), "time": ("12:00", "CT")},
    "feedercattle": {"cycle": [1, 3, 4, 5, 8, 9, 10, 11], "fut": ("last_wd", THU),  # last Thursday
                     "opt": ("last_wd", THU), "opt_cycle": [1, 3, 4, 5, 8, 9, 10, 11], "time": ("13:05", "CT")},
    "leanhogs": {"cycle": [2, 4, 5, 6, 7, 8, 10, 12], "fut": ("nth_bday", 10),   # 10th business day of the month
                 "opt": ("nth_bday", 10), "opt_cycle": [2, 4, 5, 6, 7, 8, 10, 12], "time": ("12:00", "CT")},
    # ── FX futures (CME IMM) — 2 bd before 3rd Wed (CAD 1 bd; CHF quarterly-only) ─────────
    "fx":      {"cycle": _all(), "fut": ("bdays_before_nth_wd", 2, 3, WED),
                "opt": ("kth_wd_before_nth_wd", FRI, 2, 3, WED), "opt_cycle": _all(), "time": ("09:16", "CT")},
    "fx_cad":  {"cycle": _all(), "fut": ("bdays_before_nth_wd", 1, 3, WED),
                "opt": ("kth_wd_before_nth_wd", FRI, 2, 3, WED), "opt_cycle": _all(), "time": ("09:16", "CT")},
    "fx_chf":  {"cycle": _q(), "fut": ("bdays_before_nth_wd", 2, 3, WED),
                "opt": ("kth_wd_before_nth_wd", FRI, 2, 3, WED), "opt_cycle": _all(), "time": ("09:16", "CT")},
}

# ticker -> family key. Only mapped tickers get an expiry; anything unmapped shows nothing
# (safer than a wrong asset-class guess). Non-index families appended as researched.
FAMILY_OF: dict[str, str] = {
    "ESA Index": "cme_eq", "NQA Index": "cme_eq", "RTYA Index": "cme_eq", "DMA Index": "cme_eq",
    "VGA Index": "estoxx", "GXA Index": "dax", "SMA Index": "smi", "CAA Index": "estoxx",
    "CFA Index": "cac",
    "Z A Index": "ftse", "NKA Index": "nikkei", "KMA Index": "kospi", "XPA Index": "asx",
    # cash indices — no future of their own; show the monthly index-option expiry only
    "SX5E Index": "estoxx", "SX7E Index": "estoxx", "DAX Index": "dax", "UKX Index": "ftse",
    "CAC Index": "cac", "SMI Index": "smi", "NKY Index": "nikkei", "KOSPI2 Index": "kospi",
    # Softs
    "CTA Comdty": "cotton", "JOA Comdty": "oj", "DFA Comdty": "robusta",
    "KCA Comdty": "coffee", "SBA Comdty": "sugar", "CCA Comdty": "cocoa",
    # Grains / oilseeds
    "SMA Comdty": "soymeal", "BOA Comdty": "soyoil", "RRA Comdty": "rice",
    "C A Comdty": "corn", "S A Comdty": "soybean", "W A Comdty": "wheat", "KWA Comdty": "kcwheat",
    # STIRs
    "SERA Comdty": "sofr1m", "SFRA Comdty": "sofr3m", "FFA Comdty": "fedfunds",
    "ERA Comdty": "euribor", "TKYA Comdty": "estr", "SFIA Comdty": "sonia",
    # US Treasuries
    "TUA Comdty": "ust_short", "FVA Comdty": "ust_short", "TYA Comdty": "ust_long",
    "UXYA Comdty": "ust_long", "USA Comdty": "ust_long", "WNA Comdty": "ust_long",
    # Euro govvies + gilt
    "DUA Comdty": "eur_govt", "OEA Comdty": "eur_govt", "RXA Comdty": "eur_govt",
    "UBA Comdty": "eur_govt", "OATA Comdty": "eur_govt", "G A Comdty": "gilt",
    # Energy (Ethanol CUAA intentionally unmapped — not researched)
    "CLA Comdty": "wti", "COA Comdty": "brent", "QSA Comdty": "gasoil", "XBA Comdty": "rbob",
    "HOA Comdty": "heatoil", "NGA Comdty": "natgas", "FJSA Comdty": "ttf", "MOA Comdty": "eua",
    # Metals
    "GCA Comdty": "gold", "SIA Comdty": "silver", "HGA Comdty": "copper", "PLA Comdty": "platinum",
    "PAA Comdty": "palladium", "ALEA Comdty": "aluminium", "SCOA Comdty": "ironore",
    # Livestock
    "LCA Comdty": "livecattle", "FCA Comdty": "feedercattle", "LHA Comdty": "leanhogs",
    # FX (CME IMM) — CAD 1-bd rule, CHF quarterly-only; EM crosses use the standard IMM rule
    "ECA Curncy": "fx", "BPA Curncy": "fx", "JYA Curncy": "fx", "ADA Curncy": "fx",
    "BRA Curncy": "fx", "NVA Curncy": "fx", "PEA Curncy": "fx", "RAA Curncy": "fx",
    "SIRA Curncy": "fx", "SEA Curncy": "fx", "HEA Curncy": "fx", "KOA Curncy": "fx",
    "NOA Curncy": "fx", "CCA Curncy": "fx", "ISA Curncy": "fx", "PPA Curncy": "fx",
    "CDA Curncy": "fx_cad", "SFA Curncy": "fx_chf",
}
FAMILY_BY_ASSET: dict[str, str] = {}

# Cash indices (index itself doesn't expire) — futures line suppressed, options still shown.
CASH_TICKERS: set[str] = {"SX5E Index", "SX7E Index", "DAX Index", "UKX Index",
                          "CAC Index", "SMI Index", "NKY Index", "KOSPI2 Index"}


def _holidays_for(ticker: str, asset: str) -> set[str]:
    cal = markethours.CALENDAR_OF.get(markethours.profile_id(ticker, asset))
    return set(markethours.CLOSURES.get(cal, {})) if cal else set()


def _family(ticker: str, asset: str) -> str | None:
    return FAMILY_OF.get(ticker) or FAMILY_BY_ASSET.get(asset)


def _next(rule, cycle, ref: date, hol) -> date | None:
    """Earliest expiry date >= ref across listed contract months (handles rules whose
    last-trade date falls in a month before the contract month)."""
    best = None
    for delta in range(-2, 25):                 # a couple months back (roll) → ~2yrs forward
        y, m = _shift_month(ref.year, ref.month, delta)
        if m not in cycle:
            continue
        d = _eval(rule, y, m, hol)
        if d >= ref and (best is None or d < best):
            best = d
    return best


def _fmt_date(d: date) -> str:
    return f"{d:%a %d %b %Y}"


def describe(ticker: str, asset: str, ref_date: date) -> dict:
    """Next futures & options expiry for a product. Returns dict with formatted strings
    (values are None where we don't model an expiry, e.g. cash indices' futures)."""
    fam = _family(ticker, asset)
    spec = SPECS.get(fam) if fam else None
    out = {"fut": None, "fut_time": None, "opt": None, "opt_time": None,
           "cash": ticker in CASH_TICKERS}
    if not spec:
        return out
    hol = _holidays_for(ticker, asset)
    tlabel = f"{spec['time'][0]} {spec['time'][1]}" if spec.get("time") else None
    if not out["cash"] and spec.get("fut"):
        d = _next(spec["fut"], spec["cycle"], ref_date, hol)
        if d:
            out["fut"], out["fut_time"] = _fmt_date(d), tlabel
    if spec.get("opt"):
        oc = spec.get("opt_cycle", spec["cycle"])
        d = _next(spec["opt"], oc, ref_date, hol)
        if d:
            out["opt"], out["opt_time"] = _fmt_date(d), spec.get("opt_time_label", tlabel)
    return out
