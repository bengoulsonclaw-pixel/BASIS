"""Bloomberg Codes database — symbol ⇄ product/expiry.

The load-bearing test here is `test_rules_reproduce_observed_expiries`. The rule engine in
src/expiries.py is a RECONSTRUCTION of each exchange's expiry calendar, and the weekly
open-interest capture holds the exchange's OWN dates (`OPT_EXPIRE_DT`) for eleven fixed-income
products. Cross-checking the two on 2026-08-28 found 34 mismatches across 82 expiries — two
real bugs, since fixed:

  • US Treasury / Eurex govvie options ignored the CBOT-Eurex rule that the last Friday must
    precede the prior month's last business day by at least two business days; Jul, Aug and
    Nov 2026 and Feb 2027 were all a week late.
  • ICE Euribor QUARTERLY options expire with their future (the Monday), not the Friday
    before; every quarterly was three days early. SOFR is genuinely the Friday, so its rule
    was correct and stayed put.

That check is cheap and it is the only independent evidence we have that the rules are right,
so it runs on every push. If it fails after an expiries.py edit, the edit is wrong until
proven otherwise — the parquet is the exchange's answer, not ours.
"""
from __future__ import annotations

from datetime import date

import pytest

from src import bbgcodes, expiries, universe as u


# ── the correctness anchor ────────────────────────────────────────────────────────────
def test_rules_reproduce_observed_expiries():
    """Every option expiry ever seen on the Terminal must fall out of the rule engine."""
    obs = bbgcodes.observed_expiries()
    if not obs:
        pytest.skip("no observed option-expiry capture on this box")

    misses = []
    for tk, dates in obs.items():
        if tk not in u.INSTRUMENTS:
            continue
        asset = u.INSTRUMENTS[tk][2]
        for d in sorted(dates):
            best = None
            for y in (d.year - 1, d.year, d.year + 1):
                for m in expiries.listed_months(tk, asset, "opt"):
                    r = expiries.expiry_for(tk, asset, y, m, "opt")
                    if r and (best is None or abs((r - d).days) < abs((best - d).days)):
                        best = r
            if best != d:
                misses.append(f"{tk} {d} -> rules say {best}")
    assert not misses, "rule engine disagrees with observed OPT_EXPIRE_DT:\n" + \
                       "\n".join(misses)


@pytest.mark.parametrize("ticker,asset,year,month,expected", [
    # The ≥2-business-day govvie rule: the last Friday of the prior month is TOO LATE
    # whenever it sits within two business days of that month's last business day.
    ("TYA Comdty", "Bonds", 2026, 8, date(2026, 7, 24)),    # last Fri 31 Jul IS month-end
    ("TYA Comdty", "Bonds", 2026, 9, date(2026, 8, 21)),    # 28 Aug is 1 bd before 31 Aug
    ("TYA Comdty", "Bonds", 2026, 12, date(2026, 11, 20)),  # 27 Nov is 1 bd before 30 Nov
    ("TYA Comdty", "Bonds", 2026, 10, date(2026, 9, 25)),   # 25 Sep clears — rule unchanged
    # Euribor splits its cycle; SOFR does not.
    ("ERA Comdty", "STIRs", 2026, 12, date(2026, 12, 14)),  # quarterly -> with the future
    ("ERA Comdty", "STIRs", 2026, 11, date(2026, 11, 13)),  # serial -> Friday before
    ("SFRA Comdty", "STIRs", 2026, 12, date(2026, 12, 11)),  # SOFR quarterly IS the Friday
])
def test_specific_expiry_rules(ticker, asset, year, month, expected):
    assert expiries.expiry_for(ticker, asset, year, month, "opt") == expected


# ── decoding ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("symbol,name,contract,kind", [
    ("CLZ6C 80 Comdty", "WTI Crude", "Dec 2026", "option"),
    ("TYX6C 112.5 Comdty", "US 10Y Note", "Nov 2026", "option"),
    ("ESZ6 Index", "S&P 500 E-mini", "Dec 2026", "future"),
    ("NGU26 Comdty", "Henry Hub Nat Gas", "Sep 2026", "future"),
    ("C Z6 Comdty", "Corn", "Dec 2026", "future"),          # root keeps its trailing space
])
def test_decode(symbol, name, contract, kind):
    d = bbgcodes.decode(symbol)
    assert d["ok"], d.get("reason")
    assert d["product"]["name"] == name
    assert d["contract"] == contract
    assert d["kind"] == kind


def test_decode_rejects_nonsense():
    assert not bbgcodes.decode("ZZZQ9 Comdty")["ok"]
    assert not bbgcodes.decode("")["ok"]


def test_generic_is_flagged_not_dated():
    d = bbgcodes.decode("CLA Comdty")
    assert d["ok"] and d["kind"] == "generic"
    assert d.get("contract") is None
    assert any("generic" in w.lower() for w in d["warnings"])


# ── the look-alike trap ───────────────────────────────────────────────────────────────
def test_yellow_key_separates_look_alike_roots():
    """`SM` and `CC` name two products each. Naming the wrong one is the worst failure
    this tool has, so the key must decide it — and a missing key must warn, not guess."""
    assert bbgcodes.decode("SMZ6 Comdty")["product"]["name"] == "Soybean Meal"
    assert bbgcodes.decode("SMZ6 Index")["product"]["name"] == "SMI (Swiss)"
    assert bbgcodes.decode("CCZ6 Comdty")["product"]["name"] == "Cocoa"
    assert bbgcodes.decode("CCZ6 Curncy")["product"]["name"] == "Czech Koruna (CZK)"

    bare = bbgcodes.decode("SMZ6")
    assert any("AMBIGUOUS" in w for w in bare["warnings"])


# ── serial options ────────────────────────────────────────────────────────────────────
def test_serial_option_names_its_underlying_future():
    """A November Treasury option trades on the DECEMBER future — hedging the November
    contract that doesn't exist is a real trade error, so the link is asserted."""
    d = bbgcodes.decode("TYX6C 112.5 Comdty")
    assert d["serial"] is True
    assert d["underlying"]["code"].startswith("TYZ6")
    assert d["fut_expiry"] is None          # November is not a listed TY futures month


def test_quarterly_option_is_not_flagged_serial():
    d = bbgcodes.decode("TYZ6C 112.5 Comdty")
    assert d["serial"] is False
    assert d["underlying"]["code"].startswith("TYZ6")


# ── building ──────────────────────────────────────────────────────────────────────────
def test_build_finds_exact_expiry():
    """Corn's Sep-26 option expires the last Friday of August."""
    r = bbgcodes.build("C A Comdty", date(2026, 8, 28), "opt")
    assert r["exact"] is not None
    assert r["exact"]["contract"] == "Sep 2026"
    assert r["exact"]["code"].startswith("C U6C")


def test_build_falls_back_to_the_contract_live_on_that_date():
    """A date that is nobody's expiry still has an answer: what is still trading then."""
    r = bbgcodes.build("CLA Comdty", date(2026, 12, 1), "opt")
    assert r["exact"] is None
    assert r["live_on"]["expiry"] >= date(2026, 12, 1)
    assert r["nearest"]
    assert any("Nothing expires exactly" in w or "No option expiry" in w
               for w in r["warnings"])


def test_build_always_declares_the_weekly_gap():
    """Weeklies are absent by design; the page must never let that pass unsaid."""
    r = bbgcodes.build("ESA Index", date(2026, 12, 18), "opt")
    assert any("Weekly and daily" in w for w in r["warnings"])


# ── strike formats ────────────────────────────────────────────────────────────────────
def test_strike_hint_returns_real_bloomberg_strings():
    """Strings must come from the observed tickers, not from our own float formatting —
    a code built with '220.0' where the Terminal lists '220' may simply not resolve."""
    h = bbgcodes.strike_hint("C ")
    if not h["examples"]:
        pytest.skip("no observed chain for corn on this box")
    assert all("." not in s for s in h["examples"]), h["examples"]
    assert h["step"] == 1.0


def test_option_code_round_trips_through_decode():
    code = bbgcodes.option_code("CL", 12, 2026, "80", "C")
    assert code == "CLZ6C 80 Comdty"
    d = bbgcodes.decode(code)
    assert d["ok"] and d["product"]["name"] == "WTI Crude" and d["strike"] == "80"


# ── the universe ──────────────────────────────────────────────────────────────────────
def test_every_product_has_a_root_and_no_root_is_lost():
    prods = bbgcodes.products()
    assert len(prods) >= 75
    assert all(p["root"] for p in prods)
    # both halves of each colliding root survive
    names = {p["name"] for p in prods}
    assert {"Soybean Meal", "SMI (Swiss)", "Cocoa"} <= names


def test_search_finds_by_name_and_root():
    assert any(p["name"] == "WTI Crude" for p in bbgcodes.search("wti"))
    assert any(p["name"] == "Nasdaq 100 E-mini" for p in bbgcodes.search("nasdaq"))
    assert any(p["name"] == "Corn" for p in bbgcodes.search("corn"))
    assert bbgcodes.search("") == []
