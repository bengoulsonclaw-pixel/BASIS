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


def test_build_states_the_weekly_position_either_way():
    """Whatever the state, the page says it. For a product whose weekly scheme is KNOWN
    (S&P, learned from the Terminal 2026-09-01) the caveat is the listed horizon; for one
    still unlearned it is that the roots haven't been read off the Terminal yet. Silence
    would read as 'no weeklies exist', which is never what we mean."""
    es = bbgcodes.build("ESA Index", date(2026, 12, 18), "opt")
    assert any("listed horizon" in w or "decade-old" in w.lower() for w in es["warnings"])
    assert not any("not in this database yet" in w for w in es["warnings"])

    cl = bbgcodes.build("CLA Comdty", date(2026, 12, 18), "opt")
    assert any("not in this database yet" in w for w in cl["warnings"])


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


# ── weeklies ──────────────────────────────────────────────────────────────────────────
# These dates are not our arithmetic checking itself — they are what the TERMINAL returned
# during probe_weekly_pattern.py on 2026-09-01, transcribed. If the generator ever stops
# reproducing them it has drifted from Bloomberg, and the probe output is the authority.
@pytest.mark.parametrize("year,month,week,expected", [
    (2026, 9, 1, date(2026, 9, 4)),    # NAME "S&P Emini 1st Wee Sep26"
    (2026, 9, 2, date(2026, 9, 11)),   # NAME "S&P Emini 2nd Wee Sep26"
    (2026, 9, 3, date(2026, 9, 18)),   # NAME "S&P Emini 3rd Wk  Sep26"
    (2026, 9, 4, date(2026, 9, 25)),   # NAME "S&P Emini 4th Wee Sep26"
    (2016, 12, 1, date(2016, 12, 2)),  # the one-digit-year cross-check: Z6 -> 2016
    (2016, 12, 2, date(2016, 12, 9)),
    (2016, 12, 4, date(2016, 12, 23)),
    (2017, 1, 1, date(2017, 1, 6)),
    (2017, 1, 2, date(2017, 1, 13)),
    (2017, 1, 4, date(2017, 1, 27)),
])
def test_weekly_expiries_match_the_terminal(year, month, week, expected):
    got = [w for w in bbgcodes.weekly_series("ESA Index", year, month)
           if w["week"] == week]
    assert got, f"week {week} of {month}/{year} not generated"
    assert got[0]["expiry"] == expected


def test_week_five_absent_when_the_month_has_four_fridays():
    """September 2026 has Fridays on the 4th, 11th, 18th and 25th — no fifth. The probe
    found 5EU6C did not resolve, and the generator must agree rather than inventing it."""
    weeks = {w["week"] for w in bbgcodes.weekly_series("ESA Index", 2026, 9)}
    assert weeks == {1, 2, 3, 4}
    # December 2016 does have five Fridays
    assert 5 in {w["week"] for w in bbgcodes.weekly_series("ESA Index", 2016, 12)}


def test_decode_weekly_symbol():
    d = bbgcodes.decode("2EU6C 7660 Index")
    assert d["ok"] and d["weekly"] is True and d["week"] == 2
    assert d["product"]["name"] == "S&P 500 E-mini"
    assert d["opt_expiry"] == date(2026, 9, 11)
    assert d["kind"] == "option" and d["put_call"] == "Call"


def test_weekly_root_does_not_collide_with_a_futures_root():
    """'2E' must resolve as an S&P weekly, and ordinary roots must be untouched by it."""
    assert bbgcodes.decode("2EU6C 7660 Index")["weekly"] is True
    assert bbgcodes.decode("ESZ6C 7000 Index").get("weekly") is not True
    assert bbgcodes.decode("CLZ6C 80 Comdty").get("weekly") is not True


def test_weekly_beyond_the_listed_horizon_is_flagged():
    """Bloomberg lists only a few weeks out. Past that, a one-digit year can resolve to a
    decade-old contract (Z6 returned Dec-2016 in the probe), so the page must not present
    a far-dated weekly as a tradeable code."""
    d = bbgcodes.decode("4EZ6C 2540 Index")
    assert d["ok"] and d["weekly"] is True
    assert any("BEYOND the listed horizon" in w for w in d["warnings"])


def test_impossible_week_is_flagged_not_invented():
    d = bbgcodes.decode("5EU6C 7660 Index")
    assert d["opt_expiry"] is None
    assert any("no 5th Friday" in w for w in d["warnings"])


def test_build_offers_a_weekly_on_an_exact_date():
    r = bbgcodes.build("ESA Index", date(2026, 9, 11), "opt")
    assert r["exact"] is not None
    assert r["exact"].get("weekly") is True
    assert r["exact"]["code"].startswith("2EU6C")


def test_products_without_a_learned_weekly_scheme_say_so():
    """Silence would read as 'no weeklies exist'. It must read as 'not learned yet'."""
    r = bbgcodes.build("CLA Comdty", date(2026, 12, 1), "opt")
    assert any("not in this database yet" in w for w in r["warnings"])


# ── contract with the page ────────────────────────────────────────────────────────────
# render_bbg_codes() reads these keys directly. A missing one is a KeyError in front of
# Ben mid-conversation with a client, and the Streamlit widgets cannot be driven from the
# browser harness on this box — so the page/engine contract is asserted here instead.
DECODE_KEYS = {"ok", "kind", "product", "root", "contract", "month", "month_code", "year",
               "opt_expiry", "fut_expiry", "expiry", "expiry_kind", "source", "warnings",
               "yellow_key", "underlying", "serial", "weekly"}
WEEKLY_EXTRA = {"week", "week_label"}


@pytest.mark.parametrize("symbol", [
    "2EU6C 7660 Index",      # weekly
    "CLZ6C 80 Comdty",       # monthly option
    "TYX6C 112.5 Comdty",    # serial option
    "ESZ6 Index",            # outright future
])
def test_decode_result_has_every_key_the_page_reads(symbol):
    d = bbgcodes.decode(symbol)
    missing = {k for k in DECODE_KEYS if k not in d}
    assert not missing, f"{symbol} decode is missing {sorted(missing)}"
    if d.get("weekly"):
        assert not (WEEKLY_EXTRA - set(d)), f"{symbol} missing {WEEKLY_EXTRA - set(d)}"


@pytest.mark.parametrize("ticker,target", [
    ("ESA Index", date(2026, 9, 11)),     # lands on a weekly
    ("ESA Index", date(2026, 9, 15)),     # lands on nothing
    ("CLA Comdty", date(2026, 12, 1)),    # product with no weekly scheme
])
def test_build_rows_have_every_key_the_page_reads(ticker, target):
    r = bbgcodes.build(ticker, target, "opt")
    for row in r["nearest"] + [x for x in (r["exact"], r["live_on"]) if x]:
        for k in ("code", "contract", "expiry", "days", "source"):
            assert k in row, f"{ticker} row missing {k}: {row}"
        if row.get("weekly"):
            assert "label" in row and "listed" in row
