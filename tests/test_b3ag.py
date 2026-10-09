"""The B3 ag store parses B3's daily Price Report and builds Brazil-vs-US basis series.

Each test pins a failure observed while building it (2026-10-09), not a hypothetical:

  * the report holds TWO sessions per ticker — the day's record and the next morning's
    opening record carrying the same settle — and keying on ticker alone reported the
    next morning's 8 lots instead of the day's 480;
  * options and spreads share the root prefix and must not enter the futures store, and
    roots outside PRODUCTS (SJC soy mirrors CME, so it carries no basis) stay out;
  * the 22-byte stub B3 serves for holidays AND not-yet-published days must freeze a
    past date as "no session" but leave today open for retry;
  * Bloomberg writes contracts as 'KCZ16', 'KCZ6' and space-padded 'C Z6'; a wrong
    decade on the one-digit form would match the wrong B3 month and fake the basis;
  * the unit conversions ARE the comparison — a slip there is a fake spread.

No network: every test builds its own XML bytes.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src import b3ag as b


def _rec(ticker: str, dt: str, settle: float, qty: int, last: float) -> bytes:
    return f"""
        <PricRpt>
          <TradDt><Dt>{dt}</Dt></TradDt>
          <SctyId><TckrSymb>{ticker}</TckrSymb></SctyId>
          <FinInstrmAttrbts>
            <OpnIntrst>3097</OpnIntrst>
            <FinInstrmQty>{qty}</FinInstrmQty>
            <LastPric Ccy="USD">{last}</LastPric>
            <AdjstdQt Ccy="USD">{settle}</AdjstdQt>
            <PrvsAdjstdQt Ccy="USD">361.1</PrvsAdjstdQt>
          </FinInstrmAttrbts>
        </PricRpt>""".encode()


def test_only_the_reports_own_session_is_kept():
    xml = (_rec("ICFZ26", "2026-10-08", 355.8, 480, 354.1)
           + _rec("ICFZ26", "2026-10-09", 355.8, 8, 354.0))   # next morning, later in file
    rows = b.parse_report(xml, date(2026, 10, 8))
    assert len(rows) == 1
    assert rows[0]["volume"] == 480 and rows[0]["last"] == 354.1


def test_only_listed_roots_and_plain_futures_are_kept():
    xml = (_rec("ICFZ26", "2026-10-08", 355.8, 480, 354.1)
           + _rec("CCMX26", "2026-10-08", 68.5, 900, 68.4)
           + _rec("ICFZ26C0450", "2026-10-08", 12.0, 3, 12.0)   # option
           + _rec("ICFZ26H27", "2026-10-08", 3.2, 1, 3.2)       # spread
           + _rec("SJCX26", "2026-10-08", 23.1, 50, 23.1))      # CME mirror, not a basis
    rows = b.parse_report(xml, date(2026, 10, 8))
    assert sorted((r["root"], r["ticker"]) for r in rows) == [("CCM", "CCMX26"),
                                                              ("ICF", "ICFZ26")]


def test_stub_is_holiday_only_once_safely_past(monkeypatch):
    monkeypatch.setattr(b, "_download", lambda d, timeout=90: b"\x00" * 22)
    assert b.fetch_day(date.today() - timedelta(days=10)) == []
    assert b.fetch_day(date.today()) is None


@pytest.mark.parametrize("code,root,asof,expect", [
    ("KCZ16", "KC", "2016-10-03", (2016, 12)),
    ("KCZ6", "KC", "2026-10-08", (2026, 12)),
    ("KCH7", "KC", "2026-12-20", (2027, 3)),     # December, already rolled into next year
    ("KCH0", "KC", "2019-12-20", (2020, 3)),     # decade boundary
    ("C Z6", "C", "2026-10-08", (2026, 12)),     # space-padded one-letter root
    ("LCV6", "C", "2026-10-08", None),           # wrong root never matches
])
def test_us_contract_year_resolution(code, root, asof, expect):
    assert b._us_contract_ym(code, root, pd.Timestamp(asof)) == expect


def test_unit_conversions():
    # Coffee: 355.80 USD/bag on 2026-10-08 = 268.98 c/lb
    assert b.PRODUCTS["ICF"]["to_us"](355.80, None) == pytest.approx(268.98, abs=0.01)
    # Corn: 60 kg sack = 2.362096 bu; BRL 70/sack at PTAX 5.00 = 592.7 c/bu
    assert b.BU_PER_SACK == pytest.approx(2.362096, abs=1e-5)
    assert b.PRODUCTS["CCM"]["to_us"](70.0, 5.0) == pytest.approx(592.69, abs=0.01)
    # Cattle: BRL 300/@ (15 kg carcass) at PTAX 5.00 = 181.44 USD/cwt carcass
    assert b.PRODUCTS["BGI"]["to_us"](300.0, 5.0) == pytest.approx(181.44, abs=0.01)
    # A BRL leg with no PTAX that day is NaN, never a silently stale rate
    assert pd.isna(b.PRODUCTS["CCM"]["to_us"](70.0, float("nan")))


def test_cattle_is_ratio_only():
    assert b.PRODUCTS["BGI"].get("ratio_only") is True


def test_ethanol_never_differenced_against_sugar():
    # USD/m3 minus c/lb printed a "+2,850%" basis on the first run.
    assert b.PRODUCTS["ETH"].get("parity_pending") is True


# ── brbasis: the seasonal percentile ─────────────────────────────────────────
def test_seasonal_pctl_ranks_against_same_weeks_of_prior_years_only():
    from src import brbasis
    # Basis swings with the season: +100 every Sep–Nov, -100 every April, five years.
    idx = pd.bdate_range("2020-01-01", "2025-10-10")
    vals = [100.0 if d.month in (9, 10, 11) else (-100.0 if d.month == 4 else 0.0)
            for d in idx]
    s = pd.Series(vals, index=idx)
    s.iloc[-1] = 50.0      # high on the full history, LOW for an October
    p, yrs = brbasis.seasonal_pctl(s)
    assert yrs >= 3
    assert p < 10
    assert (s <= 50.0).mean() * 100 > 70     # the raw percentile would call it rich


def test_seasonal_pctl_needs_enough_seasons():
    from src import brbasis
    idx = pd.bdate_range("2024-06-01", "2025-10-10")
    p, yrs = brbasis.seasonal_pctl(pd.Series(1.0, index=idx))
    assert pd.isna(p) and yrs < brbasis.SEAS_MIN_YEARS


# ── brbasis: sugar–ethanol parity ────────────────────────────────────────────
def test_ethanol_parity_conversion():
    from src import brbasis
    # R$2.5639/L mill-net (CEPEA SP, week of 2026-10-02) at PTAX 5.00:
    # 2.5639 / 1.6913 kg ATR/L x 1.0453 kg ATR/kg = R$1.5846/kg sugar = US$316.9/t = 14.375 c/lb
    assert brbasis.ethanol_parity_clb(2.5639, 5.0) == pytest.approx(14.375, abs=0.005)
    # a stronger real (lower PTAX) raises parity in dollar terms
    assert brbasis.ethanol_parity_clb(2.5639, 4.5) > brbasis.ethanol_parity_clb(2.5639, 5.0)


def test_sugar_line_is_us_minus_parity_with_its_own_signal_words():
    from src import brbasis
    spec = brbasis.BOOK_BY_KEY["sugar"]
    assert spec["sig_hi"].startswith("Sugar rich") and spec["sig_lo"].startswith("Sugar cheap")
