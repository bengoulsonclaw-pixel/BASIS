"""Futures last-trade rules vs the OBSERVED rolls in the deep store — never rebaseline.

src/expiries.py reconstructs each exchange's calendar; data/price_store/deep_contract.parquet
records which contract was actually front on every day. A completed front run ends ON its
contract's last trade (or a day early, where the store rolls ahead of the close), so the
store is independent evidence of the futures rule — the same discipline as
test_bbgcodes.py::test_rules_reproduce_observed_expiries for options.

Caught on the way in (2026-10-06): CME 3M SOFR and ICE 3M SONIA are named for the START of
their reference quarter and last-trade at its END. expiry_for("SFRA Comdty", "", 2026, 6)
said 16 Jun 2026; SFRM6 traded to 15 Sep. Unshifted the rule matched 0 of 33 runs for each,
shifted +3 months (SPECS `fut_shift`) 33 of 33. Euribor is fixed in advance and named for its
own month — it must stay unshifted (41 of 41).

If this fails after an expiries.py edit, the edit is wrong until proven otherwise.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src import bbgcodes, curvemon, expiries

_MIN_RUNS = 30          # 2026-10-06: 33 completed runs for SFRA/SFIA, 41 for ERA


@pytest.fixture(scope="module")
def history():
    hist = curvemon.load_history()
    if hist is None or hist.empty:
        pytest.skip("no deep store on this machine")
    return hist


@pytest.mark.parametrize("ticker", ["SFRA Comdty", "SFIA Comdty", "ERA Comdty"])
def test_every_observed_front_run_ends_on_the_rule_last_trade(history, ticker):
    runs = curvemon._runs_for(history, ticker)
    hol = expiries._holidays_for(ticker, "")
    checked, bad = 0, []
    for r in runs.iloc[:-1].itertuples(index=False):          # last run is still live
        if r.month is None or pd.isna(r.month):
            continue
        ltd = expiries.expiry_for(ticker, "", r.month.year, r.month.month, "fut")
        end = pd.Timestamp(r.end).date()
        checked += 1
        if ltd is None or not (end == ltd or end == expiries._prev_bday(ltd, hol)):
            bad.append((r.contract, end.isoformat(), ltd and ltd.isoformat()))
    assert checked >= _MIN_RUNS, (ticker, checked)
    assert not bad, (ticker, bad[:5])


@pytest.mark.parametrize("ticker,year,month,expected", [
    ("SFRA Comdty", 2026, 6, date(2026, 9, 15)),     # SFRM6: window 17 Jun → 16 Sep
    ("SFRA Comdty", 2026, 9, date(2026, 12, 15)),    # SFRU6
    ("SFIA Comdty", 2026, 6, date(2026, 9, 15)),     # SFIM6
    ("ERA Comdty", 2026, 9, date(2026, 9, 14)),      # Euribor: its own month, unshifted
])
def test_named_contract_last_trade(ticker, year, month, expected):
    assert expiries.expiry_for(ticker, "", year, month, "fut") == expected


def test_options_on_in_arrears_futures_stay_in_their_named_month():
    """Only the FUTURES rule shifts — SR3 / SONIA options expire in the month they name."""
    assert expiries.expiry_for("SFRA Comdty", "", 2026, 6, "opt") == date(2026, 6, 12)
    assert expiries.expiry_for("SFIA Comdty", "", 2026, 6, "opt") == date(2026, 6, 12)


def test_bbg_codes_decodes_sofr_june_to_a_september_last_trade():
    d = bbgcodes.decode("SFRM6 Comdty")
    assert d["ok"] and d["contract"] == "Jun 2026"
    assert d["fut_expiry"] == date(2026, 9, 15)
    assert d["opt_expiry"] == date(2026, 6, 12)
