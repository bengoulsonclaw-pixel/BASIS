"""Checks on the ANP crude feed (src/anpdata.py).

Every bug this module hit produced a plausible NUMBER rather than an error, which is
the only kind worth testing for here:

  * pre-salt is a SUBSET of offshore — summing Terra + Mar + PreSal put Brazil at
    5.5 mb/d instead of 3.5;
  * ANP ships some months UTF-8 and others latin-1 — decoding all as latin-1 never
    raises, it just mangles the header and silently yields nothing;
  * ANP mixes decimal conventions — a fixed `decimal=` turns the other style into
    NaN and drops offshore, 95% of Brazil, out of a month that still "succeeds".

The parsing tests are pure and offline. The feed test skips when the cache is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import anpdata


# ── number parsing: both decimal conventions ─────────────────────────────────
def test_num_reads_comma_decimals():
    got = anpdata._num(pd.Series(["1615,7229", "0", ",578"]))
    assert got.tolist() == pytest.approx([1615.7229, 0.0, 0.578])


def test_num_reads_dot_decimals():
    got = anpdata._num(pd.Series(["1615.7229", "0.0000", "19.1"]))
    assert got.tolist() == pytest.approx([1615.7229, 0.0, 19.1])


def test_num_treats_dot_as_thousands_when_both_present():
    assert anpdata._num(pd.Series(["1.234,56"])).iloc[0] == pytest.approx(1234.56)


def test_num_never_silently_zeroes_a_bad_value():
    """A junk value must become NaN and drop out, not read as zero production."""
    assert bool(anpdata._num(pd.Series(["n/a"])).isna().iloc[0])


# ── encoding ─────────────────────────────────────────────────────────────────
def test_decode_handles_utf8_and_latin1():
    assert "Petróleo" in anpdata._decode("Petróleo (bbl/dia)".encode("utf-8"))
    assert "Petróleo" in anpdata._decode("Petróleo (bbl/dia)".encode("latin-1"))
    assert "Petróleo" in anpdata._decode("\ufeffPetróleo".encode("utf-8"))


def test_decode_does_not_mangle_utf8_as_latin1():
    """The exact failure: latin-1 decoding UTF-8 gives 'PetrÃ³leo', which made the
    header undetectable and lost nine of twelve months without erroring."""
    assert "PetrÃ³leo" not in anpdata._decode("Petróleo".encode("utf-8"))


# ── file discovery across five naming conventions ────────────────────────────
def test_discovery_regexes_cover_every_naming_convention():
    import re
    samples = {
        "2021_03_producao.zip": ("2021", "03"),
        "2021-08-producao.zip": ("2021", "08"),
        "producao-2022-04.zip": ("2022", "04"),
    }
    for tail, want in samples.items():
        m = (re.search(r"(\d{4})[_-](\d{2})[_-]?produc", tail, re.I)
             or re.search(r"produc\w*[_-](\d{4})[_-](\d{2})", tail, re.I))
        assert m and m.groups() == want, tail
    assert re.search(r"produc\w*[_-](\d{2})\.zip", "producao-04.zip", re.I).group(1) == "04"
    assert re.search(r"por-poco-(\d{4})\.zip", "producao-por-poco-2015.zip", re.I).group(1) == "2015"


# ── the cached feed, when this box has one ───────────────────────────────────
def _cached():
    if not anpdata.STORE.exists():
        pytest.skip("no ANP cache on this box")
    return pd.read_parquet(anpdata.STORE)


def test_monthly_totals_are_plausible_for_brazil():
    """Brazil produces roughly 3-4 mb/d of crude. A month far outside that means a
    region was lost (too low) or pre-salt was double-counted (too high)."""
    df = _cached()
    totals = df.groupby("file_month")["bopd"].sum()
    for month, v in totals.items():
        assert 2.0e6 < v < 4.5e6, f"{month}: {v:,.0f} bbl/d is not a plausible Brazil total"


def test_no_month_is_a_fraction_of_the_others():
    """The onshore-only failure looked like 162k bbl/d against a 3.5m median and did
    not raise. Any month under half the median means a missing region."""
    df = _cached()
    totals = df.groupby("file_month")["bopd"].sum()
    if len(totals) < 3:
        pytest.skip("too few months cached")
    assert totals.min() >= totals.median() * 0.5


def test_petrobras_is_the_largest_operator():
    got = anpdata.by_operator(_cached())
    top = next(iter(got["operators"]))
    assert "petrobras" in top.lower()
    assert 0.5 < got["operators"][top] / got["total_bopd"] < 0.95


def test_feed_reports_its_own_vintage():
    """The drop stops in 2023, so the window has to travel with the data — nothing
    downstream may present it as current."""
    got = anpdata.by_operator(_cached())
    assert got["first_month"] and got["last_month"]
    assert got["basis"] == "operator", "must never be labelled equity"
    assert got["n_months"] >= 1
