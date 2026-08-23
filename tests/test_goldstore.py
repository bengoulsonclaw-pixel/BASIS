"""Locks on the Gold Signal Engine's point-in-time store (Milestone 1).

The spec calls this the single biggest failure mode, and it is right: a backtest
reading today's vintage of a revised series looks skilful for entirely fake reasons.
US payrolls for May 2024 first printed 158,543k and now reads 157,608k — 935k of
hindsight, available to any model that queries the table carelessly.

These tests prove the carelessness is impossible:

  * `get_series` never returns a value published after the as-of date.
  * When several vintages qualify, the latest QUALIFYING one wins — not the latest
    overall, which would leak, and not the first, which would ignore revisions.
  * The lint rule from spec §3: nothing outside goldstore.py touches the raw table.
  * Revision numbering, idempotent writes, and the staleness flags behave.

Everything runs against a temp store — never the repo's own data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import goldstore as gsd


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Redirect the store at a temp dir. Every test gets a clean table."""
    monkeypatch.setattr(gsd, "STORE_DIR", tmp_path)
    monkeypatch.setattr(gsd, "OBS_FILE", tmp_path / "observations.parquet")
    monkeypatch.setattr(gsd, "META_FILE", tmp_path / "series_meta.json")
    return gsd


def _vintages() -> pd.DataFrame:
    """One reference date revised three times, plus a neighbour — the real payrolls
    shape, hand-written so the expected answers are obvious."""
    return pd.DataFrame([
        ("2024-05-01", "2024-06-07", 158543.0),
        ("2024-05-01", "2024-07-05", 158432.0),
        ("2024-05-01", "2025-02-07", 157828.0),
        ("2024-06-01", "2024-07-05", 158700.0),
    ], columns=["reference_date", "published_at", "value"])


# ---------------------------------------------------------------------------
# the point-in-time guarantee
# ---------------------------------------------------------------------------
def test_get_series_never_returns_unpublished_data(store):
    store.put("PAYEMS", _vintages(), source="test")
    may = pd.Timestamp("2024-05-01")

    # before anything was published: the series is empty, not zero, not the first print
    assert store.get_series("PAYEMS", "2024-06-01").empty

    # each as-of sees exactly the vintage in force on that day
    assert store.get_series("PAYEMS", "2024-06-07")[may] == 158543.0
    assert store.get_series("PAYEMS", "2024-06-30")[may] == 158543.0
    assert store.get_series("PAYEMS", "2024-07-05")[may] == 158432.0
    assert store.get_series("PAYEMS", "2025-06-01")[may] == 157828.0
    assert store.get_series("PAYEMS")[may] == 157828.0          # as_of=None = latest

    # the June reference date is invisible until its own publication
    jun = pd.Timestamp("2024-06-01")
    assert jun not in store.get_series("PAYEMS", "2024-07-04").index
    assert store.get_series("PAYEMS", "2024-07-05")[jun] == 158700.0


def test_published_at_boundary_is_inclusive(store):
    """A figure released at as_of IS knowable. Off-by-one here silently discards a
    day of information at every refit."""
    store.put("X", _vintages(), source="test")
    assert not store.get_series("X", "2024-06-07").empty
    assert store.get_series("X", "2024-06-06").empty


def test_latest_qualifying_vintage_wins_not_the_latest_overall(store):
    """The subtle leak: taking max(published_at) across ALL rows rather than across
    the FILTERED rows returns today's number for every historical as-of."""
    store.put("X", _vintages(), source="test")
    may = pd.Timestamp("2024-05-01")
    got = store.get_series("X", "2024-07-06")[may]
    assert got == 158432.0
    assert got != 157828.0, "leaked the latest revision into an earlier as-of"


def test_derived_publication_uses_the_typical_lag(store):
    """A Series in means published_at = reference_date + typical_lag, so a monthly
    figure is not readable on the day it describes."""
    store.register("CPI", description="CPI", unit="index", native_freq="monthly",
                   typical_lag_days=14, bucket="risk", published_at_approximated=True)
    s = pd.Series([100.0, 101.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    store.put("CPI", s, source="test")
    assert store.get_series("CPI", "2024-02-01").empty          # not yet published
    assert len(store.get_series("CPI", "2024-02-14")) == 1
    assert len(store.get_series("CPI", "2024-03-15")) == 2


# ---------------------------------------------------------------------------
# write semantics
# ---------------------------------------------------------------------------
def test_revisions_are_numbered_by_publication_order(store):
    store.put("X", _vintages(), source="test")
    df = pd.read_parquet(store.OBS_FILE)
    may = df[df["reference_date"] == pd.Timestamp("2024-05-01")].sort_values("published_at")
    assert list(may["revision"]) == [0, 1, 2]
    assert may.iloc[0]["value"] == 158543.0, "revision 0 must be the FIRST print"


def test_reput_is_a_noop_and_restatements_that_restate_nothing_are_dropped(store):
    v = _vintages()
    assert store.put("X", v, source="test") == 4
    assert store.put("X", v, source="test") == 0, "re-pull duplicated rows"
    # a later vintage carrying an unchanged value must not manufacture a revision
    same = pd.DataFrame([("2024-05-01", "2025-09-01", 157828.0)],
                        columns=["reference_date", "published_at", "value"])
    assert store.put("X", same, source="test") == 0
    # a genuine restatement must be stored
    diff = pd.DataFrame([("2024-05-01", "2025-09-01", 157700.0)],
                        columns=["reference_date", "published_at", "value"])
    assert store.put("X", diff, source="test") == 1
    assert store.get_series("X")[pd.Timestamp("2024-05-01")] == 157700.0


def test_schema_matches_the_spec(store):
    store.put("X", _vintages(), source="test")
    df = pd.read_parquet(store.OBS_FILE)
    assert list(df.columns) == gsd.COLUMNS
    assert not df.duplicated(subset=gsd.KEY).any(), "primary key violated"
    assert df["revision"].dtype.kind == "i"
    assert df["is_synthetic"].dtype == bool


def test_register_rejects_an_unknown_bucket(store):
    with pytest.raises(ValueError):
        store.register("X", description="d", unit="u", native_freq="daily",
                       typical_lag_days=0, bucket="vibes")


def test_synthetic_data_is_flagged(store):
    """Spec §2.7 requires the pre-2003 real-yield splice to be marked. If the flag
    does not survive a round trip, synthetic data becomes indistinguishable from
    measured data in the store."""
    s = pd.Series([2.0], index=pd.to_datetime(["1995-01-31"]))
    store.put("REAL10_SYNTH", s, source="derived", is_synthetic=True)
    df = pd.read_parquet(store.OBS_FILE)
    assert bool(df["is_synthetic"].iloc[0]) is True


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------
def test_stale_flag_fires_past_twice_the_typical_lag(store):
    store.register("SLOW", description="s", unit="u", native_freq="monthly",
                   typical_lag_days=30, bucket="physical")
    store.put("SLOW", pd.DataFrame([("2024-01-31", "2024-03-01", 1.0)],
                                   columns=["reference_date", "published_at", "value"]),
              source="test")
    assert store.stale_flags("2024-04-01") == []            # 31d < 2 x 30d
    assert store.stale_flags("2024-05-15") == ["stale_series:SLOW"]


def test_missing_series_is_flagged_not_silently_ignored(store):
    store.register("NEVER", description="n", unit="u", native_freq="daily",
                   typical_lag_days=1, bucket="flows")
    assert store.stale_flags("2024-01-01") == ["missing_series:NEVER"]


# ---------------------------------------------------------------------------
# the ALFRED vintage parser
# ---------------------------------------------------------------------------
def test_vintage_matrix_unpacks_publication_dates_from_column_names():
    """ALFRED returns a matrix whose COLUMN NAMES carry the publication date
    (`PAYEMS_20240607`). Mis-parsing that suffix would date every observation
    wrongly while still producing a perfectly plausible table."""
    rows = [{"date": "2024-05-01", "PAYEMS_20240607": "158543",
             "PAYEMS_20240705": "158432", "PAYEMS_20250207": "."},
            {"date": "2024-06-01", "PAYEMS_20240705": "158700"}]
    out = gsd.parse_vintage_matrix(rows)
    assert len(out) == 3, "the '.' placeholder must be dropped, not coerced to 0"
    first = out.iloc[0]
    assert first["reference_date"] == pd.Timestamp("2024-05-01")
    assert first["published_at"] == pd.Timestamp("2024-06-07")
    assert first["value"] == 158543.0
    assert gsd.parse_vintage_matrix([]).empty


# ---------------------------------------------------------------------------
# spec §3: the accessor is the only path
# ---------------------------------------------------------------------------
def test_no_module_reads_the_observation_table_directly():
    """Spec §3: 'Add a lint rule or test that fails if any feature code reads the
    observations table directly.'

    The point-in-time guarantee is only worth anything if every reader goes through
    get_series. One pd.read_parquet on the raw file — easy to add while debugging,
    invisible in review — reinstates the leak everywhere downstream."""
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in sorted(src.glob("gold*.py")):
        if path.name == "goldstore.py":
            continue                      # the store is allowed to read its own table
        text = path.read_text(encoding="utf-8")
        for needle in ("observations.parquet", "OBS_FILE", "_read_obs", "gold_store"):
            if needle in text:
                offenders.append(f"{path.name} references {needle}")
    assert not offenders, (
        "feature code must reach observations only through goldstore.get_series: "
        + "; ".join(offenders))
