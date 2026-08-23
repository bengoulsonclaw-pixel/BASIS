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
def test_stale_flag_fires_when_the_next_release_is_overdue(store):
    """Tolerance is cadence + lag, not the spec's literal "twice the lag".

    Twice-the-lag is right for a daily series and wrong for everything else: a
    monthly figure with a 30-day lag would be "stale" 60 days after every release,
    i.e. permanently. What matters is whether the NEXT print is overdue."""
    store.register("SLOW", description="s", unit="u", native_freq="monthly",
                   typical_lag_days=30, bucket="physical")
    store.put("SLOW", pd.DataFrame([("2024-01-31", "2024-03-01", 1.0)],
                                   columns=["reference_date", "published_at", "value"]),
              source="test")
    assert store.stale_flags("2024-04-01") == []      # next print not yet due
    assert store.stale_flags("2024-07-15") == ["stale_series:SLOW"]


def test_a_daily_series_is_not_stale_over_a_weekend(store):
    """The bug this replaced: measured in CALENDAR days, every daily market series
    was flagged stale each Saturday — Friday's close is two days old against a
    one-day tolerance. That produced twenty false flags in the §8 payload and buried
    the one series that was genuinely dead."""
    store.register("PX", description="a daily price", unit="USD", native_freq="daily",
                   typical_lag_days=0, bucket="valuation")
    store.put("PX", pd.DataFrame([("2026-08-21", "2026-08-21", 100.0)],
                                 columns=["reference_date", "published_at", "value"]),
              source="test")
    assert store.stale_flags("2026-08-22") == [], "Saturday"
    assert store.stale_flags("2026-08-23") == [], "Sunday"
    assert store.stale_flags("2026-08-24") == [], "Monday, before that day's close"
    assert store.stale_flags("2026-08-27") == ["stale_series:PX"],         "three business days without a daily print IS stale"


def test_future_publication_dates_do_not_mask_a_dead_feed(store):
    """The LAGGED tier stamps published_at as reference_date + lag, so a fresh
    monthly observation can carry a future timestamp. Counting it as 'published'
    would make a stopped feed look current."""
    store.register("LAGGY", description="s", unit="u", native_freq="monthly",
                   typical_lag_days=45, bucket="physical",
                   published_at_approximated=True)
    store.put("LAGGY", pd.DataFrame([("2026-01-31", "2026-03-17", 1.0)],
                                    columns=["reference_date", "published_at", "value"]),
              source="test")
    assert store.last_published("LAGGY", "2026-08-23") == pd.Timestamp("2026-03-17")
    assert store.stale_flags("2026-08-23") == ["stale_series:LAGGY"]


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
# release cadence vs forecast horizon
# ---------------------------------------------------------------------------
def _register_representative_feeds(store):
    for sid, freq, lag, bucket in [
            ("DXY", "daily", 0, "monetary"),
            ("COT_MM_NET", "weekly", 3, "flows"),
            ("CB_RESERVES", "monthly", 45, "physical"),
            ("INDIA_IMPORTS", "monthly", 150, "physical"),
            ("WGC_DEMAND", "quarterly", 45, "physical"),
            ("MINE_PROD", "annual", 180, "valuation")]:
        store.register(sid, description=sid, unit="u", native_freq=freq,
                       typical_lag_days=lag, bucket=bucket)


def test_a_series_cannot_drive_a_window_shorter_than_its_release_cycle(store):
    """The whole point. WGC demand lands 4x a year on a 6-week lag: across a 5-day
    window it is the same number on the last day as the first, so it carries no
    information about variation over that window — however important the underlying
    economics. It earns its place at 250d, where it refreshes three or four times."""
    _register_representative_feeds(store)
    wgc = store.horizon_role("WGC_DEMAND")
    assert wgc["5d"]["role"] == "static"
    assert wgc["250d"]["role"] == "drives"

    # a daily market series is admissible everywhere
    dxy = store.horizon_role("DXY")
    assert {dxy[h]["role"] for h in ("5d", "60d", "250d")} == {"drives"}

    # weekly COT straddles a 5-day window — refreshes ~0.7x, so marginal, not drives
    assert store.horizon_role("COT_MM_NET")["5d"]["role"] == "marginal"
    assert store.horizon_role("COT_MM_NET")["60d"]["role"] == "drives"


def test_publication_lag_downgrades_independently_of_cadence(store):
    """Cadence and lag fail differently and both must bite. India's monthly import
    volumes refresh twice inside a 60-day window — cadence alone says 'drives' — but
    they arrive ~150 days late, so the window they would inform has already closed."""
    _register_representative_feeds(store)
    india = store.horizon_role("INDIA_IMPORTS")["60d"]
    cb = store.horizon_role("CB_RESERVES")["60d"]
    assert cb["role"] == "drives", "45d lag inside a 60d window is fine"
    assert india["role"] == "marginal", "150d lag on a 60d window must downgrade"
    assert "late" in india["why"]


def test_horizon_inputs_excludes_static_series(store):
    """The guard that makes this more than documentation: the 5-day model builds its
    feature set from this call, so a quarterly feed cannot quietly end up in it."""
    _register_representative_feeds(store)
    short = store.horizon_inputs("5d")
    assert "DXY" in short
    for slow in ("WGC_DEMAND", "CB_RESERVES", "INDIA_IMPORTS", "MINE_PROD"):
        assert slow not in short, f"{slow} must not be admissible at 5d"
    assert "WGC_DEMAND" in store.horizon_inputs("250d")
    with pytest.raises(ValueError):
        store.horizon_inputs("30d")


def test_horizon_matrix_covers_every_registered_series(store):
    _register_representative_feeds(store)
    m = store.horizon_matrix()
    assert len(m) == 6
    assert set(store.HORIZONS) <= set(m.columns)
    assert m.loc["MINE_PROD", "5d"] == "static"


# ---------------------------------------------------------------------------
# COT publication reconstruction — the leak fix
# ---------------------------------------------------------------------------
def test_cot_release_dates_including_holiday_slips():
    """The CFTC publishes only the Tuesday reference date, so the Friday 15:30 ET
    release has to be reconstructed. Getting this wrong reinstates the leak it
    exists to close: reading the report on its reference Tuesday is three days of
    look-ahead on a positioning signal, every week, across the whole sample.

    A federal holiday in the reference week slips the release one business day. The
    subtle case is a holiday landing ON the Friday (July 4 2025): counting the slip
    AND separately skipping the holiday double-counts it and lands on Tuesday
    instead of Monday."""
    from src.goldingest import cot_published_at
    cases = {
        "2026-08-18": "2026-08-21",   # ordinary week -> that Friday
        "2026-08-11": "2026-08-14",
        "2025-11-25": "2025-12-01",   # Thanksgiving (Thu) -> Monday
        "2025-07-01": "2025-07-07",   # July 4 falls ON the Friday -> Monday, not Tuesday
        "2025-12-23": "2025-12-29",   # Christmas (Thu) -> Monday
        "2026-01-20": "2026-01-26",   # MLK (Mon, before the Tue reference) -> Monday
        "2025-05-27": "2025-06-02",   # Memorial Day (Mon) -> Monday
        "2025-09-02": "2025-09-08",   # Labor Day (Mon) -> Monday
    }
    for ref, expected in cases.items():
        got = cot_published_at(ref)
        assert got.date().isoformat() == expected, f"{ref} -> {got.date()} != {expected}"
        assert got.hour == 15 and got.minute == 30, "release is 15:30 ET, not midnight"


def test_cot_release_time_prevents_same_morning_lookahead():
    """The 15:30 stamp is load-bearing. Stored at midnight, a model refitting on
    Friday morning would read Friday afternoon's report."""
    from src.goldingest import cot_published_at
    pub = cot_published_at("2026-08-18")
    assert pub > pd.Timestamp("2026-08-21 12:00"), "Friday noon must not see it"
    assert pub < pd.Timestamp("2026-08-21 23:59"), "Friday evening must see it"


# ---------------------------------------------------------------------------
# WGC licence enforcement at ingest
# ---------------------------------------------------------------------------
def test_wgc_price_columns_are_stripped_at_ingest():
    """The WGC licence on the LBMA price is absolute: supplied by WGC, it "may not
    be disclosed by you to anyone else". The trap is that the ETF workbook's charts
    sheet puts 'Gold Price (rhs)' directly beside the regional flow columns we do
    want, so it rides along unless something removes it deliberately.

    Note this restricts the WGC-supplied copy, not the LBMA fix itself — golddata
    pulls that straight from prices.lbma.org.uk under its own terms. Stripping here
    costs nothing because we already hold the same number from the primary source."""
    from src import wgc_fetch as w
    df = pd.DataFrame({"Date": [1], "North America": [2], "Europe": [3],
                       "Asia": [4], "Gold Price (rhs)": [5], "LBMA PM USD": [6]})
    out = w.strip_forbidden_columns(df)
    assert list(out.columns) == ["Date", "North America", "Europe", "Asia"]
    # a frame with nothing forbidden must come back untouched, not merely equal
    clean = pd.DataFrame({"Date": [1], "Tonnes": [2]})
    assert list(w.strip_forbidden_columns(clean).columns) == ["Date", "Tonnes"]


def test_publication_gate_blocks_wgc_and_flags_unregistered_sources():
    """The note-to-self, made executable. Report builders call publication_check()
    before a source's numbers reach a PDF; this locks that WGC is blocked today and
    — more importantly — that an UNKNOWN source is reported rather than waved
    through. Silence is exactly how a licensed number reaches a client."""
    from src import compliance
    blockers = compliance.publication_check("wgc")
    assert blockers, "WGC must not be publishable while compliance is undecided"
    assert any("never publish" in b for b in blockers), "the LBMA clause must surface"
    assert compliance.publication_check("not_a_registered_source"),         "an unregistered source must be flagged, not silently allowed"
    assert compliance.publication_check() == []
    assert compliance.required_citations("wgc") == ["Source: World Gold Council; Metals Focus"]


def test_wgc_client_facing_stays_gated_until_compliance_signs_off():
    """gold.org's site terms say 'personal, non-commercial use'; the workbook
    disclaimer permits 'review and commentary' with citation. Those pull in
    different directions for a broker publishing client research, so the flag stays
    False until XP compliance rules. Flipping it must be a deliberate act."""
    from src import wgc_fetch as w
    assert w.CLIENT_FACING_APPROVED is False
    assert "World Gold Council" in w.WGC_CITATION


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
        # Deliberately a raw substring scan, including inside comments. A rule that
        # tried to parse out prose could be talked into ignoring real code, and the
        # cost of the bluntness is small: a comment that needs to name a forbidden
        # handle can be reworded. When this fires, CHECK whether it is code before
        # assuming it is a false positive — it caught a genuine monkeypatch of the
        # store's private reader in goldevents.py.
        #
        # Only the handles that reach the OBSERVATIONS TABLE. An earlier version also
        # flagged the string "gold_store" and tripped on goldfeatures.py, which
        # legitimately writes its own features/targets tables into that directory —
        # a false positive on the folder name, not on the table. Guarding the
        # directory would push feature outputs somewhere arbitrary to appease a test.
        for needle in ("observations.parquet", "OBS_FILE", "_read_obs"):
            if needle in text:
                offenders.append(f"{path.name} references {needle}")
    assert not offenders, (
        "feature code must reach observations only through goldstore.get_series: "
        + "; ".join(offenders))
