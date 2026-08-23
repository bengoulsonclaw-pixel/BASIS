"""goldstore.py — the point-in-time observation store for the Gold Signal Engine.

Why this exists
---------------
The single biggest failure mode in a macro backtest is testing against data nobody
had at the time. Economic series get revised, sometimes heavily: US nonfarm payrolls
for May 2024 first printed 158,543k on 10 Jun 2024 and reads 157,608k today. A model
fitted on today's vintage has been shown 935k of information that did not exist, and
it will look skilful for entirely fake reasons.

This module makes that structurally impossible. Every observation carries the date it
DESCRIBES and the timestamp the world FIRST SAW IT, and `get_series` is the only way
to read the table. Feature code cannot accidentally see the future because it cannot
reach the raw rows at all.

Why parquet and not the SQL in the spec
---------------------------------------
BASIS has no database. There is no sqlite, postgres, duckdb or ORM anywhere in the
tree — storage is parquet and JSON files under data/. The spec's own instruction is to
extend the existing store rather than introduce a second one, so the DDL is honoured as
a SCHEMA (identical columns, identical primary key semantics) on the append-only long
parquet pattern already proven by src/eqfunda.py.

Primary key: (series_id, reference_date, revision).

Publication timestamps, and how honest each one is
--------------------------------------------------
Three tiers, and `series_meta` records which applies so nothing is silently optimistic:

  EXACT    Market data — prices, LBMA fixes, SGE closes, GLD tonnage, Yahoo closes.
           Final when printed, never revised. published_at = the close itself.
  VINTAGE  FRED/ALFRED series pulled with output_type=3, which returns the full
           revision matrix: every reference date against every vintage that changed
           it. This is genuinely point-in-time, revision numbers and all.
  LAGGED   Everything else — published_at = reference_date + typical_lag. An
           approximation. `published_at_approximated` is set True and any backtest
           leaning on these series must be described as optimistic.

Release cadence decides which horizon a series can serve
--------------------------------------------------------
Every series is registered with its `native_freq` and `typical_lag_days`, and
`horizon_role` turns those two facts into a verdict per forecast horizon:
**drives / marginal / static**. The rule it encodes is easy to state and easy to
forget in the middle of a feature list:

    A series cannot drive a forecast over a window shorter than its own
    release cycle.

WGC quarterly demand lands four times a year on a six-week lag. Across a 5-day
window it is, with probability ~0.95, the identical number on the last day and the
first — a constant. Constants say nothing about variation, so no matter how much
the underlying economics matter, that series cannot inform the 5-day call. It earns
its keep at 250 days, where it refreshes three or four times.

Cadence and lag are tested separately because they fail differently. India's import
volumes refresh twice inside a 60-day window (cadence says "drives") but arrive
about 150 days late, so the window they would have informed has already closed —
demoted to marginal. Central bank reserves, monthly on a 45-day lag, survive the
same test and drive the 60-day horizon.

None of this is hand-typed per series: it falls out of the two metadata fields, so
it cannot drift out of date the way a comment would. `horizon_matrix()` prints the
whole picture and `horizon_inputs(horizon)` is what the model layer should build
its feature set from — which makes the rule a guard rather than a note, since a
quarterly feed then cannot quietly end up in the 5-day fit.

CLI:  python src/goldstore.py [--coverage] [--stale] [--horizons]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

STORE_DIR = _ROOT / "data" / "gold_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)
OBS_FILE = STORE_DIR / "observations.parquet"
META_FILE = STORE_DIR / "series_meta.json"

# The spec's DDL, as a frame contract.
COLUMNS = ["series_id", "reference_date", "published_at", "value",
           "revision", "source", "is_synthetic"]
KEY = ["series_id", "reference_date", "revision"]

BUCKETS = ("monetary", "flows", "physical", "risk", "valuation")

# Spec §1 horizons, in calendar days.
HORIZONS = {"5d": 5, "60d": 60, "250d": 250}

# How often each frequency actually refreshes, in calendar days.
FREQ_DAYS = {"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 91, "annual": 365}


# ---------------------------------------------------------------------------
# series_meta
# ---------------------------------------------------------------------------
def _read_meta() -> dict:
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def register(series_id: str, *, description: str, unit: str, native_freq: str,
             typical_lag_days: float, bucket: str, source_url: str = "",
             published_at_approximated: bool = False) -> dict:
    """Declare a series. Idempotent — re-registering updates the metadata in place."""
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}, got {bucket!r}")
    if native_freq not in ("daily", "weekly", "monthly", "quarterly", "annual"):
        raise ValueError(f"unexpected native_freq {native_freq!r}")
    meta = _read_meta()
    meta[series_id] = {"series_id": series_id, "description": description, "unit": unit,
                       "native_freq": native_freq, "typical_lag_days": float(typical_lag_days),
                       "bucket": bucket, "source_url": source_url,
                       "published_at_approximated": bool(published_at_approximated)}
    META_FILE.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta[series_id]


def meta(series_id: str | None = None):
    m = _read_meta()
    return m if series_id is None else m.get(series_id, {})


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def _empty() -> pd.DataFrame:
    return pd.DataFrame({"series_id": pd.Series(dtype="object"),
                         "reference_date": pd.Series(dtype="datetime64[ns]"),
                         "published_at": pd.Series(dtype="datetime64[ns]"),
                         "value": pd.Series(dtype="float64"),
                         "revision": pd.Series(dtype="int64"),
                         "source": pd.Series(dtype="object"),
                         "is_synthetic": pd.Series(dtype="bool")})


def _read_obs() -> pd.DataFrame:
    if not OBS_FILE.exists():
        return _empty()
    df = pd.read_parquet(OBS_FILE)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def put(series_id: str, obs, *, source: str, is_synthetic: bool = False,
        typical_lag_days: float | None = None) -> int:
    """Write observations. Returns the number of NEW rows written.

    `obs` is either
      * a Series indexed by reference_date — one vintage. published_at is derived as
        reference_date + typical_lag (from series_meta unless overridden), and the
        series is flagged approximated unless its metadata already says otherwise; or
      * a DataFrame with columns reference_date, published_at, value — true vintages,
        e.g. straight off ALFRED.

    Revisions are assigned by publication order per reference_date. A value identical
    to the one already in force is NOT stored again: without that check a daily
    re-pull of an unrevised monthly series would manufacture a new 'revision' every
    day and the revision column would stop meaning anything."""
    if isinstance(obs, pd.Series):
        lag = typical_lag_days
        if lag is None:
            lag = meta(series_id).get("typical_lag_days", 0.0)
        new = pd.DataFrame({"reference_date": pd.to_datetime(obs.index),
                            "value": pd.to_numeric(obs.to_numpy(), errors="coerce")})
        new["published_at"] = new["reference_date"] + pd.Timedelta(days=float(lag))
    else:
        new = obs.copy()
        new["reference_date"] = pd.to_datetime(new["reference_date"])
        new["published_at"] = pd.to_datetime(new["published_at"])
        new["value"] = pd.to_numeric(new["value"], errors="coerce")
    new = new.dropna(subset=["reference_date", "published_at", "value"])
    if new.empty:
        return 0

    existing = _read_obs()
    have = existing[existing["series_id"] == series_id]

    # Drop anything already stored with the same (reference_date, published_at, value)
    # — a re-pull of unchanged history must be a no-op, not a duplicate.
    if not have.empty:
        seen = set(zip(have["reference_date"], have["published_at"]))
        new = new[[(r, p) not in seen
                   for r, p in zip(new["reference_date"], new["published_at"])]]
        if new.empty:
            return 0
        # ...and drop restatements that restate nothing: same reference date, later
        # publication, identical value.
        #
        # The in-force value has to be tracked THROUGH the incoming batch, not read
        # once from what was already stored. ALFRED hands over a whole vintage history
        # at a time, so a batch routinely contains several publications for the same
        # reference date; comparing every one of them against the pre-batch value
        # meant a genuine revert (A published, revised to B, restated back to A) had
        # its final A dropped, because A still matched the stale in-force reading.
        # The store then claimed B was current when the agency had gone back to A.
        latest = dict((have.sort_values("published_at")
                           .groupby("reference_date")["value"].last()).items())
        new = new.sort_values("published_at")
        keep = []
        for r, v in zip(new["reference_date"], new["value"]):
            prev = latest.get(r)
            dup = prev is not None and np.isclose(v, prev, rtol=0, atol=1e-12)
            keep.append(not dup)
            if not dup:
                latest[r] = v          # this row is now the value in force
        new = new[keep]
        if new.empty:
            return 0

    new["series_id"] = series_id
    new["source"] = source
    new["is_synthetic"] = bool(is_synthetic)

    combined = pd.concat([existing, new[[c for c in COLUMNS if c != "revision"]]],
                         ignore_index=True)
    combined = combined.sort_values(["series_id", "reference_date", "published_at"])
    combined["revision"] = combined.groupby(["series_id", "reference_date"]).cumcount()
    combined = combined[COLUMNS].reset_index(drop=True)
    combined.to_parquet(OBS_FILE, index=False)
    return int(len(new))


# ---------------------------------------------------------------------------
# THE ACCESSOR — the only sanctioned read path
# ---------------------------------------------------------------------------
def get_series(series_id: str, as_of=None) -> pd.Series:
    """The value of `series_id` as it stood on `as_of`, indexed by reference_date.

    This is the ONLY function feature code may use to reach the data, and the whole
    point-in-time guarantee lives in these three lines: filter to rows already
    published, take the latest publication per reference date, done. `as_of=None`
    means "everything known now" and is for reporting, never for a backtest."""
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if df.empty:
        return pd.Series(dtype=float, name=series_id)
    if as_of is not None:
        df = df[df["published_at"] <= pd.Timestamp(as_of)]
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
    out = (df.sort_values("published_at")
             .groupby("reference_date")["value"].last().sort_index())
    return out.rename(series_id)


def get_frame(series_ids, as_of=None) -> pd.DataFrame:
    """Several series on one reference-date index, each read point-in-time."""
    cols = {sid: get_series(sid, as_of) for sid in series_ids}
    cols = {k: v for k, v in cols.items() if len(v)}
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def as_known_series(series_id: str, dates: pd.DatetimeIndex) -> pd.Series:
    """The value **in force on each calendar date**, forward-filled from
    `published_at` — never from `reference_date`.

    This is spec §4.1's rule, and the distinction is the whole ball game. CPI for
    January is *about* January but is not knowable until mid-February. Filling from
    the reference date puts January's number on a January calendar row and hands the
    model three weeks of the future, every month, silently.

    Built by replaying publications in order rather than by querying per date, which
    would be O(days x rows). Walking once and snapshotting is O(n log n):

      * maintain {reference_date: value in force}, updated at each publication;
      * the level a trader would quote is the value at the LATEST reference date
        seen so far — so a revision to an *older* reference date does not move the
        current level, but a revision to the current one does;
      * `max_ref` only ever grows, so it is tracked rather than recomputed.
    """
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if df.empty:
        return pd.Series(index=dates, dtype=float, name=series_id)
    df = df.sort_values(["published_at", "reference_date", "revision"])

    values: dict = {}
    max_ref = None
    stamps, levels = [], []
    for pub, ref, val in zip(df["published_at"], df["reference_date"], df["value"]):
        values[ref] = val
        if max_ref is None or ref > max_ref:
            max_ref = ref
        stamps.append(pub)
        levels.append(values[max_ref])
    walk = pd.Series(levels, index=pd.DatetimeIndex(stamps))
    walk = walk[~walk.index.duplicated(keep="last")].sort_index()
    return walk.reindex(walk.index.union(dates)).ffill().reindex(dates).rename(series_id)


def daily_panel(series_ids, start=None, end=None) -> pd.DataFrame:
    """Several series as a point-in-time daily panel, business-day indexed.

    Every column answers "what would a trader have had on this date?", so feature
    code can treat it as an ordinary frame without having to remember the vintage
    rules on every line."""
    obs = _read_obs()
    if obs.empty:
        return pd.DataFrame()
    start = pd.Timestamp(start) if start is not None else obs["reference_date"].min()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start, end)
    cols = {sid: as_known_series(sid, dates) for sid in series_ids}
    return pd.DataFrame(cols, index=dates)


def last_reference(series_id: str, as_of=None):
    """Newest reference_date actually observed for this series (None if absent).

    daily_panel() forward-fills to today, which is right for a FEATURE (you hold the
    last known value) and wrong for a TARGET: a stalled price feed makes every
    forward return in the ffilled tail exactly zero, which is not missing data, it is
    a fabricated claim that the price did not move. Callers building forward-looking
    quantities use this to cut the panel back to observed history."""
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if as_of is not None:
        df = df[df["published_at"] <= pd.Timestamp(as_of)]
    return None if df.empty else df["reference_date"].max()


def first_publication(series_id: str) -> pd.Series:
    """First publication timestamp per reference_date — i.e. the real release dates.

    A sanctioned accessor for a question the point-in-time walk answers only slowly:
    "when did the world first see each reference month?" Doing it by stepping
    `get_series(as_of=d)` across every business day is O(days x rows), which is why
    a caller was tempted to monkeypatch `_read_obs` for speed. Reaching into the
    store's internals to go faster is exactly what the §3 lint rule exists to catch,
    so the fast path belongs here instead — same answer, one pass, no private access.

    Returns a Series indexed by reference_date holding the earliest published_at."""
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    return (df.groupby("reference_date")["published_at"].min()
              .sort_values().rename(series_id))


def last_published(series_id: str, as_of=None):
    """When this series was last updated, as known at `as_of`. Drives staleness."""
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if as_of is not None:
        df = df[df["published_at"] <= pd.Timestamp(as_of)]
    return None if df.empty else df["published_at"].max()


def stale_flags(as_of=None) -> list:
    """Flag series whose NEXT release should already have arrived.

    Spec §8 words this as "more stale than twice its typical publication lag", but
    taken literally that is wrong for anything but a daily series: a monthly figure
    with a 7-day lag would be "stale" a fortnight after every release, for ever. The
    question that matters is whether the next print is overdue, so the tolerance is
    `cadence + lag` rather than `2 x lag`.

    Measured in BUSINESS days. The calendar-day version flagged every daily market
    series as stale each weekend — on a Saturday, Friday's close is two calendar days
    old against a one-day tolerance. That produced 20 false flags in the §8 payload
    and buried the one series that is genuinely dead (Russia stopped reporting its
    reserves in 2025-11). Noisy staleness is worse than none: it trains a reader to
    ignore the field."""
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
    out = []
    for sid, m in meta().items():
        # as_of=now, not None: the LAGGED tier stamps published_at as
        # reference_date + lag, which for a fresh monthly observation lands in the
        # future. Counting those would make a dead feed look alive.
        lp = last_published(sid, now)
        if lp is None:
            out.append(f"missing_series:{sid}")
            continue
        cadence = FREQ_DAYS.get(m.get("native_freq", ""), 1.0)
        lag = max(float(m.get("typical_lag_days") or 0.0), 1.0)
        limit_bdays = int(np.ceil((cadence + lag) * 5.0 / 7.0))
        overdue_bdays = int(np.busday_count(lp.date(), now.date())) if now > lp else 0
        if overdue_bdays > limit_bdays:
            out.append(f"stale_series:{sid}")
    return sorted(out)


def horizon_role(series_id: str) -> dict:
    """Which forecast horizons this series can actually move, derived from its
    release cadence and publication lag.

    The point, and it is easy to lose: **a series cannot drive a forecast over a
    window shorter than its own release cycle.** WGC quarterly demand lands four
    times a year on a six-week lag. Across a 5-day forecast window it is, with
    probability ~0.95, exactly the same number on the last day as on the first —
    a constant. Constants carry no information about variation, so however
    important the underlying economics, that series cannot inform the 5-day call.
    It informs the 250-day one, where it refreshes three or four times.

    Two independent tests, because they fail differently:

      REFRESH  horizon / cadence. Below ~0.5 the series is static across the
               window; above ~2 it refreshes enough to carry signal.
      LAG      typical_lag > horizon means that by the time the figure is
               published the window it would have informed has already closed.
               A quarterly print arriving six weeks late is news about a period
               that a 5-day forecast has long since lived through.

    Returns {horizon: {"role", "refresh_ratio", "why"}} with role in
    drives | marginal | static. Nothing here is hand-typed per series — it falls
    out of native_freq and typical_lag_days, so it cannot drift out of date the
    way a comment in a docstring would."""
    m = meta(series_id)
    if not m:
        return {}
    cadence = FREQ_DAYS.get(m.get("native_freq", ""), 1.0)
    lag = float(m.get("typical_lag_days") or 0.0)
    out = {}
    for label, days in HORIZONS.items():
        ratio = days / cadence
        if ratio >= 2.0:
            role, why = "drives", f"refreshes ~{ratio:.0f}x per window"
        elif ratio >= 0.5:
            role, why = "marginal", f"refreshes ~{ratio:.1f}x per window"
        else:
            role = "static"
            why = (f"refreshes ~{ratio:.2f}x per window — effectively constant "
                   f"across {label}")
        if role != "static" and lag > days:
            role = "static" if role == "marginal" else "marginal"
            why += f"; published {lag:.0f}d late vs a {days}d window"
        out[label] = {"role": role, "refresh_ratio": round(ratio, 3), "why": why}
    return out


def horizon_matrix() -> pd.DataFrame:
    """Every registered series against every horizon — the table that says, in one
    place, what each feed can and cannot be expected to move.

    This is the answer to 'why is central bank buying not in the 5-day model?'
    without anyone having to remember the argument."""
    rows = []
    for sid, m in sorted(meta().items()):
        roles = horizon_role(sid)
        row = {"series_id": sid, "bucket": m.get("bucket", ""),
               "freq": m.get("native_freq", ""),
               "lag_days": m.get("typical_lag_days"),
               "approx_pub": m.get("published_at_approximated")}
        row.update({h: roles.get(h, {}).get("role", "") for h in HORIZONS})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("series_id")
    order = {"monetary": 0, "flows": 1, "physical": 2, "risk": 3, "valuation": 4}
    return df.sort_values(["bucket", "freq"], key=lambda s: s.map(order).fillna(9)
                          if s.name == "bucket" else s)


def horizon_inputs(horizon: str, roles=("drives",)) -> list:
    """Series ids admissible at `horizon`. The model layer should build its feature
    set from this rather than from a hand-kept list, so a feed whose cadence makes
    it useless at 5d cannot quietly end up in the 5d fit."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {list(HORIZONS)}, got {horizon!r}")
    return sorted(sid for sid in meta()
                  if horizon_role(sid).get(horizon, {}).get("role") in roles)


def coverage() -> pd.DataFrame:
    """One row per series: history span, row count, and how many revisions exist."""
    df = _read_obs()
    if df.empty:
        return pd.DataFrame()
    m = meta()
    now = pd.Timestamp.now()
    # The LAGGED tier stamps published_at as reference_date + lag, so a fresh monthly
    # observation legitimately carries a FUTURE publication date — it is on file but
    # not yet knowable. stale_flags() already accounts for that; this table did not,
    # and showed those stamps in `last_published` as though they had been released.
    pub = df[df["published_at"] <= now]
    g = df.groupby("series_id")
    gp = pub.groupby("series_id")
    out = pd.DataFrame({
        "rows": g.size(),
        "first": g["reference_date"].min().dt.date,
        "last": g["reference_date"].max().dt.date,
        "last_published": gp["published_at"].max().dt.date,
        # Observations held but not yet knowable at `now`.
        "pending": g["published_at"].apply(lambda s: int((s > now).sum())),
        "revised_points": g["revision"].apply(lambda s: int((s > 0).sum())),
        "synthetic": g["is_synthetic"].any(),
    })
    out["bucket"] = [m.get(s, {}).get("bucket", "") for s in out.index]
    out["freq"] = [m.get(s, {}).get("native_freq", "") for s in out.index]
    out["approx_pub"] = [m.get(s, {}).get("published_at_approximated", None)
                         for s in out.index]
    return out.sort_values(["bucket", "series_id"])


# ---------------------------------------------------------------------------
# ALFRED loader — the one ingestion helper that belongs here, because it is what
# makes the `revision` and `published_at` columns real rather than decorative.
# ---------------------------------------------------------------------------
def fred_vintages(fred_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Full revision history of a FRED series as (reference_date, published_at, value).

    Uses ALFRED `output_type=3` ("new and revised observations only"), which returns a
    matrix: one row per reference date, one column per VINTAGE DATE, populated only
    where that vintage changed the value. Unpacking it gives exactly the observations
    table — one call per series instead of one call per vintage.

    Column names come back as `SERIESID_YYYYMMDD`; the suffix IS the publication date.
    """
    import urllib.parse
    import urllib.request

    from src import macrodata
    key = macrodata.fred_key()
    if not key:
        raise RuntimeError("no FRED key — see data/fred_key.txt")
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode({
        "series_id": fred_id, "api_key": key, "file_type": "json",
        "observation_start": start,
        "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
        "output_type": 3})
    req = urllib.request.Request(url, headers={"User-Agent": "basis-gold-engine"})
    rows = json.loads(urllib.request.urlopen(req, timeout=120).read())["observations"]
    return parse_vintage_matrix(rows)


def parse_vintage_matrix(rows: list) -> pd.DataFrame:
    """Unpack ALFRED's output_type=3 matrix into long (reference_date, published_at,
    value). Split out from the fetch so the parsing — the part that can silently
    mis-date every observation in the store — is testable without a network call."""
    out = []
    for r in rows:
        try:
            ref = pd.Timestamp(r["date"])
        except Exception:
            continue
        for col, val in r.items():
            if col == "date" or val in (None, "", "."):
                continue
            stamp = col.rsplit("_", 1)[-1]
            try:
                pub = pd.Timestamp(datetime.strptime(stamp, "%Y%m%d"))
                v = float(val)
            except (ValueError, TypeError):
                continue
            out.append((ref, pub, v))
    df = pd.DataFrame(out, columns=["reference_date", "published_at", "value"])
    if df.empty:
        return df.astype({"value": "float64"})
    return df.sort_values(["reference_date", "published_at"]).reset_index(drop=True)


def main() -> int:
    if "--horizons" in sys.argv:
        m = horizon_matrix()
        print(m.to_string() if not m.empty else "no series registered yet")
        return 0
    if "--stale" in sys.argv:
        flags = stale_flags()
        print("\n".join(flags) if flags else "no stale series")
        return 0
    cov = coverage()
    if cov.empty:
        print("gold store is empty — run the Milestone 2 ingestion")
        return 0
    print(f"{OBS_FILE}  ({len(_read_obs()):,} observations, "
          f"{cov.shape[0]} series)")
    print(cov.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
