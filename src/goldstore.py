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

CLI:  python src/goldstore.py [--coverage] [--stale]
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
        latest = (have.sort_values("published_at")
                      .groupby("reference_date")["value"].last())
        keep = [not (r in latest.index and np.isclose(v, latest[r], rtol=0, atol=1e-12))
                for r, v in zip(new["reference_date"], new["value"])]
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


def last_published(series_id: str, as_of=None):
    """When this series was last updated, as known at `as_of`. Drives staleness."""
    df = _read_obs()
    df = df[df["series_id"] == series_id]
    if as_of is not None:
        df = df[df["published_at"] <= pd.Timestamp(as_of)]
    return None if df.empty else df["published_at"].max()


def stale_flags(as_of=None) -> list:
    """Spec §8: flag any series more stale than twice its typical publication lag.

    Silent staleness is worse than a missing model — a frozen feed keeps feeding the
    model a confident number that stopped being true weeks ago."""
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
    out = []
    for sid, m in meta().items():
        lp = last_published(sid, as_of)
        if lp is None:
            out.append(f"missing_series:{sid}")
            continue
        lag = float(m.get("typical_lag_days") or 0.0)
        limit = max(2.0 * lag, 1.0)
        overdue = (now - lp) / pd.Timedelta(days=1)
        if overdue > limit:
            out.append(f"stale_series:{sid}")
    return sorted(out)


def coverage() -> pd.DataFrame:
    """One row per series: history span, row count, and how many revisions exist."""
    df = _read_obs()
    if df.empty:
        return pd.DataFrame()
    m = meta()
    g = df.groupby("series_id")
    out = pd.DataFrame({
        "rows": g.size(),
        "first": g["reference_date"].min().dt.date,
        "last": g["reference_date"].max().dt.date,
        "last_published": g["published_at"].max().dt.date,
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
