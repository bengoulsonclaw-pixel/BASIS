"""metals.py — the precious-metals universe, on the same point-in-time store as gold.

The Gold Signal Engine's store is metal-agnostic: goldstore keys everything by
`series_id`, so nothing about it is gold-specific except the module name and which
series happen to be in it. This module extends the universe to silver, platinum and
palladium rather than cloning the machinery.

WHAT THE BENCHMARKS LOOK LIKE, AND WHY IT MATTERS

  metal      LBMA history   fixes/day
  gold       1968-04 ->     AM + PM
  silver     1968-01 ->     ONE (noon)
  platinum   1990-04 ->     AM + PM
  palladium  1990-04 ->     AM + PM

The gold event study found its only surviving effect in the AM->PM window — a
4.5-hour interval that brackets the 08:30 ET US releases — and found nothing over
the full 24 hours. Platinum and palladium have that window. **Silver does not.** The
LBMA Silver Price is a single noon auction, so for silver there is no intraday fix
window at all and the release study can only ever run fix-to-fix, which is precisely
the measurement that showed nothing for gold. That is a property of the benchmark,
not a modelling choice, and any silver release result has to be read knowing the
one window where an effect was detectable is unavailable.

WHY THE PGMs ARE A DIFFERENT PROBLEM FROM GOLD

Gold holds roughly sixty years of above-ground stock against annual mine supply, so
supply disruptions cannot move it and the macro drivers are all priced before they
print — which is what the gold work found. Platinum's above-ground stock is on the
order of one to two years of demand, and ~70% of mine supply is South African;
palladium is ~40% Russian with ~80% of demand in gasoline autocatalysts. Those are
thin buffers against concentrated, datable supply events. The gold negative result
should NOT be assumed to transfer.

CLI:
    python src/metals.py --lbma        LBMA benchmarks for all four metals
    python src/metals.py --cot         CFTC positioning for Ag/Pt/Pd
    python src/metals.py --all
    python src/metals.py --status
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldstore                                            # noqa: E402

LBMA_JSON = "https://prices.lbma.org.uk/json/{name}.json"

# metal -> (LBMA endpoint stem per fix, deep-store ticker, CFTC code)
#
# `fixes` maps our series suffix to the LBMA endpoint. Silver has a single entry
# because it has a single auction; giving it a synthetic "AM" and "PM" from the same
# print would manufacture a zero-width window that later code would treat as real.
METALS = {
    "GOLD": {
        "fixes": {"AM": "gold_am", "PM": "gold_pm"},
        "ticker": "GCA Comdty", "cftc": "088691", "unit": "USD/oz",
    },
    "SILVER": {
        "fixes": {"FIX": "silver"},
        "ticker": "SIA Comdty", "cftc": "084691", "unit": "USD/oz",
    },
    "PLATINUM": {
        "fixes": {"AM": "platinum_am", "PM": "platinum_pm"},
        "ticker": "PLA Comdty", "cftc": "076651", "unit": "USD/oz",
    },
    "PALLADIUM": {
        "fixes": {"AM": "palladium_am", "PM": "palladium_pm"},
        "ticker": "PAA Comdty", "cftc": "075651", "unit": "USD/oz",
    },
}

# The benchmark each metal's returns are struck on — the last fix of the London day.
BENCHMARK = {
    "GOLD": "LBMA_GOLD_PM_USD",
    "SILVER": "LBMA_SILVER_USD",
    "PLATINUM": "LBMA_PLATINUM_PM_USD",
    "PALLADIUM": "LBMA_PALLADIUM_PM_USD",
}

# Metals whose benchmark carries a morning fix as well, so an intraday release
# window exists. DERIVED, not hand-listed: a hand-written set drifts from the
# table above the moment an endpoint changes, and this one already had a
# duplicate entry where silver's absence was the whole point.
HAS_INTRADAY_WINDOW = {m for m, s in METALS.items() if len(s["fixes"]) > 1}

# LBMA publishes [USD, GBP, EUR] per date.
_CCY = {"USD": 0, "GBP": 1, "EUR": 2}


def _fetch_lbma(name: str) -> pd.DataFrame:
    """One LBMA endpoint as a frame indexed by fix date, columns USD/GBP/EUR."""
    req = urllib.request.Request(LBMA_JSON.format(name=name),
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
    rows, idx = [], []
    for r in raw:
        v = r.get("v") or []
        if not v:
            continue
        idx.append(pd.Timestamp(r["d"]))
        rows.append([pd.to_numeric(v[i], errors="coerce") if i < len(v) else None
                     for i in _CCY.values()])
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=list(_CCY))
    # LBMA carries zeros and negatives for non-fixing days on some endpoints; those
    # are absences, not prices, and a zero would wreck a log return.
    return df[df["USD"] > 0].sort_index()


def ingest_lbma(metals=None, start: str = "1990-01-01") -> int:
    """Land every metal's London benchmark in the point-in-time store.

    A fix is FINAL when printed and never restated, so published_at is the fix
    moment itself — lag 0, exactly as the gold PM fix is stamped. That is what makes
    these the only series in the store safe to use as a same-day target.
    """
    wrote = 0
    for metal in (metals or METALS):
        spec = METALS[metal]
        for suffix, endpoint in spec["fixes"].items():
            sid = (f"LBMA_{metal}_USD" if suffix == "FIX"
                   else f"LBMA_{metal}_{suffix}_USD")
            try:
                df = _fetch_lbma(endpoint)
            except Exception as e:
                print(f"  {sid:26s} FETCH FAILED: {type(e).__name__}: {e}")
                continue
            s = df["USD"]
            s = s[s.index >= pd.Timestamp(start)]
            if s.empty:
                print(f"  {sid:26s} no data after {start}")
                continue
            label = "fix" if suffix == "FIX" else f"{suffix} fix"
            goldstore.register(
                sid, description=f"LBMA {metal.title()} Price {label}",
                unit=spec["unit"], native_freq="daily", typical_lag_days=0,
                bucket="valuation", source_url="https://prices.lbma.org.uk",
                published_at_approximated=False)
            n = goldstore.put(sid, s, source="LBMA", typical_lag_days=0)
            wrote += n
            print(f"  {sid:26s} +{n:6d} rows   "
                  f"{s.index.min().date()} -> {s.index.max().date()}")
    return wrote


def ingest_cot(metals=None) -> int:
    """CFTC disaggregated positioning for the non-gold metals.

    Reuses goldingest.ingest_cot, which already carries the release-timing fix that
    matters here: the COT prints Friday 15:30 ET, and a federal holiday in the
    reference week slips it by one business day. Getting that wrong hands a model
    three days of hindsight on the most-watched positioning series there is.
    """
    from src import goldingest
    wrote = 0
    for metal in (metals or [m for m in METALS if m != "GOLD"]):
        tkr = METALS[metal]["ticker"]
        try:
            wrote += goldingest.ingest_cot(ticker=tkr)
        except Exception as e:
            print(f"  COT {metal:12s} FAILED: {type(e).__name__}: {e}")
    return wrote


def benchmark_series(metal: str, as_of=None) -> pd.Series:
    """The metal's benchmark price as it stood on `as_of`."""
    return goldstore.get_series(BENCHMARK[metal], as_of=as_of)


def panel(metals=None, start: str = "1990-01-01") -> pd.DataFrame:
    """Benchmark prices for several metals on one business-day index."""
    metals = list(metals or METALS)
    return goldstore.daily_panel([BENCHMARK[m] for m in metals], start=start) \
        .rename(columns={BENCHMARK[m]: m for m in metals})


def status() -> pd.DataFrame:
    """What the store holds per metal — the honest inventory."""
    cov = goldstore.coverage()
    rows = []
    for metal, spec in METALS.items():
        for suffix in spec["fixes"]:
            sid = (f"LBMA_{metal}_USD" if suffix == "FIX"
                   else f"LBMA_{metal}_{suffix}_USD")
            if sid in cov.index:
                r = cov.loc[sid]
                rows.append({"metal": metal, "series": sid, "rows": int(r["rows"]),
                             "first": r["first"], "last": r["last"]})
            else:
                rows.append({"metal": metal, "series": sid, "rows": 0,
                             "first": None, "last": None})
    out = pd.DataFrame(rows)
    out["intraday_window"] = out["metal"].map(
        lambda m: len(METALS[m]["fixes"]) > 1)
    return out


def main() -> int:
    args = set(sys.argv[1:]) or {"--status"}
    if args & {"--lbma", "--all"}:
        print("LBMA benchmarks:")
        ingest_lbma()
    if args & {"--cot", "--all"}:
        print("CFTC positioning:")
        ingest_cot()
    if args & {"--status", "--all"}:
        print("\nStore inventory:")
        print(status().to_string(index=False))
        missing = [m for m in METALS if len(METALS[m]["fixes"]) == 1]
        if missing:
            print(f"\nNo intraday fix window: {', '.join(missing)} — the LBMA "
                  f"benchmark is a single auction, so release studies for these "
                  f"metals can only run fix-to-fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
