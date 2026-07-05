"""USDA agricultural fundamentals data layer — free public USDA sources; the
fundamentals counterpart to cotdata.py (which already covers ag *positioning*).

Two parts, same compute-once / cache-to-parquet pattern as cotdata.py:

  1. USDA report CALENDAR  (no key, always on)
     The verified 2026 USDA release schedule (WASDE, Crop Production, Grain
     Stocks, Prospective Plantings, Acreage, Cattle on Feed, Hogs & Pigs). Drives
     "event risk" flags so the desk sees e.g. "Corn — WASDE in 2 days" *before*
     the print. Verified 2026-06-14 vs the NASS Agricultural Statistics Board
     calendar and USDA OCE (WASDE).

  2. USDA NASS QuickStats LIVE PULL  (optional — free key)
     National grain-stocks history per crop -> a tightness percentile vs the
     crop's own history. Set the free key once and it turns on automatically:
         setx NASS_API_KEY <your-key>     (register at quickstats.nass.usda.gov/api)
     Without a key, part 1 still works; the stocks flags simply stay dormant.

The ag_fundamentals strategy reads these caches and maps them onto the shared
strategy schema (src/strategies/base.py).
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SIGNALS = DATA / "signals"
CAL_FILE = SIGNALS / "ag_calendar.parquet"
NASS_FILE = SIGNALS / "ag_nass.parquet"

# ---------------------------------------------------------------------------
# 1) USDA 2026 release calendar (verified). Monthly Cattle on Feed entries are
#    the 3rd/4th-Friday releases; WASDE/NASS landmarks are exact.
# ---------------------------------------------------------------------------
REPORTS_2026 = [
    ("2026-01-12", "WASDE", "crops"),
    ("2026-01-12", "Crop Production (Annual)", "grains"),
    ("2026-01-12", "Grain Stocks", "grains"),
    ("2026-01-23", "Cattle on Feed", "cattle"),
    ("2026-02-10", "WASDE", "crops"),
    ("2026-02-20", "Cattle on Feed", "cattle"),
    ("2026-03-10", "WASDE", "crops"),
    ("2026-03-20", "Cattle on Feed", "cattle"),
    ("2026-03-26", "Hogs & Pigs", "hogs"),
    ("2026-03-31", "Prospective Plantings", "grains"),
    ("2026-03-31", "Grain Stocks", "grains"),
    ("2026-04-09", "WASDE", "crops"),
    ("2026-04-17", "Cattle on Feed", "cattle"),
    ("2026-05-12", "WASDE", "crops"),
    ("2026-05-22", "Cattle on Feed", "cattle"),
    ("2026-06-11", "WASDE", "crops"),
    ("2026-06-18", "Cattle on Feed", "cattle"),
    ("2026-06-25", "Hogs & Pigs", "hogs"),
    ("2026-06-30", "Acreage", "grains"),
    ("2026-06-30", "Grain Stocks", "grains"),
    ("2026-07-10", "WASDE", "crops"),
    ("2026-07-24", "Cattle on Feed", "cattle"),
    ("2026-08-12", "WASDE", "crops"),
    ("2026-08-21", "Cattle on Feed", "cattle"),
    ("2026-09-11", "WASDE", "crops"),
    ("2026-09-18", "Cattle on Feed", "cattle"),
    ("2026-09-24", "Hogs & Pigs", "hogs"),
    ("2026-09-30", "Grain Stocks", "grains"),
    ("2026-10-09", "WASDE", "crops"),
    ("2026-10-23", "Cattle on Feed", "cattle"),
    ("2026-11-10", "WASDE", "crops"),
    ("2026-11-20", "Cattle on Feed", "cattle"),
    ("2026-12-10", "WASDE", "crops"),
    ("2026-12-18", "Cattle on Feed", "cattle"),
    ("2026-12-23", "Hogs & Pigs", "hogs"),
]

# Which report scopes matter for each ag / soft / livestock ticker.
# (Coffee/cocoa/OJ have no major USDA domestic S&D report -> no event flag.)
SCOPES = {
    "C A Comdty": {"crops", "grains"}, "S A Comdty": {"crops", "grains"},
    "W A Comdty": {"crops", "grains"}, "KWA Comdty": {"crops", "grains"},
    "SMA Comdty": {"crops", "grains"}, "BOA Comdty": {"crops", "grains"},
    "RRA Comdty": {"crops", "grains"},
    "CTA Comdty": {"crops"}, "SBA Comdty": {"crops"},
    "LCA Comdty": {"crops", "cattle"}, "FCA Comdty": {"crops", "cattle"},
    "LHA Comdty": {"crops", "hogs"},
}


def report_calendar() -> pd.DataFrame:
    df = pd.DataFrame(REPORTS_2026, columns=["date", "report", "scope"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) USDA NASS QuickStats — free key in env var NASS_API_KEY
# ---------------------------------------------------------------------------
NASS_KEY = os.getenv("NASS_API_KEY", "").strip()
NASS_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# ticker -> NASS commodity_desc (national series)
NASS_COMMODITY = {
    "C A Comdty": "CORN", "S A Comdty": "SOYBEANS",
    "W A Comdty": "WHEAT", "KWA Comdty": "WHEAT",
}


def _nass_get(params: dict):
    q = dict(params)
    q["key"] = NASS_KEY
    q["format"] = "JSON"
    url = NASS_URL + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "strategy-monitor-ag"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r).get("data", [])
        except Exception as e:   # NASS can be slow / transient — retry a couple of times
            last = e
    raise last


def fetch_nass_stocks(commodity: str) -> pd.DataFrame:
    """Annual Dec-1 national TOTAL grain stocks (BU). Empty on any issue/no key.
    Pinning reference_period_desc to a single period keeps the payload tiny (one
    record per year) so the call is fast and unambiguous."""
    cols = ["year", "value"]
    if not NASS_KEY:
        return pd.DataFrame(columns=cols)
    try:
        rows = _nass_get({
            "commodity_desc": commodity,
            "statisticcat_desc": "STOCKS",
            "agg_level_desc": "NATIONAL",
            "domain_desc": "TOTAL",
            "reference_period_desc": "FIRST OF DEC",
            "unit_desc": "BU",
            "source_desc": "SURVEY",
            "year__GE": "2005",
        })
    except Exception:
        return pd.DataFrame(columns=cols)
    recs = []
    for d in rows:
        val = str(d.get("Value", "")).replace(",", "").strip()
        try:
            recs.append({"year": int(d.get("year")), "value": float(val)})
        except (TypeError, ValueError):
            continue
    if not recs:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(recs).groupby("year", as_index=False)["value"].max()
    return df.sort_values("year").reset_index(drop=True)


STOCKS_PERIODS = {   # NASS reference_period -> (key, month, day) for the quarterly Grain Stocks
    "FIRST OF MAR": ("MAR", 3, 1), "FIRST OF JUN": ("JUN", 6, 1),
    "FIRST OF SEP": ("SEP", 9, 1), "FIRST OF DEC": ("DEC", 12, 1),
}


def latest_stocks_period(commodity: str = "CORN"):
    """The most recent quarterly Grain Stocks reading available in NASS (i.e. the latest
    release). Returns {'key','year','asof'} (asof = pandas Timestamp) or None. Needs NASS_API_KEY."""
    if not NASS_KEY:
        return None
    try:
        rows = _nass_get({
            "commodity_desc": commodity, "statisticcat_desc": "STOCKS", "agg_level_desc": "NATIONAL",
            "unit_desc": "BU", "domain_desc": "TOTAL", "year__GE": str(pd.Timestamp.now().year - 1)})
    except Exception:
        return None
    best = None
    for d in rows:
        rp = (d.get("reference_period_desc") or "").upper()
        if rp not in STOCKS_PERIODS:
            continue
        try:
            yr = int(d.get("year"))
        except (TypeError, ValueError):
            continue
        key, mm, dd = STOCKS_PERIODS[rp]
        asof = pd.Timestamp(year=yr, month=mm, day=dd)
        if best is None or asof > best["asof"]:
            best = {"key": key, "year": yr, "asof": asof}
    return best


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_h = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600.0
    return age_h <= max_age_hours


# ---------------------------------------------------------------------------
# 3) USDA FAS PS&D balance sheets — FREE bulk CSV, no key. The real S&D layer:
#    stocks-to-use per commodity. US balance sheet for the US-driven crops;
#    world (sum of countries) for the globally-set softs (sugar, coffee).
#    StU(US)    = Ending Stocks / (Total Distribution - Ending Stocks)   [= dom use + exports]
#    StU(World) = sum(Ending Stocks) / sum(consumption attribute)
# ---------------------------------------------------------------------------
PSD_BASE = "https://apps.fas.usda.gov/psdonline/downloads/"
PSD_GROUP_FILE = {
    "grains_pulses": "psd_grains_pulses_csv.zip",
    "oilseeds": "psd_oilseeds_csv.zip",
    "cotton": "psd_cotton_csv.zip",
    "sugar": "psd_sugar_csv.zip",
    "coffee": "psd_coffee_csv.zip",
}
PSD_FILE = SIGNALS / "ag_psd.parquet"
# ticker -> (group, commodity, scope, world_use_attr)
PSD_SPEC = {
    "C A Comdty": ("grains_pulses", "Corn", "US", None),
    "W A Comdty": ("grains_pulses", "Wheat", "US", None),
    "KWA Comdty": ("grains_pulses", "Wheat", "US", None),
    "RRA Comdty": ("grains_pulses", "Rice, Milled", "US", None),
    "S A Comdty": ("oilseeds", "Oilseed, Soybean", "US", None),
    "SMA Comdty": ("oilseeds", "Meal, Soybean", "US", None),
    "BOA Comdty": ("oilseeds", "Oil, Soybean", "US", None),
    "CTA Comdty": ("cotton", "Cotton", "US", None),
    "SBA Comdty": ("sugar", "Sugar, Centrifugal", "World", "Total Disappearance"),
    "KCA Comdty": ("coffee", "Coffee, Green", "World", "Domestic Consumption"),
}


def _psd_download(group: str) -> pd.DataFrame:
    url = PSD_BASE + PSD_GROUP_FILE[group]
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "strategy-monitor-ag"}), timeout=180).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csvname = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(zf.open(csvname))
    for c in ("Commodity_Description", "Country_Name", "Attribute_Description"):
        df[c] = df[c].astype(str).str.strip()
    return df


def _stu_series(df: pd.DataFrame, commodity: str, scope: str, world_use_attr) -> pd.DataFrame:
    cols = ["market_year", "stu", "ending_stocks", "use"]
    d = df[df["Commodity_Description"] == commodity]
    if scope == "US":
        d = d[d["Country_Name"] == "United States"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    piv = d.pivot_table(index="Market_Year", columns="Attribute_Description", values="Value", aggfunc="sum")
    if "Ending Stocks" not in piv.columns:
        return pd.DataFrame(columns=cols)
    es = pd.to_numeric(piv["Ending Stocks"], errors="coerce")
    if scope == "US":
        if "Total Distribution" not in piv.columns:
            return pd.DataFrame(columns=cols)
        use = pd.to_numeric(piv["Total Distribution"], errors="coerce") - es
    else:
        if world_use_attr not in piv.columns:
            return pd.DataFrame(columns=cols)
        use = pd.to_numeric(piv[world_use_attr], errors="coerce")
    stu = es / use.where(use > 0) * 100.0
    out = pd.DataFrame({"market_year": piv.index.astype(int), "stu": stu.values,
                        "ending_stocks": es.values, "use": use.values})
    return out.dropna(subset=["stu"]).reset_index(drop=True)


def compute_psd(force: bool = False, max_age_hours: float = 20.0) -> bool:
    """Download the PS&D bulk CSVs (free, no key) and cache stocks-to-use per
    commodity to ag_psd.parquet. Reuses a fresh cache to stay fast."""
    SIGNALS.mkdir(parents=True, exist_ok=True)
    if not force and _is_fresh(PSD_FILE, max_age_hours):
        return True
    cache: dict = {}
    frames = []
    for tkr, (grp, comm, scope, wattr) in PSD_SPEC.items():
        try:
            if grp not in cache:
                cache[grp] = _psd_download(grp)
            s = _stu_series(cache[grp], comm, scope, wattr)
        except Exception:
            continue
        if not s.empty:
            frames.append(s.assign(ticker=tkr, commodity=comm, scope=scope))
    if frames:
        try:
            pd.concat(frames, ignore_index=True).to_parquet(PSD_FILE, index=False)
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# 4) USDA FAS Export Sales (ESR) — weekly export-demand pace. Needs a FAS Open
#    Data key in FAS_API_KEY (api.fas.usda.gov, X-Api-Key header). We track total
#    export COMMITMENTS for the current marketing year vs the prior MY at the same
#    week-of-year — the standard "is demand running ahead or behind?" read.
# ---------------------------------------------------------------------------
FAS_KEY = os.getenv("FAS_API_KEY", "").strip()
FAS_BASE = "https://api.fas.usda.gov/api/"
ESR_FILE = SIGNALS / "ag_esr.parquet"
# ticker -> (ESR commodity code, label)
ESR_SPEC = {
    "C A Comdty": (401, "Corn"),
    "S A Comdty": (801, "Soybeans"),
    "W A Comdty": (107, "All Wheat"),
    "KWA Comdty": (107, "All Wheat"),
    "SMA Comdty": (901, "Soybean meal"),
    "BOA Comdty": (902, "Soybean oil"),
    "CTA Comdty": (1402, "Upland cotton"),
}


def _fas_get(path: str):
    req = urllib.request.Request(FAS_BASE + path, headers={
        "X-Api-Key": FAS_KEY, "Accept": "application/json", "User-Agent": "strategy-monitor-ag"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _esr_weekly(code: int, my: int):
    """National weekly cumulative total commitments (summed across countries)."""
    try:
        data = _fas_get(f"esr/exports/commodityCode/{code}/allCountries/marketYear/{my}")
    except Exception:
        return None
    if not data:
        return None
    df = pd.DataFrame(data)
    df["wk"] = pd.to_datetime(df["weekEndingDate"])
    df["commit"] = pd.to_numeric(df["currentMYTotalCommitment"], errors="coerce")
    return df.groupby("wk")["commit"].sum().sort_index()


def _esr_pace(code: int, yr: int):
    cands = {}
    for my in (yr - 1, yr, yr + 1):
        s = _esr_weekly(code, my)
        if s is not None and len(s):
            cands[my] = s
    if not cands:
        return None
    cur_my = max(cands, key=lambda m: cands[m].index[-1])      # MY with the most recent week
    if cur_my - 1 not in cands:
        s = _esr_weekly(code, cur_my - 1)
        if s is not None and len(s):
            cands[cur_my - 1] = s
    cs = cands[cur_my]
    ps = cands.get(cur_my - 1)
    k = len(cs)
    commit = float(cs.iloc[-1])
    prior = float(ps.iloc[min(k, len(ps)) - 1]) if (ps is not None and len(ps)) else float("nan")
    yoy = (commit / prior - 1.0) * 100.0 if prior and prior == prior else float("nan")
    return {"market_year": int(cur_my), "week": cs.index[-1], "commit": commit, "prior": prior, "yoy": yoy}


def compute_esr(force: bool = False, max_age_hours: float = 20.0) -> bool:
    """Cache weekly export-commitment pace per commodity to ag_esr.parquet (FAS key)."""
    if not FAS_KEY:
        return False
    SIGNALS.mkdir(parents=True, exist_ok=True)
    if not force and _is_fresh(ESR_FILE, max_age_hours):
        return True
    yr = pd.Timestamp.now().year
    by_code: dict = {}
    rows = []
    for tkr, (code, label) in ESR_SPEC.items():
        if code not in by_code:
            by_code[code] = _esr_pace(code, yr)
        p = by_code[code]
        if p is None:
            continue
        rows.append({"ticker": tkr, "label": label, **p})
    if rows:
        try:
            pd.DataFrame(rows).to_parquet(ESR_FILE, index=False)
            return True
        except Exception:
            return False
    return False


def compute(max_age_hours: float = 20.0, force: bool = False) -> dict:
    """Refresh the calendar cache (always, local) and the NASS series (if a key is
    set and the cache is stale). Returns a status dict for the UI."""
    SIGNALS.mkdir(parents=True, exist_ok=True)
    status = {"calendar": False, "psd": False, "psd_crops": 0, "esr": False, "esr_n": 0,
              "fas_key": bool(FAS_KEY), "nass": False, "nass_key": bool(NASS_KEY), "nass_crops": 0}

    try:
        report_calendar().to_parquet(CAL_FILE, index=False)
        status["calendar"] = True
    except Exception:
        pass

    try:
        compute_psd(force=force, max_age_hours=max_age_hours)
    except Exception:
        pass
    if PSD_FILE.exists():
        try:
            status["psd"] = True
            status["psd_crops"] = int(pd.read_parquet(PSD_FILE)["ticker"].nunique())
        except Exception:
            pass

    try:
        compute_esr(force=force, max_age_hours=max_age_hours)
    except Exception:
        pass
    if ESR_FILE.exists():
        try:
            status["esr"] = True
            status["esr_n"] = int(pd.read_parquet(ESR_FILE)["ticker"].nunique())
        except Exception:
            pass

    if NASS_KEY and (force or not _is_fresh(NASS_FILE, max_age_hours)):
        frames = []
        for tkr, comm in NASS_COMMODITY.items():
            s = fetch_nass_stocks(comm)
            if not s.empty:
                frames.append(s.assign(ticker=tkr, commodity=comm))
        if frames:
            try:
                pd.concat(frames, ignore_index=True).to_parquet(NASS_FILE, index=False)
            except Exception:
                pass
    if NASS_FILE.exists():
        try:
            status["nass"] = True
            status["nass_crops"] = int(pd.read_parquet(NASS_FILE)["ticker"].nunique())
        except Exception:
            pass
    return status
