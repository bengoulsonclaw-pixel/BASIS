"""pmdata.py — data layer for the 🥇 Precious Metals Fundamentals Monitor.

Assembles everything the monthly client PDF needs into one JSON-safe dict:

  prices        datafeed.get_history (mock / snapshot / bloomberg seam)     LIVE*
  cot           CFTC managed-money via cotdata.fetch_cot / cot_index        LIVE
  fred          10y TIPS real yield (DFII10) + broad dollar (DTWEXBGS)      LIVE
  etf           total known ETF holdings per metal                          MOCK →
  comex         COMEX/NYMEX depository stocks (registered / eligible)       MOCK →
  swiss         Swiss customs gold exports by destination                   MOCK →
  central_banks monthly net central-bank gold purchases                     MOCK →
  premiums      SGE and India local premium/discount vs London              MOCK →
  mint          US Mint bullion coin sales                                  MOCK →
  pgm           WPIC platinum / palladium annual balances + auto output     MOCK →

Every block is a single fetch function returning {"live": bool, ...}; the MOCK →
ones are deterministic synthetic generators behind the same seam the live pull
will use, so wiring a source in later never touches the report engine. build()
lists every non-live block in d["mock_blocks"] and the PDF shows a DRAFT banner
until that list is empty — mock data can never reach a client silently.

* prices are only as live as DATAFEED_MODE; under "mock" the prices block is
  reported as mock too.

CLI:  python src/pmdata.py out.json [--force]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
from src import cotdata, datafeed  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SIGNALS = DATA / "signals"
SIGNALS.mkdir(parents=True, exist_ok=True)

PM_COT_FILE = SIGNALS / "pm_cot.parquet"
PM_FRED_FILE = SIGNALS / "pm_fred.parquet"
PM_COMEX_FILE = SIGNALS / "pm_comex.parquet"   # written by src/pm_fetch.py (real Chrome)
PM_MINT_FILE = SIGNALS / "pm_mint.parquet"     # written by src/pm_fetch.py (real Chrome)
PM_PGM_FILE = DATA / "pm_pgm.json"             # seeded from each WPIC / JM release

METALS = [
    {"key": "gold",      "name": "Gold",      "ticker": "GCA Comdty", "unit": "$/oz"},
    {"key": "silver",    "name": "Silver",    "ticker": "SIA Comdty", "unit": "$/oz"},
    {"key": "platinum",  "name": "Platinum",  "ticker": "PLA Comdty", "unit": "$/oz"},
    {"key": "palladium", "name": "Palladium", "ticker": "PAA Comdty", "unit": "$/oz"},
]

FRED_SERIES = {"real_yield": "DFII10", "dollar": "DTWEXBGS"}
# fred.stlouisfed.org bot-blocks plain clients; api.stlouisfed.org does not but
# needs a (free) key — data/fred_key.txt or $FRED_API_KEY, same shape as eia.py.
FRED_API = ("https://api.stlouisfed.org/fred/series/observations"
            "?series_id={sid}&api_key={key}&file_type=json")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _fred_key() -> str | None:
    key = (os.getenv("FRED_API_KEY") or "").strip()
    if not key and (DATA / "fred_key.txt").exists():
        key = (DATA / "fred_key.txt").read_text(encoding="utf-8").strip()
    return key or None


def _fred_series(sid: str) -> pd.Series:
    key = _fred_key()
    if key:
        req = urllib.request.Request(FRED_API.format(sid=sid, key=key),
                                     headers={"User-Agent": "basis-pm-monitor"})
        with urllib.request.urlopen(req, timeout=60) as r:
            obs = json.load(r)["observations"]
        s = pd.Series({o["date"]: o["value"] for o in obs})
        s.index = pd.to_datetime(s.index)
        return pd.to_numeric(s, errors="coerce")
    req = urllib.request.Request(FRED_CSV.format(sid=sid),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = pd.read_csv(io.BytesIO(r.read()))
    raw.columns = ["date", "value"]  # header name varies (DATE/observation_date)
    return pd.Series(pd.to_numeric(raw["value"], errors="coerce").values,
                     index=pd.to_datetime(raw["date"]))


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _ser(s: pd.Series, nd: int = 2) -> dict:
    """JSON-safe {dates, values} for a date-indexed series (NaN → None)."""
    s = s.dropna()
    return {"d": [d.strftime("%Y-%m-%d") for d in s.index],
            "v": [round(float(v), nd) for v in s.values]}


def _pct(series: pd.Series, days: int) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    ref = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    if ref.empty or ref.iloc[-1] == 0:
        return None
    return float(s.iloc[-1] / ref.iloc[-1] - 1.0) * 100.0


# ---------------------------------------------------------------------------
# LIVE blocks
# ---------------------------------------------------------------------------
def fetch_prices(years: int = 5) -> dict:
    """Daily settle history for the four metals through the datafeed seam."""
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    # raw=True: this feeds a CLIENT document — historical gold/silver levels must match
    # the published prints, not a roll-adjusted continuation (which lags spot by roughly
    # the contango: ~8-10% two years back on gold).
    px = datafeed.get_history([m["ticker"] for m in METALS], start=start, raw=True)
    return {"live": datafeed.MODE != "mock", "px": px}


def fetch_cot(force: bool = False, max_age_hours: float = 20.0) -> dict:
    """Weekly managed-money net + COT index for the four metals (CFTC Socrata,
    free). Cached to pm_cot.parquet so repeat builds don't re-hit the API."""
    if not force and _is_fresh(PM_COT_FILE, max_age_hours):
        return {"live": True, "hist": pd.read_parquet(PM_COT_FILE)}
    frames = []
    try:
        for m in METALS:
            df = cotdata.fetch_cot(m["ticker"])
            df["key"] = m["key"]
            df["cot_idx"] = cotdata.cot_index(df["net"])
            frames.append(df)
        hist = pd.concat(frames, ignore_index=True)
        hist.to_parquet(PM_COT_FILE, index=False)
        return {"live": True, "hist": hist}
    except Exception:
        if PM_COT_FILE.exists():  # stale cache beats synthetic
            return {"live": True, "hist": pd.read_parquet(PM_COT_FILE)}
        return {"live": False, "hist": _mock_cot()}


def fetch_fred(force: bool = False, max_age_hours: float = 12.0) -> dict:
    """10y TIPS real yield + broad dollar via FRED's keyless fredgraph CSV."""
    if not force and _is_fresh(PM_FRED_FILE, max_age_hours):
        return {"live": True, "df": pd.read_parquet(PM_FRED_FILE)}
    try:
        df = pd.DataFrame({name: _fred_series(sid)
                           for name, sid in FRED_SERIES.items()}).dropna(how="all")
        df = df[df.index >= df.index.max() - pd.DateOffset(years=6)]
        df.to_parquet(PM_FRED_FILE)
        return {"live": True, "df": df}
    except Exception:
        if PM_FRED_FILE.exists():
            return {"live": True, "df": pd.read_parquet(PM_FRED_FILE)}
        return {"live": False, "df": _mock_fred()}


# ---------------------------------------------------------------------------
# MOCK blocks — deterministic placeholders behind the seams the live pulls
# will use (ETF/COMEX → Bloomberg fields; Swiss/CB/premiums/mint/PGM → free
# web sources per the source inventory). Scales are realistic so the layout
# proofs true, but the DRAFT banner flags them until wired.
# ---------------------------------------------------------------------------
def _rng(tag: str) -> np.random.Generator:
    # crc32, not hash(): hash() is salted per process, which would re-roll the
    # placeholder data on every build.
    return np.random.default_rng(zlib.crc32(("pm-" + tag).encode()))


def _month_index(n: int) -> pd.DatetimeIndex:
    end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
    return pd.date_range(end=end, periods=n, freq="ME")


def _mock_walk(tag: str, n: int, level: float, drift: float, vol: float) -> np.ndarray:
    g = _rng(tag)
    return level * np.cumprod(1 + g.normal(drift, vol, n))


def _mock_cot() -> pd.DataFrame:
    frames = []
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="W-TUE")
    for m in METALS:
        g = _rng("cot-" + m["key"])
        net = pd.Series(np.cumsum(g.normal(0, 6000, len(dates))) + 80000, index=dates)
        df = pd.DataFrame({"date": dates, "net": net.values,
                           "oi": np.abs(net.values) * 3 + 200000})
        df["long"] = df["net"].clip(lower=0) + 50000
        df["short"] = df["long"] - df["net"]
        df["key"] = m["key"]
        df["cot_idx"] = cotdata.cot_index(df["net"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _mock_fred() -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=1300, freq="B")
    g = _rng("fred")
    return pd.DataFrame({
        "real_yield": np.clip(np.cumsum(g.normal(0, 0.02, len(idx))) + 1.8, -0.5, 3.0),
        "dollar": _mock_walk("dxy", len(idx), 121.0, 0.00002, 0.003),
    }, index=idx)


PM_ETF_FILE = SIGNALS / "pm_etf.parquet"
_OZ_PER_TONNE = 32150.7466
_ETF_SCALE = {"gold": (_OZ_PER_TONNE, "t"), "silver": (1e6, "Moz"),
              "platinum": (1e3, "koz"), "palladium": (1e3, "koz")}


def fetch_etf() -> dict:
    """Total known ETF holdings from the pm_bbg Bloomberg cache (Bloomberg
    aggregate indices, troy oz → t / Moz / koz), monthly last. Gold + silver
    are wired; Pt/Pd wait on their aggregate tickers (pm_bbg.TICKERS_ETF) and
    simply show '—' until then. Mock fallback while no cache exists."""
    if PM_ETF_FILE.exists():
        raw = pd.read_parquet(PM_ETF_FILE)
        raw["date"] = pd.to_datetime(raw["date"])
        out = {}
        for k, (div, unit) in _ETF_SCALE.items():
            m = raw[raw["metal"] == k]
            if m.empty:
                continue
            s = (m.set_index("date")["oz"] / div).sort_index()
            out[k] = {"series": s.resample("ME").last().dropna(), "unit": unit}
        if out:
            return {"live": True, "holdings": out}
    idx = _month_index(36)
    levels = {"gold": (3250.0, "t"), "silver": (720.0, "Moz"),
              "platinum": (3100.0, "koz"), "palladium": (550.0, "koz")}
    out = {}
    for k, (lvl, unit) in levels.items():
        out[k] = {"series": pd.Series(_mock_walk("etf-" + k, 36, lvl, 0.001, 0.012), index=idx),
                  "unit": unit}
    return {"live": False, "holdings": out}


_COMEX_SCALE = {"gold": (1e6, "Moz"), "silver": (1e6, "Moz"),
                "platinum": (1e3, "koz"), "palladium": (1e3, "koz")}


def fetch_comex() -> dict:
    """COMEX/NYMEX depository stocks from the pm_fetch archive (CME publishes a
    same-URL daily snapshot only, so history accumulates from first fetch —
    charts stay suppressed until the archive has a few points). Falls back to
    the deterministic mock while no archive exists."""
    if PM_COMEX_FILE.exists():
        raw = pd.read_parquet(PM_COMEX_FILE)
        raw["date"] = pd.to_datetime(raw["date"])
        out = {}
        for k, (div, unit) in _COMEX_SCALE.items():
            m = raw[raw["metal"] == k].set_index("date").sort_index()
            if m.empty:
                continue
            out[k] = {"registered": m["registered_oz"] / div,
                      "eligible": m["eligible_oz"] / div, "unit": unit}
        if out:
            return {"live": True, "stocks": out}
    idx = _month_index(24)
    levels = {"gold": (20.5, 34.0, "Moz"), "silver": (168.0, 452.0, "Moz"),
              "platinum": (320.0, 610.0, "koz"), "palladium": (48.0, 96.0, "koz")}
    out = {}
    for k, (reg, tot, unit) in levels.items():
        r = pd.Series(_mock_walk("cxr-" + k, 24, reg, -0.004, 0.03), index=idx)
        e = pd.Series(_mock_walk("cxe-" + k, 24, tot - reg, 0.002, 0.02), index=idx)
        out[k] = {"registered": r, "eligible": e, "unit": unit}
    return {"live": False, "stocks": out}


PM_SWISS_FILE = SIGNALS / "pm_swiss.parquet"
_SWISS_DESTS = {"CN": "China", "IN": "India", "GB": "UK", "US": "US", "TR": "Türkiye"}


def fetch_swiss() -> dict:
    """Swiss customs gold-bar exports (7108.1200) by destination, monthly
    tonnes, from the BAZG open-data archive maintained by pm_fetch.fetch_swiss
    (~6-week publication lag). Mock fallback while no archive exists."""
    if PM_SWISS_FILE.exists():
        raw = pd.read_parquet(PM_SWISS_FILE)
        raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1)) \
            + pd.offsets.MonthEnd(0)
        piv = raw.pivot_table(index="date", columns="iso2", values="kg", aggfunc="sum") / 1000.0
        piv = piv.sort_index().iloc[-13:]
        flows = pd.DataFrame({label: piv.get(iso, pd.Series(0.0, index=piv.index))
                              for iso, label in _SWISS_DESTS.items()}).fillna(0.0)
        others = [c for c in piv.columns if c not in _SWISS_DESTS]
        flows["Other"] = piv[others].sum(axis=1).fillna(0.0)
        return {"live": True, "exports": flows}
    idx = _month_index(13)
    dests = {"China": 38.0, "India": 30.0, "UK": 22.0, "US": 16.0, "Türkiye": 9.0, "Other": 32.0}
    g = _rng("swiss")
    flows = pd.DataFrame({d: np.clip(g.normal(mu, mu * 0.45, len(idx)), 0, None)
                          for d, mu in dests.items()}, index=idx)
    return {"live": False, "exports": flows}


PM_CB_FILE = SIGNALS / "pm_cb.parquet"
_IMF_URL = ("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IL/"
            "GX010.RGV_REVS..M?startPeriod={start}")
_OZT_TO_T = 31.1034768 / 1e6


def _from_goldstore(series_id: str):
    """A series out of the Gold Signal Engine's point-in-time store, or None.

    Preferred over this module's own pulls where the store carries the same series,
    because it holds real publication timestamps and revision history rather than a
    single latest-vintage snapshot. Returns None on any problem — a machine that has
    never run the gold ingest must keep producing this report exactly as before, so
    every caller falls back to its original path."""
    try:
        from src import goldstore
        s = goldstore.get_series(series_id)
        return s if len(s) else None
    except Exception:
        return None


def fetch_central_banks(force: bool = False, max_age_hours: float = 72.0) -> dict:
    """Reported official gold holdings, monthly net purchases = the diff.

    Reads the gold store first: it holds the same IMF series from 2010 with genuine
    publication timestamps, where this module's own call asks for a fixed FOUR-YEAR
    window — enough for a monthly chart, but it silently truncates any longer view and
    cannot say when a figure became knowable. Falls back to the original IMF pull, and
    then to the cache, so nothing here depends on the gold engine having been run.

    ~6-week lag either way, and it understates true buying (China reports partially,
    Russia not at all), so the fine print continues to cite WGC alongside."""
    # R3 (audit): prefer the store only when it is FRESH. Nothing schedules the gold
    # ingest, so preferring it on length alone meant preferring the one path that
    # never refreshes — while still reporting live: True. Reproduced by the audit: a
    # store stale by four months printed 108t of Dec-Feb buying in the client bullet
    # where the truth was 177.6t of Apr-Jun. Fall through to the self-refreshing IMF
    # pull when the store has gone quiet.
    gs = _from_goldstore("CB_GOLD_WORLD")
    if gs is not None and len(gs) > 24:
        lag_days = (pd.Timestamp.today().normalize() - gs.index.max()).days
        if lag_days <= 120:          # IMF runs ~6 weeks late; 120d allows for that
            return {"live": True, "net_purchases": gs.diff().dropna(),
                    "source": f"IMF via gold store (from {gs.index.min():%Y}, "
                              f"point-in-time)"}
    if not force and _is_fresh(PM_CB_FILE, max_age_hours):
        hold = pd.read_parquet(PM_CB_FILE)["tonnes"]
        return {"live": True, "net_purchases": hold.diff().dropna()}
    try:
        import xml.etree.ElementTree as ET
        start = (pd.Timestamp.today() - pd.DateOffset(years=4)).strftime("%Y-%m")
        req = urllib.request.Request(_IMF_URL.format(start=start),
                                     headers={"User-Agent": "basis-pm-monitor"})
        with urllib.request.urlopen(req, timeout=90) as r:
            root = ET.fromstring(r.read())
        obs = {}
        for el in root.iter():
            if el.tag.endswith("Obs"):
                p = el.attrib["TIME_PERIOD"].replace("-M", "-")
                obs[pd.Period(p, freq="M").to_timestamp(how="end").normalize()] = \
                    float(el.attrib["OBS_VALUE"]) * _OZT_TO_T
        hold = pd.Series(obs).sort_index()
        if hold.empty:
            raise RuntimeError("IMF IL returned no observations")
        pd.DataFrame({"tonnes": hold}).to_parquet(PM_CB_FILE)
        return {"live": True, "net_purchases": hold.diff().dropna()}
    except Exception:
        if PM_CB_FILE.exists():
            hold = pd.read_parquet(PM_CB_FILE)["tonnes"]
            return {"live": True, "net_purchases": hold.diff().dropna()}
        idx = _month_index(36)
        g = _rng("cb")
        buys = pd.Series(np.clip(g.normal(38, 22, len(idx)), -15, None), index=idx)
        return {"live": False, "net_purchases": buys}


PM_PREM_FILE = SIGNALS / "pm_prem.parquet"


def fetch_premiums() -> dict:
    """Shanghai physical premium/discount vs London, $/oz.

    Computed from the gold store where possible — SGE Au99.99 converted at USD/CNY
    against the LBMA AM fix. That is preferred over the Bloomberg leg for two
    reasons: the CIX (.SHGOLDOZ) resolves only under one Terminal login, so the
    figure is unavailable to anyone else and to any unattended run; and the free SGE
    feed carries history to 2016 where the CIX cache starts whenever it was first
    pulled.

    The AM fix is the right reference leg, not PM: SGE's day session closes ~08:30
    London and the AM fix is 10:30, so against the 15:00 PM fix the series carries a
    whole London session of drift that has nothing to do with Chinese demand.

    Falls back to the Bloomberg cache, then to mock, so nothing regresses."""
    try:
        from src import golddata, goldstore
        sge = _from_goldstore("SGE_AU9999")
        am = _from_goldstore("LBMA_GOLD_AM_USD")
        cny = _from_goldstore("USDCNY")
        if sge is not None and am is not None and cny is not None:
            prem = golddata.sge_premium(
                pd.DataFrame({"sge_cny_g": sge}),
                pd.DataFrame({"lbma_am_usd": am}), cny)
            if len(prem) > 52:
                return {"live": True,
                        "sge": prem.resample("W-FRI").last().dropna(),
                        "india": pd.Series(dtype=float),
                        "source": "SGE Au99.99 vs LBMA AM fix (free, 2016-)"}
    except Exception:
        pass
    if PM_PREM_FILE.exists():
        raw = pd.read_parquet(PM_PREM_FILE)
        sge = raw["premium"].resample("W-FRI").last().dropna()
        return {"live": True, "sge": sge, "india": pd.Series(dtype=float)}
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=104, freq="W-FRI")
    g = _rng("prem")
    sge = pd.Series(np.clip(np.cumsum(g.normal(0, 4, len(idx))) * 0.4 + 12, -25, 90), index=idx)
    ind = pd.Series(np.clip(np.cumsum(g.normal(0, 3, len(idx))) * 0.4 - 4, -60, 40), index=idx)
    return {"live": False, "sge": sge, "india": ind}


def fetch_mint() -> dict:
    """US Mint bullion coin sales from the pm_fetch archive (page carries the
    full-year history, so a pull rebuilds the whole series). Mock fallback."""
    if PM_MINT_FILE.exists():
        raw = pd.read_parquet(PM_MINT_FILE)
        raw["month"] = pd.to_datetime(raw["month"]) + pd.offsets.MonthEnd(0)
        g = raw[raw["metal"] == "gold"].set_index("month")["oz"].sort_index()
        s = raw[raw["metal"] == "silver"].set_index("month")["oz"].sort_index()
        if len(g) or len(s):
            return {"live": True, "gold_oz": g, "silver_oz": s}
    idx = _month_index(24)
    g = _rng("mint")
    gold = pd.Series(np.clip(g.normal(28000, 14000, len(idx)), 2000, None), index=idx)
    silver = pd.Series(np.clip(g.normal(2.4e6, 1.1e6, len(idx)), 2e5, None), index=idx)
    return {"live": False, "gold_oz": gold, "silver_oz": silver}


def fetch_pgm() -> dict:
    """PGM annual market balances (koz) from data/pm_pgm.json — hand-seeded from
    each WPIC Platinum Quarterly / JM PGM Market Report release (real published
    figures; years suffixed 'f' are the publisher's forecast). Mock fallback."""
    if PM_PGM_FILE.exists():
        try:
            seed = json.loads(PM_PGM_FILE.read_text(encoding="utf-8"))
            return {"live": True,
                    "pt_balance": seed["platinum"]["balance_koz"],
                    "pd_balance": seed["palladium"]["balance_koz"],
                    "pt_source": seed["platinum"].get("source", ""),
                    "pd_source": seed["palladium"].get("source", "")}
        except Exception:
            pass
    years = [str(y) for y in range(datetime.now().year - 4, datetime.now().year + 1)]
    g = _rng("pgm")
    pt = {y: round(float(v)) for y, v in zip(years, g.normal(-650, 220, len(years)))}
    pd_ = {y: round(float(v)) for y, v in zip(years, g.normal(150, 320, len(years)))}
    return {"live": False, "pt_balance": pt, "pd_balance": pd_,
            "pt_source": "", "pd_source": ""}


def fetch_autos() -> dict:
    """World light-vehicle production, quarterly.  MOCK → OICA statistics."""
    g = _rng("autos")
    qidx = pd.period_range(end=pd.Timestamp.today(), periods=12, freq="Q")
    autos = pd.Series(np.clip(g.normal(22.3, 0.9, 12), 18, None),
                      index=qidx.to_timestamp(how="end").normalize())
    return {"live": False, "autos": autos}


# ---------------------------------------------------------------------------
# Commentary — neutral, client-safe observations generated from the numbers.
# No recommendations, no buy/sell language (compliance).
# ---------------------------------------------------------------------------
def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


_DECS = {"t": 1, "Moz": 1, "koz": 0}


def _delta_disp(chg: float | None, unit: str) -> str | None:
    """Signed '+31.2 t' display, or None when the move is missing/NaN or rounds
    to zero (so the dashboard shows '—' and the bullet is dropped instead of
    'outflows of 0.0' / 'drew nan')."""
    decs = _DECS.get(unit, 1)
    if chg is None or not np.isfinite(chg) or abs(chg) < 0.5 * 10 ** -decs:
        return None
    return f"{chg:+,.{decs}f} {unit}"


def _chg_1m(s: pd.Series) -> float | None:
    """Change vs ~one month back, robust to the series' own cadence (the COMEX
    archive is daily snapshots, mock/mint are monthly — a plain diff() would be
    day-over-day on the former)."""
    s = s.dropna()
    if len(s) < 2:
        return None
    prev = s[s.index <= s.index[-1] - pd.Timedelta(days=28)]
    if prev.empty:
        return None
    return float(s.iloc[-1] - prev.iloc[-1])


def _posture(idx: float | None) -> str:
    if idx is None:
        return "mid-range"
    if idx >= cotdata.EXTREME_HI:
        return "toward the top of its 3-year range"
    if idx <= cotdata.EXTREME_LO:
        return "toward the bottom of its 3-year range"
    return "mid-range"


def _pgm_bullet(metal: str, balance: dict, publisher: str | None) -> str:
    """'WPIC's published balance has platinum in a deficit of 297 koz for 2026
    (forecast).' Years suffixed 'f' in pm_pgm.json are publisher forecasts."""
    year, val = list(balance.items())[-1]
    fc = year.endswith("f")
    who = f"{publisher}'s published balance has" if publisher else "Published balances have"
    return (f"{who} {metal} in a {'deficit' if val < 0 else 'surplus'} of "
            f"{abs(val):,.0f} koz for {year.rstrip('f')}{' (forecast)' if fc else ''}.")


def _metal_bullets(row: dict, extras: dict) -> list[str]:
    b = [f"Price {_fmt_pct(row['chg_1m'])} over the month and {_fmt_pct(row['chg_3m'])} "
         f"over three months; managed-money positioning sits {_posture(row['cot_idx'])} "
         f"(COT index {row['cot_idx']:.0f})." if row["cot_idx"] is not None else
         f"Price {_fmt_pct(row['chg_1m'])} over the month and {_fmt_pct(row['chg_3m'])} over three months."]
    if row["mm_4w"] is not None:
        d = "added to" if row["mm_4w"] > 0 else "trimmed"
        b.append(f"Funds {d} net length over the past four weeks "
                 f"({row['mm_4w']:+,.0f} lots to {row['mm_net']:+,.0f}).")
    if row.get("etf_disp"):
        d = "inflows" if row["etf_chg_1m"] > 0 else "outflows"
        decs = _DECS.get(row["etf_unit"], 1)
        b.append(f"ETF holdings saw {d} of {abs(row['etf_chg_1m']):,.{decs}f} {row['etf_unit']} "
                 f"on the month, to {row['etf_last']:,.{decs}f} {row['etf_unit']}.")
    if row.get("comex_disp"):
        d = "built" if row["comex_chg_1m"] > 0 else "drew"
        decs = _DECS.get(row["comex_unit"], 1)
        b.append(f"Exchange registered stocks {d} {abs(row['comex_chg_1m']):,.{decs}f} "
                 f"{row['comex_unit']} over the month.")
    b.extend(extras.get(row["key"], []))
    return b


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(force: bool = False) -> dict:
    prices = fetch_prices()
    cot = fetch_cot(force=force)
    fred = fetch_fred(force=force)
    etf = fetch_etf()
    comex = fetch_comex()
    swiss = fetch_swiss()
    cb = fetch_central_banks()
    prem = fetch_premiums()
    mint = fetch_mint()
    pgm = fetch_pgm()
    # auto production (fetch_autos) is parked: OICA's 2025 site redesign removed
    # the downloadable stats, so the chart is out of the report until a clean
    # source appears — WPIC/JM synopsis bullets carry the auto-demand story.

    # The FAIR-VALUE GAP: how much of gold's twelve-month move is NOT explained by
    # real yields, the dollar, credit and risk appetite. It belongs on this report
    # rather than only in the weekly one, because it is slow-moving by construction
    # and it closes the question these pages otherwise leave open — the real-yield
    # chart and the official-sector chart sit side by side, and the gap is the number
    # that connects them. Never fails the build: a machine without the gold engine
    # simply omits the line.
    fv = {}
    try:
        from src import goldfeatures, goldsens
        feats, _t = goldfeatures.load()
        # Phase-averaged. Reading the last point of a single sampling published
        # +15.2%/82nd percentile when the same statistic ranged 13.7-23.3% across
        # the other 20 offsets; the mean is +18.5% and the 91st. `since` carries the
        # actual span rather than letting the template assert a start year — the gap
        # series begins wherever the walk-forward fit had enough training data.
        fv = goldsens.fair_value_summary(feats, goldsens.DEFAULT_H)
        if fv:
            from src import reportkit
            fv["asof_pretty"] = reportkit.pretty_date(fv["asof"])
            # R2 of the same fit, so the gap is never published without the quality
            # of the anchor it is measured against.
            try:
                _f = goldsens.fit(feats, _t, goldsens.DEFAULT_H)
                fv["r2"] = round(float(_f["r2"]), 2) if _f else None
            except Exception:
                fv["r2"] = None
    except Exception:
        fv = {}

    mock_blocks = [name for name, blk in [
        ("prices (datafeed %s)" % datafeed.MODE, prices), ("COT", cot), ("FRED", fred),
        ("ETF holdings", etf), ("COMEX stocks", comex), ("Swiss exports", swiss),
        ("central banks", cb), ("premiums", prem), ("mint sales", mint),
        ("PGM balances", pgm),
    ] if not blk["live"]]

    px, hist, fr = prices["px"], cot["hist"], fred["df"]
    today = pd.Timestamp.today().normalize()

    metals, price_series, cot_series = [], {}, {}
    for m in METALS:
        s = px[m["ticker"]].dropna() if m["ticker"] in px.columns else pd.Series(dtype=float)
        h = hist[hist["key"] == m["key"]].set_index("date").sort_index()
        empty = {"series": pd.Series(dtype=float), "unit": "", "registered": pd.Series(dtype=float),
                 "eligible": pd.Series(dtype=float)}
        e = etf["holdings"].get(m["key"], empty)
        c = comex["stocks"].get(m["key"], empty)
        h4 = h["net"].iloc[-5] if len(h) >= 5 else None
        row = {
            "key": m["key"], "name": m["name"], "unit": m["unit"],
            "last": round(float(s.iloc[-1]), 2) if len(s) else None,
            "chg_1m": _pct(s, 30), "chg_3m": _pct(s, 91), "chg_ytd": _pct(s, (today - pd.Timestamp(today.year, 1, 1)).days),
            "mm_net": float(h["net"].iloc[-1]) if len(h) else None,
            "mm_4w": float(h["net"].iloc[-1] - h4) if h4 is not None else None,
            "cot_idx": float(h["cot_idx"].iloc[-1]) if len(h) and pd.notna(h["cot_idx"].iloc[-1]) else None,
            "cot_date": h.index[-1].strftime("%d %b") if len(h) else None,
            "etf_last": float(e["series"].iloc[-1]) if len(e["series"]) else None,
            "etf_chg_1m": _chg_1m(e["series"]), "etf_unit": e["unit"],
            "comex_reg": float(c["registered"].iloc[-1]) if len(c["registered"]) else None,
            "comex_chg_1m": _chg_1m(c["registered"]), "comex_unit": c["unit"],
        }
        row["etf_disp"] = _delta_disp(row["etf_chg_1m"], row["etf_unit"])
        row["comex_disp"] = _delta_disp(row["comex_chg_1m"], row["comex_unit"])
        metals.append(row)
        price_series[m["key"]] = _ser(s.iloc[-520:] if len(s) else s)
        cot_series[m["key"]] = {"net": _ser(h["net"].iloc[-260:], 0), "idx": _ser(h["cot_idx"].iloc[-260:], 1)}

    gold_px = px[METALS[0]["ticker"]].dropna() if METALS[0]["ticker"] in px.columns else pd.Series(dtype=float)
    ratio = (px["GCA Comdty"] / px["SIA Comdty"]).dropna() if {"GCA Comdty", "SIA Comdty"} <= set(px.columns) else pd.Series(dtype=float)

    ry = fr["real_yield"].dropna()
    extras = {
        "gold": [(lambda t: f"Central banks were net {'buyers' if t >= 0 else 'sellers'} of "
                            f"{abs(t):,.0f}t over the past three months (reported basis; "
                            f"excludes unreported official-sector buying).")(
                     cb["net_purchases"].iloc[-3:].sum()),
                 f"The Shanghai gold price last stood {prem['sge'].iloc[-1]:+,.0f} $/oz vs London"
                 + (f"; the Indian market {prem['india'].iloc[-1]:+,.0f} $/oz."
                    if len(prem["india"]) else
                    (" (5-day average, weekly)" if "SGE" in prem.get("source", "")
                     else " (daily closes)") + ".")],
        "silver": [f"The gold/silver ratio stands at {ratio.iloc[-1]:,.1f}." if len(ratio) else ""],
        "platinum": [_pgm_bullet("platinum", pgm["pt_balance"],
                                 "WPIC" if "WPIC" in pgm.get("pt_source", "") else None)],
        "palladium": [_pgm_bullet("palladium", pgm["pd_balance"],
                                  "Johnson Matthey" if "Johnson Matthey" in pgm.get("pd_source", "") else None)],
    }
    extras = {k: [x for x in v if x] for k, v in extras.items()}

    g = metals[0]
    ry_last = round(float(ry.iloc[-1]), 2) if len(ry) else None
    headline = (
        f"Gold trades at {g['last']:,.0f} $/oz ({_fmt_pct(g['chg_1m'])} over the month) with the US 10y "
        f"real yield at {ry_last:.2f}%. " if g["last"] is not None and ry_last is not None else "")
    longs = [m["name"] for m in metals if (m["cot_idx"] or 50) >= cotdata.EXTREME_HI]
    shorts = [m["name"] for m in metals if (m["cot_idx"] or 50) <= cotdata.EXTREME_LO]
    if longs:
        headline += (f"Managed-money positioning is toward the top of its 3-year range in "
                     f"{', '.join(longs)}. ")
    if shorts:
        headline += (f"Positioning sits toward the bottom of its 3-year range in "
                     f"{', '.join(shorts)}. ")
    headline += ("The pages that follow cover physical flows, official-sector activity, holdings "
                 "and the published market balances for each metal.")

    d = {
        "generated": datetime.now().strftime("%d %b %Y %H:%M"),
        "headline": headline,
        "month": today.strftime("%B %Y"),
        "asof": today.strftime("%d %b %Y"),
        "fair_value": fv,
        "mock_blocks": mock_blocks,
        "metals": metals,
        "macro": {
            "real_yield": round(float(ry.iloc[-1]), 2) if len(ry) else None,
            "real_yield_3m_bp": round(float(ry.iloc[-1] - ry[ry.index <= ry.index[-1] - pd.Timedelta(days=91)].iloc[-1]) * 100) if len(ry) > 70 else None,
            "dollar_1m": _pct(fr["dollar"].dropna(), 30),
        },
        "series": {
            "prices": price_series,
            "cot": cot_series,
            "real_yield": _ser(ry.iloc[-520:]),
            "gold": _ser(gold_px.iloc[-520:]),
            "ratio": _ser(ratio.iloc[-1300:], 1),
            "etf": {k: {"unit": v["unit"], "s": _ser(v["series"], 1)} for k, v in etf["holdings"].items()},
            "comex": {k: {"unit": v["unit"], "reg": _ser(v["registered"], 2), "eli": _ser(v["eligible"], 2)}
                      for k, v in comex["stocks"].items()},
            "swiss": {"dests": list(swiss["exports"].columns),
                      "d": [d_.strftime("%Y-%m-%d") for d_ in swiss["exports"].index],
                      "rows": [[round(float(x), 1) for x in swiss["exports"][c]] for c in swiss["exports"].columns]},
            # R1 (audit 2026-08-23): the chart window, NOT the whole series. Repointing
            # this at the gold store took it from 46 bars over four years to 437 bars
            # back to 1990, and a single +2,644t print in Dec-1998 rescaled the y-axis
            # so recent official-sector buying — the entire point of the page — became a
            # hairline at zero. Deeper history is right for the ENGINE and wrong for
            # this chart; slice here rather than shortening the source.
            "cb": _ser(cb["net_purchases"].iloc[-48:], 1),
            # R2 (audit): the fine print credited "London (daily closes, Bloomberg)"
            # for a series that had silently become a 5-day-smoothed, weekly-resampled
            # SGE-vs-LBMA-AM construction — correlating only 0.33 with the Bloomberg
            # leg it displaced. The attribution must follow the data, so the fetchers'
            # own source strings are carried through instead of discarded.
            "cb_source": cb.get("source", "IMF international liquidity"),
            "sge_source": prem.get("source",
                                   "Shanghai vs London (daily closes, Bloomberg)"),
            "sge": _ser(prem["sge"], 1), "india": _ser(prem["india"], 1),
            "mint_gold": _ser(mint["gold_oz"], 0), "mint_silver": _ser(mint["silver_oz"], 0),
            "pt_balance": pgm["pt_balance"], "pd_balance": pgm["pd_balance"],
        },
        "bullets": {m["key"]: _metal_bullets(m, extras) for m in metals},
    }
    return d


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "pm_monitor.json"
    d = build(force="--force" in sys.argv)
    out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"pm_monitor: wrote {out} (mock blocks: {', '.join(d['mock_blocks']) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Hot Sheet provider
# ---------------------------------------------------------------------------
PM_RADAR_MAX = 2            # flow lines on the sheet — mirrors weekreview.PM_MAX
PM_RADAR_STALE_DAYS = 45    # the monitor is monthly; beyond this the provider goes quiet
PM_RADAR_HEAT_SCALE = 20.0  # heat = |monthly ETF % change| × this (a 5% flow month pins 100)
PM_RADAR_HEAT_FLAT = 40.0   # gauge when only vault stocks moved (no ETF % to scale from)
PM_RADAR_HEAT_FLOOR = 30.0  # emitted lines already cleared the monitor's rounds-to-zero
                            # bar — floor the gauge so a typical flow month (~0.3% ETF
                            # change → 6 raw) isn't invisible against z-scored providers


def radar_items() -> list:
    """Hot Sheet provider: the monitor's flow lines — only metals whose ETF/vault
    numbers actually moved (the same rounds-to-zero suppression and prose as
    weekreview.collect_pm), read straight from data/pm_monitor.json. Pure disk;
    quiet once the monthly build has gone stale."""
    from datetime import date

    from src import hotsheet

    try:
        d = json.loads((DATA / "pm_monitor.json").read_text(encoding="utf-8"))
    except Exception:                        # monitor never built — quiet, not broken
        return []
    try:
        asof = datetime.strptime(d.get("asof", ""), "%d %b %Y").date()
        if (date.today() - asof).days > PM_RADAR_STALE_DAYS:
            return []
        stamp = f" (as of {d['asof']})" if (date.today() - asof).days > 10 else ""
    except Exception:
        stamp = ""
    tickers = {m["key"]: m["ticker"] for m in METALS}
    out = []
    for m in d.get("metals", []):
        parts = []
        if m.get("etf_disp"):
            word = "outflows" if (m.get("etf_chg_1m") or 0) < 0 else "inflows"
            parts.append(f"ETF holdings saw {word} of **{m['etf_disp'].lstrip('+-')}** on the month")
        if m.get("comex_disp"):
            word = "drew" if str(m["comex_disp"]).startswith("-") else "built"
            parts.append(f"exchange registered stocks {word} {m['comex_disp'].lstrip('+-')}")
        if not parts:
            continue
        # the json stores the monthly ETF change as an absolute (etf_chg_1m) plus the
        # level (etf_last), not a % — reconstruct the % to scale the heat gauge
        chg, last = m.get("etf_chg_1m"), m.get("etf_last")
        pct = (100.0 * chg / (last - chg)
               if isinstance(chg, (int, float)) and isinstance(last, (int, float))
               and np.isfinite(chg) and np.isfinite(last) and last != chg else None)
        val = chg if isinstance(chg, (int, float)) else (
            m.get("comex_chg_1m") if isinstance(m.get("comex_chg_1m"), (int, float)) else None)
        out.append(hotsheet.item(
            tag="METALS", key=f"{m['key']}:flows", section="Precious metals",
            text=f"**{m['name']}** flows moved — " + " and ".join(parts) + f"{stamp}.",
            heat=(max(PM_RADAR_HEAT_FLOOR, min(100.0, abs(pct) * PM_RADAR_HEAT_SCALE))
                  if pct is not None else PM_RADAR_HEAT_FLAT),
            metric=(m.get("etf_disp") or m.get("comex_disp") or ""), sub="on the month",
            value=val,                       # the monthly ETF (or vault) change, native units
            ticker=tickers.get(m["key"], ""), page="Precious Metals", book="ficc",
        ))
        if len(out) >= PM_RADAR_MAX:
            break
    return out
