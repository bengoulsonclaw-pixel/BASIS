"""Free macro data feed for the policy-rule engine (BASIS · Macro Rate Radar).

There is no Bloomberg anywhere in this module — by design. Every series comes from a
public statistical office or central bank, so the rule engine keeps working on a day the
Terminal is wedged, and the vintage backtest can reach back decades without burning DAPI
capacity (see the Bloomberg daily-capacity playbook for why that matters).

Sources — all verified reachable 2026-08-16
-------------------------------------------
    FRED / ALFRED    api.stlouisfed.org        FREE KEY   US everything, + real-time vintages
    ECB Data Portal  data-api.ecb.europa.eu    no key     euro-area HICP, policy rates
    Eurostat         ec.europa.eu/eurostat     no key     HICP flash (ECB fallback)
    BoE IADB         bankofengland.co.uk       no key     Bank Rate + UK series
    ONS              ons.gov.uk .../data       no key     UK CPI, wages, unemployment
    Cleveland Fed    clevelandfed.org          no key     current-month inflation nowcast
    NY Fed (HLW)     newyorkfed.org            no key     r* and output gap, US + euro area

The FRED key is the one thing a human has to fetch: free, instant, from
https://fred.stlouisfed.org/docs/api/api_key.html — drop it in data/fred_key.txt
(same convention as data/eia_key.txt) or set $FRED_API_KEY.

Gotchas, each one learned by walking into it
--------------------------------------------
* `fred.stlouisfed.org/graph/fredgraph.csv` — the no-key CSV endpoint everyone reaches for
  first — is Akamai-blocked to non-browser clients, and it TIMES OUT rather than returning
  403. Don't "fix" a hang here by raising the timeout; the keyed JSON API at
  api.stlouisfed.org is the only reliable FRED path.
* atlantafed.org and newyorkfed.org serve SOFT 404s: HTTP 200 with an HTML error page in
  the body. Status codes lie here — `_is_xlsx` checks the payload magic bytes instead.
* ONS retired api.ons.gov.uk on 25/11/2024. The website's `/data` JSON endpoint is the
  live replacement and carries full history.
* HLW is quarterly and runs ~2 quarters behind. r* moves slowly enough that this is fine,
  but every HLW series carries an explicit `stale_note` so the page can say so out loud
  rather than implying the number is current.

Everything degrades gracefully: any failure returns a Series with ok=False and a plain
reason, never a raised exception, so one dead source can't blank the whole page.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _ROOT / "data" / "macro_cache"

# A browser UA is not optional: several of these hosts (ONS, Cleveland, NY Fed) serve
# error pages or hang for default python-requests/urllib agents.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/127.0 Safari/537.36")

_TIMEOUT = 45
_DEFAULT_TTL = 6 * 3600            # macro data updates daily at best; 6h is generous


# ── series container ────────────────────────────────────────────────────────────────
@dataclass
class Series:
    """One macro time series. `obs` is [(date, float), ...] ascending, always clean:
    missing/'.'/NA points are dropped rather than carried as NaN."""
    sid: str
    title: str = ""
    source: str = ""
    units: str = ""
    freq: str = ""                                   # 'D' / 'M' / 'Q' / 'A'
    obs: list[tuple[date, float]] = field(default_factory=list)
    ok: bool = True
    reason: str = ""
    stale_note: str = ""                             # set when a source is structurally laggy

    def __bool__(self) -> bool:
        return self.ok and bool(self.obs)

    @property
    def last(self) -> tuple[date, float] | None:
        return self.obs[-1] if self.obs else None

    @property
    def latest(self) -> float | None:
        return self.obs[-1][1] if self.obs else None

    @property
    def latest_date(self) -> date | None:
        return self.obs[-1][0] if self.obs else None

    @property
    def is_projection(self) -> bool:
        """True when the series runs into the future. CBO series do: GDPPOT and NROU
        both extend a decade out."""
        return bool(self.obs) and self.obs[-1][0] > date.today()

    @property
    def latest_actual(self) -> float | None:
        """The value in force TODAY, ignoring any projected tail.

        Use this, not `.latest`, for anything CBO publishes. NROU's last observation is a
        2036 projection of 4.16% while the value in force today is 4.39% — reading
        `.latest` silently understates the natural rate by 23bp, which flows straight into
        the output gap and biases the balanced-approach rule by roughly twice that. It is
        a wrong number that looks entirely plausible."""
        hit = self.asof(date.today())
        return hit[1] if hit else None

    @property
    def latest_actual_date(self) -> date | None:
        hit = self.asof(date.today())
        return hit[0] if hit else None

    def asof(self, when: date) -> tuple[date, float] | None:
        """Last observation dated on or before `when` — the honest read for a backtest.
        NOTE this is observation-date, not publication-date: for a genuinely point-in-time
        read use the ALFRED vintage helpers, which respect release lags too."""
        prior = [o for o in self.obs if o[0] <= when]
        return prior[-1] if prior else None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.obs, columns=["date", "value"]).set_index("date")

    def yoy(self) -> "Series":
        """Year-on-year % change of a level series (monthly or quarterly index/level).
        Uses a calendar-date lookup, not a fixed row offset, so a missing month can't
        silently turn a 12m comparison into an 11m one."""
        by_date = dict(self.obs)
        out: list[tuple[date, float]] = []
        for d, v in self.obs:
            try:
                prev_d = d.replace(year=d.year - 1)
            except ValueError:                        # 29 Feb
                prev_d = d.replace(year=d.year - 1, day=28)
            prev = by_date.get(prev_d)
            if prev is None:
                # quarterly/monthly stamps sometimes drift a day; accept the nearest
                # observation within a week of the anniversary, else skip the point.
                cands = [(abs((od - prev_d).days), ov) for od, ov in self.obs
                         if abs((od - prev_d).days) <= 7]
                if not cands:
                    continue
                prev = min(cands)[1]
            if prev:
                out.append((d, (v / prev - 1.0) * 100.0))
        return Series(self.sid + "_YOY", self.title + " (y/y %)", self.source, "% y/y",
                      self.freq, out, self.ok, self.reason, self.stale_note)


def _fail(sid: str, reason: str, source: str = "") -> Series:
    return Series(sid, source=source, ok=False, reason=reason[:200])


# ── disk cache ──────────────────────────────────────────────────────────────────────
def _cache_file(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
    return _CACHE / f"{safe}.json"


def _cache_get(key: str, ttl: int):
    f = _cache_file(key)
    if not f.exists():
        return None
    try:
        blob = json.loads(f.read_text(encoding="utf-8"))
        if ttl >= 0 and time.time() - blob.get("_fetched", 0) > ttl:
            return None
        return blob.get("payload")
    except Exception:
        return None


def _cache_put(key: str, payload) -> None:
    try:
        _CACHE.mkdir(parents=True, exist_ok=True)
        _cache_file(key).write_text(
            json.dumps({"_fetched": time.time(), "payload": payload}), encoding="utf-8")
    except Exception:
        pass                                          # a cache we can't write is not an error


def _http(url: str, *, binary: bool = False, timeout: int = _TIMEOUT):
    r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    r.raise_for_status()
    if binary:
        return r.content
    # Decode explicitly rather than trusting requests' charset guess: ONS serves UTF-8
    # JSON with no charset header, and the guess turns '£' in series titles into U+FFFD.
    try:
        return r.content.decode("utf-8")
    except UnicodeDecodeError:
        return r.content.decode("latin-1")


def _is_xlsx(b: bytes) -> bool:
    """Both atlantafed.org and newyorkfed.org answer dead media URLs with HTTP 200 and an
    HTML error page, so the only trustworthy check is the zip magic of the xlsx itself."""
    return b[:2] == b"PK"


def _num(x) -> float | None:
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return None if v != v else v                      # drop NaN


# ── FRED / ALFRED ───────────────────────────────────────────────────────────────────
_FRED_BASE = "https://api.stlouisfed.org/fred/"


def fred_key() -> str | None:
    k = os.getenv("FRED_API_KEY")
    if k and k.strip():
        return k.strip()
    f = _ROOT / "data" / "fred_key.txt"
    if f.exists():
        k = f.read_text(encoding="utf-8").strip()
        if k:
            return k
    return None


def have_fred_key() -> bool:
    return fred_key() is not None


FRED_KEY_HELP = ("No FRED API key. Get a free one at "
                 "https://fred.stlouisfed.org/docs/api/api_key.html and save it to "
                 "data/fred_key.txt (or set $FRED_API_KEY).")


def fred(sid: str, *, start: str = "1990-01-01", vintage: date | str | None = None,
         units: str | None = None, ttl: int = _DEFAULT_TTL) -> Series:
    """One FRED series. With `vintage` set this becomes an ALFRED call returning the data
    AS IT STOOD on that date — revisions and release lags included, which is the whole
    point for the backtest: on 2024-06-01 you did not know 2024Q1 GDP as it reads today.

    `units` passes FRED's own transforms through ('pc1' = y/y %, 'pch' = period %, 'chg').
    Doing it server-side avoids re-deriving a transform FRED already publishes."""
    key = fred_key()
    if not key:
        return _fail(sid, FRED_KEY_HELP, "FRED")

    params = {"series_id": sid, "api_key": key, "file_type": "json",
              "observation_start": start}
    if units:
        params["units"] = units
    if vintage:
        v = vintage.isoformat() if isinstance(vintage, date) else str(vintage)
        # realtime_start == realtime_end == v asks ALFRED for the single vintage in force
        # on that day, which is exactly "what a trader could have seen that morning".
        params["realtime_start"] = v
        params["realtime_end"] = v

    ck = "fred_" + urllib.parse.urlencode({k: v for k, v in params.items() if k != "api_key"})
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            txt = _http(_FRED_BASE + "series/observations?" + urllib.parse.urlencode(params))
            cached = json.loads(txt).get("observations", [])
            _cache_put(ck, cached)
        except Exception as e:
            stale = _cache_get(ck, -1)                # expired cache beats no data at all
            if stale is not None:
                cached = stale
            else:
                return _fail(sid, f"FRED fetch failed: {e}", "FRED")

    obs = []
    for o in cached:
        v = _num(o.get("value"))                      # FRED writes '.' for missing
        if v is None:
            continue
        try:
            obs.append((date.fromisoformat(o["date"]), v))
        except (KeyError, ValueError):
            continue
    obs.sort()
    return Series(sid, _FRED_TITLES.get(sid, sid), "FRED", units or "", "", obs)


def fred_meta(sid: str, ttl: int = 30 * 24 * 3600) -> dict:
    """Series metadata (title, units, frequency, last updated). Cached hard — this changes
    about never, and it is only used to label charts honestly."""
    key = fred_key()
    if not key:
        return {}
    ck = f"fredmeta_{sid}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            txt = _http(_FRED_BASE + "series?" + urllib.parse.urlencode(
                {"series_id": sid, "api_key": key, "file_type": "json"}))
            cached = (json.loads(txt).get("seriess") or [{}])[0]
            _cache_put(ck, cached)
        except Exception:
            return {}
    return cached or {}


# Labels for the series the rule engine actually consumes — so a chart legend never
# shows a bare mnemonic to someone who doesn't live in FRED.
_FRED_TITLES = {
    "PCEPILFE": "Core PCE price index",
    "PCEPI": "Headline PCE price index",
    "CPILFESL": "Core CPI",
    "CPIAUCSL": "Headline CPI",
    "PCETRIM12M159SFRBDAL": "Trimmed-mean PCE (Dallas Fed, y/y)",
    "MEDCPIM158SFRBCLE": "Median CPI (Cleveland Fed)",
    "GDPC1": "Real GDP",
    "GDPPOT": "Real potential GDP (CBO)",
    "UNRATE": "Unemployment rate",
    "NROU": "Natural rate of unemployment (CBO)",
    "GDPNOW": "GDPNow (Atlanta Fed)",
    "DFEDTARU": "Fed target range — upper",
    "DFEDTARL": "Fed target range — lower",
    "EFFR": "Effective fed funds rate",
    "SOFR": "SOFR",
    "T5YIE": "5y breakeven inflation",
    "T10YIE": "10y breakeven inflation",
    "T5YIFR": "5y5y forward breakeven inflation",
    "SAHMREALTIME": "Sahm rule recession indicator (real-time)",
    "ICSA": "Initial jobless claims",
    "PAYEMS": "Nonfarm payrolls",
    "AHETPI": "Average hourly earnings",
}


# ── ECB Data Portal ─────────────────────────────────────────────────────────────────
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data/"


def ecb(flow: str, key_str: str, *, title: str = "", n: int = 0,
        ttl: int = _DEFAULT_TTL) -> Series:
    """ECB Data Portal series, e.g. ecb('ICP', 'M.U2.N.000000.4.ANR') for euro-area HICP
    y/y. No key needed. Returns csvdata and reads TIME_PERIOD / OBS_VALUE."""
    url = f"{_ECB_BASE}{flow}/{key_str}?format=csvdata"
    if n:
        url += f"&lastNObservations={n}"
    ck = f"ecb_{flow}_{key_str}_{n}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = _http(url)
            _cache_put(ck, cached)
        except Exception as e:
            stale = _cache_get(ck, -1)
            if stale is None:
                return _fail(f"{flow}/{key_str}", f"ECB fetch failed: {e}", "ECB")
            cached = stale
    try:
        df = pd.read_csv(io.StringIO(cached))
    except Exception as e:
        return _fail(f"{flow}/{key_str}", f"ECB parse failed: {e}", "ECB")
    if "TIME_PERIOD" not in df or "OBS_VALUE" not in df:
        return _fail(f"{flow}/{key_str}", "ECB schema changed", "ECB")

    obs = []
    for t, v in zip(df["TIME_PERIOD"], df["OBS_VALUE"]):
        d, val = _parse_period(str(t)), _num(v)
        if d and val is not None:
            obs.append((d, val))
    obs.sort()
    return Series(f"{flow}/{key_str}", title or f"{flow} {key_str}", "ECB Data Portal",
                  "", "", obs)


def _parse_period(t: str) -> date | None:
    """ECB/Eurostat period stamps: 2026-07, 2026-Q2, 2026, 2026-07-15."""
    t = t.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
            return date.fromisoformat(t)
        if re.fullmatch(r"\d{4}-\d{2}", t):
            y, m = t.split("-")
            return date(int(y), int(m), 1)
        m = re.fullmatch(r"(\d{4})-?Q([1-4])", t)
        if m:
            return date(int(m.group(1)), (int(m.group(2)) - 1) * 3 + 1, 1)
        if re.fullmatch(r"\d{4}", t):
            return date(int(t), 1, 1)
    except ValueError:
        return None
    return None


# ── Eurostat ────────────────────────────────────────────────────────────────────────
def eurostat(dataset: str, params: dict, *, title: str = "",
             ttl: int = _DEFAULT_TTL) -> Series:
    """Eurostat JSON-stat.

    This — not the ECB portal — is the live source for euro-area HICP. See `ea_hicp`
    below for why."""
    q = dict(params)
    q["format"] = "JSON"
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
           + dataset + "?" + urllib.parse.urlencode(q, doseq=True))
    ck = f"estat_{dataset}_{urllib.parse.urlencode(q, doseq=True)}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = json.loads(_http(url))
            _cache_put(ck, cached)
        except Exception as e:
            stale = _cache_get(ck, -1)
            if stale is None:
                return _fail(dataset, f"Eurostat fetch failed: {e}", "Eurostat")
            cached = stale
    try:
        # JSON-stat: values are keyed by flat index into the dimension product; with the
        # non-time dimensions pinned to one member each, that index IS the time index.
        idx = cached["dimension"]["time"]["category"]["index"]
        vals = cached["value"]
        rev = {int(v): k for k, v in idx.items()}
        obs = []
        for k, v in vals.items():
            d = _parse_period(rev.get(int(k), ""))
            val = _num(v)
            if d and val is not None:
                obs.append((d, val))
        obs.sort()
        return Series(dataset, title or dataset, "Eurostat", "", "", obs)
    except Exception as e:
        return _fail(dataset, f"Eurostat parse failed: {e}", "Eurostat")


# ECOICOP ver.2 aggregate codes (the classification HICP moved to in Jan 2026).
EA_HICP_CODES = {"headline": "TOTAL", "core": "TOT_X_NRG_FOOD"}


def ea_hicp(which: str = "headline", *, geo: str = "EA",
            ttl: int = _DEFAULT_TTL) -> Series:
    """Euro-area HICP y/y from Eurostat's `prc_hicp_minr`.

    Read this before "fixing" it to point at the ECB portal:

    From January 2026 the HICP moved to ECOICOP ver.2 and was rebased to 2025=100. The
    old Eurostat datasets (prc_hicp_manr / _midx / _fp) and the ECB's whole ICP dataflow
    were CLOSED at 2025-12 — Eurostat has literally renamed them "(1997-2025)". They
    still answer HTTP 200 with clean, plausible, well-formed data that simply stops eight
    months ago. Nothing about the response says "stale"; you only notice by looking at
    the last date. Pointing the ECB rule at the old series put euro-area inflation at
    1.9% when it was actually 2.9% — a full point of policy error, silently.

    The all-items code also changed: `CP00` became `TOTAL`. Core stayed `TOT_X_NRG_FOOD`.
    """
    code = EA_HICP_CODES.get(which)
    if not code:
        return _fail(f"hicp/{which}", f"unknown HICP aggregate {which}", "Eurostat")
    s = eurostat("prc_hicp_minr",
                 {"geo": geo, "unit": "RCH_A", "coicop18": code},
                 title=f"Euro-area HICP — {which} (y/y %)", ttl=ttl)
    s.sid = f"HICP_{which}"
    s.freq = "M"
    return s


# ── Bank of England IADB ────────────────────────────────────────────────────────────
def boe(series_code: str, *, start: str = "01/Jan/2000", title: str = "",
        ttl: int = _DEFAULT_TTL) -> Series:
    """Bank of England interactive database. `series_code` e.g. IUDBEDR (Bank Rate)."""
    today = date.today().strftime("%d/%b/%Y")
    url = ("https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?"
           f"csv.x=yes&Datefrom={start}&Dateto={today}&SeriesCodes={series_code}"
           "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    ck = f"boe_{series_code}_{start}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = _http(url)
            _cache_put(ck, cached)
        except Exception as e:
            stale = _cache_get(ck, -1)
            if stale is None:
                return _fail(series_code, f"BoE fetch failed: {e}", "BoE")
            cached = stale
    try:
        df = pd.read_csv(io.StringIO(cached))
        dcol = df.columns[0]
        vcol = [c for c in df.columns if c != dcol][0]
        obs = []
        for d, v in zip(df[dcol], df[vcol]):
            val = _num(v)
            if val is None:
                continue
            try:
                obs.append((datetime.strptime(str(d).strip(), "%d %b %Y").date(), val))
            except ValueError:
                continue
        obs.sort()
        return Series(series_code, title or series_code, "Bank of England", "", "", obs)
    except Exception as e:
        return _fail(series_code, f"BoE parse failed: {e}", "BoE")


# ── ONS ─────────────────────────────────────────────────────────────────────────────
_ONS_SERIES = {
    # series id -> (ONS website path segment, dataset)
    # DKO8 is the real UK core measure: "CPI 12mth: Excluding Energy, food, alcoholic
    # beverages & tobacco". L55O is CPIH all-items — a different thing entirely, and an
    # easy mix-up because both are plausible-looking 4-character MM23 codes.
    "D7G7": ("economy/inflationandpriceindices", "mm23"),      # CPI y/y, all items
    "DKO8": ("economy/inflationandpriceindices", "mm23"),      # CPI y/y, core
    "L55O": ("economy/inflationandpriceindices", "mm23"),      # CPIH y/y, all items
    "MGSX": ("employmentandlabourmarket/peoplenotinwork/unemployment", "lms"),
    "KAB9": ("employmentandlabourmarket/peopleinwork/earningsandworkinghours", "emp"),
}


def ons(sid: str, *, title: str = "", ttl: int = _DEFAULT_TTL) -> Series:
    """ONS time series via the website `/data` JSON endpoint.

    Their documented api.ons.gov.uk was decommissioned on 25/11/2024 — this endpoint is
    the live replacement and carries the full history in `months`/`quarters`/`years`."""
    sid = sid.upper()
    if sid not in _ONS_SERIES:
        return _fail(sid, f"unknown ONS series {sid}", "ONS")
    path, dataset = _ONS_SERIES[sid]
    url = f"https://www.ons.gov.uk/{path}/timeseries/{sid.lower()}/{dataset}/data"
    ck = f"ons_{sid}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = json.loads(_http(url))
            _cache_put(ck, cached)
        except Exception as e:
            stale = _cache_get(ck, -1)
            if stale is None:
                return _fail(sid, f"ONS fetch failed: {e}", "ONS")
            cached = stale
    try:
        obs, freq = [], "M"
        rows = cached.get("months") or []
        if not rows:
            rows, freq = cached.get("quarters") or [], "Q"
        for r in rows:
            val = _num(r.get("value"))
            if val is None:
                continue
            d = _parse_ons_date(r)
            if d:
                obs.append((d, val))
        obs.sort()
        desc = (cached.get("description") or {}).get("title") or sid
        return Series(sid, title or desc, "ONS", "", freq, obs)
    except Exception as e:
        return _fail(sid, f"ONS parse failed: {e}", "ONS")


_MONTH_ABBR = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _parse_ons_date(row: dict) -> date | None:
    y = row.get("year")
    if not y:
        return None
    try:
        year = int(str(y).strip())
    except ValueError:
        return None
    mon = str(row.get("month") or "").strip().lower()[:3]
    if mon in _MONTH_ABBR:
        return date(year, _MONTH_ABBR[mon], 1)
    q = str(row.get("quarter") or "").strip().upper()
    if q.startswith("Q") and q[1:2].isdigit():
        return date(year, (int(q[1]) - 1) * 3 + 1, 1)
    return date(year, 1, 1)


# ── Cleveland Fed inflation nowcast ─────────────────────────────────────────────────
_CLEV_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
             "inflationnowcasting/nowcast_month.json?sc_lang=en")

_CLEV_KEYS = {"CPI Inflation": "cpi", "Core CPI Inflation": "core_cpi",
              "PCE Inflation": "pce", "Core PCE Inflation": "core_pce"}


def cleveland_nowcast(ttl: int = 6 * 3600) -> dict:
    """Current-month m/m inflation nowcasts from the Cleveland Fed.

    This is the fastest honest read on inflation there is for free: it updates daily, so
    on the 14th of a month you already have an estimate of a print that lands in three
    weeks. Returns {'month': date, 'updated': str, 'cpi'/'core_cpi'/'pce'/'core_pce':
    float m/m %, 'ok': bool}.

    The payload is a FusionCharts dump — a list of ~158 chart blocks, one per target
    month, each holding the daily-revised nowcast path for that month. The last block is
    the live month; within it we take the last non-empty point of each series."""
    cached = _cache_get("cleveland_nowcast", ttl)
    if cached is None:
        try:
            cached = json.loads(_http(_CLEV_URL, timeout=90))
            _cache_put("cleveland_nowcast", cached)
        except Exception as e:
            stale = _cache_get("cleveland_nowcast", -1)
            if stale is None:
                return {"ok": False, "reason": f"Cleveland fetch failed: {e}"}
            cached = stale
    try:
        blk = cached[-1]
        chart = blk.get("chart", {})
        y, m = str(chart.get("subcaption", "")).split("-")
        out = {"ok": True, "month": date(int(y), int(m), 1),
               "updated": str(chart.get("_comment", "")).strip()}
        for s in blk.get("dataset", []):
            k = _CLEV_KEYS.get(s.get("seriesname"))
            if not k:
                continue
            vals = [_num(p.get("value")) for p in s.get("data", [])]
            vals = [v for v in vals if v is not None]
            if vals:
                out[k] = round(vals[-1], 3)
        return out
    except Exception as e:
        return {"ok": False, "reason": f"Cleveland parse failed: {e}"}


# ── Cleveland Fed simple policy rules (our validation anchor) ───────────────────────
_CLEV_RULES_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/policyrules/"
                   "policy_rules_spreadsheet.xlsx?sc_lang=en")


def cleveland_policy_rules(ttl: int = 24 * 3600) -> dict:
    """The Cleveland Fed's own seven-rule calculation — parameters, the forecasts feeding
    it, and the resulting prescriptions.

    This is the single most valuable thing in this module, and not for its numbers: it is
    an independent, machine-readable, regularly-updated computation of the same rules we
    implement. Feeding OUR engine THEIR inputs and comparing against THEIR outputs turns
    "the formula looks right" into a test that fails loudly when it isn't. See
    macrorules.validate_against_cleveland().

    Returns {'ok', 'params': {...}, 'asof': str, 'forecasts': {source: {var: {q: v}}},
    'rules': {source: {rule: {q: v}}}} where q is a float like 2026.2.
    """
    cached = _cache_get("cleveland_rules_b64", ttl)
    raw = None
    if cached:
        import base64
        try:
            raw = base64.b64decode(cached)
        except Exception:
            raw = None
    if raw is None or not _is_xlsx(raw):
        try:
            raw = _http(_CLEV_RULES_URL, binary=True, timeout=90)
        except Exception as e:
            return {"ok": False, "reason": f"Cleveland rules fetch failed: {e}"}
        if not _is_xlsx(raw):
            return {"ok": False, "reason": "Cleveland rules URL returned HTML, not xlsx"}
        import base64
        _cache_put("cleveland_rules_b64", base64.b64encode(raw).decode())

    try:
        xl = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        params, asof = {}, ""
        pdf = xl.parse("Parameters")
        for _, row in pdf.iterrows():
            label, val = row.iloc[0], row.iloc[1]
            if not isinstance(label, str):
                continue
            v = _num(val)
            low = label.lower()
            if low.startswith("based on data"):
                asof = label.strip()
            elif v is None:
                continue
            elif low.startswith("r*"):
                params["rstar"] = v
            elif low.startswith("ralt*"):
                params["rstar_alt"] = v
            elif low.startswith("uspf*"):
                params["u_spf"] = v
            elif low.startswith("usep*"):
                params["u_sep"] = v
            elif "inverse okun" in low:
                params["inv_okun"] = v
            elif "inertia" in low or "smoothing" in low:
                params["rho"] = v
            elif "target inflation" in low:
                params["target"] = v

        return {"ok": True, "params": params, "asof": asof,
                "forecasts": _clev_block(xl, "Forecasts"),
                "rules": _clev_block(xl, "Policy Rule Funds Rates")}
    except Exception as e:
        return {"ok": False, "reason": f"Cleveland rules parse failed: {e}"}


def _clev_block(xl, sheet: str) -> dict:
    """Both data sheets share a layout: column 0 names the forecast source but is filled
    in only on the first row of each block, column 1 names the variable/rule, and the
    remaining columns are quarters keyed by a 'Date (YYYY.Q format)' header row."""
    df = xl.parse(sheet, header=None)
    quarters: dict[int, float] = {}
    out: dict[str, dict] = {}
    source = None
    for _, row in df.iterrows():
        c0, c1 = row.iloc[0], row.iloc[1]
        if isinstance(c0, str) and c0.strip() and not c0.lower().startswith("source"):
            source = c0.strip()
        if not isinstance(c1, str) or not c1.strip():
            continue
        label = c1.strip()
        if label.lower().startswith("date"):
            quarters = {}
            for j in range(2, len(row)):
                q = _num(row.iloc[j])
                if q is not None:
                    quarters[j] = round(q, 1)
            continue
        if not quarters:
            continue
        vals = {}
        for j, q in quarters.items():
            v = _num(row.iloc[j]) if j < len(row) else None
            if v is not None:
                vals[q] = v
        if vals:
            out.setdefault(source or "?", {})[label] = vals
    return out


# ── NY Fed Holston-Laubach-Williams r* ──────────────────────────────────────────────
_HLW_CURRENT = ("https://www.newyorkfed.org/medialibrary/media/research/economists/"
                "williams/data/Holston_Laubach_Williams_current_estimates.xlsx")
_HLW_REALTIME = ("https://www.newyorkfed.org/medialibrary/media/research/economists/"
                 "williams/data/Holston_Laubach_Williams_real_time_estimates.xlsx")

# 'HLW Estimates' sheet, skiprows=5: Date | _ | g:US,CA,EA | _ | z:US,CA,EA | _ |
# r*:US,CA,EA | _ | gap:US,CA,EA — pandas suffixes the repeats .1/.2/.3 in that order.
_HLW_COLS = {"rstar": {"US": "US.2", "EA": "Euro Area.2", "CA": "Canada.2"},
             "gap":   {"US": "US.3", "EA": "Euro Area.3", "CA": "Canada.3"},
             "growth": {"US": "US", "EA": "Euro Area", "CA": "Canada"}}


def hlw(measure: str = "rstar", area: str = "US",
        ttl: int = 7 * 24 * 3600) -> Series:
    """Holston-Laubach-Williams natural rate / output gap / trend growth.

    measure: 'rstar' | 'gap' | 'growth'. area: 'US' | 'EA' | 'CA'.

    Quarterly and published with a long lag — typically ~2 quarters behind, which is why
    every series returned here carries a `stale_note`. That is tolerable for r* (it is a
    slow-moving trend by construction) and NOT tolerable for the output gap, so the rule
    engine prefers CBO/Okun for the gap and uses HLW's gap only as a cross-check.

    There is no UK in HLW — the BoE leg has to fall back to an Okun proxy."""
    if measure not in _HLW_COLS or area not in _HLW_COLS[measure]:
        return _fail(f"HLW/{measure}/{area}", f"unknown HLW {measure}/{area}", "NY Fed")

    cached = _cache_get("hlw_current_b64", ttl)
    raw = None
    if cached is not None:
        import base64
        try:
            raw = base64.b64decode(cached)
        except Exception:
            raw = None
    if raw is None or not _is_xlsx(raw):
        try:
            raw = _http(_HLW_CURRENT, binary=True, timeout=90)
        except Exception as e:
            return _fail(f"HLW/{measure}/{area}", f"HLW fetch failed: {e}", "NY Fed")
        if not _is_xlsx(raw):
            # newyorkfed.org answers a dead media path with 200 + an HTML error page
            return _fail(f"HLW/{measure}/{area}",
                         "HLW URL returned HTML, not a spreadsheet — the media path moved",
                         "NY Fed")
        import base64
        _cache_put("hlw_current_b64", base64.b64encode(raw).decode())

    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name="HLW Estimates",
                           skiprows=5, engine="openpyxl")
        col = _HLW_COLS[measure][area]
        if col not in df.columns:
            return _fail(f"HLW/{measure}/{area}",
                         f"HLW layout changed — no column {col}", "NY Fed")
        obs = []
        for d, v in zip(df["Date"], df[col]):
            val = _num(v)
            if val is None or pd.isna(d):
                continue
            obs.append((pd.Timestamp(d).date(), val))
        obs.sort()
        s = Series(f"HLW_{measure}_{area}",
                   {"rstar": "Natural rate r* (HLW)", "gap": "Output gap (HLW)",
                    "growth": "Trend growth (HLW)"}[measure] + f" — {area}",
                   "NY Fed (Holston-Laubach-Williams)", "%", "Q", obs)
        if obs:
            last_d = obs[-1][0]
            lag_q = (date.today() - last_d).days / 91.3
            if lag_q >= 1.5:
                # %q is not a portable strftime code — build the quarter label by hand.
                s.stale_note = (f"HLW publishes with a lag — latest estimate is "
                                f"{last_d.year} Q{(last_d.month - 1) // 3 + 1} "
                                f"({lag_q:.0f} quarters old).")
        return s
    except Exception as e:
        return _fail(f"HLW/{measure}/{area}", f"HLW parse failed: {e}", "NY Fed")


# ── BCB / Brazil ────────────────────────────────────────────────────────────────────
# Three separate free BCB services, no key on any of them:
#
#   SGS      api.bcb.gov.br/dados/serie/bcdata.sgs.{id}   time series (IPCA, Selic, …)
#   Olinda   olinda.bcb.gov.br/olinda/servico/Expectativas Focus survey expectations
#   sitebcb  www.bcb.gov.br/api/servico/sitebcb/calendario the published Copom calendar
#
# Brazil is the best-instrumented bloc after the US, and in two respects it is better
# than the euro area or the UK: the inflation TARGET is a published series (SGS 13521)
# rather than a constant we hardcode, and the Focus survey gives a per-meeting expected
# policy path — something no free source publishes for the ECB or the BoE.
_SGS_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs."

# SGS ids, each verified live 2026-08-27 against a plausibility check on its latest value.
SGS_IDS = {
    "ipca_yoy": 13522,        # IPCA accumulated 12m, % — headline inflation
    "ipca_mom": 433,          # IPCA monthly %
    "core_mom": 4466,         # IPCA trimmed-mean core, smoothed, monthly % (BCB's own core)
    "selic_target": 432,      # Meta Selic set by Copom, % — SEE the projection note below
    "selic_daily": 1178,      # Selic effective, annualised daily, %
    "unemp": 24369,           # Unemployment rate, PNAD Contínua, %
    "ibcbr": 24364,           # IBC-Br activity index, seasonally adjusted
    "infl_target": 13521,     # CMN inflation target, % (annual series)
}


def sgs(sid: int | str, *, title: str = "", start: str = "1999-01-01",
        ttl: int = _DEFAULT_TTL) -> Series:
    """One BCB SGS series. No key, no rate limit worth worrying about.

    Three traps, all real:

    * **Dates are dd/mm/yyyy.** Parsed as ISO or US order, 01/07/2026 silently becomes
      January and the whole series reorders.
    * **SGS 432 (Meta Selic) runs into the FUTURE.** The target in force is carried
      forward to the date of the next Copom decision, so its last observation is dated
      ahead of today — exactly the CBO trap `latest_actual` exists for. Read this series
      with `.asof(today)` / `.latest_actual`, never `.latest`, or the page reports a rate
      that has not been set yet as though it were current.
    * **DAILY series are capped at ten years per request** and answer a wider window with
      HTTP 406. The Selic ones (432, 1178) are daily and hit this immediately, so the
      request is PAGED in ten-year windows and concatenated.

      Retrying once inside a single ten-year window instead — which is what this did at
      first — looks like it works and is much worse than failing: the series simply
      begins nine years ago, and every caller reading further back gets None. On the
      history chart that surfaced as a Selic pinned at 0% from 2012 to 2017 (Brazil's
      policy rate was 7.25-14.25% over that stretch), because a missing policy rate was
      being substituted with zero downstream. Silent truncation is the dangerous failure
      mode here, not the 406.

    api.bcb.gov.br also times out intermittently under load, so a stale cache is
    preferred to a failure, as everywhere else here.
    """
    sid = str(sid)
    ck = f"sgs_{sid}_{start}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = json.loads(_http(f"{_SGS_BASE}{sid}/dados?formato=json"
                                      f"&dataInicial={_sgs_date(start)}"))
            _cache_put(ck, cached)
        except Exception as e:
            paged = _sgs_paged(sid, start) if "406" in str(e) else None
            if paged:
                cached = paged
                _cache_put(ck, cached)
            else:
                stale = _cache_get(ck, -1)
                if stale is None:
                    return _fail(sid, f"BCB SGS fetch failed: {e}", "BCB SGS")
                cached = stale
    try:
        obs = []
        for row in cached:
            val = _num(row.get("valor"))
            if val is None:
                continue
            try:
                obs.append((datetime.strptime(row["data"].strip(), "%d/%m/%Y").date(), val))
            except (ValueError, KeyError):
                continue
        obs.sort()
        return Series(sid, title or f"SGS {sid}", "BCB (SGS)", "%", "", obs)
    except Exception as e:
        return _fail(sid, f"BCB SGS parse failed: {e}", "BCB SGS")


def _sgs_paged(sid: str, start: str) -> list:
    """Walk a daily SGS series in ten-year windows, the most the API will serve at once.

    Each window is retried, because api.bcb.gov.br throttles rapid successive requests by
    answering **HTTP 200 with an HTML error page** rather than a 429 or a 5xx — the same
    "status codes lie" trap as the Fed regional sites, and the reason each payload is
    checked for being a list instead of being trusted. The identical request that fails
    one second succeeds the next, so a couple of retries with a pause clears it.

    Returns [] if any window is still failing after its retries, so a PARTIAL history is
    never mistaken for a complete one: silent truncation is what put a flat 0% Selic on
    the chart from 2012 to 2017 in the first place.
    """
    try:
        lo = datetime.strptime(start, "%Y-%m-%d").date()
    except ValueError:
        lo = date(1999, 1, 1)
    # A year past today, not today: SGS 432 carries the policy target forward to the next
    # Copom date, and the unpaged path returns that tail — the two must not disagree.
    end = date(date.today().year + 1, date.today().month, min(date.today().day, 28))
    rows, seen = [], set()
    while lo <= end:
        hi = min(date(lo.year + 9, lo.month, min(lo.day, 28)), end)
        chunk = None
        for attempt in range(4):
            try:
                blob = json.loads(_http(
                    f"{_SGS_BASE}{sid}/dados?formato=json"
                    f"&dataInicial={lo.strftime('%d/%m/%Y')}"
                    f"&dataFinal={hi.strftime('%d/%m/%Y')}"))
                if isinstance(blob, list):
                    chunk = blob
                    break
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        if chunk is None:
            return []
        for row in chunk:
            key = row.get("data")
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
        lo = date(hi.year, hi.month, min(hi.day, 28)) + timedelta(days=1)
        time.sleep(0.4)                   # be a good citizen between windows
    return rows


def _sgs_date(iso: str) -> str:
    """ISO -> dd/mm/yyyy, the only date format the SGS query accepts."""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return "01/01/1999"


def accum12(s: Series) -> Series:
    """Compound twelve monthly percent changes into a 12-month accumulated rate.

    Brazil publishes its core measures as MONTHLY percent changes, not as an index, so
    `Series.yoy()` — which divides index levels — cannot be used on them. Compounding is
    the correct reduction and it is what the BCB itself reports as "acumulado em 12
    meses": prod(1 + m/100) − 1, not the sum of the twelve months."""
    if not s.ok or len(s.obs) < 12:
        return _fail(s.sid + "_A12", "need 12 monthly observations to accumulate", s.source)
    out = []
    for i in range(11, len(s.obs)):
        f = 1.0
        for _d, v in s.obs[i - 11:i + 1]:
            f *= 1.0 + v / 100.0
        out.append((s.obs[i][0], (f - 1.0) * 100.0))
    return Series(s.sid + "_A12", s.title + " (accum. 12m %)", s.source, "% 12m",
                  s.freq, out, s.ok, s.reason, s.stale_note)


_OLINDA = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/")


def _focus(endpoint: str, params: str, ck: str, ttl: int):
    """One Olinda (Focus survey) query, cached, degrading to stale on failure."""
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            cached = json.loads(_http(_OLINDA + endpoint + "?" + params))
            _cache_put(ck, cached)
        except Exception:
            cached = _cache_get(ck, -1)
    if not isinstance(cached, dict):
        return []
    return cached.get("value") or []


def focus_selic_path(ttl: int = _DEFAULT_TTL) -> list[dict]:
    """The Focus survey's expected Selic AT EACH UPCOMING COPOM MEETING.

    This is the Brazil leg's answer to a STIR strip: it is what the market expects the
    policy rate to be after each decision. It is a SURVEY, not tradeable pricing, and
    every label built on it must say so — it carries no risk premium, it is a median of
    roughly 40-70 forecasters, and it updates weekly rather than continuously.

    `baseCalculo` selects the respondent window: 0 = everyone who answered in the last
    30 days (the basis the weekly Focus bulletin headlines), 1 = the last 5 business
    days only. 0 is used here to match the published bulletin.

    Returns [{"code": "R6/2026", "meeting": date|None, "median": %, "mean": %,
              "n": respondents, "asof": survey date}], ascending by meeting.
    """
    rows = _focus("ExpectativasMercadoSelic",
                  "%24top=2000&%24format=json&%24orderby=Data%20desc"
                  "&%24filter=baseCalculo%20eq%200",
                  "focus_selic", ttl)
    if not rows:
        return []
    latest = max((r.get("Data") or "") for r in rows)
    cal = copom_decisions(ttl=ttl)
    out = []
    for r in rows:
        if (r.get("Data") or "") != latest:
            continue
        code = (r.get("Reuniao") or "").strip()
        med = _num(r.get("Mediana"))
        if not code or med is None:
            continue
        out.append({"code": code, "meeting": copom_meeting_date(code, cal),
                    "median": med, "mean": _num(r.get("Media")),
                    "n": r.get("numeroRespondentes"), "asof": latest})
    # Undated meetings (a survey horizon beyond the published calendar) sort last rather
    # than crashing the comparison — the page simply has nothing to plot them against.
    out.sort(key=lambda d: (d["meeting"] is None, d["meeting"] or date.max, d["code"]))
    return out


def focus_inflation_12m(ttl: int = _DEFAULT_TTL) -> dict | None:
    """Focus 12-month-ahead IPCA expectation (smoothed), the forward-looking variable the
    BCB's own reaction function is built around.

    The BCB targets inflation on a forward horizon, so a rule fed REALISED inflation is
    describing a committee that does not exist. This is the input that makes a Brazil
    rule faithful, and it has no free equivalent for the Fed, ECB or BoE.

    "Suavizada" (smoothed) is the BCB's own published smoothing across the turn of the
    calendar year; it is the series the Focus bulletin reports for the 12-month horizon.
    """
    rows = _focus("ExpectativasMercadoInflacao12Meses",
                  "%24top=800&%24format=json&%24orderby=Data%20desc"
                  "&%24filter=Indicador%20eq%20%27IPCA%27%20and%20Suavizada%20eq%20%27S%27"
                  "%20and%20baseCalculo%20eq%200",
                  "focus_ipca12m", ttl)
    for r in rows:                       # already newest-first
        med = _num(r.get("Mediana"))
        if med is not None:
            return {"median": med, "mean": _num(r.get("Media")),
                    "n": r.get("numeroRespondentes"), "asof": r.get("Data")}
    return None


def copom_decisions(ttl: int = 24 * 3600) -> list[date]:
    """Copom DECISION dates from the BCB's published calendar.

    Two things to know about the feed:

    * Each meeting is TWO consecutive rows (day one and day two). The rate decision is
      announced on the **second** day, so consecutive pairs are collapsed to their later
      date. Treating every row as a meeting doubles the calendar.
    * www.bcb.gov.br answers unknown resources with HTTP 200 and a SharePoint error
      page, so the status code proves nothing — the payload shape is checked instead.

    Cross-checked three ways when this was written (2026-08-27): the minutes feed's
    280th meeting (5 Aug 2026) is the second day of the calendar's 4-5 Aug pair, and
    SGS 432 carries the Selic target forward to 16 Sep 2026, the next decision date.
    """
    y = date.today().year
    lo, hi = f"{y - 1}-01-01", f"{y + 2}-12-31"
    url = ("https://www.bcb.gov.br/api/servico/sitebcb/calendario"
           f"?inicioAgenda=%27{lo}%27&fimAgenda=%27{hi}%27"
           "&lista=" + urllib.parse.quote("Reuniões do Copom"))
    ck = f"copom_cal_{lo}_{hi}"
    cached = _cache_get(ck, ttl)
    if cached is None:
        try:
            blob = json.loads(_http(url))
            cached = blob.get("conteudo") if isinstance(blob, dict) else None
            if not isinstance(cached, list) or not cached:
                raise ValueError("unexpected payload — BCB soft-404?")
            _cache_put(ck, cached)
        except Exception:
            cached = _cache_get(ck, -1)
    if not isinstance(cached, list):
        return []
    days = set()
    for row in cached:
        raw = (row or {}).get("dataEvento") or ""
        try:
            days.add(datetime.strptime(raw[:10], "%Y-%m-%d").date())
        except ValueError:
            continue
    # Collapse each two-day meeting to its decision (second) day.
    out, seq = [], sorted(days)
    for i, d in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if nxt is not None and (nxt - d).days == 1:
            continue                      # day one — the decision is tomorrow's row
        out.append(d)
    return out


def copom_meeting_date(code: str, calendar: list[date] | None = None) -> date | None:
    """Focus meeting code ('R6/2026' = the 6th Copom meeting of 2026) -> decision date.

    The Focus survey identifies meetings by ordinal within the year, never by date, so
    this join against the published calendar is the only way to put a survey
    expectation on a timeline. Returns None when the code runs past the published
    calendar (Focus quotes two years further out than the BCB schedules)."""
    m = re.match(r"^R(\d+)/(\d{4})$", (code or "").strip())
    if not m:
        return None
    n, yr = int(m.group(1)), int(m.group(2))
    cal = calendar if calendar is not None else copom_decisions()
    in_year = [d for d in cal if d.year == yr]
    return in_year[n - 1] if 0 < n <= len(in_year) else None


def br_inputs(ttl: int = _DEFAULT_TTL) -> dict:
    """Brazil inputs for the rule engine.

    Better covered than the UK: inflation, the policy rate, unemployment and the
    inflation TARGET are all published series. Weaker than the US in the one place that
    matters most for the rules — nobody publishes a Brazilian NAIRU or r*, so both are
    assumptions, exactly as for the BoE, and the unemployment gap they imply dominates
    the prescription. The page must say so.
    """
    return {
        "bloc": "BR", "bank": "BCB",
        "headline_infl": sgs(SGS_IDS["ipca_yoy"], title="IPCA (accum. 12m)", ttl=ttl),
        # BCB's trimmed-mean smoothed core, published monthly and compounded here.
        "core_infl": accum12(sgs(SGS_IDS["core_mom"],
                                 title="IPCA trimmed-mean core (smoothed)", ttl=ttl)),
        "unemp": sgs(SGS_IDS["unemp"], title="Unemployment rate (PNAD Contínua)", ttl=ttl),
        # The POLICY lever is the Copom's target, not the effective daily Selic — and it
        # is future-dated (see sgs()), so every read must go through .asof().
        "policy": sgs(SGS_IDS["selic_target"], title="Selic target (Copom)", ttl=ttl),
        "selic_effective": sgs(SGS_IDS["selic_daily"],
                               title="Selic effective (annualised)", ttl=ttl),
        "activity": sgs(SGS_IDS["ibcbr"], title="IBC-Br activity index (SA)", ttl=ttl),
        "rstar": _fail("rstar/BR", "no published Brazilian r* — the BCB discusses a "
                                   "neutral real rate in Inflation Report boxes but "
                                   "publishes no series; user-set assumption", "n/a"),
        # Published, unlike the 2.0 the other legs hardcode: the CMN sets it by decree.
        "target": _target_br(ttl),
        "focus_12m": focus_inflation_12m(ttl=ttl),
    }


def _target_br(ttl: int) -> float:
    """Brazil's inflation target in force today (CMN, SGS 13521); 3.0 if unreachable."""
    s = sgs(SGS_IDS["infl_target"], title="CMN inflation target", ttl=ttl)
    v = s.latest_actual if s.ok else None
    return float(v) if v is not None else 3.0


# ── convenience bundles the rule engine consumes ────────────────────────────────────
def us_inputs(ttl: int = _DEFAULT_TTL) -> dict:
    """Everything the Fed rules need. US is the best-covered bloc by a distance: CBO
    publishes both potential GDP and the natural rate of unemployment, so the output gap
    is a real published gap rather than a proxy."""
    core = fred("PCEPILFE", ttl=ttl)
    return {
        "bloc": "US", "bank": "FED",
        "core_infl": core.yoy(),
        "headline_infl": fred("PCEPI", ttl=ttl).yoy(),
        "core_cpi": fred("CPILFESL", ttl=ttl).yoy(),
        "trimmed": fred("PCETRIM12M159SFRBDAL", ttl=ttl),
        "gdp": fred("GDPC1", ttl=ttl),
        "potential": fred("GDPPOT", ttl=ttl),
        "unemp": fred("UNRATE", ttl=ttl),
        "nairu": fred("NROU", ttl=ttl),
        "policy": fred("EFFR", start="2000-01-01", ttl=ttl),
        "rstar": hlw("rstar", "US"),
        "gdpnow": fred("GDPNOW", start="2011-01-01", ttl=ttl),
        "nowcast": cleveland_nowcast(),
        "be5y5y": fred("T5YIFR", start="2010-01-01", ttl=ttl),
        "target": 2.0,
    }


def ea_inputs(ttl: int = _DEFAULT_TTL) -> dict:
    """Euro-area inputs. No CBO equivalent, so the output gap leans on HLW's own EA gap
    (stale but model-consistent) — flagged as such on the page."""
    return {
        "bloc": "EA", "bank": "ECB",
        # Inflation from EUROSTAT, not the ECB portal — the ECB's ICP dataflow died at
        # 2025-12 with the ECOICOP ver.2 migration. See ea_hicp().
        "core_infl": ea_hicp("core", ttl=ttl),
        "headline_infl": ea_hicp("headline", ttl=ttl),
        # The ECB portal is still the right source for the policy rate and unemployment;
        # both were confirmed live (depo 2026-08-15, unemployment 2026-06).
        "policy": ecb("FM", "D.U2.EUR.4F.KR.DFR.LEV",
                      title="ECB deposit facility rate", ttl=ttl),
        "unemp": ecb("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T",
                     title="Euro-area unemployment rate", ttl=ttl),
        "rstar": hlw("rstar", "EA"),
        "gap": hlw("gap", "EA"),
        "target": 2.0,
    }


def uk_inputs(ttl: int = _DEFAULT_TTL) -> dict:
    """UK inputs. The weakest of the three legs by construction: there is no HLW UK r*
    and the ONS does not publish a potential-output series, so r* is a user-set
    assumption and the output gap is an Okun proxy off the unemployment gap. Both are
    surfaced as editable inputs rather than pretending to be measured data."""
    return {
        "bloc": "UK", "bank": "BOE",
        "headline_infl": ons("D7G7", title="UK CPI (y/y)", ttl=ttl),
        "core_infl": ons("DKO8", title="UK core CPI (y/y)", ttl=ttl),
        "unemp": ons("MGSX", title="UK unemployment rate", ttl=ttl),
        # ONS publishes AWE as a LEVEL (£/week); the pay-growth rate the MPC actually
        # watches is derived here rather than chased through another series code.
        "wages": ons("KAB9", title="UK average weekly earnings", ttl=ttl).yoy(),
        "policy": boe("IUDBEDR", title="Bank Rate", ttl=ttl),
        "rstar": _fail("HLW/rstar/UK", "HLW publishes no UK estimate — r* is a "
                                       "user-set assumption for the BoE leg", "n/a"),
        "target": 2.0,
    }


BLOC_INPUTS = {"FED": us_inputs, "ECB": ea_inputs, "BOE": uk_inputs, "BCB": br_inputs}


def source_status() -> list[dict]:
    """One row per source for the Data-health board: is it up, and how old is its latest
    observation? Deliberately cheap — it reads the cache where it can."""
    rows = []

    def add(name, s: Series, expect_days: int):
        if not s.ok:
            rows.append({"source": name, "ok": False, "detail": s.reason, "age": None})
            return
        age = (date.today() - s.latest_date).days if s.latest_date else None
        rows.append({"source": name, "ok": True, "age": age,
                     "detail": (f"latest {s.latest_date} ({age}d old)" if age is not None
                                else "no observations"),
                     "stale": age is not None and age > expect_days})

    add("FRED — core PCE", fred("PCEPILFE"), 60)
    add("Eurostat — euro-area HICP", ea_hicp("headline"), 60)
    add("ECB — deposit facility rate", ecb("FM", "D.U2.EUR.4F.KR.DFR.LEV", n=10), 7)
    add("BoE — Bank Rate", boe("IUDBEDR", start="01/Jan/2026"), 7)
    add("ONS — UK CPI", ons("D7G7"), 75)
    add("NY Fed — HLW r*", hlw("rstar", "US"), 300)
    nc = cleveland_nowcast()
    rows.append({"source": "Cleveland Fed — nowcast", "ok": bool(nc.get("ok")),
                 "detail": (f"{nc.get('month')} core PCE {nc.get('core_pce')}% m/m, "
                            f"updated {nc.get('updated')}") if nc.get("ok")
                           else nc.get("reason", "unavailable"),
                 "age": None})
    return rows
