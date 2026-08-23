"""golddata.py — the input layer for the 🥇 Gold Driver Model.

One aligned daily frame carrying every driver in the gold framework, so the
model in goldmodel.py never makes a network call and every feature has a single
documented provenance.

Design rules, learned the hard way elsewhere in this repo:

* FREE FIRST. The only Bloomberg dependency is the gold price itself, and that
  comes off the deep panama-adjusted store which is already on disk — so the
  whole model rebuilds on a day the Terminal is wedged.
* NO MOCKS. pmdata legitimately falls back to synthetic blocks so a client PDF
  can still render; a predictive model must never do that — a mock feature would
  fit and forecast exactly like a real one and nothing downstream would notice.
  Every fetcher here returns real data, the last good cache, or nothing, and says
  which in `status`.
* CACHE EVERYTHING. Each source writes its own parquet under data/signals/, so a
  dead source degrades to "stale by N days" rather than blanking the model.

Sources (every one verified reachable 2026-08-22)
------------------------------------------------
    deep store       data/price_store          GCA/SIA panama-adjusted + front2
    FRED             api.stlouisfed.org        FREE KEY  real yields, dollar, VIX, OAS
    SPDR Gold Trust  api.spdrgoldshares.com    no key    daily tonnes back to 2004
    LBMA             prices.lbma.org.uk        no key    London fix USD/GBP/EUR since 1968
    SGE              sge.com.cn/graph/Dailyhq  no key    Au99.99 CNY/g since 2016-12
    Yahoo            yfinance                  no key    DXY, GDX/GLD, USDCNY/JPY/INR
    CFTC             data/signals/cot_history  no key    managed money since 2006
    IMF              api.imf.org (SDMX)        no key    official gold holdings, monthly

Gotchas
-------
* FRED's DTWEXBGS publishes with about a week's lag — fine as a slow structural
  read, useless for timing. The tradeable dollar here is Yahoo's DX-Y.NYB, which
  is same-day. Both are carried; the model uses the timely one.
* The Shanghai premium is a CLOSE-vs-FIX comparison across time zones: SGE's day
  session ends ~08:30 London, the LBMA AM fix is 10:30, the PM fix 15:00. On a
  fast day the raw spread can print $100 of pure timing. Hence the AM fix (not
  PM) is the reference leg and the result is smoothed — see `sge_premium`.
* SPDR serves the archive as .xlsx from an api. host that 301s off the old CSV
  path; follow redirects and read bytes, never assume CSV.
* FRED serves only a rolling ~3-year window of the ICE BofA credit indices
  (BAMLH0A0HYM2 and friends) — silently, with no error. Moody's BAA10Y is the
  long-history credit leg here; see FRED_IDS.

CLI:  python src/golddata.py [--force] [--summary]
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

DATA = _ROOT / "data"
SIGNALS = DATA / "signals"
SIGNALS.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = SIGNALS / "gold_features.parquet"
STATUS_FILE = SIGNALS / "gold_features_status.json"
GLD_FILE = SIGNALS / "gold_gld.parquet"
LBMA_FILE = SIGNALS / "gold_lbma.parquet"
SGE_FILE = SIGNALS / "gold_sge.parquet"
YF_FILE = SIGNALS / "gold_yahoo.parquet"
FRED_FILE = SIGNALS / "gold_fred.parquet"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/127.0 Safari/537.36")
_TIMEOUT = 90
OZ_PER_GRAM = 31.1034768

# 1990, not 2010. The original 2010 cap was a guess that quietly became the binding
# constraint on the whole project: it left a 6.2-year evaluable window (~27 independent
# 60-day observations), on which nothing can be validated. The sources go far deeper —
# DXY to 1971, S&P to 1970, LBMA to 1968, VIX to 1990, Baa to 1986 — and the genuine
# limits are elsewhere (TIPS 2003, GLD 2004, COT 2006, SGE 2016).
START = "1990-01-01"


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def _http(url: str, data: bytes | None = None) -> bytes:
    headers = {"User-Agent": _UA}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return r.read()


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    return (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0 <= max_age_hours


def _cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _note(status: dict, key: str, df, live: bool, src: str, detail: str = "") -> None:
    """Record one block's provenance so the report can be honest about staleness."""
    last = None
    if df is not None and len(df):
        last = pd.Timestamp(df.index.max()).date().isoformat()
    status[key] = {"ok": df is not None and len(df) > 0, "live": live,
                   "source": src, "last": last, "detail": detail}


# ---------------------------------------------------------------------------
# 1. gold + metals price — the deep panama-adjusted store (Bloomberg-sourced,
#    already on disk). get_adjusted is the right frame for the target: it is the
#    only one whose returns are tradeable across a roll.
# ---------------------------------------------------------------------------
def metals_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(adjusted closes, raw closes + 2nd month). Raw is kept because the EFP and
    the carry must be measured on levels the market actually printed, not on
    roll-adjusted ones."""
    from src import deepstore
    tkrs = ["GCA Comdty", "SIA Comdty", "PLA Comdty", "PAA Comdty", "HGA Comdty"]
    adj = deepstore.get_adjusted(tkrs)
    raw = deepstore.get_raw(tkrs)
    f2 = deepstore.get_front2(["GCA Comdty"])
    raw = raw.join(f2["GCA Comdty"].rename("GC_second"), how="outer")
    return adj, raw


# ---------------------------------------------------------------------------
# 2. FRED — real yields, dollar, breakevens, risk. One keyed source covers the
#    whole "cost of holding it / fear of holding something else" axis.
# ---------------------------------------------------------------------------
FRED_IDS = {
    "real_10y": "DFII10",        # 10y TIPS — the single most important driver
    "real_5y": "DFII5",
    "breakeven_10y": "T10YIE",
    "nominal_10y": "DGS10",
    "nominal_2y": "DGS2",
    "fed_funds": "DFF",
    "dollar_broad": "DTWEXBGS",  # ~1wk publication lag; structural read only
    "vix": "VIXCLS",
    # Credit stress. BAA10Y (Moody's Baa over the 10y) is the PRIMARY leg because
    # it runs to 1986 for free. The ICE BofA high-yield OAS is the spread everyone
    # quotes, but FRED only serves a ROLLING ~3-YEAR window of the ICE BofA family
    # under licence — asking for observation_start=2010 silently returns 2023
    # onward, no error, no warning. Building a feature on it costs 70% of the
    # sample. It is carried as a secondary read, never as a model input.
    "credit_baa": "BAA10Y",
    "hy_oas": "BAMLH0A0HYM2",
    "fwd_5y5y": "T5YIFR",
}


def fred_block(force: bool = False, max_age_hours: float = 12.0) -> tuple[pd.DataFrame, bool]:
    if not force and _is_fresh(FRED_FILE, max_age_hours):
        c = _cached(FRED_FILE)
        if c is not None:
            return c, True
    from src import macrodata
    cols = {}
    for name, sid in FRED_IDS.items():
        try:
            s = macrodata.fred(sid, start=START)
            if getattr(s, "obs", None):
                cols[name] = pd.Series({pd.Timestamp(d): v for d, v in s.obs})
        except Exception:
            continue
    if not cols:
        c = _cached(FRED_FILE)
        return (c, False) if c is not None else (pd.DataFrame(), False)
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(FRED_FILE)
    return df, True


# ---------------------------------------------------------------------------
# 3. SPDR Gold Trust — daily tonnage, the Western-investor flow read.
#    The old public CSV path 301s to an .xlsx on api.spdrgoldshares.com; that
#    file carries the whole history back to the 2004 launch, so a pull rebuilds
#    the entire series rather than appending.
# ---------------------------------------------------------------------------
SPDR_URL = ("https://api.spdrgoldshares.com/api/v1/historical-archive"
            "?product=gld&exchange=NYSE&lang=en")


def spdr_gld(force: bool = False, max_age_hours: float = 20.0) -> tuple[pd.DataFrame, bool]:
    """Daily GLD tonnes. Closed days read the literal string 'US Holiday' across
    every column — to_numeric turns those into NaN and they are dropped, never
    forward-filled at source (the alignment step owns that decision)."""
    if not force and _is_fresh(GLD_FILE, max_age_hours):
        c = _cached(GLD_FILE)
        if c is not None:
            return c, True
    try:
        raw = pd.read_excel(io.BytesIO(_http(SPDR_URL)),
                            sheet_name="US GLD Historical Archive", header=0)
        raw.columns = [str(c).strip() for c in raw.columns]
        d = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], format="%d-%b-%Y", errors="coerce"),
            "gld_tonnes": pd.to_numeric(raw["Tonnes of Gold"], errors="coerce"),
        }).dropna().set_index("date").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if d.empty:
            raise RuntimeError("SPDR archive parsed empty")
        d.to_parquet(GLD_FILE)
        return d, True
    except Exception:
        c = _cached(GLD_FILE)
        return (c, False) if c is not None else (pd.DataFrame(), False)


# ---------------------------------------------------------------------------
# 4. LBMA — the London benchmark, and gold in GBP/EUR for the currency
#    cross-check ("is this a gold move or a dollar move?").
# ---------------------------------------------------------------------------
LBMA_AM = "https://prices.lbma.org.uk/json/gold_am.json"
LBMA_PM = "https://prices.lbma.org.uk/json/gold_pm.json"
LBMA_SILVER = "https://prices.lbma.org.uk/json/silver.json"


def _lbma_one(url: str, tag: str) -> pd.DataFrame:
    rows = json.loads(_http(url))
    out = {}
    for r in rows:
        v = r.get("v") or []
        try:
            d = pd.Timestamp(r["d"])
        except Exception:
            continue
        out[d] = [v[i] if len(v) > i else np.nan for i in range(3)]
    return pd.DataFrame.from_dict(
        out, orient="index",
        columns=[f"lbma_{tag}_usd", f"lbma_{tag}_gbp", f"lbma_{tag}_eur"]).sort_index()


def lbma(force: bool = False, max_age_hours: float = 20.0) -> tuple[pd.DataFrame, bool]:
    if not force and _is_fresh(LBMA_FILE, max_age_hours):
        c = _cached(LBMA_FILE)
        if c is not None:
            return c, True
    try:
        df = (_lbma_one(LBMA_AM, "am")
              .join(_lbma_one(LBMA_PM, "pm"), how="outer")
              # Silver from the same benchmark family. The deep store's SIA contract
              # only reaches 2016, which capped the gold/silver ratio — a 5y z-score
              # of a series that starts in 2016 has barely two usable years. LBMA
              # silver runs to 1968 and is free.
              .join(_lbma_one(LBMA_SILVER, "silver"), how="outer"))
        # 1990, not 2005 — another self-imposed cap. The LBMA feed carries the fix
        # back to 1968; there is no reason to throw away fifteen years of the target
        # series and the only long-history silver leg we have.
        df = df[df.index >= pd.Timestamp(START)].astype(float)
        if df.empty:
            raise RuntimeError("LBMA parsed empty")
        df.to_parquet(LBMA_FILE)
        return df, True
    except Exception:
        c = _cached(LBMA_FILE)
        return (c, False) if c is not None else (pd.DataFrame(), False)


# ---------------------------------------------------------------------------
# 5. Shanghai Gold Exchange — Au99.99, CNY per gram, daily since 2016-12. The
#    premium over London is the cleanest public read on Chinese physical
#    appetite, and this rebuilds free (with history) what the desk priced off a
#    Bloomberg CIX that only resolves under one login.
# ---------------------------------------------------------------------------
SGE_URL = "https://www.sge.com.cn/graph/Dailyhq"


def sge(force: bool = False, max_age_hours: float = 20.0) -> tuple[pd.DataFrame, bool]:
    if not force and _is_fresh(SGE_FILE, max_age_hours):
        c = _cached(SGE_FILE)
        if c is not None:
            return c, True
    try:
        raw = json.loads(_http(SGE_URL, data=b"instid=Au99.99"))["time"]
        out = {}
        for r in raw:
            try:
                d, close = pd.Timestamp(r[0]), float(r[1])
            except Exception:
                continue
            if close > 0:                       # the feed pads unlisted dates with zeros
                out[d] = close
        df = pd.DataFrame({"sge_cny_g": pd.Series(out)}).sort_index()
        if df.empty:
            raise RuntimeError("SGE parsed empty")
        df.to_parquet(SGE_FILE)
        return df, True
    except Exception:
        c = _cached(SGE_FILE)
        return (c, False) if c is not None else (pd.DataFrame(), False)


# ---------------------------------------------------------------------------
# 6. Yahoo — the timely dollar, the miners/bullion cross-check, and the FX legs
#    the Shanghai premium and the gold-in-currency reads need same-day.
# ---------------------------------------------------------------------------
YF_SYMBOLS = {
    "dxy": "DX-Y.NYB",
    "gld_px": "GLD",
    "gdx_px": "GDX",
    "usdcny": "CNY=X",
    "usdjpy": "JPY=X",
    "usdinr": "INR=X",
    "eurusd": "EURUSD=X",
    # Risk-appetite cross-check (spec §2.5). FRED's SP500 is licence-limited to a
    # rolling 10 years; ^GSPC runs to 1927 and is free.
    "spx": "^GSPC",
}


def yahoo_block(force: bool = False, max_age_hours: float = 12.0) -> tuple[pd.DataFrame, bool]:
    if not force and _is_fresh(YF_FILE, max_age_hours):
        c = _cached(YF_FILE)
        if c is not None:
            return c, True
    try:
        import yfinance as yf
        raw = yf.download(list(YF_SYMBOLS.values()), start=START, progress=False,
                          auto_adjust=True, threads=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        df = pd.DataFrame({name: close[sym] for name, sym in YF_SYMBOLS.items()
                           if sym in close.columns}).sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(how="all")
        if df.empty:
            raise RuntimeError("Yahoo returned nothing")
        df.to_parquet(YF_FILE)
        return df, True
    except Exception:
        c = _cached(YF_FILE)
        return (c, False) if c is not None else (pd.DataFrame(), False)


# ---------------------------------------------------------------------------
# 7. CFTC managed money — read straight off the COT archive the app already
#    maintains (weekly, back to 2006). Positioning is a TIMING input: an extreme
#    says how crowded the trade is, not which way it goes next.
# ---------------------------------------------------------------------------
COT_HIST = SIGNALS / "cot_history.parquet"


def cot_gold() -> tuple[pd.DataFrame, bool]:
    try:
        h = pd.read_parquet(COT_HIST)
        g = h[h["ticker"] == "GCA Comdty"].copy()
        if g.empty:
            return pd.DataFrame(), False
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date").sort_index()
        from src import cotdata
        out = pd.DataFrame({"mm_net": g["net"].astype(float)})
        out["cot_index"] = cotdata.cot_index(out["mm_net"])
        return out, True
    except Exception:
        return pd.DataFrame(), False


# ---------------------------------------------------------------------------
# 8. Central banks — IMF reported official holdings (monthly, ~6wk lag, and it
#    understates true buying since China reports partially and Russia not at
#    all). Slow floor, not a timing signal; carried as a 12m trailing sum so one
#    reporting gap cannot whipsaw it.
#
#    This pulls the IMF directly rather than reusing pmdata.fetch_central_banks:
#    that one asks for a fixed 4-year window, which is right for a monthly client
#    monitor and far too short here — a 12m trailing sum off a 4y window leaves
#    ~3y of usable feature, and the whole point is to see the post-2022 official
#    bid against the decade before it.
# ---------------------------------------------------------------------------
CB_FILE = SIGNALS / "gold_cb.parquet"
IMF_URL = ("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IL/"
           "GX010.RGV_REVS..M?startPeriod={start}")
_OZT_TO_T = 31.1034768 / 1e6


def central_banks(force: bool = False, max_age_hours: float = 72.0) -> tuple[pd.Series, bool]:
    if not force and _is_fresh(CB_FILE, max_age_hours):
        c = _cached(CB_FILE)
        if c is not None:
            return c["tonnes"].diff().dropna().rename("cb_net_tonnes"), True
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(_http(IMF_URL.format(start=START[:7])))
        obs = {}
        for el in root.iter():
            if el.tag.endswith("Obs"):
                p = el.attrib["TIME_PERIOD"].replace("-M", "-")
                obs[pd.Period(p, freq="M").to_timestamp(how="end").normalize()] =                     float(el.attrib["OBS_VALUE"]) * _OZT_TO_T
        hold = pd.Series(obs).sort_index()
        if hold.empty:
            raise RuntimeError("IMF IL returned no observations")
        pd.DataFrame({"tonnes": hold}).to_parquet(CB_FILE)
        return hold.diff().dropna().rename("cb_net_tonnes"), True
    except Exception:
        c = _cached(CB_FILE)
        if c is not None:
            return c["tonnes"].diff().dropna().rename("cb_net_tonnes"), False
        return pd.Series(dtype=float), False


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def sge_premium(sge_df: pd.DataFrame, lbma_df: pd.DataFrame,
                usdcny: pd.Series, smooth: int = 5) -> pd.Series:
    """Shanghai premium over London, $/oz.

    The reference leg is the LBMA **AM** fix, not PM. SGE's day session closes
    ~08:30 London, the AM fix is 10:30, the PM fix 15:00. Measured against PM the
    series carries a whole London session of drift that has nothing to do with
    Chinese demand — on a 2% day that is ~$90 of noise on a signal whose real
    range is tens of dollars. Even against AM the mismatch is real, hence the
    5-day smooth."""
    if sge_df.empty or lbma_df.empty or usdcny is None or usdcny.empty:
        return pd.Series(dtype=float)
    cny = usdcny.reindex(sge_df.index).ffill()
    sh_usd_oz = sge_df["sge_cny_g"] * OZ_PER_GRAM / cny
    london = lbma_df["lbma_am_usd"].reindex(sh_usd_oz.index).ffill()
    prem = (sh_usd_oz - london).dropna()
    return prem.rolling(smooth, min_periods=max(2, smooth // 2)).mean().rename("sge_premium")


def build(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """The aligned daily driver frame + a provenance/staleness dict.

    Everything lands on a business-day index and is forward-filled, because the
    inputs update on different clocks (daily prices, weekly COT, monthly central
    banks). Forward-fill is the honest choice: on a Wednesday, the last COT print
    IS what the market knows. Nothing is back-filled — that would leak the
    future into the fit."""
    status: dict = {}

    adj, raw = metals_prices()
    _note(status, "prices", adj, True, "deep store (panama-adjusted)")

    fred, fred_live = fred_block(force=force)
    _note(status, "fred", fred, fred_live, "FRED", ", ".join(FRED_IDS.values()))

    gld, gld_live = spdr_gld(force=force)
    _note(status, "spdr_gld", gld, gld_live, "SPDR Gold Trust archive")

    lb, lb_live = lbma(force=force)
    _note(status, "lbma", lb, lb_live, "LBMA benchmark fix")

    sg, sg_live = sge(force=force)
    _note(status, "sge", sg, sg_live, "Shanghai Gold Exchange Au99.99")

    yh, yh_live = yahoo_block(force=force)
    _note(status, "yahoo", yh, yh_live, "Yahoo Finance", ", ".join(YF_SYMBOLS.values()))

    cot, cot_live = cot_gold()
    _note(status, "cot", cot, cot_live, "CFTC disaggregated / managed money")

    cb, cb_live = central_banks(force=force)
    _note(status, "central_banks", cb, cb_live, "IMF international liquidity")

    idx = pd.bdate_range(START, pd.Timestamp.today().normalize())
    F = pd.DataFrame(index=idx)

    # --- target + metals ---------------------------------------------------
    F["gold"] = adj["GCA Comdty"].reindex(idx).ffill()
    F["gold_raw"] = raw["GCA Comdty"].reindex(idx).ffill()
    F["silver"] = adj["SIA Comdty"].reindex(idx).ffill()
    F["copper"] = adj["HGA Comdty"].reindex(idx).ffill()
    F["gold_silver_ratio"] = F["gold_raw"] / raw["SIA Comdty"].reindex(idx).ffill()

    # --- real rates / Fed --------------------------------------------------
    for c in fred.columns:
        F[c] = fred[c].reindex(idx).ffill()
    if {"nominal_2y", "fed_funds"} <= set(F.columns):
        # Market-priced Fed path: the 2y yield under the funds rate = cuts priced
        # in. fedpath.py backs the same read out of the live SR3 strip but keeps
        # no history, and this proxy runs to 2010 for free.
        F["cuts_priced"] = F["nominal_2y"] - F["fed_funds"]

    # --- dollar ------------------------------------------------------------
    for c in ("dxy", "usdcny", "usdjpy", "usdinr", "eurusd"):
        if c in yh.columns:
            F[c] = yh[c].reindex(idx).ffill()

    # --- flows / positioning ----------------------------------------------
    if not gld.empty:
        F["gld_tonnes"] = gld["gld_tonnes"].reindex(idx).ffill()
    if not cot.empty:
        F["mm_net"] = cot["mm_net"].reindex(idx).ffill()
        F["cot_index"] = cot["cot_index"].reindex(idx).ffill()

    # --- physical ----------------------------------------------------------
    if not lb.empty:
        F["lbma_usd"] = lb["lbma_pm_usd"].reindex(idx).ffill()
        F["gold_eur"] = lb["lbma_pm_eur"].reindex(idx).ffill()
        F["gold_gbp"] = lb["lbma_pm_gbp"].reindex(idx).ffill()
        # EFP proxy: COMEX front over the London fix. Widening = futures bid
        # relative to metal, the classic squeeze tell.
        #
        # This one MUST be differenced before the forward-fill, on dates where
        # both legs genuinely printed. The store runs a day behind the LBMA feed
        # whenever the overnight pull has not landed; differencing the ffilled
        # columns then books yesterday's COMEX against today's fix and invents an
        # EFP swing the size of the day's move — $66 of pure staleness on
        # 2026-08-21. Align on the intersection first, ffill the result after.
        efp = (raw["GCA Comdty"].dropna()
               .to_frame("comex").join(lb["lbma_pm_usd"].dropna().rename("fix"),
                                       how="inner"))
        F["efp"] = (efp["comex"] - efp["fix"]).reindex(idx).ffill()
    if "GC_second" in raw.columns:
        # Contango in $/oz. Compresses (or inverts) when metal is scarce, which
        # is the lease-rate story no free source sells directly.
        F["gc_carry"] = (raw["GC_second"] - raw["GCA Comdty"]).reindex(idx).ffill()
    prem = sge_premium(sg, lb, F["usdcny"] if "usdcny" in F.columns else None)
    if not prem.empty:
        F["sge_premium"] = prem.reindex(idx).ffill()
    if not cb.empty:
        F["cb_net_12m"] = cb.rolling(12, min_periods=6).sum().reindex(idx).ffill()

    # --- cross-checks ------------------------------------------------------
    if {"gdx_px", "gld_px"} <= set(yh.columns):
        F["miners_vs_bullion"] = (yh["gdx_px"] / yh["gld_px"]).reindex(idx).ffill()
    if {"lbma_usd", "usdinr"} <= set(F.columns):
        F["gold_inr"] = F["lbma_usd"] * F["usdinr"]
    if {"lbma_usd", "usdjpy"} <= set(F.columns):
        F["gold_jpy"] = F["lbma_usd"] * F["usdjpy"]

    F = F.dropna(subset=["gold"])
    F.to_parquet(FEATURES_FILE)
    status["_built"] = datetime.now().isoformat(timespec="seconds")
    status["_rows"] = int(len(F))
    status["_cols"] = list(F.columns)
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return F, status


def load() -> tuple[pd.DataFrame, dict]:
    """Read the cached frame; build it if it has never been made."""
    if not FEATURES_FILE.exists():
        return build()
    F = pd.read_parquet(FEATURES_FILE)
    F.index = pd.to_datetime(F.index)
    st = json.loads(STATUS_FILE.read_text(encoding="utf-8")) if STATUS_FILE.exists() else {}
    return F, st


def main() -> int:
    F, st = build(force="--force" in sys.argv)
    print(f"gold_features.parquet  rows={len(F)}  "
          f"{F.index.min().date()} -> {F.index.max().date()}")
    for k, v in st.items():
        if k.startswith("_"):
            continue
        flag = "LIVE " if v["live"] else ("cache" if v["ok"] else "DEAD ")
        print(f"  {flag} {k:14s} last={v['last']}  {v['source']}")
    if "--summary" in sys.argv:
        print()
        print(F.tail(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
