"""goldingest.py — Milestone 2: every source mapped into the point-in-time store.

One job: take the feeds golddata already fetches, plus the ones recovered during the
gap review, and land them in goldstore with an honest `published_at` on every row.
Nothing here computes a feature; nothing here reads the observations table. The model
layer talks to `goldstore.get_series(series_id, as_of)` and to nothing else.

Publication timestamps, by tier
-------------------------------
    VINTAGE  FRED/ALFRED — the real revision history, one call per series. A macro
             series ingested this way carries every restatement it ever had.
    EXACT    Market data — LBMA fixes, SGE closes, GLD tonnage, Yahoo, the deep
             store. Final when printed. published_at = the close itself.
    DERIVED  COT — the CFTC publishes only the Tuesday reference date, so the Friday
             15:30 ET release is reconstructed. See `cot_published_at`.
    LAGGED   IMF central bank reserves — approximate, flagged as such.

The COT leak this fixes
-----------------------
`cot_history.parquet` stores the Tuesday reference date and nothing else. Any
backtest that read it on the Tuesday — or Wednesday, or Thursday — was trading on a
report the market would not see until Friday afternoon: three days of look-ahead on a
positioning signal, every single week, for the whole sample. That is not a rounding
error, it is the difference between a strategy and a time machine. Every COT row
ingested here carries a real release timestamp instead.

CLI:
    python src/goldingest.py --all          everything (slow first run: ALFRED + Yahoo)
    python src/goldingest.py --fred         macro only
    python src/goldingest.py --market       prices/flows only
    python src/goldingest.py --cot          positioning only
    python src/goldingest.py --cb           IMF central bank reserves
    python src/goldingest.py --status       coverage + horizon matrix, no fetching
"""
from __future__ import annotations

import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import golddata, goldstore  # noqa: E402

OZT_TO_TONNE = 31.1034768 / 1e6


# ---------------------------------------------------------------------------
# FRED — true point-in-time via ALFRED
# ---------------------------------------------------------------------------
# series_id -> (FRED id, description, unit, freq, typical lag days, bucket)
FRED_SERIES = {
    "REAL_10Y":      ("DFII10", "10y TIPS real yield", "pct", "daily", 1, "monetary"),
    "REAL_5Y":       ("DFII5", "5y TIPS real yield", "pct", "daily", 1, "monetary"),
    "BREAKEVEN_10Y": ("T10YIE", "10y breakeven inflation", "pct", "daily", 1, "monetary"),
    "FWD_5Y5Y":      ("T5YIFR", "5y5y forward breakeven", "pct", "daily", 1, "monetary"),
    "NOMINAL_10Y":   ("DGS10", "10y Treasury yield", "pct", "daily", 1, "monetary"),
    "NOMINAL_2Y":    ("DGS2", "2y Treasury yield", "pct", "daily", 1, "monetary"),
    "FED_FUNDS":     ("DFF", "Effective fed funds rate", "pct", "daily", 1, "monetary"),
    "DOLLAR_BROAD":  ("DTWEXBGS", "Broad dollar index", "index", "daily", 7, "monetary"),
    "VIX":           ("VIXCLS", "VIX", "index", "daily", 1, "risk"),
    "CREDIT_BAA":    ("BAA10Y", "Moody's Baa over 10y", "pct", "daily", 1, "risk"),
    "CPI":           ("CPIAUCSL", "Headline CPI", "index", "monthly", 14, "risk"),
    "CORE_CPI":      ("CPILFESL", "Core CPI", "index", "monthly", 14, "risk"),
    "PCE":           ("PCEPI", "Headline PCE price index", "index", "monthly", 30, "risk"),
    "CORE_PCE":      ("PCEPILFE", "Core PCE price index", "index", "monthly", 30, "risk"),
    "PAYROLLS":      ("PAYEMS", "Nonfarm payrolls", "thousands", "monthly", 7, "risk"),
    "UNEMPLOYMENT":  ("UNRATE", "Unemployment rate", "pct", "monthly", 7, "risk"),
    # PPI Final Demand — the series the market trades, and the one with a 1990
    # history. PPIFIS is the same concept but only starts 2009, too short to compare
    # against the other releases.
    "PPI":           ("WPSFD49207", "PPI final demand", "index", "monthly", 14, "risk"),
    "RETAIL_SALES":  ("RSAFS", "Retail sales", "USD mn", "monthly", 16, "risk"),
    "M2":            ("M2SL", "M2 money supply", "USD bn", "monthly", 30, "valuation"),
    "FED_DEBT":      ("GFDEBTN", "Federal debt outstanding", "USD mn", "quarterly", 60,
                      "valuation"),
}


def ingest_fred(start: str = "1990-01-01", only=None) -> int:
    """Land every FRED series, choosing the right retrieval mode per series.

    ONLY REVISABLE SERIES GO THROUGH ALFRED. Daily market rates — TIPS yields, the
    2y, the VIX, Baa — are never restated: Tuesday's close is Tuesday's close for
    ever. Asking ALFRED for their full vintage matrix requests a grid of ~9,000
    reference dates against ~9,000 vintage columns, and FRED answers with a 400, a
    504, or a 500 depending on how far it got. That is not a transient failure to
    retry around; it is the wrong question. Those series are fetched normally and
    stamped with a one-business-day publication lag, which is exact for them.

    Monthly and quarterly macro — CPI, PCE, payrolls, M2 — genuinely IS revised, so
    those keep the ALFRED path. Payrolls comes back with 6,421 rows for 439 reference
    months: that ratio is the revision history, and it is the whole point.

    A vintage fetch that fails falls back to the plain series with an assumed lag,
    flagged `published_at_approximated=True` so the compromise stays visible."""
    n = 0
    for sid, (fred_id, desc, unit, freq, lag, bucket) in FRED_SERIES.items():
        if only and sid not in only:
            continue
        revisable = freq in ("monthly", "quarterly", "annual")
        v, mode = None, ""
        if revisable:
            try:
                v = goldstore.fred_vintages(fred_id, start=start)
                mode = "ALFRED"
            except Exception as e:
                print(f"  {sid:14s} vintage fetch failed ({str(e)[:40]}) — "
                      f"falling back to latest-only")
                v = None
        if v is None or v.empty:
            from src import macrodata
            try:
                s = macrodata.fred(fred_id, start=start)
            except Exception as e:
                print(f"  {sid:14s} FAILED  {str(e)[:70]}")
                continue
            if not getattr(s, "obs", None):
                print(f"  {sid:14s} no data")
                continue
            ser = pd.Series({pd.Timestamp(d): val for d, val in s.obs}).sort_index()
            goldstore.register(sid, description=desc, unit=unit, native_freq=freq,
                               typical_lag_days=lag, bucket=bucket,
                               source_url=f"https://fred.stlouisfed.org/series/{fred_id}",
                               published_at_approximated=revisable)
            wrote = goldstore.put(sid, ser, source="FRED", typical_lag_days=lag)
            n += wrote
            print(f"  {sid:14s} +{wrote:6d} rows   {ser.index.min().date()} -> "
                  f"{ser.index.max().date()}   [{'lagged' if revisable else 'exact'}]")
            continue

        goldstore.register(sid, description=desc, unit=unit, native_freq=freq,
                           typical_lag_days=lag, bucket=bucket,
                           source_url=f"https://fred.stlouisfed.org/series/{fred_id}",
                           published_at_approximated=False)
        wrote = goldstore.put(sid, v, source="FRED/ALFRED")
        n += wrote
        refs = v["reference_date"].nunique()
        print(f"  {sid:14s} +{wrote:6d} rows over {refs:5d} reference dates   "
              f"[{mode}, {wrote / max(refs, 1):.1f} vintages each]")
    return n


# ---------------------------------------------------------------------------
# Market data — final when printed, so published_at is the observation itself
# ---------------------------------------------------------------------------
# Series whose day-t value is NOT knowable at the London 15:00 PM fix, which is
# when the target return is struck (goldfeatures.TARGET_PRICE = LBMA_GOLD_PM_USD).
# SPDR posts day-t tonnage after the 16:00 ET close; US equity and FX closes and the
# COMEX settle all land after 15:00 London too. Stamping these lag=0 let a feature
# built from day-t data be used to predict a return measured EARLIER on day t — a
# one-day look-ahead on the freshest and most influential observation.
#
# `gld_flow_z_1y` is the case that bit: it is the second-strongest feature in the
# Stage 1 diagnostics, and correcting its stamp moves the reported 5-day hit rate
# from 55.3% to 52.5%.
POST_LONDON_FIX = {
    "GLD_TONNES", "GLD_PX", "GDX_PX", "SPX",
    "DXY", "EURUSD", "USDJPY", "USDINR", "USDCNY",
    "COMEX_GOLD_FRONT", "COMEX_GOLD_SECOND", "COMEX_GOLD_ADJ", "COMEX_SILVER_FRONT",
}


def _put_exact(sid: str, s: pd.Series, *, desc: str, unit: str, bucket: str,
               source: str, url: str = "", freq: str = "daily") -> int:
    """Market data: no revisions. published_at = reference_date + a settle lag.

    The lag is 0 only for series that print BEFORE the London PM fix (the LBMA fixes
    themselves, the SGE close). Everything in POST_LONDON_FIX gets one business day,
    because its day-t value did not exist when the target's day-t return was struck."""
    lag = 1 if sid in POST_LONDON_FIX else 0
    goldstore.register(sid, description=desc, unit=unit, native_freq=freq,
                       typical_lag_days=lag, bucket=bucket, source_url=url,
                       published_at_approximated=False)
    s = pd.Series(s).dropna()
    if s.empty:
        print(f"  {sid:20s} no data")
        return 0
    s.index = pd.to_datetime(s.index)
    wrote = goldstore.put(sid, s, source=source, typical_lag_days=lag)
    print(f"  {sid:20s} +{wrote:6d} rows   {s.index.min().date()} -> {s.index.max().date()}")
    return wrote


def ingest_market(force: bool = False) -> int:
    """LBMA, SGE, SPDR, Yahoo and the deep store — reusing golddata's fetchers so
    there is exactly one implementation of each source in the repo."""
    n = 0
    lb, _ = golddata.lbma(force=force)
    if not lb.empty:
        n += _put_exact("LBMA_GOLD_PM_USD", lb["lbma_pm_usd"], desc="LBMA gold PM fix",
                        unit="USD/oz", bucket="valuation", source="LBMA",
                        url="https://prices.lbma.org.uk")
        n += _put_exact("LBMA_GOLD_AM_USD", lb["lbma_am_usd"], desc="LBMA gold AM fix",
                        unit="USD/oz", bucket="valuation", source="LBMA")
        n += _put_exact("LBMA_GOLD_PM_EUR", lb["lbma_pm_eur"], desc="LBMA gold PM fix, EUR",
                        unit="EUR/oz", bucket="valuation", source="LBMA")
        n += _put_exact("LBMA_GOLD_PM_GBP", lb["lbma_pm_gbp"], desc="LBMA gold PM fix, GBP",
                        unit="GBP/oz", bucket="valuation", source="LBMA")
        if "lbma_silver_usd" in lb.columns:
            n += _put_exact("LBMA_SILVER_USD", lb["lbma_silver_usd"],
                            desc="LBMA silver fix", unit="USD/oz",
                            bucket="valuation", source="LBMA")

    sg, _ = golddata.sge(force=force)
    if not sg.empty:
        n += _put_exact("SGE_AU9999", sg["sge_cny_g"], desc="SGE Au99.99 close",
                        unit="CNY/g", bucket="physical", source="Shanghai Gold Exchange",
                        url="https://www.sge.com.cn")

    gld, _ = golddata.spdr_gld(force=force)
    if not gld.empty:
        n += _put_exact("GLD_TONNES", gld["gld_tonnes"], desc="SPDR Gold Shares tonnage",
                        unit="tonnes", bucket="flows", source="SPDR Gold Trust",
                        url="https://www.spdrgoldshares.com")

    yh, _ = golddata.yahoo_block(force=force)
    ymap = {"dxy": ("DXY", "Dollar index", "index", "monetary"),
            "usdcny": ("USDCNY", "USD/CNY", "rate", "monetary"),
            "usdjpy": ("USDJPY", "USD/JPY", "rate", "monetary"),
            "usdinr": ("USDINR", "USD/INR", "rate", "monetary"),
            "eurusd": ("EURUSD", "EUR/USD", "rate", "monetary"),
            "gld_px": ("GLD_PX", "SPDR Gold Shares price", "USD", "flows"),
            "gdx_px": ("GDX_PX", "VanEck Gold Miners ETF", "USD", "valuation"),
            "spx": ("SPX", "S&P 500", "index", "risk")}
    for col, (sid, desc, unit, bucket) in ymap.items():
        if col in yh.columns:
            n += _put_exact(sid, yh[col], desc=desc, unit=unit, bucket=bucket,
                            source="Yahoo Finance")

    adj, raw = golddata.metals_prices()
    n += _put_exact("COMEX_GOLD_FRONT", raw["GCA Comdty"], desc="COMEX gold front settle",
                    unit="USD/oz", bucket="valuation", source="deep store (Bloomberg)")
    if "GC_second" in raw.columns:
        n += _put_exact("COMEX_GOLD_SECOND", raw["GC_second"],
                        desc="COMEX gold 2nd month settle", unit="USD/oz",
                        bucket="valuation", source="deep store (Bloomberg)")
    n += _put_exact("COMEX_GOLD_ADJ", adj["GCA Comdty"],
                    desc="COMEX gold, panama-adjusted", unit="USD/oz",
                    bucket="valuation", source="deep store (Bloomberg)")
    n += _put_exact("COMEX_SILVER_FRONT", raw["SIA Comdty"], desc="COMEX silver front settle",
                    unit="USD/oz", bucket="valuation", source="deep store (Bloomberg)")
    return n


# ---------------------------------------------------------------------------
# COT — the leak fix
# ---------------------------------------------------------------------------
_CFTC_RELEASE_HOUR = 15.5           # 15:30 ET


def cot_published_at(ref: pd.Timestamp) -> pd.Timestamp:
    """When the market actually saw the COT report for Tuesday `ref`.

    The CFTC takes positions as of Tuesday close and publishes the following Friday
    at 15:30 ET. A US federal holiday in that week pushes the release to the next
    business day. Reference dates are almost always Tuesday but not always — 13 of
    the 1,054 gold reports in our archive are dated Monday and one Wednesday, from
    holiday-shifted weeks — so this works off the reference week rather than
    assuming a fixed +3 days.

    The 15:30 stamp matters. Stored at midnight, a Friday-morning backtest would
    read Friday's report; at 15:30 it cannot. Erring late costs a few hours of
    information, erring early is a look-ahead — so this errs late by construction."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    ref = pd.Timestamp(ref).normalize()
    monday = ref - pd.Timedelta(days=int(ref.weekday()))
    friday = monday + pd.Timedelta(days=4)
    cal = USFederalHolidayCalendar()
    hols = set(cal.holidays(start=monday - pd.Timedelta(days=7),
                            end=friday + pd.Timedelta(days=21)))

    def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in hols:
            d += pd.Timedelta(days=1)
        return d

    # One business day of slip per federal holiday in the reference week. Counting
    # the slip and then *separately* skipping a holiday that lands on the Friday
    # double-counts it: July 4 2025 fell on the Friday and that pushed the release
    # to Monday 7 Jul, not Tuesday 8 Jul. Advancing whole business days — where an
    # advance already steps over weekends and holidays — handles both shapes with
    # one rule.
    slips = len([h for h in hols if monday <= h <= friday])
    pub = friday
    for _ in range(slips):
        pub = _next_business_day(pub)
    while pub.weekday() >= 5 or pub in hols:
        pub = _next_business_day(pub)
    return pub + pd.Timedelta(hours=_CFTC_RELEASE_HOUR)


COT_FIELDS = {
    "COT_MM_LONG": ("long", "Managed money long", "contracts"),
    "COT_MM_SHORT": ("short", "Managed money short", "contracts"),
    "COT_MM_NET": ("net", "Managed money net", "contracts"),
    "COT_OI": ("oi", "Open interest", "contracts"),
}


def ingest_cot(ticker: str = "GCA Comdty") -> int:
    """COT with real release timestamps, read off the archive the app maintains."""
    hist = _ROOT / "data" / "signals" / "cot_history.parquet"
    if not hist.exists():
        print("  cot_history.parquet missing — run the COT pull first")
        return 0
    h = pd.read_parquet(hist)
    g = h[h["ticker"] == ticker].copy()
    if g.empty:
        print(f"  no COT rows for {ticker}")
        return 0
    g["reference_date"] = pd.to_datetime(g["date"])
    g["published_at"] = g["reference_date"].map(cot_published_at)
    n = 0
    for sid, (col, desc, unit) in COT_FIELDS.items():
        if col not in g.columns:
            continue
        goldstore.register(sid, description=f"CFTC gold — {desc}", unit=unit,
                           native_freq="weekly", typical_lag_days=3, bucket="flows",
                           source_url="https://publicreporting.cftc.gov",
                           published_at_approximated=False)
        frame = g[["reference_date", "published_at", col]].rename(columns={col: "value"})
        wrote = goldstore.put(sid, frame.dropna(), source="CFTC disaggregated")
        n += wrote
        print(f"  {sid:14s} +{wrote:6d} rows   ref {frame['reference_date'].min().date()}"
              f" -> {frame['reference_date'].max().date()}")
    return n


# ---------------------------------------------------------------------------
# IMF — per-country official gold reserves (recovered during the gap review)
# ---------------------------------------------------------------------------
IMF_URL = ("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IL/"
           ".RGV_REVS..M?startPeriod={start}")

# Aggregate pseudo-countries in the IMF response. Summing the raw feed without
# excluding these double-counts badly — the naive total prints ~1,600t/yr of net
# buying against the WGC's ~860t, because the world aggregate is itself a row.
IMF_AGGREGATES = {"GX010", "BIS", "ECB", "EUR", "WLD", "1C_ALLC"}

# The reserve managers worth carrying individually.
IMF_COUNTRIES = ["USA", "DEU", "ITA", "FRA", "RUS", "CHN", "CHE", "IND", "JPN",
                 "TUR", "POL", "NLD", "KAZ", "BRA", "SGP", "THA", "UZB", "QAT"]


SYNTH_SPLICE_DATE = "2003-01-02"     # first DFII10 observation


def ingest_synthetic_real_yield() -> int:
    """Spec §2.7: a synthetic pre-2003 real yield, flagged as synthetic in the data.

    FRED's DFII10 begins 2003-01-02 — the spec assumes 1997 (TIPS launch) but the
    10-year constant-maturity real series starts later. Before that the standard
    stand-in is the 10y nominal minus trailing twelve-month CPI inflation.

    Built off `goldstore.daily_panel`, so the CPI leg is the vintage that was
    PUBLISHED by each date rather than the one we know now. That matters: CPI is
    revised and arrives a fortnight late, and splicing on today's vintage would hand
    every pre-2003 row a fortnight of hindsight — precisely the failure the store
    exists to prevent.

    Rows before the splice date carry is_synthetic=True. §2.7 requires the flag, and
    without it a fitted coefficient on 1990s data would be indistinguishable from one
    fitted on measured TIPS yields."""
    panel = goldstore.daily_panel(["NOMINAL_10Y", "CPI", "REAL_10Y"], start="1990-01-01")
    if panel.empty or "NOMINAL_10Y" not in panel:
        print("  synthetic real yield: nominal 10y unavailable")
        return 0
    cpi_yoy = panel["CPI"].pct_change(252) * 100.0
    raw_synth = (panel["NOMINAL_10Y"] - cpi_yoy).dropna()

    # LEVEL-ADJUST THE SPLICE. Nominal-minus-TRAILING-CPI is a backward-looking real
    # yield; TIPS price a FORWARD-looking one. They differ by roughly the gap between
    # realised and expected inflation, and joining them raw put an 80bp step into the
    # series overnight on 2003-01-02 (1.62 -> 2.43). A z-score feature would read that
    # step as a violent, entirely fictional repricing, and it lands in the middle of
    # the history this whole exercise exists to add.
    #
    # The offset is measured on the OVERLAP — the proxy is computed forward past the
    # splice date, where the true series also exists, and the mean difference over the
    # first five years is subtracted from the synthetic segment.
    overlap = pd.concat([raw_synth.rename("proxy"), panel["REAL_10Y"].rename("true")],
                        axis=1).dropna()
    overlap = overlap[overlap.index < pd.Timestamp("2008-01-01")]
    offset = float((overlap["proxy"] - overlap["true"]).mean()) if len(overlap) else 0.0
    synth = (raw_synth - offset)
    synth = synth[synth.index < pd.Timestamp(SYNTH_SPLICE_DATE)]
    print(f"  splice offset {offset:+.3f}pp (measured on {len(overlap)} overlapping days)")

    goldstore.register("REAL_10Y_SPLICED",
                       description="10y real yield (DFII10 from 2003; before that "
                                   "10y nominal less trailing 12m CPI)",
                       unit="pct", native_freq="daily", typical_lag_days=1,
                       bucket="monetary",
                       source_url="https://fred.stlouisfed.org/series/DFII10",
                       published_at_approximated=True)
    n = goldstore.put("REAL_10Y_SPLICED", synth, source="derived (nominal less CPI)",
                      is_synthetic=True, typical_lag_days=1)
    real = panel["REAL_10Y"].dropna()
    real = real[real.index >= pd.Timestamp(SYNTH_SPLICE_DATE)]
    n += goldstore.put("REAL_10Y_SPLICED", real, source="FRED DFII10",
                       is_synthetic=False, typical_lag_days=1)
    print(f"  REAL_10Y_SPLICED   +{n:6d} rows   synthetic before {SYNTH_SPLICE_DATE}, "
          f"measured after")
    return n


def ingest_central_banks(start: str = "1990-01") -> int:
    """Per-country official gold holdings, monthly, ISO3 keyed.

    Two traps, both silent. Country codes are ISO3 — asking for `PL` returns zero
    series and no error, so a naive ISO2 mapping yields an empty store that looks
    like a dead source. And the response carries aggregate pseudo-countries
    alongside real ones, so anything that sums the feed must exclude them."""
    req = urllib.request.Request(IMF_URL.format(start=start),
                                 headers={"User-Agent": "basis-gold-engine"})
    try:
        root = ET.fromstring(urllib.request.urlopen(req, timeout=300).read())
    except Exception as e:
        print(f"  IMF fetch failed: {str(e)[:80]}")
        return 0
    n = 0
    for s in root.iter():
        if not s.tag.endswith("Series"):
            continue
        c = s.attrib.get("COUNTRY")
        if not c or (c not in IMF_COUNTRIES and c not in ("GX010",)):
            continue
        obs = {}
        for o in s:
            if not o.tag.endswith("Obs"):
                continue
            try:
                d = pd.Period(o.attrib["TIME_PERIOD"].replace("-M", "-"),
                              freq="M").to_timestamp(how="end").normalize()
                obs[d] = float(o.attrib["OBS_VALUE"]) * OZT_TO_TONNE
            except (KeyError, ValueError):
                continue
        if not obs:
            continue
        sid = "CB_GOLD_WORLD" if c == "GX010" else f"CB_GOLD_{c}"
        label = "world aggregate" if c == "GX010" else c
        goldstore.register(sid, description=f"Official gold holdings — {label}",
                           unit="tonnes", native_freq="monthly", typical_lag_days=45,
                           bucket="physical", source_url="https://api.imf.org",
                           published_at_approximated=True)
        wrote = goldstore.put(sid, pd.Series(obs).sort_index(), source="IMF SDMX (IL)")
        n += wrote
        print(f"  {sid:18s} +{wrote:5d} rows")
    return n


# ---------------------------------------------------------------------------
def status() -> None:
    cov = goldstore.coverage()
    if cov.empty:
        print("store is empty")
        return
    print(f"\n=== coverage ({len(cov)} series) ===")
    print(cov.to_string())
    print("\n=== horizon eligibility ===")
    print(goldstore.horizon_matrix().to_string())
    flags = goldstore.stale_flags()
    print(f"\n=== staleness ===\n{flags or 'all series fresh'}")


def main() -> int:
    args = set(sys.argv[1:])
    if "--status" in args:
        status()
        return 0
    do_all = "--all" in args or not args
    total = 0
    if do_all or "--fred" in args:
        print("\nFRED (ALFRED vintages):")
        total += ingest_fred()
    if do_all or "--market" in args:
        print("\nMarket data:")
        total += ingest_market()
    if do_all or "--cot" in args:
        print("\nCOT:")
        total += ingest_cot()
    if do_all or "--cb" in args:
        print("\nIMF central banks:")
        total += ingest_central_banks()
    if do_all or "--synth" in args:
        # Last, because it reads NOMINAL_10Y and CPI back out of the store.
        print("\nSynthetic pre-2003 real yield (spec §2.7):")
        total += ingest_synthetic_real_yield()
    print(f"\n{total:,} observations written")
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
