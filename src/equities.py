"""Equities data layer for the BASIS "Equities" side.

Index constituents (GICS sector tagged) and overnight moves for the major US + European indices —
S&P 500, Nasdaq 100, Dow Jones 30, Euro Stoxx 50, FTSE 100, DAX 40, CAC 40. Mirrors the futures
`datafeed` MODE pattern (bloomberg / snapshot / mock) so the whole Equities side runs offline in
mock mode here and swaps to live Bloomberg on the work PC. Standalone: it never touches the futures
`universe`/`INSTRUMENTS`, `snapshot.py`, strategies or reports.

Data seams (all mode-agnostic to callers):
  load_universe()          -> {index_key: [{ticker,name,sector,region,index}, ...]}
  get_quotes(tickers)      -> DataFrame indexed by ticker, cols [last, pct]   (pct = % vs prior close)
  get_history(tickers)     -> DataFrame date x ticker (PX_LAST) for the 21-day realized vol
  movers_frame(indexes)    -> one row per (stock, index): last, pct, sigma (overnight move in σ)

MODE is read live from `datafeed.MODE` (env DATAFEED_MODE). In mock mode everything is deterministic
per ticker (MD5 seed) so tables/heatmaps are stable across reruns and unit-testable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import datafeed  # for MODE + (on the work PC) its Bloomberg session
from src import yfin      # free Yahoo Finance pulls — the preferred equity source

# Preferred data source for the Equities side. 'yfinance' (default) pulls quotes,
# history and fundamentals free from Yahoo — Bloomberg then only supplies what
# Yahoo can't (INDX_MEMBERS index membership) and is the fallback when Yahoo is
# unreachable. Set BASIS_EQ_SOURCE=bloomberg to restore the old Terminal-first pulls.
EQ_SOURCE = os.getenv("BASIS_EQ_SOURCE", "yfinance").strip().lower()


def _use_yf() -> bool:
    return EQ_SOURCE == "yfinance" and yfin.available()

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "equities"
_CONSTITUENTS_FILE = _DATA_DIR / "constituents.json"   # bloomberg-refreshed cache (snapshot mode reads it)
_QUOTES_FILE = _DATA_DIR / "quotes.parquet"
_HISTORY_FILE = _DATA_DIR / "history.parquet"
_MANIFEST_FILE = _DATA_DIR / "manifest.json"

# index key -> (display name, Bloomberg index ticker, region)
INDICES = {
    "S&P 500":       ("S&P 500",       "SPX Index",  "US"),
    "Nasdaq 100":    ("Nasdaq 100",    "NDX Index",  "US"),
    "Dow Jones 30":  ("Dow Jones 30",  "INDU Index", "US"),
    "Euro Stoxx 50": ("Euro Stoxx 50", "SX5E Index", "EU"),
    "FTSE 100":      ("FTSE 100",      "UKX Index",  "UK"),
    "DAX 40":        ("DAX 40",        "DAX Index",  "EU"),
    "CAC 40":        ("CAC 40",        "CAC Index",  "EU"),
    "Russell 2000":  ("Russell 2000",  "RTY Index",  "US"),
}
REGION_OF = {k: v[2] for k, v in INDICES.items()}

# Indices whose live membership is huge (Russell 2000 = ~2000 names). They're OPT-IN on the
# heavy pages (all-pairs correlations, dispersion, home movers/quotes) and default-on only
# where breadth is the point (the earnings calendar). DEFAULT_INDICES is the standard
# multiselect default for the heavy pages.
HEAVY_INDICES = {"Russell 2000"}
DEFAULT_INDICES = [k for k in INDICES if k not in HEAVY_INDICES]

# GICS sectors (11) — the "category" the heatmap groups by.
GICS_SECTORS = ["Information Technology", "Communication Services", "Consumer Discretionary",
                "Consumer Staples", "Financials", "Health Care", "Industrials", "Energy",
                "Utilities", "Materials", "Real Estate"]

_IT, _CD, _CS, _Cm, _FIN = "Information Technology", "Consumer Discretionary", "Consumer Staples", "Communication Services", "Financials"
_HC, _IND, _EN, _UT, _MAT, _RE = "Health Care", "Industrials", "Energy", "Utilities", "Materials", "Real Estate"

# Curated mock universe: index -> [(ticker, short name, GICS sector)]. Real tickers + real sectors so
# the offline heatmap looks genuine; on the work PC Bloomberg's INDX_MEMBERS replaces these with full
# live membership. A stock appears under EACH index it belongs to (e.g. AAPL in S&P 500 + Nasdaq 100).
_MOCK_CONSTITUENTS = {
    "S&P 500": [
        ("AAPL US Equity", "Apple", _IT), ("MSFT US Equity", "Microsoft", _IT),
        ("NVDA US Equity", "Nvidia", _IT), ("AVGO US Equity", "Broadcom", _IT),
        ("ORCL US Equity", "Oracle", _IT), ("ADBE US Equity", "Adobe", _IT),
        ("CRM US Equity", "Salesforce", _IT), ("AMD US Equity", "AMD", _IT),
        ("ACN US Equity", "Accenture", _IT), ("TXN US Equity", "Texas Instr.", _IT),
        ("QCOM US Equity", "Qualcomm", _IT), ("IBM US Equity", "IBM", _IT),
        ("CSCO US Equity", "Cisco", _IT), ("INTC US Equity", "Intel", _IT),
        ("GOOGL US Equity", "Alphabet", _Cm), ("META US Equity", "Meta", _Cm),
        ("NFLX US Equity", "Netflix", _Cm), ("DIS US Equity", "Disney", _Cm),
        ("CMCSA US Equity", "Comcast", _Cm), ("VZ US Equity", "Verizon", _Cm),
        ("T US Equity", "AT&T", _Cm), ("TMUS US Equity", "T-Mobile", _Cm),
        ("AMZN US Equity", "Amazon", _CD), ("TSLA US Equity", "Tesla", _CD),
        ("HD US Equity", "Home Depot", _CD), ("MCD US Equity", "McDonald's", _CD),
        ("NKE US Equity", "Nike", _CD), ("LOW US Equity", "Lowe's", _CD),
        ("SBUX US Equity", "Starbucks", _CD), ("BKNG US Equity", "Booking", _CD),
        ("PG US Equity", "P&G", _CS), ("KO US Equity", "Coca-Cola", _CS),
        ("PEP US Equity", "PepsiCo", _CS), ("COST US Equity", "Costco", _CS),
        ("WMT US Equity", "Walmart", _CS), ("PM US Equity", "Philip Morris", _CS),
        ("BRK/B US Equity", "Berkshire", _FIN), ("JPM US Equity", "JPMorgan", _FIN),
        ("V US Equity", "Visa", _FIN), ("MA US Equity", "Mastercard", _FIN),
        ("BAC US Equity", "Bank of America", _FIN), ("WFC US Equity", "Wells Fargo", _FIN),
        ("GS US Equity", "Goldman Sachs", _FIN), ("MS US Equity", "Morgan Stanley", _FIN),
        ("AXP US Equity", "Amex", _FIN), ("BLK US Equity", "BlackRock", _FIN),
        ("SPGI US Equity", "S&P Global", _FIN),
        ("LLY US Equity", "Eli Lilly", _HC), ("UNH US Equity", "UnitedHealth", _HC),
        ("JNJ US Equity", "J&J", _HC), ("MRK US Equity", "Merck", _HC),
        ("ABBV US Equity", "AbbVie", _HC), ("PFE US Equity", "Pfizer", _HC),
        ("TMO US Equity", "Thermo Fisher", _HC), ("ABT US Equity", "Abbott", _HC),
        ("AMGN US Equity", "Amgen", _HC),
        ("CAT US Equity", "Caterpillar", _IND), ("BA US Equity", "Boeing", _IND),
        ("HON US Equity", "Honeywell", _IND), ("UPS US Equity", "UPS", _IND),
        ("GE US Equity", "GE Aerospace", _IND), ("RTX US Equity", "RTX", _IND),
        ("UNP US Equity", "Union Pacific", _IND), ("DE US Equity", "Deere", _IND),
        ("XOM US Equity", "Exxon", _EN), ("CVX US Equity", "Chevron", _EN),
        ("COP US Equity", "ConocoPhillips", _EN), ("SLB US Equity", "Schlumberger", _EN),
        ("NEE US Equity", "NextEra", _UT), ("DUK US Equity", "Duke Energy", _UT),
        ("SO US Equity", "Southern Co", _UT),
        ("LIN US Equity", "Linde", _MAT), ("SHW US Equity", "Sherwin-Williams", _MAT),
        ("FCX US Equity", "Freeport", _MAT),
        ("PLD US Equity", "Prologis", _RE), ("AMT US Equity", "American Tower", _RE),
        ("EQIX US Equity", "Equinix", _RE),
    ],
    "Nasdaq 100": [
        ("AAPL US Equity", "Apple", _IT), ("MSFT US Equity", "Microsoft", _IT),
        ("NVDA US Equity", "Nvidia", _IT), ("AVGO US Equity", "Broadcom", _IT),
        ("ADBE US Equity", "Adobe", _IT), ("AMD US Equity", "AMD", _IT),
        ("QCOM US Equity", "Qualcomm", _IT), ("TXN US Equity", "Texas Instr.", _IT),
        ("INTC US Equity", "Intel", _IT), ("CSCO US Equity", "Cisco", _IT),
        ("AMAT US Equity", "Applied Materials", _IT), ("MU US Equity", "Micron", _IT),
        ("LRCX US Equity", "Lam Research", _IT), ("ADI US Equity", "Analog Devices", _IT),
        ("KLAC US Equity", "KLA Corp", _IT), ("PANW US Equity", "Palo Alto", _IT),
        ("GOOGL US Equity", "Alphabet", _Cm), ("META US Equity", "Meta", _Cm),
        ("NFLX US Equity", "Netflix", _Cm), ("CMCSA US Equity", "Comcast", _Cm),
        ("TMUS US Equity", "T-Mobile", _Cm), ("WBD US Equity", "Warner Bros", _Cm),
        ("EA US Equity", "Electronic Arts", _Cm),
        ("AMZN US Equity", "Amazon", _CD), ("TSLA US Equity", "Tesla", _CD),
        ("BKNG US Equity", "Booking", _CD), ("MELI US Equity", "MercadoLibre", _CD),
        ("LULU US Equity", "Lululemon", _CD), ("ORLY US Equity", "O'Reilly", _CD),
        ("MAR US Equity", "Marriott", _CD), ("ROST US Equity", "Ross Stores", _CD),
        ("PEP US Equity", "PepsiCo", _CS), ("COST US Equity", "Costco", _CS),
        ("MDLZ US Equity", "Mondelez", _CS), ("KHC US Equity", "Kraft Heinz", _CS),
        ("MNST US Equity", "Monster", _CS), ("KDP US Equity", "Keurig Dr Pepper", _CS),
        ("AMGN US Equity", "Amgen", _HC), ("GILD US Equity", "Gilead", _HC),
        ("VRTX US Equity", "Vertex", _HC), ("REGN US Equity", "Regeneron", _HC),
        ("ISRG US Equity", "Intuitive Surg.", _HC), ("MRNA US Equity", "Moderna", _HC),
        ("HON US Equity", "Honeywell", _IND), ("ADP US Equity", "ADP", _IND),
        ("CSX US Equity", "CSX", _IND), ("PCAR US Equity", "Paccar", _IND),
        ("AEP US Equity", "American Electric", _UT), ("EXC US Equity", "Exelon", _UT),
        ("XEL US Equity", "Xcel Energy", _UT),
    ],
    "Dow Jones 30": [
        ("AAPL US Equity", "Apple", _IT), ("MSFT US Equity", "Microsoft", _IT),
        ("NVDA US Equity", "Nvidia", _IT), ("CRM US Equity", "Salesforce", _IT),
        ("IBM US Equity", "IBM", _IT), ("CSCO US Equity", "Cisco", _IT),
        ("JPM US Equity", "JPMorgan", _FIN), ("GS US Equity", "Goldman Sachs", _FIN),
        ("V US Equity", "Visa", _FIN), ("AXP US Equity", "Amex", _FIN),
        ("TRV US Equity", "Travelers", _FIN),
        ("UNH US Equity", "UnitedHealth", _HC), ("JNJ US Equity", "J&J", _HC),
        ("MRK US Equity", "Merck", _HC), ("AMGN US Equity", "Amgen", _HC),
        ("HD US Equity", "Home Depot", _CD), ("MCD US Equity", "McDonald's", _CD),
        ("NKE US Equity", "Nike", _CD), ("AMZN US Equity", "Amazon", _CD),
        ("DIS US Equity", "Disney", _Cm), ("VZ US Equity", "Verizon", _Cm),
        ("BA US Equity", "Boeing", _IND), ("CAT US Equity", "Caterpillar", _IND),
        ("HON US Equity", "Honeywell", _IND), ("MMM US Equity", "3M", _IND),
        ("PG US Equity", "P&G", _CS), ("KO US Equity", "Coca-Cola", _CS),
        ("WMT US Equity", "Walmart", _CS),
        ("CVX US Equity", "Chevron", _EN), ("SHW US Equity", "Sherwin-Williams", _MAT),
    ],
    "Euro Stoxx 50": [
        ("SAP GY Equity", "SAP", _IT), ("ASML NA Equity", "ASML", _IT),
        ("IFX GY Equity", "Infineon", _IT),
        ("MC FP Equity", "LVMH", _CD), ("RMS FP Equity", "Hermes", _CD),
        ("MBG GY Equity", "Mercedes-Benz", _CD), ("BMW GY Equity", "BMW", _CD),
        ("VOW3 GY Equity", "Volkswagen", _CD), ("STLAM IM Equity", "Stellantis", _CD),
        ("ITX SM Equity", "Inditex", _CD),
        ("OR FP Equity", "L'Oreal", _CS), ("AD NA Equity", "Ahold Delhaize", _CS),
        ("ABI BB Equity", "AB InBev", _CS), ("BN FP Equity", "Danone", _CS),
        ("SAN SM Equity", "Santander", _FIN), ("BBVA SM Equity", "BBVA", _FIN),
        ("BNP FP Equity", "BNP Paribas", _FIN), ("INGA NA Equity", "ING", _FIN),
        ("ISP IM Equity", "Intesa Sanpaolo", _FIN), ("UCG IM Equity", "UniCredit", _FIN),
        ("ALV GY Equity", "Allianz", _FIN), ("MUV2 GY Equity", "Munich Re", _FIN),
        ("CS FP Equity", "AXA", _FIN), ("DB1 GY Equity", "Deutsche Boerse", _FIN),
        ("ADYEN NA Equity", "Adyen", _FIN),
        ("SAN FP Equity", "Sanofi", _HC), ("BAYN GY Equity", "Bayer", _HC),
        ("SIE GY Equity", "Siemens", _IND), ("AIR FP Equity", "Airbus", _IND),
        ("SU FP Equity", "Schneider Elec.", _IND), ("DHL GY Equity", "DHL Group", _IND),
        ("SAF FP Equity", "Safran", _IND),
        ("DTE GY Equity", "Deutsche Telekom", _Cm), ("TEF SM Equity", "Telefonica", _Cm),
        ("ORA FP Equity", "Orange", _Cm),
        ("TTE FP Equity", "TotalEnergies", _EN), ("ENI IM Equity", "Eni", _EN),
        ("ENEL IM Equity", "Enel", _UT), ("IBE SM Equity", "Iberdrola", _UT),
        ("EOAN GY Equity", "E.ON", _UT), ("ENGI FP Equity", "Engie", _UT),
        ("BAS GY Equity", "BASF", _MAT), ("AI FP Equity", "Air Liquide", _MAT),
    ],
    "FTSE 100": [
        ("SHEL LN Equity", "Shell", _EN), ("BP/ LN Equity", "BP", _EN),
        ("AZN LN Equity", "AstraZeneca", _HC), ("GSK LN Equity", "GSK", _HC),
        ("ULVR LN Equity", "Unilever", _CS), ("DGE LN Equity", "Diageo", _CS),
        ("BATS LN Equity", "Brit. Am. Tobacco", _CS), ("RKT LN Equity", "Reckitt", _CS),
        ("TSCO LN Equity", "Tesco", _CS),
        ("HSBA LN Equity", "HSBC", _FIN), ("LLOY LN Equity", "Lloyds", _FIN),
        ("BARC LN Equity", "Barclays", _FIN), ("NWG LN Equity", "NatWest", _FIN),
        ("PRU LN Equity", "Prudential", _FIN), ("LGEN LN Equity", "Legal & General", _FIN),
        ("STAN LN Equity", "Standard Chartered", _FIN),
        ("RIO LN Equity", "Rio Tinto", _MAT), ("AAL LN Equity", "Anglo American", _MAT),
        ("GLEN LN Equity", "Glencore", _MAT),
        ("BA/ LN Equity", "BAE Systems", _IND), ("RR/ LN Equity", "Rolls-Royce", _IND),
        ("EXPN LN Equity", "Experian", _IND), ("REL LN Equity", "RELX", _IND),
        ("NXT LN Equity", "Next", _CD), ("VOD LN Equity", "Vodafone", _Cm),
        ("NG/ LN Equity", "National Grid", _UT), ("SSE LN Equity", "SSE", _UT),
        ("LAND LN Equity", "Land Securities", _RE), ("SGRO LN Equity", "Segro", _RE),
    ],
    "DAX 40": [
        ("SAP GY Equity", "SAP", _IT), ("IFX GY Equity", "Infineon", _IT),
        ("SIE GY Equity", "Siemens", _IND), ("DHL GY Equity", "DHL Group", _IND),
        ("AIR GY Equity", "Airbus", _IND), ("MTX GY Equity", "MTU Aero", _IND),
        ("RHM GY Equity", "Rheinmetall", _IND), ("ENR GY Equity", "Siemens Energy", _IND),
        ("DTG GY Equity", "Daimler Truck", _IND),
        ("ALV GY Equity", "Allianz", _FIN), ("MUV2 GY Equity", "Munich Re", _FIN),
        ("DB1 GY Equity", "Deutsche Boerse", _FIN), ("DBK GY Equity", "Deutsche Bank", _FIN),
        ("CBK GY Equity", "Commerzbank", _FIN),
        ("MBG GY Equity", "Mercedes-Benz", _CD), ("BMW GY Equity", "BMW", _CD),
        ("VOW3 GY Equity", "Volkswagen", _CD), ("ADS GY Equity", "Adidas", _CD),
        ("CON GY Equity", "Continental", _CD), ("P911 GY Equity", "Porsche AG", _CD),
        ("ZAL GY Equity", "Zalando", _CD),
        ("DTE GY Equity", "Deutsche Telekom", _Cm),
        ("BAYN GY Equity", "Bayer", _HC), ("MRK GY Equity", "Merck KGaA", _HC),
        ("FRE GY Equity", "Fresenius", _HC), ("SHL GY Equity", "Siemens Health.", _HC),
        ("HEN3 GY Equity", "Henkel", _CS), ("BEI GY Equity", "Beiersdorf", _CS),
        ("EOAN GY Equity", "E.ON", _UT), ("RWE GY Equity", "RWE", _UT),
        ("BAS GY Equity", "BASF", _MAT), ("HEI GY Equity", "Heidelberg Mat.", _MAT),
        ("SY1 GY Equity", "Symrise", _MAT), ("1COV GY Equity", "Covestro", _MAT),
        ("VNA GY Equity", "Vonovia", _RE),
    ],
    "CAC 40": [
        ("CAP FP Equity", "Capgemini", _IT), ("STM FP Equity", "STMicro", _IT),
        ("DSY FP Equity", "Dassault Syst.", _IT),
        ("MC FP Equity", "LVMH", _CD), ("RMS FP Equity", "Hermes", _CD),
        ("KER FP Equity", "Kering", _CD), ("EL FP Equity", "EssilorLuxottica", _CD),
        ("STLAP FP Equity", "Stellantis", _CD), ("ML FP Equity", "Michelin", _CD),
        ("OR FP Equity", "L'Oreal", _CS), ("BN FP Equity", "Danone", _CS),
        ("RI FP Equity", "Pernod Ricard", _CS), ("CA FP Equity", "Carrefour", _CS),
        ("BNP FP Equity", "BNP Paribas", _FIN), ("CS FP Equity", "AXA", _FIN),
        ("GLE FP Equity", "Societe Generale", _FIN), ("ACA FP Equity", "Credit Agricole", _FIN),
        ("SAN FP Equity", "Sanofi", _HC),
        ("AIR FP Equity", "Airbus", _IND), ("SU FP Equity", "Schneider Elec.", _IND),
        ("SAF FP Equity", "Safran", _IND), ("DG FP Equity", "Vinci", _IND),
        ("LR FP Equity", "Legrand", _IND), ("HO FP Equity", "Thales", _IND),
        ("SGO FP Equity", "Saint-Gobain", _IND),
        ("ORA FP Equity", "Orange", _Cm), ("PUB FP Equity", "Publicis", _Cm),
        ("TTE FP Equity", "TotalEnergies", _EN),
        ("ENGI FP Equity", "Engie", _UT), ("VIE FP Equity", "Veolia", _UT),
        ("AI FP Equity", "Air Liquide", _MAT), ("URW FP Equity", "Unibail-Rodamco", _RE),
    ],
    # Representative small caps — live Bloomberg replaces this with the full ~2000-name
    # INDX_MEMBERS membership; the mock keeps the demo calendar/heatmap looking genuine.
    "Russell 2000": [
        ("FN US Equity", "Fabrinet", _IT), ("CALX US Equity", "Calix", _IT),
        ("PLXS US Equity", "Plexus", _IT), ("ITRI US Equity", "Itron", _IT),
        ("ACLS US Equity", "Axcelis", _IT), ("SITM US Equity", "SiTime", _IT),
        ("PI US Equity", "Impinj", _IT), ("DIOD US Equity", "Diodes", _IT),
        ("SUPN US Equity", "Supernus", _HC), ("CORT US Equity", "Corcept", _HC),
        ("KRYS US Equity", "Krystal Biotech", _HC), ("ADMA US Equity", "ADMA Biologics", _HC),
        ("PTGX US Equity", "Protagonist", _HC),
        ("GBCI US Equity", "Glacier Bancorp", _FIN), ("ONB US Equity", "Old National", _FIN),
        ("UBSI US Equity", "United Bankshares", _FIN), ("EBC US Equity", "Eastern Bankshares", _FIN),
        ("WAFD US Equity", "WaFd", _FIN),
        ("BOOT US Equity", "Boot Barn", _CD), ("PLAY US Equity", "Dave & Buster's", _CD),
        ("WGO US Equity", "Winnebago", _CD), ("M US Equity", "Macy's", _CD),
        ("UNFI US Equity", "United Natural Foods", _CS), ("CHEF US Equity", "Chefs' Warehouse", _CS),
        ("ANDE US Equity", "Andersons", _CS),
        ("ALG US Equity", "Alamo Group", _IND), ("AGX US Equity", "Argan", _IND),
        ("ROAD US Equity", "Construction Partners", _IND), ("MYRG US Equity", "MYR Group", _IND),
        ("GVA US Equity", "Granite Construction", _IND),
        ("SM US Equity", "SM Energy", _EN), ("CRC US Equity", "California Resources", _EN),
        ("PTEN US Equity", "Patterson-UTI", _EN), ("LBRT US Equity", "Liberty Energy", _EN),
        ("NWE US Equity", "NorthWestern", _UT), ("AVA US Equity", "Avista", _UT),
        ("OTTR US Equity", "Otter Tail", _UT),
        ("HWKN US Equity", "Hawkins", _MAT), ("IOSP US Equity", "Innospec", _MAT),
        ("MTX US Equity", "Minerals Technologies", _MAT),
        ("APLE US Equity", "Apple Hospitality", _RE), ("SKT US Equity", "Tanger", _RE),
        ("IIPR US Equity", "Innovative Industrial", _RE),
        ("TGNA US Equity", "Tegna", _Cm), ("ZD US Equity", "Ziff Davis", _Cm),
        ("IAS US Equity", "Integral Ad Science", _Cm),
    ],
}


# ── seeds / mock generators ───────────────────────────────────────────────────
def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _mock_base(ticker: str) -> float:
    """A stable, plausible share price 20..520 per ticker."""
    return 20.0 + (_seed(ticker) % 500)


def _mock_universe() -> dict:
    out = {}
    for key, rows in _MOCK_CONSTITUENTS.items():
        region = REGION_OF.get(key, "")
        out[key] = [{"ticker": t, "name": n, "sector": s, "region": region, "index": key}
                    for (t, n, s) in rows]
    return out


def _mock_quotes(tickers) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        rng = np.random.default_rng(_seed(t) ^ 0x0EED)
        pct = float(rng.normal(0.0, 1.15))          # overnight % vs prior close, ~1.15% std
        last = _mock_base(t) * (1 + pct / 100.0)
        rows[t] = {"last": round(last, 2), "pct": round(pct, 3)}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["last", "pct"])


def _mock_history(tickers, days: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for t in tickers:
        rng = np.random.default_rng(_seed(t))
        rets = rng.normal(0.0, 0.014, size=days)    # ~1.4% daily vol
        data[t] = _mock_base(t) * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


# ── Bloomberg (live via xbbg) ─────────────────────────────────────────────────
# Reuses datafeed's normalisers so this matches how the futures side already talks to the Terminal:
# xbbg here returns **narwhals** frames in **tidy ticker/field/value** form (NOT pandas wide), so
# `_coerce_pd` → pandas, `_bdp_rows` → {security:{FIELD:value}}, `_bdh_to_wide` → date×ticker.
def _fmt_members(bds_df) -> list:
    """INDX_MEMBERS → ['AAPL UW Equity', ...]. The member codes live in the bulk data column
    ('Member Ticker and Exchange Code', values like 'AAPL UW' / 'A UN'), NOT the 'ticker'/'field'
    metadata columns xbbg prepends. Append ' Equity' to make a security."""
    if bds_df is None or getattr(bds_df, "empty", True):
        return []
    cols = [c for c in bds_df.columns if str(c).lower() not in ("ticker", "field")]
    col = cols[0] if cols else bds_df.columns[-1]
    out = []
    for raw in bds_df[col].astype(str):
        raw = raw.strip()
        if not raw or raw.lower() == "nan":
            continue
        out.append(raw if raw.endswith("Equity") else f"{raw} Equity")
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _clean_field(v, default):
    """bdp values can be None or a float NaN (missing) — a plain `x or default` misses NaN because
    float('nan') is truthy. Coerce to a clean non-'nan' string or `default`."""
    if v is None or (isinstance(v, float) and v != v):
        return default
    s = str(v).strip()
    return s if s and s.lower() != "nan" else default


def _bloomberg_constituents() -> dict:
    from xbbg import blp
    uni = {}
    for key, (_disp, idx, region) in INDICES.items():
        members = _fmt_members(datafeed._coerce_pd(blp.bds(idx, "INDX_MEMBERS")))
        meta = (datafeed._bdp_rows(datafeed._coerce_pd(
                    blp.bdp(members, ["short_name", "gics_sector_name"]))) if members else {})
        rows = []
        for t in members:
            info = meta.get(t, {})
            nm = _clean_field(info.get("SHORT_NAME"), t.split()[0])          # fallback: ticker root
            sec = _clean_field(info.get("GICS_SECTOR_NAME"), "Other")        # unclassified bucket
            rows.append({"ticker": t, "name": nm, "sector": sec, "region": region, "index": key})
        uni[key] = rows
    return uni


def _bloomberg_quotes(tickers) -> pd.DataFrame:
    from xbbg import blp
    rows = datafeed._bdp_rows(datafeed._coerce_pd(blp.bdp(list(tickers), ["px_last", "chg_pct_1d"])))
    out = pd.DataFrame(index=list(tickers), columns=["last", "pct"], dtype=float)
    for t in tickers:
        info = rows.get(t, {})
        out.loc[t, "last"] = _num(info.get("PX_LAST"))
        out.loc[t, "pct"] = _num(info.get("CHG_PCT_1D"))
    return out


def _bloomberg_history(tickers) -> pd.DataFrame:
    from xbbg import blp
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=45)     # ~31 trading days — enough for the trailing-21 σ, lighter pull
    wide = datafeed._bdh_to_wide(blp.bdh(tickers=list(tickers), flds="px_last",
                                         start_date=start.strftime("%Y-%m-%d"),
                                         end_date=end.strftime("%Y-%m-%d")))
    return wide if wide is not None else pd.DataFrame()


# ── cache I/O (bloomberg writes it; snapshot mode reads it) ───────────────────
def _write_cache(uni: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CONSTITUENTS_FILE.write_text(json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_cache() -> dict | None:
    try:
        return json.loads(_CONSTITUENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_parquet(path: Path):
    try:
        return pd.read_parquet(path) if path.exists() else None
    except Exception:
        return None


# ── public seams (mode-branched) ──────────────────────────────────────────────
def load_universe() -> dict:
    """{index_key: [{ticker,name,sector,region,index}, ...]} for every index in INDICES."""
    if datafeed.MODE == "bloomberg":
        try:
            uni = _bloomberg_constituents()
            if uni:
                _write_cache(uni)
                return uni
        except Exception:
            pass
    # Off-Terminal: the last written membership (constituents.json rides the repo, so the
    # real index membership is available at home too); curated mock only when none exists.
    cached = _read_cache()
    if cached:
        return cached
    return _mock_universe()


def refresh_constituents() -> dict:
    """Force a live Bloomberg membership pull (work PC) and cache it. Returns the universe."""
    uni = _bloomberg_constituents()
    if uni:
        _write_cache(uni)
    return uni


def cached_universe() -> dict:
    """The last WRITTEN membership (constituents.json) when one exists, else load_universe().
    For pages that only need names/sectors (Company Fundamentals): in bloomberg mode
    load_universe() re-pulls INDX_MEMBERS live — ~30s+ of dead screen on a first click —
    while the cache (refreshed by every pull / morning snapshot) answers instantly."""
    cached = _read_cache()
    return cached if cached else load_universe()


def get_quotes(tickers) -> pd.DataFrame:
    """DataFrame indexed by ticker, columns ['last','pct'] (pct = overnight % vs prior close)."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["last", "pct"])
    if _use_yf():
        try:
            q = yfin.get_quotes(tickers)
            if q["last"].notna().any():
                return q
        except Exception:
            pass                                   # offline / Yahoo down -> old chain
    if datafeed.MODE == "bloomberg":
        try:
            return _bloomberg_quotes(tickers)
        except Exception:
            pass
    if datafeed.MODE == "snapshot":
        q = _read_parquet(_QUOTES_FILE)
        if q is not None:
            return q.reindex(tickers)
    return _mock_quotes(tickers)


def get_history(tickers) -> pd.DataFrame:
    """DataFrame date x ticker (PX_LAST) — used for the 21-day realized vol behind σ."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()
    if _use_yf():
        try:
            h = yfin.get_history(tickers, sessions=31)   # enough for the trailing-21 σ
            if h is not None and not h.empty:
                return h
        except Exception:
            pass
    if datafeed.MODE == "bloomberg":
        try:
            return _bloomberg_history(tickers)
        except Exception:
            pass
    if datafeed.MODE == "snapshot":
        h = _read_parquet(_HISTORY_FILE)
        if h is not None:
            return h.reindex(columns=tickers)
    return _mock_history(tickers)


def movers_frame(index_keys=None, universe=None) -> pd.DataFrame:
    """One row per (stock, index) for the selected indices (all if None), with the overnight move
    and σ (move in units of the stock's own ~1-month daily vol). Columns:
    ticker, name, index, sector, region, last, pct, sigma. Pass `universe` to reuse an already-loaded
    (cached) membership dict instead of re-pulling it."""
    uni = universe if universe is not None else load_universe()
    keys = [k for k in (index_keys or uni.keys()) if k in uni]
    rows = []
    for k in keys:
        for c in uni.get(k, []):
            rows.append({"ticker": c["ticker"], "name": c.get("name", c["ticker"]),
                         "index": k, "sector": c.get("sector", "—"),
                         "region": c.get("region", REGION_OF.get(k, ""))})
    cols = ["ticker", "name", "index", "sector", "region", "last", "pct", "sigma"]
    if not rows:
        return pd.DataFrame(columns=cols)
    frame = pd.DataFrame(rows)
    uniq = sorted(frame["ticker"].unique())
    q = get_quotes(uniq)
    try:
        sd = get_history(uniq).pct_change().tail(21).std()
    except Exception:
        sd = pd.Series(dtype=float)
    last_map = q["last"] if "last" in q else pd.Series(dtype=float)
    pct_map = q["pct"] if "pct" in q else pd.Series(dtype=float)
    frame["last"] = frame["ticker"].map(last_map)
    frame["pct"] = frame["ticker"].map(pct_map)

    def _sig(row):
        s = sd.get(row["ticker"]) if hasattr(sd, "get") else None
        p = row["pct"]
        if s is None or s != s or s <= 1e-9 or p != p:
            return float("nan")
        return (p / 100.0) / s

    frame["sigma"] = frame.apply(_sig, axis=1)
    return frame[cols]


# ── daily snapshot (called by snapshot.py so the Equities side runs off the morning pull) ─────────
def build_snapshot() -> dict:
    """Pull the equities inputs (index membership + overnight quotes + short history) via the CURRENT
    mode and cache them to data/equities/ so the Equities pages run all day off the morning snapshot,
    Terminal-closed — the same pattern snapshot.py uses for the futures inputs. Live in bloomberg
    mode, synthetic in mock mode (an offline pipeline test). Never wipes the cache on a dead/empty
    pull. Returns a status dict."""
    uni = load_universe()
    if not uni or not any(uni.values()):
        return {"ok": False, "reason": "no constituents pulled (Terminal/API down?)"}
    tickers = sorted({c["ticker"] for rows in uni.values() for c in rows})
    q = get_quotes(tickers)
    if q is None or q.empty or int(pd.to_numeric(q.get("pct"), errors="coerce").notna().sum()) == 0:
        return {"ok": False, "reason": "no overnight quotes pulled"}
    h = get_history(tickers)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_cache(uni)                                            # constituents.json
    q.rename_axis("ticker").to_parquet(_QUOTES_FILE)            # ticker-indexed [last, pct]
    if h is not None and not h.empty:
        h.rename_axis("date").to_parquet(_HISTORY_FILE)        # date x ticker
    manifest = {
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "yfinance" if _use_yf() else datafeed.MODE,
        "as_of": str(h.index.max().date()) if (h is not None and not h.empty) else "",
        "indices": {k: len(v) for k, v in uni.items()},
        "n_memberships": sum(len(v) for v in uni.values()),
        "n_unique": len(tickers),
    }
    _MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **manifest}


def _read_manifest() -> dict:
    try:
        return json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def data_status() -> str:
    """Short 'where the equities data comes from' label for the Home caption."""
    if _use_yf():
        member = ("live Bloomberg membership" if datafeed.MODE == "bloomberg"
                  else "cached membership" if _read_cache() else "built-in membership")
        return f"live Yahoo Finance (free) · {member}"
    if datafeed.MODE == "bloomberg":
        return "live Bloomberg"
    if datafeed.MODE == "snapshot" and _QUOTES_FILE.exists() and _CONSTITUENTS_FILE.exists():
        m = _read_manifest()
        asof = m.get("as_of") or m.get("created", "")
        return f"snapshot{f' · {asof}' if asof else ''}"
    return "demo (synthetic) data"
