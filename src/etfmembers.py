"""Index membership from the tracker ETFs' published holdings files.

The one input Yahoo can't supply is WHICH stocks are in each index. Rather than
spending Bloomberg hits on INDX_MEMBERS, this module reads the daily holdings
files the tracker-fund providers publish for their own portfolios (iShares /
SSGA / Invesco) — free, keyless, machine-readable, updated daily, and in effect
the real membership (tracking differences are a handful of basis points).

Output rows match equities.py's constituent shape exactly —
{"ticker": "<ROOT EXCH Equity>", "name", "sector", "region", "index"} — with
BLOOMBERG-style tickers so every downstream seam (incl. yfin.to_yahoo) is
untouched. Sector names in the files are GICS, same vocabulary the app uses.

Per-index failures degrade gracefully: fetch_all() simply omits that index and
the caller keeps its previous membership. No Streamlit here; equities.py
imports us.
"""
from __future__ import annotations

import io
import json
import urllib.request

import pandas as pd

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_TIMEOUT = 30

# index key -> (holdings URL, kind, default Bloomberg exchange code or None=per-row)
_ISHARES_US = "https://www.ishares.com/us/products/{pid}/{slug}/1467271812596.ajax?fileType=csv&fileName={fn}&dataType=fund"
_ISHARES_UK = "https://www.ishares.com/uk/individual/en/products/{pid}/{slug}/1506575576011.ajax?fileType=csv&fileName={fn}&dataType=fund"
SOURCES = {
    "S&P 500": (_ISHARES_US.format(pid="239726", slug="ishares-core-sp-500-etf",
                                   fn="IVV_holdings"), "ishares", "US"),
    "Russell 2000": (_ISHARES_US.format(pid="239710", slug="ishares-russell-2000-etf",
                                        fn="IWM_holdings"), "ishares", "US"),
    "Nasdaq 100": ("https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
                   "?audienceType=Investor&action=download&ticker=QQQ", "invesco", "US"),
    "Dow Jones 30": ("https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
                     "fund-data/etfs/us/holdings-daily-us-en-dia.xlsx", "ssga", "US"),
    "FTSE 100": (_ISHARES_UK.format(pid="251795", slug="ishares-ftse-100-ucits-etf-inc-fund",
                                    fn="ISF_holdings"), "ishares", "LN"),
    "Euro Stoxx 50": (_ISHARES_UK.format(pid="251781", slug="ishares-euro-stoxx-50-ucits-etf-inc-fund",
                                         fn="EUE_holdings"), "ishares", None),
    "DAX 40": (_ISHARES_UK.format(pid="251464", slug="ishares-dax-ucits-etf-de-fund",
                                  fn="EXS1_holdings"), "ishares", "GY"),
    "CAC 40": ("https://www.amundietf.co.uk/en/professional/products/equity/"
               "amundi-cac-40-ucits-etf-dist/fr0007052782", "amundi", "FP"),
}

# holdings-file exchange names -> Bloomberg exchange code (Euro Stoxx spans venues)
_EXCH_CODE = {
    "new york stock exchange": "US", "nasdaq": "US", "nyse": "US", "cboe bzx": "US",
    "london stock exchange": "LN",
    "xetra": "GY", "deutsche boerse ag": "GY", "frankfurt": "GY",
    "euronext paris": "FP", "nyse euronext - euronext paris": "FP",
    "euronext amsterdam": "NA", "nyse euronext - euronext amsterdam": "NA",
    "borsa italiana": "IM", "milan": "IM",
    "bolsa de madrid": "SM", "madrid": "SM",
    "euronext brussels": "BB", "nyse euronext - euronext brussels": "BB",
    "irish stock exchange": "ID", "euronext dublin": "ID",
    "nasdaq omx helsinki": "FH", "helsinki": "FH",
    "euronext lisbon": "PL", "six swiss exchange": "SW",
}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=_TIMEOUT).read()


def _bbg_root(raw: str) -> str:
    """Holdings-file ticker -> Bloomberg root: 'BRK.B'->'BRK/B', 'BT.A'->'BT/A',
    'BP.'->'BP'. Spaces (rare share-class forms) collapse."""
    r = str(raw).strip().replace(" ", "")
    return r.replace(".", "/").rstrip("/")


def _exch_for(row_exch: str, default: str | None) -> str | None:
    if default:
        return default
    return _EXCH_CODE.get(str(row_exch).strip().lower())


def _rows_from_frame(df: pd.DataFrame, cols: dict, index_key: str, region: str,
                     default_exch: str | None) -> list:
    """Normalise a holdings frame to constituent rows. cols maps logical->actual
    column names ({'ticker','name','sector','exchange'?,'asset'?})."""
    out, seen = [], set()
    for _, r in df.iterrows():
        if cols.get("asset") and str(r.get(cols["asset"], "")).strip().lower() != "equity":
            continue
        raw = str(r.get(cols["ticker"], "") or "").strip()
        if not raw or raw in ("-", "—") or raw.lower() in ("nan", "none"):
            continue
        exch = _exch_for(r.get(cols.get("exchange", ""), ""), default_exch)
        if not exch:
            continue
        root = _bbg_root(raw)
        tick = f"{root} {exch} Equity"
        if not root or tick in seen:
            continue
        seen.add(tick)
        name = str(r.get(cols["name"], "") or root).strip().title() or root
        sec = str(r.get(cols["sector"], "") or "").strip() or "Other"
        if sec.lower() in ("cash and/or derivatives", "cash", "-", "—", "nan"):
            continue
        out.append({"ticker": tick, "name": name, "sector": sec,
                    "region": region, "index": index_key})
    return out


def _parse_ishares(blob: bytes, index_key: str, region: str, default_exch) -> list:
    """iShares holdings CSVs open with fund-metadata preamble lines; the real table
    begins at the row whose first cell is 'Ticker' / 'Issuer Ticker'."""
    text = blob.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if ln.split(",")[0].strip().strip('"') in ("Ticker", "Issuer Ticker")), None)
    if hdr is None:
        return []
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    tick_col = "Ticker" if "Ticker" in df.columns else "Issuer Ticker"
    return _rows_from_frame(df, {
        "ticker": tick_col, "name": "Name", "sector": "Sector",
        "exchange": "Exchange" if "Exchange" in df.columns else "",
        "asset": "Asset Class" if "Asset Class" in df.columns else None,
    }, index_key, region, default_exch)


def _parse_invesco(blob: bytes, index_key: str, region: str, default_exch) -> list:
    df = pd.read_csv(io.BytesIO(blob), dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    tick = next((c for c in df.columns if c.lower() in ("holding ticker", "holdingsticker", "ticker")), None)
    name = next((c for c in df.columns if c.lower() == "name"), None)
    sec = next((c for c in df.columns if c.lower() == "sector"), None)
    if not tick or not name:
        return []
    return _rows_from_frame(df, {"ticker": tick, "name": name, "sector": sec or name},
                            index_key, region, default_exch)


def _parse_ssga(blob: bytes, index_key: str, region: str, default_exch) -> list:
    """SSGA daily-holdings XLSX: preamble rows above a header row containing 'Ticker'."""
    raw = pd.read_excel(io.BytesIO(blob), header=None, dtype=str)
    hdr = next((i for i in range(min(len(raw), 12))
                if "ticker" in [str(v).strip().lower() for v in raw.iloc[i].tolist()]), None)
    if hdr is None:
        return []
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(v).strip() for v in raw.iloc[hdr].tolist()]
    cols_l = {c.lower(): c for c in df.columns}
    return _rows_from_frame(df, {
        "ticker": cols_l.get("ticker"), "name": cols_l.get("name"),
        "sector": cols_l.get("sector") or cols_l.get("name"),
    }, index_key, region, default_exch)


_PARSERS = {"ishares": _parse_ishares, "invesco": _parse_invesco, "ssga": _parse_ssga}


def fetch_index(index_key: str, region: str) -> list:
    """Constituent rows for one index from its ETF holdings file; [] on any failure
    (unknown source, network error, unparseable file)."""
    src = SOURCES.get(index_key)
    if not src:
        return []
    url, kind, default_exch = src
    parser = _PARSERS.get(kind)
    if parser is None:
        return []
    try:
        rows = parser(_fetch(url), index_key, region, default_exch)
    except Exception:
        return []
    return rows if len(rows) >= 10 else []      # a near-empty parse is a failed parse


def fetch_all(indices: dict) -> dict:
    """{index_key: rows} for every index that fetched + parsed cleanly; failed or
    unsupported indices are simply ABSENT (caller keeps its previous membership).
    `indices` is equities.INDICES ({key: (display, bbg_ticker, region)})."""
    out = {}
    for key, (_disp, _idx, region) in indices.items():
        rows = fetch_index(key, region)
        if rows:
            out[key] = rows
    return out
