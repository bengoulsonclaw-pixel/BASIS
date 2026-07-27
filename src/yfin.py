"""Yahoo Finance adapter for the BASIS Equities side.

Free, keyless replacement for the Bloomberg equity pulls: settlement/close
prices, overnight quotes, market caps, the fundamentals set and next-earnings
dates all come off Yahoo via `yfinance`. The Equities data layers
(equities / eqfunda / eqcorr) try this module FIRST when
BASIS_EQ_SOURCE=yfinance (the default) and fall back to their old
Bloomberg / snapshot / mock chains when Yahoo is unreachable — so the work-PC
Terminal now only supplies what Yahoo can't: index membership (INDX_MEMBERS).

Everything speaks BLOOMBERG tickers at the seams ('AAPL US Equity',
'BP/ LN Equity') and maps to Yahoo symbols internally (AAPL, BP.L), so no
caller changes shape. Values are normalised to the BDP conventions eqfunda
already formats: pct fields in percent, mm fields in millions, EQY_SH_OUT in
millions of shares, BEST_ANALYST_RATING 1–5 with 5 best.

Unit notes (verified against yfinance 1.5.x):
  dividendYield / debtToEquity  -> already percent, passed through
  margins / ROE / growth / payoutRatio -> fractions, x100 here
  marketCap for .L names        -> major unit (GBP) even though prices are GBp
  FCF yield is only computed when the financial currency matches the listing
  currency's major unit — a USD-reporting LN name would be off by the fx rate.

No Streamlit here. No imports from the other src modules (they import us).
"""
from __future__ import annotations

from concurrent import futures

import numpy as np
import pandas as pd

# Bloomberg exchange code -> Yahoo suffix. US composite + venue codes map to ''.
_US_CODES = {"US", "UW", "UN", "UA", "UB", "UC", "UD", "UF", "UM", "UP",
             "UQ", "UR", "UT", "UV", "UX"}
_SUFFIX = {
    "LN": ".L",                    # London
    "GY": ".DE", "GR": ".DE",      # Xetra / Germany composite
    "FP": ".PA",                   # Paris
    "NA": ".AS",                   # Amsterdam
    "SM": ".MC", "SQ": ".MC",      # Madrid
    "IM": ".MI",                   # Milan
    "BB": ".BR",                   # Brussels
    "SW": ".SW", "VX": ".SW",      # Switzerland
    "ID": ".IR",                   # Dublin
    "PL": ".LS",                   # Lisbon
    "DC": ".CO",                   # Copenhagen
    "SS": ".ST",                   # Stockholm
    "NO": ".OL",                   # Oslo
    "FH": ".HE",                   # Helsinki
    "AV": ".VI",                   # Vienna
    "CN": ".TO", "CT": ".TO",      # Toronto
    "JP": ".T", "JT": ".T",        # Tokyo
    "HK": ".HK",                   # Hong Kong
    "AU": ".AX", "AT": ".AX",      # Australia
}

_INFO_WORKERS = 6                  # parallel Ticker.info pulls — modest, Yahoo throttles bursts

# Names whose Yahoo symbol isn't the mechanical root+suffix ('ROOT EXCH' -> symbol).
_SPECIAL = {
    "STM FP": "STMPA.PA",          # STMicro Paris — Yahoo uses STMPA, STM is the NYSE line
    "NDA FH": "NDA-FI.HE",         # Nordea Helsinki — Yahoo keeps the FI country tag
}


def available() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except Exception:
        return False


def to_yahoo(ticker: str) -> str | None:
    """'BRK/B US Equity' -> 'BRK-B', 'BP/ LN Equity' -> 'BP.L', 'SAP GY Equity' ->
    'SAP.DE'. None when the exchange code has no Yahoo mapping."""
    parts = str(ticker).split()
    if len(parts) < 2:
        return None
    root, exch = parts[0], parts[1]
    if f"{root} {exch}" in _SPECIAL:
        return _SPECIAL[f"{root} {exch}"]
    root = root.rstrip("/").replace("/", "-")     # BP/ -> BP ; BRK/B -> BRK-B
    if exch in _US_CODES:
        return root
    sfx = _SUFFIX.get(exch)
    return f"{root}{sfx}" if sfx else None


def _symbol_map(tickers) -> dict:
    """{bloomberg ticker: yahoo symbol} for every mappable name, first-come on
    the rare yahoo-symbol collision."""
    out, seen = {}, set()
    for t in dict.fromkeys(tickers):
        y = to_yahoo(t)
        if y and y not in seen:
            out[t] = y
            seen.add(y)
    return out


def _close_frame(symbols: list, **kwargs) -> pd.DataFrame:
    """Batched yf.download -> date x yahoo-symbol close frame (tz-naive index)."""
    import yfinance as yf
    px = yf.download(symbols, progress=False, auto_adjust=False, threads=True, **kwargs)
    if px is None or px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"]
    else:                                          # single symbol: flat columns
        close = px[["Close"]].rename(columns={"Close": symbols[0]})
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()


def get_history(tickers, sessions: int = 60) -> pd.DataFrame:
    """DataFrame date x BLOOMBERG ticker of daily closes covering ~`sessions`
    trading days back. Empty frame when nothing maps or Yahoo is unreachable."""
    ym = _symbol_map(tickers)
    if not ym:
        return pd.DataFrame()
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(sessions * 7 / 5) + 40)
    close = _close_frame(list(ym.values()), start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), interval="1d")
    if close.empty:
        return close
    back = {y: b for b, y in ym.items()}
    close = close[[c for c in close.columns if c in back]].rename(columns=back)
    return close.dropna(how="all")


def get_quotes(tickers) -> pd.DataFrame:
    """DataFrame indexed by BLOOMBERG ticker, columns ['last','pct'] — latest
    close and % vs the prior close, off a batched 5-day pull (one request,
    not one per name)."""
    tickers = list(dict.fromkeys(tickers))
    out = pd.DataFrame(index=tickers, columns=["last", "pct"], dtype=float)
    ym = _symbol_map(tickers)
    if not ym:
        return out
    close = _close_frame(list(ym.values()), period="5d", interval="1d")
    for b, y in ym.items():
        s = close[y].dropna() if y in close.columns else pd.Series(dtype=float)
        if len(s) >= 1:
            out.loc[b, "last"] = float(s.iloc[-1])
        if len(s) >= 2 and s.iloc[-2] != 0:
            out.loc[b, "pct"] = float((s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0)
    return out


# ── fundamentals ──────────────────────────────────────────────────────────────
def _major_ccy(c: str) -> str:
    """Listing currency -> its major unit (GBp/GBX -> GBP, ZAc -> ZAR)."""
    c = (c or "").strip()
    return {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}.get(c, c.upper())


def _num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else np.nan


def _fund_values(info: dict) -> tuple:
    """One name's Ticker.info -> ({BDP numeric field: value}, {text field: str}),
    on the BDP conventions eqfunda formats (pct in %, mm in millions)."""
    g = info.get

    def pct(k):                                   # yahoo fraction -> percent
        return _num(g(k)) * 100.0

    def mm(k):                                    # full currency units -> millions
        return _num(g(k)) / 1e6

    debt, cash, ebitda = _num(g("totalDebt")), _num(g("totalCash")), _num(g("ebitda"))
    fcf, mcap = _num(g("freeCashflow")), _num(g("marketCap"))
    rec = _num(g("recommendationMean"))           # yahoo: 1 = strong buy ... 5 = sell
    fin_ccy = str(g("financialCurrency") or "").upper()
    px_ccy = _major_ccy(str(g("currency") or ""))
    vals = {
        "PE_RATIO": _num(g("trailingPE")),
        "BEST_PE_RATIO": _num(g("forwardPE")),
        "PX_TO_BOOK_RATIO": _num(g("priceToBook")),
        "PX_TO_SALES_RATIO": _num(g("priceToSalesTrailing12Months")),
        "BEST_PEG_RATIO_CUR_YR": _num(g("trailingPegRatio")),
        "EV_TO_T12M_EBITDA": _num(g("enterpriseToEbitda")),
        "GROSS_MARGIN": pct("grossMargins"),
        "OPER_MARGIN": pct("operatingMargins"),
        "PROF_MARGIN": pct("profitMargins"),
        "EBITDA_TO_REVENUE": pct("ebitdaMargins"),
        "RETURN_COM_EQY": pct("returnOnEquity"),
        "TOT_DEBT_TO_TOT_EQY": _num(g("debtToEquity")),        # already %
        "NET_DEBT_TO_EBITDA": ((debt - cash) / ebitda
                               if ebitda == ebitda and ebitda > 0 else np.nan),
        "CUR_RATIO": _num(g("currentRatio")),
        "SALES_GROWTH": pct("revenueGrowth"),
        "EPS_GROWTH": pct("earningsGrowth"),
        "CF_CASH_FROM_OPER": mm("operatingCashflow"),
        "CF_FREE_CASH_FLOW": mm("freeCashflow"),
        "NET_INCOME": mm("netIncomeToCommon"),
        "FREE_CASH_FLOW_YIELD": (fcf / mcap * 100.0
                                 if fcf == fcf and mcap == mcap and mcap > 0
                                 and fin_ccy == px_ccy else np.nan),
        "EQY_DVD_YLD_IND": _num(g("dividendYield")),           # already %
        "DVD_PAYOUT_RATIO": pct("payoutRatio"),
        "EQY_SH_OUT": _num(g("sharesOutstanding")) / 1e6,
        "CRNCY_ADJ_MKT_CAP": mm("marketCap"),
        "BEST_ANALYST_RATING": (6.0 - rec) if rec == rec else np.nan,   # -> 5 best, like BDP
        # no yahoo source: ROIC, interest cover, 3Y-avg revenue growth -> NaN ('—' on the page)
    }
    texts = {"CRNCY": fin_ccy or px_ccy or None}
    ts = g("earningsTimestampStart") or g("earningsTimestamp")
    try:
        texts["EXPECTED_REPORT_DT"] = pd.Timestamp(int(ts), unit="s").strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        texts["EXPECTED_REPORT_DT"] = None
    return vals, texts


def _info_one(sym: str) -> dict:
    import yfinance as yf
    try:
        return yf.Ticker(sym).info or {}
    except Exception:
        return {}


def fundamentals_long(tickers, asof: str, fields) -> pd.DataFrame:
    """Long-format fundamentals pull matching eqfunda's DB schema
    (as_of / ticker / field / value / text) for every field in `fields` — NaN/None
    where Yahoo has no equivalent, so the schema matches a Bloomberg pull.
    One Ticker.info request per name, threaded; a dead name contributes NaNs."""
    ym = _symbol_map(tickers)
    infos: dict = {}
    with futures.ThreadPoolExecutor(max_workers=_INFO_WORKERS) as ex:
        futs = {ex.submit(_info_one, y): b for b, y in ym.items()}
        for f in futures.as_completed(futs):
            infos[futs[f]] = f.result()
    rows = []
    for t in tickers:
        vals, texts = _fund_values(infos.get(t, {}))
        for field in fields:
            if field in ("CRNCY", "EXPECTED_REPORT_DT"):
                rows.append({"as_of": asof, "ticker": t, "field": field,
                             "value": np.nan, "text": texts.get(field)})
            else:
                rows.append({"as_of": asof, "ticker": t, "field": field,
                             "value": _num(vals.get(field, np.nan)), "text": None})
    return pd.DataFrame(rows)
