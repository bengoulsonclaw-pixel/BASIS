"""Client ETFs — the curated fund watchlist on the Equities side.

The desk's client flow trades a known list of ETFs; unlike the index pages there is
no membership problem — the list IS the universe. Everything rides free Yahoo
Finance through the yfin adapter (Bloomberg-style tickers at the seams, zero
Terminal hits): overnight quotes + a short history for the σ sizing, and a
fund-metrics tearsheet (AUM, expense ratio, yield, NAV, beta, 3Y return) from
Ticker.info, cached to data/equities/etf_info.json behind a staleness guard so
the page opens instantly and only re-pulls when asked or stale.

No Streamlit here; app.py owns the page.
"""
from __future__ import annotations

import json
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd

from src import yfin

_INFO_FILE = Path(__file__).resolve().parents[1] / "data" / "equities" / "etf_info.json"
_INFO_MAX_AGE_DAYS = 7

# root ticker · short display name · desk group (the sector-like band on the page)
ETFS = [
    ("SPY",  "S&P 500",             "US broad"),
    ("QQQ",  "Nasdaq 100",          "US broad"),
    ("MAGS", "Magnificent Seven",   "US broad"),
    ("SMH",  "Semiconductors",      "Sector & theme"),
    ("XLF",  "US Financials",       "Sector & theme"),
    ("XLU",  "US Utilities",        "Sector & theme"),
    ("ARKK", "ARK Innovation",      "Sector & theme"),
    ("EWZ",  "Brazil (MSCI)",       "Country"),
    ("ECH",  "Chile (MSCI)",        "Country"),
    ("BIL",  "1-3M T-Bills",        "Fixed income"),
    ("TLT",  "20+Y Treasuries",     "Fixed income"),
    ("HYG",  "High Yield credit",   "Fixed income"),
    ("LQD",  "IG credit",           "Fixed income"),
]
GROUPS = list(dict.fromkeys(g for _, _, g in ETFS))


def tickers() -> list:
    return [f"{root} US Equity" for root, _, _ in ETFS]


def movers_frame() -> pd.DataFrame:
    """One row per ETF: ETF · Name · Group · last · pct · sigma. σ = the overnight move
    in standard deviations of the fund's own ~1-month daily moves — the same sizing rule
    as the FICC and index-equities heatmaps. Empty frame when Yahoo is unreachable."""
    tk = tickers()
    q = yfin.get_quotes(tk)
    h = yfin.get_history(tk, sessions=24)
    rows = []
    for (root, name, group), t in zip(ETFS, tk):
        last = float(q.loc[t, "last"]) if t in q.index else np.nan
        pct = float(q.loc[t, "pct"]) if t in q.index else np.nan
        sigma = np.nan
        if t in getattr(h, "columns", []):
            dd = h[t].dropna().pct_change().dropna() * 100.0
            sd = float(dd.tail(21).std())
            if sd and sd == sd and pct == pct:
                sigma = pct / sd
        rows.append({"ETF": root, "Name": name, "Group": group,
                     "last": last, "pct": pct, "sigma": sigma})
    f = pd.DataFrame(rows)
    return f if f["pct"].notna().any() else pd.DataFrame()


# ── fund tearsheet (cached Ticker.info) ───────────────────────────────────────
_INFO_KEYS = {
    "longName": "name", "category": "category", "totalAssets": "aum",
    "navPrice": "nav", "dividendYield": "yield_pct", "netExpenseRatio": "expense_pct",
    "beta3Year": "beta_3y", "threeYearAverageReturn": "ret_3y", "trailingPE": "pe",
}


def _read_info() -> dict:
    try:
        return json.loads(_INFO_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def info_age_days() -> int | None:
    """Age of the cached tearsheet pull in days; None when never pulled."""
    stamp = _read_info().get("_pulled", "")
    try:
        return int((pd.Timestamp.today().normalize() - pd.Timestamp(stamp)).days)
    except (TypeError, ValueError):
        return None


def fund_info(force: bool = False) -> dict:
    """{root: {name, category, aum, nav, yield_pct, expense_pct, beta_3y, ret_3y, pe}}
    from Yahoo Ticker.info, cached to disk for _INFO_MAX_AGE_DAYS. A dead pull keeps
    the cached file (never wipes)."""
    age = info_age_days()
    cached = _read_info()
    if not force and age is not None and age < _INFO_MAX_AGE_DAYS:
        return {k: v for k, v in cached.items() if not k.startswith("_")}
    out = {}
    try:
        with futures.ThreadPoolExecutor(max_workers=yfin._INFO_WORKERS) as ex:
            futs = {ex.submit(yfin._info_one, root): root for root, _, _ in ETFS}
            for f in futures.as_completed(futs):
                info = f.result() or {}
                vals = {ours: info.get(theirs) for theirs, ours in _INFO_KEYS.items()}
                if any(v is not None for v in vals.values()):
                    out[futs[f]] = vals
    except Exception:
        out = {}
    if len(out) < len(ETFS) // 2:                 # mostly-dead pull -> keep the cache
        return {k: v for k, v in cached.items() if not k.startswith("_")}
    payload = dict(out)
    payload["_pulled"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    _INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INFO_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return out


def fmt_aum(v) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    return f"${v / 1e9:,.1f}bn" if v >= 1e9 else f"${v / 1e6:,.0f}mm"


# ── Hot Sheet provider ────────────────────────────────────────────────────────
RADAR_SIGMA_FLOOR = 2.0    # outsized day: |move| ≥ 2 of the fund's own ~1-month daily σ
RADAR_PCT_FLOOR = 1.5      # fallback bar (|%| move) when no history is cached for the σ
RADAR_MAX_ITEMS = 3        # editorial cap — the biggest movers only


def radar_items() -> list:
    """Hot Sheet provider: client-list funds with an outsized overnight move —
    movers_frame()'s σ sizing rule (the move in the fund's own ~1-month daily σ),
    falling back to a flat % bar when no history is cached. Pure disk: reads the
    morning equities snapshot parquets only — movers_frame() itself pulls live
    Yahoo — and only a snapshot written TODAY (yesterday's pct is not the day's
    move). Quiet until the morning pull's ticker list covers the ETFs."""
    from src import equities, hotsheet

    if not equities._file_is_todays(equities._QUOTES_FILE):
        return []
    q = equities._read_parquet(equities._QUOTES_FILE)
    if q is None or q.empty or "pct" not in q:
        return []
    h = equities._read_parquet(equities._HISTORY_FILE)
    items = []
    for (root, name, _group), t in zip(ETFS, tickers()):
        pct = float(q["pct"].get(t, np.nan))
        if pct != pct:
            continue
        sigma = np.nan
        if h is not None and t in getattr(h, "columns", []):
            dd = h[t].dropna().pct_change().dropna() * 100.0   # the movers_frame σ
            sd = float(dd.tail(21).std())
            if sd and sd == sd:
                sigma = pct / sd
        if sigma == sigma:                                     # σ path: the fund's own bar
            if abs(sigma) < RADAR_SIGMA_FLOOR:
                continue
            heat = min(100.0, abs(sigma) * 33.0)
            metric, sub = f"{sigma:+.1f}σ", "of its own 21-day daily σ"
        else:                                                  # no σ history: flat % bar
            if abs(pct) < RADAR_PCT_FLOOR:
                continue
            heat = min(100.0, abs(pct) / 3.0 * 100.0)
            metric, sub = f"{pct:+.1f}%", f"vs a ±{RADAR_PCT_FLOOR:.1f}% day bar"
        items.append(hotsheet.item(
            tag="EQ-ETF", key=f"{root}:move", section="Client ETFs",
            text=(f"**{name}** ({root}) moved **{pct:+.1f}%** on the day — "
                  "beyond its usual daily range."),
            metric=metric, sub=sub, heat=heat, value=pct,
            ticker=t, page="eq:ETFs", book="equities"))
    items.sort(key=lambda it: -it["heat"])
    return items[:RADAR_MAX_ITEMS]
