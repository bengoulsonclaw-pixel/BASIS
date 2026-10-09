"""a3ag.py — Argentine grain futures (A3 Mercados, ex-Matba-Rofex), free from the exchange.

WHY. The South-America-vs-Chicago question doesn't stop at Brazil: Argentina is the
world's top soymeal/oil exporter and a top-3 corn exporter, and Rosario is its price.
A3's Rosario futures trade in US DOLLARS per tonne, which sidesteps Argentina's
multiple-exchange-rate problem entirely — no peso conversion, no "which dollar?".

SOURCE. The exchange's public market-statistics centre (cem.matbarofex.com.ar) is backed
by an open JSON API — no login, no key (verified 2026-10-09):

    https://apicem.matbarofex.com.ar/api/v2/closing-prices
        ?product=SOJ Dolar MATba&type=FUT&from=YYYY-MM-DD&to=YYYY-MM-DD&pageSize=5000&market=ROFX

Each row: dateTime, symbol (SOJ.ROS/NOV26), settlement, OHLC, volume, openInterest.
Daily history starts mid-2020 (2019 dates return nothing — Matba and Rofex merged in
2019). A whole year for one product is one ~1.5 s call (≈1,600 rows).

    product            symbol root   quote    US pair
    SOJ Dolar MATba    SOJ.ROS       USD/t    CBOT soybeans  (S A)
    MAI Dolar MATba    MAI.ROS       USD/t    CBOT corn      (C A)

TRAPS

  * MONTH CODES ARE SPANISH: ENE FEB MAR ABR MAY JUN JUL AGO SEP OCT NOV DIC.
  * NOT A FREE-MARKET PRICE. Rosario is an INTERIOR price net of Argentina's export
    duties (derechos de exportación — soy has carried ~24-33% in this history), so the
    basis to CBOT is deeply negative and moves in STEPS when the duty rate changes. The
    z-score will flag those policy steps; the page says so.
  * The bundle's other endpoints (market-position-data) only accept TV/OI — closing-prices
    is the one that carries settlements.

UNITS. 1 t = 36.7437 bu of soybeans (27.2155 kg/bu) and 39.3683 bu of corn (25.4012 kg/bu),
so c/bu = USD/t ÷ bu_per_t × 100 — CBOT's own quote unit.

CLI:  python src/a3ag.py [--from-year YYYY]
"""
from __future__ import annotations

import functools
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
# Tracked on purpose (like data/brazil/): the post-pull auto-push carries it to the VPS.
STORE = _ROOT / "data" / "argentina"
SETTLES = STORE / "a3_settles.parquet"
META = STORE / "meta.json"

_API = "https://apicem.matbarofex.com.ar/api/v2/closing-prices?"
_UA = {"User-Agent": "Mozilla/5.0"}
FIRST_YEAR = 2020

MONTHS_ES = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12}

PRODUCTS = {
    "SOJ": dict(product="SOJ Dolar MATba", name="Soybeans, Rosario", bu_per_t=1000 / 27.2155422,
                us="S A Comdty", us_unit="c/bu"),
    "MAI": dict(product="MAI Dolar MATba", name="Corn, Rosario", bu_per_t=1000 / 25.40117272,
                us="C A Comdty", us_unit="c/bu"),
}


def _get(params: dict) -> dict:
    url = _API + urllib.parse.urlencode({**params, "market": "ROFX"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                        timeout=120) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"A3 closing-prices: {last}")


def contract_ym(symbol: str) -> tuple[int, int] | None:
    """'SOJ.ROS/NOV26' -> (2026, 11)."""
    try:
        tail = symbol.split("/")[1]
        return (2000 + int(tail[3:5]), MONTHS_ES[tail[:3]])
    except (IndexError, KeyError, ValueError):
        return None


def fetch_year(root: str, year: int) -> pd.DataFrame:
    """Every plain futures settlement for one product-year (paged defensively)."""
    p = PRODUCTS[root]
    rows, page = [], 1
    while True:
        d = _get({"product": p["product"], "type": "FUT", "pageSize": 5000, "page": page,
                  "from": f"{year}-01-01", "to": f"{year}-12-31"})
        rows += d.get("data", [])
        if len(rows) >= (d.get("totalEntries") or 0) or not d.get("data"):
            break
        page += 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Plain outrights only: ROOT.ROS/MMMYY — spreads and other venues have other shapes.
    df = df[df["symbol"].str.fullmatch(rf"{root}\.ROS/[A-Z]{{3}}\d\d")]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["dateTime"].str[:10]),
        "root": root, "ticker": df["symbol"],
        "settle": pd.to_numeric(df["settlement"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "oi": pd.to_numeric(df["openInterest"], errors="coerce"),
    })
    return out.dropna(subset=["settle"])


def _meta() -> dict:
    try:
        return json.loads(META.read_text())
    except Exception:
        return {"years_done": {}}


def update(from_year: int | None = None) -> int:
    """Past years are fetched once (frozen); the current year is re-fetched every run
    (one call per product). Atomic write; a failed product keeps what's stored."""
    STORE.mkdir(parents=True, exist_ok=True)
    m = _meta()
    done = m.get("years_done", {})
    this_year = date.today().year
    old = pd.read_parquet(SETTLES) if SETTLES.exists() else pd.DataFrame()
    parts, n = [old], 0
    for root in PRODUCTS:
        for y in range(from_year or FIRST_YEAR, this_year + 1):
            if y < this_year and y in done.get(root, []):
                continue
            try:
                df = fetch_year(root, y)
            except Exception as e:
                print(f"  (A3 {root} {y} skipped: {e})")
                continue
            parts.append(df)
            n += len(df)
            if y < this_year and not df.empty:
                done.setdefault(root, []).append(y)
    new = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else pd.DataFrame()
    if not new.empty:
        new = (new.drop_duplicates(["date", "ticker"], keep="last")
                  .sort_values(["date", "ticker"]).reset_index(drop=True))
        tmp = SETTLES.with_suffix(".tmp")
        new.to_parquet(tmp, index=False)
        tmp.replace(SETTLES)
    m["years_done"] = {k: sorted(set(v)) for k, v in done.items()}
    m["updated"] = datetime.now().isoformat(timespec="seconds")
    META.write_text(json.dumps(m, indent=1))
    _read_cached.cache_clear()
    return n


@functools.lru_cache(maxsize=2)
def _read_cached(_mtime: float) -> pd.DataFrame:
    return pd.read_parquet(SETTLES)


def _read() -> pd.DataFrame:
    if not SETTLES.exists():
        return pd.DataFrame(columns=["date", "root", "ticker", "settle", "volume", "oi"])
    return _read_cached(SETTLES.stat().st_mtime).copy()


def basis(root: str) -> pd.DataFrame:
    """Rosario most-active (max OI) vs the CBOT front RAW settle, both in c/bu."""
    from . import deepstore
    p = PRODUCTS[root]
    a3 = _read()
    a3 = a3[a3["root"] == root]
    if a3.empty:
        return pd.DataFrame()
    act = (a3.sort_values(["date", "oi", "volume"], na_position="first")
             .groupby("date").tail(1).copy())
    act["a3_px"] = act["settle"] / p["bu_per_t"] * 100
    px = deepstore.get_raw([p["us"]])
    ct = deepstore.get_contracts([p["us"]])
    if px.empty:
        return pd.DataFrame()
    us = pd.DataFrame({"us_px": px[p["us"]]})
    us["us_contract"] = ct[p["us"]] if not ct.empty else None
    us.index = pd.to_datetime(us.index)
    us = us.dropna(subset=["us_px"]).rename_axis("date").reset_index()
    for f in (us, act):
        f["date"] = f["date"].astype("datetime64[ns]")
    out = act.merge(us, on="date", how="inner")
    out["basis"] = out["a3_px"] - out["us_px"]
    out["basis_pct"] = out["basis"] / out["us_px"] * 100
    return out.rename(columns={"ticker": "a3_contract", "settle": "a3_usd_t"})[
        ["date", "a3_contract", "a3_usd_t", "a3_px", "us_contract", "us_px", "basis",
         "basis_pct", "oi", "volume"]].sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    fy = int(args[args.index("--from-year") + 1]) if "--from-year" in args else None
    n = update(fy)
    df = _read()
    print(f"A3: {n} rows fetched; store {len(df)} rows, "
          f"{df['date'].min().date() if len(df) else '-'} to {df['date'].max().date() if len(df) else '-'}")
