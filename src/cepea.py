"""cepea.py — CEPEA/ESALQ physical price indicators (Brazil), free daily spreadsheets.

WHY. Some Brazil-vs-US basis legs are not on B3 as futures. Soy is the big one: B3's SJC
contract is cash-settled ON the CME price, so it mirrors Chicago and says nothing about
Brazil. The real Brazilian soy price is the Paranaguá port indicator CEPEA (the
University of São Paulo's agri-economics centre) publishes every business day — the
reference the Brazilian trade itself quotes. Compared with the CBOT front it gives the
Brazil soy premium/discount that b3ag.py cannot.

SOURCE. https://www.cepea.esalq.usp.br/br/indicador/series/{product}.aspx?id={n}
returns the full daily history as a legacy .xls (verified 2026-10-09):

    soja id=92   INDICADOR DA SOJA CEPEA/ESALQ - PARANAGUÁ   2006-03-13 →   R$ + US$/sack
    soja id=12   INDICADOR DA SOJA CEPEA/ESALQ - PARANÁ      1997-07-29 →   (state, interior)

TWO TRAPS

  * THE .xls FAILS xlrd's CORRUPTION CHECK ("Workbook corruption: seen[2] == 4") on every
    download — CEPEA's generator writes a slightly malformed compound document. It reads
    correctly with `ignore_workbook_corruption=True`; that flag is required, not a hack
    around a bad file.
  * CEPEA PUBLISHES ITS OWN US$ COLUMN, converted at its own daily rate. We use it as-is
    rather than re-converting R$ at PTAX: it is the series the trade quotes, and a second
    conversion would only add a timing mismatch.

UNITS. 60 kg sack; a soybean bushel is 27.2155422 kg, so 1 sack = 2.204623 bu and
c/bu = US$/sack ÷ 2.204623 × 100 — CBOT soybeans' own quote unit.

CLI:  python src/cepea.py
"""
from __future__ import annotations

import io
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
STORE = _ROOT / "data" / "signals" / "cepea"

_URL = "https://www.cepea.esalq.usp.br/br/indicador/series/{product}.aspx?id={id}"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/130 Safari/537.36"}

BU_PER_SACK_SOY = 60 / 27.2155422     # 2.204623 bu of soybeans in a 60 kg sack

SERIES = {
    "soy_paranagua": dict(product="soja", id=92, name="Soy, Paranaguá (CEPEA/ESALQ)",
                          us="S A Comdty", us_unit="c/bu",
                          to_us=lambda usd_sack: usd_sack / BU_PER_SACK_SOY * 100),
    # Mill-gate hydrous ethanol, R$/LITRE, net of ICMS and PIS/Cofins, ex-freight — the
    # price a São Paulo mill weighs against sugar. Weekly. (B3's ETH settles on the
    # Paulínia DAILY indicator, which runs a steady ~R$100/m3 above this — a delivery
    # premium, NOT tax: the gap did not move when PIS/Cofins changed in 2025-05 and
    # 2026-09; checked month by month 2024-11..2026-10.) No US leg: brbasis turns it
    # into sugar parity.
    "ethanol_hydrous_sp": dict(product="etanol", id=103,
                               name="Hydrous ethanol, São Paulo (CEPEA/ESALQ, net of tax)"),
}


def _download(product: str, sid: int, timeout: int = 60) -> bytes:
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(_URL.format(product=product, id=sid), headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CEPEA {product} id={sid}: {last}")


def parse_xls(blob: bytes) -> pd.DataFrame:
    """CEPEA sheet -> date, brl, usd (per sack). Header rows are skipped by shape, not by
    position: data rows are the ones whose first cell parses as dd/mm/yyyy."""
    import os
    import xlrd
    with open(os.devnull, "w") as quiet:   # xlrd narrates the corruption it tolerates
        bk = xlrd.open_workbook(file_contents=blob, ignore_workbook_corruption=True,
                                logfile=quiet)
    raw = pd.read_excel(bk, header=None, engine="xlrd")
    dt = pd.to_datetime(raw[0].astype(str).str.strip(), format="%d/%m/%Y", errors="coerce")
    df = pd.DataFrame({"date": dt,
                       "brl": pd.to_numeric(raw[1], errors="coerce"),
                       "usd": pd.to_numeric(raw[2], errors="coerce")})
    return df.dropna(subset=["date", "usd"]).sort_values("date").reset_index(drop=True)


def _path(key: str) -> Path:
    return STORE / f"{key}.parquet"


def update(keys=None) -> dict:
    """Re-download each series in full (one ~0.5 MB file each) and replace the store.
    A failed or implausibly short download keeps the last good store."""
    STORE.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in keys or SERIES:
        s = SERIES[key]
        try:
            df = parse_xls(_download(s["product"], s["id"]))
            old = pd.read_parquet(_path(key)) if _path(key).exists() else None
            if old is not None and len(df) < 0.9 * len(old):
                raise RuntimeError(f"only {len(df)} rows vs {len(old)} stored — kept old")
            tmp = _path(key).with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(_path(key))
            out[key] = f"{len(df)} rows to {df['date'].max().date()}"
        except Exception as e:
            out[key] = f"skipped: {e}"
    return out


def load(key: str) -> pd.DataFrame:
    p = _path(key)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "brl", "usd"])


def basis(key: str = "soy_paranagua") -> pd.DataFrame:
    """Physical Brazil leg vs the US futures front (RAW settle), both in the US unit."""
    from . import deepstore
    s = SERIES[key]
    phys = load(key)
    if phys.empty:
        return pd.DataFrame()
    px = deepstore.get_raw([s["us"]])
    ct = deepstore.get_contracts([s["us"]])
    if px.empty:
        return pd.DataFrame()
    us = pd.DataFrame({"us_px": px[s["us"]]})
    us["us_contract"] = ct[s["us"]] if not ct.empty else None
    us.index = pd.to_datetime(us.index)
    us = us.dropna(subset=["us_px"]).rename_axis("date").reset_index()
    out = phys.merge(us, on="date", how="inner")
    out["br_px"] = s["to_us"](out["usd"])
    out["basis"] = out["br_px"] - out["us_px"]
    out["basis_pct"] = out["basis"] / out["us_px"] * 100
    return out[["date", "brl", "usd", "br_px", "us_contract", "us_px",
                "basis", "basis_pct"]].sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    print(f"CEPEA {datetime.now():%Y-%m-%d %H:%M}: {update()}")
