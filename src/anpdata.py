"""anpdata.py — real Brazilian crude production per company, from the regulator.

ANP (Agência Nacional do Petróleo) publishes monthly per-WELL production as open
data, and every row names its `Operador`. Summing oil by operator gives an official,
sourced answer to "who produces Brazil's crude" — replacing the desk estimates that
used to fill that table.

Two things this data is NOT, both of which the page must say out loud:

  * It is OPERATED production, not equity. Petrobras operates most of the pre-salt,
    including fields where Shell, TotalEnergies, CNOOC, Equinor and Galp hold large
    working interests, so Petrobras' operated share is far above its equity share.
    ANP publishes consortium participation separately; until that is wired, equity
    is simply not something we know.
  * It is not current. The open-data drop runs 2005 -> 2023-12 and stops; no 2024+
    files exist under any of ANP's five file-naming conventions, and the dynamic-panel
    and per-field pages 404. Verified 2026-08-21. The vintage travels with the data
    so nothing downstream can present it as today's number.

File naming changed five times across the years, so URLs are DISCOVERED by scraping
the dataset page rather than constructed — building them by pattern breaks on the
next rename.

CLI:  python src/anpdata.py [--force]
"""
from __future__ import annotations

import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
SIGNALS = _ROOT / "data" / "signals"
STORE = SIGNALS / "anp_crude.parquet"

INDEX_URL = ("https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
             "producao-de-petroleo-e-gas-natural-por-poco")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124 Safari/537.36"}
MONTHS_BACK = 12          # a year of months — one month alone is a maintenance artefact
OIL_COL = "Petróleo (bbl/dia)"
OP_COL = "Operador"
PERIOD_COL = "Período"


def _get(url: str, timeout: int = 300, tries: int = 3) -> bytes:
    """gov.br drops connections mid-download often enough to need a retry."""
    last = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=_UA), timeout=timeout).read()
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def _decode(blob: bytes) -> str:
    """ANP ships SOME months as UTF-8 and others as latin-1. Decoding everything as
    latin-1 never errors — it just silently turns 'Petróleo' into 'PetrÃ³leo', which
    made the header undetectable and nine of twelve months parse to nothing. Try
    strict UTF-8 first and only fall back."""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return blob.decode(enc)
        except UnicodeDecodeError:
            continue
    return blob.decode("latin-1", "ignore")


def _num(series: pd.Series) -> pd.Series:
    """Numbers from either decimal convention. '1.234,56' -> dot is the thousands
    separator; '1234.56' or '1234,56' -> that mark is the decimal point."""
    s = series.astype(str).str.strip()
    both = s.str.contains(r"\.", regex=True) & s.str.contains(",")
    s = s.mask(both, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def discover() -> list[tuple[str, str]]:
    """[(yyyy-mm, url)] for every monthly archive on the dataset page, oldest first.

    Handles all five naming conventions ANP has used: producao-por-poco-2015.zip,
    2021_03_producao.zip, 2021-08-producao.zip, producao-2022-04.zip, producao-04.zip
    (the last one takes its year from the folder).
    """
    html = _get(INDEX_URL, timeout=120).decode("utf-8", "ignore")
    urls = sorted(set(re.findall(
        r'href="(https://www\.gov\.br/anp/[^"]+\.zip)"', html, re.I)))
    out = []
    for u in urls:
        tail = u.rsplit("/", 1)[-1]
        folder = u.rsplit("/", 2)[-2]
        m = (re.search(r"(\d{4})[_-](\d{2})[_-]?produc", tail, re.I)
             or re.search(r"produc\w*[_-](\d{4})[_-](\d{2})", tail, re.I))
        if m:
            out.append((f"{m.group(1)}-{m.group(2)}", u))
            continue
        m = re.search(r"produc\w*[_-](\d{2})\.zip", tail, re.I)
        if m and re.fullmatch(r"\d{4}", folder):          # year lives in the folder
            out.append((f"{folder}-{m.group(1)}", u))
            continue
        m = re.search(r"por-poco-(\d{4})\.zip", tail, re.I)
        if m:
            out.append((f"{m.group(1)}-full", u))          # a whole-year archive
    return sorted(out)


def _parse_zip(raw: bytes) -> pd.DataFrame:
    """Every CSV inside one archive -> [period, operator, bopd]. ANP ships latin-1,
    semicolon-separated, comma-decimal, and sometimes a banner above the header."""
    frames = []
    zf = zipfile.ZipFile(io.BytesIO(raw))
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        # Each archive holds Terra (onshore), Mar (offshore) and PreSal — and PRE-SALT
        # IS A SUBSET OF OFFSHORE, not a third region. Summing all three double-counts
        # the pre-salt and lifts Brazil's total from ~3.5 to ~5.5 mb/d.
        if "presal" in name.lower().replace("-", "").replace("_", ""):
            continue
        lines = _decode(zf.open(name).read()).splitlines()
        hdr = next((i for i, ln in enumerate(lines) if OP_COL in ln and OIL_COL[:8] in ln), None)
        if hdr is None:
            continue
        try:
            # Read EVERYTHING as text. ANP mixes decimal conventions across months —
            # some ship "1615,7229", others "1615.7229" — and a fixed `decimal=` turns
            # the other style silently into NaN, which quietly dropped offshore (95% of
            # Brazil) out of three months while still "succeeding".
            df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";", dtype=str,
                             engine="python", on_bad_lines="skip")
        except Exception:
            continue
        if OP_COL not in df.columns:
            continue
        oil = next((c for c in df.columns if c.startswith("Petróleo")), None)
        if not oil:
            continue
        d = df[[PERIOD_COL, OP_COL, oil]].copy()
        d.columns = ["period", "operator", "bopd"]
        d["bopd"] = _num(d["bopd"])
        d["operator"] = d["operator"].astype(str).str.strip()
        frames.append(d.dropna(subset=["bopd"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refresh(force: bool = False, max_age_hours: float = 24 * 30) -> pd.DataFrame:
    """Download the most recent MONTHS_BACK monthly archives and cache oil-by-operator.
    ANP's drop is static and years old, so the cache is refreshed monthly, not daily."""
    SIGNALS.mkdir(parents=True, exist_ok=True)
    if not force and STORE.exists() and (time.time() - STORE.stat().st_mtime) < max_age_hours * 3600:
        try:
            return pd.read_parquet(STORE)
        except Exception:
            pass
    monthly = [(k, u) for k, u in discover() if not k.endswith("full")]
    if not monthly:
        raise RuntimeError("ANP dataset page listed no monthly archives — page layout changed?")
    frames, failed = [], []
    for key, url in monthly[-MONTHS_BACK:]:
        try:
            d = _parse_zip(_get(url))
        except Exception as exc:
            failed.append(f"{key}: {type(exc).__name__}")
            continue
        if d.empty:
            failed.append(f"{key}: parsed to nothing")
        else:
            frames.append(d.assign(file_month=key))
    if failed:
        # loud on purpose: silently averaging whichever months happened to parse is
        # exactly the kind of invisible gap this module exists to avoid
        print(f"  ANP: {len(failed)} of {len(monthly[-MONTHS_BACK:])} months unusable "
              f"-> {', '.join(failed)}")
    if not frames:
        raise RuntimeError("ANP archives downloaded but none parsed — schema changed?")
    out = pd.concat(frames, ignore_index=True)

    # A month can "succeed" while having lost a whole region: offshore is ~95% of
    # Brazilian crude, so an encoding or decimal quirk that drops it leaves a month
    # reading ~165k bbl/d instead of ~3.5m and nothing raises. Both happened. Any
    # month far below the median is dropped and named rather than averaged in.
    totals = out.groupby("file_month")["bopd"].sum()
    if len(totals) >= 3:
        median = float(totals.median())
        partial = [m for m, v in totals.items() if v < median * 0.5]
        if partial:
            print(f"  ANP: dropping {len(partial)} partial month(s) — "
                  f"{', '.join(f'{m} ({totals[m]:,.0f} bbl/d vs median {median:,.0f})' for m in partial)}")
            out = out[~out["file_month"].isin(partial)]
    try:
        out.to_parquet(STORE, index=False)
    except Exception:
        pass
    return out


def by_operator(df: pd.DataFrame | None = None) -> dict:
    """{operator: mean bbl/day} over the cached window, plus the window's own metadata.

    The mean across whole months is the honest annual-rate proxy: a single month can
    be distorted by a platform stoppage.
    """
    df = df if df is not None else refresh()
    if df is None or df.empty:
        return {}
    months = sorted(df["file_month"].dropna().unique())
    per_month = df.groupby(["file_month", "operator"])["bopd"].sum().reset_index()
    mean_by_op = per_month.groupby("operator")["bopd"].mean().sort_values(ascending=False)
    return {
        "operators": {k: round(float(v), 1) for k, v in mean_by_op.items() if v > 0},
        "total_bopd": round(float(mean_by_op.sum()), 1),
        "months": months, "n_months": len(months),
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "source": "ANP — Produção de petróleo e gás natural por poço (open data)",
        "basis": "operator",
    }


def main(argv: list[str]) -> int:
    force = "--force" in argv
    df = refresh(force=force)
    got = by_operator(df)
    print(f"ANP crude — {got['n_months']} months, {got['first_month']} to {got['last_month']}")
    print(f"  total {got['total_bopd']:,.0f} bbl/d across {len(got['operators'])} operators")
    for i, (op, v) in enumerate(list(got["operators"].items())[:12]):
        print(f"   {i+1:>2}. {op:<38} {v:>12,.0f} bbl/d  {v / got['total_bopd'] * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
