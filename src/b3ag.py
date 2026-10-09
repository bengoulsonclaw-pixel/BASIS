"""b3ag.py — B3's agricultural futures, free from the exchange itself, against their US pairs.

WHY. The Brazil-vs-US question in ags is a BASIS question: what the Brazilian interior or
port price is doing relative to the Chicago/NY benchmark. B3 lists the Brazilian legs.
Liquidity measured off the 2025-10-08 Price Report (contracts, all months):

    root  product                         quote      volume     OI    US pair
    CCM   corn, Campinas                  BRL/sack    6,558  124,448  CBOT corn  (C A)
    BGI   live cattle (CEPEA index)       BRL/@       4,823   45,392  CME cattle (LCA)
    ICF   arabica 4/5, SP/MG warehouses   USD/bag     1,569    6,893  ICE "C"    (KCA)
    ETH   hydrous ethanol, Paulínia       BRL/m3        270    9,228  ICE sugar  (SBA)

Deliberately NOT here: SJC soy is cash-settled ON the CME price, so it mirrors Chicago and
carries no Brazil basis (the real one is the Paranaguá premium — a CEPEA physical series,
not a B3 future). CNL conilon traded 0 lots with 90 OI. B3 lists no sugar future at all;
ethanol is the Brazil sugar lever, because mills swing cane between the two on parity.

SOURCE. B3's "Pesquisa por pregão" archive publishes the full daily Price Report
(BVBG.086.01 XML) for every listed instrument, free, no login:

    https://www.b3.com.br/pesquisapregao/download?filelist=PR{yymmdd}.zip

Each record carries the official settlement (AdjstdQt), open interest, volume, OHLC and the
closing book. Verified live 2026-10-09: the archive reaches back to 2018 (2017 dates answer
the empty 22-byte stub); ICFZ25 on 2025-10-08 settled 461.35 USD/bag, OI 3,152.

FOUR TRAPS, ALL LOAD-BEARING

  * EACH REPORT HOLDS TWO SESSIONS. Alongside the day's records, PR{yymmdd} carries the
    NEXT session's early records for the same tickers (TradDt = d+1, settle still d's).
    Same ticker, same settle, different volume/OHLC — so a parser keyed on ticker alone
    silently reports the next morning's 8 lots instead of the day's 480. Filter on each
    block's own <TradDt>.

  * NESTED ZIPS, ~110-140 MB OF XML PER DAY. The download is a zip holding a zip holding
    one to four XML versions of the same report. A real XML parse costs seconds and
    hundreds of MB for a few dozen records out of ~150k, so the <PricRpt> blocks around
    each matching <TckrSymb> are sliced out with byte searches instead. The download is
    the expensive part — which is why every product shares ONE pass over each file.

  * THE EMPTY 22-BYTE STUB. Holidays, weekends, not-yet-published days and dates before
    the archive starts all answer HTTP 200 with a 22-byte non-zip body — status codes lie
    here as they do on the BCB API. A stub is recorded as "no session" only when the date
    is safely in the past; today's not-yet-published file must be retried, never frozen
    as a holiday.

  * THE LIVE BDI TABLE IS NOT A SOURCE. arquivos.b3.com.br/bdi serves the same data as
    JSON, but keeps only ~21 days, caps pages at 1,000 rows, and its paging is unstable —
    on 2026-10-09 the same day served duplicated rows on one call and no ICF rows at all
    on another. Do not "simplify" to it.

UNITS. Every B3 leg is converted to its US pair's own quote unit (BRL legs at same-day
PTAX, BCB SGS 1) so the two sit on one axis:

    ICF  USD/bag  -> c/lb      ÷ 132.27736 lb/bag × 100
    CCM  BRL/sack -> c/bu      ÷ PTAX ÷ 2.362096 bu/sack (60 kg ÷ 25.40117 kg) × 100
    BGI  BRL/@    -> USD/cwt   ÷ PTAX ÷ 15 kg × 45.359237 kg/cwt   (CARCASS weight —
                               CME is LIVE weight, so BGI-vs-LC is a ratio, never a
                               difference; ~52-55% dressing sits between them)
    ETH  BRL/m3   -> USD/m3    ÷ PTAX   (sugar parity needs a mill-economics model on
                               top; stored raw-in-USD until that is built)

The basis uses the US leg's RAW front settle — never the panama-adjusted series, which
would difference an adjusted level against a raw one (see the spread-normalisation rule).
Coffee is matched contract month for contract month (both list H K N U Z); the others list
different month cycles, so they compare B3's most-active contract (max OI) to the US front.

CLI:  python src/b3ag.py [--backfill YYYY-MM-DD] [--days N]
"""
from __future__ import annotations

import functools
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
# Under data/brazil/, NOT the gitignored data/signals/: the laptop's post-pull auto-push
# carries it to basisterminal.com, whose sync only runs run_daily.run() (no heavy stores).
STORE = _ROOT / "data" / "brazil" / "b3_ag"
SETTLES = STORE / "settles.parquet"
META = STORE / "meta.json"

_PR_URL = "https://www.b3.com.br/pesquisapregao/download?filelist=PR{yymmdd}.zip"
_UA = {"User-Agent": "Mozilla/5.0"}

ARCHIVE_START = date(2018, 1, 2)
PTAX_SGS = 1                          # BCB SGS 1 = USD/BRL PTAX (sell)

LB_PER_BAG = 60 / 0.45359237          # 132.27736 lb in a 60 kg coffee bag
BU_PER_SACK = 60 / 25.40117272        # 2.362096 bu of corn in a 60 kg sack
KG_PER_CWT = 45.359237                # 100 lb
KG_PER_ARROBA = 15.0

PRODUCTS = {
    "ICF": dict(name="Arabica coffee", quote="USD/bag", ccy="USD",
                us="KCA Comdty", us_root="KC", us_unit="c/lb", match="month",
                to_us=lambda px, ptax: px / LB_PER_BAG * 100),
    "CCM": dict(name="Corn", quote="BRL/sack", ccy="BRL",
                us="C A Comdty", us_root="C", us_unit="c/bu", match="active",
                to_us=lambda px, ptax: px / ptax / BU_PER_SACK * 100),
    "BGI": dict(name="Live cattle", quote="BRL/@", ccy="BRL",
                us="LCA Comdty", us_root="LC", us_unit="USD/cwt", match="active",
                ratio_only=True,          # carcass vs live weight: compare as a ratio
                to_us=lambda px, ptax: px / ptax / KG_PER_ARROBA * KG_PER_CWT),
    "ETH": dict(name="Hydrous ethanol", quote="BRL/m3", ccy="BRL",
                us="SBA Comdty", us_root="SB", us_unit="USD/m3", match="active",
                parity_pending=True,      # USD/m3 vs c/lb: no basis/ratio until the
                                          # sugar-ethanol parity model exists
                to_us=lambda px, ptax: px / ptax),
}

# Single-contract tickers are exactly ROOT + month letter + 2-digit year. Longer symbols
# are options (ICFZ25C0450…) and calendar spreads; the fixed shape keeps them out.
_TICKER = re.compile(rb"<TckrSymb>((?:" + "|".join(PRODUCTS).encode()
                     + rb")[FGHJKMNQUVXZ]\d\d)</TckrSymb>")
_MONTHS = "FGHJKMNQUVXZ"

# XML field -> store column. Prices are in each product's B3 quote; volumes are contracts.
_FIELDS = {
    "AdjstdQt": "settle", "PrvsAdjstdQt": "prev_settle", "LastPric": "last",
    "FrstPric": "open", "MinPric": "low", "MaxPric": "high",
    "BestBidPric": "bid", "BestAskPric": "ask",
    "FinInstrmQty": "volume", "OpnIntrst": "oi",
}


# ── fetch + parse ────────────────────────────────────────────────────────────

def _download(d: date, timeout: int = 90) -> bytes:
    url = _PR_URL.format(yymmdd=d.strftime("%y%m%d"))
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:            # B3 drops the odd connection under load
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"B3 price report {d}: {last}")


def _xml_payload(blob: bytes) -> bytes | None:
    """Unwrap zip-in-zip down to ONE XML version (the latest-named). None for a stub."""
    if len(blob) < 1000 or not zipfile.is_zipfile(io.BytesIO(blob)):
        return None
    z = zipfile.ZipFile(io.BytesIO(blob))
    for _ in range(3):                    # outer zip -> inner zip -> xml
        names = sorted(z.namelist())
        xmls = [n for n in names if n.lower().endswith(".xml")]
        if xmls:
            return z.read(xmls[-1])
        zips = [n for n in names if n.lower().endswith(".zip")]
        if not zips:
            return None
        z = zipfile.ZipFile(io.BytesIO(z.read(zips[-1])))
    return None


def _num(block: bytes, tag: str) -> float | None:
    m = re.search(rb"<" + tag.encode() + rb"(?:\s[^>]*)?>([^<]+)</", block)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_report(xml: bytes, d: date) -> list[dict]:
    rows = []
    for m in _TICKER.finditer(xml):
        lo = xml.rfind(b"<PricRpt>", 0, m.start())
        hi = xml.find(b"</PricRpt>", m.end())
        if lo < 0 or hi < 0:
            continue
        block = xml[lo:hi]
        # The report ALSO carries the next session's opening records (TradDt = d+1,
        # yesterday's settle, a handful of early trades). Keeping those — they sit later
        # in the file, so "last occurrence wins" picks them — showed ICFZ26 trading 8
        # lots on a 480-lot day. Only the block stamped with the report's own date counts.
        dt = re.search(rb"<TradDt>\s*<Dt>([\d-]+)</Dt>", block)
        if not dt or dt.group(1).decode() != d.isoformat():
            continue
        tkr = m.group(1).decode()
        row = {"date": pd.Timestamp(d), "root": tkr[:3], "ticker": tkr}
        row.update({col: _num(block, tag) for tag, col in _FIELDS.items()})
        if row["settle"] is None:
            continue
        rows.append(row)
    return list({r["ticker"]: r for r in rows}.values())


def fetch_day(d: date) -> list[dict] | None:
    """B3 ag rows for one session; [] = confirmed no session; None = not published yet."""
    xml = _xml_payload(_download(d))
    if xml is None:
        # A stub on a recent date is "not out yet", not a holiday.
        return [] if (date.today() - d).days >= 3 else None
    return parse_report(xml, d)


# ── store ────────────────────────────────────────────────────────────────────

def _meta() -> dict:
    try:
        return json.loads(META.read_text())
    except Exception:
        return {"done": []}


def _save(new_rows: list[dict], done: set[str]) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    if new_rows:
        new = pd.DataFrame(new_rows)
        old = pd.read_parquet(SETTLES) if SETTLES.exists() else pd.DataFrame()
        df = (pd.concat([old, new], ignore_index=True)
              .drop_duplicates(["date", "ticker"], keep="last")
              .sort_values(["date", "ticker"]).reset_index(drop=True))
        tmp = SETTLES.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(SETTLES)
    m = _meta()
    m["done"] = sorted(set(m.get("done", [])) | done)
    m["roots"] = sorted(PRODUCTS)
    m["updated"] = datetime.now().isoformat(timespec="seconds")
    META.write_text(json.dumps(m, indent=1))


def update(start: date | None = None, end: date | None = None,
           verbose: bool = False, flush_every: int = 20, workers: int = 3) -> int:
    """Fetch every weekday in [start, end] not already done. Resumable; never raises
    on a single bad day. Default window = the last 10 calendar days (daily pull).
    Downloads run `workers` at a time (each day is ~7 MB; three keeps B3 happy).

    A day is "done" for the root set it was fetched with; adding a root to PRODUCTS
    means re-fetching history (clear meta.json's "done") — there is no per-root resume."""
    from concurrent.futures import ThreadPoolExecutor
    end = end or date.today()
    start = start or end - timedelta(days=10)
    m = _meta()
    done = set(m.get("done", [])) if m.get("roots", sorted(PRODUCTS)) == sorted(PRODUCTS) else set()
    todo = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in todo if d.weekday() < 5 and d.isoformat() not in done]

    def _one(d):
        try:
            return d, fetch_day(d), None
        except Exception as e:
            return d, None, e

    rows, newly, n = [], set(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, got, err in ex.map(_one, todo):
            key = d.isoformat()
            if err is not None:
                print(f"  (B3 {key} skipped: {err})", flush=True)
            elif got is not None:
                rows += got
                newly.add(key)
                n += len(got)
            if verbose and err is None:
                print(f"  {key}: {'pending' if got is None else len(got)}", flush=True)
            if len(newly) >= flush_every:
                _save(rows, newly)
                rows, newly = [], set()
    _save(rows, newly)
    _read_cached.cache_clear()
    return n


# ── readers ──────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=2)
def _read_cached(_mtime: float) -> pd.DataFrame:
    return pd.read_parquet(SETTLES)


def _read() -> pd.DataFrame:
    if not SETTLES.exists():
        return pd.DataFrame(columns=["date", "root", "ticker", *_FIELDS.values()])
    return _read_cached(SETTLES.stat().st_mtime).copy()


def _b3_contract_ym(ticker: str) -> tuple[int, int]:
    """(year, month) of a B3 contract from its ticker (always a 2-digit year)."""
    return (2000 + int(ticker[4:6]), _MONTHS.index(ticker[3]) + 1)


def ptax(start: str = "2017-12-01") -> pd.DataFrame:
    from . import macrodata
    s = macrodata.sgs(PTAX_SGS, title="USD/BRL PTAX", start=start)
    if not s.ok:
        return pd.DataFrame(columns=["date", "ptax"])
    return pd.DataFrame({"date": pd.to_datetime([d for d, _ in s.obs]),
                         "ptax": [v for _, v in s.obs]})


def settles(root: str | None = None, with_ptax: bool = True) -> pd.DataFrame:
    """Stored B3 contract-days, with the price converted to the US pair's unit
    (`settle_us`). BRL legs with no PTAX for the day get NaN, never a stale rate."""
    df = _read()
    if root:
        df = df[df["root"] == root]
    if df.empty:
        return df
    df = df.copy()
    df["contract_ym"] = [_b3_contract_ym(t) for t in df["ticker"]]
    if with_ptax:
        df = df.merge(ptax(), on="date", how="left")
        df["settle_us"] = [PRODUCTS[r]["to_us"](px, fx)
                           for r, px, fx in zip(df["root"], df["settle"], df["ptax"])]
    return df


def _us_contract_ym(code: str, root: str, asof: pd.Timestamp) -> tuple[int, int] | None:
    """Bloomberg generic-ticker strings: 'KCZ16' (old) and 'KCZ6' (new) both occur, and
    one-letter roots are space-padded ('C Z6')."""
    m = re.match(r"^" + re.escape(root) + r"\s*([FGHJKMNQUVXZ])(\d{1,2})$", str(code).strip())
    if not m:
        return None
    mon = _MONTHS.index(m.group(1)) + 1
    y = m.group(2)
    if len(y) == 2:
        return (2000 + int(y), mon)
    # One digit: the first year >= asof's year - 1 ending in that digit.
    base = asof.year - 1
    return (base + ((int(y) - base) % 10), mon)


def basis(root: str) -> pd.DataFrame:
    """Daily B3 leg vs its US pair, both in the US quote unit, raw settles.

    `basis` = B3 - US (and `basis_pct` of US). For cattle (carcass vs live weight) only
    `ratio` = B3 / US is meaningful and `basis` is left NaN on purpose; for ethanol vs
    sugar neither is (different units) until the parity model is built — both NaN."""
    from . import deepstore
    p = PRODUCTS[root]
    b3 = settles(root)
    if b3.empty or not p.get("us"):
        return pd.DataFrame()
    px = deepstore.get_raw([p["us"]])
    ct = deepstore.get_contracts([p["us"]])
    if px.empty:
        return pd.DataFrame()
    us = pd.DataFrame({"us_px": px[p["us"]]})
    us["us_contract"] = ct[p["us"]] if not ct.empty else None
    us.index = pd.to_datetime(us.index)
    us = us.dropna(subset=["us_px"]).rename_axis("date").reset_index()

    if p["match"] == "month":
        us["contract_ym"] = [_us_contract_ym(c, p["us_root"], d)
                             for c, d in zip(us["us_contract"], us["date"])]
        out = us.merge(b3, on=["date", "contract_ym"], how="inner")
    else:
        # Most-active B3 contract per day (max OI; volume breaks ties on roll days).
        act = (b3.sort_values(["date", "oi", "volume"], na_position="first")
                 .groupby("date").tail(1))
        out = us.merge(act, on="date", how="inner")
    out = out.rename(columns={"ticker": "b3_contract", "settle": "b3_native",
                              "settle_us": "b3_px", "oi": "b3_oi", "volume": "b3_volume"})
    out["ratio"] = out["b3_px"] / out["us_px"]
    if p.get("parity_pending"):
        out["ratio"] = float("nan")
        out["basis"] = float("nan")
        out["basis_pct"] = float("nan")
    elif p.get("ratio_only"):
        out["basis"] = float("nan")
        out["basis_pct"] = float("nan")
    else:
        out["basis"] = out["b3_px"] - out["us_px"]
        out["basis_pct"] = out["basis"] / out["us_px"] * 100
    keep = ["date", "b3_contract", "b3_native", "b3_px", "us_contract", "us_px",
            "basis", "basis_pct", "ratio", "b3_oi", "b3_volume", "ptax"]
    return out[keep].sort_values("date").reset_index(drop=True)


def coverage() -> dict:
    df = _read()
    if df.empty:
        return {"rows": 0}
    out = {"rows": len(df), "first": str(df["date"].min().date()),
           "last": str(df["date"].max().date()), "sessions": int(df["date"].nunique())}
    out["by_root"] = {r: int(n) for r, n in df.groupby("root")["date"].nunique().items()}
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--backfill" in args:
        s = datetime.strptime(args[args.index("--backfill") + 1], "%Y-%m-%d").date()
        print(f"B3 ag backfill ({', '.join(PRODUCTS)}) from {s}…", flush=True)
        n = update(start=s, verbose=True)
    else:
        days = int(args[args.index("--days") + 1]) if "--days" in args else 10
        n = update(start=date.today() - timedelta(days=days), verbose=True)
    print(f"B3 ag: {n} new contract-days; store {coverage()}")
