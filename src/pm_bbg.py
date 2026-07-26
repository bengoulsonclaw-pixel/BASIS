"""pm_bbg.py — Bloomberg pulls for the 🥇 Precious Metals monitor (terminal needed).

Writes the same parquet caches pmdata reads, so the monitor keeps working
offline from the last pull on non-terminal days. Hit cost is tiny: a handful
of aggregate Index tickers, PX_LAST only, pulled at most once per freshness
window — nowhere near the daily DAPI limit.

  pm_etf.parquet    total known ETF holdings (troy oz) per metal
  pm_fred.parquet   real_yield (USGGT10Y) + dollar (BBDXY) — same file/shape
                    pmdata.fetch_fred writes, so either source can feed it

Pt/Pd ETF aggregates + the SGE/Shanghai gold leg still need their exact
Bloomberg tickers (SECF) — add them to TICKERS_ETF / a premium pull below.

CLI:  python src/pm_bbg.py [--force]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SIGNALS = ROOT / "data" / "signals"
ETF_FILE = SIGNALS / "pm_etf.parquet"
FRED_FILE = SIGNALS / "pm_fred.parquet"   # shared with pmdata.fetch_fred

TICKERS_ETF = {  # verified live 16 Jul 2026; values come back in troy oz
    "gold": "ETFGTOTL Index",       # Bloomberg total known ETF holdings
    "silver": "ETSITOTL Index",     # Bloomberg total known ETF holdings
    # Pt/Pd: no all-issuer aggregate found — these are ETF Securities Ltd
    # (WisdomTree) physical holdings, the biggest Pt/Pd ETP family; a flow
    # proxy, labelled as such in the report footnote.
    "platinum": "ETFHPLAT Index",
    "palladium": "ETFHPALL Index",
}
TICKERS_RATES = {"real_yield": "USGGT10Y Index", "dollar": "BBDXY Index"}
# Shanghai premium: Ben's CIX converting the Shanghai gold price to USD/oz,
# minus London spot. CIX securities resolve under this terminal's login only.
TICKERS_PREM = {"sge_usd": ".SHGOLDOZ G Index", "xau": "XAU Curncy"}
PREM_FILE = SIGNALS / "pm_prem.parquet"


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _wide(tickers: list[str], years: int) -> pd.DataFrame:
    from xbbg import blp
    from src.datafeed import _bdh_to_wide
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    w = _bdh_to_wide(blp.bdh(tickers=tickers, flds="PX_LAST",
                             start_date=start.date(), end_date=end.date()))
    if w is None or w.empty:
        raise RuntimeError(f"empty bdh for {tickers}")
    return w


def pull_etf(years: int = 4) -> pd.DataFrame:
    w = _wide(list(TICKERS_ETF.values()), years)
    rows = []
    for metal, tkr in TICKERS_ETF.items():
        if tkr not in w.columns:
            continue
        s = w[tkr].dropna()
        rows.append(pd.DataFrame({"date": s.index, "metal": metal, "oz": s.values}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ETF_FILE, index=False)
    return out


def pull_rates(years: int = 6) -> pd.DataFrame:
    w = _wide(list(TICKERS_RATES.values()), years)
    df = pd.DataFrame({name: w[tkr] for name, tkr in TICKERS_RATES.items()
                       if tkr in w.columns}).dropna(how="all")
    df.to_parquet(FRED_FILE)
    return df


def pull_premium(years: int = 4) -> pd.DataFrame:
    w = _wide(list(TICKERS_PREM.values()), years)
    df = pd.DataFrame({name: w[tkr] for name, tkr in TICKERS_PREM.items()
                       if tkr in w.columns}).dropna()
    df["premium"] = df["sge_usd"] - df["xau"]
    df.to_parquet(PREM_FILE)
    return df


def pull(force: bool = False, max_age_hours: float = 20.0) -> dict:
    out = {}
    if force or not _is_fresh(ETF_FILE, max_age_hours):
        e = pull_etf()
        out["etf"] = f"{len(e)} rows to {e['date'].max().date()}"
    if force or not _is_fresh(FRED_FILE, max_age_hours):
        r = pull_rates()
        out["rates"] = f"{len(r)} rows to {r.index.max().date()}"
    if force or not _is_fresh(PREM_FILE, max_age_hours):
        try:  # CIX only resolves under Ben's login — never fail the whole pull
            p = pull_premium()
            out["premium"] = f"{len(p)} rows to {p.index.max().date()}"
        except Exception as e:
            out["premium"] = f"skipped ({e})"
    return out


if __name__ == "__main__":
    res = pull(force="--force" in sys.argv)
    print("pm_bbg:", res or "all caches fresh — nothing pulled")
