"""pm_fetch.py — real-Chrome fetcher for the bot-protected precious-metals sources.

CME depository stocks (Akamai-blocked .xls, same URL overwritten daily) and the
US Mint bullion-sales page (Cloudflare-challenged). Same Playwright persistent-
profile pattern as opec_fetch: a real Chrome window with a durable profile sails
through both challenges; plain HTTP clients 403.

- CME files land in data/pm_inbox/ stamped by date; each run APPENDS the day's
  registered/eligible totals to data/signals/pm_comex.parquet (CME keeps no
  public history, so the archive builds from today onward).
- US Mint monthly gold/silver bullion ounces (current + prior 2 years) overwrite
  data/signals/pm_mint.parquet (the page carries full-year history, so a fresh
  pull rebuilds the whole series).

CLI:  python src/pm_fetch.py [--comex-only|--mint-only]
"""
from __future__ import annotations

import io
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "pm_inbox"
SIGNALS = ROOT / "data" / "signals"
PROFILE = ROOT / "data" / ".pm_chrome_profile"
COMEX_FILE = SIGNALS / "pm_comex.parquet"
MINT_FILE = SIGNALS / "pm_mint.parquet"

CME_FILES = {  # metal(s) -> daily stocks file
    "gold": "https://www.cmegroup.com/delivery_reports/Gold_Stocks.xls",
    "silver": "https://www.cmegroup.com/delivery_reports/Silver_stocks.xls",
    "pgm": "https://www.cmegroup.com/delivery_reports/PA-PL_Stck_Rprt.xls",
}
MINT_LANDING = "https://www.usmint.gov/about/production-sales-figures/bullion-sales"
MINT_CSVS = {  # tidy CSVs on the same origin; fetched from inside the page so
    # the browser's Cloudflare pass applies (plain HTTP clients get challenged)
    "gold": "https://www.usmint.gov/content/dam/usmint/data/tidy/bullion-american-eagle-gold.csv",
    "silver": "https://www.usmint.gov/content/dam/usmint/data/tidy/bullion-american-eagle-silver.csv",
}
_FETCH_JS = """async (url) => {
  const r = await fetch(url, {credentials: 'include'});
  if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
  return await r.text();
}"""


def _launch(p):
    INBOX.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        str(PROFILE), channel="chrome", headless=False, accept_downloads=True,
        args=["--disable-blink-features=AutomationControlled"])


def _download(page, url: str, dest: Path, timeout=60000) -> Path:
    """Navigate straight at a file URL and capture the download (goto aborts
    with net::ERR_ABORTED once Chrome hands the response to the downloader —
    that is the success path, not an error)."""
    with page.expect_download(timeout=timeout) as di:
        try:
            page.goto(url, timeout=timeout)
        except Exception:
            pass
    di.value.save_as(str(dest))
    return dest


# ---------------------------------------------------------------------------
# CME depository stocks
# ---------------------------------------------------------------------------
def _sheet_totals(xls_path: Path) -> list[dict]:
    """Pull the per-metal grand totals out of a CME stocks workbook.

    Verified layout (all three files, Jul 2026): column 0 carries a metal
    section header (GOLD / SILVER / PLATINUM / PALLADIUM), then depository
    blocks, then grand-total rows 'TOTAL REGISTERED' / 'TOTAL ELIGIBLE' whose
    value sits in the 'TOTAL TODAY' column (index 7). Gold adds a separate
    'TOTAL PLEDGED' bucket which we record but keep out of registered. The
    'Activity Date: m/d/yyyy' stamp (column 6) dates the data."""
    metals = {"GOLD": "gold", "SILVER": "silver",
              "PLATINUM": "platinum", "PALLADIUM": "palladium"}
    out, cur, activity = [], None, None
    df = pd.read_excel(xls_path, header=None, engine="xlrd")
    for i in range(len(df)):
        c0 = str(df.iat[i, 0]).strip() if pd.notna(df.iat[i, 0]) else ""
        c6 = str(df.iat[i, 6]).strip() if df.shape[1] > 6 and pd.notna(df.iat[i, 6]) else ""
        m = re.search(r"Activity Date:\s*(\d{1,2}/\d{1,2}/\d{4})", c6)
        if m:
            activity = pd.Timestamp(m.group(1)).date().isoformat()
        if c0.upper() in metals:
            cur = {"metal": metals[c0.upper()], "activity": activity,
                   "registered": None, "eligible": None, "pledged": None}
            out.append(cur)
            continue
        if cur is None:
            continue
        key = {"TOTAL REGISTERED": "registered", "TOTAL ELIGIBLE": "eligible",
               "TOTAL PLEDGED": "pledged"}.get(c0.upper())
        if key:
            try:
                cur[key] = float(str(df.iat[i, 7]).replace(",", ""))
            except (TypeError, ValueError):
                pass
            # a metal's Activity Date row appears just after its header
            cur["activity"] = cur["activity"] or activity
    return [r for r in out if r["registered"] is not None or r["eligible"] is not None]


def fetch_comex(ctx) -> pd.DataFrame:
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    # warm the Akamai cookie set on a real page first
    page.goto("https://www.cmegroup.com/", wait_until="domcontentloaded", timeout=60000)
    time.sleep(4)
    stamp = date.today().isoformat()
    rows = []
    for key, url in CME_FILES.items():
        dest = INBOX / f"{stamp}_{Path(url).name}"
        _download(page, url, dest)
        for tot in _sheet_totals(dest):
            rows.append({"date": tot["activity"] or stamp, "metal": tot["metal"],
                         "registered_oz": tot["registered"], "eligible_oz": tot["eligible"],
                         "pledged_oz": tot["pledged"]})
    got = pd.DataFrame(rows)
    if got.empty:
        raise RuntimeError("CME stocks parse produced no totals — check the inbox files")
    if COMEX_FILE.exists():  # archive keyed by activity date: replace same-day rows
        old = pd.read_parquet(COMEX_FILE)
        got = pd.concat([old[~old["date"].isin(got["date"].unique())], got],
                        ignore_index=True)
    got = got.sort_values(["metal", "date"]).reset_index(drop=True)
    got.to_parquet(COMEX_FILE, index=False)
    return got


# ---------------------------------------------------------------------------
# US Mint bullion sales
# ---------------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}


def _parse_mint_csv(text: str, metal: str) -> list[dict]:
    """Tidy-CSV layout (verified Jul 2026): repeated 5-column groups
    (Product, Weight_oz, Year, Month, Units) separated by a blank column —
    gold has one group per coin weight (1 / 0.5 / 0.25 / 0.1 oz), silver just
    the 1 oz group. Units are COINS, so ounces = Units × Weight_oz. Future
    months of the current year are present but blank."""
    df = pd.read_csv(io.StringIO(text), header=0)
    rows = []
    for off in range(0, df.shape[1], 6):
        grp = df.iloc[:, off:off + 5]
        if grp.shape[1] < 5:
            continue
        grp.columns = ["product", "weight", "year", "month", "units"]
        for _, r in grp.iterrows():
            mname = str(r["month"]).strip()
            units = pd.to_numeric(str(r["units"]).replace(",", ""), errors="coerce")
            w = pd.to_numeric(r["weight"], errors="coerce")
            yr = pd.to_numeric(r["year"], errors="coerce")
            if mname not in _MONTHS or pd.isna(units) or pd.isna(w) or pd.isna(yr):
                continue
            rows.append({"month": f"{int(yr)}-{_MONTHS[mname]:02d}", "metal": metal,
                         "oz": float(units) * float(w)})
    return rows


def fetch_mint(ctx, years: int = 3) -> pd.DataFrame:
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(MINT_LANDING, wait_until="domcontentloaded", timeout=90000)
    time.sleep(6)  # let any Cloudflare interstitial clear (profile keeps the pass)
    rows = []
    for metal, url in MINT_CSVS.items():
        rows.extend(_parse_mint_csv(page.evaluate(_FETCH_JS, url), metal))
    got = pd.DataFrame(rows)
    if not got.empty:
        cutoff = f"{date.today().year - years + 1}-01"
        got = got[got["month"] >= cutoff]
    if got.empty:
        raise RuntimeError("US Mint parse produced no rows — check page layout / challenge")
    got = (got.groupby(["month", "metal"], as_index=False)["oz"].sum()
              .sort_values(["metal", "month"]).reset_index(drop=True))
    got.to_parquet(MINT_FILE, index=False)
    return got


# ---------------------------------------------------------------------------
# Swiss customs gold exports (no browser needed — open CSV on BAZG's CDN)
# ---------------------------------------------------------------------------
SWISS_FILE = SIGNALS / "pm_swiss.parquet"
SWISS_URL = "https://ocean.nivel.bazg.admin.ch/open-data-reports/TN8_EXP_en/TN8_EXP_en.zip"
SWISS_TN = "7108.1200"  # unwrought gold bars


def fetch_swiss() -> pd.DataFrame:
    """Monthly Swiss gold-bar exports by destination from BAZG's open-data file
    (the full tariff×country export file, ~684 MB zip refreshed monthly with the
    whole revised history — so each pull REBUILDS the archive). Stream-filters
    the ~34M-row CSV down to the 7108 lines without holding it in memory."""
    import tempfile
    import urllib.request
    import zipfile

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "TN8_EXP_en.zip"
        print(f"pm_fetch: downloading Swiss trade file (~680 MB)…")
        urllib.request.urlretrieve(SWISS_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            f = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")
            header = f.readline()
            kept = [header] + [ln for ln in f if ";7108." in ln]  # coarse; exact filter below
    df = pd.read_csv(io.StringIO("".join(kept)), sep=";", escapechar="\\",
                     dtype={"Tariffnumber8": str}, engine="python")
    df = df[df["Tariffnumber8"] == SWISS_TN]
    out = (df.groupby(["year", "month", "Country_isoAlpha2"], as_index=False)["Quantity_kg"]
             .sum().rename(columns={"Country_isoAlpha2": "iso2", "Quantity_kg": "kg"}))
    if out.empty:
        raise RuntimeError("Swiss export filter produced no rows")
    out.to_parquet(SWISS_FILE, index=False)
    return out


def main() -> int:
    from playwright.sync_api import sync_playwright
    only = next((a for a in sys.argv[1:] if a.startswith("--") and a.endswith("-only")), None)
    do = lambda name: only is None or only == f"--{name}-only"  # noqa: E731
    if do("comex") or do("mint"):
        with sync_playwright() as p:
            ctx = _launch(p)
            try:
                if do("comex"):
                    c = fetch_comex(ctx)
                    print(f"pm_fetch: COMEX ok — {len(c)} rows, latest:\n"
                          f"{c[c['date'] == c['date'].max()].to_string(index=False)}")
                if do("mint"):
                    m = fetch_mint(ctx)
                    print(f"pm_fetch: Mint ok — {len(m)} rows "
                          f"({m['month'].min()} → {m['month'].max()})")
            finally:
                ctx.close()
    if do("swiss"):
        s = fetch_swiss()
        print(f"pm_fetch: Swiss ok — {len(s)} tariff-line rows "
              f"({s['year'].min()} → {s['year'].max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
