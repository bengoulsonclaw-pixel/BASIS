"""The instrument universe, grouped by asset class and region.

Keys are the full Bloomberg generic front-month tickers (including the yellow
key: Comdty / Index / Curncy) so look-alikes stay distinct, e.g.
`SMA Index` (SMI) vs `SMA Comdty` (Soybean Meal).

Each value is a tuple:  (human name, mock base price, asset class, region)
The mock base price is used ONLY by the synthetic data feed (demo mode);
it is ignored once live Bloomberg data is wired in.

Names marked "— confirm" are best guesses — correct them here when you can.
"""
from __future__ import annotations

import json
from pathlib import Path

# The product universe is editable from the app and persisted to data/universe.json.
# The _SEED_* tables below are the built-in book used to seed that store on first
# run; afterwards the JSON is the source of truth. The live INSTRUMENTS / override
# maps / TREND_UNIVERSE are filled from the store at import (loader near the bottom)
# and refreshed IN PLACE by reload(), so modules that did `from .universe import
# INSTRUMENTS` see edits without re-importing.
_SEED_INSTRUMENTS = {
    # ── INDICES ────────────────────────────────────────────────────────────
    # NA
    "ESA Index":   ("S&P 500 E-mini",       6000.0,  "Indices", "NA"),
    "NQA Index":   ("Nasdaq 100 E-mini",   21000.0,  "Indices", "NA"),
    "RTYA Index":  ("Russell 2000 E-mini",  2300.0,  "Indices", "NA"),
    "DMA Index":   ("Dow E-mini",          43000.0,  "Indices", "NA"),
    # EMEA
    "VGA Index":   ("Euro Stoxx 50",        5100.0,  "Indices", "EMEA"),
    "CAA Index":   ("Euro Stoxx Banks",      320.0,  "Indices", "EMEA"),   # Banks-sector future (Eurex) — NOT the CAC 40 (Ben confirmed 2026-08-08)
    "CFA Index":   ("CAC 40",               8700.0,  "Indices", "EMEA"),   # Euronext CAC40 10-euro future (verified live: €10/pt)
    "Z A Index":   ("FTSE 100",             8400.0,  "Indices", "EMEA"),
    "GXA Index":   ("DAX",                 18500.0,  "Indices", "EMEA"),
    "SMA Index":   ("SMI (Swiss)",         12000.0,  "Indices", "EMEA"),
    # APAC
    "NKA Index":   ("Nikkei 225",          39000.0,  "Indices", "APAC"),
    "KMA Index":   ("KOSPI 200",   360.0,  "Indices", "APAC"),
    "XPA Index":   ("ASX SPI 200",          7900.0,  "Indices", "APAC"),
    # EMEA CASH indices — the option surface on the cash index carries the
    # constant-maturity vol that the FUTURES (VGA/GXA/Z A) don't, so these are the
    # vol-report sources (added 2026-06-04). Priced via PX_LAST (cash has no
    # settle) — see PRICE_FIELD_OVERRIDE. They are vol-report only (kept out of the
    # trend/mean-reversion universe, which keeps using the futures).
    "SX5E Index":  ("Euro Stoxx 50",        6100.0,  "Indices", "EMEA"),
    "SX7E Index":  ("Euro Stoxx Banks",      270.0,  "Indices", "EMEA"),
    "DAX Index":   ("DAX",                 24900.0,  "Indices", "EMEA"),
    "UKX Index":   ("FTSE 100",            10360.0,  "Indices", "EMEA"),
    # CAC/SMI + APAC twins added 2026-07-21 — surfaces verified LIVE on the Terminal.
    # NB KOSPI2 (the optionable KOSPI 200 the KMA futures track), NOT the KOSPI Composite.
    "CAC Index":   ("CAC 40",               7800.0,  "Indices", "EMEA"),
    "SMI Index":   ("SMI (Swiss)",         12000.0,  "Indices", "EMEA"),
    "NKY Index":   ("Nikkei 225",          42000.0,  "Indices", "APAC"),
    "KOSPI2 Index": ("KOSPI 200",            420.0,  "Indices", "APAC"),

    # ── FIXED INCOME · STIRs ───────────────────────────────────────────────
    # NA
    "SERA Comdty": ("1M SOFR",                95.7,  "STIRs", "NA"),
    "SFRA Comdty": ("3M SOFR",                96.0,  "STIRs", "NA"),
    "FFA Comdty":  ("30-Day Fed Funds",       95.8,  "STIRs", "NA"),
    # EMEA
    "ERA Comdty":  ("3M Euribor",             97.8,  "STIRs", "EMEA"),
    "TKYA Comdty": ("3M ESTR",                98.0,  "STIRs", "EMEA"),
    "SFIA Comdty": ("3M SONIA (Short Sterling)", 96.0, "STIRs", "EMEA"),

    # ── FIXED INCOME · BONDS ───────────────────────────────────────────────
    # NA
    "USA Comdty":  ("US Long Bond",          118.0,  "Bonds", "NA"),
    "WNA Comdty":  ("Ultra US Bond",         125.0,  "Bonds", "NA"),
    "UXYA Comdty": ("Ultra 10Y Note",        118.0,  "Bonds", "NA"),
    "TYA Comdty":  ("US 10Y Note",           111.0,  "Bonds", "NA"),
    "FVA Comdty":  ("US 5Y Note",            108.0,  "Bonds", "NA"),
    "TUA Comdty":  ("US 2Y Note",            103.0,  "Bonds", "NA"),
    # EMEA
    "UBA Comdty":  ("Euro Buxl (30Y)",       135.0,  "Bonds", "EMEA"),
    "RXA Comdty":  ("Euro-Bund (10Y)",       130.0,  "Bonds", "EMEA"),
    "OEA Comdty":  ("Euro-Bobl (5Y)",        117.0,  "Bonds", "EMEA"),
    "DUA Comdty":  ("Euro-Schatz (2Y)",      106.0,  "Bonds", "EMEA"),
    "OATA Comdty": ("French OAT",            125.0,  "Bonds", "EMEA"),
    "G A Comdty":  ("Long Gilt",              92.0,  "Bonds", "EMEA"),

    # ── FX ─────────────────────────────────────────────────────────────────
    "ECA Curncy":  ("Euro (EUR/USD)",          1.10,  "FX", ""),
    "BPA Curncy":  ("British Pound (GBP)",     1.27,  "FX", ""),
    "BRA Curncy":  ("Brazilian Real (BRL)",    0.18,  "FX", ""),
    "CDA Curncy":  ("Canadian Dollar (CAD)",   0.73,  "FX", ""),
    "SFA Curncy":  ("Swiss Franc (CHF)",       1.12,  "FX", ""),
    "JYA Curncy":  ("Japanese Yen (JPY)",      0.0065, "FX", ""),
    "ADA Curncy":  ("Australian Dollar (AUD)", 0.66,  "FX", ""),
    "NVA Curncy":  ("New Zealand Dollar (NZD)", 0.60, "FX", ""),
    "PEA Curncy":  ("Mexican Peso (MXN)",      0.055, "FX", ""),
    "RAA Curncy":  ("South African Rand (ZAR)", 0.055, "FX", ""),
    "SIRA Curncy": ("Indian Rupee (INR)", 0.012, "FX", ""),
    "SEA Curncy":  ("Swedish Krona (SEK)",     0.095, "FX", ""),
    "HEA Curncy":  ("Hungarian Forint (HUF)",            1.00,  "FX", ""),
    "KOA Curncy":  ("Korean Won (KRW)", 0.00075, "FX", ""),
    "NOA Curncy":  ("Norwegian Krone (NOK)",   0.094, "FX", ""),
    "CCA Curncy":  ("Czech Koruna (CZK)",            0.14,  "FX", ""),
    "ISA Curncy":  ("Israeli Shekel (ILS)", 0.27, "FX", ""),
    "PPA Curncy":  ("Polish Zloty (PLN)",            1.00,  "FX", ""),

    # ── COMMODITIES · ENERGY ───────────────────────────────────────────────
    "COA Comdty":  ("Brent Crude",            76.0,  "Energy", ""),
    "CLA Comdty":  ("WTI Crude",              72.0,  "Energy", ""),
    "XBA Comdty":  ("RBOB Gasoline",           2.3,  "Energy", ""),
    "NGA Comdty":  ("Henry Hub Nat Gas",       3.3,  "Energy", ""),
    "FJSA Comdty": ("TTF Natural Gas",         100.0,  "Energy", ""),
    "HOA Comdty":  ("Heating Oil (ULSD)",      2.4,  "Energy", ""),
    "QSA Comdty":  ("Gasoil (ICE)",          700.0,  "Energy", ""),
    "MOA Comdty":  ("EU Carbon (EUA)",          100.0,  "Energy", ""),
    "CUAA Comdty": ("Chicago Ethanol",         100.0,  "Energy", ""),

    # ── COMMODITIES · METALS ───────────────────────────────────────────────
    "GCA Comdty":  ("Gold",                 2350.0,  "Metals", ""),
    "SIA Comdty":  ("Silver",                 28.0,  "Metals", ""),
    "PLA Comdty":  ("Platinum",              980.0,  "Metals", ""),
    "PAA Comdty":  ("Palladium",            1000.0,  "Metals", ""),
    "HGA Comdty":  ("Copper (COMEX)",          4.4,  "Metals", ""),
    "ALEA Comdty": ("Aluminium (COMEX)",    2500.0,  "Metals", ""),
    "SCOA Comdty": ("Iron Ore (IODEX)",         100.0,  "Metals", ""),

    # ── COMMODITIES · AGRICULTURE ──────────────────────────────────────────
    "C A Comdty":  ("Corn",                  440.0,  "Agriculture", ""),
    "S A Comdty":  ("Soybeans",             1180.0,  "Agriculture", ""),
    "W A Comdty":  ("Wheat (Chicago)",       580.0,  "Agriculture", ""),
    "KWA Comdty":  ("Wheat (KC)",            600.0,  "Agriculture", ""),
    "SMA Comdty":  ("Soybean Meal",          340.0,  "Agriculture", ""),
    "BOA Comdty":  ("Soybean Oil",            44.0,  "Agriculture", ""),
    "RRA Comdty":  ("Rough Rice",             14.0,  "Agriculture", ""),

    # ── COMMODITIES · SOFTS ────────────────────────────────────────────────
    "KCA Comdty":  ("Coffee (Arabica)",      230.0,  "Softs", ""),
    "DFA Comdty":  ("Robusta Coffee", 4000.0, "Softs", ""),
    "CCA Comdty":  ("Cocoa",                7000.0,  "Softs", ""),
    "SBA Comdty":  ("Sugar No.11",            19.0,  "Softs", ""),
    "CTA Comdty":  ("Cotton",                 70.0,  "Softs", ""),
    "JOA Comdty":  ("Orange Juice (FCOJ)",   250.0,  "Softs", ""),
    "LHA Comdty":  ("Lean Hogs",              90.0,  "Softs", ""),
    "LCA Comdty":  ("Live Cattle",           185.0,  "Softs", ""),
    "FCA Comdty":  ("Feeder Cattle",         250.0,  "Softs", ""),
}

# Pairs for the mean-reversion / rich-cheap monitor.
#   kind "ratio" -> spread = A / B
#   kind "diff"  -> spread = A - B   (simple differential; legs share units)
PAIRS = [
    {"name": "WTI – Brent",         "a": "CLA Comdty", "b": "COA Comdty",  "kind": "diff"},
    {"name": "Gold / Silver ratio",      "a": "GCA Comdty", "b": "SIA Comdty",  "kind": "ratio"},
    {"name": "Gold / Platinum ratio",    "a": "GCA Comdty", "b": "PLA Comdty",  "kind": "ratio"},
    {"name": "Platinum / Palladium",     "a": "PLA Comdty", "b": "PAA Comdty",  "kind": "ratio"},
    {"name": "Chicago – KC Wheat",  "a": "W A Comdty", "b": "KWA Comdty",  "kind": "diff"},
    {"name": "Corn – Wheat",        "a": "C A Comdty", "b": "W A Comdty",  "kind": "diff"},
    {"name": "Soybean Oil / Meal",       "a": "BOA Comdty", "b": "SMA Comdty",  "kind": "ratio"},
    {"name": "Bund – OAT",          "a": "RXA Comdty", "b": "OATA Comdty", "kind": "diff"},
    {"name": "US 10Y – Bund",       "a": "TYA Comdty", "b": "RXA Comdty",  "kind": "diff"},
    {"name": "Gold / Copper ratio",      "a": "GCA Comdty", "b": "HGA Comdty",  "kind": "ratio"},
]

# EMEA cash indices are priced via PX_LAST (cash has no PX_SETTLE); datafeed
# get_history pulls these tickers with their override field and splices them in.
_SEED_PRICE_FIELD = {
    "SX5E Index": "PX_LAST", "SX7E Index": "PX_LAST",
    "DAX Index": "PX_LAST", "UKX Index": "PX_LAST",
    "CAC Index": "PX_LAST", "SMI Index": "PX_LAST",
    "NKY Index": "PX_LAST", "KOSPI2 Index": "PX_LAST",
}

# Where to pull 1-month ATM implied vol when LIVE (Bloomberg mode). By default
# every future uses the listed-option surface on its OWN ticker
# (datafeed.IMPLIED_VOL_FIELD). The exceptions below pull from a different
# security:
#   • FX     -> the OTC pair vol (…V1M Curncy) is the clean, liquid reference;
#              vol is the same regardless of quote direction, so the USD-quoted
#              tickers map straight onto the inverted futures (JYA, SFA, CDA…).
#   • Index  -> optionally a published 1M vol index (VIX, V2X, VDAX-NEW…).
# Format:  futures_ticker -> (source_security, bloomberg_field)
# These cover the majors — confirm on the Terminal (FLDS<GO>) and extend the
# rest (EM FX crosses, etc.) as you verify each one.
_SEED_VOL_OVERRIDE = {
    # FX — 1M ATM from the OTC pair vol (…V1M Curncy). Vol is symmetric to quote
    # direction, so the USD-quoted tickers map onto the inverted futures.
    # EUR/GBP/JPY verified live 2026-06-03; the rest are validated on each run.
    "ECA Curncy":  ("EURUSDV1M Curncy", "PX_LAST"),   # EUR
    "BPA Curncy":  ("GBPUSDV1M Curncy", "PX_LAST"),   # GBP
    "JYA Curncy":  ("USDJPYV1M Curncy", "PX_LAST"),   # JPY
    "SFA Curncy":  ("USDCHFV1M Curncy", "PX_LAST"),   # CHF
    "CDA Curncy":  ("USDCADV1M Curncy", "PX_LAST"),   # CAD
    "ADA Curncy":  ("AUDUSDV1M Curncy", "PX_LAST"),   # AUD
    "NVA Curncy":  ("NZDUSDV1M Curncy", "PX_LAST"),   # NZD
    "BRA Curncy":  ("USDBRLV1M Curncy", "PX_LAST"),   # BRL
    "PEA Curncy":  ("USDMXNV1M Curncy", "PX_LAST"),   # MXN
    "RAA Curncy":  ("USDZARV1M Curncy", "PX_LAST"),   # ZAR
    "SEA Curncy":  ("USDSEKV1M Curncy", "PX_LAST"),   # SEK
    "NOA Curncy":  ("USDNOKV1M Curncy", "PX_LAST"),   # NOK
    "KOA Curncy":  ("USDKRWV1M Curncy", "PX_LAST"),   # KRW (NDF vol)
    "SIRA Curncy": ("USDINRV1M Curncy", "PX_LAST"),   # INR (NDF vol)
    "ISA Curncy":  ("USDILSV1M Curncy", "PX_LAST"),   # ILS
    # HEA=Hungarian Forint, CCA=Czech Koruna, PPA=Polish Zloty (identities confirmed
    # 2026-06-14). OTC 1M vol (USDHUFV1M / USDCZKV1M / USDPLNV1M Curncy) not yet
    # verified, so they stay on the default surface field (RV-only) until mapped.

    # Euribor — the smoothed *_DF surface is NOT published for it on any root
    # (verified live 2026-07-21), but the listed near-ATM option implieds are:
    # the call/put MID via the composite "A+B" field syntax (datafeed averages the
    # legs), quoted in the same implied-RATE convention as the other STIR surfaces
    # (~21 vol on a 2.8% rate ≈ 3.8bp/day), 14+ months of history. NB it rolls with
    # the listed front expiry (not constant-maturity) — disclosed under the report's
    # STIR cards. €STR (TKYA) returns nothing anywhere — no listed vol market.
    "ERA Comdty":  ("ERA Comdty", "HIST_CALL_IMP_VOL+HIST_PUT_IMP_VOL"),   # 3M Euribor

    # Non-US equity indices are IGNORED for now (excluded in volatility.py). To
    # bring them back: drop the non-US-index rule in volatility._excluded() and
    # re-enable these verified vol-index sources (US indices use the surface field):
    #   "VGA Index": ("V2X Index",  "PX_LAST"),   # Euro Stoxx 50 -> VSTOXX
    #   "GXA Index": ("V1X Index",  "PX_LAST"),   # DAX           -> VDAX-NEW
    #   "SMA Index": ("V3X Index",  "PX_LAST"),   # SMI           -> VSMI
    #   "NKA Index": ("VNKY Index", "PX_LAST"),   # Nikkei 225
}


# ── Fixed income → YIELDS (the technical strategies run on yields, not futures) ─
# Rates desks read YIELD charts, and the price↔yield inversion makes futures-price
# TA misleading (a bond-future "golden cross" is falling yields = a bond rally, the
# opposite of a bullish-rates read). So the price-based technical strategies convert
# fixed income to yields first:
#   • STIRs  -> implied rate = 100 − price (exact; no extra data — see is_stir()).
#   • Bonds  -> the benchmark generic yield each future tracks, pulled from Bloomberg
#     (map below). The future→yield relationship runs through the CTD + conversion
#     factor + roll, so we DON'T invert the price — we chart the benchmark yield.
# Tickers are best-standard generics; confirm/adjust on the Terminal. The US Long
# Bond / Ultra map to the 30Y generic as a pragmatic proxy (classic-bond CTD ~15–20Y).
BOND_YIELD_SOURCE = {
    "TUA Comdty":  "USGG2YR Index",    # US 2Y Note
    "FVA Comdty":  "USGG5YR Index",    # US 5Y Note
    "TYA Comdty":  "USGG10YR Index",   # US 10Y Note
    "UXYA Comdty": "USGG10YR Index",   # Ultra 10Y ≈ 10Y
    "USA Comdty":  "USGG30YR Index",   # Long Bond ≈ 30Y (confirm)
    "WNA Comdty":  "USGG30YR Index",   # Ultra Bond ≈ 30Y
    "DUA Comdty":  "GDBR2 Index",      # German 2Y (Schatz)
    "OEA Comdty":  "GDBR5 Index",      # German 5Y (Bobl)
    "RXA Comdty":  "GDBR10 Index",     # German 10Y (Bund)
    "UBA Comdty":  "GDBR30 Index",     # German 30Y (Buxl)
    "OATA Comdty": "GTFRF10Y Govt",    # French 10Y (OAT)
    "G A Comdty":  "GUKG10 Index",     # UK 10Y (Long Gilt)
}


def is_stir(tk: str) -> bool:
    return asset(tk) == "STIRs"


def is_bond(tk: str) -> bool:
    return asset(tk) == "Bonds"


def is_fixed_income(tk: str) -> bool:
    return asset(tk) in ("STIRs", "Bonds")


def yield_source(tk: str) -> str | None:
    """The Bloomberg benchmark-yield ticker whose series the technical strategies run
    on for this BOND future; None for non-bonds (STIRs use 100 − price instead)."""
    return BOND_YIELD_SOURCE.get(tk)


def yield_name(tk: str) -> str:
    """Display label for an FI instrument once it's expressed as a yield/rate for the
    technical strategies — e.g. 'US 10Y Note (yield)', '3M SOFR (rate)'."""
    n = name(tk)
    if is_stir(tk):
        return f"{n} (rate)"
    if is_bond(tk):
        return f"{n} (yield)"
    return n


# ── Editable store (data/universe.json) ─────────────────────────────────────
STORE = Path(__file__).resolve().parents[1] / "data" / "universe.json"

# Asset classes offered as a dropdown in the app's universe editor.
ASSET_CLASSES = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]

# One editor row per instrument; this is the on-disk schema too.
_FIELDS = ["ticker", "name", "price", "asset", "region",
           "vol_source", "vol_field", "price_field"]

# Live containers — filled from the store and mutated IN PLACE by _apply()/reload()/
# save(). Never rebound, so `from .universe import INSTRUMENTS` stays valid forever.
INSTRUMENTS: dict = {}
IMPLIED_VOL_OVERRIDE: dict = {}
PRICE_FIELD_OVERRIDE: dict = {}
TREND_UNIVERSE: list = []


def _seed_records() -> list:
    """Flatten the built-in _SEED_* tables into editor rows (the first-run seed)."""
    rows = []
    for tk, (nm, px, ac, rg) in _SEED_INSTRUMENTS.items():
        vs, vf = _SEED_VOL_OVERRIDE.get(tk, ("", ""))
        rows.append({"ticker": tk, "name": nm, "price": float(px), "asset": ac,
                     "region": rg, "vol_source": vs, "vol_field": vf,
                     "price_field": _SEED_PRICE_FIELD.get(tk, "")})
    return rows


def _write_store(rows: list) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps({"instruments": rows}, indent=2), encoding="utf-8")


def load_rows() -> list:
    """The current universe as editor rows. Seeds the store from the built-in book
    on first run; falls back to that book if the file is missing or unreadable."""
    if not STORE.exists():
        _write_store(_seed_records())
    try:
        rows = json.loads(STORE.read_text(encoding="utf-8")).get("instruments", [])
        return rows or _seed_records()
    except Exception:
        return _seed_records()


def _apply(rows: list) -> None:
    """Populate the live containers IN PLACE from editor rows."""
    INSTRUMENTS.clear()
    IMPLIED_VOL_OVERRIDE.clear()
    PRICE_FIELD_OVERRIDE.clear()
    for r in rows:
        tk = str(r.get("ticker", "") or "").strip()
        if not tk:
            continue
        try:
            px = float(r.get("price") or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        INSTRUMENTS[tk] = (str(r.get("name") or tk), px,
                           str(r.get("asset") or ""), str(r.get("region") or ""))
        vs = str(r.get("vol_source") or "").strip()
        if vs:
            IMPLIED_VOL_OVERRIDE[tk] = (vs, str(r.get("vol_field") or "PX_LAST").strip() or "PX_LAST")
        pf = str(r.get("price_field") or "").strip()
        if pf:
            PRICE_FIELD_OVERRIDE[tk] = pf
    # Trend/momentum runs on the whole book minus the vol-only cash indices.
    TREND_UNIVERSE[:] = [t for t in INSTRUMENTS if t not in PRICE_FIELD_OVERRIDE]


def reload() -> None:
    """Re-read the store and refresh the live universe in place."""
    _apply(load_rows())


def save(rows: list) -> None:
    """Persist editor rows to the store, then refresh the live universe in place."""
    norm = [{k: r.get(k, "") for k in _FIELDS} for r in rows]
    _write_store(norm)
    _apply(norm)


_apply(load_rows())   # fill the live universe at import


def name(ticker: str) -> str:
    info = INSTRUMENTS.get(ticker)
    return info[0] if info else ticker


def asset(ticker: str) -> str:
    info = INSTRUMENTS.get(ticker)
    return info[2] if info else ""


def region(ticker: str) -> str:
    info = INSTRUMENTS.get(ticker)
    return info[3] if info else ""


# ── Sector / product on-off filter (data/sector_filter.json) ────────────────
# A global "focus" filter shared by the dashboard AND the report generators: an
# instrument is ON unless its whole asset class is switched off, or it is switched
# off individually. Edited on the Home page; read here by everything else.
SECTOR_FILTER = STORE.parent / "sector_filter.json"
# The user's saved STARTUP DEFAULT (data/filter_default.json): the baseline the dashboard
# loads on each launch. Set via the Home "Set default" button; empty/absent = everything on.
DEFAULT_FILTER = STORE.parent / "filter_default.json"


def filter_off() -> tuple:
    """(off_assets:set, off_tickers:set) from the saved filter; empty sets if none."""
    try:
        d = json.loads(SECTOR_FILTER.read_text(encoding="utf-8"))
        return set(d.get("off_assets", [])), set(d.get("off_tickers", []))
    except Exception:
        return set(), set()


def save_filter(off_assets, off_tickers) -> None:
    SECTOR_FILTER.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_FILTER.write_text(json.dumps(
        {"off_assets": sorted(off_assets), "off_tickers": sorted(off_tickers)}, indent=2),
        encoding="utf-8")


def enabled_tickers() -> set:
    """Tickers currently ON (asset class enabled AND not switched off individually)."""
    off_a, off_t = filter_off()
    return {tk for tk, info in INSTRUMENTS.items()
            if info[2] not in off_a and tk not in off_t}


def filter_active() -> bool:
    """True if anything is currently filtered out (so callers can show a badge)."""
    off_a, off_t = filter_off()
    return bool(off_a or off_t)


def default_off() -> tuple:
    """(off_assets, off_tickers) of the saved startup default; empty sets if none set."""
    try:
        d = json.loads(DEFAULT_FILTER.read_text(encoding="utf-8"))
        return set(d.get("off_assets", [])), set(d.get("off_tickers", []))
    except Exception:
        return set(), set()


def save_default(off_assets, off_tickers) -> None:
    """Persist the current selection as the startup default (the launch baseline)."""
    DEFAULT_FILTER.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_FILTER.write_text(json.dumps(
        {"off_assets": sorted(off_assets), "off_tickers": sorted(off_tickers)}, indent=2),
        encoding="utf-8")


# ── Client-report product exclusions (data/report_exclude.json) ──────────────
# Products the desk's CLIENTS rarely trade: held out of the client-facing Technical
# Analysis report only. They stay fully live everywhere else in BASIS — the universe,
# every strategy page, the hub scoring and the other reports. Deliberately SEPARATE
# from the Sectors & products filter above, which switches a market off app-wide.
REPORT_EXCLUDE = STORE.parent / "report_exclude.json"
# One exclusion list PER BOOK — the FICC and equities Technical Analysis reports each hold out
# their own products. `scope` selects the file; default "ficc" keeps the original path.
REPORT_EXCLUDE_FILES = {"ficc": REPORT_EXCLUDE,
                        "equities": STORE.parent / "eq_report_exclude.json"}


def report_excluded(scope: str = "ficc") -> set:
    """Tickers held out of the client Technical Analysis report for `scope` ('ficc' | 'equities');
    empty set if none."""
    f = REPORT_EXCLUDE_FILES.get(scope, REPORT_EXCLUDE)
    try:
        return set(json.loads(f.read_text(encoding="utf-8")).get("tickers", []))
    except Exception:
        return set()


def save_report_excluded(tickers, scope: str = "ficc") -> None:
    """Persist the report-only exclusion list for `scope` ('ficc' | 'equities')."""
    f = REPORT_EXCLUDE_FILES.get(scope, REPORT_EXCLUDE)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"tickers": sorted(set(tickers))}, indent=2), encoding="utf-8")
