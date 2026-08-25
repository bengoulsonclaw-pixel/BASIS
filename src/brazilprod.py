"""brazilprod.py — data layer for the 🇧🇷 Brazil Production page (FICC → Fundamentals).

Answers two questions per commodity:

  1. How much does BRAZIL produce, how much does the WORLD produce, and who else
     produces it?  Pulled live and free:
       psd      USDA FAS PS&D bulk CSVs — country x commodity x year, no key
                (the same free download agdata.py already uses for stocks-to-use)
       eia      EIA v2 international — crude production by country, free key
       curated  metals, pulp and ethanol, which have no free machine-readable
                feed: hand-maintained in data/brazil_curated.json off USGS MCS
                and the industry associations, refreshed once a year.

  2. WHICH COMPANIES produce Brazil's share?  Physical output per company exists
     nowhere free — not in Yahoo, not in the equities fundamentals DB (that holds
     ~30 FINANCIAL fields, no tonnes) — and Brazilian producers are not in the
     equity universe anyway (S&P 500 / R2000 / NDX / DJ30 / FTSE / SX5E / DAX).
     So the company layer is a curated table in data/brazil_curated.json, seeded
     from company production reports and refreshed quarterly.

The honest part.  A company breakdown is only meaningful where the industry is
concentrated — iron ore, crude, pulp, cane, beef and poultry processing. For the
row crops (soybeans, corn, coffee) production is spread across tens of thousands
of private farms and no company grows a material share, so those blocks carry an
EXPORT or CRUSH basis instead. Every block states its `basis` and `confidence`,
both surfaced on the page and in the PDF; nothing silently passes a desk estimate
off as a reported figure. Where the company units match the national units the
store also carries `coverage_pct` — what the table adds up to against the
national number — so a stale or double-counted table is visible, not hidden.

Store: data/signals/brazil_prod.json, written by the daily pull (Ben's rule —
anything slow on page-open gets a disk store; in-process caches die on restart).

CLI:  python src/brazilprod.py [--force] [--out path.json]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
DATA = _ROOT / "data"
SIGNALS = DATA / "signals"
CURATED_FILE = DATA / "brazil_curated.json"
STORE_FILE = SIGNALS / "brazil_prod.json"

BRAZIL = "Brazil"
TOP_N = 10                      # producer countries charted before the "Other" bucket
HIST_FROM = 2000                # share-over-time history starts here

# ── commodity registry ───────────────────────────────────────────────────────
# src: 'psd' (USDA bulk CSV) | 'eia' (EIA international) | 'curated' (JSON file).
# raw_unit is what the SOURCE returns; disp_div / disp_unit are presentation only.
# A company block matches on raw_unit, so kb/d against kb/d reconciles and
# "% of exports" against "1000 MT" correctly does not.
PSD_BASE = "https://apps.fas.usda.gov/psdonline/downloads/"
PSD_GROUP_FILE = {
    "grains_pulses": "psd_grains_pulses_csv.zip",
    "oilseeds": "psd_oilseeds_csv.zip",
    "cotton": "psd_cotton_csv.zip",
    "sugar": "psd_sugar_csv.zip",
    "coffee": "psd_coffee_csv.zip",
    "livestock": "psd_livestock_csv.zip",
}

AG, ENERGY, METALS, FOREST = "Agriculture", "Energy", "Metals & minerals", "Forestry"


def _c(key, label, group, icon, src, raw_unit, disp_div, disp_unit, **kw):
    return dict(key=key, label=label, group=group, icon=icon, src=src,
                raw_unit=raw_unit, disp_div=disp_div, disp_unit=disp_unit, **kw)


COMMODITIES = [
    # ── Agriculture (USDA PS&D, free bulk CSV, no key) ──
    _c("soybeans", "Soybeans", AG, "🌱", "psd", "1000 MT", 1000.0, "Mt",
       psd_group="oilseeds", psd_commodity="Oilseed, Soybean"),
    _c("corn", "Corn", AG, "🌽", "psd", "1000 MT", 1000.0, "Mt",
       psd_group="grains_pulses", psd_commodity="Corn"),
    _c("sugar", "Sugar", AG, "🍬", "psd", "1000 MT", 1000.0, "Mt",
       psd_group="sugar", psd_commodity="Sugar, Centrifugal"),
    _c("coffee", "Coffee", AG, "☕", "psd", "1000 60 KG BAGS", 1000.0, "m bags (60kg)",
       psd_group="coffee", psd_commodity="Coffee, Green"),
    _c("cotton", "Cotton", AG, "🧵", "psd", "1000 480 lb. Bales", 1000.0, "m bales",
       psd_group="cotton", psd_commodity="Cotton"),
    _c("beef", "Beef & veal", AG, "🐂", "psd", "1000 MT CWE", 1000.0, "Mt CWE",
       psd_group="livestock", psd_commodity="Meat, Beef and Veal"),
    _c("chicken", "Chicken", AG, "🐔", "psd", "1000 MT", 1000.0, "Mt",
       psd_group="livestock", psd_commodity="Meat, Chicken"),
    # ── Energy ──
    _c("crude_oil", "Crude oil", ENERGY, "🛢️", "eia", "kb/d", 1000.0, "mb/d",
       eia_product=57),          # crude oil including lease condensate
    _c("ethanol", "Fuel ethanol", ENERGY, "⛽", "curated", "bn litres", 1.0, "bn litres"),
    # ── Metals & minerals (USGS MCS, curated annually) ──
    _c("iron_ore", "Iron ore", METALS, "⛏️", "curated", "Mt", 1.0, "Mt"),
    _c("niobium", "Niobium", METALS, "🔩", "curated", "kt", 1.0, "kt"),
    _c("bauxite", "Bauxite", METALS, "🪨", "curated", "Mt", 1.0, "Mt"),
    _c("gold", "Gold", METALS, "🥇", "curated", "t", 1.0, "t"),
    _c("nickel", "Nickel", METALS, "🔗", "curated", "kt", 1.0, "kt"),
    _c("manganese", "Manganese", METALS, "🧱", "curated", "kt", 1.0, "kt"),
    _c("copper", "Copper", METALS, "🟠", "curated", "kt", 1.0, "kt"),
    # ── Forestry ──
    _c("pulp", "Wood pulp", FOREST, "🌲", "curated", "Mt", 1.0, "Mt"),
]
BY_KEY = {c["key"]: c for c in COMMODITIES}
GROUP_ORDER = [AG, ENERGY, METALS, FOREST]

# What each company-table basis MEASURES — shown verbatim on the page so a crush
# or export share is never read as a production share.
BASIS_LABEL = {
    "production": ("Production", "Output the company itself produced."),
    "equity":     ("Equity share", "Working-interest share of output, not operatorship."),
    "crush":      ("Cane crushed", "Cane processed — each group splits it between sugar and ethanol."),
    "slaughter":  ("Processing share", "Share of animals slaughtered/processed, not of the herd or flock."),
    "export":     ("Export share", "Share of what leaves the country — NOT who grew it."),
    "operated":   ("Operated production", "Output the company OPERATES — not its equity share. "
                                          "An operator runs fields in which others hold interests."),
    "sold":       ("Volume sold", "Tonnage the company COMMERCIALISED, from its royalty "
                                  "returns — not what it dug up. Stock movements and the gap "
                                  "between output and shipments both land in it."),
}
# How to finish the sentence "share of Brazil's ___" on the company chart's axis.
BASIS_AXIS = {
    "production": "production", "equity": "output", "crush": "cane crush",
    "slaughter": "processing", "export": "exports", "operated": "operated output",
    "sold": "sold volume",
}
CONFIDENCE_LABEL = {
    "reported":    ("Reported", "Straight off company production releases."),
    "association": ("Industry body", "Published by the sector association."),
    "estimate":    ("Desk estimate", "Approximate — verify before client use."),
}

# ── futures hedge equivalents ────────────────────────────────────────────────
# How many lots it would take to hedge a year of output. Contracts are the ones
# already in the desk's universe (src/universe.py), so nothing here quotes an
# instrument the book cannot trade.
#
# `per_disp` is the number of CONTRACT UNITS in one of the commodity's display
# units — the only place unit conversion happens, and the thing to check first if
# a number looks wrong. Physical-commodity conversions used:
#     1 t = 2204.62262 lb            1 t gold = 32,150.7466 troy oz
#     soybeans 1 bu = 0.0272155 t    corn 1 bu = 0.0254012 t
#     coffee 1 bag = 60 kg           cotton 1 bale = 480 lb
#     1 US gal = 3.785411784 L       crude: kb/d x 365 calendar days
#
# `proxy` marks a CROSS hedge — the contract does not settle against the thing
# Brazil actually produces, so basis risk is material and the page says so.
_LB_PER_T = 2204.62262
_OZ_PER_T = 32150.7466
DRESSING_PCT = 57.0        # carcass weight as a % of liveweight — the beef proxy's assumption


def _h(ticker, name, size, size_unit, per_disp, proxy=False, note="", verified=""):
    # `verified` records how a contract SIZE was confirmed when volbt.POINT_VALUE has
    # no entry to cross-check it against. Sizes drive every lot count, so provenance
    # is worth carrying rather than remembering.
    return {"ticker": ticker, "name": name, "size": size, "size_unit": size_unit,
            "per_disp": per_disp, "proxy": proxy, "note": note, "verified": verified}


HEDGE = {
    "iron_ore":  _h("SCOA Comdty", "Iron Ore 62% Fe (SGX TSI)", 100, "t", 1e6,
                    verified="100 metric tonnes/lot confirmed by Ben, 2026-08-21"),
    "crude_oil": _h("COA Comdty", "Brent Crude (ICE)", 1000, "bbl", 1e6 * 365,
                    note="Brazilian grades price against Brent; the differential is left unhedged."),
    "gold":      _h("GCA Comdty", "Gold (COMEX)", 100, "troy oz", _OZ_PER_T),
    "copper":    _h("HGA Comdty", "Copper (COMEX)", 25_000, "lb", 1000 * _LB_PER_T),
    "sugar":     _h("SBA Comdty", "Sugar No.11 (ICE)", 112_000, "lb", 1e6 * _LB_PER_T),
    "coffee":    _h("KCA Comdty", "Coffee 'C' Arabica (ICE)", 37_500, "lb",
                    1e6 * 60 / 1000 * _LB_PER_T),
    "soybeans":  _h("S A Comdty", "Soybeans (CBOT)", 5_000, "bu", 1e6 / 0.0272155),
    "corn":      _h("C A Comdty", "Corn (CBOT)", 5_000, "bu", 1e6 / 0.0254012),
    "cotton":    _h("CTA Comdty", "Cotton No.2 (ICE)", 50_000, "lb", 1e6 * 480),
    "beef":      _h("LCA Comdty", "Live Cattle (CME)", 40_000, "lb",
                    1e6 / (DRESSING_PCT / 100) * _LB_PER_T, proxy=True,
                    note=f"CME cattle settle against US liveweight, not Brazilian carcass beef. "
                         f"Carcass weight is grossed to liveweight at a {DRESSING_PCT:.0f}% "
                         f"dressing yield. A directional proxy only — the basis is large."),
    # CUAA is the Chicago Ethanol (PLATTS) swap future at 42,000 gal — i.e. 1,000
    # barrels, sized like an oil contract. Not the 29,000 gal CBOT ethanol future (EH).
    "ethanol":   _h("CUAA Comdty", "Chicago Ethanol (Platts) swap", 42_000, "gal",
                    1e9 / 3.785411784, proxy=True,
                    verified="42,000 gal/lot (= 1,000 bbl) confirmed by Ben, 2026-08-21",
                    note="US corn ethanol against Brazilian cane ethanol — different feedstock, "
                         "different market. Brazil's own hydrous contract trades on B3."),
}
# ── input hedges ─────────────────────────────────────────────────────────────
# Some producers have no future on what they SELL but a large, liquid hedge on what
# they BUY. A poultry integrator is the clear case: there is no chicken contract, but
# feed is most of the cost of a bird and both feed grains trade deeply. Leaving those
# names at zero lots would badly understate the brokerage they actually generate.
#
# Chicken chain, per tonne of ready-to-cook output:
#   RTC -> liveweight at a 75% yield, liveweight -> feed at a 1.75 feed-conversion
#   ratio, so 2.33 t of feed per tonne of meat; that ration is ~62% corn, ~30% meal.
# CBOT soybean meal is 100 SHORT tons = 90.718474 t, not a metric contract.
_T_PER_SHORT_TON = 0.90718474
_FEED_PER_T_MEAT = (1 / 0.75) * 1.75
INPUT_HEDGE = {
    "chicken": {
        "why": "No poultry future exists. Integrators hedge the INPUT — corn and soybean "
               "meal are the bulk of the cost of a bird — so their brokerage sits in the "
               "feed grains, not in the meat.",
        "assumption": f"{_FEED_PER_T_MEAT:.2f}t of feed per tonne of meat (75% dressing, "
                      f"1.75 feed-conversion ratio), a ration of 62% corn / 30% soybean meal.",
        "legs": [
            {"ticker": "C A Comdty", "name": "Corn (CBOT)", "size": 5_000, "size_unit": "bu",
             "per_disp": 1e6 * _FEED_PER_T_MEAT * 0.62 / 0.0254012},
            {"ticker": "SMA Comdty", "name": "Soybean Meal (CBOT)", "size": 100,
             "size_unit": "short tons",
             "per_disp": 1e6 * _FEED_PER_T_MEAT * 0.30 / _T_PER_SHORT_TON},
        ],
    },
}

# Deliberately absent, with the reason shown on the page rather than a blank cell.
NO_HEDGE = {
    "niobium":   "No niobium future exists anywhere — the market is priced by bilateral contract.",
    "pulp":      "No liquid pulp future in the desk's universe. Pulp futures trade on SHFE and "
                 "Norexeco, neither of them carried here.",
    "nickel":    "Nickel hedges on the LME, which this book does not currently carry.",
    "manganese": "No liquid manganese future.",
    "bauxite":   "No bauxite future. The chain hedges downstream in alumina or aluminium, at "
                 "roughly four tonnes of bauxite per tonne of aluminium.",
    "chicken":   "No poultry future. Integrators hedge the INPUT instead — corn and soybean "
                 "meal are the bulk of the cost of a bird.",
}


# ── small helpers ────────────────────────────────────────────────────────────
def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def load_curated() -> dict:
    """The hand-maintained country + company tables. Missing/corrupt → empty, never raises."""
    try:
        return json.loads(CURATED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rank_of(name: str, series: dict) -> int | None:
    order = sorted(series, key=lambda k: -series[k])
    return order.index(name) + 1 if name in order else None


def _country_rows(values: dict, top_n: int = TOP_N) -> list[dict]:
    """Top-N producers + an 'Other' bucket, each with its share. Brazil is always
    kept even if it falls outside the top N (nickel, copper, gold)."""
    total = sum(v for v in values.values() if v and v > 0)
    if total <= 0:
        return []
    ranked = sorted(values.items(), key=lambda kv: -kv[1])
    head = [kv for kv in ranked[:top_n]]
    if BRAZIL in values and BRAZIL not in dict(head):
        head.append((BRAZIL, values[BRAZIL]))
    named = {k for k, _ in head}
    other = sum(v for k, v in ranked if k not in named and v > 0)
    rows = [{"country": k, "value": round(float(v), 3),
             "share": round(float(v) / total * 100, 2), "is_brazil": k == BRAZIL,
             "is_other": k == "Other"}
            for k, v in head if k != "Other"]
    other += sum(v for k, v in head if k == "Other")
    if other > 0:
        rows.append({"country": "Other", "value": round(float(other), 3),
                     "share": round(other / total * 100, 2), "is_brazil": False,
                     "is_other": True})
    return rows


# ── source 1: USDA PS&D bulk CSV (free, no key) ──────────────────────────────
_PSD_CACHE: dict[str, pd.DataFrame] = {}


def _psd_download(group: str) -> pd.DataFrame:
    if group in _PSD_CACHE:
        return _PSD_CACHE[group]
    url = PSD_BASE + PSD_GROUP_FILE[group]
    raw = urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "basis-brazil-production"}), timeout=240).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(zf.open(name))
    for col in ("Commodity_Description", "Country_Name", "Attribute_Description", "Unit_Description"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    _PSD_CACHE[group] = df
    return df


def _psd_production(spec: dict) -> dict:
    """Country production by market year for one PS&D commodity.

    World = the sum of the country rows. PS&D reports the EU either as the 'European
    Union' aggregate (1999 onwards) or as individual member states (before that),
    never both in the same year, so summing per-year does not double-count. The
    same holds for the USSR and its successor states.
    """
    df = _psd_download(spec["psd_group"])
    p = df[(df["Commodity_Description"] == spec["psd_commodity"])
           & (df["Attribute_Description"] == "Production")]
    if p.empty:
        raise ValueError(f"no PS&D production rows for {spec['psd_commodity']}")
    p = p[pd.to_numeric(p["Market_Year"], errors="coerce").notna()]
    p["Market_Year"] = p["Market_Year"].astype(int)
    latest = int(p["Market_Year"].max())

    hist = []
    for yr, g in p[p["Market_Year"] >= HIST_FROM].groupby("Market_Year"):
        vals = g.groupby("Country_Name")["Value"].sum()
        world = float(vals.sum())
        if world <= 0:
            continue
        br = float(vals.get(BRAZIL, 0.0))
        hist.append({"year": int(yr), "brazil": br, "world": world,
                     "share": round(br / world * 100, 2)})

    cur = p[p["Market_Year"] == latest].groupby("Country_Name")["Value"].sum()
    values = {k: float(v) for k, v in cur.items() if v and v > 0}

    # Brazil's EXPORTS as well as its production: a trade-house's share is a share of
    # what leaves the country, so its hedgeable tonnage has to be struck off exports,
    # not off the crop.
    exp = df[(df["Commodity_Description"] == spec["psd_commodity"])
             & (df["Attribute_Description"] == "Exports")
             & (df["Country_Name"] == BRAZIL)]
    exp = exp[pd.to_numeric(exp["Market_Year"], errors="coerce") == latest]
    brazil_exports = float(exp["Value"].sum()) if not exp.empty else None

    return {"year": latest, "values": values, "history": sorted(hist, key=lambda r: r["year"]),
            "brazil_exports": brazil_exports,
            "source_label": "USDA FAS PS&D", "year_label": f"{latest - 1}/{str(latest)[-2:]} MY"}


# ── source 2: EIA v2 international (free key) ────────────────────────────────
def _eia_key() -> str | None:
    k = (os.getenv("EIA_API_KEY") or "").strip()
    if not k and (DATA / "eia_key.txt").exists():
        k = (DATA / "eia_key.txt").read_text(encoding="utf-8").strip()
    return k or None


def _eia_production(spec: dict) -> dict:
    """Annual crude production by country. The API caps a page at 5,000 rows and
    there are ~245 reporting entities, so page through with offset."""
    key = _eia_key()
    if not key:
        raise ValueError("no EIA key (data/eia_key.txt or $EIA_API_KEY)")
    base = ("https://api.eia.gov/v2/international/data/?frequency=annual&data[0]=value"
            f"&facets[productId][]={spec['eia_product']}&facets[activityId][]=1"
            f"&start={HIST_FROM}&sort[0][column]=period&sort[0][direction]=desc"
            "&length=5000&offset={off}&api_key=" + key)
    rows, off = [], 0
    while True:
        with urllib.request.urlopen(urllib.request.Request(
                base.format(off=off), headers={"User-Agent": "basis-brazil-production"}),
                timeout=120) as r:
            page = json.load(r)["response"]["data"]
        rows.extend(page)
        if len(page) < 5000:
            break
        off += 5000
        if off > 60000:                      # runaway guard
            break
    if not rows:
        raise ValueError("EIA returned no rows")

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = pd.to_numeric(df["period"], errors="coerce")
    df = df.dropna(subset=["value", "year"])
    # Countries only — the same payload carries OPEC / OECD / World aggregates,
    # which would double-count if summed alongside their members.
    ctry = df[df["countryRegionTypeId"] == "c"]
    world = df[df["countryRegionName"] == "World"]

    latest = int(ctry.loc[ctry["countryRegionName"] == BRAZIL, "year"].max())
    hist = []
    for yr, g in ctry.groupby("year"):
        w = world.loc[world["year"] == yr, "value"]
        tot = float(w.iloc[0]) if len(w) else float(g["value"].sum())
        br = float(g.loc[g["countryRegionName"] == BRAZIL, "value"].sum())
        if tot > 0 and br > 0:
            hist.append({"year": int(yr), "brazil": br, "world": tot,
                         "share": round(br / tot * 100, 2)})

    cur = ctry[ctry["year"] == latest].groupby("countryRegionName")["value"].sum()
    values = {k: float(v) for k, v in cur.items() if v and v > 0}
    return {"year": latest, "values": values, "history": sorted(hist, key=lambda r: r["year"]),
            "source_label": "EIA international", "year_label": str(latest)}


# ── source 3: curated country table (USGS MCS / associations) ────────────────
def _curated_production(spec: dict, curated: dict) -> dict:
    blk = (curated.get("countries_curated") or {}).get(spec["key"])
    if not blk:
        raise ValueError(f"no curated country block for {spec['key']}")
    values = {k: float(v) for k, v in (blk.get("countries") or {}).items() if v and v > 0}
    if not values:
        raise ValueError(f"curated block for {spec['key']} has no countries")
    # The published world total is authoritative; the listed countries are only the
    # majors, so keep the stated world and let "Other" absorb the difference. If the
    # majors sum to MORE than the stated world the table is inconsistent — say so
    # rather than silently redefining the world total to the sum.
    world = float(blk.get("world") or sum(values.values()))
    listed = sum(values.values())
    warn = None
    if world >= listed:
        if world > listed:
            values = dict(values)
            values["Other"] = world - listed
    else:
        warn = (f"listed producers sum to {listed:,.1f} {blk.get('unit', '')} against a stated world "
                f"total of {world:,.1f} — the world figure or a country figure is stale")
    yr = int(blk.get("year") or 0)
    return {"year": yr, "values": values, "history": [],
            "source_label": blk.get("source") or "curated", "year_label": str(yr),
            "note": blk.get("note"), "warning": warn}


# ── company layer ────────────────────────────────────────────────────────────
# Where a figure came from. Nothing is shown as a number unless it is live or keyed.
PROVENANCE = {
    "live":       ("Live feed", "Fetched this run from a machine-readable source."),
    "keyed":      ("Keyed from source", "Typed in from a named published document you can re-check."),
    "names_only": ("Volumes not sourced", "We know who produces it — not how much each one makes."),
}


def _anp_crude_block(brazil_share: float) -> dict | None:
    """The crude company table, straight from ANP. Operated production, not equity —
    the distinction is carried on the block so the page cannot mislabel it."""
    try:
        try:
            from src import anpdata           # imported by the app, from the repo root
        except ImportError:
            import anpdata                    # run as `python src/brazilprod.py`
        got = anpdata.by_operator()
    except Exception as exc:
        # Loud: falling back to the unsourced curated block without saying so is how a
        # live feed quietly stops being live.
        print(f"  ANP crude unavailable ({type(exc).__name__}: {exc}) — crude company "
              f"table falls back to NOT SOURCED")
        return None
    ops = (got or {}).get("operators") or {}
    total = float(got.get("total_bopd") or 0)
    if not ops or total <= 0:
        return None
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:12]
    named = sum(v for _k, v in top)
    rows = [{"company": k, "ticker": "", "volume": round(v / 1000.0, 3), "kind": "company"}
            for k, v in top]
    if total > named:
        rows.append({"company": f"Other ({len(ops) - len(top)} smaller operators)",
                     "ticker": "", "volume": round((total - named) / 1000.0, 3),
                     "kind": "other"})
    return {
        "basis": "operated", "year": (got.get("last_month") or "")[:4],
        "unit": "kb/d", "confidence": "reported", "provenance": "live",
        "source": got.get("source"),
        "note": (f"OPERATED production, not equity. ANP names who operates each well; "
                 f"Petrobras operates most of the pre-salt including fields where Shell, "
                 f"TotalEnergies, CNOOC, Equinor and Galp hold large working interests, so "
                 f"its operated share is well above its equity share. Equity would need "
                 f"ANP's consortium-participation data, which is not wired — so equity is "
                 f"not something we currently know. Window: {got.get('first_month')} to "
                 f"{got.get('last_month')} ({got.get('n_months')} months)."),
        "rows": rows,
    }


# CFEM reports tonnes; these commodities are displayed in millions of tonnes.
_ANM_DISPLAY_DIV = {"iron_ore": 1e6, "bauxite": 1e6}


def _anm_metals_block(key: str) -> dict | None:
    """A metals company table from ANM's CFEM royalty returns, when it passes the gates.

    Volume SOLD, not produced — CFEM is levied on what was commercialised. anmdata
    refuses any commodity whose sold tonnage will not reconcile against the national
    figure (CFEM reports gross ore, USGS reports contained metal for most metals), so
    a `None` here means the split is genuinely not known rather than merely missing.
    """
    div = _ANM_DISPLAY_DIV.get(key)
    if not div:
        return None
    try:
        try:
            from src import anmdata           # imported by the app, from the repo root
        except ImportError:
            import anmdata                    # run as `python src/brazilprod.py`
        got = anmdata.from_store(key)
    except Exception as exc:
        # Loud, for the same reason as ANP: a live feed that quietly stops being live
        # and falls back to the unsourced block is the failure worth preventing.
        print(f"  ANM {key} unavailable ({type(exc).__name__}: {exc}) — company table "
              f"falls back to NOT SOURCED")
        return None
    if got is None:
        print(f"  ANM {key}: no store at data/signals/anm_metals.json — run the daily "
              f"pull, or `python src/anmdata.py`. Company table falls back to NOT SOURCED")
        return None
    if not got.get("sourced") or not got.get("companies"):
        return None

    top = got["companies"][:12]
    rows = []
    for c in top:
        ticker, yahoo = anmdata.TICKERS.get(c["company"], ("", ""))
        rows.append({"company": c["company"], "ticker": ticker, "yahoo": yahoo,
                     "volume": round(c["tonnes"] / div, 3), "kind": "company"})
    named = sum(c["tonnes"] for c in top)
    rest = got["total_t"] - named
    if rest > 0 and len(got["companies"]) > len(top):
        rows.append({"company": f"Other ({len(got['companies']) - len(top)} smaller producers)",
                     "ticker": "", "volume": round(rest / div, 3), "kind": "other"})
    excluded = got.get("excluded_tonnes") or 0.0
    note = (f"Volume SOLD, from CFEM royalty returns — the tonnage each title-holder "
            f"paid royalty on, not what it mined. Filers declaring run-of-mine tonnage "
            f"are excluded, because ROM carries a fraction of the royalty per tonne and "
            f"summing it with saleable product overstates Brazil badly")
    if excluded > 0:
        note += (f": {got['excluded_filers']} filer(s) and "
                 f"{excluded / div:,.1f} {'Mt' if div >= 1e6 else 't'} left out on that test")
    note += (f". What remains reconciles to {got['reconciliation']:.2f}x Brazil's national "
             f"figure — measured, not fitted.")
    return {
        "basis": "sold", "year": got.get("year"),
        "unit": "Mt" if div >= 1e6 else "t",
        "confidence": "reported", "provenance": "live",
        "source": got.get("source"), "note": note, "rows": rows,
    }


def _company_block(key: str, spec: dict, brazil_raw: float, brazil_share: float,
                   curated: dict) -> dict | None:
    blk = (curated.get("companies") or {}).get(key)
    if key == "crude_oil":
        live = _anp_crude_block(brazil_share)
        if live:
            blk = live
    elif key in _ANM_DISPLAY_DIV:
        live = _anm_metals_block(key)
        if live:
            blk = live
    if not blk or not blk.get("rows"):
        return None

    # No source, no numbers. The producers are still listed — knowing Vale mines iron
    # ore is not a guess — but every volume, share and lot count is withheld and the
    # page says why instead of printing something that looks measured.
    if blk.get("provenance") == "names_only":
        return {
            "unsourced": True, "provenance": "names_only",
            "provenance_label": PROVENANCE["names_only"][0],
            "reason": blk.get("unsourced_reason") or "No source wired for company volumes.",
            "basis": blk.get("basis", "production"),
            "basis_label": BASIS_LABEL.get(blk.get("basis", "production"), ("", ""))[0],
            "entity_label": "Producer",
            "names": [r["company"] for r in blk["rows"]
                      if not str(r.get("company", "")).lower().startswith(("other", "artisanal"))],
            "rows": [],
        }
    rows = [dict(r) for r in blk["rows"] if r.get("volume")]
    total = sum(float(r["volume"]) for r in rows)
    if total <= 0:
        return None
    for r in rows:
        v = float(r["volume"])
        r["volume"] = round(v, 3)
        r["share_brazil"] = round(v / total * 100, 2)
        r["share_world"] = round(v / total * brazil_share, 3)
        # 'artisanal' is a real producer of Brazilian output but is NOT a company —
        # Brazil's garimpo gold being the case that forced the distinction. Rows can
        # declare their kind; anything undeclared falls back to the name convention.
        r["kind"] = (r.get("kind")
                     or ("other" if str(r.get("company", "")).lower().startswith("other")
                         else "company"))
        r["is_other"] = r["kind"] == "other"
        r["is_artisanal"] = r["kind"] == "artisanal"
        # 'group' = several companies on one line: a valid production figure, but not
        # a single client, so the brokerage roll-up skips it.
        r["is_group"] = r["kind"] == "group"
    # Companies by size, then any non-corporate producer, then the Other bucket.
    rows.sort(key=lambda r: (r["is_other"], r["is_artisanal"], -r["share_brazil"]))

    basis = blk.get("basis", "production")
    conf = blk.get("confidence", "estimate")
    # Only meaningful when the company table is denominated the same way as the
    # national figure — a "% of exports" block has nothing to reconcile against.
    coverage = None
    if blk.get("unit") == spec["raw_unit"] and brazil_raw:
        coverage = round(total / brazil_raw * 100, 1)
    return {
        "basis": basis, "basis_label": BASIS_LABEL.get(basis, (basis, ""))[0],
        "basis_note": BASIS_LABEL.get(basis, ("", ""))[1],
        "confidence": conf, "confidence_label": CONFIDENCE_LABEL.get(conf, (conf, ""))[0],
        "confidence_note": CONFIDENCE_LABEL.get(conf, ("", ""))[1],
        "axis_label": f"share of Brazil's {BASIS_AXIS.get(basis, 'output')} (%)",
        # A "% of ..." block's volume column IS its share of Brazil, so the page drops it.
        "unit_is_pct": str(blk.get("unit", "")).strip().startswith("%"),
        "year": blk.get("year"), "unit": blk.get("unit"), "source": blk.get("source"),
        "provenance": blk.get("provenance", "keyed"),
        "provenance_label": PROVENANCE.get(blk.get("provenance", "keyed"), ("", ""))[0],
        "note": blk.get("note"), "total": round(total, 3), "coverage_pct": coverage,
        "named_share": round(sum(r["share_brazil"] for r in rows
                                 if not r["is_other"] and not r["is_artisanal"]), 1),
        # A table carrying a non-corporate producer can't head its first column "Company".
        "has_artisanal": any(r["is_artisanal"] for r in rows),
        "entity_label": "Producer" if any(r["is_artisanal"] for r in rows) else "Company",
        "artisanal_share": round(sum(r["share_brazil"] for r in rows if r["is_artisanal"]), 1),
        "rows": rows,
    }


def _input_hedge_block(inp: dict, brazil_disp: float, blk: dict | None, unit: str) -> dict:
    """A multi-leg hedge on what the producer BUYS rather than what it sells. Lots per
    producer are the sum of the legs, since brokerage is earned on every one."""
    legs, national_lots = [], 0.0
    for leg in inp["legs"]:
        lots = float(brazil_disp) * leg["per_disp"] / leg["size"]
        national_lots += lots
        legs.append({"ticker": leg["ticker"], "name": leg["name"],
                     "size": leg["size"], "size_unit": leg["size_unit"],
                     "lots": int(round(lots)), "lots_per_month": int(round(lots / 12.0))})
    rows = []
    for r in (blk or {}).get("rows", []):
        lots = national_lots * r["share_brazil"] / 100.0
        rows.append({"company": r["company"], "share_brazil": r["share_brazil"],
                     "lots": int(round(lots)), "lots_per_month": int(round(lots / 12.0)),
                     "units": None, "is_other": r["is_other"],
                     "is_artisanal": r.get("is_artisanal", False),
                     "is_group": r.get("is_group", False)})
    return {
        "available": True, "is_input": True, "proxy": True,
        "ticker": " + ".join(l["ticker"].split()[0] for l in legs),
        "name": " + ".join(l["name"] for l in legs),
        "size": 0, "size_unit": "", "legs": legs,
        "note": inp["why"] + " Assumes " + inp["assumption"],
        "qty_basis": "production", "national_qty": round(float(brazil_disp), 3),
        "national_unit": unit, "national_units": None,
        "national_lots": int(round(national_lots)),
        "national_lots_per_month": int(round(national_lots / 12.0)),
        "rows": rows,
    }


def _hedge_block(key: str, brazil_disp: float, exports_disp: float | None,
                 blk: dict | None, unit: str) -> dict | None:
    """How many lots would hedge a year of Brazilian output, and each producer's slice.

    One rule throughout: a producer's hedgeable volume is its SHARE OF BRAZIL applied to
    Brazil's national quantity. That keeps volume and share on the same basis and makes
    the company lots sum to the national lots. The national quantity is Brazil's EXPORTS
    where the company table is an export share (a trade house hedges what it ships, not
    the whole crop) and Brazil's PRODUCTION everywhere else.
    """
    spec = HEDGE.get(key)
    if not spec:
        inp = INPUT_HEDGE.get(key)
        if inp and brazil_disp:
            return _input_hedge_block(inp, brazil_disp, blk, unit)
        return {"available": False, "reason": NO_HEDGE.get(
            key, "No listed future for this commodity in the desk's universe.")}

    if blk and blk.get("unsourced"):
        return {"available": False,
                "reason": "Company volumes are not sourced, so a hedge cannot be sized "
                          "per producer. " + blk.get("reason", "")}
    on_exports = bool(blk) and blk.get("basis") == "export"
    national = exports_disp if on_exports else brazil_disp
    if not national or national <= 0:
        return {"available": False,
                "reason": "No national quantity to strike the hedge against."}

    total_units = float(national) * spec["per_disp"]
    national_lots = total_units / spec["size"]
    rows = []
    for r in (blk or {}).get("rows", []):
        lots = national_lots * r["share_brazil"] / 100.0
        rows.append({"company": r["company"], "share_brazil": r["share_brazil"],
                     "lots": int(round(lots)), "lots_per_month": int(round(lots / 12.0)),
                     "units": round(total_units * r["share_brazil"] / 100.0, 1),
                     "is_other": r["is_other"], "is_artisanal": r.get("is_artisanal", False),
                     "is_group": r.get("is_group", False)})
    return {
        "available": True, "ticker": spec["ticker"], "name": spec["name"],
        "size": spec["size"], "size_unit": spec["size_unit"],
        "proxy": spec["proxy"], "note": spec["note"],
        "qty_basis": "exports" if on_exports else "production",
        "national_qty": round(float(national), 3), "national_unit": unit,
        "national_units": round(total_units, 1),
        "national_lots": int(round(national_lots)),
        "national_lots_per_month": int(round(national_lots / 12.0)),
        "rows": rows,
    }


# ── build ────────────────────────────────────────────────────────────────────
def build(force: bool = False, max_age_hours: float = 20.0) -> dict:
    """Assemble the whole page into one JSON-safe dict and cache it to STORE_FILE.
    Per-commodity failures are recorded and skipped — one dead source never takes
    the page down."""
    SIGNALS.mkdir(parents=True, exist_ok=True)
    if not force and _is_fresh(STORE_FILE, max_age_hours):
        cached = load()
        if cached:
            return cached

    curated = load_curated()
    out_c, errors = {}, []
    for spec in COMMODITIES:
        try:
            if spec["src"] == "psd":
                got = _psd_production(spec)
            elif spec["src"] == "eia":
                got = _eia_production(spec)
            else:
                got = _curated_production(spec, curated)
        except Exception as exc:
            errors.append({"key": spec["key"], "label": spec["label"],
                           "error": f"{type(exc).__name__}: {exc}"})
            continue

        values = got["values"]
        world = float(sum(values.values()))
        brazil = float(values.get(BRAZIL, 0.0))
        if world <= 0:
            errors.append({"key": spec["key"], "label": spec["label"], "error": "world total is zero"})
            continue
        share = brazil / world * 100.0
        div = spec["disp_div"]
        if got.get("warning"):
            errors.append({"key": spec["key"], "label": spec["label"],
                           "error": got["warning"], "level": "warning"})
        out_c[spec["key"]] = {
            "key": spec["key"], "label": spec["label"], "group": spec["group"],
            "icon": spec["icon"], "src": spec["src"],
            "unit": spec["disp_unit"], "raw_unit": spec["raw_unit"],
            "year": got["year"], "year_label": got.get("year_label", str(got["year"])),
            "source_label": got.get("source_label", ""), "note": got.get("note"),
            "brazil": round(brazil / div, 3), "world": round(world / div, 3),
            "brazil_raw": brazil, "share": round(share, 2),
            "rank": _rank_of(BRAZIL, values), "n_producers": len(values),
            "countries": [dict(r, value=round(r["value"] / div, 3)) for r in _country_rows(values)],
            "history": [{"year": h["year"], "brazil": round(h["brazil"] / div, 3),
                         "world": round(h["world"] / div, 3), "share": h["share"]}
                        for h in got["history"]],
            "companies": None,      # filled below — the hedge block needs it
        }
        cblk = _company_block(spec["key"], spec, brazil, share, curated)
        exports_disp = (got.get("brazil_exports") / div
                        if got.get("brazil_exports") else None)
        out_c[spec["key"]]["companies"] = cblk
        out_c[spec["key"]]["brazil_exports"] = (round(exports_disp, 3) if exports_disp else None)
        out_c[spec["key"]]["hedge"] = _hedge_block(
            spec["key"], brazil / div, exports_disp, cblk, spec["disp_unit"])

    store = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "curated_as_of": curated.get("as_of"),
        "group_order": GROUP_ORDER,
        "commodities": out_c,
        "errors": errors,
    }
    try:
        STORE_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return store


def load() -> dict | None:
    """The cached store, or None if the daily pull has never written one."""
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_or_build(max_age_hours: float = 20.0) -> dict:
    cached = load()
    if cached and _is_fresh(STORE_FILE, max_age_hours):
        return cached
    try:
        return build(max_age_hours=max_age_hours)
    except Exception:
        return cached or {"built": "—", "commodities": {}, "errors":
                          [{"key": "-", "label": "build", "error": "build failed and no cache"}]}


# ── brokerage view: lots per CLIENT, across their whole book ─────────────────
# The hedge blocks are per commodity, but a producer hedging "the whole business"
# is one client trading several commodities — JBS is beef and poultry, Vale is iron
# ore, copper and nickel, the Cosan complex is cane and crude. Brokerage is earned
# per lot, so the number that matters commercially is the sum across a client's
# entire production, not any single line.
_GROUP_OVERRIDE = {
    # ANP names each operating subsidiary; commercially they are one counterparty
    "Prio Tigris": "PRIO", "Prio Bravo": "PRIO", "Petro Rio Jaguar": "PRIO",
    "Petro Rio O&G": "PRIO", "3R Potiguar": "Brava Energia",
    "3R Petroleum": "Brava Energia", "3R Petroleum Off": "Brava Energia",
    "Equinor Brasil": "Equinor", "TotalEnergies EP": "TotalEnergies",
    "Karoon Brasil": "Karoon", "Perenco Brasil": "Perenco",
    # where stripping the bracket is not enough to land on one commercial entity
    "Raizen (Cosan / Shell)": "Cosan / Raizen",
    "Shell Brasil": "Shell",
    "BP Bioenergy": "BP",
    "CNPC / PetroChina": "CNPC",
    "Olam / ofi": "Olam",
}
# Lines that are real production but not a client anyone can broker for.
_NOT_A_CLIENT = ("other", "artisanal")


def group_key(company: str) -> str:
    """The commercial entity behind a production line: 'Vale (Salobo, Sossego)' and
    'Vale' are one client; 'JBS (Friboi)' and 'JBS (Seara)' are one client. Samarco
    stays its own name — it is a separate JV that trades in its own right."""
    name = str(company).strip()
    if name in _GROUP_OVERRIDE:
        return _GROUP_OVERRIDE[name]
    return name.split(" (")[0].strip() or name


def broker_book(store: dict | None = None, turns: float = 1.0) -> pd.DataFrame:
    """One row per prospective client: every commodity they produce, and the lots a
    full hedge of a full year would require across all of them.

    `turns` is round-turns per lot per year — 1.0 means the hedge is put on once and
    held. A producer that rolls a strip trades the same position several times over,
    and brokerage is earned each time, so raise it to model that. It is an ASSUMPTION,
    not a measurement, and the page labels it as one.

    Commodities with no listed hedge contribute nothing here: they generate no lots,
    however much of them the company produces.
    """
    store = store or load() or {}
    acc: dict = {}
    for com in (store.get("commodities") or {}).values():
        h = com.get("hedge") or {}
        if not h.get("available"):
            continue
        for r in h.get("rows", []):
            if r.get("is_other") or r.get("is_artisanal") or r.get("is_group"):
                continue
            if str(r["company"]).strip().lower().startswith(_NOT_A_CLIENT):
                continue
            g = acc.setdefault(group_key(r["company"]), {"lots": 0.0, "lines": [], "coms": []})
            g["lots"] += float(r["lots"])
            g["lines"].append(f"{com['label']}: {r['lots']:,} ({h['ticker'].split()[0]})")
            g["coms"].append(com["label"])
    rows = [{
        "Client": name,
        "Commodities": ", ".join(dict.fromkeys(v["coms"])),
        "n_commodities": len(dict.fromkeys(v["coms"])),
        "Lots (1 yr)": int(round(v["lots"] * turns)),
        "Lots / month": int(round(v["lots"] * turns / 12.0)),
        "detail": " · ".join(v["lines"]),
    } for name, v in acc.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Lots (1 yr)", ascending=False).reset_index(drop=True)


HEDGE_RATIOS = (100, 75, 50, 25)
# Each ratio breaks into year / month / day. Daily lots is the executability check —
# it is what you hold against the contract's own daily volume. 252 = trading days,
# not calendar days, so the daily figure is what could actually be worked.
TRADING_DAYS = 252
HEDGE_PERIODS = (("yr", 1), ("mth", 12), ("day", TRADING_DAYS))
# One colour per period so the eye can track a row across twelve numeric columns.
# Defined here rather than in each renderer — the page, the PDF and the email table all
# read these, and three separate copies would drift. Distinguished by HUE, not by
# lightness, so none of them is the "faint" one (house rule on readable text floors).
PERIOD_COLOUR = {                    # light canvas: the PDF and the email table
    "yr": "#1A1A1A", "mth": "#1F5FA8", "day": "#1F7A44",
}
PERIOD_COLOUR_DARK = {               # the app's dark canvas
    "yr": "#E8ECF1", "mth": "#5B9BF0", "day": "#46C58A",
}
PERIOD_WORD = {"yr": "year", "mth": "month", "day": "trading day"}


def hedge_matrix(store: dict | None = None, turns: float = 1.0,
                 include_unhedgeable: bool = True,
                 ratios: tuple = HEDGE_RATIOS) -> pd.DataFrame:
    """One row per COMPANY x PRODUCT: annual volume and the lots needed at each hedge
    ratio. A company that makes three things gets three rows, kept together and the
    company's biggest line first.

    Products with no listed future are still listed, with blank lot columns — a
    producer missing from the table would read as overlooked rather than unhedgeable.
    """
    store = store or load() or {}
    rows = []
    for com in (store.get("commodities") or {}).values():
        h = com.get("hedge") or {}
        blk = com.get("companies")
        if not blk or blk.get("unsourced"):
            continue
        avail = bool(h.get("available"))
        if not avail and not include_unhedgeable:
            continue
        # Say what the volume actually measures whenever it is not plain production.
        basis = blk.get("basis", "production")
        # For exports the VOLUME is export tonnage, so "(exports)" describes both. For
        # crush and slaughter the volume is finished product while only the SHARE comes
        # from cane or headage — say so, or the tonnage reads as cane.
        qual = {"export": "exports", "crush": "share by cane crush",
                "slaughter": "share by processing", "equity": "equity share"}.get(basis)
        product = com["label"] + (f" ({qual})" if qual else "")
        if h.get("is_input"):
            product = com["label"] + " (feed hedge)"
        contract = (" + ".join(l["ticker"].split()[0] for l in h["legs"])
                    if h.get("legs") else (h.get("ticker", "").split()[0] if avail else "—"))
        by_co = {r["company"]: r for r in h.get("rows", [])}
        for r in blk["rows"]:
            if r["is_other"] or r.get("is_artisanal") or r.get("is_group"):
                continue
            lots = float(by_co.get(r["company"], {}).get("lots", 0)) * turns if avail else None
            vol = (h["national_qty"] * r["share_brazil"] / 100.0) if avail else r["volume"]
            unit = h["national_unit"] if avail else (blk.get("unit") or "")
            row = {"Company": group_key(r["company"]), "Line": r["company"],
                   "Product": product, "Contract": contract,
                   "Annual production": round(vol, 2), "Unit": unit,
                   "_lots": lots or 0.0, "_avail": avail}
            for pct in ratios:
                at_ratio = (lots * pct / 100.0) if avail else None
                for suffix, divisor in HEDGE_PERIODS:
                    row[f"{pct}% {suffix}"] = (int(round(at_ratio / divisor))
                                               if avail else None)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # A source may file one client under several operating entities — ANP lists PRIO as
    # Prio Tigris, Prio Bravo and Petro Rio Jaguar. group_key merges the names, so the
    # rows behind them have to be summed too or the client appears three times for one
    # product with a third of its book on each line.
    agg = {"Line": lambda v: " + ".join(dict.fromkeys(v)),
           "Contract": "first", "Unit": "first", "_avail": "first",
           "Annual production": "sum", "_lots": "sum"}
    for pct in ratios:
        for suffix, _d in HEDGE_PERIODS:
            agg[f"{pct}% {suffix}"] = "sum"
    df = df.groupby(["Company", "Product"], as_index=False, sort=False).agg(agg)
    df["Annual production"] = df["Annual production"].round(2)

    # `sum` over an all-NaN group returns 0.0, not NaN — so aggregation silently turns
    # an unhedgeable product's blank lot cells into zeros, and "0 lots" reads as a
    # client with nothing to hedge rather than a product with no contract. Put the
    # blanks back. (Latent until ANM sourced bauxite, which is the first product to
    # carry real volumes and no listed future.)
    lot_cols = [f"{pct}% {suffix}" for pct in ratios for suffix, _d in HEDGE_PERIODS]
    df.loc[~df["_avail"], lot_cols] = pd.NA
    df.loc[~df["_avail"], "_lots"] = 0.0          # ordering only; never displayed

    # Company blocks ordered by the company's whole book, biggest line first within it.
    totals = df.groupby("Company")["_lots"].sum().rename("_co_total")
    df = df.join(totals, on="Company")
    return (df.sort_values(["_co_total", "Company", "_lots"], ascending=[False, True, False])
              .drop(columns=["_co_total"]).reset_index(drop=True))


def hedge_totals(mat: pd.DataFrame) -> dict:
    """Column totals for the hedge matrix — the whole addressable book in one line.

    Unhedgeable products hold NaN and drop out of the sum, so the total is lots that
    could actually be traded, not a count inflated by production nobody can hedge.
    """
    if mat is None or mat.empty:
        return {}
    out = {"_n_rows": int(len(mat)), "_n_companies": int(mat["Company"].nunique()),
           "_n_hedgeable": int(mat["_avail"].sum())}
    for pct in HEDGE_RATIOS:
        for suffix, _div in HEDGE_PERIODS:
            col = f"{pct}% {suffix}"
            out[col] = int(mat[col].sum(skipna=True)) if col in mat else 0
    return out


def headline_rows(store: dict | None = None) -> pd.DataFrame:
    """One row per commodity for the overview table — Brazil's volume, the world's,
    the share and the global rank, sorted by share descending."""
    store = store or load() or {}
    rows = []
    for c in (store.get("commodities") or {}).values():
        rows.append({"Commodity": f"{c['icon']} {c['label']}", "key": c["key"],
                     "Group": c["group"], "Year": c["year_label"],
                     "Brazil": c["brazil"], "World": c["world"], "Unit": c["unit"],
                     "Share %": c["share"], "Rank": c["rank"],
                     "Companies": bool(c.get("companies"))})
    df = pd.DataFrame(rows)
    return df.sort_values("Share %", ascending=False).reset_index(drop=True) if not df.empty else df


def main(argv: list[str]) -> int:
    force = "--force" in argv
    store = build(force=force)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = store.get("commodities") or {}
    print(f"built {store.get('built')} — {len(ok)} commodities, {len(store.get('errors') or [])} errors")
    for c in sorted(ok.values(), key=lambda x: -x["share"]):
        co = c.get("companies")
        if not co:
            tag = ""
        elif co.get("unsourced"):
            tag = f"  companies: NOT SOURCED ({len(co.get('names', []))} producers named)"
        else:
            tag = (f"  companies: {len(co['rows'])} ({co['basis_label']}, "
                   f"{co.get('confidence_label', '?')})")
        print(f"  {c['label']:<14} {c['year_label']:>9}  Brazil {c['brazil']:>10,.2f} / "
              f"world {c['world']:>10,.2f} {c['unit']:<14} = {c['share']:5.1f}%  #{c['rank']}{tag}")
    for e in store.get("errors") or []:
        print(f"  !! {e['label']}: {e['error']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
