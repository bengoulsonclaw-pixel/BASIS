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
}
# How to finish the sentence "share of Brazil's ___" on the company chart's axis.
BASIS_AXIS = {
    "production": "production", "equity": "output", "crush": "cane crush",
    "slaughter": "processing", "export": "exports",
}
CONFIDENCE_LABEL = {
    "reported":    ("Reported", "Straight off company production releases."),
    "association": ("Industry body", "Published by the sector association."),
    "estimate":    ("Desk estimate", "Approximate — verify before client use."),
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
    return {"year": latest, "values": values, "history": sorted(hist, key=lambda r: r["year"]),
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
def _company_block(key: str, spec: dict, brazil_raw: float, brazil_share: float,
                   curated: dict) -> dict | None:
    blk = (curated.get("companies") or {}).get(key)
    if not blk or not blk.get("rows"):
        return None
    rows = [dict(r) for r in blk["rows"] if r.get("volume")]
    total = sum(float(r["volume"]) for r in rows)
    if total <= 0:
        return None
    for r in rows:
        v = float(r["volume"])
        r["volume"] = round(v, 3)
        r["share_brazil"] = round(v / total * 100, 2)
        r["share_world"] = round(v / total * brazil_share, 3)
        r["is_other"] = str(r.get("company", "")).lower().startswith("other")
    rows.sort(key=lambda r: (r["is_other"], -r["share_brazil"]))

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
        "note": blk.get("note"), "total": round(total, 3), "coverage_pct": coverage,
        "named_share": round(sum(r["share_brazil"] for r in rows if not r["is_other"]), 1),
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
            "companies": _company_block(spec["key"], spec, brazil, share, curated),
        }

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
        tag = f"  companies: {len(co['rows'])} ({co['basis_label']}, {co['confidence_label']})" if co else ""
        print(f"  {c['label']:<14} {c['year_label']:>9}  Brazil {c['brazil']:>10,.2f} / "
              f"world {c['world']:>10,.2f} {c['unit']:<14} = {c['share']:5.1f}%  #{c['rank']}{tag}")
    for e in store.get("errors") or []:
        print(f"  !! {e['label']}: {e['error']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
