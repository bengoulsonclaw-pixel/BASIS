"""EIA STEO data for the OPEC synopsis — an independent forecaster to benchmark
OPEC's own numbers against (and a real Brent/WTI price series with EIA's forward path).

The API key is read from $EIA_API_KEY or data/eia_key.txt. Everything here is
best-effort and graceful: any failure (no key, network, schema change) returns an
empty result so the OPEC report still builds from OPEC's data alone.

EIA v2 STEO series used (mb/d unless noted):
  PATC_WORLD  world liquid-fuels consumption     PAPR_WORLD     world production
  PAPR_OPEC   OPEC total petroleum supply        PAPR_NONOPEC   non-OPEC liquids
  BREPUUS     Brent spot ($/b)                   WTIPUUS        WTI spot ($/b)
  T3_STCHANGE_WORLD  world net inventory withdrawals (mb/d; +ve = draw)
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BASE = "https://api.eia.gov/v2/"
_ANNUAL = ["PATC_WORLD", "PAPR_WORLD", "PAPR_OPEC", "PAPR_NONOPEC",
           "BREPUUS", "WTIPUUS", "T3_STCHANGE_WORLD"]


def _key():
    k = os.getenv("EIA_API_KEY")
    if k:
        return k.strip()
    f = _ROOT / "data" / "eia_key.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def _get(path: str, key: str):
    url = _BASE + path + ("&" if "?" in path else "?") + "api_key=" + key
    req = urllib.request.Request(url, headers={"User-Agent": "opec-report"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _series_query(freq, series, start, end):
    q = f"steo/data/?frequency={freq}&data[]=value&start={start}&end={end}&length=500"
    for s in series:
        q += "&facets[seriesId][]=" + s
    return q + "&sort[0][column]=period&sort[0][direction]=asc"


def fetch() -> dict:
    """Return {ok, ann:{series:{year:val}}, prices:[{m,brent,wti}]} — or {ok:False}."""
    key = _key()
    if not key:
        return {"ok": False, "reason": "no EIA key"}
    try:
        ann = defaultdict(dict)
        d = _get(_series_query("annual", _ANNUAL, "2024", "2027"), key)
        for r in d["response"]["data"]:
            try:
                ann[r["seriesId"]][r["period"]] = float(r["value"])
            except (TypeError, ValueError):
                pass
        prices = defaultdict(dict)
        dp = _get(_series_query("monthly", ["BREPUUS", "WTIPUUS"], "2025-01", "2026-12"), key)
        for r in dp["response"]["data"]:
            try:
                prices[r["period"]][r["seriesId"]] = round(float(r["value"]), 2)
            except (TypeError, ValueError):
                pass
        price_rows = [{"m": m, "brent": prices[m].get("BREPUUS"), "wti": prices[m].get("WTIPUUS")}
                      for m in sorted(prices)]
        return {"ok": True, "ann": dict(ann), "prices": price_rows}
    except Exception as e:                     # network / schema / auth — degrade silently
        return {"ok": False, "reason": str(e)[:160]}


def _g(ann, series, year):
    return ann.get(series, {}).get(year)


def _growth(ann, series, year):
    a, b = _g(ann, series, year), _g(ann, series, str(int(year) - 1))
    return None if a is None or b is None else round(a - b, 2)


def build_comparison(opec: dict) -> dict:
    """Build the OPEC-vs-EIA comparison rows + a divergence callout, from EIA's STEO and
    the OPEC figures already parsed. `opec` carries: demand_growth_2026, demand_level_2026,
    nondoc_growth_2026, orb_2026, drawing(bool). Returns {} if EIA is unavailable."""
    res = fetch()
    if not res.get("ok"):
        return {}
    ann = res["ann"]
    eia_dem_g = _growth(ann, "PATC_WORLD", "2026")
    eia_dem_l = _g(ann, "PATC_WORLD", "2026")
    eia_nonopec_g = _growth(ann, "PAPR_NONOPEC", "2026")
    eia_brent = _g(ann, "BREPUUS", "2026")
    eia_draw = _g(ann, "T3_STCHANGE_WORLD", "2026")

    def fmt(v, sign=False, nd=1, suf=""):
        if v is None:
            return "—"
        return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}") + suf

    rows = []
    if opec.get("demand_growth_2026") is not None and eia_dem_g is not None:
        rows.append({"metric": "World oil demand growth, 2026 (mb/d)",
                     "opec": fmt(opec["demand_growth_2026"], sign=True),
                     "eia": fmt(eia_dem_g, sign=True)})
    if opec.get("demand_level_2026") is not None and eia_dem_l is not None:
        rows.append({"metric": "World oil demand, 2026 (mb/d)",
                     "opec": fmt(opec["demand_level_2026"]), "eia": fmt(eia_dem_l)})
    if opec.get("nondoc_growth_2026") is not None and eia_nonopec_g is not None:
        rows.append({"metric": "Non-DoC / non-OPEC* supply growth, 2026 (mb/d)",
                     "opec": fmt(opec["nondoc_growth_2026"], sign=True),
                     "eia": fmt(eia_nonopec_g, sign=True)})
    if opec.get("orb_2026") is not None and eia_brent is not None:
        rows.append({"metric": "Crude price, 2026 avg ($/b)",
                     "opec": f"${opec['orb_2026']:.0f} ORB", "eia": f"${eia_brent:.0f} Brent"})
    if eia_draw is not None:
        rows.append({"metric": "2026 world stock change (mb/d)",
                     "opec": ("drawing" if opec.get("drawing") else "—"),
                     "eia": (f"draw {eia_draw:.1f}" if eia_draw > 0 else f"build {abs(eia_draw):.1f}")})

    # divergence callout — lead with demand (usually the widest, most decision-relevant gap)
    callout = None
    if opec.get("demand_growth_2026") is not None and eia_dem_g is not None:
        gap = opec["demand_growth_2026"] - eia_dem_g
        if abs(gap) >= 0.4:
            direction = "more bullish" if gap > 0 else "more cautious"
            callout = (f"EIA's STEO puts 2026 world oil demand growth at {eia_dem_g:+.1f} mb/d versus "
                       f"OPEC's {opec['demand_growth_2026']:+.1f} — OPEC is materially {direction} on demand "
                       f"(a ~{abs(gap):.1f} mb/d gap), the clearest divergence between the two outlooks.")
    return {"rows": rows, "callout": callout, "prices": res["prices"],
            "note": "*OPEC reports non-DoC supply and its Reference Basket (ORB); EIA reports "
                    "non-OPEC supply and Brent — bases differ slightly. EIA figures are STEO."}
