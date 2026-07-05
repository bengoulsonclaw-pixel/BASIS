"""USDA Grain Stocks (+ June Acreage) — Reaction note, XP house style.

Quarterly: NASS publishes Grain Stocks four times a year (Jan→Dec-1, Mar→Mar-1,
Jun→Jun-1, Sep→Sep-1). This builds a reaction note for whichever quarter is latest:

  • JUNE adds the annual Acreage report → the full note: planted area vs March
    intentions & year-ago, wheat by class, June-1 stocks (on/off-farm), implied use.
  • MAR / SEP / DEC are stocks-only: total + on-farm/off-farm + implied Mar–May /
    Jun–Aug / etc. use, vs year-ago.

NASS keeps the March intention and June actual as separate reference-period records,
so the June surprise needs no manual input. Optional consensus: data/usda_consensus.json.

Run live (auto-detects the latest quarter):  python src/usdareaction.py out.pdf
A specific quarter:                          python src/usdareaction.py out.pdf --period SEP --year 2026
Demo the format on a past quarter:           python src/usdareaction.py out.pdf --period JUN --year 2025 --demo
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import agdata   # reuse _nass_get + NASS_KEY + latest_stocks_period

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
CONSENSUS_FILE = Path(__file__).resolve().parent.parent / "data" / "usda_consensus.json"

ACREAGE_CROPS = [("CORN", "Corn"), ("SOYBEANS", "Soybeans"), ("WHEAT", "Wheat (all)"),
                 ("COTTON", "Cotton (all)"), ("SORGHUM", "Sorghum")]
STOCKS_CROPS = [("CORN", "Corn"), ("SOYBEANS", "Soybeans"), ("WHEAT", "Wheat (all)")]
WHEAT_CLASSES = [("WINTER", "Winter"), ("SPRING, (EXCL DURUM)", "Spring (excl. durum)"), ("SPRING, DURUM", "Durum")]

# Quarterly Grain Stocks periods. prior_ref/prior_off locate the previous quarter (for
# implied disappearance = prior-quarter stocks − this-quarter stocks). DEC is use_valid=False
# because the Sep–Nov quarter includes the new-crop harvest, so a simple draw isn't "use".
PERIODS = {
    "MAR": {"ref": "FIRST OF MAR", "label": "March 1", "quarter": "Dec–Feb",
            "prior_ref": "FIRST OF DEC", "prior_off": -1, "use_valid": True},
    "JUN": {"ref": "FIRST OF JUN", "label": "June 1", "quarter": "Mar–May",
            "prior_ref": "FIRST OF MAR", "prior_off": 0, "use_valid": True},
    "SEP": {"ref": "FIRST OF SEP", "label": "September 1", "quarter": "Jun–Aug",
            "prior_ref": "FIRST OF JUN", "prior_off": 0, "use_valid": True},
    "DEC": {"ref": "FIRST OF DEC", "label": "December 1", "quarter": "Sep–Nov",
            "prior_ref": "FIRST OF SEP", "prior_off": 0, "use_valid": False},
}


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _pct(a, b):
    return (a / b - 1.0) * 100.0 if (a and b) else None


def _fmt_pct(v):
    return f"{v:+.1f}%" if v is not None else "—"


def _consensus(year):
    if CONSENSUS_FILE.exists():
        try:
            return json.loads(CONSENSUS_FILE.read_text(encoding="utf-8")).get(str(year), {})
        except Exception:
            return {}
    return {}


# ---- acreage (June only) ----------------------------------------------------
def _planted(commodity, year):
    rows = agdata._nass_get({
        "commodity_desc": commodity, "statisticcat_desc": "AREA PLANTED", "agg_level_desc": "NATIONAL",
        "unit_desc": "ACRES", "domain_desc": "TOTAL", "year__GE": str(year - 1)})

    def best(yr, ref, cls=None):
        vals = [_num(d.get("Value")) for d in rows
                if str(d.get("year")) == str(yr) and d.get("reference_period_desc") == ref
                and (cls is None or d.get("class_desc") == cls)]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    return rows, best


def _wheat_classes(year):
    rows, best = _planted("WHEAT", year)
    out = []
    for cls, label in WHEAT_CLASSES:
        jun, prior = best(year, "YEAR - JUN ACREAGE", cls), best(year - 1, "YEAR - JUN ACREAGE", cls)
        vs_yr = _pct(jun, prior)
        d = -1 if (vs_yr or 0) > 0.3 else 1 if (vs_yr or 0) < -0.3 else 0
        out.append({"crop": label, "actual": f"{jun/1e6:.1f}" if jun else "—", "vs_yr": _fmt_pct(vs_yr),
                    "cls": "short" if d < 0 else "long" if d > 0 else "flat",
                    "read": "More acres" if d < 0 else "Fewer acres" if d > 0 else "≈ flat"})
    return out


def build_acreage(year):
    consensus = _consensus(year)
    acre, pending = [], True
    for code, label in ACREAGE_CROPS:
        _, best = _planted(code, year)
        jun, mar, prior = best(year, "YEAR - JUN ACREAGE"), best(year, "YEAR - MAR ACREAGE"), best(year - 1, "YEAR - JUN ACREAGE")
        if jun:
            pending = False
        vs_mar, vs_yr = _pct(jun, mar), _pct(jun, prior)
        exp = consensus.get(code, {}).get("acres")
        vs_exp = _pct(jun, exp * 1e6) if (exp and jun) else None
        ref = vs_exp if vs_exp is not None else vs_mar
        d = (-1 if ref > 0.3 else 1 if ref < -0.3 else 0) if ref is not None else 0
        acre.append({"crop": label, "actual": f"{jun/1e6:.1f}" if jun else "—", "mar": f"{mar/1e6:.1f}" if mar else "—",
                     "vs_mar": _fmt_pct(vs_mar), "vs_yr": _fmt_pct(vs_yr), "vs_exp": _fmt_pct(vs_exp),
                     "cls": "short" if d < 0 else "long" if d > 0 else "flat",
                     "read": "Bearish — more acres" if d < 0 else "Bullish — fewer acres" if d > 0 else "≈ in line"})
    return acre, pending


# ---- stocks (any quarter) ---------------------------------------------------
def _positions(commodity, ref, year):
    rows = agdata._nass_get({
        "commodity_desc": commodity, "statisticcat_desc": "STOCKS", "agg_level_desc": "NATIONAL",
        "unit_desc": "BU", "domain_desc": "TOTAL", "reference_period_desc": ref, "year__GE": str(year - 1)})

    def pick(yr, kind):
        vals = []
        for d in rows:
            if str(d.get("year")) != str(yr) or (d.get("class_desc") or "") not in ("ALL CLASSES", ""):
                continue
            sd = d.get("short_desc") or ""
            on, off = "ON FARM" in sd, "OFF FARM" in sd
            if (kind == "on" and on) or (kind == "off" and off) or (kind == "total" and not on and not off):
                v = _num(d.get("Value"))
                if v is not None:
                    vals.append(v)
        return max(vals) if vals else None

    return {y: {k: pick(y, k) for k in ("total", "on", "off")} for y in (year, year - 1)}


def build_data(period, year):
    cfg = PERIODS[period]
    ref = cfg["ref"]
    consensus = _consensus(year)
    stk, dis = [], []
    for code, label in STOCKS_CROPS:
        cur = _positions(code, ref, year)
        c, p = cur.get(year, {}), cur.get(year - 1, {})
        total, total_p = c.get("total"), p.get("total")
        on, off = c.get("on"), c.get("off")
        on_yoy, off_yoy = _pct(on, p.get("on")), _pct(off, p.get("off"))
        vs_yr = _pct(total, total_p)
        exp = consensus.get(code, {}).get("stocks")
        vs_exp = _pct(total, exp * 1e9) if (exp and total) else None
        sref = vs_exp if vs_exp is not None else vs_yr
        sd = (-1 if sref > 1 else 1 if sref < -1 else 0) if sref is not None else 0
        note = ""
        if on_yoy is not None and off_yoy is not None and on_yoy - off_yoy > 5:
            note = " · farmers holding (on-farm heavy)"
        stk.append({"crop": label, "total": f"{total/1e9:.2f}" if total else "—", "vs_yr": _fmt_pct(vs_yr),
                    "vs_exp": _fmt_pct(vs_exp), "on": f"{on/1e9:.2f}" if on else "—", "off": f"{off/1e9:.2f}" if off else "—",
                    "cls": "short" if sd < 0 else "long" if sd > 0 else "flat",
                    "read": ("Bearish — ample" if sd < 0 else "Bullish — tighter" if sd > 0 else "≈ in line") + note})
        if cfg["use_valid"]:
            pp = _positions(code, cfg["prior_ref"], year + cfg["prior_off"])
            prior_cur = pp.get(year + cfg["prior_off"], {}).get("total")
            prior_py = pp.get(year - 1 + cfg["prior_off"], {}).get("total")
            dcur = (prior_cur - total) if (prior_cur and total) else None
            # Only a POSITIVE draw is genuine disappearance; a stock build means new-crop harvest
            # landed in the quarter (e.g. winter wheat in Jun–Aug), so "use" isn't meaningful — skip it.
            if dcur is not None and dcur > 0:
                dpri = (prior_py - total_p) if (prior_py and total_p) else None
                d_yoy = _pct(dcur, dpri)
                dd = 1 if (d_yoy or 0) > 1 else -1 if (d_yoy or 0) < -1 else 0
                dis.append({"crop": label, "use": f"{dcur/1e9:.2f}", "vs_yr": _fmt_pct(d_yoy),
                            "cls": "long" if dd > 0 else "short" if dd < 0 else "flat",
                            "read": "Bullish — stronger use" if dd > 0 else "Bearish — softer use" if dd < 0 else "≈ flat"})

    full = (period == "JUN")
    acre, pending = build_acreage(year) if full else ([], False)
    wclass = _wheat_classes(year) if full else []
    return {"period": period, "label": cfg["label"], "quarter": cfg["quarter"], "year": year, "full": full,
            "acre": acre, "wclass": wclass, "stk": stk, "dis": dis,
            "has_consensus": bool(consensus), "pending": pending}


def render_html(period, year, asof, demo=False, light=False):
    from reportkit import data_uri
    data = build_data(period, year)
    title = "USDA Acreage & Grain Stocks" if data["full"] else "USDA Grain Stocks"
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("usdareaction.html").render(
        asof=asof, title=title, demo=demo, logo=data_uri(ASSETS / "logo.png"),
        watermark="" if light else data_uri(ASSETS / "building.jpg"), **data)


def build_pdf(period, year, asof, out_path, demo=False, light=False):
    from reportkit import render_pdf
    return render_pdf(render_html(period, year, asof, demo, light), out_path)


def _resolve(period, year):
    if period and year:
        return period, year
    latest = None
    try:
        latest = agdata.latest_stocks_period()
    except Exception:
        latest = None
    if latest:
        return period or latest["key"], year or latest["year"]
    return period or "JUN", year or datetime.date.today().year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_pdf", nargs="?")
    ap.add_argument("--period", choices=list(PERIODS), default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--asof", default="")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", dest="json_out", default="",
                    help="dump the structured data to JSON (for the in-app preview) and exit")
    args = ap.parse_args()
    period, year = _resolve(args.period, args.year)
    asof = args.asof or f"{PERIODS[period]['label']} {year}"
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(build_data(period, year)), encoding="utf-8")
        print(f"Wrote {args.json_out}")
        return
    if not args.out_pdf:
        ap.error("out_pdf is required unless --json is given")
    build_pdf(period, year, asof, args.out_pdf, demo=args.demo)
    print(f"Wrote {args.out_pdf} ({period} {year})")


if __name__ == "__main__":
    main()
