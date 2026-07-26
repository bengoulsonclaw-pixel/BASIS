"""WASDE reaction note — the monthly USDA World Agricultural Supply & Demand
Estimates, on the XP house style.

Built from USDA FAS PS&D, which carries the same balance sheets WASDE publishes and
is refreshed with each release (free, no key). Per crop (corn, soybeans, wheat):

  • The US balance sheet — Production, Total use, Exports, Ending stocks (million
    bushels) and Stocks-to-use — vs a year ago.
  • World ending stocks (MMT) and world stocks-to-use, vs a year ago.
  • A headline US stocks-to-use bar, tight -> loose.

Lower / falling ending stocks and stocks-to-use = tighter balance = bullish.

Run:  python src/wasdereport.py out.pdf            (latest marketing year)
      python src/wasdereport.py out.pdf --year 2025 --demo
      python src/wasdereport.py --json out.json    (structured data for the in-app preview)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import pandas as pd

import agdata   # reuse the PS&D bulk downloader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"

# crop -> (PS&D group, PS&D commodity, metric-ton -> bushel factor for the US table)
CROPS = [
    ("Corn", "grains_pulses", "Corn", 39.3679),
    ("Soybeans", "oilseeds", "Oilseed, Soybean", 36.7437),
    ("Wheat", "grains_pulses", "Wheat", 36.7437),
]
TIGHT, AMPLE = 12.5, 30.0        # US stocks-to-use % thresholds for the tight/ample read
SNAP_DIR = Path(__file__).parent.parent / "data" / "wasde_snapshots"            # monthly balance-sheet vintages (MoM)
CONSENSUS_FILE = Path(__file__).parent.parent / "data" / "wasde_consensus.json"  # pre-report trade estimates


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _pct(a, b):
    return (a / b - 1.0) * 100.0 if (a and b) else None


def _pivot(df, commodity, us_only):
    d = df[df["Commodity_Description"] == commodity]
    if us_only:
        d = d[d["Country_Name"] == "United States"]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index="Market_Year", columns="Attribute_Description", values="Value", aggfunc="sum")


def _g(piv, yr, attr):
    try:
        v = piv.loc[yr, attr]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _read(vs_yr):
    if vs_yr is None:
        return "flat", "—"
    if vs_yr < -1:
        return "long", "Bullish — tighter"
    if vs_yr > 1:
        return "short", "Bearish — looser"
    return "flat", "≈ in line"


def _wasde_month():
    """YYYY-MM of the most recent past WASDE — the release the current PS&D belongs to."""
    try:
        cal = agdata.report_calendar()
        past = cal[(cal["report"] == "WASDE") & (cal["date"] <= pd.Timestamp.now().normalize())]
        if not past.empty:
            return past["date"].max().strftime("%Y-%m")
    except Exception:
        pass
    return datetime.date.today().strftime("%Y-%m")


def _save_snapshot(month, snap):
    """Persist this release's US balance sheet per crop, so next month can diff against it."""
    try:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        (SNAP_DIR / f"{month}.json").write_text(json.dumps(snap), encoding="utf-8")
    except Exception:
        pass


def _prior_snapshot(month):
    """The most recent stored snapshot strictly before `month` (YYYY-MM); {} if none yet."""
    try:
        earlier = sorted(p.stem for p in SNAP_DIR.glob("*.json") if p.stem < month)
        if earlier:
            return json.loads((SNAP_DIR / f"{earlier[-1]}.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _consensus(month):
    """Pre-report trade estimates for `month`: {crop: {avg, low, high}} (US ending stocks, mln bu)."""
    try:
        return json.loads(CONSENSUS_FILE.read_text(encoding="utf-8")).get(month, {})
    except Exception:
        return {}


def _reax_row(crop, end, prior_end, cons):
    """One revision-&-surprise row: ending stocks vs last month + actual vs the trade estimate."""
    mom = (end - prior_end) if (end is not None and prior_end is not None) else None
    avg, lo, hi = cons.get("avg"), cons.get("low"), cons.get("high")
    surp = (end - avg) if (end is not None and avg is not None) else None
    oor = bool(surp is not None and lo is not None and hi is not None and (end > hi or end < lo))
    mom_cls = ("long" if mom < -1 else "short" if mom > 1 else "flat") if mom is not None else "flat"
    surp_cls = ("long" if surp < -1 else "short" if surp > 1 else "flat") if surp is not None else "flat"
    dom_cls = surp_cls if surp is not None else mom_cls
    parts = []
    if mom is not None and abs(mom) > 1:
        parts.append("tighter month-on-month" if mom < 0 else "looser month-on-month")
    if surp is not None and abs(surp) > 1:
        parts.append(("bullish" if surp < 0 else "bearish") + " vs trade" + (" (outside range)" if oor else ""))
    read = "; ".join(parts) if parts else "in line"
    return {"crop": crop, "end": f"{end:,.0f}" if end is not None else "—",
            "mom": (f"{mom:+,.0f}" if mom is not None else "—"), "mom_cls": mom_cls,
            "avg": (f"{avg:,.0f}" if avg is not None else "—"),
            "range": (f"{lo:,.0f}–{hi:,.0f}" if (lo is not None and hi is not None) else "—"),
            "surp": (f"{surp:+,.0f}" if surp is not None else "—"), "surp_cls": surp_cls,
            "oor": oor, "dom_cls": dom_cls, "read": read[0].upper() + read[1:]}


def build_data(year=None):
    groups = {}
    for _, grp, _, _ in CROPS:
        groups.setdefault(grp, agdata._psd_download(grp))

    us_rows, world_rows, bar, raw_us = [], [], [], []
    my_used = None
    for crop, grp, comm, fac in CROPS:
        usp = _pivot(groups[grp], comm, True)
        wp = _pivot(groups[grp], comm, False)
        if usp.empty:
            continue
        # current marketing year = latest MY with a US ending-stocks projection
        cand = [y for y in sorted(usp.index) if _g(usp, y, "Ending Stocks") is not None]
        if not cand:
            continue
        my = year if year else int(cand[-1])
        my_used = my

        def bu(v):
            return v * fac / 1000.0 if v is not None else None      # 1000 MT -> million bushels

        prod, dom = bu(_g(usp, my, "Production")), bu(_g(usp, my, "Domestic Consumption"))
        exp, end = bu(_g(usp, my, "Exports")), bu(_g(usp, my, "Ending Stocks"))
        end_p = bu(_g(usp, my - 1, "Ending Stocks"))
        use = (dom + exp) if (dom is not None and exp is not None) else None
        stu = (end / use * 100.0) if (end and use) else None
        vs_yr = _pct(end, end_p)
        cls, rd = _read(vs_yr)
        us_rows.append({
            "crop": crop,
            "prod": f"{prod:,.0f}" if prod else "—", "use": f"{use:,.0f}" if use else "—",
            "exp": f"{exp:,.0f}" if exp else "—", "end": f"{end:,.0f}" if end else "—",
            "stu": f"{stu:.1f}%" if stu else "—", "vs_yr": (f"{vs_yr:+.1f}%" if vs_yr is not None else "—"),
            "cls": cls, "read": rd, "_stu": stu if stu else 0.0})
        raw_us.append({"crop": crop, "end": end, "prod": prod, "use": use, "exp": exp})

        wend, wend_p = _g(wp, my, "Ending Stocks"), _g(wp, my - 1, "Ending Stocks")
        wdom = _g(wp, my, "Domestic Consumption")
        wstu = (wend / wdom * 100.0) if (wend and wdom) else None
        wvs = _pct(wend, wend_p)
        wcls, wrd = _read(wvs)
        world_rows.append({
            "crop": crop, "end": f"{wend/1000:,.1f}" if wend else "—",
            "stu": f"{wstu:.1f}%" if wstu else "—", "vs_yr": (f"{wvs:+.1f}%" if wvs is not None else "—"),
            "cls": wcls, "read": wrd})

        if stu:
            bar.append({"crop": crop, "stu": round(stu, 1)})

    month = _wasde_month()
    prior = _prior_snapshot(month)
    cons = _consensus(month)
    reax = [_reax_row(r["crop"], r["end"], (prior.get(r["crop"]) or {}).get("end"),
                      cons.get(r["crop"], {})) for r in raw_us]
    reax_on = any(r["mom"] != "—" or r["surp"] != "—" for r in reax)
    _save_snapshot(month, {r["crop"]: {"end": r["end"], "prod": r["prod"],
                                       "use": r["use"], "exp": r["exp"]} for r in raw_us})
    return {"my": my_used, "my_label": (f"{my_used}/{str(my_used+1)[2:]}" if my_used else "—"),
            "us": us_rows, "world": world_rows, "bar": bar, "reax": reax, "reax_on": reax_on}


def data_is_fresh() -> bool:
    """Has PS&D refreshed for a NEW WASDE? True if current US ending stocks differ from the most
    recent stored snapshot (the release is reflected); False if they still match it (not yet posted).
    No prior snapshot -> True (nothing to gate on). Lets the auto-send fire the moment the data posts
    instead of on a fixed clock time."""
    month = _wasde_month()
    prior = _prior_snapshot(month)
    if not prior:
        return True
    groups = {}
    for _, grp, _, _ in CROPS:
        groups.setdefault(grp, agdata._psd_download(grp))
    for crop, grp, comm, fac in CROPS:
        usp = _pivot(groups[grp], comm, True)
        if usp.empty:
            continue
        cand = [y for y in sorted(usp.index) if _g(usp, y, "Ending Stocks") is not None]
        if not cand:
            continue
        end = _g(usp, int(cand[-1]), "Ending Stocks")
        prior_end = (prior.get(crop) or {}).get("end")
        if end is not None and prior_end is not None and round(end * fac / 1000.0) != round(prior_end):
            return True
    return False


def stu_fig(bar):
    from reportkit import pretty_date, png
    import matplotlib.pyplot as plt
    bar = sorted(bar, key=lambda r: r["stu"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.1, max(1.1, 0.35 * len(bar) + 0.4)))
    colors = ["#2E7D32" if r["stu"] <= TIGHT else "#C62828" if r["stu"] >= AMPLE else "#9E9E9E" for r in bar]
    ax.barh(range(len(bar)), [r["stu"] for r in bar], color=colors, edgecolor="white", zorder=3)
    ax.set_yticks(range(len(bar)))
    ax.set_yticklabels([r["crop"] for r in bar], fontsize=7)
    for i, r in enumerate(bar):
        ax.text(r["stu"] + 0.3, i, f'{r["stu"]:.1f}%', va="center", fontsize=6.2, color="#333")
    ax.axvline(TIGHT, ls="--", lw=0.7, color="#2E7D32", zorder=2)
    ax.axvline(AMPLE, ls="--", lw=0.7, color="#C62828", zorder=2)
    ax.set_xlim(0, max(AMPLE + 8, max((r["stu"] for r in bar), default=0) * 1.2))
    ax.set_xlabel("US stocks-to-use %  (lower = tighter = supportive)", fontsize=6.3)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, axis="x", color="#EEE", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return png(fig)


def render_html(year, asof, demo=False, light=False):
    from reportkit import data_uri
    data = build_data(year)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("wasdereport.html").render(
        asof=pretty_date(asof), demo=demo, stu_img=(stu_fig(data["bar"]) if data["bar"] else ""),
        logo=data_uri(ASSETS / "logo.png"),
        watermark="" if light else data_uri(ASSETS / "building.jpg"), **data)


def build_pdf(year, asof, out_path, demo=False, light=False):
    from reportkit import render_pdf
    return render_pdf(render_html(year, asof, demo, light), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_pdf", nargs="?")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--asof", default="")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--check-fresh", action="store_true",
                    help="print FRESH/STALE (has PS&D refreshed for a new WASDE?) and exit; exit 3 = stale")
    args = ap.parse_args()
    if args.check_fresh:
        ok = data_is_fresh()
        print("FRESH" if ok else "STALE")
        sys.exit(0 if ok else 3)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(build_data(args.year)), encoding="utf-8")
        print(f"Wrote {args.json_out}")
        return
    if not args.out_pdf:
        ap.error("out_pdf is required unless --json is given")
    asof = args.asof or datetime.date.today().strftime("%d %b %Y")
    build_pdf(args.year, asof, args.out_pdf, demo=args.demo)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
