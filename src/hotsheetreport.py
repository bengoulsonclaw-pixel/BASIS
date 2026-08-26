"""BASIS Hot Sheet — the client-facing daily PDF of the desk's cross-asset highlights.

The print cut of the 🔥 Hot Sheet page: everything the desk's modules are ALREADY
flagging today, ranked on the sheet's common 0-100 heat scale. The report adds no
analysis of its own — every line earned its place by clearing its home module's own
live threshold, and the engine reads the SAME persisted sheet the page reads
(hotsheet.cached_collection(): the morning stamp's file, with the stale-cache
fallback the cache itself handles) — never a live Bloomberg-adjacent recompute.

The client cut, applied here exactly as the page applies it:
- meta-book items (data-health caveats) and internal_only lines are dropped;
- the Home sector filter applies to FICC items that carry a ticker
  (universe.filter_active() / enabled_tickers(), the render_hotsheet rule);
- FLOW (unusual option activity) stays OUT of the ranked top strip — it is an
  activity flag, not a signal (Ben, 2026-08-23) — and prints as its own section.

Layout mirrors the page: "Top of the sheet" (top 10 by heat, at most 2 lines per
module tag — the page's diversity rule), then the remaining items grouped by
section, FICC before Equities, sections ordered by their hottest line. Each line:
tag chip, NEW / day-n badge, the item's own prose, its sparkline as inline SVG,
the headline metric over the thin heat gauge (the weekreview bar convention).

Run standalone (the app calls report engines as subprocesses):
    python src/hotsheetreport.py data/Hot_Sheet.pdf [--no-ai]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from reportkit import pretty_date, data_uri, render_pdf, md_bold, join_names

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

TOP_N = 10                  # the page's top strip: first ten by heat…
TOP_PER_TAG = 2             # …with at most two lines per module tag (diversity rule)
FLOW_TAG = "FLOW"           # activity, not signal — never in the ranked strip
FLOW_SECTION_LAST = True    # …and its section prints after the ranked FICC sections


# ---------------------------------------------------------------------------
# data — the persisted sheet, through the page's own client cut
# ---------------------------------------------------------------------------
def client_items() -> tuple[list, dict, float]:
    """(items, provider report, collected epoch) — the cached sheet with badges
    applied, meta/internal lines dropped and the Home sector filter applied to
    FICC items exactly as app.py render_hotsheet does."""
    from src import hotsheet, universe
    items, report, collected, _from_cache = hotsheet.cached_collection()
    hotsheet.apply_badges(items)
    items = [it for it in items
             if it.get("book") != "meta" and not it.get("internal_only")]
    if universe.filter_active():                 # the Home sector filter applies here too
        en = set(universe.enabled_tickers())
        items = [it for it in items
                 if it["book"] != "ficc" or not it.get("ticker") or it["ticker"] in en]
    items.sort(key=lambda it: -it["heat"])
    return items, report, collected


def top_of_sheet(items: list) -> list:
    """The page's top-strip loop verbatim: first TOP_N by heat, at most TOP_PER_TAG
    per module tag so one module's ties can't crowd the cross-desk read. FLOW is
    excluded — activity rows carry no direction and don't rank against signals."""
    top, per_tag = [], {}
    for it in items:
        if it["tag"] == FLOW_TAG:
            continue
        if per_tag.get(it["tag"], 0) >= TOP_PER_TAG:
            continue
        per_tag[it["tag"]] = per_tag.get(it["tag"], 0) + 1
        top.append(it)
        if len(top) >= TOP_N:
            break
    return top


def book_groups(rest: list, book: str) -> list:
    """The remaining items of one book grouped by section, sections ordered by
    their hottest line; the Option Flow section (if any) prints last — on the page
    it deliberately sits outside the ranked sheet."""
    by_sect: dict = {}
    for it in rest:
        if it["book"] != book:
            continue
        by_sect.setdefault(it["section"], []).append(it)
    sects = sorted(by_sect, key=lambda s: -max(x["heat"] for x in by_sect[s]))
    if FLOW_SECTION_LAST:
        flow = [s for s in sects if any(x["tag"] == FLOW_TAG for x in by_sect[s])]
        sects = [s for s in sects if s not in flow] + flow
    return [{"title": s, "rows": by_sect[s],
             "flow": any(x["tag"] == FLOW_TAG for x in by_sect[s])} for s in sects]


# ---------------------------------------------------------------------------
# sparkline — the page's _hs_spark maths in print colours (light paper: quiet
# grey trace, the brand gold marking the latest point)
# ---------------------------------------------------------------------------
SPARK_LINE = "#A9AFB6"      # print-safe mid grey — quiet next to 7.7pt body copy
SPARK_DOT = "#F5C518"       # house gold (--yellow) endpoint…
SPARK_DOT_RING = "#C8901A"  # …ringed in the report's established amber so it holds on white


def spark_svg(vals: list) -> str:
    """Inline SVG sparkline of an item's own series (oldest→newest) — the same
    geometry as the page's _hs_spark, restyled for a light paper document."""
    if not vals or len(vals) < 2:
        return ""
    W, H, P = 110.0, 30.0, 2.5
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    xy = [(P + i * (W - 2 * P) / (n - 1), P + (H - 2 * P) * (1 - (v - lo) / rng))
          for i, v in enumerate(vals)]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    lx, ly = xy[-1]
    return (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" '
            f'style="display:block;width:100%;height:100%">'
            f'<polyline points="{pts}" fill="none" stroke="{SPARK_LINE}" '
            f'stroke-width="1.2" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.6" fill="{SPARK_DOT}" '
            f'stroke="{SPARK_DOT_RING}" stroke-width="0.6"/></svg>')


def decorate(items: list) -> None:
    """Annotate items in place with their render fields: escaped **bold** prose,
    the sparkline markup, the metric fallback (heat, the page's convention) and
    the gauge width."""
    for it in items:
        it["html"] = md_bold(it["text"])
        it["spark_svg"] = spark_svg(it.get("spark") or [])
        it["met"] = it.get("metric") or f"{it['heat']:.0f}"
        it["gw"] = int(max(4, min(100, round(it["heat"]))))


# ---------------------------------------------------------------------------
# intro — deterministic template, AI-polished into the desk voice by default
# (the weekreview chain: reportkit.ai_rewrite → Fable 5, template on ANY failure)
# ---------------------------------------------------------------------------
INTRO_SYSTEM = (
    "You are a senior futures strategist writing the opening paragraph of the desk's "
    "daily cross-asset Hot Sheet for professional clients. Rewrite the terse note you "
    "are given so it reads like a real person opening the morning note — flowing, "
    "plain-English, confident but relaxed, two to four sentences.\n"
    "HARD RULES: keep EVERY number and percentage EXACTLY as given, wrapped in the "
    "same **bold** markers; keep every product and module name; stay neutral and "
    "observational — these are screens the desk's models flagged, never advice: no "
    "buy/sell/recommend language and nothing that implies the reader should act.\n"
    "Return ONLY a JSON array with the single rewritten string.")


def intro_text(items: list, n_new: int, n_mods: int) -> str:
    n = len(items)
    n_ficc = sum(1 for it in items if it["book"] == "ficc")
    n_eq = sum(1 for it in items if it["book"] == "equities")
    parts = [f"The desk's screens carry **{n}** highlights across **{n_mods}** "
             f"modules this morning"
             + (f" — **{n_new}** of them new today." if n_new else ".")]
    if n_ficc and n_eq:
        parts.append(f"**{n_ficc}** lines sit on the FICC book and **{n_eq}** on Equities.")
    counts: dict = {}
    for it in items:
        counts[it["section"]] = counts.get(it["section"], 0) + 1
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        parts.append("The heaviest sections are "
                     + join_names([f"**{k}** ({v})" for k, v in top]) + ".")
    parts.append("Every line has already cleared its home module's own flagging bar — "
                 "quiet modules simply don't appear.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def render_html(ai_polish: bool = True) -> str:
    items, report, collected = client_items()
    collected_et = (pd.Timestamp(collected, unit="s", tz="UTC")
                    .tz_convert("America/New_York").strftime("%H:%M"))
    n_new = sum(1 for it in items if it.get("badge") == "NEW")
    n_mods = len({it.get("provider", it["tag"]) for it in items})

    intro = intro_text(items, n_new, n_mods)
    if ai_polish and items:
        from reportkit import ai_rewrite
        intro = ai_rewrite([intro], INTRO_SYSTEM)[0]

    top = top_of_sheet(items)
    top_ids = {id(it) for it in top}
    rest = [it for it in items if id(it) not in top_ids]
    books = []
    for label, book in (("FICC", "ficc"), ("Equities", "equities")):
        groups = book_groups(rest, book)
        n_rest = sum(len(g["rows"]) for g in groups)
        if groups:
            books.append({"label": label, "groups": groups, "n": n_rest})

    decorate(items)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("hotsheet_report.html").render(
        asof=pretty_date(str(date.today())),
        collected_et=collected_et,
        intro=md_bold(intro),
        top=top, books=books,
        n_items=len(items), n_mods=n_mods, n_new=n_new,
        n_ficc=sum(1 for it in items if it["book"] == "ficc"),
        n_eq=sum(1 for it in items if it["book"] == "equities"),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(out_path, ai_polish: bool = True) -> str:
    return render_pdf(render_html(ai_polish), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the AI polish of the intro (deterministic template only)")
    args = ap.parse_args()
    out = build_pdf(args.out, ai_polish=not args.no_ai)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
