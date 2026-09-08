"""Trade Idea — the desk's BLANK house template, and the layout builder behind it.

Every other report in this repo renders a computed view; this one deliberately renders
nothing but the XP layout (inset black sidebar, yellow stripe, title band, banners,
compliance disclaimer) so a colleague can put their OWN idea into it and send it on to
clients in house style.

What the builder decides (all of it arrives in one payload dict from the app):
  desk / sector   — FICC or Equities, and which sector the trade sits in. The sector rides
                    in the title band's sub-line and in the file name.
  headline        — the title-band h1, defaulted per desk.
  sections        — WHICH sections, in WHAT ORDER, from SECTIONS below. Three kinds:
                    `prose` (a writing box), `table` (blank house table, e.g. the trade
                    ideas to execute) and `chart` (a box that takes a dropped image).
  pages           — how much writing space to lay out: 1 or 2 A4 pages' worth.

Two artefacts off the same template (templates/tradeidea.html):
  • build_pdf()  — the blank PDF: what the finished page looks like.
  • build_html() — a self-contained, click-to-type copy the recipient opens in any
                   browser and prints back to PDF (Ctrl+P → Save as PDF). The toolbar,
                   dashed guides and grey prompts are screen-only, so what they print is
                   the same house page. The logo is inlined as a data URI, so the file
                   works as an email attachment with no network and no other files.

Standalone (the app calls this as a subprocess, like every other report script):
    python src/tradeidea.py out.pdf --html out.html [--payload layout.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Dual import: `python src/tradeidea.py …` (how the app runs it) AND `from src import tradeidea`
# (how the app reads the catalogue below). Put src/ on the path so `reportkit` resolves either way.
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jinja2 import Environment, FileSystemLoader                  # noqa: E402
from reportkit import pretty_date, data_uri, launch_chromium      # noqa: E402

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
LAYOUT_FILE = ROOT / "data" / "tradeidea_layout.json"

# ---------------------------------------------------------------------------
# The section catalogue. key -> (title, kind, weight, prompt/extra). `weight` is that
# section's SHARE of the page's spare vertical space (see _fit_boxes) — prose that carries
# the argument gets more room than a list of catalysts. Add a section here and it appears
# in the builder; nothing else needs touching.
# ---------------------------------------------------------------------------
SECTIONS: dict[str, dict] = {
    "trade": {
        "title": "The Trade", "kind": "prose", "weight": 1.0,
        "prompt": "The trade — what to buy or sell, which contract or expiry, size or ratio, "
                  "and the horizon.",
    },
    "context": {
        "title": "Market Context", "kind": "prose", "weight": 1.0,
        "prompt": "Where this market has come from — the move to date, the range it is in, and "
                  "what has changed recently.",
    },
    "why": {
        "title": "Rationale", "kind": "prose", "weight": 1.6,
        "prompt": "Why now — the macro or relative-value argument, the data or positioning "
                  "behind it, and what the market is currently pricing.",
    },
    "risk": {
        "title": "Risks & Invalidation", "kind": "prose", "weight": 1.1,
        "prompt": "What would make this wrong — the events, levels or data that invalidate "
                  "the argument.",
    },
    "catalysts": {
        "title": "Catalysts & Calendar", "kind": "prose", "weight": 0.8,
        "prompt": "The dates that matter between now and the horizon — data, meetings, "
                  "expiries, inventory or supply reports.",
    },
    "position": {
        "title": "Positioning & Flow", "kind": "prose", "weight": 0.8,
        "prompt": "Who is in this trade already — positioning, open interest, option flow — "
                  "and what that does to the risk.",
    },
    "alts": {
        "title": "Alternative Expressions", "kind": "prose", "weight": 0.8,
        "prompt": "Other ways to hold the same view — options, spreads, a different tenor — "
                  "and why this one was chosen.",
    },
    "levels": {
        "title": "Trade Ideas to Execute", "kind": "table", "weight": 0.0,
        "cols": ["Instrument", "Direction", "Entry", "Target", "Stop", "Horizon"],
        "prompts": ["e.g. SR3Z6/H7", "Long / Short", "level", "level", "level", "e.g. 3 months"],
        "rows": 3,
    },
    "exec": {
        "title": "Execution & Levels", "kind": "prose", "weight": 1.0,
        "prompt": "Entry, target and stop; liquidity, roll or expiry considerations.",
    },
    "chart": {
        "title": "Chart", "kind": "chart", "weight": 2.2,
        "prompt": "Drop a chart image here, or click to choose one — you can also paste "
                  "straight from Bloomberg or Excel.",
    },
    "summary": {
        "title": "Summary & Next Steps", "kind": "prose", "weight": 0.8,
        "prompt": "The take-away in two or three lines, and what you want the client to do "
                  "with it.",
    },
}

# Column sets for a table section. "Custom" is whatever the builder types — a note about
# option pricing wants different headings from one about futures levels, and both are ordinary
# house tables underneath.
TABLE_PRESETS = {
    "Trade levels": ["Instrument", "Direction", "Entry", "Target", "Stop", "Horizon"],
    "Option pricing": ["Structure", "Expiry", "Strikes", "Price", "Breakeven", "Max risk"],
    "Calendar": ["Date", "Event", "Why it matters"],
}

# The classic one-pager, and what the builder opens on.
DEFAULT_SECTIONS = ["trade", "why", "risk", "levels"]

HEADLINES = {"FICC": "Global Macro Trade Idea", "Equities": "Equity Trade Idea"}
MARKET_PROMPT = {"FICC": "Market / instrument", "Equities": "Company / index"}

# A4 content box in CSS px at 96dpi, less the 0.30in top/bottom @page margins.
PAGE_PX = 11.69 * 96 - 2 * 0.30 * 96
MIN_BOX_IN = 0.55          # a writing box never gets less than this, however many you pick


def default_payload(desk: str = "FICC", sector: str = "") -> dict:
    return {
        "desk": desk, "sector": sector, "headline": HEADLINES.get(desk, HEADLINES["FICC"]),
        "sections": list(DEFAULT_SECTIONS), "oneline": True, "subject_bar": True, "pages": 1,
        "rows": {}, "asof": None,
    }


def load_layout() -> dict | None:
    """The layout saved as the desk's default from the builder, or None."""
    try:
        return json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_layout(payload: dict) -> None:
    LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAYOUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def file_stem(payload: dict) -> str:
    """`XP_US_10y_Seasonality_Energy_TEMPLATE` — built from the HEADLINE, not a fixed name: the
    piece may be about anything, and the writer already typed what it is."""
    words = "".join(c if c.isalnum() else " " for c in
                    (payload.get("headline") or "Trade Idea")).split()
    sector = "".join(c if c.isalnum() else " " for c in (payload.get("sector") or "")).split()
    return "_".join(["XP"] + words[:8] + sector[:3] + ["TEMPLATE"])


def _sections(payload: dict) -> list[dict]:
    """Resolve the builder's section list into what the template renders.

    A section arrives either as a catalogue KEY (the quick-add presets, and every layout saved
    before the builder learned to name its own sections) or as a full INSTANCE dict — the same
    fields, but with a title, columns and prompt the writer chose. Instances are what let one
    template serve a note about anything: a section is just a titled box of a given kind, and
    the same kind can appear as many times as the piece needs.
    """
    out = []
    legacy_rows = payload.get("rows") or {}
    for i, item in enumerate(payload.get("sections") or DEFAULT_SECTIONS):
        if isinstance(item, str):                       # catalogue key (or an older layout)
            spec = SECTIONS.get(item)
            if not spec:
                continue                                # a key we no longer ship — skip, don't crash
            s = dict(spec, id=item)
            if s["kind"] == "table":
                s["rows"] = int(legacy_rows.get(item, spec.get("rows", 3)))
        else:
            s = dict(item)
        kind = s.get("kind", "prose")
        # `eid` is what the markup keys every box, cell and saved draft on. It must be unique
        # per section AND stable across rebuilds (the fillable copy's autosave is keyed on it),
        # so it comes from the instance id the builder minted, not from the position.
        s["eid"] = "".join(c if c.isalnum() else "_" for c in str(s.get("id") or f"s{i}"))
        s["kind"] = kind
        s.setdefault("title", "Section")
        s.setdefault("prompt", "")
        s.setdefault("weight", 2.2 if kind == "chart" else (0.0 if kind == "table" else 1.0))
        if kind == "table":
            s["cols"] = [c for c in (s.get("cols") or TABLE_PRESETS["Trade levels"]) if str(c).strip()]
            s["prompts"] = list(s.get("prompts") or [])[:len(s["cols"])]
            s["prompts"] += [""] * (len(s["cols"]) - len(s["prompts"]))
            s["rows"] = max(1, min(20, int(s.get("rows", 3))))
        if kind == "chart":
            s.setdefault("caption", "Source: Bloomberg")
        out.append(s)
    return out


def render_html(payload: dict | None = None, editable: bool = False) -> str:
    """The template's HTML, boxes at their floor height. `editable=True` gives the fillable
    browser copy. _fit_boxes() then bakes the measured heights over the top."""
    p = {**default_payload(), **(payload or {})}
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    secs = _sections(p)
    return env.get_template("tradeidea.html").render(
        editable=editable,
        asof=pretty_date(p.get("asof") or date.today()),
        headline=p.get("headline") or HEADLINES["FICC"],
        sector=p.get("sector") or "",
        oneline=bool(p.get("oneline", True)),
        subject_bar=bool(p.get("subject_bar", True)),
        market_ph=MARKET_PROMPT.get(p.get("desk"), MARKET_PROMPT["FICC"]),
        sections=secs,
        has_chart=any(s["kind"] == "chart" for s in secs),
        logo=data_uri(ASSETS / "logo.png"),
    )


# Measured in the browser rather than guessed: collapse every writing box, see how much of the
# page the furniture (title band, banners, tables, disclaimer) actually eats, then hand the
# leftover out in proportion to the sections' weights. This is what keeps ANY mix of sections
# filling the sheet — the alternative, hard-coded heights, only ever suits one layout.
_FIT_JS = r"""
(args) => {
  const boxes = [...document.querySelectorAll('.fitme')];
  if (!boxes.length) return "";
  boxes.forEach(b => b.style.minHeight = '0px');
  const own = boxes.map(b => b.getBoundingClientRect().height);
  const used = document.querySelector('.content').getBoundingClientRect().height;
  const leftover = args.pagePx * args.pages - used;
  const weights = boxes.map(b => parseFloat(b.dataset.w || '1') || 1);
  const total = weights.reduce((a, b) => a + b, 0);
  return boxes.map((b, i) => {
    // ADD the share to what the box already occupies (its prompt line, or the drop-zone
    // caption): a min-height only takes effect above the content height, so handing a box
    // its share outright would swallow that much of the growth and leave the page short.
    const px = Math.max(args.minPx, own[i] + leftover * args.slack * weights[i] / total);
    return '#' + b.id + '{min-height:' + (px / 96).toFixed(3) + 'in}';
  }).join('\n');
}
"""


def _fit_boxes(page, html: str, pages: int, slack: float = 1.0) -> str:
    """Load `html`, measure it, and return it with per-section min-heights baked in."""
    page.set_content(html, wait_until="load")
    css = page.evaluate(_FIT_JS, {"pagePx": PAGE_PX, "pages": max(1, int(pages)),
                                  "minPx": MIN_BOX_IN * 96, "slack": slack})
    if not css:
        return html
    return html.replace("</head>", f"<style>\n{css}\n</style></head>", 1)


def build(payload: dict | None = None, out_pdf=None, out_html=None) -> tuple[str, str]:
    """Build both artefacts from one measuring pass. Returns (pdf_html, editable_html).

    The blank PDF is rendered from the same string that gets baked into the fillable copy, so
    what the recipient prints is what the PDF showed. After rendering, the page count is
    checked: if the blank template spilled past the requested page count (a long section
    title wrapping, a table with many rows), the boxes are re-fitted a touch tighter — once,
    which is enough, rather than left to run onto a stray page nobody wants.
    """
    from playwright.sync_api import sync_playwright
    p = {**default_payload(), **(payload or {})}
    pages = max(1, int(p.get("pages", 1)))
    blank_src, edit_src = render_html(p, editable=False), render_html(p, editable=True)
    with sync_playwright() as pw:
        browser = launch_chromium(pw)
        try:
            page = browser.new_page(viewport={"width": 794, "height": 1123})
            for slack in (1.0, 0.90, 0.80):
                blank = _fit_boxes(page, blank_src, pages, slack)
                if out_pdf is None:
                    break
                page.set_content(blank, wait_until="load")
                page.pdf(path=str(out_pdf), format="A4", print_background=True,
                         margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
                if _page_count(out_pdf) <= pages:
                    break
            edit = _fit_boxes(page, edit_src, pages, slack)
        finally:
            browser.close()
    if out_html is not None:
        Path(out_html).write_text(edit, encoding="utf-8")
    return blank, edit


def _page_count(pdf_path) -> int:
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return 1        # can't check — take the render as-is rather than looping


def build_pdf(out_path, payload: dict | None = None) -> str:
    build(payload, out_pdf=out_path)
    return str(out_path)


def build_html(out_path, payload: dict | None = None) -> str:
    build(payload, out_html=out_path)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Build the blank Trade Idea template.")
    ap.add_argument("out_pdf")
    ap.add_argument("--html", default=None, help="also write the fillable browser copy here")
    ap.add_argument("--payload", default=None, help="layout JSON from the builder")
    args = ap.parse_args()
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8")) if args.payload else None
    build(payload, out_pdf=args.out_pdf, out_html=args.html)
    print(f"Wrote {args.out_pdf}")
    if args.html:
        print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
