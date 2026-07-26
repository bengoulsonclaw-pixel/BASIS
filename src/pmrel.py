"""pmrel.py — detect, fetch and parse the big precious-metals releases.

Covered publications (the OPEC-synopsis model applied to metals):
  wgc   World Gold Council "Gold Demand Trends" — quarterly, ~4-5 weeks after
        quarter end; article page at a stable URL pattern
        gold-demand-trends-q{N}-{YYYY}; figures quoted with attribution.
  wpic  WPIC "Platinum Quarterly" — quarterly, ~7 weeks after quarter end,
        pre-announced dates; PDF linked from the publication page.

Each build produces a hand-editable synopsis JSON in data/pm_releases/ (the
OPEC precedent: if a parse misses a number, edit the JSON and rebuild) which
src/pmrelreport.py renders to the one-page branded PDF.

Neither site bot-blocks plain clients (verified Jul 2026) — requests only.
"""
from __future__ import annotations

import html as htmllib
import json
import re
from datetime import date, datetime
from pathlib import Path

import requests

_MONTH_PAT = ("January|February|March|April|May|June|July|August|September|"
              "October|November|December")

ROOT = Path(__file__).resolve().parents[1]
REL_DIR = ROOT / "data" / "pm_releases"
REL_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) basis-pm-monitor"}

WGC_HUB = "https://www.gold.org/goldhub/research/gold-demand-trends"
WPIC_PAGE = "https://platinuminvestment.com/supply-and-demand/platinum-quarterly"

PUBS = {
    "wgc": {"label": "WGC Gold Demand Trends", "metal": "Gold",
            "publisher": "World Gold Council"},
    "wpic": {"label": "WPIC Platinum Quarterly", "metal": "Platinum",
             "publisher": "World Platinum Investment Council / Metals Focus"},
}


def _get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def _strip_tags(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", htmllib.unescape(txt))


def _pdf_text(pdf_path: Path, max_pages: int = 30) -> str:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    out = []
    for i, page in enumerate(pdf):
        if i >= max_pages:
            break
        out.append(page.get_textpage().get_text_range())
    return "\n".join(out)


_JUNK = re.compile(r"©|Page \d+ of|FOREWORD|Disclaimer|IMPORTANT NOTICE|www\.|"
                   r"all rights reserved|cookies", re.I)


def _num_sentences(text: str, keywords: list[str], cap: int = 6) -> list[str]:
    """Sentences that carry a keyword AND a number — the quotable highlights.
    Boilerplate (copyright, page furniture, disclaimers) is filtered out and
    PDF bullet glyphs are treated as sentence breaks."""
    text = text.replace("•", ". ")
    hits = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip().lstrip(".- ")
        if not (25 < len(s) < 320) or not re.search(r"\d", s) or _JUNK.search(s):
            continue
        if any(k in s.lower() for k in keywords):
            hits.append(re.sub(r"\s+", " ", s))
        if len(hits) >= cap:
            break
    return hits


# ---------------------------------------------------------------------------
# WGC Gold Demand Trends
# ---------------------------------------------------------------------------
def detect_wgc() -> dict | None:
    """Newest edition linked off the GDT hub page."""
    txt = _get(WGC_HUB).text
    eds = re.findall(r"gold-demand-trends-q(\d)-(\d{4})", txt)
    if not eds:
        return None
    q, y = max(((int(q), int(y)) for q, y in eds), key=lambda t: (t[1], t[0]))
    return {"pub": "wgc", "edition": f"Q{q} {y}",
            "url": f"{WGC_HUB}/gold-demand-trends-q{q}-{y}"}


def build_wgc(info: dict) -> dict:
    raw = _get(info["url"]).text
    # publication date from the page's JSON-LD ("datePublished":"2026-04-29")
    published = None
    m = re.search(r"datePublished[\"':\s]+(\d{4}-\d{2}-\d{2})", raw)
    if m:
        published = datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    text = _strip_tags(raw)
    heads = _num_sentences(text, ["demand", "central bank", "etf", "jewellery",
                                  "bar and coin", "supply", "investment"], cap=7)
    figures = []
    m = re.search(r"[Tt]otal (?:gold )?demand[^.]{0,80}?([\d,]+)\s?t", text)
    if m:
        figures.append({"label": "Total gold demand", "value": f"{m.group(1)}t"})
    m = re.search(r"[Cc]entral banks?[^.]{0,100}?([\d,]+)\s?t", text)
    if m:
        figures.append({"label": "Central-bank net buying", "value": f"{m.group(1)}t"})
    return {
        "pub": "wgc", "edition": info["edition"], "url": info["url"],
        "published": published,
        "label": PUBS["wgc"]["label"], "publisher": PUBS["wgc"]["publisher"],
        "headline": (heads[0] if heads else
                     f"The World Gold Council has published Gold Demand Trends {info['edition']}."),
        "bullets": heads[1:7], "figures": figures,
        "built": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# WPIC Platinum Quarterly
# ---------------------------------------------------------------------------
def detect_wpic() -> dict | None:
    txt = _get(WPIC_PAGE).text
    links = re.findall(r'href="([^"]*?/files/\d+/(WPIC_Platinum_Quarterly_Q(\d)_(\d{4}))[^"]*?\.pdf)"',
                       txt)
    if not links:
        return None
    url, _name, q, y = max(links, key=lambda t: (int(t[3]), int(t[2])))
    if url.startswith("/"):
        url = "https://platinuminvestment.com" + url
    return {"pub": "wpic", "edition": f"Q{int(q)} {y}", "url": url}


_WPIC_ROWS = [  # label in the PDF text -> synopsis label
    ("Total mining supply", "Mining supply"),
    ("Total recycling", "Recycling supply"),
    ("Total supply", "Total supply"),
    ("Total demand", "Total demand"),
    ("Balance", "Market balance"),
    ("Above ground stocks", "Above-ground stocks"),
]


def build_wpic(info: dict) -> dict:
    pdf_path = REL_DIR / f"WPIC_{info['edition'].replace(' ', '_')}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(_get(info["url"], timeout=120).content)
    text = _pdf_text(pdf_path)
    flat = re.sub(r"\s+", " ", text)

    # publication date: the foreword carries it ("18th May 2026") — first
    # day-month-year on the opening pages
    published = None
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_PAT + r")\s+(20\d\d)",
                  flat[:6000])
    if m:
        published = f"{int(m.group(1))} {m.group(2)[:3]} {m.group(3)}"

    # Key-figure table: each label's trailing numbers are year columns; the last
    # one is the forecast year. Best effort — a miss just drops the row (the
    # JSON stays hand-editable, like the OPEC synopsis).
    figures = []
    for pat, label in _WPIC_ROWS:
        m = re.search(re.escape(pat) + r"((?:\s+-?[\d,]{2,7}){2,8})", flat)
        if m:
            nums = m.group(1).split()
            figures.append({"label": f"{label} (latest forecast, koz)",
                            "value": f"{nums[-1]}"})
    heads = _num_sentences(text.replace("\n", " "),
                           ["deficit", "surplus", "demand", "supply", "investment",
                            "automotive", "above ground"], cap=7)
    return {
        "pub": "wpic", "edition": info["edition"], "url": info["url"],
        "published": published,
        "label": PUBS["wpic"]["label"], "publisher": PUBS["wpic"]["publisher"],
        "headline": (heads[0] if heads else
                     f"WPIC has published the Platinum Quarterly for {info['edition']}."),
        "bullets": heads[1:7], "figures": figures,
        "built": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def detect(pub: str) -> dict | None:
    return {"wgc": detect_wgc, "wpic": detect_wpic}[pub]()


def build(pub: str, info: dict | None = None) -> dict:
    info = info or detect(pub)
    if not info:
        raise RuntimeError(f"could not detect the latest {pub} edition")
    d = {"wgc": build_wgc, "wpic": build_wpic}[pub](info)
    out = REL_DIR / f"{pub}_{d['edition'].replace(' ', '_')}_synopsis.json"
    out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    d["json_path"] = str(out)
    return d


if __name__ == "__main__":
    import sys
    for pub in (sys.argv[1:] or ["wgc", "wpic"]):
        d = build(pub)
        print(f"{pub}: {d['edition']} — {len(d['bullets'])} bullets, "
              f"{len(d['figures'])} figures -> {d['json_path']}")
