"""Fundamental Reports Calendar — a month-grid (Google-Calendar style) of every
fundamental report on its release date, each with a product icon. Pure builder
(no Streamlit) so it can be unit-/render-tested; app.py renders the HTML it returns
and owns the month-navigation buttons.
"""
from __future__ import annotations

import calendar as _cmod
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import agdata, markethours, release_cal

# report -> (product icon, short label, pill colour)
USDA_ICON = {
    "WASDE": ("🌍", "WASDE", "#1565C0"),
    "Crop Production (Annual)": ("🌽", "Crop Prod.", "#2E7D32"),
    "Crop Production": ("🌽", "Crop Prod.", "#2E7D32"),
    "Grain Stocks": ("🌾", "Grain Stocks", "#2E7D32"),
    "Prospective Plantings": ("🌱", "Plantings", "#558B2F"),
    "Acreage": ("🚜", "Acreage", "#558B2F"),
    "Cattle on Feed": ("🐄", "Cattle/Feed", "#8D6E63"),
    "Hogs & Pigs": ("🐖", "Hogs & Pigs", "#8D6E63"),
}
RX = {"Grain Stocks", "Acreage"}          # USDA reports that auto-email a reaction note

CSS = """
<style>
  .rcal { display:grid; grid-template-columns:repeat(7,1fr); border:1px solid rgba(128,128,128,.28);
          border-radius:8px; overflow:hidden; margin-top:6px; }
  .rcal-dow { text-align:center; font-size:11px; font-weight:600; letter-spacing:.4px; text-transform:uppercase;
              color:rgba(128,128,128,.95); padding:7px 0 5px; border-bottom:1px solid rgba(128,128,128,.28); }
  .rcal-cell { min-height:106px; padding:4px 5px; border-right:1px solid rgba(128,128,128,.16);
               border-bottom:1px solid rgba(128,128,128,.16); }
  .rcal-cell:nth-child(7n) { border-right:0; }
  .rcal-out { opacity:.4; }
  .rcal-dnum { font-size:12px; font-weight:600; opacity:.85; padding:1px 3px; }
  .rcal-today .rcal-dnum { display:inline-block; min-width:22px; height:22px; line-height:22px; text-align:center;
               background:#F5C518; color:#111; border-radius:11px; opacity:1; font-weight:700; padding:0 5px; }
  .rcal-ev { display:block; font-size:10.5px; line-height:1.4; color:#fff; border-radius:5px; padding:1px 6px;
             margin:2px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
</style>
"""


def calendar_events() -> list:
    """All fundamental-report releases as events: {date, icon, label, color, auto, tip}."""
    ev = []
    try:
        _ag = markethours.ASSET_COLORS.get("Agriculture", "#EBC23A")   # USDA = agricultural complex
        for r in agdata.report_calendar().to_dict("records"):
            icon, lbl, _ = USDA_ICON.get(r["report"], ("📊", r["report"], None))
            ev.append({"date": pd.Timestamp(r["date"]).date(), "icon": icon, "label": lbl, "color": _ag,
                       "auto": r["report"] in RX, "tip": f"USDA {r['report']}"})
    except Exception:
        pass
    try:
        today = datetime.now(ZoneInfo("America/New_York")).date()
        _en = _darken(markethours.ASSET_COLORS.get("Energy", "#33A95E"), 0.62)   # deepen the vibrant Energy green for a legible chip
        for r in release_cal.next_12_months(today):
            for who, key in (("OPEC MOMR", "opec"), ("EIA STEO", "eia"), ("IEA OMR", "iea")):
                d = r.get(key)
                if d:
                    ev.append({"date": d, "icon": "🛢️", "label": who, "color": _en,
                               "auto": key == "opec", "tip": f"{who} (oil-balance outlook)"})
    except Exception:
        pass
    try:                                                # CFTC COT — weekly (Fri 3:30pm ET)
        today = datetime.now(ZoneInfo("America/New_York")).date()
        for r in release_cal.cot_releases(today.replace(month=1, day=1),
                                          today.replace(year=today.year + 1, month=12, day=31)):
            ev.append({"date": r["date"], "icon": "🧭", "label": "COT", "color": "#C62828",
                       "auto": True,
                       "tip": "CFTC Commitments of Traders (positioning) — Fri 3:30pm ET"})
    except Exception:
        pass
    return ev


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rel_luminance(hex_color: str) -> float:
    """WCAG relative luminance (0–1) of a #rrggbb colour."""
    h = hex_color.lstrip("#")

    def _lin(c):
        c = int(c, 16) / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(h[0:2]) + 0.7152 * _lin(h[2:4]) + 0.0722 * _lin(h[4:6])


def _darken(hex_color: str, f: float) -> str:
    """Scale a #rrggbb colour toward black by factor f (0–1) — deepens a bright sector accent
    (e.g. the vibrant Energy green) into a legible chip background for white text."""
    h = hex_color.lstrip("#")
    r, g, b = (max(0, min(255, int(int(h[i:i + 2], 16) * f))) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"


def _chip(e) -> str:
    # Pick black vs white text by whichever has more WCAG contrast on the chip colour
    # (crossover ≈ 0.179) — keeps ag-yellow AND energy-green labels legible, COT-red on white.
    light = _rel_luminance(e["color"]) > 0.179
    txt = "#111" if light else "#fff"
    star = "#6d4c00" if light else "#FFD54F"             # keep the auto-email ★ visible on both
    mk = f' <span style="color:{star}">&#9733;</span>' if e["auto"] else ""
    return (f'<span class="rcal-ev" style="background:{e["color"]};color:{txt}" title="{_esc(e["tip"])}">'
            f'{e["icon"]} {_esc(e["label"])}{mk}</span>')


def month_html(events, year, month, today) -> str:
    """Google-Calendar-style month grid (Sun-first) as an HTML string."""
    by_date: dict = {}
    for e in events:
        by_date.setdefault(e["date"], []).append(e)
    out = [CSS, '<div class="rcal">']
    for dow in ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"):
        out.append(f'<div class="rcal-dow">{dow}</div>')
    for d in _cmod.Calendar(firstweekday=6).itermonthdates(year, month):
        cls = "rcal-cell" + ("" if d.month == month else " rcal-out") + (" rcal-today" if d == today else "")
        daylabel = f"{d:%b} {d.day}" if d.day == 1 else str(d.day)
        chips = "".join(_chip(e) for e in sorted(by_date.get(d, []), key=lambda x: x["label"]))
        out.append(f'<div class="{cls}"><div class="rcal-dnum">{daylabel}</div>{chips}</div>')
    out.append('</div>')
    return "".join(out)
