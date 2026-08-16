"""Equities Earnings Calendar — turns the Company Fundamentals frame's EXPECTED_REPORT_DT
into month-grid events for repcal.month_html, the same Google-Calendar grid as the FICC
fundamental-reports calendar. Pure builder (no Streamlit) so it can be unit-tested;
app.py owns the filters and month navigation.

Earnings season puts 30-50 names on one day, so each day keeps only the biggest `CAP`
companies by market cap as chips and collapses the rest into one '⋯ N more' chip whose
hover-tooltip lists them. The GICS palette below is the dashboard's own in-app colour
code — like the futures sector colours, it stays out of client PDFs.
"""
from __future__ import annotations

import pandas as pd

# GICS sector -> chip colour (deep tones so the white chip text stays legible).
SECTOR_COLORS = {
    "Information Technology": "#1565C0",
    "Communication Services": "#6A1B9A",
    "Consumer Discretionary": "#E65100",
    "Consumer Staples": "#558B2F",
    "Financials": "#283593",
    "Health Care": "#C62828",
    "Industrials": "#37474F",
    "Energy": "#1B5E20",
    "Utilities": "#00695C",
    "Materials": "#795548",
    "Real Estate": "#AD1457",
}
OTHER_COLOR = "#616161"        # unclassified ("Other") + the overflow chip
CAP = 8                        # company chips per day before collapsing to '⋯ N more'


def _mc_str(mc: float) -> str:
    if mc != mc:
        return ""
    return f" · {mc / 1000:,.0f}bn" if mc >= 1000 else f" · {mc:,.0f}mm"


# Bloomberg ticker exchange code -> venue name (the code is the 2nd token of e.g.
# "III LN Equity"). Covers every venue in the current universe; unknown codes fall
# back to the raw code so a new market is visible rather than blank.
EXCH_NAME = {
    "UW": "NASDAQ", "UN": "NYSE", "UA": "NYSE American", "UR": "NYSE Arca", "US": "US",
    "LN": "London", "GY": "Xetra", "GR": "Xetra", "FP": "Paris", "NA": "Amsterdam",
    "BB": "Brussels", "IM": "Milan", "SM": "Madrid", "SW": "SIX Swiss", "SE": "SIX Swiss",
    "ID": "Dublin", "PL": "Lisbon", "HE": "Helsinki", "DC": "Copenhagen",
    "SS": "Stockholm", "NO": "Oslo", "AV": "Vienna", "CN": "Toronto", "CT": "Toronto",
}


def _exchange(ticker: str) -> str:
    parts = str(ticker).split()
    code = parts[1] if len(parts) > 1 else ""
    return EXCH_NAME.get(code, code)


def events(df: pd.DataFrame, cap: int = CAP) -> list:
    """Company-earnings events for repcal.month_html: {date, icon, label, color, auto, tip}.
    `df` is the Company Fundamentals frame (one row per company, EXPECTED_REPORT_DT +
    CRNCY_ADJ_MKT_CAP + name/ticker/sector/indices). Chip label = ticker root; the '⋯ N more'
    label starts with U+22EF so month_html's alphabetical chip sort keeps it last in the cell."""
    if df is None or df.empty or "EXPECTED_REPORT_DT" not in df.columns:
        return []
    d = df.copy()
    d["_dt"] = pd.to_datetime(d["EXPECTED_REPORT_DT"], errors="coerce")
    d = d[d["_dt"].notna()]
    if d.empty:
        return []
    d["_mc"] = pd.to_numeric(d.get("CRNCY_ADJ_MKT_CAP"), errors="coerce")
    ev = []
    for day, grp in d.groupby(d["_dt"].dt.date):
        grp = grp.sort_values("_mc", ascending=False, na_position="last")
        for _, r in grp.head(cap).iterrows():
            tick = str(r.get("ticker", "")).split()[0] or "?"
            # `sub` = the plain-text detail the landing day-board writes beside the
            # chip (full name · index · venue) — month cells ignore it (tooltip only).
            _bits = [str(r.get("name", tick))]
            if r.get("indices"):
                _bits.append(str(r["indices"]))
            _x = _exchange(r.get("ticker", ""))
            if _x:
                _bits.append(_x)
            ev.append({"date": day, "icon": "", "label": tick,
                       "color": SECTOR_COLORS.get(r.get("sector"), OTHER_COLOR), "auto": False,
                       "bbg": str(r.get("ticker", "")),      # full ticker — the landing
                       "sub": " · ".join(_bits),             # board's Yahoo time lookup
                       "tip": f"{r.get('name', tick)} — {r.get('sector', '—')} · "
                              f"{r.get('indices', '')} · {_exchange(r.get('ticker', ''))}"
                              f"{_mc_str(r['_mc'])}"})
        rest = grp.iloc[cap:]
        if len(rest):
            names = ", ".join(str(x) for x in rest["name"].head(40))
            ev.append({"date": day, "icon": "", "label": f"⋯ {len(rest)} more",
                       "color": OTHER_COLOR, "auto": False,
                       "tip": names + ("…" if len(rest) > 40 else "")})
    return ev


def legend_html() -> str:
    """Sector colour-dot legend rendered under the grid."""
    dots = "".join(
        f'<span style="display:inline-block;margin:0 12px 4px 0;font-size:11.5px;'
        f'opacity:.85;white-space:nowrap;">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:5px;'
        f'background:{c};margin-right:5px;vertical-align:-1px;"></span>{s}</span>'
        for s, c in SECTOR_COLORS.items())
    return f'<div style="margin-top:6px">{dots}</div>'
