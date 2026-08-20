"""Fundamental Reports Calendar — a month-grid (Google-Calendar style) of every
fundamental report on its release date, each with a product icon. Pure builder
(no Streamlit) so it can be unit-/render-tested; app.py renders the HTML it returns
and owns the month-navigation buttons.
"""
from __future__ import annotations

import calendar as _cmod
import re as _re
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
              color:var(--basis-cal-ink, #8a929c); padding:7px 0 5px; border-bottom:1px solid rgba(128,128,128,.28); }
  .rcal-cell { min-height:106px; padding:4px 5px; border-right:1px solid rgba(128,128,128,.16);
               border-bottom:1px solid rgba(128,128,128,.16); }
  .rcal-cell:nth-child(7n) { border-right:0; }
  .rcal-out { opacity:.4; }
  .rcal-dnum { font-size:12px; font-weight:600; opacity:.85; padding:1px 3px; }
  .rcal-today .rcal-dnum { display:inline-block; min-width:22px; height:22px; line-height:22px; text-align:center;
               background:#F5C518; color:#111; border-radius:11px; opacity:1; font-weight:700; padding:0 5px; }
  .rcal-ev { display:block; font-size:10.5px; line-height:1.4; color:#fff; border-radius:5px; padding:1px 6px;
             margin:2px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  /* phones: seven 1fr tracks floor at the widest chip's min-content, so the month
     grid laid itself out ~665px wide and the .rcal border-radius clip simply ATE
     Thursday to Sunday. minmax(0,1fr) lets the tracks shrink; at ~55px a chip's
     text is unreadable anyway, so each event becomes a colour bar in its day
     (the label still reads on the day board, the week view and on tap-and-hold). */
  @media (max-width:760px) {
    .rcal { grid-template-columns:repeat(7,minmax(0,1fr)); }
    .rcal-cell { min-width:0; min-height:62px; padding:3px 3px 5px; }
    .rcal-dow { font-size:9px; letter-spacing:0; padding:5px 0 4px; }
    .rcal-dnum { font-size:10.5px; padding:0 2px; }
    .rcal-today .rcal-dnum { min-width:18px; height:18px; line-height:18px; }
    .rcal-ev { font-size:0; height:5px; padding:0; margin:2px 1px; border-radius:2px; }
    .rcal-ev span { display:none; }
  }
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
                       "auto": r["report"] in RX, "tip": f"USDA {r['report']} · 12:00 ET",
                       "time": "12:00 ET"})
    except Exception:
        pass
    try:
        today = datetime.now(ZoneInfo("America/New_York")).date()
        _en = _darken(markethours.ASSET_COLORS.get("Energy", "#33A95E"), 0.62)   # deepen the vibrant Energy green for a legible chip
        # release times per release_cal's header notes (ET; ~ = typical, not fixed)
        _oil_t = {"opec": "~04:00 ET", "eia": "~12:00 ET", "iea": "~04:00 ET"}
        for r in release_cal.next_12_months(today):
            for who, key in (("OPEC MOMR", "opec"), ("EIA STEO", "eia"), ("IEA OMR", "iea")):
                d = r.get(key)
                if d:
                    ev.append({"date": d, "icon": "🛢️", "label": who, "color": _en,
                               "auto": key == "opec",
                               "tip": f"{who} (oil-balance outlook) · {_oil_t[key]}",
                               "time": _oil_t[key]})
    except Exception:
        pass
    try:                                                # CFTC COT — weekly (Fri 3:30pm ET)
        today = datetime.now(ZoneInfo("America/New_York")).date()
        for r in release_cal.cot_releases(today.replace(month=1, day=1),
                                          today.replace(year=today.year + 1, month=12, day=31)):
            ev.append({"date": r["date"], "icon": "🧭", "label": "COT", "color": "#C62828",
                       "auto": True,
                       "tip": "CFTC Commitments of Traders (positioning) — Fri 3:30pm ET",
                       "time": "15:30 ET"})
    except Exception:
        pass
    try:                                                # central-bank rate decisions (STIR Paths
        from . import stirpaths                         # calendars — same source as the module)
        _CB = {"FED": ("🏛️", "FOMC", "#7FB3F5", "14:00 ET", "14:00 ET"),   # all in the meetings-blue
               "ECB": ("💶", "ECB", "#7FB3F5", "14:15 CET", "08:15 ET"),   # of the STIR Paths pages —
               "BOE": ("💷", "BoE MPC", "#7FB3F5", "12:00 London", "07:00 ET")}  # icons tell the banks apart
        for bk, bank in stirpaths.BANKS.items():
            icon, lbl, col, t, t_et = _CB[bk]
            for m in bank.meetings:
                ev.append({"date": m, "icon": icon, "label": lbl, "color": col,
                           "auto": False,
                           "tip": f"{bank.name} — {bank.meeting_name} rate decision · {t}",
                           "time": t_et})
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


# ── the landing page's trading-week strip ───────────────────────────────────────
# Mon–Fri columns, each day split into a FICC band (fundamental reports + CB rate
# decisions, i.e. calendar_events()) on top and an EQUITIES · EARNINGS band
# (eqearncal.events) below — Ben's requested separation, 2026-08-15.
WEEK_CSS = """
<style>
  .wkcal { display:grid; grid-template-columns:96px repeat(5,1fr); border:1px solid rgba(128,128,128,.28);
           border-radius:8px; overflow:hidden; margin-top:6px; }
  /* left gutter: the desk labels sit beside their band row (FICC top, EQUITIES
     bottom) instead of being repeated in every day cell */
  .wk-gut { border-right:1px solid rgba(128,128,128,.28); display:flex; flex-direction:column; }
  .wk-gutlbl { flex:1.12; display:flex; align-items:center; justify-content:center;
               font-size:12.5px; font-weight:700; letter-spacing:.26em;
               text-transform:uppercase; color:var(--basis-cal-ink-strong, #c6ccd4); }
  .wk-gutlbl.eq { flex:1; border-top:1px dashed rgba(128,128,128,.32); }
  /* fill the viewport below the hero + nav so the week IS the page, not a strip in
     the middle; the two flex sections pin the FICC/earnings divider to the SAME
     height in every column (per-column margin:auto made the split ragged). */
  .wk-col { border-right:1px solid rgba(128,128,128,.16); display:flex; flex-direction:column;
            min-height:max(500px, calc(100vh - 480px)); min-width:0; }
  .wk-col:last-child { border-right:0; }
  .wk-head { text-align:center; font-size:12px; font-weight:600; letter-spacing:.35px;
             text-transform:uppercase; color:var(--basis-cal-ink, #8a929c); padding:10px 0 8px;
             border-bottom:1px solid rgba(128,128,128,.28); }
  .wk-head .d { font-size:14px; font-weight:700; opacity:.95; margin-left:6px; }
  .wk-today .wk-head .d { background:#F5C518; color:#111; border-radius:12px; padding:1px 9px; }
  .wk-sec { display:flex; flex-direction:column; min-height:0; }
  .wk-sec.ficc { flex:1.12; }
  .wk-sec.eq { flex:1; border-top:1px dashed rgba(128,128,128,.32); }
  .wk-band { padding:8px 7px 9px; overflow:hidden; }
  .wkcal .rcal-ev { font-size:11.5px; line-height:1.55; padding:3px 8px; margin:3px 0;
                    border-radius:6px; }
  .wk-none { font-size:11px; opacity:.55; padding:2px 5px 4px; }
</style>
"""


# ── the desk-home day timeline (2026-08-20 redesign) ────────────────────────────
# One desk's events for one day as a vertical timeline: mono time (ET + local),
# a coloured left rule (gold = expiry, blue = print/decision), past rows dimmed,
# and a gold "now" line inserted at the current moment. Pure builder.
DESK_CSS = """
<style>
  .dkl-row { display:grid; grid-template-columns:96px 1fr; gap:12px; padding:9px 16px;
             border-bottom:1px solid rgba(128,128,128,.14); }
  .dkl-row.past { opacity:.5; }
  .dkl-t { text-align:right; font-family:var(--basis-mono, monospace); font-size:12.5px;
           font-variant-numeric:tabular-nums; color:var(--basis-cal-ink-strong, #c6ccd4);
           line-height:1.35; }
  .dkl-t .loc { display:block; font-size:10.5px; opacity:.75; }
  .dkl-b { border-left:2px solid #7FB3F5; padding-left:10px; }
  .dkl-b.exp { border-left-color:#F5C518; }
  .dkl-title { font-size:13.5px; color:var(--basis-cal-ink-strong, #e7eaee); }
  .dkl-det { font-size:11.5px; color:var(--basis-cal-ink, #8a929c); margin-top:2px; }
  .dkl-star { font-size:10px; background:rgba(245,197,24,.14); color:#F5C518;
              border-radius:4px; padding:1px 6px; font-weight:700; margin-left:6px; }
  .dkl-nowrow { display:grid; grid-template-columns:96px 1fr; gap:12px; align-items:center;
                padding:2px 16px; }
  .dkl-nowt { text-align:right; font-family:var(--basis-mono, monospace); font-size:10px;
              color:#F5C518; letter-spacing:.08em; }
  .dkl-now { height:1px; background:#F5C518; position:relative; }
  .dkl-now span { position:absolute; left:0; top:-2.5px; width:5px; height:5px;
                  border-radius:3px; background:#F5C518; }
  .dkl-none { font-size:12.5px; opacity:.55; padding:12px 16px; }
</style>
"""


def _ev_dt(e, day):
    """The event's aware datetime on `day`, or None when its time is unparseable."""
    m = _re.match(r"~?(\d{1,2}):(\d{2})\s+(.+)$", e.get("time") or "")
    if not m:
        return None
    tz = _TZMAP.get(m.group(3).strip())
    if not tz:
        return None
    try:
        return datetime(day.year, day.month, day.day, int(m.group(1)), int(m.group(2)),
                        tzinfo=ZoneInfo(tz))
    except Exception:
        return None


def desk_day(events, day) -> dict:
    """{html, total, ahead, next_txt} — the desk-home day timeline for `day`.
    Events use the calendar shape; kinds are inferred: labels containing 'expir'
    rule gold, everything else (prints, decisions, reports) rules blue."""
    evs = sorted((e for e in events if e["date"] == day), key=_time_key)
    now = datetime.now(ZoneInfo("America/New_York"))
    is_today = day == now.date()
    rows, ahead, next_dt = [], 0, None
    now_done = not is_today                      # only today carries a now-line
    for e in evs:
        dt = _ev_dt(e, day)
        past = bool(is_today and dt and dt < now)
        if dt and not past:
            ahead += 1
            if next_dt is None or dt < next_dt:
                next_dt = dt
        if not now_done and (dt is None or dt >= now):
            rows.append(f'<div class="dkl-nowrow"><span class="dkl-nowt">{now:%H:%M} ET</span>'
                        f'<div class="dkl-now"><span></span></div></div>')
            now_done = True
        kind = "exp" if "expir" in str(e.get("label", "")).lower() else ""
        star = '<span class="dkl-star">EMAILS DESK</span>' if e.get("auto") else ""
        det = _esc(e.get("sub") or e.get("tip") or "")
        rows.append(
            f'<div class="dkl-row{" past" if past else ""}">'
            f'<div class="dkl-t">{_tcell(e, day)}</div>'
            f'<div class="dkl-b {kind}"><div class="dkl-title">{e.get("icon", "")} '
            f'{_esc(e.get("label", ""))}{star}</div>'
            + (f'<div class="dkl-det">{det}</div>' if det else "") + '</div></div>')
    if not now_done:
        rows.append(f'<div class="dkl-nowrow"><span class="dkl-nowt">{now:%H:%M} ET</span>'
                    f'<div class="dkl-now"><span></span></div></div>')
    if not evs:
        rows = ['<div class="dkl-none">Nothing scheduled for this desk today.</div>']
    nxt = None
    if next_dt is not None:
        mins = int((next_dt - now).total_seconds() // 60)
        nxt = f"Next in {mins}m" if mins < 95 else f"Next in {mins / 60:.1f}h"
    return {"html": CSS + DESK_CSS + "".join(rows),
            "total": len(evs), "ahead": ahead, "next_txt": nxt}


# ── the landing page's TODAY board ──────────────────────────────────────────────
# One day, two desks side by side: FICC (reports + CB decisions, each with its
# release time, ET) on the LEFT, equities earnings on the RIGHT — Ben, 2026-08-15.
DAY_CSS = """
<style>
  .daycal { display:grid; grid-template-columns:1fr 1fr; border:1px solid rgba(128,128,128,.28);
            border-radius:8px; overflow:hidden; margin-top:6px; }
  .dy-panel { min-height:max(540px, calc(100vh - 420px)); min-width:0;
              border-right:1px solid rgba(128,128,128,.28); }
  .dy-panel:last-child { border-right:0; }
  .dy-head { text-align:center; font-size:13px; font-weight:700; letter-spacing:.26em;
             text-transform:uppercase; color:var(--basis-cal-ink-strong, #c6ccd4);
             padding:12px 0 10px; border-bottom:1px solid rgba(128,128,128,.28); }
  .dy-rows { padding:12px 14px; }
  .dy-row { display:flex; align-items:center; gap:12px; margin:8px 0; }
  .dy-t { font-family:var(--basis-mono, monospace); font-size:12px; line-height:1.35;
          color:var(--basis-cal-ink, #8a929c); min-width:96px; text-align:right; flex:0 0 auto; }
  .dy-tl { display:block; font-size:10.5px; opacity:.8; }
  .dy-row .rcal-ev { display:inline-block; margin:0; font-size:12px; line-height:1.6;
                     padding:4px 11px; border-radius:6px; flex:0 0 auto; }
  .dy-sub { font-size:12.5px; color:var(--basis-cal-ink, #8a929c); min-width:0;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .dy-none { font-size:12.5px; opacity:.55; padding:10px 4px; }
  .dy-next { display:block; font-size:12px; color:var(--basis-cal-ink, #8a929c);
             padding:2px 4px 0; }
  /* phones: two 50% panels left no room for a chip like "WTI Crude — options
     expiry", so the chips spilled across the divider and printed on top of the
     other panel's text. Stack the panels, let the chips wrap, and drop the
     540px min-height (an empty EARNINGS panel was half a screen of nothing). */
  @media (max-width:760px) {
    .daycal { grid-template-columns:1fr; }
    .dy-panel { min-height:0; border-right:0;
                border-bottom:1px solid rgba(128,128,128,.28); }
    .dy-panel:last-child { border-bottom:0; }
    .dy-rows { padding:8px 10px 12px; }
    .dy-row { align-items:flex-start; gap:8px; margin:7px 0; flex-wrap:wrap; }
    .dy-t { min-width:64px; font-size:11px; padding-top:3px; }
    .dy-tl { font-size:9.5px; }
    .dy-row .rcal-ev { flex:0 1 auto; min-width:0; font-size:11.5px; padding:3px 9px;
                       white-space:normal; }
    .dy-sub { flex:1 1 100%; padding-left:72px; white-space:normal; }
    .dy-head { font-size:11.5px; letter-spacing:.2em; padding:9px 0 8px; }
  }
</style>
"""


# zone label (as used in event `time` strings / expiries specs) -> IANA zone
_TZMAP = {"ET": "America/New_York", "CT": "America/Chicago", "CET": "Europe/Paris",
          "London": "Europe/London", "Brussels": "Europe/Brussels",
          "JST": "Asia/Tokyo", "KST": "Asia/Seoul", "AEST": "Australia/Sydney"}


def _tcell(e, day) -> str:
    """The time cell: the market's own time, plus a second line converted to THIS
    machine's clock (= the user's computer time — the server runs on the laptop).
    Skipped when both wall clocks agree; day-shifted conversions carry the weekday."""
    t = e.get("time")
    if not t:
        return "—"
    m = _re.match(r"(~?)(\d{1,2}):(\d{2})\s+(.+)$", t)
    if not m:
        return _esc(t)
    approx, hh, mi, zone = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).strip()
    tzname = _TZMAP.get(zone)
    if not tzname:
        return _esc(t)
    try:
        dt = datetime(day.year, day.month, day.day, hh, mi, tzinfo=ZoneInfo(tzname))
        loc = dt.astimezone()                       # system local tz, DST-correct per date
    except Exception:
        return _esc(t)
    if loc.utcoffset() == dt.utcoffset():
        return _esc(t)                              # same wall clock — no duplicate line
    suffix = "" if loc.date() == day else f" {loc:%a}"
    return _esc(t) + f'<span class="dy-tl">{approx}{loc:%H:%M} local{suffix}</span>'


def _time_key(e) -> tuple:
    """Sort key: timed events first (by ET clock time), untimed after, then label."""
    m = _re.match(r"~?(\d{1,2}):(\d{2})", e.get("time") or "")
    return ((0, int(m.group(1)) * 60 + int(m.group(2))) if m else (1, 0), e["label"])


def day_html(ficc_events, eq_events, day, next_ficc=None, next_eq=None) -> str:
    """The landing board: `day`'s FICC releases (timed, sorted by release time) on
    the left, its expected earnings on the right. Same event shape as everywhere.
    next_ficc/next_eq: optional 'Next: …' pointer shown when that panel is empty."""
    def _rows(evs, empty_msg, nxt):
        evs = sorted((e for e in evs if e["date"] == day), key=_time_key)
        if not evs:
            more = f'<span class="dy-next">{_esc(nxt)}</span>' if nxt else ""
            return f'<div class="dy-none">{empty_msg}{more}</div>'
        return "".join(
            f'<div class="dy-row"><span class="dy-t">{_tcell(e, day)}</span>'
            f'{_chip(e)}'
            + (f'<span class="dy-sub">{_esc(e["sub"])}</span>' if e.get("sub") else "")
            + '</div>' for e in evs)
    return (CSS + DAY_CSS + '<div class="daycal">'
            '<div class="dy-panel"><div class="dy-head">FICC</div><div class="dy-rows">'
            + _rows(ficc_events, "No scheduled releases.", next_ficc) + '</div></div>'
            '<div class="dy-panel"><div class="dy-head">Equities · Earnings</div><div class="dy-rows">'
            + _rows(eq_events, "No earnings expected.", next_eq) + '</div></div></div>')


def week_html(ficc_events, eq_events, week_start, today) -> str:
    """The landing strip: Mon–Fri of `week_start`'s week, FICC on top of each day,
    earnings beneath. Both event lists use the {date, icon, label, color, auto, tip}
    shape. Pure builder like month_html — app.py owns the week navigation."""
    fd, ed = {}, {}
    for e in ficc_events:
        fd.setdefault(e["date"], []).append(e)
    for e in eq_events:
        ed.setdefault(e["date"], []).append(e)
    out = [CSS, WEEK_CSS, '<div class="wkcal">',
           '<div class="wk-gut"><div class="wk-head"><span class="d">&nbsp;</span></div>'
           '<div class="wk-gutlbl">FICC</div>'
           '<div class="wk-gutlbl eq">Equities</div></div>']
    for i in range(5):
        d = week_start + pd.Timedelta(days=i)
        d = d.date() if hasattr(d, "date") else d
        cls = "wk-col" + (" wk-today" if d == today else "")
        f_chips = "".join(_chip(e) for e in sorted(fd.get(d, []), key=lambda x: x["label"]))
        e_chips = "".join(_chip(e) for e in sorted(ed.get(d, []), key=lambda x: x["label"]))
        out.append(
            f'<div class="{cls}">'
            f'<div class="wk-head">{d:%a}<span class="d">{d.day} {d:%b}</span></div>'
            f'<div class="wk-sec ficc">'
            f'<div class="wk-band">{f_chips or "<div class=wk-none>—</div>"}</div></div>'
            f'<div class="wk-sec eq">'
            f'<div class="wk-band">{e_chips or "<div class=wk-none>—</div>"}</div></div>'
            f'</div>')
    out.append('</div>')
    return "".join(out)
