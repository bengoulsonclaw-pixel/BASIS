"""Release calendar for the two oil-balance outlooks the report uses:
OPEC's Monthly Oil Market Report (MOMR) and EIA's Short-Term Energy Outlook (STEO).

Dates are official where published (release times below are in US Eastern):
  * EIA STEO — eia.gov release schedule (rule: first Tuesday after the first Thursday;
    ~12:00 noon ET). Confirmed through Dec 2027.
  * OPEC MOMR — opec.org release calendar; 2026 confirmed (~04:00 ET = 10:00 Vienna). OPEC
    publishes the next year's schedule around December, so 2027 dates are TBC until then.
  * IEA OMR — iea.org; 2026 confirmed (~04:00 ET = 10:00 Paris); 2027 TBC.
"""
from __future__ import annotations

from datetime import date

# (year, month) -> day of month
EIA_STEO = {
    (2026, 1): 13, (2026, 2): 10, (2026, 3): 10, (2026, 4): 7, (2026, 5): 12, (2026, 6): 9,
    (2026, 7): 7, (2026, 8): 11, (2026, 9): 9, (2026, 10): 6, (2026, 11): 10, (2026, 12): 8,
    (2027, 1): 12, (2027, 2): 9, (2027, 3): 9, (2027, 4): 6, (2027, 5): 11, (2027, 6): 8,
    (2027, 7): 7, (2027, 8): 10, (2027, 9): 8, (2027, 10): 13, (2027, 11): 9, (2027, 12): 7,
}
OPEC_MOMR = {   # 2026 confirmed (opec.org); 2027 TBC
    (2026, 1): 14, (2026, 2): 11, (2026, 3): 11, (2026, 4): 13, (2026, 5): 13, (2026, 6): 11,
    (2026, 7): 13, (2026, 8): 12, (2026, 9): 10, (2026, 10): 13, (2026, 11): 11, (2026, 12): 9,
}
IEA_OMR = {   # IEA Oil Market Report — 2026 confirmed (iea.org, ~04:00 ET = 10:00 Paris); 2027 TBC
    (2026, 1): 21, (2026, 2): 12, (2026, 3): 12, (2026, 4): 14, (2026, 5): 13, (2026, 6): 17,
    (2026, 7): 10, (2026, 8): 12, (2026, 9): 11, (2026, 10): 14, (2026, 11): 13, (2026, 12): 11,
}

_MONTH = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _date(y, m, tbl):
    d = tbl.get((y, m))
    return date(y, m, d) if d else None


def next_12_months(today: date) -> list[dict]:
    """One row per month for the 12 months starting this month: each with the OPEC and EIA
    release date (or None = TBC) and whether that release is still upcoming."""
    rows = []
    for i in range(12):
        y = today.year + (today.month - 1 + i) // 12
        m = (today.month - 1 + i) % 12 + 1
        opec, eia, iea = _date(y, m, OPEC_MOMR), _date(y, m, EIA_STEO), _date(y, m, IEA_OMR)
        rows.append({
            "label": f"{_MONTH[m]} {y}", "y": y, "m": m,
            "opec": opec, "opec_upcoming": bool(opec and opec >= today),
            "eia": eia, "eia_upcoming": bool(eia and eia >= today),
            "iea": iea, "iea_upcoming": bool(iea and iea >= today),
        })
    return rows


def next_release(today: date) -> dict | None:
    """The soonest upcoming release across all three reports."""
    cand = []
    for who, tbl in (("OPEC MOMR", OPEC_MOMR), ("EIA STEO", EIA_STEO), ("IEA OMR", IEA_OMR)):
        for (y, m), d in tbl.items():
            dt = date(y, m, d)
            if dt >= today:
                cand.append((who, dt))
    if not cand:
        return None
    who, dt = min(cand, key=lambda t: t[1])
    return {"who": who, "date": dt, "days": (dt - today).days}


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders — weekly (unlike the monthly outlooks above)
# ---------------------------------------------------------------------------
def cot_releases(start: date, end: date) -> list[dict]:
    """COT release dates in [start, end]. Normally each Friday (3:30pm ET, data as-of the
    prior Tuesday); when a US federal holiday falls in the release week (Tue as-of → Fri),
    the CFTC delays it to the next business day. This rule reproduces the CFTC's published
    2026 delayed dates exactly (Jan 5, Jun 22, Jul 6, Nov 16, Nov 30, Dec 28). Each item:
    {date, delayed, normal, asof} — `normal` is the un-shifted Friday, `asof` its Tuesday."""
    from datetime import timedelta
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hols = {pd.Timestamp(h).date() for h in USFederalHolidayCalendar().holidays(
        start=pd.Timestamp(start) - pd.Timedelta(days=10),
        end=pd.Timestamp(end) + pd.Timedelta(days=10))}
    out = []
    d = start + timedelta(days=(4 - start.weekday()) % 7)          # first Friday on/after start
    while d <= end:
        asof = d - timedelta(days=3)                              # the Tuesday the data is as-of
        if {d - timedelta(days=k) for k in range(4)} & hols:      # holiday in Tue–Fri of the week
            rel = d + timedelta(days=1)
            while rel.weekday() >= 5 or rel in hols:              # → next business day (usually Mon)
                rel += timedelta(days=1)
            out.append({"date": rel, "delayed": True, "normal": d, "asof": asof})
        else:
            out.append({"date": d, "delayed": False, "normal": d, "asof": asof})
        d += timedelta(days=7)
    return out
