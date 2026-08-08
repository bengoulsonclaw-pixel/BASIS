"""Seasonality monitor — calendar patterns on ten years of deep history.

Classic desk seasonality on the deep price store: per-product year × month return
heatmaps, a month-by-month screener across the whole book, average-year seasonal
paths with the current year overlaid, an MRCI-style best-window screen, and
calendar-spread (1st−2nd) seasonality for the storage/harvest markets.

SERIES CONVENTIONS (the house rules, same as the TA book):
  • Non-FI products are measured in PERCENT per period, computed as the EXACT point
    move on the panama-adjusted continuation divided by the ACTUAL raw front level
    at the period start. Point moves are exact on the adjusted series and the raw
    denominator is a real traded level, so the % is well-defined across the full
    decade — including products whose adjusted continuation crosses zero deep in
    the past (NGA, CL post-2020), which get_ta's positive-tail trim would drop.
  • Fixed income runs in YIELD/RATE space (STIR = 100 − price, bonds = deep
    benchmark yield — get_ta again) and is measured in BASIS POINTS per period.
    A "+" month means the yield/rate ROSE — the FI pages' standing convention.
  • Calendar spreads are raw front − raw second (actual traded levels, curvemon
    convention): native points, ×100 = bp for STIRs. The level on a given calendar
    date references whatever contract pair is front then — that IS the seasonal
    curve-shape observable (NG winter, harvest carry, …).

Everything is descriptive — medians, hit rates and windows a decade of history
happened to show; nothing here is a forecast. No Streamlit — the page drives this.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import deepstore, universe

MIN_YEARS = 5          # a month/window needs at least this many observed years
WIN_MIN, WIN_MAX = 4, 16   # window-finder lengths, in weeks
HIT_STRONG = 0.70      # hit rate at/beyond this (or its mirror) flags a seasonal bias
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Display units for the classic calendar spreads (native points otherwise).
_SPREAD_UNITS = {"CLA Comdty": "$/bbl", "COA Comdty": "$/bbl", "NGA Comdty": "$/MMBtu",
                 "FJSA Comdty": "€/MWh", "HOA Comdty": "$/gal", "XBA Comdty": "$/gal",
                 "QSA Comdty": "$/t", "HGA Comdty": "¢/lb"}


def unit_of(tkr: str) -> str:
    """'bp' for FI (yield/rate space), '%' otherwise."""
    return "bp" if universe.is_fixed_income(tkr) else "%"


# ── page defaults (data/seasonality.json) ───────────────────────────────────
DEFAULTS_FILE = Path(__file__).resolve().parents[1] / "data" / "seasonality.json"


def default_sectors() -> list:
    """The saved startup sector selection for the Seasonality page. [] = all sectors —
    the file stores an explicit subset only, so a page that shows nothing at launch is
    impossible (the sector-filter all-off lesson)."""
    try:
        saved = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8")).get("sectors", [])
        return [s for s in saved if s in universe.ASSET_CLASSES]
    except Exception:
        return []


def save_default_sectors(sectors) -> None:
    """Persist the startup sector selection; pass [] (or everything) to reset to all."""
    DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULTS_FILE.write_text(json.dumps(
        {"sectors": [s for s in sectors if s in universe.ASSET_CLASSES]}, indent=2),
        encoding="utf-8")


def load_frames(tickers) -> dict:
    """The deep history the whole page runs on: FI in signal space (rates/yields, %),
    price products as full-depth adjusted continuation + raw actual levels."""
    tickers = list(tickers)
    fi = [t for t in tickers if universe.is_fixed_income(t)]
    px = [t for t in tickers if not universe.is_fixed_income(t)]
    return {"fi": deepstore.get_ta(fi), "adj": deepstore.get_adjusted(px),
            "raw": deepstore.get_raw(px)}


def _changes(frames: dict, freq: str) -> pd.DataFrame:
    """Per-period changes off period-end closes: FI = yield/rate diff in bp; price
    products = adjusted point move ÷ raw level at period start, in %."""
    out = {}
    fi = frames.get("fi", pd.DataFrame())
    if fi is not None and not fi.empty:
        ends = fi.resample(freq).last()
        for t in ends.columns:
            out[t] = ends[t].diff() * 100.0
    adj, raw = frames.get("adj", pd.DataFrame()), frames.get("raw", pd.DataFrame())
    if adj is not None and not adj.empty:
        adjE = adj.resample(freq).last()
        rawE = raw.resample(freq).last().reindex(adjE.index)
        for t in adjE.columns:
            den = rawE[t].shift() if t in rawE.columns else None
            if den is None:
                continue
            den = den.where(den > 0)          # a non-positive actual level is no denominator
            out[t] = adjE[t].diff() / den * 100.0
    return pd.DataFrame(out)


def monthly_changes(frames: dict) -> pd.DataFrame:
    """Month-end-to-month-end changes; the last row is the current month TO DATE."""
    return _changes(frames, "ME")


def weekly_changes(frames: dict) -> pd.DataFrame:
    """Friday-to-Friday changes (the seasonal-path / window-finder granularity)."""
    return _changes(frames, "W-FRI")


def _partial_stamp(monthly: pd.DataFrame):
    """(year, month) of the trailing month-to-date row — excluded from all stats."""
    if monthly is None or monthly.empty:
        return None
    last = monthly.index.max()
    today = pd.Timestamp.today()
    return (last.year, last.month) if (last.year, last.month) == (today.year, today.month) else None


# ── month screener ──────────────────────────────────────────────────────────
def screener(monthly: pd.DataFrame, month: int) -> pd.DataFrame:
    """One row per product: how `month` (1–12) behaved across the stored years.
    hit = share of complete years the month printed positive (yield rose, for FI)."""
    if monthly is None or monthly.empty:
        return pd.DataFrame()
    partial = _partial_stamp(monthly)
    rows = []
    sel = monthly[monthly.index.month == month]
    for t in monthly.columns:
        s = sel[t].dropna()
        if s.empty:
            continue
        this_year, this_partial = np.nan, False
        cur = s[s.index.year == pd.Timestamp.today().year]
        if not cur.empty:
            this_year = float(cur.iloc[-1])
            this_partial = partial is not None and partial[1] == month
            if this_partial:
                s = s.drop(cur.index)          # month-to-date never joins the stats
        if len(s) < MIN_YEARS:
            continue
        hit = float((s > 0).mean())
        rows.append({
            "ticker": t, "name": universe.yield_name(t), "asset": universe.asset(t),
            "unit": unit_of(t), "med": float(s.median()), "mean": float(s.mean()),
            "hit": hit, "n": int(len(s)),
            "best": float(s.max()), "worst": float(s.min()),
            "this_year": this_year, "this_partial": this_partial,
            "bias": "↑" if hit >= HIT_STRONG else ("↓" if hit <= 1 - HIT_STRONG else "—"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_a"] = df["asset"].map({a: i for i, a in enumerate(universe.ASSET_CLASSES)})
    df["_x"] = (df["hit"] - 0.5).abs()
    return (df.sort_values(["_a", "_x", "med"], key=lambda c: c.abs() if c.name == "med" else c,
                           ascending=[True, False, False])
              .drop(columns=["_a", "_x"]).reset_index(drop=True))


# ── per-product monthly heatmap ─────────────────────────────────────────────
def monthly_matrix(monthly: pd.DataFrame, ticker: str):
    """(matrix, stats, meta): matrix = years × 12 months of changes (the heatmap),
    stats = per-month median / hit / best / worst over COMPLETE months, meta =
    {unit, partial(y,m)|None, years}."""
    meta = {"unit": unit_of(ticker), "partial": None, "years": 0}
    if monthly is None or monthly.empty or ticker not in monthly.columns:
        return None, None, meta
    s = monthly[ticker].dropna()
    if s.empty:
        return None, None, meta
    mat = (pd.DataFrame({"year": s.index.year, "month": s.index.month, "val": s.to_numpy()})
             .pivot(index="year", columns="month", values="val")
             .reindex(columns=range(1, 13)))
    partial = _partial_stamp(monthly)
    if partial is not None and partial[0] in mat.index:
        meta["partial"] = partial
    comp = mat.copy()
    if meta["partial"]:
        comp.loc[meta["partial"][0], meta["partial"][1]] = np.nan
    stats = pd.DataFrame({
        "med": comp.median(), "hit": (comp > 0).sum() / comp.notna().sum(),
        "best": comp.max(), "worst": comp.min(), "n": comp.notna().sum(),
    }).reindex(range(1, 13))
    meta["years"] = int(len(mat))
    return mat, stats, meta


# ── average-year seasonal path ──────────────────────────────────────────────
def _iso_weekly(weekly_col: pd.Series) -> pd.DataFrame:
    """Weekly changes tagged (iso_year, woy) with week 53 folded into 52."""
    s = weekly_col.dropna()
    iso = s.index.isocalendar()
    return pd.DataFrame({"y": iso["year"].astype(int).to_numpy(),
                         "w": np.minimum(iso["week"].astype(int).to_numpy(), 52),
                         "chg": s.to_numpy()})


def _year_pivot(weekly_col: pd.Series) -> pd.DataFrame:
    """woy 1..52 × iso-year matrix of weekly changes (week 53 summed into 52)."""
    d = _iso_weekly(weekly_col)
    if d.empty:
        return pd.DataFrame()
    return (d.groupby(["w", "y"])["chg"].sum(min_count=1)
             .unstack("y").reindex(range(1, 53)))


def seasonal_path(weekly: pd.DataFrame, ticker: str, smooth: int = 3):
    """Cumulative average-year path: per iso-year cum change by week, then the
    median / 25–75% band across years, current year separate. Long frame matching
    cotseasonality.seasonal_long (woy, med, p25, p75, current, wdate) + info."""
    info = {"years": 0, "cur_year": None, "unit": unit_of(ticker)}
    if weekly is None or weekly.empty or ticker not in weekly.columns:
        return pd.DataFrame(), info
    piv = _year_pivot(weekly[ticker])
    if piv.empty:
        return pd.DataFrame(), info
    # cum path per year; a missing week contributes 0 but keeps NaN before the
    # year's first datum (partial first year of history starts where it starts)
    cum = piv.fillna(0.0).cumsum()
    cum[piv.ffill().isna()] = np.nan
    cur_year = int(piv.columns.max())
    years = [c for c in cum.columns if piv[c].notna().sum() >= 40]     # complete-ish years only
    info["years"] = len(years)
    info["cur_year"] = cur_year
    if len(years) < MIN_YEARS:
        return pd.DataFrame(), info
    prof = cum[years]
    out = pd.DataFrame({
        "woy": np.arange(1, 53),
        "med": prof.median(axis=1).to_numpy(),
        "p25": prof.quantile(0.25, axis=1).to_numpy(),
        "p75": prof.quantile(0.75, axis=1).to_numpy(),
        "current": (cum[cur_year] if cur_year in cum.columns else pd.Series(index=cum.index,
                                                                            dtype=float)).to_numpy(),
    })
    if smooth and smooth > 1:               # NO wrap — a cum path restarts at zero in January
        for c in ("med", "p25", "p75"):
            out[c] = out[c].rolling(smooth, center=True, min_periods=1).mean()
    out["wdate"] = pd.Timestamp("2001-01-01") + pd.to_timedelta((out["woy"] - 1) * 7, unit="D")
    return out, info


# ── best-window finder (MRCI-style) ─────────────────────────────────────────
def _wk_label(woy: int) -> str:
    d = pd.Timestamp("2001-01-01") + pd.Timedelta(days=(int(woy) - 1) * 7)
    return d.strftime("%d %b").lstrip("0")


def best_windows(weekly: pd.DataFrame, ticker: str, top: int = 4) -> pd.DataFrame:
    """Scan every start-week × 4–16-week window (year-end wrap included) for the
    calendar stretches this product moved one way most consistently. Windows must
    clear HIT_STRONG (or its mirror) over ≥ MIN_YEARS complete years; overlapping
    echoes of one window are collapsed to the strongest."""
    if weekly is None or weekly.empty or ticker not in weekly.columns:
        return pd.DataFrame()
    piv = _year_pivot(weekly[ticker])
    if piv.empty or piv.shape[1] < MIN_YEARS:
        return pd.DataFrame()
    yrs = sorted(piv.columns)
    nxt = piv.reindex(columns=[y + 1 for y in yrs])         # wrap: year y continues into y+1
    nxt.columns = yrs
    ext = pd.concat([piv, nxt.iloc[:WIN_MAX]], axis=0).T    # years × 68 weeks
    cands = []
    for s in range(1, 53):
        for L in range(WIN_MIN, WIN_MAX + 1):
            block = ext.iloc[:, s - 1:s - 1 + L]
            vals = block.sum(axis=1, min_count=max(2, L - 2)).dropna()
            if len(vals) < MIN_YEARS:
                continue
            hit = float((vals > 0).mean())
            if hit < HIT_STRONG and hit > 1 - HIT_STRONG:
                continue
            up = hit >= HIT_STRONG
            cands.append({
                "dir": "Higher" if up else "Lower", "start": s, "weeks": L,
                "hit": hit if up else 1 - hit, "wins": int((vals > 0).sum() if up else (vals < 0).sum()),
                "n": int(len(vals)), "med": float(vals.median()), "mean": float(vals.mean()),
                "worst": float(vals.min() if up else vals.max()),
                "label": f"{_wk_label(s)} → {_wk_label((s + L - 2) % 52 + 1)}",
            })
    if not cands:
        return pd.DataFrame()
    df = pd.DataFrame(cands).sort_values(["hit", "med"],
                                         key=lambda c: c.abs() if c.name == "med" else c,
                                         ascending=False)
    keep = []
    for _, r in df.iterrows():
        wk = {(r["start"] + i - 1) % 52 for i in range(int(r["weeks"]))}
        clash = any(r["dir"] == k["dir"] and
                    len(wk & k["_wk"]) / min(len(wk), len(k["_wk"])) >= 0.5 for k in keep)
        if clash:
            continue
        keep.append({**r, "_wk": wk})
        if sum(k["dir"] == "Higher" for k in keep) >= top and \
           sum(k["dir"] == "Lower" for k in keep) >= top:
            break
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "_wk"} for r in keep])
    out = pd.concat([out[out["dir"] == "Higher"].head(top),
                     out[out["dir"] == "Lower"].head(top)])
    return out.reset_index(drop=True)


# ── calendar-spread (1st − 2nd) seasonality ─────────────────────────────────
def spread_products() -> list:
    """Universe tickers with a stored '2' generic — the calendar-spread book —
    in universe (asset-class) order."""
    f2 = deepstore.get_front2(list(universe.INSTRUMENTS))
    have = set(getattr(f2, "columns", []))
    order = {a: i for i, a in enumerate(universe.ASSET_CLASSES)}
    return sorted([t for t in universe.INSTRUMENTS if t in have],
                  key=lambda t: (order.get(universe.asset(t), 99), universe.name(t)))


def spread_seasonal(ticker: str, smooth: int = 3):
    """Weekly 1st−2nd spread LEVEL profile across the stored years: median /
    25–75% band by week-of-year + the current year. Levels are continuous across
    the year end, so the smoothing wraps (Dec→Jan). Returns (long df, info)."""
    info = {"years": 0, "cur_year": None,
            "unit": "bp" if universe.is_stir(ticker) else _SPREAD_UNITS.get(ticker, "pts")}
    r1 = deepstore.get_raw([ticker])
    f2 = deepstore.get_front2([ticker])
    if r1.empty or f2.empty or ticker not in r1.columns or ticker not in f2.columns:
        return pd.DataFrame(), info
    sp = (r1[ticker] - f2[ticker]).dropna()
    if universe.is_stir(ticker):
        sp = sp * 100.0
    wk = sp.resample("W-FRI").last().dropna()
    iso = wk.index.isocalendar()
    d = pd.DataFrame({"y": iso["year"].astype(int).to_numpy(),
                      "w": np.minimum(iso["week"].astype(int).to_numpy(), 52),
                      "lvl": wk.to_numpy()})
    piv = d.groupby(["w", "y"])["lvl"].last().unstack("y").reindex(range(1, 53))
    cur_year = int(piv.columns.max())
    years = [c for c in piv.columns if piv[c].notna().sum() >= 40]
    info["years"] = len(years)
    info["cur_year"] = cur_year
    if len(years) < MIN_YEARS:
        return pd.DataFrame(), info
    prof = piv[years]
    out = pd.DataFrame({
        "woy": np.arange(1, 53),
        "med": prof.median(axis=1).to_numpy(),
        "p25": prof.quantile(0.25, axis=1).to_numpy(),
        "p75": prof.quantile(0.75, axis=1).to_numpy(),
        "current": (piv[cur_year] if cur_year in piv.columns
                    else pd.Series(index=piv.index, dtype=float)).to_numpy(),
    })
    if smooth and smooth > 1:               # levels ARE continuous over the year end — wrap
        for c in ("med", "p25", "p75"):
            s = out[c]
            wrapped = pd.concat([s, s, s]).rolling(smooth, center=True, min_periods=1).mean()
            out[c] = wrapped.iloc[len(s):2 * len(s)].to_numpy()
    out["wdate"] = pd.Timestamp("2001-01-01") + pd.to_timedelta((out["woy"] - 1) * 7, unit="D")
    return out, info
