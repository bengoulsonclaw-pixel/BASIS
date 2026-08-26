"""Product Correlations — engine.

Correlates the daily changes of the PRODUCTS inside (and across) sectors — gold
vs silver vs copper, Bund vs 10Y vs 5Y — over a LONG trailing window (the
pair's 'normal' relationship) and a SHORT one (the pair as it trades now).
SHORT − LONG is the regime-shift map: a strongly negative cell is a pair whose
usual co-movement has broken down. Changes are log returns for price and
vol-point diffs for the 1M ATM implied vol. Equal-weighted sector composites
exist here only to feed the diversification index (one line: is the whole book
one trade?).

Windows match the vol backtester's pair panel (volbt.correlation_stats): 21
sessions ('1M') and 252 ('1Y'), correlations of DAILY CHANGES so they read as
co-movement, not shared trend. The returns metric runs on TREND_UNIVERSE (the
book minus the vol-only cash indices, so Euro Stoxx / DAX aren't double-counted);
the IV metric runs on the full book minus stale surfaces (stale_iv_reasons), so
a frozen quote never reads as 'zero correlation'.

No Streamlit here — the page drives this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datafeed import get_history, get_implied_vol_history, stale_iv_reasons
from .universe import INSTRUMENTS, TREND_UNIVERSE, asset

SHORT, LONG = 21, 252                 # sessions: '1M' and '1Y'
LOOKBACK_DAYS = 580                   # calendar days: LONG + SHORT rolling obs + slack
SECTOR_ORDER = ["Indices", "STIRs", "Bonds", "FX",
                "Energy", "Metals", "Agriculture", "Softs"]


@dataclass
class CorrSet:
    """The three matrices behind one heatmap page: the long-window ('normal')
    correlation, the short-window ('now') one, their difference, and — where
    computed — the percentile of each short-window correlation inside a year of
    its own rolling short-window history (a 5th-percentile cell is a genuine
    breakdown; a −0.2 diff on a pair that swings ±0.4 every month is not)."""
    asof: pd.Timestamp
    labels: list                      # row/col order actually present
    long_: pd.DataFrame               # trailing LONG-session correlation
    short_: pd.DataFrame              # trailing SHORT-session correlation
    diff: pd.DataFrame                # short_ − long_ (diagonal blanked)
    pctl: pd.DataFrame | None         # 0–100 per pair, or None when skipped
    members: dict                     # label -> tickers behind it ({} at product level)
    dropped: list                     # tickers excluded (stale IV / not enough data)


def instrument_changes(metric: str, asof) -> tuple[pd.DataFrame, list]:
    """Daily changes per product up to `asof` — log returns ('returns'), 1M ATM IV
    vol-point diffs ('iv'), or 1M realized-vol changes ('realized') — plus the tickers
    excluded and why they'd mislead (stale surface, too little history)."""
    end = pd.Timestamp(asof)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    dropped: list = []
    if metric == "iv":
        raw = get_implied_vol_history(list(INSTRUMENTS), start=start, end=end)
        raw = raw.loc[:end]
        stale = stale_iv_reasons(raw)
        dropped += sorted(t for t in stale if t in raw.columns)
        chg = raw.drop(columns=dropped, errors="ignore").diff()
    elif metric == "realized":
        px = get_history(list(TREND_UNIVERSE), start=start, end=end).loc[:end]
        rv = np.log(px.where(px > 0)).diff().rolling(SHORT).std() * np.sqrt(252) * 100.0
        chg = rv.diff()                                  # daily changes in 1M realized vol
    else:
        px = get_history(list(TREND_UNIVERSE), start=start, end=end).loc[:end]
        chg = np.log(px.where(px > 0)).diff()
    thin = [c for c in chg.columns if chg[c].notna().sum() < SHORT + 5]
    dropped += thin
    return chg.drop(columns=thin), dropped


def _pair_pctl(chg: pd.DataFrame, short_: pd.DataFrame) -> pd.DataFrame:
    """Rolling-percentile matrix: where each pair's SHORT-window correlation sits
    in its own trailing year of rolling SHORT-window correlations (0–100)."""
    cols = list(chg.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            cur = short_.loc[a, b]
            if pd.isna(cur):
                continue
            roll = chg[a].rolling(SHORT).corr(chg[b]).dropna().iloc[-LONG:]
            if len(roll) >= 60:
                out.loc[a, b] = out.loc[b, a] = float((roll <= cur).mean() * 100.0)
    return out


def _corr_set(chg: pd.DataFrame, asof, *, with_pctl: bool,
              members: dict | None = None, dropped: list | None = None) -> CorrSet | None:
    chg = chg.dropna(how="all")
    if chg.shape[1] < 2 or len(chg) < SHORT + 5:
        return None
    long_ = chg.iloc[-LONG:].corr(min_periods=SHORT * 2)
    short_ = chg.iloc[-SHORT:].corr(min_periods=int(SHORT * 0.7))
    diff = (short_ - long_).mask(np.eye(len(long_), dtype=bool))
    return CorrSet(asof=pd.Timestamp(asof), labels=list(chg.columns),
                   long_=long_, short_=short_, diff=diff,
                   pctl=_pair_pctl(chg, short_) if with_pctl else None,
                   members=members or {}, dropped=dropped or [])


def sector_composites(metric: str, asof) -> tuple[pd.DataFrame, dict, list]:
    """Equal-weighted mean of each sector's members' daily changes (a sector
    composite, so one wild product doesn't masquerade as its whole sector).
    Sectors with fewer than two live members are left out."""
    chg, dropped = instrument_changes(metric, asof)
    comp, members = {}, {}
    for s in SECTOR_ORDER:
        cols = [t for t in chg.columns if asset(t) == s]
        if len(cols) >= 2:
            comp[s] = chg[cols].mean(axis=1)
            members[s] = cols
    return pd.DataFrame(comp), members, dropped


def diversification_index(metric: str, asof) -> pd.DataFrame | None:
    """The one-line 'is the whole book one trade?' series: the rolling
    SHORT-window correlation of every cross-sector composite pair, averaged.
    Columns: 'avg' (signed — regime convergence shows even when half the pairs
    are negative) and 'avg_abs' (co-movement regardless of sign — the actual
    loss of diversification). Last LONG observations."""
    comp, _, _ = sector_composites(metric, asof)
    if comp.shape[1] < 3:
        return None
    cols = list(comp.columns)
    rolls = [comp[a].rolling(SHORT).corr(comp[b])
             for i, a in enumerate(cols) for b in cols[i + 1:]]
    grid = pd.concat(rolls, axis=1)
    out = pd.DataFrame({"avg": grid.mean(axis=1),
                        "avg_abs": grid.abs().mean(axis=1)}).dropna().iloc[-LONG:]
    return out if len(out) >= 60 else None


MAX_PCTL_PRODUCTS = 30    # pctl needs a rolling series per pair — cap it


def instrument_corr(metric: str, asof, sectors: list) -> CorrSet | None:
    """Product-level CorrSet for the given sectors — gold vs silver vs copper,
    Bund vs 10Y vs 5Y, and the cross-sector detail when several are picked.
    Percentiles (a rolling series per pair) are computed up to
    MAX_PCTL_PRODUCTS products; past that the diff matrix carries the story."""
    chg, dropped = instrument_changes(metric, asof)
    cols = [t for t in chg.columns if asset(t) in set(sectors)]
    return _corr_set(chg[cols], asof, with_pctl=len(cols) <= MAX_PCTL_PRODUCTS,
                     dropped=dropped)


def top_breaks(metric: str, asof, n: int = 15) -> pd.DataFrame:
    """The n product pairs (whole book) whose SHORT-window correlation sits
    furthest from the LONG one — the names behind whatever lights up on the
    sector diff map. Percentile is computed for just these pairs."""
    chg, _ = instrument_changes(metric, asof)
    cs = _corr_set(chg, asof, with_pctl=False)
    if cs is None:
        return pd.DataFrame()
    pairs = [(a, b, float(cs.diff.loc[a, b]))
             for i, a in enumerate(cs.labels) for b in cs.labels[i + 1:]
             if pd.notna(cs.diff.loc[a, b])]
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    rows = []
    for a, b, d in pairs[:n]:
        cur = cs.short_.loc[a, b]
        roll = chg[a].rolling(SHORT).corr(chg[b]).dropna().iloc[-LONG:]
        rows.append({"a": a, "b": b, "sector_a": asset(a), "sector_b": asset(b),
                     "corr_1y": float(cs.long_.loc[a, b]), "corr_1m": float(cur),
                     "diff": d,
                     "pctl": float((roll <= cur).mean() * 100.0) if len(roll) >= 60 else np.nan})
    return pd.DataFrame(rows)


# ── alert floor: a pair needs a REAL standing relationship ──────────────────
# With |1Y| near zero there is nothing to break — a 21-session correlation
# excursion on an unrelated pair is sampling noise wearing a headline (the
# NG × Ethanol catch: -0.78 1M against +0.08 1Y, when the diff-ranked feed was
# 54-for-54 pairs with |1Y| < 0.18). The floor gates the whole alert feed at ONE
# chokepoint (percentile_extremes), so the page banner, the daily client email,
# the Weekly Review and the Hot Sheet all inherit it. Set on the Product
# Correlations page (admin), persisted to data/sectorcorr.json.
DEFAULT_MIN_BASE = 0.40
SETTINGS_FILE = Path(__file__).resolve().parents[1] / "data" / "sectorcorr.json"


def min_base() -> float:
    """The saved alert floor; DEFAULT_MIN_BASE on a missing/mangled file."""
    try:
        v = float(json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))["min_base"])
        return v if 0.0 <= v <= 0.95 else DEFAULT_MIN_BASE
    except Exception:
        return DEFAULT_MIN_BASE


def save_min_base(v: float) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"min_base": round(float(v), 2)}, indent=2),
                             encoding="utf-8")


def percentile_extremes(metric: str, asof, *, lo: float = 5.0, hi: float = 95.0,
                        min_move: float = 0.30, n: int = 60,
                        base_floor: float | None = None) -> pd.DataFrame:
    """Product pairs whose 1M correlation sits at an extreme of its own rolling
    1-year range AND has actually moved (|1M − 1Y| ≥ min_move) AND had a standing
    relationship to move from (|1Y| ≥ base_floor; None = the saved page setting)
    — the alert feed. Percentile alone fires on pairs whose correlation barely
    wobbles; the move floor keeps the list to breaks a desk would care about;
    the base floor keeps out noise excursions between unrelated products."""
    fl = min_base() if base_floor is None else float(base_floor)
    bt = top_breaks(metric, asof, n=n)
    if bt.empty:
        return bt
    hit = bt[(bt["corr_1y"].abs() >= fl)
             & (bt["diff"].abs() >= min_move)
             & ((bt["pctl"] <= lo) | (bt["pctl"] >= hi))].copy()
    hit["kind"] = np.where(hit["diff"] < 0, "breakdown", "lockstep")
    return hit.reset_index(drop=True)


RADAR_PAIRS = 4                       # Hot Sheet cut: the widest breaks, one line per pair


def radar_items() -> list:
    """Hot Sheet provider: today's correlation extremes — percentile_extremes' own
    feed (≤5th/≥95th percentile + move floor + the saved standing-relationship
    floor), deduped per unordered pair on the widest |diff|, top RADAR_PAIRS by
    |diff|. Heat is the extremeness within the pair's own rolling 1-year range."""
    from datetime import date

    from src import hotsheet
    from src.universe import name as _uname

    ex = percentile_extremes("realized", date.today())
    if ex is None or ex.empty:
        return []
    best: dict = {}                   # dedupe unordered pairs, keep the widest move
    for r in ex.to_dict("records"):
        k = tuple(sorted((r["a"], r["b"])))
        if k not in best or abs(r["diff"]) > abs(best[k]["diff"]):
            best[k] = r
    rows = sorted(best.values(), key=lambda r: -abs(r["diff"]))[:RADAR_PAIRS]
    items = []
    for r in rows:
        kind = "breaking down" if r["kind"] == "breakdown" else "moving in lockstep"
        a, b = (_uname(r["a"]) or r["a"]), (_uname(r["b"]) or r["b"])
        items.append(hotsheet.item(
            tag="CORR", key=f"{r['a']}|{r['b']}:{r['kind']}",
            section="Correlations",
            text=(f"**{a} × {b}** are {kind} — 1-month correlation "
                  f"**{r['corr_1m']:+.2f}** against {r['corr_1y']:+.2f} over the year, "
                  "an extreme of the pair's own range."),
            metric=f"Δρ {r['diff']:+.2f}", sub="1M vs 1Y",
            heat=hotsheet.heat_from_pctl(r["pctl"]),
            value=r["corr_1m"], ticker="",
            page="Product Correlations", book="ficc"))
    return items
