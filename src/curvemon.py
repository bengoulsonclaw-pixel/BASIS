"""Curve / RV monitor — the desk's spread book on the deep price store.

A fixed book of curve and relative-value spreads — rate curves, cross-market yield
spreads, STIR calendars, energy time-spreads, metal ratios — each measured against
its own history: a rolling z-score (how stretched vs the recent regime) plus a
percentile over the FULL ~10-year deep store (how stretched vs everything the last
decade has shown). The 'A'-generic live feed could never do this — Treasuries hold
~6 months of 'A' history — so this module reads the deep store directly.

SERIES CONVENTIONS (each chosen so the level is a real market observable):
  • Curve / cross-market rate spreads run on BENCHMARK YIELDS (deepstore.get_yields,
    the same series the TA book charts) in bp. No futures-price inversion, no roll
    gaps, directly comparable across a decade.
  • STIR calendars are front − second PRICE spread × 100 = bp (price convention:
    positive = front priced above next = tightening priced OUT the front… falls out
    of 100 − rate). Every pair is two consecutive quarterlies, so the level is like-
    for-like across rolls and is ranked as is.
  • Energy/metal calendars are front − second in native points — backwardation
    positive, contango negative — but their HISTORY is ranked on annualised carry
    ((F1 − F2) / F1 × 12 / gap-months), rescaled to TODAY's pair: each past day reads
    "what that day's carry would make today's spread". The raw '1'−'2' series is not
    comparable across rolls — HG's active chain (H/K/N/U/Z) alternates 2- and 3-month
    gaps, so its level stepped at every roll and the z / percentile compared different
    pairs (2026-10-06: −2.2σ / 0th pctl raw, −1.05σ / 3rd on carry). Today's value IS
    the live spread; z and percentile are scale-free, so they are the carry's own.
    NG and TTF also subtract a point-in-time seasonal norm (median carry of the same
    front month in PRIOR years — 52% / 25% of their carry variance is the front month).
  • Cross-product diffs and ratios use RAW front levels (actual traded prices, not
    panama-adjusted — an adjusted level is continuation fiction and would distort a
    ratio). The legs each jump at their own rolls; the spread of two fronts is the
    standard curve-shape observable and is what a 10-year percentile should rank.

ROLLS. Any spread with a rolling raw/'2' leg also carries `roll_breaks` — True on each
session where a leg's contract (deep_contract.parquet) differs from the session before.
The LEVEL above is what z-scores and percentiles rank; a position, though, only earns
the move WITHIN a contract pair, so the fade backtest (rvtickets) chains daily changes
and zeroes them across a break — the panama rule applied to the spread itself, both
legs raw (never an adjusted leg against a raw one). A front contract whose month sits
implausibly far from the date (TTF's generic carries 'FJSU20' back to 2016) is unknown,
and a rolling spread drops those days rather than guess where its rolls were.

Entry/objective/invalidation are mechanical, not advice: objective = the rolling
mean the z-score reverts to; invalidation = the level at ±INVAL_SIGMA σ on the
stretched side (a further-stretch stop). Client prose stays neutral.

Fallback: with no deep store on disk (fresh clone pre-backfill), yields and raw
fronts fall back to the live feed's ~400-day window; calendar spreads need the
stored '2' generic and simply drop. Depth is reported per spread so the page can
show what history each number stands on.

No Streamlit here — the page and the PDF CLI drive this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import deepstore, universe
from .volbt import currency, money_symbol, point_value

REV = 8               # bump when the book/row schema changes — busts the page's st.cache_data
WINDOW = 252          # default rolling window (sessions) for the z-score
Z_THRESHOLD = 2.0     # |z| beyond this flags the spread as stretched
INVAL_SIGMA = 3.0     # invalidation level: mean ± this many rolling σ
MIN_OVERLAP = 60      # a spread needs at least this many joint sessions

GROUPS = ["Rates — Curve", "Rates — Cross-market", "STIR Calendars", "Energy", "Metals",
          "Softs"]

# ── the spread book ─────────────────────────────────────────────────────────
# legs: [(weight, kind, ticker)] summed after each series is built, except
# kind_of_spread == "ratio" which divides the first leg by the second.
# kinds: "yield" (benchmark %, deep), "raw" ('1' actual level), "front2" ('2' level).
# scale multiplies the combined spread (100 turns % / price-points into bp).
#
# The rate-curve group carries the FULL 2/5/10/30 ladder for each market (every
# pairwise spread, generated below); `bench` marks the pair that market's desk
# actually quotes — US trades 2s10s and 5s30s (the 5Y is the policy pivot vs the
# term-premium long end), the euro market quotes off the Bund so its long-end
# steepener is 10s30s (Bund–Buxl). Starred on the page and in the PDF.
_CURVE_LADDERS = [
    ("us", "US", [("2", "TUA Comdty"), ("5", "FVA Comdty"),
                  ("10", "TYA Comdty"), ("30", "USA Comdty")]),
    ("de", "Germany", [("2", "DUA Comdty"), ("5", "OEA Comdty"),
                       ("10", "RXA Comdty"), ("30", "UBA Comdty")]),
]
BENCHMARKS = {"us_2s10s", "us_5s30s", "de_2s10s", "de_10s30s"}


def _curve_specs() -> list:
    """Tenor-pair outer loop, market inner — so the table reads like for like:
    each tenor pair is a US / Germany / US−Germany-box triplet, short end down
    to the long end."""
    out = []
    tenors = ["2", "5", "10", "30"]
    (ak, aname, alad), (bk, bname, blad) = _CURVE_LADDERS[0], _CURVE_LADDERS[1]
    alad, blad = dict(alad), dict(blad)
    for i in range(len(tenors)):
        for j in range(i + 1, len(tenors)):
            ta, tb = tenors[i], tenors[j]
            for mkey, mname, ladder in _CURVE_LADDERS:
                lad = dict(ladder)
                fa, fb = lad[ta], lad[tb]
                key = f"{mkey}_{ta}s{tb}s"
                bench = key in BENCHMARKS
                out.append({
                    "key": key, "name": f"{mname} {ta}s{tb}s", "group": "Rates — Curve",
                    "unit": "bp", "dp": 1, "bench": bench, "mkt": mname,
                    "legs": [(1, "yield", fb), (-1, "yield", fa)], "scale": 100.0,
                    "desc": f"{mname} {tb}Y minus {ta}Y benchmark yield — how steep the curve "
                            f"is between those two points (futures legs "
                            f"{fb.split()[0][:-1]} / {fa.split()[0][:-1]})."
                            + (" The benchmark quote for this market's curve." if bench else ""),
                })
            out.append({
                "key": f"box_{ta}s{tb}s", "name": f"{aname} − {bname} {ta}s{tb}s box",
                "group": "Rates — Curve", "unit": "bp", "dp": 1,
                "legs": [(1, "yield", alad[tb]), (-1, "yield", alad[ta]),
                         (-1, "yield", blad[tb]), (1, "yield", blad[ta])], "scale": 100.0,
                "desc": f"{aname} {ta}s{tb}s minus {bname} {ta}s{tb}s — is the curve steeper "
                        f"here or there; four futures legs.",
            })
    return out


# Energy/metal calendars rank their history on annualised carry rescaled to today's pair
# (see SERIES CONVENTIONS); the seasonal gas pair also nets out its front month's norm.
_CARRY_NOTE = (" History is ranked on annualised carry, shown in today's-pair terms, so a "
               "roll to a pair with a different month gap doesn't read as a move.")
_SEAS_NOTE = (" History is ranked on annualised carry net of this front month's seasonal "
              "norm (prior years), shown in today's-pair terms.")

SPREADS = _curve_specs() + [
    # Rates — cross-market 10Y spreads, in bp
    {"key": "ust_bund",  "name": "10Y Treasury − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "TYA Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "US 10Y minus German 10Y benchmark yield — the transatlantic spread, TY vs RX."},
    {"key": "oat_bund",  "name": "10Y OAT − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "OATA Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "France over Germany 10Y — the euro-area sovereign risk barometer, OAT vs RX."},
    {"key": "gilt_bund", "name": "10Y Gilt − Bund", "group": "Rates — Cross-market", "unit": "bp", "dp": 1,
     "legs": [(1, "yield", "G A Comdty"), (-1, "yield", "RXA Comdty")], "scale": 100.0,
     "desc": "UK 10Y minus German 10Y benchmark yield — Gilt vs RX."},

    # STIR calendars — front − second price spread in bp
    {"key": "sfr_cal", "name": "3M SOFR cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "SFRA Comdty"), (-1, "front2", "SFRA Comdty")], "scale": 100.0,
     "desc": "Front SOFR quarterly minus the next — the pace of easing/tightening priced "
             "between the first two IMM dates (positive = lower front rate than next)."},
    {"key": "er_cal",  "name": "3M Euribor cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "ERA Comdty"), (-1, "front2", "ERA Comdty")], "scale": 100.0,
     "desc": "Front Euribor quarterly minus the next — the ECB path between the first two IMMs."},
    {"key": "sfi_cal", "name": "3M SONIA cal (1st−2nd)", "group": "STIR Calendars", "unit": "bp", "dp": 1,
     "legs": [(1, "raw", "SFIA Comdty"), (-1, "front2", "SFIA Comdty")], "scale": 100.0,
     "desc": "Front SONIA quarterly minus the next — the BoE path between the first two IMMs."},

    # Energy — time spreads (backwardation positive) + the flagship product spread
    {"key": "cl_cal", "name": "WTI M1−M2", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "CLA Comdty"), (-1, "front2", "CLA Comdty")], "scale": 1.0,
     "basis": "carry",
     "desc": "WTI prompt spread — positive = backwardation (tight prompt barrels)." + _CARRY_NOTE},
    {"key": "co_cal", "name": "Brent M1−M2", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "COA Comdty"), (-1, "front2", "COA Comdty")], "scale": 1.0,
     "basis": "carry",
     "desc": "Brent prompt spread — positive = backwardation." + _CARRY_NOTE},
    {"key": "ng_cal", "name": "Henry Hub M1−M2", "group": "Energy", "unit": "$/MMBtu", "dp": 3,
     "legs": [(1, "raw", "NGA Comdty"), (-1, "front2", "NGA Comdty")], "scale": 1.0,
     "basis": "carry_seasonal",
     "desc": "US nat-gas prompt spread — seasonal storage economics at the front." + _SEAS_NOTE},
    {"key": "ttf_cal", "name": "TTF M1−M2", "group": "Energy", "unit": "€/MWh", "dp": 2,
     "legs": [(1, "raw", "FJSA Comdty"), (-1, "front2", "FJSA Comdty")], "scale": 1.0,
     "basis": "carry_seasonal",
     "desc": "European gas prompt spread — the storage-injection signal." + _SEAS_NOTE},
    {"key": "wti_brent", "name": "WTI − Brent", "group": "Energy", "unit": "$/bbl", "dp": 2,
     "legs": [(1, "raw", "CLA Comdty"), (-1, "raw", "COA Comdty")], "scale": 1.0,
     "desc": "Front WTI minus front Brent — the Atlantic-basin arb, actual traded levels."},

    # Softs — the coffee "arb": NY arabica (¢/lb) minus London robusta (US$/t) put on the
    # same ¢/lb axis (1 t = 2,204.62 lb, so US$/t ÷ 22.0462 = ¢/lb). Point values differ
    # ($375/¢ vs $10/t), so _dollar_sigma correctly declines to quote a 1-lot $σ.
    {"key": "kc_rc", "name": "Arabica − Robusta arb", "group": "Softs", "unit": "¢/lb", "dp": 1,
     "legs": [(1, "raw", "KCA Comdty"), (-1 / 22.0462, "raw", "DFA Comdty")], "scale": 1.0,
     "desc": "ICE NY arabica front minus ICE London robusta front, both in US cents/lb — "
             "the coffee trade's \"arb\". Roasters switch blends on it, which is what "
             "pulls it back. Front months differ (KC H/K/N/U/Z, RC F/H/K/N/U/X), so roll "
             "days are excluded from the half-life."},

    # Metals — ratios on actual levels + the copper time spread
    {"key": "gc_si", "name": "Gold / Silver ratio", "group": "Metals", "unit": "×", "dp": 1,
     "kind_of_spread": "ratio",
     "legs": [(1, "raw", "GCA Comdty"), (1, "raw", "SIA Comdty")], "scale": 1.0,
     "desc": "Ounces of silver per ounce of gold, front futures — the precious relative-value dial."},
    {"key": "gc_pl", "name": "Gold / Platinum ratio", "group": "Metals", "unit": "×", "dp": 2,
     "kind_of_spread": "ratio",
     "legs": [(1, "raw", "GCA Comdty"), (1, "raw", "PLA Comdty")], "scale": 1.0,
     "desc": "Gold over platinum, front futures — precious vs industrial-precious."},
    # HG quotes in ¢/lb (659.10, not 6.59) — volbt's point value is per ¢-point ($250).
    {"key": "hg_cal", "name": "Copper M1−M2", "group": "Metals", "unit": "¢/lb", "dp": 2,
     "legs": [(1, "raw", "HGA Comdty"), (-1, "front2", "HGA Comdty")], "scale": 1.0,
     "basis": "carry",
     "desc": "COMEX copper prompt spread — positive = backwardation (tight metal)." + _CARRY_NOTE},
]

SPREAD_BY_KEY = {s["key"]: s for s in SPREADS}


# ── history assembly ────────────────────────────────────────────────────────
def _needed(specs) -> dict:
    """{kind: sorted tickers} across `specs`."""
    need: dict = {}
    for s in specs:
        for _, kind, tkr in s["legs"]:
            need.setdefault(kind, set()).add(tkr)
    return {k: sorted(v) for k, v in need.items()}


def load_history(specs=None) -> pd.DataFrame:
    """One wide frame holding every leg series the book needs, columns keyed
    '<kind>:<ticker>'. Deep store first; yields and raw fronts fall back to the
    live feed's window when the store has nothing (fresh clone), '2' generics
    are store-only."""
    need = _needed(specs or SPREADS)
    frames = []

    def _add(df: pd.DataFrame, kind: str):
        if df is not None and not df.empty:
            frames.append(df.rename(columns={c: f"{kind}:{c}" for c in df.columns}))

    if "yield" in need:
        y = deepstore.get_yields(need["yield"])
        missing = [t for t in need["yield"] if t not in getattr(y, "columns", [])]
        _add(y, "yield")
        if missing:
            from .datafeed import get_yield_history
            _add(get_yield_history(missing), "yield")
    if "raw" in need:
        r = deepstore.get_raw(need["raw"])
        missing = [t for t in need["raw"] if t not in getattr(r, "columns", [])]
        _add(r, "raw")
        if missing:
            from .datafeed import get_history
            _add(get_history(missing, raw=True), "raw")
    if "front2" in need:
        _add(deepstore.get_front2(need["front2"]), "front2")
    # the contract behind every rolling leg — which PAIR a calendar's level belongs to, and
    # where each roll falls (object-dtype columns; spreads only ever read their own legs)
    rolling = sorted({t for k in ("raw", "front2") for t in need.get(k, [])
                      if deepstore._has_chain(t)})
    if rolling:
        _add(deepstore.get_contracts(rolling), "contract")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()].sort_index()


# ── contracts, rolls and the carry basis ─────────────────────────────────────
# A front contract's month sits within this window of the date it is front on (Euribor
# is front ~3 months before its month; SOFR/SONIA 3M are named for the START of their
# reference quarter and stay front through it — SFRM6 is front 17 Jun → 15 Sep 2026).
# Outside it the string is not a real front — TTF's generic carries 'FJSU20' on every
# day from 2016 to Aug 2020 — and the day's contract is unknown.
_FRONT_MONTH_WINDOW = (-120, 200)       # days from the date to the contract month's 1st
SEAS_MIN_YEARS = 2                      # prior same-month instances a seasonal norm needs


def _rolling_tickers(spec: dict) -> list:
    """Tickers of the spread's legs that roll (raw '1' / '2' generics on a chain)."""
    return sorted({t for _, kind, t in spec["legs"]
                   if kind in ("raw", "front2") and deepstore._has_chain(t)})


def _front_runs(ct: pd.Series) -> pd.DataFrame:
    """Consecutive runs of one front contract: [start, end, contract, month] per run, with
    implausible strings (month outside _FRONT_MONTH_WINDOW of the run's dates) blanked."""
    from .rollboard import decode_contract
    c = ct.dropna().astype(str)
    if c.empty:
        return pd.DataFrame(columns=["start", "end", "contract", "month"])
    starts = c.index[c != c.shift()]
    rows = []
    for k, d0 in enumerate(starts):
        d1 = starts[k + 1] if k + 1 < len(starts) else c.index[-1] + pd.Timedelta(days=1)
        sym = c.loc[d0]
        month = decode_contract(sym, d0)
        last = c.index[(c.index >= d0) & (c.index < d1)][-1]
        lo, hi = _FRONT_MONTH_WINDOW
        ok = (month is not None and (month - d0).days <= hi and (month - last).days >= lo)
        rows.append({"start": d0, "end": last, "contract": sym if ok else None,
                     "month": month if ok else None})
    return pd.DataFrame(rows)


def _next_gap(runs: pd.DataFrame) -> list:
    """Months from each run's front contract to the '2' contract — the NEXT contract in the
    product's observed chain. The live run's successor isn't observed yet, so it comes from
    the cycle of front months seen over the last ~3 years (HG: H/K/N/U/Z → 2- or 3-month
    gaps; CL monthly → 1; STIRs quarterly → 3)."""
    # a blanked run's month is NaT in the frame, not None — and NaT.month is a NaN whose hash
    # is its id, so letting it into `recent` made sorted() pick NaN as the live gap at random
    months = [None if pd.isna(m) else m for m in runs["month"]]
    gaps: list = []
    recent = {m.month for m in months[-36:] if m is not None}
    for k, m in enumerate(months):
        if m is None:
            gaps.append(None)
            continue
        nxt = months[k + 1] if k + 1 < len(months) else None
        g = (nxt.year - m.year) * 12 + nxt.month - m.month if nxt is not None else None
        if g is None or not 0 < g <= 6:            # live run, or a hole in the chain
            ahead = sorted(((mm - m.month) % 12) or 12 for mm in recent)
            g = ahead[0] if ahead else None
        gaps.append(g)
    return gaps


def _leg_contracts(history: pd.DataFrame, tkr: str, index) -> pd.Series | None:
    """The validated front contract for `tkr` on each date of `index` (None = unknown), or
    None when the history carries no contract column for it (fresh clone / feed fallback)."""
    runs = _runs_for(history, tkr)
    if runs is None:
        return None
    pos = _run_pos(runs, index)
    vals = runs["contract"].to_numpy(dtype=object)
    return pd.Series([vals[p] if p >= 0 else None for p in pos], index=index, dtype=object)


def _runs_for(history: pd.DataFrame, tkr: str) -> pd.DataFrame | None:
    col = f"contract:{tkr}"
    if col not in history.columns or history[col].dropna().empty:
        return None
    return _front_runs(history[col])


def _run_pos(runs: pd.DataFrame, index) -> np.ndarray:
    """Position in `runs` of the run each date falls in (−1 = before the first run, or more
    than a week past the last stamp of its run — a hole long enough to hide a roll)."""
    idx = pd.DatetimeIndex(index)
    pos = runs["start"].searchsorted(idx, side="right") - 1
    pos = np.asarray(pos, dtype=int)
    ends = pd.DatetimeIndex(runs["end"])
    ok = pos >= 0
    stale = np.zeros(len(idx), dtype=bool)
    stale[ok] = (idx[ok] - ends[pos[ok]]).days > 7
    pos[stale] = -1
    return pos


def roll_breaks(spec: dict, history: pd.DataFrame, index) -> pd.Series | None:
    """True on each date of `index` where any rolling leg's contract differs from the
    previous date's — the level there changed PAIR, so the day's change is not a market
    move. None for spreads with no rolling legs (yield curves) or no contract history."""
    tkrs = _rolling_tickers(spec)
    if not tkrs or len(index) == 0:
        return None
    cts = [_leg_contracts(history, t, index) for t in tkrs]
    if any(c is None for c in cts):
        return None
    key = pd.Series(list(zip(*[c.to_numpy() for c in cts])), index=index)
    out = key.ne(key.shift())
    out.iloc[0] = False
    return out.astype(bool)


def _carry_series(spec: dict, history: pd.DataFrame, legs: pd.DataFrame) -> pd.Series | None:
    """Energy/metal calendar on the annualised-carry basis, in TODAY's pair terms:
    carry(t) = (F1 − F2) / F1 × 12 / gap(t); level(t) = carry(t) × F1_today × gap_today / 12,
    so the last value is exactly today's F1 − F2 and every past day reads what its carry
    would make today's spread. "carry_seasonal" first subtracts the median carry of the
    same front month in PRIOR years (point-in-time: no instance sees its own or later data)
    and adds today's norm back. Days with an unknown contract or a non-positive front
    (CL, 20 Apr 2020) are dropped."""
    tkr = spec["legs"][0][2]
    runs = _runs_for(history, tkr)
    if runs is None or legs.empty:
        return None
    runs["gap"] = _next_gap(runs)
    runs["mon"] = [None if pd.isna(m) else m.month for m in runs["month"]]
    pos = _run_pos(runs, legs.index)
    known = (pos >= 0) & runs["contract"].notna().to_numpy()[pos] & runs["gap"].notna().to_numpy()[pos]
    inst = pd.Series(np.where(known, pos, -1), index=legs.index)
    gap = pd.Series(np.where(known, runs["gap"].to_numpy(dtype=object)[pos], np.nan),
                    index=legs.index, dtype=float)
    f1, f2 = legs[f"raw:{tkr}"], legs[f"front2:{tkr}"]
    carry = ((f1 - f2) / f1 * 12.0 / gap).where(f1 > 0)
    norm_now = 0.0
    if spec.get("basis") == "carry_seasonal":
        # one value per contract instance (its mean carry while front), then for each
        # instance the median over PRIOR instances of the same front month — prior years
        per = pd.DataFrame({"c": carry, "i": inst})
        per = per[(per["i"] >= 0) & per["c"].notna()]
        inst_c = per.groupby("i")["c"].mean()
        norm = {}
        for i in inst_c.index:
            same = [j for j in inst_c.index if j < i and runs["mon"].iloc[j] == runs["mon"].iloc[i]]
            norm[i] = float(inst_c.loc[same].median()) if len(same) >= SEAS_MIN_YEARS else np.nan
        nrm = inst.map(norm).astype(float)
        carry = carry - nrm
        norm_now = float(nrm.iloc[-1])
    if pd.isna(carry.iloc[-1]) or not np.isfinite(norm_now):
        return None                                   # today's pair can't be measured
    carry = carry.dropna()
    if len(carry) < MIN_OVERLAP:
        return None
    k_today = float(f1.iloc[-1]) * float(gap.iloc[-1]) / 12.0
    return ((carry + norm_now) * k_today * spec.get("scale", 1.0)).rename(spec["key"])


def _build_spread(spec: dict, history: pd.DataFrame) -> pd.Series | None:
    cols = [f"{kind}:{tkr}" for _, kind, tkr in spec["legs"]]
    if any(c not in history.columns for c in cols):
        return None
    legs = history[cols].dropna()
    # a rolling spread keeps only days whose contracts are known: a roll we can't place
    # is a jump we can't neutralise (contract-less history — feed fallback — passes as is)
    for t in _rolling_tickers(spec):
        ct = _leg_contracts(history, t, legs.index)
        if ct is not None:
            legs = legs[ct.notna().to_numpy()]
    if len(legs) < MIN_OVERLAP:
        return None
    if spec.get("basis") in ("carry", "carry_seasonal"):
        return _carry_series(spec, history, legs)
    if spec.get("kind_of_spread") == "ratio":
        s = legs.iloc[:, 0] / legs.iloc[:, 1]
    else:
        s = sum(w * legs[f"{kind}:{tkr}"] for w, kind, tkr in spec["legs"])
    return (s * spec.get("scale", 1.0)).rename(spec["key"])


def spread_with_breaks(spec: dict, history: pd.DataFrame):
    """(level series, roll_breaks aligned to it or None) — what a backtest needs."""
    s = _build_spread(spec, history)
    if s is None:
        return None, None
    return s, roll_breaks(spec, history, s.index)


def _half_life(spread: pd.Series, breaks: pd.Series | None = None) -> float:
    """Ornstein-Uhlenbeck half-life (same estimator as the Mean Reversion book). Roll-day
    changes (`breaks`) are left out of the regression — a pair switch is not reversion."""
    s = spread.dropna()
    if len(s) < 30:
        return float("nan")
    d = pd.concat([s.diff(), s.shift(1)], axis=1)
    if breaks is not None:
        d = d[~breaks.reindex(d.index, fill_value=False).to_numpy(dtype=bool)]
    d = d.dropna()
    if d.empty:
        return float("nan")
    b = np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("nan")


def _dollar_sigma(spec: dict, sigma: float) -> float | None:
    """1σ of the spread in money per 1-lot-per-leg — in the legs' own currency (the row's
    `dsig_sym`) — ONLY where the volbt point value reconciles cleanly: same-product
    calendars and unit-weight diffs whose legs share one point value and one currency.
    Yield-space and ratio spreads return None ($ needs DV01 / lot ratios — we don't
    fake it)."""
    if "pv_unit" in spec:
        return abs(sigma) * spec["pv_unit"]
    if spec.get("kind_of_spread") == "ratio" or spec["unit"] == "bp" and any(
            kind == "yield" for _, kind, _ in spec["legs"]):
        return None
    if len({currency(tkr) for _, _, tkr in spec["legs"]}) != 1:
        return None
    pvs = {point_value(tkr) for _, _, tkr in spec["legs"]}
    if len(pvs) != 1 or not pvs or 0.0 in pvs:
        return None
    pv = pvs.pop() / spec.get("scale", 1.0)      # $ per displayed spread unit
    return abs(sigma) * pv


def _row(spec: dict, s: pd.Series, window: int, threshold: float,
         breaks: pd.Series | None = None) -> dict | None:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std
    if pd.isna(z.iloc[-1]):
        # not enough history for the window — fall back to whatever we have
        if len(s) < MIN_OVERLAP:
            return None
        mean = s.expanding(MIN_OVERLAP // 2).mean()
        std = s.expanding(MIN_OVERLAP // 2).std()
        z = (s - mean) / std
        if pd.isna(z.iloc[-1]):
            return None
    level, m, sd, zi = float(s.iloc[-1]), float(mean.iloc[-1]), float(std.iloc[-1]), float(z.iloc[-1])
    pctl = float((s <= level).mean() * 100.0)
    if zi >= threshold:
        signal, direction = "Rich", -1
    elif zi <= -threshold:
        signal, direction = "Cheap", 1
    else:
        signal, direction = "—", 0
    inval = m + INVAL_SIGMA * sd * (1 if zi >= 0 else -1)
    dsig = _dollar_sigma(spec, sd)
    # a roll day's change is the switch to a new contract pair, not a move on the day
    rolled = breaks is not None and len(breaks) and bool(breaks.iloc[-1])
    chg1d = (float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 and not rolled else float("nan"))
    return {
        "key": spec["key"], "name": spec["name"], "group": spec["group"],
        "unit": spec["unit"], "dp": spec["dp"], "desc": spec["desc"],
        "bench": bool(spec.get("bench", False)), "mkt": spec.get("mkt", ""),
        "level": level, "chg1d": chg1d, "rolled_today": bool(rolled),
        "z": zi, "pctl": pctl, "mean": m, "sigma": sd,
        "hi": float(s.max()), "lo": float(s.min()),
        "half_life": _half_life(s.tail(window * 2), breaks),
        "signal": signal, "direction": direction,
        "objective": m, "invalidation": inval, "dollar_sigma": dsig,
        "dsig_sym": money_symbol(spec["legs"][0][2]),
        "basis": spec.get("basis", "level"),
        "first": s.index.min().date().isoformat(), "days": int(len(s)),
        "asof": s.index.max().date().isoformat(),
    }


def monitor(window: int = WINDOW, threshold: float = Z_THRESHOLD,
            history: pd.DataFrame | None = None) -> pd.DataFrame:
    """The whole book, one row per spread, grouped in GROUPS order then by |z|."""
    if history is None:
        history = load_history()
    rows = []
    for spec in SPREADS:
        s, brk = spread_with_breaks(spec, history)
        if s is None:
            continue
        r = _row(spec, s, window, threshold, brk)
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Fixed book order (not |z|): the curve ladder reads like-for-like — each tenor
    # pair's two markets adjacent, short end to long end — and every other group keeps
    # its curated order. The "stretched now" banner surfaces the extremes instead.
    pos = {s["key"]: i for i, s in enumerate(SPREADS)}
    df["_g"] = df["group"].map({g: i for i, g in enumerate(GROUPS)})
    df["_p"] = df["key"].map(pos)
    return df.sort_values(["_g", "_p"]).drop(columns=["_g", "_p"]).reset_index(drop=True)


def spread_chart_data(key: str, window: int = WINDOW, threshold: float = Z_THRESHOLD,
                      history: pd.DataFrame | None = None, years: float | None = None):
    """Per-spread series + summary for the detail chart (mean_reversion.pair_chart_data
    shape): DataFrame(date, spread, mean, upper, lower, z) + info dict. `years` trims
    the returned frame's left edge AFTER the stats are computed on full depth."""
    spec = SPREAD_BY_KEY[key]
    if history is None:
        history = load_history([spec])
    s, brk = spread_with_breaks(spec, history)
    if s is None:
        return pd.DataFrame(), {}
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std
    out = pd.DataFrame({
        "date": s.index, "spread": s.to_numpy(), "mean": mean.to_numpy(),
        "upper": (mean + threshold * std).to_numpy(),
        "lower": (mean - threshold * std).to_numpy(), "z": z.to_numpy(),
    })
    info = _row(spec, s, window, threshold, brk) or {}
    info["threshold"] = threshold
    info["window"] = window
    if years:
        cut = s.index.max() - pd.DateOffset(days=int(years * 365.25))
        out = out[out["date"] >= cut].reset_index(drop=True)
    return out, info


# ── Hot Sheet provider ──────────────────────────────────────────────────────
RADAR_MAX = 4         # editorial cap — the book's four most stretched flags make the sheet
# The 1y z-flag misses slow grinds: a spread that drifts to a decade extreme over
# months carries its 1y band with it (OAT−Bund hit its ~100th percentile in Aug 2026
# on a z of +1.6 and never made the sheet). The full-history percentile is the
# module's other first-class read, so decade extremes get their own item kind.
RADAR_PCTL_HI = 95.0
RADAR_PCTL_LO = 5.0
RADAR_SPARK = 250     # sessions of the spread's own level behind each line's sparkline (~1y)


def radar_items() -> list:
    """The book's reads for the Hot Sheet, two kinds: its own z-flags (|z| ≥
    Z_THRESHOLD, the weekreview.collect_curve selection and prose), plus spreads at
    a full-history percentile extreme (≥95th / ≤5th) that the 1y band hasn't
    flagged — ranked by |z| and percentile stretch respectively. Reads the deep
    store via monitor(); nothing here can trigger a pull."""
    from src import hotsheet
    from .reportkit import ordinal

    history = load_history()
    m = monitor(history=history)
    if m.empty:
        return []

    def _spark(key: str) -> list | None:
        """The spread's own recent daily levels, oldest→newest, in the same unit
        the line quotes (its last point IS the item's level) — rebuilt from the
        already-loaded history frame, so no extra store reads."""
        s = _build_spread(SPREAD_BY_KEY[key], history)
        return None if s is None else s.tail(RADAR_SPARK).tolist()

    fl = m[(m["direction"] != 0) & m["z"].notna()].copy()
    fl = fl.reindex(fl["z"].abs().sort_values(ascending=False).index).head(RADAR_MAX)
    out = []
    for r in fl.itertuples(index=False):
        pct = (f" — {ordinal(int(round(r.pctl)))} percentile of the stored history"
               if r.pctl == r.pctl else "")
        out.append(hotsheet.item(
            tag="CURVE", key=f"{r.key}:{r.signal}", section="Curve / RV",
            text=f"**{r.name}** screens **{r.signal.lower()}** at {r.level:,.{r.dp}f} "
                 f"{r.unit}{pct}.",
            # heat = the more extreme of the module's two reads, so a z-flag sitting
            # AT a decade extreme never ranks below an unflagged decade extreme
            heat=max(hotsheet.heat_from_z(r.z),
                     hotsheet.heat_from_pctl(r.pctl) if r.pctl == r.pctl else 0.0),
            metric=f"z {r.z:+.1f}", sub="vs its 1y band",
            value=float(r.level),        # the spread level — a meaningful week-on-week Δ
            spark=_spark(r.key),         # the spread's own level, same unit as quoted
            ticker="",                   # multi-leg — no single ticker for the sector filter
            page="Curve Monitor", book="ficc",
        ))
    # decade extremes the z-flag hasn't caught (direction == 0 ⇒ no double line when
    # a spread trips both bars — the z item above already quotes its percentile)
    px = m[(m["direction"] == 0) & m["pctl"].notna()
           & ((m["pctl"] >= RADAR_PCTL_HI) | (m["pctl"] <= RADAR_PCTL_LO))].copy()
    px = px.reindex((px["pctl"] - 50.0).abs().sort_values(ascending=False).index)
    for r in px.itertuples(index=False):
        if len(out) >= RADAR_MAX + 2:    # extremes may add at most two lines
            break
        side = "high" if r.pctl >= RADAR_PCTL_HI else "low"
        z_note = f" (1y z {r.z:+.1f}, inside the monitor's band)" if r.z == r.z else ""
        out.append(hotsheet.item(
            tag="CURVE", key=f"{r.key}:10y-{side}", section="Curve / RV",
            text=f"**{r.name}** sits at {r.level:,.{r.dp}f} {r.unit} — the "
                 f"{ordinal(int(round(r.pctl)))} percentile of the stored history, a "
                 f"decade **{side}**{z_note}.",
            heat=hotsheet.heat_from_pctl(r.pctl),
            metric=f"{r.pctl:.0f}th pctl", sub="of stored history",
            value=float(r.level),
            spark=_spark(r.key),
            ticker="", page="Curve Monitor", book="ficc",
        ))
    return out
