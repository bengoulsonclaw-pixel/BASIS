"""Roll board — when each product's front contract rolls, and what the roll costs.

About twenty products roll every fortnight and nothing in BASIS covered it: a client asking
"when does Brent roll and what's the spread doing?" needed a Bloomberg screen (Ben, 2026-08-26).

The roll dates here are OBSERVED, not assumed. data/price_store/deep_contract.parquet holds
FUT_CUR_GEN_TICKER — the actual contract behind the front generic — for every product on every
day back to 2016, so ten years of real roll dates are simply readable. Decoding the contract
symbol gives the month being rolled out of, and the roll then sits at a very stable offset
inside that month.

That offset is measured against the EXPIRING contract's month, not the incoming one. Both were
tried against all 4,770 observed rolls: anchoring on the expiring contract gives a median
inter-quartile spread of 1 business day with 99% of products inside 5, against 2 days and 83%
for the incoming contract. The reason is that the roll is driven by the expiring contract's own
expiry, so an irregular cycle gap (sugar's Oct->Mar, five months) corrupts the incoming anchor
and leaves the expiring one untouched.

What this is NOT: first notice day. The front generic switches when the front contract stops
being front, so this is effectively front-contract expiry. Liquidity migrates earlier than that,
and a long in a physically-delivered contract must be out before first notice, which can precede
expiry by weeks. The board says so on its face rather than implying a roll is safe until the
date shown — and the stores hold no second-contract volume, so the migration cannot be measured
here rather than guessed at.

Spreads come off the RAW store on both legs: a panama adjustment adds a constant to pre-roll
history, which is harmless for a single series and ruinous for the difference of two.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from . import universe as u

ROOT_STORE = "contract"        # deepstore._read adds the "deep_" prefix
MIN_ROLLS = 4          # fewer observed rolls than this and no offset is trustworthy
SOON_BD = 10           # "rolling soon" horizon, business days — about a fortnight
# Calendar spreads routinely run z of 3-6 against their seasonal (a grade change or a squeeze
# moves them far further than an outright), so saturating at the usual 4 sigma would tie most
# of this book at 100 and tell the Hot Sheet nothing about which roll is the unusual one.
HEAT_FULL_Z = 6.0

# Standard futures month codes.
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
# "KCZ16", "CLV6", "C Z16" — the store mixes one- and two-digit years (Bloomberg changed
# format mid-history) and some roots carry a trailing space ("C " for corn).
_SYM = re.compile(r"^([A-Z0-9]+?)\s*([FGHJKMNQUVXZ])(\d{1,2})$")


def decode_contract(symbol, ref: pd.Timestamp):
    """First day of the contract month a symbol names, or None.

    A two-digit year is unambiguous. A one-digit year is not — '6' is 2016 as easily as
    2026 — so it is resolved against the date the symbol was observed on, taking the
    candidate decade that puts the contract nearest that date.
    """
    m = _SYM.match(str(symbol).strip())
    if not m:
        return None
    month, yr = MONTH_CODE[m.group(2)], m.group(3)
    if len(yr) == 2:
        return pd.Timestamp(2000 + int(yr), month, 1)
    best = None
    for decade in (2010, 2020, 2030):
        cand = pd.Timestamp(decade + int(yr), month, 1)
        if -60 <= (cand - ref).days <= 900:
            if best is None or abs((cand - ref).days) < abs((best - ref).days):
                best = cand
    return best


def _contracts() -> pd.DataFrame:
    from . import deepstore
    df = deepstore._read(ROOT_STORE)
    return df if df is not None else pd.DataFrame()


def roll_history(ticker: str, contracts: pd.DataFrame | None = None) -> list:
    """[(roll_date, first_day_of_the_month_being_rolled_OUT_of), ...] as observed."""
    c = _contracts() if contracts is None else contracts
    if ticker not in getattr(c, "columns", []):
        return []
    s = c[ticker].dropna()
    changes = list(s[s != s.shift()].items())
    out = []
    for i in range(1, len(changes)):
        day, _new = changes[i]
        _prev_day, old = changes[i - 1]
        month = decode_contract(old, day)
        if month is not None:
            out.append((day, month))
    return out


def profile(ticker: str, contracts: pd.DataFrame | None = None) -> dict | None:
    """How this product rolls: business days into the expiring contract's month, and how
    tightly it holds. `band` is the inter-quartile spread — the honest width of the estimate."""
    hist = roll_history(ticker, contracts)
    if len(hist) < MIN_ROLLS:
        return None
    offs = [np.busday_count(month.date(), day.date()) for day, month in hist]
    q1, q3 = np.percentile(offs, [25, 75])
    return {"offset": int(np.median(offs)), "band": int(round(q3 - q1)),
            "n": len(offs), "last": hist[-1][0]}


def next_roll(ticker: str, today=None, contracts: pd.DataFrame | None = None) -> dict | None:
    """Projected next roll of the front generic."""
    c = _contracts() if contracts is None else contracts
    if ticker not in getattr(c, "columns", []):
        return None
    prof = profile(ticker, c)
    if prof is None:
        return None
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    s = c[ticker].dropna()
    front = str(s.iloc[-1])
    month = decode_contract(front, today)
    if month is None:
        return None
    when = pd.Timestamp(np.busday_offset(month.date().replace(day=1),
                                         prof["offset"], roll="forward"))
    return {"front": front, "date": when.date(), "band": prof["band"], "n": prof["n"],
            "bd": int(np.busday_count(today.date(), when.date()))}


# ── the spread being rolled across ───────────────────────────────────────────
def _seasonal() -> pd.DataFrame:
    """seasmon's front1-front2 screen: a NINE-YEAR seasonal norm for the very spread this
    board is about, so the seasonal context is read rather than recomputed. It matters:
    RBOB's Sep/Oct spread carries the summer-to-winter grade change and screens extreme
    against a flat mean every single year, but only 4.7 sigma against its own season."""
    try:
        return pd.read_parquet(
            __file__.rsplit("src", 1)[0] + "data/signals/seas_spread_screen.parquet")
    except Exception:
        return pd.DataFrame()


def spreads(tickers: list) -> dict:
    """front1 - front2 per product, in points, as a percent of the front, and in cash."""
    from . import deepstore, volbt
    p1, p2 = deepstore.get_raw(tickers), deepstore.get_front2(tickers)
    out = {}
    for t in tickers:
        if t not in getattr(p1, "columns", []) or t not in getattr(p2, "columns", []):
            continue
        a, b = p1[t].dropna(), p2[t].dropna()
        idx = a.index.intersection(b.index)
        if not len(idx):
            continue
        d = idx[-1]
        front, second = float(a.loc[d]), float(b.loc[d])
        pts = front - second
        try:
            pv, ccy = float(volbt.point_value(t)), volbt.currency(t)
        except Exception:
            pv, ccy = 0.0, ""
        # point_value() returns 0.0 for the 19 products absent from its table, so a cash
        # figure computed from it reads "this roll costs you nothing" when it means "we do
        # not know the contract size". Unknown has to stay blank.
        pv = pv if pv else float("nan")
        out[t] = {
            "front_px": front, "second_px": second, "pts": pts,
            # percent of the front price is the cross-product comparable. It is meaningless
            # for STIRs, whose price is 100 - rate, so it is withheld there rather than
            # printed as a number that invites a false comparison.
            "pct": (None if _is_stir(t) or not front else pts / abs(front) * 100.0),
            "cash": (None if not np.isfinite(pv) else pts * pv), "ccy": ccy,
            "state": "backwardation" if pts > 0 else "contango" if pts < 0 else "flat",
            "asof": d,
        }
    return out


def _is_stir(ticker: str) -> bool:
    return str(u.asset(ticker)).upper().startswith("STIR")


def board(today=None) -> pd.DataFrame:
    """Every product that rolls, soonest first, with the spread it rolls across."""
    c = _contracts()
    if c.empty:
        return pd.DataFrame()
    tickers = [t for t in c.columns if t in u.INSTRUMENTS]
    sp = spreads(tickers)
    seas = _seasonal()
    seas_by = ({r.ticker: r for r in seas.itertuples(index=False)}
               if "ticker" in getattr(seas, "columns", []) else {})
    rows = []
    for t in tickers:
        nr = next_roll(t, today, c)
        if nr is None:
            continue
        s = sp.get(t, {})
        sr = seas_by.get(t)
        rows.append({
            "ticker": t, "name": u.name(t), "asset": u.asset(t),
            "front": nr["front"], "roll_date": nr["date"], "bd": nr["bd"],
            "band": nr["band"], "rolls_seen": nr["n"],
            "front_px": s.get("front_px"), "second_px": s.get("second_px"),
            "pts": s.get("pts"), "pct": s.get("pct"),
            "cash": s.get("cash"), "ccy": s.get("ccy"), "state": s.get("state"),
            "unit": getattr(sr, "unit", None) if sr is not None else None,
            "seas_norm": float(getattr(sr, "norm", np.nan)) if sr is not None else np.nan,
            "seas_z": float(getattr(sr, "z", np.nan)) if sr is not None else np.nan,
            "seas_years": float(getattr(sr, "years", np.nan)) if sr is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["bd", "name"]).reset_index(drop=True) if not df.empty else df


# ── Hot Sheet provider ───────────────────────────────────────────────────────
def radar_items() -> list:
    """Products rolling within the fortnight whose calendar spread is unusual for the season.

    A roll on its own is a diary entry, not a highlight, so the date alone never earns a row:
    it has to be a roll AND a spread stretched against its own nine-year seasonal. That pairing
    is the thing worth a call — the client is about to be moved across a spread that is not
    where it usually is.
    """
    try:
        from . import hotsheet
        df = board()
        if df is None or df.empty:
            return []
        soon = df[(df["bd"] >= 0) & (df["bd"] <= SOON_BD) & df["seas_z"].notna()]
        soon = soon[soon["seas_z"].abs() >= 1.5]
        out = []
        for r in soon.sort_values("seas_z", key=lambda s: s.abs(), ascending=False).head(4).itertuples(index=False):
            wide = r.seas_z > 0
            unit = r.unit or "pts"
            when = "tomorrow" if r.bd == 1 else "today" if r.bd == 0 else f"in {r.bd} sessions"
            cash = (f", about {abs(r.cash):,.0f} {r.ccy} a lot"
                    if r.cash is not None and np.isfinite(r.cash) else "")
            out.append(hotsheet.item(
                tag="ROLL", key=f"{r.ticker}:roll", section="Curve",
                text=(f"**{r.name}** rolls {when} and the front spread is **{r.state}** at "
                      f"**{r.pts:+.2f} {unit}** — {'wider' if wide else 'narrower'} than its "
                      f"seasonal norm of {r.seas_norm:+.2f}{cash}."),
                heat=hotsheet.heat_from_z(r.seas_z, full=HEAT_FULL_Z),
                metric=f"z {r.seas_z:+.1f}", sub=f"front spread vs {r.seas_years:.0f}y season",
                value=float(r.seas_z), ticker=r.ticker, page="Roll Board", book="ficc"))
        return out
    except Exception:
        return []
