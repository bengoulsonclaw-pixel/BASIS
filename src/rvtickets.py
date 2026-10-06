"""RV Trade Tickets — the Curve / RV spread book turned into backtested, sized tickets.

The Curve / RV Monitor says WHICH spreads are stretched (a rolling z plus a decade
percentile) and gives the mechanical entry / objective / invalidation. This module answers
the two questions a desk asks next, for every spread that is live-stretched right now:

  1. DID fading it historically pay?  A static-level event study on the FULL deep store.
     Each time the spread first crossed into the ±threshold zone we set the SAME ticket a
     trader would — target = the rolling mean, or a nearer partial take-profit `tp_z` σ from
     the mean when one is chosen (a closer target lifts the hit-rate and shortens the hold for
     a smaller move), stop a fixed STOP_BUFFER σ FURTHER out than the entry (always beyond it,
     when the spread is already past the monitor's ±INVAL_SIGMA band; at the default 2σ flag
     the 1σ buffer lands exactly on that 3σ band) — and follow the spread's daily CLOSE
     forward until it reaches the target (win), the stop (loss), or a cap of ~5 half-lives
     elapses (timeout, marked out at the market). The position is followed on a ROLL-NEUTRAL
     path: daily changes chained within each contract pair and zeroed on a roll day
     (curvemon.roll_breaks), so a calendar's switch to a new pair can't fake a win or a stop —
     and a same-product calendar is NEVER held into its front's delivery: it is force-exited at
     the close FLAT_BUFFER_BD business days before first notice (physicals) or at last trade
     (cash-settled STIRs), outcome "notice", and never entered inside that window (the
     contango that compresses in the delivery window isn't a non-delivery desk's to bank).
     One position at a time per spread, so a
     single stretched episode is counted once, and a day that gaps through both levels is
     scored the pessimistic way (stop first). Closes only — we never invent an intraday
     high/low the store doesn't hold. Out of that: the hit-rate, the realised expectancy
     (per trade, in σ so it is comparable across a book of different units), and the
     typical number of sessions a winner took to revert.

  2. HOW to put it on, sized to a risk budget.  Honest regimes, same discipline as
     curvemon._dollar_sigma (never fake a DV01 or a lot ratio):
       • same-product calendars and unit-weight diffs — 1 lot per leg, sized off volbt's
         reconciled point value;
       • any bond-yield spread — curve, cross-market or box — DV01-weighted off the desk-
         editable CTD table (futyield.fut_dv01), each leg to an equal per-bp exposure; when
         the legs span currencies (UST–Bund, Gilt–Bund, US−DE boxes) their DV01s are put on
         ONE money at today's FX, so a 2s10s reads "Buy N× TY / Sell M× TU" and a UST–Bund
         reads "Buy N× TY / Sell M× RX" sized in USD;
       • ratios (gold/silver) and any bond spread missing a leg price / CTD / FX rate — levels
         and risk only, flagged "structure manually", rather than fake a notional split.

The ranking is a forward expected value in σ: win_rate × (reward in σ) − (1 − win_rate) ×
(risk in σ), using TODAY's stretch for the reward/stop room and the backtest's hit-rate —
so a spread that is stretched but has historically kept going shows a NEGATIVE edge and
sinks, which is exactly the read the desk needs. Timeouts and stops both count as
non-wins, so the number is deliberately conservative.

Nothing here is advice: a ticket is the mechanical consequence of the spread's own history.
Client prose stays neutral; this is a desk tool.

No Streamlit here — the page drives this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import curvemon, deepstore, futyield
from .curvemon import WINDOW, Z_THRESHOLD
from .volbt import FX_USD, currency, fx_usd_rate, money_symbol, point_value

REV = 6                 # bump when the ticket/backtest schema changes — busts the page cache
                        # (6: calendars flat by first notice / last trade — outcome "notice")

# A fade's stop sits this many σ FURTHER out than the entry, so it is always beyond the
# entry (a 2σ fade risks 1σ to make 2; the more stretched the entry, the better that
# reward:risk). 1.0 coincides with the Curve Monitor's ±INVAL_SIGMA (3σ) band at the
# default 2σ flag, and — unlike a fixed 3σ level — never lands on the entry's profit side
# when a spread is already past 3σ.
STOP_BUFFER = 1.0

# Backtest horizon: follow each episode for ~5 half-lives, bounded so a fast calendar and a
# slow curve are each given a fair but finite window; the fallback is used when the OU
# half-life is not finite (no clean reversion estimate).
HOLD_LIVES = 5.0
HOLD_MIN = 42           # ≈2 months
HOLD_MAX = 252          # ≈1 year
HOLD_FALLBACK = 126     # ≈6 months

DEFAULT_RISK = 25_000.0  # per-trade risk budget, in each ticket's own currency
MIN_TRADES = 8           # sample floor for a "trustworthy" edge (fewer is shown, flagged)

# ── must be flat by: no fade rides a front contract into delivery ────────────
# A non-delivery desk is out of a physically-delivered front before first notice, and a cash-
# settled STIR's front is gone at its last trade — so a same-product calendar fade is FORCE-
# EXITED at the close of its front's exit-by day (outcome "notice") and never entered on/after
# it. The 2026-10-06 copper evidence: fixed HG1/HG2 pairs held to last trade narrowed 10/11
# (+1.63¢); out 3 bd before first notice only 2/11 did (−0.23¢) — the contango compression the
# old backtest banked happens INSIDE the delivery window. Physical: out at the close
# FLAT_BUFFER_BD business days before first notice (broker liquidation deadline + a day; CME's
# first position day is 2 bd before FND). Where an exchange has no notice day ahead of last trade
# (NYMEX WTI / Henry Hub list FND the bd AFTER it; ICE Brent goes to EFP/cash-settle and ICE
# Endex TTF to delivery the day after), the notice day is the bd after last trade.
# Last-trade rules come from src/expiries.py, holiday-aware; every one was checked against the
# observed end of each front run in deep_contract.parquet (2026-10-06: the store rolls ON the
# rule's last trade, or a day early, for all 8 products).
FLAT_BUFFER_BD = 3
_FLAT_RULES = {
    # ticker: (settlement, first-notice rule or None = bd after last trade, LTD month shift)
    "HGA Comdty": ("physical", ("last_bday_prior", 1), 0),   # COMEX FND: last bd of prior month
    "CLA Comdty": ("physical", None, 0),
    "COA Comdty": ("physical", None, 0),
    "NGA Comdty": ("physical", None, 0),
    "FJSA Comdty": ("physical", None, 0),
    # SOFR / SONIA 3M are named for the START of their reference quarter and last-trade at its
    # END (SFRM6 trades to 15 Sep 2026) — the expiries rule evaluated 3 months on (33/33 exact).
    "SFRA Comdty": ("cash", None, 3),
    "ERA Comdty": ("cash", None, 0),
    "SFIA Comdty": ("cash", None, 3),
}

# Short per-leg labels for the structure string (fallback = ticker root).
_LEG_NAME = {
    "TUA Comdty": "TU", "FVA Comdty": "FV", "TYA Comdty": "TY", "UXYA Comdty": "UXY",
    "USA Comdty": "US", "WNA Comdty": "WN", "UBA Comdty": "UB",
    "DUA Comdty": "DU", "OEA Comdty": "OE", "RXA Comdty": "RX",
    "OATA Comdty": "OAT", "G A Comdty": "Gilt",
    "SFRA Comdty": "SOFR", "ERA Comdty": "Euribor", "SFIA Comdty": "SONIA",
    "CLA Comdty": "WTI", "COA Comdty": "Brent", "NGA Comdty": "Henry Hub",
    "FJSA Comdty": "TTF", "GCA Comdty": "Gold", "SIA Comdty": "Silver",
    "PLA Comdty": "Platinum", "HGA Comdty": "Copper",
}


def _leg_name(tkr: str) -> str:
    return _LEG_NAME.get(tkr, tkr.split()[0].rstrip("A") or tkr)


def _conf_tier(n: int) -> str:
    if n >= 20:
        return "high"
    if n >= MIN_TRADES:
        return "ok"
    if n >= 3:
        return "thin"
    return "insufficient"


_CONF_WEIGHT = {"high": 1.0, "ok": 0.9, "thin": 0.6, "insufficient": 0.3}


def _flat_ticker(spec: dict) -> str | None:
    """The front contract a same-product calendar is exposed to, when it has a flat rule."""
    tkrs = {t for _, kind, t in spec["legs"] if kind in ("raw", "front2")}
    if len(tkrs) != 1 or len(spec["legs"]) != 2:
        return None
    t = tkrs.pop()
    return t if t in _FLAT_RULES else None


def _flat_dates(tkr: str, month) -> dict | None:
    """{'flat_by', 'exit_by', 'kind'} for ONE front contract month: flat_by = first notice
    (physical) or last trade (cash); exit_by = the last close the fade may hold."""
    from . import expiries
    settle, fnd_rule, shift = _FLAT_RULES[tkr]
    spec = expiries.spec_for(tkr)
    if spec is None or month is None or pd.isna(month):
        return None
    hol = expiries._holidays_for(tkr, "")
    y, m = expiries._shift_month(month.year, month.month, shift)
    try:
        ltd = expiries._eval(spec["fut"], y, m, hol)
        if settle == "cash":
            return {"flat_by": ltd, "exit_by": ltd, "kind": "last trade"}
        fnd = (expiries._eval(fnd_rule, month.year, month.month, hol) if fnd_rule
               else expiries._next_bday(ltd, hol))
    except Exception:
        return None
    return {"flat_by": fnd, "exit_by": expiries._bdays_before(fnd, FLAT_BUFFER_BD, hol),
            "kind": "first notice"}


def flat_schedule(spec: dict, history: pd.DataFrame) -> pd.DataFrame | None:
    """One row per front-contract run of a calendar: [start, end, contract, flat_by, exit_by,
    kind]. A completed run's exit_by is capped at the run's last observed day (the store rolls
    on last trade or a day early), so a position is always out before the pair switches. None
    for spreads without a flat rule or without contract history."""
    tkr = _flat_ticker(spec)
    if tkr is None or history is None or history.empty:
        return None
    runs = curvemon._runs_for(history, tkr)
    if runs is None or runs.empty:
        return None
    rows = []
    last = len(runs) - 1
    for k, r in enumerate(runs.itertuples(index=False)):
        d = _flat_dates(tkr, r.month)
        ex = pd.Timestamp(d["exit_by"]) if d else pd.NaT
        if d and k != last:
            ex = min(ex, pd.Timestamp(r.end))
        rows.append({"start": r.start, "end": r.end, "contract": r.contract,
                     "flat_by": pd.Timestamp(d["flat_by"]) if d else pd.NaT,
                     "exit_by": ex, "kind": d["kind"] if d else None})
    return pd.DataFrame(rows)


def exit_by_series(spec: dict, history: pd.DataFrame, index) -> pd.Series | None:
    """The exit-by date of the front contract on each date of `index` (NaT = no rule / unknown
    contract) — what _episodes needs to force-exit before notice and not enter inside it."""
    sched = flat_schedule(spec, history)
    if sched is None or len(index) == 0:
        return None
    pos = curvemon._run_pos(sched, index)
    ex = pd.DatetimeIndex(sched["exit_by"]).to_numpy()
    vals = np.where(pos >= 0, ex[np.clip(pos, 0, None)], np.datetime64("NaT"))
    return pd.Series(pd.DatetimeIndex(vals), index=index)


def live_flat(spec: dict, history: pd.DataFrame) -> dict | None:
    """Today's front contract and its flat dates, for the ticket: {contract, flat_by, exit_by,
    kind, sessions_left (business days from the last data date to exit_by; 0 = exit at today's
    close), in_window (on/after exit_by — the backtest would not enter today)}."""
    sched = flat_schedule(spec, history)
    if sched is None or sched.empty:
        return None
    r = sched.iloc[-1]
    if pd.isna(r["exit_by"]):
        return None
    asof = pd.Timestamp(r["end"]).normalize()
    ex = pd.Timestamp(r["exit_by"]).normalize()
    return {"contract": r["contract"], "flat_by": pd.Timestamp(r["flat_by"]).date().isoformat(),
            "exit_by": ex.date().isoformat(), "kind": r["kind"],
            "sessions_left": int(np.busday_count(asof.date(), ex.date())) if ex >= asof else -1,
            "in_window": bool(asof >= ex)}


def _stop_level(mean: float, sigma: float, z: float) -> float:
    """The fade stop: STOP_BUFFER σ FURTHER from the mean than the entry, on the stretched
    side. risk = |stop − entry| is therefore STOP_BUFFER·σ for any entry, and reward to the
    mean is |z|·σ, so reward:risk = |z| / STOP_BUFFER — honest and monotonic in the stretch."""
    return mean + (abs(z) + STOP_BUFFER) * sigma * (1.0 if z > 0 else -1.0)


# ── the backtest (event study of the fade rule) ──────────────────────────────
def _episodes(s: pd.Series, window: int, threshold: float, hold_cap: int,
              tp_z: float = 0.0, keep_open: bool = False,
              breaks: pd.Series | None = None,
              exit_by: pd.Series | None = None) -> list[dict]:
    """Every fresh fade episode in one spread series, as a list of outcome dicts (each with the
    entry/exit DATE, so the ledger can track them forward). Entry on a FRESH crossing into the
    ±threshold zone (previous bar inside the band), target = `tp_z` σ from the mean on the
    stretched side (0 = the mean, full reversion; >0 = a nearer partial take-profit), stop a
    STOP_BUFFER σ further out, both FROZEN at entry; follow the close until first touch, cap, or
    the end of the data. One position at a time: scanning resumes only after the trade closes.
    The unresolved tail is dropped for the backtest (keep_open=False) or reported as an "open"
    episode (keep_open=True) for the forward ledger.

    `breaks` (curvemon.roll_breaks — True where the spread switched contract pair) makes the
    position ROLL-NEUTRAL: entry, z and the frozen levels still come off the spread itself, but
    what the position earns from there is the chain of within-pair daily changes, with each
    roll day's change set to zero — a held calendar is rolled into the new pair at no P&L,
    exactly as the panama store treats an outright. Without it the raw level jumps at every
    roll (HG 1−2: 2–5.6¢ against a ~0.35¢ daily σ) and a fade spanning one is scored on the
    jump.

    `exit_by` (exit_by_series — the front contract's last holdable close, per date) makes it
    a NON-DELIVERY position: no entry on/after the day's exit_by, and an open fade is force-
    exited at the close of the last session on/before the ENTRY front's exit_by (outcome
    "notice", marked at the market like a timeout — a non-win). A notice exit that lies beyond
    the data is unresolved (open / dropped), never pre-empted. Scanning resumes after the exit
    with the fresh-crossing rule unchanged, so a stretch that persists across the roll is NOT
    re-entered in the next pair — the episode is counted once."""
    s = s.dropna()
    n = len(s)
    if n < window + 10:
        return []
    exv = None
    if exit_by is not None:
        exv = pd.DatetimeIndex(exit_by.reindex(s.index)).to_numpy()
        idx_np = s.index.to_numpy()
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    z = (s - mean) / std
    vals = s.to_numpy(dtype=float)
    path = vals                                          # what a held position marks against
    if breaks is not None:
        br = breaks.reindex(s.index, fill_value=False).to_numpy(dtype=bool)
        step = np.diff(vals, prepend=vals[0])
        step[br] = 0.0
        path = vals[0] + np.cumsum(step)
    zv = z.to_numpy(dtype=float)
    mv = mean.to_numpy(dtype=float)
    sv = std.to_numpy(dtype=float)

    out: list[dict] = []
    i = window
    while i < n:
        zi, sig = zv[i], sv[i]
        if not np.isfinite(zi) or not np.isfinite(sig) or sig <= 0 or abs(zi) < threshold:
            i += 1
            continue
        zp = zv[i - 1]
        if np.isfinite(zp) and abs(zp) >= threshold:      # not a fresh crossing
            i += 1
            continue
        k_flat = None                                     # last bar this front may be held to
        if exv is not None and not np.isnat(exv[i]):
            if idx_np[i] >= exv[i]:                       # inside the front's exit window
                i += 1
                continue
            if idx_np[-1] >= exv[i]:                      # else exit-by is past the data: no cap yet
                k_flat = int(np.searchsorted(idx_np, exv[i], side="right")) - 1
                if k_flat <= i:                           # no session left to hold before it
                    i += 1
                    continue

        direction = -1 if zi > 0 else 1                   # fade: short a rich spread
        side = 1.0 if zi > 0 else -1.0
        entry = vals[i]
        off = entry - path[i]                             # marks the roll-neutral path at entry
        tgt = mv[i] + tp_z * sig * side                   # tp_z=0 → the mean (full reversion)
        stop = _stop_level(mv[i], sig, zi)

        j_end = min(n - 1, i + hold_cap)
        forced = k_flat is not None and k_flat <= j_end   # the front's exit-by binds first
        if forced:
            j_end = k_flat
        outcome = exitpx = exit_i = None
        j = i + 1
        while j <= j_end:
            px = path[j] + off
            if direction == -1:                           # short: win below target, stop above
                hit_tgt, hit_stop = px <= tgt, px >= stop
            else:                                         # long: win above target, stop below
                hit_tgt, hit_stop = px >= tgt, px <= stop
            # target and stop sit on OPPOSITE sides of the entry, so a single close can satisfy
            # at most one; check the stop first so a boundary touch is scored the sober way.
            if hit_stop:
                outcome, exitpx, exit_i = "loss", stop, j
                break
            if hit_tgt:
                outcome, exitpx, exit_i = "win", tgt, j
                break
            j += 1
        entry_date = s.index[i].date().isoformat()
        if outcome is None:
            if forced:                                    # flat before notice / last trade
                outcome, exitpx, exit_i = "notice", path[j_end] + off, j_end
            elif i + hold_cap <= n - 1:                   # full window elapsed → timeout
                outcome, exitpx, exit_i = "timeout", path[j_end] + off, j_end
            elif keep_open:                               # unresolved tail → a live open position
                out.append({
                    "dir": direction, "entry": float(entry), "target": float(tgt),
                    "stop": float(stop), "exit": None, "bars": int(j_end - i),
                    "outcome": "open", "pnl": None, "pnl_sigma": None,
                    "entry_date": entry_date, "exit_date": None,
                    "sigma": float(sig), "z": float(zi),
                })
                break
            else:                                         # ran off the end of data → dropped
                break

        pnl = direction * (exitpx - entry)                # +ve = toward the objective
        out.append({
            "dir": direction, "entry": float(entry), "target": float(tgt),
            "stop": float(stop), "exit": float(exitpx), "bars": int(exit_i - i),
            "outcome": outcome, "pnl": float(pnl), "pnl_sigma": float(pnl / sig),
            "entry_date": entry_date, "exit_date": s.index[exit_i].date().isoformat(),
            "sigma": float(sig), "z": float(zi),
        })
        i = exit_i + 1
    return out


def backtest_spread(s: pd.Series, window: int, threshold: float,
                    half_life: float, tp_z: float = 0.0,
                    breaks: pd.Series | None = None,
                    exit_by: pd.Series | None = None) -> dict:
    """Aggregate fade-rule stats for one spread: sample size, hit-rate (reach the `tp_z`
    take-profit before the stop; timeouts and forced notice exits count as non-wins), realised
    expectancy and average loss in σ, and the median sessions a winner took. A nearer target
    (tp_z > 0) lifts the hit-rate and shortens the hold at the cost of a smaller per-trade move."""
    hold_cap = (int(np.clip(round(HOLD_LIVES * half_life), HOLD_MIN, HOLD_MAX))
                if half_life == half_life and half_life > 0 else HOLD_FALLBACK)
    eps = _episodes(s, window, threshold, hold_cap, tp_z, breaks=breaks, exit_by=exit_by)
    n = len(eps)
    base = {"n": n, "wins": 0, "losses": 0, "timeouts": 0, "notices": 0,
            "win_rate": float("nan"),
            "expectancy_sigma": float("nan"), "avg_loss_sigma": float("nan"),
            "median_days_win": float("nan"), "avg_days": float("nan"), "hold_cap": hold_cap}
    if n == 0:
        return base
    wins = [e for e in eps if e["outcome"] == "win"]
    losses = [e for e in eps if e["outcome"] == "loss"]
    tos = [e for e in eps if e["outcome"] == "timeout"]
    win_days = [e["bars"] for e in wins]
    base.update({
        "wins": len(wins), "losses": len(losses), "timeouts": len(tos),
        "notices": sum(1 for e in eps if e["outcome"] == "notice"),
        "win_rate": len(wins) / n,
        "expectancy_sigma": float(np.mean([e["pnl_sigma"] for e in eps])),
        "avg_loss_sigma": float(np.mean([e["pnl_sigma"] for e in losses])) if losses else float("nan"),
        "median_days_win": float(np.median(win_days)) if win_days else float("nan"),
        "avg_days": float(np.mean([e["bars"] for e in eps])),
    })
    return base


# ── sizing ───────────────────────────────────────────────────────────────────
def _sized_legs(spec: dict, direction: int, lots: int) -> list[dict]:
    """Per-leg Buy/Sell at `lots` each (1-lot-per-leg regime). side = sign(weight×direction):
    short the spread (direction −1) sells the +weight leg; same-ticker calendars label the
    legs front / 2nd rather than repeating the product."""
    same_ticker = len({t for _, _, t in spec["legs"]}) == 1
    legs = []
    for w, kind, tkr in spec["legs"]:
        side = "Buy" if (w * direction) > 0 else "Sell"
        if same_ticker:
            label = "front" if kind == "raw" else "2nd"
        else:
            label = _leg_name(tkr)
        legs.append({"side": side, "lots": lots, "label": label})
    return legs


def _is_dv01_sizeable(spec: dict) -> bool:
    """A bond-yield spread we can DV01-weight: every leg is a benchmark-yield leg whose future
    is in the editable CTD table, with 2 or 4 legs (single- or cross-market curve, or box).
    Currency mixing is handled by the sizer via FX; ratios and STIR/price-leg spreads are out
    (they carry no CTD yield)."""
    legs = spec["legs"]
    if len(legs) not in (2, 4) or spec.get("kind_of_spread") == "ratio":
        return False
    if any(kind != "yield" for _, kind, _ in legs):
        return False
    seed = futyield.load_ctd()
    return all(t in seed for _, _, t in legs)


def size_ticket(spec: dict, row: dict, risk_budget: float, raw_last: dict | None = None,
                ctd: dict | None = None, fx: dict | None = None) -> dict:
    """Translate a spread row into a sized ticket. Returns {method, structure_legs, lots,
    per_bp, risk_money, risk_sym, note} — method ∈ {calendar, dv01, manual, none}."""
    level = float(row["level"])
    inval = float(row["invalidation"])
    sigma = float(row["sigma"])
    direction = int(row["direction"])
    stop_dist = abs(inval - level)
    none = {"method": "none", "structure_legs": [], "lots": None, "per_bp": None,
            "risk_money": None, "risk_sym": "$", "note": ""}
    if not np.isfinite(stop_dist) or stop_dist <= 0 or direction == 0:
        return none

    # A — reconciled $ per unit (same-product calendars, unit-weight diffs): 1 lot per leg.
    dsig = row.get("dollar_sigma")
    if dsig is not None and dsig == dsig and sigma > 0:
        pv_unit = dsig / sigma                               # money per displayed unit, per lot/leg
        risk_per_lot = stop_dist * pv_unit
        if risk_per_lot > 0:
            lots = max(1, round(risk_budget / risk_per_lot))
            return {"method": "calendar",
                    "structure_legs": _sized_legs(spec, direction, lots),
                    "lots": lots, "per_bp": None,
                    "risk_money": lots * risk_per_lot,
                    "risk_sym": row.get("dsig_sym") or "$",
                    "note": "1 lot per leg, sized off the reconciled point value."}

    # B — bond-yield spread (curve / cross-market / box): DV01-weight each leg to an equal
    #     per-bp exposure, putting the legs on ONE currency (USD) at today's FX when they span
    #     currencies. A long bond future is long when its yield falls, so a leg is bought when
    #     direction×weight < 0 (fade a rich spread → direction −1).
    if _is_dv01_sizeable(spec):
        ctd = ctd or futyield.load_ctd()
        raw_last, fx = raw_last or {}, fx or {}
        legs = spec["legs"]
        multi = len({currency(t) for _, _, t in legs}) > 1    # mixed currencies → convert to USD
        conv, ok = {}, True
        for _, _, t in legs:
            c = currency(t)
            if not multi or c == "USD":
                conv[t] = 1.0
            elif fx.get(c, 0) > 0:
                conv[t] = fx[c]                               # USD per 1 unit of c
            else:
                ok = False
                break
        if ok and all(raw_last.get(t) and t in ctd for _, _, t in legs):
            def _dv01(tkr):
                e = ctd[tkr]
                return futyield.fut_dv01(float(raw_last[tkr]), e["cf"], e["coupon"], e["years"],
                                         int(e["freq"]), point_value(tkr)) * conv[tkr]
            dvs = {t: _dv01(t) for _, _, t in legs}
            if all(v > 0 for v in dvs.values()):
                d_per_bp = risk_budget / stop_dist           # common-ccy money per bp of the spread
                out_legs, wsum, realised = [], 0.0, 0.0
                for w, _, t in legs:
                    n = max(1, round(d_per_bp * abs(w) / dvs[t]))
                    out_legs.append({"side": "Buy" if (direction * w) < 0 else "Sell",
                                     "lots": n, "label": _leg_name(t)})
                    wsum += abs(w)
                    realised += n * dvs[t]
                per_bp = realised / wsum if wsum else float("nan")   # realised common-ccy $/bp
                return {"method": "dv01", "structure_legs": out_legs, "lots": None,
                        "per_bp": per_bp, "risk_money": per_bp * stop_dist,
                        "risk_sym": "$" if multi else money_symbol(legs[0][2]),
                        "note": "DV01-weighted off the editable CTD table"
                                + (", legs put on one currency at today's FX" if multi else "")
                                + " — indicative; fine-tune from the delivery basket."}
        # sizeable in principle but a price / CTD / FX rate is missing → fall through to manual

    # C — manual: honest levels, no faked lots (a ratio needs a notional split; a bond spread
    #     lands here only when a leg's price, CTD entry or FX rate is unavailable).
    why = ("ratio — size by notional" if spec.get("kind_of_spread") == "ratio"
           else "needs a leg price / CTD / FX rate not on hand — DV01-weight manually")
    return {"method": "manual", "structure_legs": [], "lots": None, "per_bp": None,
            "risk_money": None, "risk_sym": money_symbol(spec["legs"][0][2]),
            "note": f"Structure manually ({why})."}


# ── ticket assembly + ranking ────────────────────────────────────────────────
def _structure_text(action_verb: str, size: dict) -> str:
    legs = size.get("structure_legs")
    if legs:
        body = " / ".join(f"{lg['side']} {lg['lots']}× {lg['label']}" for lg in legs)
        if action_verb in ("Flatten", "Steepen"):        # single-market curve verb reads on its own
            return f"{action_verb} — {body}"
        return f"{action_verb} the spread — {body}"
    return f"{action_verb} the spread — {size.get('note') or 'structure manually'}"


def tickets(window: int = WINDOW, threshold: float = Z_THRESHOLD,
            risk_budget: float = DEFAULT_RISK, groups: list | None = None,
            history: pd.DataFrame | None = None, tp_z: float = 0.0) -> list[dict]:
    """Every live-stretched spread as a ranked trade ticket (best forward edge first). `tp_z`
    sets the take-profit `tp_z` σ from the mean (0 = full reversion); the stop, and so the
    sizing, is unchanged — only the target, reward and the backtest move with it."""
    if history is None:
        history = curvemon.load_history()
    mon = curvemon.monitor(window, threshold, history=history)
    if mon is None or mon.empty:
        return []
    flagged = mon[mon["direction"] != 0]
    if groups:
        flagged = flagged[flagged["group"].isin(groups)]
    if flagged.empty:
        return []

    # One raw-price read for every bond leg that might be DV01-sized, plus the FX generics
    # needed to put cross-currency legs on one money. FX rate = USD per 1 unit of the currency.
    ctd = futyield.load_ctd()
    sizeable = [curvemon.SPREAD_BY_KEY[r["key"]] for _, r in flagged.iterrows()
                if _is_dv01_sizeable(curvemon.SPREAD_BY_KEY[r["key"]])]
    bond_tkrs = {t for sp in sizeable for _, _, t in sp["legs"]}
    fx_generics = {c: FX_USD[c] for sp in sizeable for _, _, t in sp["legs"]
                   if (c := currency(t)) != "USD" and c in FX_USD}
    raw_last, fx = {}, {}
    read_tkrs = sorted(bond_tkrs | set(fx_generics.values()))
    if read_tkrs:
        raw = deepstore.get_raw(read_tkrs)
        if raw is not None and not raw.empty:
            last = raw.ffill().iloc[-1]
            raw_last = {t: float(last[t]) for t in raw.columns if pd.notna(last[t])}
            fx = {c: fx_usd_rate(g, raw_last[g]) for c, g in fx_generics.items() if g in raw_last}

    out = []
    for _, row in flagged.iterrows():
        spec = curvemon.SPREAD_BY_KEY[row["key"]]
        s, brk = curvemon.spread_with_breaks(spec, history)
        if s is None:
            continue
        r = row.to_dict()
        # Override the monitor's fixed ±INVAL_SIGMA band with the fade stop (always beyond the
        # entry), so sizing, risk and the displayed stop are all the SAME level the backtest used.
        r["invalidation"] = _stop_level(r["mean"], r["sigma"], r["z"])
        # Target = tp_z σ from the mean on the stretched side (tp_z=0 → the mean, full reversion).
        r["objective"] = r["mean"] + tp_z * r["sigma"] * (1.0 if r["z"] > 0 else -1.0)
        sigma = float(r["sigma"]) or float("nan")
        reward_units = abs(r["level"] - r["objective"])
        if not np.isfinite(reward_units) or reward_units <= 0:   # TP at/beyond the entry — no trade
            continue
        bt = backtest_spread(s, window, threshold, r["half_life"], tp_z, breaks=brk,
                             exit_by=exit_by_series(spec, history, s.index))
        size = size_ticket(spec, r, risk_budget, raw_last, ctd, fx)
        flat = live_flat(spec, history)                  # calendars: today's front's exit-by

        risk_units = abs(r["invalidation"] - r["level"])
        reward_sigma = reward_units / sigma if sigma else float("nan")
        risk_sigma = risk_units / sigma if sigma else float("nan")
        wr = bt["win_rate"]
        ev_sigma = (wr * reward_sigma - (1 - wr) * risk_sigma
                    if wr == wr and reward_sigma == reward_sigma else float("nan"))
        conf = _conf_tier(bt["n"])

        rich = r["direction"] < 0
        single_curve = (spec.get("group") == "Rates — Curve"
                        and not spec["key"].startswith("box_"))
        if single_curve and size["method"] == "dv01":
            action_verb = "Flatten" if rich else "Steepen"
            action = f"Fade — {'flatten' if rich else 'steepen'} (DV01-weighted)"
        else:
            action_verb = "Short" if rich else "Long"
            action = (f"Fade the richness — short the spread" if rich
                      else "Fade the cheapness — long the spread")

        out.append({
            **{k: r[k] for k in ("key", "name", "group", "unit", "dp", "desc", "bench",
                                 "z", "pctl", "level", "mean", "sigma", "half_life",
                                 "signal", "direction")},
            "objective": r["objective"], "invalidation": r["invalidation"],
            "action": action,
            "structure": _structure_text(action_verb, size),
            "size_method": size["method"], "size_legs": size["structure_legs"],
            "lots": size["lots"], "per_bp": size["per_bp"],
            "risk_money": size["risk_money"], "risk_sym": size["risk_sym"],
            "size_note": size["note"],
            "reward_units": reward_units, "risk_units": risk_units,
            "reward_sigma": reward_sigma, "risk_sigma": risk_sigma,
            "rr": reward_units / risk_units if risk_units else float("nan"),
            "ev_sigma": ev_sigma,
            "bt_n": bt["n"], "bt_wins": bt["wins"], "bt_losses": bt["losses"],
            "bt_timeouts": bt["timeouts"], "bt_notices": bt["notices"], "bt_win_rate": wr,
            "flat": flat,
            "bt_expectancy_sigma": bt["expectancy_sigma"],
            "bt_median_days": bt["median_days_win"], "bt_hold_cap": bt["hold_cap"],
            "conf": conf,
            "score": (ev_sigma * _CONF_WEIGHT[conf]) if ev_sigma == ev_sigma else -9.9,
            "risk_budget": risk_budget,
        })

    out.sort(key=lambda t: t["score"], reverse=True)
    return out
