"""TA Signal Backtester — engine.

Answers: "if I'd bought/sold this product every time our technical-analysis signals cleared a
score/conviction bar, how much would I have made or lost?" Works for a single named strategy
(e.g. just Trend) or the desk's whole confluence score, on either book (FICC futures / Equities).

None of the strategy modules in `src/strategies/` emit a signal *series* over history — each one
only ever evaluates `history.iloc[-1]` (today). This engine closes that gap by walking forward
day by day, re-slicing the pulled history up to that day and calling the SAME `find_opportunities()`
functions the live pages use — so a backtest can never quietly drift from what the dashboard would
actually have said on that date. (Each strategy's own minimum-history gate just skips a ticker until
it has enough bars, exactly as it does live — no special-casing needed here.)

Fixed-income sign convention: the technical strategies compute signals on YIELDS for FI futures
(`datafeed.get_history_ta` — STIRs as 100−price, bonds as their benchmark yield; see
`src/strategies/trend.py` and friends). A "Long"/direction=+1 read there means the YIELD is expected
to rise, which is a SHORT of the bond/STIR future (the same convention the FI strategy pages already
state: "A Long / up signal there means rising yields = short the bond — and the mirror", app.py's
strategy-page header). This engine trades the actual FUTURES PRICE (so P&L uses real point values),
so every FI signal's direction is inverted before it's used to open/close a position; P&L itself is
always marked off `datafeed.get_history` (true settlement price), never the yield series.

No Streamlit here — the page drives this module, same split as `volbt.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import deepstore, tascore, universe, volbt, yfin
from .datafeed import get_history, get_history_ta, get_volume_history
from .strategies import (trend, ma_crossover, ma_crossover_swing, support_resistance,
                         fibonacci, flag_breakout, breakout_retest, momentum, bollinger,
                         elliott_wave, ichimoku, obv, mfi, mean_reversion)

# Calendar days of pre-history pulled before the backtest's start date — deep enough for every
# strategy's own warm-up window (Bollinger's is the deepest: PCTL_WINDOW(252) + BB_WINDOW(20) = 272
# rows; MA Crossover's POSITION variant needs ~205). A little slack above the deepest requirement.
LOOKBACK_BUFFER_DAYS = 550

_CLOSE_STRATS = (trend, ma_crossover, ma_crossover_swing, support_resistance, fibonacci,
                 momentum, bollinger, elliott_wave, ichimoku)
_VOL_STRATS = (flag_breakout, breakout_retest, obv, mfi)
STRATEGY_MODULES = {m.STRATEGY: m for m in (_CLOSE_STRATS + _VOL_STRATS)}
VOLUME_STRATEGIES = {m.STRATEGY for m in _VOL_STRATS}
# Mean Reversion is pair-based (universe.PAIRS) and FICC-only — its find_opportunities() ignores
# whatever single-ticker `history` it's handed and always loops the FULL PAIRS list, so it needs the
# whole pairs universe pulled, not just one ticker. Handled separately in _raw_frames, never through
# STRATEGY_MODULES: its spread signal is translated onto the target ticker's leg so it can feed a
# single-product score (it shares the Momentum axis, so the harmonic de-dup applies as usual).

EXIT_RULES = ["reversal", "score_drop", "hold_days"]


@dataclass
class Result:
    daily: pd.DataFrame        # date-indexed: price, signal_level (yield for FI), score,
                               # conviction, position (+1/-1/0), unrealized_pnl, cum_pnl
    trades: pd.DataFrame       # the blotter: one row per closed trade
    summary: dict              # stats + the run's own parameters (strategies, bars, exit rule, …)
    warnings: list = field(default_factory=list)
    series: pd.Series | None = None    # full-depth signal-space history (incl. the pre-start
                                       # warm-up buffer) — the page's indicator overlays need the
                                       # lookback so MAs/clouds are correct from the window's left edge
    volume: pd.Series | None = None    # matching volume series, when the run pulled one


def pair_for(ticker: str):
    """The universe.PAIRS entry `ticker` belongs to, or None."""
    for p in universe.PAIRS:
        if ticker in (p["a"], p["b"]):
            return p
    return None


def strategies_for_scope(scope: str) -> list:
    """Every backtestable strategy name for `scope`, in tascore.TA_STRATEGIES display order.
    Mean Reversion is FICC-only (equities has no configured pair universe)."""
    names = set(STRATEGY_MODULES) | ({"Mean Reversion"} if scope == "ficc" else set())
    return [s for s in tascore.TA_STRATEGIES if s in names]


def _validate_exit(exit_rule: str, hold_days) -> None:
    if exit_rule not in EXIT_RULES:
        raise ValueError(f"unknown exit_rule '{exit_rule}' — pick one of {EXIT_RULES}")
    if exit_rule == "hold_days" and not hold_days:
        raise ValueError("the 'hold_days' exit rule needs a positive holding period")


def _passes(sig, min_conviction: float, min_score: float, direction_filter: str) -> bool:
    if sig is None:
        return False
    if sig["conviction"] < min_conviction or abs(sig["score"]) < min_score:
        return False
    if direction_filter == "long" and sig["direction"] <= 0:
        return False
    if direction_filter == "short" and sig["direction"] >= 0:
        return False
    return True


def _raw_frames(ticker: str, strategies: list, hist_slice: pd.DataFrame,
                vol_slice: pd.DataFrame | None) -> pd.DataFrame:
    """One raw (un-reflagged) signal row per strategy for `ticker` as of the slice's last bar.
    Mean Reversion is pair-based, so its spread signal is translated onto the ticker's LEG (metric
    and direction × leg sign: "Long spread" = buy A / sell B) and keyed by the ticker itself — that
    lets it feed a single-product score exactly like any other method, sharing the Momentum axis."""
    frames = []
    for s in strategies:
        if s == "Mean Reversion":
            pr = pair_for(ticker)
            if pr is None:
                continue
            raw = mean_reversion.find_opportunities(history=hist_slice)
            row = raw[raw["instruments"] == f"{pr['a']} / {pr['b']}"].copy()
            if row.empty:
                continue
            leg_sign = 1 if ticker == pr["a"] else -1
            row["instruments"] = ticker
            row["metric"] = pd.to_numeric(row["metric"], errors="coerce") * leg_sign
            row["direction"] = row["direction"] * leg_sign
            frames.append(row)
            continue
        mod = STRATEGY_MODULES.get(s)
        if mod is None:
            continue
        kwargs = {"history": hist_slice}
        if s in VOLUME_STRATEGIES:
            kwargs["volume"] = vol_slice
        raw = mod.find_opportunities(**kwargs)
        frames.append(raw[raw["instruments"] == ticker])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _score_set(raw: pd.DataFrame, strategies: list, ticker: str, sign: int):
    """Score `ticker` on `strategies` via the exact tascore.ta_flagged() + score_products()
    pipeline the live TA hub / report use — hub-trigger re-flagging, then within-axis harmonic
    de-duplication (strongest full, next ½, ⅓ …), then the signed cross-axis sum. Returns
    {direction, conviction, score} in PRICE-trade terms (FI yield→price `sign` applied), or None
    when nothing in the set is flagging this ticker today."""
    if raw is None or raw.empty:
        return None
    flagged = tascore.ta_flagged(raw, strategies=strategies)
    if flagged is None or flagged.empty:
        return None
    scored = tascore.score_products(flagged)
    row = scored[scored["instruments"] == ticker]
    if row.empty:
        return None
    r = row.iloc[0]
    yld_dir = int(r["net_dir"])
    if yld_dir == 0:
        return None
    return {"direction": yld_dir * sign, "conviction": float(r["conviction"]),
            "score": float(r["score"]) * sign}


def _step(pos, sig, price: float, day, is_last: bool, *, exit_rule: str, hold_days,
          stop_pct, take_pct, min_conviction: float, min_score: float, direction_filter: str,
          size_mult: float, size: float):
    """Advance one flat/long/short state machine by one day. Returns (new_pos, trade_row|None)."""
    if pos is None:
        if not is_last and _passes(sig, min_conviction, min_score, direction_filter):
            pos = {"direction": sig["direction"], "entry_date": day.date(), "entry_price": price,
                   "entry_conviction": sig["conviction"], "entry_score": sig["score"]}
        return pos, None

    exit_reason = None
    if is_last:
        exit_reason = "period_end"
    else:
        if stop_pct or take_pct:
            move_pct = (price - pos["entry_price"]) / pos["entry_price"] * pos["direction"] * 100.0
            if stop_pct and move_pct <= -abs(stop_pct):
                exit_reason = "stop"
            elif take_pct and move_pct >= abs(take_pct):
                exit_reason = "take"
        if exit_reason is None:
            if exit_rule == "reversal":
                if sig is not None and sig["direction"] == -pos["direction"]:
                    exit_reason = "reversal"
            elif exit_rule == "score_drop":
                if not _passes(sig, min_conviction, min_score, direction_filter):
                    exit_reason = "score_drop"
            elif exit_rule == "hold_days":
                if (day.date() - pos["entry_date"]).days >= hold_days:
                    exit_reason = "hold_days"
    if exit_reason is None:
        return pos, None

    pnl_pts = (price - pos["entry_price"]) * pos["direction"]
    trade = {
        "entry_date": pos["entry_date"], "exit_date": day.date(),
        "direction": "Long" if pos["direction"] > 0 else "Short",
        "entry_price": pos["entry_price"], "exit_price": price, "exit_reason": exit_reason,
        "entry_conviction": pos["entry_conviction"], "entry_score": pos["entry_score"],
        "holding_days": int((day.date() - pos["entry_date"]).days),
        "pnl": pnl_pts * size_mult * size,
        "pnl_pct": (pnl_pts / pos["entry_price"] * 100.0) if pos["entry_price"] else 0.0,
    }
    new_pos = None
    if not is_last and _passes(sig, min_conviction, min_score, direction_filter):
        new_pos = {"direction": sig["direction"], "entry_date": day.date(), "entry_price": price,
                   "entry_conviction": sig["conviction"], "entry_score": sig["score"]}
    return new_pos, trade


def _summarize(trades_df: pd.DataFrame, daily: pd.DataFrame, ccy: str) -> dict:
    if trades_df is None or trades_df.empty:
        return {"total_pnl": 0.0, "n_trades": 0, "win_rate": np.nan, "avg_win": np.nan,
                "avg_loss": np.nan, "profit_factor": np.nan, "max_drawdown": 0.0,
                "avg_holding_days": np.nan, "ccy": ccy}
    wins, losses = trades_df[trades_df["pnl"] > 0], trades_df[trades_df["pnl"] < 0]
    gross_win, gross_loss = float(wins["pnl"].sum()), float(-losses["pnl"].sum())
    cum = daily["cum_pnl"] if daily is not None and not daily.empty else pd.Series([0.0])
    return {
        "total_pnl": float(trades_df["pnl"].sum()),
        "n_trades": int(len(trades_df)),
        "win_rate": float(len(wins) / len(trades_df) * 100.0),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "profit_factor": (float(gross_win / gross_loss) if gross_loss > 0
                          else (np.inf if gross_win > 0 else np.nan)),
        "max_drawdown": float((cum.cummax() - cum).max()),
        "avg_holding_days": float(trades_df["holding_days"].mean()),
        "ccy": ccy,
    }


def _fx_multiplier(ticker: str, asof, buf_start, end_ts) -> tuple:
    """USD-per-1-local-currency-unit rate as of `asof` (frozen for the run), for non-USD FICC
    contracts — mirrors volbt.py's entry-date FX conversion exactly. (1.0, "USD") for USD/equities."""
    ccy = volbt.currency(ticker)
    if ccy == "USD":
        return 1.0, ccy
    src = volbt.FX_USD.get(ccy)
    if src:
        # Deep-backtest starts predate the 400-day live window, so try the deep store's
        # RAW '1' series first (actual traded rate at `asof` — never the roll-adjusted
        # one, whose historical levels are continuation fiction), then the live feed.
        for fetch in (lambda: deepstore.get_raw([src], start=buf_start, end=end_ts),
                      lambda: get_history([src], start=buf_start, end=end_ts)):
            try:
                fxh = fetch()
                if src in fxh.columns:
                    s = fxh[src].dropna()
                    s = s[s.index <= pd.Timestamp(asof)]
                    if len(s):
                        return float(s.iloc[-1]), ccy
            except Exception:
                pass
    return 1.0, ccy


def _load_history(scope: str, ticker: str, strategies: list, start, end):
    """(signal_hist, pnl_hist, vol_hist_or_None, tickers) for `scope`. `signal_hist` is what the
    strategies score on (yields for FICC fixed income); `pnl_hist` is always the true settlement/close
    PRICE, so P&L is never computed off a yield series. Pulls the full PAIRS ticker universe when
    Mean Reversion is among `strategies` (see module docstring)."""
    eq = scope == "equities"
    needs_volume = any(s in VOLUME_STRATEGIES for s in strategies)
    buf_start = pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
    end_ts = pd.Timestamp(end)

    if eq:
        tickers = [ticker]
        days_back = max((pd.Timestamp.today().normalize() - buf_start).days, 1)
        sessions = max(int(days_back * 7 / 5) + 30, 504)
        close, vol = yfin.get_ohlcv(tickers, sessions=sessions)
        signal_hist = pnl_hist = close.reindex(columns=tickers)
        vol_hist = vol.reindex(columns=tickers) if needs_volume else None
        deep_used = []
    else:
        pair_needed = "Mean Reversion" in strategies
        tickers = (sorted({t for p in universe.PAIRS for t in (p["a"], p["b"])} | {ticker})
                  if pair_needed else [ticker])
        signal_hist = get_history_ta(tickers, start=buf_start, end=end_ts)
        pnl_hist = get_history(tickers, start=buf_start, end=end_ts)
        vol_hist = get_volume_history(tickers, start=buf_start, end=end_ts) if needs_volume else None
        # Deep-store upgrade: ~10y roll-adjusted '1'-generic history where the store has
        # it (falls straight through untouched when the store is absent — VPS / demo).
        deep_used, pnl_hist, signal_hist, vol_hist = deepstore.overlay(
            tickers, buf_start, end_ts, pnl_hist, signal_hist, vol_hist)

    if ticker not in pnl_hist.columns or pnl_hist[ticker].dropna().empty:
        raise ValueError(f"no price history for {ticker}")
    return signal_hist, pnl_hist, vol_hist, tickers, buf_start, end_ts, deep_used


def _setup(scope: str, ticker: str, strategies: list, start, end, exit_rule, hold_days):
    _validate_exit(exit_rule, hold_days)
    signal_hist, pnl_hist, vol_hist, tickers, buf_start, end_ts, deep_used = _load_history(
        scope, ticker, strategies, start, end)
    idx = pnl_hist[ticker].dropna().index
    days = idx[(idx >= pd.Timestamp(start)) & (idx <= end_ts)]
    if len(days) < 2:
        raise ValueError(f"no trading dates between {start} and {end}")
    eq = scope == "equities"
    warnings = []
    if ticker in deep_used:
        warnings.append("Deep history in use — prices are the '1' first-generic continuation "
                        "series, roll-adjusted (difference/panama, anchored to the latest "
                        "settle). Point moves and P&L are exact across rolls; absolute levels "
                        "before the most recent roll differ from what the screen showed at "
                        "the time.")
    if (days[0] - pd.Timestamp(start)).days > 7:
        warnings.append(f"History in the current data mode only reaches back to "
                        f"{days[0].date()} — the backtest starts there, not at your "
                        f"requested {start}.")
    if eq:
        size_mult, ccy = 1.0, "USD"
    else:
        pv = volbt.point_value(ticker)
        if pv <= 0:
            warnings.append(f"No point value on file for {ticker} — P&L will read as 0. "
                            "Set one in the Vol Backtester's Advanced panel (shared table).")
        fx, ccy = _fx_multiplier(ticker, days[0], buf_start, end_ts)
        if ccy != "USD" and fx == 1.0:
            warnings.append(f"{ticker} is {ccy}-denominated but no {ccy}USD rate was available — "
                            "treating 1 " + ccy + " = 1 USD.")
        elif ccy != "USD":
            warnings.append(f"{ticker} is {ccy}-denominated — P&L converted to USD at the "
                            f"backtest-start rate {ccy}USD {fx:.4f} (frozen for the run).")
        size_mult = pv * fx
    sign = -1 if (not eq and universe.is_fixed_income(ticker)) else 1
    return signal_hist, pnl_hist, vol_hist, days, size_mult, ccy, sign, warnings


def run_backtest(scope: str, ticker: str, strategies: list | None = None, *,
                 min_conviction: float = 0.0, min_score: float = 0.0, direction: str = "both",
                 start: date, end: date, exit_rule: str = "reversal", hold_days: int | None = None,
                 stop_pct: float | None = None, take_pct: float | None = None,
                 size: float = 1.0) -> Result:
    """Backtest one product on a picked SET of strategies, scored exactly like the TA hub /
    report: hub-trigger flagging, within-axis harmonic de-duplication (two methods in one axis
    don't both count in full), signed cross-axis sum. `strategies=None` falls back to the book's
    saved confluence set; a single-element list backtests that one method on its own.

    `direction`: "both" | "long" | "short" — restricts which side of a signal is ever taken.
    `exit_rule`: "reversal" (signal flips) | "score_drop" (no longer clears the entry bar) |
    "hold_days" (fixed N trading-day hold). `stop_pct`/`take_pct` are an OPTIONAL overlay that can
    close a trade before the primary rule does, checked first every day."""
    chosen = set(strategies or tascore.confluence_set(scope))
    if "Mean Reversion" in chosen and pair_for(ticker) is None:
        chosen.discard("Mean Reversion")     # pair-based — can't feed a non-pair product's score
    strategies = [s for s in tascore.TA_STRATEGIES if s in chosen]
    if not strategies:
        raise ValueError("pick at least one strategy to score on")

    signal_hist, pnl_hist, vol_hist, days, size_mult, ccy, sign, warnings = _setup(
        scope, ticker, strategies, start, end, exit_rule, hold_days)

    def _sig_on(d):
        hist_slice = signal_hist.loc[:d]
        vol_slice = vol_hist.loc[:d] if vol_hist is not None else None
        raw = _raw_frames(ticker, strategies, hist_slice, vol_slice)
        return _score_set(raw, strategies, ticker, sign)

    # signal-space series (yields for FICC fixed income) — what the chart shows so trades can be
    # read against the series the strategies actually scored on, not just the futures price
    sig_series = signal_hist[ticker] if ticker in signal_hist.columns else pd.Series(dtype=float)

    pos, rows, trades, realized = None, [], [], 0.0
    for i, d in enumerate(days):
        price = float(pnl_hist.loc[d, ticker])
        is_last = i == len(days) - 1
        sig = _sig_on(d)
        pos, trade = _step(pos, sig, price, d, is_last, exit_rule=exit_rule, hold_days=hold_days,
                           stop_pct=stop_pct, take_pct=take_pct, min_conviction=min_conviction,
                           min_score=min_score, direction_filter=direction,
                           size_mult=size_mult, size=size)
        if trade:
            trades.append(trade)
            realized += trade["pnl"]
        unreal = (price - pos["entry_price"]) * pos["direction"] * size_mult * size if pos else 0.0
        rows.append({"date": d, "price": price,
                    "signal_level": float(sig_series.get(d, np.nan)),
                    "score": (sig["score"] if sig else np.nan),
                    "conviction": (sig["conviction"] if sig else np.nan),
                    "position": (pos["direction"] if pos else 0),
                    "unrealized_pnl": unreal, "cum_pnl": realized + unreal})

    daily = pd.DataFrame(rows).set_index("date")
    trades_df = pd.DataFrame(trades)
    summary = _summarize(trades_df, daily, ccy)
    # the run's own parameters, so the page renders results from what actually RAN, not from
    # whatever the widgets say by the time the user reads them
    summary.update({
        "scope": scope, "ticker": ticker, "strategies": list(strategies),
        "min_conviction": float(min_conviction), "min_score": float(min_score),
        "direction": direction, "exit_rule": exit_rule,
        "hold_days": (int(hold_days) if hold_days else None),
        "stop_pct": (float(stop_pct) if stop_pct else 0.0),
        "take_pct": (float(take_pct) if take_pct else 0.0),
        "start": days[0].date(), "end": days[-1].date(),
        "fi": bool(scope != "equities" and universe.is_fixed_income(ticker)),
    })
    vol_series = (vol_hist[ticker].dropna()
                  if vol_hist is not None and ticker in vol_hist.columns else None)
    return Result(daily=daily, trades=trades_df, summary=summary, warnings=warnings,
                  series=sig_series.dropna(), volume=vol_series)


def compare_strategies(scope: str, ticker: str, *, min_conviction: float = 0.0,
                       min_score: float = 0.0, direction: str = "both", start: date, end: date,
                       exit_rule: str = "reversal", hold_days: int | None = None,
                       stop_pct: float | None = None, take_pct: float | None = None,
                       size: float = 1.0) -> pd.DataFrame:
    """One row per individual strategy PLUS the confluence composite, each independently backtested
    on `ticker` with the same rules — computes every strategy's raw signal ONCE per day (shared
    across all the variants' state machines) rather than paying for N separate day-loops. Every
    variant scores through the same hub pipeline as run_backtest, so a one-strategy backtest and
    that strategy's compare row always agree. Ranked by P&L."""
    strategies = strategies_for_scope(scope)
    if "Mean Reversion" in strategies and pair_for(ticker) is None:
        strategies = [s for s in strategies if s != "Mean Reversion"]
    variants = strategies + ["Confluence"]
    conf_set = [s for s in tascore.confluence_set(scope)
                if s in set(strategies)]     # drops e.g. Mean Reversion on a non-pair product
    all_needed = [s for s in tascore.TA_STRATEGIES if s in set(strategies) | set(conf_set)]

    signal_hist, pnl_hist, vol_hist, days, size_mult, ccy, sign, warnings = _setup(
        scope, ticker, all_needed, start, end, exit_rule, hold_days)

    states = {v: None for v in variants}
    trades = {v: [] for v in variants}
    realized = {v: 0.0 for v in variants}
    daily_rows = {v: [] for v in variants}

    for i, d in enumerate(days):
        price = float(pnl_hist.loc[d, ticker])
        is_last = i == len(days) - 1
        hist_slice = signal_hist.loc[:d]
        vol_slice = vol_hist.loc[:d] if vol_hist is not None else None

        raw = _raw_frames(ticker, all_needed, hist_slice, vol_slice)
        for v in variants:
            sig = _score_set(raw, conf_set if v == "Confluence" else [v], ticker, sign)
            pos, trade = _step(states[v], sig, price, d, is_last, exit_rule=exit_rule,
                               hold_days=hold_days, stop_pct=stop_pct, take_pct=take_pct,
                               min_conviction=min_conviction, min_score=min_score,
                               direction_filter=direction, size_mult=size_mult, size=size)
            states[v] = pos
            if trade:
                trades[v].append(trade)
                realized[v] += trade["pnl"]
            unreal = (price - pos["entry_price"]) * pos["direction"] * size_mult * size if pos else 0.0
            daily_rows[v].append({"date": d, "cum_pnl": realized[v] + unreal})

    out = []
    for v in variants:
        tdf = pd.DataFrame(trades[v])
        ddf = pd.DataFrame(daily_rows[v]).set_index("date") if daily_rows[v] else pd.DataFrame()
        out.append({"strategy": v, **_summarize(tdf, ddf, ccy)})
    result = pd.DataFrame(out).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    result.attrs["warnings"] = warnings
    return result
