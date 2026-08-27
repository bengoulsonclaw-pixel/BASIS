"""Technical Analysis report — weekly (or ad-hoc).

Opens with the cross-strategy conviction leaderboard and a summary table, then pulls the
highest-conviction technical setups out of the whole TA module (via the
tascore cross-strategy conviction engine), and for each one draws a single chart with
every flagging indicator overlaid, then a neutral, observational paragraph explaining
what the technicals are showing. Fixed income reads as yields (get_history_ta).

CLIENT-SAFE: neutral, observational language only (no buy/sell/recommend) — it screens as
"constructive" / "cautious", carries objective/invalidation levels, and ends with the house
sales-commentary disclaimer. Ranking is long/short under the hood; the wording stays neutral.

Run standalone (the app calls it as a subprocess):
    python src/convreport.py opportunities.parquet out.pdf --asof "2026-07-13" --top 5
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

# Make the `src` package importable (strategies / datafeed use relative imports).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportkit import data_uri, png, pretty_date, RICH, CHEAP, NEUTRAL   # noqa: E402 (sets matplotlib Agg)
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from jinja2 import Environment, FileSystemLoader             # noqa: E402

from src import tascore                                      # noqa: E402
from src import universe as u                                # noqa: E402
from src.datafeed import get_history_ta                      # noqa: E402
from src.strategies.trend import trend_chart_data            # noqa: E402
from src.strategies.ma_crossover import crossover_chart_data as _cross_cd  # noqa: E402
from src.strategies.ma_crossover_swing import crossover_chart_data as _swing_cd  # noqa: E402
from src.strategies.flag_breakout import flag_chart_data      # noqa: E402
from src.strategies.support_resistance import sr_chart_data   # noqa: E402
from src.strategies.fibonacci import fib_chart_data           # noqa: E402
from src.strategies.breakout_retest import retest_chart_data  # noqa: E402
from src.strategies.momentum import momentum_chart_data       # noqa: E402
from src.strategies.bollinger import bollinger_chart_data     # noqa: E402
from src.strategies.elliott_wave import elliott_chart_data    # noqa: E402
from src.strategies.ichimoku import ichimoku_chart_data       # noqa: E402
from src.strategies.obv import obv_chart_data                 # noqa: E402
from src.strategies.mfi import mfi_chart_data                 # noqa: E402
from src.strategies.donchian import donchian_chart_data       # noqa: E402
from src.strategies.aroon import aroon_chart_data             # noqa: E402
import matplotlib.ticker as mticker                           # noqa: E402

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"

NAVY = "#0B3D91"
GOLD = "#C9A227"

# Which strategy contributes which chart_data (only those with a useful info/overlay).
_CHART_FN = {
    "Trend": trend_chart_data, "MA Crossover": _cross_cd, "MA Swing": _swing_cd,
    "Flag Breakout": flag_chart_data, "Support & Resistance": sr_chart_data,
    "Fibonacci Retracement": fib_chart_data, "Breakout & Retest": retest_chart_data,
    "Momentum (RSI/MACD)": momentum_chart_data, "Bollinger Squeeze": bollinger_chart_data,
    "Elliott Wave": elliott_chart_data, "Ichimoku Cloud": ichimoku_chart_data,
    "On-Balance Volume": obv_chart_data, "Money Flow Index": mfi_chart_data,
    "Donchian Channel": donchian_chart_data, "Aroon": aroon_chart_data,
}


# ── history source: FICC datafeed by default, or an injected equity store ──────────────────────
# The report normally pulls its chart history from the FICC datafeed (get_history_ta → yields for
# fixed income). The Equities TA report runs the SAME engine over the equity universe by injecting a
# cached yfinance OHLCV store here: `close` + `volume` frames (date × Bloomberg-style ticker) feed
# every strategy's chart_data, and `meta` ({ticker: {'name','sector',…}}) supplies the sector label.
# When set, is_fixed_income is forced off (equities are prices, never yields) and $-risk falls back
# to % (no futures point value). FICC leaves this None and nothing below changes.
_HIST = {"close": None, "volume": None, "meta": None}


def set_history_source(close=None, volume=None, meta=None) -> None:
    """Point the report at an injected OHLCV store (equities) instead of the FICC datafeed."""
    _HIST["close"], _HIST["volume"], _HIST["meta"] = close, volume, meta


def _pf_series(tk: str):
    """The price/yield series for tk — from the injected equity store if set, else FICC yields."""
    c = _HIST["close"]
    if c is not None:
        return c[tk].dropna() if tk in c.columns else pd.Series(dtype=float)
    return get_history_ta([tk])[tk].dropna()


def _is_fi(tk) -> bool:
    """Fixed income only in FICC mode — an injected equity store is always priced, never yields."""
    return _HIST["close"] is None and u.is_fixed_income(str(tk))


def _sector_of(tk: str) -> str:
    """Sector/asset label — from the injected equity meta if set, else the FICC universe."""
    m = _HIST["meta"]
    if m is not None:
        return ((m.get(tk) or {}).get("sector")) or "—"
    return u.asset(tk) or "—"


def _call_chart(fn, tk: str, hist1, vol1):
    """Call a strategy's *_chart_data, feeding the injected equity history/volume when the function
    accepts them (all take `history`; only OBV/MFI take `volume`). In FICC mode hist1/vol1 are None
    and each function falls back to its own datafeed pull, exactly as before."""
    params = inspect.signature(fn).parameters
    kw = {}
    if hist1 is not None and "history" in params:
        kw["history"] = hist1
    if vol1 is not None and "volume" in params:
        kw["volume"] = vol1
    return fn(tk, **kw)


# ── gather each flagging strategy's read for one product ────────────────────
def _gather(tk: str, strats: set) -> dict:
    """{strategy: {'cdata':..., 'info':...}} for the strategies that flagged this product."""
    out = {}
    hist1 = vol1 = None                                 # equity mode: feed the injected OHLCV store
    c = _HIST["close"]
    if c is not None and tk in c.columns:
        hist1 = pd.DataFrame({tk: c[tk]})
        v = _HIST["volume"]
        if v is not None and tk in v.columns:
            vol1 = pd.DataFrame({tk: v[tk]})
    for s in strats:
        fn = _CHART_FN.get(s)
        if fn is None:
            continue
        try:
            cdata, info = _call_chart(fn, tk, hist1, vol1)
        except Exception:
            cdata, info = None, None
        if info is not None:
            out[s] = {"cdata": cdata, "info": info}
    return out


def _levels(gathered: dict) -> dict:
    """Aggregate the actionable levels (objective / invalidation / reward:risk) from the
    strategies that carry them — prefer Flag Breakout, then Fibonacci."""
    for s in ("Flag Breakout", "Fibonacci Retracement", "Elliott Wave", "Donchian Channel"):
        info = (gathered.get(s) or {}).get("info") or {}
        tgt, stop, rr = info.get("target"), info.get("stop"), info.get("rr")
        if tgt is not None and stop is not None and np.isfinite(tgt) and np.isfinite(stop):
            return {"source": s, "target": float(tgt), "stop": float(stop),
                    "rr": float(rr) if rr is not None and np.isfinite(rr) else None}
    return {}


def _struct_levels(pf, net_dir: int) -> dict:
    """Confirmation trigger / objective / invalidation from the price (or yield) series itself —
    real, observable swing pivots, never a fabricated target. The nearest overhead/underfoot pivot
    a close beyond which would extend the move is the trigger; the recent swing on the other side
    is the structural invalidation; a prior swing further out (only where there is genuine room to
    it) is the next objective. Used to fill in whatever a chart pattern (Flag / Fibonacci) doesn't
    already supply, so every pick carries a where-it-confirms / where-it-fails read."""
    try:
        s = pf.dropna()
        if len(s) < 30:
            return {}
        px = float(s.iloc[-1])
        hi_n, lo_n = float(s.tail(20).max()), float(s.tail(20).min())
        hi_f, lo_f = float(s.tail(120).max()), float(s.tail(120).min())
    except Exception:
        return {}
    out = {}
    if net_dir > 0:                                    # constructive: extends up, breaks down
        out["stop"] = lo_n
        overhead = sorted(h for h in (hi_n, hi_f) if h > px * 1.002)
        if overhead:
            out["trigger"] = overhead[0]               # nearest pivot to clear
            if len(overhead) > 1 and overhead[-1] > overhead[0] * 1.01:
                out["target"] = overhead[-1]           # a DISTINCT higher pivot = objective
    else:                                              # cautious: extends down, breaks up
        out["stop"] = hi_n
        below = sorted((l for l in (lo_n, lo_f) if l < px * 0.998), reverse=True)
        if below:
            out["trigger"] = below[0]
            if len(below) > 1 and below[-1] < below[0] * 0.99:
                out["target"] = below[-1]
    return out


# A reward-to-risk ratio is only meaningful when BOTH legs are a tradeable distance from spot.
# Below this (as a % of price / of yield for FI) the leg is inside the noise — an objective
# 0.4bp above spot or a stop 0.014% away produce "0.0:1" and "800:1" respectively, both of
# which a client notices in a document carrying the compliance disclaimer.
RR_MIN_LEG_PCT = 0.25


def _full_levels(gathered: dict, pf, net_dir: int) -> dict:
    """The report's actionable levels: prefer a pattern's measured move (Flag / Fibonacci) for the
    objective + invalidation, fall back to structural swing pivots, and always add a confirmation
    trigger. Every figure is an objective read of the chart — no recommendation."""
    lv = dict(_levels(gathered))                       # pattern measured move, if any
    try:
        px = float(pf.dropna().iloc[-1])
    except Exception:
        px = None
    # Drop a pattern's levels when they contradict the pick's NET direction — e.g. a bullish
    # Fibonacci measured move on a product that nets cautious (other strategies outweigh it). Left
    # in, the objective/invalidation sit on the wrong side of price and the scenario reads backwards.
    if px is not None and lv.get("target") is not None and lv.get("stop") is not None:
        if ((lv["target"] > px) != (net_dir > 0)) or ((lv["stop"] < px) != (net_dir > 0)):
            lv = {}                                    # mismatched → fall back to structural pivots
    st = _struct_levels(pf, net_dir)
    for k in ("stop", "target", "trigger"):
        if lv.get(k) is None and st.get(k) is not None:
            lv[k] = st[k]
    # R:R always from TODAY's price (a pattern's own rr is struck at detection), so the quoted
    # ratio exactly equals the reward-to-objective ÷ risk-to-stop distances shown beside it.
    if px is not None and lv.get("target") is not None and lv.get("stop") is not None:
        r0 = abs(px - lv["stop"])
        reach = abs(lv["target"] - px)
        # Both legs must be a real distance before a RATIO of them means anything. An
        # objective 0.4bp above spot (inside the bid/ask) printed "a reward-to-risk of
        # roughly 0.0:1", and a stop 0.014% away printed 800:1 — 17 of 83 live picks were
        # above 20:1 (2026-08-26). Below the floor the ratio is dropped, not rendered: the
        # prose already has a no-rr branch, and no number beats an absurd one in a client PDF.
        _floor = abs(px) * RR_MIN_LEG_PCT / 100.0
        if r0 > _floor and reach > _floor:
            lv["rr"] = reach / r0
    return lv


def _risk(tk, entry, stop) -> dict:
    """Risk to the invalidation level. The **% move** to the stop is scale-invariant and always
    reported. The **per-contract $** is added ONLY where the volbt point-value reconciles with
    the live data scale — i.e. `point_value × price` implies a plausible single-contract USD
    notional. FX, cents-quoted and non-USD contracts (whose live data scale doesn't match the
    table) and fixed income (levels are yields → need a DV01) fall back to the % alone, so a
    mis-scaled dollar figure is never shown."""
    out = {"pct": None, "usd": None}
    try:
        entry, stop = float(entry), float(stop)
    except (TypeError, ValueError):
        return out
    if not (np.isfinite(entry) and np.isfinite(stop)) or entry == 0:
        return out
    # A distance inside the noise is not a level worth quoting — the Euro-Bobl objective sat
    # 0.4bp from spot and rendered as "reward to objective 0bp". Same floor the R:R uses, so
    # the figure and the ratio appear and disappear together.
    if abs(entry - stop) < abs(entry) * RR_MIN_LEG_PCT / 100.0:
        return out
    if _is_fi(tk):
        # FI levels are YIELDS, so a "% of entry" is a percentage OF THE RATE, not of the
        # futures price — and it lands in the same leaderboard column as genuine price
        # percentages. Live example (2026-08-26): Euro-Schatz printed "11.1% away" for a
        # 31.2bp stop that is ~0.6% of the 105.475 futures price, next to "WTI 28.5% away"
        # which IS a price move. Rates ideas read 15-20x riskier than they are, and STIRs
        # the same amount safer. Quote FI risk in basis points, which needs no conversion
        # and is what a rates client asks for anyway.
        out["bp"] = abs(entry - stop) * 100.0
        return out
    out["pct"] = abs(entry - stop) / abs(entry) * 100.0
    try:
        from src.volbt import point_value
        pv = point_value(tk)
    except Exception:
        pv = 0.0
    if pv and 8_000 <= pv * abs(entry) <= 1_500_000:   # point-value reconciles with the data scale
        out["usd"] = round(abs(entry - stop) * pv)
    return out


# "New this week" baseline — the last weekly run's picks, updated only on the scheduled
# (--baseline) run so ad-hoc previews from the app don't reset what counts as "new".
_BASELINE = ROOT / "data" / "signals" / "conv_baseline.json"


def _load_baseline() -> set:
    try:
        return set(json.loads(_BASELINE.read_text(encoding="utf-8")).get("instruments", []))
    except Exception:
        return set()


def _save_baseline(picks) -> None:
    try:
        _BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE.write_text(json.dumps({"instruments": [p["instruments"] for p in picks]}, indent=2),
                             encoding="utf-8")
    except Exception:
        pass


# ── neutral, observational prose ────────────────────────────────────────────
_SHORT = {"Trend": "Trend", "MA Crossover": "MA crossover", "MA Swing": "MA swing",
          "Flag Breakout": "flag/pennant", "Support & Resistance": "support/resistance",
          "Fibonacci Retracement": "Fibonacci", "Breakout & Retest": "breakout-retest",
          "Momentum (RSI/MACD)": "RSI/MACD momentum", "Bollinger Squeeze": "Bollinger squeeze",
          "Elliott Wave": "Elliott wave", "Ichimoku Cloud": "Ichimoku",
          "On-Balance Volume": "OBV", "Money Flow Index": "money flow"}


def _phrase(strategy: str, info: dict, is_fi: bool) -> str:
    """One neutral clause describing what this strategy is reading."""
    g = info.get
    try:
        if strategy == "Trend":
            return f"a {g('mom', 0) * 100:+.0f}% three-month trend"
        if strategy in ("MA Crossover", "MA Swing"):
            st = str(g("state", "")).lower()
            per = "50/200-day" if strategy == "MA Crossover" else "20/50-day"
            cross = "a golden-cross alignment" if "golden" in st else (
                "a death-cross alignment" if "death" in st else "a moving-average alignment")
            return f"{cross} of the {per} averages"
        if strategy == "Support & Resistance":
            near = g("nearest")
            kind = "support" if g("direction", 0) > 0 else "resistance"
            return (f"price holding tested {kind} near {near:g}" if near is not None
                    and np.isfinite(near) else f"price against a tested {kind} level")
        if strategy == "Fibonacci Retracement":
            r = g("nearest_ratio", 0) * 100
            near = g("nearest")
            return (f"a reaction at the {r:.1f}% Fibonacci level ({near:g})" if near is not None
                    and np.isfinite(near) else f"a reaction at the {r:.1f}% Fibonacci level")
        if strategy == "Flag Breakout":
            return f"a {str(g('type', 'flag')).lower()} at {g('readiness', 0):.0f}/100 breakout readiness"
        if strategy == "Breakout & Retest":
            lv = g("level")
            return (f"a retest of the {lv:g} level (role reversed)" if lv is not None
                    and np.isfinite(lv) else "a retest of a broken level")
        if strategy == "Momentum (RSI/MACD)":
            div = str(g("divergence", "none"))
            divtxt = f", {div} RSI divergence" if div and div != "none" else ""
            return f"RSI at {g('rsi', 0):.0f}{divtxt}"
        if strategy == "Bollinger Squeeze":
            return f"bands compressed at the {g('bandwidth_pctl', 0):.0f}th percentile (a squeeze)"
        if strategy == "Elliott Wave":
            ph = {"W3": "wave 3 of a developing five-wave sequence",
                  "W5": "wave 5 of a maturing five-wave sequence",
                  "done": "a completed five-wave sequence, favouring a corrective phase"}
            return (f"an Elliott count in {ph.get(str(g('phase', '')), 'an impulse')} "
                    f"(wave fit {g('fit', 0):.0f}/100)")
        if strategy in ("On-Balance Volume", "Money Flow Index"):
            lbl = str(g("label", "a volume read")).lower()
            note = str(g("note", "")).strip()
            return f"{lbl} ({note})" if note else lbl
        if strategy == "Ichimoku Cloud":
            lbl = str(g("label", "an Ichimoku read")).lower()
            return f"an Ichimoku {lbl}" if not lbl.startswith(("cloud", "bull", "bear")) else lbl
        if strategy == "Donchian Channel":
            w = int(g("window", 20))
            pos, d = g("pos"), g("direction", 0)
            if d > 0:
                return (f"a fresh {w}-day channel high" if pos is not None and pos >= 1
                        else f"pressing the top of its {w}-day range")
            if d < 0:
                return (f"a fresh {w}-day channel low" if pos is not None and pos <= 0
                        else f"pressing the bottom of its {w}-day range")
            return f"a stretch across its {w}-day price channel"
        if strategy == "Aroon":
            osc = g("osc", g("metric", 0)) or 0
            side = "up" if osc >= 0 else "down"
            return (f"Aroon confirming a {side}trend (Up {g('up', 0):.0f} / Down {g('down', 0):.0f})")
    except Exception:
        pass
    return _SHORT.get(strategy, strategy)


def _prose(pick: dict, gathered: dict, lv: dict) -> str:
    """A neutral paragraph: what's stacking up, the specifics, and the where-it-confirms /
    where-it-fails / objective scenario read (all objective observations, never a recommendation)."""
    word = "constructive" if pick["net_dir"] > 0 else "cautious"
    is_fi = _is_fi(pick["instruments"])
    strats = [s for s, _d, _st in pick["tags"]]
    names = ", ".join(_SHORT.get(s, s) for s in strats)
    clauses = [c for c in (_phrase(s, (gathered.get(s) or {}).get("info") or {}, is_fi)
                           for s in strats) if c]
    # de-dup while preserving order, keep the first 4 so the sentence stays readable
    seen, uniq = set(), []
    for c in clauses:
        if c not in seen:
            seen.add(c); uniq.append(c)
    body = "; ".join(uniq[:4])

    fi_note = ""
    if is_fi:
        fi_note = (" This is a **yield-based** read — a constructive screen means yields are biased "
                   "higher (the future lower), and the reverse for cautious.") if pick["net_dir"] > 0 else (
                   " This is a **yield-based** read — a cautious screen means yields are biased lower "
                   "(the future higher).")

    up = pick["net_dir"] > 0
    trig, stop, tgt, rr = lv.get("trigger"), lv.get("stop"), lv.get("target"), lv.get("rr")
    _rk_d, _rw_d = (lv.get("risk") or {}), (lv.get("reward") or {})

    def _away(d):
        """Distance in the product's OWN dimension — basis points for fixed income (whose
        levels are yields), percent of price for everything else. Mixing the two in one
        leaderboard made rates ideas read ~15-20x riskier than they were."""
        if d.get("bp") is not None:
            return f"{d['bp']:.0f}bp", d["bp"]
        if d.get("pct") is not None:
            return f"{d['pct']:.1f}%", d["pct"]
        return None, None

    risk_txt, risk = _away(_rk_d)
    parts = []
    if trig is not None and np.isfinite(trig):
        parts.append(f"A daily close {'above' if up else 'below'} **{trig:g}** would extend the "
                     f"{word} setup.")
    if stop is not None and np.isfinite(stop):
        rk = f" ({risk_txt} away)" if risk_txt else ""
        parts.append(f"A move {'back below' if up else 'back above'} **{stop:g}**{rk} would "
                     f"negate it.")
    if tgt is not None and np.isfinite(tgt):
        rew_txt, rew = _away(_rw_d)
        src = lv.get("source")
        basis = f", measured from the {_SHORT.get(src, src)} setup" if src else ""
        if rew and risk and rr and np.isfinite(rr):
            parts.append(f"The next objective sits near **{tgt:g}**{basis} — **{rew_txt}** away, "
                         f"against the **{risk_txt}** to the invalidation, a reward-to-risk of "
                         f"roughly **{rr:.1f}:1**.")
        else:
            rrtxt = f" — roughly **{rr:.1f}:1** against the invalidation" if rr and np.isfinite(rr) else ""
            parts.append(f"The next objective sits near **{tgt:g}**{rrtxt}{basis}.")
    lvl_txt = (" " + " ".join(parts)) if parts else ""

    return (f"**{pick['market']}** screens **{word}** — flagged by **{pick['n']} of "
            f"{pick['total']}** technical strategies ({names}), conviction "
            f"**{pick['conviction']:.0f}/100**. The technicals show {body}.{lvl_txt}{fi_note}")


def _panels_for(gathered):
    """(panel names, has_momentum, has_mfi, has_obv) — which indicator sub-panels a pick's chart
    will carry. Used by BOTH the figure layout and the page-fit estimate, so the two can't drift."""
    def _has(name, col):
        cd = (gathered.get(name) or {}).get("cdata")
        return cd is not None and not getattr(cd, "empty", True) and col in getattr(cd, "columns", [])
    mom = _has("Momentum (RSI/MACD)", "rsi")
    mfi = _has("Money Flow Index", "mfi")
    obv = _has("On-Balance Volume", "obv")
    aroon = _has("Aroon", "aroon_up")
    panels = ((["osc"] if (mom or mfi) else []) + (["macd"] if mom else [])
              + (["obv"] if obv else []) + (["aroon"] if aroon else []))
    return panels, mom, mfi, obv


# Page-fit budget, in inches of the printed content column (A4 less the 0.3in @page margins,
# with a little slack). Used to decide how many picks can share the closing page WITH the
# sales-commentary disclaimer, so it never overflows onto a page of its own.
_PAGE_IN, _DISCLAIMER_IN = 10.85, 2.45   # sized for the full XP compliance disclaimer (2026-08)


def _est_section_in(pick, n_panels: int) -> float:
    """Rough printed height of one pick block: chart (figure height scaled to the 5.92in content
    column) + banner + stats + write-up. Deliberately a slight over-estimate — erring tall costs
    at worst one page, erring short pushes the disclaimer off the closing page."""
    fig_h = (2.5 + 0.6 * n_panels) if n_panels else 3.0
    chart = fig_h * (5.92 / 6.6)
    body = re.sub(r"<[^>]+>", "", pick.get("prose", "") or "")
    # Calibrated to .summary in _report_style.html — RE-DERIVE BOTH CONSTANTS IF THAT SIZE MOVES.
    # At 9.5pt/1.4 in the 5.92in column: ~89 chars per line, 9.5pt x 1.4 = 13.3pt = 0.185in a line.
    prose = 0.185 * max(3.0, len(body) / 89.0)
    return chart + 0.22 + 0.40 + prose + 0.25           # banner + stats(may wrap) + prose + margins


# ── the multi-indicator chart (every flagging line on one price/yield axis) ──
def _chart_png(tk: str, gathered: dict, net_dir: int, pf=None, lv=None) -> str:
    if pf is None:
        pf = _pf_series(tk)
    if pf is None or getattr(pf, "empty", True):
        return ""
    if lv is None:                                     # standalone use: same levels the prose cites
        lv = _full_levels(gathered, pf, net_dir)
    win = pf.tail(180)
    strats = set(gathered)
    is_fi = _is_fi(tk)
    # Oscillators can't share the price axis, so flagged indicator series get their own
    # sub-panels under the price chart: RSI/MFI share one 0–100 panel, MACD and OBV get
    # their own. Only the panels a pick actually flagged are drawn.
    panels, _mom, _hasmfi, _hasobv = _panels_for(gathered)
    _mcd = (gathered.get("Momentum (RSI/MACD)") or {}).get("cdata")
    _fcd = (gathered.get("Money Flow Index") or {}).get("cdata")
    _ocd = (gathered.get("On-Balance Volume") or {}).get("cdata")
    if panels:                                         # compact so TWO pick sections fit per page
        fig, axs = plt.subplots(1 + len(panels), 1, figsize=(6.6, 2.5 + 0.6 * len(panels)),
                                sharex=True,
                                gridspec_kw={"height_ratios": [2.5] + [0.6] * len(panels),
                                             "hspace": 0.12})
        ax = axs[0]
        _sub = dict(zip(panels, axs[1:]))
    else:
        fig, ax = plt.subplots(figsize=(6.6, 3.0))
        _sub = {}

    # Ichimoku cloud (Kumo) + Tenkan/Kijun, when Ichimoku flagged — drawn FIRST so it sits behind
    # everything; the cloud carries its 26-session forward projection (extends the x-axis right).
    ici = (gathered.get("Ichimoku Cloud") or {}).get("info") or {}
    if ici.get("cloud"):
        try:
            cl = ici["cloud"]
            cx = np.array([pd.Timestamp(r["date"]) for r in cl])
            ca = np.array([r["a"] if r["a"] is not None else np.nan for r in cl], dtype=float)
            cb = np.array([r["b"] if r["b"] is not None else np.nan for r in cl], dtype=float)
            if np.isfinite(ca).any() and np.isfinite(cb).any():
                ax.fill_between(cx, ca, cb, where=ca >= cb, color=CHEAP, alpha=0.13,
                                linewidth=0, interpolate=True, label="Ichimoku cloud")
                ax.fill_between(cx, ca, cb, where=ca < cb, color=RICH, alpha=0.13,
                                linewidth=0, interpolate=True)
                ax.plot(cx, ca, color=CHEAP, lw=0.6, alpha=0.45)
                ax.plot(cx, cb, color=RICH, lw=0.6, alpha=0.45)
            for key, col, lbl in (("tenkan", "#00838F", "Tenkan"), ("kijun", "#C2185B", "Kijun")):
                rec = ici.get(key) or []
                xs = [pd.Timestamp(r["date"]) for r in rec if r.get("val") is not None]
                ys = [r["val"] for r in rec if r.get("val") is not None]
                if xs:
                    ax.plot(xs, ys, color=col, lw=1.0, alpha=0.9, label=lbl)
        except Exception:
            pass

    # moving averages (union of the windows the flagging MA/trend strategies use)
    windows = set()
    if "MA Crossover" in strats:
        windows |= {50, 200}
    if "MA Swing" in strats:
        windows |= {20, 50}
    if "Trend" in strats:
        windows |= {20, 100}
    _MA_C = {20: "#7E57C2", 50: "#1F77B4", 100: "#E08A00", 200: "#555555"}
    for w in sorted(windows):
        ma = pf.rolling(w).mean().reindex(win.index)
        ax.plot(win.index, ma, lw=1.0, color=_MA_C.get(w, "#999"), label=f"MA{w}")

    # Bollinger band
    if "Bollinger Squeeze" in strats:
        mid = pf.rolling(20).mean(); sd = pf.rolling(20).std()
        up = (mid + 2 * sd).reindex(win.index); lo = (mid - 2 * sd).reindex(win.index)
        ax.fill_between(win.index, lo, up, color="#B0BEC5", alpha=0.18, linewidth=0, label="Bollinger 2σ")

    # Donchian channel — the prior-N high/low bands whose break is the signal
    dcd = (gathered.get("Donchian Channel") or {}).get("cdata")
    if dcd is not None and not getattr(dcd, "empty", True):
        try:
            dch = dcd[dcd["date"] >= win.index[0]][["date", "upper", "lower"]].dropna()
            if not dch.empty:
                ax.plot(dch["date"], dch["upper"], ls="--", lw=1.3, color="#5C6BC0", label="Donchian")
                ax.plot(dch["date"], dch["lower"], ls="--", lw=1.3, color="#5C6BC0")
        except Exception:
            pass

    # flag channel + breakout + pole
    flag = (gathered.get("Flag Breakout") or {})
    fcd, fi = flag.get("cdata"), (flag.get("info") or {})
    if fcd is not None and not getattr(fcd, "empty", True) and fi:
        try:
            fch = fcd[["date", "upper", "lower", "breakout"]].dropna()
            fcol = CHEAP if fi.get("sign", 0) > 0 else RICH
            ax.fill_between(fch["date"], fch["lower"], fch["upper"], color=fcol, alpha=0.12, linewidth=0)
            ax.plot(fch["date"], fch["breakout"], ls="--", lw=1.6, color=fcol, label="breakout")
            pb, pt = fi.get("pole_base"), fi.get("pole_tip")
            if pb and pt:
                ax.plot([pb[0], pt[0]], [pb[1], pt[1]], lw=2.2, color="#B0B0B0")
        except Exception:
            pass

    # Elliott count: the counted pivots (purple, labelled 0..5), clipped to the chart window
    ewi = (gathered.get("Elliott Wave") or {}).get("info") or {}
    if ewi.get("pivots"):
        try:
            t0 = win.index[0]
            pts = [p for p in ewi["pivots"] if p["date"] >= t0]
            if len(pts) >= 2:
                ax.plot([p["date"] for p in pts], [p["price"] for p in pts],
                        lw=1.5, color="#7E57C2", alpha=0.85, marker="o", ms=3.2,
                        label="Elliott count", zorder=5)
                for p in pts:
                    ax.annotate(p["label"], (p["date"], p["price"]),
                                textcoords="offset points",
                                xytext=(0, 7 if p.get("kind") == "H" else -11),
                                fontsize=7, fontweight="bold", color="#5E35B1",
                                ha="center", zorder=6)
        except Exception:
            pass

    # horizontal levels: S/R (green/red), Fib key (gold), retest (gold dashed) — each labelled
    # at the right edge so the reader can tell the lines apart without a decoder ring.
    lvl_lbls = []                                      # (y, text, colour) for the edge labels
    def _hline(y, color, ls="-", lbl=None, lw=1.2, alpha=0.75, lblcolor=None):
        if y is not None and np.isfinite(y):
            ax.axhline(y, color=color, lw=lw, ls=ls, alpha=alpha)
            if lbl and not any(abs(y - y0) < 1e-12 and t == lbl for y0, t, _c in lvl_lbls):
                lvl_lbls.append((float(y), lbl, lblcolor or color))
    for L in ((gathered.get("Support & Resistance") or {}).get("info") or {}).get("levels", []) or []:
        kind = "support" if L.get("kind") == "support" else "resistance"
        # role-reversed levels (broken, now acting from the other side) are the stronger zones
        _hline(L.get("price"), CHEAP if kind == "support" else RICH,
               lbl=kind + (" (rev)" if L.get("flipped") else ""))
    fibi = (gathered.get("Fibonacci Retracement") or {}).get("info") or {}
    if fibi.get("levels"):
        # The whole retracement ladder (0–100%), not just the golden zone: the level the note
        # reads from — the nearest reaction level, or the trigger where it lands on a fib — in
        # dark gold, the rest in a light wash so the active one stands out.
        _trig, _near = (lv or {}).get("trigger"), fibi.get("nearest")
        def _fib_active(p):
            if _trig is not None and np.isfinite(_trig) and abs(p - _trig) <= 1e-4 * abs(_trig):
                return True
            return _near is not None and abs(p - _near) <= 1e-6 * max(abs(_near), 1e-9)
        for L in fibi["levels"]:
            p = L.get("price")
            if p is None or not np.isfinite(p):
                continue
            act = _fib_active(p)
            _hline(p, "#8B6914" if act else "#DCC988", lw=1.5 if act else 0.9,
                   alpha=0.95 if act else 0.6, lbl=f"Fib {L.get('ratio', 0) * 100:g}%",
                   lblcolor="#8B6914" if act else "#A98F35")
    _hline(((gathered.get("Breakout & Retest") or {}).get("info") or {}).get("level"), GOLD,
           ls="--", lbl="retest")

    # trigger / objective / invalidation — the SAME direction-checked levels the note cites
    if lv:
        _hline(lv.get("trigger"), NAVY, ls="--", lbl="trigger")
        _hline(lv.get("target"), CHEAP, ls=":", lbl="objective")
        _hline(lv.get("stop"), RICH, ls=":", lbl="invalidation")

    # price / yield line + today
    ax.plot(win.index, win.to_numpy(dtype=float), lw=1.9, color=NAVY, label="Yield" if is_fi else "Price")
    ax.scatter([win.index[-1]], [float(win.iloc[-1])], s=34,
               color=(CHEAP if net_dir > 0 else RICH), zorder=6)

    ax.set_ylabel("Yield (%)" if is_fi else "Price")
    ax.grid(True, color="#ECECEC", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # right-edge labels for the horizontal levels, nudged apart where they'd overlap
    if lvl_lbls:
        ymin, ymax = ax.get_ylim()
        gap = (ymax - ymin) * 0.048
        lvl_lbls.sort(key=lambda t: t[0])
        ys = []
        for y, _t, _c in lvl_lbls:
            ys.append(y if not ys or y - ys[-1] >= gap else ys[-1] + gap)
        over = ys[-1] - ymax                           # if the stack ran off the top, walk it back
        if over > 0:
            ys = [y - over for y in ys]
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        tr = ax.get_yaxis_transform()                  # x in axes coords, y in data coords
        for (y0, txt, col), y in zip(lvl_lbls, ys):
            ax.text(1.006, y, txt, transform=tr, fontsize=5.6, color=col, va="center", ha="left")

    ax.legend(fontsize=6.2, ncol=3, loc="best", frameon=True, framealpha=0.9, edgecolor="#ccc")

    # the indicator sub-panels (only what this pick actually flagged)
    if "osc" in _sub:                                  # RSI and/or MFI share the 0–100 panel
        axr = _sub["osc"]
        if _mom:
            d = _mcd[_mcd["date"] >= win.index[0]]
            axr.plot(d["date"], d["rsi"], lw=1.1, color="#7E57C2", label="RSI 14")
        if _hasmfi:
            d2 = _fcd[_fcd["date"] >= win.index[0]]
            axr.plot(d2["date"], d2["mfi"], lw=1.1, color="#00897B", label="MFI 14")
        axr.axhline(70, color=RICH, lw=0.8, ls=":", alpha=0.8)
        axr.axhline(30, color=CHEAP, lw=0.8, ls=":", alpha=0.8)
        if _hasmfi:                                    # MFI's wider 80/20 flow extremes
            axr.axhline(80, color=RICH, lw=0.6, ls=":", alpha=0.45)
            axr.axhline(20, color=CHEAP, lw=0.6, ls=":", alpha=0.45)
        axr.set_ylim(0, 100)
        axr.set_ylabel("RSI / MFI" if (_mom and _hasmfi) else ("MFI 14" if _hasmfi else "RSI 14"),
                       fontsize=7)
        if _mom and _hasmfi:
            axr.legend(fontsize=5.4, ncol=2, loc="upper left", frameon=False)
    if "macd" in _sub:
        axm = _sub["macd"]
        d = _mcd[_mcd["date"] >= win.index[0]]
        axm.bar(d["date"], d["macd_hist"], width=1.0, color="#B0BEC5", alpha=0.8, label="histogram")
        axm.plot(d["date"], d["macd"], lw=1.1, color=NAVY, label="MACD")
        axm.plot(d["date"], d["macd_signal"], lw=1.0, color="#E08A00", label="signal")
        axm.axhline(0, color="#888888", lw=0.7)
        axm.set_ylabel("MACD", fontsize=7)
        axm.legend(fontsize=5.4, ncol=3, loc="upper left", frameon=False)
    if "obv" in _sub:
        axo = _sub["obv"]
        d3 = _ocd[_ocd["date"] >= win.index[0]]
        axo.plot(d3["date"], d3["obv"], lw=1.2, color="#00695C")
        axo.set_ylabel("OBV", fontsize=7)
        axo.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _p: f"{v / 1e6:,.0f}M" if abs(v) >= 1e6 else f"{v / 1e3:,.0f}k"))
    if "aroon" in _sub:                                # Aroon Up/Down 0–100 (trend strength & freshness)
        axa = _sub["aroon"]
        _acd = (gathered.get("Aroon") or {}).get("cdata")
        if _acd is not None and not getattr(_acd, "empty", True):
            da = _acd[_acd["date"] >= win.index[0]]
            axa.plot(da["date"], da["aroon_up"], lw=1.1, color=CHEAP, label="Aroon Up")
            axa.plot(da["date"], da["aroon_down"], lw=1.1, color=RICH, label="Aroon Down")
        axa.axhline(50, color="#888888", lw=0.7, ls=":")
        axa.set_ylim(0, 100)
        axa.set_ylabel("Aroon", fontsize=7)
        axa.legend(fontsize=5.4, ncol=2, loc="upper left", frameon=False)
    for _a in _sub.values():
        _a.tick_params(labelsize=6.5)
        _a.grid(True, color="#ECECEC", lw=0.6)
        _a.set_axisbelow(True)
        for s in ("top", "right"):
            _a.spines[s].set_visible(False)

    ax.set_title("Every flagging indicator" + (" + indicator panels" if panels else " on one axis")
                 + (" (yields)" if is_fi else ""))
    return png(fig)


_LEAD_MUTED = {CHEAP: "#A5D6A7", RICH: "#EF9A9A"}   # green/red → light tints for the watch bars


def _leaderboard_png(rows, watch=None, min_score=0.0) -> str:
    """Diverging horizontal bar of cross-strategy conviction (Σ signed strength) per product —
    long (green, right) vs short (red, left), labelled with the number of flagging strategies. The
    report's PICKS are drawn solid; the 'ones to watch' (clean near-misses) are appended below in a
    faded tint, with a dashed line at the ±score bar so it reads as 'just short of the cut'.
    (Ported from the retired standalone TA report; the merged Technical Analysis report opens with it.)"""
    rows = list(rows)
    watch = list(watch or [])
    if not rows and not watch:
        return ""
    allrows = rows + watch                         # picks (strong) first, watch (weaker) beneath them
    faded = [False] * len(rows) + [True] * len(watch)
    allrows, faded = allrows[::-1], faded[::-1]    # barh draws bottom-up → strongest on top
    # compact per-row height so the leaderboard clears page 1 even with a full pick list + watch
    fig, ax = plt.subplots(figsize=(6.6, max(2.4, 0.205 * len(allrows))))
    ypos = list(range(len(allrows)))
    scores = [r["score"] for r in allrows]
    colors = [(_LEAD_MUTED.get(CHEAP if r["net_dir"] > 0 else RICH) if f
               else (CHEAP if r["net_dir"] > 0 else RICH)) for r, f in zip(allrows, faded)]
    ax.barh(ypos, scores, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color="#0A0A0A", lw=0.9, zorder=4)
    if min_score:                                  # the score half of the quality bar the watch names missed
        for x in (min_score, -min_score):
            ax.axvline(x, color="#9AA0A6", ls=(0, (4, 3)), lw=0.8, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["market"] for r in allrows], fontsize=7)
    for tick, f in zip(ax.get_yticklabels(), faded):
        tick.set_color("#9AA0A6" if f else "#333")     # grey out the watch names
    span = max((abs(s) for s in scores), default=1.0) or 1.0
    for i, (r, f) in enumerate(zip(allrows, faded)):
        s = r["score"]
        ax.text(s + (span * 0.015 if s >= 0 else -span * 0.015), i, f"{r['n']}×",
                va="center", ha="left" if s >= 0 else "right", fontsize=6,
                color="#9AA0A6" if f else "#333")
    ax.set_xlim(-span * 1.22, span * 1.22)
    # keep the legend on a SECOND line — appended inline it overran the axes and clipped at the column edge
    xlab = ("cross-strategy conviction (Σ signed strength)      ← short      long →      "
            "(label = # strategies)")
    if watch:
        xlab += "\nfaded = ones to watch" + ("      ·      dashed = quality bar (score)" if min_score else "")
    ax.set_xlabel(xlab, fontsize=6.5)
    ax.set_title("Conviction leaderboard")
    ax.grid(True, axis="x", color="#ECECEC", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return png(fig)


# ── selection: strongest constructive + cautious, plus the "ones to watch" near-misses ─
# Watch = non-conflicted markets that missed the quality bar but came close. Tunable: how many, and
# how close (within this many conviction points of the floor, and this fraction of the score floor).
_WATCH_MAX = 3
_WATCH_CONV_GAP = 15.0
_WATCH_SCORE_FRAC = 0.70


def select(sig_df: pd.DataFrame, top_n: int = 5, min_strats: int = 2, mode: str = "per_side",
           strategies=None, picks=None, exclude=None,
           min_conviction: float = 0.0, min_score: float = 0.0) -> dict:
    """Choose what the report covers.

    `picks` = an explicit list of instrument tickers to write up (the desk's curated
    selection); when omitted the picks come from `mode`:
      per_side  — the strongest top_n on EACH side
      overall   — the strongest top_n regardless of side
      threshold — every setup clearing an ABSOLUTE quality bar (conviction ≥ min_conviction
                  AND |score| ≥ min_score), capped at top_n. Unlike the top-N modes this can
                  legitimately return NOTHING — the point is to stay quiet in a week where
                  nothing is worth sending, rather than pad the report with weak setups.
    `exclude` = products held out of the CLIENT report entirely (picks, leaderboard,
    summary and watchlist); defaults to the saved list in data/report_exclude.json."""
    scored = tascore.scoring_strategies(strategies)        # the confluence set (saved, or override)
    total = len(scored)
    flagged = tascore.ta_flagged(sig_df, strategies)
    prod = tascore.score_products(flagged)
    drop = set(exclude) if exclude is not None else u.report_excluded()
    if drop and not prod.empty:                            # client doesn't trade these — whole report
        prod = prod[~prod["instruments"].isin(drop)]
    if prod.empty:
        return {"constructive": [], "cautious": [], "conflicts": [], "watch": [],
                "leaderboard": [], "total": total, "scored": scored}
    prod = prod[prod["n"] >= min_strats]

    def _rows(sub):
        out = []
        for r in sub.itertuples(index=False):
            out.append({"instruments": r.instruments, "market": r.market, "n": int(r.n),
                        "net_dir": int(r.net_dir), "conviction": float(r.conviction),
                        "score": float(r.score), "tags": list(r.tags), "conflict": bool(r.conflict),
                        "sector": _sector_of(r.instruments),
                        "longs": int(r.longs), "shorts": int(r.shorts), "total": total})
        return out

    clean = prod[~prod["conflict"]]           # prod is already sorted by |conviction score| desc
    if picks:                                 # the desk's curated selection (order kept by |score|)
        chosen = clean[clean["instruments"].isin(set(picks))]
        constructive = _rows(chosen[chosen["net_dir"] > 0])
        cautious = _rows(chosen[chosen["net_dir"] < 0])
    elif mode == "threshold":                 # absolute quality bar — may return nothing, by design
        q = clean[(clean["conviction"] >= float(min_conviction))
                  & (clean["score"].abs() >= float(min_score))].head(top_n)   # top_n = safety cap
        constructive = _rows(q[q["net_dir"] > 0])
        cautious = _rows(q[q["net_dir"] < 0])
    elif mode == "overall":                   # the top_n strongest by conviction, whatever the split
        top = clean.head(top_n)
        constructive = _rows(top[top["net_dir"] > 0])
        cautious = _rows(top[top["net_dir"] < 0])
    else:                                     # balanced: top_n on EACH side
        constructive = _rows(clean[clean["net_dir"] > 0].head(top_n))
        cautious = _rows(clean[clean["net_dir"] < 0].head(top_n))
    conflicts = _rows(prod[prod["conflict"]].head(6))

    # "Ones to watch": CLEAN (non-conflicted) directional reads that came close to the quality bar
    # but fell short — surfaced on page 1 (summary table + leaderboard) as secondary, never written
    # up. NOT the conflict watchlist that was cut: these are single-minded setups a notch below the
    # cut, not markets the strategies disagree on. In the top-N modes (min_* = 0) the filter is inert
    # and this is simply the next few by |score| just outside the picks.
    pick_tks = {p["instruments"] for p in constructive + cautious}
    near = clean[(~clean["instruments"].isin(pick_tks))
                 & (clean["conviction"] >= float(min_conviction) - _WATCH_CONV_GAP)
                 & (clean["score"].abs() >= float(min_score) * _WATCH_SCORE_FRAC)]
    watch = _rows(near.head(_WATCH_MAX))                 # clean is |score|-sorted → the closest misses
    for w in watch:
        w["flagged_by"] = ", ".join(_TAG.get(s, s) + (" ▲" if d > 0 else " ▼" if d < 0 else "")
                                    for s, d, _st in w["tags"])
        w["net_txt"] = "▲ constructive" if w["net_dir"] > 0 else "▼ cautious"

    # Leaderboard = EXACTLY the report's picks, ranked by |score| (constructive+cautious splits by
    # side, so re-sort), with the watch names appended beneath. The page-1 chart can no longer show
    # more or other products than the summary table + write-ups. (It used to be a hardcoded whole-book
    # top-16 that ignored the quality bar AND top_n and even included conflicted names — so a "top 10"
    # request drew 16 bars against 8 picks, several of them conflicted. Changed 2026-07-21.)
    lead = [{"market": p["market"], "score": p["score"], "net_dir": p["net_dir"], "n": p["n"]}
            for p in sorted(constructive + cautious, key=lambda p: -abs(p["score"]))]
    return {"constructive": constructive, "cautious": cautious, "conflicts": conflicts,
            "watch": watch, "leaderboard": lead, "total": total, "scored": scored}


# Compact strategy tags for the summary table's "Flagged by" column (matches the app).
_TAG = {"Mean Reversion": "MeanRev", "Trend": "Trend", "MA Crossover": "MA×", "MA Swing": "MA∿",
        "Flag Breakout": "Flag", "Support & Resistance": "S/R", "Fibonacci Retracement": "Fib",
        "Breakout & Retest": "Retest", "Momentum (RSI/MACD)": "Mom", "Bollinger Squeeze": "BBands",
        "Elliott Wave": "Elliott", "Ichimoku Cloud": "Ichimoku",
        "On-Balance Volume": "OBV", "Money Flow Index": "MFI",
        "Donchian Channel": "Donch", "Aroon": "Aroon"}


def _enrich(pick: dict, baseline: set) -> dict:
    """Attach the chart + prose + levels ($ risk, new-this-week flag) to a pick."""
    tk = pick["instruments"]
    gathered = _gather(tk, {s for s, _d, _st in pick["tags"]})
    pf = _pf_series(tk)
    entry = float(pf.iloc[-1]) if len(pf) else None
    pick = dict(pick)
    lv = _full_levels(gathered, pf, pick["net_dir"])
    if lv.get("stop") is not None:
        lv["risk"] = _risk(tk, entry, lv.get("stop"))
    if lv.get("target") is not None:                   # same distance maths, to the objective
        lv["reward"] = _risk(tk, entry, lv.get("target"))
    pick["levels"] = lv
    pick["img"] = _chart_png(tk, gathered, pick["net_dir"], pf, lv)   # chart draws the SAME levels
    raw = _prose(pick, gathered, lv)
    pick["_raw"], pick["prose"] = raw, _md2html(raw)
    pick["sector"] = _sector_of(tk)
    pick["is_new"] = tk not in baseline
    pick["flagged_by"] = ", ".join(_TAG.get(s, s) + (" ▲" if d > 0 else " ▼" if d < 0 else "")
                                   for s, d, _st in pick["tags"])
    pick["net_txt"] = "▲ constructive" if pick["net_dir"] > 0 else "▼ cautious"
    pick["_h"] = _est_section_in(pick, len(_panels_for(gathered)[0]))   # for the closing-page fit
    return pick


def _md2html(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s or "")


def _mc_python() -> str:
    """The Morning Coffee interpreter (has anthropic + ANTHROPIC_API_KEY); '' if not found."""
    base = Path(r"C:\Users\Ben\AppData\Local\Python")
    for exe in sorted(base.glob("pythoncore-*-64/python.exe"), reverse=True):
        if exe.exists():
            return str(exe)
    import shutil
    return shutil.which("python") or ""


def _ai_polish(texts: list) -> list:
    """Rephrase the deterministic write-ups more naturally via Claude (ai_polish.py, run by the
    Morning Coffee interpreter). Returns the ORIGINALS unchanged on any failure — offline, no
    key, timeout, bad output — so the report never depends on the model being reachable."""
    if not texts:
        return texts
    import tempfile
    try:
        mc_py = _mc_python()
        if not mc_py:
            return texts
        with tempfile.TemporaryDirectory() as td:
            inp, outp = Path(td) / "in.json", Path(td) / "out.json"
            inp.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
            r = subprocess.run([mc_py, str(ROOT / "ai_polish.py"), str(inp), str(outp)],
                               capture_output=True, text=True, timeout=240)
            if r.returncode == 0 and outp.exists():
                got = json.loads(outp.read_text(encoding="utf-8"))
                if isinstance(got, list) and len(got) == len(texts):
                    return [g if (isinstance(g, str) and g.strip()) else t
                            for g, t in zip(got, texts)]
    except Exception:
        pass
    return texts


def _paginate(table):
    """Greedy-pack the picks (kept strictly in |score| order) onto pages so each page fills
    top-to-bottom, instead of the old fixed two-up flow that pooled the slack — and, on the
    closing page, a marooned single chart — at the foot. Returns a list of pages (each a list of
    picks); the caller renders every page but the last as a distributed `.pick-page`, and the last
    inside `.report-footer` so the disclaimer pins to the bottom. Heights are the same `_h`
    estimate `_enrich` stores — deliberately an over-estimate, so a packed page's real content
    never exceeds it and can't spill onto a second sheet."""
    pages, cur, used = [], [], 0.0
    for p in table:
        h = float(p.get("_h", 4.6))
        if cur and used + h > _PAGE_IN:              # next pick won't fit → start a new page
            pages.append(cur)
            cur, used = [], 0.0
        cur.append(p)
        used += h
    if cur:
        pages.append(cur)
    # The closing page must also hold the disclaimer. If its picks leave no room, peel the last one
    # onto its own page, so the disclaimer always lands under real content and never overflows.
    if pages and len(pages[-1]) > 1 \
            and sum(float(p.get("_h", 4.6)) for p in pages[-1]) + _DISCLAIMER_IN > _PAGE_IN:
        pages.append([pages[-1].pop()])
    return pages


def render_html(sig_df, asof, top_n=5, min_strats=2, update_baseline=False, ai_polish=False,
                mode="per_side", strategies=None, picks=None, exclude=None,
                min_conviction=0.0, min_score=0.0, equities=False) -> str:
    sel = select(sig_df, top_n=top_n, min_strats=min_strats, mode=mode, strategies=strategies,
                 picks=picks, exclude=exclude, min_conviction=min_conviction, min_score=min_score)
    baseline = _load_baseline()
    constructive = [_enrich(p, baseline) for p in sel["constructive"]]
    cautious = [_enrich(p, baseline) for p in sel["cautious"]]
    if update_baseline:                                # only the weekly run moves the "new" baseline
        _save_baseline(constructive + cautious)
    if ai_polish:                                      # optional natural-language pass (template fallback)
        allp = constructive + cautious
        for p, pol in zip(allp, _ai_polish([p.get("_raw", "") for p in allp])):
            p["prose"] = _md2html(pol)
    n_new = sum(1 for p in constructive + cautious if p.get("is_new"))
    table = sorted(constructive + cautious, key=lambda p: -abs(p.get("score", 0.0)))
    # Lay the picks out page by page. The old two-up flow pooled every page's slack at its foot —
    # worst on the closing page, where a lone chart floated above a sea of white. _paginate instead
    # fills each page (the template then distributes the charts down it) and keeps the disclaimer
    # under real content. The RANKING is never reordered: _paginate only chooses how many
    # consecutive picks share a page.
    pick_pages = _paginate(table)
    watch = sel.get("watch", [])
    # The dashed "quality bar" line only means something in threshold mode; in the top-N modes the
    # watch names are just the next few outside the cut, with no bar to draw.
    leaderboard_img = _leaderboard_png(
        sel.get("leaderboard", []),
        watch=[{"market": w["market"], "score": w["score"], "net_dir": w["net_dir"], "n": w["n"]}
               for w in watch],
        min_score=min_score if mode == "threshold" else 0.0)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("convreport.html").render(
        asof=pretty_date(asof), constructive=constructive, cautious=cautious, table=table,
        watch=watch,             # "ones to watch" — near-misses shown under the summary table
        pick_pages=pick_pages,   # per-page groups; template distributes charts + closes with disclaimer
        leaderboard_img=leaderboard_img,
        n_con=len(constructive), n_cau=len(cautious), n_new=n_new, total=sel["total"], mode=mode,
        scored=sel.get("scored", []),
        axes=tascore.axis_breakdown(sel.get("scored", [])),   # per-axis breakdown for the intro list
        # only explain the trend de-duplication when the set actually holds >1 trend read —
        # otherwise the report describes machinery that never fires
        dedup_active=tascore.has_intra_axis_dup(sel.get("scored", [])),
        min_conviction=min_conviction, min_score=min_score,
        equities=equities, book_label=("equity universe" if equities else "futures book"),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(sig_df, asof, out_pdf, top_n=5, min_strats=2, update_baseline=False, ai_polish=False,
              mode="per_side", strategies=None, picks=None, exclude=None,
              min_conviction=0.0, min_score=0.0, equities=False):
    from reportkit import render_pdf
    return render_pdf(render_html(sig_df, asof, top_n, min_strats, update_baseline, ai_polish, mode,
                                  strategies, picks, exclude, min_conviction, min_score,
                                  equities=equities), out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-strats", type=int, default=2)
    ap.add_argument("--baseline", action="store_true",
                    help="update the 'new this week' baseline (the weekly run passes this; ad-hoc previews don't)")
    ap.add_argument("--ai-polish", action="store_true",
                    help="rephrase the write-ups via Claude (falls back to the template offline / no key)")
    ap.add_argument("--mode", choices=["per_side", "overall", "threshold"], default="per_side",
                    help="per_side = top N constructive AND top N cautious; overall = the top N by "
                         "conviction regardless of side; threshold = only setups clearing an "
                         "absolute quality bar (may return nothing)")
    ap.add_argument("--min-conviction", type=float, default=0.0,
                    help="threshold mode: minimum Conviction (0-100)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="threshold mode: minimum |Score|")
    ap.add_argument("--strategies", default=None,
                    help="comma-separated confluence set to score (overrides the saved set in "
                         "data/confluence_set.json; the rest stay display-only)")
    ap.add_argument("--picks", default=None,
                    help="comma-separated instrument tickers to write up (the desk's curated "
                         "selection); omit to take the strongest --top automatically")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated tickers to hold out of the whole client report; omit to "
                         "use the saved list in data/report_exclude.json")
    ap.add_argument("--equities", action="store_true",
                    help="run over the EQUITIES universe: pull every chart's history from the cached "
                         "yfinance OHLCV store (src.eqta) instead of the FICC datafeed, label sectors "
                         "from the equity membership cache, and skip the FICC sector/product filter")
    args = ap.parse_args()
    _csv = lambda v: [s.strip() for s in v.split(",") if s.strip()] if v else None
    strategies, picks, exclude = _csv(args.strategies), _csv(args.picks), _csv(args.exclude)
    df = pd.read_parquet(args.signals_parquet)
    if args.equities:                          # equity report: inject the yfinance OHLCV store
        from src import eqta
        _close, _vol = eqta.load_history()
        set_history_source(_close, _vol, eqta.member_meta())
        if exclude is None:                    # bare CLI run: fall back to the equities exclude list
            exclude = sorted(u.report_excluded("equities"))
    else:
        try:                                   # honour the dashboard's sector/product filter (FICC)
            from universe import enabled_tickers as _en, filter_active as _fa
            if _fa() and "instruments" in df.columns:
                _e = _en()
                df = df[df["instruments"].map(lambda k: all(p.strip() in _e for p in str(k).split("/")))]
        except Exception:
            pass
    build_pdf(df, args.asof, args.out_pdf, top_n=args.top, min_strats=args.min_strats,
              update_baseline=args.baseline, ai_polish=args.ai_polish, mode=args.mode,
              strategies=strategies, picks=picks, exclude=exclude,
              min_conviction=args.min_conviction, min_score=args.min_score, equities=args.equities)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
