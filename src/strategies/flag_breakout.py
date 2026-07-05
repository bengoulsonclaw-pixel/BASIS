"""Flag-pattern breakout monitor — spots products coiling at the edge of a flag.

A *flag* is the classic continuation pattern (see CMC's write-up): a sharp,
near-vertical move — the **flagpole** — followed by a brief, low-slope
**consolidation** that drifts *against* the pole inside two roughly parallel
trendlines, retracing only part of the pole. The trade is the **breakout** — a
close back through the trendline in the pole's direction:

    Bull flag : strong up pole  -> price drifts down/sideways -> breaks UP
    Bear flag : strong down pole -> price drifts up/sideways  -> breaks DOWN

This page answers "what's about to break out?" It scans the book, fits the most
recent pole+flag on each product's close history, and scores how close price is
to the breakout line as a 0–100 **breakout readiness** (100 = sitting on the
line, about to go). The readiness is signed by pattern — **+ for bull, − for
bear** — so it drops straight into the dashboard's symmetric ±trigger machinery
(see src/specs.py): readiness ≥ +t flags a bull breakout setup, ≤ −t a bear one.

**Volume confirmation.** The textbook flag also dries up on volume through the
consolidation and surges on the breakout. With a volume series available
(`datafeed.get_volume_history`), each flag carries a **dry-up ratio** (flag avg
volume ÷ pole avg volume; < 1 is the classic contraction) and a **breakout
surge** (latest bar ÷ flag avg). Volume is a *confirmation*, not a gate: coverage
varies by product and the price geometry stands on its own, so a missing volume
series just shows no confirmation rather than suppressing the signal.

Close-based: the pole, the channel and the breakout are all built from settlement
closes (the same get_history seam in mock and live). The rich per-product table is
cached to data/signals/flag_breakout.parquet (+ _history) for the visual report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..datafeed import get_history, get_volume_history
from ..universe import TREND_UNIVERSE, name, asset, region
from .base import frame

STRATEGY = "Flag Breakout"

# --- detection knobs (trading days, all tunable; calibrate with backtest_flags.py) ---
MIN_HISTORY = 70      # need a decent run-up + flag to fit anything
VOL_WINDOW = 60       # window for the per-product daily-vol scale
FLAG_MIN, FLAG_MAX = 5, 15     # consolidation length, bars after the pole (~≤3 weeks; beyond
                               # that a flag starts to read as a reversal, not a continuation)
POLE_MIN, POLE_MAX = 4, 20     # flagpole length
POLE_K = 2.5          # pole must be ≥ this many σ of its own daily moves (fast + strong)
POLE_MIN_ABS = 0.04   # …and ≥ this absolute move (screens out near-flat STIRs)
RETRACE_MAX = 0.50    # flag may give back at most half the pole ("not the majority")
CHANNEL_W = 2.0       # channel half-width = this × the flag's residual σ
TIGHT = 0.75          # consolidation: full channel must be ≤ this × the pole height
SLOPE_WITH = 0.50     # flag may lean WITH the pole up to this × the pole's per-bar velocity
SLOPE_AGAINST = 1.25  # …and AGAINST it up to this × (steeper = a reversal, not a flag)
VOL_DRYUP_MAX = 0.90  # volume "confirms" when flag avg volume ≤ this × the pole's
PENNANT_CONVERGE = 0.60  # consolidation range back-half < this × front-half → pennant (else flag)

# The page's default readiness trigger (mirrors specs.py["Flag Breakout"]["default"]).
DEFAULT_TRIGGER = 70.0

# Rich per-flag table + price/volume/channel history cached here for the visual report.
_DATA = Path(__file__).resolve().parents[2] / "data" / "signals"
DETAIL_FILE = _DATA / "flag_breakout.parquet"
HISTORY_FILE = _DATA / "flag_breakout_history.parquet"
DETAIL_COLUMNS = ["market", "ticker", "asset", "region", "pattern", "sign", "pole_ret",
                  "plen", "flen", "retrace", "readiness", "metric", "dist_pct", "broke",
                  "breakout", "target", "stop", "rr", "level", "pole_base_date", "pole_tip_date",
                  "pole_base_px", "pole_tip_px", "vol_dryup", "vol_surge", "vol_confirms",
                  "signal", "direction"]


def _best_flag(c: np.ndarray, dv: float, sign: int, va: np.ndarray | None) -> dict | None:
    """Best (closest-to-breakout) pole+flag of one polarity on close array `c`.

    `sign` = +1 hunts a bull flag (up pole, breaks up), −1 a bear flag. `dv` is the
    product's recent daily-return σ, used to demand a genuinely fast pole. `va` is the
    aligned volume array (or None). Returns a geometry dict or None if no valid flag of
    this polarity sits at the right edge."""
    n = len(c)
    best = None
    for flen in range(FLAG_MIN, FLAG_MAX + 1):
        anchor = n - 1 - flen                      # pole tip / flag start
        if anchor <= POLE_MIN:
            continue

        # Pole base = the opposite extreme within POLE_MAX bars before the tip; the tip
        # itself must be (near) the extreme of that segment, else it isn't a clean pole.
        b0 = max(0, anchor - POLE_MAX)
        seg = c[b0:anchor + 1]
        base_idx = b0 + int(np.argmin(seg) if sign > 0 else np.argmax(seg))
        tip_is_extreme = (c[anchor] >= seg.max() - 1e-9) if sign > 0 else (c[anchor] <= seg.min() + 1e-9)
        if not tip_is_extreme:
            continue
        plen = anchor - base_idx
        if not (POLE_MIN <= plen <= POLE_MAX):
            continue

        pole_ret = sign * (c[anchor] / c[base_idx] - 1.0)        # signed → always ≥0 for a valid pole
        if pole_ret < max(POLE_MIN_ABS, POLE_K * dv * np.sqrt(plen)):
            continue
        pole_h = abs(c[anchor] - c[base_idx])
        if pole_h <= 0:
            continue

        flag = c[anchor:]                                        # tip … today, inclusive
        flag_low, flag_high = float(flag.min()), float(flag.max())
        # Retrace: must not unwind more than RETRACE_MAX of the pole.
        retrace = ((c[anchor] - flag_low) if sign > 0 else (flag_high - c[anchor])) / pole_h
        if retrace > RETRACE_MAX:
            continue

        # Flag (parallel channel) vs pennant (range converging into the apex): compare the
        # price range of the consolidation's back half against its front half. Both are traded
        # identically (same pole, breakout and measured move) — this only sets the label.
        shape = "flag"
        if len(flag) >= 8:
            half = len(flag) // 2
            r1, r2 = float(np.ptp(flag[:half])), float(np.ptp(flag[half:]))
            if r1 > 0 and r2 < PENNANT_CONVERGE * r1:
                shape = "pennant"

        # Parallel channel = OLS trendline ± CHANNEL_W·σ(resid) on the flag closes.
        x = np.arange(len(flag), dtype=float)
        b, a = np.polyfit(x, flag, 1)                            # slope, intercept
        sd = float((flag - (a + b * x)).std())
        if sd <= 0:
            continue
        width = 2.0 * CHANNEL_W * sd
        if width > TIGHT * pole_h:                               # not a tight consolidation
            continue

        # Slope test: a flag drifts gently against (or flat with) the pole. Express the
        # slope as a fraction of the pole's per-bar velocity so it scales across products.
        slope_frac = (sign * b) / (pole_h / plen)
        if slope_frac > SLOPE_WITH or slope_frac < -SLOPE_AGAINST:
            continue

        # Project the channel to today; readiness = position across the channel toward the
        # breakout edge (0 = far wall, 100 = on the breakout line, >100 = already through).
        xl = x[-1]
        mid = a + b * xl
        upper, lower = mid + CHANNEL_W * sd, mid - CHANNEL_W * sd
        brk = upper if sign > 0 else lower
        proximity = 100.0 * ((c[-1] - lower) if sign > 0 else (upper - c[-1])) / width
        proximity = float(np.clip(proximity, -20.0, 140.0))
        dist_pct = sign * (brk - c[-1]) / c[-1] * 100.0          # >0 still inside, <0 broken out
        broke = (c[-1] > brk) if sign > 0 else (c[-1] < brk)

        # Measured move (consensus across the desks): project the flagpole height from the
        # breakout level; stop beyond the far edge of the flag; reward:risk on a break entry.
        target = brk + sign * pole_h
        stop = flag_low if sign > 0 else flag_high
        risk = abs(brk - stop)
        rr = pole_h / risk if risk > 1e-12 else float("nan")

        # Volume confirmation: dry-up through the flag, surge on the break (NaN if no vol).
        dryup = surge = float("nan")
        confirms = None
        if va is not None:
            pole_v = float(np.nanmean(va[base_idx:anchor + 1])) if anchor > base_idx else float("nan")
            flag_v = float(np.nanmean(va[anchor:]))
            last_v = float(va[-1])
            if np.isfinite(pole_v) and pole_v > 0 and np.isfinite(flag_v):
                dryup = flag_v / pole_v
                confirms = bool(dryup <= VOL_DRYUP_MAX)
                if flag_v > 0 and np.isfinite(last_v):
                    surge = last_v / flag_v

        cand = {"sign": sign, "base_idx": base_idx, "anchor": anchor, "plen": plen,
                "flen": flen, "shape": shape, "pole_ret": pole_ret, "pole_h": pole_h, "slope": float(b),
                "intercept": float(a), "sd": sd, "width": width, "brk": float(brk),
                "proximity": proximity, "dist_pct": float(dist_pct), "broke": bool(broke),
                "retrace": float(retrace), "target": float(target), "stop": float(stop),
                "rr": float(rr), "vol_dryup": dryup, "vol_surge": surge, "vol_confirms": confirms}
        if best is None or cand["proximity"] > best["proximity"]:
            best = cand
    return best


def _detect_flag(px: pd.Series, vol: pd.Series | None = None) -> dict | None:
    """The most breakout-ready flag (bull or bear) on a product's close series, or None.
    `vol` (optional) is the product's daily volume — aligned to the close index here."""
    s = px.dropna()
    if len(s) < MIN_HISTORY:
        return None
    c = s.to_numpy(dtype=float)
    dv = float(s.pct_change().tail(VOL_WINDOW).std())
    if not np.isfinite(dv) or dv <= 0:
        return None
    va = vol.reindex(s.index).to_numpy(dtype=float) if vol is not None else None
    if va is not None:
        va = va.copy()
        va[va <= 0] = np.nan        # treat 0/blank contract volume as missing (roll/lag artifacts)
    cands = [d for d in (_best_flag(c, dv, +1, va), _best_flag(c, dv, -1, va)) if d is not None]
    if not cands:
        return None
    return max(cands, key=lambda d: d["proximity"])             # the one nearest its breakout


def _signal(metric: float, trigger: float) -> tuple[str, int]:
    """(signal, direction) at a readiness trigger — matches specs.reflag_rows so the
    table re-flags identically when the user moves the slider."""
    if metric >= trigger:
        return "Bull breakout setup", 1
    if metric <= -trigger:
        return "Bear breakout setup", -1
    return "—", 0


def _vol_note(dryup: float, confirms) -> str:
    if confirms is None or not np.isfinite(dryup):
        return ""
    return (f" · vol ✓ dry-up {dryup:.0%} of pole" if confirms
            else f" · vol ✗ no dry-up ({dryup:.0%})")


def _context(r) -> str:
    """Human note for a detail row `r` (a namedtuple from the detail table)."""
    edge = (f"broke out ({abs(r.dist_pct):.1f}% beyond line)" if r.broke
            else f"{r.dist_pct:.1f}% to breakout")
    mm = f" · target {r.target:g}, stop {r.stop:g}"
    if pd.notna(r.rr):
        mm += f" (R:R {r.rr:.1f}{' ✓2:1' if r.rr >= 2 else ''})"
    return (f"{r.pattern} · pole {r.pole_ret * 100:+.0f}% in {r.plen}d · "
            f"flag {r.flen}d, {r.retrace * 100:.0f}% retrace · {edge}{mm}{_vol_note(r.vol_dryup, r.vol_confirms)}")


def _window_frame(s: pd.Series, va: np.ndarray | None, d: dict) -> pd.DataFrame:
    """Price + volume + the fitted channel (upper/lower/breakout) over the pole-and-flag
    window. Channel columns are NaN before the flag so only the consolidation is bracketed.
    Shared by the dashboard chart and the report's history cache."""
    idx, c = s.index, s.to_numpy(dtype=float)
    start = max(0, d["base_idx"] - 5)                           # a little lead-in before the pole
    a, b, sd, sign, anchor = d["intercept"], d["slope"], d["sd"], d["sign"], d["anchor"]
    out = pd.DataFrame({
        "date": idx[start:],
        "price": c[start:],
        "volume": (va[start:] if va is not None else np.full(len(c) - start, np.nan)),
    })
    upper = np.full(len(out), np.nan)
    lower = np.full(len(out), np.nan)
    breakout = np.full(len(out), np.nan)
    for i in range(len(out)):
        g = start + i
        if g >= anchor:                                        # inside the flag
            mid = a + b * (g - anchor)
            upper[i], lower[i] = mid + CHANNEL_W * sd, mid - CHANNEL_W * sd
            breakout[i] = upper[i] if sign > 0 else lower[i]
    out["upper"], out["lower"], out["breakout"] = upper, lower, breakout
    return out


def compute_table() -> pd.DataFrame:
    """Full set of currently-detected flags with all geometry + volume confirmation, one
    row per product, ranked nearest-breakout first. Persists the per-flag price/volume/
    channel history for the visual report. The single source of truth for the dashboard
    rows and the report."""
    tickers = list(TREND_UNIVERSE)
    history = get_history(tickers)
    try:
        volume = get_volume_history(tickers)
    except Exception:
        volume = pd.DataFrame()

    recs, hist_frames = [], []
    for t in tickers:
        if t not in history.columns:
            continue
        vser = volume[t] if (not volume.empty and t in volume.columns) else None
        d = _detect_flag(history[t], vser)
        if d is None:
            continue
        s = history[t].dropna()
        va = vser.reindex(s.index).to_numpy(dtype=float) if vser is not None else None
        metric = d["sign"] * d["proximity"]
        signal, direction = _signal(metric, DEFAULT_TRIGGER)
        recs.append({
            "market": name(t), "ticker": t, "asset": asset(t), "region": region(t),
            "pattern": f"{'Bull' if d['sign'] > 0 else 'Bear'} {d['shape']}", "sign": int(d["sign"]),
            "pole_ret": round(d["sign"] * d["pole_ret"], 4), "plen": int(d["plen"]),
            "flen": int(d["flen"]), "retrace": round(d["retrace"], 3),
            "readiness": round(d["proximity"], 1), "metric": round(metric, 1),
            "dist_pct": round(d["dist_pct"], 2), "broke": bool(d["broke"]),
            "breakout": round(d["brk"], 4),
            "target": round(d["target"], 4), "stop": round(d["stop"], 4),
            "rr": round(d["rr"], 2) if np.isfinite(d["rr"]) else np.nan,
            "level": round(float(s.iloc[-1]), 4),
            "pole_base_date": s.index[d["base_idx"]], "pole_tip_date": s.index[d["anchor"]],
            "pole_base_px": round(float(s.iloc[d["base_idx"]]), 4),
            "pole_tip_px": round(float(s.iloc[d["anchor"]]), 4),
            "vol_dryup": round(d["vol_dryup"], 3) if np.isfinite(d["vol_dryup"]) else np.nan,
            "vol_surge": round(d["vol_surge"], 3) if np.isfinite(d["vol_surge"]) else np.nan,
            "vol_confirms": d["vol_confirms"], "signal": signal, "direction": direction,
        })
        hf = _window_frame(s, va, d)
        hf["ticker"] = t
        hist_frames.append(hf)

    df = pd.DataFrame(recs, columns=DETAIL_COLUMNS)
    if not df.empty:
        order = df["metric"].abs().sort_values(ascending=False).index
        df = df.reindex(order).reset_index(drop=True)
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        (pd.concat(hist_frames, ignore_index=True) if hist_frames
         else pd.DataFrame(columns=["date", "ticker", "price", "volume", "upper", "lower", "breakout"])
         ).to_parquet(HISTORY_FILE, index=False)
    except Exception:
        pass
    return df


def find_opportunities(history: pd.DataFrame | None = None) -> pd.DataFrame:
    tbl = compute_table()

    # Cache the rich table for the visual report (compute-once-daily, like the rest).
    try:
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(DETAIL_FILE, index=False)
    except Exception:
        pass

    rows = []
    for r in tbl.itertuples(index=False):
        rows.append({
            "strategy": STRATEGY,
            "market": r.market,
            "instruments": r.ticker,
            "signal": r.signal,
            "direction": int(r.direction),
            "metric": float(r.metric),
            "metric_label": "breakout readiness",
            "level": float(r.level),
            "context": _context(r),
        })
    return frame(rows)


def flag_chart_data(ticker: str, history: pd.DataFrame | None = None):
    """Price + volume + the fitted flag channel (upper / lower / breakout line) over the
    pole-and-flag window, plus an info dict for the dashboard. (DataFrame, info) or
    (None, None) if no flag is present."""
    if history is None:
        history = get_history([ticker])
    if ticker not in history.columns:
        return None, None
    try:
        vser = get_volume_history([ticker])
        vser = vser[ticker] if ticker in vser.columns else None
    except Exception:
        vser = None
    d = _detect_flag(history[ticker], vser)
    if d is None:
        return None, None

    s = history[ticker].dropna()
    va = vser.reindex(s.index).to_numpy(dtype=float) if vser is not None else None
    sub = _window_frame(s, va, d)

    metric = d["sign"] * d["proximity"]
    signal, _ = _signal(metric, DEFAULT_TRIGGER)
    info = {
        "type": f"{'Bull' if d['sign'] > 0 else 'Bear'} {d['shape']}",
        "sign": d["sign"],
        "pole_ret": d["sign"] * d["pole_ret"],                 # signed for display (+up / −down)
        "plen": d["plen"], "flen": d["flen"], "retrace": d["retrace"],
        "readiness": d["proximity"], "metric": metric,
        "dist_pct": d["dist_pct"], "broke": d["broke"], "signal": signal,
        "breakout": d["brk"], "target": d["target"], "stop": d["stop"], "rr": d["rr"],
        "price": float(s.iloc[-1]),
        "vol_dryup": d["vol_dryup"], "vol_surge": d["vol_surge"], "vol_confirms": d["vol_confirms"],
        "anchor_date": s.index[d["anchor"]],
        "pole_base": (s.index[d["base_idx"]], float(s.iloc[d["base_idx"]])),
        "pole_tip": (s.index[d["anchor"]], float(s.iloc[d["anchor"]])),
    }
    return sub, info
