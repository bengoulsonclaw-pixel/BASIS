"""Has the skew REALIZED? — our wings vs the path vol actually travels.

Ben's spec (2026-08-14): scatter the underlying (x) against ATM vol (y) over a
window, fit a line — its gradient is the REALIZED skew. Our own 90%/110% wings
charge an implied skew over the same distance. If the 110% call is marked 10 vols
under ATM but the fitted line says vol only gives back 5 when the future trades
10% higher, the call is ~5 vols CHEAP "on arrival" — when spot reaches the
strike, the option is ATM and marks near the then-ATM vol.

Verdict per wing = (line's predicted ATM at the wing distance) − (wing's implied
vol). Positive → the wing is cheap against the realized path; negative → rich.
A CONFIDENCE flag guards the known weakness of levels-on-levels fits: when the
daily-changes beta disagrees with the levels gradient in sign or by more than
2×, one trending regime dominated the window and the verdict is marked
"regime-flagged" rather than confident.

All inputs are our own local files (own implied, own wings, price store) — this
analysis never touches Bloomberg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .volmove import SNAP, _own_iv, _prices, move_frame, level_fit, change_beta

WING_PCT = 10.0        # the wings live at ±10% moneyness — the comparison distance


def _own_wings() -> pd.DataFrame:
    """Latest own wing marks per ticker: put90 / call110 / atm (settle-dated)."""
    h = pd.read_parquet(SNAP / "own_skew_history.parquet")
    h["date"] = pd.to_datetime(h["date"])
    return h.sort_values("date").groupby("ticker").tail(1).set_index("ticker")


def analyze(window: int = 126) -> pd.DataFrame:
    """Book-wide table: realized-skew gradient vs the implied wings, one verdict
    per wing per product. Only products with own wing marks are judged — the
    whole point is comparing OUR wings to OUR realized path."""
    iv, px, wings = _own_iv(), _prices(), _own_wings()
    from .universe import name
    rows = []
    for tk in wings.index:
        j = move_frame(tk, window, iv, px)
        if j.empty:
            continue
        g_lvl, r2 = level_fit(j)
        g_chg = change_beta(j)
        if not np.isfinite(g_lvl):
            continue
        w = wings.loc[tk]
        iv_now = float(j["iv"].iloc[-1])
        pred_call = iv_now + g_lvl * WING_PCT
        pred_put = iv_now - g_lvl * WING_PCT
        call_gap = pred_call - float(w["call"])
        put_gap = pred_put - float(w["put"])
        ratio = g_chg / g_lvl if np.isfinite(g_chg) and abs(g_lvl) > 1e-9 else float("nan")
        confident = bool(np.isfinite(ratio) and ratio > 0 and 0.5 <= ratio <= 2.0)
        rows.append({
            "ticker": tk, "market": name(tk),
            "g_lvl": round(g_lvl, 3), "g_chg": round(g_chg, 3) if np.isfinite(g_chg) else np.nan,
            "r2": round(r2, 2) if np.isfinite(r2) else np.nan,
            "iv_now": round(iv_now, 1),
            "put_wing": round(float(w["put"]), 1), "call_wing": round(float(w["call"]), 1),
            "pred_put": round(pred_put, 1), "pred_call": round(pred_call, 1),
            "put_gap": round(put_gap, 1), "call_gap": round(call_gap, 1),
            "confident": confident, "n": len(j),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["max_gap"] = df[["put_gap", "call_gap"]].abs().max(axis=1)
        df = df.sort_values("max_gap", ascending=False).reset_index(drop=True)
    return df


def scatter_frame(ticker: str, window: int = 126):
    """(points frame, fit dict) for the page's per-product chart: the raw
    (price, iv) path, the fitted line's endpoints, and the wing markers with
    their line-predicted twins at ±WING_PCT."""
    j = move_frame(ticker, window)
    if j.empty:
        return None, None
    g_lvl, r2 = level_fit(j)
    wings = _own_wings()
    if ticker not in wings.index or not np.isfinite(g_lvl):
        return None, None
    w = wings.loc[ticker]
    s_now, iv_now = float(j["px"].iloc[-1]), float(j["iv"].iloc[-1])
    # draw the OLS line across the union of the traded range and the wing strikes,
    # so the wing markers always sit ON the drawn line's span
    lo = min(float(j["dist"].min()), -WING_PCT)
    hi = max(float(j["dist"].max()), WING_PCT)
    xs = np.array([lo, hi])
    g, b = np.polyfit(j["dist"], j["iv"], 1)
    fit = {
        "line": pd.DataFrame({"px": s_now * (1 + xs / 100.0), "iv": g * xs + b}),
        "wings": pd.DataFrame({
            "px": [s_now * (1 - WING_PCT / 100), s_now * (1 + WING_PCT / 100)],
            "iv": [float(w["put"]), float(w["call"])],
            "kind": ["put wing (implied)", "call wing (implied)"],
        }),
        "preds": pd.DataFrame({
            "px": [s_now * (1 - WING_PCT / 100), s_now * (1 + WING_PCT / 100)],
            "iv": [g * (-WING_PCT) + b, g * WING_PCT + b],
            "kind": ["line at put strike", "line at call strike"],
        }),
        "g_lvl": g_lvl, "r2": r2, "g_chg": change_beta(j),
        "s_now": s_now, "iv_now": iv_now,
    }
    return j, fit
