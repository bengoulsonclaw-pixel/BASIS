"""How vol behaves when the underlying moves — the shared data spine.

Two consumers (Ben, 2026-08-14):
  * skewreal.py — "has the skew realized?": the levels regression of ATM vol on
    the underlying over a window, whose gradient is the REALIZED skew, compared
    against what our own wings charge at ±10% moneyness.
  * the Volatility page's vol-response section — down/up vol betas, big-day pops
    and their retention (built next).

Everything runs off data we already hold locally — our own implied history
(own30_history), our own wings (own_skew_history) and the snapshot price store.
No Bloomberg, ever.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot"


def _own_iv() -> pd.DataFrame:
    """Our constant-30d ATM implied, date × ticker (the vol book's own leg)."""
    h = pd.read_parquet(SNAP / "own30_history.parquet")
    h["date"] = pd.to_datetime(h["date"])
    return h.pivot_table(index="date", columns="ticker", values="ours30", aggfunc="last")


def _prices() -> pd.DataFrame:
    px = pd.read_parquet(SNAP / "prices.parquet")
    return px.assign(date=pd.to_datetime(px["date"])).set_index("date")


def move_frame(ticker: str, window: int, iv: pd.DataFrame = None,
               px: pd.DataFrame = None) -> pd.DataFrame:
    """Joined per-day frame for one product over the trailing `window` sessions:
    price, ATM iv, daily %-return, daily iv change, and the price distance from
    TODAY in % (the x-axis of the realized-skew scatter). Empty if uncovered."""
    iv = _own_iv() if iv is None else iv
    px = _prices() if px is None else px
    if ticker not in iv.columns or ticker not in px.columns:
        return pd.DataFrame()
    j = pd.concat([px[ticker].rename("px"), iv[ticker].rename("iv")],
                  axis=1, sort=True).dropna()
    if len(j) < 40:
        return pd.DataFrame()
    j = j.iloc[-window:].copy()
    j["ret"] = np.log(j["px"]).diff() * 100.0
    j["div"] = j["iv"].diff()
    j["dist"] = (j["px"] / float(j["px"].iloc[-1]) - 1.0) * 100.0
    return j


def level_fit(j: pd.DataFrame):
    """(gradient in vol pts per 1% of today's spot, r²) of the LEVELS best fit —
    Ben's scattergram line: ATM vol regressed on where the underlying traded."""
    if len(j) < 40:
        return float("nan"), float("nan")
    x, y = j["dist"].to_numpy(), j["iv"].to_numpy()
    g, c = np.polyfit(x, y, 1)
    pred = g * x + c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(g), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def change_beta(j: pd.DataFrame) -> float:
    """Vol pts per 1% move from DAILY CHANGES — the regime-robust companion: when
    it disagrees with the levels gradient, one trending regime dominated the window."""
    d = j.dropna(subset=["ret", "div"])
    if len(d) < 40:
        return float("nan")
    return float(np.polyfit(d["ret"], d["div"], 1)[0])


def response_table(window: int = 260) -> pd.DataFrame:
    """Per product, over our whole own-implied history: how vol PERFORMS on a move.

    down_beta / up_beta — vol points gained per 1% move on that side (sign-adjusted:
    positive ALWAYS means implied rises). big_n / big_pop — the count of |move| ≥ 2σ
    days and the median implied change on them (σ = the product's own rolling 21d
    daily vol, so "big" is self-calibrated). keep5 — the median FURTHER implied
    change over the 5 sessions after a big day: positive = the pop kept building,
    negative = it bled back (gamma paid, vega didn't). pop_pct — big_pop as % of the
    current implied level, the cross-product ranking metric."""
    iv, px = _own_iv(), _prices()
    from .universe import name
    rows = []
    for tk in iv.columns:
        j = move_frame(tk, window, iv, px)
        if j.empty:
            continue
        d = j.dropna(subset=["ret", "div"]).copy()
        if len(d) < 80:
            continue
        d["sig"] = d["ret"].rolling(21).std()
        d["sigmove"] = d["ret"] / d["sig"]
        dn, up = d[d["ret"] < 0], d[d["ret"] > 0]
        if len(dn) < 25 or len(up) < 25:
            continue
        b_dn = -float(np.polyfit(dn["ret"], dn["div"], 1)[0])
        b_up = float(np.polyfit(up["ret"], up["div"], 1)[0])
        big = d[d["sigmove"].abs() >= 2].dropna(subset=["sigmove"])
        divs = d["div"].to_numpy()
        pos = {ts: i for i, ts in enumerate(d.index)}
        keeps = [float(divs[i + 1:i + 6].sum())
                 for ts in big.index
                 for i in [pos[ts]] if i + 5 < len(divs)]
        iv_now = float(j["iv"].iloc[-1])
        big_pop = float(big["div"].median()) if len(big) else float("nan")
        rows.append({
            "ticker": tk, "market": name(tk),
            "down_beta": round(b_dn, 2), "up_beta": round(b_up, 2),
            "asym": round(b_dn - b_up, 2),
            "big_n": len(big), "big_pop": round(big_pop, 2) if np.isfinite(big_pop) else np.nan,
            "keep5": round(float(np.median(keeps)), 2) if keeps else np.nan,
            "iv_now": round(iv_now, 1),
            "pop_pct": round(100 * big_pop / iv_now, 1)
            if np.isfinite(big_pop) and iv_now > 0 else np.nan,
            "n": len(d),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pop_pct", ascending=False).reset_index(drop=True)
    return df
