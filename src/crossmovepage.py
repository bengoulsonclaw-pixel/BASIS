"""crossmovepage.py — 🧭 Macro Compass, the BASIS page for crossmove.py.

Named for what it does and does not do: it orients you against the rest of the
complex, it does not point at where anything is going. Everything on it is
contemporaneous.

Pick a driver, a move, a change horizon and a lookback; get the average co-movement
in everything else, each row carrying its t-statistic, its R-squared and the band of
what the relationship leaves unexplained.

The band is the point of the page, not decoration. Gold's average response to a 1%
dollar move is about -1.1% over the last five years with a t of -7, and one standard
deviation of what that relationship does NOT explain is 4%. Both are true, and a page
that showed only the first would be a false-precision machine. Rows where the band
swamps the implied move are marked, so the eye lands on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import crossmove as cm


def _fmt(v: float, unit: str, dp: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{dp}f}{unit}"


def render() -> None:
    st.subheader("🧭 Macro Compass")
    st.caption(
        "A move in one macro instrument, translated into the average move in the "
        "others over a window you choose. **Contemporaneous** — this sizes a move "
        "that has already happened; it does not forecast one."
    )

    c1, c2, c3, c4 = st.columns([0.30, 0.22, 0.24, 0.24])
    with c1:
        driver = st.selectbox("Move in", list(cm.INSTRUMENTS), index=4,
                              key="cm_driver")
    unit = cm.unit_of(driver)
    step = cm.INSTRUMENTS[driver][2]
    with c2:
        move = st.number_input(f"Size ({unit})", value=cm.default_move(driver),
                               step=step, format="%.2f", key="cm_move")
    with c3:
        hz_label = st.selectbox("Over", list(cm.HORIZONS), index=2, key="cm_hz")
    with c4:
        lb_label = st.selectbox("Measured across", list(cm.LOOKBACKS), index=2,
                                key="cm_lb")
    horizon = cm.HORIZONS[hz_label]
    years = cm.LOOKBACKS[lb_label]

    try:
        t = cm.translate(driver, move, horizon=horizon, years=years)
    except Exception as e:
        st.warning(f"Could not compute: {type(e).__name__}: {e}")
        return
    if t.empty:
        st.info("Not enough overlapping history for that combination.")
        return

    st.markdown(f"##### If **{driver}** moves **{move:+g}{unit}** over {hz_label.lower()}")

    show = pd.DataFrame({
        "Instrument": t["instrument"],
        "Typically moves": [_fmt(r.implied, r.unit) for r in t.itertuples()],
        "± 1 s.d. unexplained": [f"{r.band_1sd:.2f}{r.unit}" for r in t.itertuples()],
        "t": t["t"].round(1),
        "R²": t["r_squared"].round(2),
        "n": t["n"],
        # A relationship whose noise dwarfs the implied move is not actionable, and
        # saying so in the table beats hoping the reader compares two columns.
        "Read as": [
            "noise dominates" if abs(r.implied) < r.band_1sd * 0.5
            else ("weak" if abs(r.t) < 2 else "clear")
            for r in t.itertuples()],
    })
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(cm.caveat(driver, horizon, years))

    strong = t[(t["t"].abs() >= 2) & (t["implied"].abs() >= t["band_1sd"] * 0.5)]
    if strong.empty:
        st.info(
            "No instrument moves enough with this driver, at this horizon and over "
            "this window, to stand clear of its own noise. That is a result, not a "
            "gap — it says the pairing is not one to lean on."
        )

    with st.expander("Why the numbers change when you change the window"):
        st.markdown(
            "These are **pairwise** relationships: gold against the dollar alone, "
            "including whatever else moved alongside the dollar. That is the right "
            "question for *what typically happens*, and a different one from gold's "
            "response with yields held fixed — the Gold Engine's Drivers tab fits "
            "that jointly and the two disagree on purpose.\n\n"
            "Relationships also genuinely drift. Gold's sensitivity to real yields "
            "has strengthened by roughly 60% between 2002–2013 and 2014–2026, while "
            "its sensitivity to the dollar has barely moved. Comparing two lookbacks "
            "is the fastest way to see which of the two you are looking at."
        )

    with st.expander("Full matrix — every pair"):
        st.caption("Row instrument's average move per **one unit** of the column "
                   "instrument (1% for prices, 1bp for yields, 1pt for VIX). "
                   "Read down a column to shock that instrument.")
        try:
            m = cm.matrix(horizon=horizon, years=years)
            st.dataframe(m.round(3), use_container_width=True)
        except Exception as e:
            st.caption(f"Matrix unavailable: {type(e).__name__}: {e}")
