"""crossmovepage.py — 🧭 Macro Compass.

Named for what it does and does not do: it orients you against the rest of the
complex, it does not point at where anything is going. Every relationship on this
page is CONTEMPORANEOUS, and the lead-lag work behind it (golddiag, across four
metals and horizons out to sixty days) found nothing that leads any of them.

Three tabs, all answering a version of the same question — how does this thing move
with that one:

  Instruments   a move in one macro instrument, translated into the others over a
                window you choose.
  Fed repricing what a change in POLICY PRICING is worth to gold and the dollar.
                The input is the gap between a view and the curve, never a hike.
  Releases      which scheduled US releases actually move which metal.

THE BAND IS THE POINT, ON EVERY TAB

Gold's average response to a 1% dollar move is about -1.1% over the last five years
with a t of -7, and one standard deviation of what that relationship does NOT explain
is 4%. Both are true. A page showing only the first would be a false-precision
machine, so every estimate here prints its unexplained band beside it and rows where
noise dominates are marked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import crossmove as cm


def _fmt(v: float, unit: str, dp: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{dp}f}{unit}"


# ---------------------------------------------------------------------------
def _tab_instruments() -> None:
    st.caption(
        "A move in one macro instrument, translated into the average move in the "
        "others over a window you choose. **Contemporaneous** — this sizes a move "
        "that has already happened; it does not forecast one."
    )

    c1, c2, c3, c4 = st.columns([0.30, 0.22, 0.24, 0.24])
    with c1:
        driver = st.selectbox("Move in", list(cm.INSTRUMENTS), index=4, key="cm_driver")
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
    horizon, years = cm.HORIZONS[hz_label], cm.LOOKBACKS[lb_label]

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
        "Read as": ["noise dominates" if abs(r.implied) < r.band_1sd * 0.5
                    else ("weak" if abs(r.t) < 2 else "clear")
                    for r in t.itertuples()],
    })
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(cm.caveat(driver, horizon, years))

    strong = t[(t["t"].abs() >= 2) & (t["implied"].abs() >= t["band_1sd"] * 0.5)]
    if strong.empty:
        st.info("No instrument moves enough with this driver, at this horizon and "
                "over this window, to stand clear of its own noise. That is a "
                "result, not a gap — it says the pairing is not one to lean on.")

    with st.expander("Why the numbers change when you change the window"):
        st.markdown(
            "These are **pairwise** relationships: gold against the dollar alone, "
            "including whatever else moved alongside the dollar. That is the right "
            "question for *what typically happens*, and a different one from gold's "
            "response with yields held fixed — the Gold Engine's Drivers tab fits "
            "that jointly and the two disagree on purpose.\n\n"
            "Relationships also genuinely drift. Gold moves about **−2.5%** per 1% "
            "on the dollar measured over the last year, **−1.1%** over five or ten, "
            "and **−0.7%** over the whole history. Comparing two lookbacks is the "
            "fastest way to see which regime you are quoting."
        )

    with st.expander("Full matrix — every pair"):
        st.caption("Row instrument's average move per **one unit** of the column "
                   "instrument (1% for prices, 1bp for yields, 1pt for VIX). "
                   "Read down a column to shock that instrument.")
        try:
            st.dataframe(cm.matrix(horizon=horizon, years=years).round(3),
                         use_container_width=True)
        except Exception as e:
            st.caption(f"Matrix unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
def _tab_fed() -> None:
    from src import macrochain as mc

    st.caption(
        "What a change in **policy pricing** is worth to gold and the dollar. "
        "The input is a **surprise** — the gap between a view and what the curve "
        "already carries."
    )
    st.warning(
        "**If the strip already prices it, delivering it moves nothing.** A client "
        "who says *two hikes by December* usually means two hikes will happen, not "
        "two more than the market holds. Read what is priced off the SOFR strip "
        "(STIR Paths → Fed) and enter only the difference here.",
        icon="⚠️",
    )

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        bp = st.number_input("Repricing vs the curve (bp)", value=25.0, step=5.0,
                             format="%.0f", key="cm_fed_bp",
                             help="Positive = more hawkish than the curve. "
                                  "One 25bp hike more than priced = 25.")
    if bp == 0:
        st.info("A view that matches the curve implies no move. That is the point.")
        return

    try:
        s = mc.fed_scenario(bp)
    except Exception as e:
        st.warning(f"Could not compute: {type(e).__name__}: {e}")
        return

    lo, hi = s["gold_range_pct"]
    a, b, c = st.columns(3)
    a.metric("Dollar (DXY)", f"{s['dollar_pct']:+.2f}%")
    b.metric("10y real yield", f"{s['real10_bp']:+.1f}bp",
             f"breakeven {s['breakeven_bp']:+.1f}bp", delta_color="off")
    c.metric("Gold", f"{lo:+.1f}% to {hi:+.1f}%",
             f"±{s['unexplained_1sd_pct']:.1f}% unexplained", delta_color="off")

    st.markdown("##### Where gold's move comes from")
    p = s["gold_parts"]
    st.dataframe(pd.DataFrame([
        {"Channel": "Real yields", "Contribution": f"{p['via_real_yield']:+.2f}%"},
        {"Channel": "The dollar", "Contribution": f"{p['via_dollar']:+.2f}%"},
        {"Channel": "Breakevens", "Contribution": f"{p['via_breakeven']:+.2f}%"},
        {"Channel": "— structural total", "Contribution": f"{s['gold_structural_pct']:+.2f}%"},
        {"Channel": "— measured directly", "Contribution": f"{s['gold_reduced_form_pct']:+.2f}%"},
    ]), hide_index=True, use_container_width=True)

    st.caption(
        f"Two estimates on purpose. **Measured directly** regresses gold on policy "
        f"repricing in one step. **Structural** rebuilds the same effect from its "
        f"channels, which shows where the move comes from but multiplies three "
        f"estimates together. They disagree by design — quote the range, not an "
        f"endpoint. R² {s['r2']:.2f}; one standard deviation of the unexplained is "
        f"**±{s['unexplained_1sd_pct']:.1f}% over the same month**, which is several "
        f"times the effect itself."
    )

    with st.expander("The distinction that changes the sign"):
        st.markdown(
            "**Rates up → gold down holds only for *real* rates.** Nominal yield = "
            "real yield + breakeven inflation, and the halves push gold in opposite "
            "directions: real +25bp is worth about **−1.33%**, breakeven +25bp about "
            "**+0.40%**.\n\n"
            "So it matters *why* yields are moving. A Fed getting ahead of inflation "
            "(real up, breakevens contained) is bearish gold. A Fed seen as falling "
            "behind (breakevens widening) is not. Regressing gold on the nominal "
            "yield alone averages the two and gets an R² of **0.025** against "
            "**0.26** for the split version."
        )


# ---------------------------------------------------------------------------
def _tab_releases() -> None:
    from src import metalevents as me

    st.caption(
        "Which scheduled US releases actually move which metal, measured in the "
        "London AM→PM fix window that brackets the 08:30 ET print."
    )
    try:
        tab = me.comparison_table()
        res = me.cross_metal_payroll_test()
    except Exception as e:
        st.warning(f"Release study unavailable: {type(e).__name__}: {e}")
        return
    if tab.empty:
        st.info("No release study on disk yet — run `python src/metalevents.py --study`.")
        return

    metals = [m for m in me.STUDY_METALS if m in tab.columns]
    show = pd.DataFrame({"Release": tab["release"]})
    for m in metals:
        show[m.title()] = [f"{v:.2f}×" + ("  ✓" if s else "")
                           for v, s in zip(tab[m], tab[f"{m}_sig"])]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(
        "Average absolute move in the release window ÷ the same figure on ordinary "
        "days — matched for year and weekday, and excluding days carrying any of the "
        "other releases listed. **✓ survives a multiple-comparison correction across "
        "the five releases.** Last ten years."
    )

    if res.get("comparisons"):
        st.markdown("##### Is gold's payroll premium genuinely bigger?")
        st.dataframe(pd.DataFrame([
            {"Comparison": f"Gold − {c['vs'].title()}",
             "Difference": f"{c['diff']:+.2f}×",
             "95% CI": f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]",
             "p": f"{c['p']:.4f}",
             "Survives correction": "yes" if c["survives_correction"] else "no"}
            for c in res["comparisons"]]), hide_index=True, use_container_width=True)
        st.caption(
            "One result clearing a threshold and another failing it does **not** "
            "establish that the two differ — that inference is the "
            "significant-versus-not-significant fallacy. The difference is "
            f"bootstrapped directly, {res['draws']:,} resamples. {me.verdict(res)}."
        )

    st.info(
        "**Silver is not in this table**, and its absence is structural rather than "
        "an omission. The LBMA Silver Price is a single noon auction, so silver has "
        "no AM→PM window — and that window is the only one in which any effect was "
        "detectable for gold. Measured fix-to-fix over the full 24 hours, even "
        "gold's payroll premium falls to 1.16× and stops being significant.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
def render() -> None:
    st.subheader("🧭 Macro Compass")
    st.caption(
        "How the macro complex moves together — instrument against instrument, "
        "against Fed pricing, and around the releases. Everything here is "
        "**contemporaneous**: it orients you against what has happened and does not "
        "forecast what will."
    )
    t1, t2, t3 = st.tabs(["📊 Instruments", "🏛️ Fed repricing", "📅 Releases"])
    with t1:
        _tab_instruments()
    with t2:
        _tab_fed()
    with t3:
        _tab_releases()
