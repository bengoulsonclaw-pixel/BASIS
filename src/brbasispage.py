"""brbasispage.py — the 🇧🇷 Brazil basis section of Curve / RV (engine: brbasis.py).

Same grammar as the Curve/RV Monitor tab — terminal table, ±2σ bar, decade-range bar,
detail chart with a ±2σ band and a z strip — so the two read as one module.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from . import brand, brbasis


@st.cache_data(show_spinner=False, ttl=1800)
def _hist(_mtime: float) -> pd.DataFrame:
    return brbasis.load()


def _load() -> pd.DataFrame:
    mt = brbasis.HIST.stat().st_mtime if brbasis.HIST.exists() else 0.0
    return _hist(mt)


def _drivers_txt(att, dp: int) -> str:
    """'US +0.8 · BR −21.2 · BRL +22.1' — the 1w attribution, compact, biggest first."""
    if not att:
        return "—"
    parts = [("US", att["us"]), ("Local", att["local"]), ("BRL", att["fx"])]
    parts = [(k, v) for k, v in parts if abs(v) >= 0.5 * 10 ** -dp]
    parts.sort(key=lambda kv: -abs(kv[1]))
    return " · ".join(f"{k} {v:+,.{dp}f}" for k, v in parts) or "flat"


def render() -> None:
    st.caption(
        "Brazilian prices against their US benchmarks, converted to the **US contract's own "
        "unit** (BRL legs at the day's PTAX) — the Brazil-vs-US basis. Sources are free and "
        "official: B3's daily price report for coffee, corn and cattle; CEPEA/ESALQ's "
        "Paranaguá indicator for soy (B3's soy future just mirrors CME); A3 Mercados' Rosario "
        "soy and corn futures (US$/t) for **Argentina**; CEPEA's mill-net "
        "hydrous ethanol for **sugar–ethanol parity** (the price at which a Brazilian mill "
        "earns the same from sugar as from ethanol). US legs are actual "
        "front settles, never back-adjusted. Scored like the Curve/RV Monitor: a rolling "
        "z-score and a percentile of the full history.")

    hist = _load()
    c0, c1, _ = st.columns([1.15, 0.85, 2.6], vertical_alignment="bottom")
    _win_opts = ["6 months (126d)", "1 year (252d)", "2 years (504d)"]
    win_lbl = c0.selectbox("Z-score window", _win_opts, index=1, key="brb_window")
    window = int(win_lbl.split("(")[1].rstrip("d)"))
    threshold = float(c1.number_input("Flag threshold (σ)", 0.5, 4.0, brbasis.Z_THRESHOLD,
                                      0.25, key="brb_thr"))
    mon = brbasis.monitor(hist, window, threshold)
    if mon.empty:
        st.info("No Brazil basis history yet — it builds on the next pull "
                "(B3 price-report archive + CEPEA).")
        return

    flagged = mon[mon["signal"] != "—"]
    if not flagged.empty:
        st.markdown("**Stretched now:** " + " · ".join(
            f"{r['name'].split(' — ')[0]} **{r['signal']}** ({r['z']:+.1f}σ, "
            f"{r['pctl']:.0f}th %ile)" for _, r in flagged.iterrows()))

    _esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    cols = [
        {"key": "name", "label": "Basis", "help": "Hover each name for its definition"},
        {"key": "legs", "label": "Local / US", "align": "right",
         "help": "Today's two legs (local market / US benchmark), both in the US unit, with the contracts compared underneath"},
        {"key": "level_txt", "label": "Level", "align": "right",
         "help": "Brazil minus US (cattle: Brazil ÷ US), in the US contract's unit"},
        {"key": "chg5d", "label": "1w Δ", "color": True, "fmt": "{:+,.2f}",
         "help": "Change over five sessions — blank when either leg rolled inside the week"},
        {"key": "drivers", "label": "1w drivers",       # text cell: wraps, unlike mono num
         "help": "What moved it this week, summing exactly to the 1w Δ: the US leg, the "
                 "local price in its own currency, and the real (BRL). Coffee is "
                 "USD-quoted on B3, so it has no BRL part."},
        {"key": "z", "label": "Z", "align": "right", "fmt": "{:+.2f}",
         "help": "Standard deviations from the rolling mean over the chosen window"},
        {"key": "zpic", "label": "±2σ", "zbar": True, "keep_case": True,
         "help": "The z-score on a ±2σ scale"},
        {"key": "pctl", "label": "Hist %ile", "align": "right", "fmt": "{:.0f}",
         "help": "Share of the stored history with the basis at or below today"},
        {"key": "spctl_txt", "label": "Seas %ile", "align": "right",
         "help": "Harvest-seasonal lines only (◔): today ranked against the same ±2 weeks "
                 "of the calendar in prior years — 'low for the time of year?'"},
        {"key": "hl", "label": "½-life", "align": "right",
         "help": "OU half-life, roll days excluded"},
        {"key": "signal", "label": "Signal",
         "help": "Rich / cheap vs the other leg once |z| clears the threshold — an "
                 "observation against the basis's own history, not a recommendation"},
    ]
    _sub = 'style="display:block;font-size:.76em;font-weight:400;opacity:.62;white-space:nowrap"'

    def _row(r) -> dict:
        dp = int(r["dp"])
        seas = " ◔" if r["seasonal"] else ""
        short, _, rest = str(r["name"]).partition(" — ")
        # Short bold product name, the comparison underneath: keeps the first column narrow
        # enough that the mono number columns don't squeeze it into a four-line wrap.
        name = (f'<span title="{_esc(r["desc"])}" style="cursor:help;white-space:nowrap;'
                f'border-bottom:1px dotted rgba(128,128,128,.55)">{_esc(short)}</span>{seas}'
                f'<span {_sub}>{_esc(rest)}</span>')
        legs = (f'{r["br_px"]:,.1f} / {r["us_px"]:,.1f}'
                f'<span {_sub}>{_esc(r["br_contract"])} v {_esc(r["us_contract"])}</span>')
        return {
            "name": name, "legs": legs,
            "level_txt": f"{r['level']:+,.{dp}f} {r['unit']}" if r["unit"] != "×"
                         else f"{r['level']:.{dp}f}×",
            "chg5d": None if pd.isna(r["chg5d"]) else float(r["chg5d"]),
            "drivers": (f'<span style="font-size:.82em;font-family:var(--basis-mono);'
                        f'display:inline-block;min-width:11em">{_drivers_txt(r["att5"], dp)}</span>'),
            "z": float(r["z"]), "zpic": float(r["z"]), "pctl": float(r["pctl"]),
            "spctl_txt": "—" if pd.isna(r["spctl"]) else f"{r['spctl']:.0f}",
            "hl": "—" if pd.isna(r["half_life"]) else f"{r['half_life']:.0f}d",
            # z has its own column — repeating it here wrapped the cell and pushed the
            # table into a sideways scroll
            "signal": (r["signal"] if r["signal"] == "—"
                       else f'<span style="white-space:nowrap">{r["signal"]}</span>'),
        }

    is_ar = mon["key"].str.startswith("ar_")
    for title, part, src in (
            ("Brazil vs US", mon[~is_ar], "B3 · CEPEA · BCB PTAX"),
            ("Argentina vs US", mon[is_ar], "A3 Mercados Rosario, US$/t")):
        if part.empty:
            continue
        brand.panel_header(title, right=f"{src} · as of {part['asof'].max()}")
        brand.terminal_table([_row(r) for _, r in part.iterrows()], cols)
    st.caption("◔ = harvest-seasonal basis (safrinha corn lands Jun–Aug, soy Feb–Apr). For "
               "these, read **Seas %ile** — today against the same weeks in prior years — "
               "rather than the full-history percentile.")

    # ---- detail ----------------------------------------------------------------
    st.divider()
    brand.panel_header("Basis detail", right=f"window {window}d · flag ±{threshold:g}σ")
    d0, d1 = st.columns([2.2, 1.4])
    sel = d0.selectbox("Basis", mon["name"].tolist(), key="brb_sel",
                       label_visibility="collapsed")
    rng = d1.radio("Range", ["Full", "5y", "2y", "1y"], horizontal=True, key="brb_rng",
                   label_visibility="collapsed")
    row = mon[mon["name"] == sel].iloc[0]
    years = {"Full": None, "5y": 5.0, "2y": 2.0, "1y": 1.0}[rng]
    cd = brbasis.chart_data(row["key"], hist, window, years)
    st.caption(row["desc"])
    dp = int(row["dp"])
    m0, m1, m2, m3 = st.columns(4)
    m0.metric(f"Level ({row['unit']})", f"{row['level']:+,.{dp}f}",
              None if pd.isna(row["chg1d"]) else f"{row['chg1d']:+,.{dp}f} on the day",
              delta_color="off")
    m1.metric(f"Z ({window}d)", f"{row['z']:+.2f}σ")
    m2.metric("History percentile", f"{row['pctl']:.0f}th",
              help=f"Against every session since {row['first']}.")
    m3.metric("Half-life", "—" if pd.isna(row["half_life"]) else f"≈{row['half_life']:.0f}d")
    if row["seasonal"] and not pd.isna(row["spctl"]):
        st.caption(f"Seasonal percentile **{row['spctl']:.0f}th** — against the same ±2 weeks "
                   f"of the calendar in {row['seas_years']} prior seasons.")
    att = row["att5"]
    if att:
        u = row["unit"] if row["unit"] != "×" else "×"
        st.caption(
            f"**This week ({att['total']:+,.{dp}f} {u}):** US leg {att['us']:+,.{dp}f} · "
            f"local price in its own currency {att['local']:+,.{dp}f} · "
            f"BRL {att['fx']:+,.{dp}f}. The three sum exactly to the move.")

    cc = brand.chart_colors()
    x = alt.X("date:T", title=None)
    base = alt.Chart(cd)
    band = base.mark_area(opacity=0.13, color=cc["accent"]).encode(
        x=x, y=alt.Y("lower:Q", title=f"basis ({row['unit']})", scale=alt.Scale(zero=False)),
        y2="upper:Q")
    mean_ln = base.mark_line(strokeDash=[5, 3], color=cc["muted"], strokeWidth=1.5).encode(
        x=x, y="mean:Q")
    ln = base.mark_line(color=cc["series"], strokeWidth=2.1).encode(
        x=x, y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T"),
                 alt.Tooltip("value:Q", title=f"basis ({row['unit']})", format=f",.{dp}f"),
                 alt.Tooltip("br_px:Q", title=row["b3_label"], format=",.2f"),
                 alt.Tooltip("us_px:Q", title=row["us_label"], format=",.2f"),
                 alt.Tooltip("br_contract:N", title="Brazil contract"),
                 alt.Tooltip("us_contract:N", title="US contract"),
                 alt.Tooltip("z:Q", format="+.2f")])
    brand.show_chart((band + mean_ln + ln).properties(height=320))
    z_rules = alt.Chart(pd.DataFrame({"y": [threshold, 0.0, -threshold]})).mark_rule(
        color=cc["muted"], strokeDash=[4, 3], strokeWidth=1).encode(y="y:Q")
    z_ln = base.mark_line(color=cc["accent"], strokeWidth=1.6).encode(
        x=x, y=alt.Y("z:Q", title="z"), tooltip=[alt.Tooltip("date:T"),
                                                 alt.Tooltip("z:Q", format="+.2f")])
    brand.show_chart((z_rules + z_ln).properties(height=120))
    st.caption(
        f"Mean ({window}d) **{row['mean']:,.{dp}f}** · 1σ **{row['sigma']:,.{dp}f} "
        f"{row['unit']}** · history since **{row['first']}** ({row['days']:,} sessions). "
        "Levels are observations against the basis's own history, not a recommendation.")
