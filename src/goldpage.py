"""goldpage.py — the 🥇 Gold Engine page.

Everything the gold work produces, in one place. It lives in src/ rather than inside
app.py because it carries real content (four tabs, several tables) and app.py is
already 16k lines; the page function is imported and dispatched like any other.

The four tabs, and why each exists
----------------------------------
  Week Ahead   the client report — what is scheduled, and what history says about it
  Drivers      the part of this work that holds up: gold % per unit driver move, a
               scenario builder, and the fair-value gap
  Evidence     THE MOST IMPORTANT TAB. It shows each result cut several different
               ways, so a reader can judge whether a number is established rather
               than trusting a verdict. That is not a stylistic preference: over the
               course of building this, the same CPI statistic was reported three
               different ways as the slicing changed, and the instability WAS the
               answer. A page that shows only the latest cut hides exactly the thing
               a reader needs.
  Health       store coverage, staleness, and the backtest run log

What this page must never do
----------------------------
Imply a directional forecast. The engine established across 21 years of point-in-time
data that gold's direction is not forecastable from these drivers — pooled or within
any regime — and that no model beat buying and holding. Every number here is a
measured historical relationship or a stated-assumption scenario.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import reportkit

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "gold_store"
REPORT_PDF = ROOT / "reports" / "Gold_Week_Ahead.pdf"


def _load(name: str):
    p = STORE / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stale_note(path: Path) -> str:
    if not path.exists():
        return "never built"
    # pd.Timestamp(epoch, unit="s") is UTC; pd.Timestamp.now() is local. On this
    # machine (UTC-5) the difference for a file written two hours ago was -3h,
    # whose .days is -1, so the page read "built -1d ago".
    built = datetime.fromtimestamp(path.stat().st_mtime)
    age = (datetime.now() - built).days
    return "built today" if age <= 0 else f"built {age}d ago"


# ---------------------------------------------------------------------------
def _tab_week_ahead() -> None:
    st.caption("What is scheduled, how gold has behaved around those releases before, "
               "and what a given move in rates or the dollar would be worth. "
               "**No forecast of the gold price** — see Evidence for why.")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🔄 Rebuild report", type="primary", key="gold_rebuild"):
            with st.spinner("Rendering the Gold Week Ahead PDF…"):
                r = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "goldreport.py"),
                     str(REPORT_PDF)],
                    capture_output=True, text=True, timeout=900, cwd=str(ROOT))
            if r.returncode == 0:
                st.success("Rebuilt.")
            else:
                st.error("Report build failed.")
                st.code((r.stderr or r.stdout or "")[-2000:])
    with c2:
        st.caption(f"`{REPORT_PDF.name}` — {_stale_note(REPORT_PDF)}")

    if REPORT_PDF.exists():
        st.download_button("⬇️  Download Gold — Week Ahead",
                           data=REPORT_PDF.read_bytes(), file_name=REPORT_PDF.name,
                           mime="application/pdf", key="gold_dl")
    else:
        st.info("No report yet — press Rebuild.")

    ev = _load("event_study.json")
    if not ev:
        return
    st.markdown("##### Release-day behaviour, last decade")
    rows = []
    for name, b in (ev.get("results", {}).get("am_to_pm_intraday") or {}).items():
        r = b.get("recent") or {}
        if not r:
            continue
        rows.append({"Release": name.split(" (")[0],
                     "Move vs ordinary day": r.get("ratio_matched"),
                     "Releases": r.get("n_events"),
                     "Survives correction": "yes" if r.get("significant") else "no"})
    if rows:
        df = pd.DataFrame(rows).sort_values("Move vs ordinary day", ascending=False)
        st.dataframe(df.style.format({"Move vs ordinary day": "{:.2f}×"}),
                     hide_index=True, use_container_width=True)
        st.caption("Only the employment report clears a multiple-comparison correction "
                   "across the five releases tested. A ratio above 1 with 'no' in the "
                   "last column is not evidence of anything.")


# ---------------------------------------------------------------------------
def _tab_drivers() -> None:
    s = _load("sensitivities.json")
    if not s:
        st.info("No sensitivities yet — run `python src/goldsens.py`.")
        return

    # F18/F31: say WHEN this was computed. The tabs read JSON on disk that nothing
    # refreshed until the daily pull was wired up, so a reader had no way to tell a
    # figure computed this morning from one left over from a hand-run weeks ago.
    if s.get("asof"):
        st.caption(f"Fit as of **{s['asof']}**"
                   + (f" · rebuilt {s['built'][:16].replace('T', ' ')}"
                      if s.get("built") else "")
                   + " · refreshed by the daily pull.")

    a, b, c = st.columns(3)
    # R-squared is the share of VARIANCE explained, not of a move. The old wording
    # invited reading 28% as "gold moves 28% with the drivers".
    a.metric("Variance explained", f"{s['r2_macro'] * 100:.0f}%",
             help="R-squared: the share of the VARIANCE of one-month gold moves that "
                  "the exogenous drivers account for — not the share of any given "
                  "move. Adding ETF and positioning flows raises it, but they are "
                  "co-movers — they rise because gold rallied as much as the reverse "
                  "— so they are excluded.")
    b.metric("Unexplained band", f"±{s['unexplained_sd_pct']:.1f}%",
             help="One standard deviation of the move the drivers do not explain. "
                  "Wider than most of the scenario effects below.")
    gap = s.get("fair_value_gap_pct")
    fvs = s.get("fair_value") or {}
    # delta_color="off": the arrow rendered green-up on a positive gap, which reads
    # as bullish on a page whose stated rule is never to imply a directional view.
    # This is a residual, not a signal.
    c.metric("Gold vs drivers, 12m", f"{gap:+.1f}%" if gap is not None else "—",
             (f"{reportkit.ordinal(s.get('fair_value_gap_pctile') or 0)} pctile"
              + (f" · spans {fvs['lo_pct']:+.1f}% to {fvs['hi_pct']:+.1f}%"
                 if fvs.get("spread_pct") else "")) if gap is not None else "",
             delta_color="off",
             help="The part of gold's twelve-month move the drivers do NOT account "
                  "for. It is a residual, not a valuation and not a signal — the "
                  "research behind this page found no evidence that the gap predicts "
                  "gold's subsequent direction. Averaged over every non-overlapping "
                  "sampling; the span is how much it moves with that arbitrary choice.")

    st.markdown("##### What each driver is worth")
    d = pd.DataFrame(s["sensitivities"])
    d["clears |t|>2"] = d["t_stat"].abs() > 2
    st.dataframe(
        d[["driver", "move", "gold_pct", "t_stat", "clears |t|>2"]]
        .rename(columns={"driver": "Driver", "move": "Move",
                         "gold_pct": "Implied gold", "t_stat": "t"})
        .style.format({"Implied gold": "{:+.2f}%", "t": "{:+.1f}"}),
        hide_index=True, use_container_width=True)
    st.caption("Fitted jointly on point-in-time data with Newey-West errors at the "
               "one-month horizon. Rows that do not clear |t| = 2 should be read as noise.")

    st.markdown("##### Scenario")
    st.caption("State a view on the drivers; the fitted sensitivities price it. "
               "Anything left at zero is assumed unchanged, not set to an average.")
    cols = st.columns(3)
    moves, labels = {}, {r["feature"]: r for r in s["sensitivities"]}
    # (label, feature, lo, hi, default, step, format, to_native)
    #
    # `to_native` converts what the SLIDER shows into the units the FEATURE is
    # measured in. dxy_chg_20d is a fractional 20-day change, so a 5% dollar move is
    # 0.05 — and a slider running -0.05..0.05 formatted "%.1f%%" printed that as
    # "0.1%". Every dollar scenario on this page was labelled 100x too small: the
    # user dragged to what read as a tenth of a percent and priced a five-percent
    # move. The slider now runs in percent and converts on the way in.
    presets = [
        ("10y real yield", "real_yield_10y_chg_20d", -0.50, 0.50, 0.0, 0.05, "%.2fpp", 1.0),
        ("Dollar (DXY)",   "dxy_chg_20d",           -5.0,  5.0,  0.0, 0.25, "%.2f%%", 0.01),
        ("VIX (sd)",       "vix_z_1y",              -2.0,  2.0,  0.0, 0.25, "%.2f",   1.0),
    ]
    native = {}
    for i, (lbl, feat, lo, hi, dflt, step, fmt, conv) in enumerate(presets):
        if feat in labels:
            shown = cols[i % 3].slider(lbl, lo, hi, dflt, step, format=fmt,
                                       key=f"gold_sc_{feat}")
            moves[feat] = shown
            native[feat] = shown * conv
    total = 0.0
    for f, v in native.items():
        row = labels.get(f)
        if row and row["gold_pct"]:
            per_unit_step = {"real_yield_10y_chg_20d": 0.10, "dxy_chg_20d": 0.01,
                             "vix_z_1y": 1.0}.get(f, 1.0)
            total += np.log1p(row["gold_pct"] / 100.0) * (v / per_unit_step)
    st.metric("Implied gold move", f"{np.expm1(total) * 100:+.2f}%",
              f"±{s['unexplained_sd_pct']:.1f}% unexplained (1 s.d.)")

    at = s.get("attribution") or {}
    if at:
        st.markdown("##### The month just gone")
        st.write(
            f"Gold moved **{at['actual_pct']:+.2f}%** over the last "
            f"{at['window_days']} trading days. The named drivers account for "
            f"**{at['explained_pct']:+.2f}%**, gold's unconditional drift for a further "
            f"**{at.get('drift_pct', 0):+.2f}%**, and the remaining "
            f"**{at['unexplained_pct']:+.2f}%** is not explained by rates, the dollar, "
            f"credit or risk appetite. Drift is listed separately because it is not "
            f"something a driver did.")


# ---------------------------------------------------------------------------
def _tab_evidence() -> None:
    st.caption("Each result cut several ways. **A relationship that is real barely "
               "moves when the slicing changes; one that wanders was never "
               "established, whatever a single p-value says.** Judge the row, not "
               "the number.")

    ev = _load("event_study.json")
    if ev:
        st.markdown("##### Release-day premium, by period")
        # Built by goldreport.event_behaviour() — the SAME function the client PDF
        # uses. This tab had its own copy of the rule ("elevated in every period" if
        # min > 1.0), and when the era baseline was corrected the two drifted apart:
        # the page went on making a claim the report had already stopped making.
        # There is one verdict rule in this codebase and it lives in goldreport.
        from src import goldreport
        rows = []
        for e in goldreport.event_behaviour():
            row = {"Release": e["label"]}
            for b in e["buckets"]:
                row[b["label"].replace("y ago", "")] = b["ratio"]
            row["Verdict"] = e["verdict"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Equal-width, non-overlapping five-year periods counting back from "
                   "today — so the leftmost is the market you are trading now, and "
                   "each is an independent sample rather than a rolling window of "
                   "largely the same data.")

    dg = _load("diagnostics.json")
    if dg:
        st.markdown("##### Does anything LEAD gold?")
        n_lead = dg.get("n_leading", 0)
        st.write(
            f"Of {dg.get('n_tests', 0):,} simultaneous tests, "
            f"**{dg.get('n_clearing_bonferroni', 0)} features clear** a "
            f"Bonferroni-adjusted threshold — and "
            f"**{dg.get('n_coincident', 0)} of them are coincident, {n_lead} lead.**")
        st.caption("Correlation of each feature against the daily gold return at lags "
                   "0 to 60 days. A feature scoring only at lag 0 explains gold; it "
                   "does not forecast it. Run on daily returns rather than the forward "
                   "targets, because overlapping forward returns make significance "
                   "meaningless.")

    rg = _load("regime_pretest.json")
    if rg:
        st.markdown("##### …within a regime?")
        st.write(f"**{rg.get('verdict', '')}**")
        st.caption("A full-sample correlation cannot detect a lead that exists in one "
                   "regime and reverses in another, so the scan was re-run inside each "
                   "volatility/trend state. Price-derived features are excluded as "
                   "circular: the trend regime is defined from the price, so a "
                   "price-derived feature is constrained by construction inside it.")

    runs = STORE / "backtest_runs.jsonl"
    if runs.exists():
        try:
            last = [json.loads(x) for x in runs.read_text(encoding="utf-8").splitlines()
                    if x.strip()][-1]
            st.markdown("##### Did any model beat buying and holding?")
            rows = []
            for tgt, per in (last.get("results") or {}).items():
                for nm, m in per.items():
                    if m.get("insufficient"):
                        continue
                    tim = m.get("time_in_market")
                    hit = m.get("hit_rate")
                    # A row that never opened a position is not a strategy result. The
                    # harness knew this — it prints "never entered the market" on the
                    # CLI and logs a warning — but the page dropped both and rendered
                    # the row as an ordinary competitor, with "nan%" where the hit rate
                    # would be. Say it in the table instead.
                    flat = (tim is not None and tim == 0)
                    note = ("never entered the market" if flat else
                            "no directional calls" if hit != hit else "")
                    xs = m.get("excess_vs_buyhold")
                    rows.append({
                        "Horizon": tgt.replace("fwd_ret_", ""), "Model": nm,
                        "Hit": hit,
                        "In market": tim,
                        "vs always-long": m.get("edge_vs_always_long"),
                        # excess_vs_buyhold is a difference of CUMULATIVE LOG returns,
                        # which the page printed as a bare 3-decimal number beside two
                        # percentages. exp() of it is the ratio of terminal wealth, so
                        # expm1() states it as the percentage it actually is.
                        "vs buy & hold": (float(np.expm1(xs)) if xs is not None
                                          and np.isfinite(xs) else None),
                        "Note": note})
            if rows:
                st.dataframe(pd.DataFrame(rows).style.format(
                    {"Hit": "{:.1%}", "In market": "{:.0%}",
                     "vs always-long": "{:+.1%}", "vs buy & hold": "{:+.1%}"},
                    na_rep="—"),
                    hide_index=True, use_container_width=True)
                st.caption("Walk-forward, development window only. The three-year "
                           "holdout has never been opened — there is no candidate "
                           "worth spending it on. **vs buy & hold** is terminal wealth "
                           "against holding the metal over the same dates; "
                           "**In market** is the share of days holding a position.")
                for w in dict.fromkeys(last.get("warnings") or []):
                    st.caption(f"⚠️ {w}")
        except Exception as e:
            # Was `pass`. A malformed run record erased this whole section with no
            # trace, so the page looked like a design that omits backtests rather
            # than one whose backtest block just failed to read.
            st.warning(f"Backtest results could not be read: {e}")
            st.caption(f"Source: `{runs.relative_to(ROOT)}` — rerun "
                       "`python src/goldbacktest.py --deep` to rebuild it.")


# ---------------------------------------------------------------------------
def _tab_health() -> None:
    try:
        from src import goldstore
        cov = goldstore.coverage()
        flags = goldstore.stale_flags()
    except Exception as e:
        st.warning(f"Store unreadable: {e}")
        return
    if cov.empty:
        st.info("Store is empty — run `python src/goldingest.py --all`.")
        return
    a, b, c = st.columns(3)
    a.metric("Series", len(cov))
    b.metric("Observations", f"{int(cov['rows'].sum()):,}")
    c.metric("Stale feeds", len(flags), delta_color="inverse")
    if flags:
        st.warning("Overdue: " + ", ".join(f.split(":")[-1] for f in flags))
        st.caption("A feed is flagged when its NEXT release is overdue — cadence plus "
                   "publication lag, in business days. Measured in calendar days it "
                   "flagged every daily series each weekend and buried the one feed "
                   "that had genuinely stopped.")
    st.dataframe(cov.reset_index(), hide_index=True, use_container_width=True, height=340)

    st.markdown("##### Which horizons can each feed inform?")
    st.dataframe(goldstore.horizon_matrix().reset_index(), hide_index=True,
                 use_container_width=True, height=300)
    st.caption("A series cannot drive a forecast over a window shorter than its own "
               "release cycle. Derived from cadence and publication lag, so it cannot "
               "drift out of date. **This is a diagnostic, not a constraint the model "
               "layer enforces** — the backtest fits the same 28-feature frame at "
               "every horizon, so features built from monthly releases do sit in the "
               "5-day fit and this table is how you see that. It is listed here as "
               "something to read, not something already applied.")


# ---------------------------------------------------------------------------
def render() -> None:
    st.subheader("🥇 Gold Engine")
    st.caption("Driver sensitivities, release-day behaviour and the weekly client "
               "report — built on a point-in-time store in which no value is visible "
               "before the day it was published.")
    t1, t2, t3, t4 = st.tabs(["📅 Week Ahead", "🎚️ Drivers", "🔍 Evidence", "🩺 Health"])
    with t1:
        _tab_week_ahead()
    with t2:
        _tab_drivers()
    with t3:
        _tab_evidence()
    with t4:
        _tab_health()
