"""Strategy Monitor - Streamlit dashboard.

One button per strategy. Click a strategy to see today's flagged
opportunities, tick the ones you want, and export a client-style PDF.

Run:  .venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, date, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import run_daily
from src.datafeed import (MODE, get_live_quote, get_history, get_history_ta,
                          get_implied_vol_history, get_realized_vol_history,
                          get_term_structure, stale_iv_reasons)
from src.specs import (SPECS, reflag_rows, trigger_default, save_trigger_default,
                       ta_report_defaults, save_ta_report_defaults)
from src import universe
from src import brand
from src import recipients
from src import automation
from src import alerts
from src import econ
from src import gitbackup
from src import fedpath
from src import volbt
from src import sectorcorr
from src import worldclock
from src import prodsearch
from src import expiries
from src import tascore
from src import markethours
from src import blocksizes
from src import futyield
from src import optbuilder
from src import equities
from src import eqfunda
from src import eqcorr
from src import eqdisp
from src.universe import INSTRUMENTS

ROOT = Path(__file__).parent
SIGNALS_FILE = ROOT / "data" / "signals" / "opportunities.parquet"
META_FILE = ROOT / "data" / "signals" / "meta.json"
REPORT_CLI = ROOT / "src" / "report.py"
MRREPORT_CLI = ROOT / "src" / "mrreport.py"
TRENDREPORT_CLI = ROOT / "src" / "trendreport.py"
VOLREPORT_CLI = ROOT / "src" / "volreport.py"
VOL_DETAIL_FILE = ROOT / "data" / "signals" / "volatility.parquet"
SKEWREPORT_CLI = ROOT / "src" / "skewreport.py"
SKEW_DETAIL_FILE = ROOT / "data" / "signals" / "skew.parquet"
TERMREPORT_CLI = ROOT / "src" / "termreport.py"
TERM_DETAIL_FILE = ROOT / "data" / "signals" / "termstructure.parquet"
COTREPORT_CLI = ROOT / "src" / "cotreport.py"
COT_DETAIL_FILE = ROOT / "data" / "signals" / "cot.parquet"
COT_HISTORY_FILE = ROOT / "data" / "signals" / "cot_history.parquet"
PCREPORT_CLI = ROOT / "src" / "pcreport.py"
PC_DETAIL_FILE = ROOT / "data" / "signals" / "putcall.parquet"
PC_HISTORY_FILE = ROOT / "data" / "signals" / "putcall_history.parquet"
OIREPORT_CLI = ROOT / "src" / "oireport.py"
USDAREACTION_CLI = ROOT / "src" / "usdareaction.py"
WASDEREPORT_CLI = ROOT / "src" / "wasdereport.py"
FLAGREPORT_CLI = ROOT / "src" / "flagreport.py"
FLAG_DETAIL_FILE = ROOT / "data" / "signals" / "flag_breakout.parquet"
CONVREPORT_CLI = ROOT / "src" / "convreport.py"   # the merged "Technical Analysis" report (was the Conviction Screen)
SNAPSHOT_CLI = ROOT / "snapshot.py"
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "manifest.json"

# Morning Coffee — the daily global-macro briefing. A separate project: it pulls
# Bloomberg + news, writes the macro commentary, renders a branded .docx and
# emails it to the desk. We run its main.py as a subprocess.
# The separate "Morning Coffee" macro-briefing project (optional). Point BASIS_MC_DIR at
# its folder on another PC; if it's absent the MC features show a "not found" note and the
# rest of the dashboard runs normally.
MORNING_COFFEE_DIR = Path(os.getenv("BASIS_MC_DIR", r"C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements"))
MORNING_COFFEE_CLI = MORNING_COFFEE_DIR / "main.py"


def _to_et(local_str) -> str:
    """Convert the snapshot's machine-local 'YYYY-MM-DD HH:MM:SS' capture time to
    'H:MM AM ET · DD Mon YYYY'. This box runs on UTC-5, so it's converted to actual
    New York time (DST-aware), matching the Morning Coffee heatmap stamp."""
    try:
        raw = str(local_str)[:19]
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:                      # minute-precision stamps (signals as_of)
            dt = datetime.strptime(raw[:16], "%Y-%m-%d %H:%M")
        et = dt.replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(
            ZoneInfo("America/New_York"))
        t = et.strftime("%I:%M %p").lstrip("0")
        mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][et.month - 1]
        return f"{t} ET · {et.day:02d} {mon} {et.year}"
    except Exception:
        return str(local_str)


def morning_coffee_python() -> str:
    """Interpreter that can run the Morning Coffee main.py — it needs anthropic,
    python-docx, PIL and xbbg. The dashboard venv lacks anthropic/docx and the
    project's own .venv lacks xbbg, so use the global CPython that runs it from
    the shell (newest pythoncore-*), falling back to whatever is on PATH."""
    base = Path(r"C:\Users\Ben\AppData\Local\Python")
    for exe in sorted(base.glob("pythoncore-*-64/python.exe"), reverse=True):
        if exe.exists():
            return str(exe)
    return shutil.which("python") or sys.executable


def run_morning_coffee() -> bool:
    """Run the Morning Coffee main.py (separate project + interpreter) as a subprocess;
    stash its status, log and the generated .docx in session_state so the Daily Briefing
    section below can show the result + a download. Returns True on success."""
    if not MORNING_COFFEE_DIR.exists():
        st.session_state["mc_ok"] = False
        st.session_state["mc_log"] = (
            f"Morning Coffee project not found at {MORNING_COFFEE_DIR}. It's a separate project; "
            "set the BASIS_MC_DIR environment variable to its folder to enable this, or leave it "
            "disabled on this PC — the rest of the dashboard is unaffected.")
        return False
    try:
        mc = subprocess.run(
            [morning_coffee_python(), str(MORNING_COFFEE_CLI)],
            cwd=str(MORNING_COFFEE_DIR),
            capture_output=True, text=True, timeout=600,
        )
        mc_log = (mc.stdout or "") + (("\n" + mc.stderr) if mc.stderr else "")
        mc_ok = mc.returncode == 0
    except subprocess.TimeoutExpired:
        mc_log, mc_ok = "Timed out after 10 minutes.", False
    except Exception as e:
        mc_log, mc_ok = f"Could not launch Morning Coffee: {e}", False
    st.session_state["mc_ok"] = mc_ok
    st.session_state["mc_log"] = mc_log
    try:        # persist for post-mortems (session_state dies with the tab)
        (ROOT / "data" / "mc_last_run.log").write_text(
            f"ok={mc_ok}\n{mc_log}", encoding="utf-8")
    except Exception:
        pass
    st.session_state.pop("mc_docx", None)
    match = re.search(r"Report \(EN\) saved to:\s*(.+\.docx)", mc_log)
    if mc_ok and match:
        saved = Path(match.group(1).strip())
        if saved.exists():
            st.session_state["mc_docx"] = saved.read_bytes()
            st.session_state["mc_docx_name"] = saved.name
    return mc_ok


def _regen_mc_heatmap() -> bool:
    """Rebuild the Morning Coffee heatmap (shown on Home) via the MC project's
    --heatmap-only mode. Best-effort: a failure (e.g. Bloomberg down) leaves the
    previous heatmap in place and never blocks the snapshot."""
    try:
        with st.spinner("Refreshing the Morning Coffee heatmap…"):
            r = subprocess.run(
                [morning_coffee_python(), str(MORNING_COFFEE_CLI), "--heatmap-only"],
                cwd=str(MORNING_COFFEE_DIR), capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


# The strategy book, grouped by theme for the sidebar nav. STRATEGY_ORDER (the flat
# list the rest of the app loops over) is derived from it, so adding a strategy to a
# group here adds it to the nav AND everywhere else at once.
NAV_GROUPS = {
    "Technical Analysis": list(tascore.TA_STRATEGIES),
    "Volatility":         ["Volatility", "Skew Volatility", "Vol Term Structure"],
    "Positioning & Flow": ["COT Reports", "Put/Call Ratios", "Open Interest"],
    "Fundamentals":       ["AG Fundamentals"],
}
STRATEGY_ORDER = [s for group in NAV_GROUPS.values() for s in group]
# The price-based technical strategies the 🔬 Technical Analysis overview aggregates (its nav group).
TA_STRATEGIES = NAV_GROUPS["Technical Analysis"]
STRATEGY_BLURB = {
    "Mean Reversion": "Spreads / ratios stretched from their recent mean (z-score). "
                      "Stretched = a potential fade back toward the mean.",
    "Trend": "Time-series momentum: fast/slow moving-average crossover, confirmed by the 3-month return.",
    "MA Crossover": "50/200 golden / death cross, confirmed by the 15-day EMA (on the trend side of the 50) "
                    "and the 3-month return: Long on a golden cross when the 15-EMA is above the 50 and "
                    "3-month momentum is positive; Short on a death cross when it's below and negative. "
                    "The 100-day MA is drawn as chart context. The position-trade horizon.",
    "MA Swing": "Faster swing version of the MA crossover — a 20/50 golden / death cross confirmed by the "
                "9-day EMA (on the trend side of the 20) and the 1-month return. More signals than the "
                "50/200 page but more whipsaw-prone; the 200-day is drawn as the big-trend backdrop.",
    "Flag Breakout": "Flag & pennant continuation patterns near their breakout: a sharp flagpole, then a "
                     "tight consolidation (parallel = flag, converging = pennant). Ranked by breakout "
                     "readiness (0–100) — how close price is to the trendline; bull patterns break up, bear "
                     "down. Each carries a measured-move target, stop and reward:risk, with volume "
                     "confirmation where available.",
    "Support & Resistance": "Tested horizontal levels from swing pivots — buy near support, sell near "
                            "resistance. Ranked by how close price sits to a strong level (0–100 proximity); "
                            "level strength = touches, and broken levels flip role. Pure price action.",
    "Fibonacci Retracement": "Auto-Fibonacci on each product's dominant swing — flags price reacting at a "
                             "key retracement (38.2 / 50 / 61.8% golden zone): buy the dip in an up-leg, sell "
                             "the rally in a down-leg. Carries a target (prior extreme), a stop beyond 78.6% and R:R.",
    "Breakout & Retest": "A level breaks on strong volume, then price pulls back to retest it (broken "
                         "resistance → new support, and the mirror). Flags the active retests, ranked by "
                         "how close price is to the flipped level; the cleanest momentum entry.",
    "Momentum (RSI/MACD)": "RSI (14) overbought/oversold + MACD (12/26/9) crosses, with RSI divergence as "
                           "the headline reversal warning (price higher-high while RSI lower-high = bearish, "
                           "and the mirror). A signed momentum score, bullish vs bearish.",
    "Bollinger Squeeze": "Bollinger Bands (20, 2σ): a squeeze — bandwidth in a low percentile of its own "
                         "year — flags compressed volatility coiling for a breakout; a close outside the "
                         "band is the break. Ranked by squeeze intensity (0–100).",
    "Elliott Wave": "Rules-based impulse wave count: adaptive ZigZag pivots checked against the three hard "
                    "Elliott rules, flagging wave-3 setups, wave-5 setups and completed five-wave sequences "
                    "(corrective risk). Ranked by wave fit (0–100) — how textbook the count's Fibonacci "
                    "proportions are; fixed income counted on yields.",
    "Ichimoku Cloud": "Ichimoku Kinko Hyo: price vs the cloud (Kumo), scored by confluence of the "
                      "Tenkan/Kijun cross, the future cloud colour and the lagging span. Flags fresh cloud "
                      "breakouts and TK crosses (score 0–100). Close-based (n-high/low from closes); fixed "
                      "income on yields.",
    "On-Balance Volume": "Cumulative volume signed by the day's direction — is volume BEHIND the move? "
                         "Flags OBV divergences (trend-failure warnings), volume-(un)confirmed breakouts and "
                         "hidden accumulation/distribution. FX futures excluded (real FX volume is OTC); "
                         "fixed income on yields.",
    "Money Flow Index": "Volume-weighted RSI (14d money flow): overbought/oversold on REAL flow — validating "
                        "or faulting RSI signals — early distribution/accumulation against the price trend, "
                        "and 50-line flow shifts. FX futures excluded (real FX volume is OTC); fixed income "
                        "on yields.",
    "Carry": "Roll yield from the shape of each curve - needs the Bloomberg strip (front vs deferred). Coming soon.",
    "Volatility": "1-month ATM implied vs ~1-month (21d) realized vol. Rich = options dear vs "
                  "delivered (sell vol); cheap = options underpricing moves (buy vol). "
                  "Ranked by the z-score of the implied−realized spread vs its own 1-year history.",
    "Skew Volatility": "Skew (90% put − 110% call)/ATM, z-scored vs its own 1-year history. Rich = puts "
                       "dear vs calls (sell skew); cheap = puts underpriced (buy skew). "
                       "Listed: 90/110% moneyness wings off the surface; "
                       "FX: OTC 25Δ risk reversal. Bonds & STIRs excluded.",
    "Vol Term Structure": "ATM implied curve 1M/3M/6M/12M. Slope = 3M−1M, z-scored vs its own 1-year "
                          "history: steep contango = front cheap (buy front); inverted = front rich "
                          "(sell front). Includes per-tenor implied-vs-realized.",
    "COT Reports": "CFTC Commitments of Traders — managed-money (commodities) / leveraged-funds "
                   "(financials) net positioning. Long bars up, short bars down, net line + price "
                   "overlay, with the COT Index (0–100, 3-year range) flagging crowded longs/shorts.",
    "Put/Call Ratios": "Options put/call ratio (puts ÷ calls) by product — open interest (standing "
                       "positioning, the headline) with traded volume (today's flow) alongside. Each "
                       "product's OI P/C is normalised to a 0–100 percentile vs its own 1-year range; "
                       "≥80 = put-heavy (defensive), ≤20 = call-heavy (bullish), plus 1-day shifts and "
                       "flow-vs-OI divergence. Extremes often read contrarian.",
    "Open Interest": "Fixed-income listed-option open interest as a strike × expiry-month heatmap: each cell is "
                     "the total open interest (puts + calls) struck there, shaded by size. The biggest strikes "
                     "show where positioning and dealer hedging concentrate — frequent pin / magnet levels into "
                     "expiry. The focus is the 11-product rates book (the 🏛️ report); any other product can be "
                     "explored ad-hoc (toggle below, pulls live).",
    "AG Fundamentals": "USDA fundamentals: report-calendar event risk (WASDE / Crop Production / "
                       "Grain Stocks / Plantings / Acreage / Cattle on Feed / Hogs & Pigs) plus NASS "
                       "stocks-tightness percentiles. Positioning lives on the COT Reports page.",
}

# Strategies with a dedicated visual client report (a button on their page, built
# from the full cross-section rather than the hand-ticked rows).
REPORTS = {
    "Volatility": {
        "cli": VOLREPORT_CLI, "detail": VOL_DETAIL_FILE, "key": "vol_pdf",
        "file": "Volatility_Report.pdf",
        "label": "📈 Generate Volatility Report (visual PDF)",
        "blurb": "**Daily client report** — implied vs realized across the whole book, "
                 "with the spread shown as a chart (scatter + ranked bars).",
    },
    "Skew Volatility": {
        "cli": SKEWREPORT_CLI, "detail": SKEW_DETAIL_FILE, "key": "skew_pdf",
        "file": "Skew_Volatility_Report.pdf",
        "label": "📈 Generate Skew Volatility Report (visual PDF)",
        "blurb": "**Daily client report** — skew (90% put − 110% call)/ATM across the book, "
                 "shown as a put-vs-call chart + ranked bars.",
    },
    "Vol Term Structure": {
        "cli": TERMREPORT_CLI, "detail": TERM_DETAIL_FILE, "key": "term_pdf",
        "file": "Vol_Term_Structure_Report.pdf",
        "label": "📈 Generate Vol Term Structure Report (visual PDF)",
        "blurb": "**Daily client report** — the ATM implied curve (1M/3M/6M/12M) across the book: "
                 "1M-vs-3M scatter, slope-z bars, curve shapes, and VRP by tenor.",
    },
    "Flag Breakout": {
        "cli": FLAGREPORT_CLI, "detail": FLAG_DETAIL_FILE, "key": "flag_pdf",
        "file": "Flag_Breakout_Report.pdf",
        "label": "📈 Generate Flag Breakout Report (visual PDF)",
        "blurb": "**Daily client report** — every flag in the book drawn with its pole, "
                 "consolidation channel and breakout line, ranked by readiness (with volume "
                 "confirmation where available).",
    },
}

# The curated Fixed Income open-interest book (the 🏛️ button on the Open Interest page):
# ONE PRODUCT PER PAGE (full strike chain), walked in tenor order — STIRs first, then US vs
# German at 2 / 5 / 10 / 30 years (the two of each tenor land on consecutive pages). Each
# item = (ticker, mock strike step, mock OI half-width) in PRICE units — the per-tenor grid
# that makes each rate heatmap realistic (step/width are mock hints, ignored once live
# Bloomberg supplies the real chain).
FI_OI_PAGES = [
    {"tenor": "3-Month Rates",
     "items": [("SFRA Comdty", 0.125, 0.6), ("SFIA Comdty", 0.125, 0.6), ("ERA Comdty", 0.125, 0.6)]},
    {"tenor": "2-Year", "items": [("TUA Comdty", 0.25, 1.2), ("DUA Comdty", 0.25, 1.0)]},
    {"tenor": "5-Year", "items": [("FVA Comdty", 0.5, 2.0), ("OEA Comdty", 0.5, 1.8)]},
    {"tenor": "10-Year", "items": [("TYA Comdty", 0.5, 2.5), ("RXA Comdty", 0.5, 2.5)]},
    {"tenor": "Long Bond", "items": [("USA Comdty", 1.0, 4.0), ("UBA Comdty", 1.0, 4.5)]},
]

st.set_page_config(
    page_title="BASIS — Strategy Monitor",
    page_icon=str(brand.ICON_PNG),
    layout="wide",
    menu_items={"about": "BASIS — Analysis · Strategy · Indicators"},
)

# BASIS brand theme: palettes, the dark/light CSS and the primary-button label
# fix (gold tiles need a dark label) all live in src/brand.py. apply() injects
# the CSS for the active theme; the sun/moon toggle in the masthead flips it.
brand.apply()


@st.cache_data
def load_signals():
    df = pd.read_parquet(SIGNALS_FILE) if SIGNALS_FILE.exists() else run_daily.run()
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    return df, meta


@st.cache_data(show_spinner=False)
def _pdf_page_images(pdf_bytes: bytes, scale: float = 2.0):
    """Rasterise a report PDF (bytes) into page images, for an inline preview on the page."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        return [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
    finally:
        doc.close()


@st.cache_data(show_spinner=False)
def _report_recipients(report_key: str):
    """Who an 'email this report' button will send to (managed list, else the desk list)."""
    try:
        import cot_scheduled_email as _mail
        _, _, fallback = _mail.load_email_cfg()
        return _mail._managed_recipients(report_key, fallback)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _all_contacts():
    """Every known recipient (the desk list + every managed per-report list), for the
    email pickers' dropdown options."""
    contacts = set()
    try:
        import cot_scheduled_email as _mail
        _, _, desk = _mail.load_email_cfg()
        contacts.update(e for e in (desk or []) if e)
    except Exception:
        pass
    try:
        p = ROOT / "data" / "email_recipients.json"
        if p.exists():
            for v in json.loads(p.read_text(encoding="utf-8")).values():
                contacts.update(e for e in (v or []) if e and str(e).strip())
    except Exception:
        pass
    return sorted(contacts)


def _recipient_picker(state_key: str, recipients_key: str):
    """Dropdown of known contacts, pre-filled with this report's default recipients. Returns
    the picked addresses; the user can also type a new address and press Enter to add it."""
    default = [r for r in _report_recipients(recipients_key) if r]
    options = sorted(set(_all_contacts()) | set(default))
    sel = st.multiselect(
        "Send to", options, default=default, key=f"{state_key}_to",
        accept_new_options=True,
        help="Pick one or more recipients, or type a new address and press Enter.")
    if sel:
        st.caption("Sending to: " + "  ·  ".join(sel))   # full list — the chips can clip the first char at narrow widths
    return sel


def email_report_ui(state_key, recipients_key, pdf_bytes, subject, attachment_name, intro_html=None):
    """Shared 'Email this report' block — a recipient dropdown + confirm + send button. Emails
    the given report PDF bytes to the picked recipients via the generic desk sender."""
    if not pdf_bytes:
        return
    st.markdown("**Email this report**")
    to_list = _recipient_picker(state_key, recipients_key)
    c1, c2 = st.columns([1, 4])
    confirm = c1.checkbox("Confirm", key=f"{state_key}_confirm")
    if c2.button("📤 Email report now", disabled=not (confirm and to_list), key=f"{state_key}_send"):
        with st.spinner("Sending…"):
            try:
                import cot_scheduled_email as _mail
                with tempfile.TemporaryDirectory() as _t:
                    _p = Path(_t) / attachment_name
                    _p.write_bytes(pdf_bytes)
                    sent = _mail.send_report_email(
                        _p, subject=subject,
                        intro_html=intro_html or f"<p>Please find the attached {attachment_name}.</p>",
                        attachment_name=attachment_name, report_key=recipients_key, to_override=to_list)
                st.success("Emailed to " + ", ".join(sent) + ".")
            except Exception as _e:
                st.error(f"Email failed:\n\n{_e}")


def _vol_sd_label(vol, sd, dec) -> str:
    """'28.7 (±7.98)' — a vol with its 1-day 1σ price move in brackets; falls back to
    just the vol when the SD isn't stored (older cross-section without the column)."""
    try:
        if pd.notna(sd):
            return f"{float(vol):.1f} (±{float(sd):.{int(dec)}f})"
    except Exception:
        pass
    return f"{float(vol):.1f}"


def _vol_charts(threshold):
    """On-page implied-vs-realized charts (scatter + ranked z-bars), read straight from the
    cached volatility cross-section so they show on the page WITHOUT generating the PDF."""
    import altair as alt
    from src import volcorr
    if not VOL_DETAIL_FILE.exists():
        st.info("No volatility cross-section yet — compute signals first (sidebar → **Re-run signals**).")
        return
    d = _filter_signals(pd.read_parquet(VOL_DETAIL_FILE).dropna(subset=["iv", "rv", "z"]).copy())
    if d.empty:
        st.caption("No markets match the current Home-page sector filter.")
        return
    if "iv_sd" not in d.columns:                         # older cross-section — no SD stored
        d["iv_sd"] = np.nan; d["rv_sd"] = np.nan; d["px_dec"] = 1
    d["iv_lbl"] = [_vol_sd_label(v, s, dc) for v, s, dc in zip(d["iv"], d["iv_sd"], d["px_dec"])]
    d["rv_lbl"] = [_vol_sd_label(v, s, dc) for v, s, dc in zip(d["rv"], d["rv_sd"], d["px_dec"])]
    _srclbl = {"own": "our curve", "bbg": "vendor surface", "otc": "OTC pair vol"}
    d["src_lbl"] = (d["src"].map(_srclbl).fillna("vendor surface")
                    if "src" in d.columns else "vendor surface")
    if "rv5" not in d.columns:                           # older cross-section — no 5d stored
        d["rv5"] = np.nan; d["rv_state"] = ""
    _mk5 = {"decay": " ▾ event fading", "heat": " ▴ accelerating"}
    d["rv5_lbl"] = [(f"{v:.1f}{_mk5.get(s, '')}" if pd.notna(v) else "—")
                    for v, s in zip(d["rv5"], d["rv_state"].fillna(""))]
    thr = float(threshold) if threshold is not None else 1.5
    # Directional flags require the z AND the spread's sign to agree (a stretched z on
    # a negative spread = the discount narrowing, not rich vol) — sign-mismatched
    # extremes get the house-AMBER watch tier: at the trigger, sign unconfirmed.
    d["flag"] = np.where((d["z"] >= thr) & (d["spread"] > 0), "Rich — sell vol",
                np.where((d["z"] >= thr), "Discount narrowing",
                np.where((d["z"] <= -thr) & (d["spread"] < 0), "Cheap — buy vol",
                np.where((d["z"] <= -thr), "Premium compressing", "Neutral"))))

    # ---- Relative-value peer finder: return / IV-change correlation (matches Product Correlations page) ----
    _metric_lbl = st.radio("Relative-value peers — correlate on", ["Returns", "IV changes", "Realized vol"],
                           index=2, horizontal=True, key="vol_rv_metric",
                           help="The same measures as the Product Correlations page. Returns = how prices move "
                                "together (direction); IV changes = how 1M implied vols move together (what you "
                                "trade); Realized vol = how delivered vols move together (whether they turn "
                                "volatile in sync).")
    _metric = {"IV changes": "iv", "Realized vol": "realized"}.get(_metric_lbl, "returns")
    _moves = {"iv": "implied vol", "realized": "realized vol"}.get(_metric, "price")
    _corrname = {"iv": "IV-change", "realized": "realized-vol"}.get(_metric, "return")
    cm = volcorr.load(_metric)
    corr_thr = (st.slider(f"Show {_corrname}-correlated peers ≥ (%)", 50, 99, 80, key="vol_rv_thr",
                          help=f"Products whose {_moves} moves with each one at least this closely — 1-yr "
                               f"{_corrname} correlation, matching the Product Correlations page. Hover any point "
                               "or use the focus panel below to see your sell/buy-against candidates.") / 100.0
                ) if cm is not None else 0.80
    _sig = {1: "▼ cheap", -1: "▲ rich", 0: "—"}
    _mk = {r.ticker: r.market for r in d.itertuples(index=False)}
    _dir = {r.ticker: (1 if str(r.flag).startswith("Cheap") else -1 if str(r.flag).startswith("Rich") else 0)
            for r in d.itertuples(index=False)}
    _uni = list(_mk.keys())

    def _peerstr(tk):
        if cm is None:
            return "—"
        ps = volcorr.peers(tk, cm, corr_thr, universe=_uni)
        if not ps:
            return f"none ≥{int(corr_thr*100)}%"
        parts = [f"{_mk[p]} {c*100:.0f}% {_sig[_dir.get(p, 0)]}" for p, c in ps[:8]]
        return " · ".join(parts) + (f"  +{len(ps)-8} more" if len(ps) > 8 else "")
    d["peers"] = [_peerstr(tk) for tk in d["ticker"]]

    cc = brand.chart_colors()
    dom = ["Rich — sell vol", "Cheap — buy vol", "Discount narrowing", "Premium compressing", "Neutral"]
    rng = [cc["short"], cc["long"], cc["accent"], cc["accent"], cc["muted"]]
    color = alt.Color("flag:N", scale=alt.Scale(domain=dom, range=rng),
                      legend=alt.Legend(title=None, orient="top"))
    tip = [alt.Tooltip("market:N", title="Market"),
           alt.Tooltip("iv_lbl:N", title="Implied (1M)"),
           alt.Tooltip("rv_lbl:N", title="Realized (1M)"),
           alt.Tooltip("rv5_lbl:N", title="Realized 5d"),
           alt.Tooltip("spread:Q", title="Spread", format="+.1f"),
           alt.Tooltip("z:Q", title="z (1y)", format="+.2f"),
           alt.Tooltip("src_lbl:N", title="Implied source"),
           alt.Tooltip("peers:N", title=f"RV peers ≥{int(corr_thr*100)}%")]

    hi = float(max(d["iv"].max(), d["rv"].max())) * 1.05
    fair = alt.Chart(pd.DataFrame({"v": [0, hi]})).mark_line(
        strokeDash=[4, 3], color=cc["muted"], opacity=0.7).encode(x="v:Q", y="v:Q")
    pts = alt.Chart(d).mark_circle(stroke="white", strokeWidth=0.5).encode(
        x=alt.X("rv:Q", title="Realized vol (1M, ann. %)", scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("iv:Q", title="Implied vol (1M ATM, %)", scale=alt.Scale(domain=[0, hi])),
        color=color,                                                  # whole book; flagged stand out
        size=alt.condition("datum.flag == 'Neutral'", alt.value(34), alt.value(95)),
        opacity=alt.condition("datum.flag == 'Neutral'", alt.value(0.4), alt.value(0.95)),
        tooltip=tip)
    st.markdown("**Implied vs realized — the spread at a glance** &nbsp;·&nbsp; above the dashed line = "
                "options rich (sell vol); below = cheap (buy vol).")
    brand.show_chart((fair + pts).properties(height=400))

    allp = d.sort_values("z", ascending=False)
    order = allp["market"].tolist()
    bars = alt.Chart(allp).mark_bar().encode(
        x=alt.X("z:Q", title="implied − realized spread · z-score vs 1-yr"),
        y=alt.Y("market:N", title=None, sort=order),
        color=color, tooltip=tip)
    rule = alt.Chart(pd.DataFrame({"z": [thr, -thr]})).mark_rule(
        color=cc["muted"], strokeDash=[3, 3]).encode(x="z:Q")
    st.markdown(f"**All markets ranked by spread z-score** &nbsp;·&nbsp; flagged in colour; dashed lines = your trigger (±{thr:g}).")
    brand.show_chart((bars + rule).properties(height=max(260, 15 * len(allp))))
    st.caption("Implied vols are **our own constant-30-day curve** — ATM call/put settlements of every "
               "listed option expiry, inverted through Black-76 and curve-fitted per product. FX uses OTC "
               "pair vols and equity cash indices the vendor surface, which also backstops any market our "
               "build misses on the day (hover a point for each market's source). **Amber** = the z-trigger "
               "is hit but implied hasn't crossed realized — a compressing premium or narrowing discount "
               "to watch, not a signal. Hover also shows **5d realized**: ▾ = under half the 1-month for "
               "two straight sessions (an event fading out of the window — treat cheap readings with "
               "care), ▴ = more than double it (realizing accelerating past the 1-month).")

    # ---- Short-term rates: own section, rate-vol convention (1σ moves in bp) ----
    stir_path = VOL_DETAIL_FILE.parent / "stirvol.parquet"
    if stir_path.exists():
        sd = _filter_signals(pd.read_parquet(stir_path))
        if not sd.empty:
            st.markdown("#### Short-term rates — rate vol")
            sd = sd.copy()
            # Directional flag only when the z and the spread's SIGN agree (a high z on a
            # still-negative spread = the discount narrowing, not rich vol).
            sd["flag"] = np.where((sd["z"] >= thr) & (sd["spread"] > 0), "Rich — sell vol",
                         np.where((sd["z"] >= thr), "Discount narrowing",
                         np.where((sd["z"] <= -thr) & (sd["spread"] < 0), "Cheap — buy vol",
                         np.where((sd["z"] <= -thr), "Premium compressing", "—"))))
            tbl_s = sd.assign(
                Implied=[f"±{b:.1f}bp  ({v:.1f} vol)" for v, b in zip(sd["iv"], sd["iv_bp"])],
                Realized=[f"±{b:.1f}bp  ({v:.1f} vol)" for v, b in zip(sd["rv"], sd["rv_bp"])],
                Spread=sd["spread"].map("{:+.1f}".format), Z=sd["z"].map("{:+.2f}".format),
            )[["market", "ticker", "rate", "Implied", "Realized", "Spread", "Z", "pctl", "flag"]].rename(
                columns={"market": "Market", "ticker": "Instrument", "rate": "Rate %",
                         "Implied": "Implied ±1σ (bp/day)", "Realized": "Realized ±1σ (bp/day)",
                         "Z": "z (1y)", "pctl": "%ile", "flag": "Signal"})
            st.dataframe(tbl_s, use_container_width=True, hide_index=True)
            st.caption("**The headline figure is the 1-day 1σ move in basis points — the number the desk "
                       "actually trades.** Implied is **our own constant 90-day ATM**, built identically for "
                       "all three markets: listed quarterly option settlements inverted through Black-76 on "
                       "the implied rate (100 − price), ATM call/put mids interpolated in total variance. "
                       "Realized is the matching 63-day delivered vol of the implied rate; implied naturally "
                       "sits above it (options price the meetings ahead) — judge each market's z against its "
                       "own year. Bloomberg's published 3M point is carried as a cross-check where it exists "
                       "(none for Euribor). 1M SOFR / Fed Funds options are too thin for a reliable mark; "
                       "€STR has no listed options.")

    # ---- Relative-value: pick a product, see its vol-correlated peers + their signal ----
    if cm is not None:
        st.markdown("#### Relative-value — what to trade against it")
        pick = st.selectbox("Focus product — show its vol-correlated peers",
                            d.sort_values("market")["market"].tolist(), key="vol_rv_pick")
        prow = d[d["market"] == pick].iloc[0]
        st.markdown(f"**{pick}** — {prow['flag']} &nbsp;·&nbsp; implied {prow['iv']:.1f} / realized "
                    f"{prow['rv']:.1f} &nbsp;·&nbsp; z {prow['z']:+.2f}")
        _ps = volcorr.peers(prow["ticker"], cm, corr_thr, universe=_uni)
        if _ps:
            _cmap = dict(_ps)
            pk = d[d["ticker"].isin(_cmap)].copy()
            pk["_c"] = pk["ticker"].map(_cmap)
            pk = pk.sort_values("_c", ascending=False)
            ptbl = pk.assign(Corr=(pk["_c"] * 100).map("{:.0f}%".format),
                             Spread=pk["spread"].map("{:+.1f}".format),
                             Z=pk["z"].map("{:+.2f}".format))[
                ["market", "Corr", "iv_lbl", "rv_lbl", "Spread", "Z", "flag"]].rename(columns={
                "market": "Peer", "iv_lbl": "Implied (±1σ)", "rv_lbl": "Realized (±1σ)",
                "Z": "z (1y)", "flag": "Signal"})
            st.dataframe(ptbl, use_container_width=True, hide_index=True)
            st.caption(f"Peers whose {_moves} moves with **{pick}** ≥ {int(corr_thr*100)}% (1-yr {_corrname} "
                       "correlation, matching the Product Correlations page). If it screens cheap, sell vol on a "
                       "**rich** peer against it — and vice-versa.")
        else:
            st.caption(f"Nothing correlates ≥ {int(corr_thr*100)}% with {pick} — lower the threshold above to surface more.")

    # ---- Flagged opportunities: dumbbell + table + per-market 1-year history ----
    fl = d[d["flag"] != "Neutral"].reindex(d.loc[d["flag"] != "Neutral", "z"].abs()
                                           .sort_values(ascending=False).index)
    st.markdown(f"#### Flagged — {len(fl)} opportunit{'y' if len(fl) == 1 else 'ies'} at |z| ≥ {thr:g}")
    if fl.empty:
        st.caption("Nothing flagged at the current trigger — lower it to surface more.")
        return

    fls = fl.sort_values("spread")                      # cheap (neg) at bottom → rich (pos) on top
    yorder = fls["market"].tolist()
    yenc = alt.Y("market:N", sort=yorder, title=None)
    bz = alt.Chart(fls)
    seg = bz.mark_rule(color=cc["muted"], strokeWidth=2.6).encode(x=alt.X("rv:Q", title="annualised vol (%)"), x2="iv:Q", y=yenc)
    rv_pt = bz.mark_point(filled=True, size=95, color=cc["muted"], stroke="white", strokeWidth=0.6).encode(x="rv:Q", y=yenc, tooltip=tip)
    iv_pt = bz.mark_point(filled=True, size=95, stroke="white", strokeWidth=0.6).encode(x="iv:Q", y=yenc, color=color, tooltip=tip)
    st.markdown("Grey dot = realized · coloured dot = implied · the bar between them is the spread.")
    brand.show_chart((seg + rv_pt + iv_pt).properties(height=max(170, 34 * len(fls))))

    from src import seasonal
    _m = pd.to_datetime(str(meta.get("as_of", ""))[:10], errors="coerce")
    _mo = _m.month if pd.notna(_m) else datetime.now().month
    fl = fl.assign(Weather=fl["ticker"].map(lambda t: seasonal.weather_note(t, _mo)))
    if fl["Weather"].astype(str).str.strip().ne("").any():
        st.caption("☼ = an agricultural market in its seasonal weather window — a rich-vol reading there partly "
                   "prices genuine crop-weather risk, not pure mispricing, so it's premium for real risk.")
    tbl = fl.assign(Spread=fl["spread"].map("{:+.1f}".format), Z=fl["z"].map("{:+.2f}".format),
                    Wx=fl["Weather"].map(lambda s: f"☼ {s}" if s else ""))[
        ["market", "asset", "ticker", "iv_lbl", "rv_lbl", "Spread", "Z", "pctl", "signal", "Wx"]].rename(columns={
        "market": "Market", "asset": "Asset", "ticker": "Instrument", "iv_lbl": "Implied (±1σ)",
        "rv_lbl": "Realized (±1σ)", "Z": "z (1y)", "pctl": "%ile", "signal": "Signal", "Wx": "Seasonal"})
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption("Bracketed figure = the corresponding 1-day 1σ move in the contract's own price "
               "units (vol ÷ √252 × price), to the decimals it trades in.")

    hist_path = VOL_DETAIL_FILE.parent / "volatility_history.parquet"
    if hist_path.exists():
        h = pd.read_parquet(hist_path)
        have = [m for m in fl["market"].tolist()
                if fl.loc[fl["market"] == m, "ticker"].iloc[0] in set(h.get("ticker", []))]
        if have:
            st.markdown("**1-year history — implied − realized spread vs the underlying**")
            pick = st.selectbox("Flagged market", have, key="vol_hist_pick")
            tk = fl.loc[fl["market"] == pick, "ticker"].iloc[0]
            g = h[h["ticker"] == tk].sort_values("date").copy()
            g["date"] = pd.to_datetime(g["date"])
            scol = cc["short"] if fl.loc[fl["market"] == pick, "flag"].iloc[0].startswith("Rich") else cc["long"]
            sp_area = alt.Chart(g).mark_area(opacity=0.22, color=scol).encode(
                x=alt.X("date:T", title=None), y=alt.Y("spread:Q", title="implied − realized (vol pts)"))
            sp_line = alt.Chart(g).mark_line(color=scol, strokeWidth=2.1).encode(x="date:T", y="spread:Q")
            px_line = alt.Chart(g).mark_line(color=cc["ink"], strokeWidth=1.9).encode(
                x="date:T", y=alt.Y("price:Q", title="underlying price", scale=alt.Scale(zero=False)))
            brand.show_chart(alt.layer(sp_area + sp_line, px_line).resolve_scale(y="independent").properties(height=320))


def _diverging_bars(allp, color, thr, x_title, rule_lines=True):
    """Whole-book ranked z-bars (flagged in colour, rest greyed)."""
    import altair as alt
    order = allp.sort_values("z", ascending=False)["market"].tolist()
    bars = alt.Chart(allp).mark_bar().encode(
        x=alt.X("z:Q", title=x_title), y=alt.Y("market:N", title=None, sort=order), color=color,
        tooltip=[alt.Tooltip("market:N", title="Market"), alt.Tooltip("z:Q", title="z (1y)", format="+.2f")])
    layer = bars
    if rule_lines:
        layer = bars + alt.Chart(pd.DataFrame({"z": [thr, -thr]})).mark_rule(
            color=brand.chart_colors()["muted"], strokeDash=[3, 3]).encode(x="z:Q")
    brand.show_chart(layer.properties(height=max(260, 15 * len(allp))))


def _skew_charts(threshold):
    """On-page skew charts: put-vs-call scatter + whole-book ranked bars + flagged section."""
    import altair as alt
    if not SKEW_DETAIL_FILE.exists():
        st.info("No skew cross-section yet — compute signals first (sidebar → **Re-run signals**).")
        return
    d = _filter_signals(pd.read_parquet(SKEW_DETAIL_FILE).dropna(subset=["z"]).copy())
    if d.empty:
        st.caption("No markets match the current Home-page sector filter.")
        return
    st.caption("⚠️ This page still runs on the **vendor surface's** 90/110% moneyness wings. Since "
               "28 Jul 2026 we also record **our own settlement-built skew** daily (same Black-76 "
               "machinery as the Volatility/Term pages — OTM put at 0.90×F, OTM call at 1.10×F, "
               "interpolated to constant 30d; `data/snapshot/own_skew_history.parquet`). **The own "
               "history needs backfilling at some point** before this page can switch source — until "
               "then it accrues one settle per day as a validation trail.")
    thr = float(threshold) if threshold is not None else 1.5
    d["flag"] = np.where(d["z"] >= thr, "Rich — sell skew",
                np.where(d["z"] <= -thr, "Cheap — buy skew", "Neutral"))
    cc = brand.chart_colors()
    color = alt.Color("flag:N", scale=alt.Scale(domain=["Rich — sell skew", "Cheap — buy skew", "Neutral"],
                      range=[cc["short"], cc["long"], cc["muted"]]), legend=alt.Legend(title=None, orient="top"))
    tip = [alt.Tooltip("market:N", title="Market"), alt.Tooltip("put:Q", title="Put wing", format=".1f"),
           alt.Tooltip("call:Q", title="Call wing", format=".1f"), alt.Tooltip("skew:Q", title="Skew", format="+.3f"),
           alt.Tooltip("z:Q", title="z (1y)", format="+.2f")]
    dsc = d.dropna(subset=["put", "call"])
    if not dsc.empty:
        hi = float(max(dsc["put"].max(), dsc["call"].max())) * 1.05
        fair = alt.Chart(pd.DataFrame({"v": [0, hi]})).mark_line(strokeDash=[4, 3], color=cc["muted"], opacity=0.7).encode(x="v:Q", y="v:Q")
        pts = alt.Chart(dsc).mark_circle(stroke="white", strokeWidth=0.5).encode(
            x=alt.X("call:Q", title="Call-wing vol (%)", scale=alt.Scale(domain=[0, hi])),
            y=alt.Y("put:Q", title="Put-wing vol (%)", scale=alt.Scale(domain=[0, hi])), color=color,
            size=alt.condition("datum.flag == 'Neutral'", alt.value(34), alt.value(95)),
            opacity=alt.condition("datum.flag == 'Neutral'", alt.value(0.4), alt.value(0.95)), tooltip=tip)
        st.markdown("**Put vs call wing — the skew at a glance** &nbsp;·&nbsp; above the line = puts bid over calls (downside skew).")
        brand.show_chart((fair + pts).properties(height=400))
    st.markdown(f"**All markets ranked by skew z-score** &nbsp;·&nbsp; flagged in colour; dashed = trigger (±{thr:g}).")
    _diverging_bars(d, color, thr, "(put − call)/ATM skew · z-score vs 1-yr")

    fl = d[d["flag"] != "Neutral"].reindex(d.loc[d["flag"] != "Neutral", "z"].abs().sort_values(ascending=False).index)
    st.markdown(f"#### Flagged — {len(fl)} opportunit{'y' if len(fl) == 1 else 'ies'} at |z| ≥ {thr:g}")
    if fl.empty:
        st.caption("Nothing flagged at the current trigger — lower it to surface more.")
        return
    fld = fl.dropna(subset=["put", "call"]).sort_values("skew")
    if not fld.empty:
        yenc = alt.Y("market:N", sort=fld["market"].tolist(), title=None)
        bz = alt.Chart(fld)
        seg = bz.mark_rule(color=cc["muted"], strokeWidth=2.6).encode(x=alt.X("call:Q", title="wing vol (%)"), x2="put:Q", y=yenc)
        cpt = bz.mark_point(filled=True, size=95, color=cc["muted"], stroke="white", strokeWidth=0.6).encode(x="call:Q", y=yenc, tooltip=tip)
        ppt = bz.mark_point(filled=True, size=95, stroke="white", strokeWidth=0.6).encode(x="put:Q", y=yenc, color=color, tooltip=tip)
        st.markdown("Grey dot = call wing · coloured dot = put wing · the bar between them is the skew.")
        brand.show_chart((seg + cpt + ppt).properties(height=max(170, 34 * len(fld))))
    tbl = fl.assign(Skew=fl["skew"].map("{:+.3f}".format), Z=fl["z"].map("{:+.2f}".format))[
        ["market", "asset", "ticker", "put", "call", "atm", "Skew", "Z", "pctl", "signal"]].rename(columns={
        "market": "Market", "asset": "Asset", "ticker": "Instrument", "put": "Put", "call": "Call",
        "atm": "ATM", "Z": "z (1y)", "pctl": "%ile", "signal": "Signal"})
    st.dataframe(tbl, use_container_width=True, hide_index=True)


def _term_charts(threshold):
    """On-page term-structure charts: 1M-vs-3M scatter + whole-book ranked bars + flagged curves/table."""
    import altair as alt
    if not TERM_DETAIL_FILE.exists():
        st.info("No term-structure cross-section yet — compute signals first (sidebar → **Re-run signals**).")
        return
    d = _filter_signals(pd.read_parquet(TERM_DETAIL_FILE).dropna(subset=["iv_1m", "iv_3m", "z"]).copy())
    if d.empty:
        st.caption("No markets match the current Home-page sector filter.")
        return
    _tn = ["1m", "3m", "6m", "12m"]
    if "iv_sd_1m" not in d.columns:                      # older cross-section — no SD stored
        for _l in _tn:
            d[f"iv_sd_{_l}"] = np.nan
        d["px_dec"] = 1
    for _l in _tn:
        d[f"iv_{_l}_lbl"] = [_vol_sd_label(v, s, dc)
                             for v, s, dc in zip(d[f"iv_{_l}"], d[f"iv_sd_{_l}"], d["px_dec"])]
    thr = float(threshold) if threshold is not None else 1.5
    d["flag"] = np.where(d["z"] >= thr, "Steep — front cheap",
                np.where(d["z"] <= -thr, "Inverted — front rich", "Neutral"))
    cc = brand.chart_colors()
    color = alt.Color("flag:N", scale=alt.Scale(domain=["Steep — front cheap", "Inverted — front rich", "Neutral"],
                      range=[cc["long"], cc["short"], cc["muted"]]), legend=alt.Legend(title=None, orient="top"))
    _srclbl = {"own": "our curve", "bbg": "vendor surface", "otc": "OTC tenor vols"}
    d["src_lbl"] = (d["src"].map(_srclbl).fillna("vendor surface")
                    if "src" in d.columns else "vendor surface")
    tip = [alt.Tooltip("market:N", title="Market"), alt.Tooltip("iv_1m_lbl:N", title="1M"),
           alt.Tooltip("iv_3m_lbl:N", title="3M"), alt.Tooltip("slope:Q", title="3M−1M", format="+.1f"),
           alt.Tooltip("z:Q", title="z (1y)", format="+.2f"),
           alt.Tooltip("src_lbl:N", title="Tenor source")]
    hi = float(max(d["iv_1m"].max(), d["iv_3m"].max())) * 1.05
    fair = alt.Chart(pd.DataFrame({"v": [0, hi]})).mark_line(strokeDash=[4, 3], color=cc["muted"], opacity=0.7).encode(x="v:Q", y="v:Q")
    pts = alt.Chart(d).mark_circle(stroke="white", strokeWidth=0.5).encode(
        x=alt.X("iv_1m:Q", title="1M ATM vol (%)", scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("iv_3m:Q", title="3M ATM vol (%)", scale=alt.Scale(domain=[0, hi])), color=color,
        size=alt.condition("datum.flag == 'Neutral'", alt.value(34), alt.value(95)),
        opacity=alt.condition("datum.flag == 'Neutral'", alt.value(0.4), alt.value(0.95)), tooltip=tip)
    st.markdown("**1M vs 3M — the curve at a glance** &nbsp;·&nbsp; above the line = contango (3M > 1M); below = backwardation.")
    brand.show_chart((fair + pts).properties(height=400))
    st.markdown(f"**All markets ranked by 3M−1M slope z-score** &nbsp;·&nbsp; flagged in colour; dashed = trigger (±{thr:g}).")
    _diverging_bars(d, color, thr, "3M − 1M slope · z-score vs 1-yr")
    st.caption("Tenor vols are **our own curve** (the same settlement-built fit as the Volatility page, "
               "read at 1M/3M/6M/12M and emitted only where the listed expiries genuinely reach the "
               "tenor) — the vendor surface backstops longer tenors, FX (OTC tenor vols), cash indices "
               "and any market our build misses on the day (hover a point for the source). The slope "
               "z-score's one-year history is still largely the vendor's while our own term history "
               "accrues (started 27 Jul 2026).")

    fl = d[d["flag"] != "Neutral"].reindex(d.loc[d["flag"] != "Neutral", "z"].abs().sort_values(ascending=False).index)
    st.markdown(f"#### Flagged — {len(fl)} opportunit{'y' if len(fl) == 1 else 'ies'} at |z| ≥ {thr:g}")
    if fl.empty:
        st.caption("Nothing flagged at the current trigger — lower it to surface more.")
        return
    _tmap = {"iv_1m": "1M", "iv_3m": "3M", "iv_6m": "6M", "iv_12m": "12M"}
    cur = fl.melt(id_vars=["market", "flag"], value_vars=list(_tmap), var_name="tenor", value_name="iv")
    cur["tenor"] = cur["tenor"].map(_tmap)
    curve = alt.Chart(cur).mark_line(point=True, strokeWidth=2.2).encode(
        x=alt.X("tenor:N", sort=["1M", "3M", "6M", "12M"], title="tenor"),
        y=alt.Y("iv:Q", title="ATM vol (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("market:N", legend=alt.Legend(title=None, orient="right")),
        tooltip=["market:N", "tenor:N", alt.Tooltip("iv:Q", format=".1f")])
    st.markdown("**Flagged curves — 1M → 12M ATM term structure**")
    brand.show_chart(curve.properties(height=340))
    tbl = fl.assign(Slope=fl["slope"].map("{:+.1f}".format), Z=fl["z"].map("{:+.2f}".format))[
        ["market", "asset", "ticker", "iv_1m_lbl", "iv_3m_lbl", "iv_6m_lbl", "iv_12m_lbl", "Slope", "Z", "pctl", "signal"]].rename(columns={
        "market": "Market", "asset": "Asset", "ticker": "Instrument",
        "iv_1m_lbl": "1M (±1σ)", "iv_3m_lbl": "3M (±1σ)", "iv_6m_lbl": "6M (±1σ)", "iv_12m_lbl": "12M (±1σ)",
        "Z": "z (1y)", "pctl": "%ile", "signal": "Signal"})
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption("Bracketed figure = the corresponding 1-day 1σ move in the contract's own price "
               "units (vol ÷ √252 × price), to the decimals it trades in.")


# ===========================================================================
#  Navigation + the non-strategy pages (Home / Morning Coffee / Universe).
#  One Streamlit script; st.session_state.active selects the view. The sidebar
#  is pure nav; the dispatch near the bottom renders the active page.
# ===========================================================================
def _load_snap():
    return json.loads(SNAPSHOT_MANIFEST.read_text()) if SNAPSHOT_MANIFEST.exists() else None


def _go(dest: str) -> None:
    """on_click nav handler — runs before the rerun, so the highlight stays in sync."""
    st.session_state.active = dest
    st.session_state["_scroll_top_ts"] = time.time()    # new page opens at the top


def _set_side(s: str) -> None:
    """Switch between the FICC (futures) and Equities sides; land on that side's home page."""
    st.session_state.side = s
    st.session_state.active = "eq:Home" if s == "Equities" else "Home"
    st.session_state["_scroll_top_ts"] = time.time()


def _nav_button(label: str, dest: str) -> None:
    st.button(label, use_container_width=True, key=f"nav_{dest}",
              type="primary" if st.session_state.get("active") == dest else "secondary",
              on_click=_go, args=(dest,))


# Sidebar groups collapsed to a single entry — every member page shows this tab row at the top
# (rendered just before the page dispatch, so it works for generic strategy pages AND the special
# dispatched pages like the Reports Calendar and OPEC report). member = (button label, active key).
_GROUP_TABS = {
    "Market Information": [("📅 Reports Calendar", "Release Calendar"),
                           ("🕒 Market Hours", "Market Hours"),
                           ("📦 Block Sizes", "Block Sizes"),
                           ("🧮 Fut / Yield", "Fut Yield"),
                           ("🌐 Universe", "Universe")],
    "Trade Testing":      [("🏛️ Fed Path", "Fed Path"),
                           ("🧪 Vol Backtester", "Vol Backtester")],
    "Volatility":         [(s, s) for s in NAV_GROUPS["Volatility"]],
    "Positioning & Flow": [(s, s) for s in NAV_GROUPS["Positioning & Flow"]],
    "Fundamentals":       [("AG Fundamentals", "AG Fundamentals"),
                           ("🛢️ OPEC Report", "OPEC Report"),
                           ("🥇 Precious Metals", "Precious Metals")],
}
_TAB_MEMBERS_OF = {dest: members for members in _GROUP_TABS.values() for _lbl, dest in members}


def _render_group_tabs(active_page: str) -> None:
    """If `active_page` belongs to a collapsed sidebar group, render its tab-row switcher (the active
    tab highlighted). No-op for any page that isn't part of a collapsed group."""
    members = _TAB_MEMBERS_OF.get(active_page)
    if not members:
        return
    cols = st.columns(len(members))
    for col, (label, dest) in zip(cols, members):
        col.button(label, key=f"gtab_{dest}", use_container_width=True, on_click=_go, args=(dest,),
                   type="primary" if dest == active_page else "secondary")


def _data_badge(snap, side: str = "FICC") -> None:
    """Compact, always-visible data-source status for the sidebar. Healthy states render as
    a subtle caption (same voice as "Signals as of" below); only problem states — demo /
    missing data — keep the loud warning box. On the Equities desk the badge shows the
    EQUITIES pull stamp (manifest `equities_pulled`), not the FICC snapshot's."""
    if side == "Equities":
        eqp = (snap or {}).get("equities_pulled", "")
        if eqp:
            st.caption(f"📦 Equities: pulled **{_to_et(eqp)}**")
        else:
            st.caption("📦 Equities: no pull recorded yet — pages run on the last cached data")
        return
    if MODE == "bloomberg":
        st.caption("🟢 Live Bloomberg")
    elif MODE == "snapshot" and snap and snap.get("source") == "bloomberg":
        _pulled = f"<br>pulled {_to_et(snap['created'])}" if snap.get("created") else ""
        st.caption(f"📦 Snapshot: **{snap.get('as_of', '?')}**{_pulled}",
                   unsafe_allow_html=True)
    elif MODE == "snapshot" and snap:
        st.warning(f"Demo snapshot ({snap.get('source', '?')})", icon="⚠️")
    elif MODE == "snapshot":
        st.warning("No snapshot — demo data", icon="⚠️")
    else:
        st.warning("DEMO MODE — synthetic", icon="⚠️")


def _ficc_moves_frame() -> pd.DataFrame:
    """Overnight per-contract move frame — columns Market · Sector · pct · last · sigma, STIRs
    excluded, sorted by σ (biggest movers first). σ = the move ÷ the contract's own ~1-month daily
    move std. Shared by the Overnight-moves table and the on-screen FICC heatmap so both read from
    one source (the underlying Bloomberg calls are cached). Empty frame when there's no live quote."""
    live = get_live_quote(list(universe.enabled_tickers()))
    live = live.dropna(subset=["pct"]) if not live.empty else live
    if live.empty:
        return pd.DataFrame(columns=["Market", "Sector", "pct", "last", "sigma"])
    try:
        sd = get_history(list(universe.enabled_tickers())).pct_change().tail(21).std()
    except Exception:
        sd = None

    def _sigma(tk, pct):
        s = sd.get(tk) if sd is not None else None
        if s is None or s != s or s <= 1e-9:
            return float("nan")
        return (pct / 100.0) / s

    rows = [{"Market": INSTRUMENTS.get(tk, (tk, 0.0, "", ""))[0],
             "Sector": INSTRUMENTS.get(tk, (tk, 0.0, "", ""))[2],
             "pct": float(r["pct"]), "last": float(r["last"]),
             "sigma": _sigma(tk, float(r["pct"]))}
            for tk, r in live.iterrows()
            if INSTRUMENTS.get(tk, (tk, 0.0, "", ""))[2] not in {"STIRs"}]   # STIRs: price vol ≈ 0
    return pd.DataFrame(rows).sort_values("sigma", ascending=False)


def _all_filtered_off() -> bool:
    """True when the Sectors & products filter has switched EVERY instrument off — data may
    well exist, but nothing is enabled to show. Lets Home say 'the filter is off' instead of the
    misleading 'no quote available' when the movers table / heatmap come back empty."""
    try:
        return not universe.enabled_tickers()
    except Exception:
        return False


def _filtered_off_notice() -> None:
    st.warning("🗂️ **All sectors are switched off** in the Sectors & products filter above — "
               "click **All** there to bring the markets back. (Your data is fine; nothing is "
               "just enabled to display.)")


def _overnight_moves(snap) -> None:
    """Overnight net change (previous settle -> snapshot pull), expressed in σ."""
    st.subheader("Overnight moves")
    if _all_filtered_off():
        _filtered_off_notice()
        return
    _have_live = (SNAPSHOT_DIR / "live.parquet").exists()
    if MODE == "snapshot" and not _have_live:
        st.caption("No live overnight quote captured yet — click **Pull Bloomberg Snapshot** "
                   "(needs the Terminal). It records each contract's move from the previous "
                   "trading day's settle to the moment the snapshot is pulled.")
        return
    _mv = _ficc_moves_frame()
    if _mv.empty:
        st.caption("No overnight quote available.")
        return
    _asof = (snap or {}).get("live_as_of") or (snap or {}).get("created", "")
    st.caption("Move from the previous trading day's **settlement** to the snapshot pull"
               + (f" · prices as of **{_to_et(_asof)}**" if _asof else "")
               + ". Sorted by **σ (1m)** = the move in standard deviations of the contract's "
                 "own ~1-month daily moves. STIRs excluded (price vol ≈ 0 → σ is noise).")
    _rows = _mv.head(14).copy()
    _rows["last_fmt"] = _rows["last"].map(lambda v: f"{v:g}")
    brand.terminal_table(
        _rows.to_dict("records"),
        [{"key": "Market",   "label": "Market"},
         {"key": "Sector",   "label": "Sector"},
         {"key": "last_fmt", "label": "Last",  "align": "right"},
         {"key": "pct",      "label": "Chg %", "color": True, "fmt": "{:+.2f}"},
         {"key": "sigma",    "label": "σ 1M",  "color": True, "fmt": "{:+.1f}"},
         {"key": "sigma",    "label": "Z-range", "zbar": True}])


def _mc_heatmap_path() -> Path:
    return MORNING_COFFEE_DIR / "_heat_combined_en.png"


@st.cache_data(ttl=1800, show_spinner=False)
def _load_weather():
    return worldclock.fetch_weather()


@st.cache_data(ttl=21600, show_spinner=False)
def _load_city_photos(key):
    return worldclock.fetch_city_photos(key)


def _world_clocks() -> None:
    """World-clock rail: flat terminal cells — mono city label with a small weather glyph
    beside it and the current °C right-aligned, over a ticking HH:MM:SS. Weather is
    Open-Meteo via _load_weather (free, cached 30 min — zero Bloomberg hits); on any
    fetch failure the rail renders without icons/temps rather than blocking."""
    import streamlit.components.v1 as components
    pal = brand.palette()
    faint = pal.get("faint", pal["text_dim"])
    mono = "'IBM Plex Mono',Consolas,'SF Mono',Menlo,monospace"
    try:
        wx = _load_weather()
    except Exception:
        wx = []
    wx = (wx + [{"temp": None, "icon": ""}] * len(worldclock.CITIES))[:len(worldclock.CITIES)]
    cells = "".join(
        '<div class="c"><div class="city">'
        '<span class="wi">' + (w.get("icon") or "") + '</span>'
        '<span class="nm">' + c["name"].upper() + '</span>'
        + ('<span class="tmp">' + str(w["temp"]) + '&#176;</span>'
           if w.get("temp") is not None else "")
        + '</div><div class="time" data-tz="' + c["tz"] + '">--:--:--</div></div>'
        for c, w in zip(worldclock.CITIES, wx))
    html = (
        "<meta charset='utf-8'><style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{background:transparent;font-family:" + mono + "}"
        ".row{display:grid;grid-template-columns:repeat(6,minmax(150px,200px));"
        "width:fit-content;max-width:100%;"
        "background:" + pal["surface2"] + ";border:1px solid " + pal["border"] + "}"
        ".c{padding:7px 12px;min-width:0;border-right:1px solid " + pal["border"] + "}"
        ".c:last-child{border-right:none}"
        ".city{display:flex;align-items:center;gap:5px;font-size:12px;"
        "letter-spacing:.12em;color:" + faint + "}"
        ".city .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".wi{display:flex;flex:0 0 auto}.wi svg{width:17px;height:17px}"
        ".tmp{margin-left:auto;flex:0 0 auto;letter-spacing:0;font-size:11.5px;"
        "font-variant-numeric:tabular-nums;color:" + faint + "}"
        ".time{font-size:17px;font-weight:500;color:" + pal["text"] +
        ";font-variant-numeric:tabular-nums}"
        "</style><div class='row'>" + cells + "</div>"
        "<script>function t(){document.querySelectorAll('.time').forEach(function(e){"
        "e.textContent=new Intl.DateTimeFormat('en-GB',{timeZone:e.dataset.tz,"
        "hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date());});}"
        "t();setInterval(t,1000);"
        # keep the fixed top bar clear of the (drag-resizable) sidebar: mirror the
        # sidebar's live width into --basis-topbar-left on the parent document
        "function sb(){try{var d=window.parent.document;"
        "var s=d.querySelector('section[data-testid=\"stSidebar\"]');"
        # floor at 48px: a collapsed sidebar can stay mounted at ~0 width, and the
        # expand arrow needs that corner visible/clickable either way
        "var w=s?Math.round(s.getBoundingClientRect().width):0;"
        "if(w<48)w=48;"
        "d.documentElement.style.setProperty('--basis-topbar-left',w+'px');}catch(e){}}"
        "sb();setInterval(sb,500);</script>")
    components.html(html, height=68)




@st.cache_data(ttl=900, show_spinner=False)
def _load_econ_today():
    return econ.fetch_today()


def _econ_figures() -> None:
    """Today's major (high-impact) economic releases — the figures the Bloomberg ECO
    page tracks, from the free FairEconomy calendar (independent of Bloomberg)."""
    st.divider()
    st.subheader("Today's economic figures")
    try:
        rows = _load_econ_today()
    except Exception:
        rows = []
    if not rows:
        st.caption("No major (high-impact) releases scheduled today — or the calendar feed is "
                   "unavailable (weekends/holidays are normally empty). Times are US-Eastern.")
        return
    df = pd.DataFrame(rows)[["time", "country", "title", "actual", "forecast", "previous"]].rename(
        columns={"time": "Time (ET)", "country": "Ccy", "title": "Event",
                 "actual": "Actual", "forecast": "Forecast", "previous": "Prior"})

    def _bold_actual(col):
        return ["font-weight:700" if str(v).strip() else "color:#888" for v in col]

    brand.themed_dataframe(df, {}, colorers=[(["Actual"], _bold_actual)])
    st.caption(f"**{len(df)}** major release(s) today, US-Eastern · high-impact only, from the "
               "FairEconomy calendar (the figures the Bloomberg ECO page tracks). Actuals fill "
               "in as they print; refreshes ~every 15 min.")


def _home_heatmap() -> None:
    """On-screen FICC market heatmap — a native HTML treemap (tile area ∝ σ, green up / red down),
    grouped by sector, built live from the overnight-moves frame. Renders crisp via components.html
    rather than pasting the Morning Coffee report PNG."""
    from src import heatmap_html
    import streamlit.components.v1 as components
    st.divider()
    st.subheader("Market heatmap")
    if _all_filtered_off():
        _filtered_off_notice()
        return
    mvf = _ficc_moves_frame()
    mvf = mvf.dropna(subset=["sigma"]) if not mvf.empty else mvf
    if mvf.empty:
        st.caption("Appears once an overnight quote is available "
                   "(**Pull Bloomberg Snapshot**, or a Morning Coffee run).")
        return
    st.caption("**Tile size = how many σ the contract moved overnight** (vs its own ~1-month daily "
               "vol) — colour is direction (green up / red down), deepening with |σ|. Grouped by "
               "sector; hover a tile for the name, % and σ.")
    sections = []                                   # one band per sector; tiles = products (1-level)
    for sec in list(dict.fromkeys(mvf["Sector"])):
        ds = mvf[mvf["Sector"] == sec]
        items = [(r["Market"], float(r["pct"]) if r["pct"] == r["pct"] else 0.0,
                  float(r["sigma"]) if r["sigma"] == r["sigma"] else None)
                 for _, r in ds.iterrows()]
        if items:
            sections.append((sec, [(sec, items)]))
    height = int(min(700, max(320, 120 + 84 * len(sections))))
    components.html(heatmap_html.render_html(sections, height, sub_headers=False),
                    height=height + 6, scrolling=False)


def _md_money(s: str) -> str:
    """Escape '$' before handing prose to st.markdown. Streamlit renders $...$ as inline LaTeX,
    so TWO dollar prices in one paragraph (e.g. '$87 ... $81') swallow the text between them and
    render it as maths — while an odd one out ('$90') renders fine, which is what made this look
    random. Escaping keeps every price literal."""
    return str(s).replace("$", r"\$")


def _mc_commentary(log: str) -> str:
    """Pull the English Market Commentary prose out of the Morning Coffee run log —
    main.py prints it between 'Formatting commentary with Claude...' and
    'Translating to Portuguese...'."""
    if not log:
        return ""
    m = re.search(r"Formatting commentary with Claude\.\.\.\s*(.*?)\s*\n\s*Translating to Portuguese",
                  log, re.S)
    return m.group(1).strip() if m else ""


def _mc_sidecar():
    """The Morning Coffee app sidecar (commentary + news headlines), if main.py
    has written one. Preferred over scraping the run log."""
    p = MORNING_COFFEE_DIR / "results" / "_latest_briefing.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _mc_native_heatmap(side) -> bool:
    """Draw the app-native FICC treemap (same renderer as Home) for the Morning Coffee page,
    built from the report's OWN overnight moves in the sidecar (`moves`) so it matches the run
    that just happened — not the dashboard's separate snapshot. Returns False when the sidecar
    carries no move data (older runs), so the caller can fall back to the report PNG."""
    moves = [m for m in ((side or {}).get("moves") or []) if m.get("pct") is not None]
    if not moves:
        return False
    from src import heatmap_html
    import streamlit.components.v1 as components
    sections = []                                   # one band per sector; tiles = products (1-level)
    for sec in list(dict.fromkeys(m.get("sector", "") for m in moves)):
        items = [(m.get("market", ""), float(m["pct"]),
                  (float(m["sigma"]) if m.get("sigma") is not None else None))
                 for m in moves if m.get("sector", "") == sec]
        if items:
            sections.append((sec, [(sec, items)]))
    if not sections:
        return False
    height = int(min(700, max(320, 120 + 84 * len(sections))))
    components.html(heatmap_html.render_html(sections, height, sub_headers=False),
                    height=height + 6, scrolling=False)
    st.caption("**Tile size = how many σ the contract moved overnight** (vs its own ~1-month daily "
               "vol) — colour is direction (green up / red down). Grouped by sector; hover a tile "
               "for the name, % and σ.")
    return True


def _filter_signals(df):
    """Restrict any signals/detail df to the enabled instruments (the Home sector
    filter). Works off an `instruments` column (opportunities — a MR pair needs both
    legs on) or a `ticker` column (the per-strategy detail parquets)."""
    if df is None or getattr(df, "empty", True) or not universe.filter_active():
        return df
    cols = getattr(df, "columns", [])
    en = universe.enabled_tickers()
    if "instruments" in cols:
        return df[df["instruments"].map(
            lambda ins: all(p.strip() in en for p in str(ins).split("/")))]
    if "ticker" in cols:
        return df[df["ticker"].isin(en)]
    return df


# Home filter groups: the top buttons + their drill-down structure. FX carries no region in
# the universe store, so map the currencies here (used only for this filter's grouping).
_REGION_ORDER = ["NA", "EMEA", "APAC", "LATAM"]
_FX_REGION = {
    "CDA Curncy": "NA",
    "ECA Curncy": "EMEA", "BPA Curncy": "EMEA", "SFA Curncy": "EMEA", "SEA Curncy": "EMEA",
    "NOA Curncy": "EMEA", "HEA Curncy": "EMEA", "CCA Curncy": "EMEA", "PPA Curncy": "EMEA",
    "ISA Curncy": "EMEA", "RAA Curncy": "EMEA",
    "JYA Curncy": "APAC", "ADA Curncy": "APAC", "NVA Curncy": "APAC", "SIRA Curncy": "APAC",
    "KOA Curncy": "APAC",
    "BRA Curncy": "LATAM", "PEA Curncy": "LATAM",
}
# (name, member asset classes, split mode) — drives both the group button and its dropdown.
_FILTER_GROUPS = [
    ("Fixed income", ["STIRs", "Bonds"], "region_asset"),
    ("Indices",      ["Indices"],        "region"),
    ("Commodities",  ["Energy", "Metals", "Agriculture", "Softs"], "asset"),
    ("FX",           ["FX"],             "region"),
]


def _sf_region(tk) -> str:
    return universe.region(tk) or _FX_REGION.get(tk, "")


def _sf_key(group, path) -> str:
    return ("sf_pp_" + group + "_" + "_".join(path)).replace(" ", "").replace("&", "")


def _sf_sections():
    """Ordered [(group, path, tickers, key)] — one entry per drill-down sub-section (asset
    class and/or region), matching the structure requested: Fixed income by region→STIRs/Bonds,
    Indices & FX by region, Commodities by asset class."""
    by_ac = {}
    for tk, info in INSTRUMENTS.items():
        by_ac.setdefault(info[2], []).append(tk)
    out = []
    for group, classes, mode in _FILTER_GROUPS:
        if mode == "asset":
            for ac in classes:
                if by_ac.get(ac):
                    out.append((group, [ac], by_ac[ac], _sf_key(group, [ac])))
        elif mode == "region":
            gtks = [tk for ac in classes for tk in by_ac.get(ac, [])]
            for r in _REGION_ORDER:
                tks = [tk for tk in gtks if _sf_region(tk) == r]
                if tks:
                    out.append((group, [r], tks, _sf_key(group, [r])))
        else:                                            # region_asset (Fixed income)
            for r in _REGION_ORDER:
                for ac in classes:
                    tks = [tk for tk in by_ac.get(ac, []) if _sf_region(tk) == r]
                    if tks:
                        out.append((group, [r, ac], tks, _sf_key(group, [r, ac])))
    return out


def _sf_labeler(tickers):
    """format_func giving each product its display name, disambiguating any duplicate names
    within a section by appending the ticker root — the index FUTURE vs its CASH index share a
    name (VGA/SX5E both 'Euro Stoxx 50', GXA/DAX both 'DAX', Z A/UKX both 'FTSE 100'), and
    st.pills collapses options with identical labels, so the futures would silently drop out."""
    names = [INSTRUMENTS[t][0] for t in tickers]
    dup = {n for n in names if names.count(n) > 1}

    def lab(tk):
        nm = INSTRUMENTS[tk][0]
        if nm not in dup:
            return nm
        root = tk.rsplit(" ", 1)[0] if tk.rsplit(" ", 1)[-1] in ("Index", "Comdty", "Curncy") else tk
        return f"{nm} ({root})"
    return lab


def _sf_enabled() -> set:
    """Union of every section's pill selection = the currently-enabled tickers."""
    on = set()
    for _g, _p, tks, key in _sf_sections():
        on |= set(st.session_state.get(key, tks))
    return on


def _sf_current_off():
    """(off_assets, off_tickers) derived from the live pill state — a wholly-off asset class
    collapses to off_assets, everything else off is listed individually in off_tickers."""
    on = _sf_enabled()
    by_ac = {}
    for tk, info in INSTRUMENTS.items():
        by_ac.setdefault(info[2], []).append(tk)
    off_a = [a for a, tks in by_ac.items() if not any(tk in on for tk in tks)]
    off_t = [tk for tk in INSTRUMENTS if tk not in on and INSTRUMENTS[tk][2] not in off_a]
    return off_a, off_t


def _persist_filter() -> None:
    """Write the live filter json (read by the whole app + the report generators).

    Guard: never persist a *fully-off* filter. An all-off state enables nothing — it silently
    blanks every page and report, and because the live filter only resets to the default on a Home
    render it would survive restarts. A transient empty selection (just after "None", or a widget
    reset during "Re-run signals") must not become the saved state, so skip the write and leave the
    last good filter in place."""
    if not _sf_enabled():
        return
    universe.save_filter(*_sf_current_off())


def _sf_apply(setter) -> None:
    """Run setter(section)->tickers over every section, write the keys, persist and rerun."""
    for sec in _sf_sections():
        st.session_state[sec[3]] = list(setter(sec))
    _persist_filter()
    st.rerun()


def render_sector_filter() -> None:
    """Home sector/product filter: a row of group buttons (All / None / Fixed income / Indices /
    Commodities / FX) that toggle a whole sector on or off, each with a drill-down dropdown by
    region / asset class → individual contracts. Persisted and read by the whole app (and the
    report generators) via universe.enabled_tickers()."""
    pal = brand.palette()
    secs = _sf_sections()
    if "sf_init" not in st.session_state:                # each launch starts from the saved default
        st.session_state["sf_init"] = True
        universe.save_filter(*universe.default_off())
    elif not universe.enabled_tickers():                 # stuck fully-off (stale file / glitch) → self-heal
        universe.save_filter(*universe.default_off())    # restore the saved default…
        for _s in secs:                                  # …and drop the stale empty pills so they re-seed
            st.session_state.pop(_s[3], None)
    off_a, off_t = universe.filter_off()
    for _g, _p, tks, key in secs:                        # seed the pills from that (once per session)
        st.session_state.setdefault(
            key, [tk for tk in tks if INSTRUMENTS[tk][2] not in off_a and tk not in off_t])

    on = _sf_enabled()
    st.markdown(f"#### 🗂️  Sectors & products — {len(on)}/{len(INSTRUMENTS)} instruments on")
    with st.container():
        st.caption("Hit a group to switch the whole sector on or off. Open its dropdown to drill in "
                   "by region / asset class and toggle individual contracts.")
        groups = [g[0] for g in _FILTER_GROUPS]
        cols = st.columns(3 + len(groups))
        if cols[0].button("All", key="sf_b_all", use_container_width=True):
            _sf_apply(lambda s: s[2])
        if cols[1].button("None", key="sf_b_none", use_container_width=True):
            _sf_apply(lambda s: [])
        for col, group in zip(cols[2:2 + len(groups)], groups):
            gtks = {tk for s in secs if s[0] == group for tk in s[2]}
            n_on = len(gtks & on)
            if col.button(f"{group}\n\n{n_on}/{len(gtks)}", key=f"sf_b_{group}",
                          use_container_width=True, type="primary" if n_on else "secondary"):
                _sf_apply(lambda s, _g=group, _full=(n_on == len(gtks)):
                          (([] if _full else s[2]) if s[0] == _g
                           else st.session_state.get(s[3], s[2])))
        if cols[-1].button("📌 Set default", key="sf_b_setdef", use_container_width=True,
                           help="Save the current selection as the startup default — the app loads "
                                "this on every launch until you set it again."):
            universe.save_default(*_sf_current_off())
            st.toast("Saved — the dashboard will start with this selection from now on.", icon="📌")
        d_off_a, d_off_t = universe.default_off()
        n_def = len({tk for tk in INSTRUMENTS
                     if INSTRUMENTS[tk][2] not in d_off_a and tk not in d_off_t})
        st.caption(f"📌 **Startup default: {n_def}/{len(INSTRUMENTS)} markets.** Arrange the selection "
                   "how you want it, then **Set default** to change what loads each launch.")

        _gcols = st.columns(len(_FILTER_GROUPS))          # 4 group dropdowns side by side, evenly sized
        for _gi, (group, _classes, mode) in enumerate(_FILTER_GROUPS):
            gsecs = [s for s in secs if s[0] == group]
            gtks = {tk for s in gsecs for tk in s[2]}
            with _gcols[_gi], st.expander(f"{group}  —  {len(gtks & on)}/{len(gtks)} markets", expanded=False):
                last_region = None
                for _g, path, tks, key in gsecs:
                    if mode == "region_asset":
                        region, header = path[0], path[1]
                        if region != last_region:
                            st.markdown(f"<div style='margin:.5rem 0 .1rem;font-weight:800;"
                                        f"font-size:.8rem;letter-spacing:.05em;"
                                        f"color:{pal.get('gold', '#F5C518')}'>{region}</div>",
                                        unsafe_allow_html=True)
                            last_region = region
                    else:
                        header = path[0]
                    tks_sorted = sorted(tks, key=lambda t: INSTRUMENTS[t][0])
                    sel = set(st.session_state.get(key, tks))
                    st.markdown(f"**{header}** &nbsp;·&nbsp; {len(sel & set(tks))}/{len(tks)}")
                    st.pills(header, tks_sorted, selection_mode="multi", key=key,
                             format_func=_sf_labeler(tks_sorted),
                             on_change=_persist_filter, label_visibility="collapsed")


# ── Report-day alerts: a red heads-up banner (Home) + a full-screen popup at release time ─────
# Release times are US-Eastern (USDA/WAOB reports at noon; livestock 3pm; oil outlooks per release_cal).
_USDA_TIME_ET = {
    "WASDE": "12:00", "Crop Production": "12:00", "Crop Production (Annual)": "12:00",
    "Grain Stocks": "12:00", "Prospective Plantings": "12:00", "Acreage": "12:00",
    "Cattle on Feed": "15:00", "Hogs & Pigs": "15:00",
}
_OIL_RELEASES = {"opec": ("OPEC MOMR", "🛢️", "04:00"), "eia": ("EIA STEO", "🛢️", "12:00"),
                 "iea": ("IEA OMR", "🛢️", "04:00")}
# EIA WEEKLY reports (own weekly + holiday-shift logic): (name, icon, base weekday [0=Mon],
# normal ET time, holiday-shifted ET time). Petroleum Status = Wed 10:30, Nat Gas Storage = Thu 10:30;
# each slips one business day per US federal holiday earlier in its week (then the later time).
_EIA_WEEKLY = {
    "petroleum": ("EIA Petroleum Status", "🛢️", 2, "10:30", "11:00"),
    "natgas":    ("EIA Nat Gas Storage",  "🔥", 3, "10:30", "12:00"),
}

# Client-side: at each release time today, drop a full-screen, click-to-dismiss overlay on the
# parent page (+ a persistent OS notification). localStorage dedups so it shows once per release.
_REPORT_POPUP_JS = """<!doctype html><html><head><meta charset="utf-8"></head><body><script>
(function(){
  var RELS = __PAYLOAD__;
  var doc = null; try { doc = window.parent.document; } catch(e) { doc = null; }
  function acked(id){ try { return localStorage.getItem('rpt:'+id)==='1'; } catch(e){ return false; } }
  function ack(id){ try { localStorage.setItem('rpt:'+id,'1'); } catch(e){} }
  function beep(){ try { var a=new (window.AudioContext||window.webkitAudioContext)();
    var o=a.createOscillator(), g=a.createGain(); o.connect(g); g.connect(a.destination);
    o.type='sine'; o.frequency.value=880; g.gain.value=0.07; o.start();
    setTimeout(function(){ o.stop(); }, 320); } catch(e){} }
  function osnotify(r){ try { if (window.Notification && Notification.permission==='granted'){
    new Notification('\\uD83D\\uDD34 '+r.name+' \\u2014 released',
      {body:'Scheduled '+r.t+' ET. Check the numbers now.', requireInteraction:true, tag:r.id}); } } catch(e){} }
  function show(r){
    if (acked(r.id)) return;
    osnotify(r); beep();
    if (!doc || !doc.body) return;
    if (doc.getElementById('rpt-'+r.id)) return;
    var ov = doc.createElement('div'); ov.id = 'rpt-'+r.id;
    ov.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:rgba(8,8,8,.62);display:flex;align-items:center;justify-content:center;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif';
    var card = doc.createElement('div');
    card.style.cssText = 'max-width:440px;width:86%;background:#fff;border-top:7px solid #D32F2F;border-radius:12px;padding:26px 28px 24px;box-shadow:0 18px 60px rgba(0,0,0,.55);text-align:center';
    card.innerHTML = '<div style="font-size:44px;line-height:1">'+r.icon+'</div>'
      + '<div style="color:#B71C1C;font-weight:800;font-size:12px;letter-spacing:1.5px;margin-top:8px">REPORT RELEASED</div>'
      + '<div style="color:#111;font-weight:800;font-size:22px;margin:5px 0 3px">'+r.name+'</div>'
      + '<div style="color:#666;font-size:13px;margin-bottom:20px">Scheduled '+r.t+' ET &middot; check the numbers now</div>';
    var btn = doc.createElement('button'); btn.textContent = 'Dismiss';
    btn.style.cssText = 'background:#D32F2F;color:#fff;border:0;border-radius:8px;padding:11px 30px;font-size:15px;font-weight:700;cursor:pointer';
    function close(){ ack(r.id); if (ov.parentNode) ov.parentNode.removeChild(ov); }
    btn.onclick = close;
    ov.onclick = function(e){ if (e.target === ov) close(); };
    card.appendChild(btn); ov.appendChild(card); doc.body.appendChild(ov);
  }
  try { if (window.Notification && Notification.permission==='default') Notification.requestPermission(); } catch(e){}
  RELS.forEach(function(r){
    if (acked(r.id)) return;
    var delay = r.fire - Date.now();
    if (delay <= 0) show(r); else if (delay < 86400000) setTimeout(function(){ show(r); }, delay);
  });
})();
</script></body></html>"""


def _is_cot_release(today) -> bool:
    """True if the CFTC Commitments of Traders posts today — normally Friday 15:30 ET,
    delayed to the next business day when a US federal holiday falls in the report week."""
    from datetime import timedelta
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hols = {pd.Timestamp(d).date() for d in USFederalHolidayCalendar().holidays(
        start=f"{today.year - 1}-12-01", end=f"{today.year + 1}-01-31")}

    def _nbd(d):
        d += timedelta(days=1)
        while d.weekday() >= 5 or d in hols:
            d += timedelta(days=1)
        return d

    for wk in (0, 1):                                        # this week's Friday, and last week's
        friday = today - timedelta(days=today.weekday()) + timedelta(days=4 - 7 * wk)
        rel = friday
        for _ in range(sum(1 for i in range(5) if (friday - timedelta(days=4 - i)) in hols)):
            rel = _nbd(rel)                                  # each holiday Mon–Fri delays one business day
        if rel == today:
            return True
    return False


def _eia_weekly_today(today) -> list:
    """EIA weekly reports releasing today — Petroleum Status (normally Wed 10:30 ET) and Natural
    Gas Storage (Thu 10:30 ET) — each delayed one business day per US federal holiday that falls
    Mon..its release weekday of that report's week (a shifted release uses the later time)."""
    from datetime import timedelta
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hols = {pd.Timestamp(d).date() for d in USFederalHolidayCalendar().holidays(
        start=f"{today.year - 1}-12-01", end=f"{today.year + 1}-01-31")}

    def _nbd(d):
        d += timedelta(days=1)
        while d.weekday() >= 5 or d in hols:
            d += timedelta(days=1)
        return d

    out = []
    for name, icon, base_wd, t0, t1 in _EIA_WEEKLY.values():
        for wk in (0, 1):                                   # this week and last week (a late shift)
            monday = today - timedelta(days=today.weekday()) - timedelta(days=7 * wk)
            rel = monday + timedelta(days=base_wd)
            nhol = sum(1 for i in range(base_wd + 1) if (monday + timedelta(days=i)) in hols)
            for _ in range(nhol):
                rel = _nbd(rel)
            if rel == today:
                out.append({"name": name, "icon": icon, "color": "#37474F", "t": t1 if nhol else t0})
                break
    return out


def _todays_releases(today=None) -> list:
    """Fundamental reports releasing today: {name, icon, color, t (HH:MM ET), fire_ms}. ET-dated."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src import agdata, release_cal, repcal
    et = ZoneInfo("America/New_York")
    today = today or datetime.now(et).date()
    out = []
    try:
        cal = agdata.report_calendar()
        for r in cal[cal["date"].dt.date == today].to_dict("records"):
            rep = r["report"]
            icon, _lbl, color = repcal.USDA_ICON.get(rep, ("📊", rep, "#2E7D32"))
            out.append({"name": f"USDA {rep}", "icon": icon, "color": color,
                        "t": _USDA_TIME_ET.get(rep, "12:00")})
    except Exception:
        pass
    try:
        for r in release_cal.next_12_months(today):
            for key, (who, icon, t) in _OIL_RELEASES.items():
                if r.get(key) == today:
                    out.append({"name": who, "icon": icon, "color": "#37474F", "t": t})
    except Exception:
        pass
    try:
        if _is_cot_release(today):
            out.append({"name": "CFTC COT", "icon": "📊", "color": "#7B1FA2", "t": "15:30"})
    except Exception:
        pass
    try:
        out.extend(_eia_weekly_today(today))
    except Exception:
        pass
    for o in out:
        hh, mm = map(int, o["t"].split(":"))
        o["fire_ms"] = int(datetime(today.year, today.month, today.day, hh, mm, tzinfo=et).timestamp() * 1000)
        o["key"] = alerts.key_for_release(o["name"])       # for the per-report banner/popup toggles
    return out


def render_report_banner() -> None:
    """Red heads-up strip at the top of Home on days a fundamental report releases (only for reports
    whose Home banner is switched on in Alert Settings)."""
    rels = [r for r in _todays_releases() if alerts.alert_enabled(r.get("key"), "banner")]
    if not rels:
        return
    items = " &nbsp;&middot;&nbsp; ".join(f"{r['icon']} <b>{r['name']}</b> {r['t']} ET" for r in rels)
    st.markdown(
        "<div style='background:linear-gradient(90deg,#B71C1C,#E53935);color:#fff;padding:11px 16px;"
        "border-radius:9px;margin:0 0 14px;font-size:15px;border:1px solid #7f0000;"
        "box-shadow:0 2px 8px rgba(0,0,0,.28)'>&#128308; <b>REPORT DAY</b> &mdash; releasing today: "
        + items + ". <span style='opacity:.9'>A full-screen alert pops at release time.</span></div>",
        unsafe_allow_html=True)


def render_report_popup() -> None:
    """Invisible JS component: fires a full-screen, click-to-dismiss overlay (+ OS notification) at
    each report's release time today, on top of any page (only for reports whose popup is switched
    on in Alert Settings). No-op on days with no releases."""
    rels = [r for r in _todays_releases() if alerts.alert_enabled(r.get("key"), "popup")]
    if not rels:
        return
    import json as _json
    import streamlit.components.v1 as components
    payload = _json.dumps([{"id": str(r["fire_ms"]) + "|" + r["name"], "name": r["name"],
                            "icon": r["icon"], "t": r["t"], "fire": r["fire_ms"]} for r in rels])
    components.html(_REPORT_POPUP_JS.replace("__PAYLOAD__", payload), height=0)


def _render_corr_break_banner() -> None:
    """Amber strip on the Product Correlations page whenever product pairs sit at an extreme
    (≤5th / ≥95th percentile) of their own rolling 1-year correlation range — gated by the same
    Alert Settings toggle as the release banners. Fails silent: a data hiccup never blocks the page."""
    if not alerts.alert_enabled("sectorcorr", "banner"):
        return
    try:
        ex = _sc_extremes(date.today().isoformat(), MODE)
    except Exception:
        return
    if ex is None or ex.empty:
        return
    tops = [f"**{universe.name(a)} ↔ {universe.name(b)}** ({d:+.2f} vs 1Y, {p:.0f}th pctl)"
            for a, b, d, p in zip(ex["a"].head(3), ex["b"].head(3),
                                  ex["diff"].head(3), ex["pctl"].head(3))]
    more = f" — and {len(ex) - 3} more" if len(ex) > 3 else ""
    st.warning("🔗 **Correlation breaks** — pairs at an extreme of their 1-year range: "
               + " · ".join(tops) + more + ". See the correlation maps below.")


def render_home() -> None:
    render_report_banner()
    snap = _load_snap()

    # (world clocks moved to the fixed top bar — rendered on every page)
    render_sector_filter()
    st.subheader("Data")
    c1, c2, c3, c4 = st.columns(4)
    c4.button("☕  Morning Coffee", use_container_width=True, key="home_mc",
              on_click=_go, args=("Morning Coffee",),
              help="The morning report — overnight moves, levels and the day ahead.")
    def _run_ficc_pull():
        # TWO PHASES so the Terminal only needs to be open for the short one:
        # fetch (Bloomberg, ~3-5 min) -> banner flips to "close the Terminal" ->
        # compute (fits + signals, Terminal-closed). Live elapsed timers on both.
        def _phase(args, msg):
            ph = st.empty()
            t0 = time.time()
            proc = subprocess.Popen([sys.executable, str(SNAPSHOT_CLI), *args], cwd=str(ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    env={**os.environ, "DATAFEED_MODE": "bloomberg",
                                         "PYTHONUTF8": "1"})
            while proc.poll() is None:
                ph.info(msg.format(el=(time.time() - t0) / 60))
                time.sleep(5)
            out, err = proc.communicate()
            ph.empty()
            return proc.returncode, out, err, (time.time() - t0) / 60

        _t_all = time.time()
        rc, _out, _err, _m1 = _phase(
            ["--fetch"],
            "⏳ **Bloomberg phase** — {el:.1f} min elapsed (typically 3–5 min). "
            "The Terminal must stay open for THIS phase only.")
        if rc != 0:
            st.error("Snapshot fetch failed:\n\n" + (_err or _out or "no output"))
            return
        _done = st.empty()
        _done.success(f"✅ **Bloomberg finished ({_m1:.1f} min) — you can CLOSE THE "
                      "TERMINAL now.** Crunching the maths…")
        rc, _out, _err, _m2 = _phase(
            ["--compute"],
            "🧮 **Compute phase** (Terminal-closed) — {el:.1f} min elapsed: COT signals, "
            "own-vol-curve fits, manifest.")
        if rc != 0:
            st.error("Snapshot compute failed (the fetched data is safe on disk — "
                     "'Re-run signals' or retry):\n\n" + (_err or _out or "no output"))
            return
        run_daily.run(); load_signals.clear()
        _regen_mc_heatmap()          # refresh the Morning Coffee heatmap on Home
        gitbackup.push_data_async()  # fresh data → GitHub → VPS site within ~15 min
        st.session_state.pop("ficc_pull_confirm", None)
        _done.empty()
        st.success(f"Snapshot complete — Bloomberg needed {_m1:.1f} min, maths "
                   f"{_m2:.1f} min, total {(time.time() - _t_all) / 60:.1f} min.")
        st.rerun()

    # Heavy handlers are DEFERRED (flag set here, executed below the row): blocking inside a
    # column slot pauses the script mid-row, so Streamlit showed a half-drawn fresh button row
    # with the old row faded beneath it for the whole computation.
    if c1.button("📥 Pull Bloomberg Snapshot", use_container_width=True, key="home_pull",
                 help="Two phases: the Terminal is only needed for the FETCH (~3–5 min) — "
                      "the banner tells you when you can close it — then the maths (own-vol "
                      "curve, COT, signals) runs Terminal-free. Equities have their own pull "
                      "on the Equities home page."):
        # Same-day guard: a re-pull re-spends thousands of Bloomberg hits (the daily-capacity
        # budget) for near-identical data, so it asks first.
        _today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if str((snap or {}).get("created", ""))[:10] == _today:
            st.session_state["ficc_pull_confirm"] = True
        else:
            st.session_state["ficc_pull_go"] = True
    if c2.button("🔁 Re-run signals", use_container_width=True, key="home_rerun",
                 help="Recompute all strategies from the current data — instant in snapshot mode."):
        st.session_state["rerun_signals_go"] = True
    if st.session_state.get("ficc_pull_confirm"):
        st.warning(f"⚡ Snapshot **already pulled today** ({_to_et((snap or {}).get('created', ''))}). "
                   "Pulling again re-spends the day's Bloomberg data allowance on near-identical "
                   "data — worth it only if the first pull was bad or markets have moved a lot.")
        _g1, _g2, _ = st.columns([1.4, 1, 3.6])
        if _g1.button("Pull again anyway", key="ficc_pull_anyway"):
            st.session_state.pop("ficc_pull_confirm", None)
            st.session_state["ficc_pull_go"] = True
        if _g2.button("Cancel", key="ficc_pull_cancel"):
            st.session_state.pop("ficc_pull_confirm", None); st.rerun()
    if st.session_state.pop("ficc_pull_go", False):
        _run_ficc_pull()
    if st.session_state.pop("rerun_signals_go", False):
        with st.spinner("Recomputing all signals…"):
            run_daily.run()
        load_signals.clear(); st.rerun()
    if c3.button("⬇️  Export snapshot to Excel", use_container_width=True, key="home_excel",
                 disabled=not (SNAPSHOT_DIR / "prices.parquet").exists()):
        with st.spinner("Building workbook…"):
            with tempfile.TemporaryDirectory() as tmp:
                xlsx = Path(tmp) / "snapshot.xlsx"
                res = subprocess.run([sys.executable, str(SNAPSHOT_CLI), "--excel", str(xlsx)],
                                     cwd=str(ROOT), capture_output=True, text=True)
                ok = res.returncode == 0 and xlsx.exists()
                st.session_state["snap_xlsx"] = xlsx.read_bytes() if ok else None
        if not st.session_state.get("snap_xlsx"):
            st.error("Excel export failed:\n\n" + (res.stderr or res.stdout or "no output"))
    if st.session_state.get("snap_xlsx"):
        st.download_button("Download snapshot.xlsx", data=st.session_state["snap_xlsx"],
                           file_name="bloomberg_snapshot.xlsx", use_container_width=True,
                           key="home_xlsx_dl",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    _overnight_moves(snap)
    _econ_figures()
    _home_heatmap()


# ── EQUITIES side ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _eq_universe():
    """Index constituents (cached — membership changes rarely; cleared by 'Pull equities data')."""
    return equities.load_universe()


@st.cache_data(ttl=300, show_spinner=False)
def _eq_movers(index_keys: tuple):
    """Overnight-movers frame for the selected indices (cached; cleared by 'Refresh quotes')."""
    return equities.movers_frame(list(index_keys), universe=_eq_universe())


def _equities_overnight_moves(index_keys, snap) -> None:
    st.subheader("Overnight moves")
    f = _eq_movers(tuple(index_keys))
    f = f.dropna(subset=["pct"]) if not f.empty else f
    if f.empty:
        st.caption("No overnight equity quotes available. In Bloomberg mode click **Pull equities "
                   "data**; otherwise this shows the built-in demo universe.")
        return
    f, _q = prodsearch.search_row_box(f, ["ticker", "name", "sector", "index"], key="eq_home_search")
    if _q and f.empty:
        st.info(prodsearch.NO_MATCH_STOCK.format(q=_q))
        return
    # One row per COMPANY (by name) — a company in more than one selected index is listed once, with
    # every index it belongs to joined in the Index column (registry order). This also collapses a
    # cross-listed name that appears under different exchange tickers across indexes (e.g. Stellantis
    # = STLAP FP in CAC 40 + STLAM IM in Euro Stoxx). On Bloomberg the group key is `short_name`,
    # which is stable per company across its listings.
    _order = {k: i for i, k in enumerate(equities.INDICES)}
    _by_co = (f.groupby("name", as_index=False)
                .agg(sector=("sector", "first"), pct=("pct", "first"), last=("last", "first"),
                     sigma=("sigma", "first"),
                     index=("index", lambda s: ", ".join(sorted(set(s), key=lambda k: _order.get(k, 99))))))
    disp = pd.DataFrame({
        "Stock": _by_co["name"], "Index": _by_co["index"], "Sector": _by_co["sector"],
        "% (o/n)": _by_co["pct"].astype(float), "Last": _by_co["last"].astype(float),
        "σ (1m)": _by_co["sigma"].astype(float),
    }).sort_values("σ (1m)", ascending=False, na_position="last")
    st.caption("Overnight move (previous close → latest) for each company in the selected indices "
               "(a company in several indices is listed once, with all its indexes shown), sorted by "
               "**σ (1m)** = the move in standard deviations of the stock's own ~1-month daily moves."
               + ("" if MODE == "bloomberg" else f"  ·  _{equities.data_status()}_"))

    def _color_move(col):
        out = []
        for v in col:
            if v != v or v == 0:
                out.append("color:#888")
            elif v > 0:
                out.append("color:#137333;font-weight:700")
            else:
                out.append("color:#c5221f;font-weight:700")
        return out

    _fmt = {"% (o/n)": lambda v: f"{v:+.2f}%",
            "Last": lambda v: f"{v:,.2f}",
            "σ (1m)": lambda v: f"{v:+.1f}σ" if v == v else "—"}
    brand.themed_dataframe(disp, _fmt,
                           colorers=[(["% (o/n)", "σ (1m)"], _color_move)], height=440)


@st.cache_data(ttl=300, show_spinner=False)
def _eq_heatmap_sections(index_keys: tuple):
    """Nested treemap data for the Equities heatmap: [(index, [(gics_sector, [(name, pct, σ)…])…])…],
    ordered by index then canonical GICS sector ("Other"/extras last). Cached; cleared by the
    Pull/Refresh buttons. Returns [] when there's no overnight data."""
    f = _eq_movers(tuple(index_keys))
    f = f.dropna(subset=["sigma"]) if not f.empty else f
    if f.empty:
        return []
    sections = []
    for key in [k for k in index_keys if (f["index"] == k).any()]:
        di = f[f["index"] == key]
        present = list(dict.fromkeys(di["sector"]))
        ordered = ([s for s in equities.GICS_SECTORS if s in present]
                   + [s for s in present if s not in equities.GICS_SECTORS])   # extras (e.g. "Other") last
        subs = []
        for sec in ordered:
            ds = di[di["sector"] == sec]
            if ds.empty:
                continue
            subs.append((sec, [(r["name"], float(r["pct"]) if r["pct"] == r["pct"] else 0.0,
                                float(r["sigma"]) if r["sigma"] == r["sigma"] else None)
                               for _, r in ds.iterrows()]))
        if subs:
            sections.append((key, subs))
    return sections


def _equities_heatmap(index_keys) -> None:
    from src import heatmap_html
    import streamlit.components.v1 as components
    st.subheader("Market heatmap")
    st.caption("**Tile size = how many standard deviations (σ) the stock moved overnight** — colour "
               "is direction (green up / red down), deepening with |σ|. Grouped by index, then GICS "
               "sector; hover a tile for the name, % and σ."
               + ("" if MODE == "bloomberg" else f"  ·  _{equities.data_status()}_"))
    sections = _eq_heatmap_sections(tuple(index_keys))
    if not sections:
        st.caption("No overnight data to chart.")
        return
    height = int(min(1320, max(380, 40 + 172 * len(sections))))
    components.html(heatmap_html.render_html(sections, height, sub_headers=True),
                    height=height + 6, scrolling=False)


# ── Equities auto-pull (Windows Task Scheduler) ──────────────────────────────
# The ⏰ control left of "Pull equities data": a daily scheduled run of
# `snapshot.py --equities` via run_eq_autopull.bat, so e.g. a 09:00 ET pull
# lands fresh data for the US open without touching the app. The task runs
# whether or not BASIS is open (it's a Task Scheduler job, not an app thread).
_EQ_AUTOPULL_TASK = "BASIS Equities Auto Pull"
_EQ_AUTOPULL_FILE = ROOT / "data" / "eq_autopull.json"
_EQ_AUTOPULL_BAT = ROOT / "run_eq_autopull.bat"


def _eq_autopull_cfg() -> dict:
    try:
        # utf-8-sig: tolerate a BOM from hand-edits (PowerShell's Set-Content utf8 adds one)
        cfg = json.loads(_EQ_AUTOPULL_FILE.read_text(encoding="utf-8-sig"))
        # "time" is LAPTOP wall time. Older saves stored ET + its local mapping —
        # migrate via local_time so the shown time matches when the task fires.
        t = str(cfg.get("time") or cfg.get("local_time") or "08:00")
        return {"enabled": bool(cfg.get("enabled")), "time": t}
    except Exception:
        return {"enabled": False, "time": "08:00"}


def _eq_autopull_apply(enabled: bool, hhmm: str) -> tuple[bool, str]:
    """Create/refresh (or delete) the Windows scheduled task at the given LAPTOP
    wall time (user choice: no timezone conversion — what you set is when it runs)."""
    if enabled:
        cmd = ["schtasks", "/Create", "/F", "/TN", _EQ_AUTOPULL_TASK,
               "/TR", f'"{_EQ_AUTOPULL_BAT}"',
               "/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI",
               "/ST", hhmm]
    else:
        cmd = ["schtasks", "/Delete", "/F", "/TN", _EQ_AUTOPULL_TASK]
    r = subprocess.run(cmd, capture_output=True, text=True)
    err = (r.stderr or r.stdout or "").strip()
    # deleting a task that never existed is success, not failure
    ok = r.returncode == 0 or (not enabled and "cannot find" in err.lower())
    # No catch-up on purpose: the task runs only if the laptop is ON at the scheduled time; a run
    # missed because it was off/asleep is simply skipped (schtasks' default — StartWhenAvailable off).
    if ok:
        _EQ_AUTOPULL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _EQ_AUTOPULL_FILE.write_text(json.dumps(
            {"enabled": enabled, "time": hhmm}, indent=2))
    return ok, err


def _eq_autopull_control(col) -> None:
    cfg = _eq_autopull_cfg()
    _lbl = (f"⏰ Auto-pull · {cfg['time']}" if cfg["enabled"] else "⏰ Auto-pull · off")
    with col.popover(_lbl, use_container_width=True,
                     help="Schedule an automatic daily equities pull (weekdays) — runs even "
                          "when BASIS is closed, via Windows Task Scheduler."):
        _cur = dtime(*(int(x) for x in cfg["time"].split(":")))
        _t = st.time_input("Pull daily at (laptop time)", value=_cur, step=300,
                           key="eq_ap_time")
        _on = st.toggle("Automatic pull on", value=cfg["enabled"], key="eq_ap_on")
        if st.button("Save", key="eq_ap_save", use_container_width=True, type="primary"):
            ok, msg = _eq_autopull_apply(_on, _t.strftime("%H:%M"))
            if ok:
                st.session_state["eq_ap_saved"] = True
                st.rerun()
            else:
                st.error(f"Couldn't update the scheduled task:\n\n{msg or 'no output'}")
        if st.session_state.pop("eq_ap_saved", False):
            st.success("Saved.")
        st.caption((f"**On** — weekdays at {cfg['time']} laptop time. "
                    if cfg["enabled"] else "**Off.** ")
                   + "Runs the same job as **Pull equities data** (Yahoo quotes/history + "
                     "weekly fundamentals + the **Technical Analysis** backfill & signals; Bloomberg "
                     "membership only if the Terminal is up), then syncs the VPS. It runs only if the "
                     "laptop is **on** at pull time — a missed run is skipped, not caught up later. "
                     "Log: %LOCALAPPDATA%\\basis_eq_autopull.log")


def render_equities_home() -> None:
    snap = _load_snap()
    # (world clocks moved to the fixed top bar — rendered on every page)
    _keys = list(equities.INDICES.keys())
    sel = st.multiselect("Indices to show", _keys, default=list(equities.DEFAULT_INDICES),
                         key="eq_idx_filter",
                         help="Scope the movers table and heatmap to these indices. "
                              "Russell 2000 (~2000 names) is opt-in — add it here when needed.")
    sel = sel or _keys
    st.subheader("Data")
    c0, c1, c2, c3 = st.columns([1.15, 1.55, 1.55, 2.75])
    _eq_autopull_control(c0)
    try:                                   # mirror snapshot.py's equities pull switches
        from snapshot import PULL_EQUITY_CONSTITUENTS as _EQ_ON, PULL_FUNDAMENTALS as _EQF_ON
    except Exception:
        _EQ_ON = _EQF_ON = True
    _eq_pull_on = bool(_EQ_ON or _EQF_ON)
    if c1.button("📥 Pull equities data", use_container_width=True, key="eq_pull",
                 disabled=not _eq_pull_on,
                 help=("The Equities side's own data pull — overnight quotes/history and the "
                       "(weekly-guarded) fundamentals refresh come FREE from Yahoo Finance; "
                       "Bloomberg only refreshes index membership when the Terminal is up "
                       "(cached membership otherwise).") if _eq_pull_on else
                      ("OFF — individual-stock & Company-Fundamentals pulling is disabled to keep "
                       "the Bloomberg pull light. Equity INDEX numbers come from the FICC snapshot.")):
        # No app-mode gate: the app itself runs in snapshot mode all day — the pull SUBPROCESS
        # sets DATAFEED_MODE=bloomberg, same as the FICC snapshot button. If the Terminal is
        # closed the pull fails gracefully and never wipes the caches.
        # Same-day guard: an equities re-pull re-spends thousands of Bloomberg hits
        # (the daily-capacity budget) on near-identical data, so it asks first.
        _today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if str((snap or {}).get("equities_pulled", ""))[:10] == _today:
            st.session_state["eq_pull_confirm"] = True
        else:
            st.session_state["eq_pull_go"] = True
    if not _eq_pull_on:
        st.caption("ℹ️ Individual-stock & Company-Fundamentals pulling is **off** — equity "
                   "**index** numbers come from the FICC snapshot. Re-enable in snapshot.py "
                   "(`PULL_EQUITY_CONSTITUENTS` / `PULL_FUNDAMENTALS`) for per-stock data.")
    if st.session_state.get("eq_pull_confirm"):
        st.warning(f"⚡ Equities **already pulled today** "
                   f"({_to_et((snap or {}).get('equities_pulled', ''))}). Quotes and fundamentals "
                   "re-pull free from Yahoo; only the Bloomberg membership refresh re-spends "
                   "Terminal hits.")
        _g1, _g2, _ = st.columns([1.4, 1, 3.6])
        if _g1.button("Pull again anyway", key="eq_pull_anyway"):
            st.session_state.pop("eq_pull_confirm", None)
            st.session_state["eq_pull_go"] = True
        if _g2.button("Cancel", key="eq_pull_cancel"):
            st.session_state.pop("eq_pull_confirm", None); st.rerun()
    if st.session_state.pop("eq_pull_go", False):
        _ph = st.empty()
        _t0 = time.time()
        proc = subprocess.Popen([sys.executable, str(SNAPSHOT_CLI), "--equities"],
                                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env={**os.environ, "DATAFEED_MODE": "bloomberg",
                                                "PYTHONUTF8": "1"})
        while proc.poll() is None:
            _el = (time.time() - _t0) / 60
            _ph.info(f"⏳ Pulling equities — **{_el:.1f} min elapsed** (typically ~5–7 min for "
                     "the ~2,700-name universe: ETF membership + chunked Yahoo quotes/history, "
                     "plus fundamentals when their cycle is due).")
            time.sleep(5)
        _out, _err = proc.communicate()
        _ph.empty()
        if proc.returncode != 0:
            st.error("Equities pull failed:\n\n" + (_err or _out or "no output"))
        else:
            _eq_universe.clear(); _eq_movers.clear(); _eq_heatmap_sections.clear()
            _eqf_frame.clear()
            gitbackup.push_data_async()  # fresh data → GitHub → VPS site within ~15 min
            st.success(f"Equities data refreshed ({(time.time() - _t0) / 60:.1f} min).")
            st.rerun()
    if c2.button("🔄 Refresh quotes", use_container_width=True, key="eq_refresh",
                 help="Re-pull the latest closes from Yahoo Finance (free) and rebuild the "
                      "movers table and heatmap. Falls back to the cached quotes offline."):
        _eq_movers.clear(); _eq_heatmap_sections.clear()
        st.session_state["eq_refresh_note"] = True
        st.rerun()
    if st.session_state.pop("eq_refresh_note", False):
        st.info("Quotes refreshed — live Yahoo Finance when reachable, otherwise the cached "
                "pull (see the source caption).")
    _n = sum(len(v) for v in _eq_universe().values())
    c3.caption(f"**Universe:** {_n} index constituents across {len(_keys)} indices · "
               + equities.data_status() + ". Quotes, history and fundamentals ride Yahoo "
               "Finance free of charge; Bloomberg only refreshes index membership.")
    st.divider()
    _equities_overnight_moves(sel, snap)
    _econ_figures()
    _equities_heatmap(sel)


# ── Company Fundamentals (Equities) ───────────────────────────────────────────
_EQF_GOOD_CSS = "color:#137333;font-weight:700"
_EQF_BAD_CSS = "color:#c5221f;font-weight:700"

# Preset screens — thresholds are SECTOR percentiles (like-for-like within GICS sector),
# except 'raw<=' which caps the raw value (a payout ratio over ~80% strains the dividend
# whatever the sector norms are).
_EQF_PRESETS = {
    "All companies": [],
    "Quality — high ROE, low leverage": [("RETURN_COM_EQY", ">=", 70.0),
                                         ("TOT_DEBT_TO_TOT_EQY", "<=", 40.0)],
    "Cheap vs sector — fwd P/E + EV/EBITDA": [("BEST_PE_RATIO", "<=", 30.0),
                                              ("EV_TO_T12M_EBITDA", "<=", 40.0)],
    "Growth — revenue + EPS": [("SALES_GROWTH", ">=", 60.0), ("EPS_GROWTH", ">=", 60.0)],
    "Income — yield with a sustainable payout": [("EQY_DVD_YLD_IND", ">=", 70.0),
                                                 ("DVD_PAYOUT_RATIO", "raw<=", 80.0)],
}


@st.cache_data(ttl=600, show_spinner=False)
def _eqf_frame(index_keys: tuple):
    """Fundamentals frame + sector percentiles for the selected indices (cached; cleared by
    'Pull fundamentals'). Uses the CACHED membership — a live INDX_MEMBERS re-pull here made
    the first click on the page sit on a dead screen for ~30s+ in Bloomberg mode."""
    df, asof, src = eqfunda.company_frame(universe=equities.cached_universe(),
                                          index_keys=list(index_keys))
    return (eqfunda.add_sector_percentiles(df) if not df.empty else df), asof, src


def _eqf_pull_note() -> None:
    """Pull-cadence note for the fundamentals-driven pages: fundamentals move slowly, so
    each tier refreshes on a cycle and its data can lag by up to that cycle length.
    (On the Yahoo source the cycle is about pull time / rate limits, not Bloomberg hits.)"""
    s = eqfunda.staleness()
    core = (f"core indices last pulled **{s['last_pull']}** ({s['core_age']}d ago)"
            if s["last_pull"] else "core indices **never pulled**")
    heavy = ""
    if getattr(equities, "HEAVY_INDICES", None):
        heavy = ("; Russell 2000 last pulled **" + s["last_heavy_pull"] + "** "
                 f"({s['heavy_age']}d ago)" if s["last_heavy_pull"]
                 else "; Russell 2000 **never pulled**")
    src_note = ("free Yahoo Finance pulls" if equities._use_yf()
                else "to respect Bloomberg's daily data capacity, pulls are minimised")
    st.caption(f"ℹ️ **Pull-cadence note** — {src_note}; fundamentals refresh on a cycle: "
               f"core indices **weekly**, Russell 2000 **monthly**. {core}{heavy}. Figures "
               f"and expected report dates can lag by up to their cycle; **Pull equities "
               f"data** (Equities home) forces a refresh.")


def _eqf_styles(sub: pd.DataFrame, field: str) -> list:
    """Row-aligned CSS for one metric column — coloured only in the sector-percentile tails,
    direction-aware (low is the good end for multiples/leverage, high for the rest)."""
    out = []
    for _, r in sub.iterrows():
        g = eqfunda.goodness(field, r.get(field + "__pctl"))
        out.append(_EQF_GOOD_CSS if g > 0 else _EQF_BAD_CSS if g < 0 else "")
    return out


def _eqf_screener(df: pd.DataFrame) -> None:
    labels = {f["field"]: f["label"] for f in eqfunda.FIELDS}
    c1, c2 = st.columns([2, 3])
    preset = c1.selectbox("Preset screen", list(_EQF_PRESETS), key="eqf_preset")
    sectors = sorted(df["sector"].dropna().unique())
    sec_sel = c2.multiselect("Sectors", sectors, key="eqf_sectors", help="Blank = all sectors.")
    metric_opts = [f["field"] for f in eqfunda.FIELDS if f["kind"] not in ("text", "date")]
    cols = st.multiselect("Metrics (columns)", metric_opts, default=eqfunda.SCREENER_DEFAULT,
                          format_func=lambda f: labels[f], key="eqf_cols") or eqfunda.SCREENER_DEFAULT
    cols = [f for f in cols if f in df.columns]        # a field a pull didn't return can't be a column
    sub = df[df["sector"].isin(sec_sel)] if sec_sel else df
    sub, _q = prodsearch.search_row_box(sub, ["name", "ticker", "sector", "indices"], key="eqf_search")
    if _q and sub.empty:
        st.info(prodsearch.NO_MATCH_STOCK.format(q=_q))
        return
    for f, op, thr in _EQF_PRESETS[preset]:
        col = f if op == "raw<=" else f + "__pctl"
        if col not in sub.columns:
            continue                                   # metric absent from this pull — skip this leg
        v = pd.to_numeric(sub[col], errors="coerce")
        sub = sub[v.notna() & ((v >= thr) if op == ">=" else (v <= thr))]
    if sub.empty or not cols:
        st.caption("No companies pass this screen in the current selection.")
        return
    if "CRNCY_ADJ_MKT_CAP" in sub.columns:
        sub = sub.sort_values("CRNCY_ADJ_MKT_CAP", ascending=False, na_position="last")
    disp = pd.DataFrame({"Company": sub["name"].values, "Sector": sub["sector"].values,
                         "Index": sub["indices"].values})
    fmt, colorers = {}, []
    for f in cols:
        disp[labels[f]] = pd.to_numeric(sub[f], errors="coerce").values
        fmt[labels[f]] = (lambda _f: lambda v: eqfunda.fmt_value(_f, v))(f)
        colorers.append(([labels[f]], (lambda sty: lambda col: sty)(_eqf_styles(sub, f))))
    st.caption(f"**{len(disp)}** companies · sorted by market cap · **green / red = top / bottom "
               "20% of the stock's own GICS sector** on that metric, direction-aware (low is the "
               "good end for valuation multiples and leverage, high for the rest).")
    brand.themed_dataframe(disp, fmt, colorers=colorers, na_rep="—", height=520)


def _eqf_group_rows(row, peers: pd.DataFrame) -> dict:
    """{group: [{label,value,median,pctl,pctl_txt,good}, ...]} for one company vs its GICS
    sector peers — the tearsheet's (and the PDF's) building block."""
    out: dict = {}
    for spec in eqfunda.FIELDS:
        if spec["kind"] in ("text", "date"):
            continue                                   # currency / next report live in the header
        f = spec["field"]
        med = pd.to_numeric(peers[f], errors="coerce").median() if f in peers.columns else float("nan")
        p = row.get(f + "__pctl")
        p = None if (p is None or p != p) else float(p)
        out.setdefault(spec["group"], []).append({
            "label": spec["label"], "value": eqfunda.fmt_value(f, row.get(f)),
            "median": eqfunda.fmt_value(f, med),
            "pctl": p, "pctl_txt": "—" if p is None else eqfunda.ordinal(p),
            "good": eqfunda.goodness(f, p),
        })
    return out


def _eqf_tearsheet(df: pd.DataFrame, asof: str, src: str) -> None:
    d = df.sort_values("name").reset_index(drop=True)
    lab = (d["name"] + "  (" + d["ticker"] + ")").tolist()
    pick = st.selectbox("Company", lab, key="eqf_co")
    row = d.iloc[lab.index(pick)]
    bits = [row["sector"], row["indices"], row["region"],
            "Mkt cap " + eqfunda.fmt_value("CRNCY_ADJ_MKT_CAP", row.get("CRNCY_ADJ_MKT_CAP")),
            eqfunda.fmt_value("CRNCY", row.get("CRNCY")),
            "next report " + eqfunda.fmt_value("EXPECTED_REPORT_DT", row.get("EXPECTED_REPORT_DT"))]
    st.markdown(f"### {row['name']}")
    st.caption("  ·  ".join(str(b) for b in bits if b and b != "—") + f"  ·  as of {asof} ({src}).")
    peers = df[df["sector"] == row["sector"]]
    groups = _eqf_group_rows(row, peers)
    st.caption("**Sector pctl** places the value inside the stock's own GICS sector "
               f"({len(peers)} names in the current index selection) — coloured only at the "
               "tails (top/bottom 20%), direction-aware. Values at an extreme of their sector "
               "range may be worth a closer look.")
    cols2 = st.columns(2)
    for i, g in enumerate(eqfunda.GROUP_ORDER):
        rows = groups.get(g)
        if not rows:
            continue
        with cols2[i % 2]:
            st.markdown(f"**{g}**")
            tbl = pd.DataFrame([{"Metric": r["label"], "Value": r["value"],
                                 "Sector median": r["median"], "Sector pctl": r["pctl_txt"]}
                                for r in rows])
            sty = [_EQF_GOOD_CSS if r["good"] > 0 else _EQF_BAD_CSS if r["good"] < 0 else ""
                   for r in rows]
            brand.themed_dataframe(tbl, {}, colorers=[(["Sector pctl"], (lambda s: lambda col: s)(sty))],
                                   height=int(40 + 35.2 * len(rows)))

    with st.expander("📈 Metric history — builds as pulls append to the database"):
        mf = st.selectbox("Metric", [f["field"] for f in eqfunda.FIELDS
                                     if f["kind"] not in ("text", "date")],
                          format_func=lambda f: eqfunda.SPEC[f]["label"], key="eqf_hist_f")
        h = eqfunda.field_history(mf)
        s = h[row["ticker"]].dropna() if row["ticker"] in getattr(h, "columns", []) else pd.Series(dtype=float)
        if len(s) >= 2:
            st.line_chart(s, height=220)
        else:
            st.caption(f"{len(s)} stored point(s) for this name so far — the trend chart appears "
                       "once a couple of weekly pulls have accumulated.")

    st.divider()
    rc1, rc2 = st.columns([1, 2])
    if rc1.button("📄 Generate tearsheet PDF", key="eqf_pdf_btn", use_container_width=True):
        payload = {
            "asof": asof, "mode": src, "name": str(row["name"]), "ticker": str(row["ticker"]),
            "sector": str(row["sector"]), "region": str(row["region"]), "indices": str(row["indices"]),
            "mktcap": eqfunda.fmt_value("CRNCY_ADJ_MKT_CAP", row.get("CRNCY_ADJ_MKT_CAP")),
            "crncy": eqfunda.fmt_value("CRNCY", row.get("CRNCY")),
            "next_report": eqfunda.fmt_value("EXPECTED_REPORT_DT", row.get("EXPECTED_REPORT_DT")),
            "n_peers": int(len(peers)),
            "groups": [{"name": g, "rows": groups[g]} for g in eqfunda.GROUP_ORDER if g in groups],
        }
        with st.spinner("Building the tearsheet…"):
            try:
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "payload.json"
                    _out = Path(_t) / "Company_Fundamentals.pdf"
                    _in.write_text(json.dumps(payload), encoding="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "eqfundareport.py"), str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["eqf_pdf"] = _out.read_bytes()
                        st.session_state["eqf_pdf_name"] = (
                            "Company_Fundamentals_" + re.sub(r"\W+", "_", str(row["name"])) + ".pdf")
                    else:
                        st.error("Report failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed:\n\n{e}")
    rc2.caption("A branded one-page fundamentals tearsheet for this company — the four metric "
                "groups with sector medians and percentiles, on the house style.")
    if st.session_state.get("eqf_pdf"):
        st.download_button("⬇️  Download " + st.session_state.get("eqf_pdf_name", "Company_Fundamentals.pdf"),
                           data=st.session_state["eqf_pdf"],
                           file_name=st.session_state.get("eqf_pdf_name", "Company_Fundamentals.pdf"),
                           mime="application/pdf", key="eqf_pdf_dl")
        email_report_ui("eqf_email", "eqfunda", st.session_state["eqf_pdf"],
                        subject=f"BASIS — Company Fundamentals: {row['name']}",
                        attachment_name=st.session_state.get("eqf_pdf_name", "Company_Fundamentals.pdf"))


def _eqf_peers(df: pd.DataFrame) -> None:
    d = df.sort_values("name").reset_index(drop=True)
    lab = (d["name"] + "  (" + d["ticker"] + ")").tolist()
    sel = st.multiselect("Companies (2–6)", lab, key="eqf_peer_sel", max_selections=6,
                         help="Pick a name and its peers — e.g. the sector rivals across indices.")
    if len(sel) < 2:
        st.caption("Pick at least two companies to compare side by side.")
        return
    rows_d = [d.iloc[lab.index(x)] for x in sel]
    recs, best = [], {}
    for spec in eqfunda.FIELDS:
        if spec["kind"] in ("text", "date"):
            continue
        f = spec["field"]
        vals = pd.Series([pd.to_numeric(r.get(f), errors="coerce") for r in rows_d],
                         index=sel, dtype=float)
        rec = {"Group": spec["group"], "Metric": spec["label"]}
        for x in sel:
            rec[x] = eqfunda.fmt_value(f, vals[x])
        recs.append(rec)
        if spec["better"] and vals.notna().any():
            best[spec["label"]] = vals.idxmax() if spec["better"] == "high" else vals.idxmin()
    disp = pd.DataFrame(recs)
    colorers = [([x], (lambda s: lambda col: s)(
                    [(_EQF_GOOD_CSS if best.get(r["Metric"]) == x else "") for r in recs]))
                for x in sel]
    st.caption("**Green = best of the selected group** on that metric, direction-aware; context "
               "metrics with no better/worse end (yield, payout, size) stay unmarked.")
    brand.themed_dataframe(disp, {}, colorers=colorers, height=int(40 + 35.2 * len(recs)))


def render_eq_fundamentals() -> None:
    st.subheader("🏢 Company Fundamentals")
    st.caption("The research fundamentals — valuation, profitability, leverage, growth and income "
               "— for every index constituent, always ranked **within GICS sector** so a bank's "
               "P/B is judged against banks, not software. Every pull **appends** to the "
               "fundamentals database, so trends accumulate over time.")
    _eqf_pull_note()
    c1, c2 = st.columns([1, 2])
    if c1.button("📥 Pull fundamentals", use_container_width=True, key="eqf_pull",
                 help="Force a FULL manual pull (~30 fields per constituent, needs the Terminal) "
                      "and append to the database. Rarely needed — the Equities pull refreshes "
                      "fundamentals on a cycle (core weekly · Russell 2000 monthly)."):
        st.session_state["eqf_pull_confirm"] = True
    if st.session_state.get("eqf_pull_confirm"):
        _n = len({c["ticker"] for rows in equities.cached_universe().values() for c in rows})
        st.warning(f"⚡ A full manual pull is ~{_n:,} names × ~30 fields ≈ **{_n * 30:,} Bloomberg "
                   "hits** against the daily capacity. The scheduled cycle (core weekly · "
                   "Russell 2000 monthly) usually makes this unnecessary.")
        _g1, _g2, _ = st.columns([1.4, 1, 3.6])
        if _g1.button("Pull anyway", key="eqf_pull_anyway"):
            st.session_state.pop("eqf_pull_confirm", None)
            with st.spinner("Pulling fundamentals…"):
                try:
                    res = eqfunda.refresh()
                    if res.get("ok"):
                        _eqf_frame.clear()
                        st.success(f"Fundamentals pulled — {res.get('n_tickers', 0)} names appended "
                                   f"({res.get('last_pull', '')}).")
                        st.rerun()
                    else:
                        st.error(f"Pull failed: {res.get('reason', 'unknown')}")
                except Exception as e:
                    st.error(f"Pull failed: {e}")
        if _g2.button("Cancel", key="eqf_pull_cancel"):
            st.session_state.pop("eqf_pull_confirm", None); st.rerun()
    c2.caption(f"**Database:** {eqfunda.data_status()}."
               + ("  ·  _Pulls come free from Yahoo Finance — no Terminal needed. A few "
                  "Bloomberg-only fields (ROIC, interest cover, 3Y-avg revenue growth) show "
                  "'—' on Yahoo pulls._" if equities._use_yf()
                  else "" if MODE == "bloomberg"
                  else "  ·  _Off-Terminal a pull writes synthetic demo rows — live values need "
                       "Bloomberg mode on the work PC._"))
    _keys = list(equities.INDICES.keys())
    sel = st.multiselect("Indices", _keys, default=list(equities.DEFAULT_INDICES), key="eqf_idx",
                         help="Scope the screener / tearsheet / peers to these indices. "
                              "Russell 2000 (~2000 names) is opt-in — add it here when needed.")
    with st.spinner("Loading the fundamentals database…"):
        df, asof, src = _eqf_frame(tuple(sel or _keys))
    if df.empty:
        st.caption("No universe loaded — pull equities data first (Equities Home).")
        return
    t1, t2, t3 = st.tabs(["🔎 Screener", "📇 Company tearsheet", "⚖️ Peer comparison"])
    with t1:
        _eqf_screener(df)
    with t2:
        _eqf_tearsheet(df, asof, src)
    with t3:
        _eqf_peers(df)


# ── Earnings Calendar (Equities) ──────────────────────────────────────────────
def _ecal_shift(delta):
    y, m = st.session_state.get("ecal_ym", (0, 0))
    m += delta
    if m < 1:
        m, y = 12, y - 1
    elif m > 12:
        m, y = 1, y + 1
    st.session_state["ecal_ym"] = (y, m)


def _ecal_today():
    t = datetime.now(ZoneInfo("America/New_York")).date()
    st.session_state["ecal_ym"] = (t.year, t.month)


def render_eq_earnings() -> None:
    """Month-grid of every constituent's next expected earnings date — the Equities twin of
    the FICC fundamental-reports calendar (same repcal grid, chips coloured by GICS sector)."""
    from src import eqearncal, repcal
    import calendar as _cmod
    st.subheader("📅 Earnings calendar")
    today = datetime.now(ZoneInfo("America/New_York")).date()
    _keys = list(equities.INDICES.keys())
    fc1, fc2 = st.columns([3, 3])
    sel = fc1.multiselect("Indices", _keys, default=_keys, key="ecal_idx",
                          help="Scope the calendar to these indices.")
    with st.spinner("Loading the fundamentals database…"):
        df, asof, src = _eqf_frame(tuple(sel or _keys))
    if df.empty:
        st.caption("No universe loaded — pull equities data first (Equities Home).")
        return
    sectors = sorted(df["sector"].dropna().unique())
    sec_sel = fc2.multiselect("Sectors", sectors, key="ecal_sectors", help="Blank = all sectors.")
    dff = df[df["sector"].isin(sec_sel)] if sec_sel else df
    dff, _q = prodsearch.search_row_box(dff, ["name", "ticker", "sector", "indices"], key="ecal_search")
    if _q and dff.empty:
        st.info(prodsearch.NO_MATCH_STOCK.format(q=_q))
        return
    n_dated = int(pd.to_datetime(dff.get("EXPECTED_REPORT_DT"), errors="coerce").notna().sum())
    st.caption(f"**{n_dated}** of {len(dff)} companies in the selection have a Bloomberg expected "
               f"report date (from the {asof} {src} pull). Each date is the company's **next** "
               "expected report, so the grid fills roughly one quarter ahead and rolls forward "
               "with the fundamentals pull cycle. Busy days show the biggest names by market "
               "cap plus a **⋯ more** chip — hover any chip for the details.")
    _eqf_pull_note()

    # ----- month navigation: Today / ‹ / › / Month Year (same pattern as the FICC calendar) -----
    st.session_state.setdefault("ecal_ym", (today.year, today.month))
    n1, n2, n3, n4 = st.columns([1.1, 0.7, 0.7, 6])
    n1.button("Today", key="ecal_today_btn", on_click=_ecal_today, use_container_width=True)
    n2.button("‹", key="ecal_prev", on_click=_ecal_shift, args=(-1,), use_container_width=True)
    n3.button("›", key="ecal_next", on_click=_ecal_shift, args=(1,), use_container_width=True)
    cy, cm = st.session_state["ecal_ym"]
    n4.markdown(f"<div style='font-size:21px;font-weight:700;padding-top:2px'>{_cmod.month_name[cm]} {cy}</div>",
                unsafe_allow_html=True)

    st.markdown(repcal.month_html(eqearncal.events(dff), cy, cm, today), unsafe_allow_html=True)
    st.markdown(eqearncal.legend_html(), unsafe_allow_html=True)

    with st.expander("📋 List view — upcoming earnings, soonest first"):
        d = dff.copy()
        d["_dt"] = pd.to_datetime(d.get("EXPECTED_REPORT_DT"), errors="coerce")
        d = d[d["_dt"].notna() & (d["_dt"].dt.date >= today)].sort_values("_dt")
        if d.empty:
            st.caption("No dated upcoming reports in the current selection.")
        else:
            brand.themed_dataframe(pd.DataFrame({
                "Date": d["_dt"].dt.strftime("%a %d %b %Y"),
                "Company": d["name"], "Ticker": d["ticker"].str.split().str[0],
                "Sector": d["sector"], "Index": d["indices"],
                "Mkt cap": [eqfunda.fmt_value("CRNCY_ADJ_MKT_CAP", v)
                            for v in d.get("CRNCY_ADJ_MKT_CAP", pd.Series(index=d.index))],
            }), {}, height=520)


# ── Client ETFs (Equities) ────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _eqetf_movers():
    from src import eqetf
    return eqetf.movers_frame()


def render_eq_etfs() -> None:
    """The desk's curated client-ETF watchlist: overnight movers + σ heatmap by group,
    and a fund tearsheet (AUM / expense / yield / beta) — all free Yahoo data."""
    from src import eqetf, heatmap_html
    st.subheader("🧺 Client ETFs")
    st.caption("The funds the desk's clients trade — overnight moves sized in σ of each "
               "fund's own ~1-month daily vol, plus the fund tearsheet. All data rides "
               "**free Yahoo Finance** (zero Bloomberg hits).")
    if st.button("🔄 Refresh ETF data", key="eqetf_refresh",
                 help="Re-pull quotes and the fund tearsheet from Yahoo (free)."):
        st.session_state["eqetf_go"] = True
    if st.session_state.pop("eqetf_go", False):
        with st.spinner("Refreshing ETF data from Yahoo…"):
            _eqetf_movers.clear()
            eqetf.fund_info(force=True)
        st.rerun()

    f = _eqetf_movers()
    if f.empty:
        st.caption("No ETF quotes available — Yahoo unreachable. Try **Refresh ETF data**.")
        return

    # ---- overnight movers, grouped, biggest |σ| first within each group ------
    disp = f.copy()
    disp["_abs"] = disp["sigma"].abs()
    disp = (disp.sort_values(["Group", "_abs"], ascending=[True, False])
                .drop(columns="_abs")
                .rename(columns={"last": "Last", "pct": "%", "sigma": "σ (1m)"}))
    _pcol = ["color:#2E9E63" if (v == v and v >= 0) else "color:#D85A4A" for v in disp["%"]]
    brand.themed_dataframe(
        disp[["ETF", "Name", "Group", "Last", "%", "σ (1m)"]],
        {"Last": lambda v: f"{v:,.2f}" if v == v else "—",
         "%": lambda v: f"{v:+.2f}%" if v == v else "—",
         "σ (1m)": lambda v: f"{v:+.2f}" if v == v else "—"},
        colorers=[(["%"], lambda col: _pcol)],
        height=int(38 + 35 * len(disp)))

    # ---- σ heatmap by group (same renderer as the FICC / index heatmaps) -----
    sections = []
    for g in eqetf.GROUPS:
        dg = f[f["Group"] == g].dropna(subset=["pct"])
        items = [(r["ETF"], float(r["pct"]),
                  float(r["sigma"]) if r["sigma"] == r["sigma"] else None)
                 for _, r in dg.iterrows()]
        if items:
            sections.append((g, [(g, items)]))
    if sections:
        height = int(min(560, max(280, 110 + 82 * len(sections))))
        components.html(heatmap_html.render_html(sections, height, sub_headers=False),
                        height=height + 6, scrolling=False)

    # ---- fund tearsheet ------------------------------------------------------
    st.subheader("Fund tearsheet")
    info = eqetf.fund_info()
    rows = []
    for root, name, group in eqetf.ETFS:
        i = info.get(root, {})
        rows.append({
            "ETF": root, "Fund": i.get("name") or name, "Category": i.get("category") or group,
            "AUM": eqetf.fmt_aum(i.get("aum")),
            "Expense": (f"{i['expense_pct']:.2f}%" if isinstance(i.get("expense_pct"), (int, float)) else "—"),
            "Yield": (f"{i['yield_pct']:.2f}%" if isinstance(i.get("yield_pct"), (int, float)) else "—"),
            "Beta 3Y": (f"{i['beta_3y']:.2f}" if isinstance(i.get("beta_3y"), (int, float)) else "—"),
            "3Y ann.": (f"{i['ret_3y'] * 100:+.1f}%" if isinstance(i.get("ret_3y"), (int, float)) else "—"),
            "Hldgs P/E": (f"{i['pe']:.1f}" if isinstance(i.get("pe"), (int, float)) else "—"),
        })
    brand.themed_dataframe(pd.DataFrame(rows), {}, height=int(38 + 35 * len(rows)))
    _age = eqetf.info_age_days()
    st.caption("Fund metrics from Yahoo Ticker.info"
               + (f" · pulled **{_age}d ago**" if _age is not None else " · not pulled yet")
               + " (auto-refreshes weekly; the button forces it). Bond-fund yields are "
                 "distribution yields; equity-fund P/E is holdings-weighted.")


# ── Single Stock Correlations (Equities) ──────────────────────────────────────
_EQC_MAX_MAP = 60          # heatmap name cap — melted matrices must stay under Altair's 5k-row limit


@st.cache_data(ttl=1800, show_spinner=False)
def _eqc_set(index_keys: tuple, p1: str, p2: str, thr: float, mode: str):
    """Stock-pair correlation set (cached so widget reruns don't re-pull history;
    `mode` keys the cache to the data source). Uses the CACHED membership for the
    same reason the fundamentals page does — a live INDX_MEMBERS re-pull here would
    sit on a dead screen in Bloomberg mode."""
    return eqcorr.compute(equities.cached_universe(), list(index_keys), p1, p2, thr)


def render_eq_correlations() -> None:
    import altair as alt

    st.subheader("🔗  Single Stock Correlations")
    st.caption(
        "Every pair of index constituents, correlated on **daily log returns** over two trailing "
        "windows. **Time period 1 is the screen**: only pairs whose correlation clears the "
        "threshold survive, and the time-period-2 and difference maps are drawn for exactly those "
        "companies — the names that traded as one, and whether they still do. A strongly negative "
        "cell on the third map is a pair whose usual lockstep has loosened; pairs at an extreme "
        "may be worth a closer look.")

    _keys = list(equities.INDICES.keys())
    sel = st.multiselect("Indices", _keys, default=list(equities.DEFAULT_INDICES), key="eqc_idx",
                         help="Stocks from these indices form the pair universe "
                              "(a stock in several indices counts once). Russell 2000 is "
                              "opt-in — ~2000 names make the all-pairs scan much slower.")
    c1, c2, c3 = st.columns([1, 1, 1.6])
    p1 = c1.selectbox("Time period 1 (screen)", eqcorr.PERIOD_ORDER,
                      index=eqcorr.PERIOD_ORDER.index(eqcorr.DEFAULT_P1), key="eqc_p1",
                      help="The trailing window the threshold screens on.")
    p2 = c2.selectbox("Time period 2 (compare)", eqcorr.PERIOD_ORDER,
                      index=eqcorr.PERIOD_ORDER.index(eqcorr.DEFAULT_P2), key="eqc_p2",
                      help="The surviving pairs are re-correlated over this window.")
    thr = c3.slider("Keep pairs with period-1 correlation ≥", 0.0, 1.0,
                    eqcorr.DEFAULT_THRESHOLD, 0.01, key="eqc_thr")

    with st.spinner("Correlating the pairs…"):
        cs = _eqc_set(tuple(sel or _keys), p1, p2, float(thr), MODE)
    if cs is None:
        st.info("Not enough stocks with price history in that selection — pull equities "
                "data first (Equities Home).")
        return
    st.caption(f"**{cs.n_universe}** stocks screened over the trailing {p1.lower()} · "
               f"**{len(cs.corr1)}** names in **{len(cs.pairs)}** pairs ≥ {thr:.2f} · "
               f"as of {cs.asof:%Y-%m-%d} · {cs.source}."
               + (f"  ·  **{len(cs.dropped)}** excluded (too little history)."
                  if cs.dropped else ""))
    if cs.pairs.empty:
        st.info(f"No pair correlates at or above {thr:.2f} over the trailing {p1.lower()} — "
                "lower the threshold (or widen the index selection).")
        return

    # heatmaps: cap at the names behind the strongest pairs so the melt stays renderable
    kept = list(cs.corr1.columns)
    if len(kept) > _EQC_MAX_MAP:
        top = []
        for _, r in cs.pairs.iterrows():
            for t in (r["a"], r["b"]):
                if t not in top:
                    top.append(t)
            if len(top) >= _EQC_MAX_MAP:
                break
        kept = [t for t in cs.corr1.columns if t in set(top)]
        st.caption(f"Maps show the **{len(kept)}** names behind the strongest pairs "
                   f"(of {len(cs.corr1)}) — the full pair list is in the table below.")
    names = cs.names
    order = [names[t] for t in kept]
    M1 = cs.corr1.loc[kept, kept].rename(index=names, columns=names)
    M2 = cs.corr2.loc[kept, kept].rename(index=names, columns=names)
    D = cs.diff.loc[kept, kept].rename(index=names, columns=names)
    hgt = max(340, 26 * len(kept))
    text_ok = len(kept) <= 16

    def _tidy(mat):
        d = mat.copy()
        d.index.name = "row"
        return d.reset_index().melt("row", var_name="col", value_name="corr").dropna(subset=["corr"])

    def _heat(tidy, title, *, domain, fmt="+.2f", extra_tips=()):
        tips = [alt.Tooltip("row:N", title=""), alt.Tooltip("col:N", title="vs"),
                alt.Tooltip("corr:Q", title=title, format=fmt), *extra_tips]
        enc_x = alt.X("col:N", sort=order, title=None,
                      axis=alt.Axis(labelAngle=-40, labelFontSize=11, orient="top", labelLimit=140))
        enc_y = alt.Y("row:N", sort=order, title=None,
                      axis=alt.Axis(labelFontSize=11, labelLimit=140))
        base = alt.Chart(tidy)
        rect = base.mark_rect(stroke=brand.palette()["canvas"], strokeWidth=2.1).encode(
            x=enc_x, y=enc_y,
            color=alt.Color("corr:Q",
                            scale=alt.Scale(scheme="redblue", domain=domain, reverse=True),
                            legend=alt.Legend(title=None, format="+.1f", gradientLength=160)),
            tooltip=tips)
        layers = [rect]
        if text_ok:
            span_ = max(abs(domain[0]), abs(domain[1]))
            layers.append(base.mark_text(fontSize=11).encode(
                x=enc_x, y=enc_y, text=alt.Text("corr:Q", format=fmt),
                color=alt.condition(f"abs(datum.corr) > {span_ * 0.55}",
                                    alt.value("#F5F5F5"), alt.value("#1A1A1A")),
                tooltip=tips))
        return alt.layer(*layers).properties(height=hgt, title=title)

    if len(kept) <= 30:                          # side-by-side while the labels still read
        h1, h2 = st.columns(2)
        with h1:
            brand.show_chart(_heat(_tidy(M1), f"Time period 1 — {p1}", domain=[-1, 1]))
        with h2:
            brand.show_chart(_heat(_tidy(M2), f"Time period 2 — {p2}", domain=[-1, 1]))
    else:
        brand.show_chart(_heat(_tidy(M1), f"Time period 1 — {p1}", domain=[-1, 1]))
        brand.show_chart(_heat(_tidy(M2), f"Time period 2 — {p2}", domain=[-1, 1]))

    dt = _tidy(D)
    dt["cp1"] = [M1.loc[r, c] for r, c in zip(dt["row"], dt["col"])]
    dt["cp2"] = [M2.loc[r, c] for r, c in zip(dt["row"], dt["col"])]
    span = float(np.ceil(dt["corr"].abs().max() * 10) / 10) if len(dt) else 0.2
    span = max(span, 0.2)
    brand.show_chart(_heat(dt, f"{p2} − {p1} — where the relationship has shifted",
                           domain=[-span, span],
                           extra_tips=(alt.Tooltip("cp1:Q", title=p1, format="+.2f"),
                                       alt.Tooltip("cp2:Q", title=p2, format="+.2f"))))

    # ---- the pairs behind the maps -------------------------------------------
    st.divider()
    st.markdown(f"**Qualifying pairs — {p1} correlation ≥ {thr:.2f}**")
    pt = cs.pairs
    _q = st.text_input("Find a stock — shows the pairs it appears in", key="eqc_search",
                       placeholder="name, ticker or sector — e.g. Apple, AAPL, Financials").strip()
    if _q:
        _tmp = pt.copy()
        _tmp["_na"] = _tmp["a"].map(lambda x: cs.names.get(x, x))
        _tmp["_nb"] = _tmp["b"].map(lambda x: cs.names.get(x, x))
        pt = prodsearch.filter_rows(_tmp, ["a", "b", "_na", "_nb", "sector_a", "sector_b"], _q)
        if pt.empty:
            st.info(prodsearch.NO_MATCH_STOCK.format(q=_q))
            return
    shown = pt.head(200)
    disp = pd.DataFrame({
        "Pair": [f"{cs.names[a]}  ↔  {cs.names[b]}" for a, b in zip(shown["a"], shown["b"])],
        "Sectors": [sa if sa == sb else f"{sa} / {sb}"
                    for sa, sb in zip(shown["sector_a"], shown["sector_b"])],
        f"Corr ({p1})": shown["c1"].values, f"Corr ({p2})": shown["c2"].values,
        "Δ (2−1)": shown["d"].values,
    })
    brand.themed_dataframe(
        disp, {f"Corr ({p1})": "{:+.2f}", f"Corr ({p2})": "{:+.2f}", "Δ (2−1)": "{:+.2f}"},
        na_rep="—", height=min(560, int(38 + 35 * len(disp))))
    st.caption(("Top 200 of " + f"{len(pt)} pairs, " if len(pt) > 200 else "")
               + f"ranked by the {p1.lower()} correlation. Δ is the period-2 minus period-1 "
                 "correlation — the pairs whose co-movement has moved furthest from the screen "
                 "window may be worth a closer look.")


# ── Index Dispersion (Equities) ───────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _eqd_set(index_key: str, hist: str, top_n: int, mode: str):
    """One index's dispersion set (cached — a live run is a weights bds + an IV
    bdh + a price bdh per index; `mode` keys the cache to the data source). Uses
    the CACHED membership for the same dead-screen reason the correlations and
    fundamentals pages do."""
    return eqdisp.compute(equities.cached_universe(), index_key, hist, top_n)


@st.cache_data(ttl=1800, show_spinner=False)
def _eqd_bt(index_key: str, hist: str, top_n: int, weighting: str, mode: str):
    """The rolled dispersion-carry backtest (cached — the 2-year window is its own
    longer IV + price pull in live mode)."""
    return eqdisp.backtest(equities.cached_universe(), index_key, hist, top_n, weighting)


def render_eq_dispersion() -> None:
    import altair as alt

    st.subheader("🎯  Index Dispersion")
    st.caption(
        "The correlation the index options market is **pricing** against the correlation the "
        "constituents are **delivering**. Implied correlation is backed out of the index 1M ATM "
        "vol vs its top members' vols (weights renormalised); realized runs the same ratio on "
        "trailing 21-day vols. A **spread at a high percentile** of its own history is the classic "
        "backdrop for short-index-vol / long-single-name-vol dispersion structures, a low "
        "percentile for the reverse — either extreme may be worth a closer look.")

    _keys = list(equities.INDICES.keys())
    sel = st.multiselect("Indices", _keys, default=list(equities.DEFAULT_INDICES), key="eqd_idx",
                         help="Each index is monitored against its own constituent basket. "
                              "Russell 2000 (~2000 names) is opt-in — add it here when needed.")
    c1, c2 = st.columns([1, 1.6])
    hist = c1.selectbox("History window", eqdisp.HISTORY_ORDER,
                        index=eqdisp.HISTORY_ORDER.index(eqdisp.DEFAULT_HISTORY), key="eqd_hist",
                        help="The correlation history the spread percentile is ranked within.")
    top_n = c2.slider("Basket — top N members by index weight", 10, 60,
                      eqdisp.DEFAULT_TOP_N, 5, key="eqd_topn",
                      help="Indices with fewer members than N use full membership. The S&P's "
                           "tail adds noise, not signal — the top names carry the vol market.")

    with st.spinner("Backing the correlations out…"):
        sets = {k: _eqd_set(k, hist, int(top_n), MODE) for k in (sel or _keys)}
    sets = {k: v for k, v in sets.items() if v is not None}
    if not sets:
        st.info("Not enough price/vol history in that selection — pull equities data first "
                "(Equities Home).")
        return

    rows = [{
        "Index": k,
        "Implied corr": float(ds.imp.iloc[-1]),
        "Realized corr": float(ds.real.iloc[-1]),
        "Spread": float(ds.spread.iloc[-1]),
        "Pctl": ds.pctl,
        "Index IV": float(ds.idx_iv.iloc[-1]) if len(ds.idx_iv) else float("nan"),
        "Index RV": float(ds.idx_rv.iloc[-1]) if len(ds.idx_rv) else float("nan"),
        "Basket": f"{len(ds.basket)} of {ds.n_full} · {ds.coverage * 100:.0f}% wgt",
        "Signal": eqdisp.flag(ds.pctl),
    } for k, ds in sets.items()]
    ov = pd.DataFrame(rows).sort_values("Pctl", ascending=False).reset_index(drop=True)
    brand.themed_dataframe(
        ov, {"Implied corr": "{:.2f}", "Realized corr": "{:.2f}", "Spread": "{:+.2f}",
             "Pctl": "{:.0f}", "Index IV": "{:.1f}", "Index RV": "{:.1f}"},
        na_rep="—", height=int(38 + 35 * len(ov)))
    any_ds = next(iter(sets.values()))
    st.caption(f"1M ATM implied vs trailing-21-day realized · spread percentile within the "
               f"trailing {hist.lower()} · as of {any_ds.asof:%Y-%m-%d} · {any_ds.source} · "
               f"{any_ds.weight_source}. Weights are today's basket applied through history.")
    if any("proxy" in s.weight_source for s in sets.values()):
        st.caption("⚠️ **Index weights licence** — real Bloomberg member weights (`INDX_MWEIGHT`) "
                   "are only entitled here for the FTSE 100 and Dow Jones; the other indices run "
                   "on a market-cap proxy (dual share classes split, free float ignored). Full "
                   "weightings need the index provider's equity data licence (S&P DJI / NASDAQ / "
                   "STOXX / Euronext via the Bloomberg rep) — once entitled, this page picks up "
                   "the real weights automatically.")

    # ranked percentile bars — where each index sits in its own spread history
    cc = brand.chart_colors()
    bar_df = ov[["Index", "Pctl", "Spread", "Signal"]].copy()
    bar_df["flag"] = ["Rich" if p >= eqdisp.RICH_PCTL else "Cheap" if p <= eqdisp.CHEAP_PCTL
                      else "Neutral" for p in bar_df["Pctl"]]
    color = alt.Color("flag:N", scale=alt.Scale(domain=["Rich", "Cheap", "Neutral"],
                      range=[cc["short"], cc["long"], cc["muted"]]), legend=None)
    bars = alt.Chart(bar_df).mark_bar().encode(
        x=alt.X("Pctl:Q", title=f"implied − realized correlation · percentile vs {hist}",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Index:N", title=None, sort=bar_df["Index"].tolist()), color=color,
        tooltip=[alt.Tooltip("Index:N"), alt.Tooltip("Spread:Q", format="+.2f"),
                 alt.Tooltip("Pctl:Q", format=".0f"), alt.Tooltip("Signal:N")])
    rules = alt.Chart(pd.DataFrame({"v": [eqdisp.CHEAP_PCTL, eqdisp.RICH_PCTL]})).mark_rule(
        color=cc["muted"], strokeDash=[3, 3]).encode(x="v:Q")
    brand.show_chart((bars + rules).properties(height=max(120, 34 * len(bar_df))))

    # ---- one index in detail --------------------------------------------------
    st.divider()
    pick = st.selectbox("Index detail", list(sets.keys()), key="eqd_pick")
    ds = sets[pick]

    long = pd.concat([
        pd.DataFrame({"date": ds.imp.index, "corr": ds.imp.values, "kind": "Implied (1M ATM)"}),
        pd.DataFrame({"date": ds.real.index, "corr": ds.real.values, "kind": "Realized (21d)"}),
    ])
    lines = alt.Chart(long).mark_line(strokeWidth=2.1).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("corr:Q", title="correlation", scale=alt.Scale(zero=False)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Implied (1M ATM)", "Realized (21d)"],
                        range=[cc["series"], cc["ink"]]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("kind:N", title=""),
                 alt.Tooltip("corr:Q", format=".2f")])
    brand.show_chart(lines.properties(height=300, title=f"{pick} — implied vs realized correlation"))

    sp = pd.DataFrame({"date": ds.spread.index, "spread": ds.spread.values})
    scol = cc["short"] if ds.pctl >= eqdisp.RICH_PCTL else \
        cc["long"] if ds.pctl <= eqdisp.CHEAP_PCTL else cc["series"]
    sp_area = alt.Chart(sp).mark_area(opacity=0.22, color=scol).encode(
        x=alt.X("date:T", title=None), y=alt.Y("spread:Q", title="implied − realized"))
    sp_line = alt.Chart(sp).mark_line(color=scol, strokeWidth=2.1).encode(
        x="date:T", y="spread:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("spread:Q", format="+.2f")])
    brand.show_chart((sp_area + sp_line).properties(
        height=220, title=f"{pick} — the dispersion spread · now {ds.pctl:.0f}th pctl"))

    st.markdown(f"**The basket — top {len(ds.basket)} of {ds.n_full} members, "
                f"{ds.coverage * 100:.0f}% of index weight**")
    bt = ds.basket
    disp = pd.DataFrame({
        "Name": bt["name"], "Sector": bt["sector"],
        "Weight": bt["weight"] * 100.0, "IV (1M)": bt["iv"], "RV (21d)": bt["rv"],
        "IV − RV": bt["prem"],
    })
    brand.themed_dataframe(
        disp, {"Weight": "{:.1f}%", "IV (1M)": "{:.1f}", "RV (21d)": "{:.1f}", "IV − RV": "{:+.1f}"},
        na_rep="—", height=min(560, int(38 + 35 * len(disp))))
    st.caption("Weights renormalised over the kept basket. The single names whose IV sits "
               "furthest below their own realized are the natural long-vol legs of a dispersion "
               "basket; names at an extreme may be worth a closer look."
               + (f"  ·  **{len(ds.dropped)}** members excluded (no prices / no listed vol)."
                  if ds.dropped else ""))

    # ---- Phase 2: basket builder ---------------------------------------------
    st.divider()
    st.markdown(f"**🧺 Basket builder — {pick}**")
    default_dir = "reverse" if ds.pctl <= eqdisp.CHEAP_PCTL else "classic"
    b1, b2, b3 = st.columns([1.9, 1, 1.2])
    dir_lbl = b1.selectbox("Structure", list(eqdisp.DIRECTIONS.values()),
                           index=list(eqdisp.DIRECTIONS).index(default_dir), key="eqd_dir",
                           help="Defaults to the side the flag points at: a rich spread pairs "
                                "with the classic short-index / long-single-name structure, a "
                                "cheap one with the reverse.")
    direction = next(k for k, v in eqdisp.DIRECTIONS.items() if v == dir_lbl)
    vega = b2.number_input("Vega per leg", 100.0, 100000.0, 1000.0, 100.0, key="eqd_vega",
                           help="Local currency per vol point on each side. The single-name "
                                "vega is split across the basket and matched 1:1 by index vega.")
    sch_lbl = b3.selectbox("Vega split", list(eqdisp.VEGA_SCHEMES.values()), key="eqd_scheme",
                           help="Cap-weighted mirrors the index; equal-vega maximises the "
                                "idiosyncratic (dispersion) exposure per name.")
    scheme = next(k for k, v in eqdisp.VEGA_SCHEMES.items() if v == sch_lbl)

    bk = eqdisp.build_basket(ds, float(vega), scheme, direction)
    if bk is None:
        st.info("No usable vol/spot marks to size a basket from.")
    else:
        il = bk.index_leg
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net vega", f"{bk.legs['vega'].sum() + il['vega']:+,.0f}",
                  help="Zero by construction — the structure is a pure correlation position "
                       "at entry.")
        m2.metric("Net theta / day", f"{bk.net_theta:+,.0f}")
        m3.metric("Net premium", f"{bk.net_premium:+,.0f}")
        m4.metric("Earnings inside tenor", f"{bk.n_earnings} names")
        legs = bk.legs
        bdisp = pd.DataFrame({
            "Leg": [f"📌 {pick} (index)"] + list(legs["name"]),
            "Sector": ["—"] + list(legs["sector"]),
            "IV (1M)": [il["iv"]] + list(legs["iv"]),
            "Spot": [il["spot"]] + list(legs["spot"]),
            "Vega": [il["vega"]] + list(legs["vega"]),
            "Straddles": [il["contracts"]] + list(legs["contracts"]),
            "Premium": [il["premium"]] + list(legs["premium"]),
            "Theta/day": [il["theta"]] + list(legs["theta"]),
            "Earnings ≤30d": [""] + list(legs["earnings"]),
        })
        brand.themed_dataframe(
            bdisp, {"IV (1M)": "{:.1f}", "Spot": "{:,.2f}", "Vega": "{:+,.0f}",
                    "Straddles": "{:+,.1f}", "Premium": "{:+,.0f}", "Theta/day": "{:+,.0f}"},
            na_rep="—", height=min(560, int(38 + 35 * len(bdisp))))
        st.caption(
            "ATM straddles at the 1M tenor (Black-Scholes, r=0), vega-neutral at entry — "
            "positive = long. Contracts use listed multipliers (single names ×100, "
            f"{pick} index options ×{il['mult']:g}); premium and theta in local currency. "
            "Names reporting inside the tenor carry their event premium in the long-vol legs — "
            "worth weighing in the selection. Indicative sizing off the monitor's marks, "
            "not an order ticket.")

    # ---- Phase 3: rolled carry backtest ----------------------------------------
    st.divider()
    st.markdown(f"**🧪 Dispersion carry backtest — {pick}**")
    t1, t2 = st.columns([1, 1.2])
    bt_hist = t1.selectbox("Backtest window", ["1 year", "2 years"], index=1, key="eqd_bt_hist")
    bt_lbl = t2.selectbox("Vega split", list(eqdisp.VEGA_SCHEMES.values()), key="eqd_bt_scheme")
    bt_scheme = next(k for k, v in eqdisp.VEGA_SCHEMES.items() if v == bt_lbl)
    with st.spinner("Rolling the structure through history…"):
        bt = _eqd_bt(pick, bt_hist, int(top_n), bt_scheme, MODE)
    if bt is None:
        st.info("Not enough vol history to backtest this index yet.")
        return
    sx = bt.stats
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trades", f"{sx['n']}")
    k2.metric("Total P&L", f"{sx['total'] * 1000:+,.0f}", help="Per 1,000 vega per leg.")
    k3.metric("Hit rate", f"{sx['hit']:.0f}%")
    k4.metric("Worst trade", f"{sx['worst'] * 1000:+,.0f}")
    k5.metric("Pctl → P&L corr", "—" if sx["pctl_ic"] != sx["pctl_ic"] else f"{sx['pctl_ic']:+.2f}",
              help="Correlation between the entry spread percentile and the trade's P&L: "
                   "negative says richer entries paid the CLASSIC structure less (i.e. the "
                   "flag ordering carried information for the reverse side too).")

    cv = pd.DataFrame({"date": bt.curve.index, "pnl": bt.curve.values * 1000})
    ccol = cc["long"] if cv["pnl"].iloc[-1] >= 0 else cc["short"]
    cv_area = alt.Chart(cv).mark_area(opacity=0.2, color=ccol).encode(
        x=alt.X("date:T", title=None), y=alt.Y("pnl:Q", title="cumulative P&L · per 1k vega/leg"))
    cv_line = alt.Chart(cv).mark_line(color=ccol, strokeWidth=2.1, interpolate="step-after").encode(
        x="date:T", y="pnl:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("pnl:Q", format="+,.0f")])
    brand.show_chart((cv_area + cv_line).properties(
        height=240, title=f"{pick} — classic structure (long single-name vol / short index vol), "
                          f"rolled every {bt.roll} sessions"))

    sc = bt.trades.dropna(subset=["pctl"]).copy()
    if len(sc) >= 8:
        sc["pnl_k"] = sc["pnl"] * 1000
        pts = alt.Chart(sc).mark_circle(size=90, stroke="white", strokeWidth=0.5).encode(
            x=alt.X("pctl:Q", title="entry spread percentile", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("pnl_k:Q", title="trade P&L · per 1k vega"),
            color=alt.condition("datum.pnl_k > 0", alt.value(cc["long"]), alt.value(cc["short"])),
            tooltip=[alt.Tooltip("entry:T"), alt.Tooltip("pctl:Q", format=".0f"),
                     alt.Tooltip("pnl_k:Q", title="P&L", format="+,.0f")])
        zero = alt.Chart(pd.DataFrame({"v": [0.0]})).mark_rule(
            color=cc["muted"], strokeDash=[3, 3]).encode(y="v:Q")
        brand.show_chart((pts + zero).properties(
            height=240, title="Did the flag carry information? Entry percentile vs outcome"))

    tt = bt.trades
    tdisp = pd.DataFrame({
        "Entry": tt["entry"].dt.strftime("%Y-%m-%d"), "Exit": tt["exit"].dt.strftime("%Y-%m-%d"),
        "Names": tt["n_names"], "Imp corr": tt["imp_corr"], "Fwd real corr": tt["fwd_real_corr"],
        "Entry pctl": tt["pctl"], "Names leg": tt["pnl_names"] * 1000,
        "Index leg": tt["pnl_index"] * 1000, "P&L": tt["pnl"] * 1000,
    })
    brand.themed_dataframe(
        tdisp, {"Imp corr": "{:.2f}", "Fwd real corr": "{:.2f}", "Entry pctl": "{:.0f}",
                "Names leg": "{:+,.0f}", "Index leg": "{:+,.0f}", "P&L": "{:+,.0f}"},
        na_rep="—", height=min(490, int(38 + 35 * len(tdisp))))
    st.caption(
        "Vega-carry approximation: each leg is booked as vega × (realized vol over the hold − "
        "implied at entry) — the delta-hedged straddle P&L to first order, with no gamma "
        "path-dependency or transaction costs. P&L per 1,000 vega per leg, local currency; the "
        "reverse structure is the exact mirror. A period where the classic side persistently "
        "lost is a correlation regime worth understanding, not in itself an argument for "
        "either side.")


def render_morning_coffee() -> None:
    st.subheader("☕ Morning Coffee")
    st.caption("The daily global-macro briefing — pulls Bloomberg + the news, writes the "
               "commentary, emails the desk, and then opens here in English with the heatmap.")
    if st.button("☕  Generate, email & open the report", type="primary",
                 use_container_width=True, key="run_mc"):
        with st.spinner("Pulling Bloomberg, reading the news, writing the macro commentary "
                        "and emailing the report… (~1–2 min)"):
            run_morning_coffee()

    if "mc_ok" not in st.session_state:
        st.info("Tap the button to generate today's report. It emails the desk, then opens "
                "here in English with the heatmap.")
        return
    if not st.session_state["mc_ok"]:
        st.error("Morning Coffee run failed — the log below has the actual error. "
                 "(NOT necessarily Bloomberg: prices fall back to the morning snapshot "
                 "Terminal-closed — news, Gmail, the AI commentary or the email send "
                 "can each fail independently.)")
        with st.expander("Run log", expanded=True):
            st.code(st.session_state.get("mc_log", ""), language="text")
        return

    st.success("Generated and emailed to the desk. ☕")
    side = _mc_sidecar()
    if not _mc_native_heatmap(side):          # app-native treemap from the report's own moves…
        heat = _mc_heatmap_path()             # …falling back to the report PNG for older runs
        if heat.exists():
            st.image(str(heat), use_container_width=True)
    commentary = ((side or {}).get("commentary_en")
                  or _mc_commentary(st.session_state.get("mc_log", "")))
    if commentary:
        st.markdown("#### Market Commentary")
        for para in commentary.split("\n\n"):
            if para.strip():
                st.markdown(_md_money(para.strip()))
    else:
        st.caption("(Couldn't read the English text from this run — the .docx download below "
                   "has the full report.)")
    _news = (side or {}).get("headlines") or []
    if _news:
        st.markdown("#### Market news")
        for _h in _news[:20]:
            _t = str(_h.get("title", "")).strip()
            if not _t:
                continue
            _u = str(_h.get("url", "")).strip()
            _src = str(_h.get("source", "")).strip()
            st.markdown((f"- [{_md_money(_t)}]({_u})" if _u else f"- {_md_money(_t)}")
                        + (f" — *{_md_money(_src)}*" if _src else ""))
    if st.session_state.get("mc_docx"):
        st.download_button("⬇️  Download the report (.docx)", data=st.session_state["mc_docx"],
                           file_name=st.session_state.get("mc_docx_name", "Morning_Coffee.docx"),
                           key="mc_docx_dl",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with st.expander("Run log"):
        st.code(st.session_state.get("mc_log", ""), language="text")


# --- overview pages: cross-strategy confluence + data health ----------------
_STRAT_SHORT = {
    "Mean Reversion": "MeanRev", "Trend": "Trend", "MA Crossover": "MA×",
    "MA Swing": "MA∿", "Flag Breakout": "Flag", "Support & Resistance": "S/R",
    "Fibonacci Retracement": "Fib", "Breakout & Retest": "Retest",
    "Momentum (RSI/MACD)": "Mom", "Bollinger Squeeze": "BBands", "Elliott Wave": "Elliott",
    "Ichimoku Cloud": "Ichimoku", "On-Balance Volume": "OBV", "Money Flow Index": "MFI",
    "Volatility": "Vol", "Skew Volatility": "Skew",
    "Vol Term Structure": "Term", "COT Reports": "COT", "Put/Call Ratios": "P/C",
    "AG Fundamentals": "AG",
}


def _norm_mkt(m) -> str:
    """Strip the ' · Sector' suffix some strategies append → base instrument name."""
    return str(m).split(" · ")[0].strip()


def render_confluence() -> None:
    st.subheader("\U0001F3AF Confluence")
    st.caption("Instruments flagged by several strategies today — where the signals stack up. "
               "Grouped by the underlying contract; pick one below for the full breakdown.")
    df, _meta = load_signals()
    fl = df[df["signal"].ne("—")].copy() if (df is not None and "signal" in df) else None
    fl = _filter_signals(fl)
    if fl is None or fl.empty:
        st.info("No flagged signals yet — pull a snapshot or re-run signals on Home.")
        return
    fl["key"] = fl["instruments"].astype(str)
    fl["name"] = fl["market"].map(_norm_mkt)
    n_strats = int(df["strategy"].nunique())
    minc = st.slider("Show instruments flagged by at least…", 2, min(10, n_strats), 3,
                     help=f"{n_strats} strategies ran today — raise this to see only the strongest pile-ups.")
    _VOL = {"Volatility", "Skew Volatility", "Vol Term Structure"}

    rows = []
    for key, sub in fl.groupby("key"):
        strats = list(dict.fromkeys(sub["strategy"].tolist()))   # unique, order-preserving
        if len(strats) < minc:
            continue
        nm = sub["name"].mode()
        nm = nm.iat[0] if not nm.empty else sub["name"].iat[0]
        sector = "pair" if " / " in key else (INSTRUMENTS.get(key, (key, 0.0, "", ""))[2] or "—")
        dirs = sub[~sub["strategy"].isin(_VOL)]["direction"].fillna(0)
        nb, ns = int((dirs > 0).sum()), int((dirs < 0).sum())
        lean = f"▲{nb} ▼{ns}" if (nb or ns) else "—"
        rows.append({"Market": nm, "Sector": sector, "# Strats": len(strats), "Lean": lean,
                     "Flagged by": ", ".join(_STRAT_SHORT.get(s, s) for s in strats)})
    if not rows:
        st.info(f"Nothing is flagged by {minc}+ strategies right now — lower the threshold above.")
        return
    conf = pd.DataFrame(rows).sort_values(["# Strats", "Market"], ascending=[False, True])
    _q = st.text_input("Find a product", key="conf_search", placeholder=prodsearch.PLACEHOLDER).strip()
    if _q:
        conf = prodsearch.filter_frame(conf, INSTRUMENTS, _q, name_col="Market")
        if conf.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q))
            return
    st.caption(f"**{len(conf)}** instruments flagged by **{minc}+** of {n_strats} strategies. "
               "**Lean** = directional signals leaning long (▲) vs short (▼), excluding the vol strategies.")
    brand.themed_dataframe(conf, {})

    st.markdown("##### Inspect an instrument")
    pick = st.selectbox("Instrument", conf["Market"].tolist(), key="conf_pick",
                        label_visibility="collapsed")
    det = fl[fl["name"] == pick]
    det_tbl = det[["strategy", "signal", "metric_label", "context"]].rename(
        columns={"strategy": "Strategy", "signal": "Signal",
                 "metric_label": "Metric", "context": "Notes"})
    brand.themed_dataframe(det_tbl, {})
    _uniq = list(dict.fromkeys(det["strategy"].tolist()))
    st.caption("Open the strategy:")
    jcols = st.columns(min(len(_uniq), 5) or 1)
    for i, s in enumerate(_uniq):
        jcols[i % len(jcols)].button(_STRAT_SHORT.get(s, s), key=f"conf_go_{s}",
                                     use_container_width=True, on_click=_go, args=(s,))


def _ax(tk) -> str:
    """Y-axis title for a technical chart — fixed income is charted as YIELDS, not price."""
    return "Yield (%)" if universe.is_fixed_income(str(tk)) else "Price"


def _ta_quicknav(current: str | None = None, eq: bool = False) -> None:
    """Quick-switch buttons for the technical strategies — the same 2×5 set as the Technical
    Analysis hub. Shown on the hub and at the top of each technical-strategy page so the user
    can flip between them without the sidebar; the current page's button is highlighted. `eq`
    routes to the Equities per-strategy pages (`eq:<strategy>`) instead of the FICC ones."""
    cols = st.columns(5)
    for i, s in enumerate(tascore.TA_STRATEGIES):
        dest = f"eq:{s}" if eq else s
        cols[i % 5].button(
            _STRAT_SHORT.get(s, s), key=f"tanav_{'eq_' if eq else ''}{current or 'hub'}_{s}",
            use_container_width=True, type="primary" if s == current else "secondary",
            on_click=_go, args=(dest,))


def _ta_conviction_expander() -> None:
    """The shared "How Conviction & Score are calculated" explainer — identical on the FICC and
    Equities TA hubs (the maths is universe-agnostic)."""
    with st.expander("ℹ️  How “Conviction” and “Score” are calculated"):
        st.markdown(
            "Every strategy speaks its own language — a z-score, a 0–100 readiness/proximity, a momentum "
            "score, a return %, an MA-gap %. To rank products across all of them, each flagged signal is "
            "put on one common scale, then aggregated per product in **three steps**.\n\n"
            "**1 · Each flagged signal → a _strength_ (0–100).** How far the metric sits toward "
            "“full conviction”:\n\n"
            "> `strength = min(100, |metric| ÷ full-scale × 100)`\n\n"
            "where **full-scale** (the metric magnitude that scores 100) is:\n\n"
            "| Strategy | Native metric | = 100 at |\n"
            "|---|---|---|\n"
            "| Mean Reversion | \\|z-score\\| | 3.0 |\n"
            "| Trend | \\|3-month return\\| | 25% |\n"
            "| MA Crossover / MA Swing | \\|MA gap\\| | 10% |\n"
            "| Flag Breakout · S&R · Fibonacci · Breakout & Retest · Momentum · Bollinger Squeeze · "
            "Elliott Wave · Ichimoku · OBV · MFI | already 0–100 (readiness / proximity / momentum / "
            "squeeze / wave fit / Ichimoku / volume flow) | used as-is |\n\n"
            "**2 · Conviction (0–100) = the _average_ strength** of the strategies flagging that product — "
            "how strong the signals are on average, *regardless of how many* agree.\n\n"
            "**3 · Score = the _signed sum_ of those strengths** — long signals count **＋**, short **－** — "
            "displayed as **|Score|**:\n\n"
            "> `Score = | Σ (±strength) |`\n\n"
            "So Score rewards **both** confluence (more agreeing strategies) **and** strength, while "
            "opposing calls partly cancel. The **sign** of that sum sets the **Net** column "
            "(▲ long / ▼ short, or ⚠ *mixed* when both sides fire), and **|Score| ranks the table**.\n\n"
            "**Confluence set.** Only a curated, *independent* subset feeds this score — by default "
            "**Trend, Momentum (RSI/MACD), OBV, Support & Resistance and Flag Breakout**, one per axis "
            "(direction / momentum / volume / location / pattern) — so agreement is real corroboration, "
            "not the same read echoed. Edit it under **🎯 Confluence set** above; every other strategy "
            "keeps its page and chart overlays but stays out of the score. If you tick more than one "
            "method in the **same axis**, they're de-duplicated (strongest full, the next at **½**, "
            "**⅓**, …), so a single dimension can't vote twice.\n\n"
            "**Worked example.** Three strategies flag a product **Long** at strengths 90 / 80 / 70 and one "
            "flags it **Short** at 60 → Conviction = (90+80+70+60) ÷ 4 = **75**; "
            "Score = |＋90＋80＋70－60| = **180** (the short partly cancels); Net = **▲ long**. If instead all "
            "four agreed Long, Score = |90+80+70+60| = **300** — same conviction, far higher stacked score."
        )


_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


@_fragment
def _ta_reports(meta, prod=None, scope="ficc", conf_set=None) -> None:
    """The report controls, ISOLATED in a Streamlit fragment: clicking Generate (a ~15–25s PDF
    subprocess) reruns ONLY this block, so the heavy leaderboard/gallery below don't re-execute
    and the page no longer ghosts / half-redraws while the report builds. `prod` is the scored
    product table, used to offer the pick list.

    `scope` ('ficc' | 'equities') runs the SAME controls over either book: it selects the signals
    file, the instrument universe + sector labels, and the per-book persistence (report defaults +
    exclude list), and — for equities — passes ``--equities`` so the report engine draws its charts
    from the yfinance OHLCV store. `conf_set` overrides the scored confluence set (equities passes
    its own page's selection; FICC falls back to the shared session set). Every widget key is suffixed
    by scope so the two pages' fragments never collide."""
    eq = scope == "equities"
    k = f"_{scope}"                               # per-scope widget-key suffix (no cross-page collisions)
    _pdf_key = f"conv_pdf{k}"                     # session slot for the built PDF bytes
    if eq:
        from src import eqta
        sig_file = eqta.SIGNALS_FILE
        _emeta = eqta.member_meta()
        _instruments = list(_emeta.keys())
        _name_fn = lambda t: (_emeta.get(t, {}) or {}).get("name") or t
        _sector_fn = lambda t: (_emeta.get(t, {}) or {}).get("sector") or "—"
        _report_label = "Equities Technical Analysis"
    else:
        sig_file = SIGNALS_FILE
        _instruments = INSTRUMENTS
        _name_fn = universe.name
        _sector_fn = lambda t: universe.asset(t) or "—"
        _report_label = "Technical Analysis"
    # --- Technical Analysis report (merged: conviction leaderboard + the curated best-ideas screen) ---
    cc1, cc2 = st.columns([1, 3])
    _rd = ta_report_defaults(scope)               # saved build settings; also drive the weekly email
    _MODE_IX = {"per_side": 0, "overall": 1, "threshold": 2}
    _conv_mode_lbl = cc1.radio("Selection",
                               ["Balanced — N per side", "Strongest overall — top N",
                                "Quality bar — min conviction & score"],
                               key=f"conv_mode{k}", index=_MODE_IX.get(_rd["mode"], 1),
                               help="Balanced: the top N constructive AND the top N cautious. "
                                    "Strongest overall: the top N by conviction regardless of side. "
                                    "Quality bar: only setups clearing an ABSOLUTE bar — so a quiet "
                                    "week reports as quiet instead of padding out N weak charts.")
    _conv_mode = ("threshold" if "Quality" in _conv_mode_lbl
                  else "overall" if "overall" in _conv_mode_lbl else "per_side")
    _conv_top = cc1.number_input("How many", 3, 12, int(_rd["top_n"]), key=f"conv_top{k}",
                                 help="Balanced mode: this many on EACH side. Strongest-overall: "
                                      "this many in TOTAL. Quality bar: an upper CAP on how many "
                                      "qualifying setups get written up.")
    _min_conv = _min_score = 0.0
    if _conv_mode == "threshold":
        _min_conv = cc1.number_input(
            "Min conviction", 0, 100, int(_rd["min_conviction"]), 5, key=f"conv_minconv{k}",
            help="Average strength of the flagging strategies (0–100). Filters out setups that are "
                 "broad but individually weak.")
        _min_score = cc1.number_input(
            "Min |score|", 0, 600, int(_rd["min_score"]), 10, key=f"conv_minscore{k}",
            help="Score = conviction × how many strategies agree, so this is effectively a BREADTH "
                 "floor on top of conviction. With a 5-strategy set, ~150 ≈ two strong agreeing "
                 "reads; 200+ demands three or more.")
    _conv_ai = cc1.checkbox("✨ AI-polish the write-ups", key=f"conv_ai{k}", value=bool(_rd["ai_polish"]),
                            help="Rewrite each chart note in a conversational desk-analyst voice via "
                                 "Claude (numbers & levels kept exact, neutral tone); falls back to "
                                 "the plain template if the model isn't reachable. Adds ~30–90s.")
    # One place to fix the report's build settings — and the WEEKLY email obeys the same saved values.
    if cc1.button("📌 Set as default", key=f"conv_defaults_save{k}",
                  help="Save this Selection / How many / AI-polish (and the quality bar) as the "
                       "startup default — the weekly emailed report runs on exactly these."):
        save_ta_report_defaults(scope, mode=_conv_mode, top_n=int(_conv_top), ai_polish=bool(_conv_ai),
                                min_conviction=float(_min_conv), min_score=float(_min_score))
        st.toast("Report defaults saved — the weekly email will use these.", icon="📌")
    cc1.caption(f"📌 Default: **{_rd['mode'].replace('_', ' ')}**, **{int(_rd['top_n'])}** picks, "
                f"AI-polish **{'on' if _rd['ai_polish'] else 'off'}**"
                + (f", bar **{_rd['min_conviction']:g}/{_rd['min_score']:g}**"
                   if _rd["mode"] == "threshold" else ""))
    # --- products the CLIENT rarely trades: out of this report, live everywhere else in BASIS ---
    with st.expander("🚫 Products excluded from the client report", expanded=False):
        _excl = st.multiselect(
            "Held out of the report entirely (picks, leaderboard, summary and watchlist)",
            options=sorted(_instruments, key=lambda t: str(_name_fn(t))),
            default=[t for t in universe.report_excluded(scope) if t in set(_instruments)],
            format_func=lambda t: str(_name_fn(t)), key=f"rep_excl{k}",
            help="For markets your clients don't trade. They stay FULLY live everywhere else — the "
                 "universe, every strategy page, the hub scoring and the other reports. (This is not "
                 "the Home ‘Sectors & products’ filter, which switches a market off app-wide.)")
        _rx1, _rx2 = st.columns([1, 2.4])
        if _rx1.button("💾 Save as default", key=f"rep_excl_save{k}",
                       help="Persist this list — used by the weekly emailed report and on every launch."):
            universe.save_report_excluded(_excl, scope)
            st.toast(f"{len(_excl)} product(s) excluded from the report.", icon="🚫")
        _rx2.caption("**Generate report** uses whatever's selected here; **Save** also applies it to "
                     "the weekly email and future launches.")

    # --- which picks get written up: defaults to the strongest N, but the desk can adjust ---
    _picks = None
    if prod is not None and not getattr(prod, "empty", True):
        _cand = prod[(prod["n"] >= 2) & (~prod["conflict"])
                     & (~prod["instruments"].isin(set(_excl)))]
        if not _cand.empty:
            if _conv_mode == "threshold":
                _q = _cand[(_cand["conviction"] >= float(_min_conv))
                           & (_cand["score"].abs() >= float(_min_score))]
                _dflt = set(_q.head(int(_conv_top))["instruments"])
            elif _conv_mode == "overall":
                _dflt = set(_cand.head(int(_conv_top))["instruments"])
            else:
                _dflt = (set(_cand[_cand["net_dir"] > 0].head(int(_conv_top))["instruments"])
                         | set(_cand[_cand["net_dir"] < 0].head(int(_conv_top))["instruments"]))
            with st.expander(f"✅ Picks written up — {len(_dflt)} of {len(_cand)} candidates "
                             "(default = the strongest; untick / add as you like)", expanded=False):
                _view = pd.DataFrame({
                    "Include": [t in _dflt for t in _cand["instruments"]],
                    "Market": _cand["market"].tolist(),
                    "Sector": [_sector_fn(t) for t in _cand["instruments"]],
                    "#": _cand["n"].tolist(),
                    "Net": ["▲ constructive" if d > 0 else "▼ cautious" for d in _cand["net_dir"]],
                    "Conviction": [round(float(c)) for c in _cand["conviction"]],
                    "Score": [round(abs(float(s))) for s in _cand["score"]],
                })
                # key includes mode/N so changing them re-defaults; manual ticks persist otherwise
                _ed = st.data_editor(
                    _view, hide_index=True, use_container_width=True,
                    disabled=[c for c in _view.columns if c != "Include"],
                    key=f"ta_picks{k}_{_conv_mode}_{int(_conv_top)}_{len(_cand)}")
                _picks = [t for t, keep in zip(_cand["instruments"], _ed["Include"]) if keep]
                st.caption(f"**{len(_picks)}** pick(s) will be written up, ranked by |Score|. "
                           "Leave the defaults for the automatic strongest-N behaviour.")

    _fname = _report_label.replace(" ", "_") + "_Report.pdf"
    if cc1.button(f"📈 Generate {_report_label} Report (PDF)", type="primary",
                  key=f"conv_gen{k}", disabled=not sig_file.exists()):
        with st.spinner("Selecting the setups and drawing each chart…"):
            with tempfile.TemporaryDirectory() as tmp:
                out_pdf = Path(tmp) / "ta.pdf"
                cmd = [sys.executable, str(CONVREPORT_CLI), str(sig_file), str(out_pdf),
                       "--asof", str(meta.get("as_of", "")), "--top", str(int(_conv_top)),
                       "--mode", _conv_mode]
                if eq:                                       # draw charts from the yfinance OHLCV store
                    cmd.append("--equities")
                if _conv_mode == "threshold":
                    cmd += ["--min-conviction", str(float(_min_conv)),
                            "--min-score", str(float(_min_score))]
                if _conv_ai:
                    cmd.append("--ai-polish")
                _cs = list(conf_set) if conf_set else (st.session_state.get("conf_set")
                                                       or tascore.CONFLUENCE_DEFAULT)
                cmd += ["--strategies", ",".join(_cs)]       # match the on-page confluence set
                if _picks is not None:                       # the desk's curated pick list
                    cmd += ["--picks", ",".join(_picks)]
                cmd += ["--exclude", ",".join(_excl)]        # WYSIWYG even before Save
                res = subprocess.run(cmd, capture_output=True, text=True)
                ok = res.returncode == 0 and out_pdf.exists()
                st.session_state[_pdf_key] = out_pdf.read_bytes() if ok else None
        if not st.session_state.get(_pdf_key):
            st.error(f"{_report_label} report failed:\n\n" + (res.stderr or res.stdout or "no output"))
        else:
            st.success(f"{_report_label} report ready.")
    if st.session_state.get(_pdf_key):
        st.download_button(f"⬇️ Download {_fname}", data=st.session_state[_pdf_key],
                           file_name=_fname, mime="application/pdf", key=f"conv_pdf_dl{k}")
        email_report_ui(_pdf_key, "eq_convreport" if eq else "convreport", st.session_state.get(_pdf_key),
                        subject=f"{_report_label} Report", attachment_name=_fname)
    cc2.caption(f"The merged **{_report_label}** report — opens with the conviction leaderboard and the "
                "stacked-signals summary, then the strongest constructive & cautious setups by "
                "cross-strategy conviction, each with a multi-indicator chart, a plain-English read and "
                "objective / invalidation levels. Neutral, client-safe language"
                + ("; ~2,600 single-name equities off free yfinance data." if eq
                   else "; fixed income read on yields."))


@st.cache_data(show_spinner=False)
def _ta_scored(as_of, filter_key, scored_key=None):
    """Cached cross-strategy scoring for the TA overview — recomputed only when the signals
    (`as_of`), the sector filter, or the confluence set (`scored_key`) change, not on every rerun."""
    df, _ = load_signals()
    flagged = tascore.ta_flagged(_filter_signals(df), list(scored_key) if scored_key else None)
    if flagged is None or flagged.empty:
        return None, None
    flagged = flagged.copy()
    flagged["name"] = flagged["market"].map(_norm_mkt)
    flagged["dir"] = pd.to_numeric(flagged["direction"], errors="coerce").fillna(0).astype(int)
    flagged["strength"] = [tascore.strength(s, m) for s, m in zip(flagged["strategy"], flagged["metric"])]
    return flagged, tascore.score_products(flagged)


@st.cache_data(show_spinner=False)
def _ta_gallery_data(tk, strset_key, as_of):
    """Cached per-product data for the TA-overview gallery chart (the price/yield series + the
    flagging strategies' levels) so it isn't recomputed on every rerun — only when the product
    or the signals change. The heavy `*_chart_data` calls live here, behind the cache."""
    from src.strategies import (support_resistance as _sr, flag_breakout as _fb,
                                breakout_retest as _br, momentum as _mom, fibonacci as _fbn,
                                elliott_wave as _ew, ichimoku as _ich, obv as _obv, mfi as _mfi)
    strset = set(strset_key)
    out = {"pf": None, "flag": None, "sr_levels": [], "fib_levels": [], "retest_level": None,
           "mom": None, "elliott": None, "ichimoku": None, "obv": None, "mfi": None}
    try:
        out["pf"] = get_history_ta([tk])[tk].dropna()
    except Exception:
        return out
    if "Flag Breakout" in strset:
        try:
            _fcd, _fi = _fb.flag_chart_data(tk)
            if _fcd is not None and not _fcd.empty and _fi:
                out["flag"] = (_fcd[["date", "upper", "lower", "breakout"]].dropna(), _fi)
        except Exception:
            pass
    if "Support & Resistance" in strset:
        try:
            _, _isr = _sr.sr_chart_data(tk)
            out["sr_levels"] = (_isr or {}).get("levels", []) or []
        except Exception:
            pass
    if "Fibonacci Retracement" in strset:
        try:
            _, _ifb = _fbn.fib_chart_data(tk)
            out["fib_levels"] = [L for L in ((_ifb or {}).get("levels", []) or []) if L.get("key")]
        except Exception:
            pass
    if "Breakout & Retest" in strset:
        try:
            _, _ibr = _br.retest_chart_data(tk)
            out["retest_level"] = (_ibr or {}).get("level")
        except Exception:
            pass
    if "Momentum (RSI/MACD)" in strset:
        try:
            _mcd, _mi = _mom.momentum_chart_data(tk)
            if _mcd is not None and not _mcd.empty:
                out["mom"] = _mcd[["date", "rsi"]].dropna()
        except Exception:
            pass
    if "Elliott Wave" in strset and out["pf"] is not None:
        try:
            # analyse the SAME series the chart draws (get_history_ta → yields for FI), so the pivots
            # land on the chart's y-scale — passing price history would blow up the shared scale on FI.
            _, _ewi = _ew.elliott_chart_data(tk, history=pd.DataFrame({tk: out["pf"]}))
            if _ewi and _ewi.get("pivots"):
                out["elliott"] = _ewi["pivots"]     # [{date, price, label 0..5, kind H/L}]
        except Exception:
            pass
    if "Ichimoku Cloud" in strset and out["pf"] is not None:
        try:                                        # price-axis overlay → same series as the chart
            _, _ici = _ich.ichimoku_chart_data(tk, history=pd.DataFrame({tk: out["pf"]}))
            if _ici and _ici.get("cloud"):
                out["ichimoku"] = _ici              # tenkan/kijun record lists + cloud [{date,a,b}]
        except Exception:
            pass
    if "On-Balance Volume" in strset:               # own panel + own (volume) history → don't pass pf
        try:
            _od, _oi = _obv.obv_chart_data(tk)
            if _od is not None and not _od.empty:
                out["obv"] = _od[["date", "obv"]].dropna()
        except Exception:
            pass
    if "Money Flow Index" in strset:                # own 0–100 panel + own (volume) history
        try:
            _md, _mi2 = _mfi.mfi_chart_data(tk)
            if _md is not None and not _md.empty:
                out["mfi"] = _md[["date", "mfi"]].dropna()
        except Exception:
            pass
    return out


@st.cache_data(show_spinner=False)
def _eq_ta_scored(as_of, scored_key=None):
    """Equities cross-strategy scoring — the equity opportunities (from eqta) through the SAME
    tascore engine as the FICC side. Recomputed only when the signals or confluence set change."""
    from src import eqta
    df, _ = eqta.load_signals()
    if df is None or df.empty:
        return None, None
    flagged = tascore.ta_flagged(df, list(scored_key) if scored_key else None)
    if flagged is None or flagged.empty:
        return None, None
    flagged = flagged.copy()
    flagged["dir"] = pd.to_numeric(flagged["direction"], errors="coerce").fillna(0).astype(int)
    flagged["strength"] = [tascore.strength(s, m) for s, m in zip(flagged["strategy"], flagged["metric"])]
    if "name" not in flagged.columns:                 # the leaderboard labels rows by `name`
        flagged["name"] = flagged["market"]
    return flagged, tascore.score_products(flagged)


@st.cache_data(show_spinner=False)
def _eq_ta_gallery_data(tk, strset_key, as_of):
    """Per-equity gallery data — mirrors _ta_gallery_data but feeds each strategy's *_chart_data the
    cached equity close/volume (get_history_ta has no equities)."""
    from src import eqta
    from src.strategies import (support_resistance as _sr, flag_breakout as _fb,
                                breakout_retest as _br, momentum as _mom, fibonacci as _fbn,
                                elliott_wave as _ew, ichimoku as _ich, obv as _obv, mfi as _mfi)
    strset = set(strset_key)
    out = {"pf": None, "flag": None, "sr_levels": [], "fib_levels": [], "retest_level": None,
           "mom": None, "elliott": None, "ichimoku": None, "obv": None, "mfi": None}
    close, vol = eqta.load_history()
    if close.empty or tk not in close.columns:
        return out
    out["pf"] = close[tk].dropna()
    hist = pd.DataFrame({tk: out["pf"]})
    volf = vol[[tk]] if (not vol.empty and tk in vol.columns) else None
    if "Flag Breakout" in strset:
        try:
            _fcd, _fi = _fb.flag_chart_data(tk, history=hist)
            if _fcd is not None and not _fcd.empty and _fi:
                out["flag"] = (_fcd[["date", "upper", "lower", "breakout"]].dropna(), _fi)
        except Exception:
            pass
    if "Support & Resistance" in strset:
        try:
            _, _isr = _sr.sr_chart_data(tk, history=hist)
            out["sr_levels"] = (_isr or {}).get("levels", []) or []
        except Exception:
            pass
    if "Fibonacci Retracement" in strset:
        try:
            _, _ifb = _fbn.fib_chart_data(tk, history=hist)
            out["fib_levels"] = [L for L in ((_ifb or {}).get("levels", []) or []) if L.get("key")]
        except Exception:
            pass
    if "Breakout & Retest" in strset:
        try:
            _, _ibr = _br.retest_chart_data(tk, history=hist)
            out["retest_level"] = (_ibr or {}).get("level")
        except Exception:
            pass
    if "Momentum (RSI/MACD)" in strset:
        try:
            _mcd, _mi = _mom.momentum_chart_data(tk, history=hist)
            if _mcd is not None and not _mcd.empty:
                out["mom"] = _mcd[["date", "rsi"]].dropna()
        except Exception:
            pass
    if "Elliott Wave" in strset:
        try:
            _, _ewi = _ew.elliott_chart_data(tk, history=hist)
            if _ewi and _ewi.get("pivots"):
                out["elliott"] = _ewi["pivots"]
        except Exception:
            pass
    if "Ichimoku Cloud" in strset:
        try:
            _, _ici = _ich.ichimoku_chart_data(tk, history=hist)
            if _ici and _ici.get("cloud"):
                out["ichimoku"] = _ici
        except Exception:
            pass
    if "On-Balance Volume" in strset and volf is not None:
        try:
            _od, _oi = _obv.obv_chart_data(tk, history=hist, volume=volf)
            if _od is not None and not _od.empty:
                out["obv"] = _od[["date", "obv"]].dropna()
        except Exception:
            pass
    if "Money Flow Index" in strset and volf is not None:
        try:
            _md, _mi2 = _mfi.mfi_chart_data(tk, history=hist, volume=volf)
            if _md is not None and not _md.empty:
                out["mfi"] = _md[["date", "mfi"]].dropna()
        except Exception:
            pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _eq_instruments() -> dict:
    """{ticker: (name, 0.0, sector, region)} — an INSTRUMENTS-shaped map of the equity universe so
    the shared prodsearch / sector helpers work on the Equities TA page."""
    from src import eqta
    return {t: (m["name"], 0.0, m["sector"], m["region"]) for t, m in eqta.member_meta().items()}


def render_ta_overview() -> None:
    import altair as alt
    from src.strategies import (support_resistance as _sr, flag_breakout as _fb,
                                breakout_retest as _br, momentum as _mom, fibonacci as _fbn)

    st.subheader("\U0001F52C Technical Analysis")
    st.caption("Every product flagged across the technical strategies — chart patterns, tested levels, "
               "momentum and volatility bands — ranked by a cross-strategy **conviction score** (how many "
               "strategies agree × how strong each is, longs netted against shorts). Open a strategy to "
               "tune its trigger and see its charts.")

    with st.expander("ℹ️  How “Conviction” and “Score” are calculated"):
        st.markdown(
            "Every strategy speaks its own language — a z-score, a 0–100 readiness/proximity, a momentum "
            "score, a return %, an MA-gap %. To rank products across all of them, each flagged signal is "
            "put on one common scale, then aggregated per product in **three steps**.\n\n"
            "**1 · Each flagged signal → a _strength_ (0–100).** How far the metric sits toward "
            "“full conviction”:\n\n"
            "> `strength = min(100, |metric| ÷ full-scale × 100)`\n\n"
            "where **full-scale** (the metric magnitude that scores 100) is:\n\n"
            "| Strategy | Native metric | = 100 at |\n"
            "|---|---|---|\n"
            "| Mean Reversion | \\|z-score\\| | 3.0 |\n"
            "| Trend | \\|3-month return\\| | 25% |\n"
            "| MA Crossover / MA Swing | \\|MA gap\\| | 10% |\n"
            "| Flag Breakout · S&R · Fibonacci · Breakout & Retest · Momentum · Bollinger Squeeze · "
            "Elliott Wave · Ichimoku · OBV · MFI | already 0–100 (readiness / proximity / momentum / "
            "squeeze / wave fit / Ichimoku / volume flow) | used as-is |\n\n"
            "**2 · Conviction (0–100) = the _average_ strength** of the strategies flagging that product — "
            "how strong the signals are on average, *regardless of how many* agree.\n\n"
            "**3 · Score = the _signed sum_ of those strengths** — long signals count **＋**, short **－** — "
            "displayed as **|Score|**:\n\n"
            "> `Score = | Σ (±strength) |`\n\n"
            "So Score rewards **both** confluence (more agreeing strategies) **and** strength, while "
            "opposing calls partly cancel. The **sign** of that sum sets the **Net** column "
            "(▲ long / ▼ short, or ⚠ *mixed* when both sides fire), and **|Score| ranks the table**.\n\n"
            "**Confluence set.** Only a curated, *independent* subset feeds this score — by default "
            "**Trend, Momentum (RSI/MACD), OBV, Support & Resistance and Flag Breakout**, one per axis "
            "(direction / momentum / volume / location / pattern) — so agreement is real corroboration, "
            "not the same read echoed. Edit it under **🎯 Confluence set** above; every other strategy "
            "keeps its page and chart overlays but stays out of the score. If you tick more than one "
            "method in the **same axis**, they're de-duplicated (strongest full, the next at **½**, "
            "**⅓**, …), so a single dimension can't vote twice.\n\n"
            "**Worked example.** Three strategies flag a product **Long** at strengths 90 / 80 / 70 and one "
            "flags it **Short** at 60 → Conviction = (90+80+70+60) ÷ 4 = **75**; "
            "Score = |＋90＋80＋70－60| = **180** (the short partly cancels); Net = **▲ long**. If instead all "
            "four agreed Long, Score = |90+80+70+60| = **300** — same conviction, far higher stacked score."
        )

    # Quick-nav row (top of page): open any strategy's own page — trigger control, full table, charts.
    st.caption("Open a strategy for its trigger control, full table and charts:")
    _ta_quicknav()

    df, meta = load_signals()
    if _all_filtered_off():                              # the Sectors & products filter hides everything
        st.warning("🗂️ **All sectors are switched off** in the Sectors & products filter (🏠 Home) — "
                   "nothing is enabled to analyse, so this page looks empty. Your data is fine.")
        if st.button("🗂️  Turn all sectors back on", key="ta_filter_reset", type="primary"):
            universe.save_filter(set(), set())
            for _s in _sf_sections():
                st.session_state.pop(_s[3], None)        # clear stale empty pills so they re-seed on
            st.rerun()
        return
    with st.expander("🎯 Confluence set — which methods feed the score, by axis", expanded=True):
        st.caption("The five independent **axes** of technical analysis. Tick the method(s) that feed "
                   "the conviction score within each — agreement **across** axes is genuine "
                   "corroboration, while several methods in one axis are de-duplicated (strongest "
                   "counts full, the next ½, then ⅓…) so a single dimension can't vote twice. Anything "
                   "not ticked keeps its own page and chart overlays; it's simply out of the score.")
        _saved = set(tascore.confluence_set())
        _conf = []
        _acols = st.columns(len(tascore.TA_AXES))          # one column per axis → side by side
        for _col, (_ax, _methods) in zip(_acols, tascore.TA_AXES.items()):
            _picked = _col.multiselect(
                _ax, options=_methods, default=[m for m in _methods if m in _saved],
                key=f"conf_ax::{_ax}",
                help=f"The {_ax.lower()} axis — {len(_methods)} method(s) available. "
                     "Blank = this dimension sits out of the score.")
            _conf.extend(_picked)
        st.session_state["conf_set"] = _conf          # the union — the report-generate handler reads this
        _cs1, _cs2 = st.columns([1, 2.4])
        if _cs1.button("💾 Save as default", key="conf_save",
                       help="Persist this set — used by the weekly report and on every launch."):
            tascore.save_confluence_set(_conf or tascore.CONFLUENCE_DEFAULT)
            st.toast("Confluence set saved.", icon="🎯")
        _cs2.caption("**Generate report** uses whatever's ticked here; **Save** also makes it the "
                     "default for the weekly email and future launches.")
    _conf = _conf or tascore.CONFLUENCE_DEFAULT            # never score an empty set
    _fkey = tuple(sorted(universe.enabled_tickers())) if universe.filter_active() else ()
    flagged, prod = _ta_scored(meta.get("as_of", ""), _fkey, tuple(_conf))
    if flagged is None or flagged.empty:
        st.info("Nothing flagged across the technical strategies right now. Open a strategy to lower its "
                "trigger, or pull a fresh snapshot on the 🏠 Home page.")
        return
    score_map = dict(zip(prod["instruments"], prod["score"]))

    n_long, n_short = int((flagged["dir"] > 0).sum()), int((flagged["dir"] < 0).sum())
    n_multi = int((prod["n"] >= 2).sum())
    m = st.columns(4)
    m[0].metric("Flagged signals", len(flagged))
    m[1].metric("Long / Short", f"{n_long} / {n_short}")
    m[2].metric("Products", int(prod.shape[0]))
    m[3].metric("Flagged by 2+", n_multi, help="Products flagged by more than one technical strategy.")

    def _arrow(d):
        return " ▲" if d > 0 else " ▼" if d < 0 else ""

    def _sector(k):
        return "pair" if " / " in str(k) else (INSTRUMENTS.get(k, (k, 0.0, "", ""))[2] or "—")

    # --- stacked signals (2+ strategies), ranked by conviction score. The table is
    #     click-selectable: picking a row drives the per-product chart panel below. ---
    multi = prod[prod["n"] >= 2]
    _q = st.text_input("Find a product", key="ta_ov_search", placeholder=prodsearch.PLACEHOLDER).strip()
    if _q:
        multi = prodsearch.filter_frame(multi, INSTRUMENTS, _q, ticker_col="instruments")
        if multi.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q))
    sel_pos = 0
    if not multi.empty:
        st.markdown("##### Stacked signals — flagged by 2 or more strategies (ranked by conviction)")
        st.caption("**Click a product** to bring up its charts — every indicator that flagged it — below.")
        rows = []
        for r in multi.itertuples(index=False):
            tags = ", ".join(_STRAT_SHORT.get(s, s) + _arrow(d) for s, d, _st in r.tags)
            net = "⚠ mixed" if r.conflict else ("▲ long" if r.net_dir > 0 else "▼ short" if r.net_dir < 0 else "—")
            rows.append({"Market": r.market, "Sector": _sector(r.instruments), "# Str": int(r.n),
                         "Net": net, "Conviction": r.conviction, "Score": abs(r.score), "Flagged by": tags})
        # A palette-styled, single-row-selectable grid (mirrors brand.themed_dataframe's theming
        # but returns the selection event so a click can drive the charts below).
        _pal = brand.palette()
        _sty = (pd.DataFrame(rows).style
                .format({"Conviction": "{:.0f}", "Score": "{:.0f}"})
                .set_properties(**{"background-color": _pal["surface"], "color": _pal["text"]}))
        _evt = st.dataframe(_sty, use_container_width=True, hide_index=True,
                            on_select="rerun", selection_mode="single-row", key="ta_stack_table")
        try:
            _sel = _evt.selection["rows"]
        except Exception:
            _sel = []
        if _sel:
            sel_pos = int(_sel[0])
        st.caption("▲ long · ▼ short · ⚠ strategies disagree. Click a row to chart that product below. "
                   "**Score** = Σ signed strength across the strategies (confluence × strength); "
                   "**Conviction** = their mean strength (0–100).")

    # --- per-strategy counts, CONFLUENCE SET ONLY: the non-scored strategies never enter `flagged`,
    #     so they'd read 0 across the board — redundant. Only the scored methods are listed. ---
    st.markdown("##### By strategy")
    st.caption("The methods in your confluence set above — the only ones scored, so the only ones counted here.")
    counts = [{"Strategy": s, "Flagged": int((flagged["strategy"] == s).sum()),
               "Long": int(((flagged["strategy"] == s) & (flagged["dir"] > 0)).sum()),
               "Short": int(((flagged["strategy"] == s) & (flagged["dir"] < 0)).sum())}
              for s in tascore.TA_STRATEGIES if s in set(_conf)]
    brand.themed_dataframe(pd.DataFrame(counts), {}, column_config={
        # pin the three integer columns narrow so they don't over-expand and clip "Short" off the
        # right edge; the wide "Strategy" text column then absorbs the remaining container width.
        "Flagged": st.column_config.NumberColumn(width="small"),
        "Long": st.column_config.NumberColumn(width="small"),
        "Short": st.column_config.NumberColumn(width="small"),
    })

    # --- charts for the SELECTED stacked product (default: the top row), drawing the
    #     indicators that flagged it. Driven by the table selection above. ---
    gallery = multi.iloc[[sel_pos]] if not multi.empty else multi.iloc[0:0]
    if not gallery.empty:
        _sel_name = str(gallery.iloc[0]["market"])
        st.markdown(f"##### Charts — {_sel_name}")
        st.caption("Charts for the **selected** stacked product (click another row above to switch). Each "
                   "chart draws **what triggered the flags**: Bollinger bands, the moving averages "
                   "(MA crossover 50/200 · swing 20/50 · trend 20/100), the flag channel, the **Ichimoku "
                   "cloud + Tenkan/Kijun**, the **Elliott wave count (purple, 0–5)**, and the "
                   "support/resistance, broken and flag-breakout levels. Sub-panels below carry **RSI / MFI** "
                   "(when momentum or money-flow flag) and **OBV** (when volume flags). (Mean Reversion is a "
                   "pair spread, so it's noted but not overlaid here.)")
        _cc = brand.chart_colors()
        for r in gallery.itertuples(index=False):
            tk = r.instruments
            _yax = "Yield (%)" if universe.is_fixed_income(tk) else "Price"
            strset = {s for s, _, _ in r.tags}
            tags_txt = ", ".join(_STRAT_SHORT.get(s, s) + _arrow(d) for s, d, _st in r.tags)
            net = "long ▲" if r.net_dir > 0 else "short ▼" if r.net_dir < 0 else "mixed"
            st.markdown(f"**{r.market}** · {int(r.n)} strategies ({tags_txt}) · net **{net}** · score **{abs(r.score):.0f}**")
            _g = _ta_gallery_data(tk, frozenset(strset), meta.get("as_of", ""))   # cached (no recompute per rerun)
            pf = _g["pf"]
            if pf is None or pf.empty:
                st.caption("No history to chart.")
                continue
            win = pf.tail(180)
            pxdf = pd.DataFrame({"date": win.index, "price": win.to_numpy(dtype=float)})

            # Price-axis line overlays (computed on the FULL history for proper lookback, shown
            # over the window) — the bands / MAs that the flagging strategies are built on.
            lines = {}
            if "Bollinger Squeeze" in strset:
                _mid, _sd = pf.rolling(20).mean(), pf.rolling(20).std()
                lines["BB upper"], lines["BB mid"], lines["BB lower"] = _mid + 2 * _sd, _mid, _mid - 2 * _sd
            for _strat, _ws in (("MA Crossover", (50, 200)), ("MA Swing", (20, 50)), ("Trend", (20, 100))):
                if _strat in strset:
                    for _w in _ws:
                        lines.setdefault(f"MA{_w}", pf.rolling(_w).mean())

            # The flag pattern drawn in full (channel fill + edges + dashed breakout + pole, in
            # its direction colour) and the horizontal levels from the other visual strategies.
            flag_layers, rules = [], []
            if _g["flag"]:
                _fch, _fi = _g["flag"]
                _fcol = _cc["long"] if _fi["sign"] > 0 else _cc["short"]
                _fbase = alt.Chart(_fch).encode(x="date:T")
                flag_layers += [
                    _fbase.mark_area(opacity=0.22, color=_fcol).encode(y="lower:Q", y2="upper:Q"),
                    _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="upper:Q"),
                    _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="lower:Q"),
                    _fbase.mark_line(color=_fcol, strokeDash=[6, 3], strokeWidth=2.4).encode(y="breakout:Q"),
                    alt.Chart(pd.DataFrame({"date": [_fi["pole_base"][0], _fi["pole_tip"][0]],
                                            "price": [_fi["pole_base"][1], _fi["pole_tip"][1]]})).mark_line(
                        color="#B0B0B0", strokeWidth=2.8).encode(x="date:T", y="price:Q"),
                ]
            for lv in _g["sr_levels"]:
                rules.append((lv["price"], _cc["long"] if lv["kind"] == "support" else _cc["short"]))
            for _L in _g["fib_levels"]:
                rules.append((_L["price"], _cc["accent"]))
            if _g["retest_level"] is not None:
                rules.append((_g["retest_level"], _cc["accent"]))

            base = alt.Chart(pxdf).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)))
            layers = list(flag_layers)
            if lines:
                _ldf = pd.DataFrame({"date": win.index})
                for _lab, _ser in lines.items():
                    _ldf[_lab] = _ser.reindex(win.index).to_numpy(dtype=float)
                _long = _ldf.melt("date", var_name="Indicator", value_name="val").dropna(subset=["val"])
                layers.append(alt.Chart(_long).mark_line(strokeWidth=1.8).encode(
                    x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("Indicator:N", legend=alt.Legend(orient="top", title=None, labelFontSize=11)),
                    tooltip=[alt.Tooltip("Indicator:N"), alt.Tooltip("val:Q", format=",.2f")]))
            layers.append(base.mark_line(color=_cc["ink"], strokeWidth=2.3).encode(
                y=alt.Y("price:Q", title=_yax, scale=alt.Scale(zero=False), axis=alt.Axis(labelFontSize=11)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_yax, format=",.2f")]))
            for pv, cv in rules:
                if np.isfinite(pv):
                    layers.append(alt.Chart(pd.DataFrame({"y": [pv]})).mark_rule(
                        color=cv, strokeDash=[5, 3], opacity=0.85, strokeWidth=1.8).encode(y="y:Q"))
            # Elliott wave count: the labelled 0..5 pivots as a purple markered line over the price
            # (clipped to the shown window), matching the report chart. Drawn last → sits on top.
            if _g.get("elliott"):
                _piv = pd.DataFrame([p for p in _g["elliott"] if p["date"] >= win.index[0]])
                if len(_piv) >= 2:
                    layers.append(alt.Chart(_piv).mark_line(
                        color="#9575CD", strokeWidth=1.8, opacity=0.9,
                        point=alt.OverlayMarkDef(color="#9575CD", size=42)).encode(
                        x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("label:N", title="Wave"),
                                 alt.Tooltip("price:Q", title=_yax, format=",.2f")]))
                    layers.append(alt.Chart(_piv).mark_text(
                        dy=-12, fontSize=12, fontWeight="bold", color="#B39DDB").encode(
                        x="date:T", y="price:Q", text="label:N"))
            # Ichimoku Kumo (cloud) + Tenkan/Kijun — PREPENDED so the translucent cloud sits behind
            # the price (like the report); green where span-A ≥ span-B, red where below. The cloud
            # carries a 26-session forward projection, so it extends the x-axis to the right.
            _ich_layers = []
            _ich = _g.get("ichimoku")
            if _ich and _ich.get("cloud"):
                _cl = pd.DataFrame([c for c in _ich["cloud"] if c["date"] >= win.index[0]]).dropna(
                    subset=["a", "b"])
                if not _cl.empty:
                    _cl["bull"] = _cl["a"] >= _cl["b"]
                    for _fl, _col in ((True, _cc["long"]), (False, _cc["short"])):
                        _seg = _cl.copy()
                        _seg.loc[_cl["bull"] != _fl, ["a", "b"]] = None
                        _ich_layers.append(alt.Chart(_seg).mark_area(opacity=0.32).encode(
                            x="date:T", y=alt.Y("a:Q", scale=alt.Scale(zero=False)), y2="b:Q",
                            color=alt.value(_col)))
                    for _k, _c2 in (("tenkan", "#26A69A"), ("kijun", "#EC407A")):
                        _ln = pd.DataFrame([r for r in (_ich.get(_k) or []) if r["date"] >= win.index[0]]
                                           ).dropna(subset=["val"])
                        if not _ln.empty:
                            _ich_layers.append(alt.Chart(_ln).mark_line(
                                color=_c2, strokeWidth=1.2, opacity=0.85).encode(
                                x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False))))
            brand.show_chart(alt.layer(*(_ich_layers + layers)).resolve_scale(y="shared").properties(height=300))

            # Oscillator sub-panel (0–100): RSI when momentum flags, MFI when money-flow flags — they
            # share one panel like the report (RSI 70/30 guides, MFI 80/20).
            _osc, _guides = [], []
            if _g["mom"] is not None:
                _osc.append(("rsi", _g["mom"].tail(180), "#7E57C2", "RSI"))
                _guides += [(70, _cc["short"]), (30, _cc["long"])]
            if _g["mfi"] is not None:
                _osc.append(("mfi", _g["mfi"].tail(180), "#00897B", "MFI"))
                _guides += [(80, _cc["short"]), (20, _cc["long"])]
            if _osc:
                _olays = [alt.Chart(_df).mark_line(color=_c, strokeWidth=2).encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                    y=alt.Y(f"{_col_name}:Q", title="RSI / MFI", scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(values=[0, 20, 30, 50, 70, 80, 100], labelFontSize=11)))
                    for _col_name, _df, _c, _ in _osc]
                _olays += [alt.Chart(pd.DataFrame({"y": [_y]})).mark_rule(
                    color=_c, strokeDash=[4, 3]).encode(y="y:Q") for _y, _c in _guides]
                brand.show_chart(alt.layer(*_olays).resolve_scale(y="shared").properties(
                    height=130, title=" / ".join(t for _, _, _, t in _osc) + " (14)"))

            # OBV sub-panel — cumulative volume; its trend vs price (confirmation / divergence) is the read.
            if _g["obv"] is not None:
                brand.show_chart(alt.Chart(_g["obv"].tail(180)).mark_line(
                    color="#26A69A", strokeWidth=1.8).encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                    y=alt.Y("obv:Q", title="OBV", scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelFontSize=10))).properties(height=110, title="On-Balance Volume"))

    # --- full flagged leaderboard, ranked by product conviction score ---
    st.markdown("##### All flagged signals")
    fc1, fc2 = st.columns([3, 2])
    pick = fc1.multiselect("Filter by strategy", tascore.TA_STRATEGIES, default=[], key="ta_filter",
                           help="Empty = every technical strategy.")
    sidef = fc2.radio("Side", ["All", "Long", "Short"], horizontal=True, key="ta_side")
    view = flagged
    if pick:
        view = view[view["strategy"].isin(pick)]
    if sidef == "Long":
        view = view[view["dir"] > 0]
    elif sidef == "Short":
        view = view[view["dir"] < 0]
    view = view.assign(_score=view["instruments"].map(lambda k: abs(score_map.get(k, 0.0))))
    view = view.sort_values(["_score", "name", "strategy"], ascending=[False, True, True])
    show = pd.DataFrame({
        "Market": view["name"].values,
        "Sector": [_sector(k) for k in view["instruments"]],
        "Strategy": [_STRAT_SHORT.get(s, s) for s in view["strategy"]],
        "Signal": [f"{sig}{_arrow(d)}" for sig, d in zip(view["signal"], view["dir"])],
        "Conviction": view["strength"].round(0).values,
        "Score": view["_score"].round(0).values,
        "Notes": view["context"].values,
    })

    def _sig_color(col):
        return ["color:#137333;font-weight:700" if "▲" in str(v)
                else "color:#c5221f;font-weight:700" if "▼" in str(v) else "color:#888" for v in col]

    brand.themed_dataframe(show, {"Conviction": "{:.0f}", "Score": "{:.0f}"},
                           colorers=[(["Signal"], _sig_color)], height=520)
    st.caption("**Conviction** = this signal's strength 0–100 (on its strategy's scale); **Score** = the "
               "product's cross-strategy conviction (confluence × strength), so a product's rows cluster at "
               "the top when several strategies agree. The **Confluence** page covers the whole book.")

    # Report controls pinned to the FOOT of the page (a consistent "generate + email at the bottom"
    # layout across the app). Still isolated in a fragment, so a Generate click reruns only this
    # block — not the leaderboard/gallery above it.
    st.divider()
    _ta_reports(meta, prod)


def _ta_render_gallery(gallery, gallery_data_fn, as_of) -> None:
    """The per-product chart gallery — price + every flagging overlay (Bollinger, MAs, flag channel,
    Ichimoku cloud + Tenkan/Kijun, Elliott count, S&R / Fib / retest levels) and the RSI/MFI + OBV
    sub-panels. Shared by the FICC and Equities TA pages; only `gallery_data_fn` (the per-product
    data source) differs. `_yax` is Price for equities automatically (is_fixed_income is False)."""
    import altair as alt                              # render_ta_overview imports it lazily; do the same here
    if gallery is None or gallery.empty:
        return

    def _arrow(d):
        return " ▲" if d > 0 else " ▼" if d < 0 else ""

    _sel_name = str(gallery.iloc[0]["market"])
    st.markdown(f"##### Charts — {_sel_name}")
    st.caption("Charts for the **selected** stacked product (click another row above to switch). Each "
               "chart draws **what triggered the flags**: Bollinger bands, the moving averages "
               "(MA crossover 50/200 · swing 20/50 · trend 20/100), the flag channel, the **Ichimoku "
               "cloud + Tenkan/Kijun**, the **Elliott wave count (purple, 0–5)**, and the "
               "support/resistance, broken and flag-breakout levels. Sub-panels below carry **RSI / MFI** "
               "(when momentum or money-flow flag) and **OBV** (when volume flags).")
    _cc = brand.chart_colors()
    for r in gallery.itertuples(index=False):
        tk = r.instruments
        _yax = "Yield (%)" if universe.is_fixed_income(tk) else "Price"
        strset = {s for s, _, _ in r.tags}
        tags_txt = ", ".join(_STRAT_SHORT.get(s, s) + _arrow(d) for s, d, _st in r.tags)
        net = "long ▲" if r.net_dir > 0 else "short ▼" if r.net_dir < 0 else "mixed"
        st.markdown(f"**{r.market}** · {int(r.n)} strategies ({tags_txt}) · net **{net}** · score **{abs(r.score):.0f}**")
        _g = gallery_data_fn(tk, frozenset(strset), as_of)
        pf = _g["pf"]
        if pf is None or pf.empty:
            st.caption("No history to chart.")
            continue
        win = pf.tail(180)
        pxdf = pd.DataFrame({"date": win.index, "price": win.to_numpy(dtype=float)})

        lines = {}
        if "Bollinger Squeeze" in strset:
            _mid, _sd = pf.rolling(20).mean(), pf.rolling(20).std()
            lines["BB upper"], lines["BB mid"], lines["BB lower"] = _mid + 2 * _sd, _mid, _mid - 2 * _sd
        for _strat, _ws in (("MA Crossover", (50, 200)), ("MA Swing", (20, 50)), ("Trend", (20, 100))):
            if _strat in strset:
                for _w in _ws:
                    lines.setdefault(f"MA{_w}", pf.rolling(_w).mean())

        flag_layers, rules = [], []
        if _g["flag"]:
            _fch, _fi = _g["flag"]
            _fcol = _cc["long"] if _fi["sign"] > 0 else _cc["short"]
            _fbase = alt.Chart(_fch).encode(x="date:T")
            flag_layers += [
                _fbase.mark_area(opacity=0.22, color=_fcol).encode(y="lower:Q", y2="upper:Q"),
                _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="upper:Q"),
                _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="lower:Q"),
                _fbase.mark_line(color=_fcol, strokeDash=[6, 3], strokeWidth=2.4).encode(y="breakout:Q"),
                alt.Chart(pd.DataFrame({"date": [_fi["pole_base"][0], _fi["pole_tip"][0]],
                                        "price": [_fi["pole_base"][1], _fi["pole_tip"][1]]})).mark_line(
                    color="#B0B0B0", strokeWidth=2.8).encode(x="date:T", y="price:Q"),
            ]
        for lv in _g["sr_levels"]:
            rules.append((lv["price"], _cc["long"] if lv["kind"] == "support" else _cc["short"]))
        for _L in _g["fib_levels"]:
            rules.append((_L["price"], _cc["accent"]))
        if _g["retest_level"] is not None:
            rules.append((_g["retest_level"], _cc["accent"]))

        base = alt.Chart(pxdf).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)))
        layers = list(flag_layers)
        if lines:
            _ldf = pd.DataFrame({"date": win.index})
            for _lab, _ser in lines.items():
                _ldf[_lab] = _ser.reindex(win.index).to_numpy(dtype=float)
            _long = _ldf.melt("date", var_name="Indicator", value_name="val").dropna(subset=["val"])
            layers.append(alt.Chart(_long).mark_line(strokeWidth=1.8).encode(
                x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Indicator:N", legend=alt.Legend(orient="top", title=None, labelFontSize=11)),
                tooltip=[alt.Tooltip("Indicator:N"), alt.Tooltip("val:Q", format=",.2f")]))
        layers.append(base.mark_line(color=_cc["ink"], strokeWidth=2.3).encode(
            y=alt.Y("price:Q", title=_yax, scale=alt.Scale(zero=False), axis=alt.Axis(labelFontSize=11)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_yax, format=",.2f")]))
        for pv, cv in rules:
            if np.isfinite(pv):
                layers.append(alt.Chart(pd.DataFrame({"y": [pv]})).mark_rule(
                    color=cv, strokeDash=[5, 3], opacity=0.85, strokeWidth=1.8).encode(y="y:Q"))
        if _g.get("elliott"):
            _piv = pd.DataFrame([p for p in _g["elliott"] if p["date"] >= win.index[0]])
            if len(_piv) >= 2:
                layers.append(alt.Chart(_piv).mark_line(
                    color="#9575CD", strokeWidth=1.8, opacity=0.9,
                    point=alt.OverlayMarkDef(color="#9575CD", size=42)).encode(
                    x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("label:N", title="Wave"),
                             alt.Tooltip("price:Q", title=_yax, format=",.2f")]))
                layers.append(alt.Chart(_piv).mark_text(
                    dy=-12, fontSize=12, fontWeight="bold", color="#B39DDB").encode(
                    x="date:T", y="price:Q", text="label:N"))
        _ich_layers = []
        _ich = _g.get("ichimoku")
        if _ich and _ich.get("cloud"):
            _cl = pd.DataFrame([c for c in _ich["cloud"] if c["date"] >= win.index[0]]).dropna(
                subset=["a", "b"])
            if not _cl.empty:
                _cl["bull"] = _cl["a"] >= _cl["b"]
                for _fl, _col in ((True, _cc["long"]), (False, _cc["short"])):
                    _seg = _cl.copy()
                    _seg.loc[_cl["bull"] != _fl, ["a", "b"]] = None
                    _ich_layers.append(alt.Chart(_seg).mark_area(opacity=0.32).encode(
                        x="date:T", y=alt.Y("a:Q", scale=alt.Scale(zero=False)), y2="b:Q",
                        color=alt.value(_col)))
                for _k, _c2 in (("tenkan", "#26A69A"), ("kijun", "#EC407A")):
                    _ln = pd.DataFrame([rr for rr in (_ich.get(_k) or []) if rr["date"] >= win.index[0]]
                                       ).dropna(subset=["val"])
                    if not _ln.empty:
                        _ich_layers.append(alt.Chart(_ln).mark_line(
                            color=_c2, strokeWidth=1.2, opacity=0.85).encode(
                            x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False))))
        brand.show_chart(alt.layer(*(_ich_layers + layers)).resolve_scale(y="shared").properties(height=300))

        _osc, _guides = [], []
        if _g["mom"] is not None:
            _osc.append(("rsi", _g["mom"].tail(180), "#7E57C2", "RSI"))
            _guides += [(70, _cc["short"]), (30, _cc["long"])]
        if _g["mfi"] is not None:
            _osc.append(("mfi", _g["mfi"].tail(180), "#00897B", "MFI"))
            _guides += [(80, _cc["short"]), (20, _cc["long"])]
        if _osc:
            _olays = [alt.Chart(_df).mark_line(color=_c, strokeWidth=2).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                y=alt.Y(f"{_col_name}:Q", title="RSI / MFI", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=[0, 20, 30, 50, 70, 80, 100], labelFontSize=11)))
                for _col_name, _df, _c, _ in _osc]
            _olays += [alt.Chart(pd.DataFrame({"y": [_y]})).mark_rule(
                color=_c, strokeDash=[4, 3]).encode(y="y:Q") for _y, _c in _guides]
            brand.show_chart(alt.layer(*_olays).resolve_scale(y="shared").properties(
                height=130, title=" / ".join(t for _, _, _, t in _osc) + " (14)"))

        if _g["obv"] is not None:
            brand.show_chart(alt.Chart(_g["obv"].tail(180)).mark_line(
                color="#26A69A", strokeWidth=1.8).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                y=alt.Y("obv:Q", title="OBV", scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=10))).properties(height=110, title="On-Balance Volume"))


def render_eq_ta_overview() -> None:
    """Equities Technical Analysis — the FICC TA overview run on the equity universe off yfinance
    data. Same strategies, same cross-strategy scoring, same charts, and the same foot-of-page PDF
    report + email controls; the confluence set and report settings are this page's OWN, independent
    of the FICC ones."""
    from src import eqta
    st.subheader("\U0001F4C8 Equities — Technical Analysis")
    st.caption("Every equity flagged across the technical strategies, ranked by a cross-strategy "
               "**conviction score** — the same engine as the FICC page, on ~2,600 names off free "
               "yfinance data. This page keeps its **own** confluence set, independent of the FICC one.")

    df, meta = eqta.load_signals()
    if df is None or df.empty:
        st.info("No equity signals cached yet. Run the equities backfill + engine "
                "(`python -c \"from src import eqta; eqta.backfill(eqta.universe_tickers()); eqta.run()\"`).")
        return
    st.caption(f"Signals as of **{meta.get('as_of', '—')}** over **{meta.get('names', '?')}** names.")

    _ta_conviction_expander()

    # Quick-nav row (top of page): open any strategy's own EQUITIES page — trigger control, table, charts.
    st.caption("Open a strategy for its trigger control, full table and charts:")
    _ta_quicknav(eq=True)

    with st.expander("🎯 Confluence set — which methods feed the score, by axis", expanded=True):
        st.caption("The five independent **axes** of technical analysis — this **equities** page keeps its "
                   "**own** selection, separate from the FICC page. Tick the method(s) that feed the score "
                   "within each; several in one axis are de-duplicated so a dimension can't vote twice. "
                   "(Flag Breakout has no equity signals yet, so it contributes nothing here for now.)")
        _saved = set(tascore.confluence_set("equities"))
        _conf = []
        _acols = st.columns(len(tascore.TA_AXES))
        for _col, (_ax, _methods) in zip(_acols, tascore.TA_AXES.items()):
            _picked = _col.multiselect(
                _ax, options=_methods, default=[m for m in _methods if m in _saved],
                key=f"conf_ax_eq::{_ax}",
                help=f"The {_ax.lower()} axis — {len(_methods)} method(s) available. Blank = out of the score.")
            _conf.extend(_picked)
        if st.button("💾 Save as default", key="conf_save_eq",
                     help="Persist this equities confluence set — independent of the FICC one."):
            tascore.save_confluence_set(_conf or tascore.CONFLUENCE_DEFAULT, scope="equities")
            st.toast("Equities confluence set saved.", icon="🎯")
    _conf = _conf or tascore.CONFLUENCE_DEFAULT
    flagged, prod = _eq_ta_scored(meta.get("as_of", ""), tuple(_conf))
    if flagged is None or flagged.empty:
        st.info("Nothing flagged across the confluence-set strategies right now.")
        return
    score_map = dict(zip(prod["instruments"], prod["score"]))
    _inst = _eq_instruments()

    n_long, n_short = int((flagged["dir"] > 0).sum()), int((flagged["dir"] < 0).sum())
    n_multi = int((prod["n"] >= 2).sum())
    m = st.columns(4)
    m[0].metric("Flagged signals", len(flagged))
    m[1].metric("Long / Short", f"{n_long} / {n_short}")
    m[2].metric("Products", int(prod.shape[0]))
    m[3].metric("Flagged by 2+", n_multi, help="Products flagged by more than one technical strategy.")

    def _arrow(d):
        return " ▲" if d > 0 else " ▼" if d < 0 else ""

    def _sector(k):
        return _inst.get(k, (k, 0.0, "—", ""))[2] or "—"

    multi = prod[prod["n"] >= 2]
    _q = st.text_input("Find a company", key="eqta_search", placeholder=prodsearch.PLACEHOLDER).strip()
    if _q:
        multi = prodsearch.filter_frame(multi, _inst, _q, ticker_col="instruments")
        if multi.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q))
    sel_pos = 0
    if not multi.empty:
        st.markdown("##### Stacked signals — flagged by 2 or more strategies (ranked by conviction)")
        st.caption("**Click a product** to bring up its charts — every indicator that flagged it — below.")
        rows = []
        for r in multi.itertuples(index=False):
            tags = ", ".join(_STRAT_SHORT.get(s, s) + _arrow(d) for s, d, _st in r.tags)
            net = "⚠ mixed" if r.conflict else ("▲ long" if r.net_dir > 0 else "▼ short" if r.net_dir < 0 else "—")
            rows.append({"Market": r.market, "Sector": _sector(r.instruments), "# Str": int(r.n),
                         "Net": net, "Conviction": r.conviction, "Score": abs(r.score), "Flagged by": tags})
        _pal = brand.palette()
        _sty = (pd.DataFrame(rows).style
                .format({"Conviction": "{:.0f}", "Score": "{:.0f}"})
                .set_properties(**{"background-color": _pal["surface"], "color": _pal["text"]}))
        _evt = st.dataframe(_sty, use_container_width=True, hide_index=True,
                            on_select="rerun", selection_mode="single-row", key="eqta_stack_table")
        try:
            _sel = _evt.selection["rows"]
        except Exception:
            _sel = []
        if _sel:
            sel_pos = int(_sel[0])
        st.caption("▲ long · ▼ short · ⚠ strategies disagree. Click a row to chart that product below. "
                   "**Score** = Σ signed strength across the strategies; **Conviction** = their mean strength (0–100).")

    st.markdown("##### By strategy")
    st.caption("The methods in your confluence set above — the only ones scored, so the only ones counted here.")
    counts = [{"Strategy": s, "Flagged": int((flagged["strategy"] == s).sum()),
               "Long": int(((flagged["strategy"] == s) & (flagged["dir"] > 0)).sum()),
               "Short": int(((flagged["strategy"] == s) & (flagged["dir"] < 0)).sum())}
              for s in tascore.TA_STRATEGIES if s in set(_conf)]
    brand.themed_dataframe(pd.DataFrame(counts), {}, column_config={
        "Flagged": st.column_config.NumberColumn(width="small"),
        "Long": st.column_config.NumberColumn(width="small"),
        "Short": st.column_config.NumberColumn(width="small"),
    })

    gallery = multi.iloc[[sel_pos]] if not multi.empty else multi.iloc[0:0]
    _ta_render_gallery(gallery, _eq_ta_gallery_data, meta.get("as_of", ""))

    st.markdown("##### All flagged signals")
    fc1, fc2 = st.columns([3, 2])
    pick = fc1.multiselect("Filter by strategy", tascore.TA_STRATEGIES, default=[], key="eqta_filter",
                           help="Empty = every technical strategy.")
    sidef = fc2.radio("Side", ["All", "Long", "Short"], horizontal=True, key="eqta_side")
    view = flagged
    if pick:
        view = view[view["strategy"].isin(pick)]
    if sidef == "Long":
        view = view[view["dir"] > 0]
    elif sidef == "Short":
        view = view[view["dir"] < 0]
    view = view.assign(_score=view["instruments"].map(lambda k: abs(score_map.get(k, 0.0))))
    view = view.sort_values(["_score", "name", "strategy"], ascending=[False, True, True])
    show = pd.DataFrame({
        "Market": view["name"].values,
        "Sector": [_sector(k) for k in view["instruments"]],
        "Strategy": [_STRAT_SHORT.get(s, s) for s in view["strategy"]],
        "Signal": [f"{sig}{_arrow(d)}" for sig, d in zip(view["signal"], view["dir"])],
        "Conviction": view["strength"].round(0).values,
        "Score": view["_score"].round(0).values,
        "Notes": view["context"].values,
    })

    def _sig_color(col):
        return ["color:#137333;font-weight:700" if "▲" in str(v)
                else "color:#c5221f;font-weight:700" if "▼" in str(v) else "color:#888" for v in col]

    brand.themed_dataframe(show, {"Conviction": "{:.0f}", "Score": "{:.0f}"},
                           colorers=[(["Signal"], _sig_color)], height=520)

    # Report controls pinned to the FOOT of the page — the same "generate + email at the bottom"
    # layout as the FICC page, run over the equity book (scope="equities" feeds convreport the
    # yfinance OHLCV store) and scored on THIS page's confluence set.
    st.divider()
    _ta_reports(meta, prod, scope="equities", conf_set=_conf)


def render_eq_strategy(strat: str) -> None:
    """An Equities per-strategy page — the FICC strategy-page layout (quick-nav, per-strategy trigger
    control, chart and full table) run on the equity universe off yfinance data. Reached from the TA
    quick-nav (active = "eq:<strategy>"). Trigger defaults persist independently of the FICC ones."""
    from src import eqta
    st.header(strat)
    st.caption(STRATEGY_BLURB.get(strat, ""))
    _ta_quicknav(strat, eq=True)
    st.caption("💡 Equities run on **price** (free yfinance data) — no fixed-income yield inversion here; "
               "a **Long / up** read simply screens the share price higher.")

    df, meta = eqta.load_signals()
    if df is None or df.empty:
        st.info("No equity signals cached yet — run the equities backfill + engine from the "
                "📈 Technical Analysis page first.")
        return
    _inst = _eq_instruments()

    def _sector(k):
        return _inst.get(k, (k, 0.0, "—", ""))[2] or "—"

    # --- per-strategy trigger control (same as the FICC pages), persisted independently as eq:<strat> ---
    spec = SPECS.get(strat, {})
    threshold = spec.get("default")
    if threshold is not None:
        threshold = st.number_input(
            spec["label"], min_value=float(spec["min"]), max_value=float(spec["max"]),
            value=trigger_default(f"eq:{strat}", spec["default"]), step=float(spec["step"]),
            key=f"eqthr_{strat}",
            help="Changing this re-derives the flags from the stored metrics — no data re-pull. It sets "
                 "the trigger for the table below, and is independent of this strategy's FICC trigger.")
        st.info(f"**Trigger:** {spec['trigger'](threshold)}")
        _td1, _td2 = st.columns([0.74, 0.26])
        if _td2.button("📌 Set default", key=f"eqthr_def_{strat}", use_container_width=True,
                       help="Save this as the EQUITIES default trigger for this strategy (independent of FICC)."):
            save_trigger_default(f"eq:{strat}", float(threshold))
            st.toast(f"Saved {threshold:g} as the equities default trigger for {strat}.", icon="📌")
        _td1.caption(f"📌 Equities default trigger for **{strat}**: "
                     f"**{trigger_default(f'eq:{strat}', spec['default']):g}** — change it above, then **Set default**.")
    if spec.get("math"):
        with st.expander("ℹ️  How this is calculated"):
            st.markdown(spec["math"])

    # --- this strategy's flagged equity signals, reflagged at the trigger ---
    _v = df[df["strategy"] == strat].copy()
    if _v.empty:
        _why = " (not run on the equity universe)" if strat in ("Mean Reversion", "Flag Breakout") else ""
        st.info(f"No **{strat}** signals in the equity universe{_why}.")
        return
    if threshold is not None and spec.get("hi"):
        _v = reflag_rows(_v, float(threshold), spec["hi"], spec["lo"])   # equities: price, not yields
        _v = pd.concat([_v[_v["direction"] != 0], _v[_v["direction"] == 0]])
    _flagged = _v[_v["direction"] != 0].copy()

    # --- chart: reuse the shared gallery (price + this strategy's overlay + RSI/MFI/OBV sub-panels) ---
    if _flagged.empty:
        st.info("Nothing flagged at the current trigger — lower it to chart the near-misses.")
    else:
        _order = _flagged.reindex(_flagged["metric"].abs().sort_values(ascending=False).index)
        _mkts = _order["market"].tolist()
        sel = st.selectbox(f"Chart a market — {len(_mkts)} flagged at the current trigger (strongest first)",
                           _mkts, key=f"eqstrat_sel_{strat}")
        _row = _order[_order["market"] == sel].iloc[0]
        _d = int(_row["direction"])
        _stg = tascore.strength(strat, _row["metric"])
        gallery = pd.DataFrame([{
            "instruments": _row["instruments"], "market": sel, "n": 1, "net_dir": _d,
            "score": _d * _stg, "conflict": False, "tags": [(strat, _d, _stg)],
        }])
        _ta_render_gallery(gallery, _eq_ta_gallery_data, meta.get("as_of", ""))

    # --- full table for this strategy across the equity universe (tick rows → plain-table PDF) ---
    st.markdown("##### All flagged signals")
    st.caption("Tick rows to export a plain-table PDF for **this** strategy; the full multi-indicator "
               "client report (leaderboard + charts + write-ups) is on the 📈 Technical Analysis page.")
    _q = st.text_input("Find a company", key=f"eqstrat_find_{strat}",
                       placeholder=prodsearch.PLACEHOLDER).strip()
    show_src = _v
    if _q:
        show_src = prodsearch.filter_frame(show_src, _inst, _q, ticker_col="instruments")
        if show_src.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q))
    if not show_src.empty:
        _view = show_src.copy()
        _view.insert(0, "Include", _view["signal"].ne("—"))
        _view.insert(2, "Sector", [_sector(k) for k in _view["instruments"]])
        _cols = ["Include", "market", "Sector", "signal", "metric", "level", "context"]
        _edited = st.data_editor(
            _view[_cols], use_container_width=True, hide_index=True, key=f"eqstrat_editor_{strat}",
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", help="Tick to add to the PDF report"),
                "market": "Market", "Sector": st.column_config.TextColumn("Sector", width="small"),
                "signal": "Signal", "metric": "Metric", "level": "Level", "context": "Notes"},
            disabled=[c for c in _cols if c != "Include"])
        chosen = show_src.loc[[i for i, keep in zip(_view.index, _edited["Include"]) if keep]]
        st.caption(f"**{len(chosen)}** row(s) selected for the report.")
        _pk = f"eqstrat_pdf_{strat}"
        if st.button("📄 Generate PDF report", type="primary", key=f"eqstrat_gen_{strat}",
                     disabled=chosen.empty):
            with st.spinner("Rendering PDF…"):
                with tempfile.TemporaryDirectory() as tmp:
                    out_pdf, rows_json = Path(tmp) / "report.pdf", Path(tmp) / "rows.json"
                    rows_json.write_text(chosen.to_json(orient="records"), encoding="utf-8")
                    cmd = [sys.executable, str(REPORT_CLI), str(rows_json), str(out_pdf),
                           "--title", f"{strat} — Equities", "--asof", str(meta.get("as_of", "")),
                           "--trigger", (spec["trigger"](threshold) if threshold is not None else ""),
                           "--no-filter"]                       # equity tickers aren't in the FICC enabled-set
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    ok = res.returncode == 0 and out_pdf.exists()
                    st.session_state[_pk] = out_pdf.read_bytes() if ok else None
            if not st.session_state.get(_pk):
                st.error("Report failed:\n\n" + (res.stderr or res.stdout or "no output"))
            else:
                st.success("Report ready.")
        if st.session_state.get(_pk):
            _fn = f"{strat.replace(' ', '_').replace('/', '-')}_Equities.pdf"
            st.download_button(f"⬇️ Download {_fn}", data=st.session_state[_pk], file_name=_fn,
                               mime="application/pdf", key=f"eqstrat_dl_{strat}")
            email_report_ui(_pk, _pk, st.session_state.get(_pk),
                            subject=f"{strat} — Equities Technical Analysis", attachment_name=_fn)


def render_data_health() -> None:
    st.subheader("\U0001FA7A Data health")
    snap = _load_snap()
    df, _meta = load_signals()

    st.markdown("##### Snapshot")
    if not snap:
        st.warning("No snapshot manifest — the app is on demo/mock data.")
    else:
        created = snap.get("created", "")
        age_txt, age_h = "—", None
        try:
            dt = datetime.strptime(str(created)[:19], "%Y-%m-%d %H:%M:%S")
            age_h = (datetime.now() - dt).total_seconds() / 3600
            age_txt = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h / 24:.1f} days ago"
        except Exception:
            pass
        m = st.columns(4)
        m[0].metric("Source", str(snap.get("source", "?")))
        m[1].metric("Settle date", str(snap.get("as_of", "?")))
        m[2].metric("Pulled", age_txt)
        m[3].metric("Tickers", str(snap.get("n_tickers", "—")))
        m2 = st.columns(4)
        m2[0].metric("IV markets", str(snap.get("iv_markets", "—")))
        m2[1].metric("OI chains", str(snap.get("oi_markets", "—")))
        m2[2].metric("Price rows", str(snap.get("price_rows", "—")))
        m2[3].metric("Live quotes", str(snap.get("live_n", "—")))
        if snap.get("source") != "bloomberg":
            st.warning(f"Snapshot source is **{snap.get('source', '?')}** (demo) — pull a live "
                       "Bloomberg snapshot on Home for real data.")
        elif age_h is not None and age_h > 24:
            st.warning(f"Snapshot is ~{age_txt} — pull a fresh one on Home for today's data.")
        else:
            st.success("Snapshot looks current.")

    st.markdown("##### Coverage by strategy")
    if df is None or df.empty or "strategy" not in df:
        st.caption("No signals computed yet — re-run signals on Home.")
    else:
        df = _filter_signals(df)
        uni = len(universe.enabled_tickers())
        rows = []
        for strat in STRATEGY_ORDER:
            if strat == "Open Interest":            # a monitor/report, not a scored strategy
                continue
            sub = df[df["strategy"] == strat]
            n = len(sub)
            flagged = int(sub["signal"].ne("—").sum()) if "signal" in sub else 0
            rows.append({"Strategy": strat, "Markets scored": n,
                         "Coverage": f"{n} pairs" if strat == "Mean Reversion" else f"{n}/{uni}",
                         "Flagged": flagged})
        brand.themed_dataframe(pd.DataFrame(rows), {})
        st.caption(f"Universe = **{uni}** instruments. A low *Markets scored* vs the universe means "
                   "that strategy is missing data for some products (usually a thin/absent vol "
                   "surface, or no CFTC / listed-options series).")

    st.markdown("##### Known coverage gaps")
    st.markdown(
        "- **Put/Call** — Bloomberg option OI/volume covers ~55/84; absent: 14 FX (OTC), "
        "9 cash-index futures (DAX/CAC/Euro Stoxx/FTSE/Nikkei/KOSPI/ASX/Dow/SMI), 3 STIRs "
        "(1M SOFR / Fed Funds / 3M ESTR), Ethanol / EU Carbon, COMEX Aluminium.\n"
        "- **Skew / Vol term** — FX uses the OTC vol surface; bonds & STIRs are excluded by design.\n"
        "- **COT** — CFTC-listed US contracts only; no ICE-Europe, Eurex bonds, EU/APAC indices, "
        "Euribor / SONIA, minor FX, or LME metals."
    )


def render_market_hours() -> None:
    """A Gantt-style timeline of every product's trading session on a 24h axis, in a chosen
    reference timezone, with a 'now' line. Gold = the liquid window inside the full session."""
    import altair as alt
    from datetime import datetime
    from zoneinfo import ZoneInfo
    st.subheader("🕒 Market hours")

    labels = list(markethours.TZ_CHOICES.keys())
    c1, c2 = st.columns([0.4, 0.6])
    pick = c1.selectbox("Reference time zone", labels,
                        index=labels.index("New York (ET)"), key="mh_tz")
    ref_tz = markethours.TZ_CHOICES[pick]
    search = c2.text_input("Find a product", key="mh_search",
                           placeholder="type a name, ticker or sector — e.g. oil, CLA, metals").strip()
    apply_filter = st.checkbox("Show only the sectors enabled on Home", value=False, key="mh_filter")

    now = datetime.now(ZoneInfo(ref_tz))
    now_h = now.hour + now.minute / 60
    ref_date = now.date()

    tickers = list(INSTRUMENTS)
    if apply_filter and universe.filter_active():
        en = universe.enabled_tickers()
        tickers = [t for t in tickers if t in en]
    if search:                                  # free-text find: name / ticker / sector / region / alias
        tickers = prodsearch.filter_tickers(tickers, INSTRUMENTS, search)
        if not tickers:
            st.info(prodsearch.NO_MATCH.format(q=search))
            return
    _aorder = [a for a in universe.ASSET_CLASSES if a != "FX"] + ["FX"]   # FX at the bottom
    order = {a: i for i, a in enumerate(_aorder)}
    tickers.sort(key=lambda t: (order.get(INSTRUMENTS[t][2], 99), INSTRUMENTS[t][0]))
    if not tickers:
        st.info("No products selected — turn some sectors back on, or untick the filter above.")
        return

    label_of = _sf_labeler(tickers)
    y_order = [label_of(t) for t in tickers]
    rows, settle_rows, closed_rows, open_n, closed_n, half_n = [], [], [], 0, 0, 0
    for t in tickers:
        asset = INSTRUMENTS[t][2]
        seg = markethours.day_segments(t, ref_tz, ref_date, asset)
        lbl = label_of(t)
        exch_tz_short = seg["exch_tz"].split("/")[-1].replace("_", " ")
        if seg["closed"]:                       # full holiday — show a greyed "Closed" row
            closed_n += 1
            closed_rows.append({"mkt": lbl, "x0": 0.0, "x1": 24.0, "xmid": 12.0,
                                "note": f'Closed — {seg["closed"]}', "exch": seg["exchange"]})
            continue
        opn = markethours.is_open(seg["full"], now_h)
        open_n += int(opn)
        half = seg["half_day"]
        half_n += int(bool(half))
        status = ("open now" if opn else "closed now") + (f' · half-day ({half})' if half else '')
        half_txt = f' · closes early {seg["early_close"]} ({exch_tz_short})' if half else ''
        lcol = markethours.ASSET_LIQUID.get(asset, markethours.ASSET_COLORS.get(asset, "#888"))
        exp = expiries.describe(t, asset, ref_date)      # indicative next futures / options expiry
        fut_exp = (f'{exp["fut"]} · {exp["fut_time"]}' if exp["fut"]
                   else "cash index — options only" if exp["cash"] else "—")
        opt_exp = f'{exp["opt"]} · {exp["opt_time"]}' if exp["opt"] else "—"
        for a, b in seg["full"]:
            rows.append({"mkt": lbl, "asset": asset, "start": a, "end": b, "kind": "Full session",
                         "lcolor": lcol, "exch": seg["exchange"],
                         "hours": f'{seg["full_local"]} ({exch_tz_short}){half_txt}', "et": f'{seg["full_et"]} ET',
                         "status": status, "fut_exp": fut_exp, "opt_exp": opt_exp})
        # The liquid window may sit on a different venue (e.g. iron ore → Dalian) — label it.
        diff_venue = seg["liquid_exch"] != seg["exchange"]
        liq_hours = (f'{seg["liquid_local"]} · {seg["liquid_exch"]}' if diff_venue
                     else f'{seg["liquid_local"]} ({exch_tz_short})')
        for a, b in seg["liquid"]:
            rows.append({"mkt": lbl, "asset": asset, "start": a, "end": b, "kind": "Liquid window",
                         "lcolor": lcol, "exch": (seg["liquid_exch"] if diff_venue else seg["exchange"]),
                         "hours": liq_hours, "et": f'{seg["liquid_et"]} ET', "status": status,
                         "fut_exp": fut_exp, "opt_exp": opt_exp})
        if seg["settle"] is not None:
            settle_rows.append({"mkt": lbl, "settle": seg["settle"],
                                "settle_txt": f'{seg["settle_local"]} {exch_tz_short} · {seg["settle_et"]} ET'})
    df = pd.DataFrame(rows)
    df_settle = pd.DataFrame(settle_rows)
    df_closed = pd.DataFrame(closed_rows)

    extra = ""
    if closed_n:
        extra += f" **{closed_n} closed** for holidays."
    if half_n:
        extra += f" **{half_n} on a half-day** (bar truncated to the early close)."
    st.caption(f"Indicative regular-session hours, shown in **{pick}** for "
               f"**{ref_date:%a %d %b}**. Bars are **coloured by sector** (key above); the **solid** part is "
               f"the liquid window, the faded part the rest of the electronic session. The **white tick │** is "
               f"the daily **settlement**; the red line is **now** — **{open_n}/{len(tickers) - closed_n} open** "
               f"({now:%H:%M} {pick}).{extra} Sessions crossing midnight wrap to the next line.")

    xaxis = alt.X("start:Q", scale=alt.Scale(domain=[0, 24], nice=False),
                  axis=alt.Axis(title=None, values=list(range(0, 25, 2)),
                                labelExpr="(datum.value<10?'0':'')+datum.value+':00'"))
    yaxis = alt.Y("mkt:N", sort=y_order, axis=alt.Axis(title=None, labelFontSize=9, labelLimit=200))
    cscale = alt.Scale(domain=_aorder,
                       range=[markethours.ASSET_COLORS.get(a, "#888") for a in _aorder])
    tip = [alt.Tooltip("mkt:N", title="Product"), alt.Tooltip("exch:N", title="Exchange"),
           alt.Tooltip("kind:N", title="Bar"), alt.Tooltip("hours:N", title="Local (exchange)"),
           alt.Tooltip("et:N", title="Eastern (ET)"), alt.Tooltip("status:N", title="Status"),
           alt.Tooltip("fut_exp:N", title="Next future exp."), alt.Tooltip("opt_exp:N", title="Next option exp.")]
    base = alt.Chart(df)
    full = base.transform_filter(alt.datum.kind == "Full session").mark_bar(opacity=0.4).encode(
        x=xaxis, x2="end:Q", y=yaxis,
        color=alt.Color("asset:N", scale=cscale, legend=alt.Legend(title="Sector", orient="top", columns=8)),
        tooltip=tip)
    liquid = base.transform_filter(alt.datum.kind == "Liquid window").mark_bar().encode(
        x="start:Q", x2="end:Q", y=yaxis, color=alt.Color("lcolor:N", scale=None, legend=None), tooltip=tip)
    layers = []
    if not df_closed.empty:                     # faint grey band + italic note for shut markets
        ctip = [alt.Tooltip("mkt:N", title="Product"), alt.Tooltip("exch:N", title="Exchange"),
                alt.Tooltip("note:N", title="Status")]
        layers.append(alt.Chart(df_closed).mark_bar(color="#9AA0A6", opacity=0.12).encode(
            x="x0:Q", x2="x1:Q", y=yaxis, tooltip=ctip))
        layers.append(alt.Chart(df_closed).mark_text(color="#9AA0A6", fontSize=9, fontStyle="italic").encode(
            x="xmid:Q", y=yaxis, text="note:N", tooltip=ctip))
    layers += [full, liquid]
    if not df_settle.empty:
        stip = [alt.Tooltip("mkt:N", title="Product"), alt.Tooltip("settle_txt:N", title="Settlement")]
        layers.append(alt.Chart(df_settle).mark_tick(color="#000", thickness=4, size=16).encode(
            x="settle:Q", y=yaxis, tooltip=stip))      # dark halo so the tick shows on any colour
        layers.append(alt.Chart(df_settle).mark_tick(color="#FFFFFF", thickness=2, size=16).encode(
            x="settle:Q", y=yaxis, tooltip=stip))
    layers.append(alt.Chart(pd.DataFrame({"x": [now_h]})).mark_rule(color="#E53935", size=2).encode(x="x:Q"))
    chart = alt.layer(*layers).properties(height=max(320, 19 * len(tickers)))
    brand.show_chart(chart)
    st.caption("Hours, settlement times **and the holiday/half-day calendar** are indicative — "
               "tell me any you'd like corrected and I'll adjust `src/markethours.py`.")


def render_block_sizes() -> None:
    """The whole book with each exchange's minimum block-trade size for the futures and
    the listed options. Values are editable and persist to data/blocksizes.json."""
    st.subheader("📦 Minimum block sizes")

    c1, c2 = st.columns([0.55, 0.45])
    apply_filter = c1.checkbox("Show only the sectors enabled on Home", value=False, key="bs_filter")

    # The book, ordered like Market Hours (sector order, FX last); cash indices are
    # vol sources, not tradable lines, so they don't get a block-size row.
    tickers = [t for t in INSTRUMENTS if t not in universe.PRICE_FIELD_OVERRIDE]
    if apply_filter and universe.filter_active():
        en = universe.enabled_tickers()
        tickers = [t for t in tickers if t in en]
    tickers, _q = prodsearch.search_box(tickers, INSTRUMENTS, key="bs_search", container=c2)
    if _q and not tickers:
        st.info(prodsearch.NO_MATCH.format(q=_q))
        return
    _aorder = [a for a in universe.ASSET_CLASSES if a != "FX"] + ["FX"]
    order = {a: i for i, a in enumerate(_aorder)}
    tickers.sort(key=lambda t: (order.get(INSTRUMENTS[t][2], 99), INSTRUMENTS[t][0]))
    if not tickers:
        st.info("No products selected — turn some sectors back on, or untick the filter above.")
        return

    bmap = blocksizes.load_map()
    label_of = _sf_labeler(tickers)
    rows = []
    for t in tickers:
        e = bmap.get(t, {})
        rows.append({"sector": INSTRUMENTS[t][2], "product": label_of(t), "ticker": t,
                     "exchange": markethours.exchange_of(t, INSTRUMENTS[t][2]),
                     "fut": e.get("fut", ""), "opt": e.get("opt", ""),
                     "strat": e.get("strat", ""), "note": e.get("note", "")})
    df = pd.DataFrame(rows)

    n_missing = int((df["fut"].str.strip() == "").sum())
    st.caption(
        "Exchange **minimum block-trade sizes** (lots) for each product's futures and listed "
        "options — CME Rule 526 thresholds, ICE block/EFRP minimums, Eurex TES sizes, etc. "
        "CME rates/FX/equity minimums **vary by time of day** (RTH / ETH / ATH = regular / "
        "European / Asian hours, Chicago time) and some venues vary by contract month — the "
        "notes column carries those wrinkles. The **Strategies** column says how the minimum "
        "applies to spread/combination blocks — the **sum of the legs** vs **each leg** "
        "individually. Covered / volatility blocks (options vs futures) are the common "
        "exception everywhere: only the **options leg** has to meet the options minimum, "
        "with the futures leg sized to the delta. Figures are from the exchanges' published "
        "rules; **verify against the rulebook before quoting a client**, and edit any cell "
        "below to correct it (saved to `data/blocksizes.json`)."
        + (f" **{n_missing} product(s) have no figure yet** — fill them in as you confirm them." if n_missing else "")
    )

    edited = st.data_editor(
        df, use_container_width=True, height=min(1400, 42 + 35 * len(df)), hide_index=True,
        key="blocksizes_editor",
        disabled=["sector", "product", "ticker", "exchange"],
        column_config={
            "sector": st.column_config.TextColumn("Sector", width="small"),
            "product": st.column_config.TextColumn("Product"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "exchange": st.column_config.TextColumn("Exchange", width="small"),
            "fut": st.column_config.TextColumn(
                "Futures min", help="Minimum block size for the futures, in lots. "
                "RTH/ETH/ATH splits are written out, e.g. '500 RTH / 250 ETH / 125 ATH'."),
            "opt": st.column_config.TextColumn(
                "Options min", help="Minimum block size for the listed options, in lots."),
            "strat": st.column_config.TextColumn(
                "Strategies", help="How the minimum applies to spread/combination blocks: "
                "does the SUM of the legs have to meet it, or EACH leg individually?"),
            "note": st.column_config.TextColumn("Notes", width="large"),
        },
    )
    bc1, bc2 = st.columns([1, 4])
    if bc1.button("💾 Save block sizes", type="primary", key="save_blocksizes_btn"):
        blocksizes.save_map({r["ticker"]: {"fut": str(r.get("fut") or "").strip(),
                                           "opt": str(r.get("opt") or "").strip(),
                                           "strat": str(r.get("strat") or "").strip(),
                                           "note": str(r.get("note") or "").strip()}
                             for r in edited.to_dict("records")})
        st.toast("Block sizes saved.", icon="✅")
    bc2.caption("Edits persist across restarts and survive universe changes; a product added "
                "on the Universe page appears here with blank cells until you fill it in.")


def render_fut_yield() -> None:
    """Futures price ⇄ yield converter — STIRs via the IMM convention (100 − price),
    bond futures read through the CTD (price × conversion factor → solve the yield).
    Two sub-views (STIRs / Bonds) behind their own switcher row under the group tabs.
    Engine: src/futyield.py; CTD assumptions editable, persisted to data/futyield.json."""
    # ---- sub-page switcher (sits right under the Market Information tab row) ----
    view = st.session_state.setdefault("fy_view", "STIRs")
    vc1, vc2, vc3 = st.columns(3)
    if vc1.button("💵 STIRs", use_container_width=True, key="fy_tab_stirs",
                  type="primary" if view == "STIRs" else "secondary"):
        st.session_state["fy_view"] = view = "STIRs"; st.rerun()
    if vc2.button("🏦 Bonds", use_container_width=True, key="fy_tab_bonds",
                  type="primary" if view == "Bonds" else "secondary"):
        st.session_state["fy_view"] = view = "Bonds"; st.rerun()
    if vc3.button("⚖️ DV01", use_container_width=True, key="fy_tab_dv01",
                  type="primary" if view == "DV01" else "secondary"):
        st.session_state["fy_view"] = view = "DV01"; st.rerun()

    st.subheader("🧮 Fut / Yield — futures price ⇄ yield")
    st.caption(
        "Turn a futures price into the yield it implies (and back). **STIRs** are exact by "
        "construction: the IMM quote is 100 − rate, so the result is the money-market forward "
        "rate for the contract's window. **Bond futures** have no yield of their own — the "
        "market reads them through the cheapest-to-deliver: forward CTD price ≈ futures × "
        "conversion factor, and the futures' \"yield\" is the CTD's yield at that price. "
        "The two are **different conventions** (add-on ACT/360-365 vs semi/annual bond "
        "compounding) — compare each to its own curve, not to each other.")

    _fy_tks = list(futyield.STIRS) + list(futyield.load_ctd())
    try:
        _live = get_live_quote(_fy_tks)
    except Exception:
        _live = pd.DataFrame()

    def _fy_px(tk: str) -> float:
        try:
            return float(_live.loc[tk, "last"])
        except Exception:
            return float(INSTRUMENTS[tk][1])

    def _fy_chg(tk: str):
        try:
            return float(_live.loc[tk, "net"])
        except Exception:
            return float("nan")

    _src = {"bloomberg": "live Bloomberg", "snapshot": "snapshot"}.get(MODE, "demo")

    # ================= STIRs sub-page =========================================
    if view == "STIRs":
        st.markdown("#### 💵 Short-term interest rates — rate = 100 − price")
        stir_rows = []
        for tk, (idx, tenor, ccy, dc) in futyield.STIRS.items():
            px, net = _fy_px(tk), _fy_chg(tk)
            stir_rows.append({
                "Contract": INSTRUMENTS[tk][0], "Index": idx, "Tenor": tenor,
                "Quote": ccy + " · " + dc, "Price": px,
                "Implied rate %": futyield.stir_rate(px),
                "Δ day (bp)": -net * 100.0 if net == net else None,
                f"{ccy}/bp per lot": volbt.point_value(tk) / 100.0,
            })
        st.dataframe(pd.DataFrame(stir_rows), hide_index=True, use_container_width=True,
                     column_config={
                         "Price": st.column_config.NumberColumn(format="%.4f"),
                         "Implied rate %": st.column_config.NumberColumn(format="%.4f"),
                         "Δ day (bp)": st.column_config.NumberColumn(
                             format="%.1f", help="Change in the implied rate since the prior "
                             "settle — the price move with the sign flipped."),
                     })
        st.caption(f"Prices: {_src}. The implied rate is the **forward** rate for each contract's "
                   "reference window (compounded-in-arrears for SOFR/SONIA/ESTR), not today's spot fixing.")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Price → rate**")
            _p = st.number_input("Futures price", value=96.0000, step=0.0025, format="%.4f",
                                 key="fy_stir_p2r")
            st.metric("Implied rate", f"{futyield.stir_rate(_p):.4f} %")
        with s2:
            st.markdown("**Rate → price**")
            _r = st.number_input("Rate (%)", value=4.0000, step=0.0025, format="%.4f",
                                 key="fy_stir_r2p")
            st.metric("Futures price", f"{futyield.stir_price(_r):.4f}")
        with s3:
            st.markdown("**Move → P&L**")
            _ctk = st.selectbox("Contract", list(futyield.STIRS),
                                format_func=lambda t: INSTRUMENTS[t][0], key="fy_stir_pnl_tk")
            _bp = st.number_input("Move (bp)", value=1.0, step=0.25, format="%.2f", key="fy_stir_bp")
            _lots = st.number_input("Lots", value=100, step=10, min_value=1, key="fy_stir_lots")
            _ccy = futyield.STIRS[_ctk][2]
            st.metric("P&L", f"{_bp * _lots * volbt.point_value(_ctk) / 100.0:,.0f} {_ccy}",
                      help="1bp in rate = 0.01 price points; P&L per lot per bp = point value ÷ 100.")
        return

    # ================= DV01 sub-page ==========================================
    if view == "DV01":
        st.markdown("#### ⚖️ DV01 — value of a 1bp yield move, per lot")
        _ctd = futyield.load_ctd()
        dv01_of, ccy_of, dv_rows = {}, {}, []
        for tk, (idx, tenor, ccy, dc) in futyield.STIRS.items():
            d = volbt.point_value(tk) / 100.0
            dv01_of[tk], ccy_of[tk] = d, ccy
            dv_rows.append({"Contract": INSTRUMENTS[tk][0], "Type": "STIR",
                            "Price": _fy_px(tk), "DV01/lot": d, "Ccy": ccy,
                            "Basis": "Fixed by design — 1bp = 0.01 price points"})
        for tk, e in _ctd.items():
            px = _fy_px(tk)
            d = futyield.fut_dv01(px, e["cf"], e["coupon"], e["years"], int(e["freq"]),
                                  volbt.point_value(tk))
            dv01_of[tk], ccy_of[tk] = d, volbt.currency(tk)
            dv_rows.append({"Contract": INSTRUMENTS[tk][0], "Type": "Bond",
                            "Price": px, "DV01/lot": d, "Ccy": ccy_of[tk],
                            "Basis": "CTD DV01 ÷ CF at the current price"})
        st.dataframe(pd.DataFrame(dv_rows), hide_index=True, use_container_width=True,
                     height=42 + 35 * len(dv_rows),
                     column_config={
                         "Price": st.column_config.NumberColumn(format="%.4f"),
                         "DV01/lot": st.column_config.NumberColumn(
                             format="%.2f", help="Contract-currency P&L of a 1bp move in the "
                             "contract's yield, per lot."),
                     })
        st.caption(f"Prices: {_src}. STIR DV01s are constant; bond DV01s move with the market "
                   "and lean on the **CTD assumptions** on the Bonds tab — keep those current. "
                   "Each figure is in the contract's own currency.")

        _all = list(futyield.STIRS) + list(_ctd)
        _lbl = lambda t: INSTRUMENTS[t][0]
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Position → DV01**")
            _ptk = st.selectbox("Contract", _all, format_func=_lbl, key="fy_dv_pos_tk")
            _plots = st.number_input("Lots", value=100, step=10, min_value=1, key="fy_dv_pos_lots")
            st.metric("Position DV01", f"{_plots * dv01_of[_ptk]:,.0f} {ccy_of[_ptk]}/bp")
        with d2:
            st.markdown("**Target DV01 → lots**")
            _ttk = st.selectbox("Contract", _all, format_func=_lbl, key="fy_dv_tgt_tk")
            _tgt = st.number_input(f"Target DV01 ({ccy_of[_ttk]}/bp)", value=10000.0,
                                   step=500.0, min_value=0.0, format="%.0f", key="fy_dv_tgt")
            _n = _tgt / dv01_of[_ttk] if dv01_of[_ttk] else 0.0
            st.metric("Lots", f"{_n:,.1f}", help="Round to taste — the figure is exact, "
                                                 "the market trades whole lots.")
        with d3:
            st.markdown("**DV01-neutral hedge ratio**")
            _atk = st.selectbox("Leg A", _all, format_func=_lbl, key="fy_dv_ha",
                                index=_all.index("TUA Comdty") if "TUA Comdty" in _all else 0)
            _btk = st.selectbox("Leg B", _all, format_func=_lbl, key="fy_dv_hb",
                                index=_all.index("TYA Comdty") if "TYA Comdty" in _all else 0)
            _alots = st.number_input("Lots of A", value=100, step=10, min_value=1, key="fy_dv_hlots")
            _blots = _alots * dv01_of[_atk] / dv01_of[_btk] if dv01_of[_btk] else 0.0
            st.metric("Lots of B", f"{_blots:,.1f}",
                      help="Lots of B carrying the same DV01 as the A position — the ratio for "
                           "a curve trade with no outright duration.")
            if ccy_of[_atk] != ccy_of[_btk]:
                st.warning(f"Legs are in different currencies ({ccy_of[_atk]} vs {ccy_of[_btk]}) "
                           "— the ratio ignores FX; convert one leg's DV01 before sizing.",
                           icon="⚠️")
        return

    # ================= Bonds sub-page =========================================
    st.markdown("#### 🏦 Bond futures — yield read through the CTD")
    ctd = futyield.load_ctd()

    with st.expander("CTD assumptions (edit from the delivery-basket / DLV screens)"):
        st.caption(
            "Seeded with **indicative placeholders** — keep the coupon, maturity and conversion "
            "factor of the actual CTD current from the exchange delivery basket before quoting "
            "anything off this page. Set **CF = 0** to have it recomputed from the coupon / "
            "maturity / notional. Edits persist to `data/futyield.json`.")
        _ctd_df = pd.DataFrame([{"ticker": tk, "product": INSTRUMENTS[tk][0],
                                 "coupon": e["coupon"], "years": e["years"],
                                 "freq": int(e["freq"]), "notional": e["notional"],
                                 "cf": e["cf"]}
                                for tk, e in ctd.items()])
        _ed = st.data_editor(
            _ctd_df, hide_index=True, use_container_width=True, key="fy_ctd_editor",
            disabled=["ticker", "product"],
            column_config={
                "product": st.column_config.TextColumn("Product"),
                "coupon": st.column_config.NumberColumn("CTD coupon %", format="%.3f", step=0.125),
                "years": st.column_config.NumberColumn("CTD maturity (yrs)", format="%.1f", step=0.1),
                "freq": st.column_config.NumberColumn("Cpns/yr", min_value=1, max_value=2, step=1),
                "notional": st.column_config.NumberColumn("Notional cpn %", format="%.1f",
                                                          help="6% CME & Eurex, 4% Long Gilt."),
                "cf": st.column_config.NumberColumn("Conversion factor", format="%.4f",
                                                    help="0 = recompute from the CTD terms."),
            })
        if st.button("💾 Save CTD assumptions", key="fy_ctd_save"):
            out = {}
            for r in _ed.to_dict("records"):
                e = {"coupon": float(r["coupon"]), "years": float(r["years"]),
                     "freq": int(r["freq"]), "notional": float(r["notional"])}
                _auto = round(futyield.conversion_factor(e["coupon"], e["years"],
                                                         e["freq"], e["notional"]), 4)
                _cf = float(r["cf"] or 0.0)
                if _cf > 0 and abs(_cf - _auto) > 5e-5:   # only store a genuine override
                    e["cf"] = _cf
                out[r["ticker"]] = e
            futyield.save_ctd(out)
            st.toast("CTD assumptions saved.", icon="✅")
            st.rerun()

    bond_rows = []
    for tk, e in ctd.items():
        px = _fy_px(tk)
        _pv = volbt.point_value(tk)
        bond_rows.append({
            "Contract": INSTRUMENTS[tk][0], "Price": px, "CF": e["cf"],
            "Fwd CTD px": px * e["cf"],
            "Implied CTD yield %": futyield.ctd_yield(px, e["cf"], e["coupon"],
                                                      e["years"], int(e["freq"])),
            f"DV01/lot": futyield.fut_dv01(px, e["cf"], e["coupon"], e["years"],
                                           int(e["freq"]), _pv),
            "Ccy": volbt.currency(tk),
        })
    st.dataframe(pd.DataFrame(bond_rows), hide_index=True, use_container_width=True,
                 column_config={
                     "Price": st.column_config.NumberColumn(format="%.4f",
                                                            help="Decimal, not 32nds."),
                     "CF": st.column_config.NumberColumn(format="%.4f"),
                     "Fwd CTD px": st.column_config.NumberColumn(format="%.4f"),
                     "Implied CTD yield %": st.column_config.NumberColumn(format="%.3f"),
                     "DV01/lot": st.column_config.NumberColumn(
                         format="%.2f", help="Contract-currency value of a 1bp move in the "
                         "CTD yield, per lot: CTD DV01 ÷ CF × point value."),
                 })
    st.caption(f"Prices: {_src}. Standard fixed-coupon price/yield relation on the CTD — good "
               "to well under a bp for conversion, but it is **not** a delivery-option model; "
               "the CTD can switch when the curve moves.")

    b1, b2, b3 = st.columns(3)
    _btk = b1.selectbox("Contract", list(ctd), format_func=lambda t: INSTRUMENTS[t][0],
                        key="fy_bond_tk")
    _e = ctd[_btk]
    with b1:
        st.markdown("**Price → yield**")
        _bp_in = st.number_input("Futures price", value=float(round(_fy_px(_btk), 4)),
                                 step=0.01, format="%.4f", key=f"fy_bond_p2y_{_btk}")
        st.metric("Implied CTD yield",
                  f"{futyield.ctd_yield(_bp_in, _e['cf'], _e['coupon'], _e['years'], int(_e['freq'])):.3f} %")
    with b2:
        st.markdown("**Yield → price**")
        _y_in = st.number_input(
            "Target CTD yield (%)",
            value=float(round(futyield.ctd_yield(_fy_px(_btk), _e["cf"], _e["coupon"],
                                                 _e["years"], int(_e["freq"])), 3)),
            step=0.01, format="%.3f", key=f"fy_bond_y2p_{_btk}")
        st.metric("Futures price",
                  f"{futyield.fut_price_from_yield(_y_in, _e['cf'], _e['coupon'], _e['years'], int(_e['freq'])):.4f}")
    with b3:
        st.markdown("**Move → P&L**")
        _mbp = st.number_input("Yield move (bp)", value=1.0, step=0.25, format="%.2f",
                               key="fy_bond_bp")
        _mlots = st.number_input("Lots", value=100, step=10, min_value=1, key="fy_bond_lots")
        _dv = futyield.fut_dv01(_fy_px(_btk), _e["cf"], _e["coupon"], _e["years"],
                                int(_e["freq"]), volbt.point_value(_btk))
        st.metric("P&L", f"{_mbp * _mlots * _dv:,.0f} {volbt.currency(_btk)}",
                  help="Yield move × futures DV01 × lots. A yield FALL is a price RISE.")
        st.caption(f"≈ {_mbp * _dv / volbt.point_value(_btk):.4f} price points per lot.")


@st.cache_data(ttl=60, show_spinner=False)
def _report_states_cached():
    return automation.all_report_states()


def _email_toggle_cb(ekey: str, nrec: int, label: str) -> None:
    """on_change for an auto-email toggle. Turning ON needs ≥1 recipient (else revert + warn);
    both directions persist to the automation flag + Windows task and clear the state cache."""
    want = bool(st.session_state.get(f"al_em_{ekey}"))
    if want and nrec == 0:
        st.session_state[f"al_em_{ekey}"] = False          # revert the flipped toggle
        st.session_state["_al_msg"] = ("warn", f"Add at least one recipient for **{label}** below "
                                               "before switching its automatic email on.")
        return
    automation.set_report_enabled(ekey, want)
    _report_states_cached.clear()
    st.session_state["_al_msg"] = ("ok", f"Automatic email {'ON' if want else 'OFF'} — {label}.")


def _alert_toggle_cb(key: str, kind: str) -> None:
    """on_change for a banner/popup toggle — persist the one flag."""
    wkey = f"al_{'bn' if kind == 'banner' else 'pp'}_{key}"
    alerts.set_alert_flag(key, kind, bool(st.session_state.get(wkey)))


# ── Technical Analysis Report — configurable schedule (FICC + Equities) ─────────────────────
# Unlike every other auto-email (a fixed schedule baked into its Windows task, just on/off
# here), the TA report's frequency AND time are user-configurable. The ⏰ popover below
# replaces the plain toggle for these two rows only; Save creates/updates the Windows task
# (WEEKLY on Monday, or WEEKLY on all 5 weekdays) via convreport_scheduled_email.py --scope,
# then reuses automation.set_report_enabled for the same on/off gate every other report uses.
_TA_SCHEDULE_FILE = ROOT / "data" / "ta_report_schedule.json"
_TA_REPORT_CLI = ROOT / "convreport_scheduled_email.py"
_TA_SCOPE_OF = {"convreport": "ficc", "eq_convreport": "equities"}


def _ta_schedule_cfg(ekey: str) -> dict:
    try:
        d = json.loads(_TA_SCHEDULE_FILE.read_text(encoding="utf-8-sig"))
        c = d.get(ekey) or {}
    except Exception:
        c = {}
    return {"frequency": c.get("frequency", "weekly"), "time": c.get("time", "07:30")}


def _ta_schedule_apply(ekey: str, on: bool, frequency: str, hhmm: str) -> tuple[bool, str]:
    task_name = automation.REPORTS[ekey]["tasks"][0]
    days = "MON" if frequency == "weekly" else "MON,TUE,WED,THU,FRI"
    tr = f'"{sys.executable}" "{_TA_REPORT_CLI}" --scope {_TA_SCOPE_OF[ekey]}'
    r = subprocess.run(["schtasks", "/Create", "/F", "/TN", task_name, "/TR", tr,
                        "/SC", "WEEKLY", "/D", days, "/ST", hhmm],
                       capture_output=True, text=True)
    err = (r.stderr or r.stdout or "").strip()
    if r.returncode != 0:
        return False, err
    automation.set_report_enabled(ekey, on)        # enable/disable the task + the flag file
    d = {}
    try:
        d = json.loads(_TA_SCHEDULE_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        pass
    d[ekey] = {"frequency": frequency, "time": hhmm}
    _TA_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TA_SCHEDULE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return True, ""


def _ta_schedule_control(col, ekey: str, label: str, nrec: int) -> None:
    is_on = _report_states_cached().get(ekey) == "on"
    cfg = _ta_schedule_cfg(ekey)
    btn_lbl = (f"⏰ {'Weekly' if cfg['frequency'] == 'weekly' else 'Daily'} · {cfg['time']}"
               if is_on else "⏰ Off")
    with col.popover(btn_lbl, use_container_width=True,
                     help=f"Schedule an automatic {label} email — daily or weekly, any time. "
                          "Runs even when BASIS is closed, via Windows Task Scheduler."):
        _freq_lbl = st.radio("Frequency", ["Weekly (Monday)", "Daily (weekdays)"],
                             index=0 if cfg["frequency"] == "weekly" else 1, key=f"ta_freq_{ekey}")
        _cur_t = dtime(*(int(x) for x in cfg["time"].split(":")))
        _t = st.time_input("Send at (laptop time)", value=_cur_t, step=300, key=f"ta_time_{ekey}")
        _on = st.toggle("Automatic email on", value=is_on, key=f"ta_on_{ekey}")
        if st.button("Save", key=f"ta_save_{ekey}", type="primary", use_container_width=True):
            if _on and nrec == 0:
                st.error(f"Add at least one recipient for **{label}** below before switching this on.")
            else:
                freq = "weekly" if "Weekly" in _freq_lbl else "daily"
                ok, err = _ta_schedule_apply(ekey, _on, freq, _t.strftime("%H:%M"))
                if ok:
                    _report_states_cached.clear()
                    st.session_state["_al_msg"] = (
                        "ok", f"Saved — {label}: "
                              + (f"ON, {freq}, {_t.strftime('%H:%M')}." if _on else "off."))
                    st.rerun()
                else:
                    st.error(f"Couldn't update the scheduled task:\n\n{err or 'no output'}")
        st.caption("The laptop must be on (and not asleep) at send time. Runs the same report "
                   "build as the page above, using whatever's saved as its default there.")


def _render_alert_settings() -> None:
    """Per-report control of the three alerts: 📧 automatic email, 🚩 the Home 'REPORT DAY' banner,
    and 🔔 the release-time popup. Email defaults OFF and needs a recipient; banner & popup default
    ON. '—' = not applicable for that report; 'n/a' = no scheduled email task on this PC."""
    st.markdown("#### 🔔  Alerts — what fires for each report")
    st.caption("For every report, choose how you're alerted: an **automatic email** on its schedule, "
               "a **banner** on the Home page on release day, and a full-screen **popup** at the "
               "release time. Banner & popup are on‑screen only and on by default; automatic email is "
               "off until you switch it on.")
    m = st.session_state.pop("_al_msg", None)
    if m:
        (st.warning if m[0] == "warn" else st.success)(m[1])
    try:
        email_states = _report_states_cached()
    except Exception:
        email_states = {}
    data = recipients.load_all()

    hdr = st.columns([0.40, 0.20, 0.20, 0.20])
    for col, txt in zip(hdr, ["**Report**", "**📧 Auto‑email**", "**🚩 Home banner**", "**🔔 Popup**"]):
        col.markdown(txt)

    last_group = None
    for key, meta in alerts.ALERT_REPORTS.items():
        if meta["group"] != last_group:
            st.markdown(f"<div style='color:#8a8f98;font-size:.70rem;letter-spacing:.07em;"
                        f"text-transform:uppercase;margin:.5rem 0 .05rem'>{meta['group']}</div>",
                        unsafe_allow_html=True)
            last_group = meta["group"]
        c = st.columns([0.40, 0.20, 0.20, 0.20])
        c[0].markdown(f"<div style='padding-top:.35rem'>{meta['label']}</div>", unsafe_allow_html=True)

        ekey = meta["email"]
        if not ekey:
            c[1].markdown("<div style='padding-top:.35rem;color:#8a8f98'>—</div>", unsafe_allow_html=True)
        elif ekey in _TA_SCOPE_OF:
            # Configurable schedule (frequency + time), not a plain toggle — creates its own
            # Windows task on Save, so it's never "n/a" even before one exists.
            nrec = len(data.get(automation.REPORTS[ekey]["recipients"], []))
            _ta_schedule_control(c[1], ekey, meta["label"], nrec)
        elif email_states.get(ekey, "missing") == "missing":
            c[1].markdown("<div style='padding-top:.35rem;color:#8a8f98'>n/a</div>", unsafe_allow_html=True)
        else:
            nrec = len(data.get(automation.REPORTS[ekey]["recipients"], []))
            st.session_state[f"al_em_{ekey}"] = email_states.get(ekey) == "on"
            c[1].toggle("auto-email", key=f"al_em_{ekey}", label_visibility="collapsed",
                        on_change=_email_toggle_cb, args=(ekey, nrec, meta["label"]),
                        help="Email this report automatically on its schedule. Sends real emails — "
                             "needs at least one recipient set below.")

        if not meta["alerts"] and not meta.get("banner_only"):
            c[2].markdown("<div style='padding-top:.35rem;color:#8a8f98'>—</div>", unsafe_allow_html=True)
            c[3].markdown("<div style='padding-top:.35rem;color:#8a8f98'>—</div>", unsafe_allow_html=True)
        else:
            st.session_state[f"al_bn_{key}"] = alerts.alert_enabled(key, "banner")
            c[2].toggle("banner", key=f"al_bn_{key}", label_visibility="collapsed",
                        on_change=_alert_toggle_cb, args=(key, "banner"),
                        help="Show this report in the red REPORT DAY strip on Home on its release day."
                        if meta["alerts"] else
                        "Show a Home banner whenever this condition is live (no fixed release time).")
            if meta["alerts"]:
                st.session_state[f"al_pp_{key}"] = alerts.alert_enabled(key, "popup")
                c[3].toggle("popup", key=f"al_pp_{key}", label_visibility="collapsed",
                            on_change=_alert_toggle_cb, args=(key, "popup"),
                            help="Drop a full-screen popup on any page at this report's release time.")
            else:
                c[3].markdown("<div style='padding-top:.35rem;color:#8a8f98'>—</div>", unsafe_allow_html=True)

    st.caption("**📧 Auto‑email** sends real emails on schedule (off until you switch it on).  "
               "**🚩 Banner** and **🔔 popup** are on‑screen only.  “—” = not applicable · "
               "“n/a” = no scheduled task on this PC.")
    st.divider()


def render_recipients() -> None:
    st.subheader("\U0001F514 Alert settings")
    _render_alert_settings()
    st.markdown("#### \U0001F4E7 Email recipients")
    st.caption("Who receives each emailed report. Add or remove addresses — changes save "
               "immediately and are read by every report emailer.")
    data = recipients.load_all()
    for key, label in recipients.REPORTS.items():
        st.markdown(f"#### {label}")
        addrs = list(data.get(key, []))
        if not addrs:
            st.caption("_None set — this report falls back to the desk default._")
        for i, addr in enumerate(addrs):
            c1, c2 = st.columns([0.85, 0.15])
            c1.write(addr)
            if c2.button("Remove", key=f"rm_{key}_{i}", use_container_width=True):
                addrs.pop(i)
                data[key] = addrs
                recipients.save_all(data)
                st.rerun()
        with st.form(key=f"addform_{key}", clear_on_submit=True):
            fc1, fc2 = st.columns([0.78, 0.22])
            new = fc1.text_input("Add address", label_visibility="collapsed",
                                 placeholder="name@firm.com")
            add = fc2.form_submit_button("➕ Add", use_container_width=True)
        if add:
            e = (new or "").strip()
            if "@" not in e or "." not in e.split("@")[-1]:
                st.warning("Enter a valid email address (e.g. name@firm.com).")
            elif e in addrs:
                st.info(f"{e} is already on the list.")
            else:
                addrs.append(e)
                data[key] = addrs
                recipients.save_all(data)
                st.rerun()
        st.divider()


# OPEC Report page: manage the synopsis email list + the unattended automation. ---------
OPEC_DIR = ROOT.parent / "opec"
OPEC_CLI = ROOT / "opec_scheduled_email.py"
OPEC_MARKER = ROOT / "data" / "signals" / "opec_emailed.txt"
OPEC_2026_DATES = ("14 Jan · 11 Feb · 11 Mar · 13 Apr · 13 May · 11 Jun · 13 Jul · "
                   "12 Aug · 10 Sep · 13 Oct · 11 Nov · 9 Dec")


def _run_opec(args: list[str], label: str, timeout: int = 240):
    """Run the OPEC orchestrator as a subprocess (keeps Playwright/Chrome off Streamlit's
    event loop). Shows the output and refreshes the page."""
    with st.spinner(label):
        try:
            r = subprocess.run([sys.executable, str(OPEC_CLI), *args], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=timeout)
            ok = r.returncode == 0
            st.session_state["opec_log"] = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
            st.toast("OPEC job finished." if ok else "OPEC job hit an error — see the log.",
                     icon="✅" if ok else "⚠️")
        except subprocess.TimeoutExpired:
            st.session_state["opec_log"] = f"Timed out after {timeout}s."
            st.toast("OPEC job timed out.", icon="⚠️")
    st.rerun()


def render_opec() -> None:
    st.subheader("\U0001F6E2️ OPEC Monthly Oil Market Report")
    st.caption("Every month, when OPEC publishes its Monthly Oil Market Report, the desk gets a "
               "one-page synopsis + chart deck — fetched, built and emailed automatically. "
               "Manage the recipient list and run it on demand here.")

    last = OPEC_MARKER.read_text(encoding="utf-8").strip() if OPEC_MARKER.exists() else "—"
    pdfs = sorted(OPEC_DIR.glob("out/OPEC_MOMR_Synopsis_*.pdf"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    m1, m2 = st.columns(2)
    m1.metric("Last edition emailed", last)
    m2.metric("Next 2026 release", "Mon 13 Jul")
    st.caption(f"**2026 release calendar:** {OPEC_2026_DATES}  ·  ~04:00 ET (10:00 Vienna).")
    st.divider()

    # --- recipient list (the "opec" report key) ---------------------------------------
    st.markdown("#### Recipients")
    st.caption("These addresses receive the OPEC synopsis. Changes save immediately and are also "
               "used by the unattended scheduled job.")
    data = recipients.load_all()
    addrs = list(data.get("opec", []))
    if not addrs:
        st.caption("_None set — falls back to the desk default._")
    for i, addr in enumerate(addrs):
        c1, c2 = st.columns([0.85, 0.15])
        c1.write(addr)
        if c2.button("Remove", key=f"opec_rm_{i}", use_container_width=True):
            addrs.pop(i); data["opec"] = addrs; recipients.save_all(data); st.rerun()
    with st.form(key="opec_addform", clear_on_submit=True):
        fc1, fc2 = st.columns([0.78, 0.22])
        new = fc1.text_input("Add address", label_visibility="collapsed", placeholder="name@firm.com")
        if fc2.form_submit_button("➕ Add", use_container_width=True):
            e = (new or "").strip()
            if "@" not in e or "." not in e.split("@")[-1]:
                st.warning("Enter a valid email address (e.g. name@firm.com).")
            elif e in addrs:
                st.info(f"{e} is already on the list.")
            else:
                addrs.append(e); data["opec"] = addrs; recipients.save_all(data); st.rerun()
    st.divider()

    # --- actions ----------------------------------------------------------------------
    st.markdown("#### Run now")
    st.caption("The scheduled job already does this automatically on release days; these are for "
               "an on-demand run or a test. Fetching opens a brief Chrome window.")
    a1, a2, a3 = st.columns(3)
    if a1.button("📤 Fetch latest & send", type="primary", use_container_width=True,
                 help="Fetch the latest MOMR, build the synopsis and email the recipient list now."):
        _run_opec(["--force-send"], "Fetching the latest MOMR, building and sending…")
    if a2.button("👁️ Rebuild preview (no send)", use_container_width=True,
                 help="Rebuild the PDF from the last downloaded report without emailing."):
        _run_opec(["--from-inbox", "--dry-run"], "Rebuilding the synopsis from the last download…", timeout=120)
    desk1 = (recipients.get("opec") or ["benjamin.goulson@xpi.com.br"])[0]
    if a3.button("✉️ Send test to me", use_container_width=True,
                 help=f"Fetch + build, then email only {desk1}."):
        _run_opec(["--force-send", "--to", desk1], f"Building and sending a test to {desk1}…")

    if pdfs:
        st.download_button("⬇️  Download the latest synopsis PDF", data=pdfs[0].read_bytes(),
                           file_name=pdfs[0].name, mime="application/pdf")
    if st.session_state.get("opec_log"):
        with st.expander("Last run log", expanded=False):
            st.code(st.session_state["opec_log"][-4000:])


PM_CLI = ROOT / "precious_metals_scheduled_email.py"
PM_REL_CLI = ROOT / "pm_release_scheduled_email.py"
PM_PDF = ROOT / "data" / "Precious_Metals_Report.pdf"
PM_JSON = ROOT / "data" / "pm_monitor.json"
PM_MARKER = ROOT / "data" / "signals" / "pm_emailed.txt"
PM_PAGES = ROOT / "data" / "pm_pages"
PM_REL_DIR = ROOT / "data" / "pm_releases"


def _run_pm_rel(args: list[str], label: str, timeout: int = 420):
    """Run the release-synopsis check/build as a subprocess."""
    with st.spinner(label):
        try:
            r = subprocess.run([sys.executable, str(PM_REL_CLI), *args], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=timeout)
            st.session_state["pm_rel_log"] = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
            st.toast("Synopsis job finished." if r.returncode == 0 else
                     "Synopsis job hit an error — see the log.",
                     icon="✅" if r.returncode == 0 else "⚠️")
        except subprocess.TimeoutExpired:
            st.session_state["pm_rel_log"] = f"Timed out after {timeout}s."
            st.toast("Synopsis job timed out.", icon="⚠️")
    st.rerun()


def _run_pm(args: list[str], label: str, timeout: int = 300):
    """Run the Precious Metals build/send as a subprocess (keeps matplotlib/Playwright
    off Streamlit's event loop). Shows the output and refreshes the page."""
    with st.spinner(label):
        try:
            r = subprocess.run([sys.executable, str(PM_CLI), *args], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=timeout)
            ok = r.returncode == 0
            st.session_state["pm_log"] = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
            st.toast("Precious Metals job finished." if ok else
                     "Precious Metals job hit an error — see the log.",
                     icon="✅" if ok else "⚠️")
        except subprocess.TimeoutExpired:
            st.session_state["pm_log"] = f"Timed out after {timeout}s."
            st.toast("Precious Metals job timed out.", icon="⚠️")
    st.rerun()


def _pm_page_images() -> list[Path]:
    """Rasterize the report PDF so it reads inline on the page; cached beside it
    and refreshed whenever the PDF is newer than the cached images."""
    if not PM_PDF.exists():
        return []
    PM_PAGES.mkdir(parents=True, exist_ok=True)
    pngs = sorted(PM_PAGES.glob("page_*.png"))
    if pngs and pngs[0].stat().st_mtime >= PM_PDF.stat().st_mtime:
        return pngs
    try:
        import pypdfium2 as pdfium
        for old in pngs:
            old.unlink()
        pdf = pdfium.PdfDocument(str(PM_PDF))
        out = []
        for i, page in enumerate(pdf):
            p = PM_PAGES / f"page_{i + 1:02d}.png"
            page.render(scale=2.0).to_pil().save(p)
            out.append(p)
        return out
    except Exception:
        return []


def render_precious_metals() -> None:
    st.subheader("🥇 Precious Metals Fundamentals")
    st.caption("Monthly client monitor: macro & positioning across gold, silver, platinum and "
               "palladium, then physical flows, official-sector activity and published market "
               "balances per metal. The scheduled job emails it for proofreading before you "
               "forward it; manage recipients and the auto-send toggle on the Recipients page.")

    last = PM_MARKER.read_text(encoding="utf-8").strip() if PM_MARKER.exists() else "—"
    m1, m2 = st.columns(2)
    m1.metric("Last edition emailed", last)
    m2.metric("Built", datetime.fromtimestamp(PM_PDF.stat().st_mtime).strftime("%d %b %Y %H:%M")
              if PM_PDF.exists() else "—")

    if PM_JSON.exists():
        try:
            mock = json.loads(PM_JSON.read_text(encoding="utf-8")).get("mock_blocks", [])
        except Exception:
            mock = []
        if mock:
            st.warning(f"Draft — placeholder data in: {', '.join(mock)}. "
                       "The PDF carries the same banner until these sources are wired live.")

    st.markdown("#### Run now")
    a1, a2, a3 = st.columns(3)
    if a1.button("👁️ Rebuild preview (no send)", use_container_width=True,
                 help="Refresh the data pulls and rebuild the PDF without emailing."):
        _run_pm(["--dry-run"], "Rebuilding the Precious Metals monitor…")
    desk1 = (recipients.get("precious_metals") or ["benjamin.goulson@xpi.com.br"])[0]
    if a2.button("✉️ Email to me", use_container_width=True,
                 help=f"Rebuild, then email only {desk1} for proofreading."):
        _run_pm(["--force-send", "--to", desk1], f"Building and sending to {desk1}…")
    if a3.button("📤 Build & send to list", type="primary", use_container_width=True,
                 help="Rebuild and email the full recipient list now."):
        _run_pm(["--force-send"], "Building and sending to the recipient list…")

    if PM_PDF.exists():
        st.download_button("⬇️  Download the latest monitor PDF", data=PM_PDF.read_bytes(),
                           file_name=PM_PDF.name, mime="application/pdf")
    if st.session_state.get("pm_log"):
        with st.expander("Last run log", expanded=False):
            st.code(st.session_state["pm_log"][-4000:])

    # --- release synopses (WGC GDT / WPIC PQ) -----------------------------------------
    st.divider()
    st.markdown("#### Release synopses — WGC Gold Demand Trends · WPIC Platinum Quarterly")
    st.caption("A daily job watches for new editions and emails a one-page synopsis on "
               "release day (toggle on the Recipients page). Latest built synopses:")
    r1, r2 = st.columns(2)
    if r1.button("🔎 Check releases & rebuild (no send)", use_container_width=True,
                 help="Detect the latest editions, fetch, parse and rebuild the synopses."):
        _run_pm_rel(["--dry-run"], "Checking WGC / WPIC and rebuilding synopses…")
    if r2.button("✉️ Email latest synopses to me", use_container_width=True,
                 help=f"Rebuild and email the current editions to {desk1}."):
        _run_pm_rel(["--force-send", "--to", desk1], f"Building and sending to {desk1}…")
    rel_pdfs = sorted(PM_REL_DIR.glob("*_Synopsis.pdf"), key=lambda p: p.stat().st_mtime,
                      reverse=True)[:4]
    if rel_pdfs:
        cols = st.columns(len(rel_pdfs))
        for col, p in zip(cols, rel_pdfs):
            col.download_button(f"⬇️ {p.stem.replace('_', ' ')}", data=p.read_bytes(),
                                file_name=p.name, mime="application/pdf", key=f"pmrel_{p.name}")
    if st.session_state.get("pm_rel_log"):
        with st.expander("Last synopsis run log", expanded=False):
            st.code(st.session_state["pm_rel_log"][-4000:])

    pages = _pm_page_images()
    if pages:
        st.divider()
        for p in pages:
            st.image(str(p), use_container_width=True)
    else:
        st.info("No report built yet — use “Rebuild preview” above.")


def _cal_shift(delta):
    y, m = st.session_state.get("rcal_ym", (0, 0))
    m += delta
    if m < 1:
        m, y = 12, y - 1
    elif m > 12:
        m, y = 1, y + 1
    st.session_state["rcal_ym"] = (y, m)


def _cal_today():
    t = datetime.now(ZoneInfo("America/New_York")).date()
    st.session_state["rcal_ym"] = (t.year, t.month)


def render_releases() -> None:
    from src import release_cal, repcal
    import calendar as _cmod
    st.subheader("\U0001F4C5 Fundamental reports calendar")
    today = datetime.now(ZoneInfo("America/New_York")).date()
    nxt = release_cal.next_release(today)
    if nxt:
        st.caption(f"**Next:** {nxt['who']} — {nxt['date']:%a %d %b} ({nxt['days']}d). Every fundamental report is "
                   "on the month grid below, each with its product icon; a gold **★** marks the ones that auto-email the"
                   " desk (USDA Grain Stocks / Acreage, the OPEC MOMR synopsis, and the weekly COT report).")

    # ----- month navigation: Today / ‹ / › / Month Year -----
    st.session_state.setdefault("rcal_ym", (today.year, today.month))
    n1, n2, n3, n4 = st.columns([1.1, 0.7, 0.7, 6])
    n1.button("Today", key="rcal_today", on_click=_cal_today, use_container_width=True)
    n2.button("‹", key="rcal_prev", on_click=_cal_shift, args=(-1,), use_container_width=True)
    n3.button("›", key="rcal_next", on_click=_cal_shift, args=(1,), use_container_width=True)
    cy, cm = st.session_state["rcal_ym"]
    n4.markdown(f"<div style='font-size:21px;font-weight:700;padding-top:2px'>{_cmod.month_name[cm]} {cy}</div>",
                unsafe_allow_html=True)

    st.markdown(repcal.month_html(repcal.calendar_events(), cy, cm, today), unsafe_allow_html=True)
    st.caption("🌍 WASDE · 🌽 Crop Production · 🌾 Grain Stocks · 🌱 Plantings · 🚜 Acreage · 🐄 Cattle on Feed · "
               "🐖 Hogs & Pigs · 🛢️ Oil outlooks (OPEC / EIA / IEA) · 🧭 COT (weekly, Fri) &nbsp;·&nbsp; ★ = auto-emails the desk.")

    # ----- list views (the dated tables, for reference) -----
    rows = release_cal.next_12_months(today)
    with st.expander("📋 List view — oil-balance outlooks (next 12 months)"):
        def _cell(d, upcoming):
            return "TBC" if d is None else f"{d:%a %d %b %Y}" + ("" if upcoming else "  · released")
        brand.themed_dataframe(pd.DataFrame([{
            "Month": r["label"], "OPEC MOMR": _cell(r["opec"], r["opec_upcoming"]),
            "EIA STEO": _cell(r["eia"], r["eia_upcoming"]), "IEA OMR": _cell(r["iea"], r["iea_upcoming"]),
        } for r in rows]), fmt={}, height=440)
        st.caption("OPEC & IEA 2026 from opec.org / iea.org (2027 TBC); EIA STEO from eia.gov through 2027. "
                   "Times (ET): OPEC ~04:00 · IEA ~04:00 · EIA ~12:00 noon.")
    with st.expander("📋 List view — USDA releases (2026)"):
        from src import agdata as _agd
        _uup = _agd.report_calendar()
        _uup = _uup[_uup["date"] >= pd.Timestamp(today)].sort_values("date")
        if _uup.empty:
            st.caption("No remaining USDA releases on the 2026 calendar — the 2027 schedule loads in December.")
        else:
            brand.themed_dataframe(pd.DataFrame([{
                "Date": f"{r['date']:%a %d %b %Y}", "USDA report": r["report"],
                "Auto-reaction": "★ emails the desk" if r["report"] in repcal.RX else "",
            } for r in _uup.to_dict("records")]), fmt={}, height=340)
        st.caption("USDA dates verified vs the NASS Agricultural Statistics Board calendar + USDA OCE (WASDE).")
    with st.expander("📋 List view — COT (CFTC Commitments of Traders, weekly)"):
        _cot = release_cal.cot_releases(today, today.replace(year=today.year + 1, month=12, day=31))[:16]
        brand.themed_dataframe(pd.DataFrame([{
            "Release": f"{r['date']:%a %d %b %Y}" + ("  · delayed (holiday)" if r["delayed"] else ""),
            "Data as-of": f"{r['asof']:%a %d %b}",
        } for r in _cot]), fmt={}, height=340)
        st.caption("CFTC releases COT Fridays 3:30pm ET (data as-of the prior Tuesday); a federal holiday in "
                   "the release week pushes it to the next business day. The weekly report auto-emails the desk ★.")


def _fp_bands():
    """Selectable Fed target bands (25bp wide) from 2.00–2.25 up to 5.50–5.75."""
    return [f"{lo/100:.2f} – {lo/100+0.25:.2f}" for lo in range(200, 551, 25)]


def _fp_reseed(moves: dict, ver_bump: bool = True) -> None:
    """Set the scenario moves and force the editor to re-read them (bump its key)."""
    st.session_state["fp_moves"] = moves
    if ver_bump:
        st.session_state["fp_ver"] = st.session_state.get("fp_ver", 0) + 1


def render_fed_path() -> None:
    import altair as alt
    st.subheader("🏛️  Implied Fed-Path Calculator — SOFR strip (SR3)")
    st.caption(
        "Set where **you** think the Fed moves at each meeting and see where every SOFR "
        "future should trade if you're right — **alongside** the path the live strip is "
        "already pricing. The gap between your fair value and the market is the edge if "
        "your call plays out. Built on 3-month SOFR (SR3); the overnight rate is modelled "
        "as a step that only changes on FOMC dates.")

    asof = datetime.now(ZoneInfo("America/New_York")).date()

    # ---- assumptions ---------------------------------------------------------
    c1, c2, c3 = st.columns([1.3, 1, 1])
    band = c1.selectbox("Current target band (%)", _fp_bands(),
                        index=_fp_bands().index("4.25 – 4.50"),
                        help="Today's FOMC target range. Sets the starting level of the path.")
    lo = float(band.split("–")[0])
    mid = lo + 0.125
    basis_bp = c2.number_input("SOFR − target basis (bp)", value=0.0, step=0.5, format="%.1f",
                               help="SOFR trades a few bp around the target midpoint. Shifts every "
                                    "fair value in parallel; cancels out of the implied move count.")
    n_contracts = c3.slider("Contracts (SR3 quarterlies)", 4, 12, 8,
                            help="How far out the strip to price — whites + reds.")
    with st.expander("Advanced"):
        compound = st.checkbox("Compound SR3 settlement (ACT/360)", value=True,
                               help="On = true daily-compounded SR3 convention. Off = simple average "
                                    "(the convexity is <~1bp over a quarter).")
    basis = basis_bp / 100.0
    r0 = mid + basis                                   # current overnight (SOFR) level, %

    strip = fedpath.sr3_strip(asof, n_contracts)
    codes = [c.code for c in strip]

    # ---- market prices (from the feed; editable so the desk can punch in live) ----
    feed_px = fedpath.strip_prices(strip, asof, r0)
    px_sig = tuple(codes)
    if st.session_state.get("fp_px_sig") != px_sig:
        st.session_state["fp_px"] = {c: p for c, p in zip(codes, feed_px)}
        st.session_state["fp_px_sig"] = px_sig
    src_note = ("live Bloomberg" if fedpath.MODE == "bloomberg" else "synthetic demo")
    with st.expander(f"Market prices — SR3 strip  ·  {src_note}", expanded=False):
        st.caption("Seeded from the data feed. Overwrite any cell with a live quote and the whole "
                   "analysis re-prices off it.")
        px_df = pd.DataFrame({"Contract": codes,
                              "Window": [f"{c.label}" for c in strip],
                              "Market px": [st.session_state["fp_px"].get(c, p)
                                            for c, p in zip(codes, feed_px)]})
        edited_px = st.data_editor(
            px_df, hide_index=True, use_container_width=True, key="fp_px_editor",
            column_config={
                "Contract": st.column_config.TextColumn(disabled=True),
                "Window": st.column_config.TextColumn(disabled=True),
                "Market px": st.column_config.NumberColumn(format="%.4f", min_value=90.0, max_value=100.0)})
        st.session_state["fp_px"] = dict(zip(edited_px["Contract"], edited_px["Market px"]))
    prices = [float(st.session_state["fp_px"].get(c, p)) for c, p in zip(codes, feed_px)]

    # ---- market-implied path -------------------------------------------------
    ip = fedpath.implied_path(strip, prices, asof, r0)
    labels = [fedpath.meeting_label(m) for m in ip.meetings]

    # ---- scenario moves (one per covered meeting) ----------------------------
    sig = tuple(labels)
    if st.session_state.get("fp_sig") != sig:
        # Open with the market-implied path rounded to clean 25s — a natural starting point.
        rounded_cum = np.round(ip.cum_bp / 25.0) * 25.0
        seed = np.diff(np.concatenate([[0.0], rounded_cum]))
        st.session_state["fp_sig"] = sig
        _fp_reseed({lab: float(v) for lab, v in zip(labels, seed)})

    st.markdown("#### Your Fed path")
    b1, b2, b3, _ = st.columns([1, 1.3, 1, 3])
    if b1.button("Hold all", help="Zero every meeting — a flat 'no change' scenario."):
        _fp_reseed({lab: 0.0 for lab in labels}); st.rerun()
    if b2.button("Seed from market (25s)", help="Fill with the curve-implied path, rounded to 25bp steps."):
        rounded_cum = np.round(ip.cum_bp / 25.0) * 25.0
        seed = np.diff(np.concatenate([[0.0], rounded_cum]))
        _fp_reseed({lab: float(v) for lab, v in zip(labels, seed)}); st.rerun()

    moves = st.session_state["fp_moves"]
    mv_df = pd.DataFrame({"Meeting": labels,
                          "Move (bp)": [moves.get(lab, 0.0) for lab in labels]})
    edited = st.data_editor(
        mv_df, hide_index=True, use_container_width=True,
        key=f"fp_editor_{st.session_state.get('fp_ver', 0)}",
        column_config={
            "Meeting": st.column_config.TextColumn(disabled=True),
            "Move (bp)": st.column_config.NumberColumn(
                "Move (bp)", step=25.0, format="%+g",
                help="Your expected change in the target at this meeting: −25, 0, +25, …")},
        height=min(430, 45 + 35 * len(labels)))
    moves = {lab: float(v) for lab, v in zip(edited["Meeting"], edited["Move (bp)"])}
    st.session_state["fp_moves"] = moves
    move_list = [moves.get(lab, 0.0) for lab in labels]

    # ---- price the strip off both paths --------------------------------------
    scen_fn = fedpath.overnight_rate_fn(r0, ip.meetings, move_list)
    your_px = [fedpath.price(c, scen_fn, compound=compound) for c in strip]
    diff_bp = [(y - m) * 100.0 for y, m in zip(your_px, prices)]     # your − market, in bp
    diff_usd = [d * fedpath.SR3_BP_VALUE for d in diff_bp]

    # cumulative paths as TARGET MIDPOINT (%) for plotting
    cum_moves = np.cumsum(move_list)                               # bp
    your_seg_mid = np.concatenate([[mid], mid + cum_moves / 100.0])   # → target mid, %
    mkt_seg_mid = ip.seg_rates - basis                              # seg_rates are SOFR-level
    seg_dates = [asof] + [fedpath.effective_date(m) for m in ip.meetings]

    # ---- headline metrics ----------------------------------------------------
    st.markdown("#### At a glance")
    m1, m2, m3, m4 = st.columns(4)
    next_bp = float(ip.per_meeting_bp[0]) if len(ip.per_meeting_bp) else 0.0
    prob = max(0.0, min(1.0, abs(next_bp) / 25.0))
    m1.metric(f"Next meeting — {labels[0]}", f"{next_bp:+.0f} bp",
              help="Curve-implied move at the next FOMC. Probability ≈ implied ÷ 25bp (linear).")
    m1.caption(f"≈ {prob*100:.0f}% odds of a {'cut' if next_bp < 0 else 'hike' if next_bp > 0 else 'move'}")
    # cuts/hikes priced through the last meeting this calendar year
    eoy = [i for i, m in enumerate(ip.meetings) if m.year == asof.year]
    if eoy:
        yend = ip.cum_bp[eoy[-1]]
        m2.metric(f"Priced through Dec {asof.year}", f"{yend:+.0f} bp",
                  help="Cumulative move the strip prices from now to the last meeting this year.")
        m2.caption(f"≈ {yend/25:+.1f} × 25bp")
    m3.metric("Terminal (last covered mtg)", f"{(mid + ip.cum_bp[-1]/100):.2f}%",
              help=f"Curve-implied target mid at {labels[-1]}.")
    m3.caption(f"{ip.cum_bp[-1]:+.0f} bp vs today")
    your_end = mid + cum_moves[-1] / 100.0
    m4.metric("Your terminal", f"{your_end:.2f}%",
              help="Where your scenario lands the target mid at the last covered meeting.")
    m4.caption(f"{cum_moves[-1]:+.0f} bp vs today")

    # ---- chart 1: policy path (market vs your scenario) ----------------------
    cc = brand.chart_colors()
    path_df = pd.concat([
        pd.DataFrame({"date": seg_dates, "mid": mkt_seg_mid, "Path": "Market-implied"}),
        pd.DataFrame({"date": seg_dates, "mid": your_seg_mid, "Path": "Your scenario"}),
    ])
    path_df["date"] = pd.to_datetime(path_df["date"])
    dom = ["Market-implied", "Your scenario"]
    rng = [cc["series"], cc["accent"]]
    line = alt.Chart(path_df).mark_line(interpolate="step-after", strokeWidth=3).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("mid:Q", title="Target midpoint (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("Path:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("date:T", title="From"), alt.Tooltip("Path:N"),
                 alt.Tooltip("mid:Q", title="Target mid", format=".3f")])
    pts = alt.Chart(path_df).mark_point(filled=True, size=45).encode(
        x="date:T", y="mid:Q",
        color=alt.Color("Path:N", scale=alt.Scale(domain=dom, range=rng), legend=None))
    st.markdown("**Implied policy path — where the target midpoint lands at each meeting**")
    brand.show_chart((line + pts).properties(height=340))

    # ---- chart 2: per-contract mispricing (your − market) --------------------
    cmp_df = pd.DataFrame({"Contract": [c.label for c in strip], "diff": diff_bp})
    cmp_df["dir"] = np.where(np.array(diff_bp) > 0.05, "Cheap vs your view (buy)",
                     np.where(np.array(diff_bp) < -0.05, "Rich vs your view (sell)", "In line"))
    bar = alt.Chart(cmp_df).mark_bar().encode(
        x=alt.X("diff:Q", title="Your fair value − market  (bp of rate)"),
        y=alt.Y("Contract:N", sort=[c.label for c in strip], title=None),
        color=alt.Color("dir:N", scale=alt.Scale(
            domain=["Cheap vs your view (buy)", "Rich vs your view (sell)", "In line"],
            range=[cc["long"], cc["short"], cc["muted"]]),
            legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Contract:N"), alt.Tooltip("diff:Q", title="Diff (bp)", format="+.1f")])
    st.markdown("**Where your path disagrees with the strip** &nbsp;·&nbsp; green = future cheap vs "
                "your view (you'd buy); red = rich (you'd sell).")
    brand.show_chart((bar + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=cc["muted"]).encode(x="x:Q")).properties(height=max(220, 30 * len(strip))))

    # ---- table ---------------------------------------------------------------
    st.markdown("#### Contract detail")
    tbl = pd.DataFrame({
        "Contract": codes,
        "Window": [f"{c.label}" for c in strip],
        "Market": prices,
        "Mkt-implied rate": [100 - p for p in prices],
        "Your fair": your_px,
        "Diff (bp)": diff_bp,
        "Diff ($/lot)": diff_usd,
    })
    fmt = {"Market": "{:.4f}".format, "Mkt-implied rate": "{:.3f}".format,
           "Your fair": "{:.4f}".format, "Diff (bp)": "{:+.1f}".format,
           "Diff ($/lot)": "${:,.0f}".format}

    def _color_diff(col):
        out = []
        for v in col:
            if abs(v) < 0.05:
                out.append("color:#888")
            elif v > 0:
                out.append("color:#137333;font-weight:700")
            else:
                out.append("color:#c5221f;font-weight:700")
        return out
    brand.themed_dataframe(tbl, fmt, colorers=[(["Diff (bp)", "Diff ($/lot)"], _color_diff)], height=380)
    st.caption("**Market** = live/synthetic SR3 price. **Your fair** = the price implied by *your* "
               "meeting path. **Diff** = your fair − market, in bp of rate and $ per lot "
               f"(SR3 = ${fedpath.SR3_BP_VALUE:.0f}/bp). Positive = the contract looks cheap vs your "
               "view (buy it); negative = rich (sell). Per-meeting implied moves smear across each "
               "quarter (SR3 spans ~2 meetings) — read the **cumulative** path as the robust signal.")

    # ---- PDF export ----------------------------------------------------------
    st.divider()
    if st.button("📈 Generate Fed Path Report (visual PDF)", type="primary"):
        with st.spinner("Rendering the Fed-path report…"):
            try:
                payload = {
                    "asof": asof.isoformat(), "band": band, "mid": mid, "basis_bp": basis_bp,
                    "compound": compound, "n_contracts": n_contracts,
                    "codes": codes, "labels_contract": [c.label for c in strip],
                    "prices": prices, "your_px": your_px,
                    "meetings": labels, "moves": move_list,
                    "seg_dates": [d.isoformat() for d in seg_dates],
                    "mkt_mid": list(map(float, mkt_seg_mid)), "your_mid": list(map(float, your_seg_mid)),
                    "cum_bp": list(map(float, ip.cum_bp)), "per_meeting_bp": list(map(float, ip.per_meeting_bp)),
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "fedpath.json"
                    _out = Path(_t) / "Fed_Path_Report.pdf"
                    _in.write_text(json.dumps(payload))
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "fedpathreport.py"), str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["fp_pdf"] = _out.read_bytes()
                    else:
                        st.error("Report failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed:\n\n{e}")
    if st.session_state.get("fp_pdf"):
        st.download_button("⬇️  Download Fed Path Report", data=st.session_state["fp_pdf"],
                           file_name="Fed_Path_Report.pdf", mime="application/pdf")
        email_report_ui("fp_email", "fedpath", st.session_state["fp_pdf"],
                        subject="BASIS — Implied Fed Path (SOFR strip)",
                        attachment_name="Fed_Path_Report.pdf")


@st.cache_data(show_spinner=False, ttl=1800)
def _vbt_vol_tickers(mode: str) -> list:
    """Products with a real option surface in the current data mode — the vol
    backtester only offers these. Eurex futures generics (VGA/GXA) publish no
    moneyness surface; their cash twins (SX5E/DAX Index) carry the vol book, so
    hiding surface-less tickers keeps the picker from offering a guaranteed-fail
    leg under the same display name."""
    try:
        iv = get_implied_vol_history(list(INSTRUMENTS))
        keep = [t for t in INSTRUMENTS if t in iv.columns and iv[t].notna().any()]
        return keep or list(INSTRUMENTS)
    except Exception:
        return list(INSTRUMENTS)


@st.cache_data(show_spinner=False, ttl=600)
def _vbt_vol_rows(tickers: tuple, mode: str) -> list:
    """Current IV / RV / z per ticker, straight from the vol report's cached
    cross-section (volatility.parquet) — the same numbers the Volatility page
    shows. A ticker missing from the cross-section comes back with iv=None."""
    try:
        det = pd.read_parquet(VOL_DETAIL_FILE)
    except Exception:
        det = pd.DataFrame()
    out = []
    for t in tickers:
        rows = det[det["ticker"] == t] if not det.empty else pd.DataFrame()
        if len(rows):
            r = rows.iloc[0]
            out.append({"ticker": t, "name": str(r["market"]),
                        "iv": float(r["iv"]), "rv": float(r["rv"]),
                        "spread": float(r["spread"]),
                        "z": None if pd.isna(r["z"]) else float(r["z"]),
                        "pctl": None if pd.isna(r.get("pctl", np.nan)) else float(r["pctl"]),
                        "signal": str(r.get("signal", "—"))})
        else:
            out.append({"ticker": t, "name": INSTRUMENTS.get(t, (t,))[0], "iv": None,
                        "rv": None, "spread": None, "z": None, "pctl": None, "signal": "—"})
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def _vbt_pair_corr_cached(buy: str, sell: str, asof_iso: str, mode: str):
    """Pair-correlation stats, cached so widget reruns don't re-pull history
    (`mode` keys the cache to the data source). Raises instead of returning
    None so a transient feed failure is NOT cached for the TTL."""
    pc = volbt.correlation_stats(buy, sell, date.fromisoformat(asof_iso))
    if pc is None:
        raise RuntimeError("no shared history")
    return pc


def _vbt_pair_corr(buy: str, sell: str, asof_iso: str, mode: str):
    try:
        return _vbt_pair_corr_cached(buy, sell, asof_iso, mode)
    except Exception:
        return None


def render_vol_backtester() -> None:
    import altair as alt

    st.subheader("🧪  Vol Swap Backtester — delta-hedged straddle spreads")
    # same metric-card shrink the strategy pages get (they set it below the dispatch,
    # so custom pages don't inherit it) — dollar P&L values truncate without it.
    st.markdown("""
        <style>
          div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
              font-size: 1.05rem !important; line-height: 1.3 !important;
              white-space: normal !important; overflow-wrap: anywhere; }
          /* the date input renders as a flat box — give it the same gold outline
             as the product pickers so it reads as an interactive control */
          div[data-testid="stDateInput"] > div {
              border: 2px solid #F5C518 !important; border-radius: 8px; }
          div[data-testid="stDateInput"] input { font-weight: 600; }
        </style>""", unsafe_allow_html=True)
    st.caption(
        "Backtest a **buy-vol vs sell-vol** idea from the volatility report: ATM straddles on a "
        "common expiry in two products, deltas hedged with futures at **every settlement**, "
        "option marks reconstructed with Black-76 from the same surfaces the vol / skew / term "
        "reports pull. Choose the leg ratio (what nets to zero) and the re-strike discipline, "
        "then read where the P&L actually came from — gamma vs theta vs vega vs costs.")

    def _usd(v: float) -> str:
        return f"-${abs(v):,.0f}" if v < -0.5 else f"${abs(v):,.0f}"

    # ---- trade definition ----------------------------------------------------
    tickers = _vbt_vol_tickers(MODE)
    if len(tickers) < len(INSTRUMENTS):
        st.caption(f"{len(INSTRUMENTS) - len(tickers)} universe products are hidden here — "
                   "their tickers publish no option surface (e.g. Eurex futures generics; "
                   "use the cash-index twin, which is listed).")
    _NONE = "— none —"
    def _lab(t): return t if t == _NONE else f"{INSTRUMENTS[t][0]}  ·  {t}"
    opts = [_NONE] + tickers
    c1, c2 = st.columns(2)
    buy = c1.selectbox("BUY vol (long straddles) — optional", opts,
                       index=opts.index("NQA Index") if "NQA Index" in opts else 1,
                       format_func=_lab, key="vbt_buy")
    sell = c2.selectbox("SELL vol (short straddles) — optional", opts,
                        index=opts.index("ESA Index") if "ESA Index" in opts else 0,
                        format_func=_lab, key="vbt_sell")
    buy = None if buy == _NONE else buy
    sell = None if sell == _NONE else sell
    if not (buy or sell):
        st.error("Pick at least one product — one leg on its own trades that product's "
                 "implied against its own realized; two legs make the spread.")
        return
    if buy and sell and buy == sell:
        st.error("Pick two different products (or set one side to none for a single-leg trade).")
        return
    single = not (buy and sell)
    if single:
        _sname = INSTRUMENTS[(buy or sell)][0]
        st.caption(f"**Single-product mode** — {'buying' if buy else 'selling'} {_sname} straddles, "
                   "delta-hedged with its own futures at every settlement: the P&L is its implied "
                   "vol vs its own realized. Add a product on the other side to trade the spread.")
    # ---- vol-report snapshot: the numbers behind "why this product" -------------
    _vrows = _vbt_vol_rows(tuple(t for t in (buy, sell) if t), MODE)
    if _vrows:
        st.markdown("**Volatility report snapshot** — today's implied vs realized and the "
                    "spread z-score, straight from the Volatility page's cross-section.")
        _vcols = st.columns(max(len(_vrows), 2))
        for _vc, _vr in zip(_vcols, _vrows):
            with _vc:
                if _vr["iv"] is None:
                    st.metric(_vr["name"], "not scored")
                    st.caption("not in today's vol cross-section — run **Re-run signals**")
                else:
                    st.metric(f"{_vr['name']} — IV / RV",
                              f"{_vr['iv']:.1f} / {_vr['rv']:.1f}",
                              delta=f"spread {_vr['spread']:+.1f} vol · z {_vr['z']:+.2f}"
                              if _vr["z"] is not None else f"spread {_vr['spread']:+.1f} vol",
                              delta_color="off")
                    _sig = _vr["signal"] if _vr["signal"] not in ("—", "nan") else "no flag"
                    def _ord(n_):
                        n_ = int(round(n_))
                        sfx = "th" if 10 <= n_ % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n_ % 10, "th")
                        return f"{n_}{sfx}"
                    st.caption((f"{_ord(_vr['pctl'])} pctl of its 1y spread range · " if _vr["pctl"] is not None else "")
                               + f"vol report: {_sig}")

    # the pair-correlation panel renders HERE (right under the products), but is
    # filled further down once the entry date widget exists — it correlates up to entry.
    corr_slot = st.container()

    c3, c4, c5 = st.columns([1, 1, 1.6])
    entry = c3.date_input("Entry date (trades at that settlement)",
                          value=date.today() - timedelta(days=45),
                          max_value=date.today() - timedelta(days=5), key="vbt_entry")
    expiries = volbt.quarterly_expiries(entry + timedelta(days=40), 8)
    expiry = c4.selectbox("Option expiry (both legs)", expiries,
                          format_func=lambda d: f"{d:%b %Y}  ·  3rd Fri {d:%d %b}", key="vbt_exp")
    _W = {"rn_gamma": "Risk-normalised gamma (Γ·F²·σ²) — equal expected gamma earn, theta-flat",
          "gamma": "Dollar-gamma neutral — realized-vol trade",
          "vega": "Vega neutral — implied-spread trade",
          "beta_vega": "β-weighted vega — vol-market-neutral",
          "premium": "Premium flat — zero net outlay"}
    weighting = c5.selectbox("Leg ratio (nets to zero at entry / each re-strike)",
                             list(_W), format_func=_W.get, key="vbt_w", disabled=single,
                             help="Two-leg spreads only — a single leg is simply sized by its lots.")

    _R = {"never": "Never — hold the entry strikes",
          "daily": "Daily — new ATM straddles every settlement",
          "threshold": "On drift ≥ X × implied daily move"}
    r1, r2 = st.columns([2.2, 1])
    restrike = r1.radio("Re-strike (both legs together, re-ratioed on the day's greeks)",
                        list(_R), index=2, format_func=_R.get, horizontal=True, key="vbt_rs")
    restrike_mult = r2.number_input("X — implied daily moves of drift", 0.1, 5.0, 1.0, 0.25,
                                    disabled=restrike != "threshold", key="vbt_rsx")

    # ---- pair correlation — rendered into corr_slot, up top under the products ----
    with corr_slot:
        pc = _vbt_pair_corr(buy, sell, entry.isoformat(), MODE) if not single else None
        if not single:
            st.markdown(f"**Pair correlation up to {entry:%d %b %Y}** — how these two usually move "
                        "together (daily changes), and whether that link has drifted lately.")
        if single:
            pass                                    # no pair, no panel
        elif pc is None:
            st.info("Not enough shared history to correlate this pair.")
        else:
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Returns — 1Y", f"{pc.px_1y:+.2f}",
                      help="Correlation of daily log returns over the last 252 sessions before entry.")
            k2.metric("Returns — 1M", f"{pc.px_1m:+.2f}",
                      delta=f"{pc.px_1m - pc.px_1y:+.2f} vs 1Y", delta_color="off",
                      help="Same, over the last 21 sessions — the pair as it's trading now.")
            k3.metric("IV changes — 1Y", "n/a" if pd.isna(pc.iv_1y) else f"{pc.iv_1y:+.2f}",
                      help="Correlation of daily 1M ATM implied-vol changes — how tightly the two vol markets re-mark together.")
            k4.metric("IV changes — 1M", "n/a" if pd.isna(pc.iv_1m) else f"{pc.iv_1m:+.2f}",
                      delta="" if pd.isna(pc.iv_1m) or pd.isna(pc.iv_1y)
                      else f"{pc.iv_1m - pc.iv_1y:+.2f} vs 1Y", delta_color="off")
            k5.metric("Realized vol — 1Y", "n/a" if pd.isna(pc.rv_1y) else f"{pc.rv_1y:+.2f}",
                      help="Correlation of daily 1M realized-vol changes — whether the two actually turn volatile in sync.")
            k6.metric("Realized vol — 1M", "n/a" if pd.isna(pc.rv_1m) else f"{pc.rv_1m:+.2f}",
                      delta="" if pd.isna(pc.rv_1m) or pd.isna(pc.rv_1y)
                      else f"{pc.rv_1m - pc.rv_1y:+.2f} vs 1Y", delta_color="off")
            if pd.notna(pc.pctl):
                if pc.pctl <= 10:
                    st.warning(f"The 1M return correlation ({pc.px_1m:+.2f}) sits in the "
                               f"**{pc.pctl:.0f}th percentile** of its rolling 1-year range — the pair's "
                               "usual co-movement has loosened. The spread carries more outright risk "
                               "than the pair's history suggests, and relative-value logic leans on a "
                               "link that isn't currently holding.")
                elif pc.pctl >= 90:
                    st.info(f"The 1M return correlation ({pc.px_1m:+.2f}) is in the "
                            f"**{pc.pctl:.0f}th percentile** of its rolling 1-year range — unusually "
                            "tight vs history. The pair is moving near-lockstep, which flatters a "
                            "relative-value spread but often mean-reverts.")
                else:
                    st.caption(f"The 1M return correlation is in the {pc.pctl:.0f}th percentile of its "
                               "rolling 1-year range — in line with how this pair normally trades.")
            st.markdown("**Rolling 1M correlation over the past year** — the breakdown picture.")
            cc0 = brand.chart_colors()
            rp = pc.rolling_px.rename("corr").reset_index()
            rp.columns = ["date", "corr"]; rp["Series"] = "Returns (21d rolling)"
            ri = pc.rolling_iv.rename("corr").reset_index()
            ri.columns = ["date", "corr"]; ri["Series"] = "IV changes (21d rolling)"
            rv = pc.rolling_rv.rename("corr").reset_index()
            rv.columns = ["date", "corr"]; rv["Series"] = "Realized vol (21d rolling)"
            cdf = pd.concat([rp, ri.dropna(subset=["corr"]), rv.dropna(subset=["corr"])])
            cdom = ["Returns (21d rolling)", "IV changes (21d rolling)", "Realized vol (21d rolling)"]
            # fit the axis to the data (a tight pair reads as a flat line on a
            # fixed -1..1 axis); pad a touch, keep the dashed 1Y level in frame,
            # clamp to the +/-1 bounds. NaN-proof: a NaN in a Vega domain
            # renders a silently blank chart, so fall back to [-1, 1].
            _cvals = np.append(cdf["corr"].to_numpy(dtype=float), float(pc.px_1y))
            _cvals = _cvals[np.isfinite(_cvals)]
            _clo, _chi = (float(_cvals.min()), float(_cvals.max())) if len(_cvals) else (-1.0, 1.0)
            _cpad = max(0.05, (_chi - _clo) * 0.15)
            cchart = alt.Chart(cdf).mark_line(strokeWidth=3).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("corr:Q", title="correlation",
                        scale=alt.Scale(domain=[max(-1.0, _clo - _cpad),
                                                min(1.0, _chi + _cpad)]),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.Color("Series:N", scale=alt.Scale(domain=cdom,
                                range=[cc0["series"], cc0["accent"], cc0["long"]]),
                                legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Series:N"),
                         alt.Tooltip("corr:Q", title="Corr", format="+.2f")])
            lvl = alt.Chart(pd.DataFrame({"y": [pc.px_1y]})).mark_rule(
                color=cc0["muted"], strokeDash=[5, 3]).encode(y="y:Q")
            brand.show_chart((cchart + lvl).properties(
                height=270, title="Rolling 1M correlation (dashed = 1Y level)"))
            st.caption("Dashed line = the 1-year return-correlation level. A rolling line well "
                       "below it is the 'breakdown' to watch when trading one product against the other.")

    with st.expander("Advanced — sizing, costs, exit, point values"):
        a1, a2, a3, a4 = st.columns(4)
        buy_lots = a1.number_input("Buy-leg straddles (lots)", 1.0, 100000.0,
                                   trigger_default("VolBT lots", 100.0), 10.0,
                                   key="vbt_lots", help="Sell-leg lots follow from the ratio.")
        opt_cost = a2.number_input("Option cost (per lot)", 0.0, 10000.0,
                                   trigger_default("VolBT opt cost", 0.0), 0.5,
                                   key="vbt_copt", help="Charged per option contract traded (a straddle = call + put = 2 lots) — entry, exit and every re-strike. Same currency as the instrument. 0 = frictionless.")
        fut_cost = a3.number_input("Futures cost (per lot)", 0.0, 10000.0,
                                   trigger_default("VolBT fut cost", 0.0), 0.5,
                                   key="vbt_cfut", help="Charged per futures contract traded, on every delta hedge. Same currency as the instrument. 0 = frictionless.")
        exit_bd = a4.number_input("Exit N bus. days before expiry", 0, 20,
                                  int(trigger_default("VolBT exit bd", 5)), 1, key="vbt_exit",
                                  help="Pin-risk gamma in the final days isn't a realistic backtest — step aside before it.")
        b1, b2 = st.columns(2)
        mult_buy = (b1.number_input(f"{volbt.currency(buy)} per 1.0 point — {buy}", 0.0, 1e9,
                                    float(volbt.point_value(buy)), key=f"vbt_mb_{buy}")
                    if buy else None)
        mult_sell = (b2.number_input(f"{volbt.currency(sell)} per 1.0 point — {sell}", 0.0, 1e9,
                                     float(volbt.point_value(sell)), key=f"vbt_ms_{sell}")
                     if sell else None)
        if (buy and not mult_buy) or (sell and not mult_sell):
            st.warning("Unknown contract point value — greeks need it in dollars. Set it above.")
        st.caption("Enter point values in the contract's **own currency** — non-USD legs are "
                   "converted to USD automatically at the entry-date FX rate (frozen for the "
                   "run, from the FX futures in the universe), so mixed-currency pairs ratio "
                   "and net correctly in dollars.")
        _dc1, _dc2 = st.columns([1, 3])
        if _dc1.button("📌 Set as default", key="vbt_set_def", use_container_width=True,
                       help="Save lots, both costs and the exit buffer as this page's "
                            "startup defaults — they load on every launch."):
            save_trigger_default("VolBT lots", float(buy_lots))
            save_trigger_default("VolBT opt cost", float(opt_cost))
            save_trigger_default("VolBT fut cost", float(fut_cost))
            save_trigger_default("VolBT exit bd", int(exit_bd))
            st.toast(f"Saved as defaults: {buy_lots:g} lots · opt {opt_cost:g}/lot · "
                     f"fut {fut_cost:g}/lot · exit {int(exit_bd)} bd.", icon="📌")
        _dc2.caption(f"📌 Current defaults: **{trigger_default('VolBT lots', 100):g}** lots · "
                     f"option **{trigger_default('VolBT opt cost', 0):g}**/lot · "
                     f"futures **{trigger_default('VolBT fut cost', 0):g}**/lot · "
                     f"exit **{int(trigger_default('VolBT exit bd', 5))}** bus. days — change the "
                     "values above and click Set as default to update.")

    if st.button("▶  Run backtest", type="primary", key="vbt_run",
                 disabled=bool((buy and not mult_buy) or (sell and not mult_sell))):
        try:
            with st.spinner("Repricing and hedging settlement by settlement…"):
                st.session_state["vbt_res"] = volbt.run_backtest(
                    buy, sell, entry, expiry, weighting=weighting, restrike=restrike,
                    restrike_mult=float(restrike_mult), buy_lots=float(buy_lots),
                    opt_cost_lot=float(opt_cost), fut_cost_lot=float(fut_cost),
                    exit_bd_before_expiry=int(exit_bd),
                    mult_override={t: float(m) for t, m in ((buy, mult_buy), (sell, mult_sell)) if t})
            st.session_state.pop("vbt_pdf", None)
        except ValueError as e:
            st.session_state.pop("vbt_res", None)
            st.error(str(e))

    res = st.session_state.get("vbt_res")
    if res is None:
        return
    s = res.summary
    for w in res.warnings:
        st.warning(w)

    # ---- headline --------------------------------------------------------------
    src_note = ("live Bloomberg" if s["mode"] == "bloomberg"
                else "snapshot" if s["mode"] == "snapshot" else "synthetic demo")
    _single = bool(s.get("single"))
    _rs_note = (f" (X={s['restrike_mult']:g})" if s['restrike'] == 'threshold' else "")
    if _single:
        _k = "buy" if s.get("buy") else "sell"
        _pname = s["buy_name"] or s["sell_name"]
        _dirn = "Buy" if _k == "buy" else "Sell"
        st.markdown(f"#### {_dirn} {_pname} vol, delta-hedged — implied vs its own realized — "
                    f"{s['entry']:%d %b %Y} → {s['exit']:%d %b %Y}  ·  exp {s['expiry']:%d %b %Y}")
        st.caption(f"{s['buy_lots']:g} straddles  ·  re-strike: {_R[s['restrike']]}{_rs_note}"
                   f"  ·  data: {src_note}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net P&L", _usd(s["total"]))
        m1.caption(f"max drawdown {_usd(s['max_dd'])}")
        m2.metric("Entry IV", f"{s[f'entry_iv_{_k}']:.1f}")
        m2.caption("what the straddles were struck at")
        m3.metric("Realized over hold", f"{s[f'rlz_{_k}']:.1f}")
        _gap = s[f'entry_iv_{_k}'] - s[f'rlz_{_k}']
        m3.caption(f"entry IV − realized: {_gap:+.1f} vol")
        m4.metric("Re-strikes", f"{s['n_restrikes']}")
        m4.caption(f"all-in costs {_usd(s['costs'])}")
    else:
        st.markdown(f"#### Buy {s['buy_name']} vol / sell {s['sell_name']} vol — "
                    f"{s['entry']:%d %b %Y} → {s['exit']:%d %b %Y}  ·  exp {s['expiry']:%d %b %Y}")
        st.caption(f"{_W[s['weighting']]}  ·  re-strike: {_R[s['restrike']]}{_rs_note}"
                   + f"  ·  {s['buy_lots']:g} buy straddles × ratio {s['sell_per_buy_entry']:.2f} at entry"
                   + (f"  ·  vol-β {s['beta']:.2f} ({s['beta_obs']} obs)" if s['weighting'] == 'beta_vega' else "")
                   + f"  ·  data: {src_note}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Net P&L", _usd(s["total"]))
        m1.caption(f"max drawdown {_usd(s['max_dd'])}")
        m2.metric(f"Buy leg — {s['buy_name']}", _usd(s["total_buy"]))
        m2.caption(f"entry IV {s['entry_iv_buy']:.1f} · realized {s['rlz_buy']:.1f}")
        m3.metric(f"Sell leg — {s['sell_name']}", _usd(s["total_sell"]))
        m3.caption(f"entry IV {s['entry_iv_sell']:.1f} · realized {s['rlz_sell']:.1f}")
        m4.metric("IV spread @ entry", f"{s['entry_iv_spread']:+.1f} vol")
        m4.caption(f"realized spread {s['rlz_spread']:+.1f} vol over the hold")
        m5.metric("Re-strikes", f"{s['n_restrikes']}")
        m5.caption(f"all-in costs {_usd(s['costs'])}")

    # ---- cash greeks — at entry and at the latest marks; total then per product -
    # Raw greeks (blue, contract units) sit LEFT of each dollar greek (black).
    _GC = ["Straddles", "Delta (lots)", "$ Delta", "Gamma (Δ/pt)", "$ Gamma (per 1%)",
           "Vega (pts)", "$ Vega (per vol pt)", "Theta (pts/day)", "$ Theta (per day)",
           "Premium (pts)", "$ Premium"]
    _GC_RAW = ["Delta (lots)", "Gamma (Δ/pt)", "Vega (pts)", "Theta (pts/day)",
               "Premium (pts)"]

    def _cash_greeks(spec, pnl_total):
        """spec rows: (label, sign, F, K, iv, lots, mult, tau_years, cum_pnl) →
        greeks table, TOTAL (net) on top. Raw greeks are position-level in the
        contract's own units; $ greeks convert via point value (and FX). The
        TOTAL P&L is passed in because it includes costs, which belong to
        neither leg."""
        rows, tot = [], {c: 0.0 for c in _GC}
        for label, sgn, F, K, iv, n, m, tau, pnl in spec:
            g = volbt.straddle_greeks(F, K, iv, tau)
            row = {"Position": label, "Straddles": sgn * n,
                   "Delta (lots)": sgn * g.delta * n,
                   "$ Delta": sgn * g.delta * n * F * m,
                   "Gamma (Δ/pt)": sgn * g.gamma * n,
                   "$ Gamma (per 1%)": sgn * g.gamma * F * F * n * m / 100.0,
                   "Vega (pts)": sgn * g.vega * n,
                   "$ Vega (per vol pt)": sgn * g.vega * n * m,
                   "Theta (pts/day)": sgn * g.theta / 365.0 * n,
                   "$ Theta (per day)": sgn * g.theta / 365.0 * n * m,
                   "Premium (pts)": sgn * g.value * n,
                   "$ Premium": sgn * g.value * n * m,
                   "$ P&L (cum.)": pnl}
            rows.append(row)
            for c in tot:
                tot[c] += row[c]
        tot = {k2: (0.0 if abs(v) < 0.005 else v) for k2, v in tot.items()}  # kill -0 dust
        return pd.DataFrame([{"Position": "TOTAL (net)", **tot,
                              "$ P&L (cum.)": pnl_total}] + rows)

    _ev0, _evN = res.events.iloc[0], res.events.iloc[-1]
    _dN = res.daily.iloc[-1]
    _tau0 = max((s["expiry"] - s["entry"]).days, 1) / 365.0
    _tauN = max((s["expiry"] - s["exit"]).days, 1) / 365.0
    _spec0, _specN = [], []
    for _gk, _gK, _gsgn in (("buy", "Buy", 1.0), ("sell", "Sell", -1.0)):
        if not s.get(_gk):
            continue
        _lab = f"{_gK} — {s[f'{_gk}_name']}"
        _gm = float(s[f"mult_{_gk}"])
        _spec0.append((_lab, _gsgn, float(_ev0[f"{_gk}_K"]), float(_ev0[f"{_gk}_K"]),
                       float(_ev0[f"{_gk}_iv"]), float(_ev0[f"{_gk}_lots"]), _gm, _tau0,
                       0.0))
        _specN.append((_lab, _gsgn, float(_dN[f"{_gk}_F"]), float(_dN[f"{_gk}_K"]),
                       float(_dN[f"{_gk}_iv"]), float(_evN[f"{_gk}_lots"]), _gm, _tauN,
                       float(s[f"total_{_gk}"])))
    st.markdown("**Greeks** — TOTAL first, then by product. Raw greeks in "
                ":blue[**blue**] (contract units: delta in futures lots, gamma as delta per "
                "1.0 point, vega / theta / premium in price points); dollar greeks beside them "
                "convert via point value (and entry FX). **\\$ Delta** = the underlying notional "
                "the options carry (the futures hedge holds the opposite); **\\$ Gamma** = \\$ "
                "delta picked up per 1% spot move; **\\$ P&L (cum.)** = each leg's cumulative "
                "P&L to that date — the TOTAL row also carries the costs, which belong to "
                "neither leg. On a mixed pair the TOTAL of the blue columns adds different "
                "contracts' units, so read it as indicative; the \\$ columns are the comparable "
                "ones.")
    _gfmt = {"Straddles": "{:+,.1f}".format,
             "Delta (lots)": "{:+,.2f}".format, "$ Delta": "{:+,.0f}".format,
             "Gamma (Δ/pt)": "{:+,.4f}".format, "$ Gamma (per 1%)": "{:+,.0f}".format,
             "Vega (pts)": "{:+,.1f}".format, "$ Vega (per vol pt)": "{:+,.0f}".format,
             "Theta (pts/day)": "{:+,.1f}".format, "$ Theta (per day)": "{:+,.0f}".format,
             "Premium (pts)": "{:+,.0f}".format, "$ Premium": "{:+,.0f}".format,
             "$ P&L (cum.)": "{:+,.0f}".format}
    _gpal = brand.palette()
    _gsurf = str(_gpal.get("surface", "#ffffff")).lstrip("#")
    try:
        _glum = (0.299 * int(_gsurf[0:2], 16) + 0.587 * int(_gsurf[2:4], 16)
                 + 0.114 * int(_gsurf[4:6], 16))
    except Exception:
        _glum = 255.0
    # bright blue on the dark theme, deep blue on light — the theme 'series' tone
    # is too dim to read in a table cell
    _graw_blue = "#82B4FF" if _glum < 128 else "#1F5FA8"
    _gcolor = [([c for c in _GC_RAW],
                lambda col: [f"color:{_graw_blue}; font-weight:600"] * len(col))]
    st.caption(f"At entry — {s['entry']:%d %b %Y} (as struck)")
    _g0 = _cash_greeks(_spec0, float(res.daily["net"].iloc[0]))
    brand.themed_dataframe(_g0, fmt=_gfmt, colorers=_gcolor, height=45 + 35 * len(_g0))
    st.caption(f"Latest — {s['exit']:%d %b %Y} (final marks before close-out; re-strikes "
               "re-size the position along the way — see the dollar-greeks chart)")
    _gN = _cash_greeks(_specN, float(s["total"]))
    brand.themed_dataframe(_gN, fmt=_gfmt, colorers=_gcolor, height=45 + 35 * len(_gN))

    cc = brand.chart_colors()
    d = res.daily.reset_index()
    d["buy_cum"] = d["buy_pnl"].cumsum()
    d["sell_cum"] = d["sell_pnl"].cumsum()

    # ---- chart 1: cumulative P&L, net + per leg, re-strikes ticked --------------
    st.markdown("**Cumulative P&L** — net in gold"
                + ("" if _single else "; each leg (options + its hedges) faint")
                + "; ▲ marks a re-strike.")
    _frames = [pd.DataFrame({"date": d["date"], "pnl": d["cum_net"], "Series": "Net"})]
    dom = ["Net"]
    if not _single:
        _frames += [pd.DataFrame({"date": d["date"], "pnl": d["buy_cum"], "Series": f"Buy {s['buy_name']}"}),
                    pd.DataFrame({"date": d["date"], "pnl": d["sell_cum"], "Series": f"Sell {s['sell_name']}"})]
        dom += [f"Buy {s['buy_name']}", f"Sell {s['sell_name']}"]
    cum_df = pd.concat(_frames)
    _xax = alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12))
    halo = alt.Chart(cum_df[cum_df["Series"] == "Net"]).mark_line(
        color=cc["halo"], strokeWidth=5.6).encode(x=_xax, y="pnl:Q")
    line = alt.Chart(cum_df).mark_line(strokeWidth=3.4).encode(
        x=_xax,
        y=alt.Y("pnl:Q", title="cumulative P&L ($)",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, format="~s")),
        color=alt.Color("Series:N", scale=alt.Scale(domain=dom,
                        range=[cc["accent"], cc["series"], cc["short"]]),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
        opacity=alt.condition(alt.datum.Series == "Net", alt.value(1.0), alt.value(0.5)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Series:N"),
                 alt.Tooltip("pnl:Q", title="P&L ($)", format="+,.0f")])
    rs_df = d[d["restrike"] == 1]
    ticks = alt.Chart(rs_df).mark_point(shape="triangle-up", filled=True, size=60,
                                        color=cc["muted"]).encode(
        x="date:T", y=alt.value(8),
        tooltip=[alt.Tooltip("date:T", title="Re-strike")])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=cc["muted"]).encode(y="y:Q")
    brand.show_chart((halo + line + ticks + zero).properties(
        height=360, title="Cumulative P&L — net (gold) vs each leg · ▲ re-strikes"))

    # ---- chart 2: attribution ----------------------------------------------------
    st.markdown("**P&L attribution** — where it came from: gamma (realized vol) vs theta "
                "(implied paid/collected) vs vega (surface re-marks) vs costs.")
    att = pd.DataFrame({
        "component": ["Gamma (realized)", "Theta (carry)", "Vega (IV re-mark)",
                      "Higher-order (resid.)", "Costs", "NET"],
        "value": [s["gamma_pnl"], s["theta_pnl"], s["vega_pnl"], s["resid_pnl"],
                  -s["costs"], s["total"]]})
    att["value"] = att["value"].apply(lambda v: 0.0 if abs(v) < 0.5 else v)   # no "-0" labels
    att["kind"] = np.where(att["component"] == "NET", "net",
                           np.where(att["value"] >= 0, "pos", "neg"))
    _alo = min(0.0, float(att["value"].min()))
    _ahi = max(0.0, float(att["value"].max()))
    _arng = (_ahi - _alo) or 1.0
    _asort = list(att["component"])
    bar = alt.Chart(att).mark_bar().encode(
        x=alt.X("value:Q", title="P&L ($)",
                scale=alt.Scale(domain=[_alo - _arng * (0.18 if _alo < 0 else 0.02),
                                        _ahi + _arng * 0.18]),
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, format="~s")),
        y=alt.Y("component:N", sort=_asort, title=None, axis=alt.Axis(labelFontSize=12)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["pos", "neg", "net"],
                        range=[cc["long"], cc["short"], cc["accent"]]), legend=None),
        tooltip=[alt.Tooltip("component:N", title="Component"),
                 alt.Tooltip("value:Q", title="P&L ($)", format="+,.0f")])
    _tpos = alt.Chart(att[att["value"] >= 0]).mark_text(
        align="left", dx=6, fontSize=12, fontWeight="bold", color=cc["ink"]).encode(
        x="value:Q", y=alt.Y("component:N", sort=_asort),
        text=alt.Text("value:Q", format="+,.0f"))
    _tneg = alt.Chart(att[att["value"] < 0]).mark_text(
        align="right", dx=-6, fontSize=12, fontWeight="bold", color=cc["ink"]).encode(
        x="value:Q", y=alt.Y("component:N", sort=_asort),
        text=alt.Text("value:Q", format="+,.0f"))
    brand.show_chart((bar + _tpos + _tneg + alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=cc["muted"]).encode(x="x:Q")).properties(
        height=250, title="P&L attribution — labelled in $"))
    with st.expander("How these bars are computed"):
        st.markdown("""
Each bar is a daily decomposition summed over the whole backtest. Every settlement day,
each leg's option P&L is split using the **previous day's greeks** (the position actually
held overnight), netted across legs:

- **Gamma (realized)** — ½ × gamma × (price move)² × lots × point value. The convexity
  payoff: you earn on the *square* of the move, direction-irrelevant. This is the
  "realized vol" engine — big daily moves make it large, quiet days make it tiny.
- **Theta (carry)** — theta × days elapsed × lots × point value. The rent on holding the
  options — paid if long, collected if short (weekends land as three days of it on
  Monday). Theta is what the market *charges* for gamma, priced off implied vol.
- **Vega (IV re-mark)** — vega × (change in the implied vol used to mark the position).
  Pure mark-to-market from the surface moving, including the roll along the term
  structure as expiry approaches. Unlike gamma/theta it isn't "banked" daily — it's only
  locked in because the position is closed at exit.
- **Delta** — computed too but never shown: the futures hedge is reset to exactly minus
  the position delta at every settlement, so the hedge P&L cancels the delta term
  identically (it holds to the cent, every day).
- **Higher-order** — the day's actual option P&L minus all of the above. The Taylor-series
  remainder: what a first-order greek split can't capture — very large single-day jumps
  (gamma itself changes mid-move), spot and vol moving together. Small and noisy = the
  attribution is trustworthy; persistently large = read the other bars with caution.
- **Costs** — every transaction cost charged (options crossed at entry/exit/re-strikes,
  futures hedges), shown negative.

**NET = Gamma + Theta + Vega + Higher-order − Costs** — an accounting identity with the
cumulative P&L line above, not an estimate.
""")

    # ---- chart 3: the implied vols in the marks (+ spread when two legs) ---------
    if _single:
        st.markdown("**Implied vol in the marks** — the fixed-strike vol at the trade's "
                    "days-to-expiry, i.e. what the straddles were marked (and re-struck) at.")
        _kk = "buy" if s.get("buy") else "sell"
        _pn = s["buy_name"] or s["sell_name"]
        iv_df = pd.DataFrame({"date": d["date"], "iv": d[f"{_kk}_iv"], "Series": f"{_pn} IV"})
        ivdom = [f"{_pn} IV"]
    else:
        st.markdown("**Implied vols in the marks** — each leg's fixed-strike vol at the trade's "
                    "days-to-expiry, and the spread the trade is long.")
        iv_df = pd.concat([
            pd.DataFrame({"date": d["date"], "iv": d["buy_iv"], "Series": f"{s['buy_name']} IV"}),
            pd.DataFrame({"date": d["date"], "iv": d["sell_iv"], "Series": f"{s['sell_name']} IV"}),
            pd.DataFrame({"date": d["date"], "iv": d["buy_iv"] - d["sell_iv"], "Series": "Spread (buy − sell)"}),
        ])
        ivdom = [f"{s['buy_name']} IV", f"{s['sell_name']} IV", "Spread (buy − sell)"]
    ivc = alt.Chart(iv_df).mark_line(strokeWidth=3).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
        y=alt.Y("iv:Q", title="vol points",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=ivdom,
                        range=[cc["series"], cc["short"], cc["accent"]]),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Series:N"),
                 alt.Tooltip("iv:Q", title="Vol", format=".2f")])
    brand.show_chart((ivc + zero).properties(
        height=300, title="Implied vols marking the trade" +
        ("" if _single else " · spread (gold)")))

    # ---- chart 4: dollar greeks — is the neutrality / exposure decaying? ---------
    st.markdown("**Dollar greeks by leg** — how the chosen neutrality decays between "
                "re-strikes (equal lines = neutral)." if not _single else
                "**Dollar greeks** — how the position's gamma and vega evolve between "
                "re-strikes (gamma fades as spot drifts from strike; re-striking restores it).")
    _gleg = []
    if s.get("buy"):
        _gleg.append(("buy", f"Buy {s['buy_name']}", cc["series"]))
    if s.get("sell"):
        _gleg.append(("sell", f"Sell {s['sell_name']}", cc["short"]))
    g1, g2 = st.columns(2)
    for col, field, lab, _gdiv in ((g1, "gamma_usd", "$ gamma per 1% (Γ·F²·mult ÷ 100)", 100.0),
                                   (g2, "vega_usd", "$ vega (per vol pt)", 1.0)):
        gdf = pd.concat([pd.DataFrame({"date": d["date"], "v": d[f"{k}_{field}"] / _gdiv, "Leg": nm})
                         for k, nm, _c in _gleg])
        ch = alt.Chart(gdf).mark_line(strokeWidth=3).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("v:Q", title=lab,
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13, format="~s")),
            color=alt.Color("Leg:N", scale=alt.Scale(
                domain=[nm for _k, nm, _c in _gleg],
                range=[c for _k, _nm, c in _gleg]),
                legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Leg:N"),
                     alt.Tooltip("v:Q", title=lab, format=",.0f")])
        with col:
            brand.show_chart(ch.properties(height=260))

    # ---- events + daily detail ---------------------------------------------------
    st.markdown("**Trade log** — entry, every re-strike (new strikes, IVs, ratio), exit.")
    _EV = {"entry": "Entry", "daily": "Re-strike (daily)",
           "threshold": "Re-strike (drift)", "exit": "Exit"}
    ev = res.events.copy()
    ev["event"] = ev["event"].map(lambda e: _EV.get(e, e))
    ev = ev.rename(columns={
        "date": "Date", "event": "Event", "buy_K": "Buy strike",
        "sell_K": "Sell strike", "buy_iv": "Buy IV", "sell_iv": "Sell IV",
        "sell_per_buy": "Ratio (sell per buy)", "buy_lots": "Buy lots",
        "sell_lots": "Sell lots"})
    if _single:
        _drop_side = "Sell" if s.get("buy") else "Buy"
        ev = ev.drop(columns=[c for c in ev.columns if c.startswith(_drop_side)]
                     + ["Ratio (sell per buy)"])
    _ev_fmt = {"Buy strike": "{:,.2f}".format, "Sell strike": "{:,.2f}".format,
               "Buy IV": "{:.2f}".format, "Sell IV": "{:.2f}".format,
               "Ratio (sell per buy)": "{:.3f}".format, "Buy lots": "{:.1f}".format,
               "Sell lots": "{:.1f}".format}
    brand.themed_dataframe(ev, fmt={c: f for c, f in _ev_fmt.items() if c in ev.columns},
                           na_rep="—", height=min(380, 45 + 35 * len(ev)))
    if _single:
        _legs_line = (f"product = {s['buy_name'] or s['sell_name']} "
                      f"({'long' if s.get('buy') else 'short'} vol, delta-hedged)")
        _ratio_bullet = ""
        _lots_bullet = f"- **Lots** — the straddles held, fixed at {s['buy_lots']:g} throughout.\n"
    else:
        _legs_line = f"buy leg = {s['buy_name']}, sell leg = {s['sell_name']}"
        _ratio_bullet = (f"- **Ratio (sell per buy)** — sell straddles per one buy straddle, re-solved "
                         f"from that day's greeks so the chosen neutrality "
                         f"({_W[s['weighting']].split(' — ')[0]}) is restored. Blank on Exit "
                         "(nothing is struck, only closed).\n")
        _lots_bullet = (f"- **Buy/Sell lots** — the resulting position sizes: buy lots stay fixed at "
                        f"{s['buy_lots']:g}, sell lots move with the ratio.\n")
    st.markdown(f"""
*How to read the trade log — {_legs_line}.*
- **Date / Event** — each time the position was (re)struck. **Entry** opens it; each **Re-strike** closes the old straddles and strikes fresh ATM ones (*daily* = every settlement, *drift* = a settle moved ≥ X implied daily moves from its strike); **Exit** closes everything at the pre-expiry buffer (set under Advanced).
- **Strike** — the new at-the-money strike, i.e. that day's settlement.
- **IV** — the implied vol the fresh straddles were dealt at. Compare down the column to see the level the trade kept re-entering at.
{_ratio_bullet}{_lots_bullet}""")
    with st.expander("Daily detail — marks, attribution, costs"):
        cols = {}
        for _kk, _KK in (("buy", "Buy"), ("sell", "Sell")):
            if s.get(_kk):
                cols.update({f"{_kk}_F": f"{_KK} settle", f"{_kk}_K": f"{_KK} strike",
                             f"{_kk}_iv": f"{_KK} IV"})
        for _kk, _KK in (("buy", "Buy"), ("sell", "Sell")):
            if s.get(_kk):
                cols[f"{_kk}_pnl"] = f"{_KK} leg P&L" if not _single else "Leg P&L"
        cols.update({"net_gamma": "Gamma P&L", "net_theta": "Theta P&L",
                     "net_vega": "Vega P&L", "net_resid": "Higher-order",
                     "cost": "Costs", "net": "Net P&L", "cum_net": "Cumulative",
                     "restrike": "Re-strike"})
        st.dataframe(res.daily[list(cols)].round(2).rename(columns=cols),
                     use_container_width=True, height=420)
        st.markdown(f"""
**How to read this table** — {_legs_line}.

*The marks — state of each leg at that day's settlement*
- **Buy/Sell settle** — the settlement price of each leg that day: what the options are marked against and the level the deltas are hedged at.
- **Buy/Sell strike** — the strike the straddle is currently holding. On entry and every re-strike day it equals the settle (struck at-the-money); between re-strikes the settle drifts away from it, and that gap is what the threshold rule watches.
- **Buy/Sell IV** — the implied vol used to mark that leg's straddle: the fixed-strike vol at the trade's remaining days-to-expiry (surface-interpolated, smile-adjusted when settle has drifted from strike). The difference between the two IV columns is the spread the trade is long.

*The P&L — dollars, that day*
- **Buy/Sell leg P&L** — each leg's total day P&L: option mark-to-market **plus its futures hedge**, signed from your book's perspective (a short straddle losing value shows positive). The two legs minus costs sum to Net P&L.
- **Costs** — transaction costs charged that day: option crossing on entry/exit/re-strike days, hedge slippage otherwise.
- **Net P&L / Cumulative** — the package's day P&L and its running total (the gold line in the chart above).

*The attribution — where the day's net came from (these explain Net P&L, they don't add to it separately)*
- **Gamma P&L** — the realized-vol engine: ½ × gamma × (price move)², netted across legs. Positive when the leg you're long gamma on moved more than the leg you're short.
- **Theta P&L** — the carry: decay paid on the long leg minus decay collected on the short leg (Mondays carry the weekend). Gamma and theta are two sides of the same bet — over the trade you want gamma earned to beat theta paid.
- **Vega P&L** — the implied re-mark: each leg's vega × that day's IV move. Mark-to-market on the IV spread; dominates if you exit early rather than grind to expiry.
- **Higher-order** — the leftover after delta/gamma/theta/vega. Small and noisy is healthy; persistently large means the greek breakdown is straining (huge moves, vol jumping with spot).
- **Re-strike** — 1 on days the package was rolled to fresh ATM strikes and re-ratioed (the entry row shows 1 because entering is the first strike).

*Sanity identity:* for any day, Buy leg P&L + Sell leg P&L ≈ Gamma + Theta + Vega + Higher-order — the delta piece is absent because the futures hedge cancels it by construction.
""")

    with st.expander("Trade blotter — every ticket behind the P&L"):
        bl = getattr(res, "trades", None)
        if bl is None or bl.empty:
            st.info("Re-run the backtest to populate the blotter (added after this run).")
        else:
            view = st.radio("Show", ["All tickets", "Options only", "Futures hedges only"],
                            horizontal=True, key="vbt_blotter_view")
            b = bl.copy()
            if view == "Options only":
                b = b[b["instrument"] != "future"]
            elif view == "Futures hedges only":
                b = b[b["instrument"] == "future"]
            b["instrument"] = b["instrument"].str.capitalize()
            b["action"] = b["action"].str.capitalize()
            b["leg"] = b["leg"].map({"buy": "Buy vol", "sell": "Sell vol"})
            b = b.rename(columns={
                "date": "Date", "leg": "Leg", "product": "Product",
                "instrument": "Instrument", "action": "Action", "lots": "Lots",
                "strike": "Strike", "price": "Price", "cash": "Cash ($)",
                "reason": "Reason"})
            st.dataframe(b.round({"Lots": 2, "Strike": 2, "Price": 2, "Cash ($)": 0}),
                         use_container_width=True, height=420, hide_index=True)
            _esc = lambda v: _usd(v).replace("$", "\\$")   # bare $…$ pairs trigger st.markdown's LaTeX mode
            st.caption(f"Blotter check: the Cash column sums to {_esc(float(bl['cash'].sum()))} "
                       f"= Net P&L {_esc(s['total'])} + Costs {_esc(s['costs'])} — every position "
                       "opened here is also closed here, so the tickets reconstruct the P&L exactly.")
            st.markdown(f"""
*How to read the blotter — {_legs_line}.*
- **Prices are Black-76 model values at that day's settlement** (settlement price + that day's surface vol) — i.e. mid. The cost assumptions, when set, are charged separately in the Daily detail *Costs* column, never baked into these prices. At entry and re-strikes the call and put prices are identical by construction: an at-the-money-forward straddle with r = 0 has call = put.
- **Entry / Re-strike open** — striking fresh ATM calls + puts on both legs (buy-vol leg buys them, sell-vol leg sells them); Strike = that day's settle.
- **Re-strike close** — unwinding the old strikes at their current marks just before striking new ones. **Exit close** — the final unwind at the pre-expiry buffer.
- **Delta hedge** — the futures traded at settlement to flatten the package delta. Lots is the *change* in the hedge position that day (the running position is minus the leg's option delta × lots held). On re-strike days the hedge adjusts in one net ticket alongside the new strikes; *Delta hedge (close)* flattens the book on the way out.
- **Cash ($)** — signed premium/notional: sells positive, buys negative, × the contract point value.
""")

    # ---- PDF tearsheet -------------------------------------------------------------
    st.divider()
    if st.button("📈 Generate Backtest Tearsheet (PDF)", type="primary", key="vbt_pdf_btn"):
        with st.spinner("Rendering the tearsheet…"):
            try:
                payload = {
                    "summary": {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                                for k, v in s.items()},
                    "dates": [x.date().isoformat() for x in res.daily.index],
                    "cum_net": [float(x) for x in res.daily["cum_net"]],
                    "buy_cum": [float(x) for x in d["buy_cum"]],
                    "sell_cum": [float(x) for x in d["sell_cum"]],
                    "buy_iv": [float(x) for x in d["buy_iv"]],
                    "sell_iv": [float(x) for x in d["sell_iv"]],
                    "restrike": [int(x) for x in d["restrike"]],
                    "events": json.loads(res.events.to_json(orient="records", date_format="iso")),
                }
                _pcr = (_vbt_pair_corr(s["buy"], s["sell"], s["entry"].isoformat(), MODE)
                        if (s.get("buy") and s.get("sell")) else None)
                if _pcr is not None:
                    def _n(v):
                        return None if pd.isna(v) else round(float(v), 3)
                    payload["corr"] = {"px_1y": _n(_pcr.px_1y), "px_1m": _n(_pcr.px_1m),
                                       "iv_1y": _n(_pcr.iv_1y), "iv_1m": _n(_pcr.iv_1m),
                                       "pctl": _n(_pcr.pctl),
                                       "rolling": {
                                           "dates": [x.date().isoformat()
                                                     for x in _pcr.rolling_px.index],
                                           "px": [_n(v) for v in _pcr.rolling_px],
                                           "iv": [_n(v) for v in _pcr.rolling_iv],
                                           "level": _n(_pcr.px_1y)}}
                _legs_t = [t for t in (s["buy"], s["sell"]) if t]
                payload["volctx"] = _vbt_vol_rows(tuple(_legs_t), MODE)
                _ivh = get_implied_vol_history(_legs_t)
                _rvh = get_realized_vol_history(_legs_t)
                _vh = {}
                for _t in _legs_t:
                    if _t not in _ivh.columns:
                        continue
                    _ivs = _ivh[_t].dropna().iloc[-252:]
                    if _ivs.empty:
                        continue
                    _rvs = (_rvh[_t].reindex(_ivs.index) if _t in _rvh.columns
                            else pd.Series(index=_ivs.index, dtype=float))
                    _vh[_t] = {"name": INSTRUMENTS.get(_t, (_t,))[0],
                               "dates": [x.date().isoformat() for x in _ivs.index],
                               "iv": [float(v) for v in _ivs],
                               "rv": [None if pd.isna(v) else float(v) for v in _rvs]}
                payload["volhist"] = _vh
                if getattr(res, "trades", None) is not None and not res.trades.empty:
                    payload["blotter"] = json.loads(
                        res.trades.to_json(orient="records", date_format="iso"))
                payload["greeks"] = {
                    "entry_caption": f"At entry — {s['entry']:%d %b %Y} (as struck)",
                    "latest_caption": f"Latest — {s['exit']:%d %b %Y} (final marks before close-out)",
                    "entry": json.loads(_g0.to_json(orient="records")),
                    "latest": json.loads(_gN.to_json(orient="records")),
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "volbt.json"
                    _out = Path(_t) / "Vol_Backtest_Tearsheet.pdf"
                    _in.write_text(json.dumps(payload))
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "volbtreport.py"), str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["vbt_pdf"] = _out.read_bytes()
                    else:
                        st.error("Tearsheet failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Tearsheet failed:\n\n{e}")
    if st.session_state.get("vbt_pdf"):
        st.download_button("⬇️  Download Backtest Tearsheet", data=st.session_state["vbt_pdf"],
                           file_name="Vol_Backtest_Tearsheet.pdf", mime="application/pdf")
        email_report_ui("vbt_email", "volbt", st.session_state["vbt_pdf"],
                        subject="BASIS — Vol Swap Backtest",
                        attachment_name="Vol_Backtest_Tearsheet.pdf")


@st.cache_data(show_spinner=False, ttl=1800)
def _osb_atm_curve(ticker: str, mode: str) -> dict:
    """Latest ATM implied-vol term points for `ticker` — {calendar days: vol %}
    across the live 1M/3M/6M/12M surface tenors, from the SAME pull the vol /
    term reports use. A tenor that is frozen or has stopped publishing
    (stale_iv_reasons) is dropped rather than served as a live mark; an empty
    dict means the product has no usable surface and vols stay manual."""
    days_of = {"1M": 30, "3M": 91, "6M": 182, "12M": 365}
    try:
        ts = get_term_structure([ticker])
    except Exception:
        return {}
    out = {}
    for lab, frame in ts.items():
        if frame is None or ticker not in frame.columns or not frame[ticker].notna().any():
            continue
        if ticker in stale_iv_reasons(frame[[ticker]]):
            continue
        out[days_of[lab]] = float(frame[ticker].dropna().iloc[-1])
    return out


def render_strategy_builder() -> None:
    """Multi-leg option strategy builder (the optioncreator.com workflow): build a
    position from Buy/Sell x Call/Put/Future legs, read net debit/credit, max
    profit / max loss / breakevens / greeks, and see the P&L curve at expiry plus
    a re-priced 'T + d days' scenario line. Engine: src/optbuilder.py (Black-76,
    same convention as the Vol Backtester)."""
    import altair as alt
    st.subheader("🧰  Option Strategy Builder — multi-leg payoff modeller")
    st.markdown("""
        <style>
          div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
              font-size: 1.05rem !important; line-height: 1.3 !important;
              white-space: normal !important; overflow-wrap: anywhere; }
        </style>""", unsafe_allow_html=True)
    st.caption(
        "Pick a product, then enter each leg like a ticket — **month, strike, call / put / "
        "future, price paid** (leave *Price paid* blank to mark at the Black-76 model value). "
        "Or load a preset and reshape it. The chart shows P&L **at the front expiry** (solid) "
        "and a **re-priced scenario** part-way through the trade (dashed) — slide days forward "
        "and shift vol to see the position decay and re-mark.")

    # ---- listed months: the next 13 monthly expiries (3rd Friday, house convention) ----
    _today = date.today()
    _MONTHS, _MDAYS = [], {}
    _yy, _mm = _today.year, _today.month
    for _ in range(14):
        _exp = volbt.third_friday(_yy, _mm)
        if _exp > _today:
            _lbl = f"{_exp:%b %Y}"
            _MONTHS.append(_lbl)
            _MDAYS[_lbl] = (_exp - _today).days
        _mm += 1
        if _mm == 13:
            _mm, _yy = 1, _yy + 1
    _NO_MONTH = "—"

    def _closest_month(days: float) -> str:
        return min(_MONTHS, key=lambda L: abs(_MDAYS[L] - days))

    _NONE = "— manual —"
    _opts = [_NONE] + list(INSTRUMENTS)
    c1, c2, c3, c4 = st.columns([1.9, 1, 1, 1])
    prod = c1.selectbox("Product", _opts,
                        format_func=lambda t: t if t == _NONE else f"{INSTRUMENTS[t][0]}  ·  {t}",
                        key="osb_prod",
                        help="Picking a product seeds the price, point value and vol marks — "
                             "and restrikes the loaded legs at its price. '— manual —' prices "
                             "a generic underlying you set yourself.")
    # re-seed price + point value when the product changes (widget state otherwise wins),
    # and restrike the loaded legs at the new price — strikes built around the old spot
    # are meaningless on a different underlying
    if st.session_state.get("osb_prod_prev") != prod:
        _first = "osb_prod_prev" not in st.session_state
        st.session_state["osb_prod_prev"] = prod
        if prod != _NONE:
            try:
                _seed = float(get_live_quote([prod]).loc[prod, "last"])
            except Exception:
                _seed = float(INSTRUMENTS[prod][1])
            st.session_state["osb_spot"] = _seed
            st.session_state["osb_pv"] = float(volbt.point_value(prod)) or 1.0
            _cv = _osb_atm_curve(prod, MODE)
            if _cv:                               # default vol follows the surface
                _dd = _MDAYS.get(st.session_state.get("osb_dmonth", ""), 30.0)
                st.session_state["osb_dvol"] = round(optbuilder.vol_at(_cv, _dd), 1)
            if not _first:
                st.session_state.pop("osb_rows", None)
                st.session_state["osb_nonce"] = st.session_state.get("osb_nonce", 0) + 1
                st.rerun()
    F0 = c2.number_input("Underlying price", min_value=0.0001, value=100.0,
                         format="%.4f", key="osb_spot")
    pv = c3.number_input("Point value", min_value=0.0, value=1.0,
                         format="%.2f", key="osb_pv",
                         help="Contract currency per 1.00 of price, per lot — scales the "
                              "price-point P&L into money. Leave 1 to stay in price points.")
    rate = c4.number_input("Rate %", min_value=0.0, value=0.0, step=0.25,
                           format="%.2f", key="osb_rate",
                           help="Discount rate. Futures-style margining makes discounting "
                                "near-noise — 0 matches the Vol Backtester convention.")
    r = rate / 100.0
    ccy = volbt.currency(prod) if prod != _NONE else "ccy"
    in_ccy = pv != 1.0

    # ---- vol marks: the product's live ATM term curve, if it publishes one -----
    curve = _osb_atm_curve(prod, MODE) if prod != _NONE else {}
    _TENOR_LBL = {30: "1M", 91: "3M", 182: "6M", 365: "12M"}

    def _leg_vol(days: float, fallback: float) -> float:
        """Surface ATM vol interpolated to a leg's expiry — manual fallback when
        the product has no live surface."""
        return round(optbuilder.vol_at(curve, days), 1) if curve else fallback

    if prod != _NONE and curve:
        st.caption("Vol marks from the **option surface** — ATM "
                   + "  ·  ".join(f"{_TENOR_LBL[d]} {v:.1f}"
                                  for d, v in sorted(curve.items()))
                   + " — presets and blank *Vol %* cells seed from this curve, "
                     "interpolated to each leg's expiry. Type a vol to override.")
    elif prod != _NONE:
        st.caption("This product publishes **no live option surface** (or it's stale) — "
                   "vol marks stay manual.")

    # ---- presets + the defaults new/preset legs inherit -----------------------
    p1, p2, p3, p4 = st.columns([1.9, 1, 1, 1], vertical_alignment="bottom")
    preset = p1.selectbox("Preset strategy", list(optbuilder.PRESETS), key="osb_preset")
    dflt_vol = p2.number_input("Vol %", min_value=0.5, value=20.0,
                               step=0.5, format="%.1f", key="osb_dvol",
                               help="The vol presets and blank Vol % cells fall back to — "
                                    "overridden by the option surface when a product is picked.")
    dflt_month = p3.selectbox("Month", _MONTHS, key="osb_dmonth",
                              help="The expiry month presets and new legs are built with — "
                                   "legs expire on the month's 3rd Friday (the house listed-"
                                   "expiry convention). Each leg's month is editable in the "
                                   "table for calendars.")
    dflt_days = float(_MDAYS[dflt_month])

    def _leg_row(l: dict) -> dict:
        fut = l["kind"] == "Future"
        return {"Side": l["side"], "Qty": float(l["qty"]), "Type": l["kind"],
                "Strike": float(l["strike"]),
                "Month": _NO_MONTH if fut else _closest_month(l["days"]),
                "Vol %": np.nan if fut else _leg_vol(l["days"], l["vol"]),
                "Price paid": np.nan}

    def _preset_rows() -> pd.DataFrame:
        return pd.DataFrame([_leg_row(l)
                             for l in optbuilder.PRESETS[preset](F0, dflt_vol, dflt_days)])

    if p4.button("⤵️ Load preset", use_container_width=True, key="osb_load"):
        st.session_state["osb_rows"] = _preset_rows()
        st.session_state["osb_nonce"] = st.session_state.get("osb_nonce", 0) + 1
        st.rerun()

    if "osb_rows" not in st.session_state:
        st.session_state["osb_rows"] = _preset_rows()

    def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Numeric columns back to float after editor round-trips — an object
        column renders empty cells as a grey 'None' instead of blank."""
        df = df.copy()
        for c in ("Qty", "Strike", "Vol %", "Price paid"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    edited = st.data_editor(
        _clean_rows(st.session_state["osb_rows"]), num_rows="dynamic", use_container_width=True,
        key=f"osb_editor_{st.session_state.get('osb_nonce', 0)}",
        column_config={
            "Side": st.column_config.SelectboxColumn("Side", options=["Buy", "Sell"],
                                                     required=True, width="small"),
            "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, format="%d",
                                                 width="small"),
            "Type": st.column_config.SelectboxColumn("Type", options=["Call", "Put", "Future"],
                                                     required=True, width="small"),
            "Strike": st.column_config.NumberColumn(
                "Strike / entry", format="%.4f",
                help="Option strike — or the entry price for a Future leg."),
            "Month": st.column_config.SelectboxColumn(
                "Month", options=[_NO_MONTH] + _MONTHS, width="small",
                help="Option expiry month (3rd Friday). Mix months for calendars — the "
                     "payoff line draws at the FRONT month with later legs re-priced, not "
                     "expired. '—' is for Future legs."),
            "Vol %": st.column_config.NumberColumn(
                "Vol %", min_value=0.5, format="%.1f",
                help="Blank = the product's ATM surface vol at that month (or the fallback "
                     "Vol % above)."),
            "Price paid": st.column_config.NumberColumn(
                "Price paid (blank = model)", format="%.4f",
                help="The premium paid/received per unit in price points. Blank marks the "
                     "leg at the Black-76 model value — type your fill to anchor to it."),
        })
    st.session_state["osb_rows"] = edited

    legs = []
    for _r in edited.to_dict("records"):
        if not _r.get("Type") or not _r.get("Side") or pd.isna(_r.get("Strike")):
            continue
        _mlbl = _r.get("Month")
        _days = float(_MDAYS.get(_mlbl, dflt_days))
        legs.append({"side": _r["Side"], "qty": int(_r["Qty"]) if pd.notna(_r["Qty"]) else 1,
                     "kind": _r["Type"], "strike": float(_r["Strike"]), "days": _days,
                     "month": _mlbl if _mlbl in _MDAYS else
                     (_NO_MONTH if _r["Type"] == "Future" else dflt_month),
                     "vol": float(_r["Vol %"]) if pd.notna(_r["Vol %"])
                     else _leg_vol(_days, dflt_vol),
                     "premium": None if pd.isna(_r.get("Price paid")) else float(_r["Price paid"]),
                     "prem_src": "model" if pd.isna(_r.get("Price paid")) else "screen"})

    # ---- add-leg buttons: build spreads / hedge without hunting the table's "+" row ----
    def _append_leg(row: dict) -> None:
        """Append to what's ON SCREEN (edited), so in-progress tweaks survive."""
        st.session_state["osb_rows"] = pd.concat(
            [edited, pd.DataFrame([row])], ignore_index=True)
        st.session_state["osb_nonce"] = st.session_state.get("osb_nonce", 0) + 1

    _atm = optbuilder.atm_strike(F0)
    _blank_opt = lambda kind: {"Side": "Buy", "Qty": 1.0, "Type": kind, "Strike": _atm,
                               "Month": dflt_month,
                               "Vol %": _leg_vol(dflt_days, dflt_vol), "Price paid": np.nan}
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("➕ Add call leg", use_container_width=True, key="osb_add_c"):
        _append_leg(_blank_opt("Call"))
        st.rerun()
    if a2.button("➕ Add put leg", use_container_width=True, key="osb_add_p"):
        _append_leg(_blank_opt("Put"))
        st.rerun()
    if a3.button("➕ Add future @ spot", use_container_width=True, key="osb_add_f",
                 help="An outright futures leg entered at the current underlying price — "
                      "flip Side / edit the entry in the table."):
        _append_leg({"Side": "Buy", "Qty": 1.0, "Type": "Future", "Strike": round(F0, 4),
                     "Month": _NO_MONTH, "Vol %": np.nan, "Price paid": np.nan})
        st.rerun()
    _net_delta = optbuilder.totals_greeks(legs, F0, 0.0, r)["delta"] if legs else 0.0
    _hlots = int(round(abs(_net_delta)))
    if a4.button(f"⚖️ Delta-hedge  (Δ {_net_delta:+.2f})", use_container_width=True,
                 key="osb_hedge", disabled=_hlots == 0,
                 help="Adds ONE futures leg at the current spot, sized to the nearest whole "
                      "lot against the position's delta today. Greyed out when the net delta "
                      "rounds to zero lots."):
        _hside = "Sell" if _net_delta > 0 else "Buy"
        _append_leg({"Side": _hside, "Qty": float(_hlots), "Type": "Future",
                     "Strike": round(F0, 4), "Month": _NO_MONTH,
                     "Vol %": np.nan, "Price paid": np.nan})
        st.toast(f"Hedge added: {_hside} {_hlots} future(s) @ {F0:,.4g} — "
                 f"residual Δ {_net_delta - _hlots * (1 if _net_delta > 0 else -1):+.2f}",
                 icon="⚖️")
        st.rerun()
    st.caption("Rows can also be added, edited or deleted directly in the table (the blank "
               "row at the bottom adds; tick a row's left edge and press Delete to remove). "
               "The hedge sizes off **today's** delta — re-hedge after big edits.")

    if not legs:
        st.info("Add at least one leg (or load a preset) to see the payoff.")
        return

    # freeze entry premiums at BASE vols so the scenario sliders re-mark the
    # position without silently moving its entry price
    legs = [dict(l, premium=optbuilder.entry_premium(l, F0, r)) for l in legs]
    front = optbuilder.front_days(legs)

    # ---- scenario ------------------------------------------------------------
    s1, s2 = st.columns([2.2, 1])
    d_now = s1.slider("Scenario — days from today", 0, max(int(front), 1),
                      0, key="osb_dnow",
                      help="The dashed line: the position re-priced this many days in, "
                           "legs' remaining life reduced accordingly.")
    vshift = s2.slider("Vol shift (pts)", -15.0, 15.0, 0.0, 0.5, key="osb_vshift",
                       help="Added to every option leg's vol for the dashed scenario line.")
    scn_legs = [dict(l, vol=l["vol"] + vshift) for l in legs]

    # ---- headline numbers ------------------------------------------------------
    def _fmt(x: float) -> str:
        pts = f"{x:,.4f}".rstrip("0").rstrip(".")
        return f"{x * pv:,.0f} {ccy}  ({pts} pts)" if in_ccy else f"{pts} pts"

    net = optbuilder.net_premium(legs, F0, r)
    (mp, mp_unb), (ml, ml_unb) = optbuilder.max_profit_loss(legs, F0, r)
    bes = optbuilder.breakevens(legs, F0, r)
    _vols = [(l["vol"], l["qty"]) for l in legs if l["kind"] != "Future"]
    ref_vol = (sum(v * q for v, q in _vols) / sum(q for _, q in _vols)) if _vols else None
    prob = optbuilder.pop(legs, F0, ref_vol, front, r) if ref_vol and front > 0 else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Net " + ("debit" if net >= 0 else "credit"), _fmt(abs(net)),
              help="What the package costs (debit) or collects (credit) to put on, "
                   "model-priced where premiums were left blank.")
    m2.metric("Max profit", "Unlimited ⬆" if mp_unb else _fmt(mp),
              help="Best P&L at the front expiry across underlying prices (0 included).")
    m3.metric("Max loss", "Unlimited ⬇" if ml_unb else _fmt(ml),
              help="Worst P&L at the front expiry across underlying prices (0 included).")
    m4.metric("Breakeven" + ("s" if len(bes) != 1 else ""),
              "  ·  ".join(f"{b:,.2f}" for b in bes) if bes else "none",
              help="Underlying prices where the front-expiry P&L crosses zero.")
    m5.metric("P(profit)", f"{prob * 100:.0f} %" if prob is not None else "n/a",
              help="Model estimate: probability the front-expiry P&L ends positive under "
                   "a drift-free lognormal at the legs' average vol — not a market-implied "
                   "probability.")

    # ---- payoff chart ----------------------------------------------------------
    cc = brand.chart_colors()
    lo, hi = optbuilder.grid_range(legs, F0)
    xs = np.linspace(lo, hi, 241)
    exp_lbl = f"At front expiry (T+{front:.0f}d)"
    scn_lbl = f"T+{d_now}d, vol {vshift:+.1f}"
    dfc = pd.DataFrame({
        "F": xs,
        exp_lbl: [optbuilder.strategy_pnl(legs, float(x), F0, None, r) * pv for x in xs],
        scn_lbl: [optbuilder.strategy_pnl(scn_legs, float(x), F0, float(d_now), r) * pv
                  for x in xs],
    })
    y_t = f"P&L ({ccy})" if in_ccy else "P&L (price points)"
    dfc["pos"] = dfc[exp_lbl].clip(lower=0.0)
    dfc["neg"] = dfc[exp_lbl].clip(upper=0.0)
    base = alt.Chart(dfc)
    shade = base.mark_area(opacity=0.10, color=cc["long"]).encode(
        x=alt.X("F:Q", title="underlying price at expiry", scale=alt.Scale(zero=False)),
        y=alt.Y("pos:Q", title=y_t)) + \
        base.mark_area(opacity=0.10, color=cc["short"]).encode(x="F:Q", y="neg:Q")
    long_df = dfc.melt("F", value_vars=[exp_lbl, scn_lbl], var_name="Series", value_name="pnl")
    lines = alt.Chart(long_df).mark_line(strokeWidth=2.4).encode(
        x=alt.X("F:Q", title="underlying price at expiry", scale=alt.Scale(zero=False)),
        y=alt.Y("pnl:Q", title=y_t),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=[exp_lbl, scn_lbl],
                                        range=[cc["ink"], cc["accent"]])),
        strokeDash=alt.StrokeDash("Series:N", legend=None,
                                  scale=alt.Scale(domain=[exp_lbl, scn_lbl],
                                                  range=[[1, 0], [6, 4]])),
        tooltip=[alt.Tooltip("F:Q", format=",.2f", title="underlying"),
                 "Series:N", alt.Tooltip("pnl:Q", format=",.2f", title="P&L")])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=cc["muted"], strokeWidth=1).encode(y="y:Q")
    spot = alt.Chart(pd.DataFrame({"x": [F0]})).mark_rule(
        color=cc["muted"], strokeDash=[3, 3]).encode(
        x="x:Q", tooltip=[alt.Tooltip("x:Q", format=",.2f", title="spot")])
    chart = shade + zero + spot
    if bes:
        chart += alt.Chart(pd.DataFrame({"x": bes})).mark_rule(
            color=cc["series"], strokeDash=[2, 4]).encode(
            x="x:Q", tooltip=[alt.Tooltip("x:Q", format=",.2f", title="breakeven")])
    chart += lines
    brand.show_chart(chart.properties(height=420).interactive(bind_y=False))
    st.caption("Dotted vertical = current underlying; dashed blue verticals = breakevens. "
               "Scroll / drag to zoom the price axis.")

    # ---- greeks ----------------------------------------------------------------
    g = optbuilder.totals_greeks(scn_legs, F0, float(d_now), r)
    _gf = (lambda v: f"{v * pv:+,.0f} {ccy}") if in_ccy else (lambda v: f"{v:+,.4f}")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta", f"{g['delta']:+,.3f}",
              help="Underlying-equivalent position, in units of the underlying (lots when "
                   "each leg's Qty is lots).")
    g2.metric("Gamma", f"{g['gamma']:+,.5f}", help="Delta change per 1.00 move in the underlying.")
    g3.metric("Vega", _gf(g["vega"]), help="P&L per +1 vol point, all legs shifted together.")
    g4.metric("Theta / day", _gf(g["theta"]), help="P&L per calendar day, other things equal.")
    st.caption(f"Greeks at the scenario mark — spot {F0:,.4g}, T+{d_now}d, vol {vshift:+.1f}. "
               + ("Vega / theta scaled by the point value; delta and gamma stay in "
                  "underlying units." if in_ccy else
                  "Price-point units — set a point value (or pick a product) for money terms."))

    # ---- leg blotter -------------------------------------------------------------
    rows = []
    for l in legs:
        m = optbuilder.leg_model(l, F0, 0.0, r)
        rows.append({
            "Leg": f"{l['side']} {l['qty']} {l['kind']} "
                   + (f"{l['strike']:,.4g}" if l["kind"] != "Future" else f"@ {l['strike']:,.4g}")
                   + (f" · {l.get('month', '')} ({l['days']:.0f}d)"
                      if l["kind"] != "Future" else ""),
            "Vol %": l["vol"] if l["kind"] != "Future" else None,
            "Premium": l["premium"], "Model now": m["price"],
            "Delta": (1 if l["side"] == "Buy" else -1) * l["qty"] * m["delta"],
            "Vega": (1 if l["side"] == "Buy" else -1) * l["qty"] * m["vega"],
            "Theta/d": (1 if l["side"] == "Buy" else -1) * l["qty"] * m["theta"],
        })
    st.markdown("**Legs at entry** — the premium each leg trades at (typed or model) and "
                "its signed greeks today.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={
                     "Vol %": st.column_config.NumberColumn(format="%.1f"),
                     "Premium": st.column_config.NumberColumn(format="%.4f"),
                     "Model now": st.column_config.NumberColumn(format="%.4f"),
                     "Delta": st.column_config.NumberColumn(format="%+.3f"),
                     "Vega": st.column_config.NumberColumn(format="%+.4f"),
                     "Theta/d": st.column_config.NumberColumn(format="%+.4f"),
                 })
    st.caption("Premiums / greeks per unit in price points × leg Qty; multiply by the point "
               "value for money. European exercise, flat per-leg vols — a payoff modeller, "
               "not an execution price.")

    # ---- PDF ticket ------------------------------------------------------------
    st.divider()
    t1, t2 = st.columns([2.2, 1], vertical_alignment="bottom")
    ticket_title = t1.text_input("Ticket title", value=preset, key="osb_title",
                                 help="The strategy name on the PDF — defaults to the preset; "
                                      "rename it for a reshaped / custom position.")
    if t2.button("🧾 Build PDF ticket", use_container_width=True, key="osb_pdf_btn"):
        with st.spinner("Building the ticket…"):
            try:
                payload = {
                    "asof": date.today().isoformat(),
                    "title": ticket_title.strip() or preset,
                    "underlying": INSTRUMENTS[prod][0] if prod != _NONE else "Manual underlying",
                    "ticker": prod if prod != _NONE else "",
                    "spot": F0, "pv": pv, "ccy": ccy, "in_ccy": in_ccy, "rate": rate,
                    "vol_source": ("live ATM option surface (1M–12M, interpolated to each "
                                   "leg's expiry)") if curve else "manual per-leg vols",
                    "legs": legs, "net": net,
                    "mp": mp, "mp_unb": mp_unb, "ml": ml, "ml_unb": ml_unb,
                    "bes": bes, "pop": prob, "front": front,
                    "greeks": optbuilder.totals_greeks(legs, F0, 0.0, r),
                    "grid": [float(x) for x in xs],
                    "exp_pnl": dfc[exp_lbl].tolist(), "scn_pnl": dfc[scn_lbl].tolist(),
                    "exp_lbl": exp_lbl, "scn_lbl": scn_lbl,
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "optbuilder.json"
                    _out = Path(_t) / "Option_Strategy_Ticket.pdf"
                    _in.write_text(json.dumps(payload))
                    rres = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "optbuilderreport.py"),
                         str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if rres.returncode == 0 and _out.exists():
                        st.session_state["osb_pdf"] = _out.read_bytes()
                    else:
                        st.error("Ticket failed:\n\n"
                                 + (rres.stderr or rres.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Ticket failed:\n\n{e}")
    if st.session_state.get("osb_pdf"):
        st.download_button("⬇️  Download Strategy Ticket", data=st.session_state["osb_pdf"],
                           file_name="Option_Strategy_Ticket.pdf", mime="application/pdf")
        email_report_ui("osb_email", "optbuilder", st.session_state["osb_pdf"],
                        subject="BASIS — Option Strategy Ticket",
                        attachment_name="Option_Strategy_Ticket.pdf")


@st.cache_data(show_spinner=False, ttl=1800)
def _sc_instruments(metric: str, asof_iso: str, sectors: tuple, mode: str):
    """Product-level matrices, cached so widget reruns don't re-pull history
    (`mode` keys the cache to the data source)."""
    return sectorcorr.instrument_corr(metric, date.fromisoformat(asof_iso), list(sectors))


@st.cache_data(show_spinner=False, ttl=1800)
def _sc_breaks(metric: str, asof_iso: str, mode: str):
    return sectorcorr.top_breaks(metric, date.fromisoformat(asof_iso))


@st.cache_data(show_spinner=False, ttl=1800)
def _sc_divindex(metric: str, asof_iso: str, mode: str):
    return sectorcorr.diversification_index(metric, date.fromisoformat(asof_iso))


@st.cache_data(show_spinner=False, ttl=1800)
def _sc_extremes(asof_iso: str, mode: str):
    """Today's correlation-break alerts (returns metric) for the Home banner."""
    return sectorcorr.percentile_extremes("realized", date.fromisoformat(asof_iso))


def render_sector_correlations() -> None:
    import altair as alt

    st.subheader("🔗  Product Correlations — inside and across sectors")
    _render_corr_break_banner()
    st.caption(
        "Pick a sector to see how its **products** move against each other — gold vs silver vs "
        "copper, Bund vs 10Y vs 5Y — or several sectors for the cross-sector detail. Daily "
        "changes (log returns for price, vol-point diffs for the 1M ATM implied vol) are "
        "correlated over the trailing **1 year** (the normal relationship) and **1 month** (the "
        "relationship as it trades now) — the same 252 / 21 session windows as the backtester's "
        "pair panel. The third map is **1M minus 1Y**: a strongly negative cell is a pair whose "
        "usual co-movement has broken down; a strongly positive one is unusual lockstep.")

    c0, c1, c2 = st.columns([1.7, 1.1, 1])
    picks = c0.multiselect("Sectors", sectorcorr.SECTOR_ORDER, default=["Metals"],
                           key="sc_sectors",
                           help="One sector shows its internal structure; add more for the "
                                "cross-sector detail (e.g. Bonds + STIRs).")
    metric = c1.radio("Correlate", ["Returns", "IV changes", "Realized vol"], index=2, horizontal=True, key="sc_metric",
                      help="Returns = settlement-price log returns (price direction). IV changes = daily moves in "
                           "the 1M ATM implied vol (how the vol markets re-mark together). Realized vol = daily "
                           "moves in 1M realized vol (whether they actually turn volatile in sync).")
    metric_key = {"IV changes": "iv", "Realized vol": "realized"}.get(metric, "returns")
    asof = c2.date_input("As of", value=date.today(), max_value=date.today(), key="sc_asof",
                         help="Correlations use data up to this date — wind it back to see the "
                              "map as it stood before an event.")
    if not picks:
        st.info("Pick at least one sector.")
        return

    ci = _sc_instruments(metric_key, asof.isoformat(), tuple(sorted(picks)), MODE)
    if ci is None:
        st.info("Not enough products with history in that selection for this date.")
        return
    sel_dropped = [t for t in ci.dropped if universe.asset(t) in set(picks)]
    if sel_dropped:
        st.caption(f"**{len(sel_dropped)}** selected products excluded (stale vol surface or "
                   "too little history): " + ", ".join(universe.name(t) for t in sel_dropped[:6])
                   + ("…" if len(sel_dropped) > 6 else ""))

    # human names on the axes; disambiguate duplicates (cash/futures twins)
    names = {}
    for t in ci.labels:
        nm = universe.name(t)
        names[t] = f"{nm} ({t.split()[0]})" if any(
            universe.name(o) == nm for o in ci.labels if o != t) else nm
    order = [names[t] for t in ci.labels]
    L = ci.long_.rename(index=names, columns=names)
    S = ci.short_.rename(index=names, columns=names)
    D = ci.diff.rename(index=names, columns=names)
    P = ci.pctl.rename(index=names, columns=names) if ci.pctl is not None else None
    hgt = max(340, 26 * len(ci.labels))
    text_ok = len(ci.labels) <= 16

    def _tidy(mat, labels=None):
        d = mat.copy()
        if labels:
            d = d.rename(index=labels, columns=labels)
        d.index.name = "row"
        return d.reset_index().melt("row", var_name="col", value_name="corr").dropna(subset=["corr"])

    def _heat(tidy, ax_order, title, *, domain, fmt="+.2f", cell_text=True,
              height=340, extra_tips=()):
        tips = [alt.Tooltip("row:N", title=""), alt.Tooltip("col:N", title="vs"),
                alt.Tooltip("corr:Q", title=title, format=fmt), *extra_tips]
        enc_x = alt.X("col:N", sort=ax_order, title=None,
                      axis=alt.Axis(labelAngle=-40, labelFontSize=11, orient="top", labelLimit=140))
        enc_y = alt.Y("row:N", sort=ax_order, title=None,
                      axis=alt.Axis(labelFontSize=11, labelLimit=140))
        base = alt.Chart(tidy)
        rect = base.mark_rect(stroke=brand.palette()["canvas"], strokeWidth=2.1).encode(
            x=enc_x, y=enc_y,
            color=alt.Color("corr:Q",
                            scale=alt.Scale(scheme="redblue", domain=domain, reverse=True),
                            legend=alt.Legend(title=None, format="+.1f", gradientLength=160)),
            tooltip=tips)
        layers = [rect]
        if cell_text:
            # dark cells at both scale ends need light text; the pale middle needs dark
            span = max(abs(domain[0]), abs(domain[1]))
            layers.append(base.mark_text(fontSize=11).encode(
                x=enc_x, y=enc_y, text=alt.Text("corr:Q", format=fmt),
                color=alt.condition(f"abs(datum.corr) > {span * 0.55}",
                                    alt.value("#F5F5F5"), alt.value("#1A1A1A")),
                tooltip=tips))
        return alt.layer(*layers).properties(height=height, title=title)

    h1, h2 = st.columns(2)
    with h1:
        brand.show_chart(_heat(_tidy(L), order, "1-year (252 sessions)", domain=[-1, 1],
                               cell_text=text_ok, height=hgt))
    with h2:
        brand.show_chart(_heat(_tidy(S), order, "1-month (21 sessions)", domain=[-1, 1],
                               cell_text=text_ok, height=hgt))

    # ---- the regime-shift map: 1M − 1Y, with each cell's percentile context ----
    dt = _tidy(D)
    dt["c1y"] = [L.loc[r, c] for r, c in zip(dt["row"], dt["col"])]
    dt["c1m"] = [S.loc[r, c] for r, c in zip(dt["row"], dt["col"])]
    extra = [alt.Tooltip("c1y:Q", title="1Y", format="+.2f"),
             alt.Tooltip("c1m:Q", title="1M", format="+.2f")]
    if P is not None:
        dt["pctl"] = [P.loc[r, c] for r, c in zip(dt["row"], dt["col"])]
        extra.append(alt.Tooltip("pctl:Q", title="1M pctl of its 1Y range", format=".0f"))
    span = float(np.ceil(dt["corr"].abs().max() * 10) / 10) if len(dt) else 0.2
    span = max(span, 0.2)
    brand.show_chart(_heat(dt, order, "1M − 1Y — where the regime has shifted",
                           domain=[-span, span], cell_text=text_ok, height=hgt,
                           extra_tips=tuple(extra)))
    st.caption(
        "Hover a cell for its **percentile**: where this month's correlation sits inside a year "
        "of its own rolling 1-month correlations. A pair at the **5th percentile** is a genuine "
        "break; a −0.2 shift on a pair that swings ±0.4 every month is just its usual noise. "
        "The returns metric runs on the trend universe (no vol-only cash-index twins); the IV "
        "metric on every product with a live surface.")

    # ---- diversification index: is the whole book one trade? -----------------
    st.divider()
    st.markdown("**Diversification index — average cross-sector correlation, rolling 1M**")
    di = _sc_divindex(metric_key, asof.isoformat(), MODE)
    if di is None:
        st.info("Not enough history to draw the diversification index for this date.")
    else:
        cc0 = brand.chart_colors()
        dd = di.reset_index()
        dd.columns = ["date", "Average (signed)", "Average |corr|"]
        dmelt = dd.melt("date", var_name="Series", value_name="corr")
        ddom = ["Average (signed)", "Average |corr|"]
        dline = alt.Chart(dmelt).mark_line(strokeWidth=2.6).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("corr:Q", title="avg pairwise correlation",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=ddom,
                            range=[cc0["series"], cc0["accent"]]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("Series:N"),
                     alt.Tooltip("corr:Q", format="+.2f")])
        dmean = alt.Chart(pd.DataFrame({"y": [float(di["avg"].mean())]})).mark_rule(
            color=cc0["muted"], strokeDash=[5, 3]).encode(y="y:Q")
        brand.show_chart((dline + dmean).properties(height=240))
        cur, cur_abs = float(di["avg"].iloc[-1]), float(di["avg_abs"].iloc[-1])
        dpct = float((di["avg_abs"] <= cur_abs).mean() * 100.0)
        st.caption(
            f"Every cross-sector composite pair's rolling 21-session correlation, averaged. "
            f"Latest: **{cur:+.2f}** signed / **{cur_abs:.2f}** absolute — the absolute line sits "
            f"in the **{dpct:.0f}th percentile** of the past year. A rising absolute line means "
            "the sectors are increasingly moving as one trade (risk-on / risk-off), so the book "
            "carries more common risk than the same positions would in a normal regime; the "
            "signed line (dashed rule = its 1-year mean) shows which way the convergence leans.")

    # ---- the names behind the map ------------------------------------------
    st.divider()
    st.markdown(f"**Biggest correlation breaks — product pairs ({metric.lower()})**")
    bt = _sc_breaks(metric_key, asof.isoformat(), MODE)
    if bt.empty:
        st.info("Not enough data to rank product pairs for this date.")
    else:
        disp = pd.DataFrame({
            "Pair": [f"{universe.name(a)}  ↔  {universe.name(b)}" for a, b in zip(bt["a"], bt["b"])],
            "Sectors": [sa if sa == sb else f"{sa} / {sb}"
                        for sa, sb in zip(bt["sector_a"], bt["sector_b"])],
            "1Y": bt["corr_1y"], "1M": bt["corr_1m"], "Δ (1M−1Y)": bt["diff"],
            "1M pctl (1Y range)": bt["pctl"],
        })
        brand.themed_dataframe(
            disp, {"1Y": "{:+.2f}", "1M": "{:+.2f}", "Δ (1M−1Y)": "{:+.2f}",
                   "1M pctl (1Y range)": "{:.0f}%"},
            na_rep="—", height=int(38 + 35 * len(disp)))
        st.caption("Every product pair in the book, ranked by |1M − 1Y|. A pair at an extreme "
                   "percentile of its own range may be worth a closer look on the backtester's "
                   "pair panel, which draws the rolling picture behind the number.")

    # ---- branded PDF of exactly this view ------------------------------------
    st.divider()
    if st.button("📈 Generate Correlation Report (visual PDF)", type="primary", key="sc_pdf_btn"):
        with st.spinner("Rendering the correlation report…"):
            try:
                def _mat(m):
                    return [[None if pd.isna(v) else float(v) for v in row] for row in m.values]
                payload = {
                    "asof": asof.isoformat(), "sectors": picks, "mode": MODE,
                    "metric_label": {"iv": "1M ATM implied-vol changes",
                                     "realized": "1M realized-vol changes"}.get(
                                         metric_key, "settlement-price log returns"),
                    "labels": order, "m1y": _mat(L), "m1m": _mat(S), "diff": _mat(D),
                    "diff_span": span,
                }
                if di is not None:
                    payload.update(div_dates=[x.isoformat() for x in di.index.date],
                                   div_avg=[float(v) for v in di["avg"]],
                                   div_abs=[float(v) for v in di["avg_abs"]])
                if not bt.empty:
                    payload["breaks"] = [
                        {"pair": f"{universe.name(a)} ↔ {universe.name(b)}",
                         "sectors": sa if sa == sb else f"{sa} / {sb}",
                         "c1y": float(y1), "c1m": float(m1), "d": float(dd1),
                         "pctl": None if pd.isna(p1) else float(p1)}
                        for a, b, sa, sb, y1, m1, dd1, p1 in zip(
                            bt["a"], bt["b"], bt["sector_a"], bt["sector_b"],
                            bt["corr_1y"], bt["corr_1m"], bt["diff"], bt["pctl"])]
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "sectorcorr.json"
                    _out = Path(_t) / "Product_Correlations.pdf"
                    _in.write_text(json.dumps(payload), encoding="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "sectorcorrreport.py"),
                         str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["sc_pdf"] = _out.read_bytes()
                    else:
                        st.error("Report failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed:\n\n{e}")
    if st.session_state.get("sc_pdf"):
        st.download_button("⬇️  Download Correlation Report", data=st.session_state["sc_pdf"],
                           file_name="Product_Correlations.pdf", mime="application/pdf")
        email_report_ui("sc_email", "sectorcorr", st.session_state["sc_pdf"],
                        subject="BASIS — Product Correlations",
                        attachment_name="Product_Correlations.pdf")


# Seed the landing view before the sidebar renders, so its nav highlights correctly.
st.session_state.setdefault("active", "Home")
st.session_state.setdefault("side", "FICC")

# The BASIS logo doubles as the Home button: an invisible, full-size button is overlaid exactly on
# top of the logo lockup (via the keyed container below), so clicking the logo routes through the
# normal nav callback — same mechanism as every other sidebar nav item.
_LOGO_HOME_CSS = """<style>
div.st-key-basis_logo_home { position: relative; }
/* Overlay the button's element-container on the whole logo. Anchor on the element-container (the
   last child of the keyed block, which is positioned relative to it) — NOT on the inner stButton,
   whose own element-container is position:relative and collapses to height 0. The logo's markdown
   overflows its Streamlit wrapper by ~1rem (the tagline escapes it), so extend the bottom by 1rem
   to cover the full lockup; the next sidebar element begins exactly at the logo's true bottom, so
   this meets it without overlapping. */
div.st-key-basis_logo_home > div[data-testid="stElementContainer"]:last-child {
    position: absolute; top: 0; left: 0; right: 0; bottom: -1rem; margin: 0; z-index: 5;
}
div.st-key-basis_logo_home div[data-testid="stButton"] { height: 100%; }
div.st-key-basis_logo_home div[data-testid="stButton"] button {
    width: 100%; height: 100%; min-height: 0; padding: 0;
    opacity: 0; cursor: pointer; border: none; background: transparent; box-shadow: none;
}
/* Hide stale sidebar entries IMMEDIATELY on a rerun. Streamlit only removes replaced elements
   when the whole run finishes, so switching FICC <-> Equities left the old side's nav buttons
   lingering greyed-out while the new page pulled its data. Scoped to the sidebar: the main
   column keeps the default grey-out (hiding it there would blank charts on every rerun). */
section[data-testid="stSidebar"] div[data-testid="stElementContainer"][data-stale="true"] {
    display: none;
}
</style>"""

# ----- sidebar: navigation -------------------------------------------------
with st.sidebar:
    st.markdown(_LOGO_HOME_CSS, unsafe_allow_html=True)
    _side = st.session_state.get("side", "FICC")
    _home_dest = "eq:Home" if _side == "Equities" else "Home"
    # Logo + the FICC/Equities switch live in one sticky wrapper (styled in brand._CSS) so
    # they stay pinned at the top of the sidebar while the nav list scrolls beneath them.
    with st.container(key="basis_sidebar_sticky"):
        with st.container(key="basis_logo_home"):
            brand.sidebar_logo()
            st.button("Home", key="basis_logo_home_btn", on_click=_go, args=(_home_dest,),
                      use_container_width=True)
        # TERMINAL: the cross-asset home, above the desk split (handoff §sidebar).
        st.markdown('<div class="bt-sect">Terminal</div>', unsafe_allow_html=True)
        _nav_button("00 · Overview", _home_dest)
        # DESK: FICC | EQUITIES segmented control.
        st.markdown('<div class="bt-sect">Desk</div>', unsafe_allow_html=True)
        _sc1, _sc2 = st.columns(2, gap="small")
        _sc1.button("FICC", key="side_ficc", use_container_width=True,
                    type="primary" if _side == "FICC" else "secondary", on_click=_set_side, args=("FICC",))
        _sc2.button("Equities", key="side_equities", use_container_width=True,
                    type="primary" if _side == "Equities" else "secondary", on_click=_set_side, args=("Equities",))
    snap = _load_snap()
    _data_badge(snap, _side)
    df, meta = load_signals()
    try:        # desk scope row (desk-aware): FICC = markets/signals, Equities = stocks/indices
        if _side == "Equities":
            _uni = equities.cached_universe()
            _n_stk = len({c["ticker"] for rows in _uni.values() for c in rows})
            st.markdown(f'<div class="bt-sect" style="margin-top:.15rem">'
                        f'{_n_stk} stocks · {len(_uni)} indices</div>', unsafe_allow_html=True)
        else:
            _n_mkts = len(universe.enabled_tickers())
            _n_sigs = len(df) if df is not None else 0
            st.markdown(f'<div class="bt-sect" style="margin-top:.15rem">'
                        f'{_n_mkts} markets · {_n_sigs} signals</div>', unsafe_allow_html=True)
    except Exception:
        pass
    if _side == "FICC":
        st.markdown('<div class="bt-sect">FICC modules</div>', unsafe_allow_html=True)
        # Market Information (Reports Calendar / Market Hours / Block Sizes / Fut-Yield / Universe)
        # collapses to one entry; Trade Testing (Fed Path + Vol Backtester) does the same, numbered
        # in after the strategy groups. Both carry the tab-row switcher (_render_group_tabs).
        # Morning Coffee is reached from the Home page's Data row.
        _nav_button("01 · Market Information", "Release Calendar")
        _nav_button("02 · Confluence", "Confluence")
        _n_mod = 2
        for _group, _strats in NAV_GROUPS.items():
            # Each strategy group collapses to ONE sidebar entry; its members are reached from within
            # that page — the TA overview's "By strategy" drill-down grid, or the tab-row switcher
            # (_render_group_tabs) for Volatility / Positioning & Flow / Fundamentals.
            _n_mod += 1
            if _group == "Technical Analysis":
                _nav_button(f"{_n_mod:02d} · Technical Analysis", "Technical Analysis")
            elif _group == "Volatility":
                _nav_button(f"{_n_mod:02d} · Volatility", "Volatility")
            elif _group == "Positioning & Flow":
                _nav_button(f"{_n_mod:02d} · Positioning & Flow", "COT Reports")
            elif _group == "Fundamentals":
                _nav_button(f"{_n_mod:02d} · Fundamentals", "AG Fundamentals")
            else:                                   # any future group: caption + its individual buttons
                st.caption(_group)
                for _s in _strats:
                    _nav_button(_s, _s)
        _n_mod += 1
        _nav_button(f"{_n_mod:02d} · Correlations", "Product Correlations")
        _n_mod += 1
        _nav_button(f"{_n_mod:02d} · Trade Testing", "Fed Path")
    else:
        st.markdown('<div class="bt-sect">Equities modules · US + EU indices</div>',
                    unsafe_allow_html=True)
        # No "Equities Home" entry — the Equities desk segment (and the logo) already land there.
        _nav_button("01 · Technical Analysis", "eq:Technical Analysis")
        _nav_button("02 · Company Fundamentals", "eq:Fundamentals")
        _nav_button("03 · Earnings Calendar", "eq:Earnings")
        _nav_button("04 · Single Stock Correlations", "eq:Correlations")
        _nav_button("05 · Index Dispersion", "eq:Dispersion")
        _nav_button("06 · Client ETFs", "eq:ETFs")
    # Cross-asset / System: shared across BOTH desks, not FICC-only.
    st.markdown('<div class="bt-sect">Cross-asset</div>', unsafe_allow_html=True)
    _nav_button("Strategy Builder", "Strategy Builder")
    st.markdown('<div class="bt-sect">System</div>', unsafe_allow_html=True)
    _nav_button("Alert Settings", "Recipients")
    _nav_button("Data Health", "Data health")
    # footer status rows (handoff): SIGNALS · FEED · DATA
    _feed = {"bloomberg": ("BBG live", "#46C58A"),
             "snapshot": ("snapshot", "#F5C518")}.get(MODE, ("demo", "#EC6A57"))
    brand.sidebar_footer([
        ("signals", _to_et(meta.get("as_of", "n/a")), ""),
        ("feed", _feed[0], _feed[1]),
        ("data", str((snap or {}).get("as_of", "—")), ""),
    ])

# ----- fixed top bar (same on every page, stays while scrolling): world clocks
# over the masthead row (logo left · module breadcrumb · ET clock · theme toggle).
# Pinned by brand CSS (.st-key-basis_topbar wrapper -> position:fixed).
_active_dest = st.session_state.get("active", "Home")
if _active_dest in ("Home", "eq:Home"):
    _crumb = None                                      # Overview: the tagline
else:
    _crumb = f"{_side} desk · {_active_dest.removeprefix('eq:')}"
with st.container(key="basis_topbar"):
    _tb_cl, _tb_tg = st.columns([0.94, 0.06], vertical_alignment="center")
    with _tb_cl:
        _world_clocks()
    with _tb_tg:
        brand.theme_toggle()           # fills the space right of the clocks
    brand.masthead(_crumb, toggle=False)
try:      # quote rail: the day's biggest movers from the same frame as Overnight moves
    _rail = _ficc_moves_frame()
    if not _rail.empty:
        _rail = _rail.reindex(_rail["sigma"].abs().sort_values(ascending=False).index).head(8)
        brand.ticker_rail([
            {"sym": r["Market"], "last": f"{r['last']:g}",
             "chg": f"{r['pct']:+.2f}%", "up": r["pct"] >= 0}
            for _, r in _rail.iterrows()])
except Exception:
    pass

# Nav clicks flag a scroll reset — the destination page should open at the top (Streamlit
# otherwise keeps the previous page's scroll position across the rerun). A TIME WINDOW, not
# a one-shot flag: pages that st.rerun() on entry (e.g. Technical Analysis) abort the first
# run — a popped flag died with it and the page kept the old scroll. Any run starting within
# the window emits the reset, so chained reruns still land at the top.
import streamlit.components.v1 as components
_nav_ts = st.session_state.get("_scroll_top_ts", 0)
# This slot renders on EVERY run — a no-op outside the nav window. When it rendered only
# after a nav, the next run (any button click) had one fewer element above the page body,
# so Streamlit's positional diffing mismatched everything below and showed the previous
# run's widgets as faded DUPLICATES under the fresh ones during long reruns.
# keyed container so the theme CSS can hide this block entirely — the 0-height iframe
# would otherwise still eat a 16px flex gap between masthead and content
with st.container(key="basis_scroll_top"):
    if time.time() - _nav_ts < 1.5:
        components.html(
            # Re-assert top for ~2.5s (heavy pages keep drawing in and can carry the old
            # scroll offset back), but stop instantly if the user scrolls on purpose.
            # The nav timestamp is baked into the markup: identical content would let the
            # frontend REUSE the previous iframe without re-running its script (the scroll
            # then silently doesn't happen — the "works once after a restart" bug).
            f"<script>/* nav:{_nav_ts} */"
            "const P = window.parent, D = P.document;"
            "let cancelled = false;"
            "const cancel = () => { cancelled = true; };"
            "['wheel','touchstart','keydown'].forEach(ev =>"
            "  D.addEventListener(ev, cancel, {once:true, passive:true}));"
            "const up = () => {"
            "  if (cancelled) return;"
            "  D.querySelectorAll('section[data-testid=\"stMain\"],"
            " [data-testid=\"stAppViewContainer\"]')"
            "    .forEach(el => el.scrollTo({top: 0, behavior: 'instant'}));"
            "  P.scrollTo({top: 0, behavior: 'instant'});"
            "};"
            "up(); [150, 400, 900, 1600, 2500].forEach(t => setTimeout(up, t));"
            "</script>", height=0)
    else:
        components.html("<script>/* idle */</script>", height=0)

# ----- default landing view -----------------------------------------------
if "active" not in st.session_state:
    st.session_state.active = "Home"
active = st.session_state.active
side = st.session_state.get("side", "FICC")

# ----- the Universe editor page (defined here, rendered from the dispatch) --
def render_universe():
    st.subheader("🗂️  Product universe")
    st.caption(
        "The whole book — **every** strategy and report runs off this one list. Add a "
        "row to add a product everywhere at once; delete a row to drop it. **Futures** "
        "with listed options need only ticker / name / asset / region. **FX** also needs "
        "a *vol source* (the OTC 1-month vol, e.g. `EURUSDV1M Curncy`); **cash indices** "
        "need *price field* = `PX_LAST`."
    )
    _uni_df = pd.DataFrame(
        universe.load_rows(),
        columns=["ticker", "name", "price", "asset", "region",
                 "vol_source", "vol_field", "price_field"],
    )
    edited_uni = st.data_editor(
        _uni_df, num_rows="dynamic", use_container_width=True, height=430,
        key="universe_editor",
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker", required=True,
                help="Bloomberg generic ticker incl. the yellow key, e.g. 'COA Comdty', 'ESA Index', 'ECA Curncy'."),
            "name": st.column_config.TextColumn("Name", required=True),
            "price": st.column_config.NumberColumn(
                "Mock price", min_value=0.0,
                help="Demo-mode base price only — ignored once live Bloomberg is on."),
            "asset": st.column_config.SelectboxColumn(
                "Asset", options=universe.ASSET_CLASSES, required=True),
            "region": st.column_config.TextColumn("Region", help="NA / EMEA / APAC, or blank."),
            "vol_source": st.column_config.TextColumn(
                "Vol source (FX/idx)",
                help="LIVE 1-month implied-vol security, e.g. 'EURUSDV1M Curncy'. Blank = use the contract's own option surface."),
            "vol_field": st.column_config.TextColumn(
                "Vol field", help="Usually 'PX_LAST'; blank defaults to PX_LAST when a vol source is set."),
            "price_field": st.column_config.TextColumn(
                "Price field", help="Cash indices: 'PX_LAST' (no settle). Blank = settlement price."),
        },
    )
    _uc1, _uc2 = st.columns([1, 4])
    if _uc1.button("💾 Save universe", type="primary", key="save_universe_btn"):
        def _cell(_r, _k):
            _v = _r.get(_k)
            return "" if (_v is None or pd.isna(_v)) else str(_v).strip()
        cleaned, seen, errs = [], set(), []
        for r in edited_uni.to_dict("records"):
            tk = _cell(r, "ticker")
            if not tk:
                continue                                    # skip the editor's blank trailing row
            if tk in seen:
                errs.append(f"duplicate ticker '{tk}'"); continue
            nm, ac = _cell(r, "name"), _cell(r, "asset")
            if not nm or not ac:
                errs.append(f"'{tk}': name and asset are required"); continue
            try:
                px = float(r.get("price")) if _cell(r, "price") else 0.0
            except (TypeError, ValueError):
                px = 0.0
            seen.add(tk)
            cleaned.append({"ticker": tk, "name": nm, "price": px, "asset": ac,
                            "region": _cell(r, "region"), "vol_source": _cell(r, "vol_source"),
                            "vol_field": _cell(r, "vol_field"), "price_field": _cell(r, "price_field")})
        if errs:
            st.error("Fix these before saving:\n\n- " + "\n- ".join(errs))
        elif not cleaned:
            st.error("The universe can't be empty.")
        else:
            universe.save(cleaned)                          # writes data/universe.json + refreshes live
            try:
                with st.spinner(f"Saved {len(cleaned)} instruments — recomputing all strategies…"):
                    run_daily.run()
                    load_signals.clear()
                st.toast(f"Universe saved ({len(cleaned)}) — signals recomputed.", icon="✅")
            except Exception as e:
                st.toast(f"Universe saved; recompute failed ({e}). Use Re-run signals.", icon="⚠️")
            st.session_state.pop("universe_editor", None)    # reset editor to the saved state
            st.rerun()
    _uc2.caption(
        "Saving rewrites `data/universe.json` and re-runs every strategy on the new list. "
        "In **snapshot** mode a brand-new product has no data until you **Pull Bloomberg Snapshot** "
        "again; regenerate any client reports afterwards to include it."
    )

# Destinations shared across BOTH desks (Cross-asset / System sidebar sections) — reachable from
# either desk's sidebar, so they must fall through this Equities-only gate to the generic dispatch
# chain below rather than being swallowed by its else-branch back into the Equities home page.
_SHARED_DESTS = {"Recipients", "Strategy Builder", "Data health"}

# ----- EQUITIES side: its own home (and future pages), dispatched before the FICC pipeline so the
# futures report-popup + group-tab switcher never run on the Equities side -----------------------
if side == "Equities" and active not in _SHARED_DESTS:
    if active == "eq:Fundamentals":
        render_eq_fundamentals()
    elif active == "eq:Earnings":
        render_eq_earnings()
    elif active == "eq:Correlations":
        render_eq_correlations()
    elif active == "eq:ETFs":
        render_eq_etfs()
    elif active == "eq:Dispersion":
        render_eq_dispersion()
    elif active == "eq:Technical Analysis":
        render_eq_ta_overview()
    elif active.startswith("eq:") and active[3:] in tascore.TA_STRATEGIES:
        render_eq_strategy(active[3:])           # per-strategy Equities page (trigger + chart + table)
    else:
        render_equities_home()
    st.stop()

# ----- report-day alert: full-screen popup at each release time, on top of any FICC page.
# Still skipped on the Equities desk (including its view of a shared page) — unchanged intent. -----
if side != "Equities":
    render_report_popup()

# Collapsed-group tab switcher (Volatility / Positioning & Flow / Fundamentals): shows at the top of
# any member page so you can hop between the group's views from its single sidebar entry.
_render_group_tabs(active)

# ----- page dispatch: render the active view ------------------------------
if active == "Home":
    render_home(); st.stop()
if active == "Confluence":
    render_confluence(); st.stop()
if active == "Technical Analysis":
    render_ta_overview(); st.stop()
if active == "Morning Coffee":
    render_morning_coffee(); st.stop()
if active == "Market Hours":
    render_market_hours(); st.stop()
if active == "Block Sizes":
    render_block_sizes(); st.stop()
if active == "Fut Yield":
    render_fut_yield(); st.stop()
if active == "Fed Path":
    render_fed_path(); st.stop()
if active == "Vol Backtester":
    render_vol_backtester(); st.stop()
if active == "Strategy Builder":
    render_strategy_builder(); st.stop()
if active == "Product Correlations":
    render_sector_correlations(); st.stop()
if active == "Data health":
    render_data_health(); st.stop()
if active == "OPEC Report":
    render_opec(); st.stop()
if active == "Precious Metals":
    render_precious_metals(); st.stop()
if active == "Release Calendar":
    render_releases(); st.stop()
if active == "Recipients":
    render_recipients(); st.stop()
if active == "Universe":
    render_universe(); st.stop()

# ----- a strategy page is active ------------------------------------------
st.header(active)
st.caption(STRATEGY_BLURB.get(active, ""))

# Quick-switch nav between the technical strategies (same buttons as the TA hub) so the user
# can flip between strategy pages without the sidebar — on the technical-strategy pages only.
if active in tascore.TA_STRATEGIES:
    _ta_quicknav(active)
    st.caption("💡 Fixed income runs on **yields**, not futures prices (shown as *(yield)* / *(rate)*). "
               "A **Long / up** signal there means **rising yields = short the bond** — and the mirror.")

# Shrink + wrap the metric-card VALUE font so long values ("Long (buy the dip)", "50.0% @
# 616.5", "Squeeze — upside watch") fit the narrow cards instead of truncating. Applies to
# every strategy page; pages that set their own size (MA crossover) override this afterwards.
st.markdown("""
    <style>
      div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
          font-size: 1.0rem !important; line-height: 1.3 !important;
          white-space: normal !important; overflow-wrap: anywhere;
      }
    </style>
""", unsafe_allow_html=True)

# ---- per-strategy trigger control + maths explainer (all strategies) ------
spec = SPECS.get(active, {})
threshold = spec.get("default")
trigger_text = ""
if threshold is not None:
    threshold = st.number_input(
        spec["label"], min_value=float(spec["min"]), max_value=float(spec["max"]),
        value=trigger_default(active, spec["default"]), step=float(spec["step"]), key=f"thr_{active}",
        help="Changing this re-derives the flags from the stored z-scores / returns — "
             "no data re-pull. It sets the trigger used by both the report and the table below.")
    trigger_text = spec["trigger"](threshold)
    st.info(f"**Trigger:** {trigger_text}")
    _td1, _td2 = st.columns([0.74, 0.26])
    if _td2.button("📌 Set default", key=f"thr_def_{active}", use_container_width=True,
                   help="Save this trigger as the default for this strategy — it loads on every "
                        "launch until you set it again."):
        save_trigger_default(active, float(threshold))
        st.toast(f"Saved {threshold:g} as the default trigger for {active}.", icon="📌")
    _td1.caption(f"📌 Default trigger for **{active}**: **{trigger_default(active, spec['default']):g}** "
                 "— change the value above, then **Set default** to make it the new default.")
if spec.get("math"):
    with st.expander("ℹ️  How this is calculated"):
        st.markdown(spec["math"])

# Mean Reversion: per-pair spread visual — the two legs (rebased) + the spread with
# its 90-day mean and ±trigger·σ band, so you can SEE how far out of the norm it is.
if active == "Mean Reversion":
    import altair as alt
    from src.strategies import mean_reversion as _mr
    from src.universe import PAIRS as _PAIRS

    _by_name = {p["name"]: p for p in _PAIRS}
    _ordered = [n for n in _filter_signals(df[df["strategy"] == "Mean Reversion"])["market"].tolist() if n in _by_name]
    _names = _ordered + [n for n in _by_name if n not in _ordered]
    if _names:
        sel = st.selectbox("Chart a pair (most stretched first)", _names, key="mr_pair")
        try:
            cdata, info = _mr.pair_chart_data(
                _by_name[sel], threshold if threshold is not None else _mr.Z_THRESHOLD)
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            kind_lbl = "ratio (A ÷ B)" if info["kind"] == "ratio" else "differential (A − B)"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("z-score", f"{info['z']:+.2f}", help=f"flagged at ±{info['threshold']:g}σ")
            c2.metric("Percentile (1y)", "—" if info["pctl"] != info["pctl"] else f"{info['pctl']:.0f}%",
                      help="where today's spread sits in its own 1-year range")
            c3.metric("Half-life", "—" if info["half_life"] != info["half_life"] else f"{info['half_life']:.0f}d",
                      help="≈ how long the spread has historically taken to revert")
            c4.metric("Spread now", f"{info['spread']:.2f}",
                      delta=(None if info["mean"] != info["mean"] else f"{info['spread'] - info['mean']:+.2f} vs mean"),
                      delta_color="off")

            legs = (cdata[["date", "a_idx", "b_idx"]]
                    .rename(columns={"a_idx": info["a_name"], "b_idx": info["b_name"]})
                    .melt("date", var_name="Leg", value_name="idx"))
            legs_chart = alt.Chart(legs).mark_line().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("idx:Q", title="Rebased to 100", scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.Color("Leg:N", legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
            ).properties(height=300, title="The two legs, rebased to 100")

            _cc = brand.chart_colors()
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            band = base.mark_area(opacity=0.15, color=_cc["series"]).encode(
                y=alt.Y("lower:Q", title=kind_lbl, scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)), y2="upper:Q")
            mean_ln = base.mark_line(strokeDash=[4, 3], color=_cc["muted"]).encode(y="mean:Q")
            spread_ln = base.mark_line(color=_cc["series"]).encode(y="spread:Q")
            today = alt.Chart(cdata.dropna(subset=["spread"]).iloc[[-1]]).mark_point(
                color=_cc["short"], size=90, filled=True).encode(x="date:T", y="spread:Q")
            spread_chart = (band + mean_ln + spread_ln + today).properties(
                height=380, title=f"Spread vs 90-day mean ± {info['threshold']:g}σ band (today ●)")

            brand.show_chart(legs_chart)
            brand.show_chart(spread_chart)
            st.caption(f"**{sel}** is built as a **{kind_lbl}**. Shaded band = 90-day mean ± your "
                       f"trigger ({info['threshold']:g}σ); the spread poking outside the band is the "
                       "signal. Charts read the cached snapshot — no Bloomberg pull.")
    st.divider()

# Trend: price with its MA20 / MA100 and the 3-month-return window drawn on, so you can
# SEE the trend behind each flagged product (mirrors the other strategy pages). One chart
# block; the generic opportunities table renders below (no st.stop()).
if active == "Trend":
    import altair as alt
    from src.strategies import trend as _trend

    _v = _filter_signals(df[df["strategy"] == "Trend"]).copy()
    if threshold is not None and spec.get("hi"):       # flag at the page trigger; flagged first
        _v = reflag_rows(_v, float(threshold), spec["hi"], spec["lo"])
        _v = pd.concat([_v[_v["direction"] != 0], _v[_v["direction"] == 0]])
    if _v.empty:
        st.info("No Trend rows yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        _tick = dict(zip(_v["market"], _v["instruments"]))
        _n_fl = int((_v["direction"] != 0).sum())
        sel = st.selectbox(
            f"Chart a market — {_n_fl} flagged at the current trigger (strongest trend first)",
            _v["market"].tolist(), key="trend_market")
        try:
            cdata, info = _trend.trend_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            _dir = _cc["long"] if info["direction"] > 0 else _cc["short"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"] + (" · mixed" if info["mixed"] else ""),
                      help="Direction = sign of the 3-month return. 'mixed' = the "
                           f"MA{info['fast_w']}/MA{info['slow_w']} crossover disagrees with it.")
            c2.metric(f"{info['mom_window'] // 21}-month return", f"{info['mom'] * 100:+.1f}%",
                      help="Today's price ÷ price 63 trading days ago − 1 — this is the trigger metric.")
            c3.metric(f"MA{info['fast_w']} vs MA{info['slow_w']}", f"{info['ma_gap']:+.1f}%",
                      help="Gap between the fast and slow moving averages — the trend confirmation/context.")
            c4.metric("Last", f"{info['last']:g}")

            _cols = {"price": "Price", "fast": f"MA{info['fast_w']}", "slow": f"MA{info['slow_w']}"}
            _dom = list(_cols.values())
            series = (cdata.melt("date", value_vars=list(_cols), var_name="Series", value_name="val")
                      .replace({"Series": _cols}))
            lines = alt.Chart(series).mark_line().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("val:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=_dom, range=[_cc["ink"], _cc["series"], _cc["accent"]]),
                    legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                size=alt.Size("Series:N", scale=alt.Scale(domain=_dom, range=[1.9, 1.4, 1.4]), legend=None),
            )
            # the 3-month return window: a dashed leg from the price 63 sessions ago to today,
            # coloured green (up / Long) or red (down / Short) — the signal made visible.
            win = pd.DataFrame({"date": [info["mom_date"], cdata["date"].iloc[-1]],
                                "price": [info["mom_price"], info["last"]]})
            win_ln = alt.Chart(win).mark_line(strokeDash=[5, 3], strokeWidth=3, color=_dir).encode(
                x="date:T", y="price:Q")
            win_pts = alt.Chart(win).mark_point(size=95, filled=True, color=_dir).encode(
                x="date:T", y="price:Q",
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title="Price", format=",.4g")])
            chart = (lines + win_ln + win_pts).resolve_scale(color="independent").properties(
                height=420,
                title=f"{sel} — price with MA{info['fast_w']} / MA{info['slow_w']}; "
                      f"dashed = the 3-month return window ({info['mom'] * 100:+.1f}%)")
            brand.show_chart(chart)
            st.caption(
                f"**{sel}**: black **Price**, blue **MA{info['fast_w']}**, gold **MA{info['slow_w']}**, and the "
                f"**dashed leg** spanning the 3-month return window (green = up / Long, red = down / Short). "
                f"The signal is the **sign of that 3-month return** ({info['mom'] * 100:+.1f}%), with the "
                f"MA{info['fast_w']}/MA{info['slow_w']} gap ({info['ma_gap']:+.1f}%) as confirmation"
                + (" — currently **mixed** (they disagree)" if info["mixed"] else "")
                + ". Charts read the cached snapshot — no Bloomberg pull.")
    st.divider()

# MA Crossover / MA Swing: the four-MA ribbon (Price / fast-EMA / fast-SMA / context-SMA
# / slow-SMA) with the fast SMA coloured green/red by slope and the golden/death cross
# markers, plus the EMA + return confirmations. One block, config-driven per variant.
# (No st.stop() — generic table renders below.)
if active in ("MA Crossover", "MA Swing"):
    import altair as alt
    from src.strategies import ma_crossover, ma_crossover_swing

    _mac = {"MA Crossover": ma_crossover, "MA Swing": ma_crossover_swing}[active]
    cfg = _mac.CFG
    _mom_long = {"3m": "3-month", "1m": "1-month"}.get(cfg.mom_label, cfg.mom_label)

    # Shrink the metric VALUE font on these two pages so "Golden Cross" / "Death Cross"
    # fit the narrow 5-column cards instead of truncating to "Death Cr…".
    st.markdown("""
        <style>
          div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
              font-size: 1.15rem !important; line-height: 1.35 !important;
              white-space: normal !important; overflow-wrap: anywhere;
          }
        </style>
    """, unsafe_allow_html=True)

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info(f"No {active} rows yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (confirmed signals first)", _v["market"].tolist(),
                           key=f"mac_market_{active}")
        try:
            cdata, info = _mac.crossover_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(f"Cross ({cfg.fast}/{cfg.slow})", info["state"].title(),
                      help=f"{cfg.fast}-day above {cfg.slow}-day = golden cross (bullish); below = death cross (bearish).")
            c2.metric(f"{cfg.ema}-EMA vs {cfg.fast}",
                      ("Above" if info["ema_above"] else "Below") + (" ✓" if info["ema_ok"] else " ✗"),
                      help=f"Golden cross needs the {cfg.ema}-EMA ABOVE the {cfg.fast} to confirm; death cross needs it below.")
            c3.metric(f"{_mom_long} return", f"{info['mom'] * 100:+.1f}%",
                      help="Must agree in sign with the cross (positive for Long, negative for Short).")
            c4.metric("Confirmed?", "Yes ✓" if info["confirmed"] else "No ✗",
                      help=f"A trade is taken only when BOTH the {cfg.ema}-EMA and the {_mom_long} return confirm.")
            c5.metric("Signal", info["signal"])

            # Price + fast-EMA + context-SMA + slow-SMA as fixed-colour lines.
            _cc = brand.chart_colors()
            _cols = {"price": "Price", "ema": f"EMA{cfg.ema}",
                     "ctx": f"MA{cfg.ctx}", "slow": f"MA{cfg.slow}"}
            _dom = list(_cols.values())
            series = (cdata.melt("date", value_vars=list(_cols), var_name="Series", value_name="val")
                      .replace({"Series": _cols}))
            lines = alt.Chart(series).mark_line().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("val:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=_dom, range=[_cc["muted"], "#7E57C2", "#E08A00", _cc["ink"]]),
                    legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                size=alt.Size("Series:N", scale=alt.Scale(domain=_dom, range=[1.0, 1.6, 1.6, 1.8]), legend=None),
            )

            # fast SMA coloured by slope — overlap the segments by one point so the line stays continuous.
            m = cdata[["date", "fast", "fast_rising"]].dropna().reset_index(drop=True)
            m["seg"] = (m["fast_rising"] != m["fast_rising"].shift()).cumsum()
            bridge = m[(m["seg"] != m["seg"].shift()) & (m.index > 0)].copy()
            bridge["seg"] -= 1
            bridge["fast_rising"] = ~bridge["fast_rising"]
            m = pd.concat([m, bridge]).sort_values(["seg", "date"])
            _rise, _fall = f"MA{cfg.fast} rising", f"MA{cfg.fast} falling"
            m["Slope"] = np.where(m["fast_rising"], _rise, _fall)
            fast_ln = alt.Chart(m).mark_line(strokeWidth=3).encode(
                x="date:T", y="fast:Q", detail="seg:N",
                color=alt.Color("Slope:N", scale=alt.Scale(domain=[_rise, _fall], range=[_cc["long"], _cc["short"]]),
                                legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
            )

            # golden/death cross markers = where sign(fast − slow) flips
            d = cdata.dropna(subset=["fast", "slow"]).copy()
            d["g"] = np.sign(d["fast"] - d["slow"])
            d = d[d["g"].ne(d["g"].shift()) & d["g"].shift().notna()]
            crosses = d.assign(type=np.where(d["g"] > 0, "Golden cross", "Death cross"))
            marks = alt.Chart(crosses).mark_point(size=140, filled=True, opacity=0.95).encode(
                x="date:T", y="slow:Q",
                shape=alt.Shape("type:N", scale=alt.Scale(domain=["Golden cross", "Death cross"],
                                                          range=["triangle-up", "triangle-down"]), legend=None),
                color=alt.Color("type:N", scale=alt.Scale(domain=["Golden cross", "Death cross"],
                                                          range=[_cc["long"], _cc["short"]]), legend=None),
                tooltip=[alt.Tooltip("date:T", title="Crossed"), alt.Tooltip("type:N", title="Type")],
            )
            chart = (lines + fast_ln + marks).resolve_scale(color="independent").properties(
                height=420,
                title=f"{sel} — 4-MA ribbon (EMA{cfg.ema}/MA{cfg.fast}/MA{cfg.ctx}/MA{cfg.slow}); "
                      f"{cfg.fast}-day green = rising, red = falling")
            brand.show_chart(chart)
            st.caption(
                f"Four-MA ribbon: grey **Price**, purple **EMA{cfg.ema}**, orange **MA{cfg.ctx}**, black "
                f"**MA{cfg.slow}**, and the **MA{cfg.fast}** coloured green (rising) / red (falling). "
                f"Signal = a {cfg.fast}/{cfg.slow} **golden cross** (▲) Long or **death cross** (▼) Short, "
                f"taken only when the **EMA{cfg.ema}** is on the trend side of the {cfg.fast} and the "
                f"**{_mom_long} return** agrees. The MA{cfg.ctx} is context only. Charts read the cached "
                "snapshot — no Bloomberg pull.")
    st.divider()

# Flag Breakout: the price chart with the flagpole, the consolidation channel and the
# dashed breakout line drawn on, so you can SEE which products are coiling at the edge.
# One chart block; the generic opportunities table renders below (no st.stop()).
if active == "Flag Breakout":
    import altair as alt
    from src.strategies import flag_breakout as _fb

    # Shrink the metric value font so the six narrow cards (incl. "Bull breakout setup") fit.
    st.markdown("""
        <style>
          div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
              font-size: 1.05rem !important; line-height: 1.3 !important;
              white-space: normal !important; overflow-wrap: anywhere;
          }
        </style>
    """, unsafe_allow_html=True)

    _cc = brand.chart_colors()
    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    _thr = float(threshold) if threshold is not None else _fb.DEFAULT_TRIGGER
    _GALLERY_CAP = 24                      # most-ready setups to draw inline (keeps the page bounded)

    def _flag_chart(cdata, info, title_market, height=440):
        """Price + flagpole + consolidation channel + dashed breakout line + measured-move
        target (green) / stop (red) + today ● — one layered Altair chart, shared by the
        gallery and the inspector."""
        _edge = _cc["long"] if info["sign"] > 0 else _cc["short"]
        _yax = "Yield (%)" if str(title_market).endswith(("(yield)", "(rate)")) else "Price"
        base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
        band = base.mark_area(opacity=0.15, color=_edge).encode(
            y=alt.Y("lower:Q", title=_yax, scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)), y2="upper:Q")
        brk_ln = base.mark_line(color=_edge, strokeDash=[6, 3], strokeWidth=2.6).encode(y="breakout:Q")
        price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.4).encode(
            y="price:Q",
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title="Price", format=",.2f")])
        pole_df = pd.DataFrame({"date": [info["pole_base"][0], info["pole_tip"][0]],
                                "price": [info["pole_base"][1], info["pole_tip"][1]]})
        pole_ln = alt.Chart(pole_df).mark_line(color=_cc["muted"], strokeWidth=3.1).encode(x="date:T", y="price:Q")
        pole_pts = alt.Chart(pole_df).mark_point(color=_cc["muted"], size=70, filled=True).encode(x="date:T", y="price:Q")
        today = alt.Chart(cdata.iloc[[-1]]).mark_point(color=_edge, size=140, filled=True).encode(x="date:T", y="price:Q")
        tgt_ln = alt.Chart(pd.DataFrame({"y": [info["target"]]})).mark_rule(
            color=_cc["long"], strokeDash=[2, 2], strokeWidth=2).encode(y="y:Q")
        stop_ln = alt.Chart(pd.DataFrame({"y": [info["stop"]]})).mark_rule(
            color=_cc["short"], strokeDash=[2, 2], strokeWidth=2).encode(y="y:Q")
        return (band + brk_ln + tgt_ln + stop_ln + pole_ln + price_ln + pole_pts + today).properties(
            height=height,
            title=f"{title_market} — {info['type'].lower()}: flagpole, consolidation channel, dashed "
                  f"breakout line; measured-move target (green) & stop (red); today ●")

    def _flag_line(info) -> str:
        _rr = "" if not np.isfinite(info["rr"]) else f" · R:R **{info['rr']:.1f}:1**"
        _vc = (" · vol ✓" if info["vol_confirms"] is True
               else " · vol ✗" if info["vol_confirms"] is False else "")
        _edge_txt = "broke out" if info["broke"] else f"{abs(info['dist_pct']):.1f}% to breakout"
        return (f"{info['type']} · readiness **{info['readiness']:.0f}/100** · pole {info['pole_ret'] * 100:+.0f}% "
                f"in {info['plen']}d · {_edge_txt} · target **{info['target']:g}** / stop **{info['stop']:g}**"
                f"{_rr}{_vc}")

    if _v.empty:
        st.info("No flag patterns in the book right now — click **🔁 Re-run signals** on the 🏠 Home "
                "page, or check back after the next snapshot. A flag needs a strong pole followed by a "
                "tight consolidation, so on quiet days there may be none.")
    else:
        # ---- Gallery: a chart for EVERY product breaking the trigger ----
        _flagged = _v[_v["metric"].abs() >= _thr]
        st.subheader(f"Breakout setups — readiness ≥ {_thr:.0f}")
        if _flagged.empty:
            st.caption("No products are at the trigger right now — lower the trigger above, or inspect any "
                       "market (incl. the watchlist) below.")
        else:
            _cap = f"**{len(_flagged)}** product(s) within reach of a breakout" + (
                f" — drawing the top {_GALLERY_CAP} by readiness." if len(_flagged) > _GALLERY_CAP else ".")
            st.caption(_cap + " Each chart shows the flagpole, the consolidation channel, the dashed breakout "
                       "line, and the measured-move target (green) & stop (red).")
            for _m, _t in zip(_flagged["market"].head(_GALLERY_CAP), _flagged["instruments"].head(_GALLERY_CAP)):
                try:
                    _cd, _ci = _fb.flag_chart_data(_t)
                except Exception:
                    _cd, _ci = None, None
                if _cd is None or _cd.empty:
                    continue
                st.markdown(f"**{_m}** — " + _flag_line(_ci))
                brand.show_chart(_flag_chart(_cd, _ci, _m, height=320))

        # ---- Inspector: drill into ANY market (incl. the sub-trigger watchlist) + volume ----
        st.divider()
        st.markdown("##### Inspect any market")
        sel = st.selectbox("Market (closest to breakout first)", _v["market"].tolist(),
                           key="fb_market", label_visibility="collapsed")
        try:
            cdata, info = _fb.flag_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _bull = info["sign"] > 0
            _edge = _cc["long"] if _bull else _cc["short"]
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Pattern", info["type"])
            c2.metric("Flagpole", f"{info['pole_ret'] * 100:+.0f}%",
                      help=f"Move over the {info['plen']}-day flagpole; flag {info['flen']}d, "
                           f"{info['retrace'] * 100:.0f}% retraced.")
            c3.metric("Readiness", f"{info['readiness']:.0f}/100",
                      help="100 = sitting on the breakout trendline. Setups flag at ≥ the trigger above.")
            c4.metric("Target", f"{info['target']:g}",
                      help="Measured move: the flagpole height projected from the breakout level.")
            c5.metric("R:R", "—" if not np.isfinite(info["rr"]) else f"{info['rr']:.1f}:1",
                      help="Reward:risk on a break-level entry — measured-move target vs a stop beyond the far edge of the flag.")
            c6.metric("Signal", info["signal"])
            brand.show_chart(_flag_chart(cdata, info, sel, height=440))

            # Volume subpanel — the textbook flag dries up on volume through the consolidation
            # and picks up on the break. Only drawn when a volume series is available.
            if cdata["volume"].notna().any():
                vdf = cdata.dropna(subset=["volume"]).copy()
                vdf["Phase"] = np.where(vdf["date"] >= info["anchor_date"], "Flag", "Pole / run-up")
                vol_bars = alt.Chart(vdf).mark_bar().encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                    y=alt.Y("volume:Q", title="Volume", axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                    color=alt.Color("Phase:N", scale=alt.Scale(domain=["Pole / run-up", "Flag"],
                                    range=[_cc["muted"], _edge]),
                                    legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                    tooltip=[alt.Tooltip("date:T", title="Day"),
                             alt.Tooltip("volume:Q", title="Volume", format=",.0f")])
                brand.show_chart(vol_bars.properties(
                    height=150, title="Volume — should dry up through the flag, pick up on the break"))

            _dry = info["vol_dryup"]
            if info["vol_confirms"] is True:
                _voltxt = f" Volume **confirms** — the flag is trading at {_dry:.0%} of the pole's volume (dry-up)."
            elif info["vol_confirms"] is False:
                _voltxt = f" Volume does **not** confirm yet — the flag isn't drying up ({_dry:.0%} of the pole)."
            else:
                _voltxt = ""
            _edge_txt = ("already broken out" if info["broke"]
                         else f"{abs(info['dist_pct']):.1f}% from the breakout line")
            _rr_txt = "" if not np.isfinite(info["rr"]) else f" R:R ≈ **{info['rr']:.1f}:1**."
            _mm_txt = (f" Measured-move **target {info['target']:g}** (green dashes — the flagpole height "
                       f"projected from the breakout), **stop {info['stop']:g}** (red — beyond the far edge "
                       f"of the flag).{_rr_txt}")
            st.caption(
                f"**{sel}**: a {info['plen']}-day flagpole of **{info['pole_ret'] * 100:+.0f}%**, then a "
                f"**{info['flen']}-day** consolidation retracing {info['retrace'] * 100:.0f}% of it. The shaded "
                f"band is the consolidation channel (trendline ±{_fb.CHANNEL_W:g}σ); the dashed line is the "
                f"{'upper' if _bull else 'lower'} edge the {info['type'].lower()} breaks through. "
                f"Price is **{_edge_txt}** (readiness {info['readiness']:.0f}/100)." + _mm_txt + _voltxt +
                " Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Support & Resistance: price with the tested horizontal levels (support green /
# resistance red, thicker = more touches) drawn on. No st.stop() — table renders below.
if active == "Support & Resistance":
    import altair as alt
    from src.strategies import support_resistance as _sr

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No support/resistance reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (closest to a level first)", _v["market"].tolist(), key="sr_market")
        try:
            cdata, info = _sr.sr_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Nearest level", "—" if not np.isfinite(info["nearest"]) else f"{info['nearest']:g}",
                      help="The strong level price is sitting on (if any).")
            c3.metric("Distance", "—" if not np.isfinite(info["dist_pct"]) else f"{info['dist_pct']:+.1f}%",
                      help="Signed % from price to that level (>0 = level below price).")
            c4.metric("Levels mapped", str(len(info["levels"])))
            _cc = brand.chart_colors()
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            layers = [price_ln]
            if info["levels"]:
                lv = pd.DataFrame(info["levels"])
                rules = alt.Chart(lv).mark_rule(opacity=0.85).encode(
                    y="price:Q",
                    color=alt.Color("kind:N", scale=alt.Scale(domain=["support", "resistance"],
                                    range=[_cc["long"], _cc["short"]]),
                                    legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                    size=alt.Size("touches:Q", scale=alt.Scale(range=[1, 3.4]), legend=None),
                    tooltip=[alt.Tooltip("kind:N", title="Kind"), alt.Tooltip("price:Q", title="Level", format=",.2f"),
                             alt.Tooltip("touches:Q", title="Touches")])
                layers.append(rules)
            today = alt.Chart(cdata.iloc[[-1]]).mark_point(
                color=(_cc["long"] if info["direction"] > 0 else _cc["short"] if info["direction"] < 0 else _cc["muted"]),
                size=130, filled=True).encode(x="date:T", y="price:Q")
            chart = alt.layer(*layers, today).properties(
                height=420, title=f"{sel} — price with tested support (green) / resistance (red) levels; today ●")
            brand.show_chart(chart)
            st.caption(f"Horizontal lines are tested levels from swing pivots (thicker = more touches). "
                       f"**{info['signal']}**. Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Fibonacci Retracement: price with the dominant swing's retracement grid (key levels gold),
# the swing leg, and the target/stop. No st.stop() — table renders below.
if active == "Fibonacci Retracement":
    import altair as alt
    from src.strategies import fibonacci as _fbn

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No Fibonacci reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (closest to a key level first)", _v["market"].tolist(), key="fib_market")
        try:
            cdata, info = _fbn.fib_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _up = info["leg"] == "up"
            _cc = brand.chart_colors()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Leg", "Up ▲" if _up else "Down ▼", help="Dominant swing direction (sets support vs resistance).")
            c3.metric("Nearest level", f"{info['nearest_ratio'] * 100:.1f}% @ {info['nearest']:g}")
            c4.metric("R:R", "—" if not np.isfinite(info["rr"]) else f"{info['rr']:.1f}:1",
                      help="Target = the prior swing extreme; stop beyond the 78.6% level.")
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            lv = pd.DataFrame(info["levels"])
            lv["label"] = (lv["ratio"] * 100).map(lambda r: f"{r:.1f}%")
            rules = alt.Chart(lv).mark_rule(opacity=0.8, strokeWidth=1.8).encode(
                y="price:Q", color=alt.condition("datum.key", alt.value(_cc["accent"]), alt.value(_cc["muted"])))
            txt = alt.Chart(lv).mark_text(align="left", dx=3, fontSize=9, color="#777").encode(
                x=alt.value(2), y="price:Q", text="label:N")
            leg_df = pd.DataFrame({"date": [info["lo_date"], info["hi_date"]], "price": [info["lo"], info["hi"]]})
            leg_ln = alt.Chart(leg_df).mark_line(color=_cc["muted"], strokeWidth=2.6, point=True).encode(x="date:T", y="price:Q")
            tgt = alt.Chart(pd.DataFrame({"y": [info["target"]]})).mark_rule(
                color=_cc["long"], strokeDash=[2, 2], strokeWidth=1.9).encode(y="y:Q")
            stp = alt.Chart(pd.DataFrame({"y": [info["stop"]]})).mark_rule(
                color=_cc["short"], strokeDash=[2, 2], strokeWidth=1.9).encode(y="y:Q")
            today = alt.Chart(cdata.iloc[[-1]]).mark_point(
                color=(_cc["long"] if info["direction"] > 0 else _cc["short"] if info["direction"] < 0 else _cc["muted"]),
                size=130, filled=True).encode(x="date:T", y="price:Q")
            chart = (leg_ln + rules + txt + tgt + stp + price_ln + today).properties(
                height=440, title=f"{sel} — Fib retracement of the {info['leg']}-leg; gold = key levels "
                                  f"(38.2/50/61.8%) · green target · red stop; today ●")
            brand.show_chart(chart)
            st.caption(f"Retracement of the dominant **{info['leg']}-leg** ({info['lo']:g}→{info['hi']:g}). "
                       f"**{info['signal']}** at the **{info['nearest_ratio'] * 100:.1f}%** level. Close-based; "
                       "reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Breakout & Retest: price with the broken level (flipped role) + the breakout day +
# today, so you can see the pullback into the level. No st.stop().
if active == "Breakout & Retest":
    import altair as alt
    from src.strategies import breakout_retest as _br

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No active breakout-retests right now — it's a conditional pattern (a level broken on "
                "volume, then retested), so the list is often short or empty. Check back after a snapshot.")
    else:
        sel = st.selectbox("Chart a retest (tightest first)", _v["market"].tolist(), key="br_market")
        try:
            cdata, info = _br.retest_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            _up = info["kind"] == "up"
            _edge = _cc["long"] if _up else _cc["short"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Broken level", f"{info['level']:g}", help="The level that broke and is now being retested (role-flipped).")
            c3.metric("Distance", f"{info['dist_pct']:+.1f}%", help="Signed % from price to the level.")
            c4.metric("Volume", "✓ confirmed" if info["vol_confirm"] is True
                      else "✗ unconfirmed" if info["vol_confirm"] is False else "—",
                      help="Did the breakout bar trade ≥1.3× its trailing average?")
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            level_ln = alt.Chart(pd.DataFrame({"y": [info["level"]]})).mark_rule(
                color=_edge, strokeDash=[6, 3], strokeWidth=2.4).encode(y="y:Q")
            broke_ln = alt.Chart(pd.DataFrame({"x": [info["broke_date"]]})).mark_rule(
                color=_cc["muted"], strokeWidth=1.8).encode(x="x:T")
            today = alt.Chart(cdata.iloc[[-1]]).mark_point(color=_edge, size=140, filled=True).encode(x="date:T", y="price:Q")
            chart = (price_ln + level_ln + broke_ln + today).properties(
                height=420, title=f"{sel} — {info['signal'].lower()}: dashed = broken level (now "
                                  f"{'support' if _up else 'resistance'}), grey = breakout day; today ●")
            brand.show_chart(chart)
            st.caption(f"Level **{info['level']:g}** broke {'up' if _up else 'down'} and price has pulled back to "
                       f"retest it. Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Momentum (RSI/MACD): price + an RSI panel (70/30 guides) + a MACD panel
# (line/signal/histogram). No st.stop().
if active == "Momentum (RSI/MACD)":
    import altair as alt
    from src.strategies import momentum as _mom

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No momentum reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (strongest setup first)", _v["market"].tolist(), key="mom_market")
        try:
            cdata, info = _mom.momentum_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("RSI (14)", f"{info['rsi']:.0f}", help="Above 70 overbought, below 30 oversold.")
            c3.metric("MACD", info["macd_state"].title())
            c4.metric("Divergence", info["divergence"].title())
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            brand.show_chart(price_ln.properties(height=240, title=sel))

            rsi_ln = base.mark_line(color="#7E57C2", strokeWidth=2.2).encode(
                y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=[0, 30, 50, 70, 100], labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("rsi:Q", title="RSI", format=".0f")])
            ob = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(color=_cc["short"], strokeDash=[4, 3]).encode(y="y:Q")
            os_ = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule(color=_cc["long"], strokeDash=[4, 3]).encode(y="y:Q")
            mid = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=_cc["muted"], strokeDash=[2, 2]).encode(y="y:Q")
            brand.show_chart((rsi_ln + ob + os_ + mid).properties(height=170, title="RSI (14) — 70 / 30 guides"))

            hist_bars = base.mark_bar().encode(
                y=alt.Y("macd_hist:Q", title="MACD", axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.condition("datum.macd_hist >= 0", alt.value(_cc["long"]), alt.value(_cc["short"])))
            macd_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.1).encode(y="macd:Q")
            sig_ln = base.mark_line(color=_cc["accent"], strokeWidth=2.1).encode(y="macd_signal:Q")
            brand.show_chart((hist_bars + macd_ln + sig_ln).properties(
                height=170, title="MACD 12/26/9 — line (black) · signal (gold) · histogram"))
            st.caption(f"**{info['signal']}** — RSI {info['rsi']:.0f}, MACD {info['macd_state']}, "
                       f"divergence: {info['divergence']}. Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Bollinger Squeeze: price + the ±2σ envelope, plus a bandwidth panel showing the
# squeeze. No st.stop().
if active == "Bollinger Squeeze":
    import altair as alt
    from src.strategies import bollinger as _bb

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No Bollinger reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (tightest squeeze first)", _v["market"].tolist(), key="bb_market")
        try:
            cdata, info = _bb.bollinger_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Bandwidth %ile", f"{info['bandwidth_pctl']:.0f}", help="Low = tight bands (squeeze).")
            c3.metric("Squeeze?", "Yes" if info["squeeze"] else "No")
            c4.metric("Breakout", info["breakout"].title())
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            band = base.mark_area(opacity=0.12, color=_cc["series"]).encode(
                y=alt.Y("lower:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)), y2="upper:Q")
            up_ln = base.mark_line(color=_cc["muted"], strokeWidth=1.6).encode(y="upper:Q")
            lo_ln = base.mark_line(color=_cc["muted"], strokeWidth=1.6).encode(y="lower:Q")
            mid_ln = base.mark_line(color=_cc["accent"], strokeDash=[4, 3], strokeWidth=1.8).encode(y="mid:Q")
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y="price:Q", tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title="Price", format=",.2f")])
            brand.show_chart((band + up_ln + lo_ln + mid_ln + price_ln).properties(
                height=380, title=f"{sel} — Bollinger Bands (20, 2σ): mid (gold dashes), ±2σ envelope"))
            bw_ln = base.mark_line(color="#7E57C2", strokeWidth=2.1).encode(
                y=alt.Y("bandwidth:Q", title="Bandwidth", axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("bandwidth:Q", title="Bandwidth", format=".4f")])
            brand.show_chart(bw_ln.properties(height=150, title="Bandwidth — (upper−lower)/mid; lows = squeezes"))
            st.caption(f"**{info['signal']}** — bandwidth at its **{info['bandwidth_pctl']:.0f}th** percentile of the "
                       f"year. Tight bands (low bandwidth) precede sharp moves. Close-based; reads the cached "
                       "snapshot — no Bloomberg pull.")
    st.divider()

# Elliott Wave: price with the ZigZag skeleton, the counted pivots labelled 0..5, and the
# juncture's projection (green objective / red invalidation). No st.stop().
if active == "Elliott Wave":
    import altair as alt
    from src.strategies import elliott_wave as _ew

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No Elliott reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (best wave fit first)", _v["market"].tolist(), key="ew_market")
        try:
            cdata, info = _ew.elliott_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            _ph = {"W3": "Wave 3 of 5", "W5": "Wave 5 of 5", "done": "Five waves done"}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Phase", _ph.get(info["phase"], info["phase"]),
                      help="Where the count sits within the five-wave impulse.")
            c3.metric("Wave fit", f"{info['fit']:.0f}/100",
                      help="How textbook the count's Fibonacci proportions are.")
            c4.metric("R:R", "—" if not np.isfinite(info["rr"]) else f"{info['rr']:.1f}:1",
                      help="Classic projection for the juncture vs the last counted pivot.")
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            zz = pd.DataFrame(info["zigzag"])
            zz_ln = alt.Chart(zz).mark_line(color=_cc["muted"], strokeWidth=1.4, point=True,
                                            opacity=0.65).encode(x="date:T", y="price:Q")
            cp = pd.DataFrame(info["pivots"])
            cnt_ln = alt.Chart(cp).mark_line(color="#7E57C2", strokeWidth=2.4, point=True).encode(
                x="date:T", y="price:Q")
            cnt_tx = (alt.Chart(cp[cp["kind"] == "H"]).mark_text(
                          fontSize=13, fontWeight="bold", color="#7E57C2", dy=-12).encode(
                          x="date:T", y="price:Q", text="label:N")
                      + alt.Chart(cp[cp["kind"] == "L"]).mark_text(
                          fontSize=13, fontWeight="bold", color="#7E57C2", dy=14).encode(
                          x="date:T", y="price:Q", text="label:N"))
            tgt = alt.Chart(pd.DataFrame({"y": [info["target"]]})).mark_rule(
                color=_cc["long"], strokeDash=[2, 2], strokeWidth=1.9).encode(y="y:Q")
            stp = alt.Chart(pd.DataFrame({"y": [info["stop"]]})).mark_rule(
                color=_cc["short"], strokeDash=[2, 2], strokeWidth=1.9).encode(y="y:Q")
            today = alt.Chart(cdata.iloc[[-1]]).mark_point(
                color=(_cc["long"] if info["direction"] > 0 else _cc["short"] if info["direction"] < 0 else _cc["muted"]),
                size=130, filled=True).encode(x="date:T", y="price:Q")
            brand.show_chart((zz_ln + cnt_ln + price_ln + cnt_tx + tgt + stp + today).properties(
                height=440, title=f"{sel} — Elliott count (purple, waves 0–5) on the ZigZag skeleton "
                                  f"(grey); green objective · red invalidation; today ●"))
            st.caption(f"**{info['signal']}** — wave fit **{info['fit']:.0f}/100**; ZigZag reversal "
                       f"threshold **{info['thr_pct']:.1f}%** (scaled to this market's own volatility). "
                       f"Objective **{info['target']:g}**, invalidation **{info['stop']:g}**. "
                       "Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Ichimoku: price + the cloud (Kumo) with its forward projection, Tenkan/Kijun and the lagging
# span — the whole system on one axis. No st.stop().
if active == "Ichimoku Cloud":
    import altair as alt
    from src.strategies import ichimoku as _ic

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No Ichimoku events right now — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (strongest read first)", _v["market"].tolist(), key="ichi_market")
        try:
            cdata, info = _ic.ichimoku_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("Ichimoku score", f"{info['score']:.0f}/100")
            c3.metric("Tenkan / Kijun", f"{info['tenkan_now']:g} / {info['kijun_now']:g}")
            c4.metric("Cloud", f"{info['cloud_bot']:g} – {info['cloud_top']:g}",
                      help="Price above = constructive, below = cautious, inside = no signal.")
            cloud_df = pd.DataFrame(info["cloud"])
            _green = _cc.get("long", "#2E7D32")
            _red = _cc.get("short", "#C62828")
            cloud = alt.Chart(cloud_df).mark_area(opacity=0.34).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("a:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                y2="b:Q",
                color=alt.condition("datum.a >= datum.b", alt.value(_green), alt.value(_red)))
            base = alt.Chart(cdata).encode(x="date:T")
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.4).encode(
                y="price:Q", tooltip=[alt.Tooltip("date:T", title="Date"),
                                      alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            tk = alt.Chart(pd.DataFrame(info["tenkan"])).mark_line(
                color="#1F77B4", strokeWidth=1.5).encode(x="date:T", y="val:Q")
            kj = alt.Chart(pd.DataFrame(info["kijun"])).mark_line(
                color="#E08A00", strokeWidth=1.5).encode(x="date:T", y="val:Q")
            ch = alt.Chart(pd.DataFrame(info["chikou"])).mark_line(
                color="#7E57C2", strokeWidth=1.3, strokeDash=[4, 3]).encode(x="date:T", y="val:Q")
            today = alt.Chart(cdata.iloc[[-1]]).mark_point(
                color=(_cc["long"] if info["direction"] > 0 else _cc["short"] if info["direction"] < 0 else _cc["muted"]),
                size=130, filled=True).encode(x="date:T", y="price:Q")
            brand.show_chart((cloud + tk + kj + ch + price_ln + today).properties(
                height=460, title=f"{sel} — Ichimoku: cloud (green/red, projected 26 ahead), "
                                  f"Tenkan (blue) / Kijun (orange), lagging span (purple dash); today ●"))
            st.caption(f"**{info['signal']}** — {info['note']}; score **{info['score']:.0f}/100**. "
                       "Cloud is projected 26 sessions ahead (Ichimoku convention). Close-based "
                       "(n-period highs/lows from closes); reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# On-Balance Volume: price + the OBV line beneath — divergences, (un)confirmed breakouts and
# quiet accumulation/distribution are all visible as price-vs-OBV disagreement. No st.stop().
if active == "On-Balance Volume":
    import altair as alt
    from src.strategies import obv as _obv

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No OBV reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (strongest read first)", _v["market"].tolist(), key="obv_market")
        try:
            cdata, info = _obv.obv_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            c1, c2, c3 = st.columns(3)
            c1.metric("Signal", info["signal"])
            c2.metric("OBV score", f"{info['score']:.0f}/100",
                      help="Strength of the dominant read (divergence / breakout / accumulation).")
            c3.metric("Read", info["note"])
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            brand.show_chart(price_ln.properties(height=300, title=f"{sel} — price"))
            obv_ln = base.mark_line(color="#00897B", strokeWidth=2.1).encode(
                y=alt.Y("obv:Q", title="OBV", scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=11, titleFontSize=13, format="~s")),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("obv:Q", title="OBV", format=",.0f")])
            brand.show_chart(obv_ln.properties(
                height=190, title="On-Balance Volume — Σ volume signed by the day's direction"))
            st.caption(f"**{info['signal']}** — {info['note']}. Volume = FUT_AGGTE_VOL (all listed "
                       "contracts). FX futures excluded (real FX volume trades OTC). Close-based; "
                       "reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Money Flow Index: price + MFI vs RSI on one 0–100 panel — validation and flow-vs-momentum
# gaps are directly visible. No st.stop().
if active == "Money Flow Index":
    import altair as alt
    from src.strategies import mfi as _mfi

    _v = _filter_signals(df[df["strategy"] == active])
    _tick = dict(zip(_v["market"], _v["instruments"]))
    if _v.empty:
        st.info("No MFI reads yet — click **🔁 Re-run signals** on the 🏠 Home page.")
    else:
        sel = st.selectbox("Chart a market (strongest read first)", _v["market"].tolist(), key="mfi_market")
        try:
            cdata, info = _mfi.mfi_chart_data(_tick[sel])
        except Exception as e:
            cdata, info = None, None
            st.info(f"Couldn't build the chart for {sel}: {e}")
        if cdata is not None and not cdata.empty:
            _cc = brand.chart_colors()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", info["signal"])
            c2.metric("MFI (14)", f"{info['mfi']:.0f}", help="≥80 overbought on flow · ≤20 oversold.")
            c3.metric("RSI (14)", f"{info['rsi']:.0f}", help="For the validation read — momentum without flow is suspect.")
            c4.metric("MFI score", f"{info['score']:.0f}/100")
            base = alt.Chart(cdata).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
            price_ln = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
                y=alt.Y("price:Q", title=_ax(_tick[sel]), scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("price:Q", title=_ax(_tick[sel]), format=",.2f")])
            brand.show_chart(price_ln.properties(height=300, title=f"{sel} — price"))
            mfi_ln = base.mark_line(color="#00897B", strokeWidth=2.1).encode(
                y=alt.Y("mfi:Q", title="MFI / RSI", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(labelFontSize=11, titleFontSize=13)),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("mfi:Q", title="MFI", format=".0f")])
            rsi_ln = base.mark_line(color="#7E57C2", strokeWidth=1.6, strokeDash=[4, 3]).encode(
                y="rsi:Q", tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("rsi:Q", title="RSI", format=".0f")])
            _bands = alt.Chart(pd.DataFrame({"y": [80.0, 20.0]})).mark_rule(
                color=_cc["muted"], strokeDash=[2, 2], strokeWidth=1.2).encode(y="y:Q")
            brand.show_chart((_bands + mfi_ln + rsi_ln).properties(
                height=190, title="MFI (solid, volume-weighted) vs RSI (dashed) — 80/20 = flow extremes"))
            st.caption(f"**{info['signal']}** — {info['note']}. MFI = 14-day money-flow RSI "
                       "(close × volume). FX futures excluded (real FX volume trades OTC). "
                       "Close-based; reads the cached snapshot — no Bloomberg pull.")
    st.divider()

# Volatility / Skew get a dedicated visual client report (charts), built from the
# full cross-section rather than the hand-ticked rows below.
if active in REPORTS:
    cfg = REPORTS[active]
    st.markdown(cfg["blurb"])
    if active == "Volatility":
        _vol_charts(threshold)
    elif active == "Skew Volatility":
        _skew_charts(threshold)
    elif active == "Vol Term Structure":
        _term_charts(threshold)
    # The visual-report Generate/Email controls live at the FOOT of the page (below the cross-section
    # table) — see the `if active in REPORTS` block at the very bottom of this script.
    st.caption("Scroll to the foot of the page to generate the visual report; the table below is the "
               "full cross-section — tick rows for a plain-table PDF instead.")
    st.divider()

# AG Fundamentals: USDA supply/demand + report-calendar event risk. Refresh pulls the
# USDA calendar (always) + NASS stocks (if NASS_API_KEY is set); the generic
# opportunities table below then shows the flags (no st.stop() — table still renders).
if active == "AG Fundamentals":
    from src import agdata

    a_refresh, a_note = st.columns([1, 3])
    if a_refresh.button("↻ Refresh AG data (USDA / NASS)",
                        help="Refresh the USDA report calendar and pull NASS grain stocks. "
                             "The calendar works with no key; stocks need the free NASS_API_KEY."):
        with st.spinner("Refreshing USDA ag data…"):
            try:
                agdata.compute(force=True)
                run_daily.run()
                load_signals.clear()
                _ag_err = None
            except Exception as e:
                _ag_err = str(e)
        if _ag_err:
            st.error("AG refresh failed (network?):\n\n" + _ag_err)
        else:
            st.rerun()
    if agdata.NASS_KEY:
        a_note.caption("USDA **report-calendar event risk** + **NASS grain-stocks** tightness percentiles. "
                       "Managed-money positioning is on the COT Reports page.")
    else:
        a_note.caption("Showing USDA **report-calendar event risk**. For NASS stocks-tightness flags, set a "
                       "free key once — `setx NASS_API_KEY <key>` (quickstats.nass.usda.gov/api) — then Refresh.")

    # The WASDE / USDA-Reaction report generators are pinned to the FOOT of the page (below the ag
    # flags table), so the pulled data shows first — see the `if active == "AG Fundamentals"` block
    # at the very bottom of this script.

# COT Reports: a dedicated positioning page (charts + ranked bar + extremes table +
# branded PDF). Self-contained — ends with st.stop() so the generic table is skipped.
if active == "COT Reports":
    import altair as alt
    from src import cotdata, cotstudy, cotseasonality

    c_refresh, c_note = st.columns([1, 3])
    if c_refresh.button("↻ Refresh COT data (CFTC API)",
                        help="Fetch the latest weekly Commitments of Traders from the free CFTC API "
                             "(~20–40s, needs internet — no Bloomberg / Terminal)."):
        with st.spinner("Fetching CFTC Commitments of Traders…"):
            try:
                cotdata.compute(force=True)
                run_daily.run()
                load_signals.clear()
                _cot_err = None
            except Exception as e:
                _cot_err = str(e)
        if _cot_err:
            st.error("COT refresh failed (network?):\n\n" + _cot_err)
        else:
            st.rerun()
    c_note.caption("Weekly CFTC report (Tuesday as-of, published Friday), pulled from the free CFTC API "
                   "— works with the Bloomberg Terminal closed. Commodities show **Managed Money**; "
                   "financials show **Leveraged Funds**.")

    _raw_hist = pd.read_parquet(COT_HISTORY_FILE) if COT_HISTORY_FILE.exists() else pd.DataFrame()
    _raw_detail = pd.read_parquet(COT_DETAIL_FILE) if COT_DETAIL_FILE.exists() else pd.DataFrame()
    hist = _filter_signals(_raw_hist)
    detail = _filter_signals(_raw_detail)
    if _raw_detail.empty or _raw_hist.empty:
        st.info("No COT data cached yet — click **↻ Refresh COT data** above to pull it from the CFTC API.")
        st.stop()
    if detail.empty or hist.empty:
        st.warning("All COT markets are switched off by the **Home sector filter** — the data is cached, "
                   "it's just hidden. Turn sectors back on (top of the Home page) to see it.")
        st.stop()

    cutoff = st.slider("Crowded when COT Index ≥ (crowded short at ≤ 100 − this)",
                       min_value=60, max_value=95, value=int(trigger_default("COT Reports", 80)),
                       step=1, key="cot_cutoff")
    hi, lo = float(cutoff), 100.0 - float(cutoff)
    asof_dt = pd.to_datetime(detail["date"]).max()
    n_long = int((detail["cot_index"] >= hi).sum())
    n_short = int((detail["cot_index"] <= lo).sum())
    st.info(f"**Trigger:** COT Index ≥ {hi:g} (crowded long) or ≤ {lo:g} (crowded short). "
            f"Latest report **{asof_dt:%d %b %Y}** · **{n_long}** crowded long · **{n_short}** crowded "
            f"short across {int(detail['cot_index'].notna().sum())} markets.")
    _cd1, _cd2 = st.columns([0.74, 0.26])
    if _cd2.button("📌 Set default", key="cot_cutoff_def", use_container_width=True,
                   help="Save this cutoff as the default for the COT page — it loads on every launch."):
        save_trigger_default("COT Reports", int(cutoff))
        st.toast(f"Saved {int(cutoff)} as the default COT cutoff.", icon="📌")
    _cd1.caption(f"📌 Default cutoff: **{int(trigger_default('COT Reports', 80))}** — change the slider, "
                 "then **Set default** to make it the new default.")

    # --- whole-book heatmap (markets x last 26 weeks, coloured by COT Index) ---
    st.markdown("##### Whole-book positioning heatmap")
    _ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
    _last = sorted(hist["date"].unique())[-26:]
    hm = hist[hist["date"].isin(_last)].dropna(subset=["cot_index"]).copy()
    _ord = detail.copy()
    _ord["_a"] = _ord["asset"].map({a: i for i, a in enumerate(_ASSET_ORDER)}).fillna(99)
    market_order = _ord.sort_values(["_a", "cot_index"], ascending=[True, False])["market"].tolist()
    heat = alt.Chart(hm).mark_rect().encode(
        x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(labelFontSize=9, labelAngle=-45)),
        y=alt.Y("market:N", sort=market_order, title=None, axis=alt.Axis(labelFontSize=10)),
        color=alt.Color("cot_index:Q", scale=alt.Scale(scheme="redblue", domain=[0, 100]),
                        title="COT Index", legend=alt.Legend(orient="top", titleFontSize=11)),
        tooltip=[alt.Tooltip("market:N", title="Market"), alt.Tooltip("category:N", title="Bucket"),
                 alt.Tooltip("date:T", title="Week"), alt.Tooltip("cot_index:Q", title="COT Index", format=".0f")],
    ).properties(height=18 * len(market_order),
                 title="COT Index by market — last 26 weeks (red = crowded short · blue = crowded long)")
    brand.show_chart(heat)
    st.caption("Each cell is one market-week. Rows grouped by asset class, most net-long at the top of each "
               "group. The right-most column is the latest report.")
    st.divider()

    # --- per-market interactive chart ---
    labels = {r.ticker: f"{r.market} · {r.asset}" for r in detail.itertuples(index=False)}
    order = detail["ticker"].tolist()                       # most-crowded first
    sel = st.selectbox("Chart a market (most crowded first)", order,
                       format_func=lambda t: labels.get(t, t), key="cot_sel")
    win = st.radio("Window", ["1Y", "3Y", "Max"], index=1, horizontal=True, key="cot_win")
    weeks = {"1Y": 52, "3Y": 156, "Max": 100000}[win]

    g = hist[hist["ticker"] == sel].sort_values("date").tail(weeks).copy()
    g["negshort"] = -g["short"]
    drow = detail[detail["ticker"] == sel].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    idx_now = float(drow["cot_index"]) if pd.notna(drow["cot_index"]) else float("nan")
    m1.metric("COT Index", "—" if idx_now != idx_now else f"{idx_now:.0f}",
              help="0 = most net-short in 3y · 100 = most net-long")
    m2.metric("Net", f"{drow['net']:+,.0f}",
              delta=None if pd.isna(drow["chg_net"]) else f"{drow['chg_net']:+,.0f} wk")
    m3.metric("Long", f"{drow['long']:,.0f}")
    m4.metric("Short", f"{drow['short']:,.0f}")
    m5.metric("Net % OI", "—" if pd.isna(drow["net_pct_oi"]) else f"{drow['net_pct_oi']:+.0f}%")

    _cc = brand.chart_colors()
    base = alt.Chart(g).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
    long_bar = base.mark_bar(color=_cc["series"]).encode(
        y=alt.Y("long:Q", title=f"Contracts ({drow['category']})",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
        tooltip=[alt.Tooltip("date:T", title="Week"), alt.Tooltip("long:Q", title="Long", format=",")])
    short_bar = base.mark_bar(color=_cc["short"]).encode(
        y="negshort:Q",
        tooltip=[alt.Tooltip("date:T", title="Week"), alt.Tooltip("short:Q", title="Short", format=",")])
    net_outline = base.mark_line(color=_cc["halo"], strokeWidth=5.2).encode(y="net:Q")  # halo separates net
    net_line = base.mark_line(color=_cc["accent"], strokeWidth=3.4).encode(             # gold net, thick
        y="net:Q", tooltip=[alt.Tooltip("date:T", title="Week"), alt.Tooltip("net:Q", title="Net", format=",")])
    contracts = alt.layer(long_bar, short_bar, net_outline, net_line)
    price_line = base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
        y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False),
                axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
        tooltip=[alt.Tooltip("price:Q", title="Price", format=",.2f")])
    pos_chart = alt.layer(contracts, price_line).resolve_scale(y="independent").properties(
        height=360, title=f"{labels.get(sel, sel)} — long up / short down / net (gold) · price")

    osc_base = alt.Chart(g).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
    osc_line = osc_base.mark_line(color=_cc["ink"]).encode(
        y=alt.Y("cot_index:Q", title="COT Index", scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 20, 50, 80, 100], labelFontSize=12, titleFontSize=13)),
        tooltip=[alt.Tooltip("date:T", title="Week"), alt.Tooltip("cot_index:Q", title="COT Index", format=".0f")])
    hi_rule = alt.Chart(pd.DataFrame({"y": [hi]})).mark_rule(color=_cc["accent"], strokeDash=[4, 3]).encode(y="y:Q")
    lo_rule = alt.Chart(pd.DataFrame({"y": [lo]})).mark_rule(color=_cc["short"], strokeDash=[4, 3]).encode(y="y:Q")
    osc_chart = alt.layer(osc_line, hi_rule, lo_rule).properties(
        height=150, title="COT Index 0–100 (dashed = crowded long / short thresholds)")

    brand.show_chart(pos_chart)

    band_base = alt.Chart(g).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
    band_area = band_base.mark_area(color=_cc["muted"], opacity=0.30).encode(
        y=alt.Y("npo_p10:Q", title="Net % OI", axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
        y2="npo_p90:Q")
    band_med = band_base.mark_line(color=_cc["muted"], strokeDash=[4, 3], strokeWidth=1.6).encode(y="npo_med:Q")
    band_line = band_base.mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
        y="net_pct_oi:Q",
        tooltip=[alt.Tooltip("date:T", title="Week"), alt.Tooltip("net_pct_oi:Q", title="Net %OI", format=".1f")])
    band_chart = alt.layer(band_area, band_med, band_line).properties(
        height=200, title="Net positioning as % of open interest, vs its 3-year 10–90% range (grey band)")
    brand.show_chart(band_chart)

    brand.show_chart(osc_chart)
    st.caption("Top: gross long up / short down, **net** line (gold), **price** on the right axis. "
               "Middle: net as % of open interest vs its own 3-year 10–90% range. Bottom: the COT Index. "
               "Charts read the cached CFTC data — no Bloomberg pull; price uses the same datafeed as the rest of the app.")

    # --- positioning seasonality (SEAG-style): net %OI by week-of-year, across all years ---
    _seas_df, _seas_info = cotseasonality.seasonal_long(hist[hist["ticker"] == sel], metric="net_pct_oi")
    if not _seas_df.empty:
        _sx = alt.Chart(_seas_df).encode(
            x=alt.X("wdate:T", title=None,
                    axis=alt.Axis(format="%b", tickCount="month", labelFontSize=12)))
        _seas_band = _sx.mark_area(color=_cc["muted"], opacity=0.35).encode(
            y=alt.Y("p25:Q", title="Net % OI", axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            y2="p75:Q")
        _seas_med = _sx.mark_line(color=_cc["series"], strokeWidth=2.6).encode(
            y="med:Q", tooltip=[alt.Tooltip("woy:Q", title="Week"),
                                alt.Tooltip("med:Q", title="Median %OI", format=".1f")])
        _seas_halo = _sx.mark_line(color=_cc["halo"], strokeWidth=4.6).encode(y="current:Q")
        _seas_cur = _sx.mark_line(color=_cc["accent"], strokeWidth=3).encode(
            y="current:Q", tooltip=[alt.Tooltip("woy:Q", title="Week"),
                                    alt.Tooltip("current:Q", title=f"{_seas_info['cur_year']} %OI", format=".1f")])
        _seas_chart = alt.layer(_seas_band, _seas_med, _seas_halo, _seas_cur).properties(
            height=210,
            title=f"Positioning seasonality — net %OI by week of year "
                  f"(median {_seas_info['years']}y · band = 25–75% of years · gold = {_seas_info['cur_year']})")
        brand.show_chart(_seas_chart)
        st.caption("How speculative positioning (net % of open interest) typically moves through the "
                   "calendar year: **blue** = median across all years, **grey band** = 25–75% of years "
                   "(wider band = less reliable seasonality), **gold** = the current year so far. "
                   "Strongest for ags and energy; weak or flat for most financials.")

    # --- forward-return study for the selected market (independent episodes + baseline) ---
    st.markdown(
        "**Forward returns after positioning extremes** — once positioning here became crowded, what did "
        f"price tend to do next? For each time this market's COT Index *entered* a crowded zone (rose to "
        f"≥ {int(hi)} or fell to ≤ {int(lo)}), the table averages the underlying's move over the next 4 and "
        "13 weeks, and compares it to the **baseline** (its average move over *all* weeks, ignoring "
        "positioning). **Episodes (n)** = how many separate times it entered that zone — a multi-week run "
        "counts once, so it's the number of independent occurrences. A descriptive back-look, not a forecast.")
    _study = cotstudy.forward_return_study(hist[hist["ticker"] == sel], hi, lo, min_episodes=1)
    if _study.empty:
        st.caption("Not enough aligned price history for this market to compute the study.")
    else:
        _r = _study.iloc[0]

        def _pct(v):
            return "—" if pd.isna(v) else f"{v:+.1f}%"

        _fr = pd.DataFrame([
            {"Horizon": "+4 weeks", "After crowded long": _pct(_r["long_4"]),
             "After crowded short": _pct(_r["short_4"]), "Baseline (all weeks)": _pct(_r["base_4"])},
            {"Horizon": "+13 weeks", "After crowded long": _pct(_r["long_13"]),
             "After crowded short": _pct(_r["short_13"]), "Baseline (all weeks)": _pct(_r["base_13"])},
        ])
        brand.themed_dataframe(_fr, {})
        _epL, _epS = int(_r["episodes_long"]), int(_r["episodes_short"])
        _gp = pd.to_datetime(hist[hist["ticker"] == sel].dropna(subset=["price"])["date"])
        _wk, _yr = len(_gp), round(len(_gp) / 52.0, 1)
        _start = _gp.min().strftime("%b %Y") if len(_gp) else "—"
        _thin = (" ⚠️ Thin price history in this data mode — far more robust once the 10-year price database "
                 "is built on Bloomberg." if (_epL + _epS) < 4 else "")
        st.caption(f"Looks back to ~**{_start}** — ~**{_wk} weeks ({_yr} yrs)** of price history. Episodes in "
                   f"that window: **{_epL}** crowded-long, **{_epS}** crowded-short (a multi-week run counts "
                   "once). Compare each column to the baseline — that gap is what positioning has added. "
                   "Small, overlapping samples; treat as a hypothesis, not a signal." + _thin)
    st.divider()

    # --- cross-section table (all markets, flagged at the chosen cutoff) ---
    ci = pd.to_numeric(detail["cot_index"], errors="coerce")
    show = pd.DataFrame({
        "Market": detail["market"] + "  ·  " + detail["asset"],
        "Bucket": detail["category"],
        "COT Index": ci,
        "Net": detail["net"],
        "Long": detail["long"],
        "Short": detail["short"],
        "Net % OI": detail["net_pct_oi"],
        "Δ wk": detail["chg_net"],
        "Signal": np.where(ci >= hi, "Crowded long", np.where(ci <= lo, "Crowded short", "—")),
    })
    _cot_fmt = {"COT Index": "{:.0f}", "Net": "{:+,.0f}", "Long": "{:,.0f}", "Short": "{:,.0f}",
                "Net % OI": "{:+.0f}%", "Δ wk": "{:+,.0f}"}

    def _sig_color(col):
        return ["color:#137333;font-weight:700" if v == "Crowded long"
                else "color:#c5221f;font-weight:700" if v == "Crowded short"
                else "color:#888" for v in col]

    _q = st.text_input("Find a market", key="cot_search", placeholder=prodsearch.PLACEHOLDER).strip()
    if _q:
        show = prodsearch.filter_frame(show, INSTRUMENTS, _q, name_col="Market")
        if show.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q)); st.stop()
    st.caption("Full cross-section — every market with CFTC COT, most crowded first.")
    brand.themed_dataframe(show, _cot_fmt, colorers=[(["Signal"], _sig_color)],
                           na_rep="—", height=520)

    # --- branded PDF (whole-book chartbook) + on-demand email. Pinned to the FOOT of the page for a
    #     consistent "generate + email at the bottom" layout across the app. ---
    st.divider()
    st.markdown("**Daily client report** — the heatmap, ranked positioning bar, crowded-markets + "
                "forward-return tables, and a chart for every market, on the XP brand.")
    qc1, qc2 = st.columns(2)
    _gen = None
    if qc1.button("📈 Generate — screen (crisp)", type="primary", disabled=not COT_DETAIL_FILE.exists()):
        _gen = ("screen", "COT_Positioning_Report.pdf")
    if qc2.button("📧 Generate — email (smaller file)", disabled=not COT_DETAIL_FILE.exists()):
        _gen = ("email", "COT_Positioning_Report_email.pdf")
    if _gen:
        quality, fname = _gen
        with st.spinner(f"Rendering COT charts… ({quality}, whole book)"):
            with tempfile.TemporaryDirectory() as tmp:
                out_pdf = Path(tmp) / "cot.pdf"
                result = subprocess.run(
                    [sys.executable, str(COTREPORT_CLI), str(COT_DETAIL_FILE), str(out_pdf),
                     "--asof", str(meta.get("as_of", "")), "--threshold", str(cutoff),
                     "--quality", quality],
                    capture_output=True, text=True,
                )
                ok = result.returncode == 0 and out_pdf.exists()
                pdf_bytes = out_pdf.read_bytes() if ok else None
        if not ok:
            st.session_state.pop("cot_pdf", None)
            st.error("COT report failed:\n\n" + (result.stderr or result.stdout or "no output"))
        else:
            st.session_state["cot_pdf"] = pdf_bytes
            st.session_state["cot_pdf_name"] = fname
            st.session_state["cot_pdf_mb"] = len(pdf_bytes) / 1024 / 1024
            st.success(f"{quality.capitalize()} report ready — {st.session_state['cot_pdf_mb']:.1f} MB.")
    if st.session_state.get("cot_pdf"):
        st.download_button(
            f"⬇️ Download {st.session_state.get('cot_pdf_name', 'COT_Positioning_Report.pdf')} "
            f"({st.session_state.get('cot_pdf_mb', 0):.1f} MB)",
            data=st.session_state["cot_pdf"],
            file_name=st.session_state.get("cot_pdf_name", "COT_Positioning_Report.pdf"),
            mime="application/pdf")
    st.caption("Screen = crisp 160-dpi charts. Email = lighter 96-dpi for a smaller attachment. "
               "The buttons above only build a file to download — they don't email anyone.")

    # Email the report on demand — pick recipients (defaults to the desk: Ben + Said, the same
    # list as the scheduled Friday-release job). Confirm-gated against stray sends.
    st.markdown("**Email this report**")
    cot_to = _recipient_picker("cot", "cot")
    ec1, ec2 = st.columns([1, 4])
    confirm_send = ec1.checkbox("Confirm", key="cot_email_confirm")
    if ec2.button("📤 Email report now", disabled=not (confirm_send and cot_to and COT_DETAIL_FILE.exists())):
        with st.spinner("Building the email report and sending…"):
            try:
                import cot_scheduled_email as _cote
                with tempfile.TemporaryDirectory() as tmp:
                    out_pdf = Path(tmp) / "cot_desk.pdf"
                    res = subprocess.run(
                        [sys.executable, str(COTREPORT_CLI), str(COT_DETAIL_FILE), str(out_pdf),
                         "--asof", str(meta.get("as_of", "")), "--threshold", str(cutoff), "--quality", "email"],
                        capture_output=True, text=True)
                    if res.returncode != 0 or not out_pdf.exists():
                        raise RuntimeError(res.stderr or res.stdout or "report build failed")
                    asof = pd.to_datetime(detail["date"]).max().date()
                    _cote.send_email(out_pdf, asof, dry_run=False, to_override=cot_to)
                    recipients = cot_to
                ok, err = True, None
            except Exception as e:
                ok, err = False, str(e)
        if ok:
            st.success("Emailed to " + ", ".join(recipients) + ".")
        else:
            st.error("Email failed:\n\n" + err)
    st.caption("Sends the report at the current crowded-cutoff to the chosen recipients immediately, using "
               "the cached CFTC data shown on this page (hit ↻ Refresh first if you want the very latest).")
    st.stop()


# Put/Call Ratios: options put/call OI (headline) + volume monitor — its own page
# (heatmap + ranked table + per-product chart + branded PDF). Self-contained → st.stop().
if active == "Put/Call Ratios":
    import altair as alt

    detail = _filter_signals(pd.read_parquet(PC_DETAIL_FILE) if PC_DETAIL_FILE.exists() else pd.DataFrame())
    hist = _filter_signals(pd.read_parquet(PC_HISTORY_FILE) if PC_HISTORY_FILE.exists() else pd.DataFrame())
    if detail.empty:
        st.info("No put/call data cached yet — click **🔁 Re-run signals** on the 🏠 Home page.")
        st.stop()

    cutoff = st.slider("Put-heavy when OI P/C percentile ≥ (call-heavy at ≤ 100 − this)",
                       min_value=60, max_value=95, value=int(trigger_default("Put/Call Ratios", 80)),
                       step=1, key="pc_cutoff")
    hi, lo = float(cutoff), 100.0 - float(cutoff)
    oi_p = pd.to_numeric(detail["oi_pctl"], errors="coerce")
    n_put = int((oi_p >= hi).sum())
    n_call = int((oi_p <= lo).sum())
    st.info(f"**Trigger:** OI P/C percentile ≥ {hi:g} (put-heavy) or ≤ {lo:g} (call-heavy). "
            f"**{n_put}** put-heavy · **{n_call}** call-heavy across {int(oi_p.notna().sum())} markets "
            "with options data. The ratio is **puts ÷ calls** (OI basis = put OI ÷ call OI, the headline; "
            "volume basis = put volume ÷ call volume; above 1 = put-heavy, below 1 = call-heavy). Each "
            "product's OI ratio is scored 0–100 vs its own 1-year range, so high = unusually **put-heavy** "
            "(defensive / hedging demand) — often read contrarian-bullish; low = unusually **call-heavy** "
            "(bullish) — contrarian-bearish. Volume P/C (today's flow) is shown alongside the headline.")
    _pd1, _pd2 = st.columns([0.74, 0.26])
    if _pd2.button("📌 Set default", key="pc_cutoff_def", use_container_width=True,
                   help="Save this cutoff as the default for the Put/Call page — it loads on every launch."):
        save_trigger_default("Put/Call Ratios", int(cutoff))
        st.toast(f"Saved {int(cutoff)} as the default Put/Call cutoff.", icon="📌")
    _pd1.caption(f"📌 Default cutoff: **{int(trigger_default('Put/Call Ratios', 80))}** — change the slider, "
                 "then **Set default** to make it the new default.")

    # --- Yesterday's activity leaderboard: DIVERGING, each side % of its OWN 1y average ---
    # Calls right / puts left, each as % of THAT SIDE's 1-year daily average; the dashed 100%
    # line on each side is the average marker. Size-normalised; extremes kept (full scale).
    _cc_a = brand.chart_colors()
    _av = detail.copy()
    for _c in ("call_last", "put_last", "avg_call", "avg_put"):
        _av[_c] = pd.to_numeric(_av[_c], errors="coerce")
    _av["tot_last"] = _av["call_last"].fillna(0) + _av["put_last"].fillna(0)
    _av = _av[(_av["avg_call"] > 0) & (_av["avg_put"] > 0) & (_av["tot_last"] > 0)].copy()
    if not _av.empty:
        st.markdown("##### Yesterday's options activity — calls vs puts, each against its own 1-year average")
        _av["call_pct"] = _av["call_last"].fillna(0) / _av["avg_call"] * 100.0
        _av["put_pct"] = _av["put_last"].fillna(0) / _av["avg_put"] * 100.0
        _av["neg_put_pct"] = -_av["put_pct"]
        _av["rank_pct"] = _av[["call_pct", "put_pct"]].max(axis=1)
        _av["call_lbl"] = _av["call_pct"].map(lambda v: f"{v:.0f}%")
        _av["put_lbl"] = _av["put_pct"].map(lambda v: f"{v:.0f}%")
        if "vol_days" in _av.columns:                       # (Nd) = days of history behind the average
            _vd = pd.to_numeric(_av["vol_days"], errors="coerce").fillna(0).astype(int)
            _av["mkt_lbl"] = [f"{m} ({v}d)" for m, v in zip(_av["market"], _vd)]
        else:
            _av["mkt_lbl"] = _av["market"].astype(str)
        _order = _av.sort_values("rank_pct", ascending=False)["mkt_lbl"].tolist()
        _M = float(max(_av["call_pct"].max(), _av["put_pct"].max()) or 100.0)   # full scale — keep extremes
        _bar_df = pd.concat([_av.assign(Side="Calls", pct=_av["call_pct"]),
                             _av.assign(Side="Puts", pct=-_av["put_pct"])])
        _bars = alt.Chart(_bar_df).mark_bar().encode(
            y=alt.Y("mkt_lbl:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11)),
            x=alt.X("pct:Q", stack=None,
                    title="← puts traded      calls traded →   (each as % of that side's own 1-year daily average)",
                    scale=alt.Scale(domain=[-_M * 1.20, _M * 1.20]),
                    axis=alt.Axis(labelFontSize=11, labelExpr="abs(datum.value) + '%'")),
            color=alt.Color("Side:N", scale=alt.Scale(domain=["Calls", "Puts"], range=[_cc_a["long"], _cc_a["short"]]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
            tooltip=[alt.Tooltip("market:N", title="Market"), alt.Tooltip("asset:N", title="Asset"),
                     alt.Tooltip("Side:N"),
                     alt.Tooltip("call_last:Q", title="Calls (contracts)", format=",.0f"),
                     alt.Tooltip("put_last:Q", title="Puts (contracts)", format=",.0f"),
                     alt.Tooltip("call_pct:Q", title="Calls (% of avg)", format=".0f"),
                     alt.Tooltip("put_pct:Q", title="Puts (% of avg)", format=".0f"),
                     alt.Tooltip("vol_days:Q", title="History (days)", format=".0f")])
        _avg100 = alt.Chart(pd.DataFrame({"x": [100.0, -100.0]})).mark_rule(
            color=_cc_a["ink"], strokeDash=[5, 3]).encode(x="x:Q")
        _zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_cc_a["muted"]).encode(x="x:Q")
        _lblc = alt.Chart(_av).mark_text(align="left", dx=3, fontSize=9, color=_cc_a["long"]).encode(
            y=alt.Y("mkt_lbl:N", sort=_order), x=alt.X("call_pct:Q"), text="call_lbl:N")
        _lblp = alt.Chart(_av).mark_text(align="right", dx=-3, fontSize=9, color=_cc_a["short"]).encode(
            y=alt.Y("mkt_lbl:N", sort=_order), x=alt.X("neg_put_pct:Q"), text="put_lbl:N")
        _act_chart = alt.layer(_bars, _avg100, _zero, _lblc, _lblp).properties(
            height=22 * max(1, len(_order)),
            title="Yesterday's options volume — calls (green, right) / puts (red, left) as % of each side's 1-year daily average; dashed 100% = average")
        brand.show_chart(_act_chart)
        st.caption("**Calls point right, puts point left**, each as a **% of that side's own 1-year daily "
                   "average** so size doesn't distort it. The **dashed 100% line on each side is the average** — "
                   "a bar past it traded **above average**. Ranked by the bigger side. The **(Nd)** next to each "
                   "name is how many days of history the average uses — under ~120 days is still building, so "
                   "read those with caution. Hover for contract counts; full totals are in each product's detail.")
        st.divider()

    # --- whole-book heatmap (markets × last ~6 weeks, coloured by OI P/C percentile) ---
    if not hist.empty:
        st.markdown("##### Whole-book put/call heatmap (OI basis)")
        _ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
        _h = hist.copy()
        _h["date"] = pd.to_datetime(_h["date"])
        _last = sorted(_h["date"].unique())[-30:]
        _h = _h[_h["date"].isin(_last)].dropna(subset=["oi_pctl"])
        # Single (non-faceted) heatmap: Streamlit's in-browser Vega mis-renders FACETED
        # charts under use_container_width (only the first band paints), so we keep one
        # rect chart and get the grouping from the row order instead. Restrict the rows
        # to markets that actually have a percentile in the window — no blank rows.
        _present = set(_h["market"])
        _od = detail[detail["market"].isin(_present)].copy()
        _od["_a"] = _od["asset"].map({a: i for i, a in enumerate(_ASSET_ORDER)}).fillna(99)
        market_order = _od.sort_values(["_a", "oi_pctl"], ascending=[True, False])["market"].tolist()
        heat = alt.Chart(_h).mark_rect().encode(
            x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(labelFontSize=9, labelAngle=-45)),
            y=alt.Y("market:N", sort=market_order, title=None, axis=alt.Axis(labelFontSize=10)),
            color=alt.Color("oi_pctl:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True, domain=[0, 100]),
                            title="OI P/C %ile", legend=alt.Legend(orient="top", titleFontSize=11)),
            tooltip=[alt.Tooltip("market:N", title="Market"), alt.Tooltip("asset:N", title="Asset"),
                     alt.Tooltip("date:T", title="Day"),
                     alt.Tooltip("pc_oi:Q", title="OI P/C", format=".2f"),
                     alt.Tooltip("oi_pctl:Q", title="%ile", format=".0f")],
        ).properties(height=18 * max(1, len(market_order)),
                     title="OI put/call percentile by market — last 30 days (red = put-heavy · green = call-heavy)")
        brand.show_chart(heat)
        st.caption("Each cell is one market-day. Rows are **grouped by asset class** (Indices, STIRs, Bonds, "
                   "FX, Energy, Metals, Agriculture, Softs), most put-heavy at the top of each group — hover a "
                   "row for its asset. The right-most column is today. Markets with under ~60 days of put/call "
                   "history (currently parts of the bond complex) are omitted until their history builds.")
        st.divider()

    # --- per-product interactive chart ---
    labels = {r.ticker: f"{r.market} · {r.asset}" for r in detail.itertuples(index=False)}
    sel = st.selectbox("Chart a market (most extreme first)", detail["ticker"].tolist(),
                       format_func=lambda t: labels.get(t, t), key="pc_sel")
    drow = detail[detail["ticker"] == sel].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OI P/C", f"{drow['pc_oi']:.2f}", help="Standing positioning: put open interest ÷ call open interest")
    m2.metric("OI P/C %ile", "—" if pd.isna(drow["oi_pctl"]) else f"{drow['oi_pctl']:.0f}",
              help="Today's OI ratio within its own 1-year range (0 = most call-heavy · 100 = most put-heavy)")
    m3.metric("Vol P/C", "—" if pd.isna(drow["pc_vol"]) else f"{drow['pc_vol']:.2f}",
              help="Today's traded flow: put volume ÷ call volume")
    m4.metric("Flow vs OI", "—" if pd.isna(drow["divergence"]) else f"{drow['divergence']:+.0f}",
              help="Volume %ile − OI %ile; +ve = today's flow more put-heavy than the standing OI")
    _ac = f"{drow['tot_call']:,.0f} calls" if pd.notna(drow["tot_call"]) else "— calls"
    _ap = f"{drow['tot_put']:,.0f} puts" if pd.notna(drow["tot_put"]) else "— puts"
    _ad = f"~{drow['avg_day']:,.0f} contracts/day" if pd.notna(drow["avg_day"]) else "—/day"
    st.caption(f"**Options activity (last ~1y):** {_ac} traded · {_ap} traded · {_ad}")

    if not hist.empty:
        g = hist[hist["ticker"] == sel].copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.sort_values("date")
        ratios = (g[["date", "pc_oi", "pc_vol"]]
                  .rename(columns={"pc_oi": "OI P/C", "pc_vol": "Volume P/C"})
                  .melt("date", var_name="Basis", value_name="pc"))
        _cc = brand.chart_colors()
        pc_lines = alt.Chart(ratios).mark_line().encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("pc:Q", title="Put / Call ratio", scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            color=alt.Color("Basis:N", scale=alt.Scale(domain=["OI P/C", "Volume P/C"],
                                                       range=[_cc["ink"], _cc["series"]]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
            tooltip=[alt.Tooltip("date:T", title="Day"), alt.Tooltip("Basis:N"),
                     alt.Tooltip("pc:Q", title="P/C", format=".2f")])
        parity = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
            color=_cc["muted"], strokeDash=[4, 3]).encode(y="y:Q")
        price_line = alt.Chart(g).mark_line(color=_cc["short"], strokeWidth=2).encode(
            x="date:T", y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False),
                                axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            tooltip=[alt.Tooltip("price:Q", title="Price", format=",.2f")])
        # P/C ratios + parity share the left axis; price gets its own (independent) right axis.
        ratios_layer = alt.layer(pc_lines, parity)
        ratio_chart = alt.layer(ratios_layer, price_line).resolve_scale(y="independent").properties(
            height=340, title=f"{labels.get(sel, sel)} — OI P/C · volume P/C (blue) · price (red)")
        brand.show_chart(ratio_chart)

        # Daily option volume — how many calls / puts traded each day vs the ~1y daily average.
        if {"call_vol", "put_vol"}.issubset(g.columns) and g[["call_vol", "put_vol"]].notna().any().any():
            vol_long = (g[["date", "call_vol", "put_vol"]]
                        .rename(columns={"call_vol": "Calls", "put_vol": "Puts"})
                        .melt("date", var_name="Side", value_name="vol"))
            vbars = alt.Chart(vol_long).mark_bar().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("vol:Q", title="Contracts traded / day", stack=True,
                        axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
                color=alt.Color("Side:N", scale=alt.Scale(domain=["Calls", "Puts"],
                                                          range=[_cc["long"], _cc["short"]]),
                                legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                tooltip=[alt.Tooltip("date:T", title="Day"), alt.Tooltip("Side:N"),
                         alt.Tooltip("vol:Q", title="Contracts", format=",.0f")])
            vlayers = [vbars]
            if pd.notna(drow["avg_day"]):
                avg_rule = alt.Chart(pd.DataFrame({"y": [float(drow["avg_day"])]})).mark_rule(
                    color=_cc["ink"], strokeDash=[5, 3], strokeWidth=2.1).encode(y="y:Q")
                vlayers.append(avg_rule)
            vol_chart = alt.layer(*vlayers).properties(
                height=230,
                title="Daily option volume — calls (green) + puts (red) traded each day vs the ~1y daily average (dashed)")
            brand.show_chart(vol_chart)

        osc_base = alt.Chart(g).encode(x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=12)))
        osc_line = osc_base.mark_line(color=_cc["ink"]).encode(
            y=alt.Y("oi_pctl:Q", title="OI P/C %ile", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[0, 20, 50, 80, 100], labelFontSize=12, titleFontSize=13)),
            tooltip=[alt.Tooltip("date:T", title="Day"), alt.Tooltip("oi_pctl:Q", title="%ile", format=".0f")])
        hi_rule = alt.Chart(pd.DataFrame({"y": [hi]})).mark_rule(color=_cc["short"], strokeDash=[4, 3]).encode(y="y:Q")
        lo_rule = alt.Chart(pd.DataFrame({"y": [lo]})).mark_rule(color=_cc["long"], strokeDash=[4, 3]).encode(y="y:Q")
        osc_chart = alt.layer(osc_line, hi_rule, lo_rule).properties(
            height=150, title="OI P/C percentile 0–100 (dashed = put-heavy / call-heavy thresholds)")
        brand.show_chart(osc_chart)
        st.caption("Charts: (1) the **put/call ratio over time** on both bases with price (red, right axis), "
                   "parity = 1.0; (2) **daily volume** — calls + puts traded each day vs the ~1y daily average "
                   "(dashed line); (3) where today's OI ratio sits in its own 1-year range. Read from the cached "
                   "snapshot — no Bloomberg pull.")
    st.divider()

    # --- cross-section table (all markets) ---
    show = pd.DataFrame({
        "Market": detail["market"] + "  ·  " + detail["asset"],
        "OI P/C": detail["pc_oi"],
        "OI %ile": oi_p,
        "Vol P/C": detail["pc_vol"],
        "Vol %ile": detail["vol_pctl"],
        "Calls (1y)": detail["tot_call"],
        "Puts (1y)": detail["tot_put"],
        "Avg/day": detail["avg_day"],
        "Δ1d (z)": detail["oi_chg_z"],
        "Flow−OI": detail["divergence"],
        "Signal": np.where(oi_p >= hi, "Put-heavy", np.where(oi_p <= lo, "Call-heavy", "—")),
    })

    def _pc_human(v):                                  # 1,240,000 → "1.24M"; 38,400 → "38.4K"
        if pd.isna(v):
            return "—"
        v = float(v)
        if abs(v) >= 1e6:
            return f"{v / 1e6:.2f}M"
        if abs(v) >= 1e3:
            return f"{v / 1e3:.1f}K"
        return f"{v:,.0f}"

    _pc_fmt = {"OI P/C": "{:.2f}", "OI %ile": "{:.0f}", "Vol P/C": "{:.2f}", "Vol %ile": "{:.0f}",
               "Calls (1y)": _pc_human, "Puts (1y)": _pc_human, "Avg/day": _pc_human,
               "Δ1d (z)": "{:+.1f}", "Flow−OI": "{:+.0f}"}

    def _pc_sig_color(col):
        return ["color:#c5221f;font-weight:700" if v == "Put-heavy"
                else "color:#137333;font-weight:700" if v == "Call-heavy"
                else "color:#888" for v in col]

    _q = st.text_input("Find a market", key="pc_search", placeholder=prodsearch.PLACEHOLDER).strip()
    if _q:
        show = prodsearch.filter_frame(show, INSTRUMENTS, _q, name_col="Market")
        if show.empty:
            st.info(prodsearch.NO_MATCH.format(q=_q)); st.stop()
    st.caption("Full cross-section — every market with options data, most extreme first. "
               "Red = put-heavy (defensive) · green = call-heavy (bullish). **Calls / Puts (1y)** = "
               "contracts traded over the last ~1 year; **Avg/day** = average traded per day — click a "
               "column header to sort (e.g. Avg/day for the most active books).")
    brand.themed_dataframe(show, _pc_fmt, colorers=[(["Signal"], _pc_sig_color)],
                           na_rep="—", height=520)

    # --- branded PDF (whole-book chartbook) — crisp for screen, or a lighter email copy. Pinned to
    #     the FOOT of the page for a consistent "generate + email at the bottom" layout. ---
    st.divider()
    st.markdown("**Daily client report** — the heatmap, ranked put/call bar, products-of-interest table, "
                "and a put/call chart for every market, on the XP brand.")
    qc1, qc2 = st.columns(2)
    _gen = None
    if qc1.button("📈 Generate — screen (crisp)", type="primary", disabled=not PC_DETAIL_FILE.exists()):
        _gen = ("screen", "PutCall_Ratios_Report.pdf")
    if qc2.button("📧 Generate — email (smaller file)", disabled=not PC_DETAIL_FILE.exists()):
        _gen = ("email", "PutCall_Ratios_Report_email.pdf")
    if _gen:
        quality, fname = _gen
        with st.spinner(f"Rendering put/call charts… ({quality}, whole book)"):
            with tempfile.TemporaryDirectory() as tmp:
                out_pdf = Path(tmp) / "pc.pdf"
                result = subprocess.run(
                    [sys.executable, str(PCREPORT_CLI), str(PC_DETAIL_FILE), str(out_pdf),
                     "--asof", str(meta.get("as_of", "")), "--threshold", str(cutoff),
                     "--quality", quality],
                    capture_output=True, text=True,
                )
                ok = result.returncode == 0 and out_pdf.exists()
                pdf_bytes = out_pdf.read_bytes() if ok else None
        if not ok:
            st.session_state.pop("pc_pdf", None)
            st.error("Put/Call report failed:\n\n" + (result.stderr or result.stdout or "no output"))
        else:
            st.session_state["pc_pdf"] = pdf_bytes
            st.session_state["pc_pdf_name"] = fname
            st.session_state["pc_pdf_mb"] = len(pdf_bytes) / 1024 / 1024
            st.success(f"{quality.capitalize()} report ready — {st.session_state['pc_pdf_mb']:.1f} MB.")
    if st.session_state.get("pc_pdf"):
        st.download_button(
            f"⬇️ Download {st.session_state.get('pc_pdf_name', 'PutCall_Ratios_Report.pdf')} "
            f"({st.session_state.get('pc_pdf_mb', 0):.1f} MB)",
            data=st.session_state["pc_pdf"],
            file_name=st.session_state.get("pc_pdf_name", "PutCall_Ratios_Report.pdf"),
            mime="application/pdf")
        email_report_ui("pc_pdf", "pc_pdf", st.session_state.get("pc_pdf"),
                        subject="Put/Call Ratios Report",
                        attachment_name=st.session_state.get("pc_pdf_name", "PutCall_Ratios_Report.pdf"))
    st.caption("Screen = crisp 160-dpi charts. Email = lighter 96-dpi for a smaller attachment. "
               "The buttons above only build a file to download — they don't email anyone.")
    st.stop()


# Open Interest: listed-option open interest as a strike × expiry-month heatmap, per
# product — an interactive preview plus a branded PDF (this product, or the whole book).
# Self-contained → st.stop() so the generic opportunities table is skipped.
if active == "Open Interest":
    import altair as alt
    from src.datafeed import get_oi_chain, OI_SNAPSHOT_TICKERS

    _OI_ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
    _oi_order = sorted(
        universe.enabled_tickers(),
        key=lambda t: (_OI_ASSET_ORDER.index(INSTRUMENTS[t][2]) if INSTRUMENTS[t][2] in _OI_ASSET_ORDER else 99,
                       INSTRUMENTS[t][0]))
    try:
        _px = get_history(_oi_order)
        _spot_map = {t: (float(_px[t].dropna().iloc[-1]) if (t in _px and _px[t].notna().any()) else float("nan"))
                     for t in _oi_order}
    except Exception:
        _spot_map = {t: float("nan") for t in _oi_order}

    # ---- PDF builders (shared by every report button on this page) ----
    def _oi_input_frame(tickers, n_expiries, n_strikes):
        frames = []
        for t in tickers:
            try:
                c = get_oi_chain(t, n_expiries=n_expiries, n_strikes=n_strikes)
            except Exception:
                c = None
            if c is None or c.empty:
                continue
            frames.append(c.assign(ticker=t, market=INSTRUMENTS[t][0], asset=INSTRUMENTS[t][2],
                                   spot=_spot_map.get(t, float("nan"))))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _oi_fixed_income_frame():
        """The curated fixed-income book — ONE PRODUCT PER PAGE (full strike chain), in tenor
        order: STIRs, then US vs German at 2s/5s/10s/30s. Each product keeps its per-tenor
        strike grid (step, half-width) so the rate heatmap is realistic."""
        frames, missing, pg = [], [], 0
        for grp in FI_OI_PAGES:
            for tk, step, width in grp["items"]:
                if tk not in INSTRUMENTS:
                    missing.append(tk); continue
                try:
                    c = get_oi_chain(tk, n_expiries=24, n_strikes=None, step=step, width=width)
                except Exception:
                    c = None
                if c is None or c.empty:
                    missing.append(tk); continue
                frames.append(c.assign(ticker=tk, market=INSTRUMENTS[tk][0], asset=INSTRUMENTS[tk][2],
                                       spot=_spot_map.get(tk, float("nan")),
                                       page=pg, page_title=f"{grp['tenor']} — {INSTRUMENTS[tk][0]}"))
                pg += 1
        return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing

    def _oi_render_pdf(frame, scope, fname, spinner, slot="oi_pdf"):
        if frame is None or frame.empty:
            st.session_state.pop(slot, None)
            st.error("No option open interest to render for this selection.")
            return
        with st.spinner(spinner):
            with tempfile.TemporaryDirectory() as tmp:
                _cpq = Path(tmp) / "oi_chain.parquet"
                _out = Path(tmp) / "oi.pdf"
                frame.to_parquet(_cpq, index=False)
                _res = subprocess.run(
                    [sys.executable, str(OIREPORT_CLI), str(_cpq), str(_out),
                     "--asof", str(meta.get("as_of", "")), "--scope", scope],
                    capture_output=True, text=True)
                if _res.returncode == 0 and _out.exists():
                    st.session_state[slot] = _out.read_bytes()
                    st.session_state[f"{slot}_name"] = fname
                    st.session_state[f"{slot}_mb"] = len(st.session_state[slot]) / 1024 / 1024
                else:
                    st.session_state.pop(slot, None)
                    st.error("Open Interest report failed:\n\n" + (_res.stderr or _res.stdout or "no output"))
        if st.session_state.get(slot):
            st.success(f"Report ready — {st.session_state.get(f'{slot}_mb', 0):.1f} MB.")

    # ---- Fixed Income book — the headline report. Shown at the top so it's always available,
    #      independent of the single-product picker below (which st.stop()s on no-chain). ----
    st.markdown("**Fixed Income open-interest book (PDF)** — one product per page: short rates "
                "(SOFR · SONIA · Euribor), then **US vs German** at 2s / 5s / 10s / 30s.")
    if st.button("🏛️ Generate Fixed Income OI Report", type="primary", key="oi_fi_btn"):
        _fi_frame, _fi_missing = _oi_fixed_income_frame()
        _oi_render_pdf(_fi_frame, "grouped", "Fixed_Income_Open_Interest.pdf",
                       "Rendering the fixed-income open-interest book…", slot="oi_fi_pdf")
        if _fi_missing:
            st.caption("Skipped (no chain): " + ", ".join(dict.fromkeys(_fi_missing)) + ".")
    if st.session_state.get("oi_fi_pdf"):
        st.download_button(
            f"⬇️ Download Fixed_Income_Open_Interest.pdf ({st.session_state.get('oi_fi_pdf_mb', 0):.1f} MB)",
            data=st.session_state["oi_fi_pdf"], file_name="Fixed_Income_Open_Interest.pdf",
            mime="application/pdf", key="oi_fi_dl")
        email_report_ui("oi_fi_pdf", "oi_fi_pdf", st.session_state.get("oi_fi_pdf"),
                        subject="Fixed-Income Open Interest", attachment_name="Fixed_Income_Open_Interest.pdf")

    _snap = _load_snap()
    _oi_asof = (_snap or {}).get("oi_as_of") or "never"
    oc1, oc2 = st.columns([1, 2])
    if oc1.button("↻ Refresh OI data", key="oi_refresh",
                  help="Pull the 11 fixed-income option chains live from Bloomberg (Terminal must be up). "
                       "Meant to run weekly — Mondays. The report and heatmaps read this cached data."):
        with st.spinner("Pulling the 11 fixed-income option chains from Bloomberg… (~1–2 min)"):
            _r = subprocess.run([sys.executable, str(SNAPSHOT_CLI), "--oi"], cwd=str(ROOT),
                                capture_output=True, text=True,
                                env={**os.environ, "DATAFEED_MODE": "bloomberg", "PYTHONUTF8": "1"})
        if _r.returncode == 0:
            st.success("OI data refreshed."); st.rerun()
        else:
            st.error("OI refresh failed (is the Terminal logged in?):\n\n" + (_r.stderr or _r.stdout or "no output"))
    oc2.caption(f"OI is captured **weekly** (run Mondays), separate from the daily snapshot, to keep the "
                f"Bloomberg pull light. Last OI pull: **{_to_et(_oi_asof) if _oi_asof != 'never' else 'never'}**.")

    st.divider()
    st.markdown("##### Explore a single product")
    _all_products = st.checkbox(
        "Include all products (ad-hoc)", value=False, key="oi_all",
        help="Off = the 11 fixed-income products this page focuses on. On = every product — those "
             "aren't in the weekly snapshot, so they pull live on demand (run in Bloomberg mode).")
    _fi_order = [tk for grp in FI_OI_PAGES for (tk, _s, _w) in grp["items"] if tk in INSTRUMENTS]
    _pick = _oi_order if _all_products else _fi_order
    if _pick and st.session_state.get("oi_sel") not in _pick:   # keep the selection valid as the list flips
        st.session_state["oi_sel"] = _pick[0]
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    sel = sc1.selectbox("Product", _pick,
                        format_func=lambda t: f"{INSTRUMENTS[t][0]} · {INSTRUMENTS[t][2]}", key="oi_sel")
    n_exp = int(sc2.slider("Expiry months", 4, 16, 8, key="oi_nexp"))
    _strike_view = sc3.selectbox("Strikes", ["All", 41, 31, 21, 15, 11], index=0, key="oi_nk",
                                 help="Strikes shown: All = the full chain; or a window of the N nearest spot.")
    n_k = None if _strike_view == "All" else int(_strike_view)
    spot = _spot_map.get(sel, float("nan"))

    chain = get_oi_chain(sel, n_expiries=n_exp, n_strikes=n_k)
    if chain is None or chain.empty:
        if MODE == "snapshot" and sel not in OI_SNAPSHOT_TICKERS:
            st.info(f"**{INSTRUMENTS[sel][0]}** isn't in the weekly fixed-income OI capture (the 11 core "
                    "rates products). Its chain pulls live on demand: run in **Bloomberg mode** (Terminal "
                    "open) to view it.")
        else:
            st.info("No listed-option open interest is available for this product (its options may be thin or "
                    "trade OTC). Pick another product.")
        st.stop()

    chain = chain.copy()
    chain["total"] = chain["call_oi"].fillna(0) + chain["put_oi"].fillna(0)
    _tot = float(chain["total"].sum())
    _tc, _tp = float(chain["call_oi"].sum()), float(chain["put_oi"].sum())
    _pc = (_tp / _tc) if _tc else float("nan")
    _busiest = chain.groupby("expiry_label")["total"].sum().idxmax() if len(chain) else "—"
    _peak = chain.groupby("strike")["total"].sum().idxmax() if len(chain) else float("nan")

    om1, om2, om3, om4, om5 = st.columns(5)
    om1.metric("Spot", "—" if not np.isfinite(spot) else f"{spot:g}", help="Last settlement of the underlying")
    om2.metric("Total OI", f"{_tot:,.0f}", help="Put + call open interest summed across the shown strikes & expiries")
    om3.metric("P/C (OI)", "—" if not np.isfinite(_pc) else f"{_pc:.2f}",
               help="Total put OI ÷ total call OI on this grid (>1 = put-heavy)")
    om4.metric("Busiest expiry", str(_busiest), help="Expiry month holding the most open interest")
    om5.metric("Peak strike", "—" if not np.isfinite(_peak) else f"{_peak:g}",
               help="Single strike holding the most open interest")

    _col_order = (chain[["expiry", "expiry_label"]].drop_duplicates()
                  .sort_values("expiry")["expiry_label"].tolist())
    _strike_order = sorted(chain["strike"].unique(), reverse=True)
    _mx = float(chain["total"].max()) or 1.0
    _hbase = alt.Chart(chain).encode(
        x=alt.X("expiry_label:O", sort=_col_order, title="Expiry month",
                axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=13)),
        y=alt.Y("strike:O", sort=_strike_order, title="Strike",
                axis=alt.Axis(labelFontSize=11, titleFontSize=13)))
    _rect = _hbase.mark_rect().encode(
        color=alt.Color("total:Q", scale=alt.Scale(scheme="yelloworangered"),
                        title="OI (puts+calls)", legend=alt.Legend(orient="top", titleFontSize=11)),
        tooltip=[alt.Tooltip("expiry_label:N", title="Expiry"), alt.Tooltip("strike:Q", title="Strike"),
                 alt.Tooltip("call_oi:Q", title="Call OI", format=",.0f"),
                 alt.Tooltip("put_oi:Q", title="Put OI", format=",.0f"),
                 alt.Tooltip("total:Q", title="Total OI", format=",.0f")])
    _txt = _hbase.mark_text(fontSize=10, fontWeight="bold").encode(
        text=alt.Text("total:Q", format=".2~s"),
        color=alt.condition(f"datum.total > {0.58 * _mx}", alt.value("white"), alt.value("#222")))
    brand.show_chart((_rect + _txt).properties(
        height=max(300, 24 * len(_strike_order)),
        title=f"{INSTRUMENTS[sel][0]} — open interest by strike & expiry"))
    st.caption(f"Each cell is the **total open interest** (puts + calls) at that strike and expiry; deeper red = "
               f"more open interest. Hover for the put/call split. Spot is **{spot:g}** — **Strikes = All** shows the "
               "full chain; pick a number to zoom to the N nearest spot. Large concentrations often act as pin / "
               "magnet levels into expiry. Reads the cached snapshot — no Bloomberg pull.")
    st.divider()

    st.markdown("**Per-product PDF** — the selected product's heatmap on the XP brand.")
    if st.button("📈 This product's PDF", type="primary"):
        _safe = INSTRUMENTS[sel][0].replace(" ", "_").replace("/", "-")
        _oi_render_pdf(_oi_input_frame([sel], n_exp, n_k), "single",
                       f"Open_Interest_{_safe}.pdf", f"Rendering open-interest heatmap… ({INSTRUMENTS[sel][0]})")
    # Whole-book (every product) is an AD-HOC cross-asset export — only when "all products" is on,
    # since it pulls every chain live (heavy). The page's default deliverable is the FI book above.
    if _all_products and st.button("📚 Whole-book PDF (all products · ad-hoc · pulls every chain live)"):
        _oi_render_pdf(_oi_input_frame(_oi_order, 6, 13), "book",
                       "Open_Interest_Whole_Book.pdf", "Rendering open-interest heatmaps… (whole book)")

    if st.session_state.get("oi_pdf"):
        st.download_button(
            f"⬇️ Download {st.session_state.get('oi_pdf_name', 'Open_Interest.pdf')} "
            f"({st.session_state.get('oi_pdf_mb', 0):.1f} MB)",
            data=st.session_state["oi_pdf"], file_name=st.session_state.get("oi_pdf_name", "Open_Interest.pdf"),
            mime="application/pdf")
        email_report_ui("oi_pdf", "oi_pdf", st.session_state.get("oi_pdf"),
                        subject="Open Interest Report",
                        attachment_name=st.session_state.get("oi_pdf_name", "Open_Interest.pdf"))
    st.stop()


view = _filter_signals(df[df["strategy"] == active]).copy()
if spec.get("hi") and threshold is not None and not view.empty:
    view = reflag_rows(view, threshold, spec["hi"], spec["lo"],
                       fi_yield=active in tascore.TA_STRATEGIES)

_find = st.text_input("Find a product", key=f"find_{active}", placeholder=prodsearch.PLACEHOLDER).strip()
if _find:
    view = prodsearch.filter_frame(view, universe.INSTRUMENTS, _find, ticker_col="instruments")

if view.empty:
    st.info(prodsearch.NO_MATCH.format(q=_find) if _find
            else "No opportunities flagged for this strategy yet.")
else:
    view.insert(0, "Include", view["signal"].ne("—"))
    # Sector column, derived from the instrument (a " / " pair reads as "pair"). The market
    # string already carries "· sector" for some strategies, so strip that suffix for a clean
    # display and restore the untouched market for the report input below (by index).
    view.insert(view.columns.get_loc("market") + 1, "sector",
                ["pair" if " / " in str(k)
                 else (universe.INSTRUMENTS.get(k, ("", 0.0, "", ""))[2] or "—")
                 for k in view["instruments"]])
    _orig_market = view["market"].copy()
    view["market"] = [m[:-(len(s) + 3)] if s and s != "—" and str(m).endswith(f" · {s}") else m
                      for m, s in zip(view["market"], view["sector"])]
    edited = st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", help="Tick to add to the PDF report"),
            "market": "Market",
            "sector": st.column_config.TextColumn("Sector", width="small"),
            "instruments": "Instruments",
            "signal": "Signal",
            "metric": "Metric",
            "metric_label": "Metric type",
            "level": "Level",
            "context": "Notes",
            "strategy": None,
            "direction": None,
        },
        disabled=[c for c in view.columns if c != "Include"],
    )

    chosen = edited[edited["Include"]].drop(columns=["Include", "sector"])
    chosen["market"] = _orig_market.loc[chosen.index]   # restore the exact original market for the report
    st.caption(f"**{len(chosen)}** row(s) selected for the report.")

    if st.button("Generate PDF report", type="primary", disabled=chosen.empty):
        spinner_msg = ("Rendering charts + PDF…" if active in ("Mean Reversion", "Trend")
                       else "Rendering PDF...")
        with st.spinner(spinner_msg):
            with tempfile.TemporaryDirectory() as tmp:
                out_pdf = Path(tmp) / "report.pdf"
                if active == "Mean Reversion":
                    # Chart report: spread-with-band + rebased legs per selected pair.
                    pairs_json = Path(tmp) / "pairs.json"
                    pairs_json.write_text(json.dumps(chosen["market"].tolist()), encoding="utf-8")
                    cmd = [sys.executable, str(MRREPORT_CLI), str(pairs_json), str(out_pdf),
                           "--asof", str(meta.get("as_of", "")),
                           "--threshold", str(threshold if threshold is not None else 1.5)]
                elif active == "Trend":
                    # Chart report: price + MA20/MA100 + the 3-month-return leg per selected product.
                    tickers_json = Path(tmp) / "tickers.json"
                    tickers_json.write_text(json.dumps(chosen["instruments"].tolist()), encoding="utf-8")
                    cmd = [sys.executable, str(TRENDREPORT_CLI), str(tickers_json), str(out_pdf),
                           "--asof", str(meta.get("as_of", "")),
                           "--threshold", str(threshold if threshold is not None else 0.0)]
                else:
                    rows_json = Path(tmp) / "rows.json"
                    rows_json.write_text(chosen.to_json(orient="records"), encoding="utf-8")
                    cmd = [sys.executable, str(REPORT_CLI), str(rows_json), str(out_pdf),
                           "--title", active, "--asof", str(meta.get("as_of", "")),
                           "--trigger", trigger_text]
                result = subprocess.run(cmd, capture_output=True, text=True)
                # Read the PDF BEFORE this block exits and deletes the temp folder.
                ok = result.returncode == 0 and out_pdf.exists()
                pdf_bytes = out_pdf.read_bytes() if ok else None
        if not ok:
            st.session_state.pop("pdf_bytes", None)
            st.error("PDF generation failed:\n\n" + (result.stderr or result.stdout or "no output produced"))
        else:
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_name"] = f"{active.replace(' ', '_')}_opportunities.pdf"
            st.success("Report ready.")

    # Rendered outside the click block + cached in session_state so the download
    # survives the rerun that Streamlit triggers when the button is clicked.
    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "Download PDF", data=st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_name", "report.pdf"),
            mime="application/pdf",
        )
        email_report_ui(f"tbl_{active.replace(' ', '_')}", "table", st.session_state.get("pdf_bytes"),
                        subject=f"{active} — flagged opportunities",
                        attachment_name=st.session_state.get("pdf_name", "report.pdf"))


# ─── Visual client report (Vol / Skew / Term) — controls pinned to the FOOT of the page ──────────
# Relocated here from just below the charts so every page's generate/email controls sit at the
# bottom, consistently. `active`, `meta`, `threshold`, REPORTS are all in scope in this linear flow.
if active in REPORTS:
    cfg = REPORTS[active]
    st.divider()
    st.markdown(f"##### Generate the {active} report")
    if st.button(cfg["label"], type="primary", disabled=not cfg["detail"].exists()):
        with st.spinner("Rendering charts…"):
            with tempfile.TemporaryDirectory() as tmp:
                out_pdf = Path(tmp) / "report.pdf"
                result = subprocess.run(
                    [sys.executable, str(cfg["cli"]), str(cfg["detail"]), str(out_pdf),
                     "--asof", str(meta.get("as_of", "")),
                     "--threshold", str(threshold if threshold is not None else 1.5)],
                    capture_output=True, text=True,
                )
                ok = result.returncode == 0 and out_pdf.exists()
                pdf_bytes = out_pdf.read_bytes() if ok else None
        if not ok:
            st.session_state.pop(cfg["key"], None)
            st.error(f"{active} report failed:\n\n" + (result.stderr or result.stdout or "no output"))
        else:
            st.session_state[cfg["key"]] = pdf_bytes
            st.success(f"{active} report ready.")
    if st.session_state.get(cfg["key"]):
        _pdf = st.session_state[cfg["key"]]
        st.download_button(f"⬇️ Download {active} Report (PDF)", data=_pdf,
                           file_name=cfg["file"], mime="application/pdf", key=f"{cfg['key']}_dl")

        _asof = str(meta.get("as_of", ""))[:10]
        email_report_ui(cfg["key"], cfg["key"], _pdf,
                        subject=f"{active} Report" + (f" — {_asof}" if _asof else ""),
                        attachment_name=cfg["file"],
                        intro_html=f"<p>Please find today's {active} report attached.</p>")

        # Inline preview — the actual report pages (charts + table) shown on the page.
        with st.expander("👁️  Preview the report here", expanded=True):
            try:
                for _img in _pdf_page_images(_pdf):
                    st.image(_img, use_container_width=True)
            except Exception as _e:
                st.caption(f"(Inline preview needs pypdfium2 — {_e})")


# ─── AG Fundamentals report generators — pinned to the FOOT of the page, below the flags table ────
# Moved here from above the table so the pulled data shows first, then the generators (like every
# other page). Fresh `if` block → re-import agdata and recompute the WASDE as-of.
if active == "AG Fundamentals":
    from src import agdata
    _wcal = agdata.report_calendar()
    _wpast = _wcal[(_wcal["report"] == "WASDE") & (_wcal["date"] <= pd.Timestamp.now().normalize())]
    _wasof = (_wpast["date"].max().strftime("%d %b %Y") + " WASDE") if not _wpast.empty else ""
    st.divider()
    st.markdown("##### Generate the ag reports")
    _t_wasde, _t_rx = st.tabs(["🌍 WASDE — Supply & Demand", "📊 USDA Reaction — Acreage & Grain Stocks"])

    with _t_wasde:
        st.markdown("**Monthly WASDE balance-sheet note** — US & world supply/demand and stocks-to-use, plus "
                    "month-over-month ending-stocks revisions and the trade-consensus surprise (when estimates "
                    "are loaded). Auto-emails on each release when switched on in Alert Settings.")
        if st.button("🌍 Generate WASDE Report (PDF)", type="primary", key="wasde_gen"):
            with st.spinner("Building the WASDE note from USDA PS&D…"):
                with tempfile.TemporaryDirectory() as tmp:
                    out_pdf = Path(tmp) / "wasde.pdf"
                    result = subprocess.run(
                        [sys.executable, str(WASDEREPORT_CLI), str(out_pdf), "--asof", _wasof],
                        capture_output=True, text=True,
                    )
                    ok = result.returncode == 0 and out_pdf.exists()
                    pdf_bytes = out_pdf.read_bytes() if ok else None
            if not ok:
                st.session_state.pop("wasde_pdf", None)
                st.error("WASDE report failed:\n\n" + (result.stderr or result.stdout or "no output"))
            else:
                st.session_state["wasde_pdf"] = pdf_bytes
                st.success("WASDE report ready.")
        if st.session_state.get("wasde_pdf"):
            st.download_button("⬇️ Download WASDE_Report.pdf", data=st.session_state["wasde_pdf"],
                               file_name="WASDE_Report.pdf", mime="application/pdf")
            email_report_ui("wasde_pdf", "wasde", st.session_state.get("wasde_pdf"),
                            subject="USDA WASDE — Supply & Demand", attachment_name="WASDE_Report.pdf")

    with _t_rx:
        st.markdown("**USDA Reaction note — quarterly Grain Stocks (+ June Acreage).** Stocks total with the "
                    "on-farm/off-farm split and implied quarterly use; the June release also adds planted area "
                    "vs the March intentions, wheat by class, and the acreage surprise. It **auto-detects the "
                    "latest release** and **emails itself on each quarterly print** (the scheduled task is live). "
                    "Generate or preview it on demand here.")
        c_gen, c_prev = st.columns(2)
        if c_gen.button("📊 Generate PDF", type="primary", key="rx_gen"):
            with st.spinner("Pulling the latest NASS Grain Stocks…"):
                with tempfile.TemporaryDirectory() as tmp:
                    out_pdf = Path(tmp) / "rx.pdf"
                    result = subprocess.run(
                        [sys.executable, str(USDAREACTION_CLI), str(out_pdf)],
                        capture_output=True, text=True,
                    )
                    ok = result.returncode == 0 and out_pdf.exists()
                    pdf_bytes = out_pdf.read_bytes() if ok else None
            if not ok:
                st.session_state.pop("rx_pdf", None)
                st.error("USDA Reaction note failed:\n\n" + (result.stderr or result.stdout or "no output"))
            else:
                st.session_state["rx_pdf"] = pdf_bytes
                st.success("USDA Reaction note ready.")
        if c_prev.button("🔢 Preview the numbers", key="rx_prev"):
            import json as _json
            with st.spinner("Pulling the latest NASS Grain Stocks…"):
                with tempfile.TemporaryDirectory() as tmp:
                    jp = Path(tmp) / "rx.json"
                    r = subprocess.run([sys.executable, str(USDAREACTION_CLI), "--json", str(jp)],
                                       capture_output=True, text=True)
                    st.session_state["rx_data"] = (_json.loads(jp.read_text(encoding="utf-8"))
                                                   if (r.returncode == 0 and jp.exists()) else None)
        _d = st.session_state.get("rx_data")
        if _d:
            st.caption(f"Latest report: **{_d.get('label', '')} {_d.get('year', '')}** "
                       + ("— June: full note with Acreage" if _d.get("full") else "— stocks-only"))
            if _d.get("pending"):
                st.info("June Acreage isn't released yet — showing March intentions; the actuals and the "
                        "surprise fill in on release.")
            if _d.get("full"):
                st.markdown("**Planted acreage** — vs March intentions & year-ago")
                st.dataframe(pd.DataFrame(_d["acre"])[["crop", "actual", "mar", "vs_mar", "vs_yr", "read"]],
                             hide_index=True, use_container_width=True)
                st.markdown("**Wheat by class**")
                st.dataframe(pd.DataFrame(_d["wclass"])[["crop", "actual", "vs_yr", "read"]],
                             hide_index=True, use_container_width=True)
            st.markdown(f"**{_d.get('label', '')} stocks** — total, on-farm vs off-farm")
            st.dataframe(pd.DataFrame(_d["stk"])[["crop", "total", "vs_yr", "on", "off", "read"]],
                         hide_index=True, use_container_width=True)
            if _d.get("dis"):
                st.markdown(f"**Implied {_d.get('quarter', '')} use** (prior-quarter minus this-quarter stocks)")
                st.dataframe(pd.DataFrame(_d["dis"])[["crop", "use", "vs_yr", "read"]],
                             hide_index=True, use_container_width=True)
        if st.session_state.get("rx_pdf"):
            st.download_button("⬇️ Download USDA_Reaction.pdf", data=st.session_state["rx_pdf"],
                               file_name="USDA_Reaction.pdf", mime="application/pdf")
            email_report_ui("rx_pdf", "usda_reaction", st.session_state.get("rx_pdf"),
                            subject="USDA Grain Stocks — Reaction", attachment_name="USDA_Reaction.pdf")
        st.caption("Auto-send is **live** (Task Scheduler → `usda_reaction_scheduled_email.py`): it emails the "
                   "note to the **USDA Reaction** recipients on each quarterly Grain Stocks print "
                   "(Jan / Mar / Jun / Sep).")
