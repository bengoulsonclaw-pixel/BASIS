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
from datetime import datetime, date, timedelta, time as dtime, timezone
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
                       ta_report_defaults, save_ta_report_defaults,
                       tabt_defaults, save_tabt_defaults)
from src import universe
from src import brand
from src import repcal
from src import recipients
from src import automation
from src import alerts
from src import econ
from src import gitbackup
from src import fedpath
from src import stirpaths
from src import macrorules
from src import macroradar
from src import macrosurprise
from src import macrodata
from src import volbt
from src import tabt
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
from src import eqanalyst
from src import eqcorr
from src import eqdisp
from src import curvemon
from src import seasmon
from src import brazilprod
from src import auth
from src import health
from src import compliance
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
CURVEREPORT_CLI = ROOT / "src" / "curvereport.py"
SEASREPORT_CLI = ROOT / "src" / "seasreport.py"     # Seasonality Monitor client PDF (2026-08-22)
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
    """Convert a snapshot capture time to 'H:MM AM ET · DD Mon YYYY' in New York time (DST-aware).
    Timestamps tagged with an explicit offset (the UTC pull time '…+00:00') are honored as-is;
    legacy naive 'YYYY-MM-DD HH:MM[:SS]' stamps are read as this box's local time (UTC-5)."""
    try:
        raw = str(local_str).strip()
        if not raw:
            return ""
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:                      # minute-precision stamps (signals as_of)
            dt = datetime.strptime(raw[:16], "%Y-%m-%d %H:%M")
        if dt.tzinfo is None:                   # legacy naive stamp -> machine-local (UTC-5)
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        et = dt.astimezone(ZoneInfo("America/New_York"))
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
    menu_items={"about": "BASIS — Analysis · Strategies · Indicators"},
)

# BASIS brand theme: palettes, the dark/light CSS and the primary-button label
# fix (gold tiles need a dark label) all live in src/brand.py. apply() injects
# the CSS for the active theme; the sun/moon toggle in the masthead flips it.
brand.apply()

# Per-user login gate — must run before anything else renders. Stops the script here until a
# valid session exists; everything below only ever runs for an authenticated user. See src/auth.py.
CURRENT_USER = auth.require_login()
IS_ADMIN = auth.is_admin()

# Fresh sessions land on the BASIS front door (logo + the trading-week calendar);
# set BEFORE any consumer so every later .get("active", ...) default never fires.
st.session_state.setdefault("active", "Landing")


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
    """Shared 'Email this report' block. Admins get the full recipient picker (desk/report contact
    lists, unchanged). Colleagues get a single-click send to their own logged-in address only —
    the recipient comes straight from the authenticated session, never from anything a colleague's
    session can edit, so there is no path from this UI to an arbitrary address."""
    if not pdf_bytes:
        return
    user = CURRENT_USER
    st.markdown("**Email this report**")
    if IS_ADMIN:
        to_list = _recipient_picker(state_key, recipients_key)
        c1, c2 = st.columns([1, 4])
        confirm = c1.checkbox("Confirm", key=f"{state_key}_confirm")
        clicked = c2.button("📤 Email report now", disabled=not (confirm and to_list),
                            key=f"{state_key}_send")
    else:
        to_list = [user["email"]]
        st.caption(f"Sends a copy to **{user['email']}** — your own address only.")
        on_cooldown = not auth.can_send(user["email"])
        if on_cooldown:
            st.caption("Please wait a moment before sending again.")
        clicked = st.button("📤 Generate PDF and email me", disabled=on_cooldown, key=f"{state_key}_send")
    if clicked:
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
                auth.record_send(user["email"], recipients_key)
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

    _volresponse_section()

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
    try:
        from src import owncurve as _oc
        _done, _tot = _oc.skew_backfill_progress()
        _cover = f" ({_done}/{_tot} wing-capable products carry a full year of our history)"
    except Exception:
        _cover = ""
    st.caption("Skew runs on **our own settlement-built wings** since 14 Aug 2026 — OTM put at "
               "0.90×F and OTM call at 1.10×F inverted through Black-76, interpolated to constant "
               "30d, same machinery as the Volatility/Term pages" + _cover + ". The vendor surface "
               "backstops per date, FX stays the OTC 25Δ risk reversal, and short-duration bonds "
               "stay vendor (their ±10% wings are model extrapolation at unlisted strikes — no "
               "market marks exist to invert). Switchover validation: 28 products, median corr "
               "+0.83 vs vendor, 94% sign agreement.")
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

    _skewreal_section()

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


@st.cache_data(ttl=900, show_spinner=False)
def _skewreal_table(window: int):
    from src import skewreal
    return skewreal.analyze(window)


@st.cache_data(ttl=900, show_spinner=False)
def _volresponse_table():
    from src import volmove
    return volmove.response_table()


def _volresponse_section():
    """How vol PERFORMS on a move (Ben, 2026-08-14): down/up betas, the median
    implied pop on the product's own 2σ+ days, and whether the pop holds — all
    off our own implied history + the price store, no vendor anywhere."""
    import altair as alt
    from src import volmove
    st.markdown("#### Vol response — what a move pays")
    df = _volresponse_table()
    if df is None or df.empty:
        st.caption("Needs more own-implied history — builds daily.")
        return
    show = df.assign(
        Down=df["down_beta"].map("{:+.2f}".format), Up=df["up_beta"].map("{:+.2f}".format),
        Pop=[f"{p:+.1f} ({pc:+.0f}%)" if pd.notna(p) else "—"
             for p, pc in zip(df["big_pop"], df["pop_pct"])],
        Keep=df["keep5"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—"),
    )[["market", "iv_now", "Down", "Up", "big_n", "Pop", "Keep"]].rename(columns={
        "market": "Market", "iv_now": "ATM", "Down": "Down-beta /1%", "Up": "Up-beta /1%",
        "big_n": "2σ+ days", "Pop": "Median pop (of ATM)", "Keep": "Next 5d"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)
    st.caption("**Down/Up-beta** = vol points implied gains per 1% move that way (positive = vol "
               "rises — equity vol pays on the downside, energy often on rallies). **Median pop** = "
               "the implied change on the product's own 2σ+ move days, with its size relative to "
               "the ATM level in brackets — the cross-product rank. **Next 5d** = what the pop did "
               "over the following week: positive kept building, negative bled back (gamma paid, "
               "vega didn't). Everything is computed from our own implied history against settlement "
               "prices. A cheap-vol flag on a market that responds hard here is a different "
               "conversation from cheap vol that never wakes up.")

    pick = st.selectbox("Chart a market (biggest big-day pop first)", df["market"].tolist(),
                        key="vr_pick")
    tk = df.set_index("market").loc[pick, "ticker"]
    j = volmove.move_frame(tk, 260)
    d = j.dropna(subset=["ret", "div"]).copy()
    d["side"] = np.where(d["ret"] < 0, "down day", "up day")
    cc = brand.chart_colors()
    pts = alt.Chart(d.reset_index()).mark_circle(size=30, opacity=0.5, stroke="white",
                                                 strokeWidth=0.3).encode(
        x=alt.X("ret:Q", title="daily move (%)"),
        y=alt.Y("div:Q", title="implied vol change (pts)"),
        color=alt.Color("side:N", scale=alt.Scale(domain=["down day", "up day"],
                                                  range=[cc["short"], cc["long"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("ret:Q", format="+.2f"),
                 alt.Tooltip("div:Q", format="+.2f")])
    fits = []
    for sub, lo, hi in ((d[d["ret"] < 0], float(d["ret"].min()), 0.0),
                        (d[d["ret"] > 0], 0.0, float(d["ret"].max()))):
        if len(sub) >= 25:
            g, b = np.polyfit(sub["ret"], sub["div"], 1)
            fits.append(pd.DataFrame({"ret": [lo, hi], "div": [g * lo + b, g * hi + b]}))
    fitlayer = [alt.Chart(f).mark_line(color=cc["ink"], strokeDash=[5, 3],
                                       strokeWidth=1.5).encode(x="ret:Q", y="div:Q")
                for f in fits]
    r = df[df["ticker"] == tk].iloc[0]
    st.markdown(f"**{pick}** — each dot is a day (move vs implied change); dashed = the two "
                f"half-fits. Down-beta {r['down_beta']:+.2f}, up-beta {r['up_beta']:+.2f} "
                f"vol-pts per 1%.")
    ch = pts
    for f in fitlayer:
        ch = ch + f
    brand.show_chart(ch.properties(height=360))


def _skewreal_section():
    """Skew vs the realized spot-vol path (Ben, 2026-08-14): the best-fit line
    through (underlying, ATM vol) over the window is the REALIZED skew; our own
    wings charge the implied one. The gap at ±10% moneyness is each wing's
    richness/cheapness 'on arrival' — when spot reaches the strike, the option is
    ATM and marks near the then-ATM vol."""
    import altair as alt
    from src import skewreal
    st.markdown("#### Skew vs realized path — are the wings fair?")
    _wl = st.radio("Fit window", ["3M", "6M", "1Y"], index=1, horizontal=True, key="skr_win",
                   help="How much history the best-fit line sees. Short = current regime, "
                        "long = smoother gradient. The changes-beta column is the regime "
                        "check: when it disagrees with the line's gradient, one trend "
                        "dominated the window and the verdict is marked ≈ rather than ✓.")
    win = {"3M": 63, "6M": 126, "1Y": 252}[_wl]
    df = _skewreal_table(win)
    if df is None or df.empty:
        st.caption("No products with own wing marks yet — the skew history builds daily.")
        return
    show = df.assign(
        conf=np.where(df["confident"], "✓", "≈ regime"),
        Gradient=df["g_lvl"].map("{:+.2f}".format),
        Chg=df["g_chg"].map("{:+.2f}".format),
        Put=[f"{w:.1f} vs {p:.1f} → {g:+.1f}" for w, p, g in zip(df["put_wing"], df["pred_put"], df["put_gap"])],
        Call=[f"{w:.1f} vs {p:.1f} → {g:+.1f}" for w, p, g in zip(df["call_wing"], df["pred_call"], df["call_gap"])],
    )[["market", "iv_now", "Gradient", "Chg", "r2", "Put", "Call", "conf"]].rename(columns={
        "market": "Market", "iv_now": "ATM", "Gradient": "Realized grad (per 1%)",
        "Chg": "Chg-beta", "r2": "r²", "Put": "Put wing vs line → gap",
        "Call": "Call wing vs line → gap", "conf": "Conf"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)
    st.caption("**Gap > 0 = the wing looks cheap against the realized path** (the line predicts "
               "MORE vol at that strike than the wing charges); < 0 = rich. Wings are our own "
               "settlement-built 90/110% marks; the line is fitted on our own ATM history — "
               "no vendor surface anywhere in this table. **r²** = how much of the vol's "
               "variation the price level explains (1 = dots on the line, 0 = no relationship — "
               "the gradient is noise there; ≥0.5 solid, <0.2 ignore the row). ✓ = the "
               "daily-changes beta agrees with the line's gradient (sign and within 2×); "
               "≈ = one trending regime dominated the window — read the gap with care.")

    pick = st.selectbox("Chart a market (largest wing gap first)", df["market"].tolist(), key="skr_pick")
    tk = df.set_index("market").loc[pick, "ticker"]
    j, fit = skewreal.scatter_frame(tk, win)
    if j is None:
        st.caption("Not enough joined history for this product at this window.")
        return
    cc = brand.chart_colors()
    pts = alt.Chart(j.reset_index()).mark_circle(size=34, color=cc["series"], opacity=0.45,
                                                 stroke="white", strokeWidth=0.3).encode(
        x=alt.X("px:Q", title="underlying", scale=alt.Scale(zero=False)),
        y=alt.Y("iv:Q", title="ATM vol (%)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("px:Q", format=".2f"),
                 alt.Tooltip("iv:Q", format=".1f")])
    line = alt.Chart(fit["line"]).mark_line(color=cc["ink"], strokeDash=[5, 3],
                                            strokeWidth=1.6).encode(x="px:Q", y="iv:Q")
    wings = alt.Chart(fit["wings"]).mark_point(shape="diamond", size=170, filled=True,
                                               color=cc["accent"], stroke="white",
                                               strokeWidth=0.8).encode(
        x="px:Q", y="iv:Q", tooltip=["kind:N", alt.Tooltip("iv:Q", format=".1f")])
    preds = alt.Chart(fit["preds"]).mark_point(shape="circle", size=120, filled=False,
                                               color=cc["ink"], strokeWidth=1.6).encode(
        x="px:Q", y="iv:Q", tooltip=["kind:N", alt.Tooltip("iv:Q", format=".1f")])
    now = alt.Chart(fit["now"]).mark_point(shape="cross", size=200, filled=True,
                                           color=cc["short"], stroke="white",
                                           strokeWidth=0.8).encode(
        x="px:Q", y="iv:Q",
        tooltip=["kind:N", alt.Tooltip("px:Q", format=".2f"), alt.Tooltip("iv:Q", format=".1f")])
    ch = pts + line + wings + preds + now
    smile_note = ""
    if "smile" in fit:
        smile = alt.Chart(fit["smile"]).mark_line(color=cc["accent"], strokeWidth=2.2,
                                                  opacity=0.95).encode(x="px:Q", y="iv:Q")
        ch = ch + smile
        if "farpts" in fit:
            farp = alt.Chart(fit["farpts"]).mark_point(shape="diamond", size=110, filled=True,
                                                       color=cc["accent"], stroke="white",
                                                       strokeWidth=0.6, opacity=0.8).encode(
                x="px:Q", y="iv:Q",
                tooltip=["kind:N", alt.Tooltip("iv:Q", format=".1f")])
            ch = ch + farp
        _sm = fit["smile_params"]
        _kind = ("a quartic drawn exactly through all five marks" if _sm.get("method") == "quartic"
                 else f"an exact fit through the {_sm['n_marks']} available marks"
                 + (" (shape-guarded: a quartic here would roll a tail over, so a monotone "
                    "exact fit is used)" if _sm.get("method") == "monotone" else ""))
        smile_note = (f" **Solid gold curve = today's smile** — {_kind}; tails only ever rise; "
                      "small diamonds = the 80/120% marks. Where the smile sits below the dashed "
                      "line, options at that strike are cheap against the realized path; above = rich.")
    st.markdown(f"**{pick}** — dots = the last {_wl} of (underlying, ATM vol); dashed = best fit "
                "(the realized skew); **red cross = today's ATM strike** (spot, our ATM vol); "
                "**gold diamonds = our wing marks at ±10%**; hollow circles = where the line says "
                "ATM vol trades at those strikes." + smile_note)
    brand.show_chart(ch.properties(height=380))
    r = df[df["ticker"] == tk].iloc[0]
    side = "call" if abs(r["call_gap"]) >= abs(r["put_gap"]) else "put"
    gap = r[f"{side}_gap"]
    st.caption(f"Verdict: the **{side} wing** is marked {r[f'{side}_wing']:.1f} where the realized "
               f"path predicts {r[f'pred_{side}']:.1f} — **{'cheap' if gap > 0 else 'rich'} by "
               f"{abs(gap):.1f} vols on arrival** (gradient {r['g_lvl']:+.2f}/1%, changes-beta "
               f"{r['g_chg']:+.2f}, r² {r['r2']:.2f}"
               + (", regime-flagged — treat as indicative" if not r["confident"] else "") + ").")


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
                           ("🧭 Macro Compass", "Macro Compass")],
    "STIR Paths":         [("🗓️ Rates Home", "STIR Timeline"),
                           ("Fed", "Fed Path"),      # flags ride in via CSS ::before
                           ("ECB", "ECB Path"),      # (_STIR_TAB_FLAG_CSS) — emoji here
                           ("BoE", "BoE Path")],     # would double up with them
    # The old Trade Testing module dissolved (Ben 2026-08-15): TA Backtester + Signal
    # Ledger live under Technical Analysis, the Vol Backtester under Volatility.
    "Technical Analysis": [("📈 TA Hub", "Technical Analysis"),
                           ("🎯 TA Backtester", "TA Backtester"),
                           ("📒 Signal Ledger", "Signal Ledger")],
    # Equities mirrors it (Ben 2026-08-15): the eq TA Backtester rides the eq TA module
    # (the equities Signal Ledger is embedded at the foot of the TA hub page itself).
    "Equities TA":        [("📈 TA Hub", "eq:Technical Analysis"),
                           ("🎯 TA Backtester", "eq:TA Backtester"),
                           ("📒 Signal Ledger", "eq:Signal Ledger")],
    "Volatility":         [(s, s) for s in NAV_GROUPS["Volatility"]]
                          + [("🧪 Vol Backtester", "Vol Backtester")],
    "Positioning & Flow": [(s, s) for s in NAV_GROUPS["Positioning & Flow"]],
    "Fundamentals":       [("AG Fundamentals", "AG Fundamentals"),
                           ("🛢️ OPEC Report", "OPEC Report"),
                           ("🥇 Precious Metals", "Precious Metals"),
                           ("🥇 Gold Engine", "Gold Engine"),
                           ("🇧🇷 Brazil Production", "Brazil Production")],
    "Seasonality":        [("📅 Product Seasonality", "Seasonality"),
                           ("🔀 Spread Seasonality", "Seasonality Spreads")],
}
_TAB_MEMBERS_OF = {dest: members for members in _GROUP_TABS.values() for _lbl, dest in members}


def _render_group_tabs(active_page: str) -> None:
    """If `active_page` belongs to a collapsed sidebar group, render its tab-row switcher (the active
    tab highlighted). No-op for any page that isn't part of a collapsed group."""
    members = _TAB_MEMBERS_OF.get(active_page)
    if not members:
        return
    if not IS_ADMIN:
        members = [m for m in members if m[1] not in _ADMIN_ONLY_DESTS]
    cols = st.columns(len(members))
    for col, (label, dest) in zip(cols, members):
        col.button(label, key=f"gtab_{dest}", use_container_width=True, on_click=_go, args=(dest,),
                   type="primary" if dest == active_page else "secondary")
    if any(dest in ("Fed Path", "ECB Path", "BoE Path") for _l, dest in members):
        st.markdown(f"<style>{_STIR_TAB_FLAG_CSS}</style>", unsafe_allow_html=True)


def _data_badge(snap, side: str = "FICC") -> None:
    """Compact, always-visible data-source status for the sidebar. Healthy states render as
    a subtle caption (same voice as "Signals as of" below); only problem states — demo /
    missing data — keep the loud warning box. On the Equities desk the badge shows the
    EQUITIES pull stamp (manifest `equities_pulled`), not the FICC snapshot's."""
    if side == "Equities":
        _running, _started = _eq_pull_running()
        if _running:
            st.caption(f"⏳ Equities: **pull running** — started {_started}")
        else:
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
        st.caption("No live overnight quote captured yet — click **Pull Bloomberg snapshot** "
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
    # Weather sits BESIDE the clock (Ben, 2026-08-13) — the full-width cells leave
    # plenty of room right of the time; on the caption row it crowded the city name.
    cells = "".join(
        '<div class="c"><div class="city">'
        '<span class="nm">' + c["name"].upper() + '</span></div>'
        '<div class="tr"><div class="time" data-tz="' + c["tz"] + '">--:--:--</div>'
        '<span class="wx"><span class="wi">' + (w.get("icon") or "") + '</span>'
        + ('<span class="tmp">' + str(w["temp"]) + '&#176;</span>'
           if w.get("temp") is not None else "")
        + '</span></div></div>'
        for c, w in zip(worldclock.CITIES, wx))
    html = (
        "<meta charset='utf-8'><style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        # top padding = the visible gap between the BASIS masthead and the clock strip
        # text-size-adjust: mobile Chrome "font boosting" silently inflates small text
        # inside narrow blocks — it was scaling the city labels ~25% past what the CSS
        # asks for, which is half of why the rail overlapped itself on a phone.
        "html{-webkit-text-size-adjust:100%;text-size-adjust:100%}"
        "body{background:transparent;font-family:" + mono + ";padding-top:8px}"
        ".row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));"
        "width:100%;"       # full-bleed strip: half-width fit-content read as clutter
        "background:" + pal["surface2"] + ";border:1px solid " + pal["border"] + "}"
        ".c{padding:7px 12px;min-width:0;border-right:1px solid " + pal["border"] + "}"
        ".c:last-child{border-right:none}"
        ".city{display:flex;align-items:center;font-size:12px;"
        "letter-spacing:.12em;color:" + faint + "}"
        ".city .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".tr{display:flex;align-items:center;gap:10px}"
        ".wx{display:flex;align-items:center;gap:4px;flex:0 0 auto}"
        ".wi{display:flex;flex:0 0 auto}.wi svg{width:17px;height:17px}"
        ".tmp{letter-spacing:0;font-size:11.5px;"
        "font-variant-numeric:tabular-nums;color:" + faint + "}"
        ".time{font-size:17px;font-weight:500;color:" + pal["text"] +
        ";font-variant-numeric:tabular-nums}"
        # phones: six cells across a ~320px rail left the times overlapping each other
        # and the city names as "CHI…". Three columns over two rows fits, with the
        # weather moved to the cell's top-right corner (beside a clock it no longer
        # fits) — fit() below grows the frame to the wrapped rail. MUST stay last:
        # these are same-specificity overrides of the rules above.
        "@media (max-width:760px){"
        ".row{grid-template-columns:repeat(3,minmax(0,1fr))}"
        ".c{padding:5px 8px;position:relative;overflow:hidden;"
        "border-bottom:1px solid " + pal["border"] + "}"
        ".c:nth-child(3n){border-right:none}"
        ".c:nth-child(n+4){border-bottom:none}"
        # city set tight enough that SINGAPORE/SÃO PAULO print in full beside the temp
        ".city{font-size:9.5px;letter-spacing:.02em;padding-right:31px}"
        ".time{font-size:15px}"
        ".wx{position:absolute;top:4px;right:5px;gap:2px}"
        ".wi svg{width:11px;height:11px}.tmp{font-size:9px}}"
        "</style><div class='row'>" + cells + "</div>"
        "<script>function t(){document.querySelectorAll('.time').forEach(function(e){"
        "e.textContent=new Intl.DateTimeFormat('en-GB',{timeZone:e.dataset.tz,"
        "hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date());});}"
        "t();setInterval(t,1000);"
        # tick the day-timelines' gold now-line LIVE (ET + machine-local labels) —
        # the timeline itself is static HTML, so the stamped time went stale in minutes
        "function nl(){try{var d=window.parent.document,n=new Date();"
        "var et=new Intl.DateTimeFormat('en-GB',{timeZone:'America/New_York',"
        "hour:'2-digit',minute:'2-digit',hour12:false}).format(n);"
        "var lo=new Intl.DateTimeFormat('en-GB',{hour:'2-digit',minute:'2-digit',"
        "hour12:false}).format(n);"
        "d.querySelectorAll('.nowt-et').forEach(function(e){e.textContent=et+' ET';});"
        "d.querySelectorAll('.nowt-loc').forEach(function(e){e.textContent=lo+' local';});"
        "}catch(e){}}nl();setInterval(nl,1000);"
        # keep the fixed top bar clear of the (drag-resizable) sidebar: mirror the
        # sidebar's live width into --basis-topbar-left on the parent document
        "function sb(){try{var d=window.parent.document;"
        "var s=d.querySelector('section[data-testid=\"stSidebar\"]');"
        # floor at 48px: a collapsed sidebar can stay mounted at ~0 width, and the
        # expand arrow needs that corner visible/clickable either way
        "var w=s?Math.round(s.getBoundingClientRect().width):0;"
        "if(w<48)w=48;"
        # on a phone the sidebar OVERLAYS the page (it is ~70% of the screen); shifting
        # the bar by its width squeezed the clocks into a sliver whenever the nav was open
        "var vw=d.documentElement.clientWidth;if(vw<820)w=48;"
        # hand the viewport width back to PYTHON (brand.viewport_width): charts are
        # drawn server-side, and things CSS can't reach — a legend laid out in one
        # 570px row, axis label sizes — have to be decided before the spec is built.
        # st.context.cookies is read from the CONNECTING request, so a cookie set
        # mid-session isn't visible until the next load: on the first ever visit in
        # a browser (no cookie yet) reload once, guarded by sessionStorage so a
        # blocked-cookie browser can never loop. Every later visit is a plain render.
        "var had=/(^|;\\s*)basis_vw=/.test(d.cookie);"
        "d.cookie='basis_vw='+vw+';path=/;max-age=31536000;SameSite=Lax';"
        "var W=window.parent;"
        "if(!had&&!W.sessionStorage.getItem('basis_vw_done')){"
        "W.sessionStorage.setItem('basis_vw_done','1');W.location.reload();}"
        "d.documentElement.style.setProperty('--basis-topbar-left',w+'px');"
        # ...and the bar's live HEIGHT into --basis-topbar-h. The 62px iframe is
        # right on a desktop, but on a phone the rail wraps to two rows and the
        # bar's own rows stack — a hard-coded .block-container clearance then hid
        # the top of the page under the bar. Measure, don't guess.
        "var k=d.querySelector('.st-key-basis_topbar'),n=k,fx=null;"
        "while(n){if(d.defaultView.getComputedStyle(n).position==='fixed'){fx=n;break;}"
        "n=n.parentElement;}"
        "if(fx)d.documentElement.style.setProperty('--basis-topbar-h',"
        "Math.round(fx.getBoundingClientRect().height)+'px');}catch(e){}}"
        # grow OUR iframe to the wrapped rail's real height (Streamlit stamps the
        # fixed 62px; re-applied on every tick so a rerun can't clip the rail)
        # (Streamlit sizes the iframe's WRAPPER blocks to the 62px it was told about,
        # so the taller frame just overflowed the bar — the wrappers get the height too,
        # up to the fixed bar itself, which is left to grow on its own)
        "function fit(){try{var f=window.frameElement;if(!f)return;"
        "var h=Math.ceil(document.querySelector('.row').getBoundingClientRect().height)+10;"
        "if(Math.abs(f.getBoundingClientRect().height-h)>1){"
        "f.style.height=h+'px';f.setAttribute('height',h);}"
        "var n=f.parentElement,i=0,d=window.parent.document;"
        "while(n&&i<6&&d.defaultView.getComputedStyle(n).position!=='fixed'){"
        "if(n.style.minHeight!==h+'px')n.style.minHeight=h+'px';n=n.parentElement;i++;}"
        "}catch(e){}}"
        # phone only: tapping a module in the (overlaying, near-full-screen) sidebar
        # left the nav sitting on top of the page you just opened — collapse it once
        # the click has been handed to Streamlit. Bound once; desktop is untouched.
        "function auto(){try{var d=window.parent.document;"
        "if(d.documentElement.clientWidth>=820)return;"
        "var u=d.querySelector('[data-testid=\"stSidebarUserContent\"]');"
        "if(!u||u.dataset.basisAuto)return;u.dataset.basisAuto='1';"
        "u.addEventListener('click',function(ev){"
        "if(!ev.target.closest('button'))return;setTimeout(function(){"
        "var c=d.querySelector('[data-testid=\"stSidebarCollapseButton\"] button')"
        "||d.querySelector('[data-testid=\"stSidebarCollapseButton\"]');"
        "if(c)c.click();},150);},true);}catch(e){}}"
        "fit();sb();auto();setInterval(function(){fit();sb();auto();},500);"
        "window.addEventListener('resize',function(){fit();sb();});</script>")
    # 62 = 8px masthead gap + the rail's real rendered height (53px content + 1px
    # slack) — anything taller leaves a dead dark band under the clocks. On narrow
    # viewports fit() above grows the frame to the two-row rail.
    components.html(html, height=62)




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
                   "(**Pull Bloomberg snapshot**, or a Morning Coffee run).")
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
    d_off_a, d_off_t = universe.default_off()
    n_def = len({tk for tk in INSTRUMENTS
                 if INSTRUMENTS[tk][2] not in d_off_a and tk not in d_off_t})
    # design card header: summary + startup default live in the header sub-line now
    st.markdown(f'<div class="dk-h"><span class="dk-t">Sectors &amp; products</span>'
                f'<span class="dk-s">{len(on)}/{len(INSTRUMENTS)} instruments on · '
                f'startup default {n_def}/{len(INSTRUMENTS)}</span></div>',
                unsafe_allow_html=True)
    if not IS_ADMIN:
        # This filter is one shared file (data/sector_filter.json) read by every session, not a
        # per-user preference -- a colleague toggling it would change what everyone else sees, so
        # it's admin-only. Colleagues just see the current selection.
        st.caption("Set by your admin -- controls which markets show up across every report page.")
        return
    with st.container():
        st.caption("Hit a group to switch the whole sector on or off. Open its dropdown to drill in "
                   "by region / asset class and toggle individual contracts. 📌 saves the current "
                   "selection as the startup default.")
        # one wrapping row of chips (design): All · None · sector chips · Set default
        with st.container(key="sf_chiprow"):
            groups = [g[0] for g in _FILTER_GROUPS]
            if st.button("All", key="sf_b_all"):
                _sf_apply(lambda s: s[2])
            if st.button("None", key="sf_b_none"):
                _sf_apply(lambda s: [])
            for group in groups:
                gtks = {tk for s in secs if s[0] == group for tk in s[2]}
                n_on = len(gtks & on)
                if st.button(f"{group} · {n_on}/{len(gtks)}", key=f"sf_b_{group}",
                             type="primary" if n_on else "secondary"):
                    _sf_apply(lambda s, _g=group, _full=(n_on == len(gtks)):
                              (([] if _full else s[2]) if s[0] == _g
                               else st.session_state.get(s[3], s[2])))
            if st.button("📌 Set default", key="sf_b_setdef",
                         help="Save the current selection as the startup default — the app loads "
                              "this on every launch until you set it again."):
                universe.save_default(*_sf_current_off())
                st.toast("Saved — the dashboard will start with this selection from now on.",
                         icon="📌")

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
      + '<div style="color:#B71C1C;font-weight:800;font-size:12px;letter-spacing:1.5px;margin-top:8px">'+(r.hdr||'REPORT RELEASED')+'</div>'
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
    try:
        for d in stirpaths.decisions_today(today):
            out.append({"name": d["name"], "icon": d["icon"], "color": "#0D47A1",
                        "t": d["t"], "decision": True})
    except Exception:
        pass
    for o in out:
        hh, mm = map(int, o["t"].split(":"))
        o["fire_ms"] = int(datetime(today.year, today.month, today.day, hh, mm, tzinfo=et).timestamp() * 1000)
        o["key"] = alerts.key_for_release(o["name"])       # for the per-report banner/popup toggles
    return out


def render_report_banner() -> None:
    """Heads-up strips at the top of Home: red on days a fundamental report releases, navy on
    central-bank decision days (each gated by its own toggle in Alert Settings)."""
    rels = [r for r in _todays_releases() if alerts.alert_enabled(r.get("key"), "banner")]
    reports = [r for r in rels if not r.get("decision")]
    decisions = [r for r in rels if r.get("decision")]
    if reports:
        items = " &nbsp;&middot;&nbsp; ".join(f"{r['icon']} <b>{r['name']}</b> {r['t']} ET"
                                              for r in reports)
        st.markdown(
            "<div style='background:linear-gradient(90deg,#B71C1C,#E53935);color:#fff;padding:11px 16px;"
            "border-radius:9px;margin:0 0 14px;font-size:15px;border:1px solid #7f0000;"
            "box-shadow:0 2px 8px rgba(0,0,0,.28)'>&#128308; <b>REPORT DAY</b> &mdash; releasing today: "
            + items + ". <span style='opacity:.9'>A full-screen alert pops at release time.</span></div>",
            unsafe_allow_html=True)
    if decisions:
        items = " &nbsp;&middot;&nbsp; ".join(f"{r['icon']} <b>{r['name']}</b> {r['t']} ET"
                                              for r in decisions)
        st.markdown(
            "<div style='background:linear-gradient(90deg,#0D2B5E,#1565C0);color:#fff;padding:11px 16px;"
            "border-radius:9px;margin:0 0 14px;font-size:15px;border:1px solid #082044;"
            "box-shadow:0 2px 8px rgba(0,0,0,.28)'>&#127963;&#65039; <b>DECISION DAY</b> &mdash; "
            "announcing today: " + items
            + ". <span style='opacity:.9'>A full-screen alert pops at decision time.</span></div>",
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
                            "icon": r["icon"], "t": r["t"], "fire": r["fire_ms"],
                            "hdr": "RATE DECISION" if r.get("decision") else "REPORT RELEASED"}
                           for r in rels])
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


def _render_skew_backfill_banner() -> None:
    """One-time green Home banner the morning the own-skew backfill drip completes
    (Ben asked for an in-app notification, 2026-08-08). Dismiss persists to disk so
    it never nags; fails silent — a data hiccup must not block the Home page."""
    ack = SNAPSHOT_DIR.parent / "skew_backfill_ack.json"
    if ack.exists():
        return
    try:
        from src import owncurve
        done, total = owncurve.skew_backfill_progress()
        remaining = owncurve.skew_backfill_remaining()
    except Exception:
        return
    if total == 0 or remaining > 0:
        return
    c1, c2 = st.columns([5.2, 0.8])
    c1.success(f"🎉 **Own-skew backfill finished** — {done} of {total} wing-capable products "
               "carry a full year of our settlement-built skew history"
               + ("" if done == total else
                  f"; the other {total - done} (quarterly expiries / sparse wings) reconstructed "
                  "all their listed marks allow")
               + ". The Skew page now runs on our wings (switched 14 Aug 2026, validation on the "
               "page caption); stragglers keep accruing daily.")
    if c2.button("Dismiss", key="skew_backfill_ack", use_container_width=True):
        try:
            ack.write_text('{"acknowledged": true}')
        except Exception:
            pass
        st.rerun()


def _render_cb_calendar_banner() -> None:
    """Amber Home strip when a central-bank meeting calendar is running thin — same
    9-month rule as the Data-health board's 'CB calendars' line. The STIR Paths
    *_DECISIONS lists are hand-extended when the banks publish new years, so this is
    the front-page nudge; Snooze parks it for 30 days and it re-arms until the lists
    are actually extended (the snooze self-clears once they are). Fails silent — a
    hiccup here must never block the Home page."""
    ack_p = SNAPSHOT_DIR.parent / "cb_calendar_ack.json"
    try:
        cal = health.meeting_calendar_runway()
        thin = cal[cal["months_left"] < health.CB_CAL_MIN_MONTHS]
    except Exception:
        return
    if thin.empty:
        try:
            ack_p.unlink(missing_ok=True)              # extended → reset the snooze
        except Exception:
            pass
        return
    try:
        snoozed = json.loads(ack_p.read_text(encoding="utf-8")).get("when")
        if snoozed and (pd.Timestamp.now() - pd.Timestamp(snoozed)).days < 30:
            return
    except Exception:
        pass
    items = " · ".join(f"**{r.bank}** ends {r.last_meeting:%b %Y} (~{r.months_left:.0f}mo)"
                       for r in thin.itertuples(index=False))
    c1, c2 = st.columns([5.2, 0.8])
    c1.warning("🗓️ **Central-bank meeting calendar running thin** — " + items
               + ". The decision dates behind STIR Paths are a hand-kept list; ask for the "
                 "newly published year to be appended (src/fedpath.py / src/stirpaths.py) "
                 "or the path tools go blind past the last date.")
    if c2.button("Snooze 30d", key="cb_cal_ack", use_container_width=True):
        try:
            ack_p.write_text(json.dumps({"when": pd.Timestamp.now().isoformat()}),
                             encoding="utf-8")
        except Exception:
            pass
        st.rerun()


def render_weekly_review() -> None:
    """🗞️ Weekly Review — build/preview the cross-module Monday wrap (src/weekreview.py).
    Ad-hoc builds here never roll the weekly-delta baseline (the scheduled Monday email
    passes --baseline), so kicking the tyres never eats the week's 'new this week' tags."""
    st.subheader("🗞️ Weekly Review")
    st.caption("The exception-based Monday wrap: one line per flag past its home module's own "
               "live threshold — vol, curve/RV, positioning, technicals, correlation breaks, "
               "seasonal windows and metals flows — with the technical scorecard folded in and "
               "the release calendar. The review adds no analysis of its own. Also available as "
               "a scheduled Monday email (Recipients → Alert settings).")
    b1, b2 = st.columns([1, 3])
    _ai = b2.toggle("AI-polish the intro (desk voice; template fallback offline)",
                    value=True, key="wr_ai")
    if b1.button("🗞️ Build the Weekly Review (PDF)", type="primary", key="wr_build"):
        with st.spinner("Collecting the week's flags from every module…"):
            _out = ROOT / "data" / "Weekly_Review.pdf"
            cmd = [sys.executable, str(ROOT / "src" / "weekreview.py"), str(_out),
                   "--asof", datetime.now().strftime("%d %b %Y")]
            if not _ai:
                cmd.append("--no-ai")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and _out.exists():
                st.session_state["wr_pdf"] = _out.read_bytes()
            else:
                st.session_state["wr_pdf"] = None
                st.error("Weekly Review failed:\n\n" + (r.stderr or r.stdout or "no output")[-2000:])
    if st.session_state.get("wr_pdf"):
        st.download_button("⬇️ Download Weekly_Review.pdf", data=st.session_state["wr_pdf"],
                           file_name="Weekly_Review.pdf", mime="application/pdf", key="wr_dl")
        email_report_ui("wr_email", "weekreview", st.session_state["wr_pdf"],
                        subject="BASIS Weekly Review",
                        attachment_name="Weekly_Review.pdf")
        try:
            for img in _pdf_page_images(st.session_state["wr_pdf"]):
                st.image(img, use_container_width=True)
        except Exception as _e:
            st.caption(f"(Inline preview needs pypdfium2 — {_e})")


def _blp_block_probe() -> str | None:
    """One raw refdata request straight at the Terminal — the ONLY reliable way to
    tell 'not logged in' from the -4002 workflow-review block (they look identical
    from a failed pull: both return no data, and the block's signature doesn't
    always reach the pull's stderr — learned 2026-08-14, nid:19435).
    Returns the block's nid if -4002 is active, '' if data flows, None if the API
    isn't answering at all (Terminal closed / logged out / login on another PC)."""
    try:
        import blpapi
        opts = blpapi.SessionOptions()
        opts.setServerHost("localhost")
        opts.setServerPort(8194)
        s = blpapi.Session(opts)
        if not s.start():
            return None
        try:
            if not s.openService("//blp/refdata"):
                return None
            svc = s.getService("//blp/refdata")
            req = svc.createRequest("ReferenceDataRequest")
            req.getElement("securities").appendValue("CLA Comdty")
            req.getElement("fields").appendValue("PX_LAST")
            s.sendRequest(req)
            while True:
                ev = s.nextEvent(8000)
                if ev.eventType() == blpapi.Event.TIMEOUT:
                    return None
                for msg in ev:
                    t = str(msg)
                    if "WORKFLOW_REVIEW_NEEDED" in t:
                        m = re.search(r"nid:(\d+)", t)
                        return m.group(1) if m else "unknown"
                    if "PX_LAST" in t:
                        return ""
                if ev.eventType() == blpapi.Event.RESPONSE:
                    return ""
        finally:
            s.stop()
    except Exception:
        return None


def _block_error(nid: str) -> None:
    st.error(
        "🚫 **Bloomberg has blocked this account's API usage pending a workflow "
        f"review** (error -4002, nid:{nid}). The Terminal is fine — every API "
        "request is rejected until Bloomberg lifts the block.\n\n"
        "**What to do:** call the Help Desk (HELP HELP in the Terminal) and ask them "
        "to clear the workflow review, quoting the code and nid above. "
        "The existing snapshot was **kept** — nothing was overwritten.")


def _explain_fetch_failure(out: str, err: str) -> None:
    """Plain-English diagnosis of a failed Bloomberg fetch — the signatures were
    learned from the real production failures of Aug 2026: the -4002 workflow-review
    block, the Terminal session dying mid-pull (login closed / moved to another PC),
    a Terminal that was never serving, and the engine wedging silently (stall cap).
    The raw log lands in an expander so the full detail is one click away."""
    text = ((err or "") + "\n" + (out or "")).strip()
    nid = re.search(r"nid:(\d+)", text)
    n_fail = text.count("PyEngine: request failed")
    n_badsec = text.count('category="BAD_SEC"')
    kept = "The existing snapshot was **kept** — nothing was overwritten."
    if "WORKFLOW_REVIEW_NEEDED" in text or "-4002" in text:
        st.error(
            "🚫 **Bloomberg has blocked this account's API usage pending a workflow "
            "review** (error -4002" + (f", nid:{nid.group(1)}" if nid else "") + "). "
            "Every data request is rejected until Bloomberg lifts the block — keeping "
            "the Terminal open does not help.\n\n"
            "**What to do:** call the Help Desk (HELP HELP in the Terminal) and ask them "
            "to clear the workflow review, quoting the code and nid above. " + kept)
    elif "[BASIS] fetch stalled" in text:
        st.error(
            "⏱️ **The pull stalled and was stopped automatically** — it stopped making "
            "progress mid-run. Most often the Bloomberg engine lost one request and "
            "waited forever (its requests have no timeout) — **the Terminal is usually "
            "fine** and an immediate retry succeeds. The other cause is the session "
            "dying underneath it (Terminal closed/logged out, login moved to another "
            "PC).\n\n"
            "**What to do:** pull again — a pull can't resume, it restarts and takes "
            "~10–15 min. If the retry ALSO stalls, then check the Terminal login on "
            "this machine. " + kept)
    elif n_fail >= 8:
        st.error(
            "🔌 **The Bloomberg connection died mid-pull** — requests were flowing and "
            f"then {n_fail} in a row failed. That is what it looks like when the "
            "Terminal is closed, logs out, or the login moves to another PC while a "
            "pull is running.\n\n"
            "**What to do:** log the Terminal in on THIS machine and keep it open for "
            "the whole Bloomberg phase (~10–15 min), then pull again. " + kept)
    elif "NO price data" in text:
        # empty pull = EITHER not-logged-in OR the -4002 block; only a live raw
        # request can tell them apart (the block hides from the pull's own log)
        probe = _blp_block_probe()
        if probe:
            _block_error(probe)
        elif probe == "":
            st.error(
                "🖥️ **Bloomberg returned no data during the pull, but is answering "
                "now** — most likely the Terminal was logged in after the pull had "
                "already started (or briefly dropped).\n\n"
                "**What to do:** just pull again — it should work now. " + kept)
        else:
            st.error(
                "🖥️ **Bloomberg isn't answering on this machine** — the Terminal is "
                "closed, not logged in, or the login is active on another PC.\n\n"
                "**What to do:** open the Terminal on THIS machine, log in, and pull "
                "again. " + kept)
    else:
        st.error("Snapshot fetch failed — the log below has the detail. " + kept)
    if n_badsec:
        st.caption(f"The log also notes **{n_badsec} expired/invalid option strikes** "
                   "being skipped — routine housekeeping on every pull, not the cause.")
    with st.expander("Technical log"):
        st.code(text[-4000:] or "no output", language="text")


@st.cache_data(ttl=900, show_spinner=False)
def _landing_eq_events() -> list:
    """The landing page's earnings chips — FILE-backed since 2026-08-20 (Ben: the
    homepage 'takes ages each time'): the daily equities pull pre-writes
    data/equities/earnings_events.json, and reading it back is milliseconds and
    survives server restarts (in-process caches don't). The ~30s fundamentals
    compute only ever runs as a fallback when the store is missing, and then
    writes it for everyone after."""
    from src import eqearncal
    ev = eqearncal.read_events_store()
    if ev is not None:
        return ev
    eqearncal.refresh_events_store()
    return eqearncal.read_events_store() or []


_MACRO_FLAGS = {"USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵", "CAD": "🇨🇦",
                "AUD": "🇦🇺", "CHF": "🇨🇭", "CNY": "🇨🇳", "NZD": "🇳🇿"}


@st.cache_data(ttl=900, show_spinner=False)
def _landing_macro(day_iso: str) -> list:
    """High-impact economic prints (CPI, PPI, NFP… — the ECO-page majors) for one
    day as landing-board events. Short TTL: the feed fills in `actual` as figures
    print. The free feed only covers the current week — other days return []."""
    rows = econ.fetch_day(date.fromisoformat(day_iso))
    ev = []
    for r in rows:
        tip = f"{r['title']} ({r['country']})"
        if r["actual"]:
            tip += f" · actual {r['actual']}"
        if r["forecast"]:
            tip += f" · fcst {r['forecast']}"
        if r["previous"]:
            tip += f" · prev {r['previous']}"
        lbl = f"{r['country']} {r['title']}"
        if r["actual"]:                 # the print is IN — show it on the chip itself
            lbl += f" · {r['actual']}" + (f" (vs {r['forecast']}e)" if r["forecast"] else "")
        ev.append({"date": date.fromisoformat(day_iso),
                   "icon": _MACRO_FLAGS.get(r["country"], "📊"),
                   "label": lbl, "color": "#546E7A",
                   "auto": False, "tip": tip, "time": r["time_et"]})
    return ev


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _earnings_times(bbg_tickers: tuple) -> dict:
    """{bloomberg ticker: '06:30 ET' | None} via Yahoo for the day's reporters —
    Bloomberg's field is date-only, Yahoo's earnings timestamp carries the hour."""
    from src import yfin
    return {b: yfin.earnings_time_et(b) for b in bbg_tickers}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _landing_expiries(day_iso: str) -> list:
    """Products whose NEXT futures/options expiry lands ON `day` — the ⏳ rows of the
    landing FICC panel. Groups big shared expiries (index option Fridays…) into one
    row per sector so twenty products don't become twenty rows. Pure calendar maths
    (src/expiries) — zero Bloomberg."""
    from src import expiries as _exp
    d = date.fromisoformat(day_iso)
    key = f"{d:%a %d %b %Y}"                        # expiries._fmt_date format
    bucket: dict = {}
    seen: set = set()      # indices live in the universe TWICE (futures generic +
    for t in INSTRUMENTS:  # cash ticker, same display name) — one expiry row each
        name, asset = INSTRUMENTS[t][0], INSTRUMENTS[t][2]
        try:
            ex = _exp.describe(t, asset, d)
        except Exception:
            continue
        for kind, dkey, tkey in (("future", "fut", "fut_time"), ("options", "opt", "opt_time")):
            if ex.get(dkey) == key and (name, kind) not in seen:
                seen.add((name, kind))
                bucket.setdefault((asset, kind, ex.get(tkey) or ""), []).append(name)
    ev = []
    for (asset, kind, tm), names in bucket.items():
        col = markethours.ASSET_COLORS.get(asset, "#616161")
        base = {"date": d, "icon": "⏳", "color": col, "auto": False, "time": tm or None}
        if len(names) <= 2:
            for n in names:
                ev.append({**base, "label": f"{n} — {kind} expiry",
                           "tip": f"{n}: next {kind} expiry" + (f" · {tm}" if tm else "")})
        else:
            ev.append({**base, "label": f"{asset} — {kind} expiries (×{len(names)})",
                       "tip": ", ".join(sorted(names)) + (f" · {tm}" if tm else "")})
    return ev


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _landing_closures(day_iso: str) -> list:
    """'🚫 CME closed — Labor Day'-style notes for `day` (full holidays + early
    closes), one per exchange. markethours calendar maths only."""
    d = date.fromisoformat(day_iso)
    closed, half = {}, {}
    for t in INSTRUMENTS:
        try:
            seg = markethours.day_segments(t, "America/New_York", d, INSTRUMENTS[t][2])
        except Exception:
            continue
        if seg.get("closed"):
            closed.setdefault(seg.get("exchange", "?"), seg["closed"])
        elif seg.get("half_day"):
            half.setdefault(seg.get("exchange", "?"),
                            seg.get("early_close") or seg["half_day"])
    return ([f"🚫 {x} closed — {why}" for x, why in sorted(closed.items())]
            + [f"⏱️ {x} early close {v}" for x, v in sorted(half.items())])


def _land_desk_go(side: str, dest: str) -> None:
    """Landing-page deep link that may cross desks: flip the desk first, then nav."""
    st.session_state["side"] = side
    _go(dest)


def _land_day_set(off: int) -> None:
    st.session_state["land_day"] = off


def _add_weekdays(d: date, n: int) -> date:
    """d shifted by n TRADING days (Mon–Fri) — the landing ‹ › never lands on a weekend."""
    step = 1 if n >= 0 else -1
    k = abs(n)
    while k:
        d += timedelta(days=step)
        if d.weekday() < 5:
            k -= 1
    return d


def render_landing() -> None:
    """The BASIS front door — what the sidebar logo (and a fresh session) lands on:
    the brand lockup over ONE trading-week calendar covering BOTH desks. Each day
    splits into a FICC band (fundamental reports + FOMC/ECB/BoE rate decisions —
    repcal.calendar_events) on top and an EQUITIES · EARNINGS band (eqearncal off
    the fundamentals DB) beneath — Ben's separation, 2026-08-15. The two full
    month calendars stay one click away."""
    from src import repcal, eqearncal
    pal = brand.palette()
    # Hero: the ❯ mark anchored LEFT, the enlarged BASIS + strapline centred on the
    # page, tight to the top (Ben's layout, 2026-08-15). The wordmark is the REAL
    # brand SVG — a CSS text-gradient stand-in read as the wrong colours.
    st.markdown(
        # classes carry the phone treatment (brand CSS): the absolutely-positioned
        # mark is dropped below 820px, where it landed on top of the wordmark
        f'<div class="land-hero" style="position:relative;padding:.05rem 0 .4rem;min-height:112px">'
        f'<div class="land-mark" style="position:absolute;left:.3rem;top:50%;transform:translateY(-50%)">'
        f'{brand.mark_svg(pal, height=88)}</div>'
        f'<div style="display:flex;flex-direction:column;align-items:center">'
        f'{brand.word_svg(pal, height=68)}'
        f'<div class="land-tag" style="font-size:17px;font-weight:600;letter-spacing:.44em;'
        f'text-transform:uppercase;color:{pal.get("blue_bright", pal["tagline"])};'
        f'margin-top:3px">'      # same blue as the sidebar strapline (Ben, 2026-08-21)
        f'Analysis · Strategies · Indicators</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    today = datetime.now(ZoneInfo("America/New_York")).date()
    # weekends look ahead: the front door shows the NEXT trading day, labelled as such
    base = today if today.weekday() < 5 else _add_weekdays(today, 1)
    st.session_state.setdefault("land_day", 0)
    off = st.session_state["land_day"]
    day = _add_weekdays(base, off)
    _badge = ('<span style="font-size:13px;letter-spacing:.12em;text-transform:uppercase;'
              'background:#F5C518;color:#14171C;font-weight:800;padding:2px 10px;'
              'border-radius:5px;margin-right:12px;vertical-align:3px">Today</span>')
    if day == today:
        _title = f"{_badge}{day:%a %d %b %Y}"          # the desk homes' gold badge
    elif off == 0:
        _title = f"{day:%a %d %b %Y} · next trading day"
    else:
        _title = f"{day:%A %d %b %Y}"
    # centred nav cluster:  ‹  day  ›  (arrows hug the date; trading days only).
    # Keyed container so the phone CSS can hold the five columns in ONE row —
    # stacked, the arrows became two full-width empty bars around the date.
    with st.container(key="land_nav"):
        _sp1, n_prev, n_title, n_next, _sp2 = st.columns([2.4, 0.55, 3.1, 0.55, 2.4],
                                                         vertical_alignment="center")
        n_prev.button("‹", key="land_prev", on_click=_land_day_set, args=(off - 1,),
                      use_container_width=True)
        n_title.markdown(f"<div class='land-daytitle dk-vc' style='font-size:21px;font-weight:700;"
                         f"text-align:center'>{_title}</div>", unsafe_allow_html=True)
        n_next.button("›", key="land_next", on_click=_land_day_set, args=(off + 1,),
                      use_container_width=True)
    if off != 0:                       # only offer the way back once you've left
        _r1, _r2, _r3 = st.columns([4, 1.7, 4])
        _r2.button("↩ Today", key="land_today", on_click=_land_day_set, args=(0,),
                   use_container_width=True)

    # holidays / early closes for the shown day, right under the date
    try:
        _cl = _landing_closures(day.isoformat())
        if _cl:
            st.markdown(f"<div style='text-align:center;font-size:13px;"
                        f"color:{brand.palette()['caption']};padding:2px 0 4px'>"
                        f"{' &nbsp;·&nbsp; '.join(_cl)}</div>", unsafe_allow_html=True)
    except Exception:
        pass

    ficc_ev = repcal.calendar_events()
    try:                              # + the ECO-page majors (CPI, PPI, NFP…) with times
        ficc_ev = ficc_ev + _landing_macro(day.isoformat())
    except Exception:
        pass
    try:                              # + products whose future/options expire this day
        ficc_ev = ficc_ev + _landing_expiries(day.isoformat())
    except Exception:
        pass
    try:
        with st.spinner("Loading the earnings calendar…"):
            eq_ev = _landing_eq_events()
    except Exception:
        eq_ev = []
    try:                              # Yahoo carries the report HOUR the DB date lacks
        _days = tuple(e["bbg"] for e in eq_ev if e.get("bbg") and e["date"] == day)
        if _days:
            with st.spinner("Looking up earnings times…"):
                _times = _earnings_times(_days)
            for e in eq_ev:
                t = _times.get(e.get("bbg"))
                if t:
                    e["time"] = t
    except Exception:
        pass
    def _next_line(evs):
        fut = [e for e in evs if e["date"] > day]
        if not fut:
            return None
        nd = min(e["date"] for e in fut)
        labs = sorted({f'{e["icon"]} {e["label"]}'.strip() for e in fut if e["date"] == nd})
        return (f'Next: {", ".join(labs[:3])}{"…" if len(labs) > 3 else ""} · {nd:%a %d %b}')
    # ── the desk-home design on the front door (Ben, 2026-08-21): the user's
    # My Day list beside BOTH day timelines, in the same card language ──
    dkF = repcal.desk_day(ficc_ev, day)
    dkE = repcal.desk_day(eq_ev, day)

    def _day_card(dk, evs, title, sub):
        nxt = ""
        if dk["total"] == 0:
            _nl = _next_line(evs)
            if _nl:
                nxt = f'<div class="dkl-none">{repcal._esc(_nl)}</div>'
        return ('<div class="dk-card" style="min-height:352px"><div class="dk-h">'
                f'<span class="dk-t">{title}</span><span class="dk-s">{sub}</span></div>'
                + dk["html"] + nxt + '</div>')
    _c0, _c1, _c2 = st.columns([0.85, 1, 1])
    with _c0:
        _myday_card()
    _c1.markdown(_day_card(dkF, ficc_ev, "FICC", "Prints, decisions &amp; expiries"),
                 unsafe_allow_html=True)
    _c2.markdown(_day_card(dkE, eq_ev, "Equities", "Earnings reporters"),
                 unsafe_allow_html=True)
    st.markdown('<div class="dk-legend">'
                '<span><span class="bar" style="background:#F5C518"></span>Expiry</span>'
                '<span><span class="bar" style="background:#7FB3F5"></span>Print · decision · '
                'earnings</span>'
                '<span>Past events dimmed · gold line = now</span>'
                '<span>Times in ET, local beneath · earnings times via Yahoo where published'
                '</span>'
                '<span>My Day tasks are per-seat</span></div>', unsafe_allow_html=True)
    if not eq_ev:
        st.caption("No earnings dates loaded yet — pull equities data to fill the "
                   "Equities panel.")

    b1, b2, _ = st.columns([1.7, 1.7, 4.6])
    b1.button("📅 Full reports calendar", key="land_ficc_cal", use_container_width=True,
              on_click=_land_desk_go, args=("FICC", "Release Calendar"))
    b2.button("📅 Full earnings calendar", key="land_eq_cal", use_container_width=True,
              on_click=_land_desk_go, args=("Equities", "eq:Earnings"))


def _home_day_set(off: int) -> None:
    st.session_state["home_day"] = off


@st.cache_data(ttl=30, show_spinner=False)
def _pull_driver_alive() -> bool:
    """True if a run_pull.py process exists. Only consulted when the status file
    claims 'running' (rare), cached 30s; on any doubt say alive — never cry wolf."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'run_pull' }).Count"],
            capture_output=True, text=True, timeout=10)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:
        return True


def _md_add_cb(seat: str) -> None:
    """My Day 'Add' — a callback so the title box can legally be cleared."""
    from src import myday
    _d = st.session_state.get("md_date")
    _t = st.session_state.get("md_time")
    myday.add(seat, st.session_state.get("md_title", ""),
              _d.isoformat() if _d else "",
              _t.strftime("%H:%M") if _t else "")
    st.session_state["md_title"] = ""
    st.session_state["md_date"] = None      # date is optional — reset to empty
    st.session_state["md_time"] = None


def _myday_card() -> None:
    """The per-seat private task list (redesign 2026-08-20). One JSON per seat on
    disk (src/myday.py); the header seat selector decides whose list shows."""
    from src import myday
    seat = st.session_state.get("seat", "admin")
    meta = next((s for s in myday.seats() if s["id"] == seat),
                {"name": str(seat), "desk": ""})
    items = myday.load(seat)
    today_s = date.today().isoformat()
    # open = anything not done that concerns today: dated today, or undated
    # (undated tasks recur every day until completed — Ben, 2026-08-20)
    open_n = sum(1 for i in items
                 if not i.get("done") and (i.get("date") or today_s) == today_s)
    with st.container(key="dkcard_myday"):
        st.markdown(f'<div class="dk-h"><span class="dk-t">My Day</span>'
                    f'<span class="dk-s">{meta["name"]} · {open_n} open</span></div>',
                    unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns([2.9, 1.7, 1.25, 1.0], vertical_alignment="center")
        a1.text_input("Task", key="md_title", label_visibility="collapsed",
                      placeholder="Add a task…")
        a2.date_input("Date (optional)", key="md_date", label_visibility="collapsed",
                      value=None)
        a3.time_input("Time", key="md_time", label_visibility="collapsed", value=None)
        a4.button("Add", key="md_add", use_container_width=True,
                  on_click=_md_add_cb, args=(seat,))
        _f = st.session_state.setdefault("md_filter", "today")

        def _today_view(i) -> bool:
            if i.get("date"):                     # dated: only on its date
                return i["date"] == today_s
            # undated: every day until done; keep the struck row for the rest of
            # the completion day so it can still be un-toggled
            return (not i.get("done")) or i.get("done_date") == today_s
        show = [i for i in items if _f == "all" or _today_view(i)]
        show.sort(key=lambda i: (0 if i.get("date") else 1,
                                 i.get("date", ""), i.get("time") or "99:99"))
        for i in show[:12]:
            _kind = "d" if i.get("date") else "u"     # gold rule = dated, blue = until-done
            with st.container(key=f'mdrow{_kind}_{i.get("id")}'):
                r1, r2, r3 = st.columns([1.15, 4.5, 0.5], vertical_alignment="center")
                _dl = ("Today" if i.get("date") == today_s
                       else (i.get("date") or "every day"))
                r1.markdown(f'<div class="dkl-t">{i.get("time") or "—"}'
                            f'<span class="loc">{_dl}</span></div>', unsafe_allow_html=True)
                _lbl = f'~~{i.get("title", "")}~~' if i.get("done") else i.get("title", "")
                if r2.button(_lbl or "—", key=f'md_t_{i.get("id")}', use_container_width=True,
                             help="Click to mark done / not done"):
                    myday.toggle(seat, i.get("id")); st.rerun()
                if r3.button("×", key=f'md_x_{i.get("id")}', help="Remove"):
                    myday.remove(seat, i.get("id")); st.rerun()
        if not show:
            st.caption("Nothing here yet — add your first task above. Leave the date "
                       "empty for a standing task that shows every day until it's done.")
        f1, f2, f3 = st.container(key="md_filters").columns([1.0, 1.25, 3.2],
                                                            vertical_alignment="center")
        if f1.button("Today", key="md_f_today",
                     type="primary" if _f == "today" else "secondary"):
            st.session_state["md_filter"] = "today"; st.rerun()
        if f2.button("All dates", key="md_f_all",
                     type="primary" if _f == "all" else "secondary"):
            st.session_state["md_filter"] = "all"; st.rerun()
        f3.markdown('<div class="dk-s" style="text-align:right;padding:6px 4px 0 0">'
                    '<span style="display:inline-block;width:10px;height:3px;'
                    'background:#F5C518;vertical-align:middle;margin-right:4px"></span>dated'
                    '<span style="display:inline-block;width:10px;height:3px;'
                    'background:#7FB3F5;vertical-align:middle;margin:0 4px 0 10px"></span>'
                    'every day until done · private to this seat</div>',
                    unsafe_allow_html=True)


def _hotsheet_top10_card(book: str = "ficc") -> None:
    """The design's Hot Sheet table, showing what module 02 actually shows: the
    radar's top strip. Same pipeline as the Hot Sheet page — persisted collection,
    NEW badges, the Home sector filter, one desk's book, 2-per-module cap, first
    10 — so the card and the module can never disagree. (The old card ran the
    tascore conviction composite, which is a different engine — Ben, 2026-08-20.)"""
    try:
        items, _report, _collected, _ = _hs_collect()
        items = [dict(it) for it in items]      # apply_badges mutates — keep the cache pristine
        hotsheet.apply_badges(items)
    except Exception:
        items, _collected = [], 0.0
    if universe.filter_active():                 # the Home sector filter applies here too
        _en = set(universe.enabled_tickers())
        items = [it for it in items
                 if it["book"] != "ficc" or not it["ticker"] or it["ticker"] in _en]
    items = [it for it in items if it["book"] == book]
    top, _per_tag = [], {}
    for it in items:                             # mirror of the module's top strip
        if _per_tag.get(it["tag"], 0) >= 2:
            continue
        _per_tag[it["tag"]] = _per_tag.get(it["tag"], 0) + 1
        top.append(it)
        if len(top) >= 10:
            break
    _sub = "Top 10 by heat"
    if top:
        try:
            _ts = pd.Timestamp(_collected, unit="s", tz="UTC").tz_convert("America/New_York")
            _sub = (f"Top {len(top)} by heat · {len({it['provider'] for it in items})} "
                    f"modules · collected {_ts:%H:%M} ET")
        except Exception:
            pass
    st.markdown(
        '<style>'
        '.hsr{display:grid;grid-template-columns:34px 118px minmax(0,1fr) minmax(200px,295px);'
        'gap:10px;align-items:center;padding:5px 2px;'
        'border-bottom:1px solid rgba(128,128,128,.14);font-size:13px}'
        '.hsr-head{padding:7px 2px;border-bottom:1px solid rgba(128,128,128,.28)}'
        '.hsr-head div{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;'
        'font-weight:600;color:var(--basis-cal-ink,#8a929c)}'
        '.hs-n,.hs-num{font-family:var(--basis-mono,monospace);font-variant-numeric:tabular-nums}'
        '.hs-num{text-align:right;white-space:nowrap;font-weight:600;'
        'overflow:hidden;text-overflow:ellipsis}'
        '.hs-tag{white-space:nowrap}'
        '.hs-chip{display:inline-block;font:600 10px/1.7 var(--basis-mono,monospace);'
        'color:var(--basis-gold,#F5C518);border:1px solid rgba(245,197,24,.45);'
        'padding:1px 6px;text-transform:uppercase;letter-spacing:.06em}'
        '.hs-newb{display:inline-block;font:700 10px/1.7 var(--basis-mono,monospace);'
        'color:#14171C;background:var(--basis-gold,#F5C518);padding:1px 6px;margin-left:6px}'
        '.hs-story{line-height:1.45;min-width:0;overflow-wrap:break-word}'
        '.hs-subi{font:400 10.5px var(--basis-mono,monospace);'
        'color:var(--basis-cal-ink,#8a929c);font-weight:400}'
        '.hs-heat{height:3px;background:rgba(128,128,128,.22);margin-top:4px}'
        '.hs-heat div{height:3px;background:var(--basis-gold,#F5C518)}'
        '</style>', unsafe_allow_html=True)
    # a KEYED container (not one HTML blob) so each row can carry the module's
    # jump button — the same _hs_go navigation as the Hot Sheet page (Ben)
    with st.container(key="dkcard_hotsheet"):
        st.markdown(f'<div class="dk-h"><span class="dk-t">Hot Sheet'
                    f'</span><span class="dk-s">{_sub}</span></div>', unsafe_allow_html=True)
        if not top:
            st.markdown('<div class="dkl-none">No module is clearing its bar — quiet '
                        'markets, or the morning snapshot hasn&#8217;t run yet.</div>',
                        unsafe_allow_html=True)
            return
        _hh, _hg = st.columns([12, 0.8], vertical_alignment="center")
        _hh.markdown('<div class="hsr hsr-head"><div>#</div><div>Tag</div><div>Story</div>'
                     '<div style="text-align:right">Metric · heat</div></div>',
                     unsafe_allow_html=True)
        for n, it in enumerate(top, start=1):
            story = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", it["text"])
            badge = '<span class="hs-newb">NEW</span>' if it.get("badge") == "NEW" else ""
            met = it["metric"] or f"{it['heat']:.0f}"
            sub = f'<span class="hs-subi"> · {it["sub"]}</span>' if it.get("sub") else ""
            bar = f'<div class="hs-heat"><div style="width:{it["heat"]:.0f}%"></div></div>'
            _rm, _rg = st.columns([12, 0.8], vertical_alignment="center")
            _rm.markdown(
                f'<div class="hsr"><div class="hs-n">{n:02d}</div>'
                f'<div class="hs-tag"><span class="hs-chip">{it["tag"]}</span>{badge}</div>'
                f'<div class="hs-story">{story}</div>'
                f'<div class="hs-num" title="{repcal._esc(met)}'
                f'{" · " + repcal._esc(it["sub"]) if it.get("sub") else ""}">'
                f'{met}{sub}{bar}</div></div>', unsafe_allow_html=True)
            if it.get("page"):
                _rg.button("→", key=f"hs_go_{book}_{n}",
                           help=f"Open {it['page'].removeprefix('eq:')}",
                           on_click=_hs_go, args=(it["page"],))
        st.button("Open Hot Sheet →", key=f"home_open_hs_{book}", on_click=_hs_go,
                  args=("Hot Sheet" if book == "ficc" else "eq:Hot Sheet",))


_MC_HOME_FILE = ROOT / "data" / "morning_coffee_home.json"


def _mc_synopsis_card() -> None:
    """The Morning Coffee report's market commentary in full (Ben, 2026-08-20:
    replaced the overnight-moves table on this row — the moves live on in the
    Morning Coffee page's treemap). Reads the same export as the headlines card;
    falls back to the short synopsis field for older exports."""
    try:
        mc = json.loads(_MC_HOME_FILE.read_text(encoding="utf-8"))
    except Exception:
        mc = {}
    prose = (mc.get("commentary") or mc.get("synopsis") or "").strip()
    stamp = mc.get("generated_at", "")
    with st.container(key="dkcard_mcsyn"):
        st.markdown(f'<div class="dk-h"><span class="dk-t">Synopsis · Morning Coffee</span>'
                    f'<span class="dk-s">{repcal._esc(stamp) if stamp else "no run yet"}'
                    f'</span></div>', unsafe_allow_html=True)
        if prose:
            paras = "".join(
                f'<p style="margin:0 0 .7rem;font-size:15px;line-height:1.6">'
                f'{repcal._esc(p.strip())}</p>'
                for p in prose.split("\n") if p.strip())
            st.markdown(f'<div style="padding:10px 2px 2px">{paras}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No commentary exported yet — the next Morning Coffee run will "
                       "fill this card.")


def _mc_card() -> None:
    """Headlines + synopsis from the last Morning Coffee run — reads the export the
    MC pipeline writes (morning_coffee_home.json); graceful before the first run."""
    try:
        mc = json.loads(_MC_HOME_FILE.read_text(encoding="utf-8"))
    except Exception:
        mc = {}
    heads = mc.get("headlines") or []
    stamp = mc.get("generated_at", "")
    _time = stamp.split("·")[-1].strip() if "·" in stamp else stamp      # "08:49 ET"
    _srcs = list(dict.fromkeys(str(h.get("source", "")).strip()
                               for h in heads if str(h.get("source", "")).strip()))
    with st.container(key="dkcard_mc"):
        _sub = f"{len(_srcs)} sources · pulled {_time}" if heads else "no run yet"
        st.markdown(f'<div class="dk-h"><span class="dk-t">Headlines · Morning Coffee'
                    f'</span><span class="dk-s" style="margin-right:96px">{repcal._esc(_sub)}'
                    f'</span></div>', unsafe_allow_html=True)
        # the design's Run report pill, floated into the header band (admin + this
        # PC only — the pipeline needs Bloomberg, the news feeds and the Gmail token)
        if IS_ADMIN and MORNING_COFFEE_DIR.exists():
            if st.button("Run report", key="home_mc_run",
                         help="Run the Morning Coffee pipeline now — pulls Bloomberg, reads "
                              "the news, writes the commentary and emails the desk (~1–2 min). "
                              "These cards refresh when it finishes."):
                with st.spinner("Pulling Bloomberg, reading the news, writing the macro "
                                "commentary and emailing the report… (~1–2 min)"):
                    _ok = run_morning_coffee()
                if _ok:
                    st.toast("Morning Coffee sent — cards refreshed.", icon="☕")
                else:
                    st.toast("Morning Coffee failed — see the Morning Coffee page for the log.",
                             icon="⚠️")
                st.rerun()
        # (the synopsis moved to its own card beside this one, 2026-08-20)
        if heads:
            # "cited in Hot Sheet: Brent" (design): product names from today's radar
            # stories, matched as whole words against each headline
            try:
                _hs_items, *_ = _hs_collect()
                _hs_names = []
                for _it in _hs_items:
                    _m = re.match(r"\s*\*\*([^*]+)\*\*", str(_it.get("text", "")))
                    if _m:
                        _nm = re.sub(r"\s*\([^)]*\)", "", _m.group(1)).strip()   # drop "(COMEX)"
                        if len(_nm) >= 3 and _nm not in _hs_names:
                            _hs_names.append(_nm)
            except Exception:
                _hs_names = []

            def _h_sub(h) -> str:
                _bits = [str(h.get("time", "")).strip(), str(h.get("tag", "")).strip()]
                _title = str(h.get("title", ""))
                _hits = [n for n in _hs_names
                         if re.search(r"\b" + re.escape(n.split(" × ")[0].split(" − ")[0]) + r"\b",
                                      _title, flags=re.I)]
                if _hits:
                    _bits.append("cited in Hot Sheet: " + ", ".join(_hits[:2]))
                _s = " · ".join(x for x in _bits if x)
                return (f'<div class="dk-s" style="margin-top:2px">{repcal._esc(_s)}</div>'
                        if _s else "")
            _rows = "".join(
                f'<div style="display:grid;grid-template-columns:92px 1fr;gap:10px;'
                f'padding:9px 2px;border-bottom:1px solid rgba(128,128,128,.1)">'
                f'<div style="font-family:var(--basis-mono,monospace);font-size:11px;'
                f'letter-spacing:.06em;text-transform:uppercase;color:#F5C518;'
                f'padding-top:2px">{repcal._esc(str(h.get("source", "")))}</div>'
                f'<div><div style="font-size:14.5px;line-height:1.45">'
                f'{repcal._esc(str(h.get("title", "")))}</div>'
                f'{_h_sub(h)}</div></div>'
                for h in heads[:8])
            st.markdown(_rows, unsafe_allow_html=True)
        else:
            st.caption("No headlines exported yet — the next Morning Coffee run will "
                       "fill this card.")
        # footer strip (design): sources roll-call + last run · gold link into the module
        with st.container(key="mc_footer"):
            _fl, _fr = st.columns([3.4, 1.0], vertical_alignment="center")
            _ftxt = (" · ".join(_srcs) + (f" — last run {_time}" if _time else "")
                     if _srcs else "No run yet")
            _fl.markdown(f'<div class="dk-vc mc-foot">{repcal._esc(_ftxt)}</div>',
                         unsafe_allow_html=True)
            with _fr:
                st.button("Open Morning Coffee →", key="home_open_mc", on_click=_go,
                          args=("Morning Coffee",))


def render_home() -> None:
    """FICC desk overview — the 2026-08-20 Claude Design redesign: a date bar with
    live event counts + the two data actions, My Day (per-seat tasks) beside the
    FICC day timeline, the Hot Sheet top-10, overnight moves + Morning Coffee,
    and the sector filter demoted to a bottom card. The old banners and the
    Excel / Weekly Review buttons were removed per Ben."""
    snap = _load_snap()
    _today = datetime.now(ZoneInfo("America/New_York")).date()
    _base = _today if _today.weekday() < 5 else _add_weekdays(_today, 1)
    st.session_state.setdefault("home_day", 0)
    _off = st.session_state["home_day"]
    _day = _add_weekdays(_base, _off)

    ficc_ev = repcal.calendar_events()
    try:
        ficc_ev = ficc_ev + _landing_macro(_day.isoformat())
    except Exception:
        pass
    try:
        ficc_ev = ficc_ev + _landing_expiries(_day.isoformat())
    except Exception:
        pass
    dk = repcal.desk_day(ficc_ev, _day)

    # ── date bar: ‹ Today · date › + live counts + the two data actions ──
    # keyed so the phone CSS can hold ‹ date › on one row (stacked, the arrows
    # became full-width empty bars around the date — same fix as the landing nav)
    p1, p2, p3, pc, c1, c2 = st.container(key="desk_datebar").columns(
        [0.42, 1.85, 0.42, 2.35, 1.4, 1.05], vertical_alignment="center")
    p1.button("‹", key="home_prev", on_click=_home_day_set, args=(_off - 1,),
              use_container_width=True)
    _tag = ('<span style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;'
            'background:#F5C518;color:#14171C;font-weight:800;padding:2px 9px;'
            'border-radius:5px;margin-right:12px;vertical-align:2px">Today</span>'
            if _day == _today else "")
    p2.markdown(f'<div class="dk-vc" style="text-align:center;font-family:var(--basis-mono,monospace);'
                f'font-size:17px;font-weight:600">{_tag}{_day:%a %d %b %Y}</div>',
                unsafe_allow_html=True)
    p3.button("›", key="home_next", on_click=_home_day_set, args=(_off + 1,),
              use_container_width=True)
    _bits = [f"{dk['total']} events" + (" today" if _day == _today else "")]
    if _day == _today:
        _bits.append(f"{dk['ahead']} still ahead")
        if dk.get("next_txt"):
            _bits.append(dk["next_txt"])
    pc.markdown('<div class="dk-s dk-vc" style="text-align:right;letter-spacing:.06em;'
                'text-transform:uppercase">' + " · ".join(_bits) + '</div>',
                unsafe_allow_html=True)
    # Honesty guard (2026-08-21): the status file said "running" for hours after a
    # server restart killed the driver mid-run. If the status says running but no
    # run_pull.py process exists, say so instead of showing nothing.
    if IS_ADMIN:
        try:
            _pstat = json.loads((ROOT / "data" / "snapshot" /
                                 ".pull_driver_status.json").read_text(encoding="utf-8"))
        except Exception:
            _pstat = {}
        if _pstat.get("outcome") == "running" and not _pull_driver_alive():
            st.warning("⚠️ **The pull driver was killed mid-run** (status still says "
                       "'running' but no driver process exists — usually a server "
                       "restart during a pull). The fetched data may already be safe: "
                       "if `logs/pull_driver_fetch.log` ends with *BLOOMBERG PHASE "
                       "COMPLETE*, press **Re-run signals** — do **not** pull again, "
                       "that would re-spend the day's Bloomberg allowance.")

    def _run_ficc_pull():
        # ONE button, self-healing (Ben, 2026-08-20): the whole pull runs through
        # run_pull.py — pre-flight probe (a block / logged-out Terminal refuses in
        # ~2s, zero hits), fetch with an 8-min WRITE-STALL watchdog, ONE automatic
        # retry (the playbook that fixed every wedge this month), compute, git
        # backup. Press Pull, keep the Terminal open until the green banner — no
        # babysitting, no 45-minute mornings.
        _DSTAT = ROOT / "data" / "snapshot" / ".pull_driver_status.json"

        def _dstat() -> dict:
            try:
                return json.loads(_DSTAT.read_text(encoding="utf-8"))
            except Exception:
                return {}

        ph = st.empty()
        t0 = time.time()
        # DETACHED, console output to a file — never pipes (2026-08-21: a server
        # restart mid-pull broke the driver's stdout pipe and killed it right
        # after a perfect fetch; detached + file logging means a bounce, keeper
        # respawn or crashed session can no longer take a running pull with it)
        _con = (ROOT / "logs" / "pull_driver_console.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "run_pull.py")], cwd=str(ROOT),
            stdout=_con, stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           | subprocess.DETACHED_PROCESS))
        _msgs = {
            "running": "⏳ **Pulling** — {el:.1f} min elapsed (fetch + maths; ~10–15 min "
                       "healthy). Self-healing: a stalled fetch is killed after ~8 quiet "
                       "minutes and retried once automatically. Keep the Terminal open "
                       "until the green banner.",
            "retrying": "🔁 **First attempt stalled — the automatic retry is running** "
                        "({el:.1f} min total). Nothing to do; keep the Terminal open.",
        }
        while proc.poll() is None:
            el = (time.time() - t0) / 60
            ph.info(_msgs.get(_dstat().get("outcome"), _msgs["running"]).format(el=el))
            time.sleep(5)
        _con.close()
        ph.empty()
        stat = _dstat()
        outcome, detail = stat.get("outcome"), stat.get("detail", "")
        if outcome == "ok":
            run_daily.run(); load_signals.clear()
            _regen_mc_heatmap()      # refresh the Morning Coffee heatmap on Home
            st.session_state.pop("ficc_pull_confirm", None)
            st.success(f"✅ Snapshot {detail} — backup pushed. You can close the "
                       "Terminal now.")
            st.rerun()
        elif outcome == "preflight_refused":
            if "WORKFLOW_REVIEW" in detail or "-4002" in detail:
                st.error(f"🚫 **{detail}.** The Terminal itself is fine — every API "
                         "request is rejected until Bloomberg lifts the block (HELP "
                         "HELP, quote the -4002). No pull was started, no hits spent. "
                         "The existing snapshot was **kept**.")
            else:
                st.error(f"🖥️ **Bloomberg isn't answering on this machine** — {detail}. "
                         "Open the Terminal, log in, and press Pull again. (No pull "
                         "was started — no API hits were spent.)")
        elif outcome == "fetch_failed_twice":
            st.error("⏱️ **Both fetch attempts stalled** — even the automatic retry "
                     "froze, which points at a genuinely unhealthy Bloomberg session. "
                     "Restart the Terminal (log in on THIS machine), then press Pull "
                     "again. The existing snapshot was **kept** — nothing was "
                     "overwritten.")
            try:
                _tail = (ROOT / "logs" / "pull_driver_fetch_retry.log").read_text(
                    encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                _tail = "no log"
            with st.expander("Technical log"):
                st.code(_tail, language="text")
        elif outcome == "compute_failed":
            st.error("Snapshot compute failed (the fetched data is safe on disk — "
                     "'Re-run signals' or retry). " + detail)
        else:
            st.error("The pull driver ended unexpectedly — see logs/pull_driver.log. "
                     "The existing snapshot was **kept**.")

    # Heavy handlers are DEFERRED (flag set here, executed below the row): blocking inside a
    # column slot pauses the script mid-row, so Streamlit showed a half-drawn fresh button row
    # with the old row faded beneath it for the whole computation.
    if IS_ADMIN and c1.button("Pull Bloomberg snapshot", use_container_width=True, key="home_pull",
                 help="Two phases: the Terminal is only needed for the FETCH (~3–5 min) — "
                      "the banner tells you when you can close it — then the maths (own-vol "
                      "curve, COT, signals) runs Terminal-free. Equities have their own pull "
                      "on the Equities home page."):
        # Pull guard (src/pullguard.py): pre-flight review-risk check — new securities,
        # weekend/off-hours timing, same-day re-pull. Any flag -> warn + confirm before a
        # single hit is spent (Bloomberg's -4002 workflow reviews trigger on usage SHAPE).
        _today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        try:
            from src import pullguard
            _pw = pullguard.assess(same_day=str((snap or {}).get("created", ""))[:10] == _today)
        except Exception:
            _pw = (["⚡ Snapshot already pulled today — a re-pull re-spends the day's "
                    "Bloomberg allowance on near-identical data."]
                   if str((snap or {}).get("created", ""))[:10] == _today else [])
        if _pw:
            st.session_state["ficc_pull_confirm"] = _pw
        else:
            st.session_state["ficc_pull_go"] = True
    if IS_ADMIN and c2.button("Re-run signals", use_container_width=True, key="home_rerun",
                 help="Recompute all strategies from the current data — instant in snapshot mode."):
        st.session_state["rerun_signals_go"] = True
    if IS_ADMIN and st.session_state.get("ficc_pull_confirm"):
        _pw = st.session_state["ficc_pull_confirm"]
        _pw = _pw if isinstance(_pw, list) else []       # legacy True from a hot-reload
        st.warning("**This pull has Bloomberg review-risk flags** — their -4002 workflow "
                   "reviews trigger on unusual usage patterns, and this pull would look "
                   "unusual:\n\n" + "\n\n".join(f"- {w}" for w in _pw)
                   if _pw else
                   f"⚡ Snapshot **already pulled today** ({_to_et((snap or {}).get('created', ''))}).")
        _g1, _g2, _ = st.columns([1.4, 1, 3.6])
        if _g1.button("Pull anyway", key="ficc_pull_anyway",
                      help="Proceed knowingly — the flags above are warnings, not blocks."):
            st.session_state.pop("ficc_pull_confirm", None)
            st.session_state["ficc_pull_go"] = True
        if _g2.button("Cancel", key="ficc_pull_cancel"):
            st.session_state.pop("ficc_pull_confirm", None); st.rerun()
    if IS_ADMIN and st.session_state.pop("ficc_pull_go", False):
        _run_ficc_pull()
    if IS_ADMIN and st.session_state.pop("rerun_signals_go", False):
        with st.spinner("Recomputing all signals…"):
            run_daily.run()
        load_signals.clear(); st.rerun()
    # (Excel export + Weekly Review buttons and the old banners removed in the
    #  2026-08-20 redesign per Ben — Excel lives on via `snapshot.py --excel`.)

    # ── My Day beside the FICC day timeline ──
    _cl, _cr = st.columns([0.82, 1])
    with _cl:
        _myday_card()
    with _cr:
        st.markdown('<div class="dk-card" style="min-height:352px"><div class="dk-h">'
                    '<span class="dk-t">FICC</span>'
                    '<span class="dk-s">Prints, decisions &amp; expiries</span>'
                    '</div>' + dk["html"] + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="dk-legend">'
                '<span><span class="bar" style="background:#F5C518"></span>Expiry</span>'
                '<span><span class="bar" style="background:#7FB3F5"></span>Print · decision</span>'
                '<span>Past events dimmed · gold line = now</span>'
                '<span>My Day tasks are per-seat · click a task to mark done</span>'
                '<span>Times in ET, local beneath</span></div>', unsafe_allow_html=True)

    # ── Hot Sheet top-10 ──
    _hotsheet_top10_card()

    # ── Morning Coffee: synopsis (left) + headlines (right) ──
    _bl, _br = st.columns(2)
    with _bl:
        _mc_synopsis_card()
    with _br:
        _mc_card()

    # ── heatmap, then sectors & products as the very last card (Ben, 2026-08-21) ──
    _home_heatmap()
    with st.container(key="dkcard_sectors"):
        render_sector_filter()


# ── EQUITIES side ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _eq_universe():
    """Index constituents (cached — membership changes rarely; cleared by 'Pull equities data')."""
    return equities.load_universe()


@st.cache_data(ttl=300, show_spinner=False)
def _eq_movers(index_keys: tuple):
    """Overnight-movers frame for the selected indices (cached; cleared by 'Refresh quotes')."""
    return equities.movers_frame(list(index_keys), universe=_eq_universe())


def _equities_overnight_moves(index_keys, snap, show_header: bool = True) -> None:
    if show_header:
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


@st.cache_data(ttl=1800, show_spinner=False)
def _eq_recent_actions(index_keys: tuple, n_names: int = 30, days: int = 60):
    """Recent rating actions across the most-moved names in the selected indices.

    Bounded on purpose: the feed costs one Yahoo request set per name, so it follows the
    ~`n_names` biggest overnight movers (the names already on this screen) and only US-listed
    lines — Yahoo's grade history doesn't exist for other listings. eqanalyst's disk cache
    serves everything already seen instantly and re-pulls only what's stale."""
    # store-first (app-wide once-a-day rule): the equities pull writes the card's
    # final feed for the default selection — ms instead of a per-name assembly
    _st = eqanalyst.read_home_feed(list(index_keys), n_names=n_names, days=days)
    if _st is not None:
        return _st
    f = _eq_movers(tuple(index_keys))
    if f is None or getattr(f, "empty", True):
        return pd.DataFrame(), {}
    d = f.dropna(subset=["pct"]).copy()
    if d.empty:
        return pd.DataFrame(), {}
    d["_rank"] = pd.to_numeric(d["sigma"], errors="coerce").abs()
    d["_rank"] = d["_rank"].fillna(pd.to_numeric(d["pct"], errors="coerce").abs())
    d = d[[eqanalyst.is_us_line(t) for t in d["ticker"]]]
    if d.empty:
        return pd.DataFrame(), {}
    d = d.sort_values("_rank", ascending=False).drop_duplicates("ticker").head(int(n_names))
    tks = list(d["ticker"])
    try:
        feed = eqanalyst.recent_actions(tks, names=dict(zip(d["ticker"], d["name"])),
                                        days=days, limit=10, max_fetch=int(n_names))
        return feed, eqanalyst.coverage(tks)
    except Exception:
        return pd.DataFrame(), {}


def _eq_rating_actions(index_keys, show_header: bool = True) -> None:
    """The Equities home's rating-actions strip: who changed their view on the names in view."""
    if show_header:
        st.subheader("Recent rating actions")
    with st.spinner("Checking the analyst feed…"):
        feed, cov = _eq_recent_actions(tuple(index_keys))
    scope = (f"the {cov.get('asked', 0)} biggest overnight movers among the **US-listed** names "
             f"in the selected indices ({cov.get('cached', 0)} of them with published coverage)"
             if cov else "the biggest overnight movers among the US-listed names in the "
                         "selected indices")
    if feed is None or getattr(feed, "empty", True):
        st.caption(f"No upgrades, downgrades or new coverage in the last 60 days across {scope}. "
                   "(Yahoo's free grade history covers US-listed lines only; full detail for any "
                   "single company is on **Company Fundamentals → Company tearsheet → Analyst view**.)")
        return
    rows = []
    for _, r in feed.iterrows():
        grade = (f"{r['from']} → {r['to']}" if r["from"] and r["to"] and r["from"] != r["to"]
                 else (r["to"] or r["from"] or "—"))
        tgt = eqanalyst.fmt_px(r.get("target"), r.get("ccy") or "")
        rows.append({"Date": r["date"].strftime("%d %b"), "Stock": r["stock"], "Firm": r["firm"],
                     "Rating": grade, "Action": eqanalyst.action_label(r["action"]),
                     "Price target": tgt, "_dir": int(r.get("dir") or 0)})
    tbl = pd.DataFrame(rows)
    dirs = tbl.pop("_dir").tolist()
    sty = [_EQF_GOOD_CSS if d0 > 0 else _EQF_BAD_CSS if d0 < 0 else "" for d0 in dirs]
    st.caption(f"Upgrades, downgrades and new coverage published in the last 60 days across "
               f"{scope} — grade changes only (routine target tweaks are left out). Published "
               "third-party analyst views, shown for context. Full consensus, targets and the "
               "longer action history for any company: **Company Fundamentals → Company "
               "tearsheet → Analyst view**.")
    brand.themed_dataframe(tbl, {}, colorers=[(["Action"], (lambda s: lambda col: s)(sty))],
                           height=int(40 + 35.2 * min(len(tbl), 10)))


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
_EQ_PULL_LOCK = ROOT / "data" / "snapshot" / ".eq_pull.lock"


def _eq_pull_running() -> tuple[bool, str]:
    """(is_running, HH:MM it started) from the equities pull lock file (written by
    snapshot.py's run_equities() for BOTH the manual button and the unattended scheduled
    Auto-pull). Self-expires after 20 min so a killed/crashed pull can't wedge the banner on
    forever (a normal pull finishes in ~5-7 min)."""
    try:
        if not _EQ_PULL_LOCK.exists():
            return False, ""
        mtime = _EQ_PULL_LOCK.stat().st_mtime
        age_min = (time.time() - mtime) / 60
        if 0 <= age_min < 20:
            return True, datetime.fromtimestamp(mtime).strftime("%H:%M")
    except Exception:
        pass
    return False, ""


def _autorefresh_fragment(seconds: int):
    """@st.fragment(run_every=seconds), degrading to a plain no-op wrapper on a Streamlit
    build without fragments — defined standalone (not reusing the later `_fragment` alias)
    so it's usable by functions defined earlier in the module than that alias is."""
    fn = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if fn is None:
        return lambda f: f
    try:
        return fn(run_every=seconds)
    except TypeError:
        return fn


@_autorefresh_fragment(10)
def _eq_pull_banner() -> None:
    """A live-updating bar while an equities pull (manual or scheduled Auto-pull) is running,
    on the Equities home page. Auto-refreshes itself every 10s and clears on its own once the
    pull's lock file is gone — no click needed to notice it start or finish."""
    running, started = _eq_pull_running()
    if running:
        st.info(f"⏳ **Equities pull running** — started {started}, typically ~5–7 min "
                "(membership + Yahoo quotes/history, fundamentals when due). This clears "
                "on its own when it finishes.", icon="⏳")


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


def _eq_home_day_set(off: int) -> None:
    st.session_state["eq_home_day"] = off


def render_equities_home() -> None:
    """Equities desk overview — the FICC desk-home design cloned (Ben, 2026-08-21):
    date bar with the data pills, My Day beside the earnings timeline, the equities
    Hot Sheet top-10, movers + rating actions, the heatmap, and the indices scope
    demoted to the bottom card. (The econ-figures strip stays retired — the BASIS
    landing board owns the day's prints, same call as the FICC redesign.)"""
    snap = _load_snap()
    _today = datetime.now(ZoneInfo("America/New_York")).date()
    _base = _today if _today.weekday() < 5 else _add_weekdays(_today, 1)
    st.session_state.setdefault("eq_home_day", 0)
    _off = st.session_state["eq_home_day"]
    _day = _add_weekdays(_base, _off)

    _keys = list(equities.INDICES.keys())
    # the indices multiselect renders in the BOTTOM card; read its state up here
    _sel_state = st.session_state.get("eq_idx_filter")
    sel = (_sel_state or _keys) if _sel_state is not None else list(equities.DEFAULT_INDICES)

    # today's reporters, with Yahoo's report hour merged in (Bloomberg is date-only)
    try:
        eq_ev = [dict(e) for e in _landing_eq_events()]
    except Exception:
        eq_ev = []
    try:
        _dayrep = tuple(e["bbg"] for e in eq_ev if e.get("bbg") and e["date"] == _day)
        if _dayrep:
            _times = _earnings_times(_dayrep)
            for e in eq_ev:
                t = _times.get(e.get("bbg"))
                if t:
                    e["time"] = t
    except Exception:
        pass
    dk = repcal.desk_day(eq_ev, _day)

    # ── date bar: ‹ Today · date › + reporter counts + the two data actions ──
    # keyed so the phone CSS can hold ‹ date › on one row (stacked, the arrows
    # became full-width empty bars around the date — same fix as the landing nav)
    p1, p2, p3, pc, c1, c2 = st.container(key="desk_datebar").columns(
        [0.42, 1.85, 0.42, 2.35, 1.4, 1.05], vertical_alignment="center")
    p1.button("‹", key="eq_home_prev", on_click=_eq_home_day_set, args=(_off - 1,),
              use_container_width=True)
    _tag = ('<span style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;'
            'background:#F5C518;color:#14171C;font-weight:800;padding:2px 9px;'
            'border-radius:5px;margin-right:12px;vertical-align:2px">Today</span>'
            if _day == _today else "")
    p2.markdown(f'<div class="dk-vc" style="text-align:center;font-family:var(--basis-mono,monospace);'
                f'font-size:17px;font-weight:600">{_tag}{_day:%a %d %b %Y}</div>',
                unsafe_allow_html=True)
    p3.button("›", key="eq_home_next", on_click=_eq_home_day_set, args=(_off + 1,),
              use_container_width=True)
    _bits = [f"{dk['total']} reporters" + (" today" if _day == _today else "")]
    if _day == _today:
        _bits.append(f"{dk['ahead']} still ahead")
        if dk.get("next_txt"):
            _bits.append(dk["next_txt"])
    pc.markdown('<div class="dk-s dk-vc" style="text-align:right;letter-spacing:.06em;'
                'text-transform:uppercase">' + " · ".join(_bits) + '</div>',
                unsafe_allow_html=True)
    try:                                   # mirror snapshot.py's equities pull switches
        from snapshot import PULL_EQUITY_CONSTITUENTS as _EQ_ON, PULL_FUNDAMENTALS as _EQF_ON
    except Exception:
        _EQ_ON = _EQF_ON = True
    _eq_pull_on = bool(_EQ_ON or _EQF_ON)
    if IS_ADMIN and c1.button("Pull equities data", use_container_width=True, key="eq_pull",
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
    if IS_ADMIN and not _eq_pull_on:
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
        # DETACHED + file logging, same lesson as the FICC driver (2026-08-21):
        # a server restart mid-pull must never kill the pull through a dead pipe
        _con = (ROOT / "logs" / "eq_pull_console.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(SNAPSHOT_CLI), "--equities"], cwd=str(ROOT),
            stdout=_con, stderr=subprocess.STDOUT,
            env={**os.environ, "DATAFEED_MODE": "bloomberg", "PYTHONUTF8": "1"},
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           | subprocess.DETACHED_PROCESS))
        while proc.poll() is None:
            _el = (time.time() - _t0) / 60
            _ph.info(f"⏳ Pulling equities — **{_el:.1f} min elapsed** (typically ~5–7 min for "
                     "the ~2,700-name universe: ETF membership + chunked Yahoo quotes/history, "
                     "plus fundamentals when their cycle is due).")
            time.sleep(5)
        _con.close()
        _ph.empty()
        if proc.returncode != 0:
            try:
                _tail = (ROOT / "logs" / "eq_pull_console.log").read_text(
                    encoding="utf-8", errors="replace")[-3000:]
            except Exception:
                _tail = "no log"
            st.error("Equities pull failed — the tail of logs/eq_pull_console.log:")
            st.code(_tail, language="text")
        else:
            _eq_universe.clear(); _eq_movers.clear(); _eq_heatmap_sections.clear()
            _eqf_frame.clear()
            gitbackup.push_data_async()  # fresh data → GitHub → VPS site within ~15 min
            st.success(f"Equities data refreshed ({(time.time() - _t0) / 60:.1f} min).")
            st.rerun()
    if IS_ADMIN and c2.button("Refresh quotes", use_container_width=True, key="eq_refresh",
                 help="Re-pull the latest closes from Yahoo Finance (free) and rebuild the "
                      "movers table and heatmap. Falls back to the cached quotes offline."):
        _eq_movers.clear(); _eq_heatmap_sections.clear()
        st.session_state["eq_refresh_note"] = True
        st.rerun()
    if IS_ADMIN and st.session_state.pop("eq_refresh_note", False):
        st.info("Quotes refreshed — live Yahoo Finance when reachable, otherwise the cached "
                "pull (see the source caption).")
    if IS_ADMIN:
        _eq_pull_banner()          # live bar while a manual/auto pull runs

    # ── My Day beside the earnings day timeline ──
    _cl, _cr = st.columns([0.82, 1])
    with _cl:
        _myday_card()
    with _cr:
        _next = ""
        if dk["total"] == 0:                      # quiet day → point at the next reporters
            _fut = [e for e in eq_ev if e["date"] > _day]
            if _fut:
                _nd = min(e["date"] for e in _fut)
                _labs = sorted({e["label"] for e in _fut if e["date"] == _nd})
                _next = ('<div class="dkl-none">Next: '
                         + repcal._esc(", ".join(_labs[:4]))
                         + ("…" if len(_labs) > 4 else "")
                         + f' · {_nd:%a %d %b}</div>')
        st.markdown('<div class="dk-card" style="min-height:352px"><div class="dk-h">'
                    '<span class="dk-t">Equities</span>'
                    '<span class="dk-s">Earnings reporters</span>'
                    '</div>' + dk["html"] + _next + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="dk-legend">'
                '<span><span class="bar" style="background:#7FB3F5"></span>Earnings report</span>'
                '<span>Past reporters dimmed · gold line = now</span>'
                '<span>Times via Yahoo where published, — date-only otherwise</span>'
                '<span>My Day tasks are per-seat · click a task to mark done</span>'
                '</div>', unsafe_allow_html=True)

    # ── Hot Sheet top-10 (equities book) ──
    _hotsheet_top10_card("equities")

    # ── overnight movers (left) + rating actions (right) ──
    _bl, _br = st.columns(2)
    with _bl, st.container(key="dkcard_eqmoves"):
        st.markdown('<div class="dk-h"><span class="dk-t">Overnight movers</span>'
                    '<span class="dk-s">settlement → latest close · σ-ranked</span></div>',
                    unsafe_allow_html=True)
        _equities_overnight_moves(sel, snap, show_header=False)
    with _br, st.container(key="dkcard_eqrate"):
        st.markdown('<div class="dk-h"><span class="dk-t">Rating actions</span>'
                    '<span class="dk-s">upgrades · downgrades · new coverage</span></div>',
                    unsafe_allow_html=True)
        _eq_rating_actions(sel, show_header=False)

    # ── heatmap, then the indices scope as the very last card ──
    _equities_heatmap(sel)
    with st.container(key="dkcard_eqscope"):
        st.markdown(f'<div class="dk-h"><span class="dk-t">Indices &amp; universe</span>'
                    f'<span class="dk-s">{len(sel)}/{len(_keys)} indices in view</span></div>',
                    unsafe_allow_html=True)
        _mcol, _acol = st.columns([3.2, 1.1], vertical_alignment="center")
        _mcol.multiselect("Indices to show", _keys, default=list(equities.DEFAULT_INDICES),
                          key="eq_idx_filter",
                          help="Scope the movers table and heatmap to these indices. "
                               "Russell 2000 (~2000 names) is opt-in — add it here when needed.",
                          label_visibility="collapsed")
        if IS_ADMIN:
            _eq_autopull_control(_acol)
        _n = sum(len(v) for v in _eq_universe().values())
        st.caption(f"**Universe:** {_n} index constituents across {len(_keys)} indices · "
                   + equities.data_status() + ". Quotes, history and fundamentals ride Yahoo "
                   "Finance free of charge; Bloomberg only refreshes index membership.")


# ── Company Fundamentals (Equities) ───────────────────────────────────────────
_EQF_GOOD_CSS = "color:#137333;font-weight:700"
_EQF_BAD_CSS = "color:#c5221f;font-weight:700"

# Preset screens — thresholds are SECTOR percentiles (like-for-like within GICS sector),
# except 'raw<=' which caps the raw value (a payout ratio over ~80% strains the dividend
# whatever the sector norms are).
# The page's three views. A plain control (not st.tabs) so the screener can send you
# straight to a company — Streamlit can't switch tabs programmatically.
_EQF_VIEWS = ["🔎 Screener", "📇 Company", "⚖️ Compare"]

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
    # disk store (app-wide once-a-day rule): the daily equities pull warms it for
    # the default indices; any other selection computes once and keeps its own
    return eqfunda.company_frame_cached(universe=equities.cached_universe(),
                                        index_keys=list(index_keys))


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


# ── shared price / signal seams for the company page (all on-disk, no network) ─
@st.cache_data(ttl=1800, show_spinner=False)
def _eqf_closes():
    """The equities TA backfill: ~4.5 years of split+dividend-adjusted daily closes for the
    whole ~2,670-name universe, straight off disk. Refreshed by the daily equities pull —
    reading it costs nothing, so the company page charts without touching Yahoo."""
    from src import eqta
    try:
        close, _vol = eqta.load_history()
        return close if close is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def _eqf_signal_rows(ticker: str):
    """(this name's live technical signals, signals meta) from the equities TA run."""
    from src import eqta
    try:
        sig, meta = eqta.load_signals()
    except Exception:
        return pd.DataFrame(), {}
    if sig is None or getattr(sig, "empty", True):
        return pd.DataFrame(), {}
    d = sig[sig["instruments"].astype(str) == str(ticker)]
    return d.copy(), (meta or {})


def _eqf_price_stats(ticker: str) -> dict | None:
    """Last settled close + the return/σ/52-week context for the company header. None when
    the name isn't in the TA backfill (a very recent index addition, or a non-mapping line)."""
    c = _eqf_closes()
    if c.empty or ticker not in c.columns:
        return None
    s = pd.to_numeric(c[ticker], errors="coerce").dropna()
    if s.empty:
        return None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else float("nan")
    pct = (last / prev - 1.0) * 100.0 if prev == prev and prev else float("nan")
    rets = s.pct_change().dropna() * 100.0
    sd = float(rets.tail(21).std()) if len(rets) >= 5 else float("nan")
    out = {"last": last, "asof": s.index[-1], "pct": pct,
           "sigma": (pct / sd) if (sd == sd and sd > 1e-9 and pct == pct) else float("nan")}
    for lbl, n in (("r1m", 21), ("r3m", 63), ("r12m", 252)):
        out[lbl] = ((last / float(s.iloc[-n - 1]) - 1.0) * 100.0
                    if len(s) > n and float(s.iloc[-n - 1]) else float("nan"))
    yr = s.tail(252)
    hi, lo = float(yr.max()), float(yr.min())
    out.update({"hi52": hi, "lo52": lo,
                "range_pos": ((last - lo) / (hi - lo) * 100.0) if hi > lo else float("nan")})
    return out


def _eqf_days_to_report(row) -> tuple:
    """(display string, days until) for the next expected report date on this name."""
    raw = eqfunda.fmt_value("EXPECTED_REPORT_DT", row.get("EXPECTED_REPORT_DT"))
    if raw == "—":
        return "—", None
    try:
        d = pd.Timestamp(raw).normalize()
    except (TypeError, ValueError):
        return raw, None
    days = int((d - pd.Timestamp.today().normalize()).days)
    return f"{d:%d %b %Y}", days


# ── screener ─────────────────────────────────────────────────────────────────
def _eqf_range_bounds(df: pd.DataFrame, field: str):
    """(lo, hi) slider bounds for a metric — the 1st/99th percentile of the current selection,
    so one outlier can't stretch the slider into uselessness. None when nothing is valued."""
    s = pd.to_numeric(df.get(field), errors="coerce").dropna()
    if s.empty:
        return None
    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
    if not (hi > lo):
        lo, hi = float(s.min()), float(s.max())
    return (lo, hi) if hi > lo else None


def _eqf_apply_pending() -> None:
    """Apply a queued screen/company jump BEFORE the widgets are built (Streamlit refuses
    session_state writes to a key whose widget already ran this pass)."""
    spec = st.session_state.pop("_eqf_apply", None)
    if not spec:
        return
    if spec.get("preset") in _EQF_PRESETS:
        st.session_state["eqf_preset"] = spec["preset"]
    for k, key in (("sectors", "eqf_sectors"), ("cols", "eqf_cols")):
        if isinstance(spec.get(k), list):
            st.session_state[key] = spec[k]
    ranges = spec.get("ranges") or {}
    st.session_state["eqf_rng_fields"] = list(ranges)
    for f, pair in ranges.items():
        try:
            st.session_state[f"eqf_rng_{f}"] = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, IndexError):
            pass


def _eqf_screener(df: pd.DataFrame) -> None:
    labels = {f["field"]: f["label"] for f in eqfunda.FIELDS}
    metric_opts = [f["field"] for f in eqfunda.FIELDS if f["kind"] not in ("text", "date")]
    c1, c2 = st.columns([2, 3])
    preset = c1.selectbox("Preset screen", list(_EQF_PRESETS), key="eqf_preset")
    sectors = sorted(df["sector"].dropna().unique())
    sec_sel = c2.multiselect("Sectors", sectors, key="eqf_sectors", help="Blank = all sectors.")
    cols = st.multiselect("Metrics (columns)", metric_opts, default=eqfunda.SCREENER_DEFAULT,
                          format_func=lambda f: labels[f], key="eqf_cols") or eqfunda.SCREENER_DEFAULT
    cols = [f for f in cols if f in df.columns]        # a field a pull didn't return can't be a column

    # ---- value filters: raw-number ranges on top of the percentile presets ----
    ranges: dict = {}
    with st.expander("🎚️ Value filters — screen on the actual numbers, not just sector rank"):
        rng_fields = st.multiselect("Metrics to filter", metric_opts, key="eqf_rng_fields",
                                    format_func=lambda f: labels[f],
                                    help="Each metric adds a range slider below. Presets screen on "
                                         "SECTOR PERCENTILE; these screen on the raw value "
                                         "(e.g. 'yield above 3%' regardless of sector).")
        scols = st.columns(2)
        for i, f in enumerate([f for f in rng_fields if f in df.columns]):
            b = _eqf_range_bounds(df, f)
            if b is None:
                scols[i % 2].caption(f"_{labels[f]}: nothing valued in this selection._")
                continue
            lo, hi = b
            step = max((hi - lo) / 200.0, 1e-6)
            sel = scols[i % 2].slider(labels[f], lo, hi, value=(lo, hi), step=step,
                                      key=f"eqf_rng_{f}")
            if sel and (sel[0] > lo or sel[1] < hi):
                ranges[f] = [float(sel[0]), float(sel[1])]
        if rng_fields:
            st.caption("Sliders span the 1st–99th percentile of the current selection. A metric left "
                       "at full width doesn't filter; narrowing one drops names with **no value** "
                       "for it, since they can't be shown to pass.")

    # ---- saved screens ----
    saved = eqfunda.load_screens()
    s1, s2, s3 = st.columns([2, 1.4, 1.4])
    nm = s1.text_input("Save this screen as", key="eqf_screen_name",
                       placeholder="e.g. Quality compounders, EU")
    if s2.button("💾 Save screen", use_container_width=True, key="eqf_screen_save",
                 disabled=not str(nm).strip()):
        eqfunda.save_screen(nm, {"preset": preset, "sectors": list(sec_sel), "cols": list(cols),
                                 "ranges": ranges})
        st.success(f"Saved “{nm}”.")
    if saved:
        pick = s3.selectbox("Saved screens", ["—"] + sorted(saved), key="eqf_screen_pick")
        l1, l2, _ = st.columns([1.2, 1.2, 4])
        if l1.button("📂 Load", key="eqf_screen_load", disabled=pick == "—"):
            st.session_state["_eqf_apply"] = saved.get(pick, {})
            st.rerun()
        if l2.button("🗑️ Delete", key="eqf_screen_del", disabled=pick == "—"):
            eqfunda.delete_screen(pick)
            st.rerun()

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
    for f, (lo, hi) in ranges.items():
        v = pd.to_numeric(sub.get(f), errors="coerce")
        sub = sub[v.notna() & v.between(lo, hi)]
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
    # Hover text on every metric header — what the number measures and how to read it.
    col_cfg = {labels[f]: st.column_config.Column(labels[f], help=eqfunda.help_for(f))
               for f in cols}
    st.caption(f"**{len(disp)}** companies · sorted by market cap · **green / red = top / bottom "
               "20% of the stock's own GICS sector** on that metric, direction-aware (low is the "
               "good end for valuation multiples and leverage, high for the rest)."
               + ("  ·  Value filters: "
                  + ", ".join(f"{labels[f]} {eqfunda.fmt_value(f, lo)}–{eqfunda.fmt_value(f, hi)}"
                              for f, (lo, hi) in ranges.items()) if ranges else "")
               + "  ·  **Click a row to open that company**, or hover a column header for what "
                 "the metric measures.")
    # The grid key carries a counter: bumping it on a jump hands back a FRESH grid with no row
    # selected, so coming back to the screener doesn't instantly re-fire the last click.
    ev = brand.themed_dataframe(disp, fmt, colorers=colorers, na_rep="—", height=520,
                                column_config=col_cfg,
                                key=f"eqf_screen_grid_{st.session_state.get('_eqf_grid_n', 0)}",
                                selection_mode="single-row", on_select="rerun")
    picked = list(((getattr(ev, "selection", None) or {}) if ev is not None else {}).get("rows", []))
    if picked:
        r = sub.iloc[picked[0]]
        st.session_state["_eqf_grid_n"] = st.session_state.get("_eqf_grid_n", 0) + 1
        st.session_state["_eqf_goto"] = {"view": _EQF_VIEWS[1],
                                         "company": f"{r['name']}  ({r['ticker']})"}
        st.rerun()


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
            "good": eqfunda.goodness(f, p), "help": eqfunda.help_for(f),
        })
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _eqa_record(ticker: str):
    """One name's analyst record (consensus / targets / counts trend / rating actions).
    Cached per session on top of eqanalyst's own daily disk cache, so flipping between
    companies never re-hits Yahoo for a name already seen."""
    try:
        return eqanalyst.analyst(ticker)
    except Exception:
        return None


def _eqa_actions_table(rec: dict, height_rows: int = 8) -> None:
    """The rating-actions feed for one name — date · firm · from → to · action · target."""
    d = eqanalyst.actions_frame(rec, limit=12)
    if d.empty:
        st.caption("No rating actions on the free feed for this name"
                   + (" in the stored window (~2 years)."
                      if eqanalyst.is_us_line(rec.get("ticker", ""))
                      else " — Yahoo carries the upgrade/downgrade history for **US-listed** "
                           "lines only, so non-US listings show consensus and targets but an "
                           "empty action feed."))
        return
    ccy = (rec.get("targets") or {}).get("ccy") or ""
    rows = []
    for _, r in d.iterrows():
        grade = (f"{r['from']} → {r['to']}" if r["from"] and r["to"] and r["from"] != r["to"]
                 else (r["to"] or r["from"] or "—"))
        tgt = eqanalyst.fmt_px(r.get("target"), ccy)
        if r.get("prior") and r.get("target") and r["prior"] != r["target"]:
            tgt += f"  (was {eqanalyst.fmt_px(r['prior'])})"
        rows.append({"Date": r["date"].strftime("%d %b %Y"), "Firm": r["firm"],
                     "Rating": grade, "Action": eqanalyst.action_label(r["action"]),
                     "Price target": tgt,
                     "_dir": eqanalyst.action_direction(r["action"])})
    tbl = pd.DataFrame(rows)
    dirs = tbl.pop("_dir").tolist()
    sty = [_EQF_GOOD_CSS if d0 > 0 else _EQF_BAD_CSS if d0 < 0 else "" for d0 in dirs]
    brand.themed_dataframe(tbl, {}, colorers=[(["Action"], (lambda s: lambda col: s)(sty))],
                           height=int(40 + 35.2 * min(len(tbl), height_rows)))


def _eqa_tearsheet_block(ticker: str, name: str) -> None:
    """'Analyst view' — what the sell side currently publishes on this name: consensus
    rating, the buy/hold/sell split and how it has moved, price targets vs spot, and the
    recent rating actions. Free Yahoo data; descriptive only."""
    st.markdown("#### Analyst view")
    with st.spinner("Loading the analyst consensus…"):
        rec = _eqa_record(ticker)
    if not rec:
        st.caption(f"No published analyst consensus for **{name}** on the free feed "
                   "(coverage is thinnest in small caps and some non-US listings).")
        return
    con, tg = rec.get("consensus") or {}, rec.get("targets") or {}
    ccy = tg.get("ccy") or ""
    mean, n = con.get("mean"), con.get("n")
    up = eqanalyst.implied_upside(rec)
    pos = eqanalyst.target_range_pos(rec)
    trend = eqanalyst.trend_frame(rec)
    shift = eqanalyst.trend_shift(rec)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Street consensus", con.get("label") or "—",
              (f"mean {mean:.2f}" if isinstance(mean, (int, float)) else "—")
              + (f" · {n} analysts" if n else ""), delta_color="off",
              help="Yahoo's consensus mean on the **1 = most positive … 5 = least positive** "
                   "scale. The 'Consensus rating (1–5)' in the Income & Ownership table above "
                   "is the same figure on the inverted Bloomberg convention (5 = most positive).")
    m2.metric("Mean price target", eqanalyst.fmt_px(tg.get("mean"), ccy),
              (f"{up:+.1f}% vs {eqanalyst.fmt_px(tg.get('spot'))}" if up == up else "—"),
              help="Consensus mean target and the implied move from the price it was pulled "
                   "against — both in the listing currency, so the ratio is unit-safe.")
    m3.metric("Target range",
              f"{eqanalyst.fmt_px(tg.get('low'))} – {eqanalyst.fmt_px(tg.get('high'))}",
              (f"price sits {pos:.0f}% up the range" if pos is not None else "—"),
              delta_color="off",
              help="Lowest and highest published target, and where the pull-time price sits "
                   "between them.")
    _posnow = trend["pos_pct"].iloc[0] if not trend.empty and trend["pos_pct"].notna().any() else None
    m4.metric("Positive ratings", f"{_posnow:.0f}%" if _posnow is not None and _posnow == _posnow else "—",
              (f"{shift:+.0f}pp vs 3m ago" if shift == shift else "—"),
              help="Share of covering analysts at buy or strong buy, and how that share has "
                   "moved across the four stored monthly vintages.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Rating split — last four monthly vintages**")
        if trend.empty:
            st.caption("No stored buy/hold/sell split for this name.")
        else:
            lbl = {"0m": "Now", "-1m": "1m ago", "-2m": "2m ago", "-3m": "3m ago"}
            tt = pd.DataFrame({
                "Vintage": [lbl.get(p, p) for p in trend["period"]],
                "Strong buy": trend["strongBuy"], "Buy": trend["buy"], "Hold": trend["hold"],
                "Sell": trend["sell"], "Strong sell": trend["strongSell"],
                "Analysts": trend["total"],
                "% positive": [f"{v:.0f}%" if v == v and v is not None else "—"
                               for v in trend["pos_pct"]],
            })
            brand.themed_dataframe(tt, {}, height=int(40 + 35.2 * len(tt)))
    with c2:
        st.markdown("**Recent rating actions**")
        _eqa_actions_table(rec)
    _age = eqanalyst.age_days(rec)
    st.caption("Consensus, targets and rating actions are published **third-party analyst "
               "views**, summarised here for context — they are observations about the "
               "Street's positioning, not a recommendation. Source: Yahoo Finance (free), "
               f"pulled {rec.get('pulled', '—')}"
               + (" (today)." if _age == 0 else f" ({_age}d ago)." if _age else ".")
               + "  Rating actions are US-listed lines only on this feed.")


def _eqf_header_metrics(row, stats: dict | None) -> None:
    """The company page's top strip: where the price is, how it has travelled, and when the
    next set of numbers lands."""
    rep, days = _eqf_days_to_report(row)
    h = st.columns(6)
    if stats:
        h[0].metric("Last close", f"{stats['last']:,.2f}",
                    f"{stats['pct']:+.2f}%" if stats["pct"] == stats["pct"] else "—",
                    help=f"Settled close for {stats['asof']:%d %b %Y} from the equities TA "
                         "backfill (split + dividend adjusted), and the move vs the prior close.")
        h[1].metric("Move in σ", f"{stats['sigma']:+.1f}σ" if stats["sigma"] == stats["sigma"] else "—",
                    "vs its own 1m daily σ", delta_color="off",
                    help="The last session's move in standard deviations of this stock's own "
                         "~1-month daily moves — the same sizing rule as the movers table.")
        h[2].metric("3-month", f"{stats['r3m']:+.1f}%" if stats["r3m"] == stats["r3m"] else "—",
                    f"12m {stats['r12m']:+.1f}%" if stats["r12m"] == stats["r12m"] else "—",
                    delta_color="off", help="Total return over the last ~63 and ~252 sessions.")
        h[3].metric("52-week range",
                    f"{stats['range_pos']:.0f}%" if stats["range_pos"] == stats["range_pos"] else "—",
                    f"{stats['lo52']:,.0f} – {stats['hi52']:,.0f}", delta_color="off",
                    help="Where the last close sits between the 52-week low and high.")
    else:
        h[0].metric("Last close", "—", "no price history")
    h[4].metric("Market cap",
                eqfunda.fmt_value("CRNCY_ADJ_MKT_CAP", row.get("CRNCY_ADJ_MKT_CAP")),
                eqfunda.fmt_value("CRNCY", row.get("CRNCY")), delta_color="off")
    h[5].metric("Next report", rep,
                ("today" if days == 0 else f"in {days}d" if days and days > 0
                 else f"{abs(days)}d ago" if days else "—"),
                delta_color="off",
                help="Expected reporting date from the fundamentals pull. Anything inside a "
                     "week is flagged under the strip.")
    if days is not None and 0 <= days <= 5:
        st.warning(f"📅 **{row['name']} reports in {days} day(s)** ({rep}) — figures and technical "
                   "levels below can move sharply around the release.")


def _eqf_price_panel(ticker: str, name: str) -> None:
    """Adjusted price with 50/200-day averages, over a pickable window. Reads the on-disk TA
    backfill, so it costs nothing and matches exactly what the technical strategies see."""
    import altair as alt
    c = _eqf_closes()
    if c.empty or ticker not in c.columns:
        st.caption("No stored price history for this name — it joins the chart after the next "
                   "equities pull (Equities home → **Pull equities data**).")
        return
    s = pd.to_numeric(c[ticker], errors="coerce").dropna()
    if len(s) < 5:
        st.caption("Too little stored price history to chart yet.")
        return
    span = st.segmented_control("Window", ["1M", "3M", "6M", "1Y", "3Y", "Max"], default="1Y",
                                key="eqf_px_span") or "1Y"
    n = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756}.get(span)
    ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
    win = s.tail(n) if n else s
    d = pd.DataFrame({"date": win.index, "Price": win.values,
                      "50-day": ma50.reindex(win.index).values,
                      "200-day": ma200.reindex(win.index).values})
    long = d.melt("date", var_name="series", value_name="level").dropna(subset=["level"])
    cc = brand.chart_colors()
    line = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("level:Q", title="Adjusted price", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["Price", "50-day", "200-day"],
                                        range=[cc["accent"], cc["long"], cc["muted"]]),
                        legend=alt.Legend(orient="top")),
        strokeWidth=alt.condition("datum.series == 'Price'", alt.value(2.0), alt.value(1.1)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("level:Q", title="Level", format=",.2f")])
    brand.show_chart(line.properties(height=300))
    st.caption("Split- and dividend-adjusted closes from the equities backfill — the same series "
               "every technical strategy runs on, so the chart and the signals below can't "
               "disagree. Moving averages are computed on the full history, then windowed.")


def _eqf_tech_read(ticker: str, name: str) -> None:
    """Which technical strategies are currently firing on this name, straight from the daily
    equities TA run — the read the Technical Analysis page has and the tearsheet never did."""
    sig, meta = _eqf_signal_rows(ticker)
    st.markdown("#### Technical read")
    if sig.empty:
        st.caption(f"No technical strategy is currently firing on {name}"
                   + (f" (signals as of {meta.get('as_of')})." if meta.get("as_of") else ".")
                   + " That is a flat read, not missing data — most names are quiet on most days.")
        return
    sig = sig.sort_values("metric", ascending=False, na_position="last")
    # Strategies rank on different yardsticks (3m return %, distance to the level, …), so the
    # measure travels WITH each row rather than becoming a column of mostly-blank cells.
    rows = [{"Strategy": r["strategy"], "Signal": r["signal"],
             "Ranking measure": (f"{str(r['metric_label'])}: {float(r['level']):,.2f}"
                                 if pd.notna(r.get("level")) else str(r["metric_label"])),
             "What triggered it": r.get("context") or "—",
             "_dir": int(r.get("direction") or 0)} for _, r in sig.iterrows()]
    tbl = pd.DataFrame(rows)
    dirs = tbl.pop("_dir").tolist()
    sty = [_EQF_GOOD_CSS if d0 > 0 else _EQF_BAD_CSS if d0 < 0 else "" for d0 in dirs]
    brand.themed_dataframe(tbl, {}, colorers=[(["Signal"], (lambda s: lambda col: s)(sty))],
                           height=int(40 + 35.2 * min(len(tbl), 9)))
    st.caption(f"**{len(sig)}** strategy signal(s) live on this name"
               + (f" · signals as of {meta.get('as_of')}" if meta.get("as_of") else "")
               + ". Long/short is the strategy's own directional wording; the full trigger logic "
                 "and backtests live on **Equities → Technical Analysis**.")


def _eqf_group_tables(groups: dict, n_peers: int) -> None:
    """The four fundamentals groups, each metric with its value, the sector median, and a bar
    showing where in the sector the value ranks (gold in the top/bottom decile)."""
    st.caption("**Sector rank** places the value inside the stock's own GICS sector "
               f"({n_peers} names in the current index selection), direction-aware — the bar "
               "fills to the percentile and the centre tick is the sector median. Values at an "
               "extreme of their sector range may be worth a closer look. **Hover any metric "
               "name** (dotted underline) for what it measures.")
    cols2 = st.columns(2)
    for i, g in enumerate(eqfunda.GROUP_ORDER):
        rows = groups.get(g)
        if not rows:
            continue
        with cols2[i % 2]:
            brand.panel_header(g, right=f"{len(rows)} metrics")
            brand.terminal_table(
                [{"m": r["label"], "v": r["value"], "med": r["median"],
                  "p": r["pctl_txt"], "bar": r["pctl"], "tip": r.get("help", "")} for r in rows],
                [{"key": "m", "label": "Metric", "help_key": "tip"},
                 {"key": "v", "label": "Value", "align": "right"},
                 {"key": "med", "label": "Sector med", "align": "right"},
                 {"key": "p", "label": "Pctl", "align": "right"},
                 {"key": "bar", "label": "Sector rank", "pbar": True,
                  "help": "Percentile within the GICS sector — left is low, right is high."}])


def _eqf_tearsheet(df: pd.DataFrame, asof: str, src: str) -> None:
    d = df.sort_values("name").reset_index(drop=True)
    lab = (d["name"] + "  (" + d["ticker"] + ")").tolist()
    pick = st.selectbox("Company", lab, key="eqf_co")
    row = d.iloc[lab.index(pick)]
    ticker = str(row["ticker"])
    bits = [row["sector"], row["indices"], row["region"], ticker]
    st.markdown(f"### {row['name']}")
    st.caption("  ·  ".join(str(b) for b in bits if b and b != "—")
               + f"  ·  fundamentals as of {asof} ({src}).")
    stats = _eqf_price_stats(ticker)
    _eqf_header_metrics(row, stats)
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        if st.button("⚖️ Compare with its peers", key="eqf_to_peers", use_container_width=True,
                     help="Opens the Compare view pre-loaded with this name and the five "
                          "closest comparables in its sector."):
            st.session_state["_eqf_goto"] = {"view": _EQF_VIEWS[2], "peers_for": ticker}
            st.rerun()
    st.divider()
    _eqf_price_panel(ticker, str(row["name"]))
    st.divider()
    _eqf_tech_read(ticker, str(row["name"]))
    st.divider()
    peers = df[df["sector"] == row["sector"]]
    groups = _eqf_group_rows(row, peers)
    st.markdown("#### Fundamentals vs sector")
    _eqf_group_tables(groups, len(peers))

    st.divider()
    _eqa_tearsheet_block(ticker, str(row["name"]))

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


def _eqf_rebased_chart(tickers: list, names: dict, span: str) -> None:
    """Total-return price paths for the compare set, rebased to 100 at the start of the window —
    the 'who has actually outperformed' picture the numbers alone don't give."""
    import altair as alt
    c = _eqf_closes()
    have = [t for t in tickers if t in getattr(c, "columns", [])]
    if not have:
        st.caption("No stored price history for the selected names yet.")
        return
    n = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756}.get(span)
    rows = []
    for t in have:
        s = pd.to_numeric(c[t], errors="coerce").dropna()
        s = s.tail(n) if n else s
        if len(s) < 2 or not float(s.iloc[0]):
            continue
        base = float(s.iloc[0])
        rows.append(pd.DataFrame({"date": s.index, "level": s.values / base * 100.0,
                                  "name": names.get(t, t)}))
    if not rows:
        st.caption("Not enough overlapping history to rebase these names.")
        return
    d = pd.concat(rows, ignore_index=True)
    ch = alt.Chart(d).mark_line(strokeWidth=1.6).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("level:Q", title=f"Rebased to 100 ({span})", scale=alt.Scale(zero=False)),
        color=alt.Color("name:N", title=None, legend=alt.Legend(orient="top", columns=3)),
        tooltip=[alt.Tooltip("name:N", title="Company"), alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("level:Q", title="Rebased", format=",.1f")])
    brand.show_chart(ch.properties(height=290))


def _eqf_street_rows(tickers: list, names: dict) -> list:
    """Consensus / target / implied-move rows for the compare table — a bounded analyst pull
    (the compare set is six names at most, and eqanalyst serves cached ones instantly)."""
    try:
        recs = eqanalyst.bulk(tickers, max_fetch=len(tickers))
    except Exception:
        recs = {}
    if not recs:
        return []
    out = []
    for label, fn in (("Street consensus",
                       lambda r: (r.get("consensus") or {}).get("label") or "—"),
                      ("Analysts covering",
                       lambda r: str((r.get("consensus") or {}).get("n") or "—")),
                      ("Mean price target",
                       lambda r: eqanalyst.fmt_px((r.get("targets") or {}).get("mean"),
                                                  (r.get("targets") or {}).get("ccy") or "")),
                      ("Implied move to target",
                       lambda r: eqanalyst.fmt_pct(eqanalyst.implied_upside(r)))):
        rec = {"Group": "Street view", "Metric": label}
        for t in tickers:
            r = recs.get(t)
            rec[names.get(t, t)] = fn(r) if r else "—"
        out.append(rec)
    return out


def _eqf_scatter(df: pd.DataFrame, highlight: list, labels: dict) -> None:
    """Any metric against any other across a sector — where the outliers actually show up."""
    import altair as alt
    metric_opts = [f["field"] for f in eqfunda.FIELDS
                   if f["kind"] not in ("text", "date") and f["field"] in df.columns]
    if len(metric_opts) < 2:
        return
    c1, c2, c3 = st.columns([2, 2, 2])
    xf = c1.selectbox("X axis", metric_opts, key="eqf_sc_x",
                      index=metric_opts.index("BEST_PE_RATIO") if "BEST_PE_RATIO" in metric_opts else 0,
                      format_func=lambda f: labels[f])
    yf = c2.selectbox("Y axis", metric_opts, key="eqf_sc_y",
                      index=metric_opts.index("RETURN_COM_EQY") if "RETURN_COM_EQY" in metric_opts else 1,
                      format_func=lambda f: labels[f])
    sectors = sorted(df["sector"].dropna().unique())
    scope = c3.multiselect("Sectors plotted", sectors, key="eqf_sc_sec",
                           help="Blank = the sectors of the compared names.")
    if not scope:
        scope = sorted({s for s in df[df["ticker"].isin(highlight)]["sector"].dropna()}) or sectors
    d = df[df["sector"].isin(scope)].copy()
    d["x"] = pd.to_numeric(d[xf], errors="coerce")
    d["y"] = pd.to_numeric(d[yf], errors="coerce")
    d["cap"] = pd.to_numeric(d.get("CRNCY_ADJ_MKT_CAP"), errors="coerce")
    d = d[d["x"].notna() & d["y"].notna()]
    # Trim the far tails so one 900x P/E doesn't flatten every other name onto the axis — but
    # never drop a COMPARED name, which is exactly the case where the outlier is the point.
    keep = d["ticker"].isin(highlight)
    for col in ("x", "y"):
        lo, hi = d[col].quantile(0.01), d[col].quantile(0.99)
        d = d[keep.loc[d.index] | d[col].between(lo, hi)]
    if d.empty:
        st.caption("Nothing valued on both metrics in this scope.")
        return
    d["set"] = np.where(d["ticker"].isin(highlight), "Compared", "Sector")
    d["x_lbl"] = [eqfunda.fmt_value(xf, v) for v in d["x"]]
    d["y_lbl"] = [eqfunda.fmt_value(yf, v) for v in d["y"]]
    cc = brand.chart_colors()
    pts = alt.Chart(d).mark_circle(stroke="white", strokeWidth=0.4).encode(
        x=alt.X("x:Q", title=labels[xf], scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=labels[yf], scale=alt.Scale(zero=False)),
        size=alt.Size("cap:Q", title="Market cap", legend=None,
                      scale=alt.Scale(range=[25, 420])),
        color=alt.Color("set:N", title=None,
                        scale=alt.Scale(domain=["Compared", "Sector"],
                                        range=[cc["accent"], cc["muted"]]),
                        legend=alt.Legend(orient="top")),
        opacity=alt.condition("datum.set == 'Compared'", alt.value(0.95), alt.value(0.45)),
        tooltip=[alt.Tooltip("name:N", title="Company"), alt.Tooltip("sector:N", title="Sector"),
                 alt.Tooltip("x_lbl:N", title=labels[xf]),
                 alt.Tooltip("y_lbl:N", title=labels[yf])])
    txt = alt.Chart(d[d["set"] == "Compared"]).mark_text(
        dy=-11, fontSize=10, color=cc["accent"]).encode(x="x:Q", y="y:Q", text="name:N")
    brand.show_chart((pts + txt).properties(height=380))
    st.caption(f"**{len(d)}** companies across {', '.join(scope)} · dot size = market cap · the "
               "compared names are labelled. Axes are trimmed at the 1st/99th percentile so a "
               "single extreme multiple can't flatten the cloud.")


def _eqf_peers(df: pd.DataFrame) -> None:
    d = df.sort_values("name").reset_index(drop=True)
    lab = (d["name"] + "  (" + d["ticker"] + ")").tolist()
    by_ticker = {str(r["ticker"]): f"{r['name']}  ({r['ticker']})" for _, r in d.iterrows()}
    labels = {f["field"]: f["label"] for f in eqfunda.FIELDS}

    a1, a2 = st.columns([3, 2])
    anchor_lab = a1.selectbox("Anchor company", ["—"] + lab, key="eqf_peer_anchor",
                              help="The name you're comparing FROM — auto-peers builds its "
                                   "comparison set from here.")
    if a2.button("👥 Auto-peers (same sector, nearest size)", use_container_width=True,
                 key="eqf_peer_auto", disabled=anchor_lab == "—"):
        tk = str(d.iloc[lab.index(anchor_lab)]["ticker"])
        picks = [by_ticker[p] for p in eqfunda.peer_set(df, tk, 5) if p in by_ticker]
        st.session_state["_eqf_goto"] = {"view": _EQF_VIEWS[2],
                                         "peer_sel": [anchor_lab] + picks}
        st.rerun()
    sel = st.multiselect("Companies (2–6)", lab, key="eqf_peer_sel", max_selections=6,
                         help="Pick a name and its peers — e.g. the sector rivals across indices.")
    if len(sel) < 2:
        st.caption("Pick at least two companies to compare side by side — or choose an anchor "
                   "above and hit **Auto-peers**, which takes the five closest names in its GICS "
                   "sector by market cap.")
        return
    rows_d = [d.iloc[lab.index(x)] for x in sel]
    tickers = [str(r["ticker"]) for r in rows_d]
    names = {str(r["ticker"]): x for r, x in zip(rows_d, sel)}

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
    with st.spinner("Loading the Street's view on the set…"):
        street = _eqf_street_rows(tickers, names)
    recs = recs + street
    disp = pd.DataFrame(recs)
    colorers = [([x], (lambda s: lambda col: s)(
                    [(_EQF_GOOD_CSS if best.get(r["Metric"]) == x else "") for r in recs]))
                for x in sel]
    st.caption("**Green = best of the selected group** on that metric, direction-aware; context "
               "metrics with no better/worse end (yield, payout, size) stay unmarked. The "
               "**Street view** rows are published analyst consensus, shown for context.")
    brand.themed_dataframe(disp, {}, colorers=colorers, height=int(40 + 35.2 * min(len(recs), 16)))

    st.divider()
    st.markdown("#### Sector rank across the set")
    pctl_cols = [f for f in eqfunda.SCREENER_DEFAULT if f + "__pctl" in df.columns]
    if pctl_cols:
        grid = pd.DataFrame({"Company": [str(r["name"]) for r in rows_d]})
        cell_style = {}
        for f in pctl_cols:
            vals = [r.get(f + "__pctl") for r in rows_d]
            grid[labels[f]] = ["—" if (v is None or v != v) else eqfunda.ordinal(v) for v in vals]
            cell_style[labels[f]] = [_EQF_GOOD_CSS if eqfunda.goodness(f, v) > 0
                                     else _EQF_BAD_CSS if eqfunda.goodness(f, v) < 0 else ""
                                     for v in vals]
        brand.themed_dataframe(grid, {}, colorers=[([c], (lambda s: lambda col: s)(cell_style[c]))
                                                   for c in cell_style],
                               column_config={labels[f]: st.column_config.Column(
                                   labels[f], help=eqfunda.help_for(f)) for f in pctl_cols},
                               height=int(40 + 35.2 * len(grid)))
        st.caption("Each cell is the company's percentile **within its own GICS sector** — so two "
                   "names from different sectors are still judged against their own peers. Green / "
                   "red mark the top / bottom 20%, direction-aware. Hover a column header for what "
                   "the metric measures.")

    st.divider()
    st.markdown("#### Price paths")
    span = st.segmented_control("Window", ["3M", "6M", "1Y", "3Y", "Max"], default="1Y",
                                key="eqf_cmp_span") or "1Y"
    _eqf_rebased_chart(tickers, {t: str(r["name"]) for t, r in zip(tickers, rows_d)}, span)

    st.divider()
    st.markdown("#### Metric scatter")
    _eqf_scatter(df, tickers, labels)


def _eqf_take_goto() -> None:
    """Consume a queued view jump (screener row click, 'compare with peers', auto-peers) before
    any of those widgets are built this pass."""
    goto = st.session_state.pop("_eqf_goto", None)
    if not goto:
        return
    if goto.get("view") in _EQF_VIEWS:
        st.session_state["eqf_view"] = goto["view"]
    if goto.get("company"):
        st.session_state["eqf_co"] = goto["company"]
    if goto.get("peer_sel"):
        st.session_state["eqf_peer_sel"] = list(goto["peer_sel"])[:6]
    if goto.get("peers_for"):
        st.session_state["_eqf_peers_for"] = goto["peers_for"]


def render_eq_fundamentals() -> None:
    st.subheader("🏢 Company Fundamentals")
    st.caption("Everything BASIS holds on a listed company — price and technicals, the research "
               "fundamentals ranked **within GICS sector** so a bank's P/B is judged against "
               "banks not software, and the Street's published view — plus a screener over the "
               "whole universe and a side-by-side compare. Every pull **appends** to the "
               "fundamentals database, so trends accumulate over time.")
    _eqf_take_goto()
    _eqf_pull_note()
    c1, c2 = st.columns([1, 2])
    if IS_ADMIN and c1.button("📥 Pull fundamentals", use_container_width=True, key="eqf_pull",
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
                         help="Scope the screener / company page / compare to these indices. "
                              "Russell 2000 (~2000 names) is opt-in — add it here when needed.")
    with st.spinner("Loading the fundamentals database…"):
        df, asof, src = _eqf_frame(tuple(sel or _keys))
    if df.empty:
        st.caption("No universe loaded — pull equities data first (Equities Home).")
        return
    _eqf_apply_pending()
    # A queued "compare with its peers" resolves here — peer_set needs the loaded frame.
    _pf = st.session_state.pop("_eqf_peers_for", None)
    if _pf:
        _by = {str(r["ticker"]): f"{r['name']}  ({r['ticker']})" for _, r in df.iterrows()}
        if _pf in _by:
            st.session_state["eqf_peer_anchor"] = _by[_pf]
            st.session_state["eqf_peer_sel"] = ([_by[_pf]]
                                                + [_by[p] for p in eqfunda.peer_set(df, _pf, 5)
                                                   if p in _by])
    view = st.segmented_control("View", _EQF_VIEWS, default=_EQF_VIEWS[0], key="eqf_view")
    if view == _EQF_VIEWS[1]:
        _eqf_tearsheet(df, asof, src)
    elif view == _EQF_VIEWS[2]:
        _eqf_peers(df)
    else:
        _eqf_screener(df)


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
    if IS_ADMIN and st.button("🔄 Refresh ETF data", key="eqetf_refresh",
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


# --- overview pages: the Hot Sheet + data health -----------------------------
_STRAT_SHORT = {
    "Mean Reversion": "MeanRev", "Trend": "Trend", "MA Crossover": "MA×",
    "MA Swing": "MA∿", "Flag Breakout": "Flag", "Support & Resistance": "S/R",
    "Fibonacci Retracement": "Fib", "Breakout & Retest": "Retest",
    "Momentum (RSI/MACD)": "Mom", "Bollinger Squeeze": "BBands", "Elliott Wave": "Elliott",
    "Ichimoku Cloud": "Ichimoku", "On-Balance Volume": "OBV", "Money Flow Index": "MFI",
    "Donchian Channel": "Donch", "Aroon": "Aroon",
    "Volatility": "Vol", "Skew Volatility": "Skew",
    "Vol Term Structure": "Term", "COT Reports": "COT", "Put/Call Ratios": "P/C",
    "AG Fundamentals": "AG",
}


def _norm_mkt(m) -> str:
    """Strip the ' · Sector' suffix some strategies append → base instrument name."""
    return str(m).split(" · ")[0].strip()


from src import hotsheet


@st.cache_data(show_spinner="Collecting the desk's screens…")
def _hs_cached(cache_mtime: float):
    """Session-shared view of the PERSISTED sheet (hotsheet.cached_collection):
    the morning stamp writes data/signals/hotsheet_cache.json and page opens just
    read it back — keyed on the file's mtime so a new stamp or an on-page ↻
    invalidates this, and an ordinary click never runs a provider."""
    return hotsheet.cached_collection()


def _hs_collect():
    try:
        _mt = hotsheet.CACHE_FILE.stat().st_mtime
    except OSError:
        _mt = 0.0
    return _hs_cached(_mt)


def _hs_go(dest: str) -> None:
    """Jump into the module that owns an item — equities pages need the desk side
    switched too, or the sidebar highlight lands on the wrong book."""
    st.session_state.side = "Equities" if dest.startswith("eq:") else "FICC"
    _go(dest)


_HS_MONO = "'IBM Plex Mono', Consolas, monospace"


def _hs_spark(vals: list, pal: dict) -> str:
    """Inline SVG sparkline of an item's own series (oldest→newest): a quiet trace
    with the latest point marked gold, so a slow grind and a sudden snap read
    differently before the prose is even parsed. Pure markup — no chart library,
    so thirty rows render instantly."""
    W, H, P = 110.0, 30.0, 2.5
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    xy = [(P + i * (W - 2 * P) / (n - 1), P + (H - 2 * P) * (1 - (v - lo) / rng))
          for i, v in enumerate(vals)]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    lx, ly = xy[-1]
    return (f'<svg width="100%" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
            f'preserveAspectRatio="none" style="display:block">'
            f'<polyline points="{pts}" fill="none" stroke="{pal["faint"]}" '
            f'stroke-width="1.2" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.4" fill="{pal["gold"]}"/></svg>')


def _hs_row(it: dict, uid: str, pal: dict) -> None:
    """One Hot Sheet line: tag chip + badge + prose, the story's sparkline, the
    metric column with its heat gauge, and the jump into the owning module."""
    c_txt, c_spark, c_met, c_go = st.columns([8, 3, 3, 1])
    chip = (f'<span style="font:600 .6rem/1.7 {_HS_MONO};color:{pal["gold"]};'
            f'border:1px solid {pal["label_ring"]};padding:.05rem .35rem;'
            f'margin-right:.5rem;white-space:nowrap">{it["tag"]}</span>')
    badge = ""
    if it.get("badge") == "NEW":
        badge = (f'<span style="font:700 .6rem/1.7 {_HS_MONO};color:{pal["canvas"]};'
                 f'background:{pal["gold"]};padding:.05rem .35rem;margin-right:.5rem">NEW</span>')
    elif it.get("badge"):
        badge = (f'<span style="font:600 .6rem/1.7 {_HS_MONO};color:{pal["faint"]};'
                 f'border:1px solid {pal["border"]};padding:.05rem .35rem;'
                 f'margin-right:.5rem;white-space:nowrap">{it["badge"]}</span>')
    c_txt.markdown(chip + badge + it["text"], unsafe_allow_html=True)
    if it.get("spark"):
        c_spark.markdown(f'<div style="padding-top:.35rem">{_hs_spark(it["spark"], pal)}</div>',
                         unsafe_allow_html=True)
    met = it["metric"] or f"{it['heat']:.0f}"
    sub = (f'<br><span style="font:400 .62rem {_HS_MONO};color:{pal["faint"]}">{it["sub"]}</span>'
           if it["sub"] else "")
    bar = (f'<div style="height:3px;background:{pal["border"]};margin-top:.3rem">'
           f'<div style="height:3px;width:{it["heat"]:.0f}%;background:{pal["gold"]}"></div></div>')
    c_met.markdown(f'<div style="text-align:right"><span style="font:600 .78rem {_HS_MONO};'
                   f'color:{pal["text"]}">{met}</span>{sub}{bar}</div>', unsafe_allow_html=True)
    if it["page"]:
        c_go.button("→", key=f"hs_{uid}", help=f"Open {it['page'].removeprefix('eq:')}",
                    on_click=_hs_go, args=(it["page"],))


def _hs_flow_section(flow: list, pal: dict) -> None:
    """Unusual option activity — its OWN section, deliberately outside the ranked sheet
    (Ben, 2026-08-23). Every other Hot Sheet row is a dislocation: something is out of
    line and there's a trade in it, ranked by how extreme. These rows claim nothing —
    a lot of contracts changed hands in one product yesterday, go and look. Mixing the
    two would put an activity number on the same heat scale as a signal, which it isn't."""
    if not flow:
        return
    st.markdown(f'<div style="border-top:1px solid {pal["border"]};margin:1.1rem 0 .55rem"></div>',
                unsafe_allow_html=True)
    st.markdown("##### 📣 Unusual option activity")
    st.caption("Products that traded far more puts or calls in the last session than they "
               "normally do, measured against each contract's own recent daily volume. "
               "This is an **activity** flag, not a signal — it carries no direction and no "
               "view, it just marks where the flow went.")
    for i, it in enumerate(flow):
        _hs_row(it, f"flow{i}", pal)


def _hs_refresh_button(host) -> None:
    """↻ — re-run every provider now and re-persist the sheet. Ordinary opens read
    the morning stamp's file; this is for intraday freshness after an ad-hoc pull."""
    if host.button("↻", key="hs_refresh",
                   help="Re-run every provider now (10–30s). The sheet otherwise "
                        "refreshes itself with the morning snapshot."):
        with st.spinner("Re-collecting the desk's screens…"):
            hotsheet.refresh_collection()
        _hs_cached.clear()
        st.rerun()


def _hs_footer(report: dict) -> None:
    """Provider roll-call — a newly built module (anything in src/ defining
    radar_items()) visibly plugs itself in here, and a broken one visibly falls off
    instead of silently vanishing from the sheet."""
    n_on = sum(1 for v in report.values() if v["n"])
    n_quiet = sum(1 for v in report.values() if v["status"] == "quiet")
    n_fail = sum(1 for v in report.values() if v["status"] == "failed")
    try:
        _m = json.loads(hotsheet.META_FILE.read_text(encoding="utf-8"))
        _stamp = f" · history last stamped {_m.get('stamped', '—')}"
    except Exception:
        _stamp = " · history not stamped yet"
    with st.expander(f"Providers — {n_on} contributing · {n_quiet} quiet · {n_fail} failed{_stamp}"):
        st.caption("The page collects live on each render; badges and the Weekly Review read the "
                   "once-a-day history stamp written by the morning snapshot.")
        rows = [{"Provider": k, "Status": v["status"], "Items": v["n"], "ms": v["ms"],
                 "Error": v["err"]} for k, v in sorted(report.items())]
        brand.themed_dataframe(pd.DataFrame(rows), {})


def render_hotsheet(book: str = "ficc") -> None:
    """One Hot Sheet per desk (Ben, 2026-08-19): the FICC page shows the FICC book,
    the Equities side carries its own page — no cross-book toggle. Meta caveats
    (data health, ledger) ride the FICC sheet, where the stores they guard live."""
    _desk = "FICC" if book == "ficc" else "Equities"
    st.subheader("🔥 Hot Sheet")
    st.caption(f"Everything the {_desk} modules are flagging **today**, ranked on one heat "
               "scale — the morning 30-second read. Each line jumps into the module that owns "
               "it; the sheet is exception-based, so quiet modules simply don't appear.")
    items, report, _collected, _from_cache = _hs_collect()
    _col_ts = pd.Timestamp(_collected, unit="s", tz="UTC").tz_convert("America/New_York")
    hotsheet.apply_badges(items)
    if universe.filter_active():                 # the Home sector filter applies here too
        _en = set(universe.enabled_tickers())
        items = [it for it in items
                 if it["book"] != "ficc" or not it["ticker"] or it["ticker"] in _en]

    pal = brand.palette()
    meta_items = [it for it in items if it["book"] == "meta"] if book == "ficc" else []
    items = [it for it in items if it["book"] == book]
    if meta_items:                               # trust caveats first — a quiet sheet only means
        _cav = " · ".join(it["text"].replace("**", "") for it in meta_items)   # calm markets if the data is healthy
        st.markdown(f'<div style="border:1px solid {pal["border"]};border-left:3px solid #D9971C;'
                    f'background:{pal["surface"]};padding:.45rem .7rem;font-size:.8rem;'
                    f'color:{pal["text_dim"]};margin-bottom:.6rem">⚠️ {_cav}</div>',
                    unsafe_allow_html=True)

    # Unusual option activity is NOT part of the ranked sheet — it gets its own section
    # below (Ben, 2026-08-23). Split it out before the strip so it can neither take a
    # strip slot nor be buried among the "rest of the sheet" expanders.
    flow_items = [it for it in items if it["tag"] == "FLOW"]
    items = [it for it in items if it["tag"] != "FLOW"]

    if not items:
        st.info(f"No {_desk} module is clearing its bar right now — either genuinely quiet "
                "markets, or the morning snapshot hasn't run yet (see the provider roll-call "
                "below).")
        _hs_flow_section(flow_items, pal)      # activity can still be worth showing
        _hs_refresh_button(st)
        _hs_footer(report)
        return

    c1, c_r, _sp = st.columns([3, 1, 8])
    _order = c1.radio("Order", ["By heat", "New first"], horizontal=True, key="hs_order",
                      label_visibility="collapsed")
    _hs_refresh_button(c_r)
    if _order == "New first":
        items.sort(key=lambda it: (it.get("badge") != "NEW", -it["heat"]))

    n_new = sum(1 for it in items if it.get("badge") == "NEW")
    st.caption(f"**{len(items)}** items across **{len({it['provider'] for it in items})}** "
               f"{_desk} modules" + (f" — **{n_new}** new today." if n_new else ".")
               + f" Screens collected **{_col_ts:%H:%M} ET**.")
    # Top strip: at most 2 lines per module, so one module's ties (COT crowding and
    # perfect seasonal records both pin heat at 100) can't crowd the cross-desk read —
    # everything skipped here still shows in its module's expander below.
    top, _per_tag = [], {}
    for it in items:
        if _per_tag.get(it["tag"], 0) >= 2:
            continue
        _per_tag[it["tag"]] = _per_tag.get(it["tag"], 0) + 1
        top.append(it)
        if len(top) >= 10:
            break
    for i, it in enumerate(top):
        _hs_row(it, f"top{i}", pal)

    _hs_flow_section(flow_items, pal)

    _top_ids = {id(it) for it in top}
    rest = [it for it in items if id(it) not in _top_ids]
    if rest:
        st.markdown("##### The rest of the sheet")
        by_sect: dict = {}
        for it in rest:
            by_sect.setdefault(it["section"], []).append(it)
        for sect in sorted(by_sect, key=lambda s: -max(x["heat"] for x in by_sect[s])):
            with st.expander(f"{sect} · {len(by_sect[sect])}", expanded=False):
                for i, it in enumerate(by_sect[sect]):
                    _hs_row(it, f"{sect}_{i}", pal)
    _hs_footer(report)


def _ax(tk) -> str:
    """Y-axis title for a technical chart — fixed income is charted as YIELDS, not price."""
    return "Yield (%)" if universe.is_fixed_income(str(tk)) else "Price"


def _ta_quicknav(current: str | None = None, eq: bool = False) -> None:
    """Quick-switch buttons for the technical strategies — the same 2×5 set as the Technical
    Analysis hub. Shown on the hub and at the top of each technical-strategy page so the user
    can flip between them without the sidebar; the current page's button is highlighted. `eq`
    routes to the Equities per-strategy pages (`eq:<strategy>`) instead of the FICC ones."""
    cols = st.columns(8)                                # 16 strategies → a neat 2 rows of 8
    for i, s in enumerate(tascore.TA_STRATEGIES):
        dest = f"eq:{s}" if eq else s
        cols[i % 8].button(
            _STRAT_SHORT.get(s, s), key=f"tanav_{'eq_' if eq else ''}{current or 'hub'}_{s}",
            use_container_width=True, type="primary" if s == current else "secondary",
            on_click=_go, args=(dest,))


def _trigger_docs_expander(scope: str = "ficc", picked=None, only: str | None = None) -> None:
    """The audited plain-English buy/sell trigger reference (src/strategy_docs.py TRIGGER_DOCS),
    shared by the TA Backtester, both TA hubs and the per-strategy pages. `only` renders a single
    strategy's entry (the per-strategy pages); otherwise the full roster. `picked` tags the
    strategies currently in the backtester's selection."""
    from src.strategy_docs import TRIGGER_DOCS
    eq = scope == "equities"
    _title = (f"📖 When {only} buys / sells" if only
              else "📖 How each strategy decides to buy / sell")
    if only and only not in TRIGGER_DOCS:
        return
    with st.expander(_title, expanded=False):
        st.caption("What the method's actual code does — distilled during the signal audit and "
                   "locked by fixture tests (a clean textbook long must flag ▲, the mirror ▼), so "
                   "this reference can't quietly drift from the engine."
                   + ("" if eq else " Fixed income signals compute on **yields** — a buy read "
                      "means rising yields, which trades as **selling** the bond/STIR future."))
        for _sname in ([only] if only else tascore.TA_STRATEGIES):
            _doc = TRIGGER_DOCS.get(_sname)
            if _doc is None or (eq and _sname == "Mean Reversion"):
                continue
            _in_set = " · *in your current picks*" if (picked and _sname in picked) else ""
            if not only:
                st.markdown(f"**{_sname}** — {tascore.axis_of(_sname)} axis{_in_set}")
            st.markdown(f"- ▲ **Buys:** {_doc['buys']}\n- ▼ **Sells:** {_doc['sells']}")


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
    if IS_ADMIN and cc1.button("📌 Set as default", key=f"conv_defaults_save{k}",
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
        if IS_ADMIN and _rx1.button("💾 Save as default", key=f"rep_excl_save{k}",
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

    if not eq:
        # --- Weekly Signal Scorecard — the TA book's track record, filed with its reports ---
        sc1, sc2 = st.columns([1, 3])
        _ledger_ok = (ROOT / "data" / "signal_cache" / "ledger_outcomes.parquet").exists()
        if sc1.button("🗂️ Generate Weekly Signal Scorecard (PDF)", key="ta_scorecard",
                      disabled=not _ledger_ok):
            with st.spinner("Building the Weekly Signal Scorecard…"):
                _sc_out = ROOT / "data" / "Weekly_Signal_Scorecard.pdf"
                _sc = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "sigscore.py"), str(_sc_out)],
                    capture_output=True, text=True, timeout=600)
                if _sc.returncode == 0 and _sc_out.exists():
                    st.session_state["ta_scorecard_pdf"] = _sc_out.read_bytes()
                else:
                    st.error("Scorecard failed:\n\n"
                             + (_sc.stderr or _sc.stdout or "no output")[-2000:])
        if st.session_state.get("ta_scorecard_pdf"):
            st.download_button("⬇️ Download Weekly_Signal_Scorecard.pdf",
                               data=st.session_state["ta_scorecard_pdf"],
                               file_name="Weekly_Signal_Scorecard.pdf", mime="application/pdf",
                               key="ta_scorecard_dl")
            email_report_ui("ta_sc_email", "sigscore", st.session_state["ta_scorecard_pdf"],
                            subject="BASIS — Weekly Signal Scorecard",
                            attachment_name="Weekly_Signal_Scorecard.pdf")
        sc2.caption("The **Weekly Signal Scorecard** — the client-facing track record of THIS page's "
                    "signals: the week's flagged signals, the verdicts that landed on earlier calls at "
                    "5/10/21 sessions (hits and misses), the era league, the strategy × product hit-rate "
                    "map and a watch list where a live composite read meets a real track record. Data "
                    "from the 📒 Signal Ledger; also available as a scheduled Monday email "
                    "(Alert Settings). Ad-hoc builds here never roll the weekly-delta baseline.")


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
                                elliott_wave as _ew, ichimoku as _ich, obv as _obv, mfi as _mfi,
                                donchian as _dc, aroon as _ar)
    strset = set(strset_key)
    out = {"pf": None, "flag": None, "sr_levels": [], "fib_levels": [], "retest_level": None,
           "mom": None, "elliott": None, "ichimoku": None, "obv": None, "mfi": None,
           "donchian": None, "aroon": None}
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
    if "Donchian Channel" in strset and out["pf"] is not None:   # price-axis channel → same series as chart
        try:
            _dcd, _di = _dc.donchian_chart_data(tk, history=pd.DataFrame({tk: out["pf"]}))
            if _dcd is not None and not _dcd.empty:
                out["donchian"] = _dcd[["date", "upper", "lower"]].dropna()
        except Exception:
            pass
    if "Aroon" in strset and out["pf"] is not None:             # own 0–100 panel (Aroon Up/Down)
        try:
            _acd, _ai = _ar.aroon_chart_data(tk, history=pd.DataFrame({tk: out["pf"]}))
            if _acd is not None and not _acd.empty:
                out["aroon"] = _acd[["date", "aroon_up", "aroon_down"]].dropna()
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
                                elliott_wave as _ew, ichimoku as _ich, obv as _obv, mfi as _mfi,
                                donchian as _dc, aroon as _ar)
    strset = set(strset_key)
    out = {"pf": None, "flag": None, "sr_levels": [], "fib_levels": [], "retest_level": None,
           "mom": None, "elliott": None, "ichimoku": None, "obv": None, "mfi": None,
           "donchian": None, "aroon": None}
    close, vol = eqta.load_history_one(tk)     # column-pruned: ms, not a 37MB load
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
    if "Donchian Channel" in strset:                # price-axis channel bands (prior-N high/low)
        try:
            _dcd, _di = _dc.donchian_chart_data(tk, history=hist)
            if _dcd is not None and not _dcd.empty:
                out["donchian"] = _dcd[["date", "upper", "lower"]].dropna()
        except Exception:
            pass
    if "Aroon" in strset:                           # own 0–100 panel (Aroon Up/Down)
        try:
            _acd, _ai = _ar.aroon_chart_data(tk, history=hist)
            if _acd is not None and not _acd.empty:
                out["aroon"] = _acd[["date", "aroon_up", "aroon_down"]].dropna()
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

    _trigger_docs_expander("ficc")

    # Quick-nav row (top of page): open any strategy's own page — trigger control, full table, charts.
    st.caption("Open a strategy for its trigger control, full table and charts:")
    _ta_quicknav()

    df, meta = load_signals()
    if _all_filtered_off():                              # the Sectors & products filter hides everything
        st.warning("🗂️ **All sectors are switched off** in the Sectors & products filter (🏠 Home) — "
                   "nothing is enabled to analyse, so this page looks empty. Your data is fine.")
        if IS_ADMIN and st.button("🗂️  Turn all sectors back on", key="ta_filter_reset", type="primary"):
            universe.save_filter(set(), set())
            for _s in _sf_sections():
                st.session_state.pop(_s[3], None)        # clear stale empty pills so they re-seed on
            st.rerun()
        elif not IS_ADMIN:
            st.caption("Ask your admin to reset it, or visit Home — it self-heals on load.")
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
        if IS_ADMIN and _cs1.button("💾 Save as default", key="conf_save",
                       help="Persist this set — used by the weekly report and on every launch."):
            tascore.save_confluence_set(_conf or tascore.CONFLUENCE_DEFAULT)
            st.toast("Confluence set saved.", icon="🎯")
            # Re-score the Signal Ledger's COMPOSITE NOW (Ben's ask 2026-08-13) — it is
            # built from the SAVED set, so without this it would lag until the morning
            # snapshot's rebuild. rescore= re-derives only the Composite pseudo-strategy;
            # every per-strategy outcome stays frozen (the ledger is append-only).
            from src import sigledger as _sl
            if _sl.OUTCOMES_FILE.exists():
                with st.spinner("Re-scoring the Signal Ledger's Composite under the new set "
                                "(~1 min, whole 10y history)…"):
                    _sl.rebuild(log=lambda *_a, **_k: None, rescore=(_sl.CONFLUENCE,))
                st.toast("Signal Ledger rebuilt — its Composite row now tracks this set.",
                         icon="📒")
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
            if _g.get("donchian") is not None:              # Donchian channel — prior-N high/low bands
                _dch = _g["donchian"][_g["donchian"]["date"] >= win.index[0]]
                if not _dch.empty:
                    _db = alt.Chart(_dch).encode(x="date:T")
                    layers += [_db.mark_line(color="#5C6BC0", strokeDash=[5, 3], strokeWidth=1.7).encode(
                                   y=alt.Y("upper:Q", scale=alt.Scale(zero=False))),
                               _db.mark_line(color="#5C6BC0", strokeDash=[5, 3], strokeWidth=1.7).encode(
                                   y=alt.Y("lower:Q", scale=alt.Scale(zero=False)))]
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

            if _g.get("aroon") is not None:                 # Aroon Up/Down sub-panel (0–100)
                _ar = _g["aroon"][_g["aroon"]["date"] >= win.index[0]]
                if not _ar.empty:
                    _arl = _ar.melt("date", value_vars=["aroon_up", "aroon_down"],
                                    var_name="Line", value_name="val")
                    _arl["Line"] = _arl["Line"].map({"aroon_up": "Aroon Up", "aroon_down": "Aroon Down"})
                    _g50 = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
                        color=_cc["muted"], strokeDash=[4, 3]).encode(y="y:Q")
                    _arc = alt.Chart(_arl).mark_line(strokeWidth=1.9).encode(
                        x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                        y=alt.Y("val:Q", title="Aroon", scale=alt.Scale(domain=[0, 100]),
                                axis=alt.Axis(values=[0, 50, 100], labelFontSize=11)),
                        color=alt.Color("Line:N", scale=alt.Scale(domain=["Aroon Up", "Aroon Down"],
                                                                  range=[_cc["long"], _cc["short"]]),
                                        legend=alt.Legend(orient="top", title=None, labelFontSize=11)))
                    brand.show_chart(alt.layer(_g50, _arc).resolve_scale(y="shared").properties(
                        height=120, title="Aroon (25) — Up vs Down"))

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
               "the top when several strategies agree. The **Hot Sheet** covers the whole desk.")

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
        if _g.get("donchian") is not None:              # Donchian channel — prior-N high/low bands
            _dch = _g["donchian"][_g["donchian"]["date"] >= win.index[0]]
            if not _dch.empty:
                _db = alt.Chart(_dch).encode(x="date:T")
                layers += [_db.mark_line(color="#5C6BC0", strokeDash=[5, 3], strokeWidth=1.7).encode(
                               y=alt.Y("upper:Q", scale=alt.Scale(zero=False))),
                           _db.mark_line(color="#5C6BC0", strokeDash=[5, 3], strokeWidth=1.7).encode(
                               y=alt.Y("lower:Q", scale=alt.Scale(zero=False)))]
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

        if _g.get("aroon") is not None:                 # Aroon Up/Down sub-panel (0–100)
            _ar = _g["aroon"][_g["aroon"]["date"] >= win.index[0]]
            if not _ar.empty:
                _arl = _ar.melt("date", value_vars=["aroon_up", "aroon_down"],
                                var_name="Line", value_name="val")
                _arl["Line"] = _arl["Line"].map({"aroon_up": "Aroon Up", "aroon_down": "Aroon Down"})
                _g50 = alt.Chart(pd.DataFrame({"y": [50.0]})).mark_rule(
                    color=_cc["muted"], strokeDash=[4, 3]).encode(y="y:Q")
                _arc = alt.Chart(_arl).mark_line(strokeWidth=1.9).encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(labelFontSize=11)),
                    y=alt.Y("val:Q", title="Aroon", scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(values=[0, 50, 100], labelFontSize=11)),
                    color=alt.Color("Line:N", scale=alt.Scale(domain=["Aroon Up", "Aroon Down"],
                                                              range=[_cc["long"], _cc["short"]]),
                                    legend=alt.Legend(orient="top", title=None, labelFontSize=11)))
                brand.show_chart(alt.layer(_g50, _arc).resolve_scale(y="shared").properties(
                    height=120, title="Aroon (25) — Up vs Down"))


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
    _trigger_docs_expander("equities")

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
        if IS_ADMIN and st.button("💾 Save as default", key="conf_save_eq",
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
    _trigger_docs_expander("equities", only=strat)

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
        if IS_ADMIN and _td2.button("📌 Set default", key=f"eqthr_def_{strat}", use_container_width=True,
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


@st.cache_data(ttl=300, show_spinner=False)
def _health_bundle():
    """One gathered pass over every store the 🩺 Data health page reports (the engine
    re-reads parquets, so cache it briefly — the Refresh button clears it)."""
    frames = health.snapshot_frames()
    deep = health.deep_health()
    stale = health.stale_surfaces()
    caches = health.cache_health()
    board = health.checks(frames=frames, deep=deep, stale=stale, caches=caches)
    return frames, deep, stale, caches, board


def render_data_health() -> None:
    st.subheader("\U0001FA7A Data health")
    snap = _load_snap()
    df, _meta = load_signals()
    hc1, hc2 = st.columns([0.8, 0.2])
    hc1.caption("Every store the app rides — the morning snapshot, the deep price store, the "
                "vol surfaces, the accruing caches — checked in one place, so a bad pull shows "
                "up here instead of as a mysteriously thin report downstream.")
    if hc2.button("↻ Refresh checks", use_container_width=True, key="dh_refresh"):
        _health_bundle.clear()
    frames, deep, stale, caches, board = _health_bundle()

    # ---- status board — worst first --------------------------------------------------
    _sev = {"bad": 0, "warn": 1, "ok": 2}
    _icon = {"bad": st.error, "warn": st.warning, "ok": st.success}
    for c in sorted(board, key=lambda c: _sev.get(c["level"], 3)):
        _icon.get(c["level"], st.info)(f"**{c['area']}** — {c['message']}")

    st.markdown("##### Snapshot")
    if not snap:
        st.warning("No snapshot manifest — the app is on demo/mock data.")
    else:
        _created = health.parse_stamp(snap.get("created"))
        age_h = (pd.Timestamp.now(tz="UTC") - _created).total_seconds() / 3600 if _created is not None else None
        age_txt = ("—" if age_h is None
                   else f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h / 24:.1f} days ago")
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
    if not frames.empty:
        with st.expander(f"Per-frame detail — {len(frames)} files in data/snapshot/"):
            show = frames.copy()
            show["written"] = show["written"].map(lambda t: _to_et(t.isoformat()) if pd.notna(t) else "—")
            show.columns = ["Frame", "Rows", "Markets", "Last data date", "Written", "Age (h)", "MB"]
            brand.themed_dataframe(show, {"Age (h)": "{:.1f}", "MB": "{:.2f}"}, na_rep="—")
            st.caption("Core pull frames should share one *Written* stamp — a frame hours behind "
                       "the rest means that leg of the fetch failed and the old file was kept. "
                       "The history stores below them accrue daily and are covered in *Caches*.")

    st.markdown("##### Deep price store")
    cov = deep["coverage"]
    if cov.empty:
        st.info("Deep store is empty — the TA Backtester runs on the live feed's ~400 sessions "
                "until the next Bloomberg pull backfills it (src/deepstore.py, automatic).")
    else:
        n_uni = len(list(universe.INSTRUMENTS))
        d1 = st.columns(4)
        d1[0].metric("Tickers held", f"{len(cov)}/{n_uni}")
        d1[1].metric("Truncated", str(len(deep["truncated"])),
                     help=f"Held below {deep['min_days']} days — re-backfills on the next pull.")
        d1[2].metric("Median depth", f"{int(cov['days'].median()):,}d")
        d1[3].metric("Last settle", str(deep["store_last"].date()) if deep["store_last"] is not None else "—")
        _probs = cov[cov["flag"].ne("")]
        with st.expander(f"Per-ticker coverage — {len(_probs)} flagged"):
            show = cov.copy()
            show.columns = ["Ticker", "Market", "First", "Last", "Days", "Rolls", "Flag"]
            brand.themed_dataframe(show, {"Days": "{:,.0f}", "Rolls": "{:,.0f}"}, na_rep="—",
                                   height=420)
            st.caption("*truncated* = held below the self-heal floor (re-backfills next pull); "
                       "*stale* = trails the store and is silently skipped by the backtester's "
                       "deep-history overlay. Blank = healthy.")

    st.markdown("##### Vol surfaces")
    if stale.empty:
        st.success("No frozen or dead implied-vol surfaces — every market's vol book is live.")
    else:
        st.caption(f"**{stale['ticker'].nunique()} market(s)** with a stale surface — frozen at one "
                   "value or no longer publishing. The vol/skew/term strategies already leave these "
                   "unscored (`datafeed.stale_iv_reasons`); listed here so a dead surface is a known "
                   "fact, not a surprise.")
        show = stale.copy()
        show.columns = ["Ticker", "Market", "Surface", "Reason"]
        brand.themed_dataframe(show, {})

    st.markdown("##### Caches & history stores")
    if not caches.empty:
        show = caches.copy()
        show.columns = ["Store", "Last data date", "Products", "File age (h)"]
        brand.themed_dataframe(show, {"Products": "{:,.0f}", "File age (h)": "{:.1f}"}, na_rep="—")
        st.caption("Every accruing store the daily pull feeds. A *Last data date* stuck behind the "
                   "snapshot settle means that leg of the pull has been failing quietly.")

    st.markdown("##### Regression suite")
    tr = health.last_test_run()
    if not tr:
        st.info("Never run on this box. Run **run_tests.bat** (repo root), or push code — the "
                "pre-push git hook runs the suite and blocks the push if it's red.")
    else:
        r1 = st.columns(4)
        r1[0].metric("Result", "🟢 green" if tr.get("ok") else ("⏭ skipped" if tr.get("skipped") else "🔴 RED"))
        r1[1].metric("When", _to_et(str(tr.get("when", ""))) or "—")
        r1[2].metric("Tests", str(tr.get("summary", "—")))
        r1[3].metric("Duration", f"{tr.get('duration_s', 0):.0f}s")
        st.caption("Golden-file tests over the scoring/backtest/adjustment engines (tascore, tabt, "
                   "volbt, deepstore) — `tests\\` in the repo. They run automatically before any "
                   "push that touches code; re-baseline deliberate behaviour changes with "
                   "`run_tests.bat --regen`.")

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

    # ── frozen time axis: the 24h scale + a red "now" notch render as their own thin
    # chart inside a sticky wrapper pinned just below the fixed top bar, so the times
    # stay on screen however far down the (very tall) Gantt you scroll. Both charts
    # reserve an identical y gutter (minExtent=maxExtent) so their time columns line
    # up; the Gantt keeps its gridlines but drops its own axis labels.
    _yw = 214
    st.markdown(
        "<style>"
        "[data-testid='stLayoutWrapper']:has(> .st-key-mh_axis_sticky),"
        "div.st-key-mh_axis_sticky {"
        f"position:sticky; top:107px; z-index:5; background:{brand.palette()['canvas']};"
        "}</style>", unsafe_allow_html=True)
    with st.container(key="mh_axis_sticky"):
        _pad = alt.Chart(pd.DataFrame({"y": [""]})).mark_tick(opacity=0).encode(
            y=alt.Y("y:N", axis=alt.Axis(title=None, labels=False, ticks=False, domain=False,
                                         minExtent=_yw, maxExtent=_yw)))
        _now = alt.Chart(pd.DataFrame({"x": [now_h]})).mark_rule(color="#E53935", size=2).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 24], nice=False),
                    axis=alt.Axis(title=None, orient="top", values=list(range(0, 25, 2)), grid=False,
                                  labelExpr="(datum.value<10?'0':'')+datum.value+':00'")))
        brand.show_chart(alt.layer(_pad, _now).properties(height=8))

    xaxis = alt.X("start:Q", scale=alt.Scale(domain=[0, 24], nice=False),
                  axis=alt.Axis(title=None, values=list(range(0, 25, 2)), grid=True,
                                labels=False, ticks=False, domain=False))
    yaxis = alt.Y("mkt:N", sort=y_order, axis=alt.Axis(title=None, labelFontSize=9, labelLimit=200,
                                                       minExtent=_yw, maxExtent=_yw))
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
        disabled=True if not IS_ADMIN else ["sector", "product", "ticker", "exchange"],
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
    if IS_ADMIN and bc1.button("💾 Save block sizes", type="primary", key="save_blocksizes_btn"):
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
            disabled=True if not IS_ADMIN else ["ticker", "product"],
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
        if IS_ADMIN and st.button("💾 Save CTD assumptions", key="fy_ctd_save"):
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

    # ---- Failure alerts: who is emailed when a scheduled report CRASHES (not the report lists) ----
    from src import failalert
    st.markdown("#### ⚠️ Failure alerts")
    st.caption("If a scheduled report **fails to build or send**, an alert email with the error goes to "
               "these addresses (at most one per report per day). This list is separate from the report "
               "recipient lists below — failures are internal, they never go to clients.")
    fa = failalert.load_recipients()
    for i, addr in enumerate(fa):
        c1, c2 = st.columns([0.85, 0.15])
        c1.write(addr)
        if c2.button("Remove", key=f"fa_rm_{i}", use_container_width=True,
                     disabled=len(fa) == 1,
                     help="At least one address must stay on the list — otherwise failures go unseen."
                          if len(fa) == 1 else None):
            fa.pop(i)
            failalert.save_recipients(fa)
            st.rerun()
    with st.form(key="fa_addform", clear_on_submit=True):
        fc1, fc2 = st.columns([0.78, 0.22])
        fa_new = fc1.text_input("Add failure-alert address", label_visibility="collapsed",
                                placeholder="name@firm.com")
        fa_add = fc2.form_submit_button("➕ Add", use_container_width=True)
    if fa_add:
        e = (fa_new or "").strip()
        if "@" not in e or "." not in e.split("@")[-1]:
            st.warning("Enter a valid email address (e.g. name@firm.com).")
        elif e in fa:
            st.info(f"{e} is already on the list.")
        else:
            fa.append(e)
            failalert.save_recipients(fa)
            st.rerun()
    st.divider()

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


def _opec_edition_from_pdf(path):
    """(MonthName, Year) for an uploaded MOMR PDF, read from its cover text; None if unreadable."""
    try:
        import pypdfium2 as _pdf
        doc = _pdf.PdfDocument(str(path))
        txt = "\n".join(doc[i].get_textpage().get_text_range() for i in range(min(4, len(doc))))
        months = ("January February March April May June July August September October "
                  "November December").split()
        m = re.search(r"Oil Market Report\s*[–—-]\s*(" + "|".join(months) + r")\s+(\d{4})", txt)
        if m:
            return m.group(1), m.group(2)
        m = re.search(r"\b(" + "|".join(months) + r")\s+(\d{4})", txt)   # fallback: first Month Year
        return (m.group(1), m.group(2)) if m else None
    except Exception:
        return None


def render_opec() -> None:
    from src import release_cal
    st.subheader("\U0001F6E2️ OPEC Monthly Oil Market Report")
    st.caption("Each month the desk gets a one-page synopsis + chart deck of OPEC's Monthly Oil Market "
               "Report. OPEC now gates the PDF behind a registration form and a bot-check, so the "
               "**download is a quick manual step**; everything after — building the branded synopsis "
               "and emailing the recipient list — is automatic once the PDF lands in the inbox.")

    last = OPEC_MARKER.read_text(encoding="utf-8").strip() if OPEC_MARKER.exists() else "—"
    pdfs = sorted(OPEC_DIR.glob("out/OPEC_MOMR_Synopsis_*.pdf"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    today = datetime.now(ZoneInfo("America/New_York")).date()
    nopec = next((r["opec"] for r in release_cal.next_12_months(today) if r["opec_upcoming"]), None)
    m1, m2 = st.columns(2)
    m1.metric("Last edition emailed", last)
    m2.metric("Next OPEC release", f"{nopec:%a %d %b}" if nopec else "TBC")
    st.caption(f"**2026 release calendar:** {OPEC_2026_DATES}  ·  ~04:00 ET (10:00 Vienna). Full "
               "OPEC/EIA/IEA schedule on the Fundamental Reports Calendar page.")
    st.divider()

    if IS_ADMIN:
        # --- recipient list (the "opec" report key) -----------------------------------
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

        # --- upload this month's PDF --------------------------------------------------
        st.markdown("#### This month's report")
        st.caption("Download the latest MOMR from opec.org (fill OPEC's short form → **Download PDF**), "
                   "then drop the file here. I detect the edition and save it to the inbox; then hit "
                   "**Build & send** below. The scheduled job also picks it up on its next run.")
        up = st.file_uploader("Upload the MOMR PDF", type=["pdf"], key="opec_upload",
                              label_visibility="collapsed")
        if up is not None:
            try:
                inbox = OPEC_DIR / "inbox"; inbox.mkdir(parents=True, exist_ok=True)
                tmp = inbox / "_uploaded.pdf"; tmp.write_bytes(up.getvalue())
                ed = _opec_edition_from_pdf(tmp)
                if ed:
                    mon, yr = ed
                    dest = inbox / f"MOMR_{mon}{yr}.pdf"; tmp.replace(dest)
                    st.success(f"Saved **{mon} {yr}** to the inbox ({dest.name}). Use **Build & send** below.")
                else:
                    tmp.unlink(missing_ok=True)
                    st.warning("Couldn't read the edition month from that PDF — is it the MOMR report file?")
            except Exception as e:
                st.error(f"Upload failed: {e}")
        st.divider()

        # --- actions ------------------------------------------------------------------
        st.markdown("#### Build & send")
        st.caption("Builds the synopsis from the newest MOMR PDF in the inbox and emails it — no fetch, "
                   "no Chrome. The scheduled job does the same automatically once a new PDF is in the inbox.")
        a1, a2, a3 = st.columns(3)
        if a1.button("📤 Build & send latest", type="primary", use_container_width=True,
                     help="Build from the newest inbox MOMR and email the recipient list now."):
            _run_opec(["--force-send"], "Building the synopsis and sending…")
        if a2.button("👁️ Rebuild preview (no send)", use_container_width=True,
                     help="Rebuild the PDF from the newest inbox report without emailing."):
            _run_opec(["--from-inbox", "--dry-run"], "Rebuilding the synopsis…", timeout=120)
        desk1 = (recipients.get("opec") or ["benjamin.goulson@xpi.com.br"])[0]
        if a3.button("✉️ Send test to me", use_container_width=True,
                     help=f"Build from the inbox, then email only {desk1}."):
            _run_opec(["--force-send", "--to", desk1], f"Building and sending a test to {desk1}…")
        if st.session_state.get("opec_log"):
            with st.expander("Last run log", expanded=False):
                st.code(st.session_state["opec_log"][-4000:])

    if pdfs:
        st.download_button("⬇️  Download the latest synopsis PDF", data=pdfs[0].read_bytes(),
                           file_name=pdfs[0].name, mime="application/pdf")


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

    if IS_ADMIN:
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
        if st.session_state.get("pm_log"):
            with st.expander("Last run log", expanded=False):
                st.code(st.session_state["pm_log"][-4000:])

    if PM_PDF.exists():
        st.download_button("⬇️  Download the latest monitor PDF", data=PM_PDF.read_bytes(),
                           file_name=PM_PDF.name, mime="application/pdf")

    # --- release synopses (WGC GDT / WPIC PQ) -----------------------------------------
    st.divider()
    st.markdown("#### Release synopses — WGC Gold Demand Trends · WPIC Platinum Quarterly")
    st.caption("A daily job watches for new editions and emails a one-page synopsis on "
               "release day (toggle on the Recipients page). Latest built synopses:")
    if IS_ADMIN:
        r1, r2 = st.columns(2)
        if r1.button("🔎 Check releases & rebuild (no send)", use_container_width=True,
                     help="Detect the latest editions, fetch, parse and rebuild the synopses."):
            _run_pm_rel(["--dry-run"], "Checking WGC / WPIC and rebuilding synopses…")
        if r2.button("✉️ Email latest synopses to me", use_container_width=True,
                     help=f"Rebuild and email the current editions to {desk1}."):
            _run_pm_rel(["--force-send", "--to", desk1], f"Building and sending to {desk1}…")
        if st.session_state.get("pm_rel_log"):
            with st.expander("Last synopsis run log", expanded=False):
                st.code(st.session_state["pm_rel_log"][-4000:])
    rel_pdfs = sorted(PM_REL_DIR.glob("*_Synopsis.pdf"), key=lambda p: p.stat().st_mtime,
                      reverse=True)[:4]
    if rel_pdfs:
        cols = st.columns(len(rel_pdfs))
        for col, p in zip(cols, rel_pdfs):
            col.download_button(f"⬇️ {p.stem.replace('_', ' ')}", data=p.read_bytes(),
                                file_name=p.name, mime="application/pdf", key=f"pmrel_{p.name}")

    pages = _pm_page_images()
    if pages:
        st.divider()
        for p in pages:
            st.image(str(p), use_container_width=True)
    else:
        st.info("No report built yet — use “Rebuild preview” above.")


# ── Brazil Production (FICC → Fundamentals) ──────────────────────────────────
# Two layers per commodity: how much Brazil produces against the world (pulled free
# from USDA PS&D / EIA, or curated from USGS where no free feed exists), then which
# companies produce Brazil's share (curated — physical output per company is not in
# any free feed, and Brazilian producers are not in the equity universe). The company
# tables state their BASIS on the page because an export or crush share must never be
# read as a production share. Engine + the honest-data rules: src/brazilprod.py.
BRAZIL_PDF = ROOT / "reports" / "Brazil_Production.pdf"


@st.cache_data(ttl=1800, show_spinner=False)
def _brazil_store() -> dict:
    return brazilprod.load_or_build()


@st.cache_data(ttl=900, show_spinner=False)
def _brazil_quotes(tickers: tuple) -> dict:
    """{bloomberg ticker: (last, pct)} for the listed producers, off free Yahoo. The
    B3 lines resolve through yfin's new BZ -> .SA mapping; anything unmappable (the
    HK and Santiago lines, the private groups) simply comes back empty."""
    if not tickers:
        return {}
    try:
        from src import yfin
        q = yfin.get_quotes([f"{t} Equity" for t in tickers])
    except Exception:
        return {}
    out = {}
    for t in tickers:
        key = f"{t} Equity"
        if key in q.index:
            last, pct = q.loc[key, "last"], q.loc[key, "pct"]
            if pd.notna(last):
                out[t] = (float(last), None if pd.isna(pct) else float(pct))
    return out


def _brazil_share_chart(com: dict, cc: dict):
    """Top producing countries, Brazil in gold and everyone else muted."""
    import altair as alt
    rows = com.get("countries") or []
    if not rows:
        return None
    d = pd.DataFrame(rows)
    # Sorted by size but with the "Other" bucket pinned last — mid-table it reads as
    # if it were a country.
    order = (d[~d["is_other"]].sort_values("value", ascending=False)["country"].tolist()
             + d[d["is_other"]]["country"].tolist())
    # Three-way colouring needs a scale — alt.condition takes one test, not a chain.
    d["kind"] = np.where(d["is_brazil"], "Brazil",
                         np.where(d["is_other"], "Other", "Producer"))
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("value:Q", title=f"{com['year_label']} production ({com['unit']})"),
        y=alt.Y("country:N", title=None, sort=order),
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["Brazil", "Producer", "Other"],
                                        range=[cc["accent"], cc["series"], cc["muted"]])),
        opacity=alt.condition("datum.is_brazil", alt.value(1.0), alt.value(0.55)),
        tooltip=[alt.Tooltip("country:N", title="Country"),
                 alt.Tooltip("value:Q", title=f"Production ({com['unit']})", format=",.2f"),
                 alt.Tooltip("share:Q", title="Share of world", format=".2f")],
    ).properties(height=max(220, 26 * len(d)))


def _brazil_history_chart(com: dict, cc: dict):
    """Brazil's share of world production over time — the 'is this gaining or
    losing ground?' read. Only the PS&D and EIA sources carry history."""
    import altair as alt
    hist = com.get("history") or []
    if len(hist) < 3:
        return None
    d = pd.DataFrame(hist)
    line = alt.Chart(d).mark_line(color=cc["accent"], strokeWidth=2.4).encode(
        x=alt.X("year:O", title=None,
                axis=alt.Axis(values=[y for y in d["year"] if y % 5 == 0])),
        y=alt.Y("share:Q", title="Brazil's share of world (%)",
                scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("year:O", title="Year"),
                 alt.Tooltip("brazil:Q", title=f"Brazil ({com['unit']})", format=",.2f"),
                 alt.Tooltip("world:Q", title=f"World ({com['unit']})", format=",.2f"),
                 alt.Tooltip("share:Q", title="Share", format=".2f")])
    return line.properties(height=230)


def _brazil_company_chart(blk: dict, cc: dict):
    import altair as alt
    rows = [r for r in blk["rows"]]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    order = d["company"].tolist()          # already sorted: companies, artisanal, Other
    # Three-way: companies in gold, a non-corporate producer (garimpo) in blue so it can
    # never be mistaken for a company, the Other bucket muted.
    d["kind_lbl"] = np.where(d.get("is_artisanal", False), "Not a company",
                             np.where(d["is_other"], "Other", "Company"))
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("share_brazil:Q", title=blk.get("axis_label") or "share of Brazil (%)"),
        y=alt.Y("company:N", title=None, sort=order),
        color=alt.Color("kind_lbl:N", legend=None,
                        scale=alt.Scale(domain=["Company", "Not a company", "Other"],
                                        range=[cc["accent"], cc["series"], cc["muted"]])),
        opacity=alt.condition("datum.is_other", alt.value(0.45), alt.value(0.9)),
        tooltip=[alt.Tooltip("company:N", title="Company"),
                 alt.Tooltip("volume:Q", title=f"Volume ({blk['unit']})", format=",.2f"),
                 alt.Tooltip("share_brazil:Q", title="Share of Brazil", format=".2f"),
                 alt.Tooltip("share_world:Q", title="Share of WORLD", format=".2f")],
    ).properties(height=max(200, 27 * len(d)))


def _brazil_hedge_section(com: dict, blk: dict | None) -> None:
    """How many lots would hedge a year of this commodity's Brazilian output, nationally
    and per producer. Deliberately a SEPARATE section from the production table: the
    production numbers are measurements, these are a derived what-if built on a stack of
    conversions, and the two shouldn't be read with the same confidence."""
    h = com.get("hedge") or {}
    st.markdown("##### Hedging a year of that output")
    if not h.get("available"):
        st.info(f"**No hedge sized for {com['label'].lower()}.** {h.get('reason', '')}")
        return

    basis_word = "exports" if h["qty_basis"] == "exports" else "production"
    m1, m2, m3 = st.columns(3)
    m1.metric("Contract", h["ticker"].split()[0],
              help=f"{h['name']} — {h['size']:,} {h['size_unit']} per lot.")
    m2.metric("Lots to hedge all of Brazil", f"{h['national_lots']:,}",
              help=f"Brazil's annual {basis_word} of {h['national_qty']:,.2f} "
                   f"{h['national_unit']}, hedged in full.")
    m3.metric("Spread over 12 months", f"{h['national_lots_per_month']:,} / mth",
              help="The same hedge laid evenly across a 12-month strip, which is closer to "
                   "how it would actually be executed than one giant front-month clip.")

    if h["proxy"]:
        st.warning(f"**Cross hedge — {h['name']} is a proxy, not a match.** {h['note']}")
    elif h["note"]:
        st.caption(h["note"])

    if blk and h.get("rows"):
        hd = pd.DataFrame(h["rows"])
        show = hd.assign(**{
            "% of Brazil": hd["share_brazil"].map("{:.1f}".format),
            "Lots (1 yr)": hd["lots"].map("{:,}".format),
            "Lots / month": hd["lots_per_month"].map("{:,}".format),
        }).rename(columns={"company": blk.get("entity_label") or "Company"})
        st.dataframe(show[[blk.get("entity_label") or "Company", "% of Brazil",
                           "Lots (1 yr)", "Lots / month"]],
                     use_container_width=True, hide_index=True)

    st.caption(f"Each producer's hedgeable volume is its **share of Brazil applied to Brazil's "
               f"national {basis_word}** ({h['national_qty']:,.2f} {h['national_unit']}), so the "
               f"rows add back to the national figure. One lot is **{h['size']:,} "
               f"{h['size_unit']}**. This is the notional size of a *full* hedge of a *whole* "
               f"year — a producer would hedge a fraction of that, rolling a strip, and would "
               f"still carry quality, location and timing basis the flat price does not cover.")


def render_brazil_production() -> None:
    import altair as alt
    st.subheader("🇧🇷 Brazil Production")
    st.caption("What Brazil produces, how much of the world's supply that is, who else "
               "produces it — and which companies produce Brazil's share. Country data is "
               "free and refreshes with the daily pull (USDA PS&D, EIA); the metals, pulp "
               "and company tables are hand-maintained, and every company block states what "
               "it actually measures.")

    store = _brazil_store()
    coms = store.get("commodities") or {}
    if not coms:
        st.warning("No Brazil production store yet — run the daily pull, or use “Rebuild now” below.")

    cc = brand.chart_colors()
    b1, b2, b3 = st.columns(3)
    b1.metric("Store built", store.get("built") or "—")
    b2.metric("Commodities", len(coms))
    b3.metric("Curated tables as of", store.get("curated_as_of") or "—")

    warn = [e for e in (store.get("errors") or []) if e.get("level") == "warning"]
    hard = [e for e in (store.get("errors") or []) if e.get("level") != "warning"]
    if hard:
        st.error("Sources that failed: " + "; ".join(f"**{e['label']}** — {e['error']}" for e in hard))
    if warn:
        st.warning("Curated table needs a look: "
                   + "; ".join(f"**{e['label']}** — {e['error']}" for e in warn))

    a1, a2 = st.columns(2)
    if IS_ADMIN and a1.button("🔄 Rebuild now", key="brz_rebuild", use_container_width=True,
                              help="Re-download the USDA PS&D and EIA data and re-read the "
                                   "curated tables. Normally the daily pull does this."):
        with st.spinner("Rebuilding Brazil production…"):
            try:
                brazilprod.build(force=True)
                _brazil_store.clear()
                st.success("Rebuilt.")
                st.rerun()
            except Exception as exc:
                st.error(f"Rebuild failed — {type(exc).__name__}: {exc}")
    if coms and a2.button("📈 Generate PDF report", key="brz_pdf_btn", use_container_width=True,
                          type="primary",
                          help="A branded client PDF: the ranked share-of-world chart on the front "
                               "page, then one page per commodity — Brazil's share of world "
                               "production and, where the industry is concentrated enough to have "
                               "an answer, which companies produce what share of Brazil's own "
                               "output. Each page prints what its table measures."):
        with st.spinner("Building the Brazil Production report…"):
            try:
                BRAZIL_PDF.parent.mkdir(parents=True, exist_ok=True)
                res = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "brazilreport.py"), str(BRAZIL_PDF)],
                    capture_output=True, text=True, timeout=600, cwd=str(ROOT))
                if res.returncode == 0 and BRAZIL_PDF.exists():
                    st.session_state["brz_pdf_ready"] = True
                else:
                    st.error("Report build failed.")
                    st.code((res.stderr or res.stdout or "")[-3000:])
            except Exception as exc:
                st.error(f"Report build failed — {type(exc).__name__}: {exc}")
    if st.session_state.get("brz_pdf_ready") and BRAZIL_PDF.exists():
        st.download_button("⬇️  Download Brazil_Production.pdf", data=BRAZIL_PDF.read_bytes(),
                           file_name=BRAZIL_PDF.name, mime="application/pdf", key="brz_pdf_dl")

    if not coms:
        return

    # ── 1. the whole book at a glance ────────────────────────────────────────
    st.divider()
    st.markdown("#### Where Brazil sits in world supply")
    head = brazilprod.headline_rows(store)
    # Clean field names for Vega-Lite — '%' and spaces in a field name are a trap.
    hd = head.rename(columns={"Commodity": "commodity", "Share %": "share", "Brazil": "brazil",
                              "World": "world", "Unit": "unit", "Year": "yr", "Rank": "rank"})
    bars = alt.Chart(hd).mark_bar(color=cc["accent"]).encode(
        x=alt.X("share:Q", title="Brazil's share of world production (%)"),
        y=alt.Y("commodity:N", title=None, sort=hd["commodity"].tolist()),
        tooltip=[alt.Tooltip("commodity:N", title="Commodity"),
                 alt.Tooltip("yr:N", title="Year"),
                 alt.Tooltip("brazil:Q", title="Brazil", format=",.2f"),
                 alt.Tooltip("world:Q", title="World", format=",.2f"),
                 alt.Tooltip("unit:N", title="Unit"),
                 alt.Tooltip("share:Q", title="Share of world", format=".1f"),
                 alt.Tooltip("rank:Q", title="World rank")])
    brand.show_chart(bars.properties(height=max(280, 26 * len(hd))))
    st.dataframe(
        head.assign(**{"Brazil": head["Brazil"].map("{:,.2f}".format),
                       "World": head["World"].map("{:,.2f}".format),
                       # spelled out — "Share %" alone never says a share OF WHAT
                       "% of world production": head["Share %"].map("{:.1f}%".format),
                       "Rank": head["Rank"].map(lambda r: f"#{int(r)}" if pd.notna(r) else "—"),
                       "Companies": head["Companies"].map({True: "✓", False: "—"})})
            [["Commodity", "Group", "Year", "Brazil", "World", "Unit",
              "% of world production", "Rank", "Companies"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Brazil": st.column_config.TextColumn(help="Brazil's production, in the row's unit."),
            "World": st.column_config.TextColumn(help="World production, same year and unit."),
            "% of world production": st.column_config.TextColumn(
                help="Brazil's production as a percentage of the world's."),
            "Companies": st.column_config.TextColumn(
                "Co. table", help="A company-level breakdown exists for this commodity.")})
    st.caption("**% of world production** is Brazil's production divided by world production in the same year "
               "and unit. Agricultural years are USDA marketing years, not calendar years, so a "
               "2025/26 crop and a 2024 mining figure are not the same window — each row states "
               "its own. **Rank** counts every reporting country.")

    # ── 2. the commercial view: lots per prospective client ─────────────────
    # Brokerage is earned per lot, so the number that sizes the opportunity is the
    # lot count a client would trade hedging its WHOLE production — summed across
    # every commodity it produces, not read off one commodity's page. Internal only:
    # this deliberately does not go in the client PDF.
    st.divider()
    st.markdown("#### Brokerage potential — lots by client")
    st.caption("Every producer above, rolled up across all the commodities they make, sized as "
               "the lots a **full hedge of a full year's output** would require. Brokerage is "
               "per lot, so this is the addressable ticket count, not a notional. "
               "‘Other’ buckets, non-corporate production and multi-company lines are excluded — "
               "there is nobody to call.")

    tc1, tc2, tc3 = st.columns([1, 1, 2])
    turns = tc1.number_input("Round-turns per lot per year", min_value=0.5, max_value=12.0,
                             value=1.0, step=0.5, key="brz_turns",
                             help="1.0 = the hedge goes on once and is held to expiry. A producer "
                                  "rolling a strip trades the same position again at every roll "
                                  "and pays brokerage each time, so raise this to model that. "
                                  "It is an assumption, not a measurement.")
    book = brazilprod.broker_book(store, turns=float(turns))
    if book.empty:
        st.info("No hedgeable production in the store yet.")
    else:
        tc2.metric("Prospective clients", f"{len(book):,}")
        tc3.metric("Total addressable lots / year", f"{book['Lots (1 yr)'].sum():,.0f}",
                   help="Sum across every named producer. Commodities with no listed hedge "
                        "(niobium, pulp, nickel, manganese, bauxite) contribute nothing.")

        top = book.head(15).copy()
        st.markdown("**Top 15 by annual lots**")
        brand.show_chart(alt.Chart(top).mark_bar(color=cc["accent"]).encode(
            x=alt.X("Lots (1 yr):Q", title="lots per year, full hedge of full output"),
            y=alt.Y("Client:N", title=None, sort=top["Client"].tolist()),
            tooltip=[alt.Tooltip("Client:N"), alt.Tooltip("Commodities:N"),
                     alt.Tooltip("Lots (1 yr):Q", format=","),
                     alt.Tooltip("Lots / month:Q", format=",")],
        ).properties(height=max(240, 26 * len(top))))

        st.dataframe(book.assign(**{
            "Lots (1 yr)": book["Lots (1 yr)"].map("{:,}".format),
            "Lots / month": book["Lots / month"].map("{:,}".format)})
            [["Client", "Commodities", "Lots (1 yr)", "Lots / month", "detail"]],
            use_container_width=True, hide_index=True,
            column_config={"detail": st.column_config.TextColumn(
                "Breakdown", help="Lots per commodity and the contract they trade in.")})
        # Company x product detail — the call-sheet view. A company that makes three
        # things gets three rows, because each is a different contract and a different
        # conversation.
        st.markdown("**By company and product** &nbsp;·&nbsp; lots at each hedge ratio")
        h1, h2 = st.columns([1, 3])
        show_unhedgeable = h1.checkbox("Include products with no future", value=True,
                                       key="brz_show_nohedge",
                                       help="Suzano's pulp, CBMM's niobium and Vale's nickel have "
                                            "no listed hedge. Shown with blank lot columns so a "
                                            "producer never looks overlooked when it is simply "
                                            "unhedgeable.")
        mat = brazilprod.hedge_matrix(store, turns=float(turns),
                                      include_unhedgeable=show_unhedgeable)
        if not mat.empty:
            # Each ratio spans three sub-columns (yr / mth / day). Streamlit has no
            # grouped headers, so the ratio is carried on the first of the three and
            # the period on each — "100% · yr", "· mth", "· day".
            _period_word = {"yr": "year", "mth": "month", "day": "trading day"}
            # Labels must be UNIQUE — a shared "· mth" across the four ratio groups gives
            # the frame duplicate column names, which breaks the Styler outright.
            ratio_cols, col_cfg = [], {}
            for pct in brazilprod.HEDGE_RATIOS:
                for suffix, _div in brazilprod.HEDGE_PERIODS:
                    label = f"{pct}% {suffix}"
                    ratio_cols.append((f"{pct}% {suffix}", label))
                    col_cfg[label] = st.column_config.TextColumn(
                        label, help=f"Lots to hedge {pct}% of that year's output, "
                                    f"per {_period_word[suffix]}.")
            disp = mat.assign(**{
                "Annual production": [f"{v:,.2f} {u}" for v, u in
                                      zip(mat["Annual production"], mat["Unit"])]})
            for src, label in ratio_cols:
                disp[label] = mat[src].map(lambda x: "—" if pd.isna(x) else f"{x:,.0f}")
            # TOTAL pinned to the foot — the whole addressable book in one line.
            tot = brazilprod.hedge_totals(mat)
            total_row = {"Company": "TOTAL", "Product": f"{tot['_n_hedgeable']} hedgeable lines",
                         "Annual production": "", "Contract": ""}
            for src, label in ratio_cols:
                total_row[label] = f"{tot.get(src, 0):,.0f}"
            disp = pd.concat([disp, pd.DataFrame([total_row])], ignore_index=True)
            h2.metric("Company × product rows", f"{len(mat):,}",
                      help=f"{mat['Company'].nunique()} companies; "
                           f"{int((~mat['_avail']).sum())} lines have no listed hedge.")
            # One hue per period (year / month / day) so a row can be tracked across
            # twelve numeric columns. Colours come from brazilprod so the page, the PDF
            # and the email table cannot drift apart.
            pal = (brazilprod.PERIOD_COLOUR_DARK if brand.theme() == "dark"
                   else brazilprod.PERIOD_COLOUR)
            shown_cols = (["Company", "Product", "Annual production", "Contract"]
                          + [lbl for _s, lbl in ratio_cols])
            styler = disp[shown_cols].style
            for (src, lbl) in ratio_cols:
                suffix = src.split()[-1]
                styler = styler.set_properties(subset=[lbl], **{"color": pal[suffix]})
            st.dataframe(styler, use_container_width=True, hide_index=True,
                         column_config=col_cfg)
            st.caption(f"Colour marks the period, not the value: "
                       f"<span style='color:{pal['yr']}'><b>per year</b></span> · "
                       f"<span style='color:{pal['mth']}'><b>per month</b></span> · "
                       f"<span style='color:{pal['day']}'><b>per trading day</b></span>.",
                       unsafe_allow_html=True)
            st.download_button(
                "⬇️  Download as CSV", key="brz_matrix_csv",
                data=mat.drop(columns=["_lots", "_avail"]).to_csv(index=False).encode("utf-8"),
                file_name="Brazil_hedge_by_company_product.csv", mime="text/csv",
                help="The full table, for working up a call list off-app.")
            st.caption("**Contract** is the future each line would hedge in — a multi-product "
                       "company trades several, which is the point of splitting the rows. "
                       "A product qualifier in brackets says what the share measures: "
                       "*exports* for the trade houses, *share by processing* for the meat "
                       "packers, *equity share* for the oil partners. Poultry carries no meat "
                       "future, so JBS and BRF are sized on the **feed** hedge (corn + soybean "
                       "meal) against their bird output.")

        st.caption("A client's lots are its share of Brazil applied to Brazil's national "
                   "output, converted at each contract's size. Poultry has no meat future, so "
                   "the integrators (JBS Seara, BRF) are sized on their **feed** hedge — corn "
                   "and soybean meal — which is where their brokerage actually sits. Treat the "
                   "whole table as an upper bound on ticket count: it assumes 100% of output is "
                   "hedged, on-exchange, through one broker.")

    # ── 3. one commodity in depth ───────────────────────────────────────────
    st.divider()
    st.markdown("#### A commodity in depth")
    ordered = [k for g in (store.get("group_order") or [])
               for k in sorted(coms, key=lambda x: -coms[x]["share"]) if coms[k]["group"] == g]
    labels = {k: f"{coms[k]['icon']} {coms[k]['label']}  ·  {coms[k]['group']}" for k in ordered}
    pick = st.selectbox("Commodity", ordered, format_func=lambda k: labels[k], key="brz_pick")
    com = coms[pick]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Brazil produces ({com['unit']})", f"{com['brazil']:,.2f}")
    m2.metric(f"World produces ({com['unit']})", f"{com['world']:,.2f}")
    m3.metric("Brazil's share", f"{com['share']:.1f}%")
    m4.metric("World rank", f"#{com['rank']}" if com.get("rank") else "—",
              help=f"Out of {com['n_producers']} reporting producers.")

    hist = _brazil_history_chart(com, cc)
    if hist is not None:
        cl, cr = st.columns([3, 2])
        with cl:
            st.markdown("**Who produces it** &nbsp;·&nbsp; Brazil in gold")
            ch = _brazil_share_chart(com, cc)
            if ch is not None:
                brand.show_chart(ch)
        with cr:
            st.markdown("**Brazil's share over time**")
            brand.show_chart(hist)
    else:
        st.markdown("**Who produces it** &nbsp;·&nbsp; Brazil in gold")
        ch = _brazil_share_chart(com, cc)
        if ch is not None:
            brand.show_chart(ch)

    src_bits = [f"Source: **{com['source_label']}**", f"year **{com['year_label']}**"]
    if com["src"] == "curated":
        src_bits.append("hand-maintained — refreshed annually, no free machine-readable feed exists")
    st.caption(" · ".join(src_bits) + ".")
    if com.get("note"):
        st.info(com["note"])

    # ── 4. who inside Brazil produces it ────────────────────────────────────
    st.divider()
    blk = com.get("companies")
    st.markdown(f"#### Who produces Brazil's {com['label'].lower()}")
    if not blk:
        st.info(f"No company table for {com['label'].lower()}. "
                + ("Brazilian output here is spread across many private operators with no "
                   "published company-level split worth charting."
                   if com["src"] == "curated" else
                   "Add one to `data/brazil_curated.json` under `companies` and it appears here."))
        # The national hedge still stands even with no company split to apportion it across.
        _brazil_hedge_section(com, None)
        return

    # Producers known, volumes not. Names are shown because knowing Vale mines iron ore
    # is not a guess — but every number is withheld rather than estimated, and the page
    # says exactly what would be needed to fill it in.
    if blk.get("unsourced"):
        st.warning(f"**We do not know how much each company produces.** "
                   f"{blk.get('reason', '')}")
        if blk.get("names"):
            st.markdown("**Producers** &nbsp;·&nbsp; names only, no volumes")
            st.dataframe(pd.DataFrame({
                blk.get("entity_label", "Producer"): blk["names"],
                f"Output ({com['unit']})": ["—"] * len(blk["names"]),
                "% of Brazil": ["—"] * len(blk["names"]),
                "% of WORLD": ["—"] * len(blk["names"])}),
                use_container_width=True, hide_index=True)
        st.caption(f"Brazil's national total — **{com['brazil']:,.2f} {com['unit']}**, "
                   f"{com['share']:.1f}% of world — is sourced ({com['source_label']}). "
                   f"It is only the split between companies that is missing. Nothing here is "
                   f"estimated: a blank means we do not know, not that the producer is small.")
        _brazil_hedge_section(com, blk)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Measures", blk["basis_label"], help=blk["basis_note"])
    c2.metric("Source", blk.get("provenance_label") or blk.get("confidence_label", "—"),
              help=(blk.get("source") or "") + "  " + blk.get("confidence_note", ""))
    c3.metric("Named companies cover", f"{blk['named_share']:.0f}%",
              help="Share of Brazil's total accounted for by the named companies; the "
                   "remainder sits in the 'Other' bucket"
                   + (" and in non-corporate production." if blk.get("has_artisanal") else "."))
    if blk.get("has_artisanal"):
        st.warning(f"**{blk['artisanal_share']:.0f}% of Brazil's {com['label'].lower()} has no "
                   f"corporate producer.** That line is shown in blue on the chart below and is "
                   f"counted in the totals, but it is not a company — which is why this table's "
                   f"first column reads *Producer*, not *Company*.")

    if blk["basis"] == "export":
        st.warning(f"**This is an export share, not a production share.** {blk['note']}")
    elif blk.get("note"):
        st.info(blk["note"])
    if blk["confidence"] == "estimate":
        st.caption("⚠️ These are desk estimates assembled from company disclosures and sector "
                   "bodies — sound enough to frame a conversation, but verify a number before "
                   "it goes in front of a client.")

    ch = _brazil_company_chart(blk, cc)
    if ch is not None:
        brand.show_chart(ch)

    quotes = _brazil_quotes(tuple(r["ticker"] for r in blk["rows"] if r.get("ticker")))
    tbl = pd.DataFrame(blk["rows"])
    tbl["Last"] = [f"{quotes[t][0]:,.2f}" if t in quotes else "—"
                   for t in tbl.get("ticker", pd.Series([""] * len(tbl)))]
    tbl["1d %"] = [("—" if t not in quotes or quotes[t][1] is None else f"{quotes[t][1]:+.2f}")
                   for t in tbl.get("ticker", pd.Series([""] * len(tbl)))]
    vol_col = f"Volume ({blk['unit']})"
    show = tbl.assign(**{
        vol_col: tbl["volume"].map("{:,.2f}".format),
        "% of Brazil": tbl["share_brazil"].map("{:.1f}".format),
        "% of WORLD": tbl["share_world"].map("{:.2f}".format),
        "Listing": tbl["ticker"].fillna("").replace("", "—"),
    }).rename(columns={"company": blk.get("entity_label") or "Company"})
    # A "% of exports"-style block has no volume to show — that column would just
    # repeat "% of Brazil".
    cols = [blk.get("entity_label") or "Company"] + ([] if blk.get("unit_is_pct") else [vol_col])
    st.dataframe(show[cols + ["% of Brazil", "% of WORLD", "Listing", "Last", "1d %"]],
                 use_container_width=True, hide_index=True)

    _brazil_hedge_section(com, blk)

    foot = [f"**{blk['basis_label']}**, {blk['year']} — {blk['source']}"]
    if blk.get("coverage_pct") is not None:
        foot.append(f"the table totals **{blk['coverage_pct']:.0f}%** of Brazil's national figure")
    st.caption(" · ".join(foot) + ". **% of WORLD** chains each company's share of Brazil through "
               "Brazil's share of world supply — what the company is worth to global balances. "
               "Prices are free Yahoo closes for the listed lines (B3 or the ADR); private "
               "groups, co-operatives and the Hong Kong / Santiago lines show “—”.")


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
               "🐖 Hogs & Pigs · 🛢️ Oil outlooks (OPEC / EIA / IEA) · 🧭 COT (weekly, Fri) · "
               "🏛️ FOMC · 💶 ECB · 💷 BoE MPC rate decisions &nbsp;·&nbsp; ★ = auto-emails the desk.")

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


# ─────────────────────────────────────────────────────────────────────────────
# STIR Paths — expiry timeline + per-bank implied-path / meeting-scenario tools
# (module pages: 🗓️ Expiry Timeline · 🏛️ Fed · 💶 ECB · 💷 BoE · ⚖️ Cross-Bank;
#  engine src/stirpaths.py — the Fed page keeps the fedpath PDF report)
# ─────────────────────────────────────────────────────────────────────────────
_STIR_STORE = ROOT / "data" / "stirpaths.json"
_STIR_HOUSE = {                                   # shipped defaults ("↺ Default" fallback)
    "timeline": list(stirpaths.PRODUCTS),         # overview: the whole book
    "FED": ["SFRA Comdty"],                       # bank pages: the strip the desk trades
    "ECB": ["ERA Comdty", "TKYA Comdty"],
    "BOE": ["SFIA Comdty"],
}
# Real flags, not emoji: Windows renders flag emoji as bare letter pairs
# ("US"), so the banks wear small self-contained SVG flags (base64 data URIs —
# no assets, no network). Used in headers, cards, mover chips and (via CSS
# ::before) the module tab row.
def _stir_flag_svgs() -> dict:
    import base64
    import math
    us = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">',
          '<rect width="60" height="40" fill="#fff"/>']
    for i in range(0, 13, 2):
        us.append(f'<rect y="{i * 40 / 13:.2f}" width="60" height="{40 / 13:.2f}" '
                  'fill="#B22234"/>')
    us.append('<rect width="24" height="21.54" fill="#3C3B6E"/>')
    for r_ in range(4):
        for k_ in range(5):
            us.append(f'<circle cx="{2.6 + k_ * 4.7:.1f}" cy="{2.8 + r_ * 5.3:.1f}" '
                      'r="1.05" fill="#fff"/>')
    us.append('</svg>')
    eu = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">',
          '<rect width="60" height="40" fill="#039"/>']
    for j in range(12):
        a0 = j * math.pi / 6
        cx, cy = 30 + 12 * math.sin(a0), 20 - 12 * math.cos(a0)
        pts = []
        for kk in range(10):
            rr = 2.4 if kk % 2 == 0 else 0.95
            aa = kk * math.pi / 5
            pts.append(f"{cx + rr * math.sin(aa):.2f},{cy - rr * math.cos(aa):.2f}")
        eu.append(f'<polygon points="{" ".join(pts)}" fill="#FC0"/>')
    eu.append('</svg>')
    uk = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 40">'
          '<rect width="60" height="40" fill="#012169"/>'
          '<path d="M0 0L60 40M60 0L0 40" stroke="#fff" stroke-width="8"/>'
          '<path d="M0 0L60 40M60 0L0 40" stroke="#C8102E" stroke-width="3.4"/>'
          '<rect x="24" width="12" height="40" fill="#fff"/>'
          '<rect y="14" width="60" height="12" fill="#fff"/>'
          '<rect x="26.6" width="6.8" height="40" fill="#C8102E"/>'
          '<rect y="16.6" width="60" height="6.8" fill="#C8102E"/></svg>')
    enc = lambda s: base64.b64encode(s.encode()).decode()
    return {"FED": enc("".join(us)), "ECB": enc("".join(eu)), "BOE": enc(uk)}


_STIR_FLAG_B64 = _stir_flag_svgs()


def _flag_img(bk: str, h: int = 13) -> str:
    return (f"<img src='data:image/svg+xml;base64,{_STIR_FLAG_B64[bk]}' "
            f"style='height:{h}px;width:auto;vertical-align:-1px;border-radius:2px;"
            f"box-shadow:0 0 0 1px rgba(255,255,255,0.22)'/>")


# flags on the module tab-row buttons (button labels cannot carry HTML — the
# flag rides in as a CSS ::before on the button text, keyed per tab)
_STIR_TAB_FLAG_CSS = "".join(
    f".st-key-gtab_{dest.replace(' ', '-')} button p::before {{"
    "content:''; display:inline-block; width:19px; height:12.5px;"
    f"background:url(data:image/svg+xml;base64,{_STIR_FLAG_B64[bk]}) center/cover;"
    "margin-right:8px; border-radius:2px; vertical-align:-1.5px;"
    # NB this segment is NOT an f-string: a doubled }} here stays literal, the
    # stray brace made the CSS parser drop the following rule (ECB/BoE flags
    # vanished while Fed's — the first rule — survived)
    "box-shadow:0 0 0 1px rgba(255,255,255,0.22);}"
    for dest, bk in (("Fed Path", "FED"), ("ECB Path", "ECB"), ("BoE Path", "BOE")))
# Bank identity colours — Ben's explicit preference (2026-08-11): keep the
# red/blue/green even though red/green mean direction elsewhere; a product-
# colour palette (SR3 gold / ER purple / SONIA orange) was tried and rejected.
_STIR_BANK_COLOR = {"FED": "#E53935", "ECB": "#1E88E5", "BOE": "#43A047"}


def _stir_defaults() -> dict:
    out = {k: list(v) for k, v in _STIR_HOUSE.items()}
    if _STIR_STORE.exists():
        try:
            saved = json.loads(_STIR_STORE.read_text(encoding="utf-8"))
            out.update({k: v for k, v in saved.items() if k in _STIR_HOUSE and v})
        except Exception:
            pass
    return out


def _stir_save_default(key: str, tickers: list) -> None:
    cur = _stir_defaults()
    cur[key] = tickers
    _STIR_STORE.parent.mkdir(parents=True, exist_ok=True)
    _STIR_STORE.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def _stir_set_sel(skey: str, values: list) -> None:
    st.session_state[skey] = list(values)


_STIR_SCEN_STORE = ROOT / "data" / "stirpaths_scenarios.json"


def _stir_scen_all() -> dict:
    try:
        return json.loads(_STIR_SCEN_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _stir_scen_save(bank_key: str, name: str, vals: dict) -> None:
    d = _stir_scen_all()
    d.setdefault(bank_key, {})[name] = vals
    _STIR_SCEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    _STIR_SCEN_STORE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _stir_scen_delete(bank_key: str, name: str) -> None:
    d = _stir_scen_all()
    d.get(bank_key, {}).pop(name, None)
    if d.get("_defaults", {}).get(bank_key) == name:   # deleting the ★ default
        d["_defaults"].pop(bank_key, None)             # clears the pointer too
    _STIR_SCEN_STORE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _stir_scen_default(bank_key: str) -> str | None:
    """The bank's ★ default scenario name — honoured only while it still exists."""
    d = _stir_scen_all()
    name = d.get("_defaults", {}).get(bank_key)
    return name if name in d.get(bank_key, {}) else None


def _stir_scen_set_default(bank_key: str, name: str | None) -> None:
    d = _stir_scen_all()
    if name is None:
        d.get("_defaults", {}).pop(bank_key, None)
    else:
        d.setdefault("_defaults", {})[bank_key] = name
    _STIR_SCEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    _STIR_SCEN_STORE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _stir_picker(store_key: str, products: list, label: str = "Products shown") -> list:
    """Product include/exclude multiselect with a persisted default: 💾 saves the
    current selection as the page default (data/stirpaths.json), ↺ restores it."""
    dflt = _stir_defaults()[store_key]
    have = [p.ticker for p in products]
    skey = f"stir_sel_{store_key}"
    c1, c2, c3 = st.columns([5, 1.1, 1.1], vertical_alignment="bottom")
    sel = c1.multiselect(
        label, have, default=[t for t in dflt if t in have] or have[:1], key=skey,
        format_func=lambda t: f"{stirpaths.PRODUCTS[t].short} · {stirpaths.PRODUCTS[t].name}")
    c2.button("💾 Save default", key=f"stir_sv_{store_key}", use_container_width=True,
              help="Persist the current selection as this page's default.",
              on_click=lambda: (_stir_save_default(store_key, list(sel)),
                                st.toast("Saved as default.")))
    c3.button("↺ Default", key=f"stir_rst_{store_key}", use_container_width=True,
              help="Back to the saved default selection.",
              on_click=_stir_set_sel, args=(skey, [t for t in dflt if t in have]))
    return sel


def _stir_meeting_rules(banks: list, asof, hor_end, x_scale, mute=False):
    """Dashed vertical rule per rate decision inside the horizon. On multi-bank
    charts the rules carry the bank colour + legend (they ARE the bank marker);
    on single-bank pages they mute to grey so the in-band ticks pop instead."""
    import altair as alt
    mrows = []
    for bk in banks:
        b = stirpaths.BANKS[bk]
        for m in b.meetings:
            if asof <= m < hor_end:
                mrows.append({"Bank": bk, "Date": pd.Timestamp(m),
                              "What": f"{b.meeting_name} decision · {m:%a %d %b %Y}"})
    if not mrows:
        return None
    base = alt.Chart(pd.DataFrame(mrows)).mark_rule(
        strokeDash=[5, 4], strokeWidth=1.4 if mute else 1.3,
        opacity=0.35 if mute else 0.55).encode(
        x=alt.X("Date:T", scale=x_scale),
        tooltip=[alt.Tooltip("What:N", title="Meeting")])
    if mute:
        return base.encode(color=alt.value("#8A93A6"))
    bdom = [b for b in ("FED", "ECB", "BOE") if b in banks]
    return base.encode(color=alt.Color("Bank:N", scale=alt.Scale(
        domain=bdom, range=[_STIR_BANK_COLOR[b] for b in bdom]),
        legend=alt.Legend(title="Rate decisions", orient="top") if len(bdom) > 1 else None))


def _stir_timeline(sel: list, banks: list, asof, months: int,
                   key: str = "tl", default_view: str = "Contract windows",
                   ann: dict | None = None) -> None:
    """The module's centrepiece, in two flavours: 'Contract windows' = one row per
    CONTRACT with its reference period as a shaded band (the how-many-decisions-
    inside-each-period story, Ben's core ask); 'Compact rows' = one row per product
    with expiry marks only."""
    prods = [stirpaths.PRODUCTS[t] for t in sel]
    if not prods:
        st.info("Pick at least one product to draw the timeline.")
        return
    view = st.radio("Timeline view", ["Contract windows", "Compact rows"],
                    index=["Contract windows", "Compact rows"].index(default_view),
                    horizontal=True, key=f"stir_view_{key}",
                    help="Contract windows draws each contract's reference period as a band, "
                         "so the decisions inside each period are countable at a glance. "
                         "Compact rows is the one-line-per-product expiry overview.")
    if view == "Contract windows":
        _stir_window_chart(prods, banks, asof, months, ann)
    else:
        _stir_compact_chart(prods, banks, asof, months)


def _stir_compact_chart(prods: list, banks: list, asof, months: int) -> None:
    """One row per product: futures ● (true last-trade day, labelled with the
    contract that actually dies — the in-arrears quarterlies expire at their
    window END) and options ◇, with decisions as dashed rules."""
    import altair as alt
    ey, em = stirpaths._add_months(asof.year, asof.month, months)
    hor_end = date(ey, em, 1)
    x_scale = alt.Scale(domain=[pd.Timestamp(asof).isoformat(),
                                pd.Timestamp(hor_end).isoformat()])
    rows = []
    for p in prods:
        n = (months // 3 + 4) if p.quarterly else months + 3
        for c in stirpaths.strip(p, asof, n):
            lt = stirpaths.fut_last_trade(p, c)
            if asof <= lt < hor_end:
                rows.append({"Product": p.short, "Kind": "Future", "Date": pd.Timestamp(lt),
                             "What": f"{p.short} {c.label} future last trade · {lt:%a %d %b %Y} "
                                     f"(window {c.start:%d %b} → {c.end:%d %b %y})"})
        if p.has_options:
            for r in stirpaths.expiry_rows(p, asof, months):
                if r.kind == "Option":
                    rows.append({"Product": p.short, "Kind": "Option", "Date": pd.Timestamp(r.expiry),
                                 "What": f"{p.short} {r.month} option · {r.expiry:%a %d %b %Y}"})
    order = [p.short for p in prods]
    h = max(190, 48 * len(prods) + 95)
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Date:T", title=None, scale=x_scale,
                axis=alt.Axis(format="%b %y", labelAngle=0)),
        y=alt.Y("Product:N", sort=order, title=None,
                scale=alt.Scale(paddingOuter=1.0)),
        color=alt.Color("Product:N", scale=alt.Scale(domain=order, range=[p.color for p in prods]),
                        legend=None),
        tooltip=[alt.Tooltip("What:N", title="Expiry")])
    layers = [base.transform_filter(alt.datum.Kind == "Future").mark_point(
                  filled=True, size=120, shape="circle"),
              base.transform_filter(alt.datum.Kind == "Option").mark_point(
                  filled=False, size=120, shape="diamond", strokeWidth=2.2)]
    rules = _stir_meeting_rules(banks, asof, hor_end, x_scale)
    if rules is not None:
        layers.append(rules)
    # decision LANES: each bank gets a slim top lane with the decision's FULL
    # date ("16 Sep 26") in the bank colour, sat just right of its dashed rule
    # so the rule never runs through the text; per-bank lanes mean labels can't
    # collide across banks
    lanes = [b for b in ("FED", "ECB", "BOE") if b in banks]
    for li, bk in enumerate(lanes):
        b = stirpaths.BANKS[bk]
        lrows = [{"Date": pd.Timestamp(m), "lab": f"{m:%d %b %y}",
                  "What": f"{b.meeting_name} decision · {m:%a %d %b %Y}"}
                 for m in b.meetings if asof <= m < hor_end]
        if lrows:
            layers.append(alt.Chart(pd.DataFrame(lrows)).mark_text(
                fontSize=10.5, fontWeight="bold", align="left", dx=4).encode(
                x=alt.X("Date:T", scale=x_scale), y=alt.value(10 + 14 * li),
                text="lab:N", color=alt.value(_STIR_BANK_COLOR[bk]),
                tooltip=[alt.Tooltip("What:N", title="Decision")]))
    brand.show_chart(alt.layer(*layers).properties(height=h))
    lane_note = " / ".join(f"{stirpaths.BANKS[b].meeting_name}" for b in lanes)
    st.caption("Top lanes: **rate decision dates**, one lane per bank "
               f"({lane_note}, top to bottom), each label just right of its dashed rule. "
               "● futures last-trade day &nbsp;·&nbsp; ◇ monthly options expiry (standard "
               "cycle, rule-based, exchange-holiday aware).")


def _stir_window_chart(prods: list, banks: list, asof, months: int,
                       ann: dict | None = None) -> None:
    """One row per CONTRACT: a shaded band spanning its reference window, the
    decisions still ahead inside it as ticks, ● last-trade day, ◇ its options.
    With `ann` (the bank pages pass it) the chart becomes the desk cockpit:
    each future's market price above its ●, the front contract top-left, the
    market-implied odds above each decision rule, YOUR odds below it, and a
    market-vs-your-scenario price overlay on a right-hand axis."""
    import altair as alt
    cc = brand.chart_colors()
    ey, em = stirpaths._add_months(asof.year, asof.month, months)
    hor_end = date(ey, em, 1)
    x_scale = alt.Scale(domain=[pd.Timestamp(asof).isoformat(),
                                pd.Timestamp(hor_end).isoformat()])
    px_of = (ann or {}).get("px", {})
    fair_of = (ann or {}).get("fair", {})

    bars, ticks, futs, opts, mcs, dels, row_order = [], [], [], [], [], [], []
    overflow: dict[str, int] = {}                   # monthly rows beyond the density cap
    row_of = {}                                     # (product short, contract label) -> row id
    for p in prods:
        bank = stirpaths.BANKS[p.bank]
        n = (months // 3 + 4) if p.quarterly else months + 3
        shown = 0
        for c in stirpaths.strip(p, asof, n):
            if c.end <= asof or c.start >= hor_end:
                continue
            if not p.quarterly and shown >= 12:     # density guard: a 24mo horizon of
                overflow[p.short] = overflow.get(p.short, 0) + 1   # monthlies drowns the chart
                continue
            shown += 1
            left = [m for m in stirpaths.meetings_in_window(bank, c) if m > asof]
            row = f"{p.short} {c.label} · {len(left)} mtg" + ("" if len(left) == 1 else "s")
            row_order.append(row)
            row_of[(p.short, c.label)] = row
            seg_s = pd.Timestamp(max(c.start, asof))
            seg_e = pd.Timestamp(min(c.end, hor_end))
            bars.append({"Row": row, "Product": p.short, "start": seg_s, "end": seg_e,
                         "What": f"{p.short} {c.label} — reference window {c.start:%d %b %y} → "
                                 f"{c.end:%d %b %y} · {len(left)} decision(s) still ahead inside it"})
            for m in left:
                if m < hor_end:
                    ticks.append({"Row": row, "Date": pd.Timestamp(m), "Bank": p.bank,
                                  "What": f"{bank.meeting_name} {m:%a %d %b %Y} — inside the "
                                          f"{p.short} {c.label} window"})
            lt = stirpaths.fut_last_trade(p, c)
            if asof <= lt < hor_end:
                futs.append({"Row": row, "Product": p.short, "Date": pd.Timestamp(lt),
                             "px": f"{px_of[c.code]:.3f}" if c.code in px_of else "",
                             "What": f"{p.short} {c.label} future last trade · {lt:%a %d %b %Y}"
                                     + (f" · trading {px_of[c.code]:.4f}" if c.code in px_of else "")})
            if c.code in px_of and c.code in fair_of:
                d_bp = (fair_of[c.code] - px_of[c.code]) * 100.0
                dels.append({"Row": row, "Date": pd.Timestamp(hor_end), "lab": f"{d_bp:+.1f}",
                             "col": (cc["long"] if d_bp > 0.05 else
                                     cc["short"] if d_bp < -0.05 else cc["muted"]),
                             "What": f"{c.code}: your fair {fair_of[c.code]:.4f} vs market "
                                     f"{px_of[c.code]:.4f} → {d_bp:+.1f}bp "
                                     f"({'cheap vs your view (buy)' if d_bp > 0.05 else 'rich vs your view (sell)' if d_bp < -0.05 else 'in line'})"})
        if p.has_options:
            for r in stirpaths.expiry_rows(p, asof, months):
                if r.kind != "Option":
                    continue
                u = stirpaths.option_underlying(p, r.year, r.mon)
                row = row_of.get((p.short, u.label))
                if row:
                    opts.append({"Row": row, "Product": p.short, "Date": pd.Timestamp(r.expiry),
                                 "What": f"{p.short} {r.month} option · expires {r.expiry:%a %d %b %Y} "
                                         f"· exercises into {u.code}"})
            for r in stirpaths.midcurve_expiries(p, asof, months):
                u = stirpaths.option_underlying_mc(p, r.year, r.mon)
                row = row_of.get((p.short, u.label))
                if row:
                    mcs.append({"Row": row, "Product": p.short, "Date": pd.Timestamp(r.expiry),
                                "What": f"{p.short} {r.month} 1Y-MIDCURVE option · expires "
                                        f"{r.expiry:%a %d %b %Y} · exercises into the deferred "
                                        f"{u.code} — the instrument for far-out meeting views"})
    if not bars:
        st.info("No contract windows intersect the horizon.")
        return

    # The odds get their own labelled ROWS at the top of the band scale (not
    # pixel-pinned text) — allocated space, so they can never collide with the
    # first contract row's price tags.
    if ann:
        row_order = ["Market %", "Your call %"] + row_order
    h = max(190, 34 * len(row_order) + 70)
    y_enc = alt.Y("Row:N", sort=row_order, title=None, scale=alt.Scale(paddingOuter=0.5))

    def _enc(chart, tooltip_title):
        return chart.encode(
            y=y_enc,
            color=alt.Color("Product:N", legend=None, scale=alt.Scale(
                domain=[p.short for p in prods], range=[p.color for p in prods])),
            tooltip=[alt.Tooltip("What:N", title=tooltip_title)])

    def _sign_col(v):
        return cc["long"] if v > 0 else cc["short"] if v < 0 else cc["muted"]

    layers = [_enc(alt.Chart(pd.DataFrame(bars)).mark_bar(fillOpacity=0.20, size=17),
                   "Contract").encode(x=alt.X("start:T", title=None, scale=x_scale,
                                              axis=alt.Axis(format="%b %y", labelAngle=0)),
                                      x2="end:T")]
    rules = _stir_meeting_rules(banks, asof, hor_end, x_scale, mute=bool(ann))
    if rules is not None:
        layers.append(rules)
    bdom = [b for b in ("FED", "ECB", "BOE") if b in banks]
    if ticks:
        layers.append(alt.Chart(pd.DataFrame(ticks)).mark_tick(
            thickness=3, size=17, opacity=0.95).encode(
            x=alt.X("Date:T", scale=x_scale), y=y_enc,
            color=alt.Color("Bank:N", legend=None, scale=alt.Scale(
                domain=bdom, range=[_STIR_BANK_COLOR[b] for b in bdom])),
            tooltip=[alt.Tooltip("What:N", title="Decision")]))
    if futs:
        fdf = pd.DataFrame(futs)
        layers.append(_enc(alt.Chart(fdf).mark_point(
            filled=True, size=100, shape="circle"), "Future").encode(
            x=alt.X("Date:T", scale=x_scale)))
        if ann:                                     # (1) market price above each futures expiry
            layers.append(_enc(alt.Chart(fdf).mark_text(
                dy=-14, fontSize=11, fontWeight="bold"), "Future").encode(
                x=alt.X("Date:T", scale=x_scale), text="px:N"))
    if opts:
        layers.append(_enc(alt.Chart(pd.DataFrame(opts)).mark_point(
            filled=False, size=100, shape="diamond", strokeWidth=2.2), "Option").encode(
            x=alt.X("Date:T", scale=x_scale)))
    if mcs:
        layers.append(_enc(alt.Chart(pd.DataFrame(mcs)).mark_point(
            filled=False, size=95, shape="triangle-up", strokeWidth=2.2), "1Y midcurve").encode(
            x=alt.X("Date:T", scale=x_scale)))
    if ann:                                         # (3)+(4) odds rows: market, then your call
        mrows = [{"Row": "Market %", "Date": pd.Timestamp(m), "lab": f"{v:+.0f}",
                  "col": _sign_col(v),
                  "What": f"Market prices {v:+.0f}% ({'hike' if v > 0 else 'cut' if v < 0 else 'hold'}) "
                          f"at the {m:%d %b %y} decision"}
                 for m, v in (ann.get("mkt") or {}).items() if asof <= m < hor_end]
        yrows = [{"Row": "Your call %", "Date": pd.Timestamp(m), "lab": f"{v:+.0f}",
                  "col": _sign_col(v),
                  "What": f"Your call: {v:+.0f}% ({'hike' if v > 0 else 'cut' if v < 0 else 'hold'}) "
                          f"at the {m:%d %b %y} decision — edit below the chart"}
                 for m, v in (ann.get("you") or {}).items() if asof <= m < hor_end]
        if mrows or yrows:
            layers.append(alt.Chart(pd.DataFrame(mrows + yrows)).mark_text(
                fontSize=11, fontWeight="bold").encode(
                x=alt.X("Date:T", scale=x_scale), y=y_enc, text="lab:N",
                color=alt.Color("col:N", scale=None),
                tooltip=[alt.Tooltip("What:N", title="Odds")]))
    if dels:                                        # (5) your fair vs market, one Δbp per row
        layers.append(alt.Chart(pd.DataFrame(dels)).mark_text(
            align="right", dx=-2, fontSize=11, fontWeight="bold").encode(
            x=alt.X("Date:T", scale=x_scale), y=y_enc, text="lab:N",
            color=alt.Color("col:N", scale=None),
            tooltip=[alt.Tooltip("What:N", title="Your fair − market")]))

    props = {"height": h}
    if ann and ann.get("front"):                    # (2) front-month quote, top-left
        props["title"] = alt.TitleParams(ann["front"], anchor="start", fontSize=13)
    brand.show_chart(alt.layer(*layers).properties(**props))
    cap = ("Each row = one contract; the shaded band is its **reference window** and the row label "
           "counts the decisions still ahead inside it. Coloured ticks = those decisions "
           "&nbsp;·&nbsp; ● last-trade day &nbsp;·&nbsp; ◇ its options (serials sit on the "
           "quarterly they exercise into) &nbsp;·&nbsp; △ 1Y midcurves (short-dated premium on the "
           "deferred quarterly) &nbsp;·&nbsp; dashed rules = decision dates.")
    if ann:
        cap += (" &nbsp;·&nbsp; Numbers above each ● = the future's current price. **Market %** = "
                "the odds the strip prices at each decision, **Your call %** = yours (green hike / "
                "red cut, edit below). Right edge: **Δbp** = your fair − market per contract — "
                "green cheap vs your view, red rich.")
    if overflow:
        cap += (" &nbsp;·&nbsp; Monthly rows capped at 12: "
                + ", ".join(f"{k} +{v} more" for k, v in overflow.items())
                + " (shorten the horizon or use Compact rows).")
    st.caption(cap + " Hover anything for detail.")


def _stir_window_table(prods: list, bank_keys: list, asof, n_q: int) -> None:
    """Meetings-per-contract: each front contract's reference window and exactly
    which decisions land inside it — the thesis behind the module, as a table."""
    rows = []
    for p in prods:
        bank = stirpaths.BANKS[p.bank]
        for c in stirpaths.strip(p, asof, n_q if p.quarterly else min(3 * n_q, 12)):
            mtgs = stirpaths.meetings_in_window(bank, c)
            rows.append({
                "Product": p.short, "Contract": c.code,
                "Window": f"{c.start:%d %b %y} → {c.end:%d %b %y}",
                "Meetings in window": len(mtgs),
                "Which": ", ".join(f"{m:%d %b %y}" for m in mtgs) or "—",
            })
    df = pd.DataFrame(rows)
    brand.themed_dataframe(df, fmt={}, height=min(420, 45 + 35 * len(df)))
    st.caption("A decision counts when its *effective* day (the business day after the announcement) "
               "falls inside the contract's reference window — i.e. it moves that contract's settlement.")


def render_stir_overview() -> None:
    """The module's home: the state of global rate expectations — bank cards,
    what repriced, the cross-bank divergence chart (absorbed from the old
    Cross tab), the next fortnight's events, then the full expiry timeline."""
    import altair as alt
    st.subheader("🗓️  STIR Paths — the state of rate expectations")
    asof = datetime.now(ZoneInfo("America/New_York")).date()
    fits = {bk: stirpaths.default_bank_fit(bk, asof) for bk in stirpaths.BANKS}
    src, src_asof = stirpaths.strip_source(
        stirpaths.strip(stirpaths.PRODUCTS["SFRA Comdty"], asof, 8))
    st.caption("Each bank's strip inverted into the meeting-step path it prices — prices from the "
               + (f"**morning snapshot · {src_asof}**" if src == "snapshot"
                  else "**synthetic demo feed**")
               + ". Open a bank's cockpit to set your own odds against it.")

    # ---- bank cards ----------------------------------------------------------
    _dest = {"FED": "Fed Path", "ECB": "ECB Path", "BOE": "BoE Path"}
    cards = st.columns(3)
    for col, (bk, bank) in zip(cards, stirpaths.BANKS.items()):
        ip = fits[bk]
        with col:
            if ip is None or not len(ip.meetings):
                st.info(f"{bank.name}: no strip priced.")
                continue
            nxt = ip.meetings[0]
            bp0 = float(ip.per_meeting_bp[0])
            dec0 = _stir_signed_pct(bp0, bank.step_bp, bank.step_bp) / 100.0
            eoy = [i for i, m in enumerate(ip.meetings) if m.year == asof.year]
            yend = float(ip.cum_bp[eoy[-1]]) if eoy else float(ip.cum_bp[-1])
            lvl0 = bank.default_rate
            term = lvl0 + float(ip.cum_bp[-1]) / 100.0
            col_c = _STIR_BANK_COLOR[bk]
            rate_big = (f"{lvl0 - 0.125:.2f}–{lvl0 + 0.125:.2f}" if bk == "FED"
                        else f"{lvl0:.2f}")   # Fed band derives from default_rate
            rate_tip = {"FED": "Current FOMC target band (%)",
                        "ECB": "Current deposit facility rate (%)",
                        "BOE": "Current Bank Rate (%)"}[bk]
            st.markdown(
                f"<div style='border:1px solid rgba(128,128,128,0.28);border-left:4px solid "
                f"{col_c};border-radius:8px;padding:0.7rem 0.9rem 0.55rem;min-height:10.6rem'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div style='font-weight:700;font-size:0.95rem'>{_flag_img(bk)} {bank.name}"
                f"</div>"
                f"<div style='text-align:right;white-space:nowrap' title='{rate_tip}'>"
                f"<span style='font-size:1.25rem;font-weight:700'>{rate_big}</span>"
                f"<br><span style='color:#9AA4B0;font-size:0.66rem;"
                f"letter-spacing:0.05em'>NOW</span></div>"
                f"</div>"
                f"<div style='color:#C3CAD3;font-size:0.78rem;margin-top:0.1rem'>"
                f"Next: <b>{bank.meeting_name}</b></div>"
                f"<div style='color:#C3CAD3;font-size:0.78rem'>{nxt:%a %d %b} · in "
                f"<b>{(nxt - asof).days}d</b></div>"
                f"<div style='font-size:1.4rem;font-weight:700;margin:0.2rem 0 0.1rem'>"
                f"{_stir_odds_str(bp0, bank.step_bp)} <span style='font-size:0.85rem;"
                f"color:#C3CAD3'>{bp0:+.1f}bp</span></div>"
                f"<div style='color:#C3CAD3;font-size:0.78rem'>Thru Dec {asof.year}: "
                f"<b>{yend:+.0f}bp</b> &nbsp;·&nbsp; Terminal: <b>{term:.2f}%</b></div>"
                f"</div>", unsafe_allow_html=True)
            st.button(f"Open {bk} cockpit →", key=f"stir_card_{bk}",
                      use_container_width=True, on_click=_go, args=(_dest[bk],))

    # ---- what repriced (the daily ledger) ------------------------------------
    rep = stirpaths.meeting_repricing(asof)
    movers = []
    for bk, mp in rep.items():
        for iso, (bp, d) in mp.items():
            if abs(d) >= 1.0:
                movers.append((abs(d), bk, date.fromisoformat(iso), bp, d))
    if movers:
        movers.sort(reverse=True)
        chips = " &nbsp;·&nbsp; ".join(
            f"<span style='color:{'#66BB6A' if d > 0 else '#EF5350'};font-weight:700'>"
            f"{_flag_img(bk, 11)} {m:%b %y} {d:+.1f}bp</span>"
            f"<span style='color:#9AA4B0;font-size:0.78rem'> (now {bp:+.1f})</span>"
            for _, bk, m, bp, d in movers[:5])
        st.markdown("**Repriced since the last snapshot:** &nbsp;" + chips,
                    unsafe_allow_html=True)
    else:
        st.caption("Repricing movers appear here once the morning snapshot has recorded two "
                   "days of implied paths (data/stir_meeting_history.json).")

    # ---- cross-bank divergence (absorbed from the old Cross tab) -------------
    _sp = st.columns([5, 1.6])
    _sp[0].markdown("#### Fed · ECB · BoE — where the cycles diverge")
    view = _sp[1].radio("Show", ["Cumulative bp", "Rate level"], key="stir_home_view",
                        horizontal=True, label_visibility="collapsed")
    frames, mrows = [], []
    for bk, bank in stirpaths.BANKS.items():
        ip = fits[bk]
        if ip is None or not len(ip.meetings):
            continue
        seg_dates = [asof] + [fedpath.effective_date(m) for m in ip.meetings]
        lvl = ip.seg_rates - stirpaths.BANK_BASIS_SEED[bk] / 100.0
        cum = np.concatenate([[0.0], (lvl[1:] - lvl[0]) * 100.0])
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(seg_dates),
            "value": cum if view == "Cumulative bp" else lvl,
            "Bank": bank.name}))
        for m, bp, cm in zip(ip.meetings, ip.per_meeting_bp, ip.cum_bp):
            mrows.append({"_d": m, "Decision": f"{m:%a %d %b %y}",
                          "Bank": {"FED": "Fed", "ECB": "ECB", "BOE": "BoE"}[bk],
                          "In": f"{(m - asof).days}d",
                          "Implied (bp)": float(bp),
                          "Odds": _stir_odds_str(float(bp), bank.step_bp),
                          "Cum (bp)": float(cm)})
    if frames:
        dom = [stirpaths.BANKS[b].name for b in ("FED", "ECB", "BOE")]
        rng = [_STIR_BANK_COLOR[b] for b in ("FED", "ECB", "BOE")]
        line = alt.Chart(pd.concat(frames)).mark_line(
            interpolate="step-after", strokeWidth=3).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q",
                    title="Cumulative bp vs today" if view == "Cumulative bp"
                    else "Policy rate (%)",
                    scale=alt.Scale(zero=(view == "Cumulative bp"))),
            color=alt.Color("Bank:N", scale=alt.Scale(domain=dom, range=rng),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("date:T", title="From"), alt.Tooltip("Bank:N"),
                     alt.Tooltip("value:Q", format=".2f")])
        brand.show_chart(line.properties(height=330))
    with st.expander("📋 Every decision, every bank — what's priced"):
        if mrows:
            mdf = (pd.DataFrame(sorted(mrows, key=lambda r: r["_d"]))
                   .drop(columns="_d").head(24))
            brand.themed_dataframe(
                mdf, fmt={"Implied (bp)": "{:+.1f}".format, "Cum (bp)": "{:+.1f}".format},
                height=min(500, 45 + 35 * len(mdf)))
            st.caption("Chronological across all three banks — the screenshot for a client "
                       "chat. Odds = implied move ÷ the bank's step, FedWatch-style.")

    # ---- the next fortnight --------------------------------------------------
    st.markdown("#### The next two weeks")
    ev = []
    horizon14 = asof + timedelta(days=14)
    for bk, bank in stirpaths.BANKS.items():
        for m in bank.meetings:
            if asof <= m <= horizon14:
                ev.append((m, f"{bank.meeting_name} rate decision",
                           bank.name))
    for p in stirpaths.PRODUCTS.values():
        for c in stirpaths.strip(p, asof, 14):
            lt = stirpaths.fut_last_trade(p, c)
            if asof <= lt <= horizon14:
                ev.append((lt, f"● {p.short} {c.label} futures last trade", c.code))
        if p.has_options:
            for r in stirpaths.expiry_rows(p, asof, 2):
                if r.kind == "Option" and asof <= r.expiry <= horizon14:
                    u = stirpaths.option_underlying(p, r.year, r.mon)
                    ev.append((r.expiry, f"◇ {p.short} {r.month} options expire",
                               f"into {u.code}"))
            for r in stirpaths.midcurve_expiries(p, asof, 2):
                if asof <= r.expiry <= horizon14:
                    u = stirpaths.option_underlying_mc(p, r.year, r.mon)
                    ev.append((r.expiry, f"△ {p.short} {r.month} 1Y-midcurve expires",
                               f"into {u.code}"))
    if ev:
        ev.sort(key=lambda t: t[0])
        brand.themed_dataframe(pd.DataFrame(
            [{"Date": f"{d:%a %d %b}", "In": f"{(d - asof).days}d",
              "Event": what, "Detail": det} for d, what, det in ev]),
            fmt={}, height=min(420, 45 + 35 * len(ev)))
    else:
        st.caption("Nothing lands in the next two weeks — the calendar's quiet.")

    # ---- full expiry timeline (the original page, demoted to a section) ------
    st.markdown("#### Full expiry timeline — futures, options & rate decisions")
    sel = _stir_picker("timeline", list(stirpaths.PRODUCTS.values()))
    months = st.slider("Horizon (months)", 6, 24, 15, key="stir_tl_months")
    banks = sorted({stirpaths.PRODUCTS[t].bank for t in sel},
                   key=["FED", "ECB", "BOE"].index) if sel else []
    _stir_timeline(sel, banks, asof, months, key="ov", default_view="Compact rows")

    ey, em = stirpaths._add_months(asof.year, asof.month, months)
    up = sorted((m, b) for b in banks for m in stirpaths.BANKS[b].meetings
                if asof <= m < date(ey, em, 1))
    if up:
        with st.expander(f"📋 Upcoming rate decisions in the window ({len(up)})"):
            brand.themed_dataframe(pd.DataFrame([{
                "Date": f"{m:%a %d %b %Y}", "In": f"{(m - asof).days}d",
                "Bank": stirpaths.BANKS[b].name,
                "Meeting": stirpaths.BANKS[b].meeting_name} for m, b in up]),
                fmt={}, height=min(420, 45 + 35 * len(up)))
    with st.expander("🎯 Meetings inside each contract window"):
        _stir_window_table([stirpaths.PRODUCTS[t] for t in sel], banks, asof, 4)


def _stir_fed_bands():
    """Selectable Fed target bands (25bp wide) from 2.00–2.25 up to 5.50–5.75."""
    return [f"{lo/100:.2f} – {lo/100+0.25:.2f}" for lo in range(200, 551, 25)]


def _stir_reseed(bank_key: str, views: dict, ver_bump: bool = True) -> None:
    """Set the scenario probabilities and force the editor to re-read them."""
    st.session_state[f"sp{bank_key}_views"] = views
    if ver_bump:
        st.session_state[f"sp{bank_key}_ver"] = st.session_state.get(f"sp{bank_key}_ver", 0) + 1


def _stir_seed_from_market(per_bp: dict, hike_bp: float, cut_bp: float) -> dict:
    """Implied per-meeting move → ONE signed 'your odds' number per meeting:
    +% = chance of a hike, −% = chance of a cut (desk shorthand), snapped to 5%
    steps. Beyond ±100% = more than one full step priced (−150 = 1.5 cuts);
    capped at ±300."""
    out = {}
    for lab, bp in per_bp.items():
        v = (bp / hike_bp if bp > 0 else bp / cut_bp) * 100.0
        out[lab] = float(min(300.0, max(-300.0, round(v / 5) * 5.0)))
    return out


def _stir_signed_pct(bp: float, hike_bp: float, cut_bp: float) -> float:
    """Un-snapped signed odds %, for the chart labels (−82 = 82% of a cut priced)."""
    v = (bp / hike_bp if bp > 0 else bp / cut_bp) * 100.0
    return float(min(300.0, max(-300.0, v)))


def _stir_odds_str(bp: float, step: float) -> str:
    d, p = stirpaths.implied_odds(bp, step)
    if d == "hold":
        return "hold"
    if abs(bp) > step:                      # more than one full step priced
        full = int(abs(bp) // step)
        rem = (abs(bp) - full * step) / step * 100.0
        return f"{full}×{step:.0f} {d} + {rem:.0f}%"
    return f"{p * 100:.0f}% {d}"


def _stir_term_chart(prods: list, bank, strips: dict, px_of: dict, fair_of: dict,
                     asof, front_note: str = "") -> None:
    """The centrepiece: the futures TERM STRUCTURE — market curve (solid, filled ●
    at each contract's true last-trade day) vs the desk-scenario curve (dashed
    gold-edged, open points) — with every rate decision as a prominent vertical
    rule and the option / midcurve expiries pinned along the bottom rail."""
    import altair as alt
    from datetime import timedelta as _td
    cc = brand.chart_colors()
    # the chart ends where KNOWLEDGE ends: the last published decision (+3 weeks) —
    # contracts expiring beyond it are dropped rather than drawn into a zone that
    # would falsely read as meeting-free
    horizon = max(bank.meetings) + _td(days=21)
    ups = [m for m in bank.meetings if m >= asof]

    def _rel(lt):
        near = min(bank.meetings, key=lambda m: abs((lt - m).days))
        d = (lt - near).days
        word = "AFTER" if d > 0 else "BEFORE" if d < 0 else "ON"
        return (f"{d:+d}d", f"expires {abs(d)}d {word} the {near:%d %b %y} decision"
                            if d else f"expires ON the {near:%d %b %y} decision day")

    rows, orail, n_dropped = [], [], 0
    for p in prods:
        for c in strips[p.ticker]:
            lt = stirpaths.fut_last_trade(p, c)
            if lt < asof:
                continue
            if lt > horizon:
                n_dropped += 1
                continue
            rel, rel_tip = _rel(lt)
            n_left = sum(1 for m in ups if m < lt)
            if c.code in px_of:
                rows.append({"Date": pd.Timestamp(lt), "px": px_of[c.code], "Path": "Market",
                             "Product": p.short, "tag": c.code[-2:], "rel": rel,
                             "What": f"{c.code} ({p.short} {c.label}) — market {px_of[c.code]:.4f}, "
                                     f"last trade {lt:%a %d %b %y} · {rel_tip} · "
                                     f"{n_left} decision(s) before it dies"})
            if c.code in fair_of:
                rows.append({"Date": pd.Timestamp(lt), "px": fair_of[c.code], "Path": "Your view",
                             "Product": p.short,
                             "What": f"{c.code} ({p.short} {c.label}) — your fair {fair_of[c.code]:.4f} "
                                     f"({(fair_of[c.code] - px_of.get(c.code, fair_of[c.code])) * 100:+.1f}bp vs market)"})
        if p.has_options:
            for r in stirpaths.expiry_rows(p, asof, 25):
                if r.kind == "Option" and asof <= r.expiry <= horizon:
                    u = stirpaths.option_underlying(p, r.year, r.mon)
                    orail.append({"Date": pd.Timestamp(r.expiry), "Kind": "Option", "Product": p.short,
                                  "What": f"{p.short} {r.month} option expiry · {r.expiry:%a %d %b %y} "
                                          f"· exercises into {u.code}"})
            for r in stirpaths.midcurve_expiries(p, asof, 25):
                if asof <= r.expiry <= horizon:
                    u = stirpaths.option_underlying_mc(p, r.year, r.mon)
                    orail.append({"Date": pd.Timestamp(r.expiry), "Kind": "1Y midcurve", "Product": p.short,
                                  "What": f"{p.short} {r.month} 1Y-MIDCURVE expiry · {r.expiry:%a %d %b %y} "
                                          f"· exercises into the deferred {u.code}"})
    if not rows:
        st.info("No priced contracts to chart.")
        return
    h = 400
    x_scale = alt.Scale(domain=[pd.Timestamp(asof - _td(days=4)).isoformat(),
                                pd.Timestamp(horizon).isoformat()])
    df = pd.DataFrame(rows)
    pdom = ["Market", "Your view"]
    prng = [cc["series"], cc["accent"]]
    base = alt.Chart(df).encode(
        x=alt.X("Date:T", title=None, scale=x_scale),
        y=alt.Y("px:Q", title="Futures price", scale=alt.Scale(zero=False)),
        color=alt.Color("Path:N", scale=alt.Scale(domain=pdom, range=prng),
                        legend=alt.Legend(title=None, orient="top")),
        detail="Product:N",
        tooltip=[alt.Tooltip("What:N", title="Contract")])
    lines = base.mark_line(strokeWidth=2.6).encode(
        strokeDash=alt.StrokeDash("Path:N", legend=None, scale=alt.Scale(
            domain=pdom, range=[[1, 0], [7, 4]])))
    mkt_pts = base.transform_filter(alt.datum.Path == "Market").mark_point(
        filled=True, size=130, shape="circle")
    you_pts = base.transform_filter(alt.datum.Path == "Your view").mark_point(
        filled=False, size=130, shape="circle", strokeWidth=2.4)
    tags = base.transform_filter(alt.datum.Path == "Market").mark_text(
        dy=-15, fontSize=10.5, fontWeight="bold").encode(text="tag:N")
    rels = base.transform_filter(alt.datum.Path == "Market").mark_text(
        dy=17, fontSize=9.5, fontWeight="normal").encode(
        text="rel:N", color=alt.value("#9AA4B0"))
    layers = [lines, mkt_pts, you_pts, tags, rels]
    # rate decisions — the most visible overlay on the chart
    mdf = pd.DataFrame([{"Date": pd.Timestamp(m),
                         "What": f"{bank.meeting_name} decision · {m:%a %d %b %Y}"}
                        for m in bank.meetings if asof <= m <= horizon])
    if len(mdf):
        layers.append(alt.Chart(mdf).mark_rule(
            strokeDash=[6, 4], strokeWidth=1.8, opacity=0.65).encode(
            x=alt.X("Date:T", scale=x_scale),
            color=alt.value("#E8EAED"),
            tooltip=[alt.Tooltip("What:N", title="Decision")]))
        layers.append(alt.Chart(mdf).mark_text(
            fontSize=10, fontWeight="bold", angle=270, dx=-6, align="right",
            baseline="line-bottom", color="#E8EAED").encode(
            x=alt.X("Date:T", scale=x_scale), y=alt.value(0),
            text=alt.Text("Date:T", format="%d %b %y"),
            tooltip=[alt.Tooltip("What:N", title="Decision")]))
    # options / midcurves along the bottom rail
    if orail:
        odf = pd.DataFrame(orail)
        layers.append(alt.Chart(odf.query("Kind == 'Option'")).mark_point(
            filled=False, size=80, shape="diamond", strokeWidth=2).encode(
            x=alt.X("Date:T", scale=x_scale), y=alt.value(h - 12),
            color=alt.value(cc["muted"]),
            tooltip=[alt.Tooltip("What:N", title="Option expiry")]))
        mc = odf.query("Kind == '1Y midcurve'")
        if len(mc):
            layers.append(alt.Chart(mc).mark_point(
                filled=False, size=85, shape="triangle-up", strokeWidth=2).encode(
                x=alt.X("Date:T", scale=x_scale), y=alt.value(h - 26),
                color=alt.value(cc["muted"]),
                tooltip=[alt.Tooltip("What:N", title="Midcurve expiry")]))
    props = {"height": h}
    if front_note:
        props["title"] = alt.TitleParams(front_note, anchor="start", fontSize=13)
    brand.show_chart(alt.layer(*layers).properties(**props))
    st.caption(f"Solid = the market's term structure (contract codes above the points, ● at each "
               f"contract's last-trade day) · dashed gold = where the curve sits **if your odds "
               f"are right**. Dashed white verticals = {bank.meeting_name} decisions (dated at "
               "the top). Under each point, **±Nd** = days the contract expires before (−) or "
               "after (+) its nearest decision — hover for the full read including how many "
               "decisions it lives through. The axis is real calendar time and **ends at the "
               "last published decision**"
               + (f"; {n_dropped} contract(s) expiring beyond it aren't drawn"
                  if n_dropped else "")
               + ". Bottom rail: ◇ option expiries · △ 1Y midcurves.")


def render_stir_bank(bank_key: str) -> None:
    import altair as alt
    bank = stirpaths.BANKS[bank_key]
    prods_all = stirpaths.bank_products(bank_key)
    st.markdown(f"<h3>{_flag_img(bank_key, 17)}&nbsp; {bank.name} — implied path "
                "&amp; meeting scenarios</h3>", unsafe_allow_html=True)
    st.caption(
        "Top to bottom: where the futures trade now → what the market prices at each "
        f"**{bank.meeting_name}** decision → **your** odds (+% hike / −% cut) → where the futures "
        "land if you're right — then the term structure, market vs your view, with every decision "
        "and option expiry on it.")

    asof = datetime.now(ZoneInfo("America/New_York")).date()

    # ---- assumptions ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.5, 1.1, 1, 1])
    if bank_key == "FED":
        _b0 = f"{bank.default_rate - 0.125:.2f} – {bank.default_rate + 0.125:.2f}"
        band = c1.selectbox("Current target band (%)", _stir_fed_bands(),
                            index=(_stir_fed_bands().index(_b0)
                                   if _b0 in _stir_fed_bands() else 0),
                            key="spFED_band",
                            help="Today's FOMC target range. Sets the starting level of the path.")
        policy = float(band.split("–")[0]) + 0.125
    else:
        band = None
        policy = c1.number_input(f"{bank.rate_name} (%)", value=bank.default_rate,
                                 step=0.25, format="%.2f", key=f"sp{bank_key}_rate",
                                 help="Today's policy rate. Sets the starting level of the path.")
    proxy = {"FED": "SOFR", "ECB": "€STR", "BOE": "SONIA"}[bank_key]
    basis_bp = c2.number_input(f"{proxy} − policy basis (bp)",
                               value=stirpaths.BANK_BASIS_SEED[bank_key], step=0.5,
                               format="%.1f", key=f"sp{bank_key}_basis",
                               help=f"{proxy} trades a few bp around the policy rate. Shifts every "
                                    "fair value in parallel; cancels out of the implied move count.")
    n_q = c3.slider("Quarterly contracts", 4, 12, 8, key=f"sp{bank_key}_nq",
                    help="How far out the strip to price — whites + reds.")
    months = c4.slider("Timeline horizon (mo)", 6, 24, 15, key=f"sp{bank_key}_mo",
                       help="Horizon of the contract-windows timeline (in the expander below).")

    q0 = next(p for p in prods_all if p.quarterly)
    front_start = stirpaths.strip(q0, asof, 1)[0].start
    auto_stub = stirpaths.realized_stub_avg(bank, front_start, asof)
    with st.expander("Advanced"):
        a1, a2, a3 = st.columns(3)
        compound = a1.checkbox("Compound settlements (ACT/360)", value=True,
                               key=f"sp{bank_key}_cmp",
                               help="On = true daily-compounded convention for the 3M contracts. "
                                    "Off = simple average (convexity <~1bp/quarter).")
        hike_bp = a1.number_input("Hike step (bp)", value=bank.step_bp, step=5.0,
                                  format="%.0f", key=f"sp{bank_key}_hk")
        cut_bp = a2.number_input("Cut step (bp)", value=bank.step_bp, step=5.0,
                                 format="%.0f", key=f"sp{bank_key}_ct")
        haircut = a2.number_input("Term-premium haircut (bp/yr)", value=0.0, step=0.5,
                                  format="%.1f", key=f"sp{bank_key}_hc",
                                  help="Shaves bp per year off the DISPLAYED odds/cumulative path "
                                       "(prices and fair values stay untouched). 0 = off.")
        er_spread = 0.0
        if bank_key == "ECB":
            er_spread = a3.number_input("Euribor − €STR spread (bp)",
                                        value=stirpaths.PRODUCTS["ERA Comdty"].spread_bp,
                                        step=1.0, format="%.1f", key="spECB_ersp",
                                        help="Seeds the per-contract Spread column in the market-"
                                             "prices grid (Euribor is a forward-looking term fix). "
                                             "Default measured off the real strips vs €STR-fair.")
        stub = a3.number_input(
            f"Realized o/n avg since {front_start:%d %b} (%)",
            value=round(float(auto_stub), 3) if auto_stub is not None
            else round(policy + basis_bp / 100.0, 3),
            step=0.005, format="%.3f", key=f"sp{bank_key}_stub",
            help="Average overnight fixing over the front window's already-elapsed days — fixes "
                 "the front contract's odds after a mid-window move. Auto-seeded from the fixings "
                 "cache when available.")
        a3.caption(("auto from fixings cache" if auto_stub is not None
                    else "no fixings cache — seeded from today's rate; adjust after a recent move"))
        if stirpaths.MODE == "bloomberg" and IS_ADMIN:
            if a3.button("↻ Pull o/n fixings", key=f"sp{bank_key}_fixpull",
                         help="Refresh data/stir_fixings.json from the Terminal (3 index bdh pulls)."):
                got = stirpaths.refresh_fixings(asof)
                st.session_state.pop(f"sp{bank_key}_stub", None)
                st.session_state.pop(f"sp{bank_key}_stub_seed0", None)
                st.toast(f"Fixings refreshed: {got}" if got else "Pull failed (blocked/offline).")
                st.rerun()
    # remember the stub input's FIRST-render seed: the widget keeps its old
    # value when policy/basis change, so comparing against a RECOMPUTED seed
    # mistook the stale auto-seed for a deliberate user stub and silently
    # disabled the market solve
    _st0_key = f"sp{bank_key}_stub_seed0"
    st.session_state.setdefault(_st0_key, round(float(stub), 3))

    r0 = policy + basis_bp / 100.0                  # overnight-proxy level today, %
    # The market's own current-rate read (a no-meeting monthly = pure average
    # of the prevailing o/n rate) is the fit's starting level whenever the
    # policy input is UNTOUCHED (still the registry default) — the Aug FF/SR1
    # clean month knows the effective rate is 3.63x when the band mid says
    # 3.625, worth several points on the front meeting's odds, and it stays
    # right even when the registry has gone stale by a whole move. Touch the
    # band/rate input and your what-if always wins.
    _anchor = stirpaths.clean_month_anchor(bank_key, asof)
    _policy_touched = abs(policy - bank.default_rate) > 1e-9
    _anchor_used = (_anchor is not None and not _policy_touched
                    and abs(_anchor[0] - bank.default_rate) <= 0.40)
    if _anchor_used:
        r0 = _anchor[0] + basis_bp / 100.0

    sel = _stir_picker(bank_key, prods_all, label="Products in the tools")
    prods = [stirpaths.PRODUCTS[t] for t in sel] or [prods_all[0]]

    # ---- strips + market prices + per-contract spreads (editable) ------------
    strips = {p.ticker: stirpaths.strip(p, asof, n_q if p.quarterly else min(3 * n_q, 12))
              for p in prods}
    contracts, owner = [], []
    for p in prods:
        for c in strips[p.ticker]:
            contracts.append(c)
            owner.append(p)
    codes = [c.code for c in contracts]
    seed_spd = [er_spread if p.ticker == "ERA Comdty" else 0.0 for p in owner]

    px_key, spd_key, sig_key = (f"sp{bank_key}_px", f"sp{bank_key}_spd", f"sp{bank_key}_px_sig")
    src_key = f"sp{bank_key}_px_src"
    if st.session_state.get(sig_key) != tuple(codes):
        # seed ONLY here — no per-rerun feed work, and never a Bloomberg call: prices
        # come from the morning-snapshot store (mock offline). Live quotes arrive
        # solely via the ⚡ button below.
        feed_px = []
        for p in prods:
            feed_px += stirpaths.strip_prices(p, bank, strips[p.ticker], asof, r0)
        src, src_asof = stirpaths.strip_source(contracts)
        st.session_state[px_key] = dict(zip(codes, feed_px))
        st.session_state[spd_key] = dict(zip(codes, seed_spd))
        st.session_state[sig_key] = tuple(codes)
        # the untouched seed values, kept so the fit can tell a REAL page edit
        # (or store price) from a mock-backfilled display seed — mock-seeded
        # untouched codes must never enter a live fit as overrides
        st.session_state[f"sp{bank_key}_px_seed0"] = dict(zip(codes, feed_px))
        st.session_state[src_key] = ("morning snapshot · " + src_asof if src == "snapshot"
                                     else "synthetic demo")
    src_note = st.session_state.get(src_key, "synthetic demo")
    if stirpaths.MODE == "bloomberg" and IS_ADMIN:
        lp1, lp2 = st.columns([5.2, 1.2])
        lp1.caption(f"Prices: **{src_note}** — the page never pulls Bloomberg on its own; "
                    "the ⚡ button requests THIS strip's tickers only.")
        # the pull covers this bank's whole FIT universe (incl. fit-only
        # serials/monthlies) — refreshing only the displayed contracts left the
        # hidden instruments at morning vintage, and the serial-pair
        # differences turned that gap into phantom "pinned" meeting moves
        _live_cs = [c for p_, c in stirpaths.pull_universe(asof) if p_.bank == bank_key]
        if lp2.button("⚡ Live pull", key=f"sp{bank_key}_livepull", use_container_width=True,
                      help="One request for exactly this bank's strip + serial/monthly "
                           f"tickers ({len(_live_cs)} contracts) — nothing else touches "
                           "Bloomberg."):
            got = stirpaths.live_strip_prices(_live_cs)
            if got:
                cur = dict(st.session_state[px_key])
                cur.update(got)
                st.session_state[px_key] = cur
                st.session_state[src_key] = (f"⚡ live pull "
                                             f"{datetime.now(ZoneInfo('America/New_York')):%H:%M ET}"
                                             + (f" · {len(got)}/{len(codes)}"
                                                if len(got) < len(codes) else ""))
                st.rerun()
            else:
                st.error("Live pull returned nothing (blocked/offline?) — keeping the "
                         "existing prices. Nothing else was requested from Bloomberg.")
    else:
        st.caption(f"Prices: **{src_note}**.")
    with st.expander(f"✏️ Edit market prices / spreads — {' + '.join(p.short for p in prods)} "
                     f"strip · {src_note}"):
        st.caption("Overwrite any price with a live quote and the whole analysis re-prices off "
                   "it. **Spread (bp)** = settlement-index spread vs the overnight proxy per "
                   "contract.")
        px_df = pd.DataFrame({"Product": [p.short for p in owner], "Contract": codes,
                              "Window": [c.label for c in contracts],
                              "Market px": [st.session_state[px_key].get(c) for c in codes],
                              "Spread (bp)": [st.session_state[spd_key].get(c, s)
                                              for c, s in zip(codes, seed_spd)]})
        edited_px = st.data_editor(
            px_df, hide_index=True, use_container_width=True,
            # key carries the contract-set signature: a roll/product change discards
            # stale positional edits instead of re-applying them to different rows
            key=f"sp{bank_key}_px_editor_{abs(hash(tuple(codes))) % 99991}",
            column_config={
                "Product": st.column_config.TextColumn(disabled=True),
                "Contract": st.column_config.TextColumn(disabled=True),
                "Window": st.column_config.TextColumn(disabled=True),
                "Market px": st.column_config.NumberColumn(format="%.4f",
                                                           min_value=90.0, max_value=100.0),
                "Spread (bp)": st.column_config.NumberColumn(format="%.1f", step=0.5,
                                                             min_value=-50.0, max_value=50.0)})
        st.session_state[px_key] = dict(zip(edited_px["Contract"], edited_px["Market px"]))
        st.session_state[spd_key] = dict(zip(edited_px["Contract"], edited_px["Spread (bp)"]))
    prices = [float(st.session_state[px_key].get(c, 96.0)) for c in codes]
    spreads = [float(st.session_state[spd_key].get(c, s)) for c, s in zip(codes, seed_spd)]
    spread_of = dict(zip(codes, spreads))

    # ---- market-implied path: the FULL liquid-instrument fit ------------------
    # Display and fit are decoupled: the odds below come from EVERY liquid
    # instrument for this bank (quarterlies + SR1/FF monthlies + ER/€STR
    # serials), whatever the picker shows. Page-edited prices/spreads override
    # the store for the contracts on screen; the front stub is SOLVED from the
    # market unless real fixings exist or the Advanced input was changed
    # (changed vs its FIRST-render seed — see _st0_key above).
    _stub_arg = stub if (auto_stub is not None
                         or round(float(stub), 3) != st.session_state.get(
                             _st0_key, round(float(stub), 3))) else None
    # overrides = store-priced codes, genuine page edits, and live-pull
    # additions ONLY — an untouched display seed for a code the store lacks is
    # a MOCK number and must not slip into a real fit through the override door
    _have = stirpaths.store_codes()
    _seed0 = st.session_state.get(f"sp{bank_key}_px_seed0", {})
    _ov_px = {k: float(v) for k, v in (st.session_state.get(px_key) or {}).items()
              if k in _have or k not in _seed0 or abs(float(v) - float(_seed0[k])) > 1e-9}
    if bank_key == "ECB":
        # fit-only ER serials must price off the SAME basis as the page's
        # spread input — mixing the registry default with an edited page value
        # inside one least-squares printed phantom "pinned" meeting moves
        for _c in stirpaths.serial_strip(stirpaths.PRODUCTS["ERA Comdty"], asof):
            spread_of.setdefault(_c.code, er_spread)
    bf = stirpaths.bank_fit(bank_key, asof, r0=r0, stub_rate=_stub_arg,
                            override_prices=_ov_px,
                            override_spreads=spread_of)
    if bf is not None:
        ip = bf.implied
    else:                                           # nothing liquid (shouldn't happen)
        ip = stirpaths.implied_path(bank, contracts, prices, asof, r0, spreads,
                                    stub_rate=stub)
    # the stub every scenario/landing price shares: solved > fixings/manual input
    stub = float(ip.stub) if ip.stub is not None else stub
    # residuals by CODE: the fit universe is wider than the displayed contracts,
    # so positional zips against ip.* would misalign (or crash) — always map
    _resid_of = {c.code: float(r) for c, r in zip(ip.contracts, ip.residual_bp)}
    labels = [fedpath.meeting_label(m) for m in ip.meetings]
    yrs = np.array([(fedpath.effective_date(m) - asof).days / 365.25 for m in ip.meetings])
    cum_disp = np.array(ip.cum_bp) - haircut * yrs
    per_disp_arr = np.diff(np.concatenate([[0.0], cum_disp]))
    per_disp = dict(zip(labels, [float(v) for v in per_disp_arr]))
    mcol = [f"{m:%d %b %y}" for m in ip.meetings]   # meeting column headers

    # ---- scenario state: ONE signed % per meeting (+hike / −cut) -------------
    # 'mv2' namespace: the original keys picked up clamp-poisoned widget state on
    # long-lived tabs that no in-code heal could reliably exorcise — fresh keys
    # guarantee every session starts from the market seed.
    def mv_key(lab):
        return f"sp{bank_key}_mv2_{lab}"

    # the market's call in exact decimal odds — the seed AND the white/green/red anchor
    _mkt_dec = {lab: round(_stir_signed_pct(per_disp[lab], hike_bp, cut_bp) / 100.0, 3)
                for lab in labels}
    # per-meeting seeding: only meetings never seen get seeded — from the bank's
    # ★ default scenario when one is saved (Named scenarios expander), else from
    # the market's call; changing the contract slider / product set / horizon
    # never wipes odds already set
    _def_name = _stir_scen_default(bank_key)
    _def_vals = _stir_scen_all().get(bank_key, {}).get(_def_name, {}) if _def_name else {}
    # labels whose value came from a SCENARIO (default seed or Load) — exempt
    # from the rail-pin heal below, so a deliberate ±3.00 view isn't wiped
    _scn_seeded = st.session_state.setdefault(f"sp{bank_key}_scn_seeded", set())
    for lab in labels:
        if mv_key(lab) not in st.session_state:
            if lab in _def_vals:
                # scenarios ALWAYS store percent units (Save writes decimal*100)
                # — the conversion is unconditionally /100; an abs<=3 "already
                # decimal" heuristic corrupts small-odds views 100x
                st.session_state[mv_key(lab)] = float(_def_vals[lab]) / 100.0
                _scn_seeded.add(lab)
            else:
                st.session_state[mv_key(lab)] = _mkt_dec[lab]
        if abs(float(st.session_state[mv_key(lab)])) > 3.0:    # migrate old %-unit state
            st.session_state[mv_key(lab)] = float(st.session_state[mv_key(lab)]) / 100.0
    # heal the one-off clamp artifact: a value sitting EXACTLY on the ±3.00 rail is
    # (practically) never a real view — reseed that meeting from the market.
    # Scenario-sourced values are exempt: a saved ±3.00 was a deliberate choice.
    for lab in labels:
        if abs(float(st.session_state[mv_key(lab)])) == 3.0 and lab not in _scn_seeded:
            st.session_state[mv_key(lab)] = _mkt_dec.get(lab, 0.0)

    px_of_now = dict(zip(codes, prices))

    # your scenario, read from session up front (the editable widgets render in
    # section 2, but their values live in session state from the previous
    # interaction, so section 1 can already price off them)
    vals = {lab: float(st.session_state[mv_key(lab)]) * 100.0 for lab in labels}
    views = [stirpaths.MeetingView(m, max(vals[lab], 0.0) / 100.0, max(-vals[lab], 0.0) / 100.0,
                                   hike_bp, cut_bp)
             for m, lab in zip(ip.meetings, labels)]
    exp_moves = [v.expected_bp for v in views]
    scen_fn = stirpaths.scenario_rate_fn(r0, views, asof=asof, stub_rate=stub)

    def _fair(p, c):
        return fedpath.price(c, scen_fn, compound=(p.compound and compound)) \
            - spread_of.get(c.code, 0.0) / 100.0

    your_px = [_fair(p, c) for p, c in zip(owner, contracts)]
    fair_of = dict(zip(codes, your_px))
    diff_bp = [(y - m) * 100.0 for y, m in zip(your_px, prices)]

    _MKT_C, _YOU_C = "#E8EAED", "#F5C518"           # market/futures white · your gold
    _MTG_C, _FUT_C = "#7FB3F5", _MKT_C              # meetings blue · contracts = market white
    _MTG_TXT = "#BCD6F9"                            # meetings blue for TEXT — #7FB3F5 is a
                                                    # chart-accent weight, too dark to read
                                                    # at label size (Ben 2026-08-16); rules,
                                                    # tints and borders keep the deeper blue
    st.markdown("""<style>
      /* label recipe E (see brand.py): sentence case, app font, readable sizes */
      .sp-hdr { font-size: 0.85rem; letter-spacing: 0.01em; text-transform: none;
                font-weight: 600; padding: 0 0.1rem; }
      .sp-cell { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);
                 border-radius: 6px; height: 1.95rem; line-height: 1.85rem; padding: 0 0.45rem;
                 font-size: 0.85rem; white-space: nowrap; overflow: hidden; }
      .sp-sub { font-size: 0.72rem; color: #AEB7C2; }
      .sp-lab { font-weight: 600; font-size: 0.85rem; padding-top: 0.35rem;
                letter-spacing: 0.01em; }
    </style>""", unsafe_allow_html=True)
    _code_tip = {c.code: f"{p.short} {c.label}" for p, c in zip(owner, contracts)}

    def _rail(main, color, sub="", tip=""):
        s = (f"<div style='font-weight:400;font-size:0.68rem;color:#AEB7C2;"
             f"letter-spacing:0;text-transform:none;line-height:1.2'>{sub}</div>") if sub else ""
        return (f"<div class='sp-lab' style='color:{color};padding-top:0.1rem' "
                f"title='{tip}'>{main}{s}</div>")

    def _grid_hdr(cols_, title, items, color, tips=None, sub="", tip=""):
        cols_[0].markdown(_rail(title, color, sub, tip), unsafe_allow_html=True)
        for c_, it in zip(cols_[1:], items):
            t = f" title='{(tips or {}).get(it, '')}'" if tips else ""
            c_.markdown(f"<div class='sp-hdr' style='color:{color}'{t}>{it}</div>",
                        unsafe_allow_html=True)

    def _grid_row(cols_, label, label_color, cells, sub="", tip=""):
        cols_[0].markdown(_rail(label, label_color, sub, tip), unsafe_allow_html=True)
        for c_, html in zip(cols_[1:], cells):
            c_.markdown(f"<div class='sp-cell'>{html}</div>", unsafe_allow_html=True)

    # ---- 1 · the futures: market price over your fair, gap beneath -----------
    st.markdown(f"**1 · Futures — the market's prices over "
                f"<span style='color:{_YOU_C}'>yours</span>** &nbsp;·&nbsp; Δ row = the gap "
                "in bp (green = cheap vs your view / buy, red = rich / sell)",
                unsafe_allow_html=True)
    gC = [1] + [1] * len(codes)
    # code + its last-trade (expiry) date on a small second line
    _hdr_items, _hdr_tips = [], {}
    for p_, c_ in zip(owner, contracts):
        _lt = stirpaths.fut_last_trade(p_, c_)
        _it = (f"{c_.code}<div style='font-weight:400;font-size:0.68rem;"
               f"color:#AEB7C2;letter-spacing:0'>exp {_lt:%d %b %y}</div>")
        _hdr_items.append(_it)
        _hdr_tips[_it] = (f"{p_.short} {c_.label} · last trading day "
                          f"{_lt:%a %d %b %Y}")
    _grid_hdr(st.columns(gC, gap="small"), "Futures", _hdr_items, _FUT_C, _hdr_tips,
              sub="code · expiry", tip="Each column is one listed futures contract; "
                                       "the small date is its last trading day")
    _grid_row(st.columns(gC, gap="small"), "Market", _MKT_C,
              [f"{px_of_now[c]:.4f}" for c in codes],
              sub="price trading now", tip="The live market price of each contract")
    # editable BOTH ways: these price cells re-solve the §2 odds when typed into,
    # and re-derive from the §2 odds whenever those change
    def _fpx_edit():
        tgt = [float(st.session_state.get(f"sp{bank_key}_fpx_{c}", px_of_now[c]))
               for c in codes]
        # invert (linear), then refine: the inversion is simple-average while the
        # forward pricing compounds, so iterate the ~1-2bp convexity wedge away
        adj, ip2 = list(tgt), None
        for _ in range(3):
            ip2 = stirpaths.implied_path(bank, contracts, adj, asof, r0, spreads,
                                         stub_rate=stub)
            vs = [stirpaths.MeetingView(m_, max(float(b_), 0.0) / hike_bp,
                                        max(-float(b_), 0.0) / cut_bp, hike_bp, cut_bp)
                  for m_, b_ in zip(ip2.meetings, ip2.per_meeting_bp)]
            fn_ = stirpaths.scenario_rate_fn(r0, vs, asof=asof, stub_rate=stub)
            fwd = [fedpath.price(c_, fn_, compound=(p_.compound and compound))
                   - spread_of.get(c_.code, 0.0) / 100.0
                   for p_, c_ in zip(owner, contracts)]
            adj = [a_ + (t_ - f_) for a_, t_, f_ in zip(adj, tgt, fwd)]
        for m_, bp_ in zip(ip2.meetings, ip2.per_meeting_bp):
            lab_ = fedpath.meeting_label(m_)
            if lab_ in labels:
                st.session_state[mv_key(lab_)] = round(
                    _stir_signed_pct(float(bp_), hike_bp, cut_bp) / 100.0, 3)

    def _fpx_bump(k: str, d: float) -> None:
        st.session_state[k] = round(min(100.0, max(90.0,
                                    float(st.session_state.get(k, 96.0)) + d)), 4)
        _fpx_edit()                                 # a stepped price re-solves the odds too

    _fwrap = f"sp{bank_key}_fpxwrap"
    st.markdown(f"""<style>
      .st-key-{_fwrap} div[data-testid="stVerticalBlock"] {{ gap: 0.08rem; }}
      .st-key-{_fwrap} div[data-testid="stHorizontalBlock"] {{ gap: 0.25rem; }}
      .st-key-{_fwrap} div[data-testid="stElementContainer"] {{
          margin: 0; padding: 0; min-height: 0; }}
      .st-key-{_fwrap} button {{
          min-height: 0.9rem; height: 0.9rem; padding: 0; border-radius: 3px;
          width: 100%; display: flex; align-items: center; justify-content: center; }}
      .st-key-{_fwrap} button p {{
          font-size: 0.65rem; line-height: 1; margin: 0; padding: 0; }}
      .st-key-{_fwrap} div[data-testid="stNumberInputContainer"] {{
          height: 1.95rem; background: rgba(255,255,255,0.06);
          border: 1px solid rgba(245,197,24,0.55) !important; border-radius: 6px; }}
      .st-key-{_fwrap} div[data-testid="stNumberInputContainer"] > div {{
          background: transparent; }}
      .st-key-{_fwrap} div[data-testid="stNumberInput"] input {{
          padding: 0.2rem 0.45rem; font-size: 0.85rem; font-weight: 600; }}
    </style>""", unsafe_allow_html=True)
    with st.container(key=_fwrap):
        # DISPLAY-ONLY since the vertical meetings table took over the editing
        # (its YOUR FUT column uses these same sp_fpx keys — a key can only own
        # one widget). Slaving the keys here keeps _fpx_edit's inversion seeing
        # odds-consistent values for every displayed contract.
        ycols = st.columns(gC, gap="small")
        ycols[0].markdown(_rail("Your call", _YOU_C, "your fair — edit in §2 ▼",
                                "Your fair value per contract, re-priced from your odds. "
                                "Editing is two-way in the meetings table below: type odds "
                                "OR a YOUR FUT price and the other re-solves"),
                          unsafe_allow_html=True)
        for c_, code in zip(ycols[1:], codes):
            st.session_state[f"sp{bank_key}_fpx_{code}"] = round(fair_of[code], 4)
            c_.markdown(f"<div class='sp-cell' style='border:1px solid "
                        f"rgba(245,197,24,0.35);color:#F5C518;font-weight:600' "
                        f"title='Your fair for {code} under your §2 odds'>"
                        f"{fair_of[code]:.4f}</div>", unsafe_allow_html=True)
    _d_cols = st.columns(gC, gap="small")
    _d_cols[0].markdown(_rail("Δ bp", "#E8EAED", "your fair − market price",
                              "The gap between your fair value and the live market, in "
                              "basis points — green = cheap vs your view, red = rich"),
                        unsafe_allow_html=True)
    for c_, d in zip(_d_cols[1:], diff_bp):
        tone = ("#9AA4B0" if abs(d) < 0.05 else "#66BB6A" if d > 0 else "#EF5350")
        c_.markdown(f"<div class='sp-cell' style='color:{tone};font-weight:700'>{d:+.1f}</div>",
                    unsafe_allow_html=True)

    def _sp_gap(h: float = 0.9) -> None:
        st.markdown(f"<div style='height:{h}rem'></div>", unsafe_allow_html=True)

    _sp_gap(0.5)
    with st.expander("💾 Named scenarios — save / load the desk's cases"):
        sc_all = _stir_scen_all().get(bank_key, {})
        cur_def = _stir_scen_default(bank_key)
        s1, s2, s3, s4, s5 = st.columns([1.7, 0.8, 1.6, 0.8, 1.1],
                                        vertical_alignment="bottom")
        new_name = s1.text_input("Save current odds as", key=f"sp{bank_key}_scn_name",
                                 placeholder="base / hawkish / dovish …")
        if s2.button("Save", key=f"sp{bank_key}_scn_save", use_container_width=True):
            if new_name.strip():
                _stir_scen_save(bank_key, new_name.strip(), vals)
                st.toast(f"Saved scenario '{new_name.strip()}'."); st.rerun()
        pickn = s3.selectbox("Saved scenarios", ["—"] + sorted(sc_all),
                             key=f"sp{bank_key}_scn_pick",
                             format_func=lambda n: f"★ {n}" if n == cur_def else n)

        def _scn_load(name):
            # runs as a CALLBACK (before the next render), so writing the odds
            # widgets' keys is always legal regardless of where this expander
            # sits relative to the inputs
            for lab_, v_ in _stir_scen_all().get(bank_key, {}).get(name, {}).items():
                if lab_ in labels:
                    # percent -> decimal, unconditionally (see the seed loop note:
                    # an abs<=3 heuristic corrupts small-odds views 100x)
                    st.session_state[mv_key(lab_)] = float(v_) / 100.0
                    st.session_state.setdefault(f"sp{bank_key}_scn_seeded", set()).add(lab_)

        s4.button("Load", key=f"sp{bank_key}_scn_load", use_container_width=True,
                  disabled=(pickn == "—"), on_click=_scn_load, args=(pickn,))
        is_def = pickn != "—" and pickn == cur_def
        if s5.button("☆ Clear default" if is_def else "★ Set as default",
                     key=f"sp{bank_key}_scn_def", use_container_width=True,
                     disabled=(pickn == "—"),
                     help="The default scenario seeds YOUR CALL automatically when "
                          "this page opens (instead of the market's odds)."):
            _stir_scen_set_default(bank_key, None if is_def else pickn)
            st.toast("Default cleared." if is_def
                     else f"'{pickn}' now loads on open."); st.rerun()
        if pickn != "—" and st.button(f"🗑 Delete '{pickn}'", key=f"sp{bank_key}_scn_del"):
            _stir_scen_delete(bank_key, pickn); st.rerun()
        _tail = "Scenarios persist to data/stirpaths_scenarios.json."
        st.caption((f"★ **{cur_def}** seeds YOUR CALL whenever this page opens — "
                    f"meetings it doesn't cover seed from the market. {_tail}")
                   if cur_def else
                   f"★ Set as default makes a scenario load automatically on open. {_tail}")

    # ---- 2+3 · one table: the market's call, your call directly beneath ------
    _sp_gap()
    h1_, h2_, h3_ = st.columns([4.2, 0.9, 1.5])
    h1_.markdown(f"**2 · The odds, decision by decision — two-way with the "
                 f"<span style='color:{_YOU_C}'>YOUR CALL</span> prices in section 1** "
                 "(edit either; the other re-solves) &nbsp;·&nbsp; decimal odds: "
                 "−0.66 = 66% cut odds · +0.50 = 50% hike odds · beyond ±1 = more than one step"
                 + (f" · after the {haircut:.1f}bp/yr haircut" if haircut else ""),
                 unsafe_allow_html=True)
    if h2_.button("Hold all", key=f"sp{bank_key}_hold", use_container_width=True,
                  help="Zero every meeting — a flat 'no change' scenario."):
        for lab in labels:
            st.session_state[mv_key(lab)] = 0.0
        st.rerun()
    if h3_.button("Seed from market", key=f"sp{bank_key}_seed", use_container_width=True,
                  help="Set every meeting to EXACTLY what the strip prices — everything "
                       "starts white; only your edits colour up."):
        for lab in labels:
            st.session_state[mv_key(lab)] = _mkt_dec[lab]
        st.rerun()

    # Streamlit's native steppers aren't rendered at these widths (width-gated in
    # the React tree), so each YOUR-CALL cell is [number][＋ over －]: our own
    # buttons, shrunk to spinner size by CSS scoped to this container. The strip
    # sits flush under the MARKET row so the two read as one table.
    def _stir_bump(k: str, d: float) -> None:
        v = float(st.session_state.get(k, 0.0))
        g = round(v / 0.05) * 0.05
        if abs(v - g) > 1e-9:                       # off-grid (exact market seed):
            nv = (np.floor if d < 0 else np.ceil)(v / 0.05) * 0.05   # snap in click direction
        else:
            nv = v + d
        st.session_state[k] = round(float(min(3.0, max(-3.0, nv))), 2)

    _wrap = f"sp{bank_key}_oddswrap"
    st.markdown(f"""<style>
      .st-key-{_wrap} div[data-testid="stVerticalBlock"] {{ gap: 0.08rem; }}
      .st-key-{_wrap} div[data-testid="stHorizontalBlock"] {{ gap: 0.25rem; }}
      .st-key-{_wrap} div[data-testid="stElementContainer"] {{
          margin: 0; padding: 0; min-height: 0; }}
      .st-key-{_wrap} button {{
          min-height: 0.9rem; height: 0.9rem; padding: 0; border-radius: 3px;
          width: 100%; display: flex; align-items: center; justify-content: center; }}
      .st-key-{_wrap} button p {{
          font-size: 0.65rem; line-height: 1; margin: 0; padding: 0; }}
      .st-key-{_wrap} div[data-testid="stNumberInputContainer"] {{
          height: 1.95rem; background: rgba(255,255,255,0.06);
          border: 1px solid rgba(245,197,24,0.55) !important; border-radius: 6px; }}
      .st-key-{_wrap} div[data-testid="stNumberInputContainer"] > div {{
          background: transparent; }}
      .st-key-{_wrap} div[data-testid="stNumberInput"] input {{
          padding: 0.2rem 0.45rem; font-size: 0.85rem; font-weight: 600; }}
      /* the vertical table's columns are wide enough that Streamlit's NATIVE
         steppers render (they were width-gated out of the old narrow cells) —
         hide them, ours carry the snap/re-solve logic */
      .st-key-{_wrap} button[data-testid="stNumberInputStepUp"],
      .st-key-{_wrap} button[data-testid="stNumberInputStepDown"] {{
          display: none !important; }}
      /* Streamlit gives stMarkdownContainer margin-bottom:-14px, collapsing its
         layout box to 13px while our 1.95rem sp-cell OVERFLOWS it — the row's
         centre-alignment then aligns the 13px box, printing text cells 7px
         lower than the widget cells (measured in the DOM). Restore true height. */
      .st-key-{_wrap} [data-testid="stMarkdownContainer"] {{
          margin-bottom: 0 !important; }}
    </style>""", unsafe_allow_html=True)
    pairs = list(zip(ip.meetings, labels))
    # WIRP-style VERTICAL table (Ben, 2026-08-15): one ROW per meeting, columns
    # mirroring the WIRP screen (%Hike/Cut · #Hikes/Cuts · Imp Rate Δ · Implied
    # Rate) plus our SETTLES INTO and the editable YOUR CALL — compact, and
    # familiar to anyone who has lived on WIRP.
    _vgrid = [1.0, 0.5, 0.8, 0.95, 0.6, 0.6, 0.72, 1.08, 0.5, 1.18]
    # which quarterly future each decision SETTLES INTO (windows tile, so each
    # meeting maps to exactly one) — shading groups meetings sharing a contract.
    # Beside it: that contract's latest SETTLE from the snapshot store (the
    # morning pull's numbers — manual Terminal entry while the pull is blocked),
    # deliberately NOT the page-edited price: this column is last night's mark.
    _sstore = stirpaths._load_strip_store()
    _spx = {**_sstore.get("settles", {}), **_sstore.get("prices", {})}
    _s_asof = _sstore.get("asof", "")
    qprod = next((p for p in prods if p.quarterly), prods[0])
    qstrip = stirpaths.strip(qprod, asof, n_q + 4)
    win_cells, _tints, _ti, _last = [], ["rgba(127,179,245,0.10)",
                                         "rgba(127,179,245,0.24)"], 0, None
    for m, lab in pairs:
        eff = stirpaths.bank_effective_date(bank, m)
        wc = next((c for c in qstrip if c.start <= eff < c.end), None)
        tag = wc.code[-2:] if wc else "—"
        first = tag != _last                        # first row of a contract group
        if first:
            _ti, _last = 1 - _ti, tag
        tip = (f"{bank.meeting_name} {m:%d %b %y} settles inside {wc.code} "
               f"(window {wc.start:%d %b %y} → {wc.end:%d %b %y})" if wc else "")
        _sv = _spx.get(wc.code) if wc else None
        settle_txt = f"{_sv:.4f}" if _sv is not None else "—"
        settle_tip = (f"{wc.code} settlement, snapshot store · {_s_asof}"
                      if wc and _sv is not None else
                      "no stored settlement for this contract")
        win_cells.append((tag, tip, _ti, settle_txt, settle_tip,
                          wc.code if wc else None, first))
    # blue tints for the market-side INTO/SETTLE groups, gold twins for the
    # your-side repeat tag — blue = market, gold = your call, everywhere
    _gold_tints = ["rgba(245,197,24,0.08)", "rgba(245,197,24,0.20)"]
    _mkt_wash = "rgba(127,179,245,0.07)"
    # your number: green when above the market's call, red when below
    _tone_css = []
    for i, (m, lab) in enumerate(pairs):
        mine = float(st.session_state[mv_key(lab)])
        tone = ("#66BB6A" if mine > _mkt_dec[lab] + 0.012 else
                "#EF5350" if mine < _mkt_dec[lab] - 0.012 else "#E8EAED")
        _tone_css.append(f".st-key-sp{bank_key}_oc_{i} input "
                         f"{{ color: {tone} !important; }}")
    st.markdown("<style>" + "\n".join(_tone_css) + "</style>", unsafe_allow_html=True)
    _pin_of = dict(zip(labels, bf.pinned)) if bf is not None else {}
    step_bp = float(hike_bp)
    with st.container(key=_wrap):
        # group band: BLUE = the market's side, GOLD = yours — the split at a glance
        band = st.columns([sum(_vgrid[:7]), sum(_vgrid[7:])], gap="small")
        band[0].markdown(f"<div style='color:{_MTG_TXT};border-bottom:2px solid {_MTG_C};"
                         "font-size:1.0rem;letter-spacing:0.02em;font-weight:700;"
                         "padding:0 0.1rem 2px' title='Everything in blue is the MARKET: "
                         "what the futures strip prices right now'>Market — priced in now"
                         "</div>", unsafe_allow_html=True)
        band[1].markdown(f"<div style='color:{_YOU_C};border-bottom:2px solid {_YOU_C};"
                         "font-size:1.0rem;letter-spacing:0.02em;font-weight:700;"
                         "padding:0 0.1rem 2px' title='Everything in gold is YOURS: "
                         "editable, two-way'>Your call — editable</div>",
                         unsafe_allow_html=True)
        # NB: tooltips render inside title='…' HTML attributes — an apostrophe
        # in the text TERMINATES the attribute and mangles the tag (a header
        # once rendered unstyled because of a stray "you're"). Keep them
        # apostrophe-free.
        hdr = st.columns(_vgrid, gap="small", vertical_alignment="bottom")
        for c_, (txt, col, tip) in zip(hdr, [
                ("Meeting", _MTG_TXT, f"Scheduled {bank.meeting_name} decision dates"),
                ("Into", _MTG_TXT, "The futures contract whose settlement this decision "
                                 "feeds into — codes match section 1"),
                ("Settle", _MTG_TXT, "The latest stored settlement of that contract "
                                   f"(morning snapshot store · {_s_asof or 'demo'}) — "
                                   "the overnight mark, not the editable page price"),
                ("% hike/cut", _MTG_TXT, "The move the strip prices AT this meeting: "
                                       "+0.35 = 35% odds of one hike, −0.66 = 66% odds "
                                       "of a cut · ≈ = interpolated, no contract "
                                       "isolates this meeting"),
                ("# h/c", _MTG_TXT, "Cumulative hikes/cuts priced THROUGH this meeting "
                                  "(in steps) — the #Hikes/Cuts column on WIRP"),
                ("Cum bp", _MTG_TXT, "Cumulative bp priced through this meeting — "
                                   "the Imp Rate Δ column on WIRP"),
                ("Implied", _MTG_TXT, "The overnight-proxy level the strip implies "
                                    "after this meeting — the Implied Rate column "
                                    "on WIRP"),
                ("Your call", _YOU_C, "Your odds per decision (type or ＋/－) — "
                                      "Your fut, the Δ bp row and the gold curve "
                                      "re-price from these"),
                ("Into", _YOU_C, "The same contract tag again, so the price being "
                                 "edited is never ambiguous — gold groups match "
                                 "the blue ones on the left"),
                ("Your fut settle", _YOU_C, "Where the Into future lands under YOUR odds — "
                                     "two-way: type a target price here and the odds "
                                     "of the meetings inside that contract re-solve. "
                                     "One editor per contract, on its first row")]):
            c_.markdown(f"<div class='sp-hdr' style='color:{col}' title='{tip}'>{txt}</div>",
                        unsafe_allow_html=True)
        for i, (m, lab) in enumerate(pairs):
            row = st.columns(_vgrid, gap="small", vertical_alignment="center")
            row[0].markdown(f"<div class='sp-cell' style='color:{_MTG_TXT};font-weight:700' "
                            f"title='{bank.meeting_name} decision · {m:%A %d %B %Y}'>"
                            f"{m:%d %b %y}</div>", unsafe_allow_html=True)
            tag, tip, _gi, settle_txt, settle_tip, wc_code, first = win_cells[i]
            _bt = ["rgba(127,179,245,0.10)", "rgba(127,179,245,0.24)"][_gi]
            _gt = _gold_tints[_gi]
            row[1].markdown(f"<div class='sp-cell' style='background:{_bt};"
                            f"font-size:0.75rem;font-weight:700' title='{tip}'>{tag}</div>",
                            unsafe_allow_html=True)
            row[2].markdown(f"<div class='sp-cell' style='background:{_bt};"
                            f"font-size:0.8rem' title='{settle_tip}'>{settle_txt}</div>",
                            unsafe_allow_html=True)
            _pinned = _pin_of.get(lab, True)
            _tip = (f"{bank.meeting_name} {m:%a %d %b %Y} · "
                    + ("pinned by its own contract — trust it" if _pinned else
                       "no single contract isolates this meeting — interpolated, indicative"))
            row[3].markdown(f"<div class='sp-cell' style='background:{_mkt_wash}' "
                            f"title='{_tip}'>"
                            f"{'' if _pinned else '≈ '}{_mkt_dec[lab]:+.2f} "
                            f"<span class='sp-sub'>({per_disp[lab]:+.1f}bp)</span></div>",
                            unsafe_allow_html=True)
            row[4].markdown(f"<div class='sp-cell' style='background:{_mkt_wash}'>"
                            f"{cum_disp[i] / step_bp:+.2f}</div>",
                            unsafe_allow_html=True)
            row[5].markdown(f"<div class='sp-cell' style='background:{_mkt_wash}'>"
                            f"{cum_disp[i]:+.1f}</div>",
                            unsafe_allow_html=True)
            row[6].markdown(f"<div class='sp-cell' style='background:{_mkt_wash}'>"
                            f"{r0 + cum_disp[i] / 100.0:.3f}</div>",
                            unsafe_allow_html=True)
            with row[7]:
                with st.container(key=f"sp{bank_key}_oc_{i}"):
                    cin, cbtn = st.columns([3.1, 0.9], gap="small")
                    cin.number_input(lab, min_value=-3.0, max_value=3.0, step=0.05,
                                     format="%.2f", key=mv_key(lab),
                                     label_visibility="collapsed",
                                     help=f"{bank.meeting_name} {m:%a %d %b %Y}")
                    with cbtn:
                        st.button("＋", key=f"sp{bank_key}_up_{lab}",
                                  use_container_width=True,
                                  on_click=_stir_bump, args=(mv_key(lab), 0.05))
                        st.button("－", key=f"sp{bank_key}_dn_{lab}",
                                  use_container_width=True,
                                  on_click=_stir_bump, args=(mv_key(lab), -0.05))
            row[8].markdown(f"<div class='sp-cell' style='background:{_gt};color:#FFE08C;"
                            f"font-size:0.75rem;font-weight:700' title='{tip}'>{tag}</div>",
                            unsafe_allow_html=True)
            with row[9]:
                # YOUR FUT: two-way editor on the INTO contract — first row of
                # each contract group only (one contract = one price; later rows
                # of the group show it dimmed). Shares §1's sp_fpx keys, slaved
                # to the odds each run; typing re-solves the odds via _fpx_edit.
                if wc_code and wc_code in fair_of and first:
                    _fk = f"sp{bank_key}_fpx_{wc_code}"
                    with st.container(key=f"sp{bank_key}_vf_{i}"):
                        fin, fbtn = st.columns([3.1, 0.9], gap="small")
                        fin.number_input(wc_code, min_value=90.0, max_value=100.0,
                                         step=0.005, format="%.4f", key=_fk,
                                         label_visibility="collapsed",
                                         on_change=_fpx_edit,
                                         help=f"Where {wc_code} lands under your odds — "
                                              "type a target and the odds of its "
                                              "meetings re-solve")
                        with fbtn:
                            st.button("＋", key=f"sp{bank_key}_fup_{wc_code}",
                                      use_container_width=True,
                                      on_click=_fpx_bump, args=(_fk, 0.005))
                            st.button("－", key=f"sp{bank_key}_fdn_{wc_code}",
                                      use_container_width=True,
                                      on_click=_fpx_bump, args=(_fk, -0.005))
                elif wc_code and wc_code in fair_of:
                    st.markdown(f"<div class='sp-cell' style='opacity:0.45;color:#F5C518' "
                                f"title='Same contract ({wc_code}) — edit on its first "
                                f"row above'>{fair_of[wc_code]:.4f}</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("<div class='sp-cell'>—</div>", unsafe_allow_html=True)
    st.caption("One row per decision, WIRP's column conventions: **%HIKE/CUT** = the move "
               "priced at that meeting alone (≈ = interpolated) · **#H/C** and **CUM BP** = "
               "cumulative through it · **IMPLIED** = the o/n level after it. The two gold "
               "columns are **two-way**: edit **YOUR CALL** odds and YOUR FUT re-prices, or "
               "type a target in **YOUR FUT** and the odds of that contract's meetings "
               "re-solve (one price editor per contract, on its first row — green/red = "
               "more hawkish/dovish than the market). **INTO** / **SETTLE** = the quarterly "
               "future the decision settles into and its latest stored settlement.")
    if bf is not None:
        _bits = [f"market odds fit from **{bf.n_instruments} liquid instruments** — quarterlies "
                 "+ monthlies/serials, independent of the products displayed above"]
        if ip.stub is not None and _stub_arg is None:
            _bits.append(f"front stub **solved from the market**: {ip.stub:.3f}%")
        if _anchor is not None:
            _av = _anchor[0] + basis_bp / 100.0
            _bits.append(f"current {proxy} per the {_anchor[1]} clean month: **{_av:.3f}%**"
                         + (" (anchors the fit)" if _anchor_used
                            else " — your page setting drives the fit instead"))
        st.caption(" &nbsp;·&nbsp; ".join(_bits))
        if _anchor is not None and abs(_anchor[0] - bank.default_rate) > 0.10:
            _gap = abs(_anchor[0] - bank.default_rate) * 100
            st.warning(f"The market's clean-month read of the current policy rate "
                       f"({_anchor[0]:.2f}%) sits {_gap:.0f}bp from the app's registry setting "
                       f"({bank.default_rate:.2f}%) — the registry looks stale. "
                       + ("The fit already uses the market's read; update stirpaths.BANKS "
                          "default_rate when convenient."
                          if _anchor_used else
                          "The fit is running on YOUR page setting, which disagrees with the "
                          "market's read — double-check the level before quoting these odds."))
    # ---- at a glance ---------------------------------------------------------
    st.markdown("#### At a glance")
    m1, m2, m3, m4 = st.columns(4)
    next_bp = per_disp[labels[0]] if labels else 0.0
    m1.metric(f"Next meeting — {labels[0] if labels else '—'}", f"{next_bp:+.0f} bp",
              help=f"Curve-implied move at the next {bank.meeting_name} decision.")
    m1.caption(f"market: {_stir_odds_str(next_bp, bank.step_bp)}")
    eoy = [i for i, m in enumerate(ip.meetings) if m.year == asof.year]
    if eoy:
        yend = float(cum_disp[eoy[-1]])
        m2.metric(f"Priced through Dec {asof.year}", f"{yend:+.0f} bp",
                  help="Cumulative move the strip prices from now to the last meeting this year.")
        m2.caption(f"≈ {yend / bank.step_bp:+.1f} × {bank.step_bp:.0f}bp")
    m3.metric("Terminal (last covered mtg)", f"{policy + float(cum_disp[-1]) / 100:.2f}%",
              help=f"Curve-implied {bank.rate_name.lower()} at {labels[-1] if labels else '—'}.")
    m3.caption(f"{float(cum_disp[-1]):+.0f} bp vs today"
               + (f" · {haircut:.1f}bp/yr haircut on" if haircut else ""))
    scen_cum = float(np.sum(exp_moves))
    m4.metric("Your terminal", f"{policy + scen_cum / 100:.2f}%",
              help="Probability-weighted landing rate under your meeting odds.")
    m4.caption(f"{scen_cum:+.0f} bp expected vs today")

    # ---- what's priced, meeting by meeting -----------------------------------
    with st.expander("📋 What's priced, meeting by meeting (table)"):
        ot = pd.DataFrame([{
            "Decision": f"{m:%a %d %b %y}",
            "In": f"{(m - asof).days}d",
            "Implied (bp)": per_disp[lab],
            "Market odds": _stir_odds_str(per_disp[lab], bank.step_bp),
            "Your call (%)": vals[lab],
            "E[move] (bp)": v.expected_bp,
            "Cum (bp)": float(cum_disp[i]),
        } for i, (m, lab, v) in enumerate(zip(ip.meetings, labels, views))])
        brand.themed_dataframe(
            ot, fmt={"Implied (bp)": "{:+.1f}".format, "Your call (%)": "{:+.0f}".format,
                     "E[move] (bp)": "{:+.1f}".format, "Cum (bp)": "{:+.1f}".format},
            height=min(430, 45 + 35 * len(ot)))

    # ---- the term structure: market vs your view -----------------------------
    _sp_gap()
    st.markdown("#### Term structure — market vs your view")
    fronts = []
    for p in prods:
        live = [c for c in strips[p.ticker] if stirpaths.fut_last_trade(p, c) >= asof]
        if live and live[0].code in px_of_now:
            fronts.append(f"{p.short} front {live[0].code} @ {px_of_now[live[0].code]:.4f}")
    _stir_term_chart(prods, bank, strips, px_of_now, fair_of, asof,
                     front_note="   ·   ".join(fronts))

    # ---- WIRP-style combined chart: rate lines over cumulative-steps bars ----
    # WIRP's bars and line are the SAME series in two units (rate = current +
    # cum steps x step size), so double-encoding the market this way adds no
    # noise — and your view rides on top as ONE extra dashed gold line.
    cc = brand.chart_colors()
    step_bp_ = float(hike_bp) or 25.0
    lab_x = ["Current"] + [f"{m:%d %b %y}" for m in ip.meetings]
    mkt_rate = np.concatenate([[policy], policy + cum_disp / 100.0])
    your_rate = np.concatenate([[policy], policy + np.cumsum(exp_moves) / 100.0])
    bars_df = pd.DataFrame({"Meeting": lab_x,
                            "Steps": np.concatenate([[0.0], cum_disp / step_bp_])})
    dom = ["Market-implied", "Your scenario (expected)"]
    rng = [cc["series"], cc["accent"]]
    lines_df = pd.concat([
        pd.DataFrame({"Meeting": lab_x, "rate": mkt_rate, "Path": dom[0]}),
        pd.DataFrame({"Meeting": lab_x, "rate": your_rate, "Path": dom[1]}),
    ])
    x_enc = alt.X("Meeting:N", sort=lab_x, title=None,
                  axis=alt.Axis(labelAngle=-35))
    bars = alt.Chart(bars_df).mark_bar(size=24, opacity=0.35, color=_MTG_C).encode(
        x=x_enc,
        y=alt.Y("Steps:Q", title="# hikes/cuts priced (cum)",
                axis=alt.Axis(orient="right")),
        tooltip=[alt.Tooltip("Meeting:N"),
                 alt.Tooltip("Steps:Q", title="Cum steps (market)", format="+.2f")])
    lines = alt.Chart(lines_df).mark_line(strokeWidth=3).encode(
        x=x_enc,
        y=alt.Y("rate:Q", title=f"{bank.rate_name} (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("Path:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("Path:N", scale=alt.Scale(
            domain=dom, range=[[1, 0], [6, 4]]), legend=None),
        tooltip=[alt.Tooltip("Meeting:N"), alt.Tooltip("Path:N"),
                 alt.Tooltip("rate:Q", title="Rate", format=".3f")])
    pts = alt.Chart(lines_df).mark_point(filled=True, size=45).encode(
        x=x_enc, y=alt.Y("rate:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Path:N", scale=alt.Scale(domain=dom, range=rng), legend=None))
    st.markdown(f"**Implied {bank.rate_name.lower()} & hikes/cuts priced — "
                "market vs your view**")
    brand.show_chart(alt.layer(bars, lines + pts).resolve_scale(y="independent")
                     .properties(height=330))
    st.caption("The WIRP read: bars = cumulative hikes/cuts the market prices through "
               "each meeting (right axis) · solid line = the same thing as a rate level "
               "(left axis) · dashed gold = where YOUR odds put the rate. This chart is "
               "the futures chart above turned upside-down — price = 100 − rate, so a "
               "falling strip IS a rising path.")

    # ---- contract-windows timeline (the meeting-risk Gantt), now secondary ---
    with st.expander("🗓️ Contract windows & meeting-risk timeline"):
        ann = {"px": px_of_now, "fair": fair_of,
               "front": "",
               "mkt": {m: _stir_signed_pct(per_disp[lab], hike_bp, cut_bp)
                       for m, lab in zip(ip.meetings, labels)},
               "you": {m: vals[lab] for m, lab in zip(ip.meetings, labels)}}
        _stir_timeline([p.ticker for p in prods], [bank_key], asof, months,
                       key=bank_key, default_view="Contract windows", ann=ann)
        _stir_window_table(prods, [bank_key], asof, n_q)

    # ---- contract detail (per product) ---------------------------------------
    st.markdown("#### Contract detail")
    tbl = pd.DataFrame({
        "Product": [p.short for p in owner],
        "Contract": codes,
        "Window": [c.label for c in contracts],
        "Market": prices,
        "Mkt-implied rate": [100 - p for p in prices],
        "Your fair": your_px,
        "Diff (bp)": diff_bp,
        "Diff (/lot)": [d * p.bp_value for d, p in zip(diff_bp, owner)],
        "vs fit (bp)": [_resid_of.get(c.code, float("nan")) for c in contracts],
    })
    fmt = {"Market": "{:.4f}".format, "Mkt-implied rate": "{:.3f}".format,
           "Your fair": "{:.4f}".format, "Diff (bp)": "{:+.1f}".format,
           "Diff (/lot)": (bank.ccy + "{:,.0f}").format,
           "vs fit (bp)": (lambda v: "—" if v != v else f"{v:+.1f}")}

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
    brand.themed_dataframe(tbl, fmt,
                           colorers=[(["Diff (bp)", "Diff (/lot)", "vs fit (bp)"], _color_diff)],
                           height=min(420, 45 + 35 * len(tbl)))
    bp_note = " · ".join(f"{p.short} {bank.ccy}{p.bp_value:.2f}/bp" for p in prods)
    st.caption("**Your fair** = the price implied by your probability-weighted meeting path. "
               "Positive diff = the contract looks cheap vs your view (buy); negative = rich (sell). "
               "**vs fit** = market − the smoothed implied path's fair value (view-free curve RV). "
               f"Point values: {bp_note}."
               + (" ER is priced as compounded-expected €STR + its per-contract spread."
                  if bank_key == "ECB" else ""))

    # ---- option-expiry landings ----------------------------------------------
    opt_prods = [p for p in prods if p.has_options]
    if opt_prods:
        st.markdown("#### Where futures land at each option expiry")
        st.caption("Each **listed monthly option and 1Y midcurve**, the future it exercises into, "
                   "and that future's fair price under **your scenario** on the option's expiry day "
                   "— with the decisions the option lives through vs the ones still open in the "
                   "underlying's window after it dies.")
        px_now = dict(zip(codes, prices))
        lrows, lkeys = [], []
        for p in opt_prods:
            for L in stirpaths.landings(p, bank, asof, views, r0, months,
                                        spread_bp=er_spread if p.ticker == "ERA Comdty" else 0.0,
                                        stub_rate=stub, include_midcurves=True):
                mkt = px_now.get(L.underlying.code)
                lrows.append({
                    "Option": f"{p.short} {'1Y-MC ' if L.series != 'Std' else ''}{L.opt_month}",
                    "Expiry": f"{L.expiry:%a %d %b %y}",
                    "Underlying": L.underlying.code,
                    "Mtgs before expiry": len(L.meetings_decided),
                    "Still open after": len(L.meetings_open),
                    "Landing px": L.fair,
                    "Mkt px now": mkt if mkt is not None else float("nan"),
                    "Diff (bp)": (L.fair - mkt) * 100.0 if mkt is not None else float("nan"),
                })
                lkeys.append((p, L))
        ldf = pd.DataFrame(lrows)
        brand.themed_dataframe(
            ldf, fmt={"Landing px": "{:.4f}".format, "Mkt px now": "{:.4f}".format,
                      "Diff (bp)": "{:+.1f}".format},
            colorers=[(["Diff (bp)"], _color_diff)],
            height=min(430, 45 + 35 * len(ldf)))
        st.caption("**Mtgs before expiry** = decisions the option lives through. **Still open "
                   "after** = decisions inside the underlying's window landing after the option "
                   "dies. Blank market price = the underlying sits beyond the priced strip.")

        # hand-off → Strategy Builder, with the outcome distribution
        h1, h2 = st.columns([3, 1.4])
        pick = h1.selectbox(
            "Build a client structure on…",
            list(range(len(lkeys))),
            format_func=lambda i: (f"{lkeys[i][0].short} "
                                   f"{'1Y-MC ' if lkeys[i][1].series != 'Std' else ''}"
                                   f"{lkeys[i][1].opt_month} option · "
                                   f"exp {lkeys[i][1].expiry:%d %b %y} → "
                                   f"{lkeys[i][1].underlying.code} @ {lkeys[i][1].fair:.4f}"),
            key=f"sp{bank_key}_ho_pick")
        p_pick, L_pick = lkeys[pick]
        dist = stirpaths.landing_distribution(
            p_pick, bank, L_pick.underlying, asof, views, r0, upto=L_pick.expiry,
            spread_bp=er_spread if p_pick.ticker == "ERA Comdty" else 0.0, stub_rate=stub)
        if len(dist) > 1:
            ddf = pd.DataFrame({"px": [d[0] for d in dist], "p": [d[1] * 100.0 for d in dist]})
            dbar = alt.Chart(ddf).mark_bar(size=7).encode(
                x=alt.X("px:Q", title="Underlying price at option expiry",
                        scale=alt.Scale(zero=False), axis=alt.Axis(format=".3f")),
                y=alt.Y("p:Q", title="Probability (%)"),
                color=alt.value(cc["series"]),
                tooltip=[alt.Tooltip("px:Q", format=".4f", title="Price"),
                         alt.Tooltip("p:Q", format=".1f", title="Prob %")])
            mean_rule = alt.Chart(pd.DataFrame({"px": [L_pick.fair]})).mark_rule(
                strokeWidth=2.5, strokeDash=[6, 4]).encode(
                x="px:Q", color=alt.value(cc["accent"]),
                tooltip=[alt.Tooltip("px:Q", format=".4f", title="Expected landing")])
            st.markdown("**Outcome distribution at this expiry — your odds, meeting by meeting**")
            brand.show_chart((dbar + mean_rule).properties(height=190))
            top2 = sorted(dist, key=lambda t: -t[1])[:2]
            st.caption("Meetings decided by the option's expiry realise as discrete outcomes; "
                       "later in-window meetings enter at expected value. Dashed gold = the "
                       "probability-weighted landing "
                       + " · ".join(f"mode {px:.3f} ({pr * 100:.0f}%)" for px, pr in top2)
                       + ". A landing in a valley between modes is the fly/condor tell.")
        else:
            st.caption("Your scenario is deterministic through this expiry — a single landing "
                       "point, no distribution.")
        if h2.button("🧰 Model in Strategy Builder", key=f"sp{bank_key}_ho_go",
                     use_container_width=True,
                     help="Opens the Strategy Builder with this product marked at the scenario "
                          "landing price, the leg month preselected, and your outcome "
                          "distribution attached for scenario-expected P&L."):
            st.session_state["osb_handoff"] = {
                "prod": p_pick.ticker, "spot": float(L_pick.fair),
                "days": max(1, (L_pick.expiry - asof).days),
                "expiry": L_pick.expiry.isoformat(),
                "nodes": [[float(px), float(pr)] for px, pr in dist],
                "note": (f"{p_pick.short} {'1Y-MC ' if L_pick.series != 'Std' else ''}"
                         f"{L_pick.opt_month} option (exp {L_pick.expiry:%d %b %Y}) on "
                         f"{L_pick.underlying.code} — scenario landing {L_pick.fair:.4f}")}
            _go("Strategy Builder")
            st.rerun()

    # ---- PDF export — the Meeting-Risk Map (all three banks) -----------------
    st.divider()
    bank_short = {"FED": "Fed", "ECB": "ECB", "BOE": "BoE"}[bank_key]
    if st.button(f"📈 Generate {bank_short} Meeting-Risk Map (PDF)", type="primary",
                 key=f"sp{bank_key}_pdfgen"):
        with st.spinner("Rendering the Meeting-Risk Map…"):
            try:
                ey_, em_ = stirpaths._add_months(asof.year, asof.month, months)
                hor_end_ = date(ey_, em_, 1)
                grows, growmap = [], {}
                for p in prods:
                    shown = 0
                    for c in strips[p.ticker]:
                        if c.end <= asof or c.start >= hor_end_:
                            continue
                        if not p.quarterly and shown >= 8:
                            continue
                        shown += 1
                        gleft = [m for m in stirpaths.meetings_in_window(bank, c) if m > asof]
                        glt = stirpaths.fut_last_trade(p, c)
                        row = {"label": f"{p.short} {c.label} · {len(gleft)}m",
                               "color": p.color,
                               "start": max(c.start, asof).isoformat(),
                               "end": min(c.end, hor_end_).isoformat(),
                               "mtgs": [m.isoformat() for m in gleft if m < hor_end_],
                               "fut": glt.isoformat() if asof <= glt < hor_end_ else None,
                               "px": px_of_now.get(c.code), "opts": [], "mcs": []}
                        grows.append(row)
                        growmap[(p.short, c.label)] = row
                    if p.has_options:
                        for rr in stirpaths.expiry_rows(p, asof, months):
                            if rr.kind != "Option":
                                continue
                            u = stirpaths.option_underlying(p, rr.year, rr.mon)
                            if (p.short, u.label) in growmap:
                                growmap[(p.short, u.label)]["opts"].append(rr.expiry.isoformat())
                        for rr in stirpaths.midcurve_expiries(p, asof, months):
                            u = stirpaths.option_underlying_mc(p, rr.year, rr.mon)
                            if (p.short, u.label) in growmap:
                                growmap[(p.short, u.label)]["mcs"].append(rr.expiry.isoformat())

                def _nan_dash(v, fmtstr="{:.4f}"):
                    return "—" if v is None or v != v else fmtstr.format(v)

                payload = {
                    "bank": bank_key, "bank_name": bank.name,
                    "meeting_name": bank.meeting_name, "rate_name": bank.rate_name,
                    "ccy": bank.ccy, "asof": asof.isoformat(),
                    "policy": float(policy), "band": band, "mode": stirpaths.MODE,
                    "haircut": float(haircut),
                    "products": [p.short for p in prods],
                    "scen_cum": float(np.sum(exp_moves)),
                    "gantt": {"hor_start": asof.isoformat(), "hor_end": hor_end_.isoformat(),
                              "meetings": [m.isoformat() for m in bank.meetings
                                           if asof <= m < hor_end_],
                              "rows": grows},
                    "odds": [{"decision": f"{m:%a %d %b %y}", "in_days": f"{(m - asof).days}d",
                              "implied": float(per_disp[lab]),
                              "odds": _stir_odds_str(per_disp[lab], bank.step_bp),
                              "you": float(vals[lab]), "emove": float(v.expected_bp),
                              "cum": float(cum_disp[i])}
                             for i, (m, lab, v) in enumerate(zip(ip.meetings, labels, views))],
                    "seg_dates": [x.isoformat() for x in seg_dates],
                    "mkt_seg": list(map(float, mkt_seg)),
                    "your_seg": list(map(float, your_seg)),
                    "contracts": [{"code": codes[i],
                                   "window": f"{owner[i].short} {contracts[i].label}",
                                   "market": f"{prices[i]:.4f}",
                                   "rate": f"{100 - prices[i]:.3f}",
                                   "your": f"{your_px[i]:.4f}",
                                   "diff_bp": f"{diff_bp[i]:+.1f}",
                                   "diff_ccy": f"{bank.ccy}{diff_bp[i] * owner[i].bp_value:,.0f}",
                                   "resid": (f"{_resid_of[codes[i]]:+.1f}"
                                             if codes[i] in _resid_of else "—"),
                                   "dir": 1 if diff_bp[i] > 0.05 else
                                          (-1 if diff_bp[i] < -0.05 else 0)}
                                  for i in range(len(codes))],
                    "landings": [{"option": lr["Option"],
                                  "expiry": lr["Expiry"].split(" ", 1)[-1],
                                  "under": lr["Underlying"],
                                  "before": lr["Mtgs before expiry"],
                                  "open": lr["Still open after"],
                                  "landing": f"{lr['Landing px']:.4f}",
                                  "mkt": _nan_dash(lr["Mkt px now"]),
                                  "diff": _nan_dash(lr["Diff (bp)"], "{:+.1f}"),
                                  "dir": (0 if lr["Diff (bp)"] != lr["Diff (bp)"] else
                                          1 if lr["Diff (bp)"] > 0.05 else
                                          -1 if lr["Diff (bp)"] < -0.05 else 0)}
                                 for lr in (lrows if opt_prods else [])][:14],
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "stirpath.json"
                    _out = Path(_t) / "Meeting_Risk_Map.pdf"
                    _in.write_text(json.dumps(payload), encoding="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "src" / "stirpathreport.py"),
                         str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state[f"sp{bank_key}_pdfb"] = _out.read_bytes()
                    else:
                        st.error("Report failed:\n\n"
                                 + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed:\n\n{e}")
    if st.session_state.get(f"sp{bank_key}_pdfb"):
        fname = f"{bank_short}_Meeting_Risk_Map.pdf"
        st.download_button(f"⬇️  Download {bank_short} Meeting-Risk Map",
                           data=st.session_state[f"sp{bank_key}_pdfb"],
                           file_name=fname, mime="application/pdf",
                           key=f"sp{bank_key}_pdfdl")
        email_report_ui(f"sp{bank_key}_email", "stirpath",
                        st.session_state[f"sp{bank_key}_pdfb"],
                        subject=f"BASIS — {bank.name} Meeting-Risk Map (STIR)",
                        attachment_name=fname)


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
        if IS_ADMIN and _dc1.button("📌 Set as default", key="vbt_set_def", use_container_width=True,
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


@st.cache_data(show_spinner=False, ttl=900)
def _tabt_overlays(tk: str, strats: tuple, hist: pd.DataFrame, vol: pd.DataFrame | None,
                   sessions: int = 0) -> dict:
    """Indicator overlays for the backtester's 'why it traded' chart — the SAME per-strategy chart
    payloads the TA hub gallery draws (each strategy's *_chart_data), computed on the backtest's own
    injected history so FICC fixed income overlays sit on the yield series the signals scored on,
    and equities on their yfinance closes. Cached: the heavy chart_data calls only rerun when the
    backtest itself changes, not on every widget touch."""
    from src.strategies import (support_resistance as _sr, flag_breakout as _fb,
                                breakout_retest as _br, momentum as _mom, fibonacci as _fbn,
                                elliott_wave as _ew, ichimoku as _ich, obv as _obv, mfi as _mfi)
    strset = set(strats)
    out = {"flag": None, "sr_levels": [], "fib_levels": [], "retest_level": None,
           "mom": None, "elliott": None, "ichimoku": None, "obv": None, "mfi": None}
    if "Flag Breakout" in strset:
        try:
            _fcd, _fi = _fb.flag_chart_data(tk, history=hist, volume=vol)
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
            # window = the full backtest span, so the cloud covers every trade (the hub's
            # default 170-session tail would leave older trades floating cloud-less)
            _, _ici = _ich.ichimoku_chart_data(tk, history=hist,
                                               window=(sessions + 30) if sessions else None)
            if _ici and _ici.get("cloud"):
                out["ichimoku"] = _ici
        except Exception:
            pass
    if "On-Balance Volume" in strset:
        try:
            _od, _oi = _obv.obv_chart_data(tk, history=hist, volume=vol)
            if _od is not None and not _od.empty:
                out["obv"] = _od[["date", "obv"]].dropna()
        except Exception:
            pass
    if "Money Flow Index" in strset:
        try:
            _md, _mi2 = _mfi.mfi_chart_data(tk, history=hist, volume=vol)
            if _md is not None and not _md.empty:
                out["mfi"] = _md[["date", "mfi"]].dropna()
        except Exception:
            pass
    return out


def render_ta_backtester(scope: str = "ficc") -> None:
    """TA Signal Backtester: pick a product, a score/conviction bar, and either one individual
    technical strategy or the whole confluence score, then walk history day by day taking the SAME
    signal the live TA pages would have shown on each date — so it can never quietly drift from what
    the dashboard actually says. `scope` ('ficc' | 'equities') runs the identical page over either
    book, same convention as `_ta_reports` (per-scope widget keys, universe, and history source)."""
    import altair as alt

    eq = scope == "equities"
    k = f"_{scope}"
    st.subheader("🎯  TA Signal Backtester")
    st.markdown("""
        <style>
          div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
              font-size: 1.05rem !important; line-height: 1.3 !important;
              white-space: normal !important; overflow-wrap: anywhere; }
        </style>""", unsafe_allow_html=True)
    st.caption(
        "Would following our technical-analysis signals have made or lost money? Walks the "
        "chosen product's history day by day, takes the SAME signal the live TA pages would have "
        "shown on that date (one strategy on its own, or the whole confluence score), and marks a "
        "position to market until your exit rule closes it. Not investment advice — a historical "
        "read of the signal, not a guarantee of future performance."
    )

    def _usd(v: float) -> str:
        return f"-${abs(v):,.0f}" if v < -0.5 else f"${abs(v):,.0f}"

    def _usd_md(v: float) -> str:
        # st.caption/st.markdown treat a $...$ PAIR as inline LaTeX — escape whenever two
        # dollar amounts might land in the same caption string (a single $ alone is safe).
        return _usd(v).replace("$", r"\$")

    if eq:
        from src import eqta
        _emeta = eqta.member_meta()
        tickers = sorted(_emeta.keys(), key=lambda t: str(_emeta.get(t, {}).get("name") or t))
        def _lab(t): return f"{_emeta.get(t, {}).get('name') or t}  ·  {t}"
    else:
        tickers = sorted(universe.TREND_UNIVERSE, key=lambda t: universe.name(t))
        def _lab(t): return f"{universe.name(t)}  ·  {t}"
    if not tickers:
        st.info("No products in the universe yet.")
        return

    _dflt = tabt_defaults(scope)                  # saved settings — seeds every control below

    ticker = st.selectbox("Product", tickers, format_func=_lab, key=f"tabt_tk{k}")

    # --- strategy picker, by axis — the SAME structure as the TA hub's confluence set, and the
    # same scoring rules: several methods ticked within one axis are de-duplicated (strongest
    # counts full, the next ½, then ⅓ …), agreement ACROSS axes counts in full.
    _avail = set(tabt.strategies_for_scope(scope))
    with st.expander("🎯 Strategies in the score — pick per axis (same rules as the TA hub)",
                     expanded=True):
        st.caption("Tick one method to backtest it on its own, or several to backtest their "
                   "combined score. Methods ticked within the **same axis** are de-duplicated "
                   "(strongest counts full, the next **½**, then **⅓** …) so a single dimension "
                   "can't vote twice; agreement **across** axes counts in full — exactly how the "
                   "TA hub and the report score.")
        # .get(): survives a Streamlit hot-reload where the cached src.specs module predates
        # the "strategies" key (imported modules aren't re-imported on rerun — only app.py is).
        _saved_bt = set(_dflt.get("strategies") or tascore.confluence_set(scope))
        picked = []
        _acols = st.columns(len(tascore.TA_AXES))
        for _col, (_ax, _methods) in zip(_acols, tascore.TA_AXES.items()):
            _opts = [m for m in _methods if m in _avail]
            if not _opts:
                continue
            _picked = _col.multiselect(
                _ax, options=_opts, default=[m for m in _opts if m in _saved_bt],
                key=f"tabt_ax{k}::{_ax}",
                help=f"The {_ax.lower()} axis — {len(_opts)} method(s) available. "
                     "Blank = this dimension sits out of the score.")
            picked.extend(_picked)
        if not picked:
            st.warning("Pick at least one strategy — nothing is in the score right now.")
        elif tascore.has_intra_axis_dup(picked):
            st.caption("↕️ De-dup active: you've ticked more than one method in an axis, so within "
                       "it the strongest counts full and the rest at ½, ⅓ … — the axis still only "
                       "votes once.")
    if not eq and "Mean Reversion" in picked and not tabt.pair_for(ticker):
        st.caption("Mean Reversion is pair-based and this product isn't part of a configured pair "
                   "(see the Universe page's rich-cheap monitor) — it will sit out of this "
                   "backtest's score.")

    # --- plain-English trigger reference: how every strategy decides to buy / sell ---
    _trigger_docs_expander(scope, picked=picked)

    t1, t2, t3 = st.columns(3)
    min_conviction = t1.number_input("Min conviction", 0, 100, int(_dflt["min_conviction"]), 5,
                                     key=f"tabt_mc{k}",
                                     help="Average strength of the flagging strategy/strategies "
                                          "(0-100) required to open a trade.")
    min_score = t2.number_input("Min |score|", 0, 600, int(_dflt["min_score"]), 10, key=f"tabt_ms{k}",
                                help="Signed conviction score required to open a trade — for a "
                                     "single strategy this equals its own conviction; for "
                                     "Confluence it also demands breadth (several strategies "
                                     "agreeing), same as the TA report's quality bar.")
    _DIR_OPTS = ["Both", "Long only", "Short only"]
    _DIR_IX = {"both": 0, "long": 1, "short": 2}
    direction_lbl = t3.radio("Direction", _DIR_OPTS, key=f"tabt_dir{k}", horizontal=True,
                             index=_DIR_IX.get(_dflt["direction"], 0))
    direction = {"Both": "both", "Long only": "long", "Short only": "short"}[direction_lbl]

    d1, d2 = st.columns(2)
    end = d2.date_input("To", value=date.today() - timedelta(days=5),
                        max_value=date.today() - timedelta(days=5), key=f"tabt_end{k}")
    start = d1.date_input("From", value=end - timedelta(days=182),
                          max_value=end - timedelta(days=30), key=f"tabt_start{k}")

    e1, e2 = st.columns([2, 1])
    _EXIT = {"reversal": "Signal reversal — exit when the signal flips direction",
             "score_drop": "Score drop — exit once it no longer clears the bar above",
             "hold_days": "Fixed holding period — exit N calendar days after entry"}
    _exit_keys = list(_EXIT)
    exit_lbl = e1.radio("Exit rule", _exit_keys, format_func=_EXIT.get, key=f"tabt_exit{k}",
                        horizontal=False,
                        index=_exit_keys.index(_dflt["exit_rule"]) if _dflt["exit_rule"] in _exit_keys else 0)
    hold_days = (e2.number_input("Holding days (N)", 1, 250, int(_dflt["hold_days"]), 1,
                                 key=f"tabt_hold{k}")
                if exit_lbl == "hold_days" else None)

    with st.expander("Stop/take overlay, position size & transaction costs"):
        st.caption("Stop/take are layered ON TOP of the exit rule above — whichever triggers "
                   "first closes the trade. Leave either at 0 to disable that side.")
        o1, o2, o3 = st.columns(3)
        stop_pct = o1.number_input("Stop-loss (%, 0 = off)", 0.0, 100.0, float(_dflt["stop_pct"]), 0.5,
                                   key=f"tabt_stop{k}")
        take_pct = o2.number_input("Take-profit (%, 0 = off)", 0.0, 200.0, float(_dflt["take_pct"]), 0.5,
                                   key=f"tabt_take{k}")
        _dflt_size = _dflt["size"] or (100.0 if eq else 1.0)
        size = o3.number_input("Shares" if eq else "Lots", 1.0, 1_000_000.0, _dflt_size, 1.0,
                               key=f"tabt_size{k}")
        st.caption("**Transaction costs** — charged per SIDE as incurred (entry day and exit "
                   "day), volbt convention: 0 = frictionless. Every P&L figure on this page is "
                   "then **net of costs**.")
        c1_, c2_ = st.columns(2)
        commission = c1_.number_input(
            f"Commission ($ per {'share' if eq else 'contract'} per side)", 0.0, 1000.0,
            float(_dflt.get("commission", 0.0)), 0.25, key=f"tabt_comm{k}",
            help="All-in broker/exchange/clearing charge per side, in USD.")
        slippage_pts = c2_.number_input(
            "Slippage (price points per side)", 0.0, 100.0,
            float(_dflt.get("slippage_pts", 0.0)), 0.005, format="%.3f", key=f"tabt_slip{k}",
            help=("Execution give-up per side in PRICE POINTS of the quote — converted to $ "
                  "through the same contract point-value (and FX) as the P&L itself, so e.g. "
                  "one tick of ES slippage (0.25 pt) costs 0.25 × $50 = $12.50 a side."
                  if not eq else
                  "Execution give-up per side in $ per share (half the bid/ask spread is the "
                  "usual working assumption)."))

    dc1, dc2 = st.columns([1, 3])
    if IS_ADMIN and dc1.button("📌 Set as default", key=f"tabt_set_def{k}", use_container_width=True,
                  help="Save the strategy picks, thresholds, direction, exit rule and overlay as "
                       "this book's startup default for the TA Backtester — they load on every launch."):
        save_tabt_defaults(scope, strategies=list(picked),
                           min_conviction=float(min_conviction), min_score=float(min_score),
                           direction=direction, exit_rule=exit_lbl,
                           hold_days=float(hold_days or _dflt["hold_days"]),
                           stop_pct=float(stop_pct), take_pct=float(take_pct), size=float(size),
                           commission=float(commission), slippage_pts=float(slippage_pts))
        st.toast("TA Backtester defaults saved for this book.", icon="📌")
    _dflt = tabt_defaults(scope)   # re-read — reflects a just-saved value on this same rerun
    _dflt_strats = _dflt.get("strategies") or list(tascore.confluence_set(scope))
    dc2.caption(f"📌 Default: **{len(_dflt_strats)}** strategies"
               + ("" if _dflt.get("strategies") else " (the book's confluence set)")
               + f", conviction **{_dflt['min_conviction']:g}**, score **{_dflt['min_score']:g}**, "
               f"**{_dflt['direction']}**, exit **{_dflt['exit_rule'].replace('_', ' ')}**"
               + (f" ({_dflt['hold_days']:g}d)" if _dflt["exit_rule"] == "hold_days" else "")
               + (f", stop/take {_dflt['stop_pct']:g}%/{_dflt['take_pct']:g}%"
                  if _dflt["stop_pct"] or _dflt["take_pct"] else "")
               + (f", costs \\${_dflt.get('commission', 0):g} + "
                  f"{_dflt.get('slippage_pts', 0):g}pt/side"
                  if _dflt.get("commission") or _dflt.get("slippage_pts") else ""))

    st.caption("A wide date range (or **Compare all strategies**, which backtests every strategy "
              "at once) can take a while — each day re-checks the live signal logic, not a shortcut "
              "approximation. Narrow the date range for a quicker look.")

    kwargs = dict(min_conviction=float(min_conviction), min_score=float(min_score),
                 direction=direction, start=start, end=end, exit_rule=exit_lbl,
                 hold_days=int(hold_days) if hold_days else None,
                 stop_pct=float(stop_pct) or None, take_pct=float(take_pct) or None,
                 size=float(size), commission=float(commission),
                 slippage_pts=float(slippage_pts))

    b1, b2 = st.columns(2)
    if b1.button("▶  Run backtest", type="primary", key=f"tabt_run{k}", disabled=not picked):
        try:
            with st.spinner("Walking the signal day by day…"):
                st.session_state[f"tabt_res{k}"] = tabt.run_backtest(scope, ticker, list(picked), **kwargs)
            st.session_state.pop(f"tabt_cmp{k}", None)
        except ValueError as e:
            st.session_state.pop(f"tabt_res{k}", None)
            st.error(str(e))
    if b2.button("📊  Compare all strategies", key=f"tabt_cmp_btn{k}"):
        try:
            with st.spinner("Backtesting every strategy on this product — this can take a few "
                            "minutes over a wide date range…"):
                st.session_state[f"tabt_cmp{k}"] = tabt.compare_strategies(scope, ticker, **kwargs)
            st.session_state.pop(f"tabt_res{k}", None)
        except ValueError as e:
            st.session_state.pop(f"tabt_cmp{k}", None)
            st.error(str(e))

    cmp_df = st.session_state.get(f"tabt_cmp{k}")
    if cmp_df is not None:
        for w in cmp_df.attrs.get("warnings", []):
            st.warning(w)
        _cmp_costs = float(cmp_df["costs"].sum()) if "costs" in cmp_df.columns else 0.0
        st.markdown(f"#### Every strategy vs Confluence — {_lab(ticker)}, "
                    f"{start:%d %b %Y} → {end:%d %b %Y}"
                    + (" (net of costs)" if _cmp_costs else ""))
        # tag each row with its axis ("sector" of technical analysis) — short forms keep the
        # y labels readable; Confluence spans them all
        _AX_SHORT = {"Trend": "Trend", "Momentum / Oscillators": "Momentum", "Volume": "Volume",
                     "Support & Resistance": "S&R", "Patterns & Breakouts": "Patterns"}
        cmp_df = cmp_df.assign(
            axis=[("all axes" if s2 == "Confluence"
                   else _AX_SHORT.get(tascore.axis_of(s2), tascore.axis_of(s2)))
                  for s2 in cmp_df["strategy"]])
        cmp_df = cmp_df.assign(
            label=[f"{s2}  ·  {a2}" for s2, a2 in zip(cmp_df["strategy"], cmp_df["axis"])])
        bars = alt.Chart(cmp_df).mark_bar().encode(
            x=alt.X("total_pnl:Q", title="Total P&L ($)" + (" net" if _cmp_costs else "")),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition("datum.total_pnl >= 0", alt.value("#46C58A"), alt.value("#EC6A57")),
            tooltip=[alt.Tooltip("strategy:N"), alt.Tooltip("axis:N", title="Axis"),
                    alt.Tooltip("total_pnl:Q", format="+,.0f"),
                    alt.Tooltip("n_trades:Q", title="trades"),
                    alt.Tooltip("win_rate:Q", format=".0f", title="win rate %")])
        brand.show_chart(bars.properties(height=32 * max(len(cmp_df), 4) + 40))
        view = cmp_df.assign(
            **{"Axis": cmp_df["axis"],
               "Total P&L": cmp_df["total_pnl"].map(_usd),
               "Trades": cmp_df["n_trades"],
               "Win rate": cmp_df["win_rate"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}%"),
               "Avg win": cmp_df["avg_win"].map(lambda v: "—" if pd.isna(v) else _usd(v)),
               "Avg loss": cmp_df["avg_loss"].map(lambda v: "—" if pd.isna(v) else _usd(v)),
               "Profit factor": cmp_df["profit_factor"].map(
                   lambda v: "∞" if v == np.inf else ("—" if pd.isna(v) else f"{v:.2f}")),
               "Max drawdown": cmp_df["max_drawdown"].map(_usd),
               "Costs": (cmp_df["costs"] if "costs" in cmp_df.columns
                         else pd.Series(0.0, index=cmp_df.index)).map(_usd),
               "Avg hold (days)": cmp_df["avg_holding_days"].map(
                   lambda v: "—" if pd.isna(v) else f"{v:.0f}")}
        )[["strategy", "Axis", "Total P&L", "Trades", "Win rate", "Avg win", "Avg loss",
           "Profit factor", "Max drawdown"] + (["Costs"] if _cmp_costs else [])
          + ["Avg hold (days)"]].rename(columns={"strategy": "Strategy"})
        st.dataframe(view, hide_index=True, use_container_width=True)
        return

    res = st.session_state.get(f"tabt_res{k}")
    if res is None:
        return
    for w in res.warnings:
        st.warning(w)
    s = res.summary
    if s["n_trades"] == 0:
        st.info("No trades cleared the bar over this window — try a lower conviction/score "
               "threshold, a wider date range, or a different strategy.")
        return
    # render from the RUN's own parameters (stored in the summary), not the widgets — the desk
    # may have retuned the controls since the run and the header must describe what actually ran
    _rs = s.get("strategies") or list(picked)
    _rtk = s.get("ticker", ticker)
    _rstart, _rend = s.get("start", start), s.get("end", end)
    _strat_note = (_rs[0] if len(_rs) == 1
                  else f"{len(_rs)}-strategy score ({', '.join(_rs)})" if _rs else "—")
    _rdir = {"both": "Both", "long": "Long only", "short": "Short only"}.get(
        s.get("direction", direction), "Both")
    _rstop, _rtake = s.get("stop_pct", 0.0), s.get("take_pct", 0.0)
    _rcomm, _rslip = s.get("commission", 0.0), s.get("slippage_pts", 0.0)
    _has_costs = bool(_rcomm or _rslip)
    st.markdown(f"#### {_lab(_rtk)} — {_strat_note} — {_rstart:%d %b %Y} → {_rend:%d %b %Y}")
    st.caption(f"Min conviction {s.get('min_conviction', min_conviction):g} · "
              f"Min |score| {s.get('min_score', min_score):g} · {_rdir} · "
              f"exit: {_EXIT.get(s.get('exit_rule', exit_lbl), s.get('exit_rule', exit_lbl))}"
              + (f" · stop {_rstop:g}% / take {_rtake:g}%" if _rstop or _rtake else "")
              + (f" · costs \\${_rcomm:g} + {_rslip:g}pt per side" if _has_costs
                 else " · frictionless (no costs applied)"))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total P&L" + (" (net)" if _has_costs else ""), _usd(s["total_pnl"]))
    m1.caption(f"max drawdown {_usd_md(s['max_drawdown'])}"
              + (f" · net of {_usd_md(s.get('costs', 0.0))} costs" if _has_costs else ""))
    m2.metric("Trades", f"{s['n_trades']}")
    m2.caption(f"avg hold {s['avg_holding_days']:.0f}d")
    m3.metric("Win rate", f"{s['win_rate']:.0f}%")
    m3.caption(f"avg win {_usd_md(s['avg_win'])} · avg loss {_usd_md(s['avg_loss'])}")
    m4.metric("Profit factor", "∞" if s["profit_factor"] == np.inf
              else ("—" if pd.isna(s["profit_factor"]) else f"{s['profit_factor']:.2f}"))
    m4.caption("gross win ÷ gross loss")
    m5.metric("Size", f"{size:g} {'shares' if eq else 'lots'}")
    m5.caption("USD P&L" if eq else "USD P&L (FX-converted where needed)")

    _cc = brand.chart_colors()
    dd = res.daily.reset_index()
    # a result produced by an OLDER engine (before per-day score/signal recording) has no data
    # for these panels — say so rather than drawing empty charts, and backfill so nothing breaks.
    # Distinguish "old result, current engine" (a re-run fixes it) from "the ENGINE ITSELF is
    # stale in memory" (Streamlit hot-reloads app.py but caches src/ modules until the process
    # restarts — a re-run on a stale process just reproduces the same gap).
    if any(_c not in dd.columns for _c in ("score", "conviction", "signal_level")):
        import dataclasses as _dc
        _engine_current = "series" in {f.name for f in _dc.fields(tabt.Result)}
        if _engine_current:
            st.info("⚠️ This result predates the latest engine — the charts below will be partly "
                    "empty. Hit **▶ Run backtest** again to regenerate it in full.")
        else:
            st.warning("⚠️ BASIS is running an **older engine still cached in memory** — the app "
                      "picks up page changes instantly, but engine (src/) changes only load on a "
                      "full restart. **Close the Terminal and relaunch it**, then run the backtest "
                      "again; until then these charts will stay partly empty no matter how many "
                      "times you re-run.")
        for _c in ("score", "conviction", "signal_level"):
            if _c not in dd.columns:
                dd[_c] = np.nan

    # position segments + window bounds, shared by EVERY panel below: all charts pin their
    # x-axis to the same [start, end] domain and carry the same long/short bands, so the
    # shading lines up column-for-column from the P&L curve to the score bars. Every panel's
    # y-axis also reserves the SAME fixed gutter (minExtent=maxExtent) — otherwise "−30,000"
    # P&L labels vs "95.5" price labels give each plot area a different left edge and the
    # bands drift out of column-alignment even on identical date domains.
    _win_start = pd.Timestamp(dd["date"].iloc[0])
    _win_end = pd.Timestamp(dd["date"].iloc[-1])
    _win_index = pd.DatetimeIndex(dd["date"])
    _xsc = alt.Scale(domain=[str(_win_start.date()), str(_win_end.date())])
    _YEXT = {"minExtent": 84, "maxExtent": 84}
    _segs, _cur, _t0 = [], 0, None
    for _r in dd.itertuples():
        _p = int(_r.position)
        if _p != _cur:
            if _cur != 0:
                _segs.append({"start": _t0, "end": _r.date, "side": "Long" if _cur > 0 else "Short"})
            _cur, _t0 = _p, _r.date
    if _cur != 0 and len(dd):
        _segs.append({"start": _t0, "end": dd["date"].iloc[-1], "side": "Long" if _cur > 0 else "Short"})

    def _band_layer():
        return alt.Chart(pd.DataFrame(_segs)).mark_rect(opacity=0.10).encode(
            x="start:T", x2="end:T",
            color=alt.Color("side:N", scale=alt.Scale(domain=["Long", "Short"],
                            range=[_cc["long"], _cc["short"]]), legend=None))

    # green while the running total is above water, red while it's under — not one colour for
    # the whole run, so a drawdown through zero reads as losing money at that point in time
    dd["gain"] = dd["cum_pnl"].clip(lower=0.0)
    dd["loss"] = dd["cum_pnl"].clip(upper=0.0)
    cv_layers = [_band_layer()] if _segs else []
    cv_layers.append(alt.Chart(dd).mark_area(opacity=0.25, color=_cc["long"],
                                             interpolate="step-after").encode(
        x=alt.X("date:T", title=None, scale=_xsc),
        y=alt.Y("gain:Q", title="cumulative P&L ($)", axis=alt.Axis(**_YEXT))))
    cv_layers.append(alt.Chart(dd).mark_area(opacity=0.25, color=_cc["short"],
                                             interpolate="step-after").encode(
        x="date:T", y="loss:Q"))
    cv_layers.append(alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=_cc["muted"], opacity=0.6).encode(y="y:Q"))
    cv_layers.append(alt.Chart(dd).mark_line(color=_cc["ink"], strokeWidth=1.8,
                                             interpolate="step-after").encode(
        x="date:T", y="cum_pnl:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("cum_pnl:Q", format="+,.0f"),
                alt.Tooltip("position:Q", title="position")]))
    brand.show_chart(alt.layer(*cv_layers).properties(
        height=280, title="Cumulative P&L — fill green above water / red below; "
                          "bands = position on (green long / red short)"))

    # ---- why it traded: the series the signals scored on, the picked strategies' own
    #      indicators drawn over it (cloud / MAs / bands / levels — same as the TA hub gallery),
    #      and every entry/exit marked -------------------------------------------------------
    _has_lvl = "signal_level" in dd.columns and dd["signal_level"].notna().any()
    _ycol = "signal_level" if _has_lvl else "price"
    _fi_chart = bool(s.get("fi")) and _has_lvl
    _ytitle = "Yield (%)" if _fi_chart else "Price"
    st.markdown("##### Why it traded — the picked strategies' own indicators, with every entry/exit")

    # full-depth signal series (incl. warm-up buffer) so overlays have their lookback
    _pf = res.series if getattr(res, "series", None) is not None and len(res.series) \
        else res.daily[_ycol].dropna()
    _vol_df = (pd.DataFrame({_rtk: res.volume})
               if getattr(res, "volume", None) is not None else None)
    _ov = _tabt_overlays(_rtk, tuple(sorted(_rs)), pd.DataFrame({_rtk: _pf}), _vol_df,
                         sessions=len(dd))

    layers = []
    if _segs:
        layers.append(_band_layer())

    # Ichimoku Kumo (cloud) + Tenkan/Kijun — behind the price like the hub/report; green where
    # span-A ≥ span-B, red below. Clipped to the backtest window on BOTH sides: the cloud's
    # 26-session forward projection would otherwise stretch this chart's x-axis past the other
    # panels' and knock every band out of column-alignment with them.
    _ich = _ov.get("ichimoku")
    if _ich and _ich.get("cloud"):
        _cl = pd.DataFrame([c for c in _ich["cloud"]
                            if _win_start <= c["date"] <= _win_end]).dropna(subset=["a", "b"])
        if not _cl.empty:
            _cl["bull"] = _cl["a"] >= _cl["b"]
            for _fl, _col in ((True, _cc["long"]), (False, _cc["short"])):
                _seg = _cl.copy()
                _seg.loc[_cl["bull"] != _fl, ["a", "b"]] = None
                layers.append(alt.Chart(_seg).mark_area(opacity=0.32).encode(
                    x="date:T", y=alt.Y("a:Q", scale=alt.Scale(zero=False)), y2="b:Q",
                    color=alt.value(_col)))
            for _k2, _c2 in (("tenkan", "#26A69A"), ("kijun", "#EC407A")):
                _ln = pd.DataFrame([r2 for r2 in (_ich.get(_k2) or []) if r2["date"] >= _win_start]
                                   ).dropna(subset=["val"])
                if not _ln.empty:
                    layers.append(alt.Chart(_ln).mark_line(
                        color=_c2, strokeWidth=1.2, opacity=0.85).encode(
                        x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False))))

    # flag channel (fill + edges + dashed breakout + pole), in its direction colour
    if _ov.get("flag"):
        _fch, _fi2 = _ov["flag"]
        _fcol = _cc["long"] if _fi2["sign"] > 0 else _cc["short"]
        _fbase = alt.Chart(_fch).encode(x="date:T")
        layers += [
            _fbase.mark_area(opacity=0.22, color=_fcol).encode(y="lower:Q", y2="upper:Q"),
            _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="upper:Q"),
            _fbase.mark_line(color=_fcol, strokeWidth=1.6).encode(y="lower:Q"),
            _fbase.mark_line(color=_fcol, strokeDash=[6, 3], strokeWidth=2.4).encode(y="breakout:Q"),
        ]

    # MA / Bollinger line overlays — computed on the FULL buffered history (correct lookback),
    # shown over the backtest window; same widths as the hub gallery
    _mls = {}
    if "Bollinger Squeeze" in _rs:
        _mid, _sd = _pf.rolling(20).mean(), _pf.rolling(20).std()
        _mls["BB upper"], _mls["BB mid"], _mls["BB lower"] = _mid + 2 * _sd, _mid, _mid - 2 * _sd
    for _strat, _ws in (("MA Crossover", (50, 200)), ("MA Swing", (20, 50)), ("Trend", (20, 100))):
        if _strat in _rs:
            for _w in _ws:
                _mls.setdefault(f"MA{_w}", _pf.rolling(_w).mean())
    if _mls:
        _ldf = pd.DataFrame({"date": _win_index})
        for _lab, _ser in _mls.items():
            _ldf[_lab] = _ser.reindex(_win_index).to_numpy(dtype=float)
        _long = _ldf.melt("date", var_name="Indicator", value_name="val").dropna(subset=["val"])
        layers.append(alt.Chart(_long).mark_line(strokeWidth=1.8).encode(
            x="date:T", y=alt.Y("val:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Indicator:N", legend=alt.Legend(orient="top", title=None,
                                                             labelFontSize=11)),
            tooltip=[alt.Tooltip("Indicator:N"), alt.Tooltip("val:Q", format=",.2f")]))

    # horizontal levels: support/resistance, Fibonacci, broken-level retest
    for _lv in _ov.get("sr_levels", []):
        _lc = _cc["long"] if _lv["kind"] == "support" else _cc["short"]
        if np.isfinite(_lv["price"]):
            layers.append(alt.Chart(pd.DataFrame({"y": [_lv["price"]]})).mark_rule(
                color=_lc, strokeDash=[5, 3], opacity=0.85, strokeWidth=1.8).encode(y="y:Q"))
    for _L in _ov.get("fib_levels", []):
        if np.isfinite(_L["price"]):
            layers.append(alt.Chart(pd.DataFrame({"y": [_L["price"]]})).mark_rule(
                color=_cc["accent"], strokeDash=[5, 3], opacity=0.85, strokeWidth=1.8).encode(y="y:Q"))
    if _ov.get("retest_level") is not None and np.isfinite(_ov["retest_level"]):
        layers.append(alt.Chart(pd.DataFrame({"y": [_ov["retest_level"]]})).mark_rule(
            color=_cc["accent"], strokeDash=[5, 3], opacity=0.85, strokeWidth=1.8).encode(y="y:Q"))

    layers.append(alt.Chart(dd).mark_line(color=_cc["ink"], strokeWidth=2.2).encode(
        x=alt.X("date:T", title=None, scale=_xsc),
        y=alt.Y(f"{_ycol}:Q", title=_ytitle, scale=alt.Scale(zero=False),
                axis=alt.Axis(**_YEXT)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{_ycol}:Q", title=_ytitle, format=",.3f"),
                alt.Tooltip("score:Q", title="Score", format="+.1f"),
                alt.Tooltip("conviction:Q", title="Conviction", format=".0f")]))

    # Elliott wave count (purple 0-5 pivots), on top of the price like the hub
    if _ov.get("elliott"):
        _piv = pd.DataFrame([p for p in _ov["elliott"] if p["date"] >= _win_start])
        if len(_piv) >= 2:
            layers.append(alt.Chart(_piv).mark_line(
                color="#9575CD", strokeWidth=1.8, opacity=0.9,
                point=alt.OverlayMarkDef(color="#9575CD", size=42)).encode(
                x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("label:N", title="Wave"),
                        alt.Tooltip("price:Q", title=_ytitle, format=",.2f")]))
            layers.append(alt.Chart(_piv).mark_text(
                dy=-12, fontSize=12, fontWeight="bold", color="#B39DDB").encode(
                x="date:T", y="price:Q", text="label:N"))

    _tr = res.trades.copy()
    _lvl = res.daily[_ycol]
    _tr["entry_lvl"] = [float(_lvl.get(pd.Timestamp(x), np.nan)) for x in _tr["entry_date"]]
    _tr["exit_lvl"] = [float(_lvl.get(pd.Timestamp(x), np.nan)) for x in _tr["exit_date"]]
    layers.append(alt.Chart(_tr).mark_point(size=140, filled=True,
                                            stroke="white", strokeWidth=0.6).encode(
        x="entry_date:T", y="entry_lvl:Q",
        shape=alt.Shape("direction:N", scale=alt.Scale(domain=["Long", "Short"],
                        range=["triangle-up", "triangle-down"]), legend=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Long", "Short"],
                        range=[_cc["long"], _cc["short"]]),
                        legend=alt.Legend(title="Entry", orient="top")),
        tooltip=[alt.Tooltip("entry_date:T", title="Entry"), alt.Tooltip("direction:N", title="Dir"),
                alt.Tooltip("entry_price:Q", title="Entry px", format=",.3f"),
                alt.Tooltip("entry_conviction:Q", title="Conviction", format=".0f"),
                alt.Tooltip("entry_score:Q", title="Score", format="+.0f")]))
    layers.append(alt.Chart(_tr).mark_point(size=120, shape="cross", filled=True,
                                            color=_cc["accent"]).encode(
        x="exit_date:T", y="exit_lvl:Q",
        tooltip=[alt.Tooltip("exit_date:T", title="Exit"), alt.Tooltip("exit_reason:N", title="Reason"),
                alt.Tooltip("exit_price:Q", title="Exit px", format=",.3f"),
                alt.Tooltip("pnl:Q", title="P&L", format="+,.0f")]))

    # PATTERN LEVELS AS OF EACH ENTRY — the read that actually pulled the trigger. The
    # full-width dashed rules above are TODAY's levels (last-180-session swing etc.), which
    # say nothing about a trade taken a year ago; here each trade gets the levels recomputed
    # from history up to ITS entry day. Drawn in CYAN (a colour nothing else on this chart
    # uses — Ben's call: colour, not line style, separates then-vs-now) and only over ±5
    # sessions around the entry, so with many trades each cluster stays pinned to its own
    # marker instead of span-length segments overlapping each other.
    _ENTRY_LVL_COLOR = "#4DD0E1"
    _PAT = {"Fibonacci Retracement", "Support & Resistance", "Breakout & Retest"} & set(_rs)
    if _PAT and len(_tr) and _pf is not None and len(_pf):
        from src.strategies import (fibonacci as _fbn2, support_resistance as _sr2,
                                    breakout_retest as _br2)
        _seg_rows = []
        for _t2 in _tr.itertuples():
            _h2 = pd.DataFrame({_rtk: _pf.loc[:pd.Timestamp(_t2.entry_date)]})
            if len(_h2) < 60:
                continue
            _ei = _win_index.searchsorted(pd.Timestamp(_t2.entry_date))
            _x0 = _win_index[max(0, _ei - 5)]
            _x1 = _win_index[min(len(_win_index) - 1, _ei + 5)]
            try:
                _when = pd.Timestamp(_t2.entry_date).strftime("%d %b %y")
                if "Fibonacci Retracement" in _PAT:
                    _, _fi4 = _fbn2.fib_chart_data(_rtk, history=_h2)
                    for _L4 in ((_fi4 or {}).get("levels") or []):
                        if _L4.get("key") and np.isfinite(_L4["price"]):
                            _seg_rows.append({"start": _x0, "end": _x1, "y": _L4["price"],
                                              "what": f"Fib {_L4['ratio']:.3f} at {_when} entry"})
                if "Support & Resistance" in _PAT:
                    _, _si4 = _sr2.sr_chart_data(_rtk, history=_h2)
                    for _L4 in ((_si4 or {}).get("levels") or []):
                        if np.isfinite(_L4["price"]):
                            _seg_rows.append({"start": _x0, "end": _x1, "y": _L4["price"],
                                              "what": f"{_L4['kind']} at {_when} entry"})
                if "Breakout & Retest" in _PAT:
                    _, _bi4 = _br2.retest_chart_data(_rtk, history=_h2)
                    _lv4 = (_bi4 or {}).get("level")
                    if _lv4 is not None and np.isfinite(_lv4):
                        _seg_rows.append({"start": _x0, "end": _x1, "y": _lv4,
                                          "what": f"retest level at {_when} entry"})
            except Exception:
                pass
        if _seg_rows:
            layers.append(alt.Chart(pd.DataFrame(_seg_rows)).mark_rule(
                color=_ENTRY_LVL_COLOR, strokeWidth=2.4, opacity=0.95).encode(
                x="start:T", x2="end:T", y="y:Q",
                tooltip=[alt.Tooltip("what:N", title=""),
                        alt.Tooltip("y:Q", title="Level", format=",.2f")]))

    brand.show_chart(alt.layer(*layers).resolve_scale(y="shared").properties(
        height=340, title=f"{_ytitle}, the picked strategies' indicators & every trade"))
    # caption describes ONLY the overlays this run's picked strategies actually draw
    _OV_FULL = {"Ichimoku Cloud": "the **Ichimoku cloud + Tenkan/Kijun**",
                "MA Crossover": "the **50/200 moving averages**",
                "MA Swing": "the **20/50 moving averages**",
                "Trend": "the **20/100 moving averages**",
                "Bollinger Squeeze": "the **Bollinger bands**"}
    _OV_EOW = {"Flag Breakout": "the **flag channel**",
               "Elliott Wave": "the **Elliott count**",
               "Support & Resistance": "the **support/resistance levels**",
               "Fibonacci Retracement": "the **Fibonacci levels**",
               "Breakout & Retest": "the **retest level**"}
    _full_ovs = [_OV_FULL[s2] for s2 in _rs if s2 in _OV_FULL]
    _eow_ovs = [_OV_EOW[s2] for s2 in _rs if s2 in _OV_EOW]
    _cap = ("Every decision was made from your picked strategies **recomputed as of that "
            "historical day** — nothing is read off this drawing.")
    if _full_ovs:
        _cap += f" Drawn over the full window: {', '.join(_full_ovs)}."
    if _eow_ovs:
        _cap += (f" {', '.join(_eow_ovs).capitalize()} are drawn twice: **gold/green/red dashed "
                 "full-width** = today's read (context for now), **cyan segments** = the levels "
                 "as they stood **at each trade's entry**, pinned ±5 sessions around that entry "
                 "marker — the read that actually pulled the trigger (hover a segment for which "
                 "level and which entry). Flag channel and Elliott count stay end-of-window "
                 "snapshots.")
    _cap += (" Shaded bands = days a position was on (green long / red short). ▲ / ▼ = entries, "
             "✕ = exits — hover any marker for conviction, score, reason and P&L.")
    if _fi_chart:
        _cap += (" Fixed income charts the **yield** the signals score on, so a **Long** (buy "
                 "the future) entry sits on a **falling-yield** signal — the usual FI mirror.")
    st.caption(_cap)

    # oscillator / volume sub-panels, when those strategies are in the score (hub convention)
    _osc, _guides = [], []
    if _ov.get("mom") is not None:
        _osc.append(("rsi", _ov["mom"][_ov["mom"]["date"] >= _win_start], "#7E57C2", "RSI"))
        _guides += [(70, _cc["short"]), (30, _cc["long"])]
    if _ov.get("mfi") is not None:
        _osc.append(("mfi", _ov["mfi"][_ov["mfi"]["date"] >= _win_start], "#00897B", "MFI"))
        _guides += [(80, _cc["short"]), (20, _cc["long"])]
    if _osc:
        _olays = [alt.Chart(_df).mark_line(color=_c, strokeWidth=2).encode(
            x=alt.X("date:T", title=None, scale=_xsc, axis=alt.Axis(labelFontSize=11)),
            y=alt.Y(f"{_col_name}:Q", title="RSI / MFI", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[0, 20, 30, 50, 70, 80, 100], labelFontSize=11,
                                  **_YEXT)))
            for _col_name, _df, _c, _ in _osc if not _df.empty]
        _olays += [alt.Chart(pd.DataFrame({"y": [_y]})).mark_rule(
            color=_c, strokeDash=[4, 3]).encode(y="y:Q") for _y, _c in _guides]
        if _olays:
            brand.show_chart(alt.layer(*_olays).resolve_scale(y="shared").properties(
                height=130, title=" / ".join(t for _, _, _, t in _osc) + " (14)"))
    if _ov.get("obv") is not None:
        _od = _ov["obv"][_ov["obv"]["date"] >= _win_start]
        if not _od.empty:
            brand.show_chart(alt.Chart(_od).mark_line(
                color="#26A69A", strokeWidth=1.8).encode(
                x=alt.X("date:T", title=None, scale=_xsc, axis=alt.Axis(labelFontSize=11)),
                y=alt.Y("obv:Q", title="OBV", scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=10, **_YEXT))).properties(
                height=110, title="On-Balance Volume"))

    # the daily score behind the trades — tucked away: the price chart above already tells the
    # story visually, this is the numeric trigger for anyone who wants to audit it
    with st.expander("🔬 Under the hood — the daily score that pulled the trigger", expanded=False):
        st.caption("Each bar is **one day's combined read** from your picked strategies, on the "
                  "same signed scale as the TA hub: bar **up** = the set read long that day, bar "
                  "**down** = short; taller = stronger and broader agreement. The dashed lines "
                  "are your **Min |score|** entry bar — a trade opens the day a bar first pokes "
                  "past them (with the conviction floor met) on your chosen side, and a reversal "
                  "exit fires the day the bars flip side. **No bar = nothing flagged that day** — "
                  "event-driven methods (Ichimoku, flag, retest …) only speak on their event days, "
                  "which is why a position can sit unchanged for weeks between bars.")
        _mbar = float(s.get("min_score", min_score) or 0.0)
        _sc_layers = [_band_layer()] if _segs else []
        _sc_layers.append(alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
            color=_cc["muted"], opacity=0.6).encode(y="y:Q"))
        if _mbar:
            for _b in (_mbar, -_mbar):
                _sc_layers.append(alt.Chart(pd.DataFrame({"y": [_b]})).mark_rule(
                    color=_cc["accent"], strokeDash=[5, 3]).encode(y="y:Q"))
        _sc_layers.append(alt.Chart(dd.dropna(subset=["score"])).mark_bar(size=3).encode(
            x=alt.X("date:T", title=None, scale=_xsc),
            y=alt.Y("score:Q", title="daily score (signed)", axis=alt.Axis(**_YEXT)),
            color=alt.condition("datum.score >= 0", alt.value(_cc["long"]),
                                alt.value(_cc["short"])),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("score:Q", format="+.1f"),
                    alt.Tooltip("conviction:Q", title="Conviction", format=".0f")]))
        brand.show_chart(alt.layer(*_sc_layers).properties(
            height=190, title="Daily signed score (dashed = your |score| entry bar)"))

    st.markdown("##### Trade blotter" + (" — P&L net of costs" if _has_costs else ""))
    _tcost = res.trades["cost"] if "cost" in res.trades.columns else pd.Series(0.0, index=res.trades.index)
    tv = res.trades.assign(
        **{"Entry": res.trades["entry_date"].astype(str), "Exit": res.trades["exit_date"].astype(str),
           "Dir": res.trades["direction"],
           "Entry px": res.trades["entry_price"].map(lambda v: f"{v:,.3f}"),
           "Exit px": res.trades["exit_price"].map(lambda v: f"{v:,.3f}"),
           "Reason": res.trades["exit_reason"],
           "Conviction": res.trades["entry_conviction"].map(lambda v: f"{v:.0f}"),
           "Hold (d)": res.trades["holding_days"],
           "Cost": _tcost.map(_usd),
           "P&L": res.trades["pnl"].map(_usd),
           "P&L %": res.trades["pnl_pct"].map(lambda v: f"{v:+.1f}%")}
    )[["Entry", "Exit", "Dir", "Entry px", "Exit px", "Reason", "Conviction", "Hold (d)"]
      + (["Cost"] if _has_costs else []) + ["P&L", "P&L %"]]
    st.dataframe(tv, hide_index=True, use_container_width=True, height=min(400, 40 + 35 * len(tv)))


@st.cache_data(show_spinner="Loading the signal ledger …", max_entries=2)
def _ledger_cached(scope: str, stamp: float) -> pd.DataFrame:
    from src import sigledger
    return sigledger.load(scope)


@st.cache_data(show_spinner="Loading the axis votes …", max_entries=2)
def _ledger_votes_cached(scope: str, stamp: float) -> pd.DataFrame:
    from src import sigledger
    return sigledger.load_votes(scope)


def _ledger_stamp(scope: str) -> float:
    """Cache key: the outcome files' newest mtime — a morning rebuild busts the cache."""
    from src import sigledger
    paths = (sigledger._eq_year_paths() if scope == "equities"
             else [sigledger.OUTCOMES_FILE])
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def render_signal_ledger(scope: str = "ficc") -> None:
    """Signal Ledger: every TA signal the hub would have flagged, tracked forward — hit
    rates by strategy / product + the confluence composite. FICC = its own page under the
    Technical Analysis module's tab row; equities = embedded at the foot of the Equities
    TA page (fully separate books per Ben). Reads the precomputed outcomes parquets (src/sigledger.py, rebuilt by the
    daily pulls); aggregation is pandas over cached frames — the 10.7M-row equities book
    loads once per rebuild, not once per widget click."""
    import altair as alt
    from src import sigledger

    st.subheader("📒  Signal Ledger" + ("  —  Equities" if scope == "equities" else ""))

    st.caption(
        "Every technical signal the TA hub would have flagged, tracked forward: did the market "
        "go the signal's way 5, 10 and 21 sessions later? Judged in **signal space** — "
        + ("split- and dividend-adjusted closes for every name (the same series the Equities "
           "TA page scores). "
           if scope == "equities" else
           "yields for fixed income (a Long there means rising yields = short the future, the "
           "FI pages' convention), pair spreads for Mean Reversion, prices elsewhere. ")
        + "Each move is also expressed in the product's own trailing-volatility units (σ) so "
          "moves compare fairly across names. A historical read of the signals' accuracy, not "
          "investment advice — dollar P&L with exits and sizing lives in the TA Backtester."
    )

    out = _ledger_cached(scope, _ledger_stamp(scope))
    if out.empty:
        st.info("No ledger on disk yet for this book — it builds from the signal cache. Run "
                f"`python backfill_signals.py{' --equities' if scope == 'equities' else ''}` "
                "once (then the daily pull keeps it fresh).")
        return

    _guard = sigledger.guard_refusal(scope)
    if _guard:
        st.warning(
            f"⚠️ The last ledger update ({_guard.get('when', '?')[:16]}) was **refused**: the "
            f"day's price frame would have re-marked {_guard.get('flips', 0):.1%} of settled "
            f"historical outcomes, which never legitimately happens — the frame was corrupt "
            f"(typically a wedged/partial Bloomberg morning; see 2026-08-13). The track record "
            f"below is intact and simply hasn't gained that day's signals yet; the next clean "
            f"morning rebuild clears this notice automatically.")

    # ---- controls ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.0, 1.2, 1.6, 0.9])
    horizon = c1.selectbox("Horizon (sessions)", list(sigledger.HORIZONS),
                           index=len(sigledger.HORIZONS) - 1, key="sl_h")
    _wins = {"All history": None, "Last year": 1, "Last 3 years": 3, "Last 5 years": 5}
    win = c2.selectbox("Window", list(_wins), key="sl_w")
    mkts = c3.multiselect("Products", sorted(out["market"].dropna().unique()),
                          default=[], key=f"sl_m_{scope}", help="Blank = the whole book.")
    min_n = c4.number_input("Min signals", 1, 500, 25, step=5, key="sl_n",
                            help="League rows with fewer signals than this are hidden — "
                                 "a 3-signal 100% hit rate isn't a track record.")
    view = out
    if _wins[win]:
        view = view[view["date"] >= view["date"].max() - pd.DateOffset(years=_wins[win])]
    if mkts:
        view = view[view["market"].isin(set(mkts))]
    if view.empty:
        st.info("Nothing matches these filters.")
        return

    hcol, scol = f"hit{horizon}", f"sig{horizon}"
    evaluable = view[view[hcol].notna()]

    # ---- headline ------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Signals flagged", f"{len(view):,}")
    k2.metric("Evaluable at horizon", f"{len(evaluable):,}")
    k3.metric(f"Overall hit rate ({horizon}d)",
              f"{evaluable[hcol].mean() * 100:.1f}%" if len(evaluable) else "—")
    lg_all = sigledger.league(view, "strategy")
    lg_ok = lg_all[lg_all["n"] >= min_n]
    k4.metric("Best strategy at horizon",
              lg_ok.sort_values(f"hit{horizon}", ascending=False)["strategy"].iloc[0]
              if len(lg_ok) else "—")

    # Shared cell styling for every table on the page.
    def _div_bg(v, centre: float, span: float) -> str:
        if pd.isna(v):
            return ""
        x = max(-1.0, min(1.0, (float(v) - centre) / span))
        r, g, b = (30, 132, 73) if x > 0 else (192, 57, 43)     # #1e8449 / #c0392b
        return f"background-color: rgba({r},{g},{b},{0.12 + 0.43 * abs(x):.2f})"

    def _sig_fg(v) -> str:
        # σ-move styling: no fill, italic, text tinted by drift sign — visually distinct
        # from the hit columns' filled cells while keeping green=good / red=bad. Mid-tone
        # colours stay legible on both themes; ~zero drift keeps the theme's own text.
        if pd.isna(v) or abs(float(v)) < 0.02:
            return "font-style: italic"
        c = "#27ae60" if float(v) > 0 else "#e05545"
        return f"font-style: italic; color: {c}; font-weight: 600"

    # ---- league table --------------------------------------------------------------
    # Landing state is persisted (💾) — factory default Product + one-vote (Ben
    # 2026-08-13); the saved choice leads each radio so the default option sits LEFT.
    _SL_PREFS = Path("data/sigledger_prefs.json")

    def _sl_prefs() -> dict:
        try:
            return json.loads(_SL_PREFS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    _prefs = _sl_prefs()
    _by_saved = _prefs.get("by", "Product")
    _cnt_saved = _prefs.get("counting", "One vote per axis / day")
    _by_opts = [_by_saved] + [o for o in ("Product", "Strategy") if o != _by_saved]
    _cnt_opts = [_cnt_saved] + [o for o in ("One vote per axis / day", "All signals")
                                if o != _cnt_saved]
    _lb1, _lb2, _lb3 = st.columns([1, 2.0, 0.9])
    by = _lb1.radio("League by", _by_opts, horizontal=True, key="sl_by")
    _cnt = _lb2.radio(
        "Counting", _cnt_opts, horizontal=True, key="sl_cnt",
        help="**All signals** pools every flagged (day, strategy) row — five trend methods "
             "echoing the same call count five times, so axes with many members dominate. "
             "**One vote per axis / day** collapses each day's methods within an axis to its "
             "net direction (majority; ties drop) — one independent call per dimension per "
             "day, the same double-count the confluence score's de-dup guards against.")
    _lb3.markdown("<div style='height:1.72em'></div>", unsafe_allow_html=True)
    if _lb3.button("💾 Save as default", key="sl_prefs_save",
                   help="Make the current League by + Counting the page's landing state "
                        "(here and on the VPS after the next sync)."):
        _SL_PREFS.write_text(json.dumps({"by": by, "counting": _cnt}), encoding="utf-8")
        st.toast(f"Ledger default saved: {by} · {_cnt}.", icon="💾")
    vote_mode = _cnt != "All signals"
    if vote_mode:
        # Cached full-book votes, then the same row filters as `view` — collapsing 10.7M
        # equities rows per widget click would drag; filtering after the collapse is
        # equivalent (both are row filters on date/market).
        _v = _ledger_votes_cached(scope, _ledger_stamp(scope))
        if _wins[win]:
            _v = _v[_v["date"] >= out["date"].max() - pd.DateOffset(years=_wins[win])]
        if mkts:
            _v = _v[_v["market"].isin(set(mkts))]
        _lg_src = _v
    else:
        _lg_src = view
    _strat_col = "Axis" if vote_mode else "Strategy"
    lg = sigledger.league(_lg_src, "strategy" if by == "Strategy" else "market")
    lg = lg[lg["n"] >= min_n].sort_values(f"hit{horizon}", ascending=False)
    def _render_league(frame: pd.DataFrame, first_col: str) -> None:
        """One league table (main or drill-down): first column + optional Category,
        Signals, then hit/σ-move interleaved per horizon with the shared styling."""
        hit_cols = [f"Hit {h}d" for h in sigledger.HORIZONS]
        sig_cols = [f"σ-move {h}d" for h in sigledger.HORIZONS]
        num = pd.DataFrame({
            first_col: frame.iloc[:, 0].to_numpy(),
            **({"Category": frame.iloc[:, 0].map(tascore.axis_tag).to_numpy()}
               if first_col == "Strategy" else {}),
            "Signals": frame["n"].to_numpy(),
            # hit + σ-move interleaved per horizon (Ben's preferred reading order)
            **{col: frame[src].to_numpy() for h in sigledger.HORIZONS
               for col, src in ((f"Hit {h}d", f"hit{h}"), (f"σ-move {h}d", f"sig{h}"))},
        })
        # Opened-sector member rows (└) get a light BASIS-gold backdrop on their label
        # cells so they read as a band under their sector — the hit/σ cells keep the
        # diverging colour code untouched.
        _label_cols = [c for c in num.columns if c not in hit_cols + sig_cols]

        def _member_bg(row):
            gold = ("background-color: rgba(245,197,24,0.14)"
                    if str(row.iloc[0]).lstrip().startswith("└") else "")
            return [gold if c in _label_cols else "" for c in row.index]

        sty = (num.style
               .format({"Signals": "{:,.0f}",
                        **{c: "{:.1f}%" for c in hit_cols},
                        **{c: "{:+.2f}" for c in sig_cols}}, na_rep="—")
               .apply(_member_bg, axis=1)
               # hits: 50% = coin flip = neutral; ±5pp = full colour
               .map(lambda v: _div_bg(v, 50.0, 5.0), subset=hit_cols)
               # σ-moves deliberately DON'T get the fill (Ben: make the two measures look
               # different) — plain cells, italic, text tinted by drift direction instead.
               .map(_sig_fg, subset=sig_cols))
        st.dataframe(sty, hide_index=True, use_container_width=True,
                     height=min(560, 40 + 35 * len(num)))

    if lg.empty:
        st.info("No rows clear the min-signals bar — lower it to see thin samples.")
    else:
        _display, _fc, _opened = lg, (_strat_col if by == "Strategy" else by), []
        if vote_mode and by == "Strategy":
            # Open a sector IN PLACE: its member strategies splice in as indented rows
            # right under their axis row — one table, same columns and shading.
            _opened = st.multiselect(
                "Open sectors — show their member strategies as their own rows",
                list(lg.iloc[:, 0]), default=[], key=f"sl_axopen_{scope}",
                help="An opened sector keeps its de-duplicated vote row; the indented "
                     "rows beneath it are its individual strategies, pooled counting.")
            if _opened:
                _frames = []
                for _, _r in lg.iterrows():
                    _frames.append(_r.to_frame().T)
                    _tag = _r.iloc[0]
                    if _tag in _opened:
                        _ax = next((a for a, t in tascore.AXIS_TAGS.items() if t == _tag),
                                   None)
                        dl = sigledger.league(
                            view[view["strategy"].isin(set(tascore.TA_AXES.get(_ax, ())))],
                            "strategy")
                        dl = dl[dl["n"] >= min_n].sort_values(f"hit{horizon}",
                                                              ascending=False)
                        dl.iloc[:, 0] = "└  " + dl.iloc[:, 0].astype(str)
                        _frames.append(dl)
                _display = pd.concat(_frames, ignore_index=True)
                for _c in _display.columns[1:]:
                    _display[_c] = pd.to_numeric(_display[_c], errors="coerce")
                _fc = "Axis / strategy"
        _render_league(_display, _fc)
        st.caption(("One row per (day, product, axis) net call — echoes within an axis "
                    "already collapsed, so these are independent daily votes. "
                    if vote_mode else
                    "Every flagged (day, strategy) row counts once — correlated same-day "
                    "signals from one axis each count in full (see the Counting toggle). ")
                   + ("Indented └ rows are an opened sector's individual strategies — "
                      "they pool all their signals, so they can read stronger or weaker "
                      "than the sector's single de-duplicated daily vote. "
                      if _opened else "")
                   + "Hit = the signal-space move went the signal's way by the horizon. "
                     "σ-move = the mean signed move in trailing-21-session σ units — the "
                     "honest size of the edge, not just its frequency.")
        if not vote_mode:
            _cs = tascore.confluence_set(scope)
            st.caption("🧭 The **Confluence** row is the composite scored on the saved "
                       f"confluence set — currently **{' · '.join(_cs)}** "
                       f"({len(_cs)} method{'s' if len(_cs) != 1 else ''}). Change it from "
                       "the TA hub's Confluence-set picker — **Save as default** re-scores "
                       "the row's whole history under the new set.")

        # ---- product drill-down: which strategies/axes drive that product's record ----
        if by == "Product":
            pick = st.selectbox(
                f"🔍 Drill into a product — its per-{'axis' if vote_mode else 'strategy'} breakdown",
                ["—"] + list(lg.iloc[:, 0]), key=f"sl_drill_{scope}")
            if pick != "—":
                dmin = max(5, int(min_n) // 5)
                dl = sigledger.league(_lg_src[_lg_src["market"] == pick], "strategy")
                dl = dl[dl["n"] >= dmin].sort_values(f"hit{horizon}", ascending=False)
                if dl.empty:
                    st.info(f"Nothing clears {dmin} signals on {pick} with these filters.")
                else:
                    _render_league(dl, _strat_col)
                    st.caption(f"{'Axes' if vote_mode else 'Strategies'} ranked on **{pick}** "
                               f"alone, same filters as the league. Per-product samples are "
                               f"thinner, so the bar here is min {dmin} signals per row "
                               f"(⅕ of the league's, floor 5).")

    # ---- strategy × product/sector heat --------------------------------------------
    _heat_by = "sector" if scope == "equities" else "market"
    hm = sigledger.heat(view, horizon, by=_heat_by)
    hm = hm[hm["n"] >= max(5, min_n // 5)]
    if not hm.empty:
        st.markdown(f"##### Hit rate by strategy × "
                    f"{'sector' if scope == 'equities' else 'product'} ({horizon}d)")
        st.altair_chart(alt.Chart(hm).mark_rect().encode(
            x=alt.X("market:N", title=None, sort=alt.SortField("market")),
            y=alt.Y("strategy:N", title=None),
            color=alt.Color("hit:Q", title="Hit %",
                            scale=alt.Scale(domain=[30, 50, 70],
                                            range=["#c0392b", "#3a4454", "#1e8449"])),
            tooltip=[alt.Tooltip("strategy:N"), alt.Tooltip("market:N"),
                     alt.Tooltip("hit:Q", format=".1f", title=f"hit % ({horizon}d)"),
                     alt.Tooltip("n:Q", title="signals")],
        ).properties(height=26 * hm["strategy"].nunique() + 40), use_container_width=True)
        st.caption("Green = the signal family has historically been RIGHT on that product at this "
                   "horizon; red = fade-worthy. Thin cells (few signals) are dropped.")

    # ---- era league + regime read (dropdown) — the regime-rotation story -----------
    # Deliberately ignores the Window filter (it IS all the windows at once); the product
    # filter and min-signals bar still apply.
    _base = out[out["market"].isin(set(mkts))] if mkts else out
    wl = sigledger.windows_league(_base, horizon, min_n=int(min_n))
    if not wl.empty:
        _yrs = out["date"].max().year - out["date"].min().year
        _wlbl = {"Full": f"Full {_yrs}y", "5y": "5y", "3y": "3y", "1y": "1y"}
        with st.expander(f"🧭 Era league — hit rate by lookback window ({horizon}d), "
                         f"with the regime read"):
            _rr = sigledger.regime_read(out)
            if _rr:
                st.markdown(_rr["text"])
                st.caption(f"Written by the ledger itself from the full unfiltered book "
                           f"(21-session horizon, min-sample gated) as of "
                           f"{_rr['asof'].date()} — it re-writes with every morning snapshot, "
                           f"so when signal leadership rotates, this paragraph rotates with it.")
            wl_cols = [_wlbl[label] for _, label in sigledger.WINDOWS]
            wnum = pd.DataFrame({
                "Strategy": wl["strategy"].to_numpy(),
                "Category": wl["strategy"].map(tascore.axis_tag).to_numpy(),
                "Signals": wl["n Full"].to_numpy(),
                **{_wlbl[label]: wl[label].to_numpy() for _, label in sigledger.WINDOWS},
                f"Δ 1y vs full": wl["delta"].to_numpy(),
            })
            wsty = (wnum.style
                    .format({"Signals": "{:,.0f}",
                             **{c: "{:.1f}%" for c in wl_cols},
                             "Δ 1y vs full": "{:+.1f}pp"}, na_rep="—")
                    .map(lambda v: _div_bg(v, 50.0, 5.0), subset=wl_cols)
                    .map(lambda v: _div_bg(v, 0.0, 5.0), subset=["Δ 1y vs full"]))
            st.dataframe(wsty, hide_index=True, use_container_width=True,
                         height=min(560, 40 + 35 * len(wnum)))
            st.caption("Each column is the hit rate over that trailing window (all ending "
                       "today), shortest on the left — so one row tells a strategy's whole "
                       "regime story: green on the left fading right is a strategy that works "
                       "NOW but didn't before, green all the way across is persistence. "
                       "Ignores the Window filter by design; products and min-signals still "
                       "apply. Blank cells have fewer signals in that window than the "
                       "min-signals bar.")

        # ---- year-by-year breakdown (dropdown) --------------------------------------
        with st.expander(f"📆 Year-by-year breakdown ({horizon}d)"):
            yl = sigledger.year_league(_base, horizon, min_n=int(min_n))
            if yl.empty:
                st.info("No year clears the min-signals bar with these filters.")
            else:
                yl = yl.reindex([s for s in wl["strategy"] if s in yl.index])
                ynum = pd.DataFrame({
                    "Strategy": yl.index.to_numpy(),
                    "Category": yl.index.map(tascore.axis_tag).to_numpy(),
                    **{str(y): yl[y].to_numpy() for y in yl.columns},
                })
                yr_cols = [str(y) for y in yl.columns]
                ysty = (ynum.style
                        .format({c: "{:.1f}%" for c in yr_cols}, na_rep="—")
                        .map(lambda v: _div_bg(v, 50.0, 5.0), subset=yr_cols))
                st.dataframe(ysty, hide_index=True, use_container_width=True,
                             height=min(560, 40 + 35 * len(ynum)))
                st.caption(f"Hit rate per calendar year at the {horizon}-session horizon, rows "
                           f"in the era league's order. The first and last years are partial "
                           f"({_base['date'].min():%b %Y} start, ledger runs to "
                           f"{_base['date'].max():%b %Y}). Blank cells have fewer signals that "
                           f"year than the min-signals bar.")

    # ---- recent signals blotter ----------------------------------------------------
    st.markdown("##### Latest signals — and how they resolved")
    recent = view.sort_values("date", ascending=False).head(30)
    bl = pd.DataFrame({
        "Date": recent["date"].dt.date.astype(str),
        "Strategy": recent["strategy"], "Product": recent["market"],
        "Signal": recent["signal"],
        "Entry level": recent["entry_level"].map(
            lambda v: f"{v:,.3f}" if pd.notna(v) else "—"),
        **{f"{h}d": recent.apply(
            lambda r, h=h: ("pending" if pd.isna(r[f"hit{h}"])
                            else ("✓" if r[f"hit{h}"] else "✗") + f" {r[f'sig{h}']:+.1f}σ"),
            axis=1) for h in sigledger.HORIZONS},
    })
    _out_cols = [f"{h}d" for h in sigledger.HORIZONS]

    def _outcome_bg(v: str) -> str:
        if isinstance(v, str) and v.startswith("✓"):
            return "background-color: rgba(30,132,73,0.30)"
        if isinstance(v, str) and v.startswith("✗"):
            return "background-color: rgba(192,57,43,0.30)"
        return "opacity: 0.55"                                   # pending — visibly muted
    st.dataframe(bl.style.map(_outcome_bg, subset=_out_cols),
                 hide_index=True, use_container_width=True,
                 height=min(560, 40 + 35 * len(bl)))
    st.caption(f"Ledger spans {out['date'].min().date()} → {out['date'].max().date()} · "
               f"{len(out):,} flagged signals · rebuilt with each morning snapshot.")

    st.caption("🗂️ The **Weekly Signal Scorecard** — this page's numbers as a client PDF — "
               "is built from the 📈 Technical Analysis page's report controls (and as a "
               "scheduled Monday email via Alert Settings).")


def render_strategy_builder() -> None:
    """Multi-leg option strategy builder (the optioncreator.com workflow): build a
    position from Buy/Sell x Call/Put/Future legs, read net debit/credit, max
    profit / max loss / breakevens / greeks, and see the P&L curve at expiry plus
    a re-priced 'T + d days' scenario line. Engine: src/optbuilder.py (Black-76,
    same convention as the Vol Backtester)."""
    import altair as alt
    # ---- hand-off from STIR Paths: mark the underlying at the scenario landing ----
    _ho = st.session_state.pop("osb_handoff", None)
    if _ho:
        st.session_state["osb_prod"] = _ho["prod"]
        st.session_state["osb_prod_prev"] = _ho["prod"]   # skip the live-quote reseed below
        st.session_state["osb_spot"] = float(_ho["spot"])
        st.session_state["osb_pv"] = float(volbt.point_value(_ho["prod"])) or 1.0
        st.session_state.pop("osb_rows", None)
        st.session_state["osb_nonce"] = st.session_state.get("osb_nonce", 0) + 1
        st.session_state["osb_ho_note"] = _ho["note"]
        st.session_state["osb_ho_days"] = _ho.get("days")       # → preselect the leg month
        st.session_state["osb_ho_nodes"] = _ho.get("nodes")     # → scenario-expected P&L
    st.subheader("🧰  Option Strategy Builder — multi-leg payoff modeller")
    if st.session_state.get("osb_ho_note"):
        st.info(f"Seeded from STIR Paths: {st.session_state['osb_ho_note']}. The underlying is "
                "marked at the scenario landing price — build the legs against it, or reset the "
                "price to live from the product picker.")
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
                st.session_state.pop("osb_ho_note", None)   # a STIR hand-off dies with its product
                st.session_state.pop("osb_ho_nodes", None)
                st.session_state.pop("osb_ho_days", None)
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
    _hd = st.session_state.pop("osb_ho_days", None)
    if _hd is not None:                          # STIR hand-off: leg month ≈ the option's expiry
        st.session_state["osb_dmonth"] = _closest_month(float(_hd))
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
    _nodes = st.session_state.get("osb_ho_nodes")
    if _nodes:
        _epnl = sum(pr * optbuilder.strategy_pnl(legs, float(px), F0, None, r)
                    for px, pr in _nodes)
        _ppos = sum(pr for px, pr in _nodes
                    if optbuilder.strategy_pnl(legs, float(px), F0, None, r) > 0)
        e1, e2, _ = st.columns([1.2, 1.2, 3])
        e1.metric("Scenario E[P&L]", ("+" if _epnl >= 0 else "−") + _fmt(abs(_epnl)),
                  help="Probability-weighted P&L at the front expiry over your STIR scenario's "
                       "outcome distribution (carried in from the STIR Paths hand-off) — the "
                       "structure's expected value if your meeting odds are right.")
        e2.metric("P(profit | scenario)", f"{_ppos * 100:.0f} %",
                  help="Share of your scenario's probability mass landing where this structure "
                       "makes money — YOUR odds, vs the lognormal P(profit) beside it.")

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
    if _nodes:
        _nd = pd.DataFrame({"x": [float(px) for px, _ in _nodes],
                            "p": [float(pr) * 100.0 for _, pr in _nodes]})
        chart += alt.Chart(_nd).mark_point(filled=True, shape="circle", opacity=0.9).encode(
            x="x:Q", y=alt.value(408),
            size=alt.Size("p:Q", scale=alt.Scale(range=[25, 320]), legend=None),
            color=alt.value(cc["accent"]),
            tooltip=[alt.Tooltip("x:Q", format=",.4f", title="scenario landing"),
                     alt.Tooltip("p:Q", format=".1f", title="probability %")])
    brand.show_chart(chart.properties(height=420).interactive(bind_y=False))
    st.caption("Dotted vertical = current underlying; dashed blue verticals = breakevens. "
               + ("Gold dots along the bottom = your STIR scenario's landing outcomes "
                  "(size = probability). " if _nodes else "")
               + "Scroll / drag to zoom the price axis.")

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
def _cm_monitor(window: int, threshold: float, mode: str, rev: int = 0):
    """The curve/RV spread book, cached so widget reruns don't re-read the deep store
    (`rev` = curvemon.REV keys the cache to the engine's book/schema version, so an
    engine change never serves a stale frame for the rest of the TTL)."""
    return curvemon.monitor(window, threshold)


@st.cache_data(show_spinner=False, ttl=1800)
def _cm_chart(key: str, window: int, threshold: float, years, mode: str, rev: int = 0):
    return curvemon.spread_chart_data(key, window, threshold, years=years)


_CM_PREFS = ROOT / "data" / "curvemon_prefs.json"


def _cm_load_prefs() -> dict:
    """Saved page defaults (groups / window / threshold / sort) from 📌 Set as default."""
    try:
        return json.loads(_CM_PREFS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_curve_monitor() -> None:
    import altair as alt

    st.subheader("📐  Curve / RV Monitor — the spread book on ten years of history")
    st.caption(
        "A fixed book of curve and relative-value spreads — rate curves and cross-market "
        "spreads on **benchmark yields** (bp), STIR calendars, energy time-spreads and metal "
        "ratios — each scored two ways: a **rolling z-score** (stretch vs the recent regime) "
        "and a **percentile of the full ~10-year deep store** (stretch vs everything the last "
        "decade has shown). Levels are real market observables — actual front prices and "
        "benchmark yields, never back-adjusted continuations.")

    prefs = _cm_load_prefs()
    _win_opts = ["3 months (63d)", "6 months (126d)", "1 year (252d)", "2 years (504d)"]
    _win_idx = next((i for i, o in enumerate(_win_opts)
                     if f"({prefs.get('window')}d)" in o), 2)
    _def_groups = [g for g in prefs.get("groups", curvemon.GROUPS) if g in curvemon.GROUPS] \
        or curvemon.GROUPS

    c0, c1, c2, c3 = st.columns([1.15, 0.85, 1.6, 1.0], vertical_alignment="bottom")
    win_lbl = c0.selectbox("Z-score window", _win_opts, index=_win_idx, key="cm_window",
                           help="Sessions behind the rolling mean/σ the z-score measures against. "
                                "The 10-year percentile always uses the full store.")
    window = int(win_lbl.split("(")[1].rstrip("d)"))
    threshold = float(c1.number_input("Flag threshold (σ)", 0.5, 4.0,
                                      float(prefs.get("threshold", curvemon.Z_THRESHOLD)),
                                      0.25, key="cm_thr"))
    groups = c2.multiselect("Groups", curvemon.GROUPS, default=_def_groups, key="cm_groups")
    if c3.button("📌 Set as default", key="cm_setdef", use_container_width=True,
                 help="Save the current groups, window, threshold and sort as this page's "
                      "startup defaults (e.g. to drop sectors you don't watch)."):
        try:
            _CM_PREFS.write_text(json.dumps({
                "groups": groups, "window": window, "threshold": threshold,
                "sort": st.session_state.get("cm_sort", "Term"),
            }, indent=2), encoding="utf-8")
            st.toast("Saved — the Curve Monitor will open like this from now on.", icon="📌")
        except Exception as e:
            st.warning(f"Couldn't save defaults: {e}")

    mon = _cm_monitor(window, threshold, MODE, curvemon.REV)
    if mon is None or mon.empty:
        st.info("No spread history yet — the deep price store hasn't been built on this "
                "machine (it backfills on the next Bloomberg session).")
        return
    if groups:
        mon = mon[mon["group"].isin(groups)]

    flagged = mon[mon["signal"] != "—"]
    if not flagged.empty:
        st.markdown("**Stretched now:** " + " · ".join(
            f"{r['name']} **{r['signal']}** ({r['z']:+.1f}σ, {r['pctl']:.0f}th %ile of 10y)"
            for _, r in flagged.iterrows()))
    st.caption("★ = the pair that market's desk actually quotes (US 2s10s & 5s30s; Germany "
               "2s10s & 10s30s — the euro long end trades off the Bund). Hover any spread's "
               "name for what it is.")

    _sort_opts = ["Term", "Country", "A–Z"]
    _sort_idx = _sort_opts.index(prefs["sort"]) if prefs.get("sort") in _sort_opts else 0
    s0, _s1 = st.columns([1.3, 2.7])
    sort_by = s0.radio("Sort", _sort_opts, index=_sort_idx, horizontal=True, key="cm_sort",
                       help="Term = like-for-like, each tenor pair's two markets adjacent, "
                            "short end to long end. Country = US block then Germany. "
                            "A–Z = alphabetical. (The curve group is where they differ.)")
    if sort_by == "A–Z":
        mon = mon.sort_values("name", kind="stable")
    elif sort_by == "Country" and "mkt" in mon.columns:
        _mkt_rank = {"US": 0, "Germany": 1}      # ladder markets first, cross-market/box after
        mon = mon.sort_values("mkt", key=lambda c: c.map(lambda m: _mkt_rank.get(m, 99)),
                              kind="stable")
    # "Term" = the engine's book order: tenor pairs short-to-long, markets adjacent

    _cols = [
        {"key": "name", "label": "Spread",
         "help": "The relationship being tracked — hover each name for its definition"},
        {"key": "level_txt", "label": "Level", "align": "right",
         "help": "Today's level of the spread, in its native unit"},
        {"key": "chg1d", "label": "1d Δ", "color": True, "fmt": "{:+,.2f}",
         "help": "Change on the day, same unit (green up / red down)"},
        {"key": "z", "label": "Z", "align": "right", "fmt": "{:+.2f}",
         "help": "How stretched vs the recent regime: standard deviations from the rolling "
                 "mean over the chosen z-score window"},
        {"key": "zpic", "label": "±2σ", "zbar": True, "keep_case": True,
         "help": "The z-score drawn on a ±2σ scale from the centre tick — a full-length bar "
                 "is at or beyond the flag threshold"},
        {"key": "pctl", "label": "10y %ile", "align": "right", "fmt": "{:.0f}",
         "help": "How stretched vs the whole stored decade: share of ~10 years of sessions "
                 "with the spread at or below today (0 = decade low, 100 = decade high)"},
        {"key": "pctl", "label": "10y range", "pbar": True,
         "help": "The same percentile drawn 0–100 — tick = the decade's median; the fill "
                 "turns gold inside the top or bottom decile"},
        {"key": "hl", "label": "½-life", "align": "right",
         "help": "Ornstein–Uhlenbeck half-life: the typical number of sessions a stretch has "
                 "taken to close half-way back to the mean"},
        {"key": "signal", "label": "Signal",
         "help": "Rich / Cheap once |z| clears the flag threshold — an observation against "
                 "the spread's own history, not a recommendation"},
    ]
    for g in [g for g in curvemon.GROUPS if g in set(mon["group"])]:
        sub = mon[mon["group"] == g]
        brand.panel_header(g, right=f"{len(sub)} spreads")
        rows = []
        _esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
        _gold = brand.palette()["gold"]
        for _, r in sub.iterrows():
            _star = (f' <span style="color:{_gold}" title="The benchmark quote for this '
                     f'market&#39;s curve">★</span>') if r.get("bench") else ""
            rows.append({
                "name": (f'<span title="{_esc(r["desc"])}" style="cursor:help;border-bottom:'
                         f'1px dotted rgba(128,128,128,.55)">{_esc(r["name"])}</span>{_star}'),
                "level_txt": f"{r['level']:,.{int(r['dp'])}f} {r['unit']}",
                "chg1d": None if pd.isna(r["chg1d"]) else float(r["chg1d"]),
                "z": float(r["z"]), "zpic": float(r["z"]),
                "pctl": float(r["pctl"]),
                "hl": "—" if pd.isna(r["half_life"]) else f"{r['half_life']:.0f}d",
                "signal": r["signal"] if r["signal"] == "—" else f"{r['signal']} {r['z']:+.1f}σ",
            })
        brand.terminal_table(rows, _cols)

    # ---- spread detail --------------------------------------------------------
    st.divider()
    brand.panel_header("Spread detail",
                       right=f"window {window}d · flag ±{threshold:g}σ")
    d0, d1 = st.columns([2.2, 1.4])
    _bench_of = (dict(zip(mon["name"], mon["bench"])) if "bench" in mon.columns else {})
    sel_name = d0.selectbox("Spread", mon["name"].tolist(), key="cm_sel",
                            format_func=lambda n: f"★ {n}" if _bench_of.get(n) else n,
                            label_visibility="collapsed")
    rng = d1.radio("Range", ["Full", "5y", "2y", "1y"], horizontal=True, key="cm_rng",
                   label_visibility="collapsed")
    row = mon[mon["name"] == sel_name].iloc[0]
    years = {"Full": None, "5y": 5.0, "2y": 2.0, "1y": 1.0}[rng]
    cd, info = _cm_chart(row["key"], window, threshold, years, MODE, curvemon.REV)
    if cd.empty:
        st.info("No history for this spread.")
        return
    st.caption(row["desc"])

    dp = int(row["dp"])
    m0, m1, m2, m3 = st.columns(4)
    m0.metric(f"Level ({row['unit']})", f"{row['level']:,.{dp}f}",
              None if pd.isna(row["chg1d"]) else f"{row['chg1d']:+,.{dp}f} on the day",
              delta_color="off")
    m1.metric(f"Z ({window}d)", f"{row['z']:+.2f}σ")
    m2.metric("10-year percentile", f"{row['pctl']:.0f}th",
              help="Share of the last decade's sessions with the spread at or below today.")
    m3.metric("Half-life", "—" if pd.isna(row["half_life"]) else f"≈{row['half_life']:.0f}d",
              help="Ornstein-Uhlenbeck estimate of how long a stretch takes to close halfway.")

    cc = brand.chart_colors()
    x_enc = alt.X("date:T", title=None)
    base = alt.Chart(cd)
    band = base.mark_area(opacity=0.13, color=cc["accent"]).encode(
        x=x_enc, y=alt.Y("lower:Q", title=f"spread ({row['unit']})",
                         scale=alt.Scale(zero=False)), y2="upper:Q")
    mean_ln = base.mark_line(strokeDash=[5, 3], color=cc["muted"], strokeWidth=1.5).encode(
        x=x_enc, y="mean:Q")
    spread_ln = base.mark_line(color=cc["series"], strokeWidth=2.1).encode(
        x=x_enc, y=alt.Y("spread:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T"),
                 alt.Tooltip("spread:Q", title=f"spread ({row['unit']})", format=f",.{dp}f"),
                 alt.Tooltip("z:Q", format="+.2f")])
    brand.show_chart((band + mean_ln + spread_ln).properties(height=320))
    z_rules = alt.Chart(pd.DataFrame({"y": [threshold, 0.0, -threshold]})).mark_rule(
        color=cc["muted"], strokeDash=[4, 3], strokeWidth=1).encode(y="y:Q")
    z_ln = base.mark_line(color=cc["accent"], strokeWidth=1.6).encode(
        x=x_enc, y=alt.Y("z:Q", title="z"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("z:Q", format="+.2f")])
    brand.show_chart((z_rules + z_ln).properties(height=120))

    dsig = ("" if pd.isna(row["dollar_sigma"]) or row["dollar_sigma"] is None
            else f" (≈${row['dollar_sigma']:,.0f} per 1-lot spread)")
    st.caption(
        f"Mean ({window}d) **{row['mean']:,.{dp}f} {row['unit']}** — the level a reversion "
        f"points back to · invalidation reference **{row['invalidation']:,.{dp}f}** "
        f"({'+' if row['z'] >= 0 else '−'}{curvemon.INVAL_SIGMA:g}σ on the stretched side) · "
        f"1σ = **{row['sigma']:,.{dp}f} {row['unit']}**{dsig} · history since "
        f"**{row['first']}** ({row['days']:,} sessions). Levels are observations against the "
        "spread's own history, not a recommendation.")

    # ---- client PDF ------------------------------------------------------------
    st.divider()
    if st.button("📐 Generate Curve / RV Report (visual PDF)", type="primary", key="cm_pdf_btn"):
        with st.spinner("Rendering the Curve / RV report…"):
            try:
                # chart the four most-stretched spreads — flagged ones rank first by |z|
                focus = mon.reindex(mon["z"].abs().sort_values(ascending=False).index).head(4)
                charts = []
                for _, r in focus.iterrows():
                    cdf, cinfo = _cm_chart(r["key"], window, threshold, None, MODE, curvemon.REV)
                    if cdf.empty:
                        continue
                    charts.append({
                        "key": r["key"], "name": r["name"], "unit": r["unit"],
                        "dp": int(r["dp"]), "desc": r["desc"],
                        "dates": [d.strftime("%Y-%m-%d") for d in cdf["date"]],
                        "spread": [None if pd.isna(v) else float(v) for v in cdf["spread"]],
                        "mean": [None if pd.isna(v) else float(v) for v in cdf["mean"]],
                        "upper": [None if pd.isna(v) else float(v) for v in cdf["upper"]],
                        "lower": [None if pd.isna(v) else float(v) for v in cdf["lower"]],
                        "info": {k: (None if isinstance(v, float) and pd.isna(v) else v)
                                 for k, v in cinfo.items()},
                    })
                payload = {
                    "asof": str(mon["asof"].max()), "window": window, "threshold": threshold,
                    "groups": curvemon.GROUPS,
                    "rows": [{k: (None if isinstance(v, float) and pd.isna(v) else v)
                              for k, v in r.items()} for r in mon.to_dict("records")],
                    "charts": charts,
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "curve.json"
                    _out = Path(_t) / "Curve_RV_Monitor.pdf"
                    _in.write_text(json.dumps(payload), encoding="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(CURVEREPORT_CLI), str(_in), str(_out)],
                        capture_output=True, text=True, timeout=180)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["cm_pdf"] = _out.read_bytes()
                    else:
                        st.error("Report failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed: {e}")
    if st.session_state.get("cm_pdf"):
        st.download_button("⬇️  Download Curve / RV Report", data=st.session_state["cm_pdf"],
                           file_name="Curve_RV_Monitor.pdf", mime="application/pdf",
                           key="cm_pdf_dl")
        email_report_ui("cm_email", "curve", st.session_state["cm_pdf"],
                        subject="BASIS — Curve / RV Monitor",
                        attachment_name="Curve_RV_Monitor.pdf")


# ---------------------------------------------------------------------------
# Seasonality — calendar patterns on the deep store
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def _seas_changes(mode: str):
    """Monthly + weekly change frames for the whole book — reads the daily disk
    store (~ms; the 7.5s deep-store scan only runs if the store is stale)."""
    return seasmon.changes_cached()


@st.cache_data(show_spinner=False, ttl=1800)
def _seas_windows(ticker: str, mode: str):
    _mo, wk = _seas_changes(mode)
    return seasmon.best_windows(wk, ticker)


@st.cache_data(show_spinner=False, ttl=1800)
def _seas_spread_products(mode: str):
    return seasmon.spread_products()


@st.cache_data(show_spinner=False, ttl=1800)
def _seas_spread(ticker: str, mode: str):
    return seasmon.spread_seasonal(ticker)


@st.cache_data(show_spinner=False, ttl=1800)
def _seas_spread_screener(mode: str):
    return seasmon.spread_screener_cached()      # daily disk store, ms on open


@st.cache_data(show_spinner=False, ttl=1800)
def _seas_open_windows(mode: str):
    """Whole-book open/upcoming windows — the SEAS radar's source list (JSON scan
    cache per data day + ISO week, so this is normally a disk read)."""
    return seasmon.open_windows()


def _seas_fmt(unit: str) -> str:
    """Signed number format for a seasonality unit — bp whole, % one decimal."""
    return "{:+,.0f}" if unit == "bp" else "{:+,.1f}"


def _seas_wspan(start, weeks) -> str:
    """'W35 → W45' — a window's ISO-week span, the weekly score's exact basis
    (the paired date span is the fixed-date score's basis)."""
    return f"W{int(start)} → W{(int(start) + int(weeks) - 2) % 52 + 1}"


def render_seasonality() -> None:
    import altair as alt

    st.subheader("📅  Product Seasonality — calendar patterns on ten years of history")
    st.caption(
        "How each product's calendar year has actually traded across the deep store: "
        "per-month return heatmaps, a month screener over the whole book, the average-year "
        "path with the current year overlaid, and the calendar windows a decade rewarded most "
        "consistently. Price products are measured in **%** of the actual front level; fixed "
        "income runs in **yield/rate space (bp)** — a positive month means the yield ROSE. "
        "Everything here is a description of history, not a forecast.")

    monthly, weekly = _seas_changes(MODE)
    if monthly is None or monthly.empty:
        st.info("No seasonal history yet — the deep price store hasn't been built on this "
                "machine (it backfills on the next Bloomberg session).")
        return

    # ---- month screener -----------------------------------------------------
    c0, c1, c2 = st.columns([1.1, 2.35, 0.6], vertical_alignment="bottom")
    mo_sel = c0.selectbox("Month", seasmon.MONTH_LABELS, index=date.today().month - 1,
                          key="seas_month",
                          help="Which calendar month to screen the book on. Defaults to the "
                               "current month; the in-progress month never joins the stats.")
    month = seasmon.MONTH_LABELS.index(mo_sel) + 1
    scr = seasmon.screener(monthly, month)
    if scr.empty:
        st.info("Not enough stored years to screen this month.")
        return
    assets = [a for a in universe.ASSET_CLASSES if a in set(scr["asset"])]
    _saved = [a for a in seasmon.default_sectors() if a in assets]
    sel_assets = c1.multiselect("Sectors", assets, default=_saved or assets, key="seas_assets")
    if IS_ADMIN and c2.button("📌 Set default", key="seas_set_def", use_container_width=True,
                              help="Save the current sector picks as this page's startup "
                                   "selection — they load on every launch. Set default with "
                                   "every sector (or none) selected to reset to all."):
        seasmon.save_default_sectors([] if set(sel_assets) >= set(assets) else sel_assets)
        st.toast("Seasonality sector default saved.", icon="📌")
    _saved = seasmon.default_sectors()     # re-read — reflects a just-saved value this rerun
    if _saved:
        st.caption("📌 Startup default: **" + " + ".join(_saved) + "** — the other sectors "
                   "stay one click away above.")
    if sel_assets:
        scr = scr[scr["asset"].isin(sel_assets)]

    strong = scr[scr["bias"] != "—"].copy()
    strong["_x"] = (strong["hit"] - 0.5).abs()
    strong = strong.sort_values("_x", ascending=False).head(8)
    if not strong.empty:
        st.markdown(f"**Seasonal bias in {mo_sel}:** " + " · ".join(
            f"{r['name']} **{r['bias']} {round(r['hit'] * r['n'])}/{r['n']}**"
            for _, r in strong.iterrows()))

    cur_month = month == date.today().month
    for a in [a for a in assets if a in set(scr["asset"])]:
        sub = scr[scr["asset"] == a]
        unit = sub["unit"].iloc[0]
        fmt = _seas_fmt(unit)
        brand.panel_header(a, right=f"{len(sub)} products · {unit}")
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "name": r["name"],
                "med": float(r["med"]), "hitpic": float((r["hit"] - 0.5) * 4),
                "hit": f"{round(r['hit'] * r['n'])}/{r['n']} {r['bias']}".replace(" —", ""),
                "mean": float(r["mean"]), "best": float(r["best"]), "worst": float(r["worst"]),
                "this": None if pd.isna(r["this_year"]) else float(r["this_year"]),
            })
        brand.terminal_table(rows, [
            {"key": "name", "label": "Product"},
            {"key": "med", "label": f"Med {unit}", "color": True, "fmt": fmt},
            {"key": "hitpic", "label": "Hit ±", "zbar": True},
            {"key": "hit", "label": "Years up", "align": "right"},
            {"key": "mean", "label": f"Mean {unit}", "align": "right", "fmt": fmt},
            {"key": "best", "label": "Best", "align": "right", "fmt": fmt},
            {"key": "worst", "label": "Worst", "align": "right", "fmt": fmt},
            {"key": "this", "label": "This yr (MTD)" if cur_month else "This yr",
             "color": True, "fmt": fmt},
        ])
    st.caption(
        f"**Med / Mean / Best / Worst** = that month's change across the stored years "
        f"(complete months only — the in-progress month is shown under *This yr* but never "
        f"counted). **Years up** = years the month printed positive, with a bias arrow at "
        f"{seasmon.HIT_STRONG:.0%} agreement or better; the centre bar pictures the same "
        "hit rate. Fixed income is the change in the benchmark yield / STIR rate — for a "
        "bond future, a ↑ month means yields typically rose (futures fell).")

    # ---- product detail -----------------------------------------------------
    st.divider()
    tickers = list(scr["ticker"]) if not scr.empty else []
    if not tickers:
        return
    default_t = "NGA Comdty" if "NGA Comdty" in tickers else tickers[0]
    brand.panel_header("Product seasonality", right="ten-year detail")
    tkr = st.selectbox("Product", tickers, index=tickers.index(default_t),
                       format_func=lambda t: f"{universe.yield_name(t)}  ·  {t}",
                       key="seas_tkr", label_visibility="collapsed")
    unit = seasmon.unit_of(tkr)
    fmt = _seas_fmt(unit)
    mat, stats, meta = seasmon.monthly_matrix(monthly, tkr)
    if mat is None or stats is None:
        st.info("Not enough deep history for this product.")
        return

    srow = stats.loc[month]
    m0, m1, m2, m3 = st.columns(4)
    m0.metric(f"{mo_sel} median ({unit})", "—" if pd.isna(srow["med"]) else fmt.format(srow["med"]),
              None if pd.isna(srow["hit"]) else
              f"higher in {round(srow['hit'] * srow['n'])} of {int(srow['n'])} years",
              delta_color="off")
    _c = stats.dropna(subset=["med"])
    if not _c.empty:
        _b, _w = int(_c["med"].idxmax()), int(_c["med"].idxmin())
        m1.metric("Strongest month", seasmon.MONTH_LABELS[_b - 1], fmt.format(_c.loc[_b, "med"]),
                  delta_color="off")
        m2.metric("Weakest month", seasmon.MONTH_LABELS[_w - 1], fmt.format(_c.loc[_w, "med"]),
                  delta_color="off")
    m3.metric("Years on the store", f"{meta['years']}")

    # year × month heatmap
    cc = brand.chart_colors()
    pal = brand.palette()
    hm = mat.reset_index().melt("year", var_name="month", value_name="val").dropna()
    hm["mon"] = hm["month"].map(lambda m: seasmon.MONTH_LABELS[int(m) - 1])
    hm["txt"] = hm["val"].map(lambda v: fmt.format(v))
    if meta["partial"]:
        py, pm = meta["partial"]
        hm.loc[(hm["year"] == py) & (hm["month"] == pm), "txt"] += "*"
    vmax = float(hm["val"].abs().quantile(0.90)) or 1.0
    base = alt.Chart(hm).encode(
        x=alt.X("mon:N", sort=seasmon.MONTH_LABELS, title=None,
                axis=alt.Axis(labelAngle=0, orient="top", labelFontSize=12)),
        y=alt.Y("year:O", sort="descending", title=None, axis=alt.Axis(labelFontSize=12)))
    cells = base.mark_rect(stroke=pal["canvas"], strokeWidth=1.4).encode(
        color=alt.Color("val:Q", legend=None,
                        scale=alt.Scale(domain=[-vmax, 0, vmax],
                                        range=[cc["short"], pal["surface"], cc["long"]],
                                        clamp=True)),
        tooltip=[alt.Tooltip("year:O"), alt.Tooltip("mon:N", title="Month"),
                 alt.Tooltip("val:Q", title=f"Change ({unit})", format="+,.1f")])
    labels = base.mark_text(fontSize=11.5, font="monospace").encode(
        text="txt:N",
        color=alt.condition(f"abs(datum.val) > {vmax * 0.55}",
                            alt.value("#F2F4F6"), alt.value(pal["text"])))
    brand.show_chart((cells + labels).properties(
        height=max(230, 26 * mat.shape[0] + 40),
        title=f"{universe.yield_name(tkr)} — monthly change ({unit}), year × month"))
    _n = " · *current month to date (excluded from stats)" if meta["partial"] else ""
    st.caption(f"Colour is clamped at ±{vmax:,.1f} {unit} (the 90th percentile of "
               f"|monthly moves|) so one outlier month doesn't wash the map out{_n}.")

    srows = []
    for lbl, k in [("Median", "med"), ("Years up", "hit")]:
        row = {"lbl": lbl}
        for m in range(1, 13):
            v = stats.loc[m, k]
            if pd.isna(v):
                row[seasmon.MONTH_LABELS[m - 1]] = "—"
            elif k == "hit":
                row[seasmon.MONTH_LABELS[m - 1]] = f"{round(v * stats.loc[m, 'n'])}/{int(stats.loc[m, 'n'])}"
            else:
                row[seasmon.MONTH_LABELS[m - 1]] = fmt.format(v)
        srows.append(row)
    brand.terminal_table(srows, [{"key": "lbl", "label": ""}] + [
        {"key": lab, "label": lab, "align": "right"} for lab in seasmon.MONTH_LABELS])

    # average-year path
    spd, sinfo = seasmon.seasonal_path(weekly, tkr)
    if not spd.empty:
        _sx = alt.Chart(spd).encode(
            x=alt.X("wdate:T", title=None,
                    axis=alt.Axis(format="%b", tickCount="month", labelFontSize=12)))
        _band = _sx.mark_area(color=cc["muted"], opacity=0.35).encode(
            y=alt.Y("p25:Q", title=f"cumulative {unit}",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            y2="p75:Q")
        _med = _sx.mark_line(color=cc["series"], strokeWidth=2.6).encode(
            y="med:Q", tooltip=[alt.Tooltip("woy:Q", title="Week"),
                                alt.Tooltip("med:Q", title=f"Median cum {unit}", format="+,.1f")])
        _halo = _sx.mark_line(color=cc["halo"], strokeWidth=4.6).encode(y="current:Q")
        _cur = _sx.mark_line(color=cc["accent"], strokeWidth=3).encode(
            y="current:Q",
            tooltip=[alt.Tooltip("woy:Q", title="Week"),
                     alt.Tooltip("current:Q", title=f"{sinfo['cur_year']} cum {unit}",
                                 format="+,.1f")])
        _zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
            color=cc["muted"], strokeDash=[4, 3], strokeWidth=1).encode(y="y:Q")
        brand.show_chart(alt.layer(_zero, _band, _med, _halo, _cur).properties(
            height=240,
            title=f"Average year — cumulative change from 1 Jan (median of {sinfo['years']}y "
                  f"· band = 25–75% of years · gold = {sinfo['cur_year']})"))

    # best windows
    bw = _seas_windows(tkr, MODE)
    if bw is not None and not bw.empty:
        w0, w1 = st.columns(2)
        for col, direction, head in ((w0, "Higher", "Historically higher windows"),
                                     (w1, "Lower", "Historically lower windows")):
            sub = bw[bw["dir"] == direction]
            with col:
                brand.panel_header(head, right=f"hit ≥ {seasmon.HIT_STRONG:.0%}")
                if sub.empty:
                    st.caption("No calendar window cleared the agreement bar here.")
                    continue
                brand.terminal_table(
                    [{"wspan": _seas_wspan(r["start"], r["weeks"]),
                      "hit": f"{int(r['wins'])}/{int(r['n'])}",
                      "win": r["label"],
                      "dhit": (f"{int(r['date_wins'])}/{int(r['date_n'])}"
                               if int(r.get("date_n", 0) or 0) else "—"),
                      "med": float(r["med"]), "worst": float(r["worst"])}
                     for _, r in sub.iterrows()],
                    [{"key": "win", "label": "Date x → y"},
                     {"key": "dhit", "label": "Hit (dates)", "align": "right"},
                     {"key": "wspan", "label": "Week x → y"},
                     {"key": "hit", "label": "Hit (weeks)", "align": "right"},
                     {"key": "med", "label": f"Med {unit}", "color": True, "fmt": fmt},
                     {"key": "worst", "label": "Worst", "align": "right", "fmt": fmt}])
        st.caption(
            "Every 4–16-week calendar stretch (year-end wrap included) screened for the "
            f"windows this product moved one way in ≥ {seasmon.HIT_STRONG:.0%} of the stored "
            "years; overlapping echoes collapse to the strongest. Each window is scored "
            "twice: **Date x → y / Hit (dates)** replays the fixed calendar dates (the "
            "Bloomberg-SEAG convention); **Week x → y / Hit (weeks)** replays the same "
            "numbered weeks of each year (edges drift a few days year to year). Trust the "
            "windows where the two scores agree. **Worst** = the most adverse single year "
            "inside the window — even a 9-of-10 pattern has an exception. Descriptive "
            "history, not a signal.")

    # ---- seasonal windows board: the Hot Sheet's SEAS radar, in full ---------
    st.divider()
    with st.spinner("Scanning the book's seasonal windows… (first open of the day "
                    "pays the scan; the morning pull normally has it cached)"):
        wb = _seas_open_windows(MODE)
    if wb is not None and not wb.empty:
        n_open = int((wb["status"] == "open").sum())
        brand.panel_header("Seasonal windows — open now & opening soon",
                           right=f"whole book · {n_open} open · hit ≥ {seasmon.HIT_STRONG:.0%}")
        wc0, wc1 = st.columns([1.6, 2.4])
        w_show = wc0.radio("Show", ["Entering now & soon", "Everything open or upcoming"],
                           horizontal=True, key="seas_wb_show", label_visibility="collapsed",
                           help="Entering = windows in their first ~3 weeks or starting within "
                                "the next fortnight (the Hot Sheet's framing). Everything = "
                                "every window currently running or starting within a month.")
        w_hit = wc1.radio("Agreement", ["≥ 80% of years", f"All (≥ {seasmon.HIT_STRONG:.0%})",
                                        "Perfect record"],
                          horizontal=True, key="seas_wb_hit", label_visibility="collapsed")
        if w_show == "Entering now & soon":
            wb = wb[((wb["status"] == "open") & (wb["into"] <= 3)) |
                    ((wb["status"] == "upcoming") & (wb["ahead"] <= 2))]
        if w_hit == "≥ 80% of years":
            wb = wb[wb["hit"] >= 0.80]
        elif w_hit == "Perfect record":
            wb = wb[wb["hit"] >= 0.999]
        if wb.empty:
            st.caption("No window clears these filters right now — widen either control.")
        w_rows = []
        for _, r in wb.iterrows():
            w_rows.append({
                "st": (f"open · wk {int(r['into'])} of {int(r['weeks'])}"
                       if r["status"] == "open" else
                       ("opens next week" if int(r["ahead"]) == 1
                        else f"opens in {int(r['ahead'])}w")),
                "name": r["name"],
                "wspan": _seas_wspan(r["start"], r["weeks"]), "win": r["label"],
                "dir": "↑ higher" if r["dir"] == "Higher" else "↓ lower",
                "hit": f"{int(r['wins'])}/{int(r['n'])}",
                "dhit": (f"{int(r['date_wins'])}/{int(r['date_n'])}"
                         if int(r.get("date_n", 0) or 0) else "—"),
                "med": float(r["med"]), "worst": float(r["worst"]), "unit": r["unit"],
            })
        brand.terminal_table(w_rows, [
            {"key": "st", "label": "Status"},
            {"key": "name", "label": "Product"},
            {"key": "dir", "label": "Direction"},
            {"key": "win", "label": "Date x → y"},
            {"key": "dhit", "label": "Hit (dates)", "align": "right"},
            {"key": "wspan", "label": "Week x → y"},
            {"key": "hit", "label": "Hit (weeks)", "align": "right"},
            {"key": "med", "label": "Med", "color": True, "fmt": "{:+,.1f}"},
            {"key": "worst", "label": "Worst", "align": "right", "fmt": "{:+,.1f}"},
            {"key": "unit", "label": "Unit"},
        ])
        st.caption(
            "**This is the list the Hot Sheet's SEAS stories come from** — calendar windows "
            "across the **whole book** (deliberately ignoring the sector filter above, so a "
            "Hot Sheet story always has its row here). A *window* is a 4–16-week calendar "
            f"stretch this product moved one way in ≥ {seasmon.HIT_STRONG:.0%} of the stored "
            "years (≥ 5 complete years; overlapping echoes collapsed to the strongest — the "
            "same finder that fills the per-product tables under the detail below). "
            "**Med / Worst** = the median and most adverse single-year move over the window, "
            "in the product's own unit (% of price, bp of yield for FI). Every window is "
            "scored twice: **Date x → y / Hit (dates)** replays the fixed calendar dates "
            "(the Bloomberg-SEAG convention); **Week x → y / Hit (weeks)** replays the "
            "same numbered weeks of each year, whose edges drift up to ±6 days against "
            "the calendar. A record that softens badly under fixed dates was riding whatever "
            "the drifting week-edges caught — early-November election weeks, in one live "
            "example. Trust windows where the two scores agree. The left control narrows to "
            "windows just entering (the Hot Sheet's framing) or widens to every window "
            "running; the right one sets the agreement bar. Windows found by searching a "
            "decade of history are descriptive, not a signal.")

        with st.expander("❓ What is a seasonal window — and how do I read one?"):
            st.markdown(
                "**The rule.** For every product the finder tests every possible calendar "
                "stretch — starting any week of the year, lasting 4 to 16 weeks, about 676 "
                "stretches — and asks one question of each year on the store: *did this "
                "stretch finish higher or lower than it started?* If at least "
                f"**{seasmon.HIT_STRONG:.0%} of the years agreed** on the direction (over at "
                "least 5 complete years), it's a window. 6-of-10 is barely better than a "
                "coin flip, so it isn't one. Nearly-identical overlapping stretches collapse "
                "into the single strongest, so one pattern shows once.")
            if not wb.empty:
                _top = wb.iloc[0]
                _wy = seasmon.window_years(weekly, _top["ticker"], int(_top["start"]),
                                           int(_top["weeks"]))
                if not _wy.empty:
                    _wfmt = _seas_fmt(_top["unit"])
                    st.markdown(
                        f"**A live example — {_top['name']}, {_top['label']}** (the "
                        "strongest window on the board right now). The same stretch, "
                        "measured in every stored year:")
                    brand.terminal_table(
                        [{str(int(y)): float(v) for y, v in _wy.items()}],
                        [{"key": str(int(y)), "label": str(int(y)), "color": True,
                          "fmt": _wfmt} for y in _wy.index])
                    st.caption(
                        f"That table **is** the window: {int(_top['wins'])} of "
                        f"{int(_top['n'])} years one way, median "
                        f"{_wfmt.format(_top['med'])}{_top['unit']} — the board's Med column "
                        "is the middle value of exactly these numbers, and Worst is the "
                        "most adverse one.")
            st.markdown(
                "**Why they exist.** For physical commodities the *cause* repeats on the "
                "calendar, so the price pattern does too: natural gas prices the storage "
                "cycle (injection vs withdrawal), RBOB's February collapse is the "
                "winter→summer grade switch written into refinery regulation, grains fade "
                "into harvest, cattle and hogs follow the feedlot cycle. **Financial "
                "products have seasonal patterns too, but flow- and behaviour-driven** — "
                "the *Sell-in-May / Halloween* effect, September's long record as the weak "
                "equity month, year-end rallies, tax-loss and fund year-end flows, index "
                "calendars dominated by dividends and carry. Those mechanisms are real but "
                "weaker than a storage cycle, and a decade of equity drift flatters every "
                "long-side equity window — read them with an extra grain of salt.\n\n"
                "**How a desk uses one.** (1) *Timing an existing intention* — establish "
                "length you wanted anyway ahead of the strong stretch, not into the weak "
                "one. (2) *A yardstick for current price action* — a market rallying "
                "through its seasonally weak window is fighting the tide, which is "
                "information; a rally inside the strong window is partly 'just the "
                "season'. (3) *Risk framing* — same hit rate, different stakes: a window "
                "whose worst year was flat is a different proposition from one whose worst "
                "year lost 20%.\n\n"
                "**The caveat that keeps this honest.** ~676 stretches are tested per "
                "product, so a few 8- or 9-of-10 records will exist by pure luck — the way "
                "someone in a room of 676 coin-flippers flips eight heads. And because the "
                "finder aligns years by week-of-year, a window's real start and end dates "
                "drift up to ±6 days across years — the **Hit (dates)** column re-measures "
                "every window on the fixed calendar dates shown (the Bloomberg-SEAG "
                "convention), and a record that softens badly there was riding whatever the "
                "drifting week-edges caught (a late-Aug Dow window quietly sweeping US "
                "election weeks was the live example). Before reading anything into a window, ask *is there a "
                "story?* — a storage cycle is a story; 'this index went up in most "
                "Octobers' may just be the decade — and trust the windows where both "
                "columns agree. Windows describe history — they promise nothing about "
                "year eleven.")

    # ---- client PDF (2026-08-22: the last module without one) --------------------
    st.divider()
    if st.button("📅 Generate Seasonality Report (visual PDF)", type="primary", key="seas_pdf_btn"):
        with st.spinner("Rendering the Seasonality report…"):
            try:
                def _clean(d: dict) -> dict:
                    return {k: (None if isinstance(v, float) and pd.isna(v) else v)
                            for k, v in d.items()}
                # detail panels: the four most seasonal names in the screened month
                _det = scr.copy()
                _det["_x"] = (_det["hit"] - 0.5).abs()
                _det = _det.sort_values(["_x", "med"], key=lambda c: c.abs() if c.name == "med" else c,
                                        ascending=False).head(4)
                products = []
                for _, r in _det.iterrows():
                    _t = r["ticker"]
                    _mat, _stats, _meta = seasmon.monthly_matrix(monthly, _t)
                    if _mat is None:
                        continue
                    _spd, _sinfo = seasmon.seasonal_path(weekly, _t)
                    _bw = _seas_windows(_t, MODE)
                    products.append({
                        "ticker": _t, "name": r["name"], "unit": _meta["unit"],
                        "years": [int(y) for y in _mat.index],
                        "matrix": [[None if pd.isna(v) else float(v) for v in row]
                                   for row in _mat.to_numpy()],
                        "partial": list(_meta["partial"]) if _meta["partial"] else None,
                        "stats": {str(int(m)): _clean({"med": _stats.loc[m, "med"],
                                                       "hit": _stats.loc[m, "hit"],
                                                       "n": int(_stats.loc[m, "n"]),
                                                       "best": _stats.loc[m, "best"],
                                                       "worst": _stats.loc[m, "worst"]})
                                  for m in range(1, 13)},
                        "path": (None if _spd.empty else {
                            c: [None if pd.isna(v) else float(v) for v in _spd[c]]
                            for c in ("woy", "med", "p25", "p75", "current")}),
                        "path_years": int(_sinfo.get("years", 0)),
                        "cur_year": _sinfo.get("cur_year"),
                        "windows": ([] if _bw is None or _bw.empty else
                                    [_clean(w) for w in _bw.to_dict("records")]),
                    })
                try:
                    # the physical + rate books only, like the spreads page's default:
                    # index / bond / FX calendars are dividend, carry and roll mechanics
                    _sp = seasmon.spread_screener_cached()
                    _sp = _sp[_sp["asset"].isin(("STIRs", "Energy", "Metals", "Agriculture",
                                                 "Softs"))] if not _sp.empty else _sp
                    spreads = [_clean(w) for w in _sp.to_dict("records")] if not _sp.empty else []
                except Exception:
                    spreads = []
                payload = {
                    # the data date = the snapshot's as-of (the month bucket's end date
                    # would read as the 31st of an in-progress month)
                    "asof": str((_load_snap() or {}).get("as_of") or monthly.index.max().date()),
                    "month": int(month),
                    "month_label": mo_sel, "hit_strong": float(seasmon.HIT_STRONG),
                    "seas_z_flag": float(seasmon.SEAS_Z_FLAG),
                    "years": int(max((p["path_years"] for p in products), default=0)) or "~10",
                    "sectors": [a for a in assets if a in set(scr["asset"])],
                    "screener": [_clean(r) for r in scr.to_dict("records")],
                    "products": products, "spreads": spreads,
                }
                with tempfile.TemporaryDirectory() as _t:
                    _in = Path(_t) / "seas.json"
                    _out = Path(_t) / "Seasonality_Monitor.pdf"
                    _in.write_text(json.dumps(payload), encoding="utf-8")
                    r = subprocess.run(
                        [sys.executable, str(SEASREPORT_CLI), str(_in), str(_out)],
                        capture_output=True, text=True, timeout=240)
                    if r.returncode == 0 and _out.exists():
                        st.session_state["seas_pdf"] = _out.read_bytes()
                    else:
                        st.error("Report failed:\n\n" + (r.stderr or r.stdout or "unknown error")[-2000:])
            except Exception as e:
                st.error(f"Report failed: {e}")
    if st.session_state.get("seas_pdf"):
        st.download_button("⬇️  Download Seasonality Report", data=st.session_state["seas_pdf"],
                           file_name="Seasonality_Monitor.pdf", mime="application/pdf",
                           key="seas_pdf_dl")
        email_report_ui("seas_email", "seasonality", st.session_state["seas_pdf"],
                        subject="BASIS — Seasonality Monitor",
                        attachment_name="Seasonality_Monitor.pdf")


def render_seasonality_spreads() -> None:
    import altair as alt

    st.subheader("🔀  Spread Seasonality — the front calendar spread through the year")
    st.caption(
        "For every product with a stored second generic: the **front calendar spread** — the "
        "1st futures contract minus the 2nd (**XB1 − XB2**, **NG1 − NG2**, …), both **actual "
        "traded settles**, never a cash/spot leg — walked through the calendar year. Positive "
        "= backwardation (prompt over the next month), negative = contango; STIR calendars in "
        "bp. The contract pair changes as the year rolls forward — that is the point: this is "
        "the shape the front of the curve typically takes each season (injection vs "
        "withdrawal, harvest vs old-crop).")

    sps = _seas_spread_products(MODE)
    if not sps:
        st.info("No stored second-generic history yet — the deep price store hasn't been "
                "built on this machine (it backfills on the next Bloomberg session).")
        return

    # ---- spread screener: stretch vs season + seasonality strength ----------
    scr = _seas_spread_screener(MODE)
    top_stretch = None
    if scr is not None and not scr.empty:
        _sc0, _sc1 = st.columns([2.6, 1.2], vertical_alignment="bottom")
        _spa = [a for a in universe.ASSET_CLASSES if a in set(scr["asset"])]
        # dividend / carry / roll mechanics dominate index, bond and FX calendars —
        # real seasonality lives in the physical + rate books, so those are the default
        _phys = [a for a in _spa if a in ("STIRs", "Energy", "Metals", "Agriculture", "Softs")]
        sp_assets = _sc0.multiselect("Sectors", _spa, default=_phys or _spa,
                                     key="seas_sp_assets")
        sp_sort = _sc1.radio("Rank by", ["Stretched now", "Most seasonal"], horizontal=True,
                             key="seas_sp_sort", label_visibility="collapsed")
        view = scr[scr["asset"].isin(sp_assets)] if sp_assets else scr
        if sp_sort == "Most seasonal":
            view = view.sort_values("strength", ascending=False)
        if not view.empty:
            flagged = view[view["signal"] != "—"].head(6)
            if not flagged.empty:
                st.markdown("**Stretched vs season now:** " + " · ".join(
                    f"{r['name']} **{r['z']:+.1f}σ** "
                    f"({'rich' if r['z'] > 0 else 'cheap'} for this time of year)"
                    for _, r in flagged.iterrows()))
            most = view.sort_values("strength", ascending=False).head(4)
            st.markdown("**Most seasonal books:** " + " · ".join(
                f"{r['name']} **{r['strength']:.0f}%** (peak {r['peak']} · trough {r['trough']})"
                for _, r in most.iterrows()))
            brand.panel_header("Spread screener",
                               right=f"{len(view)} spreads · vs this week's 10y norm")
            rows = []
            for _, r in view.iterrows():
                rows.append({
                    "name": r["name"], "legs": r["legs"],
                    "now": f"{r['now']:,.2f} {r['unit']}", "norm": f"{r['norm']:,.2f}",
                    "dev": float(r["dev"]), "zpic": 0.0 if pd.isna(r["z"]) else float(r["z"]),
                    "z": None if pd.isna(r["z"]) else float(r["z"]),
                    "ssn": None if pd.isna(r["strength"]) else float(r["strength"]),
                    "peak": r["peak"], "trough": r["trough"],
                })
            brand.terminal_table(rows, [
                {"key": "name", "label": "Product"},
                {"key": "legs", "label": "Spread"},
                {"key": "now", "label": "Now", "align": "right"},
                {"key": "norm", "label": "Wk norm", "align": "right"},
                {"key": "dev", "label": "Δ vs norm", "color": True, "fmt": "{:+,.2f}"},
                {"key": "zpic", "label": "±2σ", "zbar": True},
                {"key": "z", "label": "Seas z", "align": "right", "fmt": "{:+.1f}"},
                {"key": "ssn", "label": "Ssn %", "align": "right", "fmt": "{:.0f}"},
                {"key": "peak", "label": "Peak", "align": "right"},
                {"key": "trough", "label": "Trough", "align": "right"},
            ])
            st.caption(
                "**Wk norm / Seas z** — today's spread against the SAME week of year across "
                "the stored years (±1 week to absorb roll-date drift, current year excluded "
                f"from its own norm); |z| ≥ {seasmon.SEAS_Z_FLAG:g}σ flags rich/cheap **for "
                "this time of year** — the unconditional read is the Curve / RV monitor's "
                "job. **Ssn %** — how much of the spread's behaviour is calendar: typical "
                "seasonal swing vs year-to-year noise at the same week (robust to blowup "
                "years). **Peak / Trough** — the months the profile typically tops (most "
                "backwardated) and bottoms (deepest contango): NG peaks in winter withdrawal, "
                "troughs in injection season. Index / bond / FX calendars are one click away "
                "above, but their spreads are mostly dividend, carry and roll mechanics — "
                "read their rows with that in mind. Descriptive history, not a signal.")
            top_stretch = view.iloc[0]["ticker"]

    st.divider()
    brand.panel_header("Front calendar spread", right="1st − 2nd futures · actual settles")
    _last = top_stretch or st.session_state.get("seas_tkr")
    sp_default = _last if _last in sps else ("NGA Comdty" if "NGA Comdty" in sps else sps[0])
    sp_tkr = st.selectbox("Spread product", sps, index=sps.index(sp_default),
                          format_func=lambda t: f"{universe.name(t)}  ·  "
                                                f"{seasmon.spread_label(t)}",
                          key="seas_sp", label_visibility="collapsed")
    legs = seasmon.spread_label(sp_tkr)
    spdf, spinfo = _seas_spread(sp_tkr, MODE)
    if spdf.empty:
        st.info("Not enough stored second-generic history for this product.")
        return
    su = spinfo["unit"]
    cc = brand.chart_colors()
    _sx = alt.Chart(spdf).encode(
        x=alt.X("wdate:T", title=None,
                axis=alt.Axis(format="%b", tickCount="month", labelFontSize=12)))
    _band = _sx.mark_area(color=cc["muted"], opacity=0.35).encode(
        y=alt.Y("p25:Q", title=f"{legs} ({su})",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13)), y2="p75:Q")
    _med = _sx.mark_line(color=cc["series"], strokeWidth=2.6).encode(
        y="med:Q", tooltip=[alt.Tooltip("woy:Q", title="Week"),
                            alt.Tooltip("med:Q", title=f"Median {legs} ({su})", format="+,.2f")])
    _halo = _sx.mark_line(color=cc["halo"], strokeWidth=4.6).encode(y="current:Q")
    _cur = _sx.mark_line(color=cc["accent"], strokeWidth=3).encode(
        y="current:Q", tooltip=[alt.Tooltip("woy:Q", title="Week"),
                                alt.Tooltip("current:Q", format="+,.2f",
                                            title=f"{spinfo['cur_year']} {legs} ({su})")])
    _zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=cc["muted"], strokeDash=[4, 3], strokeWidth=1).encode(y="y:Q")
    brand.show_chart(alt.layer(_zero, _band, _med, _halo, _cur).properties(
        height=280,
        title=f"{universe.name(sp_tkr)} — {legs} front calendar spread by week of year "
              f"(median {spinfo['years']}y · band = 25–75% · gold = {spinfo['cur_year']})"))
    st.caption(
        f"**{legs}** = the front futures settle minus the second — with September front, the "
        "Sep contract minus Oct; next month the pair itself rolls forward. **Blue** = the "
        "median across the stored years, **grey band** = 25–75% of years, **gold** = the "
        "current year so far. Around roll points the band naturally widens (exact roll dates "
        "drift a few days year to year). Descriptive history, not a signal.")


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

# ══════════════════════════════════════════════════════════════════════════════════════
# Macro Rate Radar — policy rules vs what the curve prices
#   (engines: src/macrodata.py · src/macrorules.py · src/macroradar.py ·
#             src/macrosurprise.py · src/macrobt.py — all free data, no Bloomberg)
# ══════════════════════════════════════════════════════════════════════════════════════
_RADAR_PREFS = ROOT / "data" / "macroradar.json"

_RADAR_RULES = [("balanced", "Balanced approach", macrorules.balanced),
                ("taylor93", "Taylor (1993)", macrorules.taylor93),
                ("shortfalls", "Shortfalls", macrorules.shortfalls),
                ("inertial", "Inertial", macrorules.inertial),
                ("firstdiff", "First difference", macrorules.first_difference)]
_RADAR_RULE_FN = {k: fn for k, _n, fn in _RADAR_RULES}

# Flag chips for the bank buttons — same inline-SVG ::before trick as the STIR Paths
# tabs (_STIR_TAB_FLAG_CSS), reusing its _STIR_FLAG_B64 assets. Emoji flags are not an
# option: Windows ships no flag glyphs, so 🇺🇸 renders as the letters "US". Keep the
# closing brace OUT of the f-string segments — a doubled }} inside one stays literal
# and silently kills every rule after the first (see the note on _STIR_TAB_FLAG_CSS).
_RADAR_TAB_FLAG_CSS = "".join(
    f".st-key-radar_bk_{bk} button p::before {{"
    "content:''; display:inline-block; width:19px; height:12.5px;"
    f"background:url(data:image/svg+xml;base64,{_STIR_FLAG_B64[bk]}) center/cover;"
    "margin-right:8px; border-radius:2px; vertical-align:-1.5px;"
    "box-shadow:0 0 0 1px rgba(255,255,255,0.22);}"
    for bk in ("FED", "ECB", "BOE"))


def _radar_prefs() -> dict:
    try:
        return json.loads(_RADAR_PREFS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _radar_save_prefs(blob: dict) -> None:
    try:
        _RADAR_PREFS.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    except Exception:
        pass


def render_macro_radar() -> None:
    """Policy-rule dashboard: what the macro says the policy rate should be, against what
    the strip has already priced. Every input here is free public data — no Bloomberg."""
    prefs = _radar_prefs()

    st.subheader("🏛️  Macro Rate Radar — policy rules vs what's priced")
    st.caption(
        "The formula rates desks argue about: **i = r\\* + π + 0.5(π − π\\*) + b·gap**. "
        "Five rules, three central banks, run on public data only — FRED/ALFRED, Eurostat, "
        "the ECB, the BoE, the ONS and the regional Feds. The number that matters is the "
        "**spread against what the curve already prices** (from the STIR Paths fit), not "
        "the rule level on its own.")

    st.warning(
        "**Policy rules are prescriptive, not predictive.** No central bank follows one "
        "mechanically, and the gap between prescription and actual policy routinely runs "
        "100bp+ for years. What tends to carry information is the **change** in the "
        "prescription, the **spread versus priced**, and the **dispersion** across rules — "
        "not the level.", icon="⚠️")

    with st.expander("📖  What the symbols mean", expanded=False):
        st.markdown(
            "| Symbol | Meaning | Where it comes from |\n"
            "|---|---|---|\n"
            "| **i** | The prescribed policy rate — what the rule says the central bank "
            "should set | The rule's output |\n"
            "| **r\\*** | The real neutral rate: the inflation-adjusted rate that neither "
            "stimulates nor restrains the economy | Holston-Laubach-Williams estimate "
            "(Fed/ECB); an editable assumption for the BoE |\n"
            "| **π** | Inflation now — core, year-on-year | Core CPI/PCE, HICP ex energy "
            "& food, or UK core CPI |\n"
            "| **π\\*** | The inflation target | 2% at all three banks |\n"
            "| **π − π\\*** | The inflation gap: how far inflation sits from target | "
            "Computed |\n"
            "| **u** | The unemployment rate now | Latest official print |\n"
            "| **u\\*** | The natural rate of unemployment (NAIRU): the rate consistent "
            "with stable inflation | CBO estimate (US); an editable assumption for the "
            "ECB and BoE |\n"
            "| **u gap** | u\\* − u, the labour-market gap. Positive = the labour market "
            "is running hot; each point ≈ 2 points of output gap (Okun's law) | "
            "Computed |\n"
            "| **b·gap** | The real-economy term: the u gap times the rule's weight — "
            "1.0 in Taylor (1993), 2.0 in the balanced approach, 2.0 but floored at "
            "zero in the shortfalls variant | Computed |\n"
            "| **i₋₁** | Last period's policy setting, which the inertial rule weights "
            "at 85% and the first-difference rule steps from | Actual policy rate |\n"
            "| **Δgap** | The change in the u gap versus four quarters ago — the "
            "first-difference rule reacts to direction, and r\\* cancels out of it "
            "entirely | Computed |")

    # Flags via the STIR Paths CSS trick, NOT emoji: Windows has no flag emoji font —
    # 🇺🇸 degrades to the letters "US" on every Windows box (Ben hit this on day one).
    bank_lbl = {"FED": "Fed", "ECB": "ECB", "BOE": "BoE"}
    st.markdown(f"<style>{_RADAR_TAB_FLAG_CSS}</style>", unsafe_allow_html=True)
    bcols = st.columns(3)
    bank = st.session_state.setdefault("radar_bank", prefs.get("bank", "FED"))
    for col, bk in zip(bcols, ("FED", "ECB", "BOE")):
        if col.button(bank_lbl[bk], use_container_width=True, key=f"radar_bk_{bk}",
                      type="primary" if bank == bk else "secondary"):
            st.session_state["radar_bank"] = bank = bk
            st.rerun()

    rule_names = [n for _k, n, _f in _RADAR_RULES]
    _ALL_PICK = "All rules — overview"
    default_rule = prefs.get("rule", "balanced")
    _rule_keys = [k for k, _n, _f in _RADAR_RULES]
    _def_idx = (len(rule_names) if default_rule == "all"
                else _rule_keys.index(default_rule) if default_rule in _rule_keys else 0)
    pick = st.radio("Rule", rule_names + [_ALL_PICK], horizontal=True,
                    index=_def_idx, key="radar_rule_pick")
    all_mode = pick == _ALL_PICK
    # all-mode still needs one rule for compare(); balanced is the house baseline
    rule_key = "all" if all_mode else [k for k, n, _f in _RADAR_RULES if n == pick][0]
    rule_fn = _RADAR_RULE_FN["balanced" if all_mode else rule_key]

    # ---- assumption inputs (r* / NAIRU) — editable because for some blocs nobody publishes them
    saved = prefs.get("overrides", {}).get(bank, {})
    with st.expander("⚙️  Assumptions — r\\* and the natural rate of unemployment", expanded=False):
        st.caption(
            "Where a bloc publishes these, the published value is used and these boxes stay "
            "empty. **The BoE leg has neither**: there is no UK estimate in Holston-Laubach-"
            "Williams and the ONS publishes no potential output, so both numbers below are "
            "assumptions, and the BoE prescription moves roughly one-for-one with r\\*.")
        ac1, ac2, ac3 = st.columns([1, 1, 1])
        use_override = ac1.checkbox("Override", value=bool(saved),
                                    key=f"radar_ovr_{bank}")
        rstar_in = ac2.number_input("r\\* (real neutral, %)", value=float(saved.get("rstar", 0.75)),
                                    step=0.05, format="%.2f", key=f"radar_rs_{bank}",
                                    disabled=not use_override)
        nairu_in = ac3.number_input("NAIRU / u\\* (%)", value=float(saved.get("nairu", 4.25)),
                                    step=0.05, format="%.2f", key=f"radar_nu_{bank}",
                                    disabled=not use_override)
        if st.button("💾 Save as this bank's default", key=f"radar_save_{bank}"):
            blob = _radar_prefs()
            blob.setdefault("overrides", {})[bank] = (
                {"rstar": rstar_in, "nairu": nairu_in} if use_override else {})
            blob["bank"], blob["rule"] = bank, rule_key
            _radar_save_prefs(blob)
            st.success("Saved.")

    ov_rstar = rstar_in if use_override else None
    ov_nairu = nairu_in if use_override else None

    # ---- scenario sliders --------------------------------------------------------------
    with st.expander("🎛️  Scenario — shift the macro and watch the prescribed path move",
                     expanded=False):
        st.caption(
            "The forward path assumes bland mean reversion — inflation decays toward target, "
            "the gap closes — rather than a house forecast, so what you see is a transparent "
            "baseline you can inspect, not a view smuggled in. Shocks below are added to "
            "today's readings.")
        s1, s2, s3 = st.columns(3)
        infl_shock = s1.slider("Inflation shock (pp)", -2.0, 2.0, 0.0, 0.1, key="radar_sh_i")
        gap_shock = s2.slider("Output-gap shock (pp)", -4.0, 4.0, 0.0, 0.1, key="radar_sh_g")
        rstar_shift = s3.slider("r\\* shift (pp)", -1.0, 1.0, 0.0, 0.05, key="radar_sh_r")
        h1, h2 = st.columns(2)
        infl_hl = h1.slider("Inflation half-life (quarters)", 1.0, 16.0, 6.0, 0.5,
                            key="radar_hl_i")
        gap_hl = h2.slider("Gap half-life (quarters)", 1.0, 16.0, 8.0, 0.5, key="radar_hl_g")
    assume = macrorules.PathAssumption(infl_half_life_q=infl_hl, gap_half_life_q=gap_hl,
                                       infl_shock=infl_shock, gap_shock=gap_shock,
                                       rstar_shift=rstar_shift)

    # ---- compute ------------------------------------------------------------------------
    with st.spinner("Pulling free macro data…"):
        try:
            res = macroradar.compare(bank, rule=rule_fn, nairu=ov_nairu, rstar=ov_rstar,
                                     assume=assume)
        except Exception as e:
            st.error(f"Could not build the Radar for {bank}: {e}")
            return

    if res.summary is None:
        st.error("No rule could be evaluated — the macro inputs are unavailable. "
                 "Check the Data health board.")
        return

    x, prov = macrorules.inputs_from_data(bank, nairu=ov_nairu, rstar=ov_rstar)

    # ---- headline row -------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Policy rate now", f"{res.policy_now:.2f}%")
    if all_mode:
        m2.metric("Rule range",
                  "—" if res.summary.lo is None
                  else f"{res.summary.lo:.2f}–{res.summary.hi:.2f}%",
                  help="Lowest to highest of the five prescriptions today.")
    else:
        m2.metric(f"{res.rule_name}",
                  "—" if res.prescribed_now is None else f"{res.prescribed_now:.2f}%",
                  None if res.prescribed_now is None
                  else f"{(res.prescribed_now - res.policy_now) * 100:+.0f}bp vs policy",
                  delta_color="off")   # hawkish/dovish is a direction, not good/bad news
    m3.metric("Median of 5 rules",
              "—" if res.summary.median is None else f"{res.summary.median:.2f}%",
              None if res.summary.median_gap_bp is None
              else f"{res.summary.median_gap_bp:+.0f}bp vs policy",
              delta_color="off")
    # Dispersion context: percentile/z against the ALFRED vintage history (US only).
    # Compared 4-rule vs 4-rule — the vintage store cannot evaluate first-difference,
    # so today's spread is recomputed without it before standardising.
    disp_ctx = disp4 = None
    if bank == "FED":
        try:
            from src import macrobt as _mbt
            _core = [r.prescribed for r in res.summary.results
                     if r.ok and r.prescribed is not None and r.key != "firstdiff"]
            if len(_core) >= 2:
                disp4 = (max(_core) - min(_core)) * 100.0
                disp_ctx = _mbt.dispersion_context(disp4)
        except Exception:
            disp_ctx = None
    m4.metric("Rule dispersion",
              "—" if res.summary.dispersion_bp is None else f"{res.summary.dispersion_bp:.0f}bp",
              None if disp_ctx is None or disp_ctx.get("z") is None
              else f"{disp_ctx['pct']:.0f}th %ile · z {disp_ctx['z']:+.1f}",
              delta_color="off",
              help="Highest minus lowest prescription. Wide = the committee has genuine "
                   "latitude and the outcome distribution is fat — an options view more "
                   "than a directional one."
                   + ("" if disp_ctx is None else
                      f"\n\n**Is that high?** Against the same four rules rebuilt from "
                      f"US vintage data {disp_ctx['start'].year}–{disp_ctx['end'].year} "
                      f"(first-difference excluded — no year-ago gap in the store): "
                      f"roughly **under {disp_ctx['q1']:.0f}bp the rules speak with one "
                      f"voice, over {disp_ctx['q3']:.0f}bp they genuinely disagree**; "
                      f"the all-time median is {disp_ctx['median']:.0f}bp."
                      + (("\n\nFor feel, the era medians: "
                          + " · ".join(f"{lbl} ~{v:.0f}bp" for lbl, v in disp_ctx["eras"])
                          + f". The COVID/inflation shock peaked at "
                            f"{disp_ctx['hi']:,.0f}bp — that is what \"the formulas "
                            f"have no idea\" looks like. The distribution is heavily "
                            f"right-skewed, so trust the percentile over the z-score "
                            f"on big readings.")
                         if disp_ctx.get("eras") else "")
                      + " Context is US-only — no free vintage archive exists for the "
                        "ECB or BoE."))

    if prov.assumed:
        st.info(f"**Assumed, not measured:** {', '.join(prov.assumed)} — nobody publishes "
                f"{'these' if len(prov.assumed) > 1 else 'this'} for the "
                f"{bank_lbl[bank].split()[-1]}. Set them under *Assumptions* above; the "
                f"prescription moves with them.", icon="✏️")
    if prov.stale:
        st.caption(f"⏳ Stale source: {', '.join(prov.stale)} — r\\* comes from "
                   f"Holston-Laubach-Williams, which publishes about two quarters behind.")
    if prov.missing:
        st.warning(f"Missing inputs: {', '.join(prov.missing)}", icon="⚠️")

    # ---- the five rules -----------------------------------------------------------------
    st.markdown("#### The five rules")
    rule_rows = []
    for r in res.summary.results:
        rule_rows.append({
            "Rule": r.name,
            "Prescribed": "—" if r.prescribed is None else f"{r.prescribed:.2f}%",
            "vs policy": ("—" if r.prescribed is None
                          else f"{r.vs_actual(res.policy_now):+.0f}bp"),
            "Working": r.formula or r.reason,
            "Note": r.note,
        })
    st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)
    st.caption(res.summary.verdict)

    # ---- prescribed vs priced -----------------------------------------------------------
    st.markdown("#### Prescribed vs priced")
    if not res.ok or not res.meetings:
        st.warning(f"No market-implied path available: {res.reason}", icon="⚠️")
    elif all_mode:
        st.caption(
            f"All five rule paths against the market path from the STIR Paths fit of the "
            f"live strip (store as-of **{res.strip_asof}**). Positive spread = the rule "
            f"wants a **higher** policy rate than the curve has priced.")
        meeting_ds = [m.meeting for m in res.meetings]
        paths = {}
        for _k, _n, _f in _RADAR_RULES:
            try:
                paths[_n] = dict(macrorules.prescribed_path(x, meeting_ds, rule=_f,
                                                            assume=assume,
                                                            start=res.asof))
            except Exception:
                paths[_n] = {}

        try:
            import altair as alt
            chart_rows = [{"meeting": m.meeting.isoformat(), "rate": m.priced_policy,
                           "series": "Priced by the STIR Strip"} for m in res.meetings]
            for _n, pth in paths.items():
                for m in res.meetings:
                    p = pth.get(m.meeting)
                    if p is not None:
                        chart_rows.append({"meeting": m.meeting.isoformat(),
                                           "rate": p, "series": _n})
            cdf = pd.DataFrame(chart_rows)
            dom = ["Priced by the STIR Strip"] + list(paths)
            rng = ["#F5C518", "#64B5F6", "#BA68C8", "#4DB6AC", "#FF8A65", "#F06292"]
            # ticks/grid on the meeting dates themselves — the axis is linear calendar
            # time, and anchoring the labels to the meetings makes the uneven 6-7 week
            # FOMC spacing readable instead of leaving arbitrary fortnightly ticks
            _mticks = [m.meeting.isoformat() for m in res.meetings]
            base = alt.Chart(cdf).encode(
                x=alt.X("meeting:T", title=None,
                        axis=alt.Axis(values=_mticks, format="%d %b %y",
                                      labelAngle=-40, grid=True,
                                      gridOpacity=0.25, gridDash=[2, 3])),
                y=alt.Y("rate:Q", title="Policy rate (%)", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N",
                                scale=alt.Scale(domain=dom, range=rng[:len(dom)]),
                                legend=alt.Legend(title=None, orient="top", labelLimit=0,
                                                  columns=3)),
                # the strip is the reference — draw it heavier than the rule paths
                size=alt.condition(alt.datum.series == "Priced by the STIR Strip",
                                   alt.value(3.5), alt.value(1.6)),
                tooltip=[alt.Tooltip("meeting:T", title="Meeting"),
                         alt.Tooltip("series:N", title=""),
                         alt.Tooltip("rate:Q", title="Rate", format=".2f")])
            lines = base.mark_line(interpolate="step-after")
            today_rule = alt.Chart(pd.DataFrame([{"y": res.policy_now}])).mark_rule(
                strokeDash=[4, 3], color="#9AA4B0").encode(y="y:Q")
            st.altair_chart((lines + today_rule).properties(height=320),
                            use_container_width=True)
        except Exception:
            pass

        # per-meeting spread of every rule against the strip
        sp_rows = []
        for m in res.meetings:
            row = {"Meeting": m.meeting.strftime("%d %b %Y"),
                   "Priced policy": f"{m.priced_policy:.3f}%"}
            for _n, pth in paths.items():
                p = pth.get(m.meeting)
                row[_n] = "—" if p is None else f"{(p - m.priced_policy) * 100:+.0f}bp"
            sp_rows.append(row)
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

        # opportunity read: meetings where every evaluable rule sits the same side of
        # the strip by 25bp+ — dispersion collapsing onto one side is the interesting
        # state; observation language only (house compliance rule)
        agree = []
        for m in res.meetings:
            sps = [(pth[m.meeting] - m.priced_policy) * 100
                   for pth in paths.values() if m.meeting in pth]
            if len(sps) >= 4 and (all(s >= 25 for s in sps) or
                                  all(s <= -25 for s in sps)):
                agree.append((m, min(sps, key=abs), max(sps, key=abs)))
        if agree:
            m, lo_s, hi_s = max(agree, key=lambda t: abs(t[1]))
            side = "above" if lo_s > 0 else "below"
            st.info(
                f"**All rules on one side at {len(agree)} "
                f"meeting{'s' if len(agree) > 1 else ''}.** The strongest case is the "
                f"{m.meeting:%d %b %Y} meeting, where every evaluable rule sits "
                f"{abs(lo_s):.0f}–{abs(hi_s):.0f}bp {side} the strip — even the most "
                f"conservative prescription disagrees with pricing there, which may be "
                f"worth a closer look. Pick that rule above to price the trade in the "
                f"contract-edge table.", icon="📌")
        else:
            st.caption(
                "No meeting where all five rules sit on the same side of the strip by "
                "25bp or more — the rule set does not make a unified case against "
                "pricing here. Disagreement between the rules themselves is a "
                "dispersion (options) observation, not a directional one.")
    else:
        st.caption(
            f"Market path from the STIR Paths fit of the live strip "
            f"(store as-of **{res.strip_asof}**). Positive spread = the rules want a "
            f"**higher** policy rate than the curve has priced.")
        rows = []
        for m in res.meetings:
            rows.append({
                "Meeting": m.meeting.strftime("%d %b %Y"),
                "Priced policy": f"{m.priced_policy:.3f}%",
                "Priced cumulative": f"{m.priced_cum_bp:+.1f}bp",
                "Rule prescribes": "—" if m.prescribed is None else f"{m.prescribed:.2f}%",
                "Spread": "—" if m.spread_bp is None else f"{m.spread_bp:+.0f}bp",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        try:
            import altair as alt
            chart_rows = []
            for m in res.meetings:
                chart_rows.append({"meeting": m.meeting.isoformat(),
                                   "rate": m.priced_policy, "series": "Priced by the STIR Strip"})
                if m.prescribed is not None:
                    chart_rows.append({"meeting": m.meeting.isoformat(),
                                       "rate": m.prescribed,
                                       "series": f"{res.rule_name} prescribes"})
            cdf = pd.DataFrame(chart_rows)
            dom = ["Priced by the STIR Strip", f"{res.rule_name} prescribes"]
            rng = ["#F5C518", "#64B5F6"]
            _mticks = [m.meeting.isoformat() for m in res.meetings]
            base = alt.Chart(cdf).encode(
                x=alt.X("meeting:T", title=None,
                        axis=alt.Axis(values=_mticks, format="%d %b %y",
                                      labelAngle=-40, grid=True,
                                      gridOpacity=0.25, gridDash=[2, 3])),
                y=alt.Y("rate:Q", title="Policy rate (%)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng),
                                legend=alt.Legend(title=None, orient="top",
                                                  labelLimit=0)),
                tooltip=[alt.Tooltip("meeting:T", title="Meeting"),
                         alt.Tooltip("series:N", title=""),
                         alt.Tooltip("rate:Q", title="Rate", format=".2f")])
            lines = base.mark_line(interpolate="step-after", point=True)
            today_rule = alt.Chart(pd.DataFrame([{"y": res.policy_now}])).mark_rule(
                strokeDash=[4, 3], color="#9AA4B0").encode(y="y:Q")
            st.altair_chart((lines + today_rule).properties(height=280),
                            use_container_width=True)
        except Exception:
            pass

        md = res.max_divergence
        if md is not None:
            st.info(f"**Widest disagreement: {md.spread_bp:+.0f}bp at the "
                    f"{md.meeting:%d %b %Y} meeting** — the curve prices "
                    f"{md.priced_policy:.2f}% there against a rule prescription of "
                    f"{md.prescribed:.2f}%.", icon="📌")

    # ---- contract edge -------------------------------------------------------------------
    with st.expander("💷  If the rule path is right — contract edge and P&L per lot",
                     expanded=False):
        if all_mode:
            st.caption(
                "The edge table prices the strip off ONE prescribed path — pick a single "
                "rule above to see it. In the overview the spread table serves the same "
                "purpose per rule.")
            edges = []
        else:
            st.caption(
                "Every contract in the bank's strip repriced off the rule path. **Signs are "
                "for a long position:** a negative edge means the contract is rich to the "
                "rule (the rule wants higher rates than the curve) so the future should "
                "fall. This is a sizing aid, not a forecast — it inherits every assumption "
                "above.")
            try:
                edges = macroradar.contract_edges(bank, rule=rule_fn, nairu=ov_nairu,
                                                  rstar=ov_rstar, assume=assume, n=8)
            except Exception as e:
                edges = []
                st.warning(f"Could not price the strip: {e}")
        if edges:
            st.dataframe(pd.DataFrame([{
                "Contract": e.code, "Product": e.short,
                "Market": f"{e.market_price:.3f}", "Rule fair": f"{e.rule_fair:.3f}",
                "Edge": f"{e.edge_bp:+.1f}bp",
                "P&L / lot (long)": f"{e.ccy}{e.pnl_per_lot:,.0f}",
            } for e in edges]), use_container_width=True, hide_index=True)
            if st.button("🧰 Model in Strategy Builder", key="radar_to_sb"):
                _go("Strategy Builder")

    # ---- surprise index -------------------------------------------------------------------
    st.markdown("#### Economic surprise index")
    # Opportunistic top-up (once per day per session): the ledger cannot be backfilled, so
    # if the morning snapshot ever dies before its macrosurprise block (e.g. a wedged
    # Bloomberg pull upstream), merely opening this page still captures the week's prints.
    _sk = f"radar_surprise_refreshed_{date.today().isoformat()}"
    if not st.session_state.get(_sk):
        try:
            macrosurprise.refresh()
        except Exception:
            pass
        st.session_state[_sk] = True
    ready = macrosurprise.readiness()
    idx = macrosurprise.index(bank)
    if idx.get("ok"):
        s1, s2, s3 = st.columns(3)
        s1.metric("Surprise index", f"{idx['value']:+.2f}",
                  help="Exponentially decayed sum of standardised (actual − consensus) "
                       "surprises. Positive = data beating expectations.")
        gi = macrosurprise.index(bank, "growth")
        ii = macrosurprise.index(bank, "inflation")
        s2.metric("Growth", "—" if not gi.get("ok") else f"{gi['value']:+.2f}")
        s3.metric("Inflation", "—" if not ii.get("ok") else f"{ii['value']:+.2f}")
        rec = macrosurprise.recent(bank, 12)
        if rec:
            st.dataframe(pd.DataFrame([{
                "Date": r["when"], "Release": r["title"],
                "Actual": r["actual"], "Consensus": r["forecast"],
                "Surprise (σ)": "—" if r["z"] is None else f"{r['z']:+.2f}",
            } for r in rec]), use_container_width=True, hide_index=True)
    else:
        st.info(
            f"**Still accruing** — {ready['total']} releases recorded so far"
            + (f" since {ready['first_event']}" if ready["first_event"] else "")
            + f". {idx.get('reason', '')}\n\n"
            "This index cannot be backfilled: the free calendar feed carries the current "
            "week only, and no free source has a history of consensus forecasts. It fills "
            "as the daily refresh runs, and stays silent rather than showing a confident "
            "number built on a handful of points.", icon="⏳")
    for g in macrosurprise.gaps():
        st.caption(f"⚠️ Recording gap: {g['from']} → {g['to']} ({g['days']} days) — the "
                   f"index understates that stretch.")

    # ---- vintage backtest (stored result — a cold run is ~an hour of ALFRED calls) ----
    st.markdown("#### Does the rule gap predict anything? — real-time backtest")
    try:
        from src import macrobt
        bt = macrobt.stored_summary()
    except Exception:
        bt = None
    if bt and bt.get("ok"):
        st.caption(
            f"US only (ALFRED is the only free vintage archive). {bt['n_obs']} monthly "
            f"observations, {bt['first']} → {bt['last']}, each rebuilt from the data "
            f"**as it stood on the day** — revisions and publication lags included. "
            f"Last run {bt.get('ran', '—')} — refreshed monthly by the 'BASIS Macro "
            f"Backtest Refresh' task (3rd, 08:00).")
        # The refresh task is fire-and-forget, so its silent death would leave this
        # section quietly presenting old results as current. Say so instead.
        try:
            _bt_age = (date.today() - date.fromisoformat(bt["ran"])).days
            if _bt_age > 45:
                st.warning(f"This backtest is {_bt_age} days old — the monthly refresh "
                           f"task appears not to have run. Check 'BASIS Macro Backtest "
                           f"Refresh' in Task Scheduler, or run `python -m src.macrobt`.",
                           icon="⏳")
        except Exception:
            pass
        _rule_names = {k: n for k, n, _f in _RADAR_RULES}
        sel = st.selectbox("Rule tested", list(bt.get("analyses", {}).keys()) or ["balanced"],
                           format_func=lambda k: _rule_names.get(k, k),
                           key="radar_bt_rule")
        a = bt.get("analyses", {}).get(sel) or bt.get("analysis", {})
        rows = []
        for h, s in sorted(a.get("horizons", {}).items(), key=lambda kv: int(kv[0])):
            rows.append({
                "Horizon": f"{h}m",
                "Correlation (gap → move)": ("—" if s.get("corr") is None
                                             else f"{s['corr']:+.2f}"),
                "Direction hit rate": ("—" if s.get("hit_rate") is None
                                       else f"{s['hit_rate']:.0%} of {s['n_moved']} moves"),
                "Mean move": f"{s.get('mean_move_bp', 0):+.0f}bp",
                "Mean move when |gap|>100bp": ("—" if s.get("mean_move_when_wide_bp") is None
                                               else f"{s['mean_move_when_wide_bp']:+.0f}bp "
                                                    f"(n={s.get('n_wide', 0)})"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(macrobt.verdict(a) if a else "")
    else:
        st.info("No backtest stored yet — run `python -m src.macrobt` once (about an "
                "hour cold; minutes thereafter). It measures, on point-in-time ALFRED "
                "vintages, whether the rule gap predicted subsequent policy moves — the "
                "check that decides how much weight this page deserves.", icon="🧪")

    # ---- provenance + validation ------------------------------------------------------
    with st.expander("🔍  Where every number came from, and the correctness check"):
        st.markdown("**Inputs**")
        st.dataframe(pd.DataFrame(
            [{"Input": k, "Source": v} for k, v in (prov.sources or {}).items()]),
            use_container_width=True, hide_index=True)
        st.markdown("**Cross-check against the Cleveland Fed**")
        st.caption(
            "The Cleveland Fed publishes its own seven-rule calculation. We feed our engine "
            "their inputs and compare with their published prescriptions — an independent "
            "test that fails loudly if a coefficient ever drifts.")
        try:
            v = macrorules.validate_against_cleveland()
            if v["checks"]:
                n_ok = sum(1 for c in v["checks"] if c["ok"])
                worst = max(abs(c["diff_bp"]) for c in v["checks"])
                (st.success if v["ok"] else st.error)(
                    f"{n_ok}/{len(v['checks'])} checks pass — largest difference "
                    f"{worst:.2f}bp. {v['asof']}")
                if not v["ok"]:
                    st.dataframe(pd.DataFrame(
                        [c for c in v["checks"] if not c["ok"]]),
                        use_container_width=True, hide_index=True)
            else:
                st.warning(f"Cross-check unavailable: {v.get('reason', '')}")
        except Exception as e:
            st.warning(f"Cross-check unavailable: {e}")

        st.markdown("**Free-data sources**")
        try:
            st.dataframe(pd.DataFrame(macrodata.source_status()),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"(source status unavailable: {e})")

    # ---- report ---------------------------------------------------------------------------
    st.divider()
    rc1, rc2 = st.columns([1, 3])
    if rc1.button("📄 Build PDF report", key="radar_pdf", use_container_width=True):
        with st.spinner("Rendering the Macro Rate Radar report…"):
            try:
                # Lazy import: reportkit pulls matplotlib in, which the server process
                # deliberately doesn't carry — only the report path pays for it.
                from src import macroradarreport
                out = macroradarreport.build(bank=bank, rule_key=rule_key,
                                             nairu=ov_nairu, rstar=ov_rstar)
                st.session_state["radar_pdf_path"] = str(out)
            except Exception as e:
                st.error(f"Report failed: {e}")
    p = st.session_state.get("radar_pdf_path")
    if p and Path(p).exists():
        with open(p, "rb") as fh:
            rc2.download_button("⬇️ Download the report", fh.read(),
                                file_name=Path(p).name, mime="application/pdf",
                                use_container_width=True, key="radar_dl")

# ----- sidebar: navigation -------------------------------------------------
with st.sidebar:
    st.markdown(_LOGO_HOME_CSS, unsafe_allow_html=True)
    _side = st.session_state.get("side", "FICC")
    # Logo + the FICC/Equities switch live in one sticky wrapper (styled in brand._CSS) so
    # they stay pinned at the top of the sidebar while the nav list scrolls beneath them.
    with st.container(key="basis_sidebar_sticky"):
        with st.container(key="basis_logo_home"):
            brand.sidebar_logo()
            st.button("Home", key="basis_logo_home_btn", on_click=_go, args=("Landing",),
                      use_container_width=True)
        # ("00 · Overview" + its TERMINAL header dropped 2026-08-15 — redundant since
        # the desk buttons below land on the same overview pages and the logo owns
        # the front door.)
        # DESK: FICC | EQUITIES segmented control. On the desk-neutral Landing page
        # NEITHER side highlights (Ben, 2026-08-15) — the gold state means "you are
        # on this desk", and the front door belongs to both.
        _on_landing = st.session_state.get("active") == "Landing"
        st.markdown('<div class="bt-sect">Desk</div>', unsafe_allow_html=True)
        _sc1, _sc2 = st.columns(2, gap="small")
        _sc1.button("FICC", key="side_ficc", use_container_width=True,
                    type="primary" if (_side == "FICC" and not _on_landing) else "secondary",
                    on_click=_set_side, args=("FICC",))
        _sc2.button("Equities", key="side_equities", use_container_width=True,
                    type="primary" if (_side == "Equities" and not _on_landing) else "secondary",
                    on_click=_set_side, args=("Equities",))
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
        # Market Information (Reports Calendar / Market Hours / Block Sizes / Fut-Yield)
        # collapses to one entry; STIR Paths and Technical Analysis (hub + TA Backtester +
        # Signal Ledger) do the same, numbered in after the strategy groups, each carrying
        # the tab-row switcher (_render_group_tabs). Trade Testing is the Vol Backtester
        # alone. Morning Coffee is reached from the Home page's Data row.
        _nav_button("01 · Market Information", "Release Calendar")
        _nav_button("02 · Hot Sheet", "Hot Sheet")
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
        _nav_button(f"{_n_mod:02d} · Curve / RV", "Curve Monitor")
        _n_mod += 1
        _nav_button(f"{_n_mod:02d} · Seasonality", "Seasonality")
        _n_mod += 1
        _nav_button(f"{_n_mod:02d} · STIR Paths", "STIR Timeline")
        _n_mod += 1
        _nav_button(f"{_n_mod:02d} · Macro Rate Radar", "Macro Radar")
    else:
        st.markdown('<div class="bt-sect">Equities modules · US + EU indices</div>',
                    unsafe_allow_html=True)
        # No "Equities Home" entry — the Equities desk segment (and the logo) already land there.
        # Technical Analysis carries its tab row (hub + TA Backtester; the equities
        # Signal Ledger is embedded at the hub's foot) — no separate Backtester entry.
        _nav_button("01 · Hot Sheet", "eq:Hot Sheet")
        _nav_button("02 · Technical Analysis", "eq:Technical Analysis")
        _nav_button("03 · Company Fundamentals", "eq:Fundamentals")
        _nav_button("04 · Earnings Calendar", "eq:Earnings")
        _nav_button("05 · Single Stock Correlations", "eq:Correlations")
        _nav_button("06 · Index Dispersion", "eq:Dispersion")
        _nav_button("07 · Client ETFs", "eq:ETFs")
    # Cross-asset / System: shared across BOTH desks, not FICC-only.
    st.markdown('<div class="bt-sect">Cross-asset</div>', unsafe_allow_html=True)
    _nav_button("Strategy Builder", "Strategy Builder")
    if IS_ADMIN:
        st.markdown('<div class="bt-sect">System</div>', unsafe_allow_html=True)
        _nav_button("Alert Settings", "Recipients")
        _nav_button("Data Health", "Data health")
        _nav_button("Universe", "Universe")
        _nav_button("Colleague Access", "Colleague Access")
        _nav_button("Compliance", "Compliance")
    if auth.REQUIRE_LOGIN:
        st.caption(f"Logged in as **{CURRENT_USER['name']}**")
        auth.render_logout_button()
    # footer status rows (handoff): SIGNALS · FEED · DATA
    _feed = {"bloomberg": ("BBG live", "#46C58A"),
             "snapshot": ("snapshot", "#F5C518")}.get(MODE, ("demo", "#EC6A57"))
    _data_s = str((snap or {}).get("as_of", "—"))
    try:        # ISO -> the same "19 Aug 2026" style as the other footer dates
        _data_s = datetime.strptime(_data_s[:10], "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        pass
    brand.sidebar_footer([
        ("signals", _to_et(meta.get("as_of", "n/a")), ""),
        ("feed", _feed[0], _feed[1]),
        ("data", _data_s, ""),
    ])

# ----- fixed top bar (same on every page, stays while scrolling): world clocks
# over the masthead row (logo left · module breadcrumb · ET clock · theme toggle).
# Pinned by brand CSS (.st-key-basis_topbar wrapper -> position:fixed).
_active_dest = st.session_state.get("active", "Home")
if _active_dest in ("Home", "eq:Home", "Landing"):
    _crumb = None                                      # Overview/front door: the tagline
else:
    _crumb = f"{_side} desk · {_active_dest.removeprefix('eq:')}"
with st.container(key="basis_topbar"):
    # Masthead row: BASIS + crumb left, the SEAT selector right (redesign 2026-08-20:
    # a "seat" is a person — locally the admin can flip seats to view/assign another
    # colleague's My Day list; a logged-in VPS session is pinned to its own seat).
    _mh_l, _mh_r = st.columns([0.78, 0.22], vertical_alignment="center")
    with _mh_l:
        brand.masthead(_crumb, toggle=False)      # BASIS on top, clocks underneath
    with _mh_r:
        from src import myday as _myday
        _seats = _myday.seats()
        if auth.REQUIRE_LOGIN and not IS_ADMIN:
            _uid = str(CURRENT_USER.get("email") or CURRENT_USER.get("name") or "admin")
            st.session_state["seat"] = _uid
        else:
            with st.container(key="basis_seat"):
                # the design's pill: gold initials chip · "Name · Desk" · chevron
                _opts = {f'{s["name"]} · {s["desk"]}': s["id"] for s in _seats}
                _names = list(_opts)
                _cur = st.session_state.get("basis_seat_pick") or (_names[0] if _names else "")
                _init = "".join(w[0] for w in _cur.split("·")[0].replace(".", " ").split()[:2]).upper() or "?"
                _sc_chip, _sc_sel = st.columns([0.16, 0.84], vertical_alignment="center")
                _sc_chip.markdown(f'<div class="seat-chip">{_init}</div>', unsafe_allow_html=True)
                with _sc_sel:
                    _pick = st.selectbox("Seat", _names, key="basis_seat_pick",
                                         label_visibility="collapsed")
                st.session_state["seat"] = _opts.get(_pick, "admin")
    _tb_cl, _tb_tg = st.columns([0.94, 0.06], vertical_alignment="center")
    with _tb_cl:
        _world_clocks()
    with _tb_tg:
        brand.theme_toggle()           # fills the space right of the clocks
# Top-of-page market ticker rail removed 2026-07-31 (Ben: no on-screen value). The masthead
# + world clocks + theme toggle above are the top bar now. (brand.ticker_rail is left in place
# but unused.)

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
        "In **snapshot** mode a brand-new product has no data until you **Pull Bloomberg snapshot** "
        "again; regenerate any client reports afterwards to include it."
    )

# Destinations shared across BOTH desks (Cross-asset / System sidebar sections) — reachable from
# either desk's sidebar, so they must fall through this Equities-only gate to the generic dispatch
# chain below rather than being swallowed by its else-branch back into the Equities home page.
_SHARED_DESTS = {"Recipients", "Strategy Builder", "Data health", "Universe", "Colleague Access",
                 "Compliance", "User Admin", "User Activity", "Landing"}

# Defense-in-depth: even though colleague sessions never see the nav buttons/tabs that set `active`
# to one of these admin-only destinations, refuse to render them for a non-admin session regardless
# of how `active` got set. Redirects to Home rather than erroring, since this should never happen
# in normal use.
_ADMIN_ONLY_DESTS = {"Recipients", "Data health", "Universe", "Colleague Access", "Compliance",
                     "User Admin", "User Activity"}
if active in _ADMIN_ONLY_DESTS and not IS_ADMIN:
    active = st.session_state.active = "Home"

# "What did they look at" for the Colleague Activity page -- logs once per navigation (when the
# destination actually changes), not on every widget interaction within a page. No-op on the local
# Terminal (auth.REQUIRE_LOGIN False), where there's no session to attribute a view to.
if auth.REQUIRE_LOGIN and st.session_state.get("_last_logged_page") != active:
    st.session_state["_last_logged_page"] = active
    auth.record_page_view(CURRENT_USER["email"], active)

# ----- EQUITIES side: its own home (and future pages), dispatched before the FICC pipeline so the
# futures report-popup + group-tab switcher never run on the Equities side -----------------------
if side == "Equities" and active not in _SHARED_DESTS:
    # The equities modules' own tab rows (e.g. TA Hub / TA Backtester / Signal Ledger) —
    # the FICC switcher below never runs on this side, so render it here.
    _render_group_tabs(active)
    if active == "eq:Hot Sheet":
        render_hotsheet("equities")
    elif active == "eq:Fundamentals":
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
    elif active == "eq:TA Backtester":
        render_ta_backtester("equities")
    elif active == "eq:Signal Ledger":
        render_signal_ledger("equities")
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
if active == "Landing":
    render_landing(); st.stop()
if active == "Home":
    render_home(); st.stop()
if active in ("Hot Sheet", "Confluence"):        # old saved-state deep links land here too
    render_hotsheet("ficc"); st.stop()
if active == "Technical Analysis":
    render_ta_overview(); st.stop()
if active == "Morning Coffee":
    (render_morning_coffee if IS_ADMIN else render_home)(); st.stop()
if active == "Weekly Review":
    (render_weekly_review if IS_ADMIN else render_home)(); st.stop()
if active == "Market Hours":
    render_market_hours(); st.stop()
if active == "Block Sizes":
    render_block_sizes(); st.stop()
if active == "Fut Yield":
    render_fut_yield(); st.stop()
if active == "STIR Timeline":
    render_stir_overview(); st.stop()
if active == "Fed Path":
    render_stir_bank("FED"); st.stop()
if active == "ECB Path":
    render_stir_bank("ECB"); st.stop()
if active == "BoE Path":
    render_stir_bank("BOE"); st.stop()
if active == "STIR Cross":
    render_stir_overview(); st.stop()   # Cross merged into the STIR home (legacy links)
if active == "Vol Backtester":
    render_vol_backtester(); st.stop()
if active == "TA Backtester":
    render_ta_backtester("ficc"); st.stop()
if active == "Signal Ledger":
    render_signal_ledger(); st.stop()
if active == "Strategy Builder":
    render_strategy_builder(); st.stop()
if active == "Product Correlations":
    render_sector_correlations(); st.stop()
if active == "Curve Monitor":
    render_curve_monitor(); st.stop()
if active == "Macro Radar":
    render_macro_radar(); st.stop()
if active == "Seasonality":
    render_seasonality(); st.stop()
if active == "Seasonality Spreads":
    render_seasonality_spreads(); st.stop()
if active == "Data health":
    render_data_health(); st.stop()
if active == "OPEC Report":
    render_opec(); st.stop()
if active == "Precious Metals":
    render_precious_metals(); st.stop()
if active == "Macro Compass":
    # Body in src/crossmovepage.py — one macro move translated into the others.
    from src import crossmovepage
    crossmovepage.render(); st.stop()
if active == "Gold Engine":
    # Body lives in src/goldpage.py — app.py is already 16k lines and this page
    # carries four tabs of real content.
    from src import goldpage
    goldpage.render(); st.stop()
if active == "Brazil Production":
    render_brazil_production(); st.stop()
if active == "Release Calendar":
    render_releases(); st.stop()
if active == "Recipients":
    render_recipients(); st.stop()
if active == "Universe":
    render_universe(); st.stop()
if active == "Compliance":
    compliance.render_page(); st.stop()
if active == "Colleague Access":
    st.subheader("👥 Colleague Access")
    _sec = st.segmented_control("Section", ["Accounts", "Activity"], default="Accounts",
                                key="colleague_section", label_visibility="collapsed")
    st.divider()
    (auth.render_activity if _sec == "Activity" else auth.render_user_admin)()
    st.stop()
if active == "User Admin":            # kept for any deep links / saved state
    auth.render_user_admin(); st.stop()
if active == "User Activity":
    auth.render_activity(); st.stop()

# ----- a strategy page is active ------------------------------------------
st.header(active)
st.caption(STRATEGY_BLURB.get(active, ""))

# Quick-switch nav between the technical strategies (same buttons as the TA hub) so the user
# can flip between strategy pages without the sidebar — on the technical-strategy pages only.
if active in tascore.TA_STRATEGIES:
    _ta_quicknav(active)
    st.caption("💡 Fixed income runs on **yields**, not futures prices (shown as *(yield)* / *(rate)*). "
               "A **Long / up** signal there means **rising yields = short the bond** — and the mirror.")
    _trigger_docs_expander("ficc", only=active)

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
    if IS_ADMIN and _td2.button("📌 Set default", key=f"thr_def_{active}", use_container_width=True,
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
    if IS_ADMIN and a_refresh.button("↻ Refresh AG data (USDA / NASS)",
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
    if IS_ADMIN and c_refresh.button("↻ Refresh COT data (CFTC API)",
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
    if IS_ADMIN and _cd2.button("📌 Set default", key="cot_cutoff_def", use_container_width=True,
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
    if IS_ADMIN and _pd2.button("📌 Set default", key="pc_cutoff_def", use_container_width=True,
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
    if IS_ADMIN and oc1.button("↻ Refresh OI data", key="oi_refresh",
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
        help="Off = the 11 fixed-income products this page focuses on. On = every product — but only "
             "the weekly capture is served: products outside it show no chain. (On-demand live chain "
             "pulls were removed 2026-08-18 — they were the app's one unbounded Bloomberg spend.)")
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
        if sel not in OI_SNAPSHOT_TICKERS:
            st.info(f"**{INSTRUMENTS[sel][0]}** isn't in the weekly fixed-income OI capture (the 11 core "
                    "rates products), and on-demand live chain pulls are disabled (Bloomberg budget, "
                    "2026-08-18). Ask to add it to the weekly capture if it's needed regularly.")
        elif _oi_asof == "never":
            st.warning("The weekly OI capture hasn't run yet, so there's no chain data to show for ANY "
                       "product — this is not a data problem with this product. Click **↻ Refresh OI "
                       "data** above with the Terminal logged in (Mondays) to build it.")
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
    # Whole-book is an AD-HOC cross-asset export — only when "all products" is on. Since
    # 2026-08-18 it renders ONLY products present in the weekly capture (live chain pulls
    # removed); products outside it are listed as skipped rather than silently dropped.
    if IS_ADMIN and _all_products and st.button("📚 Whole-book PDF (all captured products · ad-hoc)"):
        _book_frame = _oi_input_frame(_oi_order, 6, 13)
        _oi_render_pdf(_book_frame, "book",
                       "Open_Interest_Whole_Book.pdf", "Rendering open-interest heatmaps… (whole book)")
        _in_book = set(_book_frame["ticker"]) if not _book_frame.empty else set()
        _book_missing = [INSTRUMENTS[t][0] for t in _oi_order if t not in _in_book]
        if _book_missing:
            st.caption(f"Skipped (not in the weekly OI capture): {len(_book_missing)} products — "
                       + ", ".join(_book_missing[:12]) + (" …" if len(_book_missing) > 12 else "."))

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
