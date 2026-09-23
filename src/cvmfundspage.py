"""cvmfundspage.py — the 🇧🇷 Brazil Funds page.

The screener over CVM's daily fund filings. Engine, methodology and every honest-data
rule: src/cvmfunds.py. This module is presentation only — nothing here computes a
return, because everything is precomputed into data/signals/cvm_funds by the daily pull
(app-wide rule: a page must not do 7m rows of work on open).

Three sections, switched by the segmented control (the house pattern for in-page
sections — Company Fundamentals and Colleague Access use the same one):

  Explore    one table, one filter. A grain switch (by fund / by manager) and three
             column sets (Performance / Flows / Risk). This replaced a Screener and a
             Flows section whose columns were subsets of one another and which each
             carried their OWN copy of the filter row — so a filter set in one was gone
             the moment you moved to the other. Ticking ONE row draws that fund's whole
             tearsheet underneath the table; ticking several shortlists them instead.
  Fund       the same tearsheet, reached through the manager drill-down: a seven-window
             return ladder with the excess over CDI under each, peer rank inside its own
             ANBIMA strategy, NAV against CDI, drawdown, where the capital came from, and
             gross subscriptions against redemptions by month.
  Watchlist  the classes this user has starred. Per user, because on the VPS several
             colleagues share one deployment.

WHY EXCESS OVER CDI IS ON EVERY VIEW

CDI compounded 14.6% over the twelve months to 2026-09-21, so a Brazilian fund up 14%
LOST to cash. A return column on its own invites exactly the wrong read, which is why the
excess travels with it everywhere rather than living in one corner of the page.

EVERYTHING DESCRIPTIVE IS SHOWN IN ENGLISH

CVM publishes in Portuguese. Classes, ANBIMA strategies and investor types are
translated (src/cvmfunds.py holds the vocabulary), and fund and manager names are tidied
down to the part that identifies them — the legal boilerplate every Brazilian fund
carries says nothing about which fund it is. Names are never TRANSLATED, only trimmed,
and the Fund section prints the full registered name and CNPJ, because that is what you
quote to a client and type into a vendor system.

WHY %CDI IS BLANK SO OFTEN

"% do CDI" only means something when both legs are positive. A fund down 3% against a
CDI up 14% is not "-21% of CDI" — it just lost money, and the percentage is a number
that reads as a ratio while carrying no information. Those cells are blank and the
return column tells the story.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import auth, brand, cvmfunds

_BN = 1e9
_MM = 1e6

_VIEWS = ["📊 Explore", "📈 Fund", "★ Watchlist"]


# ── helpers ─────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _store() -> tuple[pd.DataFrame, dict]:
    return cvmfunds.load()


def _brl(v: float, unit: str = "bn", dp: int = 1) -> str:
    """The minus sign goes before the currency, not after it — "R$-14,689m" reads as a
    typo, and net flows are negative often enough for it to matter."""
    if v is None or v != v:
        return "—"
    div, suf = (_BN, "bn") if unit == "bn" else (_MM, "m")
    return f"{'−' if v < 0 else ''}R${abs(v) / div:,.{dp}f}{suf}"


def _pct(v: float, dp: int = 1, signed: bool = True) -> str:
    if v is None or v != v:
        return "—"
    # A value that rounds to zero loses its sign: "-0.00%" reads as a typo, and on the
    # 1-day column — where most funds move a basis point — it was most of the table.
    if round(float(v), dp) == 0:
        return f"{0:.{dp}f}%"
    return f"{v:{'+' if signed else ''}.{dp}f}%"


def _md(text: str) -> str:
    """Escape currency before it reaches Streamlit markdown.

    Streamlit reads `$...$` as LaTeX, so TWO amounts in one caption swallow everything
    between them: "a simple average lets an R$8m launch outvote an R$8bn flagship"
    rendered as "lets an R8mlaunchoutvoteanR8bn flagship", italicised. A single amount is
    harmless — it takes a pair — so this is only needed where a string carries more than
    one, which is easy to introduce by concatenating two captions that each carry one.
    """
    return text.replace("$", r"\$")


def _num(v: float, dp: int = 0, signed: bool = False) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:{'+' if signed else ''},.{dp}f}"


# Every number reaches the grid as a STRING, because Streamlit 1.58 renders a numeric
# NaN as the literal word "None" — verified in isolation across all four routes: plain
# st.dataframe, NumberColumn with a format, a Styler with a NaN-safe formatter, and a
# Styler with a colorer. Neither pandas' `na_rep` nor its display values survive; the
# grid substitutes its own null text. On a page whose whole point is that a blank means
# "we do not know" (a fund with no 12-month history has no 12-month return), a column of
# percentages reading "None" is the one rendering that cannot stand.
#
# The cost is that the grid's own header sort goes alphabetical. The screener carries its
# own "Sort by" control for exactly that reason, and the other two tables arrive sorted.
PCT = dict(fn="pct")            # signed: a return
PCT_U = dict(fn="pct_u")        # unsigned: a volatility, a drawdown
NUM = dict(fn="num")
NUM_S = dict(fn="num_s")        # signed: a flow
PCT_0 = dict(fn="pct0")         # signed, whole percent — a flow as a share of assets
PCT_R = dict(fn="pctr")         # a RATIO, not a move: "53% of CDI" takes no sign
NUM_1 = dict(fn="num1")
NUM_2 = dict(fn="num2")         # a Sharpe ratio, where the second decimal is the signal
PCT_2 = dict(fn="pct2")         # a single day's move, which rounds to 0.0% at one decimal

_FORMATTERS = {
    "pct": _pct,
    "pct_u": lambda v: _pct(v, signed=False),
    "pct0": lambda v: _pct(v, 0),
    "pctr": lambda v: _pct(v, 0, signed=False),
    "num": _num,
    "num1": lambda v: _num(v, 1),
    "num_s": lambda v: _num(v, signed=True),
    "num2": lambda v: _num(v, 2),
    "pct2": lambda v: _pct(v, 2),
}


def _as_text(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Render the named columns to strings so blanks stay blank. See the note above."""
    out = df.copy()
    for col, how in spec.items():
        if col in out:
            out[col] = out[col].map(_FORMATTERS[how["fn"]])
    return out


def _move_colour(col):
    """Green up / red down, matching the rest of the app's move columns.

    Reads the RENDERED cell, not the number, because the frame reaching the grid is
    already text (see the formatting note below). The sign character is the signal, and
    "—" — a value we do not have — is grey rather than either.
    """
    out = []
    for v in col:
        t = str(v).strip()
        if t.startswith("+"):
            out.append("color:#137333;font-weight:700")
        elif t.startswith(("-", "−")):
            out.append("color:#c5221f;font-weight:700")
        else:
            out.append("color:#888")
    return out


def _filters(met: pd.DataFrame, key: str) -> tuple[pd.DataFrame, dict]:
    """The filter row, and the screened frame it produces.

    Returns the active settings too, so each tab can print the basis it is standing on.
    A table of Brazilian fund assets with no stated basis is not a fact, it is one of
    three numbers that differ by 90%.
    """
    c1, c2, c3, c4 = st.columns([1.1, 1.5, 1.2, 1.0])
    klass = c1.selectbox("Asset class", cvmfunds.CVM_CLASSES_EN + ["(all)"], index=0,
                         key=f"{key}_class",
                         help="Multi-strategy (CVM's *multimercado*) is Brazil's hedge-fund "
                              "analogue — free to hold rates, FX, equities and offshore risk "
                              "at once. Not a pure synonym: bank-distributed balanced funds "
                              "sit in it too.")
    pool = met if klass == "(all)" else met[met["class_en"] == klass]
    strats = sorted(x for x in pool["strategy_en"].dropna().unique() if x)
    picked = c2.multiselect("Strategy", strats, default=[], key=f"{key}_anbima",
                            placeholder="all strategies",
                            help="ANBIMA's finer cut, translated — Macro, Long/Short, "
                                 "Offshore, Rates & FX, and so on.")
    gest = c3.text_input("Manager contains", "", key=f"{key}_gestor", placeholder="e.g. Kapitalo")
    min_aum = c4.number_input("Min assets (R$m)", min_value=0, value=100, step=50,
                              key=f"{key}_minaum") * _MM

    o1, o2, o3, o4 = st.columns(4)
    feeders = o1.toggle("Include feeders", value=False, key=f"{key}_feed",
                        help="Fund-of-quotas (FIC) classes hold another fund's units, so their "
                             "assets ARE that fund's assets counted twice. Including them "
                             "overstates multimercado by ~58%. On only to reconcile against a "
                             "vendor table that does the same.")
    excl = o2.toggle("Include exclusive", value=False, key=f"{key}_excl",
                     help="Single-family and single-institution vehicles. Real money, but "
                          "nobody can buy them.")
    prev = o3.toggle("Include pension funds", value=False, key=f"{key}_prev",
                     help="Pension wrappers (*previdência*). CVM files them under what they "
                          "invest in; ANBIMA counts them separately — the main reason a "
                          "CVM-derived total will not match an ANBIMA headline.")
    retail = o4.toggle("Retail-available only", value=False, key=f"{key}_retail",
                       help="Funds open to the general public — drops the qualified- and "
                            "professional-investor-only funds.")

    d = cvmfunds.screen(met, cvm_class=None if klass == "(all)" else klass,
                        include_feeders=feeders, include_exclusive=excl,
                        include_prev=prev, gestor=gest or None, min_aum=min_aum,
                        publico=["Retail"] if retail else None)
    if picked:
        d = d[d["strategy_en"].isin(picked)]
    return d, {"class": klass, "feeders": feeders, "exclusive": excl, "prev": prev,
               "retail": retail, "min_aum": min_aum}


def _basis_line(opts: dict, n: int, aum: float) -> None:
    bits = []
    bits.append("including feeders (**double-counted**)" if opts["feeders"] else "ex-feeder")
    bits.append("incl. exclusive" if opts["exclusive"] else "ex-exclusive")
    bits.append("incl. pension" if opts["prev"] else "ex-pension")
    if opts["retail"]:
        bits.append("retail-available only")
    st.caption(f"**{n:,}** share classes · **{_brl(aum)}** · {opts['class']} · " + ", ".join(bits))


# ── tabs ────────────────────────────────────────────────────────────────────────────
def _fund_keys(d: pd.DataFrame) -> pd.Series:
    return d["cnpj"] + "|" + d["subclass"].fillna("")


# The three column sets. One table, one filter, one grain switch — the page used to carry
# a Screener and a Flows section whose columns were a subset of one another, each with its
# OWN copy of the filter row, which reset every time you moved between them.
_COLSETS = ["Performance", "Flows", "Risk"]
_GRAINS = ["By fund", "By manager"]
_EXPLORE_HELP = {
    "Performance": "Returns are net of fees — the published quota already is. "
                   "**vs CDI** is excess over the Brazilian cash benchmark, which is the "
                   "number that says whether a fund was worth owning. ⚠ marks a class "
                   "that restruck its quota in the window — a filing artefact, not a "
                   "market move — and those sort last.",
    "Flows": "Net subscriptions minus redemptions, in **R$m**. A performance table cannot "
             "answer this — a fund can be up 20% and bleeding.",
    "Risk": "Volatility is annualised from daily quotas; drawdown is peak-to-trough over "
            "the cached window. Sharpe is excess-over-CDI divided by that volatility.",
}


def _explore_funds(d: pd.DataFrame, colset: str) -> tuple[pd.DataFrame, dict, list]:
    """(display frame, text-format spec, columns to colour) for the per-fund grain."""
    warn = np.where(d["glitch"].fillna(False), "⚠ ", "")
    base = {"Fund": warn + d["name_en"], "Manager": d["gestor_en"]}
    if colset == "Flows":
        disp = pd.DataFrame({**base,
                             "Assets": d["aum"] / _MM,
                             "Flow 1m": d["flow_1m"] / _MM,
                             "Flow 3m": d["flow_3m"] / _MM,
                             "Flow 12m": d["flow_12m"] / _MM,
                             "% of assets": d["flow_3m"] / d["aum"].replace(0, np.nan) * 100.0,
                             "Holders": d["holders"]})
        spec = {"Assets": NUM, "Flow 1m": NUM_S, "Flow 3m": NUM_S, "Flow 12m": NUM_S,
                "% of assets": PCT_0, "Holders": NUM}
        return disp, spec, ["Flow 1m", "Flow 3m", "Flow 12m", "% of assets"]
    if colset == "Risk":
        disp = pd.DataFrame({**base,
                             "Vol": d["vol"], "Max DD": d["max_dd"],
                             "Sharpe": d.get("sharpe"), "12m": d["ret_12m"],
                             "vs CDI 12m": d.get("exc_12m"),
                             "Assets": d["aum"] / _MM})
        spec = {"Vol": PCT_U, "Max DD": PCT_U, "Sharpe": NUM_2, "12m": PCT,
                "vs CDI 12m": PCT, "Assets": NUM}
        return disp, spec, ["12m", "vs CDI 12m", "Max DD"]
    disp = pd.DataFrame({**base, "Strategy": d["strategy_en"],
                         "1d": d.get("ret_1d"), "1w": d.get("ret_1w"),
                         "1m": d["ret_1m"], "3m": d["ret_3m"], "6m": d.get("ret_6m"),
                         "YTD": d["ret_ytd"], "12m": d["ret_12m"],
                         "vs CDI 12m": d.get("exc_12m"), "Assets": d["aum"] / _MM})
    spec = {"1d": PCT_2, "1w": PCT, "1m": PCT, "3m": PCT, "6m": PCT, "YTD": PCT,
            "12m": PCT, "vs CDI 12m": PCT, "Assets": NUM}
    return disp, spec, ["1d", "1w", "1m", "3m", "6m", "YTD", "12m", "vs CDI 12m"]


def _explore_managers(lt: pd.DataFrame, colset: str) -> tuple[pd.DataFrame, dict, list]:
    """The same three column sets, one row per manager. Returns are ASSET-weighted."""
    base = {"Manager": lt["label"]}
    if colset == "Flows":
        disp = pd.DataFrame({**base, "Funds": lt["funds"], "Assets": lt["aum"] / _BN,
                             "Flow 3m": lt["flow_3m"] / _MM,
                             "Flow 12m": lt["flow_12m"] / _MM, "Holders": lt["holders"]})
        return disp, {"Funds": NUM, "Assets": NUM_1, "Flow 3m": NUM_S,
                      "Flow 12m": NUM_S, "Holders": NUM}, ["Flow 3m", "Flow 12m"]
    if colset == "Risk":
        disp = pd.DataFrame({**base, "Funds": lt["funds"], "Assets": lt["aum"] / _BN,
                             "Vol": lt.get("vol"), "12m": lt.get("ret_12m"),
                             "vs CDI 12m": lt.get("exc_12m")})
        return disp, {"Funds": NUM, "Assets": NUM_1, "Vol": PCT_U, "12m": PCT,
                      "vs CDI 12m": PCT}, ["12m", "vs CDI 12m"]
    disp = pd.DataFrame({**base, "Funds": lt["funds"], "Assets": lt["aum"] / _BN,
                         "Share": lt["share"], "3m": lt.get("ret_3m"),
                         "YTD": lt.get("ret_ytd"), "12m": lt.get("ret_12m"),
                         "vs CDI 12m": lt.get("exc_12m")})
    return disp, {"Funds": NUM, "Assets": NUM_1, "Share": PCT_U, "3m": PCT, "YTD": PCT,
                  "12m": PCT, "vs CDI 12m": PCT}, ["3m", "YTD", "12m", "vs CDI 12m"]


def _selected_rows(sel) -> list:
    try:
        return list(sel["selection"]["rows"]) if sel else []
    except (KeyError, TypeError):
        return []


def _after_table(met: pd.DataFrame, d: pd.DataFrame, sel, key: str) -> None:
    """Star bar for a multi-row pick; the full tearsheet when exactly one row is ticked.

    One row means "show me this fund", several means "shortlist these" — the same gesture
    reads as both, so the count decides rather than a second control.
    """
    # `_add`, not `_star`: the tearsheet below has its own star button keyed `{key}_star`,
    # and both rendering under one key is a Streamlit error, not a styling quirk.
    _star_bar(d, sel, key=f"{key}_add")
    rows = _selected_rows(sel)
    if len(rows) == 1:
        st.divider()
        _fund_detail(met, d.iloc[rows[0]], key=key)


def _star_bar(d: pd.DataFrame, sel, key: str = "ex_star") -> None:
    """Add the rows ticked in the table to the watchlist.

    A per-row star button is not possible inside st.dataframe — the grid is a canvas, not
    DOM — so selecting rows and pressing once is the honest version of a ★ column.
    """
    rows = _selected_rows(sel)
    c1, c2 = st.columns([1.1, 3])
    if not rows:
        c2.caption("Tick one row to chart that fund, or several to shortlist them.")
        return
    if c1.button(f"★  Add {len(rows)} to watchlist", key=key, type="primary",
                 use_container_width=True):
        keys = _fund_keys(d.iloc[rows]).tolist()
        added = cvmfunds.watch_add(keys)
        c2.success(f"Added {added}." if added == len(keys)
                   else f"Added {added} — {len(keys) - added} already on the list.")


def _tab_explore(met: pd.DataFrame) -> None:
    d, opts = _filters(met, "ex")
    if d.empty:
        st.info("No funds match that filter.")
        return
    g1, g2 = st.columns([1, 1.4])
    grain = g1.segmented_control("Group by", _GRAINS, default=_GRAINS[0], key="ex_grain",
                                 label_visibility="collapsed") or _GRAINS[0]
    colset = g2.segmented_control("Columns", _COLSETS, default=_COLSETS[0], key="ex_cols",
                                  label_visibility="collapsed") or _COLSETS[0]
    _basis_line(opts, d["cnpj"].nunique(), d["aum"].sum())

    if grain == _GRAINS[1]:
        _explore_by_manager(met, d, colset)
        return

    # Each column set arrives sorted by the thing it is about, so the table opens on the
    # interesting end without anyone touching a control.
    sort_col = {"Performance": "ret_12m", "Flows": "flow_3m", "Risk": "vol"}[colset]
    asc = colset == "Risk"                      # least volatile first is the useful end
    # Re-based quotas sort LAST, whatever the column. They are real filings and stay on
    # the page behind their ⚠, but a class whose quota was restruck prints a "+488%"
    # twelve-month return and would otherwise own the top of every performance ranking —
    # a filing artefact presented as the best fund in Brazil.
    d = (d.assign(_glitch=d["glitch"].fillna(False))
          .sort_values(["_glitch", sort_col], ascending=[True, asc], na_position="last")
          .drop(columns="_glitch").head(300))
    disp, spec, moves = _explore_funds(d, colset)
    st.caption(_EXPLORE_HELP[colset])

    sel = brand.themed_dataframe(_as_text(disp, spec), {}, height=520,
                                 colorers=[(moves, _move_colour)],
                                 on_select="rerun", selection_mode="multi-row",
                                 key=f"ex_tbl_{colset}")
    _after_table(met, d, sel, key="exf")
    if len(d) == 300:
        st.caption("Showing the first 300 rows of this filter — narrow it to see the rest.")


def _explore_by_manager(met: pd.DataFrame, d: pd.DataFrame, colset: str) -> None:
    """The league table, plus the reconciliation block that says what each counting basis
    is worth. The gap between bases is wider than the gap between the top three managers,
    which is why it stays on the page instead of being resolved silently — folded into an
    expander now, because it is context you read once, not every visit."""
    pal = brand.palette()
    by_firm = st.toggle(
        "Group the entities of one house together", value=False, key="ex_firm",
        help="BTG Pactual runs three separately registered gestores and Itaú two. Off, "
             "each is its own row (CVM's unit, and what everything else here counts). On, "
             "they merge into one house — the question people actually ask, and how "
             "ANBIMA consolidates.")
    lt = cvmfunds.by_gestor(d, by_firm=by_firm)
    tot = cvmfunds.industry_totals(met)
    if tot:
        with st.expander("How much the counting basis is worth", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gross, all classes", _brl(tot["gross"], dp=0),
                      help="Every multimercado class summed. Wrong, and the number most "
                           "often quoted.")
            # ASCII hyphen, not U+2212: Streamlit parses the delta string to pick its arrow
            # direction, and a typographic minus reads as unparseable — which drew an UP
            # arrow on a number smaller than gross. delta_color="off" keeps it grey.
            m2.metric("Ex-feeder", _brl(tot["ex_feeder"], dp=0),
                      delta=f"-{tot['feeder_overstate']:.0f}% vs gross", delta_color="off",
                      help="Fund-of-quotas classes removed — each real pool counted once.")
            m3.metric("Ex-feeder, ex-pension", _brl(tot["ex_feeder_ex_prev"], dp=0),
                      help="The basis closest to an ANBIMA headline.")
            m4.metric("Pension carve-out", _brl(tot["prev"], dp=0),
                      help="Pension (previdência) multi-strategy, which ANBIMA counts as "
                           "its own category.")
            st.caption("ANBIMA published R$1,519bn of multimercado with Itaú Asset at 9.8%. "
                       "The third column here is the comparable basis; the residual gap is "
                       "ANBIMA consolidating economic groups — their one BTG Pactual is "
                       "three separately registered gestores in CVM's file — plus a vintage "
                       "difference.")
    disp, spec, moves = _explore_managers(lt, colset)
    st.caption(_md(
        "Returns are **asset-weighted** across each manager's funds — a simple average "
        "lets an R$8m launch outvote an R$8bn flagship. " + _EXPLORE_HELP[colset]))
    # On its own line, not buried mid-caption. Read as one sentence in a four-line block
    # of methodology it was invisible: the drill-down shipped and looked like nothing had
    # changed, because nothing does until a row is ticked.
    st.markdown(f"<div style='color:{pal['gold']};font-size:.85rem;margin:.1rem 0 .3rem'>"
                f"☑&nbsp; Tick a manager below to list its funds underneath."
                f"</div>", unsafe_allow_html=True)
    sel = brand.themed_dataframe(_as_text(disp, spec), {}, height=460,
                                 colorers=[(moves, _move_colour)],
                                 on_select="rerun", selection_mode="single-row",
                                 key=f"ex_mgr_{colset}_{by_firm}")
    _manager_drilldown(met, d, lt, sel, colset, by_firm)


def _manager_drilldown(met: pd.DataFrame, d: pd.DataFrame, lt: pd.DataFrame, sel,
                       colset: str, by_firm: bool) -> None:
    """The funds inside the manager picked in the league table.

    Opened by SELECTION rather than by an expander per row: Streamlit runs the body of a
    collapsed expander, so forty of them would build every manager's fund rows on every
    rerun. Selecting also keeps the league table itself sortable, which is what it is for.
    """
    try:
        rows = list(sel["selection"]["rows"]) if sel else []
    except (KeyError, TypeError):
        rows = []
    if not rows:
        return
    # Grouped on the REGISTERED gestor unless the firm merge is on — the same key
    # by_gestor grouped by, or the funds underneath would belong to a different entity.
    key = lt.index[rows[0]]
    sub = d[(d["firm"] if by_firm else d["gestor"]) == key]
    if sub.empty:
        return
    sub = sub.sort_values("aum", ascending=False)
    st.markdown(f"**{lt.iloc[rows[0]]['label']}** — {len(sub)} "
                f"{'fund' if len(sub) == 1 else 'funds'}, "
                f"{_brl(sub['aum'].sum())}")
    fdisp, fspec, fmoves = _explore_funds(sub.head(_FUND_PAGE), colset)
    fsel = brand.themed_dataframe(_as_text(fdisp, fspec), {},
                                  height=min(420, 60 + 36 * len(sub)),
                                  colorers=[(fmoves, _move_colour)],
                                  on_select="rerun", selection_mode="multi-row",
                                  key=f"ex_sub_{colset}")
    if len(sub) > _FUND_PAGE:
        st.caption(f"Showing the largest {_FUND_PAGE} of {len(sub)} — the Fund section "
                   f"searches the rest.")
    _after_table(met, sub.head(_FUND_PAGE), fsel, key="exm")


# The picker shows managers first and opens ONE at a time. Streamlit executes the body of
# a collapsed expander, so a real expander per manager would have built ~20,000 fund
# buttons on every rerun; rendering only the open manager's funds keeps it to a few dozen.
_LIST_H = 430             # the picker is a bounded scroll box, so the chart stays in view
_MGR_PAGE = 40            # managers listed before the search box is the way through
_FUND_PAGE = 60           # funds listed inside one manager — Itaú alone runs over 1,200
# The app styles every button as chrome — uppercase, letter-spaced, centred — which is
# right for a nav tab and wrong for a row of data. These rows carry names the page went to
# some trouble to tidy, and "BB Top Fixed Income Short Term Automático II" shouted back as
# "BB TOP FIXED INCOME SHORT TERM AUTOMÁTICO II" undoes that. Scoped to these two keys via
# Streamlit's st-key-<key> container class so no other button is touched.
# One grid for both row types. They used to differ — managers [6, 2, 1.6], funds
# [0.35, 5.65, 2, 1.6] — which put a fund's 12-month return underneath the manager's
# FUND COUNT: two unrelated quantities sharing a column, with no header to tell them
# apart. Fund rows now indent into the same four columns and leave the count blank,
# because a count is a thing a manager has and a fund does not.
_COLS = [5.2, 1.6, 1.0, 1.1]
_INDENT = 0.35

# Header label -> (column it sorts, the direction a FIRST click gives it). A name reads
# naturally A-Z; a number is interesting largest-first.
_SORTS = {"Manager": ("label", True), "Assets": ("aum", False),
          "Funds": ("funds", False), "12m": ("ret_12m", False)}
# The same sort applied to the funds inside an open manager. A fund has no fund count,
# so that one falls back to size.
_FUND_SORT = {"label": "name_en", "aum": "aum", "funds": "aum", "ret_12m": "ret_12m"}


def _row_css(pal: dict) -> str:
    """The app styles every button as chrome — uppercase, letter-spaced, centred — which
    is right for a nav tab and wrong for a row of data. These rows carry names the page
    went to some trouble to tidy, and "BB Top Fixed Income Short Term Automático II"
    shouted back in caps undoes that. The header buttons go the other way: they must read
    as column headings, not as things to press, so they lose their box entirely.

    All of it is scoped by Streamlit's st-key-<key> container class, so no other button
    on the page is touched.
    """
    return f"""<style>
.bf-cell{{text-align:right;padding-top:.55rem;font-variant-numeric:tabular-nums;
         font-size:.86rem;line-height:1.2}}
[class*="st-key-bfm_"] button, [class*="st-key-bff_"] button,
[class*="st-key-bfm_"] button *, [class*="st-key-bff_"] button *{{
  text-transform:none!important; letter-spacing:0!important;
  justify-content:flex-start!important; text-align:left!important}}
[class*="st-key-bff_"] button{{font-weight:400!important}}
[class*="st-key-bfm_"] button{{font-weight:600!important}}
[class*="st-key-bfh"] button{{
  background:transparent!important; border:none!important; box-shadow:none!important;
  min-height:0!important; height:auto!important; padding:.1rem .1rem .35rem!important;
  color:{pal['text_dim']}!important; font-size:.72rem!important; font-weight:600!important;
  letter-spacing:.09em!important; text-transform:uppercase!important}}
[class*="st-key-bfh"] button:hover{{color:{pal['gold']}!important}}
[class*="st-key-bfh_"] button, [class*="st-key-bfh_"] button *{{
  justify-content:flex-start!important; text-align:left!important}}
[class*="st-key-bfhr_"] button, [class*="st-key-bfhr_"] button *{{
  justify-content:flex-end!important; text-align:right!important}}
/* the header row lives OUTSIDE the bordered scroll box so it stays put while the list
   moves under it, which also means it does not inherit the box's padding — these put
   the labels back over their own columns */
[class*="st-key-bfh_"] button{{padding-left:1.7rem!important}}
[class*="st-key-bfhr_"] button{{padding-right:1rem!important}}
/* Streamlit stacks st.columns on a narrow screen, which turns this table into a column
   of loose values and the header into four orphaned words. These rows are a table at
   every width — held side by side, just smaller. :has() scopes it to the picker's own
   rows, and min-width:0 is what lets a column shrink instead of forcing a scrollbar. */
@media (max-width:760px){{
  [data-testid="stHorizontalBlock"]:has([class*="st-key-bf"]){{flex-wrap:nowrap!important}}
  [data-testid="stHorizontalBlock"]:has([class*="st-key-bf"]) > div{{min-width:0!important}}
  [class*="st-key-bfm_"] button, [class*="st-key-bff_"] button{{
    font-size:.76rem!important; padding-left:.45rem!important; padding-right:.2rem!important}}
  /* held on one line: a value that wraps ("R$1,159.4b / n") is worse than a small one,
     and a header that breaks mid-word ("FUN / DS") stops being a label at all */
  .bf-cell{{font-size:.68rem!important; padding-top:.5rem!important; white-space:nowrap}}
  [class*="st-key-bfh"] button, [class*="st-key-bfh"] button *{{
    font-size:.58rem!important; letter-spacing:.01em!important; white-space:nowrap!important}}
  [class*="st-key-bfh_"] button{{padding-left:.6rem!important}}
  [class*="st-key-bfhr_"] button{{padding-right:.35rem!important}}
}}
</style>"""


def _fund_key(row) -> str:
    """A share class is a CNPJ plus a subclass — the CNPJ alone is not unique."""
    return f"{row['cnpj']}|{row['subclass'] or ''}"


def _open_manager(gestor: str) -> None:
    """Clicking the open manager closes it; clicking another opens that one instead."""
    st.session_state["fnd_mgr"] = None if st.session_state.get("fnd_mgr") == gestor else gestor


def _select_fund(key: str) -> None:
    st.session_state["fnd_key"] = key


def _cell(col, text: str, colour: str) -> None:
    col.markdown(f"<div class='bf-cell' style='color:{colour}'>{text}</div>",
                 unsafe_allow_html=True)


def _ret_colour(v: float, pal: dict, cc: dict) -> str:
    """Grey for a return we do not have — neither green nor red is honest about a blank."""
    return pal["text_dim"] if v != v else (cc["long"] if v > 0 else cc["short"])


def _sort_click(label: str) -> None:
    """The same header again flips the direction; a different one starts in its natural
    one, so a first click never lands you on the least interesting end of a column."""
    cur, asc = st.session_state.get("fnd_sort", ("Assets", False))
    st.session_state["fnd_sort"] = (label, not asc) if cur == label else (label, _SORTS[label][1])


def _sorted(df: pd.DataFrame, key: str, asc: bool) -> pd.DataFrame:
    """Names sort case-insensitively; numbers keep their blanks at the bottom either way,
    because a fund with no 12-month return is not the worst performer."""
    if key not in df:
        return df
    if df[key].dtype == object:
        return df.sort_values(key, ascending=asc, key=lambda c: c.str.lower())
    return df.sort_values(key, ascending=asc, na_position="last")


def _header_row(active: str, asc: bool) -> None:
    """Four clickable column headings. They sit OUTSIDE the scroll box so they stay put
    while the list moves under them."""
    for col, label in zip(st.columns(_COLS), _SORTS):
        mark = (" ▲" if asc else " ▼") if label == active else ""
        col.button(label + mark,
                   key=f"{'bfh' if label == 'Manager' else 'bfhr'}_{label}",
                   on_click=_sort_click, args=(label,), use_container_width=True,
                   help=f"Sort by {label.lower()}" + (" — click again to reverse"
                                                      if label == active else ""))


def _fund_picker(met: pd.DataFrame):
    """Managers by assets, one expanding to its funds. Returns the chosen row, or None.

    Search matches funds AND managers, on the English label and the registered name
    alike, so a fund outside its manager's top few is still reachable — the old flat
    dropdown was capped at 4,000 of ~20,000 classes and the rest simply could not be
    opened.
    """
    pal, cc = brand.palette(), brand.chart_colors()
    f1, f2 = st.columns([3, 1])
    q = f1.text_input("Search", "", key="fnd_q", placeholder="fund, manager or CNPJ",
                      label_visibility="collapsed").strip()
    floor = f2.number_input("Min assets (R$m)", min_value=0, value=50, step=25,
                            key="fnd_floor", label_visibility="collapsed") * _MM

    d = cvmfunds.screen(met, cvm_class=None, min_aum=floor)
    if d.empty:
        st.info("Nothing in the store at that size.")
        return None
    if q:
        digits = "".join(ch for ch in q if ch.isdigit())
        hit = False
        for col in ("name_en", "name", "gestor_en", "gestor"):
            hit = hit | d[col].str.contains(q, case=False, na=False, regex=False)
        if len(digits) >= 6:
            hit = hit | d["cnpj"].str.contains(digits, na=False, regex=False)
        d = d[hit]
        if d.empty:
            st.info(f"Nothing matches “{q}”.")
            return None

    # Grouped on the REGISTERED gestor, never the label — see _resolve_label_clashes.
    agg = (d.groupby("gestor")
            .agg(label=("gestor_en", "first"), aum=("aum", "sum"), funds=("cnpj", "nunique")))
    # ASSET-weighted, never averaged: a simple mean lets a manager's R$8m launch-year fund
    # outvote its R$8bn flagship, which is the same rule the Managers league table runs on.
    w = d[["gestor", "aum", "ret_12m"]].dropna()
    if not w.empty:
        agg["ret_12m"] = ((w["ret_12m"] * w["aum"]).groupby(w["gestor"]).sum()
                          / w.groupby("gestor")["aum"].sum())
    else:
        agg["ret_12m"] = np.nan
    scol, asc = st.session_state.get("fnd_sort", ("Assets", False))
    skey = _SORTS[scol][0]
    agg = _sorted(agg, skey, asc)

    # The first render opens the biggest manager; after that the state belongs to the
    # user and None is a LEGITIMATE value — every manager closed. Treating None as
    # "nothing valid, fall back to the top one" is what made the open manager impossible
    # to close: clicking it set None, and the fallback immediately reopened it.
    if "fnd_mgr" not in st.session_state:
        st.session_state["fnd_mgr"] = agg.index[0]
    open_mgr = st.session_state["fnd_mgr"]
    if open_mgr is not None and open_mgr not in agg.index:
        open_mgr = None                      # a search filtered it away: close, do not jump

    shown = agg.head(_MGR_PAGE)
    if open_mgr is not None and open_mgr not in shown.index:
        shown = pd.concat([agg.loc[[open_mgr]], shown])   # keep it visible however deep
    # Sorting also sorts the funds INSIDE the open manager, so the open one is kept on
    # screen rather than sorted off the page — losing the list you were re-sorting is the
    # one thing the sort must not do. Said out loud, or a row above its alphabetical place
    # just reads as a broken sort.
    pinned = open_mgr is not None and open_mgr not in agg.head(_MGR_PAGE).index
    st.caption(f"**{len(agg):,}** managers · **{d['cnpj'].nunique():,}** funds · "
               f"showing the first {len(shown)} by {scol.lower()}"
               + (" — search to reach the rest" if len(agg) > len(shown) else "")
               + (" · the open manager is held at the top" if pinned else ""))
    st.markdown(_row_css(pal), unsafe_allow_html=True)
    _header_row(scol, asc)

    sel_key = st.session_state.get("fnd_key")
    # A bounded, scrolling box. Open a manager with 60 funds and an unbounded list pushes
    # the chart thousands of pixels down the page, so clicking a fund looks like it did
    # nothing at all — the thing you asked for happened where you could not see it.
    with st.container(height=_LIST_H):
        for gestor, m in shown.iterrows():
            is_open = gestor == open_mgr
            c0, c1, c2, c3 = st.columns(_COLS)
            c0.button(f"{'▾' if is_open else '▸'}  {m['label']}", key=f"bfm_{gestor}",
                      on_click=_open_manager, args=(gestor,), use_container_width=True,
                      type="primary" if is_open else "secondary")
            _cell(c1, _brl(m["aum"]), pal["text"])
            _cell(c2, f"{m['funds']:,}", pal["text_dim"])   # the word lives in the header now
            _cell(c3, _pct(m["ret_12m"]), _ret_colour(m["ret_12m"], pal, cc))
            if not is_open:
                continue
            funds = _sorted(d[d["gestor"] == gestor], _FUND_SORT[skey], asc)
            for _, r in funds.head(_FUND_PAGE).iterrows():
                k = _fund_key(r)
                s0, s1, s2, s3, s4 = st.columns([_INDENT, _COLS[0] - _INDENT, *_COLS[1:]])
                s0.write("")
                label = r["name_en"] + (f"  ·  {r['subclass']}" if r["subclass"] else "")
                s1.button(label, key=f"bff_{k}", on_click=_select_fund, args=(k,),
                          use_container_width=True,
                          type="primary" if k == sel_key else "secondary")
                _cell(s2, _brl(r["aum"], "m", 0), pal["text_dim"])
                s3.write("")                    # a fund count belongs to a manager, not a fund
                _cell(s4, _pct(r["ret_12m"]), _ret_colour(r["ret_12m"], pal, cc))
            if len(funds) > _FUND_PAGE:
                st.caption(f"    …{len(funds) - _FUND_PAGE:,} more under this manager — "
                           f"search by fund name to reach them.")

    # The click lands on the NEXT rerun, so a fresh session (or a search that filtered the
    # selection away) still needs a fund to show.
    keys = d.apply(_fund_key, axis=1)
    if sel_key in set(keys):
        return d[keys == sel_key].iloc[0]
    # The click lands on the NEXT rerun, so a fresh session still needs something to draw.
    # With every manager closed there is nothing to fall back TO, and guessing a fund the
    # user never picked would be worse than saying so.
    if open_mgr is None:
        st.info("Open a manager above and pick one of its funds to chart it.")
        return None
    top = d[d["gestor"] == open_mgr].sort_values("aum", ascending=False)
    if top.empty:
        return None
    sel_key = _fund_key(top.iloc[0])
    st.session_state["fnd_key"] = sel_key
    return d[keys == sel_key].iloc[0]

def _flow_chart(h: pd.DataFrame) -> None:
    """Subscriptions and redemptions GROSS, by month, with the net on top.

    The cumulative-net line elsewhere answers "did money arrive"; it cannot answer "how
    much churned". A fund taking R$800m and paying out R$780m is a very different animal
    from one quietly taking R$20m, and netted to a single line the two are identical.
    Monthly, because daily subscription bars on thirteen months of history are a haystack.
    """
    cc = brand.chart_colors()
    m = h[["date", "subs", "redem"]].copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    g = m.groupby("month", as_index=False).agg(subs=("subs", "sum"), redem=("redem", "sum"))
    if g.empty or (g["subs"].abs().sum() + g["redem"].abs().sum()) == 0:
        st.caption("No subscription or redemption activity recorded in the window.")
        return
    g["net"] = (g["subs"] - g["redem"]) / _MM
    bars = pd.concat([
        pd.DataFrame({"month": g["month"], "v": g["subs"] / _MM, "side": "Subscriptions"}),
        # redemptions plot DOWNWARD so the two sides read against each other rather than
        # stacking into a total nobody asked for
        pd.DataFrame({"month": g["month"], "v": -g["redem"] / _MM, "side": "Redemptions"}),
    ], ignore_index=True)

    base = alt.Chart(bars).mark_bar().encode(
        x=alt.X("yearmonth(month):O", title=None),
        y=alt.Y("v:Q", title="R$m"),
        color=alt.Color("side:N", title=None,
                        scale=alt.Scale(domain=["Subscriptions", "Redemptions"],
                                        range=[cc["long"], cc["short"]])),
        tooltip=[alt.Tooltip("yearmonth(month):O", title="Month"),
                 alt.Tooltip("side:N", title=""),
                 alt.Tooltip("v:Q", title="R$m", format=",.0f")])
    line = (alt.Chart(g).mark_line(point=True, color=cc["ink"], strokeWidth=2)
            .encode(x=alt.X("yearmonth(month):O", title=None),
                    y=alt.Y("net:Q", title="R$m"),
                    tooltip=[alt.Tooltip("yearmonth(month):O", title="Month"),
                             alt.Tooltip("net:Q", title="Net R$m", format=",.0f")]))
    brand.show_chart((base + line).properties(
        height=250, title="Subscriptions and redemptions by month (net in white)"))
    gross_in, gross_out = g["subs"].sum(), g["redem"].sum()
    st.caption(_md(
        f"Took in **{_brl(gross_in, 'm', 0)}**, paid out **{_brl(gross_out, 'm', 0)}** — "
        f"net **{_brl(gross_in - gross_out, 'm', 0)}** over the cached window."))


def _fund_detail(met: pd.DataFrame, row: pd.Series, *, key: str) -> None:
    """The whole tearsheet for one share class, callable from anywhere.

    `key` namespaces the widgets, because this now renders both on the Fund section and
    underneath the Explore table, and two star buttons with one key is a Streamlit error.
    """
    fkey = f"{row['cnpj']}|{row['subclass'] or ''}"
    starred = fkey in cvmfunds.watchlist()
    h1, h2 = st.columns([5, 1.2])
    h1.markdown(f"### {row['name_en']}")
    if h2.button("★  Starred" if starred else "☆  Add to watchlist", key=f"{key}_star",
                 use_container_width=True, type="primary" if starred else "secondary"):
        cvmfunds.watch_toggle(fkey)
        st.rerun()

    _ladder(row)
    st.markdown("")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Assets", _brl(row["aum"], "m", 0))
    m2.metric("Vol (ann.)", _pct(row["vol"], signed=False))
    m3.metric("Max drawdown", _pct(row["max_dd"], signed=False))
    m4.metric("Sharpe", _num(row.get("sharpe"), 2),
              help="Excess over CDI divided by annualised volatility. Blank for a class "
                   "under 1% vol, where the ratio is a division artefact.")
    m5.metric("Holders", f"{row['holders']:,.0f}" if row["holders"] == row["holders"] else "—")
    rank = _peer_rank(met, row)
    if rank:
        st.caption("Ranks " + rank)
    st.caption(f"**{row['gestor_en']}** · {row['strategy_en'] or row['class_en']} · "
               f"{row['audience_en']} investors"
               + ("  ·  ⚠️ feeder (fund-of-quotas)" if row["is_feeder"] else "")
               + ("  ·  exclusive" if row["is_exclusive"] else ""))
    # The REGISTERED name, in full and in Portuguese: it is what you quote to a client,
    # search in a vendor system and match against a CNPJ. The tidied English label above
    # is for reading, not for identifying.
    st.caption(f"Registered as **{row['name'].strip()}** · {row['gestor'].strip()} · "
               f"CNPJ {row['cnpj']}" + (f" · subclass {row['subclass']}"
                                        if row["subclass"] else ""))
    if row.get("glitch"):
        st.warning("This class printed a single-day quota move over 50% in the window — "
                   "usually a re-based quota or an amortisation, not a market move. Its "
                   "volatility is computed on clipped returns; read the NAV chart before "
                   "trusting the risk numbers.", icon="⚠️")

    hist = cvmfunds.history(row["cnpj"], row["subclass"] or "")
    if hist.empty:
        st.info("No cached daily history for this class.")
        return

    cc = brand.chart_colors()
    h = hist.dropna(subset=["quota"]).copy()
    h["NAV"] = h["quota"] / h["quota"].iloc[0] * 100.0
    cdi = cvmfunds.cdi_index(h["date"].min().date())
    long = [h[["date", "NAV"]].assign(series="Fund")]
    if not cdi.empty:
        c = cdi.reindex(pd.DatetimeIndex(h["date"])).ffill()
        base = c.dropna()
        if not base.empty:
            long.append(pd.DataFrame({"date": h["date"].values,
                                      "NAV": (c / base.iloc[0] * 100.0).values,
                                      "series": "CDI"}))
    plot = pd.concat(long, ignore_index=True).dropna(subset=["NAV"])
    brand.show_chart(
        alt.Chart(plot).mark_line()
        .encode(x=alt.X("date:T", title=None),
                y=alt.Y("NAV:Q", title="rebased to 100", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(domain=["Fund", "CDI"],
                                                range=[cc["accent"], cc["muted"]])),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                         alt.Tooltip("NAV:Q", format=",.1f")])
        .properties(height=300, title="NAV against CDI"))

    a1, a2 = st.columns(2)
    with a1:
        dd = pd.DataFrame({"date": h["date"],
                           "dd": (h["quota"] / h["quota"].cummax() - 1.0) * 100.0})
        brand.show_chart(alt.Chart(dd).mark_area(color=cc["short"], opacity=0.7)
                         .encode(x=alt.X("date:T", title=None),
                                 y=alt.Y("dd:Q", title="drawdown (%)"),
                                 tooltip=[alt.Tooltip("date:T"),
                                          alt.Tooltip("dd:Q", format=".2f")])
                         .properties(height=230, title="Drawdown"))
    with a2:
        _aum_bridge(h, row)

    _flow_chart(h)

    fl = h[["date", "subs", "redem", "pl"]].copy()
    fl["net"] = (fl["subs"].fillna(0) - fl["redem"].fillna(0)).cumsum() / _MM
    fl["assets"] = fl["pl"] / _MM
    melted = fl.melt("date", ["net", "assets"], var_name="series", value_name="v")
    melted["series"] = melted["series"].map({"net": "Cumulative net flow", "assets": "Assets"})
    brand.show_chart(alt.Chart(melted).mark_line()
                     .encode(x=alt.X("date:T", title=None),
                             y=alt.Y("v:Q", title="R$m", scale=alt.Scale(zero=False)),
                             color=alt.Color("series:N", title=None,
                                             scale=alt.Scale(range=[cc["series"], cc["ink"]])),
                             tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                                      alt.Tooltip("v:Q", format=",.0f")])
                     .properties(height=230, title="Assets and cumulative net flow"))


def _tab_fund(met: pd.DataFrame) -> None:
    row = _fund_picker(met)
    if row is None:
        return
    st.divider()
    _fund_detail(met, row, key="fnd")


# The return ladder. Each window carries its excess over CDI underneath, because in Brazil
# the raw number alone does not say whether a fund was worth owning — CDI compounded 14.6%
# over the last twelve months, so +14% is a loss against cash.
_LADDER = [("1d", "ret_1d", "exc_1d"), ("1w", "ret_1w", "exc_1w"),
           ("1m", "ret_1m", "exc_1m"), ("3m", "ret_3m", "exc_3m"),
           ("6m", "ret_6m", "exc_6m"), ("YTD", "ret_ytd", "exc_ytd"),
           ("12m", "ret_12m", "exc_12m")]


def _ladder(row: pd.Series) -> None:
    """Seven windows across, each with its excess over cash beneath it."""
    cc = brand.chart_colors()
    for col, (label, ret_key, exc_key) in zip(st.columns(len(_LADDER)), _LADDER):
        val = row.get(ret_key)
        dp = 2 if label == "1d" else 1
        col.metric(label, _pct(val, dp))
        if exc_key and exc_key in row.index and row.get(exc_key) == row.get(exc_key):
            exc = row[exc_key]
            tone = cc["long"] if exc > 0 else cc["short"]
            col.markdown(
                f"<div style='margin-top:-.7rem;font-size:.72rem;color:{tone}'>"
                f"{_pct(exc, dp)} vs CDI</div>", unsafe_allow_html=True)
        elif exc_key:
            col.markdown("<div style='margin-top:-.7rem;font-size:.72rem;color:#8b929c'>"
                         "— vs CDI</div>", unsafe_allow_html=True)


def _peer_rank(met: pd.DataFrame, row: pd.Series) -> str:
    """Where this class sits inside its own ANBIMA strategy over 12 months.

    "+12%" means nothing on its own when the whole category did 14%. The peer set is the
    same strategy on the same defensible basis — ex-feeder, ex-exclusive — so the rank is
    against products, not against a pile of duplicate feeder classes.
    """
    strat, mine = row.get("strategy_en"), row.get("ret_12m")
    if not strat or mine != mine:
        return ""
    peers = cvmfunds.screen(met, cvm_class=None, min_aum=0)
    peers = peers[(peers["strategy_en"] == strat) & peers["ret_12m"].notna()]
    if len(peers) < 5:
        return ""
    better = int((peers["ret_12m"] > mine).sum())
    pct = (1 - better / len(peers)) * 100
    return (f"**{better + 1}** of **{len(peers)}** in {strat} over 12m "
            f"({pct:.0f}th percentile)")


def _aum_bridge(h: pd.DataFrame, row: pd.Series) -> None:
    """Where the change in capital actually came from.

    Assets move for two completely different reasons — the market, and investors handing
    money over or taking it back — and a single assets line cannot tell you which. A fund
    can grow while bleeding clients, which is the case worth spotting. This splits the
    window's change into the two, and they sum to it by construction.
    """
    fl = h.dropna(subset=["pl"]).copy()
    if len(fl) < 2:
        return
    start_pl, end_pl = float(fl["pl"].iloc[0]), float(fl["pl"].iloc[-1])
    net_flow = float((fl["subs"].fillna(0) - fl["redem"].fillna(0)).sum())
    # Performance is the RESIDUAL, not a separate estimate: whatever the change in assets
    # is not explained by money moving in or out is what the market did to it.
    perf = (end_pl - start_pl) - net_flow

    bars = pd.DataFrame({
        "part": ["Start", "Net flow", "Performance", "End"],
        "value": [start_pl / _MM, net_flow / _MM, perf / _MM, end_pl / _MM],
        "kind": ["level", "flow", "perf", "level"],
    })
    cc = brand.chart_colors()
    chart = (alt.Chart(bars).mark_bar()
             .encode(x=alt.X("part:N", sort=None, title=None),
                     y=alt.Y("value:Q", title="R$m"),
                     color=alt.Color("kind:N", legend=None,
                                     scale=alt.Scale(domain=["level", "flow", "perf"],
                                                     range=[cc["muted"], cc["series"],
                                                            cc["accent"]])),
                     tooltip=[alt.Tooltip("part:N", title=""),
                              alt.Tooltip("value:Q", title="R$m", format=",.0f")])
             .properties(height=230, title="Where the capital came from"))
    brand.show_chart(chart)
    grew = end_pl >= start_pl
    st.caption(_md(
        f"Over the cached window assets went {'up' if grew else 'down'} "
        f"**{_brl(abs(end_pl - start_pl), 'm', 0)}** — "
        f"**{_brl(net_flow, 'm', 0)}** of investor money "
        f"{'in' if net_flow >= 0 else 'out'}, **{_brl(perf, 'm', 0)}** from performance."
        + ("  Growing while investors withdraw."
           if grew and net_flow < 0 else
           "  Shrinking despite money coming in." if not grew and net_flow > 0 else "")))


def _tab_watchlist(met: pd.DataFrame) -> None:
    keys = cvmfunds.watchlist()
    if not keys:
        st.info("Nothing starred yet. Tick rows in **Explore** and press "
                "“★ Add to watchlist”, or star a fund from its own page.")
        return
    all_funds = cvmfunds.screen(met, cvm_class=None, min_aum=0,
                               include_feeders=True, include_exclusive=True,
                               include_prev=True)
    ident = _fund_keys(all_funds)
    d = all_funds[ident.isin(keys)].copy()
    # Order follows the watchlist, newest star first, rather than whatever the store held.
    d["_rank"] = _fund_keys(d).map({k: i for i, k in enumerate(keys)})
    d = d.sort_values("_rank")

    missing = len(keys) - len(d)
    st.caption(f"**{len(d)}** starred · assets and flows in **R$m**"
               + (f" · {missing} no longer in the store (wound up, or below the size floor)"
                  if missing else ""))
    colset = st.segmented_control("Columns", _COLSETS, default=_COLSETS[0], key="wl_cols",
                                  label_visibility="collapsed") or _COLSETS[0]
    disp, spec, moves = _explore_funds(d, colset)
    sel = brand.themed_dataframe(_as_text(disp, spec), {}, height=420,
                                 colorers=[(moves, _move_colour)],
                                 on_select="rerun", selection_mode="multi-row",
                                 key=f"wl_tbl_{colset}")
    rows = _selected_rows(sel)
    c1, c2 = st.columns([1.1, 3])
    if rows and c1.button(f"Remove {len(rows)}", key="wl_drop", use_container_width=True):
        drop = set(_fund_keys(d.iloc[rows]))
        cvmfunds.watch_set([k for k in keys if k not in drop])
        st.rerun()
    if not rows:
        c2.caption("Tick one row to chart that fund, or several to remove them.")
    elif len(rows) == 1:
        st.divider()
        _fund_detail(met, d.iloc[rows[0]], key="wl")


# ── page ────────────────────────────────────────────────────────────────────────────
def render() -> None:
    st.subheader("🇧🇷 Brazil Funds")
    st.caption("Every regulated Brazilian fund files a **daily** report with the CVM — NAV "
               "per share, net assets, subscriptions, redemptions, holders — and the CVM "
               "republishes it as free bulk data. No vendor, no size threshold, no "
               "quarterly guessing. This is that file, joined to the fund registry so each "
               "class carries its manager, strategy and investor type — shown in English. "
               "**Multi-strategy** (CVM's *multimercado*) is Brazil's hedge-fund analogue "
               "and the default filter.")

    met, meta = _store()
    if met.empty:
        st.warning("No CVM fund store yet. It builds with the daily pull, or press "
                   "“Rebuild now” below — the first build downloads ~13 months of daily "
                   "filings (~160MB) and takes several minutes.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("As of", meta.get("as_of") or "—",
              help="The last date on which the INDUSTRY reported, not the newest row in the "
                   "file. Administrators have one business day to file and use it unevenly, "
                   "so the newest date carries only the fastest filers — on one drop it held "
                   "9 classes out of 25,162.")
    c2.metric("Share classes", f"{meta.get('n_units', 0):,}")
    c3.metric("Managers", f"{meta.get('n_gestores', 0):,}")
    c4.metric("Store built", (meta.get("built") or "—")[:16].replace("T", " "))

    if auth.is_admin():
        if st.button("🔄 Rebuild now", key="cvm_rebuild",
                     help="Re-download the registry and the last two months of daily "
                          "filings, then recompute. The daily pull normally does this."):
            with st.spinner("Rebuilding the CVM fund store — this takes a few minutes…"):
                try:
                    cvmfunds.build()
                    _store.clear()
                    st.success("Rebuilt.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rebuild failed — {type(exc).__name__}: {exc}")

    if met.empty:
        return

    st.divider()
    # A segmented control, not st.tabs: the same in-page section switcher Company
    # Fundamentals and Colleague Access use. Tabs mark the active one with a thin
    # underline that reads as decoration rather than as "you are here".
    view = st.segmented_control("Section", _VIEWS, default=_VIEWS[0], key="cvm_view",
                                label_visibility="collapsed")
    if view == _VIEWS[1]:
        _tab_fund(met)
    elif view == _VIEWS[2]:
        _tab_watchlist(met)
    else:                      # clicking the active segment deselects it — stay put
        _tab_explore(met)

    st.divider()
    st.caption("Source: **CVM — Portal Dados Abertos** (daily filings + the fund/class/"
               "subclass registry), free and unlicensed · benchmark **CDI** from BCB SGS 12. "
               "Classifications are translated from the Portuguese; fund and manager names "
               "are trimmed to what identifies them, never translated, and the Fund tab "
               "carries the full registered name and CNPJ. Offshore feeders — the Cayman and "
               "Luxembourg vehicles where much of the foreign money sits — do not file with "
               "CVM, so onshore assets understate the big global-macro houses. Nothing here "
               "is a recommendation.")
