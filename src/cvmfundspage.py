"""cvmfundspage.py — the 🇧🇷 Brazil Funds page.

The screener over CVM's daily fund filings. Engine, methodology and every honest-data
rule: src/cvmfunds.py. This module is presentation only — nothing here computes a
return, because everything is precomputed into data/signals/cvm_funds by the daily pull
(app-wide rule: a page must not do 7m rows of work on open).

Four tabs:

  Screener   the fund table, filtered. The default filter is the defensible one —
             multi-strategy, ex-feeder, ex-exclusive, ex-pension — and the page says so
             above the table rather than leaving the reader to assume a raw dump.
  Managers   the league table, and the reconciliation block that shows what each
             counting basis is worth. The gap between bases is bigger than the gap
             between the top three managers, which is the point.
  Flows      who raised and who bled. A performance table cannot answer this, and for
             a broker it is usually the more useful question.
  Fund       one share class: NAV against CDI, drawdown, assets and flows.

EVERYTHING DESCRIPTIVE IS SHOWN IN ENGLISH

CVM publishes in Portuguese. Classes, ANBIMA strategies and investor types are
translated (src/cvmfunds.py holds the vocabulary), and fund and manager names are tidied
down to the part that identifies them — the legal boilerplate every Brazilian fund
carries says nothing about which fund it is. Names are never TRANSLATED, only trimmed,
and the Fund tab prints the full registered name and CNPJ, because that is what you
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
    return f"{v:{'+' if signed else ''}.{dp}f}%"


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

_FORMATTERS = {
    "pct": _pct,
    "pct_u": lambda v: _pct(v, signed=False),
    "pct0": lambda v: _pct(v, 0),
    "pctr": lambda v: _pct(v, 0, signed=False),
    "num": _num,
    "num1": lambda v: _num(v, 1),
    "num_s": lambda v: _num(v, signed=True),
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
def _tab_screener(met: pd.DataFrame) -> None:
    d, opts = _filters(met, "scr")
    if d.empty:
        st.info("No funds match that filter.")
        return
    _basis_line(opts, d["cnpj"].nunique(), d["aum"].sum())

    sort_by = st.radio("Sort by", ["Assets", "12m return", "YTD return", "3m flow",
                                   "Vol (low first)", "Max drawdown"],
                       horizontal=True, key="scr_sort", label_visibility="collapsed")
    col, asc = {"Assets": ("aum", False), "12m return": ("ret_12m", False),
                "YTD return": ("ret_ytd", False), "3m flow": ("flow_3m", False),
                "Vol (low first)": ("vol", True), "Max drawdown": ("max_dd", False)}[sort_by]
    d = d.sort_values(col, ascending=asc, na_position="last").head(300)

    disp = pd.DataFrame({
        # A class that re-based its quota or amortised prints a ±50% day that is not a
        # market move. Its risk numbers are computed on clipped returns and are still
        # not trustworthy, so the row is MARKED rather than hidden — one such fund sits
        # in the ten largest multimercados and would otherwise read as a 69%-vol
        # blow-up next to Verde and Kapitalo.
        "Fund": np.where(d["glitch"].fillna(False), "⚠ ", "") + d["name_en"],
        "Manager": d["gestor_en"],
        "Strategy": d["strategy_en"],
        "Assets": d["aum"] / _MM,
        "1m": d["ret_1m"], "3m": d["ret_3m"], "12m": d["ret_12m"], "YTD": d["ret_ytd"],
        "%CDI 12m": d.get("cdi_12m"),
        "Vol": d["vol"], "Max DD": d["max_dd"],
        "Flow 3m": d["flow_3m"] / _MM,
        "Holders": d["holders"],
    })
    disp = _as_text(disp, {"Assets": NUM, "1m": PCT, "3m": PCT, "12m": PCT, "YTD": PCT,
                           "%CDI 12m": PCT_R, "Vol": PCT_U, "Max DD": PCT_U,
                           "Flow 3m": NUM_S, "Holders": NUM})
    n_flag = int(d["glitch"].fillna(False).sum())
    st.caption("Assets and 3m flow in **R$m**. Returns are net of fees — the published quota "
               "already is. **%CDI** is shown only where fund and benchmark are both positive."
               + (f"  ·  **⚠** marks the {n_flag} class(es) here that printed a single-day "
                  "quota move over 50% — a re-basing or an amortisation, not a market move. "
                  "Their volatility and drawdown are not comparable."
                  if n_flag else ""))
    brand.themed_dataframe(disp, {}, height=560,
                           colorers=[(["1m", "3m", "12m", "YTD", "Max DD", "Flow 3m"],
                                      _move_colour)])
    if len(d) == 300:
        st.caption("Showing the first 300 rows of the current filter — narrow it to see the rest.")


def _tab_managers(met: pd.DataFrame) -> None:
    d, opts = _filters(met, "mgr")
    if d.empty:
        st.info("No funds match that filter.")
        return
    by_firm = st.toggle(
        "Group the entities of one house together", value=False, key="mgr_firm",
        help="BTG Pactual runs three separately registered gestores and Itaú two. Off, "
             "each is its own row (CVM's unit, and what everything else here counts). On, "
             "they merge into one house — the question people actually ask, and how "
             "ANBIMA consolidates.")
    lt = cvmfunds.by_gestor(d, by_firm=by_firm)
    _basis_line(opts, d["cnpj"].nunique(), d["aum"].sum())

    tot = cvmfunds.industry_totals(met)
    if tot:
        st.markdown("**How much the counting basis is worth**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gross, all classes", _brl(tot["gross"], dp=0),
                  help="Every multimercado class summed. Wrong, and the number most often "
                       "quoted.")
        # ASCII hyphen, not U+2212: Streamlit parses the delta string to pick its arrow
        # direction, and a typographic minus reads as unparseable — which drew an UP
        # arrow on a number that is smaller than gross. delta_color="off" keeps it grey.
        m2.metric("Ex-feeder", _brl(tot["ex_feeder"], dp=0),
                  delta=f"-{tot['feeder_overstate']:.0f}% vs gross", delta_color="off",
                  help="Fund-of-quotas classes removed — each real pool counted once.")
        m3.metric("Ex-feeder, ex-pension", _brl(tot["ex_feeder_ex_prev"], dp=0),
                  help="The basis closest to an ANBIMA headline.")
        m4.metric("Pension carve-out", _brl(tot["prev"], dp=0),
                  help="Pension (*previdência*) multi-strategy, which ANBIMA counts as its "
                       "own category.")
        st.caption("ANBIMA published R$1,519bn of multimercado with Itaú Asset at 9.8%. "
                   "The third column here is the comparable basis; the residual gap is "
                   "ANBIMA consolidating economic groups — their one “BTG Pactual” is three "
                   "separately registered gestores in CVM's file — plus a vintage difference.")
        st.divider()

    top = lt.head(25).reset_index(drop=True)
    cc = brand.chart_colors()
    chart = (alt.Chart(top.assign(bn=top["aum"] / _BN,
                                  Manager=top["label"].str.slice(0, 44)))
             .mark_bar(color=cc["series"])
             .encode(x=alt.X("bn:Q", title="assets (R$bn)"),
                     y=alt.Y("Manager:N", sort="-x", title=None),
                     tooltip=[alt.Tooltip("Manager:N"),
                              alt.Tooltip("bn:Q", title="R$bn", format=",.1f"),
                              alt.Tooltip("share:Q", title="share %", format=".1f"),
                              alt.Tooltip("funds:Q", title="funds")])
             .properties(height=min(620, 26 * len(top)) or 100))
    brand.show_chart(chart)

    disp = pd.DataFrame({
        "Manager": lt["label"],
        "Assets": lt["aum"] / _BN,
        "Share": lt["share"],
        "Funds": lt["funds"],
        "12m": lt.get("ret_12m"), "YTD": lt.get("ret_ytd"), "Vol": lt.get("vol"),
        "Flow 3m": lt["flow_3m"] / _MM, "Flow 12m": lt["flow_12m"] / _MM,
        "Holders": lt["holders"],
    }).head(60)
    st.caption("Returns here are **asset-weighted** across each manager's funds — a simple "
               "average lets an R$8m launch outvote an R$8bn flagship. Assets in R$bn, "
               "flows in R$m.")
    disp = _as_text(disp, {"Assets": NUM_1, "Share": PCT_U, "Funds": NUM, "12m": PCT,
                           "YTD": PCT, "Vol": PCT_U, "Flow 3m": NUM_S,
                           "Flow 12m": NUM_S, "Holders": NUM})
    brand.themed_dataframe(
        disp, {}, height=520,
        colorers=[(["12m", "YTD", "Flow 3m", "Flow 12m"], _move_colour)])


def _tab_flows(met: pd.DataFrame) -> None:
    d, opts = _filters(met, "flw")
    if d.empty:
        st.info("No funds match that filter.")
        return
    win = st.radio("Window", ["1m", "3m", "12m"], index=1, horizontal=True, key="flw_win")
    col = f"flow_{win}"
    _basis_line(opts, d["cnpj"].nunique(), d["aum"].sum())
    st.caption("Net subscriptions minus redemptions, in **R$m**. This is the question a "
               "performance table cannot answer — a fund can be up 20% and bleeding assets, "
               "and the flow is the part that shows up as client activity.")

    net = d[col].sum()
    g = d.groupby("gestor_en")[col].sum().sort_values()
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Net flow, {win}", _brl(net, "m", 0))
    k2.metric("Managers raising", f"{int((g > 0).sum()):,}")
    k3.metric("Managers losing", f"{int((g < 0).sum()):,}")

    both = pd.concat([g.head(15), g.tail(15)]).drop_duplicates()
    cc = brand.chart_colors()
    frame = both.reset_index()
    frame.columns = ["Manager", "flow"]
    frame["mm"] = frame["flow"] / _MM
    frame["Manager"] = frame["Manager"].str.title().str.slice(0, 44)
    chart = (alt.Chart(frame).mark_bar()
             .encode(x=alt.X("mm:Q", title=f"net flow, {win} (R$m)"),
                     y=alt.Y("Manager:N", sort="-x", title=None),
                     color=alt.condition(alt.datum.mm > 0, alt.value(cc["long"]),
                                         alt.value(cc["short"])),
                     tooltip=[alt.Tooltip("Manager:N"),
                              alt.Tooltip("mm:Q", title="R$m", format="+,.0f")])
             .properties(height=min(680, 24 * len(frame)) or 100))
    brand.show_chart(chart)

    st.markdown(f"**Biggest single-fund moves, {win}**")
    d2 = d.sort_values(col, na_position="last")
    ends = pd.concat([d2.tail(12).iloc[::-1], d2.head(12)])
    disp = pd.DataFrame({
        "Fund": ends["name_en"], "Manager": ends["gestor_en"],
        "Assets": ends["aum"] / _MM, f"Flow {win}": ends[col] / _MM,
        "% of assets": ends[col] / ends["aum"].replace(0, np.nan) * 100.0,
        "12m": ends["ret_12m"], "Holders": ends["holders"],
    })
    disp = _as_text(disp, {"Assets": NUM, f"Flow {win}": NUM_S, "% of assets": PCT_0,
                           "12m": PCT, "Holders": NUM})
    brand.themed_dataframe(
        disp, {}, height=460,
        colorers=[([f"Flow {win}", "% of assets", "12m"], _move_colour)])


def _tab_fund(met: pd.DataFrame) -> None:
    d = cvmfunds.screen(met, cvm_class=None, min_aum=50 * _MM)
    if d.empty:
        st.info("Nothing in the store yet.")
        return
    d = d.sort_values("aum", ascending=False)
    labels = (d["name_en"] + "  ·  " + d["gestor_en"]
              + d["subclass"].apply(lambda s: f"  ·  {s}" if s else ""))
    pick = st.selectbox("Fund", labels.tolist()[:4000], index=0, key="fnd_pick")
    row = d.iloc[labels.tolist()[:4000].index(pick)]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Assets", _brl(row["aum"], "m", 0))
    m2.metric("12m", _pct(row["ret_12m"]))
    m3.metric("Vol (ann.)", _pct(row["vol"], signed=False))
    m4.metric("Max drawdown", _pct(row["max_dd"], signed=False))
    m5.metric("Holders", f"{row['holders']:,.0f}" if row["holders"] == row["holders"] else "—")
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
    chart = (alt.Chart(plot).mark_line()
             .encode(x=alt.X("date:T", title=None),
                     y=alt.Y("NAV:Q", title="rebased to 100", scale=alt.Scale(zero=False)),
                     color=alt.Color("series:N", title=None,
                                     scale=alt.Scale(domain=["Fund", "CDI"],
                                                     range=[cc["accent"], cc["muted"]])),
                     tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                              alt.Tooltip("NAV:Q", format=",.1f")])
             .properties(height=300, title="NAV against CDI"))
    brand.show_chart(chart)

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
        fl = h[["date", "subs", "redem", "pl"]].copy()
        fl["net"] = (fl["subs"].fillna(0) - fl["redem"].fillna(0)).cumsum() / _MM
        fl["assets"] = fl["pl"] / _MM
        melted = fl.melt("date", ["net", "assets"], var_name="series", value_name="v")
        melted["series"] = melted["series"].map({"net": "Cumulative net flow",
                                                 "assets": "Assets"})
        brand.show_chart(alt.Chart(melted).mark_line()
                         .encode(x=alt.X("date:T", title=None),
                                 y=alt.Y("v:Q", title="R$m", scale=alt.Scale(zero=False)),
                                 color=alt.Color("series:N", title=None,
                                                 scale=alt.Scale(range=[cc["series"],
                                                                        cc["ink"]])),
                                 tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                                          alt.Tooltip("v:Q", format=",.0f")])
                         .properties(height=230, title="Assets and cumulative net flow"))


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
    tabs = st.tabs(["🔎 Screener", "🏦 Managers", "💸 Flows", "📈 Fund"])
    with tabs[0]:
        _tab_screener(met)
    with tabs[1]:
        _tab_managers(met)
    with tabs[2]:
        _tab_flows(met)
    with tabs[3]:
        _tab_fund(met)

    st.divider()
    st.caption("Source: **CVM — Portal Dados Abertos** (daily filings + the fund/class/"
               "subclass registry), free and unlicensed · benchmark **CDI** from BCB SGS 12. "
               "Classifications are translated from the Portuguese; fund and manager names "
               "are trimmed to what identifies them, never translated, and the Fund tab "
               "carries the full registered name and CNPJ. Offshore feeders — the Cayman and "
               "Luxembourg vehicles where much of the foreign money sits — do not file with "
               "CVM, so onshore assets understate the big global-macro houses. Nothing here "
               "is a recommendation.")
