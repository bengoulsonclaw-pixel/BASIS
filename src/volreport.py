"""Daily Volatility Report — a branded, client-style PDF whose hero is a VISUAL
of the implied-vs-realized spread across the book:
  1. Scatter — implied (y) vs realized (x) with the 45° fair line; a dot's
     distance from the line IS its vol-risk premium (the spread).
  2. Diverging bar — most stretched markets ranked by the spread's 1-yr z-score.
Plus a compact table of the flagged opportunities. (Skew has its own report:
src/skewreport.py.)

Run standalone (the app calls it as a subprocess):
    python src/volreport.py data/signals/volatility.parquet out.pdf --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

from reportkit import (pretty_date, data_uri, png, color, legend, bar_png, flagged, reflag, render_pdf,
                       BLACK, RICH, CHEAP, NEUTRAL, WARN, WARN_TX, WARN_SIGNALS, snapshot_stamp,
                       ai_rewrite as _ai_rewrite, md_bold as _md_bold, ordinal as _ord,
                       when_phrase as _when, history_panel as _hist_panel, last_episode as _last_ep)
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
ROOT = Path(__file__).resolve().parent.parent


def scatter_png(df: pd.DataFrame) -> str:
    d = df.dropna(subset=["iv", "rv"])
    fig, ax = plt.subplots(figsize=(6.1, 4.0))   # a touch shorter so the swaps block fits page 1
    hi = float(max(d["iv"].max(), d["rv"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "implied = realized", rotation=45, ha="right",
            va="bottom", fontsize=7, color="#555", rotation_mode="anchor",
            transform_rotates_text=True)   # rotation in DATA space → text hugs the 45° line
    # Whole book plotted; FLAGGED markets highlighted (coloured + named), gold-watch
    # markets (z at trigger, sign unconfirmed) mid-weight, the rest de-emphasised grey.
    is_warn = d["signal"].isin(WARN_SIGNALS) if "signal" in d.columns else pd.Series(False, index=d.index)
    neu = d[(d["direction"] == 0) & ~is_warn]
    wn = d[(d["direction"] == 0) & is_warn]
    fl = d[d["direction"] != 0]
    ax.scatter(neu["rv"], neu["iv"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    if not wn.empty:
        ax.scatter(wn["rv"], wn["iv"], s=42, c=WARN, alpha=0.95, edgecolor="white",
                   linewidth=0.6, zorder=3)
        for i, r in enumerate(wn.itertuples(index=False)):
            ax.annotate(r.market, (r.rv, r.iv), fontsize=5.8, color=WARN_TX,
                        xytext=(4, 3 if i % 2 == 0 else -8), textcoords="offset points")
    ax.scatter(fl["rv"], fl["iv"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.rv, r.iv), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Rich — sell vol", "Cheap — buy vol",
           "At trigger — sign unconfirmed" if not wn.empty else None)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Realized vol  (1M, ann. %)")
    ax.set_ylabel("Implied vol  (1M ATM, %)")
    ax.set_title("Above the line → options rich (sell vol);  below → cheap (buy vol)")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return png(fig)


def iv_rv_png(df: pd.DataFrame, n: int = 18) -> str:
    """Dumbbell chart of the FLAGGED markets: a grey dot for realized and a
    signal-coloured dot for implied, joined by a line whose length is the spread —
    so the reader sees each opportunity's implied-vs-realized gap at a glance."""
    d = flagged(df)
    if d.empty:
        return ""
    if len(d) > n:                                   # keep it readable — most stretched
        d = d.reindex(d["z"].abs().sort_values(ascending=False).index).head(n)
    d = d.sort_values("spread")                      # cheap (neg spread) bottom → rich top
    y = list(range(len(d)))
    iv, rv, dirn = d["iv"].to_numpy(), d["rv"].to_numpy(), d["direction"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.1, max(1.7, 0.36 * len(d) + 0.6)))
    for i in y:
        ax.plot([rv[i], iv[i]], [i, i], color="#CFCFCF", lw=1.6, zorder=1)
    ax.scatter(rv, y, s=46, color=NEUTRAL, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(iv, y, s=46, color=[color(x) for x in dirn], edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["market"], fontsize=6.8)
    ax.set_ylim(-0.6, len(d) - 0.4)        # uniform half-row padding → no sparse stretch when few flagged
    ax.set_xlabel("annualised vol (%)")
    ax.set_title("Flagged markets — realized → implied (the line is the spread)")
    ax.grid(True, axis="x", color="#EEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=NEUTRAL, mec="white", ms=8, label="Realized"),
               Line2D([0], [0], marker="o", ls="", mfc=RICH, mec="white", ms=8, label="Implied — rich (sell vol)"),
               Line2D([0], [0], marker="o", ls="", mfc=CHEAP, mec="white", ms=8, label="Implied — cheap (buy vol)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=6.3, frameon=False)
    return png(fig)


def _hist_note(r, g: pd.DataFrame, asof: pd.Timestamp) -> str:
    """Deterministic **bold**-marked caption facts for one flagged market — every claim
    computed from the cached history (levels, the last comparable episode and what price
    and realized were doing then, what is driving the gap now, the seasonal window).
    The AI rewrite may compress but can only use what is stated here."""
    rich = int(r.direction) < 0
    s = g.set_index(pd.to_datetime(g["date"]))
    spread, px = s["spread"].dropna(), s["price"].dropna()
    bits = [f"Implied is **{r.iv:.1f}** against **{r.rv:.1f}** realized — a spread of "
            f"**{r.spread:+.1f}** vol points"
            + (f", the **{_ord(int(r.pctl))}** percentile of the past year" if pd.notna(r.pctl) else "")
            + "."]

    # The last time the spread sat here (excluding the current ~month-long episode).
    hit = _last_ep(spread, high_side=rich)
    if hit:
        dt = hit[0]
        ep = [f"It was last this {'wide' if rich else 'depressed'} in **{_when(dt, asof)}**"]
        try:                                     # what the market was doing around that episode
            pin = px.loc[:dt]
            mv = (float(pin.iloc[-1]) / float(pin.iloc[-22]) - 1.0) * 100.0
            if abs(mv) >= 1.0:
                ep.append(f"when the underlying had moved **{mv:+.0f}%** in a month")
        except Exception:
            pass
        if "rv" in s.columns:
            try:
                ep.append(f"with realized running **{float(s['rv'].dropna().loc[:dt].iloc[-1]):.0f}**")
            except Exception:
                pass
        bits.append(", ".join(ep) + ".")
    else:
        bits.append(f"That is its most {'stretched' if rich else 'depressed'} reading of the past year.")

    # What is driving the gap NOW — which leg moved over the past month, and the price move.
    now = []
    if "iv" in s.columns and "rv" in s.columns:
        try:
            iv_s, rv_s = s["iv"].dropna(), s["rv"].dropna()
            div, drv = float(iv_s.iloc[-1] - iv_s.iloc[-22]), float(rv_s.iloc[-1] - rv_s.iloc[-22])
            now.append(f"over the past month implied has moved **{div:+.1f}** points against "
                       f"**{drv:+.1f}** for realized")
        except Exception:
            pass
    try:
        mv_now = (float(px.iloc[-1]) / float(px.iloc[-22]) - 1.0) * 100.0
        if abs(mv_now) >= 1.0:
            now.append(f"with the underlying **{mv_now:+.0f}%** over the same stretch")
    except Exception:
        pass
    if now:
        bits.append("This time, " + " ".join(now) + ".")
    if getattr(r, "seasonal", ""):
        bits.append(f"It is also in its seasonal weather window ({r.seasonal}), when elevated "
                    "implied partly prices genuine crop risk.")
    return " ".join(bits)


def history_sections(df: pd.DataFrame, hist: pd.DataFrame, asof, ai: bool = True) -> list:
    """ALL flagged markets as [{img, cap}] — a 1-year dual-axis panel each, with a 1-2
    sentence data-grounded caption (Fable-polished when `ai`; the deterministic facts
    otherwise or on any model failure)."""
    if hist is None or len(hist) == 0:
        return []
    fl = flagged(df)
    if fl.empty:
        return []
    asof_ts = pd.to_datetime(asof, errors="coerce")
    asof_ts = asof_ts if pd.notna(asof_ts) else pd.Timestamp.today()
    have = set(hist["ticker"].unique())
    picks = fl.reindex(fl["z"].abs().sort_values(ascending=False).index)
    rows = [r for r in picks.itertuples(index=False) if r.ticker in have]
    groups = [hist[hist["ticker"] == r.ticker].sort_values("date") for r in rows]
    notes = [_hist_note(r, g, asof_ts) for r, g in zip(rows, groups)]
    caps = _ai_rewrite(notes, _CAPTION_SYSTEM) if ai else notes
    from reportkit import history_fill_height
    _h = history_fill_height(len(rows))   # pages fill evenly; never one lone panel on a page
    return [{"img": _hist_panel(r.market, int(r.direction), g, "spread", height=_h), "cap": _md_bold(c)}
            for r, g, c in zip(rows, groups, caps)]


# Shared with the skew/term/COT report openers (moved to reportkit 2026-07-21).
from reportkit import num_word as _num, join_names as _join_names, names_by_z as _names_by_z


def build_synopsis(df: pd.DataFrame, threshold: float = 1.5, seasonal=None) -> str:
    """Plain-language executive summary of the day's flags, built from the data.
    Returns an HTML fragment (rendered with |safe); market names are escaped.
    `seasonal` is the set of tickers currently in their seasonal weather window — a
    rich reading there partly prices genuine crop-weather risk, so we flag that."""
    seasonal = set(seasonal or ())
    fl = df[df["direction"] != 0]
    cheap, rich = fl[fl["direction"] > 0], fl[fl["direction"] < 0]
    nflag, nc, nr = len(fl), len(cheap), len(rich)
    if nflag == 0:
        return ("Nothing stands out today: implied and realized volatility are broadly "
                "aligned across the book, with no market's spread stretched beyond its "
                "own one-year norm.")
    # Findings only — the punchline. The screen's mechanics live in the yellow
    # "How the screen works" box in the template, so the opener stays readable.
    lead = f"<b>{_num(nflag).capitalize()} market{'s' if nflag != 1 else ''} stand{'s' if nflag == 1 else ''} out today.</b>"
    out = [lead]
    if nc:
        sentence = (
            f"{_num(nc).capitalize()} {'looks' if nc == 1 else 'look'} potentially "
            f'<span class="cheap">cheap</span>, with implied sitting below the volatility '
            f"the market has recently realized: <b>{_join_names(_names_by_z(cheap))}</b>.")
        season_c = _names_by_z(cheap[cheap["ticker"].isin(seasonal)])
        if season_c:
            whose = "its" if len(season_c) == 1 else "their"
            window = "window" if len(season_c) == 1 else "windows"
            sentence += (
                f" Notably, <b>{_join_names(season_c)}</b> screen{'s' if len(season_c) == 1 else ''} "
                f"cheap even with {whose} seasonal crop-weather {window} (&#9788;) open — implied "
                f"sitting low into a period that has historically carried genuine weather risk.")
        out.append(sentence)
    if nr:
        sentence = (
            f"{_num(nr).capitalize()} {'looks' if nr == 1 else 'look'} potentially "
            f'<span class="rich">rich</span>, with implied sitting above recently realized '
            f"volatility: <b>{_join_names(_names_by_z(rich))}</b>.")
        season = _names_by_z(rich[rich["ticker"].isin(seasonal)])
        if season:
            whose = "its" if len(season) == 1 else "their"
            window = "window" if len(season) == 1 else "windows"
            scope = "" if len(season) == nr else " in particular"
            sentence += (
                f" For <b>{_join_names(season)}</b>{scope}, that largely reflects {whose} "
                f"seasonal crop-weather {window} (&#9788;), where elevated implied volatility "
                f"prices genuine weather risk rather than necessarily a mispricing.")
        out.append(sentence)
    # 5-day caveat, only where it changes how a FLAG should be read: a cheap name
    # whose delivered vol is fading (the event is rolling out of the 1M window), or
    # a rich name where realizing is accelerating past the 1M number.
    if "rv_state" in fl.columns:
        fade = fl[(fl["direction"] > 0) & (fl["rv_state"] == "decay")]
        heat = fl[(fl["direction"] < 0) & (fl["rv_state"] == "heat")]
        for sub, verb in ((fade, "fade"), (heat, "heat")):
            if len(sub) == 0:
                continue
            names = _names_by_z(sub)
            if verb == "fade":
                if len(sub) == 1:
                    r0 = sub.iloc[0]
                    detail = (f"the last five sessions have realized only {r0.rv5:.0f} against "
                              f"the month's {r0.rv:.0f}")
                else:
                    detail = ("the last five sessions have realized less than half the "
                              "one-month number")
                out.append(
                    f"One caveat (&#9662;): for <b>{_join_names(names)}</b> the delivered number is "
                    f"fading — {detail}, so part of the cheap reading reflects an event rolling "
                    f"out of the one-month window rather than live turbulence.")
            else:
                out.append(
                    f"Note (&#9652;) that <b>{_join_names(names)}</b> {'is' if len(names) == 1 else 'are'} "
                    f"realizing far faster over the past five sessions than the one-month number shows — "
                    f"the rich reading may partly be options pricing the new regime, not excess premium.")
    out.append("They are highlighted purely as markets where the implied/realized "
               "relationship currently looks stretched, and may be worth a closer look.")
    return " ".join(out)


# ---------------------------------------------------------------------------
# Desk comment on the swaps table — a deterministic template note, rewritten into
# a natural voice by Claude (ai_polish.py run by the Morning Coffee interpreter —
# the same machinery as the Technical Analysis report). Falls back to the template
# on ANY failure, so generation never depends on the model being reachable.
# ---------------------------------------------------------------------------
_COMMENT_SYSTEM = (
    "You are a senior volatility strategist adding a short comment to the 'Possible Vol "
    "Swaps' table in a daily client report. The input is a JSON array holding ONE terse "
    "machine-built note summarising today's table; rewrite it so it reads like a person "
    "talking a client through what stands out today — 2-3 flowing, conversational but "
    "professional sentences, 70 words maximum (a seasoned desk analyst's voice, not a "
    "data read-out; it sits above a tight table, so brevity matters).\n"
    "HARD RULES — never break these:\n"
    "(1) Never invent a product, figure or relationship that is not in the input. Any "
    "figure you state must be copied EXACTLY as given, wrapped in the same **bold** "
    "markers. You may compress or summarise the secondary names for flow, but keep the "
    "standout pairing(s) with their numbers.\n"
    "(2) Neutral and observational only — this is client-safe commentary, NOT advice. "
    "Never say buy, sell, long, short, recommend, or imply the reader should act; "
    "'may be worth a closer look' is as far as you go.\n"
    "(3) Plain English, no exclamation marks, no headline, no greeting.\n"
    "Return ONLY a JSON array with the single rewritten string — nothing else.")


def _swaps_template(rv_peers: list) -> str:
    """Deterministic one-note summary of the swaps table — the AI's raw material AND the
    no-model fallback. Numbers/names carry **bold** markers the rewrite must keep."""
    side = lambda cls: "rich" if cls == "rich" else "cheap"
    pairs, seen = [], set()
    for row in rv_peers:
        for p in row["peers"]:
            if not p["opp"]:
                continue
            key = frozenset({row["market"], p["name"]})
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                f"**{row['market']}** (flagged {side(row['zcls'])}, z **{row['z']}**) against "
                f"**{p['name']}** (z **{p['z']}**, correlated **{p['corr']}** over the year, "
                f"**{p['corr1m']}** over the past month)")
    same_way = [r["market"] for r in rv_peers if not any(p["opp"] for p in r["peers"])]
    if pairs:
        note = ("Divergent pairing" + ("s" if len(pairs) > 1 else "") + " today: "
                + "; ".join(pairs) + ".")
        if same_way:
            note += (" The rest — " + ", ".join(f"**{m}**" for m in same_way) + " — are stretched "
                     "the same way as their closest relations, so they offer no natural offset.")
    else:
        note = ("No divergent pairings today: every flagged market's closest relations are "
                "stretched the same way it is, so nothing in the group offers a natural offset.")
    return note


def _ai_comment(note: str) -> str:
    return _ai_rewrite([note], _COMMENT_SYSTEM)[0]


_CAPTION_SYSTEM = (
    "You are a senior volatility strategist writing the one-to-two sentence caption under each "
    "market's spread-history chart in a daily client report (the chart shows the implied-minus-"
    "realized vol spread over the past year against the underlying price). The input is a JSON "
    "array of terse machine-built notes, one per market, in order; rewrite EACH as a flowing, "
    "conversational but professional caption — a desk analyst pointing out what the chart shows.\n"
    "HARD RULES — never break these:\n"
    "(1) Every figure, date and product name must be copied EXACTLY as given, wrapped in the same "
    "**bold** markers. NEVER invent an event, cause, or figure not in the note — you may only "
    "describe what the note states (levels, when the spread was last here, what price/vol were "
    "doing then and now, a seasonal window). No outside knowledge.\n"
    "(2) Neutral and observational — client-safe, NOT advice: never say buy, sell, long, short, "
    "recommend, or imply the reader should act.\n"
    "(3) 1-2 sentences per caption, vary the openings, no exclamation marks.\n"
    "Return ONLY a JSON array of strings, the same length and order as the input — nothing else.")


def _sd_str(v, dec) -> str:
    """The 1-day 1σ price move formatted to the contract's native decimals, '' if n/a."""
    try:
        if v is None or not pd.notna(v):
            return ""
        return f"{float(v):.{int(dec)}f}"
    except Exception:
        return ""


def render_html(df: pd.DataFrame, asof: str, threshold: float = 1.5, hist: pd.DataFrame = None,
                corr_threshold: float = 0.6, corr_metric: str = "realized", ai: bool = True,
                stir: pd.DataFrame = None, stir_hist: pd.DataFrame = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Rich — sell vol", -1), ("Cheap — buy vol", 1), gate_col="spread")
    try:                                   # ag weather-premium flag, keyed to the report's as-of month
        from seasonal import weather_note as _wx
    except Exception:
        def _wx(_t, _m): return ""
    _mo = pd.to_datetime(asof, errors="coerce")
    _mo = _mo.month if pd.notna(_mo) else pd.Timestamp.today().month
    seasonal_set = {r.ticker for r in df.itertuples(index=False) if _wx(r.ticker, _mo)}
    df = df.assign(seasonal=df["ticker"].map(lambda t: _wx(t, _mo)))   # history captions read this
    def _div_gap(r):
        """Signed ours-vs-surface gap when today's implied is our own curve AND the two
        constructions disagree materially (>=3 vols) — the †-marked rows."""
        v = getattr(r, "vs_bbg", float("nan"))
        if getattr(r, "src", "") == "own" and pd.notna(v) and abs(float(v)) >= 3.0:
            return f"{float(v):+.1f}"
        return ""

    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "iv": f"{r.iv:.1f}", "rv": f"{r.rv:.1f}", "spread": f"{r.spread:+.1f}",
        "iv_sd": _sd_str(getattr(r, "iv_sd", None), getattr(r, "px_dec", 1)),
        "rv_sd": _sd_str(getattr(r, "rv_sd", None), getattr(r, "px_dec", 1)),
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
        "seasonal": _wx(r.ticker, _mo),
        "div": _div_gap(r),
        "warn": r.signal in WARN_SIGNALS,
        "rv5": (f"{r.rv5:.1f}" if pd.notna(getattr(r, "rv5", float("nan"))) else "—"),
        "rvmark": {"decay": "▾", "heat": "▴"}.get(getattr(r, "rv_state", ""), ""),
        "rvpct": (f"{100 * r.rv5 / r.rv:.0f}%"
                  if getattr(r, "rv_state", "") and pd.notna(getattr(r, "rv5", float("nan"))) and r.rv > 0
                  else ""),
    } for r in df.dropna(subset=["z"]).sort_values("z", ascending=False).itertuples(index=False)]

    # legend for the † rows — spelled out with the signed gap, since a printed dagger
    # carries no tooltip; and the own-curve coverage count for the provenance line
    own_div = [(r["market"], r["div"]) for r in rows if r["div"]]
    n_own = int((df.get("src", pd.Series(dtype=str)) == "own").sum())

    # Every open seasonal window, spelled out for the table legend (the ☼ in a printed
    # PDF has no hover-tooltip, so the label must be visible text).
    seasonal_open = []
    for r in rows:
        if r["seasonal"] and (r["market"], r["seasonal"]) not in seasonal_open:
            seasonal_open.append((r["market"], r["seasonal"]))

    # Relative-value peers — co-movement over BOTH windows (1y picks the peers at the fixed
    # threshold; the 1m twin is shown beside it so a recently-loosened link is visible).
    rv_peers = []
    _corr_lbl = {"iv": "IV-change", "realized": "realized-vol"}.get(corr_metric, "return")
    _corr_moves = {"iv": "implied vol", "realized": "realized vol"}.get(corr_metric, "price")
    try:
        import volcorr
        _cm = volcorr.load(corr_metric)
        _cm1m = volcorr.load(corr_metric, "1m")
        if _cm is not None:
            _mk = {r.ticker: r.market for r in df.itertuples(index=False)}
            _dr = {r.ticker: int(r.direction) for r in df.itertuples(index=False)}
            _zv = {r.ticker: (float(r.z) if pd.notna(r.z) else None) for r in df.itertuples(index=False)}
            _uni = list(_mk.keys())
            # Colour by the z's SIGN (rich = z>0, cheap = z<0) so a divergent-but-unflagged peer
            # still shows the swap. The flagged leg is |z|>=1.5, so any opposite-sign peer is a real gap.
            _zc = lambda z: "rich" if (z is not None and z > 0) else "cheap" if (z is not None and z < 0) else "neu"
            _zf = lambda tk: (f"{_zv[tk]:+.1f}" if _zv.get(tk) is not None else "—")

            def _c1m(a, b):
                try:
                    v = float(_cm1m.loc[a, b])
                    return v if pd.notna(v) else None
                except Exception:
                    return None

            for r in flagged(df).itertuples(index=False):
                ps = volcorr.peers(r.ticker, _cm, corr_threshold, universe=_uni)
                if not ps:
                    continue
                _fz = _zv.get(r.ticker)
                peers = []
                for p, c in ps[:5]:
                    m1 = _c1m(r.ticker, p)
                    peers.append({
                        "name": _mk[p], "corr": f"{c*100:.0f}%",
                        "corr1m": f"{m1*100:.0f}%" if m1 is not None else "—",
                        "loose": m1 is not None and (c - m1) >= 0.30,   # link has drifted lately
                        "z": _zf(p), "zcls": _zc(_zv.get(p)),
                        "opp": (_fz is not None and _zv.get(p) is not None and (_fz > 0) != (_zv.get(p) > 0)),
                    })
                rv_peers.append({
                    "market": r.market, "signal": r.signal, "direction": int(r.direction),
                    "z": _zf(r.ticker), "zcls": _zc(_fz), "peers": peers,
                })
    except Exception:
        rv_peers = []

    # Desk comment: deterministic note -> conversational rewrite (template on failure).
    rv_comment = ""
    if rv_peers:
        _note = _swaps_template(rv_peers)
        rv_comment = _md_bold(_ai_comment(_note) if ai else _note)

    # Short-term rates — own section, rate-vol convention, 1σ moves in basis points.
    # Lives on the closing page with a 1-year panel per FLAGGED market (spread vs the
    # implied rate) above the definitions grid.
    stir_rows, stir_panels = [], []
    if stir is not None and len(stir):
        s = reflag(stir, threshold, ("Rich — sell vol", -1), ("Cheap — buy vol", 1),
                   gate_col="spread")
        stir_rows = [{
            "market": r.market, "ticker": r.ticker, "rate": f"{r.rate:.2f}",
            "iv": f"{r.iv:.1f}", "rv": f"{r.rv:.1f}",
            "iv_bp": f"{r.iv_bp:.1f}", "rv_bp": f"{r.rv_bp:.1f}",
            "spread": f"{r.spread:+.1f}",
            "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
            "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
            "vs_bbg": (f"{r.vs_bbg:+.1f}" if pd.notna(getattr(r, "vs_bbg", float("nan"))) else "n/a"),
            "signal": r.signal, "direction": int(r.direction),
        } for r in s.dropna(subset=["z"]).sort_values("z", ascending=False).itertuples(index=False)]
        if stir_hist is not None and len(stir_hist):
            fl_s = s[s["direction"] != 0]
            fl_s = fl_s.reindex(fl_s["z"].abs().sort_values(ascending=False).index)
            for r in fl_s.itertuples(index=False):
                g = stir_hist[stir_hist["ticker"] == r.ticker].sort_values("date")
                if g.empty:
                    continue
                g = g.rename(columns={"rate": "price"})   # panel's grey right axis = the implied rate
                stir_panels.append(_hist_panel(r.market, int(r.direction), g, "spread"))

    return env.get_template("volreport.html").render(
        asof=pretty_date(asof), rv_peers=rv_peers, rv_comment=rv_comment,
        stir_rows=stir_rows, stir_panels=stir_panels, src=snapshot_stamp(),
        seasonal_open=seasonal_open,
        corr_pct=f"{corr_threshold*100:.0f}",
        corr_lbl=_corr_lbl, corr_moves=_corr_moves,
        synopsis=build_synopsis(df, threshold, seasonal_set),
        scatter=scatter_png(df),
        bars=bar_png(df, value_col="spread", value_fmt="{:+.1f}", n=None, max_h=12.6, label_fs=7.0,
                     xlabel="implied − realized spread, z vs 1-yr · amber = at trigger, "
                            "sign unconfirmed · label = spread"),
        history=history_sections(df, hist, asof, ai),
        rows=rows, n_markets=len(df), zflag=f"{threshold:g}",
        own_div=own_div, n_own=n_own,
        any_rvmark=any(r["rvmark"] for r in rows),
        any_seasonal=any(r["seasonal"] for r in rows),
        n_rich=int((df["direction"] < 0).sum()), n_cheap=int((df["direction"] > 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = 1.5, hist=None,
              corr_threshold: float = 0.6, corr_metric: str = "realized", ai: bool = True,
              stir=None, stir_hist=None) -> str:
    return render_pdf(render_html(df, asof, threshold, hist, corr_threshold, corr_metric, ai,
                                  stir, stir_hist), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--corr-threshold", type=float, default=0.6)
    ap.add_argument("--corr-metric", choices=["returns", "iv", "realized"], default="realized")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the Claude rewrite of the swaps comment (template text only)")
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "volatility_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    stir_path = detail.parent / "stirvol.parquet"
    stir = pd.read_parquet(stir_path) if stir_path.exists() else None
    stir_hist_path = detail.parent / "stirvol_history.parquet"
    stir_hist = pd.read_parquet(stir_hist_path) if stir_hist_path.exists() else None
    _d = pd.read_parquet(detail)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            if "ticker" in _d:
                _d = _d[_d["ticker"].isin(_e)]
            if hist is not None and "ticker" in getattr(hist, "columns", []):
                hist = hist[hist["ticker"].isin(_e)]
            if stir is not None and "ticker" in getattr(stir, "columns", []):
                stir = stir[stir["ticker"].isin(_e)]
            if stir_hist is not None and "ticker" in getattr(stir_hist, "columns", []):
                stir_hist = stir_hist[stir_hist["ticker"].isin(_e)]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf, args.threshold, hist, args.corr_threshold,
              args.corr_metric, ai=not args.no_ai, stir=stir, stir_hist=stir_hist)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
