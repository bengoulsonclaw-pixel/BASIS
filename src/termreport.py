"""Vol Term Structure Report — a branded client PDF for the ATM implied-vol curve
across the book (1M/3M/6M/12M):
  1. Scatter — 1M (x) vs 3M (y) with the 45° flat line; above = contango, below =
     backwardation.
  2. Diverging bar — most stretched curves ranked by the 3M−1M slope z-score.
  3. Curve small-multiples — the 1M→12M curve for the most inverted / most contango.
  4. Vol-risk premium by tenor — where on the curve options are richest vs realized.
Plus a flagged-opportunities table.

Run standalone (the app calls it as a subprocess):
    python src/termreport.py data/signals/termstructure.parquet out.pdf --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportkit import (pretty_date, data_uri, png, color, legend, bar_png, flagged, reflag,
                       render_pdf, BLACK, RICH, CHEAP, NEUTRAL, snapshot_stamp,
                       ai_rewrite, md_bold, ordinal, when_phrase, history_panel, last_episode)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"

TENORS = ["1M", "3M", "6M", "12M"]
TENOR_MONTHS = [1, 3, 6, 12]
IVCOLS = [f"iv_{t.lower()}" for t in TENORS]
RVCOLS = [f"rv_{t.lower()}" for t in TENORS]
VRPCOLS = [f"vrp_{t.lower()}" for t in TENORS]
Z_FLAG = 1.5   # signal threshold — keep in sync with strategies/volatility.Z_FLAG
               # (this script runs standalone and can't import the strategy package)


def ts_scatter_png(df: pd.DataFrame) -> str:
    d = df.dropna(subset=["iv_1m", "iv_3m"])
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    hi = float(max(d["iv_1m"].max(), d["iv_3m"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "1M = 3M (flat)", rotation=45, ha="right", va="bottom",
            fontsize=7, color="#555", rotation_mode="anchor",
            transform_rotates_text=True)   # rotation in DATA space → text hugs the 45° line
    # Whole book plotted; only FLAGGED markets highlighted (coloured + named), rest muted grey.
    neu, fl = d[d["direction"] == 0], d[d["direction"] != 0]
    ax.scatter(neu["iv_1m"], neu["iv_3m"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.scatter(fl["iv_1m"], fl["iv_3m"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.iv_1m, r.iv_3m), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Inverted — front rich", "Steep — front cheap")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("1M ATM vol  (%)")
    ax.set_ylabel("3M ATM vol  (%)")
    ax.set_title("Above the line → contango (3M > 1M);  below → backwardation")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return png(fig)


def curves_png(df: pd.DataFrame, n: int = 4) -> str:
    """Small-multiples for the n most inverted + n most contango: the implied curve
    (solid, signal-coloured) with the matched REALIZED curve overlaid (dashed grey)
    — the gap at each tenor is that tenor's vol-risk premium."""
    d = df.dropna(subset=["z"])
    pick = pd.concat([d.sort_values("z").head(n),
                      d.sort_values("z", ascending=False).head(n)]).drop_duplicates("ticker")
    ncol = 4
    nrow = max(1, int(np.ceil(len(pick) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.1, 1.85 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(pick.itertuples(index=False)):
        ax = axes.flat[i]
        ax.axis("on")
        iv = [getattr(r, c) for c in IVCOLS]
        rv = [getattr(r, c) for c in RVCOLS]
        ax.plot(TENOR_MONTHS, iv, marker="o", ms=3, lw=1.5, color=color(r.direction))
        ax.plot(TENOR_MONTHS, rv, marker="s", ms=2.5, lw=1.1, ls="--", color="#8A8A8A")
        ax.set_title(r.market, fontsize=6.4)
        ax.set_xticks(TENOR_MONTHS)
        ax.set_xticklabels(TENORS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(True, color="#EEE", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#444", lw=1.5, marker="o", ms=3, label="Implied"),
               Line2D([0], [0], color="#8A8A8A", lw=1.1, ls="--", marker="s", ms=2.5,
                      label="Realized (matched window)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.975))
    return png(fig)


def vrp_png(df: pd.DataFrame) -> str:
    means = [float(df[c].mean()) for c in VRPCOLS]
    fig, ax = plt.subplots(figsize=(6.1, 2.3))
    ax.bar(range(len(TENORS)), means, width=0.6, zorder=3, edgecolor="white",
           color=[CHEAP if m >= 0 else RICH for m in means])
    ax.axhline(0, color=BLACK, lw=0.8)
    pad = (max(abs(m) for m in means) or 1.0) * 0.04
    for i, m in enumerate(means):
        ax.text(i, m + (pad if m >= 0 else -pad), f"{m:+.1f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=7, color="#333")
    ax.set_xticks(range(len(TENORS)))
    ax.set_xticklabels(TENORS)
    ax.set_ylabel("mean IV − RV (vol pts)")
    ax.set_title("Vol-risk premium by tenor (book average)")
    ax.grid(True, axis="y", color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def _sd_str(v, dec) -> str:
    """The 1-day 1σ price move formatted to the contract's native decimals, '' if n/a."""
    try:
        if v is None or not pd.notna(v):
            return ""
        return f"{float(v):.{int(dec)}f}"
    except Exception:
        return ""


def build_synopsis(df: pd.DataFrame) -> str:
    """Findings-only opener (the screen's mechanics live in the template's yellow box)."""
    from reportkit import num_word, join_names, names_by_z
    fl = df[df["direction"] != 0]
    steep, inv = fl[fl["direction"] > 0], fl[fl["direction"] < 0]
    if fl.empty:
        return ("Nothing stands out today: no market's vol curve is stretched beyond its "
                "own one-year norm.")
    out = [f"<b>{num_word(len(fl)).capitalize()} curve{'s' if len(fl) != 1 else ''} "
           f"stand{'s' if len(fl) == 1 else ''} out today.</b>"]
    if len(inv):
        out.append(f"{num_word(len(inv)).capitalize()} {'is' if len(inv) == 1 else 'are'} "
                   f'unusually <span class="rich">inverted</span>, near-term volatility bid '
                   f"above the deferred: <b>{join_names(names_by_z(inv))}</b>.")
    if len(steep):
        out.append(f"{num_word(len(steep)).capitalize()} {'is' if len(steep) == 1 else 'are'} "
                   f'unusually <span class="cheap">steep</span>, the front trading cheap '
                   f"against the back of the curve: <b>{join_names(names_by_z(steep))}</b>.")
    out.append("They are highlighted purely as markets where the curve currently looks "
               "stretched, and may be worth a closer look.")
    return " ".join(out)


_CAPTION_SYSTEM = (
    "You are a senior volatility strategist writing the one-to-two sentence caption under each "
    "market's vol-curve history chart in a daily client report (the chart shows the 3M-minus-1M "
    "ATM implied-vol slope over the past year against the underlying price; a negative slope means "
    "the curve is inverted, with near-term vol bid above the deferred). The input is a JSON array "
    "of terse machine-built notes, one per market, in order; rewrite EACH as a flowing, "
    "conversational but professional caption — a desk analyst pointing out what the chart shows.\n"
    "HARD RULES — never break these:\n"
    "(1) Every figure, date and product name must be copied EXACTLY as given, wrapped in the same "
    "**bold** markers. NEVER invent an event, cause, or figure not in the note — you may only "
    "describe what the note states (levels, when the slope was last here, what price/tenors were "
    "doing then and now). No outside knowledge.\n"
    "(2) Neutral and observational — client-safe, NOT advice: never say buy, sell, long, short, "
    "recommend, or imply the reader should act.\n"
    "(3) 1-2 sentences per caption, vary the openings, no exclamation marks.\n"
    "Return ONLY a JSON array of strings, the same length and order as the input — nothing else.")


def _hist_note(r, g: pd.DataFrame, asof: pd.Timestamp) -> str:
    """Deterministic **bold**-marked caption facts for one flagged market — every claim
    computed from the cached history (the slope and its legs, the last comparable
    episode, which tenor has moved lately, the underlying's move)."""
    steep = int(r.direction) > 0                     # steep = slope high side; inverted = low
    s = g.set_index(pd.to_datetime(g["date"]))
    slope, px = s["slope"].dropna(), s["price"].dropna()
    bits = [f"The 3M−1M slope sits at **{r.slope:+.1f}** vol points (1M **{r.iv_1m:.1f}** vs 3M "
            f"**{r.iv_3m:.1f}**)"
            + (f" — the **{ordinal(r.pctl)}** percentile of the past year" if pd.notna(r.pctl) else "")
            + "."]
    hit = last_episode(slope, high_side=steep)
    if hit:
        dt = hit[0]
        ep = [f"The curve was last this {'steep' if steep else 'inverted'} in "
              f"**{when_phrase(dt, asof)}**"]
        try:
            pin = px.loc[:dt]
            mv = (float(pin.iloc[-1]) / float(pin.iloc[-22]) - 1.0) * 100.0
            if abs(mv) >= 1.0:
                ep.append(f"when the underlying had moved **{mv:+.0f}%** in a month")
        except Exception:
            pass
        bits.append(", ".join(ep) + ".")
    else:
        bits.append(f"That is its most {'steeply upward-sloping' if steep else 'inverted'} "
                    "reading of the past year.")
    now = []
    if "iv1m" in s.columns and "iv3m" in s.columns:
        try:
            f_s, b_s = s["iv1m"].dropna(), s["iv3m"].dropna()
            d1, d3 = float(f_s.iloc[-1] - f_s.iloc[-22]), float(b_s.iloc[-1] - b_s.iloc[-22])
            now.append(f"over the past month the 1M leg has moved **{d1:+.1f}** points against "
                       f"**{d3:+.1f}** for 3M")
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
    return " ".join(bits)


def history_sections(df: pd.DataFrame, hist: pd.DataFrame, asof, ai: bool = True) -> list:
    """ALL flagged markets as [{img, cap}] — a 1-year slope-vs-underlying panel each,
    with a 1-2 sentence data-grounded caption (Fable-polished when `ai`)."""
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
    caps = ai_rewrite(notes, _CAPTION_SYSTEM) if ai else notes
    from reportkit import history_fill_height
    _h = history_fill_height(len(rows))   # pages fill evenly; never one lone panel on a page
    return [{"img": history_panel(r.market, int(r.direction), g, "slope", height=_h), "cap": md_bold(c)}
            for r, g, c in zip(rows, groups, caps)]


def render_html(df: pd.DataFrame, asof: str, threshold: float = Z_FLAG, hist: pd.DataFrame = None,
                ai: bool = True) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Steep — front cheap", 1), ("Inverted — front rich", -1))
    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "iv_1m": f"{r.iv_1m:.1f}", "iv_3m": f"{r.iv_3m:.1f}",
        "iv_6m": f"{r.iv_6m:.1f}", "iv_12m": f"{r.iv_12m:.1f}",
        "iv_1m_sd": _sd_str(getattr(r, "iv_sd_1m", None), getattr(r, "px_dec", 1)),
        "iv_3m_sd": _sd_str(getattr(r, "iv_sd_3m", None), getattr(r, "px_dec", 1)),
        "iv_6m_sd": _sd_str(getattr(r, "iv_sd_6m", None), getattr(r, "px_dec", 1)),
        "iv_12m_sd": _sd_str(getattr(r, "iv_sd_12m", None), getattr(r, "px_dec", 1)),
        "slope": f"{r.slope:+.1f}", "ratio": f"{r.ratio:.2f}" if pd.notna(r.ratio) else "—",
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in flagged(df).itertuples(index=False)]
    return env.get_template("termreport.html").render(
        asof=pretty_date(asof), src=snapshot_stamp(),
        synopsis=build_synopsis(df),
        history=history_sections(df, hist, asof, ai),
        scatter=ts_scatter_png(df),
        bars=bar_png(df, value_col="slope", value_fmt="{:+.1f}", n=None,
                     xlabel="3M − 1M slope,  z-score vs 1-yr   (label = slope, vol pts)"),
        curves=curves_png(df), vrp=vrp_png(df), rows=rows,
        n_markets=len(df), zflag=f"{threshold:g}",
        n_steep=int((df["direction"] > 0).sum()), n_inv=int((df["direction"] < 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = Z_FLAG, hist=None,
              ai: bool = True) -> str:
    return render_pdf(render_html(df, asof, threshold, hist, ai), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=Z_FLAG)
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the Claude rewrite of the history captions (template text only)")
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "termstructure_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    _d = pd.read_parquet(detail)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa() and "ticker" in _d:
            _d = _d[_d["ticker"].isin(_en())]
            if hist is not None and "ticker" in getattr(hist, "columns", []):
                hist = hist[hist["ticker"].isin(_en())]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf, args.threshold, hist, ai=not args.no_ai)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
