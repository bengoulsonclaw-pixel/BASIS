"""Skew Volatility Report — a branded, client-style PDF for option skew across the
book: the normalized skew (90% put − 110% call)/ATM, z-scored over a trailing year
(positive = puts richer than calls).
  1. Scatter — 90% put vol (y) vs 110% call vol (x) with the 45° no-skew line; a
     dot's distance from the line is its put−call skew.
  2. Diverging bar — most stretched skews ranked by the 1-yr z-score.
  3. Per-flagged-product dumbbell (110% call → 90% put wing) + 1-year skew-vs-price
     small-multiples — the same per-product view the Volatility Report uses.
Plus a flagged-opportunities table.

Listed markets use the fixed 90/110% moneyness wings of the option surface; FX uses
the native OTC 25Δ risk reversal (it has no moneyness surface). Bonds and STIRs are
excluded (low price vol pushes the 90/110% strikes far OTM into extrapolation).

Run standalone (the app calls it as a subprocess):
    python src/skewreport.py data/signals/skew.parquet out.pdf --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportkit import (pretty_date, data_uri, png, color, legend, bar_png, flagged, reflag, render_pdf,
                       BLACK, RICH, CHEAP, NEUTRAL, snapshot_stamp,
                       ai_rewrite, md_bold, ordinal, when_phrase, history_panel, last_episode)
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def scatter_png(df: pd.DataFrame) -> str:
    """90% put vol (y) vs 110% call vol (x), with the 45° no-skew line; distance from
    the line is the put−call skew (above = puts richer = downside skew)."""
    d = df.dropna(subset=["put", "call"])
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    hi = float(max(d["put"].max(), d["call"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "put = call (no skew)", rotation=45, ha="right",
            va="bottom", fontsize=7, color="#555", rotation_mode="anchor",
            transform_rotates_text=True)   # rotation in DATA space → text hugs the 45° line
    # Whole book plotted; only FLAGGED markets highlighted (coloured + named), rest muted grey.
    neu, fl = d[d["direction"] == 0], d[d["direction"] != 0]
    ax.scatter(neu["call"], neu["put"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.scatter(fl["call"], fl["put"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.call, r.put), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Rich — sell skew", "Cheap — buy skew")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("110% moneyness call vol  (%)")
    ax.set_ylabel("90% moneyness put vol  (%)")
    ax.set_title("Above the line → puts richer than calls (downside skew)")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return png(fig)


def wings_png(df: pd.DataFrame, n: int = 18) -> str:
    """Dumbbell of the FLAGGED markets: the 110% call wing (grey) and the 90% put
    wing (signal-coloured) joined by a line whose length IS the put−call gap — so the
    reader sees each flagged product's skew at a glance. (FX wings are the 25Δ RR.)"""
    d = flagged(df)
    if d.empty:
        return ""
    if len(d) > n:                                   # keep it readable — most stretched
        d = d.reindex(d["z"].abs().sort_values(ascending=False).index).head(n)
    d = d.sort_values("skew")                        # call-skew (neg) bottom → put-skew (pos) top
    y = list(range(len(d)))
    put, call, dirn = d["put"].to_numpy(), d["call"].to_numpy(), d["direction"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.1, max(1.7, 0.36 * len(d) + 0.6)))
    for i in y:
        ax.plot([call[i], put[i]], [i, i], color="#CFCFCF", lw=1.6, zorder=1)
    ax.scatter(call, y, s=46, color=NEUTRAL, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(put, y, s=46, color=[color(x) for x in dirn],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["market"], fontsize=6.8)
    ax.set_ylim(-0.6, len(d) - 0.4)        # uniform half-row padding → no sparse stretch
    ax.set_xlabel("wing implied vol (%)")
    ax.set_title("Flagged markets — 110% call → 90% put (the line is the skew)")
    ax.grid(True, axis="x", color="#EEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=NEUTRAL, mec="white", ms=8, label="110% call wing"),
               Line2D([0], [0], marker="o", ls="", mfc=RICH, mec="white", ms=8, label="90% put — rich (sell skew)"),
               Line2D([0], [0], marker="o", ls="", mfc=CHEAP, mec="white", ms=8, label="90% put — cheap (buy skew)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=6.3, frameon=False)
    return png(fig)


_CAPTION_SYSTEM = (
    "You are a senior volatility strategist writing the one-to-two sentence caption under each "
    "market's skew-history chart in a daily client report (the chart shows the normalized "
    "put-call skew over the past year against the underlying price). The input is a JSON array "
    "of terse machine-built notes, one per market, in order; rewrite EACH as a flowing, "
    "conversational but professional caption — a desk analyst pointing out what the chart shows.\n"
    "HARD RULES — never break these:\n"
    "(1) Every figure, date and product name must be copied EXACTLY as given, wrapped in the same "
    "**bold** markers. NEVER invent an event, cause, or figure not in the note — you may only "
    "describe what the note states (levels, when the skew was last here, what price/wings were "
    "doing then and now). No outside knowledge.\n"
    "(2) Neutral and observational — client-safe, NOT advice: never say buy, sell, long, short, "
    "recommend, or imply the reader should act.\n"
    "(3) 1-2 sentences per caption, vary the openings, no exclamation marks.\n"
    "Return ONLY a JSON array of strings, the same length and order as the input — nothing else.")


def _hist_note(r, g: pd.DataFrame, asof: pd.Timestamp) -> str:
    """Deterministic **bold**-marked caption facts for one flagged market — every claim
    computed from the cached history (wing levels, the last comparable episode, which
    wing has moved lately, the underlying's move)."""
    rich = int(r.direction) < 0                      # rich = puts dear = skew high side
    s = g.set_index(pd.to_datetime(g["date"]))
    skew, px = s["skew"].dropna(), s["price"].dropna()
    bits = [f"The 90% put wing is marked at **{r.put:.1f}** against **{r.call:.1f}** for the 110% "
            f"call (ATM **{r.atm:.1f}**) — a normalized skew of **{r.skew:+.2f}**"
            + (f", the **{ordinal(r.pctl)}** percentile of the past year" if pd.notna(r.pctl) else "")
            + "."]
    hit = last_episode(skew, high_side=rich)
    if hit:
        dt = hit[0]
        ep = [f"Puts were last this {'dear' if rich else 'cheap'} relative to calls in "
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
        bits.append(f"That is its most {'put-skewed' if rich else 'call-skewed'} reading of the past year.")
    now = []
    if "put" in s.columns and "call" in s.columns:
        try:
            p_s, c_s = s["put"].dropna(), s["call"].dropna()
            dp, dc = float(p_s.iloc[-1] - p_s.iloc[-22]), float(c_s.iloc[-1] - c_s.iloc[-22])
            now.append(f"over the past month the put wing has moved **{dp:+.1f}** points against "
                       f"**{dc:+.1f}** for the calls")
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
    """ALL flagged markets as [{img, cap}] — a 1-year skew-vs-underlying panel each,
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
    return [{"img": history_panel(r.market, int(r.direction), g, "skew", height=_h), "cap": md_bold(c)}
            for r, g, c in zip(rows, groups, caps)]


def build_synopsis(df: pd.DataFrame) -> str:
    """Findings-only opener (the screen's mechanics live in the template's yellow box)."""
    from reportkit import num_word, join_names, names_by_z
    fl = df[df["direction"] != 0]
    rich, cheap = fl[fl["direction"] < 0], fl[fl["direction"] > 0]
    if fl.empty:
        return ("Nothing stands out today: no market's put&ndash;call skew is stretched "
                "beyond its own one-year norm.")
    out = [f"<b>{num_word(len(fl)).capitalize()} market{'s' if len(fl) != 1 else ''} "
           f"stand{'s' if len(fl) == 1 else ''} out today.</b>"]
    if len(rich):
        out.append(f"{num_word(len(rich)).capitalize()} carr{'ies' if len(rich) == 1 else 'y'} "
                   f'unusually <span class="rich">rich</span> put skew, the market paying up for '
                   f"downside protection: <b>{join_names(names_by_z(rich))}</b>.")
    if len(cheap):
        out.append(f"{num_word(len(cheap)).capitalize()} look{'s' if len(cheap) == 1 else ''} "
                   f'<span class="cheap">cheap</span>, with downside protection under-priced '
                   f"against its own history: <b>{join_names(names_by_z(cheap))}</b>.")
    out.append("They are highlighted purely as markets where the skew currently looks "
               "stretched, and may be worth a closer look.")
    return " ".join(out)


def render_html(df: pd.DataFrame, asof: str, threshold: float = 1.5, hist: pd.DataFrame = None,
                ai: bool = True) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Rich — sell skew", -1), ("Cheap — buy skew", 1))
    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "put": f"{r.put:.1f}", "call": f"{r.call:.1f}", "atm": f"{r.atm:.1f}",
        "skew": f"{r.skew:+.2f}",
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in flagged(df).itertuples(index=False)]
    return env.get_template("skewreport.html").render(
        asof=pretty_date(asof), src=snapshot_stamp(),
        synopsis=build_synopsis(df),
        scatter=scatter_png(df),
        bars=bar_png(df, value_col="skew", value_fmt="{:+.2f}", n=None,
                     xlabel="(90% put − 110% call) / ATM skew,  z-score vs 1-yr   (label = skew)"),
        wings=wings_png(df),
        history=history_sections(df, hist, asof, ai),
        rows=rows, n_markets=len(df), zflag=f"{threshold:g}",
        n_rich=int((df["direction"] < 0).sum()), n_cheap=int((df["direction"] > 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = 1.5, hist=None,
              ai: bool = True) -> str:
    return render_pdf(render_html(df, asof, threshold, hist, ai), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the Claude rewrite of the history captions (template text only)")
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "skew_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else None
    _d = pd.read_parquet(detail)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            if "ticker" in _d:
                _d = _d[_d["ticker"].isin(_e)]
            if hist is not None and "ticker" in getattr(hist, "columns", []):
                hist = hist[hist["ticker"].isin(_e)]
    except Exception:
        pass
    build_pdf(_d, args.asof, args.out_pdf, args.threshold, hist, ai=not args.no_ai)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
