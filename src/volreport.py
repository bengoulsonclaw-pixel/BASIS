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
from pathlib import Path

from reportkit import (data_uri, png, color, legend, bar_png, flagged, reflag, render_pdf,
                       BLACK, RICH, CHEAP, NEUTRAL)
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def scatter_png(df: pd.DataFrame) -> str:
    d = df.dropna(subset=["iv", "rv"])
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    hi = float(max(d["iv"].max(), d["rv"].max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=BLACK, zorder=1)
    ax.text(hi * 0.98, hi * 0.98, "implied = realized", rotation=45, ha="right",
            va="bottom", fontsize=7, color="#555", rotation_mode="anchor")
    # Whole book plotted; only the FLAGGED markets are highlighted (coloured + named),
    # the rest are de-emphasised grey so the signals stand out.
    neu = d[d["direction"] == 0]
    fl = d[d["direction"] != 0]
    ax.scatter(neu["rv"], neu["iv"], s=20, c=NEUTRAL, alpha=0.45, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.scatter(fl["rv"], fl["iv"], s=58, c=[color(x) for x in fl["direction"]],
               edgecolor="white", linewidth=0.8, zorder=4)
    for r in fl.itertuples(index=False):
        ax.annotate(r.market, (r.rv, r.iv), fontsize=6.3, color="#222",
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, "Rich — sell vol", "Cheap — buy vol")
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


def _history_chart(rows, hist) -> str:
    """One full-width dual-axis panel per market, stacked one per line."""
    import matplotlib.dates as mdates
    nrow = max(1, len(rows))
    fig, axes = plt.subplots(nrow, 1, figsize=(6.1, 1.5 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(rows):
        ax = axes.flat[i]
        ax.axis("on")
        h = hist[hist["ticker"] == r.ticker].sort_values("date")
        dates = pd.to_datetime(h["date"])
        c = color(r.direction)
        ax.axhline(0, color="#BBB", lw=0.6, zorder=1)
        ax.fill_between(dates, h["spread"], 0, color=c, alpha=0.18, zorder=2)
        ax.plot(dates, h["spread"], color=c, lw=1.1, zorder=3)
        ax.set_title(r.market, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=5, colors=c, length=2)
        ax.tick_params(axis="x", labelsize=5, length=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax2 = ax.twinx()
        ax2.plot(dates, h["price"], color="#444", lw=1.0, zorder=4)
        ax2.tick_params(axis="y", labelsize=5, colors="#444", length=2)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    fig.suptitle("left / shaded: implied − realized (vol pts)      right / grey: underlying price",
                 fontsize=6.6, y=1.0)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.96))
    return png(fig)


def history_pngs(df: pd.DataFrame, hist: pd.DataFrame, per_chart: int = 6) -> list:
    """ALL flagged markets as 1-year dual-axis panels — implied−realized spread
    (left, shaded, signal-coloured) vs the underlying price (right, grey). Returns a
    list of images, ≤per_chart panels each, so every chunk fits on a page."""
    if hist is None or len(hist) == 0:
        return []
    fl = flagged(df)
    if fl.empty:
        return []
    have = set(hist["ticker"].unique())
    picks = fl.reindex(fl["z"].abs().sort_values(ascending=False).index)
    rows = [r for r in picks.itertuples(index=False) if r.ticker in have]
    return [_history_chart(rows[i:i + per_chart], hist) for i in range(0, len(rows), per_chart)]


_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten", "eleven", "twelve"]


def _num(n: int) -> str:
    """Spell out small counts (one … twelve); fall back to digits beyond that."""
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else str(n)


def _join_names(names) -> str:
    """['A', 'B', 'C'] -> 'A, B and C' (names HTML-escaped for safe inlining)."""
    names = [html.escape(str(x)) for x in names]
    if len(names) <= 1:
        return names[0] if names else ""
    if len(names) == 2:
        return names[0] + " and " + names[1]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _names_by_z(g: pd.DataFrame) -> list:
    """Market names, most-stretched (|z|) first."""
    return list(g.reindex(g["z"].abs().sort_values(ascending=False).index)["market"])


def build_synopsis(df: pd.DataFrame, threshold: float = 1.5, seasonal=None) -> str:
    """Plain-language executive summary of the day's flags, built from the data.
    Returns an HTML fragment (rendered with |safe); market names are escaped.
    `seasonal` is the set of tickers currently in their seasonal weather window — a
    rich reading there partly prices genuine crop-weather risk, so we flag that."""
    seasonal = set(seasonal or ())
    n = len(df)
    fl = df[df["direction"] != 0]
    cheap, rich = fl[fl["direction"] > 0], fl[fl["direction"] < 0]
    nflag, nc, nr = len(fl), len(cheap), len(rich)
    z = f"{threshold:g}"
    lead = (f"Today's Volatility Report screens 1-month implied against 21-day realized "
            f"volatility across {n} markets")
    if nflag == 0:
        return (f"{lead} and flags none today: implied and realized are broadly aligned, "
                f"with no market's spread stretched beyond {z} standard deviations of its "
                f"trailing one-year average.")
    lead += (f" and flags {_num(nflag)} where the two have diverged materially — a market "
             f"is flagged when its implied-minus-realized spread sits more than {z} standard "
             f"deviations (z-score) from its own trailing one-year average.")
    out = [lead]
    if nc:
        out.append(
            f"{_num(nc).capitalize()} {'looks' if nc == 1 else 'look'} potentially "
            f'<span class="cheap">cheap</span>, with implied sitting below the volatility '
            f"the market has recently realized: <b>{_join_names(_names_by_z(cheap))}</b>.")
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
    out.append("They are highlighted purely as markets where the implied/realized "
               "relationship currently looks stretched, and may be worth a closer look.")
    return " ".join(out)


def _sd_str(v, dec) -> str:
    """The 1-day 1σ price move formatted to the contract's native decimals, '' if n/a."""
    try:
        if v is None or not pd.notna(v):
            return ""
        return f"{float(v):.{int(dec)}f}"
    except Exception:
        return ""


def render_html(df: pd.DataFrame, asof: str, threshold: float = 1.5, hist: pd.DataFrame = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = reflag(df, threshold, ("Rich — sell vol", -1), ("Cheap — buy vol", 1))
    try:                                   # ag weather-premium flag, keyed to the report's as-of month
        from seasonal import weather_note as _wx
    except Exception:
        def _wx(_t, _m): return ""
    _mo = pd.to_datetime(asof, errors="coerce")
    _mo = _mo.month if pd.notna(_mo) else pd.Timestamp.today().month
    seasonal_set = {r.ticker for r in df.itertuples(index=False) if _wx(r.ticker, _mo)}
    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "iv": f"{r.iv:.1f}", "rv": f"{r.rv:.1f}", "spread": f"{r.spread:+.1f}",
        "iv_sd": _sd_str(getattr(r, "iv_sd", None), getattr(r, "px_dec", 1)),
        "rv_sd": _sd_str(getattr(r, "rv_sd", None), getattr(r, "px_dec", 1)),
        "z": f"{r.z:+.2f}" if pd.notna(r.z) else "—",
        "pctl": f"{int(r.pctl)}" if pd.notna(r.pctl) else "—",
        "signal": r.signal, "direction": int(r.direction),
        "seasonal": _wx(r.ticker, _mo),
    } for r in df.dropna(subset=["z"]).sort_values("z", ascending=False).itertuples(index=False)]
    return env.get_template("volreport.html").render(
        asof=asof,
        synopsis=build_synopsis(df, threshold, seasonal_set),
        scatter=scatter_png(df),
        bars=bar_png(df, value_col="spread", value_fmt="{:+.1f}", n=None,
                     xlabel="implied − realized spread,  z-score vs 1-yr   (flagged in colour · label = spread)"),
        iv_rv=iv_rv_png(df),
        history=history_pngs(df, hist),
        rows=rows, n_markets=len(df), zflag=f"{threshold:g}",
        any_seasonal=any(r["seasonal"] for r in rows),
        n_rich=int((df["direction"] < 0).sum()), n_cheap=int((df["direction"] > 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = 1.5, hist=None) -> str:
    return render_pdf(render_html(df, asof, threshold, hist), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "volatility_history.parquet"
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
    build_pdf(_d, args.asof, args.out_pdf, args.threshold, hist)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
