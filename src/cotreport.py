"""COT Positioning Report — a branded client PDF of CFTC Commitments-of-Traders
positioning across the book.

  1. Ranked bar — every market by its COT Index (0 = most net-short in 3y, 100 =
     most net-long); crowded-long (blue) / crowded-short (red) extremes stand out.
  2. Per-product charts — the classic COT view for each market: gross LONG bars up
     (blue) / gross SHORT bars down (red), the NET line over the top (black), and
     the underlying price on a second axis (grey); a COT-Index oscillator panel
     below with the crowded zones shaded.
Plus a flagged-extremes table.

Run standalone (the app calls it as a subprocess):
    python src/cotreport.py data/signals/cot.parquet out.pdf --asof "2026-06-12" --threshold 80
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from reportkit import data_uri, png, render_pdf, render_png, BLACK, YELLOW
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

import cotstudy   # shared forward-return study (also imported by the dashboard)
import cotseasonality   # shared positioning-seasonality profile (also imported by the dashboard)

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "signals" / "cot_history.parquet"

LONG_BLUE = "#1F5FA8"    # gross long bars (above zero)
SHORT_RED = "#C62828"    # gross short bars (below zero) — house red
NET_LINE = "#0A0A0A"     # net %OI line (black, lower panel)
PRICE_LINE = "#0A0A0A"   # underlying price (black, right axis)
DISPLAY_WEEKS = 156      # ~3 years shown on each chart
DEFAULT_CUTOFF = 80.0    # COT Index ≥ cutoff → crowded long; ≤ 100−cutoff → crowded short
ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]

# Embedded-chart resolution. "screen" is crisp; "email" trades sharpness for a much
# smaller file (the report is image-heavy: ~50 charts embedded as PNGs). Set via --quality.
CHART_DPI = 160


def _png(fig) -> str:
    return png(fig, dpi=CHART_DPI)


def _runs(mask, dates):
    """Contiguous (start, end) date spans where boolean `mask` is True."""
    out, start = [], None
    arr = list(mask)
    for i, m in enumerate(arr):
        if m and start is None:
            start = i
        elif (not m) and start is not None:
            out.append((dates[start], dates[i - 1])); start = None
    if start is not None:
        out.append((dates[start], dates[-1]))
    return out


def _shade_extreme(ax, dates, ci, hi, lo):
    d = list(dates)
    for a, b in _runs((ci >= hi).fillna(False).tolist(), d):
        ax.axvspan(a, b, color=YELLOW, alpha=0.13, zorder=0)
    for a, b in _runs((ci <= lo).fillna(False).tolist(), d):
        ax.axvspan(a, b, color=SHORT_RED, alpha=0.06, zorder=0)


def product_fig(g: pd.DataFrame, hi: float, lo: float, title: str, category: str):
    """The per-product COT chart, three stacked panels sharing a time axis:
      1. long/short bars + net line + price overlay,
      2. net % of open interest vs its rolling 3-year 10–90% band,
      3. the COT-Index oscillator with the crowded zones shaded."""
    g = g.dropna(subset=["net"]).sort_values("date")
    dates = pd.to_datetime(g["date"]).tolist()
    ci = pd.to_numeric(g["cot_index"], errors="coerce")

    fig, (ax, axb, axo) = plt.subplots(
        3, 1, figsize=(6.1, 3.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1], "hspace": 0.12})

    # --- panel 1: positioning + price ---
    _shade_extreme(ax, dates, ci, hi, lo)
    ax.bar(dates, g["long"], width=5, color=LONG_BLUE, zorder=2, label="MM long")
    ax.bar(dates, -g["short"], width=5, color=SHORT_RED, zorder=2, label="MM short")
    # Net line in XP yellow, thick, with a dark outline so it reads clearly on white.
    ax.plot(dates, g["net"], color=YELLOW, lw=2.6, zorder=4, label="Net",
            path_effects=[pe.Stroke(linewidth=4.2, foreground="#5A4500"), pe.Normal()])
    ax.axhline(0, color="#444", lw=0.7, zorder=3)
    ax.set_ylabel("Contracts", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.margins(x=0.01)
    ax.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if g["price"].notna().any():
        axp = ax.twinx()
        axp.plot(dates, g["price"], color=PRICE_LINE, lw=1.2, zorder=5, label="Price")
        axp.set_ylabel("Price", fontsize=6.5, color=PRICE_LINE)
        axp.tick_params(axis="y", labelsize=5.8, colors=PRICE_LINE)
        axp.margins(x=0.01)
        axp.spines["top"].set_visible(False)
        h2, l2 = axp.get_legend_handles_labels()
        handles += h2; labels += l2

    idx_now = ci.dropna()
    idx_txt = f"COT Index {idx_now.iloc[-1]:.0f}" if len(idx_now) else "COT Index —"
    ax.set_title(f"{title}  ·  {category}  ·  {idx_txt}", fontsize=7.2, loc="left", fontweight="bold")
    ax.legend(handles, labels, loc="upper left", fontsize=5.3, frameon=True, framealpha=0.85,
              edgecolor="#ccc", ncol=4, columnspacing=0.8, handlelength=1.1, borderpad=0.3)

    # --- panel 2: net % of open interest vs its 3-year range ---
    if g["net_pct_oi"].notna().any():
        axb.fill_between(dates, g["npo_p10"], g["npo_p90"], color="#9E9E9E", alpha=0.30,
                         lw=0, zorder=0, label="3y 10–90%")
        axb.plot(dates, g["npo_med"], color="#9E9E9E", lw=0.7, ls="--", zorder=1)
        axb.plot(dates, g["net_pct_oi"], color=NET_LINE, lw=1.1, zorder=2)
    axb.axhline(0, color="#444", lw=0.6, zorder=1)
    axb.set_ylabel("Net %OI", fontsize=6.0)
    axb.tick_params(axis="y", labelsize=5.5)
    axb.margins(x=0.01)
    axb.legend(loc="upper left", fontsize=5.0, frameon=True, framealpha=0.8, edgecolor="#ccc", borderpad=0.3)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # --- panel 3: COT Index oscillator ---
    axo.plot(dates, ci, color="#333", lw=1.0, zorder=3)
    axo.axhspan(hi, 100, color=YELLOW, alpha=0.22, zorder=0)
    axo.axhspan(0, lo, color=SHORT_RED, alpha=0.10, zorder=0)
    axo.axhline(hi, color="#B89000", lw=0.6, ls="--", zorder=1)
    axo.axhline(lo, color=SHORT_RED, lw=0.6, ls="--", zorder=1)
    axo.set_ylim(0, 100)
    axo.set_yticks([0, 50, 100])
    axo.set_ylabel("COT Idx", fontsize=6.0)
    axo.tick_params(axis="y", labelsize=5.5)
    axo.tick_params(axis="x", labelsize=5.8)
    for s in ("top", "right"):
        axo.spines[s].set_visible(False)
    return fig


def seasonality_fig(g_full: pd.DataFrame, title: str):
    """SEAG-style positioning seasonality: median net %OI by week-of-year with the
    25-75%-of-years band and the current year overlaid (XP yellow). Uses the FULL
    history, not the display window — so it shows the whole seasonal sample."""
    seas, current, info = cotseasonality.seasonal_profile(g_full, metric="net_pct_oi")
    fig, ax = plt.subplots(figsize=(6.1, 1.5))
    if seas is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}  ·  seasonality needs ≥4y of history",
                ha="center", va="center", fontsize=6.5, color="#999")
        return fig
    x = seas.index
    ax.fill_between(x, seas["p25"], seas["p75"], color="#BBD6F2", alpha=0.75, lw=0,
                    zorder=0, label="25–75% of years")
    ax.plot(x, seas["med"], color=LONG_BLUE, lw=1.6, zorder=2, label=f"Median ({info['years']}y)")
    if current is not None and current.notna().any():
        ax.plot(current.index, current.values, color=YELLOW, lw=1.9, zorder=3,
                label=str(info["cur_year"]),
                path_effects=[pe.Stroke(linewidth=3.2, foreground="#5A4500"), pe.Normal()])
    ax.axhline(0, color="#444", lw=0.6, zorder=1)
    ax.set_xticks(cotseasonality.MONTH_WEEKS)
    ax.set_xticklabels(cotseasonality.MONTH_LABELS, fontsize=5.8)
    ax.set_xlim(1, 52)
    ax.set_ylabel("Net %OI", fontsize=6.0)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_title(f"{title}  ·  positioning seasonality — net %OI by week of year",
                 fontsize=6.8, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=5.0, frameon=True, framealpha=0.85,
              edgecolor="#ccc", ncol=3, columnspacing=0.8, handlelength=1.1, borderpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
    return fig


def rank_fig(detail: pd.DataFrame):
    """Every market ranked by current COT Index, coloured by crowding."""
    d = detail.dropna(subset=["cot_index"]).sort_values("cot_index")
    fig, ax = plt.subplots(figsize=(6.1, max(3.0, 0.165 * len(d))))
    colors = [LONG_BLUE if x >= detail.attrs.get("hi", DEFAULT_CUTOFF)
              else SHORT_RED if x <= detail.attrs.get("lo", 100 - DEFAULT_CUTOFF)
              else "#9E9E9E" for x in d["cot_index"]]
    y = range(len(d))
    ax.barh(list(y), d["cot_index"], color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(50, color="#444", lw=0.8, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["market"], fontsize=5.7)
    for i, (v, m) in enumerate(zip(d["cot_index"], d["market"])):
        ax.text(v + (1.2 if v < 92 else -1.2), i, f"{v:.0f}",
                va="center", ha="left" if v < 92 else "right", fontsize=5.3, color="#333")
    ax.set_xlim(0, 100)
    ax.set_xlabel("COT Index   (0 = most net-short in 3y · 50 = mid · 100 = most net-long)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, axis="x", color="#EEE", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return fig


def heatmap_fig(hist: pd.DataFrame, detail: pd.DataFrame, weeks: int = 26):
    """Whole-book COT Index heatmap: markets (rows, grouped by asset class) × the last
    `weeks` weeks (cols), coloured red (net-short extreme) → white → blue (net-long
    extreme). The current (right-most) value is annotated in each row."""
    h = hist.dropna(subset=["cot_index"]).copy()
    if h.empty:
        fig, ax = plt.subplots(figsize=(6.1, 1.2)); ax.axis("off"); return fig
    h["date"] = pd.to_datetime(h["date"])
    last_dates = sorted(h["date"].unique())[-weeks:]
    h = h[h["date"].isin(last_dates)]
    piv = h.pivot_table(index="ticker", columns="date", values="cot_index", aggfunc="last")
    meta = detail.set_index("ticker")

    def _key(t):
        a = meta.loc[t, "asset"] if t in meta.index else ""
        ai = ASSET_ORDER.index(a) if a in ASSET_ORDER else len(ASSET_ORDER)
        cur = meta.loc[t, "cot_index"] if t in meta.index else 50
        return (ai, -(cur if pd.notna(cur) else 50))

    rows = sorted(piv.index, key=_key)
    piv = piv.reindex(rows)
    labels = [str(meta.loc[t, "market"]) if t in meta.index else t for t in rows]
    cols = list(piv.columns)
    data = piv.values.astype(float)

    fig, ax = plt.subplots(figsize=(6.1, max(4.0, 0.135 * len(rows))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=0, vmax=100)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=5.3)
    step = max(1, len(cols) // 8)
    xt = list(range(0, len(cols), step))
    ax.set_xticks(xt)
    ax.set_xticklabels([pd.Timestamp(cols[i]).strftime("%b %y") for i in xt], fontsize=5.5)
    for i in range(len(rows)):                       # annotate the current (last) column
        v = data[i, -1]
        if pd.notna(v):
            ax.text(len(cols) - 1, i, f"{v:.0f}", va="center", ha="center", fontsize=4.4,
                    color="white" if (v < 26 or v > 74) else "#222")
    ax.set_title(f"COT Index, last {len(cols)} weeks  ·  red = net-short extreme · blue = net-long extreme",
                 fontsize=7.2, loc="left", fontweight="bold")
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_ticks([0, 50, 100])
    cbar.ax.tick_params(labelsize=5)
    return fig


def _reflag(detail: pd.DataFrame, hi: float, lo: float) -> pd.DataFrame:
    d = detail.copy()
    ci = pd.to_numeric(d["cot_index"], errors="coerce")
    d["direction"] = np.where(ci >= hi, 1, np.where(ci <= lo, -1, 0))
    d["signal"] = np.where(ci >= hi, "Crowded long",
                           np.where(ci <= lo, "Crowded short", "—"))
    d.attrs["hi"], d.attrs["lo"] = hi, lo
    return d


def commentary(detail: pd.DataFrame, hi: float, lo: float) -> str:
    """A single plain-English paragraph summarising the week's outcomes — which markets
    are crowded long/short, the overall tilt, and the biggest week-on-week shifts.
    Returns HTML (market names escaped) to be rendered with the |safe filter."""
    d = detail.dropna(subset=["cot_index"]).copy()
    if d.empty:
        return "No CFTC positioning data is available for the current universe."
    longs = d[d["cot_index"] >= hi].sort_values("cot_index", ascending=False)
    shorts = d[d["cot_index"] <= lo].sort_values("cot_index")
    med = float(d["cot_index"].median())
    tilt = ("tilted net long" if med >= 55 else "tilted net short" if med <= 45 else "broadly balanced")

    def _mk(df):
        return ", ".join(f"<b>{html.escape(str(m))}</b> ({v:.0f})"
                         for m, v in zip(df["market"], df["cot_index"]))

    sent = []
    if len(longs):
        sent.append(f"speculative money is most crowded <b>long</b> in {_mk(longs.head(3))}")
    if len(shorts):
        lead = "and most crowded <b>short</b> in" if sent else "speculative money is most crowded <b>short</b> in"
        sent.append(f"{lead} {_mk(shorts.head(3))}")
    joined = "; ".join(sent)
    extremes = (joined[:1].upper() + joined[1:] + ".") if sent else \
        "No market currently sits at a 3-year positioning extreme."

    sw = d.reindex(d["chg_net"].abs().sort_values(ascending=False).index)
    sw = sw[sw["chg_net"].abs() > 0].head(2)
    swing = ""
    if len(sw):
        bits = ", ".join(
            f"<b>{html.escape(str(r.market))}</b> (net {'buying' if r.chg_net > 0 else 'selling'}, {r.chg_net:+,.0f})"
            for r in sw.itertuples(index=False))
        swing = f" The biggest week-on-week shifts came in {bits}."

    return (f"Across <b>{len(d)}</b> markets with CFTC data, <b>{len(longs)}</b> sit at a 3-year "
            f"<b>long</b> extreme (COT Index &ge; {hi:g}) and <b>{len(shorts)}</b> at a <b>short</b> "
            f"extreme (&le; {lo:g}); positioning is {tilt} overall (median index {med:.0f}). "
            f"{extremes}{swing} Extremes are frequently read as contrarian — a crowded long can "
            f"struggle to find new buyers, a crowded short new sellers.")


def render_html(detail: pd.DataFrame, hist: pd.DataFrame, asof: str, cutoff: float = DEFAULT_CUTOFF,
                light: bool = False) -> str:
    hi, lo = cutoff, 100.0 - cutoff
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    detail = _reflag(detail, hi, lo)
    try:                                            # header shows the DATE only (drop any time)
        asof = pd.to_datetime(asof).strftime("%d %b %Y") if str(asof).strip() else asof
    except Exception:
        asof = str(asof).split(" ")[0]
    try:                                            # latest CFTC report as-of date in the data
        report_date = pd.to_datetime(detail["date"]).max().strftime("%d %b %y")
    except Exception:
        report_date = ""

    products = []
    # Chartbook order: by sector (asset class), then alphabetically by product within each sector.
    for r in detail.sort_values(["asset", "market"], kind="stable").itertuples(index=False):
        gh = hist[hist["ticker"] == r.ticker].sort_values("date")   # full history (seasonality)
        g = gh.tail(DISPLAY_WEEKS)                                   # display window (3-panel chart)
        if g.empty:
            continue
        products.append({
            "img": _png(product_fig(g, hi, lo, r.market, r.category)),
            "seas": _png(seasonality_fig(gh, r.market)),
            "sector": r.asset,                  # for the chartbook sector dividers
        })

    def _fmt(v, sign=False):
        if pd.isna(v):
            return "—"
        return f"{v:+,.0f}" if sign else f"{v:,.0f}"

    rows = [{
        "market": r.market, "ticker": r.ticker, "category": r.category,
        "long": _fmt(r.long), "short": _fmt(r.short), "net": _fmt(r.net, True),
        "pct_oi": f"{r.net_pct_oi:+.0f}%" if pd.notna(r.net_pct_oi) else "—",
        "chg": _fmt(r.chg_net, True),
        "idx": f"{r.cot_index:.0f}" if pd.notna(r.cot_index) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in detail[detail["direction"] != 0].itertuples(index=False)]

    study = cotstudy.forward_return_study(hist, hi, lo)

    def _pct(v):
        return f"{v:+.1f}%" if pd.notna(v) else "—"

    def _sgn(v):
        return int(np.sign(v)) if pd.notna(v) else 0

    # Limit the study table to the markets flagged CROWDED right now (the crowded-markets
    # table above), kept in that same most-extreme-first order.
    # Defensive: if the study came back empty (e.g. no aligned price at all) it has no
    # "ticker" column — don't let that crash the whole report; just show no study table.
    _study_by_tkr = study.set_index("ticker") if (not study.empty and "ticker" in study.columns) else pd.DataFrame()
    fwd_rows = []
    for cr in detail[detail["direction"] != 0].itertuples(index=False):
        if cr.ticker not in _study_by_tkr.index:
            continue
        r = _study_by_tkr.loc[cr.ticker]
        fwd_rows.append({
            "market": r["market"], "asset": r["asset"], "yrs": f"{r['years']:.1f}",
            "epL": int(r["episodes_long"]),
            "long4": _pct(r["long_4"]), "long4_dir": _sgn(r["long_4"]),
            "long13": _pct(r["long_13"]), "long13_dir": _sgn(r["long_13"]),
            "epS": int(r["episodes_short"]),
            "short4": _pct(r["short_4"]), "short4_dir": _sgn(r["short_4"]),
            "short13": _pct(r["short_13"]), "short13_dir": _sgn(r["short_13"]),
            "base4": _pct(r["base_4"]), "base4_dir": _sgn(r["base_4"]),
            "base13": _pct(r["base_13"]), "base13_dir": _sgn(r["base_13"]),
        })

    # how much price history the study draws on (median weeks of aligned price per market)
    _pw = (hist.dropna(subset=["price"]).groupby("ticker").size()
           if ("price" in hist.columns and not hist.empty) else pd.Series(dtype=int))
    study_weeks = int(_pw.median()) if len(_pw) else 0
    study_years = round(study_weeks / 52.0, 1)
    _pdates = (pd.to_datetime(hist.dropna(subset=["price"])["date"])
               if ("price" in hist.columns and not hist.empty) else pd.Series(dtype="datetime64[ns]"))
    study_start = _pdates.min().strftime("%b %Y") if len(_pdates) else "—"
    study_end = _pdates.max().strftime("%b %Y") if len(_pdates) else "—"

    return env.get_template("cotreport.html").render(
        asof=asof, report_date=report_date, commentary=commentary(detail, hi, lo),
        heatmap=_png(heatmap_fig(hist, detail)),
        rank=_png(rank_fig(detail)), products=products, rows=rows, fwd_rows=fwd_rows,
        study_weeks=study_weeks, study_years=study_years,
        study_start=study_start, study_end=study_end,
        n_markets=int(detail["cot_index"].notna().sum()),
        n_long=int((detail["direction"] > 0).sum()),
        n_short=int((detail["direction"] < 0).sum()),
        hi=f"{hi:g}", lo=f"{lo:g}",
        logo=data_uri(ASSETS / "logo.png"),
        # The fixed watermark is re-embedded by Chromium on EVERY page (~0.5MB each), so
        # the email build drops it — the single biggest file-size saving on a long report.
        watermark="" if light else data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(detail: pd.DataFrame, hist: pd.DataFrame, asof: str, out_path, cutoff: float = DEFAULT_CUTOFF,
              light: bool = False) -> str:
    return render_pdf(render_html(detail, hist, asof, cutoff, light), out_path)


def build_email_image(detail: pd.DataFrame, hist: pd.DataFrame, asof: str, cutoff, out_png) -> str:
    """Render the continuous EMAIL-overview image (one full-height sidebar, no pagination):
    commentary, heatmap, ranked bar, crowded-markets table, and the crowded-only forward study."""
    hi, lo = float(cutoff), 100.0 - float(cutoff)
    detail = _reflag(detail, hi, lo)
    try:
        report_date = pd.to_datetime(detail["date"]).max().strftime("%d %b %y")
    except Exception:
        report_date = ""

    def _f(v, sign=False):
        return "—" if pd.isna(v) else (f"{v:+,.0f}" if sign else f"{v:,.0f}")

    def _pct(v):
        return f"{v:+.1f}%" if pd.notna(v) else "—"

    def _sgn(v):
        return int(np.sign(v)) if pd.notna(v) else 0

    crowded = [{
        "market": r.market, "category": r.category, "net": _f(r.net, True),
        "pct_oi": f"{r.net_pct_oi:+.0f}%" if pd.notna(r.net_pct_oi) else "—",
        "idx": f"{r.cot_index:.0f}" if pd.notna(r.cot_index) else "—",
        "signal": r.signal, "direction": int(r.direction),
    } for r in detail[detail["direction"] != 0].itertuples(index=False)]

    sbt = cotstudy.forward_return_study(hist, hi, lo).set_index("ticker")
    fwd = []
    for cr in detail[detail["direction"] != 0].itertuples(index=False):
        if cr.ticker not in sbt.index:
            continue
        r = sbt.loc[cr.ticker]
        fwd.append({
            "market": r["market"], "yrs": f"{r['years']:.1f}",
            "epL": int(r["episodes_long"]), "long13": _pct(r["long_13"]), "long13_dir": _sgn(r["long_13"]),
            "epS": int(r["episodes_short"]), "short13": _pct(r["short_13"]), "short13_dir": _sgn(r["short_13"]),
            "base13": _pct(r["base_13"]), "base13_dir": _sgn(r["base_13"]),
        })

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    html = env.get_template("cotreport_email.html").render(
        report_date=report_date, commentary=commentary(detail, hi, lo),
        heatmap=_png(heatmap_fig(hist, detail)), rank=_png(rank_fig(detail)),
        crowded=crowded, fwd=fwd, logo=data_uri(ASSETS / "logo.png"))
    return render_png(html, out_png, width=640)


def main():
    global CHART_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=DEFAULT_CUTOFF,
                    help="crowded-long cutoff on the COT Index (crowded-short = 100 − this)")
    ap.add_argument("--quality", choices=["screen", "email"], default="screen",
                    help="screen = crisp 160dpi charts; email = lighter 96dpi for a smaller attachment")
    ap.add_argument("--email-image", default="",
                    help="if set, render the continuous email-overview PNG to this path instead of the PDF")
    args = ap.parse_args()
    CHART_DPI = 160 if args.quality == "screen" else 96
    detail = pd.read_parquet(args.detail_parquet)
    hist = pd.read_parquet(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    # The COT report ALWAYS covers the full book — it is intentionally INDEPENDENT of the
    # dashboard's personal sector/product filter (a Home-page view toggle). A client / scheduled
    # report must include every product regardless of what the user has switched off for their
    # own on-screen viewing, so the filter is deliberately NOT applied here.
    if args.email_image:
        build_email_image(detail, hist, args.asof, args.threshold, args.email_image)
        print(f"Wrote {args.email_image}")
        return
    build_pdf(detail, hist, args.asof, args.out_pdf, args.threshold, light=(args.quality == "email"))
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
