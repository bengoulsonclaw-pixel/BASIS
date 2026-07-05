"""Open Interest Report — a branded client PDF of listed-option open interest as a
strike × expiry-month heatmap (puts + calls per cell, shaded by size).

Two scopes, auto-detected from the input (or forced with --scope):
  • single — one product: a large heatmap + a plain-English read of where the open
             interest concentrates (busiest expiry, peak strike, put/call tilt).
  • book   — the whole universe: one compact heatmap per product (a chartbook).

The strike×expiry open-interest grid is produced upstream by the app
(datafeed.get_oi_chain) and handed to this renderer as a tidy parquet with columns
[ticker, market, asset, spot, expiry, expiry_label, strike, call_oi, put_oi]; this
script only reads that frame and draws — no Bloomberg / datafeed import here, exactly
like the other *report.py renderers.

Run standalone (the app calls it as a subprocess):
    python src/oireport.py oi_chain.parquet out.pdf --asof "2026-06-20" --scope single
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

from reportkit import data_uri, png, render_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"

OI_CMAP = "YlOrRd"        # sequential: pale = light OI · deep red = where OI is largest
ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]
CHART_DPI = 160


def _png(fig) -> str:
    return png(fig, dpi=CHART_DPI)


def _human(v) -> str:
    """Compact contract count: 1,240,000 → '1.24M', 38,400 → '38K'."""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:,.0f}"


def _decimals_of(x: float) -> int:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def _strike_decimals(strikes) -> int:
    """Decimals for the strike axis — enough to render the strike SPACING exactly, so fine
    grids (SOFR 0.125, FX 0.0002) stay legible and round ones (index 25) stay clean."""
    s = sorted({float(x) for x in strikes if pd.notna(x)})
    if not s:
        return 0
    if len(s) == 1:
        return 0 if s[0] >= 100 else 2
    gap = min((b - a) for a, b in zip(s, s[1:]) if b > a)
    return min(5, _decimals_of(round(gap, 6)))


def _fmt_strike(v, dec) -> str:
    return "—" if (v is None or pd.isna(v)) else f"{v:,.{dec}f}"


def _draw_heatmap(ax, g: pd.DataFrame, fig, fs: float = 1.0):
    """Draw one product's open-interest heatmap into `ax` (colorbar on `fig`): strikes
    (rows, high at top) × expiry months (cols), each cell put + call OI shaded
    pale→deep-red, dashed line at spot. `fs` scales the fonts (smaller when stacked)."""
    g = g.copy()
    g["total"] = g["call_oi"].fillna(0) + g["put_oi"].fillna(0)
    spot = float(g["spot"].iloc[0]) if ("spot" in g and pd.notna(g["spot"].iloc[0])) else float("nan")
    market = str(g["market"].iloc[0]) if "market" in g else str(g["ticker"].iloc[0])
    asset = str(g["asset"].iloc[0]) if "asset" in g else ""

    col_order = (g[["expiry", "expiry_label"]].drop_duplicates()
                 .sort_values("expiry")["expiry_label"].tolist())
    piv = (g.pivot_table(index="strike", columns="expiry_label", values="total", aggfunc="sum")
           .reindex(columns=col_order).sort_index(ascending=False))
    # Drop fully-empty strikes (rows) and expiries (cols) so the board fills the page with where
    # the OI actually sits — a real chain lists far strikes and serial months that carry none.
    _rsum, _csum = piv.fillna(0).sum(axis=1), piv.fillna(0).sum(axis=0)
    if (_rsum > 0).any() and (_csum > 0).any():
        piv = piv.loc[_rsum > 0, _csum > 0]
    strikes = list(piv.index)
    cols = list(piv.columns)
    data = piv.values.astype(float)
    finite = data[np.isfinite(data)]
    vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    dec = _strike_decimals(strikes)

    fs_cell, fs_tick = 5.6 * fs, 6.6 * fs
    im = ax.imshow(data, aspect="auto", cmap=OI_CMAP, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=fs_tick)
    ax.set_yticks(range(len(strikes)))
    ax.set_yticklabels([_fmt_strike(s, dec) for s in strikes], fontsize=fs_tick)
    ax.set_ylabel("Strike", fontsize=fs_tick + 1.5)
    ax.tick_params(length=0)

    for iy in range(len(strikes)):
        for ix in range(len(cols)):
            v = data[iy, ix]
            if np.isfinite(v) and v > 0:
                ax.text(ix, iy, _human(v), ha="center", va="center", fontsize=fs_cell,
                        color="white" if v > 0.58 * vmax else "#222")

    if np.isfinite(spot) and len(strikes) > 1:
        ys = np.arange(len(strikes))
        y_spot = float(np.interp(spot, strikes[::-1], ys[::-1]))   # strikes descend down the rows
        ax.axhline(y_spot, color="#0A0A0A", lw=1.0, ls="--", zorder=6)

    spot_txt = f"  ·  spot {_fmt_strike(spot, dec)} (dashed)" if np.isfinite(spot) else ""
    ax.set_title(f"{market}  ·  {asset}{spot_txt}", fontsize=8.4 * fs, loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("OI (puts + calls)", fontsize=fs_tick)
    cbar.ax.tick_params(labelsize=max(3.0, fs_tick - 1))


def heatmap_fig(g: pd.DataFrame, compact: bool = False):
    """One product's heatmap as a standalone figure (single-product page / book panel)."""
    n_strikes = int(g["strike"].nunique())
    cell = 0.20 if compact else 0.30
    figh = min(3.8 if compact else 7.6, max(2.2, cell * n_strikes + 0.9))
    fig, ax = plt.subplots(figsize=(6.1, figh))
    _draw_heatmap(ax, g, fig, fs=0.82 if compact else 1.0)
    return fig


def page_fig(group: pd.DataFrame, landscape: bool = False):
    """A whole report page: the product heatmaps for one curated group in one figure sized to
    fill the A4 content area — so exactly these products land together on one page. Used by the
    grouped (Fixed Income) report. `landscape` widens the figure (and shortens it) so the long
    rates strip — SOFR runs ~20 quarterly expiries — keeps its columns legible."""
    tickers = list(dict.fromkeys(group["ticker"].tolist()))
    k = max(1, len(tickers))
    if landscape:                              # wide page: room for the full expiry strip
        width, fs = 9.4, 0.84
        per_h = {1: 5.8, 2: 3.4, 3: 2.3}.get(k, 6.5 / k)   # short enough to fit a landscape page
    else:
        width = 6.1
        per_h = {1: 8.6, 2: 4.2, 3: 2.9}.get(k, 8.7 / k)   # k=1 fills the portrait page
        fs = {1: 1.0, 2: 0.92, 3: 0.74}.get(k, 0.7)
    fig, axes = plt.subplots(k, 1, figsize=(width, per_h * k + 0.4), constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), tickers):
        _draw_heatmap(ax, group[group["ticker"] == t], fig, fs=fs)
    return fig


def product_commentary(g: pd.DataFrame) -> str:
    """A plain-English read of where one product's open interest sits. Returns HTML."""
    g = g.copy()
    g["total"] = g["call_oi"].fillna(0) + g["put_oi"].fillna(0)
    market = str(g["market"].iloc[0])
    spot = float(g["spot"].iloc[0]) if pd.notna(g["spot"].iloc[0]) else float("nan")
    dec = _strike_decimals(g["strike"])
    tot = float(g["total"].sum())
    tot_c, tot_p = float(g["call_oi"].sum()), float(g["put_oi"].sum())
    pc = (tot_p / tot_c) if tot_c else float("nan")
    by_exp = g.groupby(["expiry", "expiry_label"])["total"].sum().sort_values(ascending=False)
    top_exp = str(by_exp.index[0][1]) if len(by_exp) else "—"
    top_share = (by_exp.iloc[0] / tot * 100) if (tot and len(by_exp)) else float("nan")
    peak_strike = g.groupby("strike")["total"].sum().idxmax() if len(g) else float("nan")
    big = g.loc[g["total"].idxmax()] if len(g) else None

    pc_txt = "—" if not np.isfinite(pc) else f"{pc:.2f}"
    tilt = ("put-skewed" if (np.isfinite(pc) and pc > 1.15)
            else "call-skewed" if (np.isfinite(pc) and pc < 0.87) else "broadly balanced")
    parts = [
        f"<b>{html.escape(market)}</b> carries <b>{_human(tot)}</b> contracts of listed-option open interest "
        f"across the strikes and expiries shown (<b>{_human(tot_c)}</b> calls / <b>{_human(tot_p)}</b> puts, "
        f"P/C <b>{pc_txt}</b> &mdash; {tilt}).",
        f"It is concentrated in the <b>{html.escape(top_exp)}</b> expiry"
        + (f" (~{top_share:.0f}% of the total)" if np.isfinite(top_share) else "")
        + f", and the heaviest single strike is <b>{_fmt_strike(peak_strike, dec)}</b> against a spot of "
        f"<b>{_fmt_strike(spot, dec)}</b>.",
    ]
    if big is not None:
        parts.append(
            f"The single largest cell is the <b>{html.escape(str(big['expiry_label']))}</b> "
            f"<b>{_fmt_strike(float(big['strike']), dec)}</b> line at <b>{_human(float(big['total']))}</b> "
            "contracts. Big open-interest strikes often act as pin / magnet levels into expiry and mark where "
            "dealer hedging is heaviest.")
    return " ".join(parts)


def book_commentary(products: list, n_products: int) -> str:
    """One paragraph framing the whole-book chartbook. Returns HTML."""
    top = sorted(products, key=lambda p: p["_tot"], reverse=True)[:4]
    names = ", ".join(f"<b>{html.escape(p['market'])}</b> ({_human(p['_tot'])})" for p in top)
    return (f"Listed-option <b>open interest</b> across <b>{n_products}</b> products &mdash; each panel is one "
            "market's strike &times; expiry-month grid, every cell the sum of put and call open interest, shaded "
            "pale&rarr;deep-red by size. The deepest cells are where positioning is concentrated and dealer "
            f"hedging heaviest; the dashed line marks spot. The largest books shown are {names}.")


def render_html(df: pd.DataFrame, asof: str, scope: str, light: bool = False) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    common = dict(asof=asof, logo=data_uri(ASSETS / "logo.png"),
                  watermark="" if light else data_uri(ASSETS / "building.jpg"))

    # Grouped (Fixed Income): a curated `page` order, each page its own banner + a single
    # stacked figure of that page's products. The input carries `page` / `page_title`.
    if scope == "grouped" or "page" in df.columns:
        d = df.copy()
        d["page"] = pd.to_numeric(d.get("page"), errors="coerce").fillna(0).astype(int)
        pages = []
        for pidx in sorted(d["page"].unique()):
            gp = d[d["page"] == pidx]
            title = str(gp["page_title"].iloc[0]) if ("page_title" in gp.columns
                                                      and pd.notna(gp["page_title"].iloc[0])) else ""
            pages.append({"title": title, "img": _png(page_fig(gp, landscape=True))})
        return env.get_template("oireport.html").render(
            mode="grouped", heading="Open Interest — Fixed Income",
            pages=pages, n_pages=len(pages), **common)

    tickers = list(dict.fromkeys(df["ticker"].tolist()))

    def _key(t):
        a = str(df.loc[df["ticker"] == t, "asset"].iloc[0])
        ai = ASSET_ORDER.index(a) if a in ASSET_ORDER else len(ASSET_ORDER)
        return (ai, str(df.loc[df["ticker"] == t, "market"].iloc[0]))

    tickers = sorted(tickers, key=_key)
    single = (scope == "single") or (len(tickers) <= 1)
    compact = not single

    products = []
    for t in tickers:
        g = df[df["ticker"] == t]
        if g.empty:
            continue
        products.append({
            "market": str(g["market"].iloc[0]), "asset": str(g["asset"].iloc[0]),
            "img": _png(heatmap_fig(g, compact=compact)),
            "commentary": product_commentary(g) if single else "",
            "_tot": float((g["call_oi"].fillna(0) + g["put_oi"].fillna(0)).sum()),
        })

    return env.get_template("oireport.html").render(
        mode=("single" if single else "book"), heading="Open Interest",
        single=single, products=products,
        commentary=("" if single else book_commentary(products, len(products))),
        n_products=len(products), total_oi=_human(sum(p["_tot"] for p in products)),
        **common)


def build_pdf(df: pd.DataFrame, asof: str, out_path, scope: str = "auto", light: bool = False) -> str:
    _landscape = (scope == "grouped") or ("page" in df.columns)   # the wide fixed-income strip
    return render_pdf(render_html(df, asof, scope, light), out_path, landscape=_landscape)


def main():
    global CHART_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("chain_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--scope", choices=["auto", "single", "book", "grouped"], default="auto")
    ap.add_argument("--quality", choices=["screen", "email"], default="screen",
                    help="screen = crisp 160dpi; email = lighter 96dpi for a smaller attachment")
    args = ap.parse_args()
    CHART_DPI = 160 if args.quality == "screen" else 96
    df = pd.read_parquet(args.chain_parquet)
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa() and "ticker" in df:
            df = df[df["ticker"].isin(_en())]
    except Exception:
        pass
    scope = args.scope
    if scope == "auto":
        scope = "grouped" if "page" in df.columns else ("single" if df["ticker"].nunique() <= 1 else "book")
    build_pdf(df, args.asof, args.out_pdf, scope, light=(args.quality == "email"))
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
