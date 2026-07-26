"""Daily Flag Breakout Report — a branded, client-style PDF whose hero is the
VISUAL of every flag in the book drawn on its own price chart:
  1. Ranked bar — markets by signed breakout readiness (bull up / bear down),
     labelled with each flagpole's size.
  2. Per-product panels — price with the flagpole, the shaded consolidation
     channel and the dashed breakout line, one small chart per flagged market
     (a ✓vol tick where volume confirms the dry-up).
Plus a compact table of the flagged setups.

Run standalone (the app calls it as a subprocess):
    python src/flagreport.py data/signals/flag_breakout.parquet out.pdf --asof "2026-06-20"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reportkit import pretty_date, data_uri, png, color, render_pdf, BLACK
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def _vc(v):
    """Robust True / False / None from the parquet's vol_confirms cell (object or
    nullable-bool, possibly NaN/None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return bool(v)
    except (TypeError, ValueError):
        return None


def _reflag(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Re-derive (signal, direction) from the stored signed readiness at a chosen
    trigger — so the report's trigger can be tuned at generation time without re-running
    the daily job (mirrors specs.reflag_rows / the dashboard table)."""
    out = df.copy()
    m = pd.to_numeric(out["metric"], errors="coerce")
    out["signal"] = (pd.Series("—", index=out.index, dtype=object)
                     .mask(m >= threshold, "Bull breakout setup").mask(m <= -threshold, "Bear breakout setup"))
    out["direction"] = (pd.Series(0, index=out.index, dtype="int64")
                        .mask(m >= threshold, 1).mask(m <= -threshold, -1))
    return out


def flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged rows: bull (strongest readiness first) then bear (strongest first)."""
    f = df[df["direction"] != 0]
    return pd.concat([f[f["direction"] > 0].sort_values("metric", ascending=False),
                      f[f["direction"] < 0].sort_values("metric")])


def readiness_bar_png(df: pd.DataFrame, n: int = 14) -> str:
    """Diverging horizontal bar of signed breakout readiness — bull (green, breaks up) vs
    bear (red, breaks down); each bar labelled with its flagpole size."""
    d = df.dropna(subset=["metric"])
    sub = pd.concat([d[d["direction"] > 0].sort_values("metric", ascending=False).head(n),
                     d[d["direction"] < 0].sort_values("metric").head(n)]).drop_duplicates("ticker")
    if sub.empty:
        return ""
    sub = sub.sort_values("metric")                      # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(6.1, max(2.4, 0.32 * len(sub))))
    ypos = range(len(sub))
    ax.barh(list(ypos), sub["metric"], color=[color(x) for x in sub["direction"]],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(sub["market"], fontsize=6.8)
    for i, (mt, pr) in enumerate(zip(sub["metric"], sub["pole_ret"])):
        ax.text(mt + (2.5 if mt >= 0 else -2.5), i, f"{pr * 100:+.0f}%",
                va="center", ha="left" if mt >= 0 else "right", fontsize=6, color="#333")
    ax.set_xlim(-140, 140)
    ax.axvspan(70, 140, color="#2E7D32", alpha=0.05, zorder=0)    # bull setup zone
    ax.axvspan(-140, -70, color="#C62828", alpha=0.05, zorder=0)  # bear setup zone
    ax.set_xlabel("← bear (breaks down)      signed breakout readiness      bull (breaks up) →   (label = flagpole)")
    ax.set_title("Closest to breakout")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)


CHART_CAP = 24   # most-ready flagged setups to chart individually (keeps the report bounded)


def flag_chart_png(h: pd.DataFrame, r) -> str:
    """One trigger-breaking product, full width: price with the **flagpole** (grey),
    the shaded **consolidation channel** + its trendlines, and the dashed **breakout
    line** it's about to cross — with a marker + label on today's close showing how close
    it is. The flag is drawn right on the price chart."""
    h = h.sort_values("date")
    dates = pd.to_datetime(h["date"])
    c = color(r.direction)
    fig, ax = plt.subplots(figsize=(6.1, 2.5))

    mask = h["upper"].notna().to_numpy()
    if mask.any():                                              # the flag: channel + breakout line
        ax.fill_between(dates, h["lower"], h["upper"], where=mask, color=c, alpha=0.13, zorder=1)
        ax.plot(dates, h["upper"], color=c, lw=0.9, zorder=3)
        ax.plot(dates, h["lower"], color=c, lw=0.9, zorder=3)
        ax.plot(dates, h["breakout"], color=c, lw=1.5, ls="--", zorder=5)

    px = [pd.to_datetime(r.pole_base_date), pd.to_datetime(r.pole_tip_date)]   # the flagpole
    py = [r.pole_base_px, r.pole_tip_px]
    ax.plot(px, py, color="#8A8A8A", lw=2.6, solid_capstyle="round", zorder=2)
    ax.scatter(px, py, color="#8A8A8A", s=22, zorder=3)

    ax.plot(dates, h["price"], color="#222", lw=1.4, zorder=4)                 # price + today marker
    lastx, lasty = dates.iloc[-1], float(h["price"].iloc[-1])
    ax.scatter([lastx], [lasty], color=c, s=72, edgecolor="white", linewidth=0.8, zorder=6)
    edge = "broke out" if bool(r.broke) else f"{r.dist_pct:.1f}% to breakout"
    ax.annotate(edge, (lastx, lasty), fontsize=6.4, color=c, fontweight="bold",
                xytext=(-5, 9 if r.sign > 0 else -14), textcoords="offset points", ha="right")

    # Measured-move target (green) and stop (red) levels.
    ax.axhline(r.target, color="#2E7D32", lw=1.0, ls=":", zorder=2)
    ax.axhline(r.stop, color="#C62828", lw=1.0, ls=":", zorder=2)
    ax.text(dates.iloc[0], r.target, "target ", fontsize=5.6, color="#2E7D32", va="bottom", ha="left")
    ax.text(dates.iloc[0], r.stop, "stop ", fontsize=5.6, color="#C62828", va="top", ha="left")

    tick = "  ✓vol" if _vc(r.vol_confirms) is True else ""
    rr = f" · R:R {r.rr:.1f}" if pd.notna(r.rr) else ""
    ax.set_title(f"{r.market} · {r.asset} — {r.pattern}, pole {r.pole_ret * 100:+.0f}% in {int(r.plen)}d · "
                 f"readiness {r.readiness:.0f}/100{rr}{tick}", fontsize=7.3)
    ax.tick_params(labelsize=6, length=2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, color="#EEE", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return png(fig)


def flag_charts(df: pd.DataFrame, hist: pd.DataFrame, cap: int = CHART_CAP):
    """One detailed chart per flagged (trigger-breaking) setup, most-ready first.
    Returns (images, n_flagged) so the report can note if it capped the count."""
    if hist is None or len(hist) == 0:
        return [], 0
    have = set(hist["ticker"].unique())
    fl = flagged(df)
    fl = fl.reindex(fl["readiness"].sort_values(ascending=False).index)       # closest to breakout first
    rows = [r for r in fl.itertuples(index=False) if r.ticker in have]
    imgs = [flag_chart_png(hist[hist["ticker"] == r.ticker], r) for r in rows[:cap]]
    return imgs, len(rows)


def render_html(df: pd.DataFrame, asof: str, threshold: float = 70.0, hist: pd.DataFrame = None) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    df = _reflag(df, threshold)
    fl = flagged(df)

    def _vol(r):
        v = _vc(r.vol_confirms)
        if v is None:
            return "—"
        return f"✓ {r.vol_dryup:.0%}" if (v and pd.notna(r.vol_dryup)) else "✗"

    rows = [{
        "market": r.market, "asset": r.asset, "ticker": r.ticker,
        "signal": r.signal, "direction": int(r.direction), "pattern": r.pattern,
        "pole": f"{r.pole_ret * 100:+.0f}% / {r.plen}d",
        "readiness": f"{r.readiness:.0f}",
        "edge": "broke out" if r.broke else f"{r.dist_pct:.1f}%",
        "target": f"{r.target:g}", "stop": f"{r.stop:g}",
        "rr": "—" if pd.isna(r.rr) else f"{r.rr:.1f}",
        "vol": _vol(r),
    } for r in fl.itertuples(index=False)]

    charts, n_flagged = flag_charts(df, hist)
    return env.get_template("flagreport.html").render(
        asof=pretty_date(asof),
        bars=readiness_bar_png(df),
        charts=charts, n_shown=len(charts), n_flagged=n_flagged,
        rows=rows, n_markets=len(df), tflag=f"{threshold:g}",
        n_bull=int((fl["direction"] > 0).sum()), n_bear=int((fl["direction"] < 0).sum()),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(df: pd.DataFrame, asof: str, out_path, threshold: float = 70.0, hist=None) -> str:
    return render_pdf(render_html(df, asof, threshold, hist), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=70.0)
    args = ap.parse_args()
    detail = Path(args.detail_parquet)
    hist_path = detail.parent / "flag_breakout_history.parquet"
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
