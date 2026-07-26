"""Put/Call Ratios Report — a branded client PDF of options put/call positioning
across the book.

  1. Heatmap — every market's OI put/call percentile over the last weeks (red =
     put-heavy extreme · green = call-heavy extreme).
  2. Ranked bar — every market by its current OI P/C percentile.
  3. Products of interest — a table of the put-heavy / call-heavy extremes plus any
     fresh 1-day shifts and flow-vs-OI divergences.
  4. Per-product charts — OI P/C (black) and volume P/C (blue) over time with the
     underlying price (red, right axis), and a percentile oscillator below.

Run standalone (the app calls it as a subprocess):
    python src/pcreport.py data/signals/putcall.parquet out.pdf --asof "2026-06-18" --threshold 80
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "signals" / "putcall_history.parquet"

OI_LINE = "#0A0A0A"      # OI put/call line (black) — the headline basis
VOL_LINE = "#1F5FA8"     # volume put/call line (blue) — today's flow
PRICE_LINE = "#C62828"   # underlying price (red, right axis)
PUT_RED = "#C62828"      # put-heavy extreme — house red
CALL_GREEN = "#2E7D32"   # call-heavy extreme — house green
DISPLAY_DAYS = 120       # ~6 months shown on each per-product chart
HEATMAP_DAYS = 30        # last ~6 weeks of daily columns on the heatmap
DEFAULT_CUTOFF = 80.0    # OI P/C percentile ≥ cutoff → put-heavy; ≤ 100−cutoff → call-heavy
ASSET_ORDER = ["Indices", "STIRs", "Bonds", "FX", "Energy", "Metals", "Agriculture", "Softs"]

# Secondary-flag thresholds — kept in step with src/strategies/putcall.py (the detail
# parquet already carries oi_chg_z and divergence; here we only apply the cutoffs).
SPIKE_Z = 2.0
DIVERGENCE = 35.0
THIN_DAYS = 120          # < this many days of put/call history → flag (its average is still building)

CHART_DPI = 160


def _png(fig) -> str:
    return png(fig, dpi=CHART_DPI)


def _human(v) -> str:
    """Compact contract count: 1,240,000 → '1.24M', 38,400 → '38.4K'."""
    if pd.isna(v):
        return "—"
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


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


def _shade_extreme(ax, dates, pctl, hi, lo):
    d = list(dates)
    for a, b in _runs((pctl >= hi).fillna(False).tolist(), d):
        ax.axvspan(a, b, color=PUT_RED, alpha=0.08, zorder=0)
    for a, b in _runs((pctl <= lo).fillna(False).tolist(), d):
        ax.axvspan(a, b, color=CALL_GREEN, alpha=0.07, zorder=0)


def product_fig(g: pd.DataFrame, hi: float, lo: float, title: str, asset: str, avg_day=None):
    """Per-product chart: OI & volume put/call + price (top), OI-%ile oscillator (mid), and
    daily call/put volume traded vs the ~1y daily average (bottom, when volume is present)."""
    g = g.dropna(subset=["pc_oi"]).sort_values("date")
    dates = pd.to_datetime(g["date"]).tolist()
    pctl = pd.to_numeric(g["oi_pctl"], errors="coerce")

    has_vol = ({"call_vol", "put_vol"}.issubset(g.columns)
               and g[["call_vol", "put_vol"]].notna().any().any())
    if has_vol:
        fig, (ax, axo, axv) = plt.subplots(
            3, 1, figsize=(6.1, 3.9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1.5], "hspace": 0.12})
    else:
        fig, (ax, axo) = plt.subplots(
            2, 1, figsize=(6.1, 2.9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
        axv = None

    # --- panel 1: put/call ratios + price ---
    _shade_extreme(ax, dates, pctl, hi, lo)
    ax.plot(dates, g["pc_oi"], color=OI_LINE, lw=1.4, zorder=4, label="OI P/C")
    if g["pc_vol"].notna().any():
        ax.plot(dates, g["pc_vol"], color=VOL_LINE, lw=0.9, alpha=0.85, zorder=3, label="Vol P/C")
    ax.axhline(1.0, color="#888", lw=0.7, ls="--", zorder=2)
    ax.set_ylabel("Put / Call", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.margins(x=0.01)
    ax.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if g["price"].notna().any():
        axp = ax.twinx()
        axp.plot(dates, g["price"], color=PRICE_LINE, lw=1.1, zorder=5, label="Price")
        axp.set_ylabel("Price", fontsize=6.5, color=PRICE_LINE)
        axp.tick_params(axis="y", labelsize=5.8, colors=PRICE_LINE)
        axp.margins(x=0.01)
        axp.spines["top"].set_visible(False)
        h2, l2 = axp.get_legend_handles_labels()
        handles += h2; labels += l2

    p_now = pctl.dropna()
    p_txt = f"OI P/C %ile {p_now.iloc[-1]:.0f}" if len(p_now) else "OI P/C %ile —"
    ax.set_title(f"{title}  ·  {asset}  ·  {p_txt}", fontsize=7.2, loc="left", fontweight="bold")
    ax.legend(handles, labels, loc="upper left", fontsize=5.3, frameon=True, framealpha=0.85,
              edgecolor="#ccc", ncol=3, columnspacing=0.8, handlelength=1.1, borderpad=0.3)

    # --- panel 2: OI P/C percentile oscillator ---
    axo.plot(dates, pctl, color="#333", lw=1.0, zorder=3)
    axo.axhspan(hi, 100, color=PUT_RED, alpha=0.10, zorder=0)
    axo.axhspan(0, lo, color=CALL_GREEN, alpha=0.10, zorder=0)
    axo.axhline(hi, color=PUT_RED, lw=0.6, ls="--", zorder=1)
    axo.axhline(lo, color=CALL_GREEN, lw=0.6, ls="--", zorder=1)
    axo.set_ylim(0, 100)
    axo.set_yticks([0, 50, 100])
    axo.set_ylabel("%ile", fontsize=6.0)
    axo.tick_params(axis="y", labelsize=5.5)
    axo.tick_params(axis="x", labelsize=5.8)
    for s in ("top", "right"):
        axo.spines[s].set_visible(False)

    # --- panel 3: daily call/put volume traded vs the ~1y daily average ---
    if axv is not None:
        cv = pd.to_numeric(g["call_vol"], errors="coerce").fillna(0.0).to_numpy()
        pv = pd.to_numeric(g["put_vol"], errors="coerce").fillna(0.0).to_numpy()
        axv.bar(dates, cv, width=1.0, color=CALL_GREEN, label="Calls", zorder=2)
        axv.bar(dates, pv, bottom=cv, width=1.0, color=PUT_RED, label="Puts", zorder=2)
        if avg_day is not None and np.isfinite(avg_day) and avg_day > 0:
            axv.axhline(avg_day, color=OI_LINE, lw=0.9, ls="--", zorder=4, label="~1y avg/day")
        axv.set_ylabel("Vol/day", fontsize=6.0)
        axv.tick_params(axis="y", labelsize=5.5)
        axv.tick_params(axis="x", labelsize=5.8)
        axv.margins(x=0.01)
        for s in ("top", "right"):
            axv.spines[s].set_visible(False)
        axv.legend(loc="upper left", fontsize=5.0, frameon=True, framealpha=0.85,
                   edgecolor="#ccc", ncol=3, columnspacing=0.8, handlelength=1.1, borderpad=0.3)
    return fig


def rank_fig(detail: pd.DataFrame, hi: float, lo: float):
    """Every market ranked by current OI P/C percentile, coloured by extreme. Height is
    capped so the whole-book bar (this runs the full ~84-name universe, unlike COT's ~45)
    still fits one A4 page rather than overflowing to a clipped one."""
    d = detail.dropna(subset=["oi_pctl"]).sort_values("oi_pctl")
    fig, ax = plt.subplots(figsize=(6.1, min(10.2, max(3.0, 0.165 * len(d)))))
    colors = [PUT_RED if x >= hi else CALL_GREEN if x <= lo else "#9E9E9E" for x in d["oi_pctl"]]
    y = range(len(d))
    ax.barh(list(y), d["oi_pctl"], color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(50, color="#444", lw=0.8, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["market"], fontsize=5.7)
    for i, v in enumerate(d["oi_pctl"]):
        ax.text(v + (1.2 if v < 92 else -1.2), i, f"{v:.0f}",
                va="center", ha="left" if v < 92 else "right", fontsize=5.3, color="#333")
    ax.set_xlim(0, 100)
    ax.set_xlabel("OI put/call percentile   (0 = most call-heavy in 1y · 50 = mid · 100 = most put-heavy)",
                  fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, axis="x", color="#EEE", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return fig


def activity_fig(detail: pd.DataFrame, top: int | None = None):
    """The daily activity leaderboard — DIVERGING and in % terms so size doesn't distort it.
    Calls point RIGHT (green) and puts point LEFT (red), each as a % of THAT SIDE's own
    1-year daily average. The dashed line at 100% on each side IS the average marker: a bar
    reaching past it traded above average that day, short of it below. Sorted by the bigger
    side's %, so the most extreme movers vs their own norm are on top. (Absolute contract
    counts live in the per-product detail.)"""
    from matplotlib.ticker import FuncFormatter
    d = detail.copy()
    for c in ("call_last", "put_last", "avg_call", "avg_put"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["avg_call"] > 0) & (d["avg_put"] > 0)
          & ((d["call_last"].fillna(0) + d["put_last"].fillna(0)) > 0)].copy()
    d["call_pct"] = d["call_last"].fillna(0) / d["avg_call"] * 100.0      # today's calls, % of avg calls
    d["put_pct"] = d["put_last"].fillna(0) / d["avg_put"] * 100.0         # today's puts,  % of avg puts
    d["rank_pct"] = d[["call_pct", "put_pct"]].max(axis=1)
    d = d.sort_values("rank_pct", ascending=False)
    if top:
        d = d.head(top)
    d = d.iloc[::-1]                       # barh draws bottom-up → most extreme ends on top
    n = len(d)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6.1, 1.2)); ax.axis("off"); return fig

    fig, ax = plt.subplots(figsize=(6.1, min(10.2, max(3.0, 0.26 * n))))
    y = list(range(n))
    ax.barh(y, d["call_pct"], color=CALL_GREEN, edgecolor="white", linewidth=0.3, zorder=3, label="Calls")
    ax.barh(y, -d["put_pct"], color=PUT_RED, edgecolor="white", linewidth=0.3, zorder=3, label="Puts")
    # 100% = that side's 1-year average — the marker line on each side
    ax.axvline(100, color="#222", lw=1.0, ls="--", zorder=5)
    ax.axvline(-100, color="#222", lw=1.0, ls="--", zorder=5)
    ax.axvline(0, color="#222", lw=0.9, zorder=4)
    # Each label carries (Nd) = days of history behind the average; thin (< THIN_DAYS) greyed.
    if "vol_days" in d.columns:
        vd = pd.to_numeric(d["vol_days"], errors="coerce").fillna(0).astype(int).tolist()
    else:
        vd = [None] * n
    thin = [(v is not None and v < THIN_DAYS) for v in vd]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} ({v}d)" if v is not None else str(m) for m, v in zip(d["market"], vd)],
                       fontsize=6.0)
    for _tick, _t in zip(ax.get_yticklabels(), thin):
        if _t:
            _tick.set_color("#999"); _tick.set_fontstyle("italic")
    M = float(max(d["call_pct"].max(), d["put_pct"].max()) or 100.0)      # full scale — keep the extremes
    for i, (cp, pp) in enumerate(zip(d["call_pct"], d["put_pct"])):
        ax.text(cp + M * 0.012, i, f"{cp:.0f}%", va="center", ha="left", fontsize=5.0, color="#1B5E20")
        ax.text(-pp - M * 0.012, i, f"{pp:.0f}%", va="center", ha="right", fontsize=5.0, color="#B71C1C")
    ax.set_xlim(-M * 1.22, M * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))
    ax.set_xlabel("← puts traded      calls traded →      (each as % of that side's own 1-year daily average; "
                  "dashed line = 100% = average · (Nd) = days of history behind the average · grey = under 120 days)",
                  fontsize=5.4)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, axis="x", color="#EEE", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc")
    return fig


def heatmap_fig(hist: pd.DataFrame, detail: pd.DataFrame, days: int = HEATMAP_DAYS):
    """Whole-book OI P/C percentile heatmap: markets (rows, grouped by asset class) ×
    the last `days` days (cols), red (put-heavy) → green (call-heavy)."""
    h = hist.dropna(subset=["oi_pctl"]).copy()
    if h.empty:
        fig, ax = plt.subplots(figsize=(6.1, 1.2)); ax.axis("off"); return fig
    h["date"] = pd.to_datetime(h["date"])
    last_dates = sorted(h["date"].unique())[-days:]
    h = h[h["date"].isin(last_dates)]
    piv = h.pivot_table(index="ticker", columns="date", values="oi_pctl", aggfunc="last")
    meta = detail.set_index("ticker")

    def _key(t):
        a = meta.loc[t, "asset"] if t in meta.index else ""
        ai = ASSET_ORDER.index(a) if a in ASSET_ORDER else len(ASSET_ORDER)
        cur = meta.loc[t, "oi_pctl"] if t in meta.index else 50
        return (ai, -(cur if pd.notna(cur) else 50))

    rows = sorted(piv.index, key=_key)
    piv = piv.reindex(rows)
    labels = [str(meta.loc[t, "market"]) if t in meta.index else t for t in rows]
    row_assets = [str(meta.loc[t, "asset"]) if t in meta.index else "" for t in rows]
    cols = list(piv.columns)
    data = piv.values.astype(float)

    # Height capped so the whole-book heatmap fits BELOW the summary + commentary on
    # page 1 (uncapped, 84 rows × 0.135in ≈ 11in fills a page on its own and spills).
    fig, ax = plt.subplots(figsize=(6.1, min(8.0, max(4.0, 0.135 * len(rows)))))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=5.3)
    step = max(1, len(cols) // 8)
    xt = list(range(0, len(cols), step))
    ax.set_xticks(xt)
    ax.set_xticklabels([pd.Timestamp(cols[i]).strftime("%d %b") for i in xt], fontsize=5.5)
    for i in range(len(rows)):                       # annotate the current (last) column
        v = data[i, -1]
        if pd.notna(v):
            ax.text(len(cols) - 1, i, f"{v:.0f}", va="center", ha="center", fontsize=4.4,
                    color="white" if (v < 26 or v > 74) else "#222")

    # Asset-class group labels (rotated, in the left margin) + white separator lines
    # between groups. Rows are already sorted asset-then-percentile, so each asset is a
    # contiguous block; label it at the block centre and rule a line at each boundary.
    g0 = 0
    for i in range(1, len(row_assets) + 1):
        if i == len(row_assets) or row_assets[i] != row_assets[g0]:
            a = row_assets[g0]
            if a:
                ax.text(-0.27, (g0 + i - 1) / 2.0, a, transform=ax.get_yaxis_transform(),
                        rotation=90, va="center", ha="center", fontsize=6.0,
                        fontweight="bold", color="#111", clip_on=False)
            if i < len(row_assets):                  # divider between this group and the next
                ax.axhline(i - 0.5, color="white", lw=1.6, zorder=6)
            g0 = i

    ax.set_title(f"OI put/call %ile, last {len(cols)} days  ·  red = put-heavy · green = call-heavy",
                 fontsize=7.2, loc="left", fontweight="bold")
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_ticks([0, 50, 100])
    cbar.ax.tick_params(labelsize=5)
    return fig


def _reflag(detail: pd.DataFrame, hi: float, lo: float) -> pd.DataFrame:
    d = detail.copy()
    p = pd.to_numeric(d["oi_pctl"], errors="coerce")
    d["direction"] = np.where(p >= hi, -1, np.where(p <= lo, 1, 0))   # put-heavy=-1 (red), call-heavy=+1 (green)
    d["signal"] = np.where(p >= hi, "Put-heavy", np.where(p <= lo, "Call-heavy", "—"))
    return d


def commentary(detail: pd.DataFrame, hi: float, lo: float) -> str:
    """A single plain-English paragraph summarising the put-heavy / call-heavy extremes,
    the overall tilt and the most notable flow-vs-positioning divergences. Returns HTML."""
    p = pd.to_numeric(detail["oi_pctl"], errors="coerce")
    d = detail[p.notna()].copy()
    if d.empty:
        return "No options put/call data is available for the current universe."
    p = pd.to_numeric(d["oi_pctl"], errors="coerce")
    puts = d[p >= hi].sort_values("oi_pctl", ascending=False)
    calls = d[p <= lo].sort_values("oi_pctl")
    med = float(p.median())
    tilt = ("tilted put-heavy" if med >= 55 else "tilted call-heavy" if med <= 45 else "broadly balanced")

    def _mk(df):
        return ", ".join(f"<b>{html.escape(str(m))}</b> ({v:.0f})"
                         for m, v in zip(df["market"], df["oi_pctl"]))

    sent = []
    if len(puts):
        sent.append(f"open interest is most <b>put-heavy</b> in {_mk(puts.head(3))}")
    if len(calls):
        lead = "and most <b>call-heavy</b> in" if sent else "open interest is most <b>call-heavy</b> in"
        sent.append(f"{lead} {_mk(calls.head(3))}")
    joined = "; ".join(sent)
    extremes = (joined[:1].upper() + joined[1:] + ".") if sent else \
        "No market currently sits at a 1-year put/call extreme."

    dv = d.assign(_d=pd.to_numeric(d["divergence"], errors="coerce").abs())
    dv = dv[dv["_d"] >= DIVERGENCE].sort_values("_d", ascending=False).head(2)
    diver = ""
    if len(dv):
        bits = ", ".join(
            f"<b>{html.escape(str(r.market))}</b> (today's flow {'more put-heavy' if r.divergence > 0 else 'more call-heavy'} than its standing OI)"
            for r in dv.itertuples(index=False))
        diver = f" Notable flow-vs-positioning divergence in {bits}."

    return (f"Across <b>{len(d)}</b> markets with listed-option data, <b>{len(puts)}</b> sit at a 1-year "
            f"<b>put-heavy</b> extreme (OI P/C percentile &ge; {hi:g}) and <b>{len(calls)}</b> at a "
            f"<b>call-heavy</b> extreme (&le; {lo:g}); the book is {tilt} overall (median percentile "
            f"{med:.0f}). {extremes}{diver} A put/call ratio is a sentiment gauge — heavy put demand is "
            f"defensive / bearish, heavy call demand bullish — and extremes are frequently read as contrarian.")


def _activity_lead(detail: pd.DataFrame) -> str:
    """One plain-English line framed RELATIVE to each market's own average (matching the %
    leaderboard): the busiest-vs-normal books and their call/put lean, plus the single
    largest book by raw volume for context."""
    d = detail.copy()
    for c in ("call_last", "put_last", "avg_day"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tot_last"] = d["call_last"].fillna(0) + d["put_last"].fillna(0)
    d = d[(d["avg_day"] > 0) & (d["tot_last"] > 0)].copy()
    if d.empty:
        return "No previous-day options volume is available."
    d["pct"] = d["tot_last"] / d["avg_day"] * 100.0
    # Headline the "busiest" claim on markets with enough history (≥120d) so a thin baseline
    # (e.g. a 2-day average) can't top the lead; fall back to all if too few qualify.
    base = d
    if "vol_days" in d.columns:
        full = d[pd.to_numeric(d["vol_days"], errors="coerce").fillna(0) >= THIN_DAYS]
        if len(full) >= 3:
            base = full
    rows = list(base.sort_values("pct", ascending=False).itertuples(index=False))

    def _one(r):
        lean = "call-led" if (r.call_last or 0) >= (r.put_last or 0) else "put-led"
        return f"<b>{html.escape(str(r.market))}</b> ({r.pct:.0f}% of its 1-year daily average, {lean})"

    lead = ("Relative to their own 1-year daily average, the busiest books were "
            + ", ".join(_one(r) for r in rows[:3]) + ".")
    big = d.sort_values("tot_last", ascending=False).iloc[0]
    lead += (f" The largest book by raw volume was <b>{html.escape(str(big.market))}</b> "
             f"(~{_human(big.tot_last)} contracts).")
    return lead


def render_html(detail: pd.DataFrame, hist: pd.DataFrame, asof: str,
                cutoff: float = DEFAULT_CUTOFF, light: bool = False) -> str:
    hi, lo = cutoff, 100.0 - cutoff
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    detail = _reflag(detail, hi, lo)

    products = []
    for r in detail.itertuples(index=False):
        g = (hist[hist["ticker"] == r.ticker].sort_values("date").tail(DISPLAY_DAYS)
             if not hist.empty else pd.DataFrame())
        if g.empty:
            continue
        products.append({"img": _png(product_fig(g, hi, lo, r.market, r.asset, r.avg_day))})

    def _f2(v):
        return "—" if pd.isna(v) else f"{v:.2f}"

    def _f0(v):
        return "—" if pd.isna(v) else f"{v:.0f}"

    def _fz(v):
        return "—" if pd.isna(v) else f"{v:+.1f}"

    def _fd(v):
        return "—" if pd.isna(v) else f"{v:+.0f}"

    chg = pd.to_numeric(detail["oi_chg_z"], errors="coerce")
    div = pd.to_numeric(detail["divergence"], errors="coerce")
    interest = detail[(detail["direction"] != 0) | (chg.abs() >= SPIKE_Z) | (div.abs() >= DIVERGENCE)]
    rows = [{
        "market": r.market, "asset": r.asset, "signal": r.signal, "direction": int(r.direction),
        "pc_oi": _f2(r.pc_oi), "oi_pctl": _f0(r.oi_pctl),
        "pc_vol": _f2(r.pc_vol), "vol_pctl": _f0(r.vol_pctl),
        "chg": _fz(r.oi_chg_z), "div": _fd(r.divergence),
        "spike": bool(pd.notna(r.oi_chg_z) and abs(r.oi_chg_z) >= SPIKE_Z),
        "diverge": bool(pd.notna(r.divergence) and abs(r.divergence) >= DIVERGENCE),
    } for r in interest.itertuples(index=False)]

    # Whole-book options activity — total puts/calls traded + average per day, most
    # active first, so the heaviest-traded books are obvious at a glance.
    act = detail.dropna(subset=["avg_day"]).sort_values("avg_day", ascending=False)
    activity = [{
        "market": r.market, "asset": r.asset,
        "calls": _human(r.tot_call), "puts": _human(r.tot_put), "avg": _human(r.avg_day),
        "pc_vol": _f2(r.pc_vol),
    } for r in act.itertuples(index=False)]

    try:
        prev_day = pd.to_datetime(hist["date"]).max().strftime("%d %b %Y") if not hist.empty else asof
    except Exception:
        prev_day = asof

    return env.get_template("pcreport.html").render(
        asof=pretty_date(asof), commentary=commentary(detail, hi, lo),
        activity_bar=_png(activity_fig(detail)), prev_day=prev_day, act_lead=_activity_lead(detail),
        heatmap=_png(heatmap_fig(hist, detail)),
        rank=_png(rank_fig(detail, hi, lo)),
        products=products, rows=rows, activity=activity,
        n_markets=int(pd.to_numeric(detail["oi_pctl"], errors="coerce").notna().sum()),
        n_put=int((detail["direction"] < 0).sum()),
        n_call=int((detail["direction"] > 0).sum()),
        spike_z=f"{SPIKE_Z:g}", diverge=f"{DIVERGENCE:g}",
        hi=f"{hi:g}", lo=f"{lo:g}",
        logo=data_uri(ASSETS / "logo.png"),
        watermark="" if light else data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(detail: pd.DataFrame, hist: pd.DataFrame, asof: str, out_path,
              cutoff: float = DEFAULT_CUTOFF, light: bool = False) -> str:
    return render_pdf(render_html(detail, hist, asof, cutoff, light), out_path)


def main():
    global CHART_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("detail_parquet")
    ap.add_argument("out_pdf")
    ap.add_argument("--asof", default="")
    ap.add_argument("--threshold", type=float, default=DEFAULT_CUTOFF,
                    help="put-heavy cutoff on the OI P/C percentile (call-heavy = 100 − this)")
    ap.add_argument("--quality", choices=["screen", "email"], default="screen",
                    help="screen = crisp 160dpi charts; email = lighter 96dpi for a smaller attachment")
    args = ap.parse_args()
    CHART_DPI = 160 if args.quality == "screen" else 96
    detail = pd.read_parquet(args.detail_parquet)
    hist = pd.read_parquet(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            detail = detail[detail["ticker"].isin(_e)]
            if not hist.empty and "ticker" in hist:
                hist = hist[hist["ticker"].isin(_e)]
    except Exception:
        pass
    build_pdf(detail, hist, args.asof, args.out_pdf, args.threshold, light=(args.quality == "email"))
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
