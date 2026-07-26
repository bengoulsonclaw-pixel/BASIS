"""Vol Backtest Tearsheet — a branded PDF of one Vol Swap Backtester run: the
delta-hedged ATM straddle spread (buy vol in one product, sell in another), its
cumulative P&L, the gamma/theta/vega/cost attribution, the implied vols behind
the marks and the entry/re-strike/exit log.

Driven by a JSON payload the app writes (so the PDF reproduces exactly what's on
screen). Run standalone (the app calls it as a subprocess):
    python src/volbtreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import data_uri, png, render_pdf, BLACK, RICH, CHEAP, NEUTRAL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"
# product colour scheme (user-set): product 1 / buy = BLUE, product 2 / sell = RED,
# any spread between them = GOLD; implied = dark shade, realized = light shade.
P1, P1_LT = "#1F5FA8", "#7EB3E3"
P2, P2_LT = "#B22222", "#EB9090"
MAX_EVENT_ROWS = 36      # the trade log has its own page now; daily re-striking still caps


def _usd(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < -0.5 else f"${abs(v):,.0f}"


def _dates(d: dict) -> list[date]:
    return [date.fromisoformat(x) for x in d["dates"]]


def cum_png(d: dict) -> str:
    """Cumulative P&L — net (gold) with each leg faint; re-strikes ticked below."""
    s = d["summary"]
    single = bool(s.get("single"))
    dates = _dates(d)
    fig, ax = plt.subplots(figsize=(6.1, 2.15))
    if not single:
        ax.plot(dates, d["buy_cum"], color=P1, lw=1.2, alpha=0.65,
                label=f"Buy {s['buy_name']}")
        ax.plot(dates, d["sell_cum"], color=P2, lw=1.2, alpha=0.65,
                label=f"Sell {s['sell_name']}")
    ax.plot(dates, d["cum_net"], color=GOLD, lw=2.2, label="Net")
    ax.axhline(0, color=BLACK, lw=0.8)
    lo = (min(d["cum_net"]) if single else
          min(min(d["cum_net"]), min(d["buy_cum"]), min(d["sell_cum"])))
    rs = [dt for dt, f in zip(dates, d["restrike"]) if f]
    if rs:
        ax.plot(rs, [lo] * len(rs), ls="", marker="^", ms=3.2, color=NEUTRAL,
                label=f"Re-strike ({len(rs) - 1})")
    ax.set_ylabel("cumulative P&L ($)", fontsize=6.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def att_png(d: dict) -> str:
    """Attribution bars: gamma / theta / vega / higher-order / costs / net."""
    s = d["summary"]
    labels = ["Gamma (realized)", "Theta (carry)", "Vega (IV re-mark)",
              "Higher-order", "Costs", "NET"]
    vals = [s["gamma_pnl"], s["theta_pnl"], s["vega_pnl"], s["resid_pnl"],
            -s["costs"], s["total"]]
    colors = [GOLD if l == "NET" else (CHEAP if v >= 0 else RICH)
              for l, v in zip(labels, vals)]
    fig, ax = plt.subplots(figsize=(6.1, 1.55))
    y = range(len(labels))[::-1]
    ax.barh(list(y), vals, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    # fit the axis to the data (zero sits where the numbers put it, not centred)
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    rng = (hi - lo) or 1.0
    for yy, v in zip(y, vals):
        ax.text(v + (rng * 0.012 if v >= 0 else -rng * 0.012), yy, _usd(v),
                va="center", ha="left" if v >= 0 else "right", fontsize=6, color="#333")
    ax.set_xlim(lo - rng * (0.16 if lo < 0 else 0.02), hi + rng * 0.16)
    ax.grid(True, axis="x", color="#ECECEC", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticklabels([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    return png(fig)


def iv_png(d: dict) -> str:
    """The implied vols behind the daily marks, plus the spread the trade is long."""
    s = d["summary"]
    single = bool(s.get("single"))
    dates = _dates(d)
    fig, ax = plt.subplots(figsize=(6.1, 1.85))
    if single:
        key = "buy_iv" if s.get("buy") else "sell_iv"
        name = s["buy_name"] or s["sell_name"]
        ax.plot(dates, d[key], color=P1, lw=1.8, label=f"{name} IV")
    else:
        spread = [b - v for b, v in zip(d["buy_iv"], d["sell_iv"])]
        ax.plot(dates, d["buy_iv"], color=P1, lw=1.8, label=f"{s['buy_name']} IV")
        ax.plot(dates, d["sell_iv"], color=P2, lw=1.8, label=f"{s['sell_name']} IV")
        ax.plot(dates, spread, color=GOLD, lw=1.8, label="Spread (buy − sell)")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_ylabel("vol points", fontsize=6.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def _vol_phrase(v: dict) -> str:
    """One neutral, observational clause per product from its vol-report row."""
    base = (f"{v['name']} implied is {v['iv']:.1f} against {v['rv']:.1f} realized "
            f"(spread {v['spread']:+.1f} vol)")
    z = v.get("z")
    if z is None:
        return base
    if z <= -1.5:
        tone = "screens notably cheap versus its own implied–realized history"
    elif z <= -0.5:
        tone = "sits on the cheap side of its usual range"
    elif z < 0.5:
        tone = "is close to its normal level"
    elif z < 1.5:
        tone = "sits on the rich side of its usual range"
    else:
        tone = "screens notably rich versus its own implied–realized history"
    return f"{base}; at a z-score of {z:+.1f} the spread {tone}"


def vol_commentary(d: dict) -> str:
    """The 'why this trade' paragraph — built from the vol report's numbers,
    worded as observation (sales commentary), never as a recommendation."""
    s = d["summary"]
    ctx = [v for v in (d.get("volctx") or []) if v.get("iv") is not None]
    if not ctx:
        return ""
    if len(ctx) == 1 or bool(s.get("single")):
        v = ctx[0]
        side = "long" if s.get("buy") else "short"
        return (f"{_vol_phrase(v)}. That backdrop is what puts the product on the radar; "
                f"the backtest below examines how a delta-hedged {side}-volatility position "
                "would have behaved over the chosen window.")
    vb = next((v for v in ctx if v["ticker"] == s.get("buy")), ctx[0])
    vs = next((v for v in ctx if v["ticker"] == s.get("sell")), ctx[-1])
    txt = f"{_vol_phrase(vb)}. {_vol_phrase(vs)}. "
    if vb.get("z") is not None and vs.get("z") is not None:
        gap = vb["z"] - vs["z"]
        if abs(gap) >= 0.5:
            cheaper, richer = (vb, vs) if gap < 0 else (vs, vb)
            txt += (f"Relative to their own histories, {cheaper['name']} vol screens the "
                    f"cheaper of the two and {richer['name']} the richer "
                    f"(z {cheaper['z']:+.1f} vs {richer['z']:+.1f}) — the kind of divergence "
                    "that can make the pair worth a closer look as a spread. ")
        else:
            txt += ("The two screen similarly versus their own histories, so the pair reads "
                    "as a realized-vol comparison more than a re-rating story. ")
    txt += ("The backtest below examines how the delta-hedged spread — long the first "
            "product's volatility against the second — would have behaved.")
    return txt


def volctx_png(d: dict) -> str:
    """Implied-vs-realized (1y) panels, one per product, SIDE BY SIDE, entry marked."""
    vh = d.get("volhist") or {}
    if not vh:
        return ""
    s = d["summary"]
    try:
        entry = date.fromisoformat(str(s["entry"])[:10])
    except Exception:
        entry = None
    n = len(vh)
    fig, axes = plt.subplots(ncols=n, figsize=(6.1, 2.0), squeeze=False)
    for i, (ax, (t, h)) in enumerate(zip(axes[0], vh.items())):
        c_iv, c_rv = (P1, P1_LT) if i == 0 else (P2, P2_LT)   # dark = implied, light = realized
        dates = [date.fromisoformat(x) for x in h["dates"]]
        ax.plot(dates, h["iv"], color=c_iv, lw=1.8, label="Implied (1M ATM)")
        rv = [float("nan") if v is None else v for v in h["rv"]]
        ax.plot(dates, rv, color=c_rv, lw=1.6, label="Realized (1M)")
        spread = [iv_ - rv_ for iv_, rv_ in zip(h["iv"], rv)]
        ax.plot(dates, spread, color=GOLD, lw=1.4, label="Spread (IV − RV)")
        ax.axhline(0, color="#999999", lw=0.6)
        if entry and dates[0] <= entry <= dates[-1]:
            ax.axvline(entry, color=GOLD, lw=1.1, ls="--")
            ax.text(entry, ax.get_ylim()[1], " entry", color=GOLD, fontsize=5.5,
                    va="top", ha="left")
        ax.set_ylabel("vol points", fontsize=6)
        ax.set_title(h["name"], fontsize=7.5, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(labelsize=5.5)
        ax.grid(True, color="#EEE", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=5.2, frameon=True, framealpha=0.9, edgecolor="#ccc")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return png(fig)


def corrroll_png(d: dict) -> str:
    """Rolling 1M pair correlation (returns + IV changes) — the app chart, for
    page 1. Dashed grey = the 1-year return-correlation level; axis fits the data."""
    c = (d.get("corr") or {}).get("rolling")
    if not c or not c.get("px"):
        return ""
    dates = [date.fromisoformat(x) for x in c["dates"]]
    px = [float("nan") if v is None else v for v in c["px"]]
    iv = [float("nan") if v is None else v for v in c["iv"]]
    fig, ax = plt.subplots(figsize=(6.1, 1.7))
    ax.plot(dates, px, color=P1, lw=1.8, label="Returns (21d rolling)")
    ax.plot(dates, iv, color=GOLD, lw=1.8, label="IV changes (21d rolling)")
    lvl = c.get("level")
    if lvl is not None:
        ax.axhline(lvl, color="#777777", lw=0.9, ls="--")
    vals = [v for v in px + iv + ([lvl] if lvl is not None else []) if v == v]
    lo, hi = min(vals), max(vals)
    pad = max(0.03, (hi - lo) * 0.12)
    ax.set_ylim(max(-1.0, lo - pad), min(1.0, hi + pad))
    ax.set_ylabel("correlation", fontsize=6.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def volz_png(d: dict) -> str:
    """Z-score of each product's implied−realized spread through the past year —
    the metric the vol report flags on. Expanding window (min 60 obs), so the
    final point is the report's own 1-year z."""
    vh = d.get("volhist") or {}
    s = d["summary"]
    series = []
    for t, h in vh.items():
        dates = [date.fromisoformat(x) for x in h["dates"]]
        zd, zs, vals = [], [], []
        for dt_, iv, rv in zip(dates, h["iv"], h["rv"]):
            if rv is None or rv != rv:
                continue
            vals.append(iv - rv)
            if len(vals) >= 60:
                sd = float(np.std(vals))
                if sd > 0:
                    zd.append(dt_)
                    zs.append((vals[-1] - float(np.mean(vals))) / sd)
        if zs:
            series.append((h["name"], zd, zs))
    if not series:
        return ""
    try:
        entry = date.fromisoformat(str(s["entry"])[:10])
    except Exception:
        entry = None
    fig, ax = plt.subplots(figsize=(6.1, 1.9))
    for (name, zd, zs), col in zip(series, (P1, P2)):
        ax.plot(zd, zs, color=col, lw=1.8, label=name)
    if len(series) == 2:                       # the RELATIVE read: z spread, leg1 − leg2
        _z2 = dict(zip(series[1][1], series[1][2]))
        _sd = [(dt_, z1 - _z2[dt_]) for dt_, z1 in zip(series[0][1], series[0][2])
               if dt_ in _z2]
        if _sd:
            ax.plot([x[0] for x in _sd], [x[1] for x in _sd], color=GOLD, lw=2.2,
                    label="Z spread (buy − sell)")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.axhline(1.5, color=RICH, lw=0.9, ls="--")
    ax.axhline(-1.5, color=CHEAP, lw=0.9, ls="--")
    ax.text(ax.get_xlim()[0], 1.5, " implied rich", color=RICH, fontsize=6, va="bottom")
    ax.text(ax.get_xlim()[0], -1.5, " implied cheap", color=CHEAP, fontsize=6, va="top")
    if entry:
        ax.axvline(entry, color=NEUTRAL, lw=1.0, ls=":")
    ax.set_ylabel("z-score of IV − RV", fontsize=6.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


# greeks table: (column key, format, is_raw) — raw greeks render blue, $ black
GK_COLS = [("Straddles", "{:+,.1f}", False),
           ("Delta (lots)", "{:+,.2f}", True), ("$ Delta", "{:+,.0f}", False),
           ("Gamma (Δ/pt)", "{:+,.4f}", True), ("$ Gamma (per 1%)", "{:+,.0f}", False),
           ("Vega (pts)", "{:+,.1f}", True), ("$ Vega (per vol pt)", "{:+,.0f}", False),
           ("Theta (pts/day)", "{:+,.1f}", True), ("$ Theta (per day)", "{:+,.0f}", False),
           ("Premium (pts)", "{:+,.0f}", True), ("$ Premium", "{:+,.0f}", False),
           ("$ P&L (cum.)", "{:+,.0f}", False)]
GK_HEADS = ["Straddles", "Δ (lots)", "$ Δ", "Γ (Δ/pt)", "$ Γ per 1%", "Vega (pts)",
            "$ Vega /vol pt", "θ (pts/day)", "$ θ /day", "Prem (pts)", "$ Prem", "$ P&L"]


def _greek_rows(recs: list) -> list:
    out = []
    for r in recs:
        cells = []
        for c, f, raw in GK_COLS:
            v = r.get(c)
            if isinstance(v, (int, float)) and abs(v) < 0.005:
                v = 0.0                              # kill "-0" float dust
            cells.append(("—" if v is None else f.format(v), raw))
        out.append({"pos": r.get("Position", ""), "cells": cells})
    return out


WEIGHT_LABEL = {"gamma": "dollar-gamma neutral",
                "rn_gamma": "risk-normalised gamma (Γ·F²·σ², theta-flat)",
                "vega": "vega neutral",
                "beta_vega": "β-weighted vega", "premium": "premium flat"}
RESTRIKE_LABEL = {"never": "no re-striking", "daily": "re-struck ATM daily",
                  "threshold": "re-struck on drift ≥ {x:g}× the implied daily move"}
EVENT_LABEL = {"entry": "Entry", "daily": "Re-strike (daily)",
               "threshold": "Re-strike (drift)", "exit": "Exit"}


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    s = d["summary"]
    single = bool(s.get("single"))
    side_k = "buy" if s.get("buy") else "sell"
    prod = s["buy_name"] or s["sell_name"]
    events = d.get("events", [])
    shown = events[:MAX_EVENT_ROWS - 1] + events[-1:] if len(events) > MAX_EVENT_ROWS else events
    if single:
        rows = [{
            "date": date.fromisoformat(str(e.get("date", ""))[:10]).strftime("%d %b %y"),
            "event": EVENT_LABEL.get(e.get("event", ""), e.get("event", "")),
            "one_k": f"{e[f'{side_k}_K']:,.2f}", "one_iv": f"{e[f'{side_k}_iv']:.2f}",
            "lots": f"{e[f'{side_k}_lots']:.1f}",
        } for e in shown]
    else:
        rows = [{
            "date": date.fromisoformat(str(e.get("date", ""))[:10]).strftime("%d %b %y"),
            "event": EVENT_LABEL.get(e.get("event", ""), e.get("event", "")),
            "buy_k": f"{e['buy_K']:,.2f}", "sell_k": f"{e['sell_K']:,.2f}",
            "buy_iv": f"{e['buy_iv']:.2f}", "sell_iv": f"{e['sell_iv']:.2f}",
            "ratio": ("—" if e.get("sell_per_buy") in (None, "") or e["event"] == "exit"
                      else f"{float(e['sell_per_buy']):.3f}"),
            "lots": f"{e['buy_lots']:.1f} / {e['sell_lots']:.1f}",
        } for e in shown]
    fmt_d = lambda iso: date.fromisoformat(iso[:10]).strftime("%d %b %Y")
    restrike_note = RESTRIKE_LABEL[s["restrike"]].format(x=s.get("restrike_mult", 1.0))
    blot = []
    for e in d.get("blotter") or []:
        blot.append({
            "date": date.fromisoformat(str(e.get("date", ""))[:10]).strftime("%d %b %y"),
            "product": e.get("product", ""),
            "inst": str(e.get("instrument", "")).capitalize(),
            "action": str(e.get("action", "")).capitalize(),
            "lots": f"{e['lots']:,.2f}",
            "strike": "—" if e.get("strike") is None or e["strike"] != e["strike"]
                      else f"{e['strike']:,.2f}",
            "price": f"{e['price']:,.2f}",
            "cash": f"{e['cash']:+,.0f}",
            "reason": e.get("reason", ""),
        })
    blot_total = f"{sum(e.get('cash', 0.0) for e in d.get('blotter') or []):+,.0f}"
    fx_note = ""
    for _leg, _f in (s.get("fx") or {}).items():
        nm = s["buy_name"] if _leg == "buy" else s["sell_name"]
        fx_note += (f"{nm} converted at {_f['ccy']}USD {_f['rate']:.4f} "
                    "(entry-date rate, frozen). ")
    corr, corr_note = d.get("corr"), ""
    if corr and corr.get("px_1y") is not None and corr.get("px_1m") is not None:
        corr_note = (f"Pair context at entry: 1Y / 1M daily-return correlation "
                     f"{corr['px_1y']:+.2f} / {corr['px_1m']:+.2f}")
        if corr.get("pctl") is not None:
            corr_note += f" (1M in the {corr['pctl']:.0f}th percentile of its rolling year)"
        if corr.get("iv_1y") is not None and corr.get("iv_1m") is not None:
            corr_note += f"; implied-vol-change correlation {corr['iv_1y']:+.2f} / {corr['iv_1m']:+.2f}"
        corr_note += "."
    return env.get_template("volbtreport.html").render(
        buy=s["buy_name"], sell=s["sell_name"], single=single, prod=prod,
        side_word="long" if side_k == "buy" else "short",
        deal_line=(f"{'Long' if side_k == 'buy' else 'Short'} {prod} vol, delta-hedged"
                   if single else f"Buy {s['buy_name']} vol / sell {s['sell_name']} vol"),
        one_iv=f"{s[f'entry_iv_{side_k}']:.1f}", one_rlz=f"{s[f'rlz_{side_k}']:.1f}",
        one_gap=f"{s[f'entry_iv_{side_k}'] - s[f'rlz_{side_k}']:+.1f}",
        entry=fmt_d(s["entry"]), exit=fmt_d(s["exit"]), expiry=fmt_d(s["expiry"]),
        weighting=WEIGHT_LABEL.get(s["weighting"], s["weighting"]),
        restrike_note=restrike_note, corr_note=corr_note, fx_note=fx_note,
        commentary=vol_commentary(d), vol_chart=volctx_png(d),
        corr_roll_chart=corrroll_png(d),
        volz_chart=volz_png(d), blotter=blot, blotter_total=blot_total,
        gk_heads=GK_HEADS,
        greeks_entry=_greek_rows((d.get("greeks") or {}).get("entry") or []),
        greeks_latest=_greek_rows((d.get("greeks") or {}).get("latest") or []),
        greeks_entry_caption=(d.get("greeks") or {}).get("entry_caption", ""),
        greeks_latest_caption=(d.get("greeks") or {}).get("latest_caption", ""),
        n_days=s["n_days"],
        buy_lots=f"{s['buy_lots']:g}", ratio=f"{s['sell_per_buy_entry']:.2f}",
        mode=s["mode"],
        total=_usd(s["total"]), total_buy=_usd(s["total_buy"]), total_sell=_usd(s["total_sell"]),
        max_dd=_usd(s["max_dd"]), costs=_usd(s["costs"]), n_restrikes=s["n_restrikes"],
        iv_spread=f"{s['entry_iv_spread']:+.1f}", rlz_spread=f"{s['rlz_spread']:+.1f}",
        buy_iv=f"{s['entry_iv_buy']:.1f}", sell_iv=f"{s['entry_iv_sell']:.1f}",
        buy_rlz=f"{s['rlz_buy']:.1f}", sell_rlz=f"{s['rlz_sell']:.1f}",
        cum_chart=cum_png(d), att_chart=att_png(d), iv_chart=iv_png(d),
        rows=rows, n_hidden=max(0, len(events) - len(shown)),
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_path) -> str:
    return render_pdf(render_html(d), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload_json")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    d = json.loads(Path(args.payload_json).read_text())
    build_pdf(d, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
