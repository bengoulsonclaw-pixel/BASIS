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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"
MAX_EVENT_ROWS = 16      # daily re-striking logs 60+ rows — keep the sheet one page


def _usd(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _dates(d: dict) -> list[date]:
    return [date.fromisoformat(x) for x in d["dates"]]


def cum_png(d: dict) -> str:
    """Cumulative P&L — net (gold) with each leg faint; re-strikes ticked below."""
    s = d["summary"]
    dates = _dates(d)
    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    ax.plot(dates, d["buy_cum"], color=CHEAP, lw=1.2, alpha=0.55,
            label=f"Buy {s['buy_name']}")
    ax.plot(dates, d["sell_cum"], color=RICH, lw=1.2, alpha=0.55,
            label=f"Sell {s['sell_name']}")
    ax.plot(dates, d["cum_net"], color=GOLD, lw=2.2, label="Net")
    ax.axhline(0, color=BLACK, lw=0.8)
    lo = min(min(d["cum_net"]), min(d["buy_cum"]), min(d["sell_cum"]))
    rs = [dt for dt, f in zip(dates, d["restrike"]) if f]
    if rs:
        ax.plot(rs, [lo] * len(rs), ls="", marker="^", ms=3.2, color=NEUTRAL,
                label=f"Re-strike ({len(rs) - 1})")
    ax.set_ylabel("cumulative P&L ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="#ccc")
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
    fig, ax = plt.subplots(figsize=(6.1, 2.0))
    y = range(len(labels))[::-1]
    ax.barh(list(y), vals, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    span = max(abs(v) for v in vals) or 1.0
    for yy, v in zip(y, vals):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), yy, _usd(v),
                va="center", ha="left" if v >= 0 else "right", fontsize=6.4, color="#333")
    ax.set_xlim(-span * 1.28, span * 1.28)
    ax.grid(True, axis="x", color="#ECECEC", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticklabels([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    return png(fig)


def iv_png(d: dict) -> str:
    """The implied vols behind the daily marks, plus the spread the trade is long."""
    s = d["summary"]
    dates = _dates(d)
    spread = [b - v for b, v in zip(d["buy_iv"], d["sell_iv"])]
    fig, ax = plt.subplots(figsize=(6.1, 2.4))
    ax.plot(dates, d["buy_iv"], color=CHEAP, lw=1.8, label=f"{s['buy_name']} IV")
    ax.plot(dates, d["sell_iv"], color=RICH, lw=1.8, label=f"{s['sell_name']} IV")
    ax.plot(dates, spread, color=GOLD, lw=1.8, label="Spread (buy − sell)")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_ylabel("vol points")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


WEIGHT_LABEL = {"gamma": "dollar-gamma neutral", "vega": "vega neutral",
                "beta_vega": "β-weighted vega", "premium": "premium flat"}
RESTRIKE_LABEL = {"never": "no re-striking", "daily": "re-struck ATM daily",
                  "threshold": "re-struck on drift ≥ {x:g}× the implied daily move"}


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    s = d["summary"]
    events = d.get("events", [])
    shown = events[:MAX_EVENT_ROWS - 1] + events[-1:] if len(events) > MAX_EVENT_ROWS else events
    rows = [{
        "date": date.fromisoformat(str(e.get("date", ""))[:10]).strftime("%d %b %y"),
        "event": e.get("event", ""),
        "buy_k": f"{e['buy_K']:,.2f}", "sell_k": f"{e['sell_K']:,.2f}",
        "buy_iv": f"{e['buy_iv']:.2f}", "sell_iv": f"{e['sell_iv']:.2f}",
        "ratio": ("—" if e.get("sell_per_buy") in (None, "") or e["event"] == "exit"
                  else f"{float(e['sell_per_buy']):.3f}"),
        "lots": f"{e['buy_lots']:.1f} / {e['sell_lots']:.1f}",
    } for e in shown]
    fmt_d = lambda iso: date.fromisoformat(iso[:10]).strftime("%d %b %Y")
    restrike_note = RESTRIKE_LABEL[s["restrike"]].format(x=s.get("restrike_mult", 1.0))
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
        buy=s["buy_name"], sell=s["sell_name"],
        entry=fmt_d(s["entry"]), exit=fmt_d(s["exit"]), expiry=fmt_d(s["expiry"]),
        weighting=WEIGHT_LABEL.get(s["weighting"], s["weighting"]),
        restrike_note=restrike_note, corr_note=corr_note, n_days=s["n_days"],
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
