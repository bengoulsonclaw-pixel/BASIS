"""Option Strategy Ticket — a branded one-page PDF snapshot of a position built
in the BASIS Strategy Builder: the legs, the headline risk numbers (net premium,
max P&L, breakevens, P(profit), greeks) and the payoff chart exactly as shown
on screen (front-expiry line + the chosen T+d / vol-shift scenario line).

Driven by a small JSON payload the app writes (grid + both P&L series included,
so the PDF reproduces the screen rather than re-deriving it). Run standalone
(the app calls it as a subprocess):
    python src/optbuilderreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import data_uri, png, pretty_date, render_pdf, BLACK, RICH, CHEAP
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
GOLD, BLUE = "#C8901A", "#1F5FA8"


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def payoff_png(d: dict) -> str:
    """The builder's payoff picture: solid black front-expiry P&L with green/red
    profit shading, gold dashed scenario line, spot + breakeven verticals."""
    xs, exp, scn = d["grid"], d["exp_pnl"], d["scn_pnl"]
    fig, ax = plt.subplots(figsize=(6.1, 3.2))
    ax.fill_between(xs, exp, 0, where=[y >= 0 for y in exp],
                    color=CHEAP, alpha=0.12, zorder=1)
    ax.fill_between(xs, exp, 0, where=[y <= 0 for y in exp],
                    color=RICH, alpha=0.12, zorder=1)
    ax.plot(xs, exp, color=BLACK, lw=2.0, zorder=4, label=d["exp_lbl"])
    ax.plot(xs, scn, color=GOLD, lw=1.8, ls="--", zorder=4, label=d["scn_lbl"])
    ax.axhline(0, color="#888", lw=0.8, zorder=2)
    ax.axvline(d["spot"], color="#999", lw=0.9, ls=":", zorder=3)
    for b in d["bes"]:
        ax.axvline(b, color=BLUE, lw=0.9, ls=(0, (2, 3)), zorder=3)
    ax.set_xlabel("underlying price at expiry")
    ax.set_ylabel(f"P&L ({d['ccy']})" if d["in_ccy"] else "P&L (price points)")
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0)
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def _money(d: dict, pts: float, dec_pts: int = 4) -> str:
    """'12,345 USD (123.4500 pts)' when a point value applies, else 'x pts'."""
    p = f"{pts:,.{dec_pts}f}".rstrip("0").rstrip(".")
    if d["in_ccy"]:
        return f"{pts * d['pv']:,.0f} {d['ccy']}  ({p} pts)"
    return f"{p} pts"


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    rows = []
    for l in d["legs"]:
        fut = l["kind"] == "Future"
        rows.append({
            "side": l["side"], "qty": l["qty"], "kind": l["kind"],
            "strike": f"{l['strike']:,.4f}".rstrip("0").rstrip("."),
            "days": "—" if fut else (f"{l['month']}  ({l['days']:.0f}d)"
                                     if l.get("month") else f"{l['days']:.0f}d"),
            "vol": "—" if fut else f"{l['vol']:.1f}",
            "prem": "—" if fut else f"{l['premium']:,.4f}",
            "prem_ccy": ("—" if fut else f"{l['premium'] * d['pv'] * l['qty']:,.0f}")
            if d["in_ccy"] else None,
            "src": l.get("prem_src", "model"),
            "long": l["side"] == "Buy",
        })
    g = d["greeks"]
    _g = (lambda v: f"{v * d['pv']:+,.0f} {d['ccy']}") if d["in_ccy"] else (lambda v: f"{v:+,.4f}")
    return env.get_template("optbuilderreport.html").render(
        asof=_fmt_asof(d["asof"]), title=d["title"],
        underlying=d["underlying"], ticker=d["ticker"], spot=f"{d['spot']:,.4f}",
        pv=f"{d['pv']:,.2f}", ccy=d["ccy"], in_ccy=d["in_ccy"],
        n_legs=len(d["legs"]), front=f"{d['front']:.0f}",
        rate=f"{d['rate']:.2f}", vol_source=d["vol_source"],
        net_lab="Net debit" if d["net"] >= 0 else "Net credit",
        net=_money(d, abs(d["net"])),
        max_profit="Unlimited" if d["mp_unb"] else _money(d, d["mp"]),
        max_loss="Unlimited" if d["ml_unb"] else _money(d, d["ml"]),
        breakevens="  ·  ".join(f"{b:,.2f}" for b in d["bes"]) if d["bes"] else "none",
        pop=f"{d['pop'] * 100:.0f}%" if d.get("pop") is not None else "n/a",
        delta=f"{g['delta']:+,.3f}", gamma=f"{g['gamma']:+,.5f}",
        vega=_g(g["vega"]), theta=_g(g["theta"]),
        scn_lbl=d["scn_lbl"], chart=payoff_png(d), rows=rows,
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
