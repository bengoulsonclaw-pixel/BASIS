"""STIR Meeting-Risk Map — a branded client PDF for ONE central bank from the
BASIS STIR Paths cockpit: which rate decisions sit inside each futures contract's
window and each option's life, what the strip prices per meeting, the desk
scenario, and where every contract/option-landing sits against it.

Sections:
  1. Contract windows & decisions — the Gantt (bands, decision ticks, last-trade
     dots + prices, listed-option ◇ and 1Y-midcurve △ expiries).
  2. What's priced, meeting by meeting — implied bp, FedWatch-style odds, the
     desk scenario and its expected move.
  3. Policy path — market-implied vs the scenario.
  4. Contract detail + option-expiry landings.

Driven by a JSON payload the app writes (the PDF reproduces exactly what's on
screen, including a SYNTHETIC-DATA banner when built off the Terminal). Run
standalone (the app calls it as a subprocess):
    python src/stirpathreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import pretty_date, data_uri, png, render_pdf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
BLUE = "#1F5FA8"
GOLD = "#C8901A"
RED = "#C0392B"


def _d(iso: str) -> date:
    return date.fromisoformat(iso)


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def gantt_png(d: dict) -> str:
    """The meeting-risk map: one band per contract window, decision ticks inside
    it, last-trade ● with the live price, option ◇ / midcurve △ expiries."""
    g = d["gantt"]
    rows = g["rows"]
    n = max(1, len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * n + 1.15))
    x0, x1 = _d(g["hor_start"]), _d(g["hor_end"])
    for m in g["meetings"]:
        ax.axvline(_d(m), color="#C9C9C9", lw=0.7, ls=(0, (4, 3)), zorder=1)
    for i, r in enumerate(reversed(rows)):          # first row ends up on top
        y = i
        ax.barh(y, (_d(r["end"]) - _d(r["start"])).days, left=_d(r["start"]), height=0.52,
                color=r["color"], alpha=0.28, edgecolor=r["color"], lw=0.8, zorder=2)
        for m in r["mtgs"]:
            ax.plot([_d(m), _d(m)], [y - 0.26, y + 0.26], color=RED, lw=1.8, zorder=4)
        if r.get("fut"):
            ax.plot(_d(r["fut"]), y, "o", ms=5, color=r["color"],
                    mec="#555", mew=0.4, zorder=5)
            if r.get("px") is not None:
                ax.annotate(f"{r['px']:.4f}", (_d(r["fut"]), y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=5.6,
                            fontweight="bold", color="#333", zorder=6)
        for o in r.get("opts", []):
            ax.plot(_d(o), y, "D", ms=4.2, mfc="none", mec=r["color"], mew=1.1, zorder=4)
        for o in r.get("mcs", []):
            ax.plot(_d(o), y, "^", ms=4.8, mfc="none", mec=r["color"], mew=1.1, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=6.4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.65, n - 0.35)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis="x", labelsize=6.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [
        Line2D([], [], color=RED, lw=1.8, label="decision inside window"),
        Line2D([], [], marker="o", color="#666", ls="", ms=5, label="last trade (price above)"),
        Line2D([], [], marker="D", mfc="none", mec="#666", ls="", ms=4.2, label="listed option"),
        Line2D([], [], marker="^", mfc="none", mec="#666", ls="", ms=4.8, label="1Y midcurve"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.8, frameon=False)
    fig.subplots_adjust(bottom=max(0.16, 0.9 / (0.34 * n + 1.15)))
    return png(fig)


def path_png(d: dict) -> str:
    """Step chart: the policy rate path, market-implied (blue) vs scenario (gold)."""
    dates = [_d(x) for x in d["seg_dates"]]
    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    ax.step(dates, d["mkt_seg"], where="post", color=BLUE, lw=2.2, label="Market-implied")
    ax.step(dates, d["your_seg"], where="post", color=GOLD, lw=2.2, label="Desk scenario (expected)")
    ax.scatter(dates, d["mkt_seg"], s=16, color=BLUE, zorder=3)
    ax.scatter(dates, d["your_seg"], s=16, color=GOLD, zorder=3)
    ax.set_ylabel(f"{d['rate_name']}  (%)", fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    odds = d["odds"]
    nxt = odds[0] if odds else {}
    eoy = [o for o in odds if o["decision"].endswith(str(_d(d["asof"]).year % 100).zfill(2))]
    return env.get_template("stirpathreport.html").render(
        bank_name=d["bank_name"], meeting_name=d["meeting_name"], rate_name=d["rate_name"],
        asof=_fmt_asof(d["asof"]), policy=f"{d['policy']:.3f}",
        band=d.get("band"), demo=(d.get("mode") != "bloomberg"),
        haircut=d.get("haircut") or 0.0,
        products=" + ".join(d["products"]),
        next_meeting=nxt.get("decision", "—"), next_bp=f"{nxt.get('implied', 0):+.0f}",
        next_odds=nxt.get("odds", "—"),
        yend_year=_d(d["asof"]).year,
        yend_bp=f"{(eoy[-1]['cum'] if eoy else (odds[-1]['cum'] if odds else 0)):+.0f}",
        terminal=f"{d['policy'] + (odds[-1]['cum'] if odds else 0) / 100:.2f}",
        terminal_bp=f"{(odds[-1]['cum'] if odds else 0):+.0f}",
        terminal_mtg=odds[-1]["decision"] if odds else "—",
        your_terminal=f"{d['policy'] + d['scen_cum'] / 100:.2f}",
        your_terminal_bp=f"{d['scen_cum']:+.0f}",
        scenario_name=d.get("scenario_name"),
        gantt_chart=gantt_png(d), path_chart=path_png(d),
        odds_rows=odds, contract_rows=d["contracts"], landing_rows=d["landings"],
        ccy=d["ccy"],
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(d: dict, out_path) -> str:
    return render_pdf(render_html(d), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload_json")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    d = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))
    build_pdf(d, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
