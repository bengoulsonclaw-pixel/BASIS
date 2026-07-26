"""Fed Path Report — a branded client PDF snapshot of the implied Fed path off the
SOFR strip, plus the desk's chosen scenario, from the BASIS Fed-Path Calculator.

Sections:
  1. Policy path — the target-midpoint step the market is pricing vs your scenario.
  2. Cumulative move by meeting — bp of easing/tightening priced at each FOMC.
  3. Contract detail — market price vs your fair value, diff in bp and $/lot.

Driven by a small JSON payload the app writes (so the PDF reproduces exactly what's
on screen). Run standalone (the app calls it as a subprocess):
    python src/fedpathreport.py payload.json out.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from reportkit import pretty_date, data_uri, png, render_pdf, BLACK, YELLOW, RICH, CHEAP, NEUTRAL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"
BLUE = "#1F5FA8"


def _fmt_asof(iso: str) -> str:
    try:
        return pretty_date(date.fromisoformat(iso))
    except Exception:
        return iso


def path_png(d: dict) -> str:
    """Step chart: target-midpoint path, market-implied (blue) vs your scenario (gold)."""
    dates = [date.fromisoformat(x) for x in d["seg_dates"]]
    fig, ax = plt.subplots(figsize=(6.1, 3.1))
    ax.step(dates, d["mkt_mid"], where="post", color=BLUE, lw=2.2, label="Market-implied")
    ax.step(dates, d["your_mid"], where="post", color="#C8901A", lw=2.2, label="Your scenario")
    ax.scatter(dates, d["mkt_mid"], s=16, color=BLUE, zorder=3)
    ax.scatter(dates, d["your_mid"], s=16, color="#C8901A", zorder=3)
    ax.set_ylabel("Target midpoint  (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def cum_png(d: dict) -> str:
    """Cumulative move (bp from today) priced at each meeting — market bars + your line."""
    labels = d["meetings"]
    cum = d["cum_bp"]
    # your cumulative from the per-meeting moves
    your_cum, run = [], 0.0
    for m in d["moves"][:len(labels)]:
        run += m
        your_cum.append(run)
    fig, ax = plt.subplots(figsize=(6.1, 2.7))
    x = range(len(labels))
    ax.bar(x, cum, width=0.62, color=[CHEAP if v >= 0 else RICH for v in cum],
           edgecolor="white", zorder=3, label="Market-implied")
    ax.plot(x, your_cum, color="#C8901A", lw=1.8, marker="o", ms=3, zorder=4, label="Your scenario")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.2)
    ax.set_ylabel("cumulative bp vs today")
    ax.grid(True, axis="y", color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    mid = d["mid"]
    cum = d["cum_bp"]
    rows = []
    for i, code in enumerate(d["codes"]):
        diff_bp = (d["your_px"][i] - d["prices"][i]) * 100.0
        rows.append({
            "code": code, "window": d["labels_contract"][i],
            "market": f"{d['prices'][i]:.4f}", "rate": f"{100 - d['prices'][i]:.3f}",
            "your": f"{d['your_px'][i]:.4f}", "diff_bp": f"{diff_bp:+.1f}",
            "diff_usd": f"${diff_bp * 25.0:,.0f}",
            "dir": 1 if diff_bp > 0.05 else (-1 if diff_bp < -0.05 else 0),
        })
    # headline figures
    next_bp = d["per_meeting_bp"][0] if d["per_meeting_bp"] else 0.0
    eoy_idx = [i for i, m in enumerate(d["meetings"]) if m.endswith(str(date.fromisoformat(d["asof"]).year))]
    yend = cum[eoy_idx[-1]] if eoy_idx else cum[-1] if cum else 0.0
    scen_cum = sum(d["moves"][:len(d["meetings"])])
    return env.get_template("fedpathreport.html").render(
        asof=_fmt_asof(d["asof"]), band=d["band"], mid=f"{mid:.3f}",
        n_contracts=d["n_contracts"], n_meetings=len(d["meetings"]),
        next_meeting=d["meetings"][0] if d["meetings"] else "—", next_bp=f"{next_bp:+.0f}",
        yend_bp=f"{yend:+.0f}", yend_year=date.fromisoformat(d["asof"]).year,
        terminal=f"{mid + (cum[-1] if cum else 0)/100:.2f}", terminal_bp=f"{(cum[-1] if cum else 0):+.0f}",
        terminal_mtg=d["meetings"][-1] if d["meetings"] else "—",
        your_terminal=f"{mid + scen_cum/100:.2f}", your_terminal_bp=f"{scen_cum:+.0f}",
        path_chart=path_png(d), cum_chart=cum_png(d), rows=rows,
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
