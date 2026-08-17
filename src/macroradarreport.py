"""Macro Rate Radar Report — branded client PDF of the policy-rule dashboard.

One page per requested bank: the five rule prescriptions against the current policy
setting, the prescribed-vs-priced path chart off the STIR Paths fit, and the meeting
table with the spread in bp. Same house machinery as every other BASIS report
(reportkit + templates/_report_style.html), rendered via headless Chromium.

Prose is deliberately observational — "sits Xbp above what the strip prices", never
"buy/sell" — per the house compliance rule for client-facing documents. The contract-edge
table stays IN THE APP only: per-lot P&L framed off a model path reads as advice on paper.

Run standalone (the app calls build() in-process, but the CLI mirrors the other reports):
    python -m src.macroradarreport [--bank FED] [--rule balanced] [--out path.pdf]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:                       # standalone-run import seam
    sys.path.insert(0, str(_SRC))
    sys.path.insert(0, str(_SRC.parent))

from reportkit import pretty_date, data_uri, png, render_pdf, YELLOW  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.dates as mdates                   # noqa: E402
from jinja2 import Environment, FileSystemLoader    # noqa: E402

from src import macrorules, macroradar              # noqa: E402

TEMPLATES = _SRC.parent / "templates"
ASSETS = TEMPLATES / "assets"
OUT_DIR = _SRC.parent / "data" / "reports"

BLUE = "#1F5FA8"
BANK_TITLE = {"FED": "Federal Reserve", "ECB": "European Central Bank",
              "BOE": "Bank of England"}
RULE_FN = {"balanced": macrorules.balanced, "taylor93": macrorules.taylor93,
           "shortfalls": macrorules.shortfalls, "inertial": macrorules.inertial,
           "firstdiff": macrorules.first_difference}


def _path_png(res: "macroradar.RadarResult") -> str:
    """Priced step path vs the rule's prescribed path, policy-rate space."""
    ms = [m for m in res.meetings if m.prescribed is not None]
    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    xs = [m.meeting for m in res.meetings]
    ax.step(xs, [m.priced_policy for m in res.meetings], where="post", color=BLUE,
            lw=2.2, label="Priced by the strip")
    ax.scatter(xs, [m.priced_policy for m in res.meetings], s=14, color=BLUE, zorder=3)
    if ms:
        ax.plot([m.meeting for m in ms], [m.prescribed for m in ms], color="#C8901A",
                lw=2.0, marker="o", ms=3.5, label=f"{res.rule_name} prescription")
    ax.axhline(res.policy_now, color="#999", ls="--", lw=1, label="Policy rate today")
    ax.set_ylabel("Policy rate (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.grid(True, color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.9, edgecolor="#ccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    return png(fig)


def _rules_png(summary: "macrorules.RuleSummary", policy_now: float) -> str:
    """Horizontal bars: each rule's prescription against today's policy setting."""
    rows = [(r.name, r.prescribed) for r in summary.results
            if r.ok and r.prescribed is not None]
    fig, ax = plt.subplots(figsize=(6.1, 2.2))
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    colors = ["#C62828" if v > policy_now else "#2E7D32" for v in vals]
    ax.barh(names, vals, color=colors, alpha=0.75, height=0.55, zorder=2)
    ax.axvline(policy_now, color="#111", lw=1.6, zorder=3)
    ax.annotate(f"policy {policy_now:.2f}%", (policy_now, len(rows) - 0.4),
                fontsize=7, ha="center", va="bottom")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}%", (v, i), fontsize=7,
                    xytext=(4 if v >= policy_now else -4, 0),
                    textcoords="offset points",
                    ha="left" if v >= policy_now else "right", va="center")
    ax.set_xlabel("Prescribed policy rate (%)")
    # Start the axis near the data, not at zero: at zero the whole spread of prescriptions
    # squeezes into the right-hand edge and the chart says nothing.
    lo = min(vals + [policy_now]); hi = max(vals + [policy_now])
    pad = max(0.35, (hi - lo) * 0.18)
    ax.set_xlim(lo - pad, hi + pad)
    ax.grid(True, axis="x", color="#EEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return png(fig)


def _summary_prose(bank: str, res: "macroradar.RadarResult",
                   prov: "macrorules.InputProvenance") -> str:
    s = res.summary
    bits = []
    if s and s.median is not None:
        gap = s.median_gap_bp
        side = "above" if gap > 0 else "below"
        bits.append(f"The median of the five rules sits at <b>{s.median:.2f}%</b>, "
                    f"{abs(gap):.0f}bp {side} the current policy setting of "
                    f"{res.policy_now:.2f}%, with {s.dispersion_bp:.0f}bp of dispersion "
                    f"across rules.")
    hb = res.headline_bp
    if hb is not None and res.meetings:
        last = [m for m in res.meetings if m.spread_bp is not None][-1]
        side = "more tightening" if hb > 0 else "more easing"
        bits.append(f"Taken to the {last.meeting:%B %Y} meeting, the rule path implies "
                    f"{abs(hb):.0f}bp {side} than the futures strip currently prices — "
                    f"a divergence that may be worth a closer look.")
    if prov and prov.assumed:
        bits.append(f"Note: {', '.join(prov.assumed)} "
                    f"{'are' if len(prov.assumed) > 1 else 'is'} an assumption for this "
                    f"bloc rather than a published estimate; the prescription moves with it.")
    return " ".join(bits)


def build(bank: str = "FED", rule_key: str = "balanced", *,
          nairu: float | None = None, rstar: float | None = None,
          out: Path | None = None) -> Path:
    bank = bank.upper()
    rule = RULE_FN.get(rule_key, macrorules.balanced)
    res = macroradar.compare(bank, rule=rule, nairu=nairu, rstar=rstar)
    x, prov = macrorules.inputs_from_data(bank, nairu=nairu, rstar=rstar)

    rule_rows = []
    for r in (res.summary.results if res.summary else []):
        if r.ok and r.prescribed is not None:
            d = r.vs_actual(res.policy_now)
            rule_rows.append({"name": r.name, "value": f"{r.prescribed:.2f}%",
                              "vs": f"{d:+.0f}", "dir": 1 if d > 25 else (-1 if d < -25 else 0),
                              "working": r.formula})
    meet_rows = []
    for m in res.meetings:
        meet_rows.append({
            "meeting": m.meeting.strftime("%d %b %Y"),
            "priced": f"{m.priced_policy:.3f}",
            "cum": f"{m.priced_cum_bp:+.1f}",
            "presc": "—" if m.prescribed is None else f"{m.prescribed:.2f}",
            "spread": "—" if m.spread_bp is None else f"{m.spread_bp:+.0f}",
            "dir": 0 if m.spread_bp is None else (1 if m.spread_bp > 25
                                                  else (-1 if m.spread_bp < -25 else 0)),
        })

    inputs_rows = [{"k": k, "v": v} for k, v in (prov.sources or {}).items()]

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    html = env.get_template("macroradarreport.html").render(
        logo=data_uri(ASSETS / "xp_logo.png") if (ASSETS / "xp_logo.png").exists() else "",
        watermark=data_uri(ASSETS / "xp_mark.png") if (ASSETS / "xp_mark.png").exists() else "",
        asof=pretty_date(date.today()),
        bank=bank, bank_title=BANK_TITLE.get(bank, bank),
        rule_name=res.rule_name,
        policy_now=f"{res.policy_now:.2f}",
        median=("—" if not res.summary or res.summary.median is None
                else f"{res.summary.median:.2f}"),
        median_gap=("—" if not res.summary or res.summary.median_gap_bp is None
                    else f"{res.summary.median_gap_bp:+.0f}"),
        dispersion=("—" if not res.summary or res.summary.dispersion_bp is None
                    else f"{res.summary.dispersion_bp:.0f}"),
        headline=("—" if res.headline_bp is None else f"{res.headline_bp:+.0f}"),
        summary_prose=_summary_prose(bank, res, prov),
        rules_chart=_rules_png(res.summary, res.policy_now) if res.summary else "",
        path_chart=_path_png(res) if res.ok and res.meetings else "",
        have_path=bool(res.ok and res.meetings),
        path_reason=res.reason,
        strip_asof=res.strip_asof or "—",
        rule_rows=rule_rows, meet_rows=meet_rows, inputs_rows=inputs_rows,
        infl=f"{x.infl:.2f}", target=f"{x.target:.2f}", rstar=f"{x.rstar:.2f}",
        gap=("—" if x.gap() is None else f"{x.gap():+.2f}"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = out or OUT_DIR / f"Macro_Rate_Radar_{bank}.pdf"
    render_pdf(html, out)
    return Path(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="FED", choices=["FED", "ECB", "BOE"])
    ap.add_argument("--rule", default="balanced", choices=sorted(RULE_FN))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = build(a.bank, a.rule, out=Path(a.out) if a.out else None)
    print(f"wrote {p}")
