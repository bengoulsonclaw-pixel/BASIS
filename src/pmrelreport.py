"""pmrelreport.py — one-page branded PDF for a precious-metals release synopsis.

Consumes the synopsis JSON built by src/pmrel.py and adds one context chart
from the monitor's own live caches: the WPIC synopsis carries the published
Pt/Pd balance-by-year chart (data/pm_pgm.json), the WGC synopsis carries the
reported central-bank net-purchases chart (data/signals/pm_cb.parquet).

CLI:  python src/pmrelreport.py <synopsis.json> <out.pdf>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reportkit import data_uri, render_pdf  # noqa: E402  (sets Agg first)
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

import pmreport  # noqa: E402  (chart helpers, same palette)
import compliance  # noqa: E402  source licence gate

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"
TITLES = {"wgc": "Gold Demand Trends", "wpic": "Platinum Quarterly"}


def _context_chart(pub: str):
    try:
        if pub == "wpic":
            seed = json.loads((ROOT / "data" / "pm_pgm.json").read_text(encoding="utf-8"))
            return (pmreport.chart_pgm_balance(seed["platinum"]["balance_koz"],
                                               seed["palladium"]["balance_koz"]),
                    "Published market balances — platinum & palladium")
        if pub == "wgc":
            cb = pd.read_parquet(ROOT / "data" / "signals" / "pm_cb.parquet")["tonnes"]
            net = cb.diff().dropna()
            blob = {"d": [d.strftime("%Y-%m-%d") for d in net.index],
                    "v": [round(float(v), 1) for v in net.values]}
            return pmreport.chart_cb(blob), "Central-bank net gold purchases (reported, IMF)"
    except Exception:
        pass
    return None, None


class LicenceBlocked(RuntimeError):
    """A source in this synopsis is not cleared for client distribution."""


def licence_blockers(pub: str) -> list:
    """Publication blockers for this synopsis's source. Empty = clear to send."""
    return compliance.publication_check(pub)


def build_pdf(d: dict, out_pdf: Path, *, client_facing: bool = True) -> Path:
    """Render the synopsis.

    client_facing=True enforces the source licence. WGC is `client_ok: False`
    pending XP compliance, and this report auto-sends (`pm_releases` is enabled in
    data/automation.json), so without the gate a licensed extract leaves the
    building unattended. The gate existed but had no caller — the audit found the
    registry, publication_check() and required_citations() all unreferenced outside
    their own module.

    Pass client_facing=False for the in-app preview, which stays internal.
    """
    d = dict(d)
    if client_facing:
        blockers = licence_blockers(d["pub"])
        if blockers:
            raise LicenceBlocked(
                d["pub"].upper() + " synopsis is not cleared for client "
                "distribution: " + "; ".join(blockers))
    # Citation is mandatory wherever the figures appear, internal or not.
    d["citations"] = compliance.required_citations(d["pub"])
    d["title"] = TITLES.get(d["pub"], d["label"])
    chart, chart_title = _context_chart(d["pub"])
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    html = env.get_template("pmrelreport.html").render(
        d=d, chart=chart, chart_title=chart_title,
        logo=data_uri(ASSETS / "logo.png"),
        watermark=data_uri(ASSETS / "building.jpg"),
    )
    render_pdf(html, out_pdf)
    return out_pdf


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: pmrelreport.py <synopsis.json> <out.pdf>")
        return 2
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = build_pdf(d, Path(sys.argv[2]))
    print(f"pmrelreport: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
