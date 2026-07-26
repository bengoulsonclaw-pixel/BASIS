"""Render selected opportunities to a branded PDF via headless Chrome.

Runnable standalone (the app calls it as a subprocess to keep Playwright off
Streamlit's event loop):

    python src/report.py rows.json out.pdf --title "Mean Reversion" --asof "2026-06-03"
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent.parent / "templates"
ASSETS = TEMPLATES / "assets"


def _data_uri(path: Path) -> str:
    """Embed an image as a base64 data URI (so the HTML is self-contained)."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    if data[:4] == b"\x89PNG":
        mime = "image/png"
    elif data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    else:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def render_html(rows, title, asof, trigger="") -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    table = []
    for r in rows:
        metric = f"{r.get('metric', '')} {r.get('metric_label', '')}".strip()
        table.append({
            "market": r.get("market", ""),
            "instruments": r.get("instruments", ""),
            "signal": r.get("signal", ""),
            "metric": metric,
            "level": r.get("level", ""),
            "context": r.get("context", ""),
            "direction": r.get("direction", 0) or 0,
        })
    return env.get_template("report.html").render(
        title=title, asof=pretty_date(asof), rows=table, trigger=trigger,
        logo=_data_uri(ASSETS / "logo.png"),
        watermark=_data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(rows, title, asof, out_path, trigger=""):
    from reportkit import pretty_date, render_pdf   # self-healing headless-Chromium renderer
    return render_pdf(render_html(rows, title, asof, trigger), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_json")
    ap.add_argument("out_pdf")
    ap.add_argument("--title", default="Trading Opportunities")
    ap.add_argument("--asof", default="")
    ap.add_argument("--trigger", default="")
    args = ap.parse_args()
    rows = json.loads(Path(args.rows_json).read_text(encoding="utf-8"))
    try:                                       # honour the dashboard's sector/product filter
        from universe import enabled_tickers as _en, filter_active as _fa
        if _fa():
            _e = _en()
            rows = [r for r in rows
                    if not str(r.get("instruments", "")).strip()
                    or all(p.strip() in _e for p in str(r.get("instruments", "")).split("/"))]
    except Exception:
        pass
    build_pdf(rows, args.title, args.asof, args.out_pdf, args.trigger)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
