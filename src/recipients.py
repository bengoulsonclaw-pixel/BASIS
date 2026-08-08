"""Per-report email recipient lists, edited on the dashboard's Recipients page and
read by every report emailer (COT, daily digest, and the Morning Coffee project).

Stored in data/email_recipients.json. If the file is absent or a report key is
missing, callers fall back to the Morning Coffee project's hardcoded _EMAIL_TO, so
nothing breaks before the list is ever edited.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

_FILE = Path(__file__).resolve().parents[1] / "data" / "email_recipients.json"
_MC_MAIN = Path(os.getenv("BASIS_MC_DIR", r"C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements")) / "main.py"

# report key -> display label, in the order shown on the Recipients page.
REPORTS = {
    "morning_coffee": "Morning Coffee report",
    "cot": "COT Positioning report",
    "convreport": "Technical Analysis Report — FICC",
    "eq_convreport": "Technical Analysis Report — Equities",
    "digest": "Daily signal digest",
    "opec": "OPEC MOMR synopsis",
    "usda_reaction": "USDA Reaction (Acreage & Grain Stocks)",
    "wasde": "USDA WASDE report",
    "sectorcorr": "Product Correlations alert",
    "sigscore": "Weekly Signal Scorecard — Technical",
    "curve": "Curve / RV Monitor report",
    "precious_metals": "Precious Metals Fundamentals monitor",
    "pm_releases": "Precious Metals release synopses (WGC / WPIC)",
}


def _desk_default() -> list:
    """The current desk list, read from the Morning Coffee project's _EMAIL_TO."""
    try:
        tree = ast.parse(_MC_MAIN.read_text(encoding="utf-8", errors="ignore"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_EMAIL_TO":
                        desk = [str(e).strip() for e in ast.literal_eval(node.value) if str(e).strip()]
                        if desk:
                            return desk
    except Exception:
        pass
    return ["benjamin.goulson@xpi.com.br"]


def _defaults() -> dict:
    desk = _desk_default()
    return {"morning_coffee": list(desk), "cot": list(desk), "digest": [desk[0]],
            "opec": list(desk), "usda_reaction": list(desk),
            "wasde": list(desk), "sectorcorr": [desk[0]], "sigscore": [desk[0]],
            # proofread flow: the monitor goes to the desk only; Ben forwards
            # the checked copy to clients himself.
            "precious_metals": [desk[0]], "pm_releases": [desk[0]]}


def load_all() -> dict:
    """All report lists, with every known report key guaranteed present (seeded
    from the desk default if the file or a key is missing)."""
    try:
        data = json.loads(_FILE.read_text(encoding="utf-8")) if _FILE.exists() else {}
    except Exception:
        data = {}
    defaults = _defaults()
    out = {}
    for k in REPORTS:
        lst = data.get(k)
        # .get with a desk fallback: a REPORTS key absent from _defaults() (e.g.
        # convreport) must not KeyError the whole Recipients page.
        out[k] = ([e.strip() for e in lst if e and e.strip()] if isinstance(lst, list)
                  else defaults.get(k, _desk_default()))
    return out


def save_all(data: dict) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: [e.strip() for e in (data.get(k) or []) if e and e.strip()] for k in REPORTS}
    _FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def get(report: str, fallback: list | None = None) -> list:
    """Recipients for one report; `fallback` (or the desk default) if none set."""
    lst = load_all().get(report, [])
    if lst:
        return lst
    return fallback if fallback is not None else _defaults().get(report, [])
