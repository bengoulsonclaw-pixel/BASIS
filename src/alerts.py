"""In-app on/off control for the on-screen report alerts (the Home "REPORT DAY" banner and the
release-time popup).

These are pure-UI flags — no Windows Scheduled Task involved (that's automation.py, which governs
the *email* side of a report). Each report's banner and popup can be switched off independently on
the Alert Settings page. Both default **ON**: the alerts fired for every report before this control
existed, so nothing changes until the user deliberately turns one off.

ALERT_REPORTS is the single registry the Alert Settings page iterates. Each entry declares which
release name(s) the app's alert engine emits for that report (so key_for_release maps a live
release back to its toggle) and, where the report also emails, the matching automation.py key.

Flags live in data/alerts.json as {key: {"banner": bool, "popup": bool}}.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

FLAG_FILE = Path(__file__).resolve().parents[1] / "data" / "alerts.json"

# key -> {label, group, email (automation.REPORTS key or None), alerts (release names it covers)}.
# `alerts` empty => the report has no release-day alert (e.g. the daily Morning Coffee) so only its
# email toggle shows. `email` None => no automatic emailer (alert-only) so only banner/popup show.
ALERT_REPORTS = OrderedDict([
    ("morning_coffee", {"label": "Morning Coffee briefing", "group": "Desk & positioning",
                        "email": "morning_coffee", "alerts": []}),
    ("convreport", {"label": "Technical Analysis Report — FICC", "group": "Desk & positioning",
                    "email": "convreport", "alerts": []}),
    ("eq_convreport", {"label": "Technical Analysis Report — Equities", "group": "Desk & positioning",
                       "email": "eq_convreport", "alerts": []}),
    ("sigscore", {"label": "Weekly Signal Scorecard — Technical", "group": "Desk & positioning",
                  "email": "sigscore", "alerts": []}),
    ("weekreview", {"label": "Weekly Review — cross-module wrap", "group": "Desk & positioning",
                    "email": "weekreview", "alerts": []}),
    ("cot", {"label": "CFTC COT positioning", "group": "Desk & positioning",
             "email": "cot", "alerts": ["CFTC COT"]}),
    # Not a calendar release: the banner fires whenever a product pair's 1M correlation sits at
    # an extreme of its own 1-year range (banner_only — there is no release time to popup at).
    ("sectorcorr", {"label": "Correlation breaks (product pairs)", "group": "Desk & positioning",
                    "email": "sectorcorr", "alerts": [], "banner_only": True}),
    ("wasde", {"label": "USDA WASDE", "group": "USDA agriculture",
               "email": "wasde", "alerts": ["USDA WASDE"]}),
    ("usda_reaction", {"label": "USDA Acreage & Grain Stocks", "group": "USDA agriculture",
                       "email": "usda_reaction",
                       "alerts": ["USDA Acreage", "USDA Grain Stocks", "USDA Prospective Plantings"]}),
    ("usda_crop", {"label": "USDA Crop Production & Livestock", "group": "USDA agriculture",
                   "email": None,
                   "alerts": ["USDA Crop Production (Annual)", "USDA Cattle on Feed", "USDA Hogs & Pigs"]}),
    ("opec", {"label": "OPEC Oil Market Report", "group": "Energy",
              "email": "opec", "alerts": ["OPEC MOMR"]}),
    ("precious_metals", {"label": "Precious Metals Fundamentals (monthly)", "group": "Metals",
                         "email": "precious_metals", "alerts": []}),
    ("pm_releases", {"label": "PM release synopses (WGC / WPIC)", "group": "Metals",
                     "email": "pm_releases", "alerts": []}),
    ("iea", {"label": "IEA Oil Market Report", "group": "Energy",
             "email": None, "alerts": ["IEA OMR"]}),
    ("eia_steo", {"label": "EIA Short-Term Energy Outlook", "group": "Energy",
                  "email": None, "alerts": ["EIA STEO"]}),
    ("eia_petroleum", {"label": "EIA Weekly Petroleum Status", "group": "Energy",
                       "email": None, "alerts": ["EIA Petroleum Status"]}),
    ("eia_natgas", {"label": "EIA Weekly Natural Gas Storage", "group": "Energy",
                    "email": None, "alerts": ["EIA Nat Gas Storage"]}),
])

# reverse map: a live release name -> its alert-report key (built once from the registry).
_RELEASE_TO_KEY = {name: key for key, meta in ALERT_REPORTS.items() for name in meta["alerts"]}

KINDS = ("banner", "popup")


def key_for_release(name: str):
    """The alert-report key for a release the app's alert engine produced, or None if unmapped
    (an unmapped release is treated as always-on, so a new report is never silently hidden)."""
    return _RELEASE_TO_KEY.get(name)


def load_flags() -> dict:
    try:
        return json.loads(FLAG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def alert_enabled(key, kind: str) -> bool:
    """Is the `kind` ('banner'|'popup') alert switched on for this report? Defaults to ON
    (fail-open): an unknown/unmapped key, or a report the user has never toggled, shows its alert."""
    if not key:
        return True
    rec = load_flags().get(key)
    if not isinstance(rec, dict) or kind not in rec:
        return True
    return bool(rec[kind])


def set_alert_flag(key: str, kind: str, on: bool) -> None:
    """Persist one banner/popup flag for one report."""
    d = load_flags()
    rec = d.get(key)
    if not isinstance(rec, dict):
        rec = {}
    rec[kind] = bool(on)
    d[key] = rec
    FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLAG_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
