"""In-app on/off control for the automatic (scheduled) reports.

Each auto-report is a Windows Scheduled Task (created separately). The dashboard's
Recipients page enables/disables those tasks and records the intent in
data/automation.json. The scheduled emailer scripts ALSO check that flag and refuse to
send when it's off — a second safety layer on top of the disabled task, so a report only
ever emails when it has been deliberately switched on in the app.

Everything defaults to OFF (report_enabled -> False when unset).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

FLAG_FILE = Path(__file__).resolve().parents[1] / "data" / "automation.json"

# report key -> metadata. `tasks` are the Windows Scheduled Task names; `recipients` is the
# key into data/email_recipients.json (managed on the same Recipients page).
REPORTS = {
    "cot": {"label": "COT Positioning report",
            "schedule": "Fridays, on the CFTC release",
            "tasks": ["COT Report Emailer (release)"],
            "recipients": "cot"},
    "opec": {"label": "OPEC Oil Market Report synopsis",
             "schedule": "Monthly, on the OPEC MOMR release",
             "tasks": ["OPEC MOMR Synopsis"], "recipients": "opec"},
    "usda_reaction": {"label": "USDA reaction note",
                      "schedule": "On each USDA quarterly grain-stocks release",
                      "tasks": ["USDA Reaction Emailer (release)"], "recipients": "usda_reaction"},
    "wasde": {"label": "USDA WASDE report",
              "schedule": "Monthly, on the WASDE release (~noon ET)",
              "tasks": ["WASDE Report Emailer (release)"], "recipients": "wasde"},
    "morning_coffee": {"label": "Morning Coffee briefing",
                       "schedule": "Daily, on weekday mornings",
                       "tasks": ["Morning Coffee Report"], "recipients": "morning_coffee"},
    "convreport": {"label": "Technical Analysis Report — FICC",
                   "schedule": "Configurable — daily or weekly, any time (see the ⏰ control)",
                   "tasks": ["BASIS Technical Analysis Report (FICC)"],
                   "recipients": "convreport"},
    "eq_convreport": {"label": "Technical Analysis Report — Equities",
                      "schedule": "Configurable — daily or weekly, any time (see the ⏰ control)",
                      "tasks": ["BASIS Technical Analysis Report (Equities)"],
                      "recipients": "eq_convreport"},
    "sectorcorr": {"label": "Product Correlations alert",
                   "schedule": "Daily after the close — sends only when a pair hits an extreme",
                   "tasks": ["Product Correlations Alert (daily)"],
                   "recipients": "sectorcorr"},
    "precious_metals": {"label": "Precious Metals Fundamentals monitor",
                        "schedule": "Monthly, 5th at 07:00 (after LBMA vault data lands)",
                        "tasks": ["Precious Metals Monitor (monthly)"],
                        "recipients": "precious_metals"},
    "sigscore": {"label": "Weekly Signal Scorecard — Technical",
                 "schedule": "Mondays 07:45 — the prior trading week's signal verdicts",
                 "tasks": ["BASIS Weekly Signal Scorecard"],
                 "recipients": "sigscore"},
    "weekreview": {"label": "Weekly Review — cross-module wrap",
                   "schedule": "Mondays 07:55 — the week ahead, off every module's own flags",
                   "tasks": ["BASIS Weekly Review"],
                   "recipients": "weekreview"},
    "pm_releases": {"label": "Precious Metals release synopses (WGC / WPIC)",
                    "schedule": "Daily 08:30 check — sends only on a new release",
                    "tasks": ["PM Release Synopses (daily)"],
                    "recipients": "pm_releases"},
}


def _ps(cmd: str):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                              capture_output=True, text=True, timeout=30)
    except Exception:
        return None


def _task_enabled(name: str) -> str:
    """'Enabled' / 'Disabled' / 'Missing' via the fast native schtasks.exe (Get-ScheduledTask's
    PS module is slow to cold-load, which stalled the render)."""
    try:
        r = subprocess.run(["schtasks", "/query", "/tn", name, "/fo", "LIST", "/v"],
                           capture_output=True, text=True, timeout=8)
    except Exception:
        return "Missing"
    if not r or r.returncode != 0:
        return "Missing"
    for line in r.stdout.splitlines():
        if "Scheduled Task State:" in line:
            return line.split(":", 1)[1].strip()      # Enabled / Disabled
    return "Missing"


def all_report_states() -> dict:
    """{report key: 'on' | 'off' | 'missing'} from the live Windows task state.
    'on' = at least one of the report's tasks is enabled; 'missing' = none exist on this PC."""
    raw = {t: _task_enabled(t) for t in {t for r in REPORTS.values() for t in r["tasks"]}}
    out = {}
    for key, meta in REPORTS.items():
        sts = [raw.get(t, "Missing") for t in meta["tasks"]]
        if all(s == "Missing" for s in sts):
            out[key] = "missing"
        else:
            out[key] = "on" if any(s == "Enabled" for s in sts) else "off"
    return out


def set_report_enabled(key: str, on: bool) -> None:
    """Enable/disable every existing task for a report AND record the flag the scripts honour."""
    verb = "Enable" if on else "Disable"
    for t in REPORTS.get(key, {}).get("tasks", []):
        _ps(f"{verb}-ScheduledTask -TaskName '{t.replace(chr(39), chr(39) * 2)}' "
            f"-ErrorAction SilentlyContinue | Out-Null")
    d = load_flags()
    d[key] = bool(on)
    FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLAG_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def load_flags() -> dict:
    try:
        return json.loads(FLAG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def report_enabled(key: str) -> bool:
    """For the scheduled emailer scripts: send ONLY if explicitly switched on (default OFF)."""
    return bool(load_flags().get(key, False))
