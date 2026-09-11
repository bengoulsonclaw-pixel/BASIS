"""Auto-fetch the pre-WASDE trade estimates into data/wasde_consensus.json.

The WASDE note's *surprise* column compares the printed US ending stocks to the trade's
pre-report expectation (the Dow Jones survey of analysts, published 1-3 days before each
release and carried by the farm wires — DTN's monthly WASDE preview has printed the full
avg/low/high table every month). Loading it was a manual step and the Sept 2026 note went
out with the column blank — this module makes the scheduled sender fetch it itself.

How: a Claude Messages-API call over plain HTTPS (the BASIS venv has no anthropic SDK) with
the API's `web_search` server tool, on the house model chain (fable-5 -> sonnet-5 -> haiku),
asked to find THIS release's survey and answer in strict JSON. NOTHING is written unless it
passes hard validation:
  * all three crops present with numeric avg/low/high, low <= avg <= high, low < high;
  * magnitudes plausible for mln bu (corn 500-4000, soy 100-1000, wheat 300-1500);
  * the marketing year equals the one the report will print (new-crop);
  * each average within +/-35% of LAST month's printed ending stocks (the stored snapshot)
    — a wrong-crop-year or hallucinated table trips these;
  * the model must cite a source URL, and may answer {"error": "not found"} instead.
On any failure nothing is written and the note degrades exactly as before (level + MoM,
surprise column hidden) — the fetch can never block or corrupt the send.

Timing: wasde_scheduled_email calls prefetch_if_due() on every poll; it acts only from the
day BEFORE a WASDE through release day (the survey exists, the actuals mostly don't), and
failed attempts are throttled to one per 90 minutes because the sender polls every 10.

CLI (testing):  python src/wasde_consensus_fetch.py 2026-09 --dry [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONSENSUS_FILE = ROOT / "data" / "wasde_consensus.json"
SNAP_DIR = ROOT / "data" / "wasde_snapshots"
ATTEMPT_FILE = ROOT / "data" / "signals" / "wasde_consensus_attempt.txt"
ATTEMPT_COOLDOWN_MIN = 90

MODELS = ["claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5"]   # house chain (ai_polish.py)
API_URL = "https://api.anthropic.com/v1/messages"
CROPS = ("Corn", "Soybeans", "Wheat")
BOUNDS = {"Corn": (500, 4000), "Soybeans": (100, 1000), "Wheat": (300, 1500)}   # mln bu sanity
PRIOR_TOL = 0.35            # estimate avg must sit within +/-35% of last month's printed stocks

SYSTEM = (
    "You are a precise data extractor for a futures desk. You find the pre-report analyst "
    "survey ahead of a specific USDA WASDE release and return it as strict JSON. You never "
    "guess: if you cannot find the survey figures with confidence, you say so.")


def _api_key() -> str:
    try:
        from ai_polish import _api_key as k          # single source of truth (env, else MC .env)
        return k()
    except Exception:
        import os
        return os.getenv("ANTHROPIC_API_KEY", "").strip()


def has_month(month: str) -> bool:
    """True if data/wasde_consensus.json already carries usable estimates for YYYY-MM."""
    try:
        d = json.loads(CONSENSUS_FILE.read_text(encoding="utf-8")).get(month) or {}
        return any(isinstance(c, dict) and c.get("avg") is not None for c in d.values())
    except Exception:
        return False


def _expected_my(month: str) -> str:
    """The new-crop marketing-year label the report prints for a given WASDE month:
    the May WASDE introduces the new crop year, so May-Dec show year/(year+1) and
    Jan-Apr still show (year-1)/year — e.g. 2026-10 -> '2026/27', 2027-03 -> '2026/27'."""
    y, m = (int(x) for x in month.split("-"))
    y0 = y if m >= 5 else y - 1
    return f"{y0}/{str(y0 + 1)[2:]}"


def _prior_ends(month: str) -> dict:
    """{crop: last month's printed US ending stocks (mln bu)} from the newest stored
    snapshot strictly before `month`; {} if none."""
    try:
        earlier = sorted(p.stem for p in SNAP_DIR.glob("*.json") if p.stem < month)
        if not earlier:
            return {}
        snap = json.loads((SNAP_DIR / f"{earlier[-1]}.json").read_text(encoding="utf-8"))
        return {c: float(v["end"]) for c, v in snap.items()
                if isinstance(v, dict) and v.get("end") is not None}
    except Exception:
        return {}


def _release_date(month: str):
    """The WASDE release date (Timestamp) inside YYYY-MM, from the USDA calendar; None if absent."""
    try:
        from src import agdata
        import pandas as pd
        cal = agdata.report_calendar()
        w = cal[(cal["report"] == "WASDE") & (cal["date"].dt.strftime("%Y-%m") == month)]
        return w["date"].min() if not w.empty else None
    except Exception:
        return None


def _prompt(month: str, my_label: str) -> str:
    rel = _release_date(month)
    rel_s = rel.strftime("%d %B %Y") if rel is not None else month
    mon_name = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
    return (
        f"Find the PRE-REPORT analyst survey published BEFORE the USDA WASDE report of {mon_name} "
        f"(released {rel_s}) — the Dow Jones survey of analysts as carried by DTN / the farm wires "
        f"(search e.g. \"WASDE {mon_name} preview Dow Jones survey ending stocks\").\n"
        f"I need the trade's PRE-REPORT estimates — NOT the figures USDA actually printed — for "
        f"**US {my_label} (new-crop) ending stocks** in MILLION BUSHELS for corn, soybeans and wheat: "
        f"the survey AVERAGE and the LOW/HIGH of the range of analyst estimates.\n"
        f"Respond with ONLY this JSON object and nothing else:\n"
        f'{{"marketing_year": "{my_label}", "source": "<url of the article you took the numbers from>", '
        f'"Corn": {{"avg": N, "low": N, "high": N}}, "Soybeans": {{"avg": N, "low": N, "high": N}}, '
        f'"Wheat": {{"avg": N, "low": N, "high": N}}}}\n'
        f"HARD RULES: numbers must come from a real article you found via search (cite it in source); "
        f"use the {my_label} NEW-CROP column, never the old-crop one; if you cannot find the pre-report "
        f'survey with confidence, respond with exactly {{"error": "not found"}} instead. Never invent figures.')


def _call_model(model: str, prompt: str, key: str) -> str:
    """One Messages-API call with the web_search server tool; returns concatenated text blocks."""
    import requests
    body = {                                             # no `temperature`: rejected by Claude 5
        "model": model, "max_tokens": 3000, "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
    }
    r = requests.post(API_URL, timeout=280, json=body,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"{model} HTTP {r.status_code}: {r.text[:300]}")
    blocks = r.json().get("content") or []
    return "".join(b.get("text", "") or "" for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text").strip()


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)                 # tolerate prose / ``` fences around it
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    return json.loads(m.group(0))


def _validate(d: dict, my_label: str, prior: dict) -> dict:
    if d.get("error"):
        raise ValueError(f"model reports: {d['error']}")
    my = str(d.get("marketing_year", "")).replace("-", "/").strip()
    if my != my_label:
        raise ValueError(f"marketing year {my!r} != expected {my_label!r}")
    src = str(d.get("source", ""))
    if not src.startswith("http"):
        raise ValueError(f"no source URL cited ({src!r})")
    clean = {}
    for crop in CROPS:
        c = d.get(crop)
        if not isinstance(c, dict):
            raise ValueError(f"{crop} missing")
        try:
            avg, lo, hi = (round(float(c[k])) for k in ("avg", "low", "high"))
        except Exception:
            raise ValueError(f"{crop} values not numeric: {c}")
        b0, b1 = BOUNDS[crop]
        if not (b0 <= avg <= b1) or not (lo <= avg <= hi) or not (lo < hi):
            raise ValueError(f"{crop} implausible: avg={avg} range={lo}-{hi}")
        if crop in prior and prior[crop]:
            drift = abs(avg / prior[crop] - 1.0)
            if drift > PRIOR_TOL:
                raise ValueError(f"{crop} avg {avg} is {drift:+.0%} vs last month's print "
                                 f"{prior[crop]:.0f} (>±{PRIOR_TOL:.0%}) — wrong series?")
        clean[crop] = {"avg": avg, "low": lo, "high": hi}
    return clean


def fetch(month: str) -> tuple:
    """Fetch + validate the survey for YYYY-MM. Returns (estimates, source_url); raises on failure."""
    key = _api_key()
    if not key:
        raise RuntimeError("no ANTHROPIC_API_KEY available")
    my_label = _expected_my(month)
    prior = _prior_ends(month)
    prompt = _prompt(month, my_label)
    last_err = None
    for model in MODELS:
        try:
            raw = _extract_json(_call_model(model, prompt, key))
            clean = _validate(raw, my_label, prior)
            print(f"[consensus] {model} found the {month} survey (source: {raw.get('source')})")
            return clean, str(raw.get("source", ""))
        except Exception as e:
            last_err = e
            print(f"[consensus] {model}: {e}")
    raise RuntimeError(f"all models failed; last: {last_err}")


def _write(month: str, est: dict) -> None:
    try:
        d = json.loads(CONSENSUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        d = {"_format": "Keyed by WASDE month (YYYY-MM); avg/low/high = the pre-report trade "
                        "estimate of US ending stocks in million bushels."}
    d[month] = est
    CONSENSUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONSENSUS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _recently_attempted() -> bool:
    try:
        last = datetime.fromisoformat(ATTEMPT_FILE.read_text().strip())
        return datetime.now() - last < timedelta(minutes=ATTEMPT_COOLDOWN_MIN)
    except Exception:
        return False


def ensure(month: str, force: bool = False) -> str:
    """Idempotent: make sure estimates for YYYY-MM are on disk. Returns 'present' /
    'fetched' / 'failed' / 'throttled'; never raises."""
    try:
        if not force and has_month(month):
            return "present"
        if not force and _recently_attempted():
            print(f"[consensus] {month}: attempted < {ATTEMPT_COOLDOWN_MIN} min ago; waiting.")
            return "throttled"
        est, _src = fetch(month)
        _write(month, est)
        print(f"[consensus] wrote {month} estimates: " +
              ", ".join(f"{c} {v['avg']} ({v['low']}-{v['high']})" for c, v in est.items()))
        return "fetched"
    except Exception as e:
        print(f"[consensus] {month} fetch failed: {e} — the note will fall back to level + MoM.")
        try:
            ATTEMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
            ATTEMPT_FILE.write_text(datetime.now().isoformat())
        except Exception:
            pass
        return "failed"


def prefetch_if_due() -> None:
    """Called by the scheduled sender on every poll: from the day BEFORE a WASDE through
    release day, make sure that release's estimates are loaded. Silent when not due."""
    try:
        from src import agdata
        import pandas as pd
        cal = agdata.report_calendar()
        today = pd.Timestamp.now().normalize()
        nxt = cal[(cal["report"] == "WASDE") & (cal["date"] >= today)]["date"].min()
        if nxt is None or pd.isna(nxt) or (nxt - today).days > 1:
            return
        ensure(nxt.strftime("%Y-%m"))
    except Exception as e:
        print(f"[consensus] prefetch skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("month", help="WASDE month YYYY-MM")
    ap.add_argument("--dry", action="store_true", help="fetch + validate + print, write nothing")
    ap.add_argument("--force", action="store_true", help="ignore presence + attempt throttle")
    args = ap.parse_args()
    if args.dry:
        est, src = fetch(args.month)
        print(json.dumps({"month": args.month, "source": src, **est}, indent=2))
    else:
        print("->", ensure(args.month, force=args.force))


if __name__ == "__main__":
    main()
