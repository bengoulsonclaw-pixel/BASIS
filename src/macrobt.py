"""Vintage backtest for the policy-rule signal (BASIS · Macro Rate Radar).

The question this module exists to answer, before anyone trades off the Radar:

    Did the rule gap — the distance between what the rules prescribed and where policy
    actually sat — predict where short rates went next?

If the answer is no, the Radar is a commentary tool and should be presented as one. That
finding is worth having either way, so this module is written to report an honest failure
as readily as a success.

Why vintages are the whole point
--------------------------------
A backtest run on today's data is a lie. Real GDP for 2024Q1 reads 23,082 today; on
1 June 2024 it read 22,750 — a 332bn revision. Core PCE, potential output and the natural
rate are all revised, sometimes for years. A rule fed today's history would have "known"
things nobody knew, and would look far better than it was.

So every observation here is rebuilt from ALFRED vintages: for a test date D we ask FRED
what the series looked like ON D, using realtime_start = realtime_end = D. That captures
both revisions and publication lag — on 1 March, Q4 GDP may simply not exist yet, and the
rule has to cope with that exactly as it would have at the time.

This is slow (one API call per series per date) and heavily cached; a 20-year monthly
backtest is thousands of calls. Run it once, keep the store.

Scope
-----
US only. ALFRED is a FRED service and there is no euro-area or UK equivalent — no free
source publishes the vintages needed to do this honestly for the ECB or BoE. Rather than
run a fake real-time test on revised data for those blocs, this module refuses.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from src import macrodata, macrorules

_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "data" / "macro_bt"

# Series that genuinely need vintaging, because they get revised.
_VINTAGE_SERIES = {
    "core_pce": "PCEPILFE",     # index; y/y derived. Revised for years.
    "unemp": "UNRATE",          # revised via seasonal-adjustment updates
    "nairu": "NROU",            # CBO reissues this a couple of times a year
}

# The policy rate is fetched ONCE, un-vintaged, and read as-of each test date.
#
# This is not a shortcut: the daily fed funds rate is an observed settlement and is never
# revised, so its vintage equals its current value by construction. Vintaging it anyway
# cost about 2 seconds per observation, because ALFRED returns the whole ~9,000-point
# daily history on every call — which was most of the runtime of the entire backtest.
_POLICY_SERIES = "DFF"

# Horizons tested, in months. 1-3m is where a macro signal plausibly acts on the strip;
# 12m is included to show whether any edge is just slow mean reversion.
HORIZONS_M = (1, 3, 6, 12)


@dataclass
class Observation:
    when: date
    infl: float
    unemp: float
    nairu: float
    policy: float
    gap_bp: dict = field(default_factory=dict)      # rule key -> (prescribed − policy) bp
    future_policy: dict = field(default_factory=dict)   # horizon months -> policy then
    move_bp: dict = field(default_factory=dict)         # horizon months -> actual move, bp


def _vintage_scalar(sid: str, when: date, *, yoy: bool = False) -> float | None:
    """One series as it stood on `when`, reduced to the latest value then available."""
    s = macrodata.fred(sid, start="1990-01-01", vintage=when, ttl=-1)
    if not s.ok or not s.obs:
        return None
    if yoy:
        s = s.yoy()
        if not s.obs:
            return None
    hit = s.asof(when)
    return hit[1] if hit else None


def build(start: date, end: date, step_months: int = 1,
          progress=None) -> list[Observation]:
    """Assemble point-in-time observations. Cached to disk per date, because each one
    costs four ALFRED calls and the whole point is to be able to re-run the analysis
    without re-fetching a decade of vintages."""
    _STORE.mkdir(parents=True, exist_ok=True)
    policy_series = macrodata.fred(_POLICY_SERIES, start="1990-01-01")
    obs: list[Observation] = []
    d = start
    while d <= end:
        cache = _STORE / f"obs_{d.isoformat()}.json"
        rec = None
        if cache.exists():
            try:
                rec = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                rec = None
        if rec is None:
            infl = _vintage_scalar(_VINTAGE_SERIES["core_pce"], d, yoy=True)
            u = _vintage_scalar(_VINTAGE_SERIES["unemp"], d)
            nr = _vintage_scalar(_VINTAGE_SERIES["nairu"], d)
            hit = policy_series.asof(d) if policy_series.ok else None
            pol = hit[1] if hit else None
            rec = {"when": d.isoformat(), "infl": infl, "unemp": u,
                   "nairu": nr, "policy": pol}
            try:
                cache.write_text(json.dumps(rec), encoding="utf-8")
            except Exception:
                pass
        if progress:
            progress(d, rec)
        if all(rec.get(k) is not None for k in ("infl", "unemp", "nairu", "policy")):
            obs.append(Observation(date.fromisoformat(rec["when"]), rec["infl"],
                                   rec["unemp"], rec["nairu"], rec["policy"]))
        d = _add_months(d, step_months)
    return obs


def _add_months(d: date, n: int) -> date:
    y, m = d.year + (d.month - 1 + n) // 12, (d.month - 1 + n) % 12 + 1
    return date(y, m, min(d.day, 28))


def score(obs: list[Observation], rstar: float = 0.75) -> list[Observation]:
    """Attach each observation's rule gaps and the policy move that actually followed.

    r* is held FIXED rather than taken from HLW vintages. That is a deliberate
    simplification and it matters: HLW's real-time series is published irregularly and
    would add a second moving part to a test whose subject is the rule gap, not the
    neutral rate. It biases the LEVEL of every prescription but not the direction of its
    changes, which is what the correlation below actually measures."""
    by_date = {o.when: o for o in obs}
    for o in obs:
        x = macrorules.RuleInputs(bank="FED", infl=o.infl, rstar=rstar, unemp=o.unemp,
                                  nairu=o.nairu, policy_rate=o.policy,
                                  prev_policy_rate=o.policy)
        for r in macrorules.evaluate(x):
            if r.ok and r.prescribed is not None:
                o.gap_bp[r.key] = (r.prescribed - o.policy) * 100.0
        for h in HORIZONS_M:
            target = _add_months(o.when, h)
            # nearest stored observation within ~3 weeks of the horizon date
            best = min((c for c in by_date if abs((c - target).days) <= 21),
                       key=lambda c: abs((c - target).days), default=None)
            if best is not None:
                o.future_policy[h] = by_date[best].policy
                o.move_bp[h] = (by_date[best].policy - o.policy) * 100.0
    return obs


def dispersion_history(rstar: float = 0.75,
                       rule_keys=("taylor93", "balanced", "shortfalls",
                                  "inertial")) -> list[tuple[date, float]]:
    """Monthly rule dispersion (max − min prescription, bp) rebuilt from the cached
    vintage observations — no network. US only, like everything in this module.

    First-difference is excluded: the store carries no year-ago gap, so that rule
    cannot be evaluated historically, and comparing a 5-rule spread today against a
    4-rule spread in history would overstate the present. Callers must compare a
    4-rule dispersion computed the same way."""
    out = []
    for f in sorted(_STORE.glob("obs_*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(rec.get(k) is None for k in ("infl", "unemp", "nairu", "policy")):
            continue
        x = macrorules.RuleInputs(bank="FED", infl=rec["infl"], rstar=rstar,
                                  unemp=rec["unemp"], nairu=rec["nairu"],
                                  policy_rate=rec["policy"],
                                  prev_policy_rate=rec["policy"])
        vals = [r.prescribed for r in macrorules.evaluate(x)
                if r.ok and r.prescribed is not None and r.key in rule_keys]
        if len(vals) == len(rule_keys):
            out.append((date.fromisoformat(rec["when"]),
                        (max(vals) - min(vals)) * 100.0))
    return out


def dispersion_context(current_bp: float) -> dict | None:
    """Where a 4-rule dispersion reading sits against the vintage history.

    Returns percentile, z-score and summary stats, or None when the store holds too
    little history for the standardisation to mean anything."""
    hist = dispersion_history()
    if len(hist) < 24:
        return None
    vals = [v for _d, v in hist]
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals)
    n_le = sum(1 for v in vals if v <= current_bp)
    # Era medians for narrative feel — what "calm" and "blown out" actually looked like.
    eras = []
    for lbl, lo_y, hi_y in (("2011–15 post-GFC", 2011, 2015),
                            ("2016–19 calm hiking cycle", 2016, 2019),
                            ("2020–22 COVID/inflation shock", 2020, 2022),
                            ("2023– normalisation", 2023, 2100)):
        sub = [v for d, v in hist if lo_y <= d.year <= hi_y]
        if len(sub) >= 12:
            eras.append((lbl, statistics.median(sub)))
    q1, _q2, q3 = statistics.quantiles(vals, n=4)
    return {"n": len(vals), "start": hist[0][0], "end": hist[-1][0],
            "mean": mean, "sd": sd, "median": statistics.median(vals),
            "lo": min(vals), "hi": max(vals), "q1": q1, "q3": q3,
            "eras": eras,
            "pct": 100.0 * n_le / len(vals),
            "z": None if sd == 0 else (current_bp - mean) / sd}


def _corr(xs, ys) -> float | None:
    if len(xs) < 8:
        return None
    try:
        return statistics.correlation(xs, ys)
    except Exception:
        return None


def analyse(obs: list[Observation], rule_key: str = "balanced") -> dict:
    """Does the gap predict the move?

    Reports three things per horizon, because any one alone can flatter:
      * correlation of gap with the subsequent policy move
      * hit rate — how often the gap's SIGN matched the direction of the move, counting
        only observations where policy actually moved more than 10bp (a sign test against
        a sea of unchanged months is meaningless)
      * mean move conditioned on the gap being wide (|gap| > 100bp)
    """
    out = {"rule": rule_key, "n": len(obs), "horizons": {}}
    for h in HORIZONS_M:
        pairs = [(o.gap_bp.get(rule_key), o.move_bp.get(h)) for o in obs]
        pairs = [(g, m) for g, m in pairs if g is not None and m is not None]
        if not pairs:
            continue
        gs, ms = [p[0] for p in pairs], [p[1] for p in pairs]
        moved = [(g, m) for g, m in pairs if abs(m) > 10]
        hits = sum(1 for g, m in moved if (g > 0) == (m > 0))
        wide = [m for g, m in pairs if abs(g) > 100]
        out["horizons"][h] = {
            "n": len(pairs),
            "corr": _corr(gs, ms),
            "n_moved": len(moved),
            "hit_rate": (hits / len(moved)) if moved else None,
            "mean_move_bp": round(sum(ms) / len(ms), 1),
            "mean_move_when_wide_bp": (round(sum(wide) / len(wide), 1) if wide else None),
            "n_wide": len(wide),
        }
    return out


def verdict(analysis: dict) -> str:
    """Plain-language read, written to be as willing to say 'no signal' as 'signal'."""
    hs = analysis.get("horizons", {})
    best, best_c = None, 0.0
    for h, s in hs.items():
        c = s.get("corr")
        if c is not None and abs(c) > abs(best_c):
            best, best_c = h, c
    if best is None:
        return "Not enough overlapping history to judge whether the rule gap predicts anything."
    s = hs[best]
    hr = s.get("hit_rate")
    hr_txt = f", direction matched {hr:.0%} of the time when policy moved" if hr else ""
    if abs(best_c) < 0.2:
        return (f"No meaningful predictive relationship: the strongest correlation across "
                f"horizons is {best_c:+.2f} at {best} months{hr_txt}. On this evidence the "
                f"rule gap is a description of policy stance, not a forecast of it.")
    return (f"The rule gap carries some information: correlation {best_c:+.2f} with the "
            f"policy move {best} months out{hr_txt}. That is a tendency, not a trading "
            f"rule on its own.")


SUMMARY_FILE = _STORE / "summary.json"


def run(start: date | None = None, end: date | None = None,
        step_months: int = 1, rule_key: str = "balanced", progress=None) -> dict:
    """Full backtest. Defaults to 2000-onwards, monthly.

    Persists its summary to data/macro_bt/summary.json — the page reads THAT, because a
    cold run is thousands of ALFRED calls and takes the better part of an hour. Re-run
    from the CLI (python -m src.macrobt) whenever fresh months have accrued; the per-date
    cache makes an incremental re-run cheap."""
    if not macrodata.have_fred_key():
        return {"ok": False, "reason": macrodata.FRED_KEY_HELP}
    start = start or date(2000, 1, 1)
    end = end or _add_months(date.today(), -13)   # need room for the 12m horizon
    obs = score(build(start, end, step_months, progress=progress))
    if not obs:
        return {"ok": False, "reason": "no usable vintage observations were assembled"}
    analyses = {k: analyse(obs, k) for k in ("balanced", "taylor93", "shortfalls",
                                             "inertial", "firstdiff")}
    a = analyses[rule_key]
    out = {"ok": True, "analysis": a, "analyses": analyses, "verdict": verdict(a),
           "n_obs": len(obs), "first": obs[0].when.isoformat(),
           "last": obs[-1].when.isoformat(), "ran": date.today().isoformat()}
    try:
        _STORE.mkdir(parents=True, exist_ok=True)
        SUMMARY_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    except Exception:
        pass
    out["observations"] = obs
    return out


def stored_summary() -> dict | None:
    """The last persisted run, or None. This is what the page shows."""
    try:
        if SUMMARY_FILE.exists():
            return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import time

    # The scheduled task pipes stdout to a log through cmd.exe, whose default cp1252
    # stream can't encode this file's arrows/dashes — the run then dies at the final
    # cosmetic print AFTER doing all its real work, leaving a traceback in the log and
    # a 0x1 in Task Scheduler every month. Reconfigure once instead of chasing glyphs.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    n = [0]

    def _prog(d, rec):
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"  … {n[0]} dates ({d}), {time.time() - t0:.0f}s", flush=True)

    res = run(progress=_prog)
    if not res["ok"]:
        print("FAILED:", res["reason"])
        sys.exit(1)
    print(f"\n{res['n_obs']} observations, {res['first']} → {res['last']}")
    for hk, s in sorted(res["analysis"]["horizons"].items(), key=lambda kv: int(kv[0])):
        c = "—" if s["corr"] is None else f"{s['corr']:+.2f}"
        hr = "—" if s["hit_rate"] is None else f"{s['hit_rate']:.0%}"
        print(f"  {hk:>3}m  corr {c}  hit-rate {hr} (n_moved={s['n_moved']})  "
              f"wide-gap mean move {s['mean_move_when_wide_bp']}bp (n={s['n_wide']})")
    print("\nVERDICT:", res["verdict"])
