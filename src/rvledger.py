"""RV Ticket Ledger — the RV Trade Tickets engine tracked forward (an out-of-sample scorecard).

The backtest in src/rvtickets.py is an event study over the FULL deep store — in-sample by
construction. This module turns the SAME fade rule into an accountability record going forward:
from the day tracking starts, every fade episode the engine has open (and every one it opens
afterwards) is logged and followed to its outcome, so the realised forward hit-rate and σ-move can
be held up against what the backtest predicted. The question it answers is the one a backtest can't:
"is the edge still paying out of sample?"

DISCIPLINE (the Signal Ledger's hard-won rule, src/sigledger.py). A SETTLED episode is a historical
fact: once win / loss / timeout is written it is NEVER re-measured or rebaselined. Each update only
(a) settles episodes that were still open and (b) appends episodes that have newly opened — the
point is a frozen log, not a number that drifts when the store is re-read. The only deliberate reset
is deleting the store (which restarts the forward track from that day).

The rule is FIXED at the tracker's config (window / threshold / tp_z — default 252 / 2.0σ / full
mean). A track record means ONE unchanging rule; the RV Tickets page's sliders are for exploration,
this is the committed measure. Reads curvemon's own stable series (benchmark yields + raw/second
fronts — the ACTUAL levels, never the re-anchored adjusted store), so a transient bad pull can at
worst leave an open position unresolved for a day; it cannot flip a settled one.

update() runs on each pull (and lazily, cached, on page open); the page aggregates from disk.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import curvemon, rvtickets

STORE = Path(__file__).resolve().parents[1] / "data" / "signals" / "rv_ledger.json"

# The FIXED tracked rule — a track record is one unchanging rule (the page's sliders don't touch it).
WINDOW = rvtickets.WINDOW            # 252
THRESHOLD = rvtickets.Z_THRESHOLD    # 2.0σ flag
TP_Z = 0.0                           # full reversion to the mean


def _hold_cap(half_life: float) -> int:
    return (int(np.clip(round(rvtickets.HOLD_LIVES * half_life),
                        rvtickets.HOLD_MIN, rvtickets.HOLD_MAX))
            if half_life == half_life and half_life > 0 else rvtickets.HOLD_FALLBACK)


def _spread_episodes(spec: dict, history: pd.DataFrame, window: int, threshold: float,
                     tp_z: float) -> list[dict]:
    """Every episode for one spread (open tail kept), with the spread's meta attached."""
    s = curvemon._build_spread(spec, history)
    if s is None or s.dropna().shape[0] < window + 10:
        return []
    hl = curvemon._half_life(s.dropna().tail(window * 2))
    eps = rvtickets._episodes(s, window, threshold, _hold_cap(hl), tp_z, keep_open=True)
    for e in eps:
        e.update({"key": spec["key"], "name": spec["name"], "group": spec["group"],
                  "unit": spec["unit"], "dp": spec["dp"],
                  "signal": "Rich" if e["dir"] < 0 else "Cheap"})
    return eps


def _book_episodes(history: pd.DataFrame, window: int, threshold: float,
                   tp_z: float) -> dict:
    """{key: [episodes]} across the whole spread book — the one deterministic pass."""
    return {spec["key"]: _spread_episodes(spec, history, window, threshold, tp_z)
            for spec in curvemon.SPREADS}


# ── the persisted, append-only store ─────────────────────────────────────────
def load() -> dict:
    try:
        d = json.loads(STORE.read_text(encoding="utf-8"))
        d.setdefault("episodes", [])
        d.setdefault("config", {})
        return d
    except Exception:
        return {"config": {}, "episodes": []}


def _save(d: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def update(history: pd.DataFrame | None = None, today: str | None = None,
           persist: bool = True) -> dict:
    """Settle open episodes, append newly-opened ones, FREEZE everything already settled. On the
    first run the forward track starts at `since` = the latest data date, and the currently-open
    positions (today's stretched book) are taken on and followed from there."""
    if history is None:
        history = curvemon.load_history()
    store = load()
    cfg = dict(store.get("config") or {})
    window = int(cfg.get("window", WINDOW))
    threshold = float(cfg.get("threshold", THRESHOLD))
    tp_z = float(cfg.get("tp_z", TP_Z))

    book = _book_episodes(history, window, threshold, tp_z)
    now_eps, latest = {}, None
    for eps in book.values():
        for e in eps:
            now_eps[(e["key"], e["entry_date"])] = e
            d = e.get("exit_date") or e["entry_date"]
            latest = d if latest is None or d > latest else latest

    since = cfg.get("since")
    if not since:                                     # first run — start the forward track now
        since = latest or (today or date.today().isoformat())
        cfg = {"since": since, "window": window, "threshold": threshold, "tp_z": tp_z}

    def _tracked(e: dict) -> bool:
        # in scope going forward: open now, opened on/after `since`, or RESOLVED on/after `since`
        # (so a position open when tracking began is followed through to its outcome).
        if e["outcome"] == "open" or e["entry_date"] >= since:
            return True
        return bool(e.get("exit_date")) and e["exit_date"] >= since

    stored = {(e["key"], e["entry_date"]): e for e in store["episodes"]}
    out, seen = [], set()
    for k, e in stored.items():                       # refresh stored rows; settled = frozen
        seen.add(k)
        out.append(e if e["outcome"] != "open" else now_eps.get(k, e))
    for k, e in now_eps.items():                      # append newly-opened tracked episodes
        if k not in seen and _tracked(e):
            out.append(e)

    out.sort(key=lambda e: (e["entry_date"], e["key"]))
    result = {"config": cfg, "updated": (today or date.today().isoformat()), "episodes": out}
    if persist:
        _save(result)
    return result


# ── aggregation for the page ─────────────────────────────────────────────────
def _agg(eps: list[dict]) -> dict:
    """Settled hit-rate + realised σ-move for a set of episodes; opens counted but not scored."""
    settled = [e for e in eps if e["outcome"] != "open"]
    wins = [e for e in settled if e["outcome"] == "win"]
    ps = [e["pnl_sigma"] for e in settled if e.get("pnl_sigma") is not None]
    bars = [e["bars"] for e in wins]
    n = len(settled)
    return {
        "n": n, "wins": len(wins),
        "losses": sum(1 for e in settled if e["outcome"] == "loss"),
        "timeouts": sum(1 for e in settled if e["outcome"] == "timeout"),
        "open": sum(1 for e in eps if e["outcome"] == "open"),
        "hit_rate": (len(wins) / n) if n else float("nan"),
        "exp_sigma": float(np.mean(ps)) if ps else float("nan"),
        "med_bars_win": float(np.median(bars)) if bars else float("nan"),
    }


def scorecard(store: dict | None = None, history: pd.DataFrame | None = None) -> dict:
    """Per-spread and overall: the full-history BACKTEST baseline (the 'expected') next to the
    LIVE forward track since tracking began (the 'realised'). Returns rows + an overall summary."""
    if history is None:
        history = curvemon.load_history()
    store = store or load()
    cfg = store.get("config", {})
    window = int(cfg.get("window", WINDOW))
    threshold = float(cfg.get("threshold", THRESHOLD))
    tp_z = float(cfg.get("tp_z", TP_Z))

    baseline = _book_episodes(history, window, threshold, tp_z)       # full-history, recomputed
    live_by_key: dict = {}
    for e in store.get("episodes", []):
        live_by_key.setdefault(e["key"], []).append(e)

    meta = {spec["key"]: spec for spec in curvemon.SPREADS}
    rows = []
    for key, spec in meta.items():
        bt = _agg(baseline.get(key, []))
        lv = _agg(live_by_key.get(key, []))
        if bt["n"] == 0 and lv["n"] == 0 and lv["open"] == 0:
            continue
        rows.append({
            "key": key, "name": spec["name"], "group": spec["group"],
            "bt_n": bt["n"], "bt_hit": bt["hit_rate"], "bt_exp": bt["exp_sigma"],
            "live_n": lv["n"], "live_open": lv["open"], "live_wins": lv["wins"],
            "live_hit": lv["hit_rate"], "live_exp": lv["exp_sigma"],
        })
    # most live activity first, then backtest edge
    rows.sort(key=lambda r: (r["live_n"] + r["live_open"],
                             r["bt_exp"] if r["bt_exp"] == r["bt_exp"] else -9), reverse=True)

    overall = {
        "backtest": _agg([e for eps in baseline.values() for e in eps]),
        "live": _agg([e for eps in live_by_key.values() for e in eps]),
    }
    return {"rows": rows, "overall": overall, "config": cfg,
            "since": cfg.get("since"), "updated": store.get("updated"),
            "window": window, "threshold": threshold, "tp_z": tp_z}
