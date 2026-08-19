"""Opportunistic vol-surface healer.

Bloomberg's smoothed *_DF surfaces (implied vol / skew / term) publish HOURS
after settlement — by 2026-07 the morning pull could miss ~50 markets and the
values often weren't up even by the NEXT morning's pull (see the 2026-07-27
incident). This job runs hourly through the day and heals the gap the moment
it can:

    1. no-op unless the latest SETTLED row of implied_vol.parquet is still
       missing markets that have vendor history (a local read — zero hits);
    2. no-op unless the Bloomberg Terminal is actually reachable right now
       (a 5s local socket probe — the Terminal may be closed; that's fine,
       the next hourly run tries again);
    3. otherwise: re-pull ONLY the surface family (~hundreds of hits, not a
       full snapshot), rebuild signals, and commit+push so the website heals
       too. At most once per day — after a successful heal, step 1 no-ops.

Scheduled task: "BASIS Surface Topup" (hourly 11:00-18:00 weekdays).
Log: %LOCALAPPDATA%\basis_surface_topup.log
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

import pandas as pd  # noqa: E402


def log(*a) -> None:
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def surface_incomplete() -> bool:
    """True when the latest SETTLED implied-vol row is still missing markets.

    'Settled row' = the last row whose date matches the price snapshot's settle
    date (the pull-day intraday row is expected to be sparse and is ignored).
    A market counts as missing when its column has real history but no value
    on the settled row — so permanently-dead surfaces never trigger pulls."""
    try:
        iv = pd.read_parquet("data/snapshot/implied_vol.parquet").set_index("date")
        iv.index = pd.to_datetime(iv.index)
        px = pd.read_parquet("data/snapshot/prices.parquet")
        settle = pd.to_datetime(px["date"]).max().normalize()
        if settle not in iv.index:
            return True
        row = iv.loc[settle]
        recent = iv.loc[iv.index < settle].tail(10)
        had_data = recent.notna().any()
        missing = [t for t in iv.columns if had_data.get(t, False)
                   and (row[t] != row[t])]
        if missing:
            log(f"settled row {settle.date()} missing {len(missing)} markets "
                f"(e.g. {', '.join(missing[:5])})")
        return bool(missing)
    except Exception as e:
        log(f"completeness check failed ({e}) — treating as complete, no pull")
        return False


def terminal_reachable() -> bool:
    try:
        import blpapi
        opts = blpapi.SessionOptions()
        opts.setServerHost("localhost"); opts.setServerPort(8194)
        opts.setNumStartAttempts(1)
        s = blpapi.Session(opts)
        ok = s.start()
        try:
            s.stop()
        except Exception:
            pass
        return bool(ok)
    except Exception:
        return False


def main() -> int:
    if not surface_incomplete():
        return 0                                   # healed already — silent no-op
    if not terminal_reachable():
        log("surface incomplete but Terminal not reachable — will retry next hour")
        return 0
    log("healing: re-pulling implied vol / skew / term from Bloomberg…")
    os.environ["DATAFEED_MODE"] = "bloomberg"
    import snapshot
    from src.datafeed import (INSTRUMENTS, get_implied_vol_history,
                              get_skew_components, get_term_structure)
    tickers = list(INSTRUMENTS)
    iv = get_implied_vol_history(tickers)
    if iv is None or iv.empty:
        log("pull returned nothing — kept existing files"); return 0
    snapshot._save(iv, "implied_vol")
    skew = get_skew_components(tickers, atm=iv)     # share the ATM frame — same dedupe
    snapshot._save(skew["put"], "skew_put")         # as the morning snapshot (2026-08-18)
    snapshot._save(skew["call"], "skew_call")
    snapshot._save(skew["atm"], "skew_atm")
    ts = get_term_structure(tickers, atm=iv)
    for lab in snapshot.TENOR_LABELS:
        snapshot._save(ts[lab], f"term_{lab.lower()}")
    cov = int(iv.tail(2).notna().sum(axis=1).iloc[0])
    log(f"surfaces saved (settled-row coverage now ~{cov})")
    # Signals rebuild in a CLEAN subprocess (snapshot mode) — this process imported
    # datafeed in bloomberg mode and module state doesn't un-flip reliably.
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "DATAFEED_MODE"}
    res = subprocess.run([sys.executable, "run_daily.py"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=1800)
    log("signals rebuild:", (res.stdout or res.stderr or "").strip().splitlines()[-1]
        if (res.stdout or res.stderr) else f"exit {res.returncode}")
    try:
        from src import gitbackup
        gitbackup._push()                          # synchronous: commit + push now
        log("data pushed to GitHub (website follows within 15 min)")
    except Exception as e:
        log(f"push skipped ({e}) — nightly backup will sweep it up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
