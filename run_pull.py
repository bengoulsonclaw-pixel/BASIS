"""Self-healing Bloomberg pull driver — ends the daily morning babysitting.

Born 2026-08-20 after a week in which every wedge needed a human: the xbbg engine
has no per-request timeout, so a single lost response freezes a pull forever at
~0 CPU (Terminal usually perfectly healthy; an immediate retry succeeds). This
driver automates the whole proven playbook:

  1. PRE-FLIGHT: one raw blpapi request (~2s). Refuses cleanly when the Terminal
     isn't serving or Bloomberg's -4002 WORKFLOW_REVIEW_NEEDED block is active —
     a doomed 10-min run costing hits is never started.
  2. FETCH with a PROGRESS watchdog: healthy pulls write to data/snapshot every
     few minutes (longest observed healthy gap ~5 min); a wedge shows 16-23 min
     of dead air. No new writes for STALL_MIN minutes -> taskkill the whole tree.
  3. ONE automatic retry after a stall/failure.
  4. COMPUTE (local maths, no Terminal needed), then the git data backup.

--auto (the "BASIS Morning Pull" scheduled task, every 15 min 06:00-11:00 wkdays)
adds the trigger conditions: weekday morning + Terminal serving + today's
snapshot still stale + not already running + <3 attempts today. So the whole
morning routine is: open Bloomberg, walk away.

Outcome + timings land in data/snapshot/.pull_driver_status.json and
logs/pull_driver.log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "data" / "snapshot"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "pull_driver.log"
STATUS = SNAP / ".pull_driver_status.json"
LOCK = SNAP / ".pull_driver.lock"

STALL_MIN = 8          # kill the fetch after this many minutes without a snapshot write
FETCH_CAP_MIN = 30     # absolute fetch ceiling per attempt
COMPUTE_CAP_MIN = 45   # compute is local maths; failures here are code bugs, no retry
MAX_AUTO_RUNS = 3      # per day, so a broken morning can't hammer Bloomberg


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _status(**kw) -> None:
    try:
        cur = json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        cur = {}
    if cur.get("date") != date.today().isoformat():
        cur = {"date": date.today().isoformat(), "runs": 0}
    cur.update(kw)
    cur["when"] = datetime.now().isoformat(timespec="seconds")
    SNAP.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(cur, indent=1), encoding="utf-8")


def _read_status() -> dict:
    try:
        cur = json.loads(STATUS.read_text(encoding="utf-8"))
        return cur if cur.get("date") == date.today().isoformat() else {}
    except Exception:
        return {}


def _preflight() -> str:
    """'' = Bloomberg serving; else a human-readable refusal reason."""
    try:
        import blpapi
        opts = blpapi.SessionOptions()
        opts.setServerHost("localhost")
        opts.setServerPort(8194)
        s = blpapi.Session(opts)
        if not s.start():
            return "Terminal API not reachable (Terminal closed or logged out)"
        try:
            if not s.openService("//blp/refdata"):
                return "refdata service refused (not logged in?)"
            svc = s.getService("//blp/refdata")
            req = svc.createRequest("ReferenceDataRequest")
            req.getElement("securities").appendValue("CLA Comdty")
            req.getElement("fields").appendValue("PX_LAST")
            s.sendRequest(req)
            while True:
                ev = s.nextEvent(8000)
                if ev.eventType() == blpapi.Event.TIMEOUT:
                    return "probe timed out"
                for msg in ev:
                    t = str(msg)
                    if "WORKFLOW_REVIEW_NEEDED" in t:
                        return ("Bloomberg -4002 WORKFLOW_REVIEW_NEEDED block active "
                                "- ring the Help Desk; pulling is pointless until cleared")
                    if "PX_LAST" in t:
                        return ""
                if ev.eventType() == blpapi.Event.RESPONSE:
                    return ""
        finally:
            s.stop()
    except Exception as e:
        return f"probe failed: {e!r}"


def _newest_write() -> float:
    try:
        return max(p.stat().st_mtime for p in SNAP.iterdir())
    except Exception:
        return 0.0


_LAST_RC = 0          # exit code of the last phase — 2 = compute ran but some steps died


def _run_phase(args: list[str], stall_min: float | None, cap_min: float, tag: str) -> bool:
    """Run snapshot.py with the given args; kill on write-stall or the hard cap."""
    import os
    env = {**os.environ, "DATAFEED_MODE": "bloomberg", "PYTHONUTF8": "1"}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    out = (ROOT / "logs" / f"pull_driver_{tag}.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([str(PY), "-u", str(ROOT / "snapshot.py"), *args],
                            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    try:
        while proc.poll() is None:
            time.sleep(15)
            el = (time.time() - t0) / 60
            quiet = (time.time() - _newest_write()) / 60
            if el > cap_min or (stall_min and quiet > stall_min and el > stall_min):
                _log(f"{tag}: KILLING — elapsed {el:.1f} min, {quiet:.1f} min since last "
                     f"snapshot write (stall_min={stall_min}, cap={cap_min})")
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                proc.wait(timeout=30)
                return False
        global _LAST_RC
        _LAST_RC = proc.returncode
        # rc 2 from the compute phase means "the snapshot is written, but some independent
        # downstream steps died" — a real thing to report, but NOT a reason to fail the pull
        # and tempt a re-pull that re-spends the day's Bloomberg allowance (2026-08-26).
        ok = proc.returncode in (0, 2)
        _log(f"{tag}: exited rc={proc.returncode} after {(time.time() - t0) / 60:.1f} min")
        return ok
    finally:
        out.close()


def _snapshot_fresh_today() -> bool:
    try:
        m = json.loads((SNAP / "manifest.json").read_text(encoding="utf-8"))
        created = m.get("created", "")
        # created is UTC ISO; date part compared loosely — a pull that finished
        # today in ANY nearby zone counts as fresh
        return created[:10] >= date.today().isoformat()
    except Exception:
        return False


def _terminal_running() -> bool:
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq wintrv.exe", "/FO", "CSV", "/NH"],
                       capture_output=True, text=True)
    return "wintrv.exe" in (r.stdout or "")


def main() -> int:
    auto = "--auto" in sys.argv

    if auto:
        now = datetime.now()
        if now.weekday() >= 5 or not (6 <= now.hour < 11):
            return 0                                    # outside the morning window
        if _snapshot_fresh_today():
            return 0                                    # nothing to do — silent no-op
        if not _terminal_running():
            _log("auto: Terminal not running — waiting for it (next tick retries)")
            return 0
        if _read_status().get("runs", 0) >= MAX_AUTO_RUNS:
            _log("auto: max driver runs reached for today — standing down")
            return 0

    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 90 * 60:
        _log("another driver run is active — exiting")
        return 0
    LOCK.write_text(str(datetime.now()), encoding="utf-8")
    try:
        runs = _read_status().get("runs", 0) + 1
        _status(runs=runs, outcome="running")
        _log(f"=== driver run #{runs} (auto={auto}) ===")

        reason = _preflight()
        if reason:
            _log(f"pre-flight refused: {reason}")
            _status(outcome="preflight_refused", detail=reason)
            return 1
        _log("pre-flight OK — Bloomberg serving")

        t0 = time.time()
        ok = _run_phase(["--fetch"], STALL_MIN, FETCH_CAP_MIN, "fetch")
        if not ok:
            _log("fetch attempt 1 failed/stalled — ONE clean retry (the playbook)")
            _status(outcome="retrying")
            time.sleep(10)
            ok = _run_phase(["--fetch"], STALL_MIN, FETCH_CAP_MIN, "fetch_retry")
        if not ok:
            _log("fetch failed twice — giving up for this run")
            _status(outcome="fetch_failed_twice",
                    detail="both fetch attempts stalled/failed — see logs/pull_driver_*.log")
            return 1

        ok = _run_phase(["--compute"], None, COMPUTE_CAP_MIN, "compute")
        if not ok:
            _status(outcome="compute_failed",
                    detail="fetched data is safe on disk — 'Re-run signals' in the app")
            return 1
        _compute_partial = _LAST_RC == 2

        _log("pushing the data backup…")
        try:
            sys.path.insert(0, str(ROOT))
            from src import gitbackup
            gitbackup._push()
        except Exception as e:
            _log(f"backup push failed (non-fatal): {e!r}")

        mins = (time.time() - t0) / 60
        if _compute_partial:
            _status(outcome="compute_partial",
                    detail=f"complete in {mins:.1f} min, but some compute steps FAILED — "
                           f"see logs/pull_driver_compute.log; those stores are stale until "
                           f"'Re-run signals'. Do NOT re-pull, the Bloomberg data is fine")
            _log(f"=== DONE in {mins:.1f} min — WITH FAILED COMPUTE STEPS ===")
            return 0
        _status(outcome="ok", detail=f"complete in {mins:.1f} min")
        _log(f"=== DONE in {mins:.1f} min ===")
        return 0
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
