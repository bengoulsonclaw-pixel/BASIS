"""BASIS regression-suite runner.

    python run_tests.py             run the suite (what the pre-push hook calls)
    python run_tests.py --regen     re-baseline the golden files after a DELIBERATE
                                    behaviour change (then eyeball + commit the diff)
    python run_tests.py --install-hook   (re)install the pre-push git hook from hooks/

Runs pytest over tests/ in mock data mode (never touches Bloomberg) and writes the
outcome to logs/last_test_run.json so the 🩺 Data health page can show whether the
safety net is green, red, or has never run. Exit code = pytest's, so the pre-push
hook can block a push on red.

A machine without pytest (VPS, un-provisioned work PC) records a SKIP and exits 0 —
the suite must never brick a data backup push on a box that can't run it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "last_test_run.json"
HOOK_SRC = ROOT / "hooks" / "pre-push"
HOOK_DST = ROOT / ".git" / "hooks" / "pre-push"


def _write_log(payload: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def install_hook() -> int:
    if not HOOK_SRC.exists():
        print(f"hook source missing: {HOOK_SRC}")
        return 1
    HOOK_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HOOK_SRC, HOOK_DST)
    print(f"pre-push hook installed -> {HOOK_DST}")
    return 0


def main(argv: list) -> int:
    if "--install-hook" in argv:
        return install_hook()
    regen = "--regen" in argv
    extra = [a for a in argv if a not in ("--regen",)]

    when = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    if importlib.util.find_spec("pytest") is None:
        print("pytest is not installed in this environment — regression suite SKIPPED "
              "(pip install pytest to enable it).")
        _write_log({"when": when, "ok": None, "skipped": "pytest not installed",
                    "summary": "skipped", "duration_s": 0.0, "regen": False})
        return 0

    env = os.environ.copy()
    env["DATAFEED_MODE"] = "mock"          # belt & braces — conftest forces it too
    if regen:
        env["BASIS_REGEN_GOLDENS"] = "1"
        print("REGEN mode — golden files in tests/goldens/ will be REWRITTEN from "
              "current behaviour. Review the diff before committing.")
    t0 = time.time()
    # -p no:cacheprovider: no .pytest_cache in the repo (OneDrive denies the rename
    # dance it uses, and the churn would sync pointlessly)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q",
                        "-p", "no:cacheprovider", *extra], cwd=str(ROOT), env=env)
    dur = time.time() - t0

    # last line like "24 passed in 3.21s" / "1 failed, 23 passed in 4.0s" — re-derive it
    # cheaply rather than capture (keeping pytest's own output live on the console)
    summary = "passed" if r.returncode == 0 else f"exit {r.returncode}"
    try:                                   # one silent re-collect for an exact count is
        rc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q",  # cheap (<1s)
                             "--collect-only", "-p", "no:cacheprovider"],
                            cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=120)
        m = re.search(r"(\d+) tests? collected", rc.stdout)
        if m:
            summary = (f"{m.group(1)} tests, all green" if r.returncode == 0
                       else f"failures ({m.group(1)} tests collected)")
    except Exception:
        pass

    _write_log({"when": when, "ok": r.returncode == 0, "summary": summary,
                "duration_s": round(dur, 1), "regen": regen})
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
