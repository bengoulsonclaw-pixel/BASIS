"""Post-pull git backup.

After a successful Bloomberg pull the fresh data is committed and pushed to
GitHub in a background thread, so the VPS site (which syncs from GitHub every
15 min) shows same-day data minutes after a pull instead of waiting for the
22:00 nightly backup task.

Failure-tolerant by design: a pull must never be spoiled by a backup hiccup
(no internet, git lock held by the nightly task, etc.) — errors are swallowed
and the nightly task sweeps up anything missed.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)


def _push() -> None:
    try:
        _git("add", "-A")
        if _git("diff", "--cached", "--quiet").returncode != 0:
            _git("commit", "-m", "Data backup after Bloomberg pull")
        _git("push")
    except Exception:
        pass


def push_data_async() -> None:
    """Fire-and-forget: commit + push the repo without blocking the app."""
    threading.Thread(target=_push, name="gitbackup", daemon=True).start()
