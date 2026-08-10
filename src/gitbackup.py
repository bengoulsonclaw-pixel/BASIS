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
        _replicate_data_to_main()
    except Exception:
        pass


def _replicate_data_to_main() -> None:
    """The tree often sits on a session's preview branch, which strands the daily
    data commits off main — and the VPS site deploys main only (2026-08-10: the
    site ran a business day stale this way). After any backup on a non-main
    branch, replicate the data/ state to main via a throwaway worktree. DATA
    PATHS ONLY — never code, which on a branch may be another session's WIP."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch in ("main", "HEAD", ""):
        return
    import tempfile
    wt = Path(tempfile.mkdtemp(prefix="basis-data-main-"))
    try:
        if _git("worktree", "add", str(wt), "main").returncode != 0:
            return
        def _w(*args):
            return subprocess.run(["git", *args], cwd=str(wt),
                                  capture_output=True, text=True, timeout=300)
        _w("pull", "--ff-only")
        _w("checkout", branch, "--", "data")
        _w("add", "data")
        if _w("diff", "--cached", "--quiet").returncode != 0:
            _w("commit", "-m", f"Data sync from {branch} auto-backup")
            _w("push")
    finally:
        _git("worktree", "remove", "--force", str(wt))


def push_data_async() -> None:
    """Fire-and-forget: commit + push the repo without blocking the app."""
    threading.Thread(target=_push, name="gitbackup", daemon=True).start()
