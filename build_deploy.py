#!/usr/bin/env python
"""Build a clean, self-contained BASIS bundle to copy to another PC (e.g. the work PC).

Run it on THIS (development) PC with the project's venv:

    .venv\\Scripts\\python build_deploy.py

It produces a folder (default: C:\\Users\\<you>\\BASIS_deploy) containing:
  • the app code + the data it needs to start (no .venv, no scratch, no giant PDFs)
  • wheels\\            — every Python package pre-downloaded, so the work PC installs
                          OFFLINE (its restricted/proxy internet is never touched)
  • playwright-browsers\\ — the headless-Chromium the PDF reports use (no download needed)
  • SETUP_WORK_PC.md + setup_work_pc.bat — what to run over there

Re-run it any time you've updated BASIS here to refresh the bundle, then copy the folder
across again. Options:
    --out PATH      where to build the bundle (default ~/BASIS_deploy)
    --pyver X.Y     target Python version for the wheels (default: this PC's version).
                    Set this if the WORK PC runs a different Python than this one.
    --no-wheels     skip the offline package download (code/data only)
    --no-browser    skip copying the Playwright browser
    --zip           also produce BASIS_deploy.zip next to the folder
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BBG_INDEX = "https://blpapi.bloomberg.com/repository/releases/python/simple/"

# Names ignored anywhere in the tree when copying the code.
IGNORE_NAMES = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", ".mypy_cache",
                ".ipynb_checkpoints", "BASIS_deploy"}
IGNORE_SUFFIX = (".pyc", ".pyo")
# Scratch / dev-only files at the project root that the app doesn't need.
IGNORE_GLOBS = ("_*.py", "_*.png", "_*.html", "diag_*.py", "diag_*.bat", "probe_*.py",
                "history_check.txt", "*.log")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ignore(dir_path, names):
    out = set()
    d = Path(dir_path)
    for n in names:
        if n in IGNORE_NAMES or n.endswith(IGNORE_SUFFIX):
            out.add(n)
            continue
        # only apply scratch globs at the project root (don't nuke src/_something)
        if d == PROJECT:
            for pat in IGNORE_GLOBS:
                if Path(n).match(pat):
                    out.add(n)
                    break
    return out


def copy_code(out: Path) -> None:
    _log(f"  copying code -> {out}")
    shutil.copytree(PROJECT, out, ignore=_ignore, dirs_exist_ok=True)


def trim_data(out: Path) -> None:
    """Drop the large, regenerable artefacts from the copied data/ folder — keep only
    what BASIS needs to start (config + the cached snapshot/signals it reads)."""
    data = out / "data"
    if not data.exists():
        return
    removed = 0
    for p in list(data.glob("*.pdf")) + list(data.glob("_*.png")):
        p.unlink(missing_ok=True); removed += 1
    for d in ("_oitmp",):
        if (data / d).exists():
            shutil.rmtree(data / d, ignore_errors=True); removed += 1
    # transient per-session prefs (the work PC re-creates these on first run)
    for f in ("sector_filter.json", "ui_prefs.json"):
        (data / f).unlink(missing_ok=True)
    _log(f"  trimmed {removed} large/temp item(s) from data/")


def build_wheels(out: Path, pyver: str | None) -> bool:
    wheels = out / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "pip", "download", "-r", str(out / "requirements.txt"),
            "-d", str(wheels), "--extra-index-url", BBG_INDEX]
    if pyver:                                   # cross-version download (binary wheels only)
        base += ["--only-binary=:all:", "--python-version", pyver]
    _log(f"  downloading wheels -> {wheels}  (this is the big step; needs internet here)")
    r = subprocess.run(base)
    if r.returncode != 0:
        _log("  [!] full download (incl. Bloomberg blpapi) failed — retrying CORE packages only.")
        core = [a for a in base if a not in ("--extra-index-url", BBG_INDEX)]
        r2 = subprocess.run(core)               # same, minus the Bloomberg index
        if r2.returncode != 0:
            _log("  [X] wheel download failed. Re-run with internet, or use the proxy-pip path "
                 "in SETUP_WORK_PC.md.")
            return False
        _log("  [!] blpapi/xbbg are NOT in the bundle — install them online on the work PC "
             "(Bloomberg's index is normally whitelisted); see SETUP_WORK_PC.md.")
    n = len(list(wheels.glob('*.whl'))) + len(list(wheels.glob('*.tar.gz')))
    _log(f"  {n} package files in wheels/")
    return True


def copy_browser(out: Path) -> bool:
    src = Path.home() / "AppData" / "Local" / "ms-playwright"
    if not src.exists():
        _log(f"  [!] no Playwright browser cache at {src} — run `playwright install chromium` here first.")
        return False
    dst = out / "playwright-browsers"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for d in src.iterdir():
        if d.is_dir() and d.name.startswith("chromium"):
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True); copied += 1
    _log(f"  copied {copied} Chromium folder(s) -> {dst}")
    return copied > 0


def make_zip(out: Path) -> None:
    zpath = out.with_suffix(".zip")
    _log(f"  zipping -> {zpath} (large; may take a few minutes)")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    _log(f"  zip ready: {zpath}  ({zpath.stat().st_size/1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a deployable BASIS bundle.")
    ap.add_argument("--out", default=str(Path.home() / "BASIS_deploy"))
    ap.add_argument("--pyver", default=None, help="target Python version for wheels, e.g. 3.12")
    ap.add_argument("--no-wheels", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    _log(f"=== Building BASIS deployment bundle ===")
    if out.exists():
        _log(f"  clearing existing {out}")
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    copy_code(out)
    trim_data(out)
    if not args.no_wheels:
        build_wheels(out, args.pyver)
    else:
        _log("  (skipping wheels)")
    if not args.no_browser:
        copy_browser(out)
    else:
        _log("  (skipping browser)")
    if args.zip:
        make_zip(out)

    _log("")
    _log("=== Done ===")
    _log(f"Bundle: {out}")
    _log("Copy that whole folder to the work PC, then follow SETUP_WORK_PC.md inside it.")


if __name__ == "__main__":
    main()
