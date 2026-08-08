"""Shared test rig for the BASIS regression suite.

WHAT THIS SUITE IS. A small set of golden-file locks over the engines where a silent
regression costs real money or credibility: the cross-strategy scoring pipeline
(tascore), the TA backtester's state machine + FI sign convention (tabt), the option
maths and point-value table (volbt), and the deep store's panama roll-adjustment
(deepstore). It runs in seconds, touches no network and no Bloomberg, and is executed
by run_tests.py — manually, or automatically by the pre-push git hook whenever a push
contains code changes.

GOLDENS. Expected outputs live in tests/goldens/*.json and are committed. A test
compares live output against its golden with float tolerance. When a behaviour change
is DELIBERATE, re-baseline with `run_tests.bat --regen` (sets BASIS_REGEN_GOLDENS=1,
rewrites the files) and commit the diff — the review of that diff IS the review of the
behaviour change.

Every test must be deterministic: fixed dates, seeded RNGs, synthetic frames built in
the test — never "today", never live pulls, never repo data stores (test_health's
read-only structural checks are the one deliberate exception).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Force the offline data mode BEFORE any src import (src.datafeed reads the env at
# import time). Tests must never touch Bloomberg, whatever shell they run from.
os.environ["DATAFEED_MODE"] = "mock"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

GOLDENS = Path(__file__).parent / "goldens"
REGEN = os.environ.get("BASIS_REGEN_GOLDENS") == "1"

REL_TOL = 1e-9
ABS_TOL = 1e-12


def _jsonable(obj):
    """Recursively convert frames/series/numpy/dates into plain JSON-able values."""
    import numpy as np
    if isinstance(obj, pd.DataFrame):
        return {"columns": [str(c) for c in obj.columns],
                "records": [[_jsonable(v) for v in row] for row in obj.itertuples(index=False)]}
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)) or hasattr(obj, "isoformat"):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _assert_close(path: str, expected, got):
    if isinstance(expected, float) or isinstance(got, float):
        e, g = float(expected), float(got)
        if math.isnan(e) and math.isnan(g):
            return
        assert math.isclose(e, g, rel_tol=REL_TOL, abs_tol=ABS_TOL), \
            f"{path}: {g!r} != golden {e!r}"
        return
    if isinstance(expected, dict):
        assert isinstance(got, dict), f"{path}: got {type(got).__name__}, golden is a dict"
        assert set(expected) == set(got), \
            f"{path}: keys differ — extra {sorted(set(got) - set(expected))}, " \
            f"missing {sorted(set(expected) - set(got))}"
        for k in expected:
            _assert_close(f"{path}.{k}", expected[k], got[k])
        return
    if isinstance(expected, list):
        assert isinstance(got, list), f"{path}: got {type(got).__name__}, golden is a list"
        assert len(expected) == len(got), f"{path}: length {len(got)} != golden {len(expected)}"
        for i, (e, g) in enumerate(zip(expected, got)):
            _assert_close(f"{path}[{i}]", e, g)
        return
    assert expected == got, f"{path}: {got!r} != golden {expected!r}"


@pytest.fixture
def golden():
    """golden(name, obj) — compare obj against tests/goldens/<name>.json (float-tolerant),
    or rewrite the file when BASIS_REGEN_GOLDENS=1 (run_tests.bat --regen)."""
    def _check(name: str, obj):
        p = GOLDENS / f"{name}.json"
        data = _jsonable(obj)
        if REGEN:
            GOLDENS.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True),
                         encoding="utf-8")
            return
        assert p.exists(), (f"golden '{name}' missing — run  run_tests.bat --regen  once, "
                            f"eyeball the new tests/goldens/{name}.json, and commit it")
        _assert_close(name, json.loads(p.read_text(encoding="utf-8")), data)
    return _check
