"""Structural checks on the data-health engine (src/health.py) — it must always
return well-formed results, both against whatever real stores this box has
(read-only) and against an entirely EMPTY box (fresh clone / VPS), where every
reader has to degrade gracefully instead of raising."""
from __future__ import annotations

import pandas as pd

from src import health

_FRAME_COLS = ["frame", "rows", "markets", "last_date", "written", "age_h", "size_mb"]


def test_snapshot_frames_shape():
    f = health.snapshot_frames()
    assert list(f.columns) == _FRAME_COLS


def test_deep_health_keys():
    dh = health.deep_health()
    for k in ("coverage", "truncated", "missing", "laggards", "bonds_no_yield",
              "store_last", "lag_vs_snap_days", "min_days", "fresh_tol_days"):
        assert k in dh
    assert isinstance(dh["coverage"], pd.DataFrame)


def test_stale_surfaces_shape():
    s = health.stale_surfaces()
    assert list(s.columns) == ["ticker", "market", "surface", "reason"]


def test_cache_health_shape():
    c = health.cache_health()
    assert list(c.columns) == ["store", "last_date", "items", "age_h"]
    assert len(c) >= 5


def test_checks_wellformed():
    board = health.checks()
    assert board, "the status board must never come back empty"
    for c in board:
        assert c["level"] in ("ok", "warn", "bad")
        assert c["area"] and isinstance(c["message"], str) and c["message"]


def test_empty_box_degrades_gracefully(monkeypatch, tmp_path):
    """A machine with NO stores at all (fresh clone) still gets a valid board —
    'no snapshot' is a finding, not a crash."""
    from src import deepstore
    monkeypatch.setattr(health, "SNAP", tmp_path / "snapshot")
    monkeypatch.setattr(health, "STORE", tmp_path / "price_store")
    monkeypatch.setattr(health, "TEST_LOG", tmp_path / "none.json")
    monkeypatch.setattr(deepstore, "STORE_DIR", tmp_path / "price_store")
    assert health.load_manifest() == {}
    assert health.snapshot_frames().empty
    assert health.missing_core_frames() == list(health.CORE_FRAMES)
    dh = health.deep_health()
    assert dh["coverage"].empty and dh["missing"]         # whole universe missing
    board = health.checks(stale=health.stale_surfaces())  # stale check still live-computes
    levels = {c["level"] for c in board}
    assert "bad" in levels                                # "no snapshot manifest" fires
    assert health.last_test_run() == {}


def test_parse_stamp_conventions():
    """Offset-tagged stamps honored; legacy naive stamps read as UTC (they were file
    mtimes) — the +5h 'prices as of' bug must stay dead."""
    a = health.parse_stamp("2026-08-07T23:07:19+00:00")
    b = health.parse_stamp("2026-08-07 23:07:19")
    assert a == b
    assert str(a.tz) == "UTC"
    assert health.parse_stamp("") is None and health.parse_stamp(None) is None
