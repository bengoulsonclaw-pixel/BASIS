"""Lock over the mock-clobber guard in run_daily.run() (2026-09-11).

datafeed.MODE defaults to 'mock' when DATAFEED_MODE is unset, so a stray env-less
run_daily.run() (not the app or the pull — both set snapshot) overwrites the real signal
caches (opportunities.parquet + EVERY strategy cache: putcall, skew, volatility, …) with
synthetic demo data on top of a real Bloomberg pull. That happened: the Put/Call page showed
a mock 13,599 for Euro-Bund while the snapshot store held the real 208,975 (the Hot Sheet reads
the store directly, so it stayed correct). The guard refuses a MOCK-mode rebuild when a real
bloomberg snapshot is on disk and returns the existing signals unchanged. Never weaken this.
"""
from __future__ import annotations

import json

import pandas as pd

import run_daily


def _boom():
    raise AssertionError("guard did not fire — run() proceeded to the heavy rebuild")


def _wire(tmp_path, monkeypatch, source, existing=True):
    snap = tmp_path / "snapshot"; snap.mkdir()
    sig = tmp_path / "signals"; sig.mkdir()
    if source is not None:
        (snap / ".fetch_meta.json").write_text(json.dumps({"source": source}), encoding="utf-8")
    monkeypatch.setattr(run_daily, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(run_daily, "SIGNALS_DIR", sig)
    monkeypatch.setattr(run_daily, "SIGNALS_FILE", sig / "opportunities.parquet")
    if existing:
        pd.DataFrame({"strategy": ["SENTINEL"], "market": ["keep me"]}).to_parquet(
            sig / "opportunities.parquet", index=False)
    return snap, sig


def test_mock_refuses_over_real_snapshot_and_keeps_signals(tmp_path, monkeypatch):
    """MOCK mode + a real bloomberg snapshot ⇒ refuse and return the existing signals verbatim.
    universe.reload() is booby-trapped, so if the guard fails to short-circuit the test explodes
    instead of silently running (and clobbering with) a mock rebuild."""
    _wire(tmp_path, monkeypatch, "bloomberg")
    monkeypatch.setattr(run_daily, "MODE", "mock")
    monkeypatch.setattr(run_daily.universe, "reload", _boom)
    out = run_daily.run()
    assert list(out["strategy"]) == ["SENTINEL"]        # existing signals returned, NOT rebuilt


def test_snapshot_source_reads_fetch_meta_then_manifest(tmp_path, monkeypatch):
    snap, _ = _wire(tmp_path, monkeypatch, "bloomberg", existing=False)
    assert run_daily._snapshot_source() == "bloomberg"
    (snap / ".fetch_meta.json").unlink()                # manifest.json is the fallback
    (snap / "manifest.json").write_text(json.dumps({"source": "bloomberg"}), encoding="utf-8")
    assert run_daily._snapshot_source() == "bloomberg"


def test_guard_is_narrow_mock_or_missing_snapshot_not_flagged(tmp_path, monkeypatch):
    """Only a real ('bloomberg') snapshot arms the guard — a genuine offline/demo run fetches a
    mock snapshot first, and a fresh checkout has none, so neither is blocked."""
    _wire(tmp_path, monkeypatch, "mock", existing=False)
    assert run_daily._snapshot_source() == "mock"       # != 'bloomberg' ⇒ guard condition False
    (tmp_path / "snapshot" / ".fetch_meta.json").unlink()
    assert run_daily._snapshot_source() == ""           # no snapshot ⇒ guard condition False
