"""Locks over the correlation alert gate (sectorcorr.percentile_extremes):
Ben's 2026-09-10 rule — a break only matters when a REAL relationship changed.
0.99→0.60 flags; 0.30→0.02 does not; 0.45→0.06 (the SAME raw drop as 0.99→0.60)
does not either — the Fisher-z floor measures relationship lost, not points
moved. Synthetic frame via a monkeypatched top_breaks — no stores."""
from __future__ import annotations

import pandas as pd
import pytest

from src import sectorcorr


def _frame():
    rows = [
        ("A1", "B1", 0.99, 0.60),   # tight pair collapsing — THE story
        ("A2", "B2", 0.30, 0.02),   # strangers drifting — noise
        ("A3", "B3", 0.45, 0.06),   # marginal base, same raw drop as row 1 — still noise
        ("A4", "B4", 0.90, 0.60),   # smaller raw drop on a tight pair — real
        ("A5", "B5", 0.10, -0.78),  # the NG × Ethanol shape — killed by the base floor
    ]
    return pd.DataFrame([
        {"a": a, "b": b, "sector_a": "x", "sector_b": "y",
         "corr_1y": y, "corr_1m": m, "diff": m - y, "pctl": 0.0}
        for a, b, y, m in rows])


def test_break_gate_measures_relationship_lost(monkeypatch):
    monkeypatch.setattr(sectorcorr, "top_breaks", lambda *a, **k: _frame())
    hit = sectorcorr.percentile_extremes("realized", None, base_floor=0.40)
    assert list(hit["a"]) == ["A1", "A4"]
    assert (hit["kind"] == "breakdown").all()
    assert float(hit["dz"].iloc[0]) > 1.5       # 0.99→0.60 is a z-scale collapse


def test_fisher_dz_matches_the_examples():
    assert float(sectorcorr.fisher_dz(0.99, 0.60)) > 1.5
    assert float(sectorcorr.fisher_dz(0.30, 0.02)) < sectorcorr.BREAK_DZ_MIN
    assert float(sectorcorr.fisher_dz(0.45, 0.06)) < sectorcorr.BREAK_DZ_MIN
    assert float(sectorcorr.fisher_dz(0.90, 0.60)) >= sectorcorr.BREAK_DZ_MIN
    # the clip: a rounding tremor between two near-1 readings is not a break
    assert float(sectorcorr.fisher_dz(0.995, 0.97)) < sectorcorr.BREAK_DZ_MIN
