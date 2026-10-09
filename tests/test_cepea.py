"""CEPEA physical indicators: the Paranaguá soy leg of the Brazil-vs-US basis.

Pins the two things that would make the soy basis confidently wrong: the sack->bushel
conversion, and a truncated download silently replacing ten years of history.
No network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import cepea


def test_sack_to_cents_per_bushel():
    assert cepea.BU_PER_SACK_SOY == pytest.approx(2.204623, abs=1e-5)
    # 30.84 US$/sack on 2026-10-08 = 1398.88 c/bu
    assert cepea.SERIES["soy_paranagua"]["to_us"](30.84) == pytest.approx(1398.88, abs=0.01)


def test_short_download_keeps_the_stored_history(tmp_path, monkeypatch):
    monkeypatch.setattr(cepea, "STORE", tmp_path)
    full = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=100),
                         "brl": 100.0, "usd": 25.0})
    full.to_parquet(tmp_path / "soy_paranagua.parquet", index=False)
    monkeypatch.setattr(cepea, "_download", lambda product, sid, timeout=60: b"x")
    monkeypatch.setattr(cepea, "parse_xls", lambda blob: full.head(10))
    res = cepea.update(["soy_paranagua"])
    assert "kept old" in res["soy_paranagua"]
    assert len(pd.read_parquet(tmp_path / "soy_paranagua.parquet")) == 100
