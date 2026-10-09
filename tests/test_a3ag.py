"""A3 Mercados (Argentina) Rosario futures — the Argentine leg of the South-America basis.

Pins: Spanish month codes, the outright-only filter (options/spreads share the root), and
the US$/t -> c/bu conversion that IS the comparison with CBOT. No network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import a3ag


@pytest.mark.parametrize("sym,expect", [
    ("SOJ.ROS/NOV26", (2026, 11)), ("MAI.ROS/ENE27", (2027, 1)),
    ("SOJ.ROS/AGO25", (2025, 8)), ("MAI.ROS/DIC26", (2026, 12)),
    ("SOJ.ROS/XYZ26", None), ("garbage", None),
])
def test_spanish_contract_months(sym, expect):
    assert a3ag.contract_ym(sym) == expect


def test_usd_per_tonne_to_cents_per_bushel():
    assert a3ag.PRODUCTS["SOJ"]["bu_per_t"] == pytest.approx(36.7437, abs=1e-3)
    assert a3ag.PRODUCTS["MAI"]["bu_per_t"] == pytest.approx(39.3683, abs=1e-3)
    # Rosario soy 358.8 US$/t on 2026-10-08 = 976.5 c/bu
    assert 358.8 / a3ag.PRODUCTS["SOJ"]["bu_per_t"] * 100 == pytest.approx(976.49, abs=0.01)


def test_only_plain_outrights_are_stored(monkeypatch):
    rows = [{"dateTime": "2026-10-08T00:00:00.000Z", "symbol": s, "settlement": 358.8,
             "volume": 10, "openInterest": 100}
            for s in ("SOJ.ROS/NOV26", "SOJ.ROS/NOV26 344 C", "SOJ.ROS/NOV26/ENE27",
                      "SOJ.MIN/NOV26")]
    monkeypatch.setattr(a3ag, "_get", lambda params: {"data": rows, "totalEntries": len(rows)})
    df = a3ag.fetch_year("SOJ", 2026)
    assert df["ticker"].tolist() == ["SOJ.ROS/NOV26"]
