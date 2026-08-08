"""Golden locks on the cross-strategy scoring pipeline (src/tascore.py) — the maths
every TA surface shares: the hub, the Technical Analysis report, the backtester and
the signal ledger all score through these exact functions."""
from __future__ import annotations

import pandas as pd
import pytest

from src import tascore


def test_strength_scaling():
    # z-score strategies top out at 3, Trend at 25%, MA gap at 10%, 0-100 metrics identity
    assert tascore.strength("Mean Reversion", 1.5) == 50.0
    assert tascore.strength("Trend", -12.5) == 50.0          # magnitude, sign-free
    assert tascore.strength("Trend", 99.0) == 100.0          # clipped at full
    assert tascore.strength("MA Crossover", 5.0) == 50.0
    assert tascore.strength("Flag Breakout", 43.0) == 43.0   # identity scale
    assert tascore.strength("Trend", float("nan")) == 0.0
    assert tascore.strength("Trend", "junk") == 0.0
    assert tascore.strength("No Such Strategy", 50.0) == 50.0  # DEFAULT_SCALE=100


def test_axes_partition_and_dup_detection():
    # every strategy sits in exactly one axis (the module asserts the partition itself)
    for s in tascore.TA_STRATEGIES:
        assert tascore.axis_of(s) in tascore.TA_AXES
    # unknown methods de-duplicate alone rather than folding into a real axis
    assert tascore.axis_of("Mystery Method") == "Mystery Method"
    assert tascore.has_intra_axis_dup(["Trend", "MA Crossover"])          # both Trend axis
    assert not tascore.has_intra_axis_dup(tascore.CONFLUENCE_DEFAULT)     # one per axis by design


def _flagged() -> pd.DataFrame:
    """A hand-built flagged frame: CLA with three signals across two axes (two of them
    sharing the Trend axis, one conflicting short), GCA with a lone stronger short."""
    return pd.DataFrame([
        # CLA: Trend +1 @ 3m ret 15% (strength 60) | MA Crossover +1 @ gap 4% (strength 40,
        # same Trend axis -> harmonic 1/2) | OBV -1 @ score 70 (strength 70, own axis)
        {"strategy": "Trend", "market": "WTI Crude", "instruments": "CLA Comdty",
         "signal": "Long", "direction": 1, "metric": 15.0},
        {"strategy": "MA Crossover", "market": "WTI Crude", "instruments": "CLA Comdty",
         "signal": "Long", "direction": 1, "metric": 4.0},
        {"strategy": "On-Balance Volume", "market": "WTI Crude", "instruments": "CLA Comdty",
         "signal": "Short", "direction": -1, "metric": 70.0},
        # GCA: one clean short, strength 80
        {"strategy": "Trend", "market": "Gold", "instruments": "GCA Comdty",
         "signal": "Short", "direction": -1, "metric": -20.0},
    ])


def test_score_products_harmonic_dedup_and_netting(golden):
    out = tascore.score_products(_flagged())
    # ranked by |score|: GCA's clean 80 beats CLA's netted 10
    assert list(out["instruments"]) == ["GCA Comdty", "CLA Comdty"]
    cla = out[out["instruments"] == "CLA Comdty"].iloc[0]
    # 60·1 (strongest of the Trend axis) + 40·1/2 (second in-axis) − 70·1 = +10
    assert cla["score"] == pytest.approx(10.0)
    assert cla["net_dir"] == 1 and bool(cla["conflict"]) and cla["n"] == 3
    assert cla["conviction"] == pytest.approx(56.7)          # mean(60, 40, 70) rounded
    gca = out[out["instruments"] == "GCA Comdty"].iloc[0]
    assert gca["score"] == pytest.approx(-80.0) and gca["net_dir"] == -1
    golden("tascore_score_products", out)


def test_score_products_empty():
    out = tascore.score_products(pd.DataFrame())
    assert out.empty and "score" in out.columns


def test_ta_flagged_hub_trigger_and_fi_action(golden):
    """Re-flagging at the overview triggers: Trend's hub bar is ±10 (its page default is 0),
    and fixed-income rows get the futures action appended ('Long · sell the bond')."""
    df = pd.DataFrame([
        {"strategy": "Trend", "market": "WTI Crude", "instruments": "CLA Comdty",
         "signal": "—", "direction": 0, "metric": 12.0},     # above bar -> Long
        {"strategy": "Trend", "market": "US 10yr", "instruments": "TYA Comdty",
         "signal": "—", "direction": 0, "metric": 12.0},     # FI: Long yields = sell the bond
        {"strategy": "Trend", "market": "Gold", "instruments": "GCA Comdty",
         "signal": "—", "direction": 0, "metric": -15.0},    # below -bar -> Short
        {"strategy": "Trend", "market": "Copper", "instruments": "HGA Comdty",
         "signal": "Long", "direction": 1, "metric": 8.0},   # inside the bar -> dropped
    ])
    out = tascore.ta_flagged(df, strategies=["Trend"])
    assert set(out["instruments"]) == {"CLA Comdty", "TYA Comdty", "GCA Comdty"}
    tya = out[out["instruments"] == "TYA Comdty"].iloc[0]
    assert "sell the bond" in tya["signal"]                  # the FI yield->futures wording
    assert tya["direction"] == 1                             # direction stays in YIELD space here
    golden("tascore_ta_flagged", out)
