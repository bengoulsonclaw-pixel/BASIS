"""Signal Ledger append-only guarantees (src/sigledger._merge_frozen + the flip guard).

Locks the 2026-08-13 lesson: a settled outcome is a historical fact. A daily rebuild may
only (a) settle outcomes that were pending, (b) append rows for new days, and (c) re-take
strategies explicitly listed in `rescore` — and must REFUSE everything when the fresh
frame disagrees with settled history (the corrupt-frame morning that silently re-marked
10 years of track record, 50.0% -> 44.7% full-book 21d hit rate).

All frames are synthetic and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import sigledger


def _ledger(dates, strategy="Ichimoku", instrument="CLA Comdty", hit21=1.0, **over):
    n = len(dates)
    base = {
        "date": pd.to_datetime(dates), "strategy": strategy, "market": "WTI",
        "instruments": instrument, "signal": "Long", "direction": 1, "metric": 1.0,
        "level": 70.0, "entry_level": 70.0, "sigma": 1.0,
    }
    for h in sigledger.HORIZONS:
        base[f"move{h}"] = 1.0
        base[f"sig{h}"] = 0.5
        base[f"hit{h}"] = hit21
    base.update(over)
    return pd.DataFrame({k: (v if hasattr(v, "__len__") and not isinstance(v, str)
                             else [v] * n) for k, v in base.items()})


def _days(start, n):
    return pd.bdate_range(start, periods=n)


def test_settled_outcomes_are_frozen_verbatim():
    days = _days("2024-01-01", 600)
    prior = _ledger(days, hit21=1.0)
    fresh = _ledger(days, hit21=1.0)
    # fresh frame re-marks ONE old row's move/sig (below the refuse threshold) — the
    # prior row must still win verbatim
    fresh.loc[0, [f"move{h}" for h in sigledger.HORIZONS]] = -9.0
    fresh.loc[0, [f"sig{h}" for h in sigledger.HORIZONS]] = -9.0
    merged = sigledger._merge_frozen(prior, fresh, log=lambda *a: None)
    row = merged.sort_values("date").iloc[0]
    assert row["move21"] == 1.0 and row["sig21"] == 0.5 and row["hit21"] == 1.0


def test_pending_outcomes_settle_and_new_rows_append():
    days = _days("2024-01-01", 600)
    prior = _ledger(days)
    # last prior row was pending at 21 sessions
    pend = [f"{p}21" for p in ("move", "sig", "hit")]
    prior.loc[prior.index[-1], pend] = np.nan
    extra = _days(days[-1] + pd.Timedelta(days=1), 5)
    fresh = _ledger(days.append(extra), hit21=1.0)
    merged = sigledger._merge_frozen(prior, fresh, log=lambda *a: None)
    merged = merged.sort_values("date").reset_index(drop=True)
    assert len(merged) == 605                            # 600 frozen + 5 appended
    assert merged.loc[599, "hit21"] == 1.0               # pending -> settled from fresh
    assert merged.loc[599, "move21"] == 1.0


def test_corrupt_frame_is_refused_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(sigledger, "_guard_path",
                        lambda scope="ficc": tmp_path / f"guard_{scope}.json")
    days = _days("2024-01-01", 600)
    prior = _ledger(days, hit21=1.0)
    fresh = _ledger(days, hit21=1.0)
    flip = fresh.index[:60]                              # 10% of settled hits re-marked
    fresh.loc[flip, [f"hit{h}" for h in sigledger.HORIZONS]] = 0.0
    out = sigledger._merge_frozen(prior, fresh, log=lambda *a: None)
    assert out is None
    assert sigledger.guard_refusal() is not None         # marker written for the page
    # a clean follow-up merge clears the marker
    ok = sigledger._merge_frozen(prior, _ledger(days, hit21=1.0), log=lambda *a: None)
    assert ok is not None and sigledger.guard_refusal() is None


def test_rescore_retakes_only_that_strategy():
    days = _days("2024-01-01", 600)
    prior = pd.concat([_ledger(days, strategy="Ichimoku", hit21=1.0),
                       _ledger(days, strategy=sigledger.CONFLUENCE, hit21=1.0)],
                      ignore_index=True)
    fresh = pd.concat([_ledger(days, strategy="Ichimoku", hit21=1.0),
                       _ledger(days, strategy=sigledger.CONFLUENCE, hit21=0.0)],
                      ignore_index=True)
    merged = sigledger._merge_frozen(prior, fresh, rescore=(sigledger.CONFLUENCE,),
                                     log=lambda *a: None)
    conf = merged[merged["strategy"] == sigledger.CONFLUENCE]
    core = merged[merged["strategy"] == "Ichimoku"]
    assert (conf["hit21"] == 0.0).all()                  # composite re-derived whole
    assert (core["hit21"] == 1.0).all()                  # everything else frozen


def test_thin_overlap_merges_without_judging():
    days = _days("2026-06-01", 30)                       # << GUARD_MIN_OVERLAP settled rows
    prior = _ledger(days, hit21=1.0)
    fresh = _ledger(days, hit21=0.0)                     # total disagreement, but too thin
    merged = sigledger._merge_frozen(prior, fresh, log=lambda *a: None)
    assert merged is not None
    assert (merged["hit21"] == 1.0).all()                # prior still wins verbatim
