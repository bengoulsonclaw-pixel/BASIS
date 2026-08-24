"""The Signal Ledger's MEASURING STICK must not depend on how deep the live feed reaches.

Regression for the 2026-08-14 → 08-24 freeze. The ledger scores every historical signal
against a 10-year price frame that is rebuilt from scratch each morning. `sigcache.book_frames`
asked for `today − 10y − 30d` while the deep store's floor is fixed at 2016-08-08, so in
BLOOMBERG mode (which the pull's compute phase runs in, run_pull.py:123) the live 'A' generic
reached back PAST the store. `deepstore.overlay`'s depth heuristic then read that as "the store
adds no depth here" and refused the panama upgrade, leaving those products on their RAW
roll-gapped series for the whole ten years. A raw generic carries the roll gap as a real price
move, so whenever a roll fell inside a signal's 21-session window the gap entered the measured
move and could reverse its sign — re-marking ~6% of settled outcomes every morning. The
append-only guard correctly refused the merge, and the track record sat frozen for 10 days.

The invariant that was missing: the frame is a function of the deep store, not of feed depth.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import deepstore


@pytest.fixture
def store(monkeypatch):
    """A tiny synthetic deep store: 300 business days, two products, floored well after
    the feed a 'bloomberg-like' caller would supply."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"AAA Comdty": range(100, 400), "BBB Comdty": range(200, 500)},
                      index=idx, dtype=float)
    monkeypatch.setattr(deepstore, "_read", lambda name: px.copy())
    monkeypatch.setattr(deepstore, "get_adjusted", lambda t, s=None, e=None: px.copy())
    monkeypatch.setattr(deepstore, "get_ta", lambda t, s=None, e=None: px.copy())
    monkeypatch.setattr(deepstore, "get_volume", lambda t, s=None, e=None: px.copy())
    return px


def _feed(store_px, extra_days: int):
    """The live feed. extra_days > 0 reaches back FURTHER than the store — the bloomberg
    case that tripped the depth heuristic."""
    if extra_days <= 0:
        return store_px.iloc[50:].copy()                      # shallower than the store
    head = pd.bdate_range(end=store_px.index.min() - pd.Timedelta(days=1), periods=extra_days)
    return pd.concat([pd.DataFrame(1.0, index=head, columns=store_px.columns), store_px])


def test_depth_heuristic_drops_the_adjusted_series_when_the_feed_is_deeper(store):
    """The defect itself, pinned: with a deeper feed the default overlay upgrades NOTHING."""
    feed = _feed(store, extra_days=10)
    used, *_ = deepstore.overlay(list(store.columns), None, None,
                                 feed.copy(), feed.copy(), feed.copy())
    assert used == [], "control: this is the behaviour that froze the ledger"


def test_require_deep_makes_the_frame_independent_of_feed_depth(store):
    """The fix: a measurement always takes the panama-adjusted series, however far back
    the feed happens to reach."""
    tickers = list(store.columns)
    seen = {}
    for label, extra in (("shallow", 0), ("deep", 10)):
        feed = _feed(store, extra)
        used, pnl, sig, _ = deepstore.overlay(tickers, None, None,
                                              feed.copy(), feed.copy(), feed.copy(),
                                              require_deep=True)
        seen[label] = (sorted(used), sig)
    assert seen["shallow"][0] == tickers, "every product must take the store"
    assert seen["deep"][0] == tickers, "…including when the feed is deeper than the store"
    # and the resulting signal frames must agree wherever they overlap
    a, b = seen["shallow"][1], seen["deep"][1]
    common = a.index.intersection(b.index)
    assert len(common) > 200
    pd.testing.assert_frame_equal(a.loc[common, tickers], b.loc[common, tickers])


def test_require_deep_still_honours_the_freshness_test(store, monkeypatch):
    """`require_deep` relaxes DEPTH only. A stale store must still lose to the live feed,
    or a lapsed store would silently truncate the right edge."""
    stale = store.iloc[:-60]                                  # store ends ~3 months early
    monkeypatch.setattr(deepstore, "get_adjusted", lambda t, s=None, e=None: stale.copy())
    monkeypatch.setattr(deepstore, "get_ta", lambda t, s=None, e=None: stale.copy())
    feed = _feed(store, extra_days=10)
    used, *_ = deepstore.overlay(list(store.columns), None, None,
                                 feed.copy(), feed.copy(), feed.copy(), require_deep=True)
    assert used == [], "a stale store must not be forced in"


def test_first_date_exposes_the_store_floor(store):
    """Callers clamp their request to this; asking for history the store cannot serve is
    what tripped the heuristic in the first place."""
    assert deepstore.first_date() == store.index.min()


def test_book_frames_never_asks_for_history_before_the_store_floor():
    """The clamp itself — a single assert that would have failed on day one."""
    import inspect
    from src import sigcache
    src = inspect.getsource(sigcache.book_frames)
    assert "first_date()" in src and "max(start" in src, \
        "book_frames must clamp its start to the deep store's floor"
    assert "require_deep=True" in src, \
        "book_frames must require the panama upgrade — it is measuring, not backtesting"
