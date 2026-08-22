"""Locks over the Hot Sheet engine (src/hotsheet.py): provider fault isolation and
the per-provider cap, the history stamp's freeze semantics (past days immutable,
same-day replace, all-failed empty stamps refused — the signal-ledger lesson), badge
derivation, and the week_view aggregation the Weekly Review's front page will read.
Synthetic providers + tmp stores only — never the repo's data/ or real providers."""
from __future__ import annotations

import sys
import types
from datetime import date

import pandas as pd
import pytest

from src import hotsheet


def _it(key, heat=50.0, tag="TST", value=None, book="ficc", **kw):
    return hotsheet.item(tag=tag, key=key, section=kw.pop("section", "Testing"),
                         text=kw.pop("text", f"**{key}** screens notable."),
                         heat=heat, value=value, book=book, **kw)


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(hotsheet, "HISTORY_FILE", tmp_path / "hist.parquet")
    monkeypatch.setattr(hotsheet, "META_FILE", tmp_path / "meta.json")
    monkeypatch.setattr(hotsheet, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(hotsheet, "TOP10_FILE", tmp_path / "top10.json")   # 2026-08-22: the
    monkeypatch.setattr(hotsheet, "SIG_DIR", tmp_path)    # unpatched export leaked TST rows live
    return tmp_path


def _fake_provider(monkeypatch, mapping):
    """Register synthetic src.<name> modules and pin discovery to exactly them.
    import_module returns a sys.modules hit, so no import machinery is patched."""
    for name, fn in mapping.items():
        mod = types.ModuleType(f"src.{name}")
        mod.radar_items = fn
        monkeypatch.setitem(sys.modules, f"src.{name}", mod)
    monkeypatch.setattr(hotsheet, "discover", lambda: list(mapping))


# --- item factory + heat helpers -------------------------------------------
def test_item_contract():
    it = _it("wti:cheap", heat=250.0)              # heat clamps into 0-100
    assert it["key"] == "TST:wti:cheap" and it["heat"] == 100.0
    assert it["weekly"] is True and it["internal_only"] is False
    with pytest.raises(ValueError):
        hotsheet.item(tag="TST", key="x", section="S", text="t", heat=1, book="bonds")
    with pytest.raises(ValueError):
        hotsheet.item(tag="", key="x", section="S", text="t", heat=1)


def test_spark_validation():
    it = _it("s", spark=[1, 2, float("nan"), 3, None, 4, 5, 6, 7, 8])
    assert it["spark"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]   # NaN/None dropped
    assert _it("s2", spark=[1, 2, 3])["spark"] is None               # too short = no shape
    sp = _it("s3", spark=list(range(500)))["spark"]
    assert len(sp) == hotsheet.SPARK_MAX
    assert sp[0] == 0.0 and sp[-1] == 499.0                          # stride keeps both ends
    assert _it("s4")["spark"] is None


def test_stamp_drops_spark(tmp_store):
    rep = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
    hotsheet.stamp([_it("a", spark=list(range(20)))], rep,
                   asof=date(2026, 8, 17), log=lambda *a: None)
    h = hotsheet.load_history()
    assert "spark" not in h.columns and len(h) == 1  # render-only, never persisted


def test_heat_helpers():
    assert hotsheet.heat_from_z(2.0) == 50.0
    assert hotsheet.heat_from_z(-9.0) == 100.0
    assert hotsheet.heat_from_pctl(50) == 0.0
    assert hotsheet.heat_from_pctl(97) == pytest.approx(94.0)
    assert hotsheet.heat_from_pctl(0) == 100.0


# --- collect: fault isolation + the per-provider cap ------------------------
def test_collect_isolates_failures_and_caps(monkeypatch):
    def ok():
        return [_it(f"k{i}", heat=10.0 * i) for i in range(1, 8)]   # 7 items -> cap 5

    def broken():
        raise RuntimeError("store missing")

    _fake_provider(monkeypatch, {"okmod": ok, "badmod": broken, "quietmod": list})
    items, report = hotsheet.collect()
    assert report["okmod"]["status"] == "ok" and report["okmod"]["over_cap"] == 2
    assert report["badmod"]["status"] == "failed" and "store missing" in report["badmod"]["err"]
    assert report["quietmod"]["status"] == "quiet"
    assert len(items) == hotsheet.PROVIDER_CAP
    assert [i["heat"] for i in items] == sorted([i["heat"] for i in items], reverse=True)
    assert all(i["provider"] == "okmod" for i in items)


def test_collect_book_filter(monkeypatch):
    _fake_provider(monkeypatch, {"mix": lambda: [_it("a", book="ficc"),
                                                 _it("b", book="equities")]})
    items, _ = hotsheet.collect(book="equities")
    assert [i["key"] for i in items] == ["TST:b"]


# --- history stamp: past frozen, today replaced, bogus empties refused ------
def test_stamp_freezes_past_and_replaces_today(tmp_store):
    ok_rep = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
    assert hotsheet.stamp([_it("day1")], ok_rep, asof=date(2026, 8, 17), log=lambda *a: None) == 1
    assert hotsheet.stamp([_it("day2-v1")], ok_rep, asof=date(2026, 8, 18), log=lambda *a: None) == 1
    hotsheet.stamp([_it("day2-v2")], ok_rep, asof=date(2026, 8, 18), log=lambda *a: None)
    h = hotsheet.load_history()
    assert set(h[h["date"] == "2026-08-17"]["key"]) == {"TST:day1"}      # past untouched
    assert set(h[h["date"] == "2026-08-18"]["key"]) == {"TST:day2-v2"}   # today replaced


def test_stamp_refuses_all_failed_empty(tmp_store):
    ok_rep = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
    hotsheet.stamp([_it("real")], ok_rep, asof=date(2026, 8, 17), log=lambda *a: None)
    bad_rep = {"p": {"status": "failed", "n": 0, "ms": 1, "err": "boom", "over_cap": 0}}
    assert hotsheet.stamp([], bad_rep, asof=date(2026, 8, 18), log=lambda *a: None) == 0
    assert len(hotsheet.load_history()) == 1     # the wedged morning wrote nothing
    # a genuinely quiet day (providers fine, zero items) IS a valid stamp
    quiet_rep = {"p": {"status": "quiet", "n": 0, "ms": 1, "err": "", "over_cap": 0}}
    hotsheet.stamp([], quiet_rep, asof=date(2026, 8, 18), log=lambda *a: None)
    assert len(hotsheet.load_history()) == 1


# --- badges: NEW / nth-day streaks off the stamped day axis -----------------
def test_apply_badges(tmp_store):
    rep = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
    hotsheet.stamp([_it("a")], rep, asof=date(2026, 8, 14), log=lambda *a: None)
    hotsheet.stamp([_it("a"), _it("b")], rep, asof=date(2026, 8, 17), log=lambda *a: None)
    today = [_it("a"), _it("b"), _it("c")]
    hotsheet.apply_badges(today, asof=date(2026, 8, 18))
    badges = {i["key"]: i["badge"] for i in today}
    assert badges["TST:a"] == "day 3"            # weekends don't break a stamped-day streak
    assert badges["TST:b"] == ""                 # 2nd day — persistence isn't a story yet
    assert badges["TST:c"] == "NEW"


def test_apply_badges_day_one_is_quiet(tmp_store):
    today = [_it("a")]
    hotsheet.apply_badges(today, asof=date(2026, 8, 18))
    assert today[0]["badge"] == ""               # no history — everything NEW would be noise


# --- the persisted sheet: page opens read a file, never run providers -------
_OK_REP = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
_BAD_REP = {"p": {"status": "failed", "n": 0, "ms": 1, "err": "boom", "over_cap": 0}}


def test_cached_collection_serves_fresh_file(tmp_store, monkeypatch):
    hotsheet._persist_collection([_it("a")], _OK_REP)
    monkeypatch.setattr(hotsheet, "collect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran providers")))
    items, report, _, from_cache = hotsheet.cached_collection()
    assert from_cache and [i["key"] for i in items] == ["TST:a"] and "p" in report


def test_cached_collection_stale_recollects(tmp_store, monkeypatch):
    import json, time
    hotsheet._persist_collection([_it("old")], _OK_REP)
    c = json.loads(hotsheet.CACHE_FILE.read_text(encoding="utf-8"))
    c["collected"] = time.time() - (hotsheet.CACHE_STALE_H + 1) * 3600
    hotsheet.CACHE_FILE.write_text(json.dumps(c), encoding="utf-8")
    monkeypatch.setattr(hotsheet, "collect", lambda *a, **k: ([_it("new")], dict(_OK_REP)))
    items, _, _, from_cache = hotsheet.cached_collection()
    assert not from_cache and [i["key"] for i in items] == ["TST:new"]
    refreshed = json.loads(hotsheet.CACHE_FILE.read_text(encoding="utf-8"))
    assert refreshed["items"][0]["key"] == "TST:new"        # the file was re-filled


def test_refresh_all_failed_keeps_cache(tmp_store, monkeypatch):
    import json
    hotsheet._persist_collection([_it("good")], _OK_REP)
    monkeypatch.setattr(hotsheet, "collect", lambda *a, **k: ([], dict(_BAD_REP)))
    items, _, _, from_cache = hotsheet.refresh_collection()
    assert from_cache and [i["key"] for i in items] == ["TST:good"]
    kept = json.loads(hotsheet.CACHE_FILE.read_text(encoding="utf-8"))
    assert kept["items"][0]["key"] == "TST:good"            # never blanked by a wedged run


def test_stamp_today_persists_sheet(tmp_store, monkeypatch):
    monkeypatch.setattr(hotsheet, "collect", lambda *a, **k: ([_it("a")], dict(_OK_REP)))
    hotsheet.stamp_today(log=lambda *a: None)
    assert hotsheet.CACHE_FILE.exists()
    monkeypatch.setattr(hotsheet, "collect", lambda *a, **k: ([], dict(_BAD_REP)))
    hotsheet.stamp_today(log=lambda *a: None)               # wedged morning: file untouched
    import json
    assert json.loads(hotsheet.CACHE_FILE.read_text(encoding="utf-8"))["items"][0]["key"] == "TST:a"


# --- week_view: what the Weekly Review's front page reads -------------------
def test_week_view_aggregation(tmp_store):
    rep = {"p": {"status": "ok", "n": 1, "ms": 1, "err": "", "over_cap": 0}}
    hotsheet.stamp([_it("x", heat=40, value=1.0), _it("eq", book="equities")],
                   rep, asof=date(2026, 8, 12), log=lambda *a: None)
    hotsheet.stamp([_it("x", heat=90, value=2.0), _it("y", heat=99, value=5.0),
                    _it("desk", internal_only=True), _it("daily", weekly=False)],
                   rep, asof=date(2026, 8, 13), log=lambda *a: None)
    hotsheet.stamp([_it("x", heat=60, value=3.0)],
                   rep, asof=date(2026, 8, 14), log=lambda *a: None)
    wk = hotsheet.week_view(days=7, book="ficc", asof=date(2026, 8, 15))
    assert list(wk["key"]) == ["TST:x", "TST:y"]  # persistence first; internal/daily/eq cut
    x = wk.iloc[0]
    assert x["days_on"] == 3 and x["heat"] == 90.0          # the most extreme reading kept
    assert x["wk_delta"] == pytest.approx(2.0)              # generic value drift over the window
    assert wk.iloc[1]["days_on"] == 1
