"""Correctness checks on the Brazil Production engine (src/brazilprod.py).

The risky parts of this page are not the arithmetic, they are the CLAIMS:

  * a company table denominated in "% of exports" must never be reconciled
    against, or presented as, national production;
  * shares must chain honestly — a company's share of the world is its share of
    Brazil times Brazil's share of the world, no more;
  * the curated country tables must not silently redefine the world total when a
    hand edit makes the listed producers exceed it.

These run offline against the curated file and a synthetic store, so they do not
need the network or a built store. The one live-ish test skips when the daily
pull has never run on this box.
"""
from __future__ import annotations

import json

import pytest

from src import brazilprod


# ── the curated file is the hand-maintained input: keep it well-formed ───────
def test_curated_file_loads():
    cur = brazilprod.load_curated()
    assert cur.get("companies"), "curated company tables must exist"
    assert cur.get("countries_curated"), "curated country tables must exist"


def test_every_company_block_declares_basis_and_confidence():
    """An undeclared basis is the one failure that would let an export share be
    read as production — the whole point of the page's honesty rules."""
    cur = brazilprod.load_curated()
    for key, blk in cur["companies"].items():
        assert blk.get("basis") in brazilprod.BASIS_LABEL, f"{key}: unknown/missing basis"
        assert blk.get("confidence") in brazilprod.CONFIDENCE_LABEL, f"{key}: bad confidence"
        assert blk.get("source"), f"{key}: every table must name its source"
        assert blk.get("rows"), f"{key}: no rows"
        for r in blk["rows"]:
            assert r.get("company"), f"{key}: a row with no company name"
            assert isinstance(r.get("volume"), (int, float)), f"{key}: {r.get('company')} volume"


def test_row_crops_are_not_claimed_as_production():
    """Soybeans, corn and coffee are grown by tens of thousands of farms. If a
    future edit ever relabels those blocks 'production', this test is the
    tripwire — the page and PDF would start asserting something untrue."""
    cur = brazilprod.load_curated()
    for key in ("soybeans", "corn", "coffee"):
        assert cur["companies"][key]["basis"] == "export", (
            f"{key} must stay an EXPORT basis — no company grows a material share of it")


def test_curated_countries_include_brazil_and_a_world_total():
    cur = brazilprod.load_curated()
    for key, blk in cur["countries_curated"].items():
        assert blk.get("world"), f"{key}: needs a published world total"
        assert "Brazil" in (blk.get("countries") or {}), f"{key}: Brazil missing"
        assert blk.get("source"), f"{key}: needs a source"


def test_curated_world_total_is_not_exceeded_by_its_own_country_list():
    """A hand edit that pushes the listed majors past the stated world total must
    surface as a warning, never be papered over by redefining 'world'."""
    cur = brazilprod.load_curated()
    for key, blk in cur["countries_curated"].items():
        listed = sum(v for v in blk["countries"].values() if v)
        assert listed <= blk["world"] * 1.001, (
            f"{key}: listed producers ({listed:,.1f}) exceed the stated world total "
            f"({blk['world']:,.1f}) — refresh the world figure or a country figure")


# ── share maths ──────────────────────────────────────────────────────────────
def _spec(raw_unit="Mt"):
    return {"key": "iron_ore", "raw_unit": raw_unit}


def test_company_shares_sum_to_one_hundred():
    blk = brazilprod._company_block("iron_ore", _spec(), 440.0, 17.6, brazilprod.load_curated())
    assert blk is not None
    assert sum(r["share_brazil"] for r in blk["rows"]) == pytest.approx(100.0, abs=0.1)


def test_world_share_chains_through_brazil():
    """A company's share of WORLD supply is its share of Brazil scaled by Brazil's
    share of the world — the number the desk actually cares about."""
    br_share = 17.6
    blk = brazilprod._company_block("iron_ore", _spec(), 440.0, br_share,
                                    brazilprod.load_curated())
    for r in blk["rows"]:
        assert r["share_world"] == pytest.approx(r["share_brazil"] / 100 * br_share, abs=0.01)
    assert sum(r["share_world"] for r in blk["rows"]) == pytest.approx(br_share, abs=0.05)


def test_other_bucket_sorts_last():
    blk = brazilprod._company_block("iron_ore", _spec(), 440.0, 17.6, brazilprod.load_curated())
    assert blk["rows"][-1]["is_other"], "the 'Other' bucket must sort last"
    assert not any(r["is_other"] for r in blk["rows"][:-1])


def test_coverage_only_computed_when_units_match():
    """Iron ore is Mt against Mt, so the table can be reconciled. The same block
    measured against a kb/d national figure must refuse to reconcile rather than
    print a meaningless percentage."""
    cur = brazilprod.load_curated()
    same = brazilprod._company_block("iron_ore", _spec("Mt"), 440.0, 17.6, cur)
    assert same["coverage_pct"] == pytest.approx(100.0, abs=1.0)
    mismatched = brazilprod._company_block("iron_ore", _spec("kb/d"), 440.0, 17.6, cur)
    assert mismatched["coverage_pct"] is None


def test_export_blocks_never_reconcile():
    """A '% of exports' table has nothing to reconcile against national production."""
    cur = brazilprod.load_curated()
    blk = brazilprod._company_block("soybeans", {"key": "soybeans", "raw_unit": "1000 MT"},
                                    186000.0, 42.1, cur)
    assert blk["coverage_pct"] is None
    assert blk["unit_is_pct"] is True


# ── country ranking ──────────────────────────────────────────────────────────
def test_country_rows_keep_brazil_even_outside_the_top_n():
    """Brazil is 12th in copper — it must still appear, or the page silently drops
    the one country the whole module exists to show."""
    values = {f"C{i}": 100 - i for i in range(12)}
    values["Brazil"] = 1.0
    rows = brazilprod._country_rows(values, top_n=5)
    assert any(r["is_brazil"] for r in rows), "Brazil must survive the top-N cut"
    assert rows[-1]["country"] == "Other"


def test_country_shares_sum_to_one_hundred():
    values = {"Brazil": 40.0, "A": 30.0, "B": 20.0, "C": 10.0}
    rows = brazilprod._country_rows(values, top_n=2)
    assert sum(r["share"] for r in rows) == pytest.approx(100.0, abs=0.1)


def test_rank_of_brazil():
    assert brazilprod._rank_of("Brazil", {"A": 10, "Brazil": 9, "B": 1}) == 2


# ── the built store, when this box has one ───────────────────────────────────
def test_store_is_wellformed_if_present():
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box — the daily pull has not run")
    for key, com in (store.get("commodities") or {}).items():
        assert com["world"] > 0, f"{key}: world total must be positive"
        assert 0 <= com["share"] <= 100, f"{key}: share out of range"
        assert com["unit"], f"{key}: needs a display unit"
        blk = com.get("companies")
        if blk:
            assert blk["basis_label"], f"{key}: company block must state its basis"
            assert sum(r["share_brazil"] for r in blk["rows"]) == pytest.approx(100.0, abs=0.2)


def test_headline_rows_shape():
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    df = brazilprod.headline_rows(store)
    assert list(df.columns) == ["Commodity", "key", "Group", "Year", "Brazil", "World",
                                "Unit", "Share %", "Rank", "Companies"]
    assert df["Share %"].is_monotonic_decreasing, "ranked by Brazil's share, descending"
