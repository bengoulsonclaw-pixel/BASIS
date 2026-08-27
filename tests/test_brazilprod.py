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


def test_keyed_rows_each_carry_a_document_and_url():
    """`keyed` means "typed in from a named published document you can re-check", so
    every number in such a block has to say WHICH document and WHERE.

    This is the tripwire against the failure that caused the purge: numbers assembled
    from recall and given a plausible-sounding block-level source. A row with no URL
    cannot be re-checked, which makes it indistinguishable from an invented one.
    """
    cur = brazilprod.load_curated()
    for key, blk in cur["companies"].items():
        if blk.get("provenance") != "keyed":
            continue
        for r in blk["rows"]:
            if r.get("kind") in ("unsourced", "other", "artisanal"):
                continue          # a residual has nothing to cite by definition
            assert r.get("source_url", "").startswith("http"), \
                f"{key}/{r['company']}: keyed row has no source_url"
            assert r.get("document"), f"{key}/{r['company']}: keyed row names no document"
            assert r.get("as_of"), f"{key}/{r['company']}: keyed row has no as-of date"


def test_unsourced_remainder_is_never_counted_as_coverage():
    """Coverage must measure what we SOURCED. Counting the remainder would print 100%
    for a table that names two of three producers — the fitted-residual mistake by
    another route."""
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    for key, com in (store.get("commodities") or {}).items():
        blk = com.get("companies") or {}
        if blk.get("unsourced") or not blk.get("rows"):
            continue
        gap = blk.get("unsourced_share") or 0
        if gap <= 0:
            continue
        assert blk["coverage_pct"] is None or blk["coverage_pct"] < 99.9, \
            f"{key}: coverage {blk['coverage_pct']}% despite {gap}% unattributed"
        assert blk["named_share"] + gap == pytest.approx(100.0, abs=0.6), key


def test_internal_only_blocks_never_reach_the_client_pdf():
    """Soy and corn name SLC Agricola, which is under 1% of the crop. That is useful
    for sizing brokerage and misleading as a producer ranking, so those company tables
    are marked internal_only and the PDF must drop them while keeping the country page.
    """
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    internal = [k for k, c in (store.get("commodities") or {}).items()
                if (c.get("companies") or {}).get("internal_only")]
    assert internal, "expected at least one internal-only company block"
    from src import brazilreport
    for key in internal:
        got = brazilreport._block(store["commodities"][key])
        assert not got.get("has_co"), f"{key}: internal-only block reached the PDF"
        assert not got.get("rows"), f"{key}: internal-only rows reached the PDF"


def test_no_corporate_producer_rows_are_not_hedge_clients():
    """The 99% of Brazil's soy grown by independent farms is not a client to call, and
    must not inflate the brokerage book the way the garimpo line would for gold."""
    m = brazilprod.hedge_matrix(include_unhedgeable=True)
    if m.empty:
        pytest.skip("no Brazil store on this box")
    assert not m["Company"].str.contains("Independent farms", case=False).any()


def test_national_hedge_does_not_depend_on_knowing_the_producers():
    """The bug this guards: an unsourced company table used to disable the WHOLE hedge,
    including the national figure — so Brazil's coffee, cotton and cattle showed 'no
    contract' when all three have liquid futures. How big the country's hedge is and
    who would place it are separate questions.
    """
    nat = brazilprod.national_hedge()
    if not nat["rows"]:
        pytest.skip("no Brazil store on this box")
    store = brazilprod.load()
    for r in nat["rows"]:
        assert r["lots_yr"] > 0, f"{r['label']}: sized at zero lots"
    # at least one commodity must be sized nationally while having no producer split
    assert nat["n_with_company_split"] < nat["n_hedgeable"], (
        "expected some commodity hedged nationally with no company breakdown")
    # and no 'no contract' reason may blame missing company data
    for m in nat["missing"]:
        assert "not sourced" not in m["reason"].lower(), (
            f"{m['label']}: dropped from the pool because producers are unknown, "
            f"which has nothing to do with whether a contract exists")


def test_national_pool_totals_are_internally_consistent():
    nat = brazilprod.national_hedge()
    if not nat["rows"]:
        pytest.skip("no Brazil store on this box")
    assert nat["total_yr"] == sum(r["lots_yr"] for r in nat["rows"])
    assert nat["total_mth"] == pytest.approx(nat["total_yr"] / 12, rel=0.01)
    assert nat["total_day"] == pytest.approx(nat["total_yr"] / brazilprod.TRADING_DAYS, rel=0.01)


def test_named_clients_never_exceed_the_national_pool():
    """The client roll-up is a SUBSET of the pool — every client's lots come from its
    share of a national figure. If it ever exceeds the pool, a share is being applied
    to the wrong denominator."""
    nat = brazilprod.national_hedge()
    book = brazilprod.broker_book()
    if not nat["rows"] or book.empty:
        pytest.skip("no Brazil store on this box")
    assert book["Lots (1 yr)"].sum() <= nat["total_yr"] * 1.001


def test_pool_ratio_columns_are_fractions_of_the_full_hedge():
    """100 / 50 / 25 must be the same number scaled, not three separate calculations —
    and the totals row must be the sum of its own column, not of the 100% column."""
    nat = brazilprod.national_hedge()
    if not nat["rows"]:
        pytest.skip("no Brazil store on this box")
    assert nat["ratios"] == (100, 50, 25)
    for r in nat["rows"]:
        for pct in nat["ratios"]:
            assert r[f"{pct}% yr"] == pytest.approx(r["lots_yr"] * pct / 100, rel=0.001, abs=1)
            assert r[f"{pct}% mth"] == pytest.approx(r[f"{pct}% yr"] / 12, rel=0.02, abs=1)
            assert r[f"{pct}% day"] == pytest.approx(
                r[f"{pct}% yr"] / brazilprod.TRADING_DAYS, rel=0.02, abs=1)
    for pct in nat["ratios"]:
        for suffix, _d in brazilprod.HEDGE_PERIODS:
            col = f"{pct}% {suffix}"
            assert nat["totals"][col] == sum(r[col] for r in nat["rows"]), col


def test_every_unit_the_page_prints_has_a_plain_english_expansion():
    """'mb/d' is million barrels a day here, but M is the Roman thousand elsewhere — and
    this page shows Brazil in mb/d directly above Petrobras in kb/d. Any unit that
    reaches a metric or a table header must be explainable."""
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    for key, com in (store.get("commodities") or {}).items():
        if str(com.get("unit", "")).startswith("%"):
            continue
        assert brazilprod.unit_help(com["unit"]), f"{key}: unit {com['unit']!r} unexplained"
        blk = com.get("companies") or {}
        u = blk.get("unit")
        if u and not str(u).startswith("%"):
            assert brazilprod.unit_help(u), f"{key}: company unit {u!r} unexplained"


def test_barrel_units_say_which_multiple_they_are():
    """The exact confusion: 3.77 mb/d beside 3,016.96 kb/d is the same quantity."""
    assert "million" in brazilprod.unit_help("mb/d")
    assert "thousand" in brazilprod.unit_help("kb/d")
    assert "1 mb/d" in brazilprod.unit_help("kb/d")


def test_tonnes_are_declared_metric():
    """Asked and answered once already — the only short ton in this module is the CBOT
    soybean-meal contract, and it must not leak into a production unit."""
    for u in ("Mt", "kt", "t", "Mt CWE"):
        assert "metric" in brazilprod.unit_help(u), u
        assert "short" not in brazilprod.unit_help(u), u


def test_round_turns_scale_the_pool_linearly():
    one = brazilprod.national_hedge(turns=1.0)
    two = brazilprod.national_hedge(turns=2.0)
    if not one["rows"]:
        pytest.skip("no Brazil store on this box")
    assert two["total_yr"] == pytest.approx(one["total_yr"] * 2, rel=0.001)


def test_unsourced_remainder_is_not_a_hedge_client():
    """There is nobody to call for the part of Brazil we cannot name, so it must not
    appear in the brokerage roll-up."""
    m = brazilprod.hedge_matrix(include_unhedgeable=True)
    if m.empty:
        pytest.skip("no Brazil store on this box")
    assert not m["Company"].str.contains("Not sourced", case=False).any()


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


# The trade houses. None of them GROWS anything — they buy, crush and ship. Their
# export share was once printed as though it were production, which is the specific
# error the row-crop tripwires below exist to prevent.
TRADE_HOUSES = ("bunge", "cargill", "adm", "archer", "louis dreyfus", "ldc",
                "cofco", "amaggi", "glencore")


def test_row_crops_never_show_a_trade_house_as_a_producer():
    """Soybeans, corn and coffee are grown by tens of thousands of farms.

    The original tripwire pinned these blocks to an EXPORT basis. That guarded the
    mechanism rather than the rule, and blocked a legitimate change: naming the listed
    GROWERS, who really do produce, alongside an explicit non-corporate bucket. So this
    now guards the rule itself — a trade house must never appear under a production
    basis, whatever else the block does.
    """
    cur = brazilprod.load_curated()
    for key in ("soybeans", "corn", "coffee"):
        blk = cur["companies"][key]
        if blk.get("basis") != "production":
            continue
        for r in blk["rows"]:
            name = str(r.get("company", "")).lower()
            assert not any(h in name for h in TRADE_HOUSES), (
                f"{key}: {r['company']} is a trade house — its share is what it SHIPS, "
                f"not what it grew, and this block claims production")


def test_row_crop_producer_tables_are_overwhelmingly_non_corporate():
    """If a row crop does name growers, the chart must still show that essentially
    none of the crop has a corporate producer — otherwise naming SLC at 0.8% reads as
    a producer ranking rather than a prospect list."""
    cur = brazilprod.load_curated()
    for key in ("soybeans", "corn"):
        blk = cur["companies"][key]
        if blk.get("provenance") == "names_only":
            continue
        assert blk.get("internal_only") is True, f"{key}: grower table must stay internal"
        total = sum(float(r["volume"]) for r in blk["rows"])
        nonco = sum(float(r["volume"]) for r in blk["rows"]
                    if r.get("kind") in ("artisanal", "other", "unsourced"))
        assert nonco / total > 0.95, (
            f"{key}: only {nonco / total:.1%} is non-corporate — a grower table that "
            f"large would be claiming companies grow the crop")


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




# ── fixtures ─────────────────────────────────────────────────────────────────
# Every curated company block is now `names_only` — no volumes. The share and hedge
# maths still has to be tested, so it runs against a synthetic SOURCED block rather
# than against data that no longer exists.
def _sourced(basis="production", unit="Mt", provenance="live"):
    return {"companies": {"fixture": {
        "basis": basis, "year": 2024, "unit": unit, "confidence": "reported",
        "provenance": provenance, "source": "test fixture",
        "rows": [{"company": "Alpha", "ticker": "AAA BZ", "volume": 300.0},
                 {"company": "Beta", "ticker": "", "volume": 100.0},
                 {"company": "Other (small)", "ticker": "", "volume": 40.0}]}}}


def _fix_block(basis="production", unit="Mt", brazil=440.0, share=17.6):
    return brazilprod._company_block("fixture", {"key": "fixture", "raw_unit": unit},
                                     brazil, share, _sourced(basis, unit))


# ── share maths ──────────────────────────────────────────────────────────────
def _spec(raw_unit="Mt"):
    return {"key": "iron_ore", "raw_unit": raw_unit}


def test_company_shares_sum_to_one_hundred():
    blk = _fix_block()
    assert sum(r["share_brazil"] for r in blk["rows"]) == pytest.approx(100.0, abs=0.1)


def test_world_share_chains_through_brazil():
    """A company's share of WORLD supply is its share of Brazil scaled by Brazil's
    share of the world — the number the desk actually cares about."""
    br_share = 17.6
    blk = _fix_block(share=br_share)
    for r in blk["rows"]:
        assert r["share_world"] == pytest.approx(r["share_brazil"] / 100 * br_share, abs=0.01)
    assert sum(r["share_world"] for r in blk["rows"]) == pytest.approx(br_share, abs=0.05)


def test_other_bucket_sorts_last():
    blk = _fix_block()
    assert blk["rows"][-1]["is_other"], "the 'Other' bucket must sort last"
    assert not any(r["is_other"] for r in blk["rows"][:-1])


def test_coverage_only_computed_when_units_match():
    """Iron ore is Mt against Mt, so the table can be reconciled. The same block
    measured against a kb/d national figure must refuse to reconcile rather than
    print a meaningless percentage."""
    same = _fix_block(unit="Mt", brazil=440.0)
    assert same["coverage_pct"] == pytest.approx(100.0, abs=1.0)
    mismatched = brazilprod._company_block(
        "fixture", {"key": "fixture", "raw_unit": "kb/d"}, 440.0, 17.6, _sourced(unit="Mt"))
    assert mismatched["coverage_pct"] is None


def test_export_share_is_not_publicly_sourceable():
    """Brazil publishes no exporter-level customs data — verified 2026-08-25 against
    Comex Stat's own API, whose filters are country, bloc, state, transport mode,
    customs unit and product code, with no exporter field. Coffee is the block that
    still rests on it, and it must stay blank AND say so.
    """
    blk = brazilprod._company_block("coffee", {"key": "coffee", "raw_unit": "1000 60kg bags"},
                                    71900.0, 37.9, brazilprod.load_curated())
    assert blk["unsourced"] is True
    reason = blk["reason"].lower()
    assert any(s in reason for s in ("exporter", "confidential", "bill-of-lading")), reason


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
        if blk and not blk.get("unsourced"):
            assert blk["basis_label"], f"{key}: company block must state its basis"
            assert sum(r["share_brazil"] for r in blk["rows"]) == pytest.approx(100.0, abs=0.2)
        elif blk:
            assert blk.get("reason"), f"{key}: a blank table must say why it is blank"
            assert not blk.get("rows"), f"{key}: unsourced block must carry no numbers"


def test_headline_rows_shape():
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    df = brazilprod.headline_rows(store)
    assert list(df.columns) == ["Commodity", "key", "Group", "Year", "Brazil", "World",
                                "Unit", "Share %", "Rank", "Companies"]
    assert df["Share %"].is_monotonic_decreasing, "ranked by Brazil's share, descending"


# ── futures hedge equivalents ────────────────────────────────────────────────
# Unit conversion is the whole risk here: one wrong constant and the desk sizes a
# hedge an order of magnitude off. Each case below is checked against the contract
# spec by hand rather than against whatever the code currently returns.
def _hedge(key, brazil, exports=None, blk=None, unit="Mt"):
    return brazilprod._hedge_block(key, brazil, exports, blk, unit)


def test_iron_ore_lots_match_contract_size():
    """440 Mt at 100 t a lot = 4.4m lots."""
    h = _hedge("iron_ore", 440.0)
    assert h["available"] and h["national_lots"] == 4_400_000


def test_gold_tonnes_convert_through_troy_ounces():
    """60 t = 1,929,045 troy oz; COMEX gold is 100 oz a lot."""
    h = _hedge("gold", 60.0, unit="t")
    assert h["national_lots"] == pytest.approx(19_290, abs=2)


def test_soybean_bushel_conversion():
    """1 bu of soybeans is 0.0272155 t, so a 5,000 bu lot is 136.08 t."""
    h = _hedge("soybeans", 186.0)
    assert h["national_lots"] == pytest.approx(1e6 / 136.0775 * 186.0, rel=1e-4)


def test_corn_and_soybean_bushels_differ():
    """A corn bushel is lighter than a soybean one — using one constant for both is
    a classic error worth a test."""
    corn = _hedge("corn", 100.0)["national_lots"]
    soy = _hedge("soybeans", 100.0)["national_lots"]
    assert corn > soy, "the same tonnage is MORE corn lots than soybean lots"
    assert corn / soy == pytest.approx(0.0272155 / 0.0254012, rel=1e-3)


def test_crude_annualises_a_daily_rate():
    """Crude is quoted mb/d — the hedge is a year of it, at 1,000 bbl a lot."""
    h = _hedge("crude_oil", 1.0, unit="mb/d")
    assert h["national_lots"] == pytest.approx(365_000, rel=1e-6)


def test_export_basis_strikes_the_hedge_off_exports():
    """A trade house hedges what it ships, so an export-basis block must size off
    exports, never off the whole crop. Tested on a fixture — the real export blocks
    are unsourced and produce no hedge at all."""
    blk = _fix_block(basis="export", unit="1000 MT")
    h = _hedge("soybeans", 186.0, exports=118.0, blk=blk)
    assert h["qty_basis"] == "exports"
    assert h["national_qty"] == pytest.approx(118.0)
    assert h["national_lots"] < _hedge("soybeans", 186.0)["national_lots"]


def test_company_lots_sum_to_the_national_hedge():
    blk = _fix_block()
    h = _hedge("iron_ore", 440.0, blk=blk)
    assert sum(r["lots"] for r in h["rows"]) == pytest.approx(h["national_lots"], rel=1e-3)


def test_commodities_without_a_future_say_why():
    # Chicken is deliberately NOT here: it has no meat future but does have a feed
    # hedge, covered by test_poultry_is_sized_on_feed_not_meat.
    for key in ("niobium", "pulp", "nickel", "manganese", "bauxite"):
        h = _hedge(key, 100.0)
        assert h["available"] is False
        assert h["reason"], f"{key}: an unhedgeable commodity must explain itself"


def test_cross_hedges_are_flagged_as_proxies():
    """Live cattle against Brazilian carcass beef, and US corn ethanol against cane
    ethanol, are proxies. Losing that flag would let a basis-heavy trade read as clean."""
    for key, unit in (("beef", "Mt CWE"), ("ethanol", "bn litres")):
        h = _hedge(key, 10.0, unit=unit)
        assert h["available"] and h["proxy"] is True and h["note"]


def test_every_hedge_contract_is_in_the_desk_universe():
    """Never quote a hedge in an instrument the book cannot trade."""
    from src.universe import INSTRUMENTS
    for key, spec in brazilprod.HEDGE.items():
        assert spec["ticker"] in INSTRUMENTS, f"{key}: {spec['ticker']} not in the universe"


def test_hedge_present_for_every_commodity_in_the_store():
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    for key, com in store["commodities"].items():
        h = com.get("hedge")
        assert h is not None, f"{key}: every commodity must resolve to a hedge or a reason"
        assert h.get("available") or h.get("reason")


# ── brokerage roll-up ────────────────────────────────────────────────────────
def test_group_key_merges_a_client_across_commodities():
    """Vale's iron ore and copper lines are one client; JBS's beef and poultry are
    one client. Failing to merge them understates the ticket count on a call list."""
    assert brazilprod.group_key("Vale (Salobo, Sossego)") == "Vale"
    assert brazilprod.group_key("Vale") == "Vale"
    assert brazilprod.group_key("JBS (Friboi)") == brazilprod.group_key("JBS (Seara)") == "JBS"
    assert brazilprod.group_key("Anglo American (Minas-Rio)") == "Anglo American"
    # Samarco is a separate JV that trades in its own right, NOT part of Vale
    assert brazilprod.group_key("Samarco (Vale / BHP JV)") == "Samarco"


def test_broker_book_excludes_everything_you_cannot_call():
    """'Other' buckets, garimpo and multi-company lines are production, not clients."""
    book = brazilprod.broker_book()
    if book.empty:
        pytest.skip("no Brazil store on this box")
    names = " ".join(book["Client"]).lower()
    assert "other" not in names
    assert "garimpo" not in names and "artisanal" not in names
    assert "three processors" not in names


def test_broker_book_is_ranked_and_multi_commodity_clients_merge():
    book = brazilprod.broker_book()
    if book.empty:
        pytest.skip("no Brazil store on this box")
    assert book["Lots (1 yr)"].is_monotonic_decreasing
    assert book["Client"].is_unique, "a client must appear once, with its book summed"


def test_turns_scale_the_ticket_count_linearly():
    once = brazilprod.broker_book(turns=1.0)
    rolled = brazilprod.broker_book(turns=4.0)
    if once.empty:
        pytest.skip("no Brazil store on this box")
    assert rolled["Lots (1 yr)"].sum() == pytest.approx(once["Lots (1 yr)"].sum() * 4, rel=1e-3)


def test_poultry_is_sized_on_feed_not_meat():
    """There is no chicken future. An integrator's brokerage sits in corn and soybean
    meal, so poultry producers must NOT come back as zero lots."""
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    h = store["commodities"]["chicken"]["hedge"]
    assert h["available"] and h.get("is_input") and len(h["legs"]) == 2
    assert {l["ticker"] for l in h["legs"]} == {"C A Comdty", "SMA Comdty"}
    assert sum(l["lots"] for l in h["legs"]) == pytest.approx(h["national_lots"], rel=1e-3)


def test_soybean_meal_contract_is_short_tons():
    """CBOT meal is 100 SHORT tons (90.72 t), not 100 metric — a 10% error if missed."""
    store = brazilprod.load()
    if not store:
        pytest.skip("no Brazil store on this box")
    leg = [l for l in store["commodities"]["chicken"]["hedge"]["legs"]
           if l["ticker"] == "SMA Comdty"][0]
    brazil_chicken = store["commodities"]["chicken"]["brazil"]
    expected = brazil_chicken * 1e6 * (1 / 0.75 * 1.75) * 0.30 / 0.90718474 / 100
    assert leg["lots"] == pytest.approx(expected, rel=1e-3)


# ── contract sizes ───────────────────────────────────────────────────────────
# Contract sizes are hand-entered in brazilprod.HEDGE and drive every lot count on
# the page. volbt.POINT_VALUE is maintained separately, for a different module, and
# has already been through a $/quote-unit audit — so it is a genuine second source.
#   POINT_VALUE = $ per 1 point of the QUOTE, therefore  size / quote_divisor == PV.
# A contract absent from POINT_VALUE simply is not cross-checkable here; the test
# says which ones those are rather than passing silently over them.
_QUOTE_DIV = {          # 100 = the contract quotes in CENTS, 1 = quotes in dollars
    "SCOA Comdty": 1, "COA Comdty": 1, "GCA Comdty": 1, "SMA Comdty": 1,
    "CUAA Comdty": 1, "HGA Comdty": 100, "SBA Comdty": 100, "KCA Comdty": 100,
    "CTA Comdty": 100, "C A Comdty": 100, "S A Comdty": 100, "LCA Comdty": 100,
}
# Missing from POINT_VALUE, so no arithmetic second source. Anything here must carry
# a `verified` provenance note in HEDGE instead — except the names still on this
# pending list, which are genuinely unconfirmed and should shrink over time.
_NO_SECOND_SOURCE = {"SCOA Comdty", "CUAA Comdty"}
_UNVERIFIED_PENDING: set = set()          # both now desk-confirmed


def _all_contract_specs():
    out = [(k, s["ticker"], s["size"]) for k, s in brazilprod.HEDGE.items()]
    for k, inp in brazilprod.INPUT_HEDGE.items():
        out += [(f"{k}:input", leg["ticker"], leg["size"]) for leg in inp["legs"]]
    return out


def test_every_contract_has_a_quote_convention_recorded():
    for _key, tkr, _size in _all_contract_specs():
        assert tkr in _QUOTE_DIV, f"{tkr}: no quote convention — cents or dollars?"


def test_contract_sizes_agree_with_the_point_value_table():
    """The load-bearing check: a wrong contract size scales every lot count for that
    commodity, and the error is invisible because the number still looks plausible."""
    from src import volbt
    for key, tkr, size in _all_contract_specs():
        pv = volbt.POINT_VALUE.get(tkr)
        if pv is None:
            continue
        assert pv == pytest.approx(size / _QUOTE_DIV[tkr], abs=0.51), (
            f"{key}: {tkr} size {size:,} implies ${size / _QUOTE_DIV[tkr]:,.0f}/point but "
            f"volbt.POINT_VALUE says ${pv:,.0f} — one of the two tables is wrong")


def test_the_uncrosscheckable_contracts_are_the_expected_ones():
    """A contract with no POINT_VALUE entry has no arithmetic second source, so if one
    appears that nobody has checked, it should be noticed rather than trusted."""
    from src import volbt
    missing = {tkr for _k, tkr, _s in _all_contract_specs() if tkr not in volbt.POINT_VALUE}
    assert missing == _NO_SECOND_SOURCE, (
        f"the set of contracts with no second source changed: {missing}")


def test_contracts_with_no_second_source_carry_their_provenance():
    """Iron ore sizes Vale, the largest lot count on the page, and cannot be checked
    against POINT_VALUE — so it has to say where its 100 t/lot came from. Only the
    names on the pending list are allowed to have no provenance at all."""
    for key, spec in brazilprod.HEDGE.items():
        if spec["ticker"] in _NO_SECOND_SOURCE - _UNVERIFIED_PENDING:
            assert spec.get("verified"), (
                f"{key}: {spec['ticker']} has no arithmetic cross-check, so it must record "
                f"how its contract size was confirmed")


# ── company x product hedge matrix ───────────────────────────────────────────
def test_hedge_matrix_gives_a_row_per_company_product():
    m = brazilprod.hedge_matrix()
    if m.empty:
        pytest.skip("no Brazil store on this box")
    assert m.groupby(["Company", "Product"]).size().max() == 1, "one row per company x product"


def test_hedge_ratios_scale_the_lots():
    m = brazilprod.hedge_matrix()
    if m.empty:
        pytest.skip("no Brazil store on this box")
    hedged = m[m["_avail"]]
    for pct in brazilprod.HEDGE_RATIOS:
        assert (hedged[f"{pct}% yr"] <= hedged["100% yr"]).all()
        assert hedged[f"{pct}% yr"].sum() == pytest.approx(
            hedged["100% yr"].sum() * pct / 100.0, rel=1e-3)


def test_each_ratio_breaks_into_year_month_and_trading_day():
    """252 trading days, not 365 calendar days — the daily figure is meant to be held
    against the contract's own daily volume, so it has to be a day it trades on."""
    m = brazilprod.hedge_matrix()
    if m.empty:
        pytest.skip("no Brazil store on this box")
    hedged = m[m["_avail"]]
    assert brazilprod.TRADING_DAYS == 252
    for pct in brazilprod.HEDGE_RATIOS:
        yr = hedged[f"{pct}% yr"]
        assert hedged[f"{pct}% mth"].sum() == pytest.approx(yr.sum() / 12, rel=2e-3)
        assert hedged[f"{pct}% day"].sum() == pytest.approx(yr.sum() / 252, rel=2e-3)


def test_unhedgeable_lines_are_listed_but_carry_no_lots():
    """Suzano's pulp and CBMM's niobium must appear — a missing producer reads as an
    oversight, a blank lot column reads as 'there is no hedge'."""
    m = brazilprod.hedge_matrix(include_unhedgeable=True)
    if m.empty:
        pytest.skip("no Brazil store on this box")
    nohedge = m[~m["_avail"]]
    assert nohedge["100% yr"].isna().all() if len(nohedge) else True
    assert set(brazilprod.hedge_matrix(include_unhedgeable=False)["_avail"]) == {True}


def test_unhedgeable_lots_survive_the_subsidiary_aggregation_as_blanks():
    """A blank must not become a zero on the way through groupby.

    `sum` over an all-NaN group returns 0.0, so the aggregation that merges a client's
    subsidiaries silently turned "no contract exists" into "0 lots" — which reads as a
    client with nothing to hedge. Bauxite is the case: real sold tonnage from ANM, and
    no listed future anywhere in the desk's universe.
    """
    m = brazilprod.hedge_matrix(include_unhedgeable=True)
    if m.empty:
        pytest.skip("no Brazil store on this box")
    nohedge = m[~m["_avail"]]
    if not len(nohedge):
        pytest.skip("every product in this store is hedgeable")
    for col in [c for c in m.columns if c.endswith((" yr", " mth", " day"))]:
        assert nohedge[col].isna().all(), f"{col} shows a number for an unhedgeable line"
        assert not (nohedge[col] == 0).any(), f"{col} shows 0 where it must be blank"


def test_matrix_lots_reconcile_with_the_client_roll_up():
    """The per-product table and the per-client table are two views of one number."""
    m = brazilprod.hedge_matrix()
    book = brazilprod.broker_book()
    if m.empty or book.empty:
        pytest.skip("no Brazil store on this box")
    assert m["100% yr"].sum(skipna=True) == pytest.approx(book["Lots (1 yr)"].sum(), rel=1e-3)


def test_matrix_excludes_non_clients():
    m = brazilprod.hedge_matrix()
    if m.empty:
        pytest.skip("no Brazil store on this box")
    names = " ".join(m["Company"]).lower()
    assert "other" not in names and "garimpo" not in names and "processors" not in names


def test_hedge_totals_sum_the_matrix():
    """The TOTAL row is the whole addressable book; unhedgeable lines hold NaN and must
    drop out, so the total is lots that could actually be traded."""
    m = brazilprod.hedge_matrix()
    if m.empty:
        pytest.skip("no Brazil store on this box")
    t = brazilprod.hedge_totals(m)
    assert t["_n_rows"] == len(m)
    assert t["_n_hedgeable"] == int(m["_avail"].sum()) <= len(m)
    for pct in brazilprod.HEDGE_RATIOS:
        assert t[f"{pct}% yr"] == pytest.approx(m[f"{pct}% yr"].sum(skipna=True), rel=1e-6)
    assert t["25% yr"] == pytest.approx(t["100% yr"] / 4, rel=2e-3)
    assert t["100% day"] == pytest.approx(t["100% yr"] / 252, rel=5e-3)
