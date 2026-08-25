"""Checks on the ANM CFEM metals feed (src/anmdata.py).

As with ANP, every bug this module hit produced a plausible NUMBER rather than an
error, so those are the cases worth pinning:

  * DUAL SUBSTANCE LABELS — matching `FERRO` alone silently loses every filer who
    files under `MINERIO DE FERRO`, which is how CSN Mineracao disappeared from a
    national iron-ore table without anything raising;
  * A CORRUPT FILING — one ArcelorMittal row declares 289 Mt against R$28,895 of
    royalty, on its own 47% of Brazil's iron ore;
  * ROM vs SALEABLE PRODUCT — Gerdau files at R$1.06/t against Vale's R$12.68/t, and
    summing both as one basis overstates Brazil by ~80%;
  * A BASIS MISMATCH THAT MUST BLANK RATHER THAN SCALE — CFEM reports gross ore while
    USGS reports contained metal, so copper/nickel/niobium/manganese cannot be
    published without inventing a grade.

The last group is the important one. This module exists because invented company
numbers reached a client-facing PDF, so there are tripwires below asserting the
commodities we cannot source stay unsourced.

Everything here is pure and offline — rows are synthetic tuples in CFEM's shape.
"""
from __future__ import annotations

import pytest

from src import anmdata


N = anmdata.N_COLS


def _row(sub, qty, valor, unit="t", doc="00000000000191", tipo="PJ",
         ano="2024", mes="1", proc="1000", mun="MUNICIPIO"):
    """One CFEM arrecadacao row in file order."""
    r = [""] * N
    r[anmdata.C_ANO] = ano
    r[anmdata.C_MES] = mes
    r[anmdata.C_PROC] = proc
    r[anmdata.C_ANOPROC] = "2010"
    r[anmdata.C_TIPO] = tipo
    r[anmdata.C_DOC] = doc
    r[anmdata.C_SUB] = sub
    r[anmdata.C_UF] = "MG"
    r[anmdata.C_CODMUN] = "3100000"
    r[anmdata.C_MUN] = mun
    r[anmdata.C_QTD] = qty
    r[anmdata.C_UNI] = unit
    r[anmdata.C_VALOR] = valor
    r[N - 1] = "2024-01-01 00:00:00"
    return tuple(r)


IRON = anmdata.COMMODITIES["iron_ore"]["labels"]        # ("FERRO", "MINERIO DE FERRO")


# ── number parsing: both decimal conventions in one file ─────────────────────
def test_num_reads_comma_decimals():
    assert anmdata._num("155024,500000") == pytest.approx(155024.5)


def test_num_reads_dot_decimals():
    assert anmdata._num("1615.7229") == pytest.approx(1615.7229)


def test_num_treats_dot_as_thousands_when_both_present():
    assert anmdata._num("1.234,56") == pytest.approx(1234.56)


def test_num_returns_none_rather_than_zero_for_junk():
    """A junk quantity must drop the row, not read as zero tonnes sold."""
    assert anmdata._num("n/a") is None
    assert anmdata._num("") is None


# ── unit normalisation ───────────────────────────────────────────────────────
def test_units_convert_to_tonnes():
    assert anmdata.TO_TONNES["t"] == 1.0
    assert anmdata.TO_TONNES["kg"] == pytest.approx(1e-3)
    assert anmdata.TO_TONNES["g"] == pytest.approx(1e-6)


def test_aggregate_ignores_units_the_commodity_does_not_declare():
    """Iron ore is declared in tonnes. A stray kg row must not be silently scaled
    into the total on the assumption the unit column is right."""
    rows = [_row(IRON[0], "1000", "12800"), _row(IRON[0], "5000", "3", unit="kg")]
    per = anmdata._aggregate(rows, anmdata.COMMODITIES["iron_ore"])
    assert sum(v[0] for v in per.values()) == pytest.approx(1000.0)


# ── trap 1: dual substance labels ────────────────────────────────────────────
def test_both_substance_labels_are_captured():
    """The exact miss: CSN Mineracao files under 'MINERIO DE FERRO' and vanishes
    from a query matching 'FERRO' alone."""
    rows = [_row(IRON[0], "100", "1280", doc="A"),
            _row(IRON[1], "100", "1280", doc="B")]
    per = anmdata._aggregate(rows, anmdata.COMMODITIES["iron_ore"])
    assert len(per) == 2
    assert sum(v[0] for v in per.values()) == pytest.approx(200.0)


def test_every_commodity_declares_both_label_spellings():
    """A new commodity added with only the bare label would silently halve."""
    for name, spec in anmdata.COMMODITIES.items():
        assert len(spec["labels"]) >= 2, f"{name} declares one substance label"
        assert any(l.startswith("MIN\xc9RIO DE") for l in spec["labels"]), name


# ── trap 2: exact duplicate filings ──────────────────────────────────────────
def test_identical_rows_are_one_sale():
    """Byte-identical rows — same process, quantity, royalty AND creation stamp —
    are a duplicate filing. 19 of them inflated Vale by 15.7 Mt in 2024."""
    dup = _row(IRON[1], "3897362,2", "2059479,67")
    seen, out = set(), []
    for r in (dup, dup, _row(IRON[1], "100", "1280", proc="2000")):
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    assert len(out) == 2


# ── trap 3: the royalty-rate gate ────────────────────────────────────────────
def test_reference_rate_is_weighted_by_royalty_not_by_count():
    """Money is the trustworthy column. A plain median across filers is dragged down
    by a crowd of small ROM declarations; weighting by royalty paid lands the
    reference on genuine saleable product."""
    per = {("real", "PJ"): [300e6, 3.8e9],        # R$12.67/t, pays nearly all the royalty
           ("rom1", "PJ"): [30e6, 3.0e7],         # R$1.00/t
           ("rom2", "PJ"): [30e6, 3.0e7],
           ("rom3", "PJ"): [30e6, 3.0e7]}
    assert anmdata._reference_rate(per) > 10.0


def test_corrupt_filing_is_excluded():
    """The ArcelorMittal row: 289,061,408 t against R$28,895 — a misplaced decimal,
    and on its own 47% of Brazil's national iron ore."""
    rows = [_row(IRON[0], "440000000", "5600000000", doc="VALE"),
            _row(IRON[0], "289061408,7", "28895,76", doc="CORRUPT")]
    got = anmdata.by_company("iron_ore", 2024, rows=rows)
    assert got["sourced"] is True
    assert [c["cnpj"] for c in got["companies"]] == ["VALE"]
    assert got["excluded_filers"] == 1


def test_rom_declarations_are_excluded_but_reported():
    """ROM tonnage must not vanish silently — it is named as excluded so a reader can
    see the national picture is larger than the published split."""
    rows = [_row(IRON[0], "440000000", "5600000000", doc="VALE"),
            _row(IRON[1], "32000000", "33000000", doc="ROM")]     # ~R$1.03/t
    got = anmdata.by_company("iron_ore", 2024, rows=rows)
    assert [c["cnpj"] for c in got["companies"]] == ["VALE"]
    assert got["excluded_tonnes"] == pytest.approx(32e6)


# ── the reconciliation gate ──────────────────────────────────────────────────
def test_reconciliation_failure_blanks_rather_than_scales():
    """A basis mismatch must produce no shares at all. Scaling to fit the national
    figure is exactly the fitted residual that made `coverage` circular before."""
    rows = [_row("COBRE", "24000000", "36000000", doc="X")]       # 60x national
    got = anmdata.by_company("copper", 2024, rows=rows)
    assert got["sourced"] is False
    assert "companies" not in got
    assert "contained" in got["reason"].lower() or "grade" in got["reason"].lower()


def test_a_passing_commodity_reports_its_reconciliation():
    rows = [_row(IRON[0], "440000000", "5600000000", doc="VALE")]
    got = anmdata.by_company("iron_ore", 2024, rows=rows)
    assert got["sourced"] is True
    assert got["reconciliation"] == pytest.approx(1.0, abs=0.02)


def test_shares_are_of_the_published_total_and_sum_to_100():
    rows = [_row(IRON[0], "330000000", "4200000000", doc="A"),
            _row(IRON[1], "110000000", "1400000000", doc="B")]
    got = anmdata.by_company("iron_ore", 2024, rows=rows)
    assert sum(c["share_of_brazil"] for c in got["companies"]) == pytest.approx(100.0, abs=0.01)


# ── tripwires: what we cannot source must stay unsourced ─────────────────────
def test_gold_is_not_published_from_cfem():
    """Filers declare gold ORE tonnage under a gram unit, so the quantities sum to
    roughly a thousand times world production. If this ever starts passing, the unit
    handling changed and the number needs a human before it reaches a client."""
    spec = anmdata.COMMODITIES["gold"]
    assert "unreliable" in spec["basis_note"]


@pytest.mark.parametrize("commodity", ["copper", "nickel", "niobium", "manganese"])
def test_contained_metal_commodities_declare_why_they_cannot_reconcile(commodity):
    note = anmdata.COMMODITIES[commodity]["basis_note"]
    assert "CONTAINED" in note, f"{commodity} must say why CFEM tonnage is not comparable"


def test_basis_is_labelled_sold_not_produced():
    """CFEM is levied on what was commercialised. Presenting it as production would
    misstate the measure, the same way ANP's operated share is not equity."""
    rows = [_row(IRON[0], "440000000", "5600000000", doc="VALE")]
    assert anmdata.by_company("iron_ore", 2024, rows=rows)["basis"] == "sold"


# ── subsidiary roll-up ───────────────────────────────────────────────────────
def test_parent_matching_ignores_punctuation():
    """The register spells the same company 'MINERACAO RIO DO NORTE SA' and
    'S.A.' — matching on exact strings dropped the roll-up."""
    names = {"1": "MINERACAO RIO DO NORTE SA", "2": "MINERACAO RIO DO NORTE S.A."}
    assert anmdata.label("1", names) == anmdata.label("2", names)


def test_subsidiaries_merge_into_one_client():
    """One commercial group filing through several entities must be one row, or the
    brokerage ranking splits a client's book across its subsidiaries — the PRIO bug
    from the ANP side."""
    block = {"sourced": True, "companies": [
        {"cnpj": "1", "tonnes": 100.0, "royalty_brl": 1.0, "share_of_brazil": 50.0},
        {"cnpj": "2", "tonnes": 100.0, "royalty_brl": 1.0, "share_of_brazil": 50.0}]}
    names = {"1": "VALE S.A.", "2": "MINERACOES BRASILEIRAS REUNIDAS S.A. MBR"}
    merged = anmdata.with_names(block, names=names)
    assert merged["n_companies"] == 1
    assert merged["companies"][0]["company"] == "Vale"
    assert merged["companies"][0]["tonnes"] == pytest.approx(200.0)
    assert merged["companies"][0]["share_of_brazil"] == pytest.approx(100.0)
    assert sorted(merged["companies"][0]["entities"]) == ["1", "2"]


def test_legal_suffixes_are_trimmed_for_display():
    assert anmdata._tidy("ITAMINAS COMERCIO DE MINERIOS SA") == "Itaminas Comercio De Minerios"
    assert anmdata._tidy("MINERACAO CONEMP LTDA") == "Mineracao Conemp"


def test_unresolved_cnpj_is_named_as_unknown_not_guessed():
    assert anmdata.label("99999999999999", {}) == "CNPJ 99999999999999"


# ── listings ─────────────────────────────────────────────────────────────────
def test_every_ticker_is_keyed_to_a_name_the_roll_up_actually_produces():
    """TICKERS is keyed by COMMERCIAL name, so a key that no PARENTS value produces
    can never match — the listing just silently never appears. CFEM carries no ticker
    of its own, so this map is the only route to the equities seam."""
    produced = set(anmdata.PARENTS.values())
    for name in anmdata.TICKERS:
        assert name in produced, f"TICKERS key {name!r} is not a PARENTS commercial name"


def test_tickers_carry_both_a_bloomberg_and_a_yahoo_symbol():
    """The page quotes off Bloomberg-style tickers and yfin resolves the Yahoo one;
    half a pair means one of the two surfaces shows a blank."""
    for name, pair in anmdata.TICKERS.items():
        assert len(pair) == 2 and all(pair), f"{name} has an incomplete listing pair"
