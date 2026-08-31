"""The CVM fund store turns Brazil's daily regulatory filings into a screener.

The data itself is clean and complete — the risk here is not bad numbers, it is
CONFIDENT WRONG ONES, because every trap in this file produces a well-formed table that
a reader has no way to tell apart from a right one. Every test below is about a failure
that was observed while building it, not a hypothetical:

  * summing net assets double-counts every master-feeder pair (measured: +58% on
    multimercado, R$2.5tn against R$1.6tn);
  * the newest date in the file is only the fastest filers (measured: 9 classes out of
    25,162 on the 2026-08 drop) and a snapshot taken there is arbitrary;
  * a fund launched last month will happily report a "12-month" return measured from its
    first ever day unless the anchor is age-checked;
  * the two files spell CNPJs differently, and a failed join reads downstream as "no
    funds found" rather than as a bug;
  * "% do CDI" is meaningless on a negative fund and reads as a ratio anyway;
  * a manager league table averaged rather than asset-weighted puts an R$8m launch ahead
    of an R$8bn flagship;
  * the display layer translates CVM's Portuguese, and two failure modes matter there —
    a phrase-order slip turns "Free Duration Free Credit" into "Unconstrained
    Unconstrained", and cutting a manager down to its brand can merge BTG Pactual's
    three registered entities into one league-table row.

None of these touch the network — every test builds its own frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cvmfunds as cf


# ── fixtures ─────────────────────────────────────────────────────────────────
def _registry() -> pd.DataFrame:
    """Two real pools and one feeder stacked on the second, plus an exclusive vehicle."""
    return pd.DataFrame([
        # cnpj, name, gestor, feeder?, exclusive?, previdência?
        ("11111111000111", "MACRO MASTER FIF", "KAPITALO", False, False, False),
        ("22222222000122", "EQUITY MASTER FIF", "SPX", False, False, False),
        ("33333333000133", "MACRO FIC FIF", "KAPITALO", True, False, False),
        ("44444444000144", "FAMILY OFFICE FIF", "SPX", False, True, False),
        ("55555555000155", "PREV MULTI FIF", "SPX", False, False, True),
    ], columns=["cnpj", "name", "gestor", "is_feeder", "is_exclusive", "is_prev"]).assign(
        cvm_class="Multimercado", anbima="Multimercados Macro", admin="X",
        tipo_classe="Classes de Cotas de Fundos FIF", publico="Profissional",
        situacao=cf._OPERATING, operating=True, esg=False,
        start=pd.Timestamp("2020-01-01"))


def _nav(days: int = 400, quota_growth: float = 0.0005) -> pd.DataFrame:
    """A daily panel where every class reports every business day."""
    dates = pd.bdate_range(end="2026-08-26", periods=days)
    rows = []
    for i, cnpj in enumerate(_registry()["cnpj"]):
        q = 100.0 * np.cumprod(np.full(len(dates), 1.0 + quota_growth))
        rows.append(pd.DataFrame({
            "cnpj": cnpj, "subclass": "", "date": dates, "quota": q,
            "pl": 1_000_000_000.0 * (i + 1), "subs": 0.0, "redem": 0.0, "holders": 50,
        }))
    return pd.concat(rows, ignore_index=True)


# ── 1. master-feeder double counting ─────────────────────────────────────────
def test_feeders_are_excluded_by_default():
    """A FIC's assets ARE its master's assets. Counting both inflates the industry.

    This is the single biggest number on the page and the easiest to get wrong: the
    gross multimercado figure is R$2.5tn and the defensible one is R$1.6tn, and nothing
    in the shape of either says which is which.
    """
    reg = _registry()
    met = reg.assign(aum=[1e9, 2e9, 1e9, 4e9, 5e9], class_aum=[1e9, 2e9, 1e9, 4e9, 5e9])

    default = cf.screen(met)
    assert set(default["cnpj"]) == {"11111111000111", "22222222000122"}, \
        "default screen must drop the feeder, the exclusive and the previdência class"

    gross = cf.screen(met, include_feeders=True, include_exclusive=True, include_prev=True)
    assert gross["aum"].sum() == pytest.approx(13e9)
    assert default["aum"].sum() == pytest.approx(3e9)


def test_industry_totals_expose_every_basis():
    """The reconciliation block must show the gap, not resolve it silently."""
    reg = _registry()
    met = reg.assign(aum=[10e9, 10e9, 5e9, 3e9, 4e9])
    tot = cf.industry_totals(met)
    assert tot["gross"] == pytest.approx(32e9)            # everything
    assert tot["ex_feeder"] == pytest.approx(27e9)        # less the R$5bn FIC
    assert tot["ex_feeder_ex_prev"] == pytest.approx(23e9)  # less the R$4bn previdência
    assert tot["prev"] == pytest.approx(4e9)
    assert tot["feeder_overstate"] == pytest.approx(32 / 27 * 100 - 100)


# ── 2. the snapshot date ─────────────────────────────────────────────────────
def test_snapshot_date_ignores_the_fastest_filers():
    """Administrators have one business day to file and use it unevenly.

    Observed on the real 2026-08 drop: the newest date carried 9 classes out of 25,162,
    and taking the cross-section there produced a league table of two unrecognisable
    names with a 63% "market share". max(date) is never the right answer.
    """
    nav = _nav(days=30)
    full = nav["date"].max()
    # one eager filer posts a day ahead of the industry
    ahead = nav[nav["cnpj"] == "11111111000111"].tail(1).copy()
    ahead["date"] = full + pd.Timedelta(days=1)
    nav = pd.concat([nav, ahead], ignore_index=True)

    assert nav["date"].max() > full
    assert cf.last_full_date(nav) == full, \
        "the snapshot must fall back to the last date the industry actually reported"


def test_snapshot_falls_back_loudly_when_nothing_is_complete():
    """A store with no complete day must still produce a date, not raise — but it takes
    the fullest one rather than the newest."""
    dates = pd.bdate_range(end="2026-08-26", periods=4)
    nav = pd.DataFrame({
        "cnpj": ["a", "b", "c", "a", "b", "a"],
        "date": [dates[0], dates[0], dates[0], dates[1], dates[1], dates[2]],
        "quota": 1.0, "pl": 1.0, "subs": 0.0, "redem": 0.0, "holders": 1, "subclass": "",
    })
    assert cf.last_full_date(nav, coverage=1.5) == dates[0]


# ── 3. the partial-window trap ───────────────────────────────────────────────
def test_a_young_fund_gets_no_twelve_month_return():
    """A partial window that looks complete is worse than a blank.

    Without the age check on the anchor, a fund with three months of history reports a
    three-month number in the 12-month column and ranks at the top of the screener.
    """
    dates = pd.bdate_range(end="2026-08-26", periods=400)
    young = pd.Series(np.nan, index=dates)
    young.iloc[-60:] = np.linspace(100, 130, 60)          # only the last ~3 months exist
    old = pd.Series(100.0, index=dates)                   # flat, so the anchor is unambiguous
    wide = pd.DataFrame({"young|": young, "old|": old})

    target = dates[-1] - pd.DateOffset(months=12)
    base = cf._anchor(wide, target)
    assert np.isnan(base["young|"]), "a fund that did not exist 12m ago must anchor to NaN"
    assert base["old|"] == pytest.approx(100.0), "a fund that did exist must anchor normally"


def test_anchor_does_not_carry_a_stale_quota_forward():
    """ffill is right for a fund that skipped a day and wrong for one that stopped
    reporting six months ago — the tolerance is what separates them."""
    dates = pd.bdate_range(end="2026-08-26", periods=200)
    dead = pd.Series(np.nan, index=dates)
    dead.iloc[:100] = 100.0                                # stopped filing halfway
    wide = pd.DataFrame({"dead|": dead})
    assert np.isnan(cf._anchor(wide, dates[-1], tol_days=8)["dead|"])
    assert cf._anchor(wide, dates[99], tol_days=8)["dead|"] == pytest.approx(100.0)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_anchor_survives_any_datetime_resolution(unit):
    """The regression that cost a whole build.

    Parquet hands a DatetimeIndex back as datetime64[**us**], while `Timestamp.value` is
    always nanoseconds. Ageing the anchor by subtracting raw int64 epochs therefore
    compared microseconds against nanoseconds, made every fund look 20,670 days stale,
    and blanked EVERY return column — while `vol`, which needs no anchor, stayed
    populated and the table stayed well-formed. Nothing about the output said "broken".

    So the anchor is checked at every resolution pandas can hand it, not just the one
    the in-memory frame happens to use.
    """
    dates = pd.bdate_range(end="2026-08-26", periods=300).astype(f"datetime64[{unit}]")
    wide = pd.DataFrame({"live|": np.linspace(100, 120, len(dates))}, index=dates)
    got = cf._anchor(wide, pd.Timestamp("2026-08-26") - pd.DateOffset(months=6))
    assert got["live|"] == pytest.approx(wide["live|"].iloc[len(dates) // 2], rel=0.05), \
        f"anchor must resolve on a datetime64[{unit}] index"


# ── 4. the join key ──────────────────────────────────────────────────────────
def test_cnpj_normalisation_bridges_the_two_spellings():
    """The daily file writes '00.017.024/0001-53'; the registry writes the bare digits.

    Joining without this matches zero rows, and an empty result renders as "no funds
    found" — a silent failure with a plausible-looking page attached.
    """
    got = cf._digits(pd.Series(["00.017.024/0001-53", "00332266000131", None, ""]))
    assert list(got) == ["00017024000153", "00332266000131", "", ""]


# ── 5. %CDI on a losing fund ─────────────────────────────────────────────────
def test_pct_cdi_is_blank_when_the_fund_lost_money():
    """A fund down 3% against a CDI up 14% is not "-21% of CDI" — it just lost money.

    The ratio is arithmetically computable and informationally empty, and it reads as a
    performance figure. It has to be absent, with the return column carrying the story.
    """
    reg = _registry().head(1)
    nav = _nav(days=300)
    nav = nav[nav["cnpj"] == "11111111000111"].copy()
    # force a loss: quota ends below where it started
    nav["quota"] = np.linspace(100.0, 90.0, len(nav))

    met = cf.compute_metrics(nav, reg, min_aum=0)
    assert not met.empty
    row = met.iloc[0]
    assert row["ret_3m"] < 0
    if "cdi_3m" in met.columns:            # only present when BCB answered
        assert np.isnan(row["cdi_3m"]), "%CDI must be blank on a negative return"


# ── 6. the league table ──────────────────────────────────────────────────────
def test_manager_returns_are_asset_weighted():
    """A simple mean lets a tiny fund outvote a flagship and puts unrecognisable names at
    the top of a table titled "performance"."""
    d = pd.DataFrame({
        "gestor": ["ACME", "ACME"],
        "cnpj": ["1", "2"],
        "aum": [9_900e6, 100e6],
        "ret_12m": [10.0, 200.0],          # the tiny fund had a spectacular year
        "ret_ytd": [5.0, 100.0], "ret_3m": [1.0, 20.0], "vol": [5.0, 80.0],
        "holders": [100, 1], "flow_3m": [0.0, 0.0], "flow_12m": [0.0, 0.0],
    })
    lt = cf.by_gestor(d)
    weighted = (10.0 * 9_900 + 200.0 * 100) / 10_000
    assert lt.loc["ACME", "ret_12m"] == pytest.approx(weighted)
    assert lt.loc["ACME", "ret_12m"] < 15.0, "an asset-weighted 12m must not read as 105%"
    assert lt.loc["ACME", "share"] == pytest.approx(100.0)


# ── 7. subclasses partition a class, never duplicate it ──────────────────────
def test_subclass_rows_sum_to_the_class_without_double_counting():
    """Verified on the real 2026-07-31 cross-section: of 486 classes carrying subclass
    rows, none also carried a class-level row. So a class's assets are the SUM over its
    rows — and the performance stays per subclass, each being a real fee class."""
    dates = pd.bdate_range(end="2026-08-26", periods=120)
    rows = []
    for sub, pl, q in (("AAA", 60e6, 100.0), ("BBB", 40e6, 200.0)):
        rows.append(pd.DataFrame({
            "cnpj": "11111111000111", "subclass": sub, "date": dates,
            "quota": q * np.cumprod(np.full(len(dates), 1.001)),
            "pl": pl, "subs": 0.0, "redem": 0.0, "holders": 10}))
    nav = pd.concat(rows, ignore_index=True)

    met = cf.compute_metrics(nav, _registry(), min_aum=0)
    assert len(met) == 2, "each subclass is its own investable share class"
    assert met["class_aum"].unique().tolist() == [100e6], "class assets are the sum"
    assert met["aum"].sum() == pytest.approx(100e6)


# ── 8. the English display layer ─────────────────────────────────────────────
@pytest.mark.parametrize("pt, en", [
    # the ordering trap: "Crédito Livre" and "Duração Livre" must both be consumed
    # before a bare "Livre" is, or both collapse to "Unconstrained Unconstrained"
    ("Renda Fixa Duração Livre Crédito Livre",
     "Fixed Income — Free Duration Free Credit"),
    # ANBIMA truncates the field at 40 chars, so the same category also ships cut short
    ("Previdência RF Duração Livre Crédito Liv",
     "Pension — Fixed Income Free Duration Free Credit"),
    ("Multimercados Macro", "Multi-strategy — Macro"),
    ("Multimercados Invest. no Exterior", "Multi-strategy — Offshore"),
    ("Multimercados L/S - Neutro", "Multi-strategy — Long/Short Neutral"),
    ("Ações Livre", "Equity — Unconstrained"),
    ("Cambial", "FX"),
])
def test_strategy_translation(pt, en):
    assert cf.english_strategy(pt) == en


def test_no_anbima_category_comes_out_portuguese():
    """The phrase table is compositional so that a category nobody anticipated still
    reads as English. Anything left in Portuguese is a missing phrase, not a shrug."""
    leftovers = ("gestao", "duracao", "credito", "livre", "acoes", "previdencia",
                 "renda", "fixa", "soberano", "indexados", "juros", "moedas")
    for pt in ("Multimercados Juros e Moedas", "Previdência Ações Indexados",
               "Renda Fixa Duração Média Grau de Inv.", "Renda Fixa Dívida Externa"):
        got = cf._deaccent(cf.english_strategy(pt)).lower()
        assert not any(w in got.split() for w in leftovers), f"{pt} -> {got}"


@pytest.mark.parametrize("raw, want", [
    ("SPX GESTÃO DE RECURSOS LTDA", "SPX"),                    # acronym stays upper
    ("KAPITALO INVESTIMENTOS LTDA.", "Kapitalo"),
    ("BANCO BRADESCO S.A.", "Banco Bradesco"),                 # no descriptor: form still goes
    ("GENOA CAPITAL GESTORA DE RECURSOS LTDA", "Genoa Capital"),
    ("OCEANA INVESTIMENTOS ADMINISTRADORA DE CARTEIRA", "Oceana"),
])
def test_manager_is_cut_to_its_brand(raw, want):
    assert cf.tidy_manager(raw) == want


def test_two_registered_gestores_never_share_a_label():
    """BTG Pactual registers three entities whose brands are identical. Merging them in a
    league table is the economic-group consolidation this module refuses to do to the
    numbers, so it must not happen by accident in the labels either."""
    raw = pd.Series(["BTG PACTUAL ASSET MANAGEMENT S/A DTVM",
                     "BTG PACTUAL GESTORA DE RECURSOS LTDA",
                     "BTG PACTUAL GESTÃO E CONSULTORIA DE INVESTIMENTOS LTDA",
                     "KAPITALO INVESTIMENTOS LTDA."])
    label = cf._resolve_label_clashes(raw, raw.map(cf.tidy_manager))
    assert label.nunique() == 4, "each registered gestor keeps a label of its own"
    assert label.iloc[3] == "Kapitalo", "a manager with no clash keeps the short brand"


@pytest.mark.parametrize("raw, want", [
    ("KAPITALO K10 MASTER FUNDO DE INVESTIMENTO FINANCEIRO MULTIMERCADO",
     "Kapitalo K10 Master"),
    ("VERDE MASTER FUNDO DE INVESTIMENTO FINANCEIRO MULTIMERCADO RESPONSABILIDADE LIMITADA",
     "Verde Master"),
    # accents survive: the tokens are sliced from the original, never from the
    # deaccented copy used for matching — a Brazil desk should read "Itaú"
    ("ITAÚ SINFONIA MULTIMERCADO CRÉDITO PRIVADO FUNDO DE INVESTIMENTO FINANCEIRO",
     "Itaú Sinfonia Private Credit"),
])
def test_fund_name_is_trimmed_not_translated(raw, want):
    assert cf.tidy_fund_name(raw) == want


@pytest.mark.parametrize("raw, want", [
    ("BB TOP FIXED INCOME SHORT TERM AUTOMATICO II", "BB Top Fixed Income Short Term Automatico II"),
    ("XP MACRO FUNDO DE INVESTIMENTO FINANCEIRO MULTIMERCADO III", "XP Macro III"),
])
def test_roman_series_markers_survive_title_casing(raw, want):
    """Brazilian managers number their series, and the ordinary rule reads a roman numeral
    as a word: II title-cases to "Ii" and III to "Iii"."""
    assert cf.tidy_fund_name(raw) == want


def test_a_fund_that_is_all_boilerplate_keeps_its_full_name():
    """Stripping must never leave a row with nothing to identify it by."""
    assert cf.tidy_fund_name("FUNDO DE INVESTIMENTO FINANCEIRO") != ""


def test_screen_speaks_english_and_filters_on_the_portuguese():
    """The UI offers English; the stored column is CVM's Portuguese. If the mapping back
    breaks, every screen silently returns nothing — which reads as "no funds match"."""
    reg = _registry()
    met = cf.add_english(reg.assign(aum=1e9, class_en="", strategy_en="", audience_en="",
                                    name_en="", gestor_en="", firm=""))
    got = cf.screen(met, cvm_class="Multi-strategy")
    assert len(got) == 2, "English class name must map back to 'Multimercado'"
    assert cf.screen(met, cvm_class="Equity").empty
