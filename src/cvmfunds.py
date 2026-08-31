"""cvmfunds.py — the Brazilian fund industry, from the regulator, for the 🇧🇷 Brazil Funds page.

Brazil is the most transparent large fund market in the world. Every regulated fund
files a DAILY report with the CVM — NAV per share, net assets, subscriptions,
redemptions, holder count — and the CVM republishes the lot as free bulk CSV. No
13F-style quarterly guessing, no size threshold, no vendor subscription. This module
turns that firehose into a screener: who manages what, how big, and how it has done.

WHAT COUNTS AS A "HEDGE FUND" HERE

CVM's `Multimercado` class is the Brazilian analogue — free to hold rates, FX, equities
and offshore risk at once — and it is the default filter on the page. It is NOT a clean
synonym: the class also contains conservative bank-distributed balanced funds and the
whole `Previdência Multimercado` pension book. Both are labelled, never silently
dropped, so the user chooses the definition rather than inheriting ours.

THREE TRAPS, ALL LOAD-BEARING

  * MASTER-FEEDER DOUBLE COUNTING. Brazilian managers run a master holding the
    positions with several feeder (FIC) classes stacked on top, and the feeder's assets
    ARE the master's assets seen twice. Summing net assets naively inflates multimercado
    from ~R$1.9tn to ~R$2.9tn — a 53% overstatement. `Classe_Cotas == 'S'` marks a
    fund-of-quotas; excluding those is the dedupe, and `include_feeders` defaults False.

  * EXCLUSIVE VEHICLES ARE NOT PRODUCTS. 13k of 36k registered classes are single-family
    or single-institution vehicles (`Exclusivo == 'S'`). They are real money and real
    filings, but nobody can buy them, so counting them in a league table answers a
    question nobody asked. Excluded by default, restorable with one toggle.

  * THE SCHEMA MOVED UNDER RESOLUÇÃO CVM 175. The daily file is keyed on
    `CNPJ_FUNDO_CLASSE` + `ID_SUBCLASSE`, not the old `CNPJ_FUNDO`, and the registry
    split into fundo -> classe -> subclasse across three files. Anything joining on
    fund CNPJ alone fragments silently. A class reports EITHER one class-level row
    (`ID_SUBCLASSE` empty) OR one row per subclass, never both — verified on 2026-07-31,
    486 classes with subclasses, zero carrying both — so a class's assets are the SUM
    over its rows and its performance is per subclass, each being a real fee class.

RECONCILIATION, SO THE NUMBER CAN BE DEFENDED

Ex-feeder, ex-previdência, this reproduces R$1,602bn of multimercado against ANBIMA's
published R$1,519bn, with Itaú Asset at 10.6% against their 9.8% (2026-07-31). The
residual is ANBIMA consolidating economic groups — their one "BTG Pactual" is our three
registered gestores — plus a vintage difference. Close enough to trust, far enough apart
that the page says which basis it is on rather than implying it IS the ANBIMA table.

WHAT THIS CANNOT SEE

Offshore feeders. The Cayman and Luxembourg vehicles where much of the foreign money
sits do not file with CVM, so onshore assets understate the big global-macro houses.

CLI:  python src/cvmfunds.py [--force] [--months N] [--min-aum M]
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
STORE = _ROOT / "data" / "signals" / "cvm_funds"
REGISTRY = STORE / "registry.parquet"
METRICS = STORE / "metrics.parquet"
META = STORE / "meta.json"

_INF_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ym}.zip"
_REG_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124 Safari/537.36"}

MONTHS_BACK = 13          # 13, not 12: a 12-month return needs an anchor BEFORE the window
MIN_AUM = 10_000_000.0    # R$10m — below this a class is a shell or a wind-down, not a product
TRADING_DAYS = 252
CDI_SGS = 12              # BCB SGS 12 = CDI, annualised daily factor in percent

# CVM's own class, off registro_classe.Classificacao. Multimercado is the hedge-fund
# analogue; the rest are here so the page can widen without a code change.
CVM_CLASSES = ["Multimercado", "Ações", "Renda Fixa", "Cambial"]
CVM_CLASSES_EN = ["Multi-strategy", "Equity", "Fixed Income", "FX"]

# The ANBIMA sub-classification prefixes that mark a pension wrapper. ANBIMA counts
# these as their own category; CVM files them under whatever they invest in, which is
# the single biggest reason a CVM-derived total will not match an ANBIMA headline.
_PREV_PREFIX = "Previd"

_OPERATING = "Em Funcionamento Normal"


# ── Portuguese -> English ───────────────────────────────────────────────────────────
# CVM publishes in Portuguese and the desk does not read it, so everything DESCRIPTIVE is
# translated for display. Everything IDENTIFYING is not: a fund's and a manager's
# registered name is how you look it up, quote it to a client and match it in a vendor
# system, so those keep their words and only shed legal-form boilerplate.
#
# Translation happens in load(), not in the store, so fixing a term here takes effect on
# the next page load instead of needing a 200-second rebuild.

CVM_CLASS_EN = {"Multimercado": "Multi-strategy", "Ações": "Equity",
                "Renda Fixa": "Fixed Income", "Cambial": "FX"}
_CLASS_PT = {v: k for k, v in CVM_CLASS_EN.items()}

AUDIENCE_EN = {"Público Geral": "Retail", "Qualificado": "Qualified",
               "Profissional": "Professional"}
_AUDIENCE_PT = {v: k for k, v in AUDIENCE_EN.items()}

# ANBIMA's 63 sub-classifications are compositional ("Previdência RF Duração Livre Crédito
# Livre"), so this translates PHRASES rather than mapping all 63 — a new ANBIMA category
# then still comes out readable instead of coming out Portuguese.
#
# Order is load-bearing, longest first: "Crédito Livre" and "Duração Livre" must both be
# consumed before a bare "Livre" is, or "Free Duration Free Credit" degrades to
# "Unconstrained Unconstrained". The truncated spellings are real — ANBIMA cuts the field
# at 40 characters, so "Crédito Livre" also ships as "Crédito Livr" and "Crédito Liv".
_PT_PHRASES = [
    ("balanceados acima de 49", "Balanced (over 49% equity)"),
    ("balanceados de 15-30", "Balanced (15-30% equity)"),
    ("balanceados de 30-49", "Balanced (30-49% equity)"),
    ("balanceados ate 15", "Balanced (up to 15% equity)"),
    ("investimento no exterior", "Offshore"),
    ("invest. no exterior", "Offshore"),
    ("estrat. especifica", "Specific Strategy"),
    ("fundos de mono acao", "Single-Stock"),
    ("capital protegido", "Capital Protected"),
    ("fechados de acoes", "Closed-End Equity"),
    ("valor/crescimento", "Value/Growth"),
    ("grau de invest.", "Investment Grade"),
    ("grau de inv.", "Investment Grade"),
    ("grau de inv", "Investment Grade"),
    ("juros e moedas", "Rates & FX"),
    ("l/s - direcional", "Long/Short Directional"),
    ("l/s - neutro", "Long/Short Neutral"),
    ("divida externa", "External Debt"),
    ("credito livre", "Free Credit"),
    ("credito livr", "Free Credit"),
    ("credito liv", "Free Credit"),
    ("duracao alta", "High Duration"),
    ("duracao baixa", "Low Duration"),
    ("duracao media", "Mid Duration"),
    ("duracao livre", "Free Duration"),
    ("indice ativo", "Active Index"),
    ("data alvo", "Target Date"),
    ("fmp - fgts", "FMP-FGTS"),
    ("multimercados", "Multi-strategy"),
    ("multimercado", "Multi-strategy"),
    ("previdencia", "Pension"),
    ("renda fixa", "Fixed Income"),
    ("balanceados", "Balanced"),
    ("dividendos", "Dividend"),
    ("small caps", "Small Caps"),
    ("indexados", "Index"),
    ("setoriais", "Sector"),
    ("soberano", "Sovereign"),
    ("dinamico", "Dynamic"),
    ("cambial", "FX"),
    ("simples", "Simple"),
    ("trading", "Trading"),
    ("acoes", "Equity"),
    ("ativo", "Active"),
    ("macro", "Macro"),
    ("livre", "Unconstrained"),
]
# The family word each strategy leads with, so the rest can be set off after a dash.
_FAMILIES = ("Pension", "Multi-strategy", "Fixed Income", "Equity", "FX")

# Vehicle boilerplate in a fund's registered name. Every Brazilian fund carries some of
# it and none of it distinguishes one fund from another — "Kapitalo K10 Master Fundo De
# Investimento Financeiro Multimercado" is Kapitalo K10 Master. Stripped for display
# only; the full registered name stays on the Fund tab.
_NAME_NOISE = [
    "fundo de investimento em cotas de fundos de investimento",
    "em cotas de fundos de investimento",
    "fundo de investimento em cotas",
    "fundos de investimento financeiro",
    "fundo de investimento financeiro",
    "classe de cotas de fundo de investimento",
    "fundos de investimento", "fundo de investimento", "classe de investimento",
    "responsabilidade limitada", "resp. limitada", "resp limitada", "resp ltda",
    "de responsabilidade limitada", "classe unica", "de cotas",
]
# Single words that carry no information once the class and strategy are their own
# columns. "classe" and "fundo" are the RCVM-175 wrapper words and appear on thousands
# of names; "multimercado" is already the Strategy column.
_NAME_WORDS = {"fif", "fic", "fi", "ci", "multimercado", "multimercados", "multim",
               "mult", "rf", "financeiro", "financeira", "classe", "fundo", "fundos",
               "investimento", "investimentos", "inv", "invest", "cotas", "ltda"}
_NAME_PHRASES = [("titulos publicos", "Government Bonds"),
                 ("debentures incentivadas", "Tax-Exempt Debentures"),
                 ("debenture incentivada", "Tax-Exempt Debenture"),
                 ("investimento no exterior", "Offshore"),
                 ("invest no exterior", "Offshore"),
                 ("credito privado", "Private Credit"), ("cred priv", "Private Credit"),
                 ("curto prazo", "Short Term"), ("longo prazo", "Long Term"),
                 ("renda fixa", "Fixed Income"),
                 ("infraestrutura", "Infrastructure"),
                 ("previdenciario", "Pension"), ("previdencia", "Pension"),
                 ("referenciado", "Indexed"), ("referenciada", "Indexed"),
                 ("incentivado", "Tax-Exempt"), ("incentivadas", "Tax-Exempt"),
                 ("incentivada", "Tax-Exempt"), ("incentivados", "Tax-Exempt"),
                 ("institucional", "Institutional"), ("liquidez", "Liquidity"),
                 ("soberano", "Sovereign"), ("simples", "Simple"),
                 ("credito", "Credit"), ("mutuo", "Mutual"), ("acoes", "Equity")]

# Manager names. A registered name is brand + legal business description + legal form
# ("SPX GESTÃO DE RECURSOS LTDA"), and only the BRAND identifies anybody. Hunting an
# English rendering for each of CVM's ~40 descriptor spellings was a losing game that
# still left 101 of 839 managers half-translated ("Oceana Investments Administradora De
# Carteira De Valores Mobiliarios"), so the tail is CUT instead and the label is "Oceana".
# Where cutting makes two registered gestores collide, that group falls back to the full
# name minus its legal form — see _resolve_label_clashes.
_MGR_DESCRIPTORS = {
    "gestao", "gestora", "gestor", "gestoes", "administradora", "administracao",
    "distribuidora", "corretora", "consultoria", "asset", "investimentos",
    "investimento", "patrimonial", "patrimonio", "recursos", "wealth", "servicos",
    "sociedade", "participacoes", "empreendimentos", "management",
}
_MGR_FORMS = {"ltda", "ltda.", "s.a.", "s.a", "sa", "s/a", "dtvm", "ctvm", "cctvm",
              "cvmc", "eireli", "me", "epp", "inc", "llc"}

# Joining words that stay lower-case inside a name, once it is not the first word.
_LOWER_WORDS = {"de", "do", "da", "dos", "das", "e", "of", "the", "and", "no", "na"}

# Series markers, and they are everywhere in Brazilian fund names. Left to the ordinary
# rule they carry vowels and title-case into "Ii" and "Iii".
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}

_MARK = "\x00"


def _deaccent(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def _apply(text: str, phrases: list[tuple[str, str]]) -> str:
    """Replace each phrase once, fencing the result so a later, shorter rule cannot match
    inside an already-translated span."""
    out = _deaccent(text).lower()
    for pt, en in phrases:
        if pt in out:
            out = out.replace(pt, f"{_MARK}{en}{_MARK}")
    return out


def _reassemble(marked: str, drop: set[str] | None = None) -> str:
    """Split on the fences: translated spans survive verbatim, untranslated leftovers are
    title-cased so a term we have no rule for still reads as a name, not as raw data."""
    parts, keep = marked.split(_MARK), []
    for i, seg in enumerate(parts):
        if i % 2:                                  # an already-translated span
            keep.append(seg)
            continue
        words = [w for w in seg.split() if not drop or w.strip(".,-") not in drop]
        if words:
            keep.append(" ".join(w.title() for w in words))
    return re.sub(r"\s+", " ", " ".join(keep)).strip(" -,")


def english_strategy(term: str) -> str:
    """One ANBIMA sub-classification in English, e.g.
    'Previdência RF Duração Livre Crédito Liv' -> 'Pension — Fixed Income Free Duration
    Free Credit'."""
    if not term or not str(term).strip():
        return ""
    marked = _apply(term, _PT_PHRASES)
    # a bare 'RF' is Renda Fixa; as a whole word only, or it eats the 'rf' inside a name
    marked = re.sub(r"(?<![a-z])rf(?![a-z])", f"{_MARK}Fixed Income{_MARK}", marked)
    out = _reassemble(marked)
    for fam in _FAMILIES:                          # set the family off from the detail
        if out.startswith(fam) and len(out) > len(fam) + 1:
            return f"{fam} — {out[len(fam):].strip()}"
    return out


def english_class(term: str) -> str:
    return CVM_CLASS_EN.get(str(term).strip(), str(term or "").strip())


def english_audience(term: str) -> str:
    return AUDIENCE_EN.get(str(term).strip(), str(term or "").strip())


_NOISE_SPANS = sorted((tuple(_deaccent(p).lower().split()) for p in _NAME_NOISE),
                      key=len, reverse=True)
_PHRASE_SPANS = sorted(((tuple(_deaccent(p).lower().split()), en) for p, en in _NAME_PHRASES),
                       key=lambda x: len(x[0]), reverse=True)


def tidy_fund_name(name: str) -> str:
    """A fund's registered name minus the vehicle boilerplate.

    Works on token INDICES rather than string replacement so the original tokens survive
    with their accents — "Itaú Sinfonia", not "Itau Sinfonia". Falls back to the full name
    whenever stripping would leave too little to identify the fund by.
    """
    words = str(name or "").split()
    if not words:
        return ""
    flat = [_deaccent(w).strip(".,-()").lower() for w in words]
    out, i = [], 0
    while i < len(words):
        hit = next((sp for sp in _PHRASE_SPANS if flat[i:i + len(sp[0])] == list(sp[0])), None)
        if hit:                                   # a meaningful descriptor: translate it
            out.extend(hit[1].split())
            i += len(hit[0])
            continue
        span = next((sp for sp in _NOISE_SPANS if flat[i:i + len(sp)] == list(sp)), None)
        if span:                                  # vehicle boilerplate: drop it
            i += len(span)
            continue
        if flat[i] not in _NAME_WORDS:
            out.append(words[i])
        i += 1
    tidied = _smart_title(out)
    return tidied if len(tidied) >= 3 else _smart_title(words)


def _smart_title(words: list[str]) -> str:
    """Title-case a registered name without wrecking it.

    CVM ships every name in caps, so a plain .title() turns BTG into "Btg", SPX into
    "Spx" and BB into "Bb". A short all-consonant token is an acronym and stays upper;
    joining words stay lower; accents are preserved because the ORIGINAL token is cased,
    never the deaccented copy used for matching — a Brazil desk should read "Itaú".
    """
    out = []
    for i, w in enumerate(words):
        core = _deaccent(w).strip(".,-()&/").lower()
        if not core:
            out.append(w)
        elif core in _ROMAN or (len(core) <= 4 and not re.search(r"[aeiouy]", core)):
            out.append(w.upper())                       # BTG, SPX, UBS, BB, JGP, G5, III
        elif i and core in _LOWER_WORDS:
            out.append(w.lower())
        else:
            out.append("-".join(part.capitalize() for part in w.split("-")))
    return re.sub(r"\s+", " ", " ".join(out)).strip(" -,")


def tidy_manager(name: str) -> str:
    """A gestor's brand — the registered name up to where its legal description starts."""
    words = str(name or "").split()
    if not words:
        return ""
    flat = [_deaccent(w).strip(".,-()&/").lower() for w in words]
    cut = next((i for i, w in enumerate(flat) if w in _MGR_DESCRIPTORS), len(words))
    # legal form too: a name with no descriptor to cut at ("BANCO BRADESCO S.A.") would
    # otherwise keep its suffix and read as "Banco Bradesco S.a."
    brand = _smart_title([w for w, f in zip(words[:cut], flat[:cut])
                          if f.strip(".") not in _MGR_FORMS])
    # A name that is ALL description ("Gestora de Recursos XYZ") has no brand to cut to.
    return brand if len(brand) >= 2 else _strip_legal_form(name)


def _strip_legal_form(name: str) -> str:
    """The registered name with ONLY the legal form removed — nothing cut, nothing
    translated. The collision fallback: guaranteed distinct, because the registered names
    it is built from are distinct."""
    words = [w for w in str(name or "").split()
             if _deaccent(w).strip(".,").lower() not in _MGR_FORMS]
    return _smart_title(words) or _smart_title(str(name).split())


def _resolve_label_clashes(raw: pd.Series, label: pd.Series) -> pd.Series:
    """Two DIFFERENT registered gestores must never wear the same label.

    Translating the business descriptor is what creates the risk: "BTG Pactual Asset
    Management S/A DTVM", "BTG Pactual Gestão e Consultoria de Investimentos" and "BTG
    Pactual Gestora de Recursos" are three separately registered entities that all
    translate to "BTG Pactual Asset Management" — three identical rows in a league table,
    which is exactly the economic-group merge this module refuses to do to the numbers.

    Where a label collides, that group falls back to the registered name with only the
    legal form dropped. Less English, but each row is still the entity it says it is.
    """
    pairs = pd.DataFrame({"raw": raw, "label": label}).drop_duplicates()
    clashing = set(pairs[pairs.duplicated("label", keep=False)]["raw"])
    if not clashing:
        return label
    return label.where(~raw.isin(clashing), raw.map(_strip_legal_form))


def add_english(df: pd.DataFrame) -> pd.DataFrame:
    """The display vocabulary, derived rather than stored. Portuguese columns stay beside
    it — `screen()` still filters on them, and the Fund tab still shows the registered
    name, because that is what you quote to a client."""
    if df.empty:
        return df
    d = df.copy()
    d["class_en"] = d["cvm_class"].map(english_class)
    d["strategy_en"] = d["anbima"].map(english_strategy)
    d["audience_en"] = d["publico"].map(english_audience)
    d["name_en"] = d["name"].map(tidy_fund_name)
    # Two labels, deliberately. `firm` is the brand and MAY repeat across the several
    # entities one house registers; `gestor_en` never repeats, so a league table on
    # registered gestores can always tell its rows apart.
    d["firm"] = d["gestor"].map(tidy_manager)
    d["gestor_en"] = _resolve_label_clashes(d["gestor"], d["firm"])
    return d


# ── fetch ───────────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 300, tries: int = 4) -> bytes | None:
    """dados.cvm.gov.br truncates mid-download often enough that a single attempt is not
    a fetch, it is a coin toss — the 12MB daily files failed on the first try in testing
    with IncompleteRead. Returns None on a genuine 404 (a month that does not exist yet)
    so the caller can tell "not published" from "download broke"."""
    last = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=_UA), timeout=timeout).read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except Exception as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CVM download failed after {tries} tries: {url} ({last})")


def _read_csv(blob: bytes, member: str | None = None, **kw) -> pd.DataFrame:
    """CVM ships latin-1, semicolon-separated CSV inside a zip. Everything is read as
    text and coerced later — the files mix blank, '0.00' and absent for the same idea."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = member or zf.namelist()[0]
    return pd.read_csv(zf.open(name), sep=";", dtype=str, encoding="latin-1",
                       low_memory=False, **kw)


def _digits(s: pd.Series) -> pd.Series:
    """CNPJs arrive formatted ('00.017.024/0001-53') in the daily file and bare
    ('00332266000131') in the registry. Joining without this matches nothing, and an
    empty join reads downstream as "no funds found" rather than as a bug."""
    return s.fillna("").astype(str).str.replace(r"[^0-9]", "", regex=True)


def _months(n: int = MONTHS_BACK) -> list[str]:
    """The n most recent YYYYMM keys, oldest first, ending with the current month."""
    today = date.today()
    out, y, m = [], today.year, today.month
    for _ in range(n):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


# ── registry ────────────────────────────────────────────────────────────────────────
def build_registry() -> pd.DataFrame:
    """fundo -> classe -> subclasse, flattened to one row per CLASS.

    The gestor (who actually runs the money) lives on the FUND, the classification and
    the feeder/exclusive flags live on the CLASS, and the daily file is keyed on the
    class — so all three files are needed to answer "who manages this and what is it".
    """
    blob = _get(_REG_URL)
    if blob is None:
        raise RuntimeError("CVM registry 404 — registro_fundo_classe.zip moved?")
    cls = _read_csv(blob, "registro_classe.csv")
    fnd = _read_csv(blob, "registro_fundo.csv")

    keep_f = ["ID_Registro_Fundo", "CNPJ_Fundo", "Gestor", "Administrador",
              "Tipo_Fundo", "Denominacao_Social"]
    fnd = fnd[[c for c in keep_f if c in fnd.columns]].rename(
        columns={"Denominacao_Social": "fund_name"})

    reg = cls.merge(fnd, on="ID_Registro_Fundo", how="left", suffixes=("", "_fund"))
    reg["cnpj"] = reg["CNPJ_Classe"].fillna("").str.strip()
    reg = reg[reg["cnpj"].str.len() > 0].copy()

    out = pd.DataFrame({
        "cnpj":        reg["cnpj"],
        "name":        reg["Denominacao_Social"].fillna(reg.get("fund_name", "")),
        "gestor":      reg["Gestor"].fillna("(não informado)").str.strip(),
        "admin":       reg["Administrador"].fillna("").str.strip(),
        "tipo_classe": reg["Tipo_Classe"].fillna(""),
        "cvm_class":   reg["Classificacao"].fillna(""),
        "anbima":      reg["Classificacao_Anbima"].fillna(""),
        "is_feeder":   reg["Classe_Cotas"].fillna("") == "S",
        "is_exclusive": reg["Exclusivo"].fillna("") == "S",
        "publico":     reg["Publico_Alvo"].fillna(""),
        "situacao":    reg["Situacao"].fillna(""),
        "start":       pd.to_datetime(reg["Data_Inicio"], errors="coerce"),
        "esg":         reg.get("Classe_ESG", pd.Series(index=reg.index, dtype=str)).fillna("") == "S",
    })
    out["is_prev"] = out["anbima"].str.startswith(_PREV_PREFIX)
    # A class in pré-operacional or liquidação still files, and its numbers are real but
    # not comparable — a fund winding down prints redemptions that look like outflows.
    out["operating"] = out["situacao"] == _OPERATING
    return out.drop_duplicates(subset=["cnpj"]).reset_index(drop=True)


# ── daily NAV ───────────────────────────────────────────────────────────────────────
def _nav_path(ym: str) -> Path:
    return STORE / f"nav_{ym}.parquet"


def fetch_month(ym: str, universe: set[str] | None = None) -> pd.DataFrame | None:
    """One month of daily reports, filtered to `universe` (class CNPJs) if given.

    Returns None when the month is not published yet — the current month exists from its
    first business day, but a month can legitimately 404 at a year boundary.
    """
    blob = _get(_INF_URL.format(ym=ym))
    if blob is None:
        return None
    df = _read_csv(blob)
    df["cnpj"] = _digits(df["CNPJ_FUNDO_CLASSE"])
    if universe is not None:
        df = df[df["cnpj"].isin(universe)]
    out = pd.DataFrame({
        "cnpj":     df["cnpj"],
        # Empty for a single-class fund, populated when the class splits into fee
        # classes. Each subclass carries its OWN quota, so it is part of the row key.
        "subclass": df["ID_SUBCLASSE"].fillna(""),
        "date":     pd.to_datetime(df["DT_COMPTC"], errors="coerce"),
        "quota":    pd.to_numeric(df["VL_QUOTA"], errors="coerce"),
        "pl":       pd.to_numeric(df["VL_PATRIM_LIQ"], errors="coerce"),
        "subs":     pd.to_numeric(df["CAPTC_DIA"], errors="coerce"),
        "redem":    pd.to_numeric(df["RESG_DIA"], errors="coerce"),
        "holders":  pd.to_numeric(df["NR_COTST"], errors="coerce"),
    })
    return out.dropna(subset=["date"]).reset_index(drop=True)


def refresh_nav(months: list[str], universe: set[str], force: bool = False) -> list[str]:
    """Cache each month as its own parquet. Only M and M-1 are re-downloaded — the CVM
    revises those two daily and freezes the rest — so a routine refresh moves ~25MB
    rather than the ~160MB a full rebuild costs.

    Returns the months actually present on disk afterwards.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    live = set(months[-2:])                       # M and M-1: still being revised
    have = []
    for ym in months:
        path = _nav_path(ym)
        if path.exists() and not force and ym not in live:
            have.append(ym)
            continue
        try:
            df = fetch_month(ym, universe)
        except Exception as exc:
            # A dead month is survivable; pretending it was empty is not. Keep whatever
            # is already cached and say which month is stale.
            print(f"  CVM {ym}: {type(exc).__name__} — {exc}")
            if path.exists():
                have.append(ym)
            continue
        if df is None:
            print(f"  CVM {ym}: not published yet")
            continue
        if df.empty:
            print(f"  CVM {ym}: parsed to nothing — schema changed?")
            continue
        df.to_parquet(path, index=False)
        have.append(ym)
        print(f"  CVM {ym}: {len(df):,} rows, {df['cnpj'].nunique():,} classes")
    # Months that fell out of the window are dead weight; the portal only serves ~12.
    for stale in STORE.glob("nav_*.parquet"):
        if stale.stem.replace("nav_", "") not in months:
            stale.unlink(missing_ok=True)
    return sorted(have)


def load_nav(months: list[str] | None = None) -> pd.DataFrame:
    """Every cached month, concatenated and de-duplicated on (fund, date).

    The overlap matters: a resubmission can leave the same fund-day in two monthly
    files with different numbers, and the LATER file is the corrected one.
    """
    files = sorted(STORE.glob("nav_*.parquet"))
    if months is not None:
        files = [f for f in files if f.stem.replace("nav_", "") in months]
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return (df.sort_values("date")
              .drop_duplicates(subset=["cnpj", "subclass", "date"], keep="last")
              .reset_index(drop=True))


# ── benchmark ───────────────────────────────────────────────────────────────────────
def cdi_index(start: date) -> pd.Series:
    """Cumulative CDI growth factor, indexed by date, 1.0 at `start`.

    SGS 12 publishes the DAILY factor as a percent (0.05166 = 0.05166% that day), so the
    index is a running product, never a sum: over a year the difference between
    compounding and adding is worth ~90bp on a 14% rate, which is the whole margin
    between beating CDI and missing it.

    Returns an empty Series if BCB is unreachable — every %CDI figure downstream then
    reads as unavailable rather than as 0.
    """
    try:
        from src import macrodata
        s = macrodata.sgs(CDI_SGS, title="CDI", start=start.isoformat())
        if not s:
            return pd.Series(dtype=float)
        idx = pd.Series({pd.Timestamp(d): v for d, v in s.obs}).sort_index()
        return (1.0 + idx / 100.0).cumprod()
    except Exception:
        return pd.Series(dtype=float)


# ── metrics ─────────────────────────────────────────────────────────────────────────
def _anchor(wide: pd.DataFrame, target: pd.Timestamp, tol_days: int = 12) -> pd.Series:
    """The last quota on or before `target`, per fund, but ONLY where that observation
    is within `tol_days` of it.

    Without the tolerance a fund launched last month gets its 12-month return measured
    from its first ever day and prints a headline number that is really a one-month
    number. That is the silent-truncation failure this codebase keeps re-learning: a
    partial window that looks complete is worse than a blank.
    """
    sub = wide.loc[wide.index <= target]
    if sub.empty:
        return pd.Series(index=wide.columns, dtype=float)
    vals = sub.ffill().iloc[-1]
    # Age of each fund's last REAL observation at the anchor date — ffill above would
    # otherwise happily carry a 2025 quota forward to answer a 2026 question.
    #
    # Found by row POSITION and mapped back through the index, never by arithmetic on a
    # raw int64 epoch. That shortcut is a unit trap and it cost a full build: a parquet
    # round-trip hands the index back as datetime64[**us**] while `Timestamp.value` is
    # always nanoseconds, so the two disagree by 1000x, every fund measured as 20,670
    # days stale, and EVERY return column came back blank — with vol, which needs no
    # anchor, still populated and the table still looking well-formed.
    mask = sub.notna()
    pos = pd.Series(np.arange(len(sub), dtype=float), index=sub.index)
    last_pos = mask.mul(pos, axis=0).where(mask).max()
    idx_ns = sub.index.to_numpy(dtype="datetime64[ns]")
    stamps = np.full(len(sub.columns), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = last_pos.notna().to_numpy()
    if ok.any():
        stamps[ok] = idx_ns[last_pos[ok].to_numpy(dtype=int)]
    age = (pd.Timestamp(target).to_datetime64() - stamps) / np.timedelta64(1, "D")
    return vals.where(pd.Series(age, index=sub.columns) <= tol_days)


def last_full_date(nav: pd.DataFrame, coverage: float = 0.80) -> pd.Timestamp:
    """The most recent date on which the INDUSTRY has actually reported.

    This is not `nav["date"].max()`, and using the max is the single most destructive
    mistake available here. Administrators have one business day to file and they use it
    unevenly, so the newest date in the file carries only the fastest filers — on the
    2026-08 drop the last date held 9 classes out of 25,162. A snapshot taken there is
    not a small sample of the industry, it is an arbitrary nine funds, and every total,
    league table and league-table share computed from it is meaningless while looking
    perfectly well-formed.

    So: walk back to the newest date whose reporting count is at least `coverage` of the
    median trading day's, and take the whole cross-section there.
    """
    counts = nav.groupby("date")["cnpj"].count()
    if counts.empty:
        raise RuntimeError("CVM: no dated rows in the NAV cache")
    full = counts[counts >= counts.median() * coverage]
    if full.empty:                       # nothing is complete — fall back, loudly
        print(f"  CVM: no date reaches {coverage:.0%} coverage; using the fullest "
              f"({counts.idxmax().date()}, {counts.max():,} classes)")
        return counts.idxmax()
    return full.index.max()


def _drawdown(wide: pd.DataFrame) -> pd.Series:
    """Worst peak-to-trough on the quota series, per fund, over the cached window."""
    filled = wide.ffill()
    return (filled / filled.cummax() - 1.0).min()


def compute_metrics(nav: pd.DataFrame, registry: pd.DataFrame,
                    min_aum: float = MIN_AUM) -> pd.DataFrame:
    """One row per SUBCLASS (the real investable share class) with assets, returns,
    risk and flows. This is what the page reads; nothing here runs on page-open.
    """
    if nav.empty:
        return pd.DataFrame()
    nav = nav.copy()
    nav["fund_id"] = nav["cnpj"] + "|" + nav["subclass"].fillna("")

    last_date = last_full_date(nav)
    latest = nav[nav["date"] == last_date]

    # Assets are summed to the CLASS (a class's subclasses partition its money) but a
    # subclass row still carries its own slice, so both are kept: `aum` is the tradeable
    # share class, `class_aum` is what belongs in a league table.
    class_aum = latest.groupby("cnpj")["pl"].sum().rename("class_aum").reset_index()

    quota = nav.pivot_table(index="date", columns="fund_id", values="quota", aggfunc="last")
    quota = quota.where(quota > 0)                 # a zero quota is a filing artefact
    last_q = _anchor(quota, last_date, tol_days=8)

    # Start the benchmark BEFORE the NAV window. A 12-month anchor lands within a day or
    # two of the oldest cached month, and a CDI index that begins on exactly that day has
    # nothing to `asof` back to — every 12-month %CDI silently drops out.
    cdi = cdi_index((nav["date"].min() - pd.Timedelta(days=45)).date())
    rows = {}
    for label, months in (("1m", 1), ("3m", 3), ("6m", 6), ("12m", 12)):
        target = last_date - pd.DateOffset(months=months)
        base = _anchor(quota, target)
        rows[f"ret_{label}"] = (last_q / base - 1.0) * 100.0
        if not cdi.empty:
            c_now = cdi.asof(last_date)
            c_then = cdi.asof(target)
            if pd.notna(c_now) and pd.notna(c_then) and c_then:
                bench = c_now / c_then - 1.0
                # "% do CDI" is the Brazilian yardstick. It is only meaningful for a
                # POSITIVE benchmark leg and a positive fund leg — a fund down 3% against
                # a CDI up 14% is not "-21% of CDI", it just lost money — so the negative
                # case is left blank and the return column carries the story.
                ret = rows[f"ret_{label}"]
                rows[f"cdi_{label}"] = pd.Series(
                    np.where((ret > 0) & (bench > 0),
                             ret / (bench * 100.0) * 100.0, np.nan), index=ret.index)
                rows[f"exc_{label}"] = ret - bench * 100.0

    ytd_target = pd.Timestamp(year=last_date.year, month=1, day=1)
    rows["ret_ytd"] = (last_q / _anchor(quota, ytd_target) - 1.0) * 100.0

    # fill_method=None on purpose: a fund that skips a day should leave a GAP, not a
    # forward-filled 0.0% return. Padding invents flat days that drag annualised vol down
    # by a third on anything that reports weekly.
    daily = quota.pct_change(fill_method=None)
    # One filing glitch — a re-based quota, an amortisation — shows up as a ±50% day and
    # would dominate an annualised vol. Clipped, and flagged so the page can say so.
    glitch = (daily.abs() > 0.5).any()
    rows["vol"] = daily.clip(-0.5, 0.5).std() * np.sqrt(TRADING_DAYS) * 100.0
    rows["max_dd"] = _drawdown(quota) * 100.0
    rows["glitch"] = glitch
    rows["obs"] = quota.notna().sum()

    out = pd.DataFrame(rows)
    out["sharpe"] = np.where(out["vol"] > 0, out["exc_12m"] / out["vol"], np.nan) \
        if "exc_12m" in out else np.nan

    # Flows: net money in, over the same windows. This is the one thing a performance
    # table cannot tell you — a fund can be up 20% and bleeding.
    flows = nav.set_index("date")
    for label, months in (("1m", 1), ("3m", 3), ("12m", 12)):
        cut = last_date - pd.DateOffset(months=months)
        w = flows[flows.index > cut]
        net = (w.groupby("fund_id")["subs"].sum() - w.groupby("fund_id")["redem"].sum())
        out[f"flow_{label}"] = net

    tail = latest.set_index("fund_id")
    out["aum"] = tail["pl"]
    out["holders"] = tail["holders"]
    out["cnpj"] = tail["cnpj"]
    out["subclass"] = tail["subclass"]
    out = out[out["cnpj"].notna()]

    out = out.reset_index().rename(columns={"index": "fund_id"})
    out = out.merge(class_aum, on="cnpj", how="left")
    out = out.merge(registry, on="cnpj", how="left")
    out["as_of"] = last_date
    out = out[out["class_aum"].fillna(0) >= min_aum]
    return out.sort_values("aum", ascending=False).reset_index(drop=True)


# ── build / load ────────────────────────────────────────────────────────────────────
def build(force: bool = False, months_back: int = MONTHS_BACK,
          min_aum: float = MIN_AUM) -> pd.DataFrame:
    """Full refresh: registry, the monthly NAV cache, then metrics. Run by the daily pull.

    The universe is screened BEFORE the NAV cache is written, not after: keeping all 25k
    reporting classes would store ~7m rows a year for a table nobody can use, while the
    operating, non-exclusive, above-threshold classes are a few thousand.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("CVM funds: registry…")
    reg = build_registry()
    reg.to_parquet(REGISTRY, index=False)
    print(f"  {len(reg):,} registered classes, {reg['gestor'].nunique():,} gestores")

    # FIF only. FIDC / FIP / FII are real, large and completely different animals with
    # their own datasets, disclosure rules and NAV conventions — mixing them into a
    # multimercado screen would be a category error, not extra coverage.
    universe = reg[reg["operating"] & reg["tipo_classe"].str.contains("FIF", na=False)]
    ids = set(universe["cnpj"])
    print(f"  {len(ids):,} operating FIF classes in scope")

    months = _months(months_back)
    have = refresh_nav(months, ids, force=force)
    if not have:
        raise RuntimeError("CVM: no monthly NAV files cached — download or schema broke")

    nav = load_nav(have)
    print(f"CVM funds: {len(nav):,} fund-days over {len(have)} months")
    met = compute_metrics(nav, reg, min_aum=min_aum)
    met.to_parquet(METRICS, index=False)

    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        # The last COMPLETE cross-section, which lags the newest row in the file by a day
        # or two. Reporting the file's max here would put a date on the page that the
        # numbers underneath it do not come from.
        "as_of": str(pd.Timestamp(met["as_of"].iloc[0]).date()) if not met.empty else None,
        "file_max": str(nav["date"].max().date()),
        "months": have,
        "n_classes": int(met["cnpj"].nunique()) if not met.empty else 0,
        "n_units": int(len(met)),
        "n_gestores": int(met["gestor"].nunique()) if not met.empty else 0,
        "min_aum": min_aum,
        "source": "CVM — Dados Abertos (informe diário + registro fundo/classe)",
        "seconds": round(time.time() - t0, 1),
    }
    META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"CVM funds: {meta['n_units']:,} share classes, as of {meta['as_of']} "
          f"({meta['seconds']}s)")
    return met


def load() -> tuple[pd.DataFrame, dict]:
    """(metrics, meta) off disk. Empty frame + empty dict when nothing is built yet —
    the page renders a "not built" state rather than raising."""
    if not METRICS.exists():
        return pd.DataFrame(), {}
    try:
        met = add_english(pd.read_parquet(METRICS))
        meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
        return met, meta
    except Exception:
        return pd.DataFrame(), {}


def history(cnpj: str, subclass: str = "") -> pd.DataFrame:
    """The cached daily series for one share class — quota, assets, flows, holders."""
    nav = load_nav()
    if nav.empty:
        return nav
    hit = nav[(nav["cnpj"] == cnpj) & (nav["subclass"].fillna("") == (subclass or ""))]
    return hit.sort_values("date").reset_index(drop=True)


# ── screening ───────────────────────────────────────────────────────────────────────
def screen(met: pd.DataFrame, *, cvm_class: str | None = "Multimercado",
           include_feeders: bool = False, include_exclusive: bool = False,
           include_prev: bool = False, publico: list[str] | None = None,
           gestor: str | None = None, min_aum: float = 0.0,
           min_obs: int = 0) -> pd.DataFrame:
    """The default screen is the defensible one: real products, counted once.

    Every exclusion here is a judgement, so each is a separate flag and the page prints
    which are active. `include_feeders=True` is the number that double-counts — it exists
    because the gross figure is what reconciles to some vendor tables, not because it is
    right.
    """
    if met.empty:
        return met
    d = met
    if cvm_class:
        # the UI speaks English; the stored column is CVM's Portuguese
        d = d[d["cvm_class"] == _CLASS_PT.get(cvm_class, cvm_class)]
    if not include_feeders:
        d = d[~d["is_feeder"].fillna(False)]
    if not include_exclusive:
        d = d[~d["is_exclusive"].fillna(False)]
    if not include_prev:
        d = d[~d["is_prev"].fillna(False)]
    if publico:
        d = d[d["publico"].isin([_AUDIENCE_PT.get(p, p) for p in publico])]
    if gestor:
        # match either spelling, so typing "SPX Asset Management" or "SPX Gestão" both work
        hit = d["gestor"].str.contains(gestor, case=False, na=False, regex=False)
        if "gestor_en" in d:
            hit = hit | d["gestor_en"].str.contains(gestor, case=False, na=False, regex=False)
        d = d[hit]
    if min_aum:
        d = d[d["aum"].fillna(0) >= min_aum]
    if min_obs:
        d = d[d["obs"].fillna(0) >= min_obs]
    return d


def by_gestor(d: pd.DataFrame, by_firm: bool = False) -> pd.DataFrame:
    """League table: assets, fund count and asset-weighted performance per manager.

    Returns are ASSET-WEIGHTED, not averaged. A simple mean lets a manager's R$8m
    launch-year fund outvote its R$8bn flagship and puts unrecognisable names at the top
    of a table titled "performance".

    `by_firm` merges the several entities one house registers — BTG Pactual runs three,
    Itaú two — into a single row. Off by default because CVM's unit is the registered
    gestor and that is what the rest of this module counts; on, it answers the question
    people actually ask ("how big is BTG?") and matches how ANBIMA consolidates.
    """
    if d.empty:
        return d
    key = "firm" if (by_firm and "firm" in d) else "gestor"
    g = d.groupby(key)
    out = pd.DataFrame({
        "label": (g[key].first() if key == "firm" else
                  (g["gestor_en"].first() if "gestor_en" in d else g["gestor"].first())),
        "aum": g["aum"].sum(),
        "funds": g["cnpj"].nunique(),
        "holders": g["holders"].sum(),
        "flow_3m": g["flow_3m"].sum(),
        "flow_12m": g["flow_12m"].sum(),
    })
    for col in ("ret_12m", "ret_ytd", "ret_3m", "vol"):
        if col not in d:
            continue
        w = d[[key, "aum", col]].dropna()
        num = (w[col] * w["aum"]).groupby(w[key]).sum()
        den = w.groupby(key)["aum"].sum()
        out[col] = num / den.replace(0, np.nan)
    out["share"] = out["aum"] / out["aum"].sum() * 100.0
    return out.sort_values("aum", ascending=False)


def industry_totals(met: pd.DataFrame) -> dict:
    """The reconciliation block — every basis at once, so the gap is visible rather than
    argued about. This is what the page prints under its headline number."""
    if met.empty:
        return {}
    mm = met[met["cvm_class"] == "Multimercado"]
    gross = mm["aum"].sum()
    exf = mm[~mm["is_feeder"].fillna(False)]["aum"].sum()
    core = mm[(~mm["is_feeder"].fillna(False)) & (~mm["is_prev"].fillna(False))]["aum"].sum()
    return {"gross": gross, "ex_feeder": exf, "ex_feeder_ex_prev": core,
            "feeder_overstate": (gross / exf - 1.0) * 100.0 if exf else np.nan,
            "prev": exf - core}


def main(argv: list[str]) -> int:
    force = "--force" in argv
    mb = MONTHS_BACK
    ma = MIN_AUM
    if "--months" in argv:
        mb = int(argv[argv.index("--months") + 1])
    if "--min-aum" in argv:
        ma = float(argv[argv.index("--min-aum") + 1])
    met = build(force=force, months_back=mb, min_aum=ma)
    tot = industry_totals(met)
    print(f"\nMultimercado, R$bn:  gross {tot['gross']/1e9:,.0f}  "
          f"ex-feeder {tot['ex_feeder']/1e9:,.0f}  "
          f"ex-feeder ex-prev {tot['ex_feeder_ex_prev']/1e9:,.0f}   "
          f"(feeders overstate by {tot['feeder_overstate']:.0f}%)")
    lt = by_gestor(screen(met)).head(15)
    print(f"\nTop 15 gestores — multimercado, ex-feeder, ex-exclusive, ex-previdência:")
    for i, (name, r) in enumerate(lt.iterrows(), 1):
        print(f" {i:>2}. {str(name)[:46]:<46} R${r['aum']/1e9:>7.1f}bn  "
              f"{r['share']:>4.1f}%  {int(r['funds']):>4} funds  "
              f"12m {r.get('ret_12m', float('nan')):>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
