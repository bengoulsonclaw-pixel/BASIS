"""anmdata.py — real Brazilian metals production per company, from the regulator.

Companion to `anpdata.py`, which does the same job for crude. Same contract: what
this returns is SOURCED, and anything it cannot source it refuses to return.

WHERE THE DATA COMES FROM, AND WHY NOT THE OBVIOUS PLACE
--------------------------------------------------------
The obvious place is ANM's Anuário Mineral Brasileiro (AMB), built from the RAL
filings every mining title-holder must submit. It is NOT usable for this: the
published files are aggregated to `Ano base x UF x Classe Substancia x Substancia`
and carry NO title-holder. Verified 2026-08-24 against the live files — the AMB
route gives better national totals and nothing at all about companies.

What does carry the company is CFEM — the royalty every title-holder pays on what
it SELLS. ANM publishes the returns, and `CPF_CNPJ` is unmasked for companies:

    Ano | Mes | Processo | Tipo_PF_PJ | CPF_CNPJ | Substancia | UF | Municipio
        | QuantidadeComercializada | UnidadeDeMedida | ValorRecolhido

Monthly, back to 2002, and refreshed daily — materially more current than ANP's
crude drop, which dies at 2023-12.

WHAT THIS DATA IS NOT
---------------------
  * It is SOLD, not produced. CFEM is levied on commercialised volume, so stock
    movements and the gap between output and shipments both land in it. For hedging
    and brokerage that is arguably the better basis — sold tonnes are priced tonnes —
    but it is a different question from production and must be labelled as one.
  * It is a GROSS tonnage, not contained metal. That is why only iron ore and
    bauxite survive the gates below: their national reference figures are also
    reported as ore. Copper, nickel, niobium and manganese are published by USGS as
    CONTAINED METAL, and converting CFEM's ore tonnage to metal content needs a
    grade assumption per mine — which would be inventing a number. Those stay blank.
  * `UnidadeDeMedida` cannot be trusted on its own. Gold is the worst case: filers
    declare ore tonnage under a gram unit, so summing the unit field gives Brazil
    3.4 MILLION tonnes of gold against a world total near 3,300. Gold is excluded
    for that reason and the reason travels with the exclusion.

THE THREE TRAPS, ALL OF WHICH PRODUCE PLAUSIBLE NUMBERS RATHER THAN ERRORS
--------------------------------------------------------------------------
  1. DUAL SUBSTANCE LABELS. Every metal appears as both `FERRO` and `MINERIO DE
     FERRO` (and so on). Filtering on one silently drops roughly half the rows —
     CSN Mineracao vanishes completely from an iron-ore query that matches `FERRO`
     alone, because it files under the other label.
  2. CORRUPT FILINGS. One ArcelorMittal row declares 289,061,408 t against R$28,895
     of royalty — a misplaced decimal, and on its own 47% of Brazil's national iron
     ore. Money is the trustworthy field here: royalty is actually paid, so the
     implied R$/t exposes a typo the tonnage column cannot.
  3. ROM vs SALEABLE PRODUCT. Some filers declare run-of-mine gross tonnage, which
     carries a fraction of the royalty per tonne because it is worth a fraction as
     much. Gerdau files iron ore at R$1.06/t against Vale's R$12.68/t. Summing both
     as though they were the same thing overstates Brazil by ~80%.

CLI:  python src/anmdata.py [--force] [--year YYYY] [--commodity iron_ore]
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data" / "cache"
SIGNALS = _ROOT / "data" / "signals"
STORE = SIGNALS / "anm_metals.json"
NAMES = _ROOT / "data" / "cnpj_names.json"
KEY_FILE = _ROOT / "data" / "dadosgov_key.txt"

# dados.gov.br catalogue. The key is needed for the CATALOGUE call that lists the
# dataset's resources; the CSVs it points at are served openly from dadosabertos.
CATALOGUE = "https://dados.gov.br/dados/api/publico/conjuntos-dados"
# `nomeConjuntoDados` is a fuzzy TEXT search, not a slug lookup — passing the slug
# `sistema-arrecadacao` matches an unrelated dataset at a federal university. Search
# the acronym and pin the result by slug + owning organisation.
CFEM_SEARCH = "CFEM"
CFEM_DATASET = "sistema-arrecadacao"
CFEM_ORG = "agencia-nacional-de-mineracao"
AUTH_HEADER = "chave-api-dados-abertos"      # confirmed against the live API
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124 Safari/537.36"}

# Fallback only. discover() asks the catalogue for the real resource list so a
# rename upstream is survivable; this is what to try when the API itself is down.
CFEM_FALLBACK = "https://dadosabertos.anm.gov.br/CFEM/CFEM_Arrecadacao_2022_2026.csv"

# Column positions. The header carries accented names that arrive in latin-1, and
# matching them by string has already broken once, so index positionally and assert
# the header shape instead.
C_ANO, C_MES, C_PROC, C_ANOPROC, C_TIPO, C_DOC = 0, 1, 2, 3, 4, 5
C_SUB, C_UF, C_CODMUN, C_MUN, C_QTD, C_UNI, C_VALOR = 6, 7, 8, 9, 10, 11, 12
N_COLS = 14

TO_TONNES = {"t": 1.0, "kg": 1e-3, "g": 1e-6}

# The base year the national reference figures below are quoted for. CFEM carries
# later years, but comparing a 2026 sold tonnage against a 2024 USGS national figure
# would make the reconciliation gate meaningless — so both move together or neither.
LATEST_YEAR = 2024

# National reference figures, USGS Mineral Commodity Summaries 2025, base year 2024.
# These mirror `countries_curated` in data/brazil_curated.json and exist here so the
# reconciliation gate has something independent to check against.
COMMODITIES = {
    "iron_ore": {
        "labels": ("FERRO", "MIN\xc9RIO DE FERRO"),
        "units": ("t",),
        "national_t": 440e6,
        "basis_note": "usable ore, comparable to CFEM's sold tonnage",
    },
    "bauxite": {
        "labels": ("BAUXITA", "MIN\xc9RIO DE ALUM\xcdNIO"),
        "units": ("t",),
        "national_t": 31e6,
        "basis_note": "dry-equivalent ore, comparable to CFEM's sold tonnage",
    },
    "copper": {
        "labels": ("COBRE", "MIN\xc9RIO DE COBRE"),
        "units": ("t",),
        "national_t": 400e3,
        "basis_note": "USGS publishes CONTAINED COPPER; CFEM declares ore and "
                      "concentrate, and the conversion needs a per-mine grade",
    },
    "nickel": {
        "labels": ("N\xcdQUEL", "MIN\xc9RIO DE N\xcdQUEL"),
        "units": ("t",),
        "national_t": 88e3,
        "basis_note": "USGS publishes CONTAINED NICKEL; CFEM declares ore",
    },
    "niobium": {
        "labels": ("NI\xd3BIO", "MIN\xc9RIO DE NI\xd3BIO"),
        "units": ("t",),
        "national_t": 75e3,
        "basis_note": "USGS publishes CONTAINED NIOBIUM; CFEM declares ore "
                      "and concentrate",
    },
    "manganese": {
        "labels": ("MANGAN\xcaS", "MIN\xc9RIO DE MANGAN\xcaS"),
        "units": ("t",),
        "national_t": 620e3,
        "basis_note": "USGS publishes CONTAINED MANGANESE; CFEM declares ore",
    },
    "gold": {
        "labels": ("OURO", "MIN\xc9RIO DE OURO"),
        "units": ("g", "kg"),
        "national_t": 60.0,
        "basis_note": "the unit field is unreliable for gold — filers declare ore "
                      "tonnage under a gram unit, so the declared quantities sum to "
                      "roughly a thousand times world production",
    },
}

# CFEM names the legal entity that holds the title. Several commercial groups file
# through more than one, exactly as PRIO and 3R do in the ANP data — so roll them up
# or one client appears several times with a fraction of its book on each row.
PARENTS = {
    "VALE S.A.": "Vale",
    "MINERACOES BRASILEIRAS REUNIDAS S.A. MBR": "Vale",
    "CSN MINERACAO S.A.": "CSN Minera\xe7\xe3o",
    "ANGLO AMERICAN MINERIO DE FERRO BRASIL S/A": "Anglo American",
    "ANGLO AMERICAN NIQUEL BRASIL LTDA": "Anglo American",
    "SAMARCO MINERACAO S.A.": "Samarco",
    "MINERACAO USIMINAS S.A.": "Usiminas",
    "GERDAU ACOMINAS S/A": "Gerdau",
    "ARCELORMITTAL BRASIL S.A.": "ArcelorMittal",
    "MINERACAO RIO DO NORTE S.A.": "Minera\xe7\xe3o Rio do Norte",
    "ALCOA WORLD ALUMINA BRASIL LTDA": "Alcoa",
    "MINERACAO PARAGOMINAS S.A.": "Hydro Paragominas",
    "NORSK HYDRO BRASIL LTDA": "Hydro Paragominas",
    "COMPANHIA BRASILEIRA DE ALUMINIO": "CBA",
    "CMOC BRASIL MINERACAO, INDUSTRIA E PARTICIPACOES LTDA.": "CMOC Brasil",
    "MINERACAO CARAIBA S/A": "Ero Copper",
    "MINERACAO MARACA INDUSTRIA E COMERCIO S/A": "Lundin Mining",
}

# Listings for the names that have one, keyed by the COMMERCIAL name above.
# CFEM identifies a title-holder by CNPJ and never by ticker, so without this the
# page loses the equities seam entirely and every Listing cell prints an em-dash.
# (ticker, yahoo) — `BZ` resolves to B3 via the mapping in src/yfin.py.
TICKERS = {
    "Vale": ("VALE3 BZ", "VALE"),
    "CSN Minera\xe7\xe3o": ("CMIN3 BZ", "CMIN3.SA"),
    "Anglo American": ("AAL LN", "AAL.L"),
    "Usiminas": ("USIM5 BZ", "USIM5.SA"),
    "Gerdau": ("GGBR4 BZ", "GGBR4.SA"),
    "ArcelorMittal": ("MT NA", "MT"),
    "CBA": ("CBAV3 BZ", "CBAV3.SA"),
    "Alcoa": ("AA US", "AA"),
    "Hydro Paragominas": ("NHY NO", "NHY.OL"),
    "CMOC Brasil": ("3993 HK", "3993.HK"),
    "Ero Copper": ("ERO CN", "ERO.TO"),
    "Lundin Mining": ("LUN CN", "LUN.TO"),
    # Samarco (Vale/BHP JV), Minera\xe7\xe3o Rio do Norte (consortium), Vallourec Tubos
    # do Brasil and the smaller miners are unlisted or have no clean Brazilian line.
}


# ---------------------------------------------------------------- fetch


def _key() -> str | None:
    try:
        k = KEY_FILE.read_text(encoding="utf-8").strip()
        return k or None
    except OSError:
        return None


def _get(url: str, timeout: int = 600, tries: int = 3,
         headers: dict | None = None) -> bytes:
    """gov.br drops connections mid-download often enough to need a retry."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={**_UA, **(headers or {})})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as exc:                      # noqa: BLE001 — retried below
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def discover() -> list[str]:
    """CFEM arrecadacao CSV URLs, newest segment last, via the dados.gov.br catalogue.

    The catalogue call is the one that needs the API key — without it dados.gov.br
    returns 401, which is what blocked this module being written at all. The files
    it points at are then served openly.
    """
    key = _key()
    if not key:
        raise RuntimeError(
            f"No dados.gov.br API key. Put it in {KEY_FILE.relative_to(_ROOT)} — "
            "free, issued instantly from a gov.br login at https://dados.gov.br.")
    hdr = {AUTH_HEADER: key, "Accept": "application/json"}
    listing = json.loads(_get(f"{CATALOGUE}?nomeConjuntoDados={CFEM_SEARCH}&pagina=1",
                              timeout=120, headers=hdr).decode("utf-8"))
    ds = next((x for x in listing
               if x.get("nome") == CFEM_DATASET
               and x.get("nomeOrganizacao") == CFEM_ORG), None)
    if ds is None:
        raise RuntimeError("CFEM dataset not found in the dados.gov.br catalogue")
    detail = json.loads(_get(f"{CATALOGUE}/{ds['id']}", timeout=120,
                             headers=hdr).decode("utf-8"))
    urls = [r.get("link", "") for r in detail.get("recursos", [])]
    # Only the year-segmented arrecadacao files. The unsegmented CFEM_Arrecadacao.csv
    # is the whole 2002-> base and duplicates every segment, so taking both would
    # double-count everything.
    out = sorted(u for u in urls
                 if "CFEM_Arrecadacao_" in u and u.lower().endswith(".csv"))
    return out or [CFEM_FALLBACK]


def _cached(url: str, force: bool = False, max_age_hours: float = 24 * 7) -> Path:
    """Download to data/cache/ (gitignored — CFEM alone is 93 MB)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / url.rsplit("/", 1)[-1]
    if not force and dst.exists() and \
            (time.time() - dst.stat().st_mtime) < max_age_hours * 3600:
        return dst
    blob = _get(url)
    dst.write_bytes(blob)
    return dst


# ---------------------------------------------------------------- parse


def _num(raw: str) -> float | None:
    """CFEM mixes decimal conventions within one file, exactly as ANP does.
    '1.234,56' -> dot is the thousands separator; '1234.56' -> dot is the point."""
    s = (raw or "").strip()
    if not s:
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(year: int, force: bool = False) -> list[tuple]:
    """Every CFEM arrecadacao row for `year`, exact duplicates removed.

    Byte-identical rows — same process, municipality, quantity, royalty AND creation
    timestamp — are duplicate filings, not two real sales. 19 of them sat in 2024 iron
    ore alone and inflated Vale by 15.7 Mt.
    """
    seen: set[tuple] = set()
    out: list[tuple] = []
    for url in discover():
        path = _cached(url, force=force)
        with open(path, encoding="latin-1", newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if not header or len(header) != N_COLS:
                print(f"  ANM: {path.name} has {len(header or [])} columns, "
                      f"expected {N_COLS} — schema changed, skipping")
                continue
            for row in rd:
                if len(row) < N_COLS - 1 or row[C_ANO] != str(year):
                    continue
                t = tuple(row)
                if t in seen:
                    continue
                seen.add(t)
                out.append(t)
    return out


def _aggregate(rows: list[tuple], spec: dict) -> dict[tuple[str, str], list]:
    """{(cnpj, PF/PJ): [tonnes, royalty]} for one commodity's substance labels."""
    labels = set(spec["labels"])
    units = set(spec["units"])
    per: dict[tuple[str, str], list] = {}
    for r in rows:
        if r[C_SUB] not in labels:
            continue
        unit = r[C_UNI].strip()
        if unit not in units or unit not in TO_TONNES:
            continue
        qty, val = _num(r[C_QTD]), _num(r[C_VALOR])
        if qty is None or val is None or qty <= 0:
            continue
        acc = per.setdefault((r[C_DOC], r[C_TIPO]), [0.0, 0.0])
        acc[0] += qty * TO_TONNES[unit]
        acc[1] += val
    return per


def _reference_rate(per: dict) -> float:
    """Royalty-weighted median of implied R$/tonne.

    Royalty is the trustworthy column — it is money actually paid, so a filer cannot
    understate it the way a tonnage typo overstates quantity. Weighting the median by
    royalty means ROM declarations, which pay little against a large tonnage, carry
    almost no weight, and the reference lands on genuine saleable product.
    """
    pairs = sorted(((v / q, v) for q, v in per.values() if q > 0 and v > 0),
                   key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return 0.0
    run = 0.0
    for rate, w in pairs:
        run += w
        if run >= total / 2:
            return rate
    return pairs[-1][0]


def by_company(commodity: str, year: int, rows: list[tuple] | None = None,
               rate_floor: float = 0.4, tolerance: float = 0.25) -> dict:
    """Sourced per-company sold volumes for one commodity, or a stated refusal.

    Two gates, and the commodity is only published if it passes both:

      RATE GATE     — a filer whose implied royalty per tonne is below
                      `rate_floor` x the royalty-weighted median is declaring ROM or
                      has a corrupt quantity. Excluded from the split, and its
                      tonnage is reported separately rather than silently dropped.
      RECONCILE GATE— what survives must land within `tolerance` of the independent
                      national figure. Failing it means the basis does not match
                      (ore tonnage against contained metal, typically), and no share
                      is published at all.
    """
    spec = COMMODITIES[commodity]
    rows = load_rows(year) if rows is None else rows
    per = _aggregate(rows, spec)
    if not per:
        return {"commodity": commodity, "year": year, "sourced": False,
                "reason": f"No CFEM rows for {year} under "
                          f"{' / '.join(spec['labels'])}."}

    ref = _reference_rate(per)
    floor = ref * rate_floor
    kept = {k: v for k, v in per.items() if v[0] > 0 and (v[1] / v[0]) >= floor}
    dropped = {k: v for k, v in per.items() if k not in kept}
    total = sum(v[0] for v in kept.values())
    national = spec["national_t"]
    ratio = total / national if national else 0.0

    base = {
        "commodity": commodity, "year": year,
        "reference_rate_brl_per_t": round(ref, 4),
        "excluded_filers": len(dropped),
        "excluded_tonnes": round(sum(v[0] for v in dropped.values()), 1),
        "national_reference_t": national,
        "reconciliation": round(ratio, 4),
        "source": "ANM — CFEM arrecada\xe7\xe3o (open data)",
        "basis": "sold",
    }
    if abs(ratio - 1.0) > tolerance:
        return {**base, "sourced": False,
                "reason": (f"CFEM's sold tonnage reconciles to {ratio:.1f}x the "
                           f"national figure, outside the {tolerance:.0%} tolerance — "
                           f"{spec['basis_note']}.")}

    companies = []
    for (doc, tipo), (tonnes, royalty) in sorted(kept.items(), key=lambda x: -x[1][0]):
        companies.append({
            "cnpj": doc, "type": tipo,
            "tonnes": round(tonnes, 1),
            "share_of_brazil": round(tonnes / total * 100, 3) if total else 0.0,
            "royalty_brl": round(royalty, 2),
            "implied_brl_per_t": round(royalty / tonnes, 3) if tonnes else 0.0,
        })
    return {**base, "sourced": True, "total_t": round(total, 1),
            "n_companies": len(companies), "companies": companies}


# ---------------------------------------------------------------- names


def _load_names() -> dict:
    try:
        return json.loads(NAMES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve_names(cnpjs: list[str], sleep: float = 0.6) -> dict:
    """CNPJ -> registered company name, cached on disk.

    BrasilAPI mirrors the Receita Federal register and is free and unauthenticated,
    but it is rate-limited, so every hit is cached and only unknown CNPJs are fetched.
    A lookup that fails is left absent rather than stored as a guess.
    """
    cache = _load_names()
    todo = [c for c in dict.fromkeys(cnpjs) if c and c not in cache]
    for i, cnpj in enumerate(todo):
        try:
            blob = _get(f"https://brasilapi.com.br/api/cnpj/v1/{cnpj}",
                        timeout=45, tries=2)
            d = json.loads(blob.decode("utf-8"))
            name = (d.get("razao_social") or "").strip()
            if name:
                cache[cnpj] = name
        except Exception as exc:                      # noqa: BLE001
            print(f"  ANM: CNPJ {cnpj} unresolved ({type(exc).__name__})")
        if i < len(todo) - 1:
            time.sleep(sleep)
    if todo:
        NAMES.parent.mkdir(parents=True, exist_ok=True)
        NAMES.write_text(json.dumps(cache, ensure_ascii=False, indent=1,
                                    sort_keys=True), encoding="utf-8")
    return cache


def _norm(name: str) -> str:
    """Match key for PARENTS. The register is inconsistent about punctuation — the
    same company is 'MINERACAO RIO DO NORTE SA' here and 'S.A.' elsewhere — so match
    on letters and digits only rather than maintaining every spelling."""
    return "".join(ch for ch in name.upper() if ch.isalnum())


_PARENTS_NORM = {_norm(k): v for k, v in PARENTS.items()}

# Legal-form suffixes. A call list wants 'Itaminas', not 'Itaminas Comercio De
# Minerios Sa' — but only strip them from the END, since 'SA' appears inside real words.
_SUFFIXES = ("S/A", "S.A.", "SA", "LTDA.", "LTDA", "EIRELI", "ME", "EPP", "HOLDING")


def _tidy(raw: str) -> str:
    words = raw.title().replace("  ", " ").strip().split()
    while words and _norm(words[-1]) in {_norm(s) for s in _SUFFIXES}:
        words.pop()
    return " ".join(words).rstrip(" -–") or raw.title()


def label(cnpj: str, names: dict) -> str:
    """Commercial name where we roll subsidiaries up, otherwise the registered one."""
    raw = names.get(cnpj)
    if not raw:
        return f"CNPJ {cnpj}"
    return _PARENTS_NORM.get(_norm(raw)) or _tidy(raw)


def with_names(block: dict, names: dict | None = None) -> dict:
    """Attach resolved names and merge a group's subsidiaries into one row.

    `names` is injectable so the roll-up can be tested without network access.
    """
    if not block.get("sourced"):
        return block
    if names is None:
        names = resolve_names([c["cnpj"] for c in block["companies"]])
    merged: dict[str, dict] = {}
    for c in block["companies"]:
        nm = label(c["cnpj"], names)
        m = merged.setdefault(nm, {"company": nm, "tonnes": 0.0, "royalty_brl": 0.0,
                                   "entities": []})
        m["tonnes"] += c["tonnes"]
        m["royalty_brl"] += c["royalty_brl"]
        m["entities"].append(c["cnpj"])
    total = sum(m["tonnes"] for m in merged.values())
    out = []
    for m in sorted(merged.values(), key=lambda x: -x["tonnes"]):
        out.append({**m,
                    "tonnes": round(m["tonnes"], 1),
                    "royalty_brl": round(m["royalty_brl"], 2),
                    "share_of_brazil": round(m["tonnes"] / total * 100, 3) if total else 0.0})
    return {**block, "companies": out, "n_companies": len(out)}


# ---------------------------------------------------------------- driver


def from_store(commodity: str) -> dict | None:
    """One commodity's gated, named block from the precomputed store.

    Read-only ON PURPOSE. Building it means parsing a 93 MB CSV, which belongs to the
    daily pull, not to a page open — so a missing store is reported rather than
    silently triggering a minutes-long rebuild in front of a user.
    """
    try:
        blob = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return (blob.get("commodities") or {}).get(commodity)


def refresh(year: int = LATEST_YEAR, force: bool = False) -> dict:
    """Every commodity, gated, named, and written to the signals store."""
    rows = load_rows(year, force=force)
    out = {"year": year, "n_rows": len(rows), "commodities": {}}
    for name in COMMODITIES:
        block = by_company(name, year, rows=rows)
        out["commodities"][name] = with_names(block)
    SIGNALS.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    force = "--force" in argv
    year = 2024
    if "--year" in argv:
        year = int(argv[argv.index("--year") + 1])
    only = argv[argv.index("--commodity") + 1] if "--commodity" in argv else None

    built = refresh(year=year, force=force)
    print(f"ANM CFEM — {built['n_rows']:,} deduplicated rows for {year}")
    print(f"  store: {STORE.relative_to(_ROOT)}\n")
    for name in ([only] if only else list(COMMODITIES)):
        block = built["commodities"][name]
        if not block.get("sourced"):
            print(f"  {name:<11} BLANK — {block['reason']}")
            continue
        print(f"  {name:<11} {block['total_t']/1e6:>8,.2f} Mt sold, "
              f"{block['n_companies']} companies, "
              f"reconciles {block['reconciliation']:.2f}x national "
              f"(excluded {block['excluded_filers']} filers / "
              f"{block['excluded_tonnes']/1e6:,.1f} Mt as ROM or corrupt)")
        for c in block["companies"][:8]:
            print(f"       {c['company']:<34} {c['tonnes']/1e6:>8,.2f} Mt "
                  f"{c['share_of_brazil']:>6.2f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
