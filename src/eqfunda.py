"""Company Fundamentals data layer + database for the BASIS Equities side.

Pulls a ~30-field fundamentals snapshot (valuation / profitability / leverage / growth / income)
for every index constituent via Bloomberg BDP, and APPENDS each pull to a long-format parquet DB —
so after a few months of weekly pulls the page gets history for free (share-count trends, margin
trajectory, "P/E vs its own range"). Mirrors the `equities.py` MODE pattern (bloomberg / snapshot /
mock): mock mode generates deterministic per-ticker fundamentals (MD5 seed) so the whole page works
offline; snapshot mode reads the last Bloomberg-written DB.

Field mnemonics are standard BDP fields but a few vary by terminal build (INTEREST_COVERAGE_RATIO,
BEST_* consensus fields) — anything the terminal doesn't recognise simply comes back empty and the
page shows "—"; verify odd ones with FLDS on the work PC.

Data seams (all mode-agnostic to callers):
  latest_wide(tickers)        -> (DataFrame ticker x FIELD, asof 'YYYY-MM-DD', source label)
  company_frame(universe)     -> one row per ticker: name/sector/region/indices + every field
  add_sector_percentiles(df)  -> adds '<FIELD>__pctl' columns (percent rank within GICS sector)
  field_history(field)        -> DataFrame as_of x ticker (grows as the weekly pulls accumulate)
  refresh(tickers)            -> pull via the CURRENT mode and append to the DB
  maybe_refresh(max_age_days) -> refresh only if the last pull is older (snapshot.py's weekly guard)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import datafeed  # for MODE + the shared xbbg normalisers
from src import equities
from src import yfin
from src.equities import _seed

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "equities"
_DB_FILE = _DATA_DIR / "fundamentals.parquet"          # long: as_of, ticker, field, value, text
_MANIFEST_FILE = _DATA_DIR / "fundamentals_manifest.json"

_BDP_CHUNK = 60          # tickers per BDP request (~30 fields x 60 names keeps requests modest)

# ── field registry ────────────────────────────────────────────────────────────
# kind: how to format ('x' multiple, 'pct' already-in-percent, 'ratio' plain, 'mm' currency
# millions, 'text'/'date' strings). better: which way is "good" for the sector-percentile
# colouring ('high' / 'low' / None = context-dependent, shown uncoloured).
# mock: deterministic generator — ('logn', centre, sigma) or ('norm', mu, sd, lo, hi).
GROUP_ORDER = ["Valuation", "Profitability", "Leverage & Liquidity",
               "Growth & Cash Flow", "Income & Ownership"]


def _f(field, label, group, kind, better, mock=None, derived=False):
    return {"field": field, "label": label, "group": group, "kind": kind,
            "better": better, "mock": mock, "derived": derived}


FIELDS = [
    # Valuation — cheap vs expensive relative to earnings / assets / sales
    _f("PE_RATIO",            "P/E (trailing)",         "Valuation", "x", "low",  ("logn", 18, 0.45)),
    _f("BEST_PE_RATIO",       "P/E (forward, BEst)",    "Valuation", "x", "low",  ("logn", 16, 0.45)),
    _f("PX_TO_BOOK_RATIO",    "Price / Book",           "Valuation", "x", "low",  ("logn", 3.0, 0.70)),
    _f("PX_TO_SALES_RATIO",   "Price / Sales",          "Valuation", "x", "low",  ("logn", 2.5, 0.80)),
    _f("BEST_PEG_RATIO_CUR_YR", "PEG (BEst, cur yr)",    "Valuation", "x", "low",  ("logn", 1.8, 0.50)),
    _f("EV_TO_T12M_EBITDA",   "EV / EBITDA (T12M)",     "Valuation", "x", "low",  ("logn", 12, 0.45)),
    # Profitability — how well revenue turns into profit
    _f("GROSS_MARGIN",        "Gross margin",           "Profitability", "pct", "high", ("norm", 40, 15, 3, 92)),
    _f("OPER_MARGIN",         "Operating margin",       "Profitability", "pct", "high", ("norm", 17, 10, -8, 55)),
    _f("PROF_MARGIN",         "Net margin",             "Profitability", "pct", "high", ("norm", 11, 8, -12, 45)),
    _f("EBITDA_TO_REVENUE",   "EBITDA margin",          "Profitability", "pct", "high", ("norm", 22, 11, -5, 65)),
    _f("RETURN_COM_EQY",      "ROE",                    "Profitability", "pct", "high", ("norm", 15, 10, -15, 55)),
    _f("RETURN_ON_INV_CAPITAL", "ROIC",                 "Profitability", "pct", "high", ("norm", 11, 7, -10, 40)),
    # Leverage & liquidity — financial risk / ability to ride out a downturn
    _f("TOT_DEBT_TO_TOT_EQY", "Total debt / equity",    "Leverage & Liquidity", "pct", "low",  ("logn", 80, 0.75)),
    _f("NET_DEBT_TO_EBITDA",  "Net debt / EBITDA",      "Leverage & Liquidity", "x",   "low",  ("norm", 1.8, 1.3, -2, 6)),
    _f("CUR_RATIO",           "Current ratio",          "Leverage & Liquidity", "ratio", "high", ("norm", 1.6, 0.55, 0.4, 4)),
    _f("INTEREST_COVERAGE_RATIO", "Interest cover",     "Leverage & Liquidity", "x",   "high", ("logn", 9, 0.80)),
    # Growth & cash flow — trajectory and earnings quality
    _f("SALES_GROWTH",        "Revenue growth (YoY)",   "Growth & Cash Flow", "pct", "high", ("norm", 6, 9, -25, 45)),
    _f("SALES_3YR_AVG_GROWTH", "Revenue growth (3Y avg)", "Growth & Cash Flow", "pct", "high", ("norm", 7, 7, -15, 35)),
    _f("EPS_GROWTH",          "EPS growth (YoY)",       "Growth & Cash Flow", "pct", "high", ("norm", 9, 16, -40, 70)),
    _f("CF_CASH_FROM_OPER",   "Operating cash flow",    "Growth & Cash Flow", "mm",  "high", ("logn", 4500, 1.1)),
    _f("CF_FREE_CASH_FLOW",   "Free cash flow",         "Growth & Cash Flow", "mm",  "high", ("logn", 3000, 1.1)),
    _f("FREE_CASH_FLOW_YIELD", "FCF yield",             "Growth & Cash Flow", "pct", "high", ("norm", 4.5, 2.6, -2, 12)),
    _f("NET_INCOME",          "Net income",             "Growth & Cash Flow", "mm",  "high", ("logn", 3200, 1.1)),
    _f("OCF_TO_NET_INCOME",   "Cash conversion (OCF / NI)", "Growth & Cash Flow", "ratio", "high", None, derived=True),
    # Income & ownership — payout, size, the Street
    _f("EQY_DVD_YLD_IND",     "Dividend yield",         "Income & Ownership", "pct", None, ("norm", 2.2, 1.5, 0, 8)),
    _f("DVD_PAYOUT_RATIO",    "Payout ratio",           "Income & Ownership", "pct", None, ("norm", 42, 24, 0, 115)),
    _f("EQY_SH_OUT",          "Shares out (mm)",        "Income & Ownership", "ratio", None, ("logn", 900, 1.0)),
    # CRNCY_ADJ_MKT_CAP (not CUR_MKT_CAP): millions, like the CF_*/NET_INCOME fundamentals —
    # CUR_MKT_CAP comes back in FULL currency units and breaks the shared mm formatting.
    _f("CRNCY_ADJ_MKT_CAP",   "Market cap",             "Income & Ownership", "mm",  None, ("logn", 60000, 1.2)),
    _f("BEST_ANALYST_RATING", "Consensus rating (1–5)", "Income & Ownership", "ratio", "high", ("norm", 3.8, 0.55, 1.5, 5)),
    _f("CRNCY",               "Currency",               "Income & Ownership", "text", None),
    _f("EXPECTED_REPORT_DT",  "Next expected report",   "Income & Ownership", "date", None),
]

# Plain-English "what is this and how do I read it" for every metric — the hover text on the
# screener's column headers, the company page's metric rows and the compare grid. Kept out of
# the _f() rows so the registry stays readable, and neutral in tone: what the number measures
# and what moves it, never what to do about it.
FIELD_HELP = {
    "PE_RATIO":
        "Share price ÷ the last 12 months' earnings per share — how many years of current "
        "profit the price represents. Lower looks cheaper, but a low multiple usually reflects "
        "doubt that those earnings persist.",
    "BEST_PE_RATIO":
        "Price ÷ the earnings analysts expect over the coming year. Forward-looking, so it "
        "moves whenever estimates are revised, not just when the price does.",
    "PX_TO_BOOK_RATIO":
        "Price ÷ book value (assets minus liabilities). Most meaningful for banks and insurers "
        "where book value tracks the real asset base; below 1.0x the market is valuing the "
        "company under its accounting net worth.",
    "PX_TO_SALES_RATIO":
        "Price ÷ revenue. Useful where earnings are negative or depressed — revenue can't be "
        "flattered by cost cuts or one-off charges.",
    "BEST_PEG_RATIO_CUR_YR":
        "Forward P/E ÷ expected earnings growth. Around 1.0 is the traditional 'growth fairly "
        "priced' line; below it the growth is cheap relative to the multiple being paid.",
    "EV_TO_T12M_EBITDA":
        "Enterprise value (market cap plus net debt) ÷ EBITDA — the takeover-style multiple. "
        "Because it counts debt, it compares a leveraged company with an unleveraged one fairly.",
    "GROSS_MARGIN":
        "Revenue left after the direct cost of producing the product, as a % of revenue — the "
        "cleanest read on pricing power.",
    "OPER_MARGIN":
        "Profit from operations as a % of revenue: after the costs of running the business, "
        "before interest and tax.",
    "PROF_MARGIN":
        "Bottom-line profit as a % of revenue, after everything including interest and tax.",
    "EBITDA_TO_REVENUE":
        "EBITDA as a % of revenue — margin before depreciation and amortisation, so "
        "capital-heavy and capital-light businesses compare more evenly.",
    "RETURN_COM_EQY":
        "Profit as a % of shareholders' equity — how hard the owners' capital works. Leverage "
        "flatters it, so read it alongside debt / equity.",
    "RETURN_ON_INV_CAPITAL":
        "Profit as a % of ALL capital employed, debt included — the leverage-neutral version of "
        "ROE. No Yahoo source, so it shows '—' on Yahoo pulls.",
    "TOT_DEBT_TO_TOT_EQY":
        "Total debt as a % of equity. Above 100% the company runs on more borrowed than owned "
        "capital; what counts as normal varies enormously by sector, hence the sector ranking.",
    "NET_DEBT_TO_EBITDA":
        "Years of EBITDA needed to repay net debt. Around 3x is where lenders typically start "
        "to tighten terms in most sectors; a negative figure means net cash.",
    "CUR_RATIO":
        "Current assets ÷ current liabilities — whether the next year's obligations are "
        "covered. Below 1.0 isn't automatically a strain for businesses paid up front.",
    "INTEREST_COVERAGE_RATIO":
        "Operating profit ÷ the interest bill — how many times over the interest is covered. "
        "No Yahoo source, so it shows '—' on Yahoo pulls.",
    "SALES_GROWTH":
        "Revenue against the same period a year earlier.",
    "SALES_3YR_AVG_GROWTH":
        "Average annual revenue growth over three years — smooths out one exceptional year. "
        "No Yahoo source, so it shows '—' on Yahoo pulls.",
    "EPS_GROWTH":
        "Earnings per share against a year earlier. Measured per share, so buybacks lift it "
        "even when total profit is flat.",
    "CF_CASH_FROM_OPER":
        "Cash the business itself generated, before investment and financing.",
    "CF_FREE_CASH_FLOW":
        "Operating cash flow after capital spending — the cash actually available for "
        "dividends, buybacks and paying down debt.",
    "FREE_CASH_FLOW_YIELD":
        "Free cash flow as a % of market cap — the cash-based answer to what the business "
        "throws off relative to its price, and harder to massage than earnings.",
    "NET_INCOME":
        "Bottom-line accounting profit for the period.",
    "OCF_TO_NET_INCOME":
        "Operating cash flow ÷ net income. At or above 1.0 reported profit is arriving as "
        "cash; persistently below it raises questions about earnings quality.",
    "EQY_DVD_YLD_IND":
        "Indicated annual dividend as a % of the share price.",
    "DVD_PAYOUT_RATIO":
        "The share of earnings paid out as dividends. Above roughly 80% there is little "
        "headroom if profits dip.",
    "EQY_SH_OUT":
        "Shares in issue, in millions. A falling count means buybacks; a rising one means "
        "issuance diluting existing holders.",
    "CRNCY_ADJ_MKT_CAP":
        "Market value of all shares in issue, in millions of the reporting currency.",
    "BEST_ANALYST_RATING":
        "Consensus analyst rating on the Bloomberg convention — 5 = most positive, 1 = least. "
        "Careful: the Analyst view block quotes Yahoo's consensus mean, which runs the OTHER "
        "way (1 = most positive).",
    "CRNCY":
        "The currency the fundamentals are reported in. LSE lines trade in pence but report "
        "in pounds.",
    "EXPECTED_REPORT_DT":
        "The next expected reporting date, from the fundamentals pull.",
}


SPEC = {f["field"]: f for f in FIELDS}


def help_for(field: str) -> str:
    """Hover text for a metric — falls back to the label when no description is written."""
    return FIELD_HELP.get(field) or SPEC.get(field, {}).get("label", field)
NUMERIC_FIELDS = [f["field"] for f in FIELDS if f["kind"] not in ("text", "date")]
_PULL_FIELDS = [f["field"] for f in FIELDS if not f["derived"]]

# The screener's default column set — a readable cross-section, one or two per group.
SCREENER_DEFAULT = ["BEST_PE_RATIO", "EV_TO_T12M_EBITDA", "PX_TO_BOOK_RATIO", "RETURN_COM_EQY",
                    "OPER_MARGIN", "TOT_DEBT_TO_TOT_EQY", "SALES_GROWTH", "FREE_CASH_FLOW_YIELD",
                    "EQY_DVD_YLD_IND", "CRNCY_ADJ_MKT_CAP"]


# ── formatting / colouring helpers ────────────────────────────────────────────
def _fmt_mm(v: float) -> str:
    """Millions of local currency -> '4.2bn' / '640mm'."""
    return f"{v / 1000:,.1f}bn" if abs(v) >= 1000 else f"{v:,.0f}mm"


def fmt_value(field: str, v) -> str:
    spec = SPEC.get(field)
    if spec is None:
        return "—" if v is None or v != v else str(v)
    if spec["kind"] in ("text", "date"):
        s = "" if v is None else str(v).strip()
        if field == "CRNCY" and s.lower() in ("gbp", "gbx"):
            s = "GBP"          # LN stocks trade in pence but the fundamentals/mkt cap are £mm
        return s if s and s.lower() != "nan" else "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    if spec["kind"] == "x":
        return f"{v:,.1f}x"
    if spec["kind"] == "pct":
        return f"{v:,.1f}%"
    if spec["kind"] == "mm":
        return _fmt_mm(v)
    return f"{v:,.2f}"


def ordinal(p) -> str:
    """55 -> '55th', 23 -> '23rd' — for the sector-percentile columns."""
    n = int(round(float(p)))
    sfx = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sfx}"


def goodness(field: str, pctl) -> int:
    """+1 good / -1 poor / 0 neutral for a sector percentile, honouring the field's
    'better' direction. Only the tails (>=80th / <=20th) are flagged."""
    spec = SPEC.get(field)
    if spec is None or spec["better"] is None or pctl is None or pctl != pctl:
        return 0
    hi = pctl >= 80
    lo = pctl <= 20
    if not (hi or lo):
        return 0
    good_end = hi if spec["better"] == "high" else lo
    return 1 if good_end else -1


# ── mock generator (deterministic per ticker+field, like equities._mock_*) ────
_CCY_BY_EXCH = {"US": "USD", "UW": "USD", "UN": "USD", "LN": "GBP"}   # everything else in the book: EUR


def _mock_ccy(ticker: str) -> str:
    parts = ticker.split()
    return _CCY_BY_EXCH.get(parts[1] if len(parts) > 1 else "", "EUR")


def _mock_value(ticker: str, spec: dict) -> float:
    rng = np.random.default_rng(_seed(f"{ticker}|{spec['field']}"))
    m = spec["mock"]
    if m[0] == "logn":
        return float(m[1] * np.exp(rng.normal(0.0, m[2])))
    v = float(rng.normal(m[1], m[2]))
    return min(max(v, m[3]), m[4])


def _mock_long(tickers, asof: str) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    rows = []
    for t in tickers:
        for spec in FIELDS:
            if spec["derived"]:
                continue
            if spec["field"] == "CRNCY":
                rows.append({"as_of": asof, "ticker": t, "field": "CRNCY",
                             "value": np.nan, "text": _mock_ccy(t)})
            elif spec["field"] == "EXPECTED_REPORT_DT":
                dt = today + pd.Timedelta(days=int(_seed(t) % 85) + 3)
                rows.append({"as_of": asof, "ticker": t, "field": "EXPECTED_REPORT_DT",
                             "value": np.nan, "text": dt.strftime("%Y-%m-%d")})
            else:
                rows.append({"as_of": asof, "ticker": t, "field": spec["field"],
                             "value": _mock_value(t, spec), "text": None})
    return pd.DataFrame(rows)


# ── Bloomberg (live via xbbg, through datafeed's normalisers) ─────────────────
def _bloomberg_long(tickers, asof: str) -> pd.DataFrame:
    from . import bbg as blp
    info = {}
    flds = [f.lower() for f in _PULL_FIELDS]
    for i in range(0, len(tickers), _BDP_CHUNK):
        chunk = list(tickers[i:i + _BDP_CHUNK])
        info.update(datafeed._bdp_rows(datafeed._coerce_pd(blp.bdp(chunk, flds))))
    rows = []
    for t in tickers:
        got = info.get(t, {})
        for spec in FIELDS:
            if spec["derived"]:
                continue
            raw = got.get(spec["field"])
            if spec["kind"] in ("text", "date"):
                s = "" if raw is None else str(raw).strip()
                if spec["kind"] == "date":
                    s = s[:10]
                rows.append({"as_of": asof, "ticker": t, "field": spec["field"],
                             "value": np.nan, "text": s if s and s.lower() != "nan" else None})
            else:
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    v = float("nan")
                rows.append({"as_of": asof, "ticker": t, "field": spec["field"],
                             "value": v, "text": None})
    return pd.DataFrame(rows)


# ── database I/O (append-only long parquet + manifest) ────────────────────────
def _read_db() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(_DB_FILE) if _DB_FILE.exists() else None
    except Exception:
        return None


def _read_manifest() -> dict:
    try:
        return json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append_db(long_df: pd.DataFrame, asof: str, source: str) -> dict:
    """Append a pull to the DB (idempotent per ticker+date: a same-day re-pull replaces that
    date's rows FOR THE TICKERS PULLED, never duplicates them — and never touches other
    tickers pulled the same day, so a Russell-only top-up can't wipe that morning's core
    pull). Never wipes history."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = _read_db()
    if db is not None and not db.empty:
        pulled = set(long_df["ticker"])
        db = db[~((db["as_of"] == asof) & (db["ticker"].isin(pulled)))]
        long_df = pd.concat([db, long_df], ignore_index=True)
    long_df.to_parquet(_DB_FILE, index=False)
    manifest = {
        "last_pull": asof,
        "source": source,
        "n_tickers": int(long_df[long_df["as_of"] == asof]["ticker"].nunique()),
        "n_fields": len(_PULL_FIELDS),
        "n_dates": int(long_df["as_of"].nunique()),
    }
    _MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _universe_tickers() -> list:
    return sorted({c["ticker"] for rows in equities.load_universe().values() for c in rows})


def refresh(tickers=None) -> dict:
    """Pull fundamentals via the CURRENT mode and append to the DB. Live on the work PC
    (DATAFEED_MODE=bloomberg); in mock mode it writes deterministic synthetic rows — an
    offline test of the whole pipeline, same as equities.build_snapshot(). Returns a status
    dict; never wipes the DB on a dead pull."""
    tickers = list(tickers) if tickers else _universe_tickers()
    asof = pd.Timestamp.today().strftime("%Y-%m-%d")
    long_df, source = pd.DataFrame(), ""
    if equities._use_yf():                   # free Yahoo pull first (no Terminal, no hit limit)
        try:
            long_df = yfin.fundamentals_long(tickers, asof, _PULL_FIELDS)
            source = "yfinance"
        except Exception:
            long_df = pd.DataFrame()
    if long_df.empty or int(long_df["value"].notna().sum()) == 0:
        if datafeed.MODE == "bloomberg":
            long_df = _bloomberg_long(tickers, asof)
            source = "bloomberg"
        else:
            long_df = _mock_long(tickers, asof)
            source = datafeed.MODE
    n_vals = int(long_df["value"].notna().sum())
    if long_df.empty or n_vals == 0:
        return {"ok": False, "reason": "no fundamentals returned (Terminal/API down?)"}
    manifest = _append_db(long_df, asof, source)
    return {"ok": True, "n_values": n_vals, **manifest}


def _age_days(stamp: str) -> int:
    try:
        return int((pd.Timestamp.today().normalize() - pd.Timestamp(stamp)).days)
    except (TypeError, ValueError):
        return 10 ** 6                       # never pulled -> due now


def maybe_refresh(max_age_days: int = 7, heavy_age_days: int = 30, tickers=None) -> dict:
    """The pull guard — fundamentals move slowly, so only re-pull what's stale. TIERED for
    Bloomberg's daily data capacity: the core indices refresh weekly, but the HEAVY indices
    (Russell 2000 — ~2000 names x ~30 fields ≈ 70k hits per pull) only every `heavy_age_days`.
    An explicit `tickers` list bypasses the tiering (pull exactly those, guarded weekly)."""
    m0 = _read_manifest()
    if tickers is not None:                  # explicit list: old single-cycle behaviour
        if _age_days(m0.get("last_pull", "")) < max_age_days:
            return {"ok": True, "skipped": True, "last_pull": m0.get("last_pull"),
                    "age_days": _age_days(m0.get("last_pull", ""))}
        return refresh(tickers)

    uni = equities.load_universe()
    heavy_keys = getattr(equities, "HEAVY_INDICES", set())
    core = {c["ticker"] for k, rows in uni.items() if k not in heavy_keys for c in rows}
    heavy = {c["ticker"] for k, rows in uni.items() if k in heavy_keys for c in rows} - core

    core_due = _age_days(m0.get("last_pull", "")) >= max_age_days
    heavy_due = bool(heavy) and _age_days(m0.get("last_heavy_pull", "")) >= heavy_age_days
    if not core_due and not heavy_due:
        return {"ok": True, "skipped": True, "last_pull": m0.get("last_pull"),
                "age_days": _age_days(m0.get("last_pull", "")),
                "last_heavy_pull": m0.get("last_heavy_pull", "")}

    to_pull = sorted((core if core_due else set()) | (heavy if heavy_due else set()))
    res = refresh(to_pull)
    if res.get("ok"):
        # refresh()->_append_db rewrote the manifest — re-stamp the per-tier pull dates it
        # doesn't know about (core-only pulls must not look like a fresh heavy pull and
        # vice versa).
        m = _read_manifest()
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        m["last_heavy_pull"] = today if heavy_due else m0.get("last_heavy_pull", "")
        if not core_due:                     # heavy-only pull: core stamp stays put
            m["last_pull"] = m0.get("last_pull", today)
        _MANIFEST_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        res.update({"pulled_core": core_due, "pulled_heavy": heavy_due})
    return res


def staleness() -> dict:
    """Pull-age summary for the UI notes: when each tier was last pulled and how old it is."""
    m = _read_manifest()
    return {"last_pull": m.get("last_pull", ""), "core_age": _age_days(m.get("last_pull", "")),
            "last_heavy_pull": m.get("last_heavy_pull", ""),
            "heavy_age": _age_days(m.get("last_heavy_pull", ""))}


# ── read seams ────────────────────────────────────────────────────────────────
def _pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    num = long_df[long_df["value"].notna()]
    wide = (num.pivot_table(index="ticker", columns="field", values="value", aggfunc="last")
            if not num.empty else pd.DataFrame())
    txt = long_df[long_df["text"].notna()]
    if not txt.empty:
        tw = txt.pivot_table(index="ticker", columns="field", values="text", aggfunc="last")
        wide = wide.join(tw, how="outer") if not wide.empty else tw
    return wide


def _add_derived(wide: pd.DataFrame) -> pd.DataFrame:
    if {"CF_CASH_FROM_OPER", "NET_INCOME"}.issubset(wide.columns):
        ni = pd.to_numeric(wide["NET_INCOME"], errors="coerce")
        ocf = pd.to_numeric(wide["CF_CASH_FROM_OPER"], errors="coerce")
        wide["OCF_TO_NET_INCOME"] = (ocf / ni).where(ni > 0)   # meaningless on a loss-maker
    return wide


def latest_wide(tickers=None):
    """(DataFrame ticker x FIELD — each ticker's NEWEST stored values, asof, source). Falls
    back to on-the-fly mock values (not written to the DB) when no pull exists yet.
    Per-ticker latest (not latest-date-only): core indices refresh weekly but heavy ones
    (Russell 2000) monthly to respect Bloomberg's daily capacity, so the freshest rows for
    different names legitimately live on different pull dates."""
    db = _read_db()
    if db is not None and not db.empty:
        asof = str(db["as_of"].max())
        wide = _pivot(db.sort_values("as_of"))     # aggfunc="last" -> newest row per ticker/field
        source = _read_manifest().get("source", "?")
    else:
        tk = list(tickers) if tickers else _universe_tickers()
        asof = pd.Timestamp.today().strftime("%Y-%m-%d")
        wide = _pivot(_mock_long(tk, asof))
        source = "demo"
    wide = _add_derived(wide)
    if tickers is not None:
        wide = wide.reindex(list(tickers))
    return wide, asof, source


def field_history(field: str) -> pd.DataFrame:
    """DataFrame as_of x ticker of one numeric field across every stored pull — the trend seam
    (share-count drift, margin trajectory, valuation vs own range). Empty until pulls accumulate."""
    db = _read_db()
    if db is None or db.empty:
        return pd.DataFrame()
    d = db[(db["field"] == field) & db["value"].notna()]
    if d.empty:
        return pd.DataFrame()
    out = d.pivot_table(index="as_of", columns="ticker", values="value", aggfunc="last")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def company_frame(universe=None, index_keys=None):
    """One row per COMPANY across the selected indices: name / sector / region / joined index
    list + every fundamentals field. Live Bloomberg membership returns the same stock under
    DIFFERENT exchange codes in different indices (e.g. NVDA UW in the S&P vs another code in
    the Dow) and cross-listings under different tickers (Stellantis STLAM IM / STLAP FP), so
    rows are merged by short name — the same rule as the Home movers table. Share classes keep
    separate rows (Alphabet A/C have distinct short names). Returns (frame, asof, source)."""
    uni = universe if universe is not None else equities.load_universe()
    keys = [k for k in (index_keys or uni.keys()) if k in uni]
    order = {k: i for i, k in enumerate(equities.INDICES)}
    meta: dict = {}
    for k in keys:
        for c in uni.get(k, []):
            m = meta.setdefault(c["ticker"], {
                "ticker": c["ticker"], "name": c.get("name", c["ticker"]),
                "sector": c.get("sector", "—"),
                "region": c.get("region", equities.REGION_OF.get(k, "")), "indices": []})
            if k not in m["indices"]:
                m["indices"].append(k)
    if not meta:
        return pd.DataFrame(), "", ""
    frame = pd.DataFrame(list(meta.values())).set_index("ticker")
    wide, asof, source = latest_wide(list(frame.index))
    joined = frame.join(wide)
    rows = []
    for name, grp in joined.groupby("name", sort=False):
        r = {"ticker": grp.index[0], "name": name, "sector": grp["sector"].iloc[0],
             "region": grp["region"].iloc[0],
             "indices": ", ".join(sorted(dict.fromkeys(k for lst in grp["indices"] for k in lst),
                                         key=lambda k: order.get(k, 99)))}
        for c in wide.columns:                    # first non-null across the merged listings
            v = grp[c].dropna()
            r[c] = v.iloc[0] if len(v) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("name")
    return out, asof, source


def _frame_store_paths(index_keys):
    import hashlib
    h = hashlib.md5("|".join(sorted(index_keys)).encode()).hexdigest()[:10]
    base = _DATA_DIR / f"fund_frame_{h}"
    return base.with_suffix(".parquet"), base.with_suffix(".key.json")


def _frame_store_key(index_keys) -> dict:
    def _mt(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    from src import equities as _eq
    return {"db": _mt(_DB_FILE), "manifest": _mt(_eq._MANIFEST_FILE),
            "constituents": _mt(_eq._CONSTITUENTS_FILE), "keys": sorted(index_keys)}


def company_frame_cached(universe=None, index_keys=None):
    """company_frame() + sector percentiles served from a DISK store (app-wide once-a-day
    rule, 2026-08-22): the ~2.5s pivot + per-company merge across ~700 names ran on every
    page open after a server restart. The store is keyed on the fundamentals DB, the
    membership files and the index set; the daily equities pull warms it for the default
    indices, any other selection computes once and writes its own. Returns the same
    (frame, asof, source) triple."""
    import json
    keys = sorted(index_keys or [])
    fpath, kpath = _frame_store_paths(keys)
    want = _frame_store_key(keys)
    try:
        have = json.loads(kpath.read_text(encoding="utf-8"))
        if have.get("key") == want and fpath.exists():
            return pd.read_parquet(fpath), have.get("asof", ""), have.get("source", "")
    except Exception:
        pass
    df, asof, source = company_frame(universe=universe, index_keys=keys)
    out = add_sector_percentiles(df) if not df.empty else df
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(fpath, index=False)
        kpath.write_text(json.dumps({"key": want, "asof": asof, "source": source}),
                         encoding="utf-8")
    except Exception:
        pass
    return out, asof, source


def add_sector_percentiles(df: pd.DataFrame, fields=None) -> pd.DataFrame:
    """Adds '<FIELD>__pctl' columns: the value's percent rank (0–100) WITHIN its GICS sector —
    the like-for-like frame (a bank's P/B vs banks, not vs software). Sectors with fewer than
    3 valued names get NaN (a percentile of two is noise)."""
    out = df.copy()
    grp = out["sector"].fillna("—")
    for f in (fields or NUMERIC_FIELDS):
        if f not in out.columns:
            continue
        s = pd.to_numeric(out[f], errors="coerce")
        pct = s.groupby(grp).rank(pct=True) * 100
        n = s.notna().groupby(grp).transform("sum")
        out[f + "__pctl"] = pct.where(n >= 3)
    return out


def peer_set(df: pd.DataFrame, ticker: str, n: int = 5) -> list:
    """The `n` closest comparables to `ticker` in the frame: same GICS sector, ranked by how
    near their market cap is on a LOG scale (a 60bn name compares to a 40bn one, not to the
    600bn megacap that happens to sit next to it in absolute terms). Falls back to the same
    index when the sector has too few valued names. Returns tickers, nearest first."""
    if df.empty or "ticker" not in df.columns or ticker not in set(df["ticker"]):
        return []
    me = df[df["ticker"] == ticker].iloc[0]
    cap_col = "CRNCY_ADJ_MKT_CAP"
    pool = df[(df["sector"] == me.get("sector")) & (df["ticker"] != ticker)]
    if len(pool) < 2 and "indices" in df.columns:      # thin sector — widen to index-mates
        pool = df[(df["indices"] == me.get("indices")) & (df["ticker"] != ticker)]
    if pool.empty:
        return []
    my_cap = pd.to_numeric(pd.Series([me.get(cap_col)]), errors="coerce").iloc[0] \
        if cap_col in df.columns else np.nan
    if my_cap != my_cap or my_cap <= 0 or cap_col not in pool.columns:
        return list(pool["ticker"].head(n))            # no cap to rank on — first names in sector
    caps = pd.to_numeric(pool[cap_col], errors="coerce")
    dist = (np.log(caps.where(caps > 0)) - np.log(my_cap)).abs()
    return list(pool.assign(_d=dist).sort_values("_d", na_position="last")["ticker"].head(n))


# ── saved screens (the screener's named filter sets) ─────────────────────────
_SCREENS_FILE = _DATA_DIR / "screens.json"


def load_screens() -> dict:
    """{name: {preset, sectors, cols, ranges}} — the user's saved screener setups."""
    try:
        d = json.loads(_SCREENS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_screen(name: str, spec: dict) -> dict:
    name = str(name).strip()
    if not name:
        return load_screens()
    screens = load_screens()
    screens[name] = spec
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SCREENS_FILE.write_text(json.dumps(screens, ensure_ascii=False, indent=1), encoding="utf-8")
    return screens


def delete_screen(name: str) -> dict:
    screens = load_screens()
    if screens.pop(str(name), None) is not None:
        _SCREENS_FILE.write_text(json.dumps(screens, ensure_ascii=False, indent=1), encoding="utf-8")
    return screens


def data_status() -> str:
    """Short 'where the fundamentals come from' label for page captions."""
    m = _read_manifest()
    if m.get("last_pull"):
        return (f"{m.get('source', '?')} pull · {m['last_pull']} · "
                f"{m.get('n_tickers', 0)} names · {m.get('n_dates', 1)} stored date(s)")
    return "no pull yet — demo (synthetic) values"
