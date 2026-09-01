"""Bloomberg Codes database — symbol ⇄ product/expiry, both directions.

The desk quotes Bloomberg symbols all day and there was nowhere in BASIS to answer either
half of the obvious question (Ben, 2026-08-28): *"what is `CLZ6C 80 Comdty`?"* and *"which
code is the Nasdaq option expiring on this date?"*.

WHAT THIS RUNS ON — and what it deliberately does NOT.
Nothing here pulls from Bloomberg. Every symbol is derived from data already on disk:

  • data/snapshot/opt_chain_cache.json — 288 REAL option series stems harvested from live
    OPT_CHAIN (asof 2026-08-27). This is the authoritative half: an observed stem is proof
    the series exists and proof of its exact string form (strike spacing, the trailing
    space in `C ` for corn, Henry Hub's two-digit futures year).
  • src/universe.py — 89 products: ticker → name, asset class, region.
  • src/expiries.py — 50 contract families of holiday-aware expiry rules.
  • data/price_store/deep_contract.parquet — a decade of contracts actually observed as front.

COVERAGE, STATED HONESTLY. The observed cache holds quarterlies and SERIAL MONTHLIES only —
every stem in it is root+month+year, none carry a weekly/daily root. `expiries.py` says the
same on its own face ("weeklies/dailies ignored"). So weekly, daily and midcurve series are
NOT in this database, and this module reports them as unharvested rather than guessing: their
Bloomberg roots follow no rule that can be derived from the futures root, and only OPT_CHAIN
knows them. `harvest.json` below is the slot they land in if that pull is ever made.

PROVENANCE IS PART OF THE ANSWER. Every code and date carries a source label — "observed"
(seen in a live chain / harvested), or "rule" (derived from expiries.py, indicative). The
page shows it. A rule-derived expiry is a good estimate of the standard cycle, not the
exchange's calendar, and must never be handed to a client as though it were.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from . import expiries, universe as u

ROOT = Path(__file__).resolve().parents[1]
CHAIN_CACHE = ROOT / "data" / "snapshot" / "opt_chain_cache.json"
STORE_DIR = ROOT / "data" / "bbgcodes"
HARVEST_FILE = STORE_DIR / "harvest.json"       # weekly/daily series, if ever harvested

MONTH_CODE = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
              7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_MONTH = {v: k for k, v in MONTH_CODE.items()}
MONTH_NAME = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

YELLOW_KEYS = ("Comdty", "Index", "Curncy", "Equity", "Corp", "Govt", "Curncy")

# Products whose universe key is a CASH index — no futures root of their own.
CASH = set(expiries.CASH_TICKERS)


# ── roots ─────────────────────────────────────────────────────────────────────────────
def _strip_key(sym: str) -> tuple[str, str]:
    """'CLZ6 Comdty' -> ('CLZ6', 'Comdty'). Unkeyed input keeps an empty suffix."""
    s = re.sub(r"\s+", " ", str(sym)).strip()
    for k in YELLOW_KEYS:
        if s.endswith(" " + k):
            return s[: -(len(k) + 1)].rstrip() or s[: -(len(k) + 1)], k
    return s, ""


def _root_from_generic(ticker: str) -> str | None:
    """Futures ROOT behind a universe generic: 'CLA Comdty'->'CL', 'C A Comdty'->'C ',
    'OD1 Comdty'->'OD'. The trailing 'A' is Bloomberg's front-generic marker, not part of
    the root — and corn/wheat/gilt/FTSE roots keep the space the marker sat on."""
    if ticker in CASH:
        return None
    body, _ = _strip_key(ticker)
    if not body:
        return None
    if body[-1] == "A" and len(body) > 1:
        return body[:-1]                        # 'CLA'->'CL', 'C A'->'C ', 'UXYA'->'UXY'
    return re.sub(r"\d+$", "", body) or None    # numbered generic, e.g. 'OD1'->'OD'


def _build_root_map() -> dict:
    """(root, yellow key) -> universe ticker.

    The key is HALF THE IDENTITY, not decoration. Two roots in this universe are genuinely
    ambiguous without it — `SM` is the SMI (Index) or Soybean Meal (Comdty), `CC` is the
    Czech Koruna (Curncy) or Cocoa (Comdty) — and naming the wrong product is the worst
    failure this tool has. Keying on the pair keeps both, and `roots_named()` below hands
    the caller every candidate when the key is missing rather than picking one."""
    out = {}
    for tk in u.INSTRUMENTS:
        r = _root_from_generic(tk)
        if r:
            out.setdefault((r, _strip_key(tk)[1]), tk)
    return out


def roots_named(root: str, key: str = "") -> list:
    """Universe tickers matching a root, narrowed by yellow key when one is given."""
    return [tk for (r, k), tk in _build_root_map().items()
            if r == root and (not key or k == key)]


# ── observed series (the authoritative half) ──────────────────────────────────────────
def _stem_parts(stem: str):
    """'CLU6'->('CL','U',6,1); 'NGU26'->('NG','U',26,2); 'C U6'->('C ','U',6,1).
    Year digits are counted, not assumed — Henry Hub lists futures years out and uses a
    two-digit year where its options stay one-digit."""
    if len(stem) >= 4 and stem[-2:].isdigit() and stem[-3] in CODE_MONTH:
        return stem[:-3], stem[-3], int(stem[-2:]), 2
    if len(stem) >= 3 and stem[-1].isdigit() and stem[-2] in CODE_MONTH:
        return stem[:-2], stem[-2], int(stem[-1]), 1
    return None


_observed_cache: dict | None = None


def observed(refresh: bool = False) -> dict:
    """Everything the live-chain capture proves, keyed by futures root:

        {root: {"suffix": 'Comdty', "asof": '2026-08-27',
                "opt_series": {(year, month), ...},     # option series months seen
                "fut_contracts": {(year, month), ...},  # dated futures seen
                "opt_yd": 1|2, "fut_yd": 1|2,           # year digits in the ticker string
                "strike_fmt": {'100', '4500.5', ...}}}  # exact strike strings seen

    Empty (not an error) when the cache is missing — the page then runs rules-only."""
    global _observed_cache
    if _observed_cache is not None and not refresh:
        return _observed_cache
    out: dict = {}
    try:
        raw = json.loads(CHAIN_CACHE.read_text(encoding="utf-8"))
    except Exception:
        _observed_cache = {}
        return _observed_cache
    for contract, blob in (raw or {}).items():
        body, key = _strip_key(contract)
        fp = _stem_parts(body)
        if not fp:
            continue
        root = fp[0]
        rec = out.setdefault(root, {"suffix": key or "Comdty", "asof": blob.get("asof"),
                                    "opt_series": set(), "fut_contracts": set(),
                                    "opt_yd": 1, "fut_yd": 1, "strike_fmt": set()})
        rec["fut_yd"] = max(rec["fut_yd"], fp[3])
        rec["fut_contracts"].add((_year(fp[2], fp[3]), CODE_MONTH[fp[1]]))
        for stem, strikes in (blob.get("series") or {}).items():
            sp = _stem_parts(stem)
            if not sp or sp[0] != root:
                continue
            rec["opt_yd"] = max(rec["opt_yd"], sp[3])
            rec["opt_series"].add((_year(sp[2], sp[3]), CODE_MONTH[sp[1]]))
            # The cache keys strikes by their FLOAT repr ('220.0'); the real Bloomberg
            # string lives inside the ticker it maps to ('C U6C  220 Comdty' -> '220').
            # Show the latter — a code built with '220.0' may simply not resolve, and the
            # whole point of this page is a symbol that can be pasted into the Terminal.
            if isinstance(strikes, dict):
                for tickers in strikes.values():
                    for one in (tickers if isinstance(tickers, list) else [tickers]):
                        tail = _strip_key(str(one))[0].split()
                        if len(tail) >= 2:
                            rec["strike_fmt"].add(tail[-1])
    _observed_cache = out
    return out


def _year(yy: int, digits: int, ref: int | None = None) -> int:
    """Resolve a ticker's year digits. Two digits are unambiguous. ONE digit is not —
    'Z6' is 2026 as readily as 2036 — so it resolves to the nearest listing decade,
    which is why `decode()` returns an explicit ambiguity flag for it."""
    if digits >= 2:
        return 2000 + (yy % 100)
    ref = ref or date.today().year
    y = (ref // 10) * 10 + yy
    if y < ref - 1:
        y += 10
    return y


OI_CHAIN = ROOT / "data" / "snapshot" / "oi_chain.parquet"
_expiry_cache: dict | None = None


def observed_expiries(refresh: bool = False) -> dict:
    """{universe ticker: {date, ...}} — option expiries OBSERVED on the Terminal.

    The weekly FI open-interest capture stores real `OPT_EXPIRE_DT` values, so for the 11
    fixed-income products we hold the exchange's own dates rather than our reconstruction
    of them. Cross-checking the two on 2026-08-28 caught two genuine bugs in the rule
    engine (govvie options ignoring CBOT/Eurex's ≥2-business-day rule, and Euribor
    quarterlies expiring with the future rather than the Friday before); after the fix all
    82 observed expiries reproduce exactly. Treat this file as the correctness anchor."""
    global _expiry_cache
    if _expiry_cache is not None and not refresh:
        return _expiry_cache
    out: dict = {}
    try:
        import pandas as pd
        df = pd.read_parquet(OI_CHAIN)
        for tk, g in df.groupby("ticker"):
            out[str(tk)] = {d.date() for d in pd.to_datetime(g["expiry"].unique())}
    except Exception:
        out = {}
    _expiry_cache = out
    return out


def verified(ticker: str, d) -> bool:
    """True when this exact option expiry was seen in a live chain — not merely derived."""
    return bool(d) and d in observed_expiries().get(ticker, set())


def harvested() -> dict:
    """Weekly/daily/midcurve series, if a harvest has ever been run. Absent by design
    until then — see the module docstring. Shape: {root: {"series": [...], "asof": ...}}."""
    try:
        return json.loads(HARVEST_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ── product metadata ──────────────────────────────────────────────────────────────────
def _record(tk: str, root: str) -> dict:
    name, _px, asset, region = u.INSTRUMENTS[tk]
    return {"ticker": tk, "root": root, "name": name, "asset": asset, "region": region,
            "key": _strip_key(tk)[1]}


def product_of(root: str, key: str = "") -> dict | None:
    """Product record behind a root, narrowed by yellow key. None if unknown; when the root
    is ambiguous and no key is given, the FIRST match is returned and `look_alikes()` tells
    the caller to warn — never silently resolve an ambiguous root as though it were certain."""
    hits = roots_named(root, key)
    return _record(hits[0], root) if hits else None


def products() -> list:
    """Every product with a usable futures root, grouped by asset class — the picker list."""
    rows = [_record(tk, root) for (root, _k), tk in _build_root_map().items()]
    return sorted(rows, key=lambda r: (r["asset"], r["name"]))


def look_alikes(root: str) -> list:
    """Every universe product sharing this root across yellow keys — returned only when
    there IS a clash. `SMA Index` (SMI) vs `SMA Comdty` (Soybean Meal) and `CCA Curncy`
    (Czech Koruna) vs `CCA Comdty` (Cocoa) are the two live in this universe."""
    out = [_record(tk, root) for tk in roots_named(root)]
    return out if len(out) > 1 else []


# ── WEEKLY series ─────────────────────────────────────────────────────────────────────
# Weekly options are NOT reachable through any option chain — OPT_CHAIN on the future
# returns only standard series, and CHAIN_TICKERS / OPT_CHAIN_FULL return nothing at all
# (probe_weekly_opts.py, 2026-09-01). They are only reachable if you already know the root,
# and the root is not derivable from the futures root. So each product's weekly scheme has
# to be LEARNED ONCE and recorded here; after that every weekly is pure arithmetic and no
# Bloomberg request is ever needed again.
#
# S&P confirmed by probe_weekly_pattern.py on 2026-09-01: week N carries root "<N>E", the
# ordinary calendar month code, and expires on the Nth Friday. Verified 10/10 across three
# months in two different decades (Sep-26 weeks 1-4; Dec-16 weeks 1,2,4; Jan-17 weeks 1,2,4)
# — and week 5 correctly does NOT exist in a month with only four Fridays.
#
# TO ADD A PRODUCT: read ONE weekly ticker off the Terminal (Bloomberg names them
# "... 1st Wee ..."), confirm the week-to-root mapping, and add an entry. Nothing else.
FRI = 4
WEEKLY_SPECS: dict = {
    "ESA Index": {
        "roots": {1: "1E", 2: "2E", 3: "3E", 4: "4E", 5: "5E"},
        "weekday": FRI, "label": "Friday weekly",
        "verified": "probe 2026-09-01, 10/10 across Sep-26 / Dec-16 / Jan-17",
    },
}

# Bloomberg lists only a few weeks of weeklies at a time. Beyond that horizon a constructed
# code does not merely fail — with a one-digit year it can RESOLVE TO A DECADE-OLD EXPIRED
# CONTRACT (asking for Z6 = Dec-2026 returned Dec-2016 during the probe). So generated
# weeklies are bounded, and anything past the bound is labelled unlisted rather than served.
WEEKLY_HORIZON_DAYS = 70


def weekly_root_map() -> dict:
    """(weekly root, yellow key) -> (product ticker, week number)."""
    out = {}
    for tk, spec in WEEKLY_SPECS.items():
        key = _strip_key(tk)[1]
        for wk, root in spec["roots"].items():
            out[(root, key)] = (tk, wk)
    return out


def has_weeklies(ticker: str) -> bool:
    """True when this product's weekly scheme is known. False means NOT YET LEARNED — never
    that the product has no weeklies."""
    return ticker in WEEKLY_SPECS


def nth_weekday_of(year: int, month: int, n: int, weekday: int):
    """The n-th <weekday> of a month, or None when the month doesn't have that many."""
    first = date(year, month, 1)
    d = first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    return d if d.month == month else None


def weekly_series(ticker: str, year: int, month: int) -> list:
    """Every listed weekly series for one product-month.

    Week N expires on the Nth <weekday> of the month, stepped back to the previous exchange
    day if that falls on a holiday (the same holiday calendars the rest of the expiry engine
    uses). A month with four Fridays simply has no week 5 — the absence is the rule working,
    not a gap."""
    spec = WEEKLY_SPECS.get(ticker)
    if not spec:
        return []
    asset = u.INSTRUMENTS.get(ticker, ("", 0, "", ""))[2]
    hol = expiries._holidays_for(ticker, asset)
    key = _strip_key(ticker)[1]
    out = []
    for wk, root in sorted(spec["roots"].items()):
        raw = nth_weekday_of(year, month, wk, spec["weekday"])
        if raw is None:
            continue
        exp = expiries._on_or_before_bday(raw, hol)
        out.append({
            "week": wk, "root": root,
            "stem": f"{root}{MONTH_CODE[month]}{year % 10}",
            "code": f"{root}{MONTH_CODE[month]}{year % 10}C <strike> {key}",
            "expiry": exp, "moved": exp != raw,
            "contract": f"{MONTH_NAME[month]} {year}",
            "label": f"week {wk} ({spec['label']})",
        })
    return out


def weeklies_between(ticker: str, start: date, end: date) -> list:
    """Weekly series expiring in a date window, across month boundaries."""
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        for rec in weekly_series(ticker, y, m):
            if start <= rec["expiry"] <= end:
                out.append(rec)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(out, key=lambda r: r["expiry"])


def strike_hint(root: str, n: int = 6) -> dict:
    """How this product's strikes are WRITTEN, and how far apart they sit.

    Examples come from the middle of the observed ladder, not the bottom: the deep tails of
    an equity chain ('100', '200') tell you nothing about the format you'd actually type.
    Note the quirks this exposes — Henry Hub lists 0.25 as `.25`, corn as a bare integer —
    which is exactly why the strings are shown rather than a number we format ourselves."""
    fmts = (observed().get(root) or {}).get("strike_fmt") or set()

    def _num(s):
        try:
            return float(s)
        except ValueError:
            return None

    pairs = sorted(((_num(s), s) for s in fmts if _num(s) is not None))
    if not pairs:
        return {"step": None, "examples": []}
    diffs = [round(b[0] - a[0], 8) for a, b in zip(pairs, pairs[1:]) if b[0] > a[0]]
    mid = len(pairs) // 2
    lo = max(0, mid - n // 2)
    return {"step": min(diffs) if diffs else None,
            "examples": [s for _v, s in pairs[lo:lo + n]]}


def _decode_weekly(root: str, code: str, yy: int, digits: int, key: str,
                   ticker: str, week: int) -> dict:
    """Decode a weekly series stem, e.g. '2EU6' -> S&P week 2 of Sep 2026."""
    month, year = CODE_MONTH[code], _year(yy, digits)
    prod = _record(ticker, root)
    spec = WEEKLY_SPECS[ticker]
    series = [w for w in weekly_series(ticker, year, month) if w["week"] == week]
    exp = series[0]["expiry"] if series else None
    horizon = date.today() + timedelta(days=WEEKLY_HORIZON_DAYS)
    out = {"ok": True, "root": root, "product": prod, "month_code": code, "month": month,
           "year": year, "contract": f"{MONTH_NAME[month]} {year}",
           "fut_expiry": None, "opt_expiry": exp,
           "expiry_time": expiries.expiry_time(ticker, prod["asset"]),
           "source": "rule", "expiry_verified": False, "weekly": True, "week": week,
           "week_label": f"week {week} ({spec['label']})",
           "underlying": underlying_future(ticker, _root_from_generic(ticker) or "",
                                           month, year),
           "serial": False, "listed_observed": False, "warnings": []}
    if exp is None:
        out["warnings"].append(
            f"{MONTH_NAME[month]} {year} has no {week}th "
            f"{'Friday' if spec['weekday'] == FRI else 'weekday'}, so this series should "
            "not exist. If the Terminal resolves it, the week numbering differs from what "
            "we recorded — tell me and I'll re-probe.")
    elif exp > horizon:
        out["warnings"].append(
            "This is BEYOND the listed horizon — Bloomberg lists only a few weeks of "
            "weeklies at a time. The date below is arithmetic, and the code is not yet "
            "tradeable. Worse, with a one-digit year the Terminal may resolve it to a "
            "decade-old expired contract instead of refusing it.")
    if series and series[0]["moved"]:
        out["warnings"].append(
            "The nominal expiry fell on an exchange holiday, so it steps back to the "
            "previous business day.")
    return out


def underlying_future(ticker: str, root: str, month: int, year: int) -> dict | None:
    """The futures contract a given option month is written on.

    Serial options do NOT have a future of their own month — a November Treasury option
    trades on the DECEMBER future — so the underlying is the first listed delivery month
    on or after the option's, rolling into next year when the option sits past the last
    quarterly. Getting this wrong is how someone hedges the wrong contract, so the page
    states it explicitly whenever option month and future month differ."""
    asset = (product_of(root, _strip_key(ticker)[1]) or {}).get("asset", "")
    cycle = expiries.listed_months(ticker, asset, "fut")
    if not cycle:
        return None
    m = next((c for c in cycle if c >= month), None)
    y = year if m else year + 1
    m = m or min(cycle)
    obs = observed().get(root, {})
    yd = obs.get("fut_yd") or 1
    suffix = _strip_key(ticker)[1] or obs.get("suffix") or "Comdty"
    return {"month": m, "year": y, "contract": f"{MONTH_NAME[m]} {y}",
            "code": f"{root}{MONTH_CODE[m]}{y % 100 if yd == 2 else y % 10} {suffix}",
            "expiry": expiries.expiry_for(ticker, asset, y, m, "fut")}


# ── DECODE: symbol in → product + expiry out ──────────────────────────────────────────
_OPT_RE = re.compile(r"^(?P<stem>.+?)(?P<pc>[CP])\s+(?P<strike>[.\d]+)$")


def decode(symbol: str) -> dict:
    """Parse any Bloomberg futures/option symbol the desk might paste.

    Handles the outright (`CLZ6 Comdty`), the option (`CLZ6C 80 Comdty`, `ESU6P 4500 Index`),
    and the generic (`CLA Comdty`, `CL1 Comdty`). Returns a dict with `ok`, a component
    breakdown, the product, both expiries, provenance and any ambiguity warnings. Never
    raises — an unparseable string comes back with `ok=False` and a reason."""
    raw = re.sub(r"\s+", " ", str(symbol or "")).strip()
    out = {"input": raw, "ok": False, "reason": None, "kind": None, "warnings": [],
           "source": None}
    if not raw:
        out["reason"] = "empty"
        return out

    body, key = _strip_key(raw)
    out["yellow_key"] = key or None
    if not key:
        out["warnings"].append(
            "No yellow key given — a bare root is ambiguous on the Terminal "
            "(e.g. SMA Index is the SMI, SMA Comdty is Soybean Meal).")

    # option first: the C/P sits immediately before the strike
    m = _OPT_RE.match(body)
    if m:
        stem, pc, strike = m.group("stem").rstrip(), m.group("pc"), m.group("strike")
        parsed = _decode_stem(stem, key)
        if parsed["ok"]:
            out.update(parsed)
            out["kind"] = "option"
            out["put_call"] = "Call" if pc == "C" else "Put"
            out["strike"] = strike
            out["expiry"] = parsed.get("opt_expiry")
            out["expiry_kind"] = "Option expiry"
            return out

    parsed = _decode_stem(body, key)
    if parsed["ok"]:
        out.update(parsed)
        out["kind"] = "future"
        out["expiry"] = parsed.get("fut_expiry")
        out["expiry_kind"] = "Futures last trade"
        return out

    # a generic (front-month) ticker names a product but no contract month
    root = _root_from_generic(raw) or _root_from_generic(body + " " + (key or "Comdty"))
    prod = product_of(root, key) if root else None
    if prod:
        out.update({"ok": True, "kind": "generic", "product": prod, "root": root,
                    "source": "universe"})
        out["warnings"].append(
            "This is a GENERIC (front-month) ticker — it rolls, and names no single "
            "contract. Use the Build tab for a dated code.")
        return out

    out["reason"] = "not a recognised root + month code + year"
    return out


def _decode_stem(stem: str, key: str) -> dict:
    """Decode `root + month code + year` and attach product + both expiries."""
    parts = _stem_parts(stem)
    if not parts:
        return {"ok": False}
    root, code, yy, digits = parts

    # A WEEKLY root ('2E') is not a futures root and resolves through its own map.
    wk_hit = weekly_root_map().get((root, key)) or (
        weekly_root_map().get((root, "Index")) if not key else None)
    if wk_hit:
        return _decode_weekly(root, code, yy, digits, key, *wk_hit)

    prod = product_of(root, key)
    if not prod:
        return {"ok": False}
    month = CODE_MONTH[code]
    year = _year(yy, digits)
    obs = observed().get(root, {})
    seen = (year, month) in (obs.get("opt_series") or set()) \
        or (year, month) in (obs.get("fut_contracts") or set())

    asset = prod["asset"]
    fx = expiries.expiry_for(prod["ticker"], asset, year, month, "fut")
    ox = expiries.expiry_for(prod["ticker"], asset, year, month, "opt")
    exp_verified = verified(prod["ticker"], ox)
    und = underlying_future(prod["ticker"], root, month, year)
    out = {"ok": True, "root": root, "product": prod, "month_code": code,
           "month": month, "year": year,
           "contract": f"{MONTH_NAME[month]} {year}",
           "fut_expiry": fx, "opt_expiry": ox,
           "expiry_time": expiries.expiry_time(prod["ticker"], asset),
           "source": "observed" if (seen or exp_verified) else "rule",
           "expiry_verified": exp_verified, "underlying": und, "weekly": False,
           "serial": bool(und and (und["month"], und["year"]) != (month, year)),
           "listed_observed": seen, "warnings": []}
    # The one-digit year is only worth flagging when the decade is genuinely contestable.
    # Every desk reads CLZ6 as 2026, and warning on all of them trains the reader to skip
    # warnings — so it fires only where the contract is beyond the observed listing horizon.
    if digits == 1 and not seen and year > date.today().year + 2:
        out["warnings"].append(
            f"One-digit year '{yy}' is ambiguous — read here as {year}, but "
            f"{year + 10} uses the same code, and this is far enough out that both are "
            f"plausible. Confirm on the Terminal.")
    others = [a for a in look_alikes(root) if a["ticker"] != prod["ticker"]]
    if others and not key:
        out["warnings"].append(
            "Root '%s' is AMBIGUOUS without a yellow key — read here as %s, but it is "
            "also %s. Re-enter with the key to be certain."
            % (root.strip(), prod["name"],
               " / ".join(f"{a['name']} ({a['key']})" for a in others)))
    if not seen and not exp_verified:
        out["warnings"].append(
            "Not in the last live chain capture — the expiry below is RULE-DERIVED "
            "(indicative standard cycle), not observed.")
    if fx is None and ox is None:
        out["warnings"].append(
            f"{MONTH_NAME[month]} is not a listed month for this product's standard cycle.")
    return out


# ── BUILD: product + date in → symbol out ─────────────────────────────────────────────
def build(ticker: str, target: date, kind: str = "opt", n: int = 3) -> dict:
    """Codes for a product around a target date.

    Returns `exact` (the contract whose expiry IS the target date, if any), `nearest`
    (the n listed expiries either side) and `live_on` (the contract still trading ON that
    date — usually the more useful answer, since a calendar date rarely IS an expiry).

    `kind` is 'opt' or 'fut'. Every candidate carries its provenance."""
    prod = None
    for p in products():
        if p["ticker"] == ticker:
            prod = p
            break
    out = {"product": prod, "target": target, "kind": kind,
           "exact": None, "nearest": [], "live_on": None, "warnings": []}
    if not prod:
        out["warnings"].append("Unknown product.")
        return out

    root, asset = prod["root"], prod["asset"]
    obs = observed().get(root, {})
    yd = (obs.get("opt_yd") if kind == "opt" else obs.get("fut_yd")) or 1
    suffix = _strip_key(ticker)[1] or obs.get("suffix") or "Comdty"

    cands = []
    for y in range(target.year - 1, target.year + 3):
        for mth in expiries.listed_months(ticker, asset, kind):
            d = expiries.expiry_for(ticker, asset, y, mth, kind)
            if d is None:
                continue
            seen = (y, mth) in ((obs.get("opt_series") if kind == "opt"
                                 else obs.get("fut_contracts")) or set())
            vf = kind == "opt" and verified(ticker, d)
            cands.append({
                "code": _code(root, mth, y, yd, suffix, kind),
                "stem": f"{root}{MONTH_CODE[mth]}{y % 100 if yd == 2 else y % 10}",
                "contract": f"{MONTH_NAME[mth]} {y}", "month": mth, "year": y,
                "expiry": d, "source": "observed" if (seen or vf) else "rule",
                "expiry_verified": vf, "days": (d - target).days,
            })
    if not cands:
        out["warnings"].append(
            "This product's contract family isn't modelled in the expiry engine, so no "
            "code can be built. Add it to expiries.SPECS to enable it.")
        return out

    cands.sort(key=lambda c: c["expiry"])
    for c in cands:
        if c["expiry"] == target:
            out["exact"] = c
            break
    out["nearest"] = sorted(cands, key=lambda c: abs(c["days"]))[:n]
    out["nearest"].sort(key=lambda c: c["expiry"])
    live = [c for c in cands if c["expiry"] >= target]
    out["live_on"] = live[0] if live else None

    # ---- weeklies, where the product's scheme is known ------------------------------
    if kind == "opt" and has_weeklies(ticker):
        horizon = date.today() + timedelta(days=WEEKLY_HORIZON_DAYS)
        for w in weeklies_between(ticker, target - timedelta(days=21),
                                  target + timedelta(days=21)):
            rec = {"code": w["code"], "stem": w["stem"], "contract": w["contract"],
                   "month": month_of(w["expiry"]), "year": w["expiry"].year,
                   "expiry": w["expiry"], "source": "rule", "weekly": True,
                   "week": w["week"], "label": w["label"],
                   "listed": w["expiry"] <= horizon,
                   "days": (w["expiry"] - target).days}
            cands.append(rec)
            if w["expiry"] == target and (out["exact"] is None
                                          or not out["exact"].get("weekly")):
                out["exact"] = rec
        cands.sort(key=lambda c: c["expiry"])
        out["nearest"] = sorted(cands, key=lambda c: abs(c["days"]))[:max(n, 5)]
        out["nearest"].sort(key=lambda c: c["expiry"])
        live = [c for c in cands if c["expiry"] >= target]
        out["live_on"] = live[0] if live else out["live_on"]
        out["weeklies"] = True

    if out["exact"] is None:
        out["warnings"].append(
            f"No {'option' if kind == 'opt' else 'futures'} expiry falls exactly on "
            f"{target:%a %d %b %Y} — the nearest listed expiries are shown instead.")
    if kind == "opt" and has_weeklies(ticker):
        out["warnings"].append(
            "Bloomberg lists only a few weeks of weeklies at a time. A code beyond that "
            "horizon does not simply fail — with a one-digit year it can resolve to a "
            "DECADE-OLD expired contract (asking for Z6 = Dec-2026 returned Dec-2016). "
            "Rows marked 'not yet listed' are arithmetic, not tradeable codes.")
    elif kind == "opt":
        out["warnings"].append(
            f"Weekly and daily series for {prod['name']} are not in this database yet — "
            "only quarterlies and serial monthlies. The Bloomberg root for a weekly can't "
            "be derived from the futures root; it has to be read off the Terminal once "
            "(they're named \"… 1st Wee …\") and added to bbgcodes.WEEKLY_SPECS.")
    return out


def month_of(d: date) -> int:
    return d.month


def _code(root: str, month: int, year: int, yd: int, suffix: str, kind: str) -> str:
    """The Bloomberg symbol. Options append C/P + strike at the call site — the STEM is
    what's product-determined; the strike is the user's."""
    stem = f"{root}{MONTH_CODE[month]}{year % 100 if yd == 2 else year % 10}"
    return f"{stem} {suffix}" if kind == "fut" else f"{stem}C <strike> {suffix}"


def option_code(root: str, month: int, year: int, strike, put_call: str = "C",
                yd: int | None = None, suffix: str | None = None) -> str:
    """A complete option symbol once the strike is known, e.g. 'CLZ6C 80 Comdty'."""
    obs = observed().get(root, {})
    yd = yd or obs.get("opt_yd") or 1
    suffix = suffix or obs.get("suffix") or "Comdty"
    stem = f"{root}{MONTH_CODE[month]}{year % 100 if yd == 2 else year % 10}"
    return f"{stem}{put_call.upper()} {strike} {suffix}"


# ── SEARCH ────────────────────────────────────────────────────────────────────────────
def search(query: str, limit: int = 12) -> list:
    """Fuzzy product lookup — name, root or ticker. You half-know the symbol in practice,
    so 'soy', 'S ', 'ZS' and 'soybean' should all land somewhere useful."""
    q = str(query or "").strip().lower()
    if not q:
        return []
    scored = []
    for p in products():
        hay = f"{p['name']} {p['root']} {p['ticker']}".lower()
        if q == p["root"].strip().lower() or q == p["ticker"].lower():
            s = 0
        elif hay.startswith(q) or p["name"].lower().startswith(q):
            s = 1
        elif q in hay:
            s = 2
        else:
            continue
        scored.append((s, p["name"], p))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _s, _n, p in scored[:limit]]
