"""Prescribed-vs-priced bridge (BASIS · Macro Rate Radar).

The orchestration layer: it joins the macro side (src/macrodata + src/macrorules) to the
market side (src/stirpaths, which inverts the live strip into a meeting-step path) and
produces the one number the whole module exists for —

        what the macro says the policy rate should be
      − what the curve has already priced
      = the size of the disagreement, in basis points, per meeting

A rule level on its own is close to useless for trading: it can sit 150bp from the actual
policy rate for years without anything happening. What is actionable is the SPREAD against
pricing, because that is the part you can put on. This module produces that spread, and
then converts it into P&L on the contracts the desk actually trades.

Rate conventions, and one trap
------------------------------
stirpaths fits the OVERNIGHT PROXY (SOFR / €STR / SONIA), not the policy rate itself. The
two differ by a small basis held in stirpaths.BANK_BASIS_SEED — €STR fixes about 8bp below
the ECB's deposit rate, while SOFR and SONIA sit essentially on top of theirs. Comparing a
rule prescription (a POLICY rate) against a fitted path (an OVERNIGHT rate) without
converting would put a spurious 8bp of "divergence" on every ECB meeting forever. The
conversion is verified: the ECB fit's front segment of 2.17% maps to a deposit rate of
2.25%, which is exactly where the ECB has it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src import macrorules, stirpaths


# ── which banks this module covers, and what to call them ───────────────────────────
# The roster is the union of stirpaths.BANKS and the extras below, because a bloc can
# have a macro leg here before it has a priced strip there. Brazil was exactly that: rich
# free BCB data and no fit, since DI1 is a different instrument (compounded to a fixed
# maturity on a business-252 count, not a 100−rate averaging contract) and modelling it
# was its own build. The extras are a FALLBACK for display metadata and for the survey
# path — once a bloc gains a real strip it is served from stirpaths automatically.
@dataclass(frozen=True)
class _BankMeta:
    key: str
    name: str
    meeting_name: str
    rate_name: str
    ccy: str = "$"


_EXTRA_BANKS = {
    "BCB": _BankMeta("BCB", "Banco Central do Brasil", "Copom", "Selic target", "R$"),
}

# dict.fromkeys de-duplicates while preserving order: a bank listed here twice renders
# two buttons with the same Streamlit key and crashes the page. That is not theoretical —
# it happened the moment BCB gained a strip in stirpaths.BANKS while still being carried
# in _EXTRA_BANKS as a survey-only bloc.
RADAR_BANKS: list[str] = list(dict.fromkeys(list(stirpaths.BANKS) + list(_EXTRA_BANKS)))


def bank_meta(bank: str):
    """Display metadata for any radar bank, whether or not it has a fitted strip."""
    bank = bank.upper()
    return stirpaths.BANKS.get(bank) or _EXTRA_BANKS[bank]


def is_survey_bank(bank: str) -> bool:
    """True where this bloc has no strip in stirpaths at all.

    Deliberately NOT the last word on whether a given comparison is survey-based: a bank
    can be in stirpaths.BANKS and still return no fit (its strip half-built, or simply not
    in the morning store yet). compare() therefore decides per call, from what the fit
    actually yields, and a bloc upgrades itself from survey to market pricing the moment a
    real path appears — with every label following, because they key off the result."""
    return bank.upper() not in stirpaths.BANKS


def policy_from_overnight(bank: str, overnight: float) -> float:
    """Overnight proxy -> policy rate, undoing the bank's fixing basis.

    Call this on a STRIP fit only. Survey paths are quoted in the policy rate itself, so
    they must bypass it — do not rely on the bank being absent from BANK_BASIS_SEED to
    make it a no-op, because a bloc gains a basis the moment someone builds its strip."""
    return overnight - stirpaths.BANK_BASIS_SEED.get(bank.upper(), 0.0) / 100.0


def overnight_from_policy(bank: str, policy: float) -> float:
    return policy + stirpaths.BANK_BASIS_SEED.get(bank.upper(), 0.0) / 100.0


@dataclass
class MeetingCompare:
    meeting: date
    priced_policy: float          # policy rate the strip implies AFTER this meeting, %
    priced_cum_bp: float          # cumulative move priced from today, bp
    prescribed: float | None      # rule prescription for that date, %
    spread_bp: float | None       # prescribed − priced, bp. +ve = rules want MORE hikes
                                  #   than the curve has in it


@dataclass
class RadarResult:
    bank: str
    ok: bool
    asof: date
    policy_now: float
    rule_key: str
    rule_name: str
    prescribed_now: float | None
    summary: "macrorules.RuleSummary | None"
    meetings: list[MeetingCompare] = field(default_factory=list)
    provenance: "macrorules.InputProvenance | None" = None
    strip_asof: str | None = None
    strip_source: str = ""
    reason: str = ""
    # True when the comparison is against a SURVEY of forecasters rather than against
    # tradeable pricing. Every label the user sees has to change with this: a survey
    # median carries no risk premium, cannot be executed, and moves once a week.
    path_is_survey: bool = False

    @property
    def headline_bp(self) -> float | None:
        """Divergence at the furthest meeting we have both sides for — the cleanest single
        statement of how far apart the macro and the curve are."""
        usable = [m for m in self.meetings if m.spread_bp is not None]
        return usable[-1].spread_bp if usable else None

    @property
    def max_divergence(self) -> MeetingCompare | None:
        usable = [m for m in self.meetings if m.spread_bp is not None]
        return max(usable, key=lambda m: abs(m.spread_bp)) if usable else None


@dataclass
class _SurveyFit:
    """The shape compare() needs from a strip fit, filled from a survey instead.

    Mirrors the stirpaths fit contract exactly — `seg_rates[0]` is today and
    `seg_rates[i+1]` is the level after meeting i — so the comparison below is written
    once and works for both. The rates here are POLICY rates already, which is why the
    basis conversion is a no-op for these banks."""
    meetings: list[date]
    seg_rates: list[float]
    cum_bp: list[float]
    source: str = ""
    asof_label: str = ""


def _survey_fit(bank: str, policy_now: float, asof: date) -> "_SurveyFit | None":
    """Build a meeting path from the BCB's Focus survey of professional forecasters.

    This is Brazil's stand-in for a fitted strip until the DI1 curve is modelled. It is
    a genuinely different object from a market-implied path and the difference is not
    pedantic: a survey median is a central expectation with no term premium in it, so
    the spread against a rule prescription means "the rules disagree with economists",
    not "the rules disagree with the price of risk"."""
    from src import macrodata

    rows = [r for r in macrodata.focus_selic_path() if r.get("meeting")]
    rows = [r for r in rows if r["meeting"] > asof]
    if not rows:
        return None
    meetings = [r["meeting"] for r in rows]
    seg = [policy_now] + [float(r["median"]) for r in rows]
    cum = [(float(r["median"]) - policy_now) * 100.0 for r in rows]
    n = rows[0].get("n")
    return _SurveyFit(meetings, seg, cum,
                      source=f"BCB Focus survey — median of {n} forecasters"
                             if n else "BCB Focus survey — median",
                      asof_label=str(rows[0].get("asof") or ""))


def compare(bank: str, *, rule=macrorules.balanced, asof: date | None = None,
            nairu: float | None = None, rstar: float | None = None,
            assume: "macrorules.PathAssumption | None" = None,
            use_expectations: bool = False,
            max_meetings: int = 8) -> RadarResult:
    """Prescribed path vs priced path for one bank."""
    bank = bank.upper()
    asof = asof or date.today()

    x, prov = macrorules.inputs_from_data(bank, nairu=nairu, rstar=rstar,
                                          use_expectations=use_expectations)
    res = macrorules.evaluate(x)
    summary = macrorules.summarise(res, x.policy_rate)
    now = rule(x)

    # Prefer tradeable pricing; fall back to a survey of forecasters only where there is
    # no priced path to be had. The spread against a strip and the spread against a
    # survey are different claims, so which one produced this result is recorded on it.
    survey, fit = False, None
    try:
        if bank in stirpaths.BANKS:
            try:
                fit = stirpaths.default_bank_fit(bank, asof)
            except Exception:
                fit = None
        if fit is None or not len(getattr(fit, "meetings", ())):
            sfit = _survey_fit(bank, x.policy_rate, asof)
            if sfit is not None:
                fit, survey = sfit, True
    except Exception as e:
        return RadarResult(bank, False, asof, x.policy_rate, now.key, now.name,
                           now.prescribed, summary, [], prov,
                           reason=f"no expected path: {e}" if survey
                                  else f"no market-implied path: {e}",
                           path_is_survey=survey)
    if fit is None or not len(fit.meetings):
        return RadarResult(bank, False, asof, x.policy_rate, now.key, now.name,
                           now.prescribed, summary, [], prov,
                           reason=("no Focus survey path available — the BCB feed is "
                                   "unreachable or quotes no dated meeting"
                                   if survey else
                                   "no market-implied path available for this bank"),
                           path_is_survey=survey)

    meetings = list(fit.meetings)[:max_meetings]
    path = dict(macrorules.prescribed_path(x, meetings, rule=rule, assume=assume,
                                           start=asof))

    rows = []
    for i, m in enumerate(meetings):
        # seg_rates[0] is today; seg_rates[i+1] is the level AFTER meeting i.
        seg = float(fit.seg_rates[i + 1]) if i + 1 < len(fit.seg_rates) else None
        if seg is None:
            continue
        # Only a STRIP fit needs the basis conversion: it prices the overnight proxy.
        # A survey is quoted in the policy rate itself, so converting it would invent a
        # divergence out of the fixing basis — the same error the module docstring warns
        # about for the ECB, in reverse. This bit once: BCB gained a -10bp seed when its
        # DI strip was built, and every Focus-survey meeting silently moved 10bp.
        priced = seg if survey else policy_from_overnight(bank, seg)
        pres = path.get(m)
        rows.append(MeetingCompare(m, priced, float(fit.cum_bp[i]), pres,
                                   None if pres is None else (pres - priced) * 100.0))

    return RadarResult(bank, True, asof, x.policy_rate, now.key, now.name, now.prescribed,
                       summary, rows, prov,
                       strip_asof=(fit.asof_label if survey
                                   else stirpaths.strip_store_asof()),
                       strip_source=(fit.source if survey else ""),
                       path_is_survey=survey)


# ── turning a divergence into a trade ───────────────────────────────────────────────
@dataclass
class ContractEdge:
    """One contract's mispricing under the rule path.

    Both figures are stated for a LONG position, and the sign is the trade direction:

        edge_bp < 0  the contract is RICH to the rule path — the rule wants higher rates
                     than the curve, so the future should fall. Sell it.
        edge_bp > 0  the contract is CHEAP — the rule wants lower rates. Buy it.

    edge_bp is a PRICE difference expressed in bp (fair price − market price, ×100). It is
    not a rate difference; because a future prices at 100 − rate the two carry opposite
    signs, and conflating them inverts every trade on the page."""
    ticker: str
    short: str
    code: str
    market_price: float
    rule_fair: float
    edge_bp: float                # price edge in bp, LONG convention (see above)
    ccy: str
    pnl_per_lot: float            # currency P&L per lot HELD LONG if the rule path is right


def contract_edges(bank: str, *, rule=macrorules.balanced, asof: date | None = None,
                   nairu: float | None = None, rstar: float | None = None,
                   assume: "macrorules.PathAssumption | None" = None,
                   n: int = 8) -> list[ContractEdge]:
    """Price every contract in the bank's strip off the RULE-implied path and compare with
    the market. This is the Strategy Builder bridge: it answers "if the macro is right and
    the curve is wrong, which contract pays, and how much per lot?"

    Treat the output as a sizing aid, not a forecast — it inherits every assumption in the
    rule inputs, and for the BoE that includes r* and NAIRU, which nobody publishes.
    """
    bank = bank.upper()
    asof = asof or date.today()
    x, _prov = macrorules.inputs_from_data(bank, nairu=nairu, rstar=rstar)
    bank_obj = stirpaths.BANKS[bank]

    try:
        fit = stirpaths.default_bank_fit(bank, asof)
    except Exception:
        fit = None
    meetings = list(fit.meetings) if fit is not None else list(bank_obj.meetings)
    meetings = [m for m in meetings if m > asof][:12]
    path = dict(macrorules.prescribed_path(x, meetings, rule=rule, assume=assume,
                                           start=asof))
    if not path:
        return []

    # Step function of the OVERNIGHT proxy under the rule path: flat between decisions,
    # changing the business day after each one — the same convention stirpaths prices on.
    steps = sorted((stirpaths.bank_effective_date(bank_obj, m),
                    overnight_from_policy(bank, p)) for m, p in path.items())
    start_rate = overnight_from_policy(bank, x.policy_rate)

    def rate_fn(d: date) -> float:
        cur = start_rate
        for eff, r in steps:
            if d >= eff:
                cur = r
            else:
                break
        return cur

    out = []
    for prod in stirpaths.bank_products(bank):
        if not prod.in_strip:
            continue
        contracts = stirpaths.strip(prod, asof, n)
        if not contracts:
            continue
        try:
            prices = stirpaths.strip_prices(prod, bank_obj, contracts, asof, x.policy_rate)
        except Exception:
            continue
        for c, px in zip(contracts, prices):
            try:
                fair = stirpaths.fair_price(prod, c, rate_fn)
            except Exception:
                continue
            # Futures price = 100 − rate, so a price difference in POINTS is a rate
            # difference of the opposite sign; ×100 puts it in bp.
            edge_bp = (fair - px) * 100.0
            out.append(ContractEdge(prod.ticker, prod.short, c.code, round(px, 4),
                                    round(fair, 4), round(edge_bp, 1), bank_obj.ccy,
                                    round(edge_bp * prod.bp_value, 2)))
    return out


# ── one-call assembly for the page and the PDF ──────────────────────────────────────
def dashboard(banks=("FED", "ECB", "BOE"), *, rule=macrorules.balanced,
              asof: date | None = None, overrides: dict | None = None) -> dict:
    """Everything the page needs, in one call. `overrides` is {bank: {nairu, rstar}}."""
    from src import macrosurprise

    overrides = overrides or {}
    out = {"asof": asof or date.today(), "banks": {}, "rule": getattr(rule, "__name__", "")}
    for b in banks:
        o = overrides.get(b, {})
        try:
            out["banks"][b] = compare(b, rule=rule, asof=asof,
                                      nairu=o.get("nairu"), rstar=o.get("rstar"))
        except Exception as e:                    # one dead bank must not blank the page
            out["banks"][b] = RadarResult(b, False, asof or date.today(), 0.0, "", "",
                                          None, None, [], None, reason=str(e)[:200])
    out["surprise"] = {b: macrosurprise.index(b) for b in banks}
    out["surprise_readiness"] = macrosurprise.readiness()
    return out


# ── Hot Sheet provider (src/hotsheet.py discovers radar_items by name) ────────────────
# Prescribed-vs-priced divergence worth surfacing, in bp. The module publishes no
# threshold of its own (level gaps of 100bp+ persist for years — see the docstring);
# 50bp = two conventional steps, comfortably past rule-input noise.
GAP_MIN_BP = 50.0
# …and BECAUSE those level gaps persist for years, a level bar alone parks a
# permanent line on the exception-based sheet (the BoE's assumed-input gap sat at
# heat 100 from day one). So a standing gap must also have MOVED since it last
# made the sheet — a print moving the prescription, or the market repricing
# against the rule — or crossed sides. First-ever appearances still show once.
GAP_MOVE_BP = 20.0

# The bank's flagship strip ticker, for the Hot Sheet's sector filter. Brazil has no
# entry: its DI1 curve is not in the universe, and pointing the filter at BRL FX would
# file a rates signal under a currency.
_RADAR_TICKER = {"FED": "SFRA Comdty", "ECB": "ERA Comdty", "BOE": "SFIA Comdty"}


def _last_stamped_gap(bk: str) -> float | None:
    """The gap recorded the last time this bank's line was on a STAMPED sheet
    (data/signals/hotsheet_history.parquet `value` column). Stamps only move once
    a day, so the move-gate compares against the last appearance, not the last
    render — slow drift accumulates until it clears GAP_MOVE_BP and re-flags."""
    try:
        from src import hotsheet
        h = hotsheet.load_history()
        h = h[h["key"] == f"MACRO:{bk}:rulegap"].dropna(subset=["value"])
        if h.empty:
            return None
        return float(h.sort_values("date")["value"].iloc[-1])
    except Exception:
        return None


def radar_items() -> list:
    """Daily Hot Sheet items: banks where the policy-rule prescription sits notably
    away from what the curve prices.

    STRICTLY CACHE-ONLY. compare() lets the macrodata fetchers refresh any cache
    older than its TTL — a network call this context must never make (the Hot
    Sheet runs inside snapshot compute and page loads). macrodata was built to
    degrade to its stale disk cache whenever HTTP fails, so for the duration of
    the call its fetch hook is swapped for one that always raises: every series
    then comes off data/macro_cache exactly as last written (or reports ok=False
    and drops out), and nothing can touch the network. The market side is already
    store-only (stirpaths.default_bank_fit reads the morning strip store)."""
    from src import hotsheet, macrodata

    def _no_network(*a, **k):
        raise RuntimeError("hotsheet provider is cache-only — no network fetches")

    out = []
    orig_http = macrodata._http
    macrodata._http = _no_network
    try:
        for bk in RADAR_BANKS:
            try:
                res = compare(bk)
            except Exception:                     # one dead bank never blanks the sheet
                continue
            if not res.ok:
                continue
            usable = [m for m in res.meetings if m.spread_bp is not None]
            if not usable:
                continue
            last = usable[-1]                     # the headline_bp meeting
            gap = float(last.spread_bp)
            if abs(gap) < GAP_MIN_BP:
                continue
            prev = _last_stamped_gap(bk)          # the gap when this line last made the sheet
            flipped = prev is not None and (gap > 0) != (prev > 0)
            if prev is not None and abs(gap - prev) < GAP_MOVE_BP and not flipped:
                continue                          # standing gap, unmoved — wallpaper, not news
            # Per-bank honesty (module convention): say so when the rule ran on
            # assumed inputs rather than measured ones — the BoE leg always does.
            caveat = ""
            if res.provenance is not None and res.provenance.assumed:
                caveat = f" ({' and '.join(res.provenance.assumed)} assumed for this bloc)"
            side = "above" if gap > 0 else "below"
            versus = ("what the **Focus survey** expects for the"
                      if res.path_is_survey else "what the market prices for the")
            if flipped:
                move = " The prescription has **crossed** to the other side of the market."
            elif prev is not None:
                word = "widened" if abs(gap) > abs(prev) else "narrowed"
                move = (f" The gap has **{word} {abs(gap - prev):.0f}bp** since it "
                        f"last made the sheet.")
            else:
                move = ""
            out.append(hotsheet.item(
                tag="MACRO", key=f"{bk}:rulegap", section="Policy rules",
                text=(f"The **{res.rule_name}** rule prescribes a policy rate "
                      f"**{abs(gap):.0f}bp {side}** {versus} "
                      f"**{bank_meta(bk).name}**{caveat}.{move}"),
                metric=f"{gap:+.0f} bp", sub=f"by the {last.meeting:%b %Y} meeting",
                heat=(min(100.0, abs(gap - prev) / 50.0 * 100.0) if prev is not None
                      else min(100.0, abs(gap) / 150.0 * 100.0)),
                value=gap, ticker=_RADAR_TICKER.get(bk, ""),
                page="Macro Radar", book="ficc"))
    finally:
        macrodata._http = orig_http
    return out
