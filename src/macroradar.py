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


def policy_from_overnight(bank: str, overnight: float) -> float:
    """Overnight proxy -> policy rate, undoing the bank's fixing basis."""
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


def compare(bank: str, *, rule=macrorules.balanced, asof: date | None = None,
            nairu: float | None = None, rstar: float | None = None,
            assume: "macrorules.PathAssumption | None" = None,
            max_meetings: int = 8) -> RadarResult:
    """Prescribed path vs priced path for one bank."""
    bank = bank.upper()
    asof = asof or date.today()

    x, prov = macrorules.inputs_from_data(bank, nairu=nairu, rstar=rstar)
    res = macrorules.evaluate(x)
    summary = macrorules.summarise(res, x.policy_rate)
    now = rule(x)

    fit = None
    try:
        fit = stirpaths.default_bank_fit(bank, asof)
    except Exception as e:
        return RadarResult(bank, False, asof, x.policy_rate, now.key, now.name,
                           now.prescribed, summary, [], prov,
                           reason=f"no market-implied path: {e}")
    if fit is None or not len(fit.meetings):
        return RadarResult(bank, False, asof, x.policy_rate, now.key, now.name,
                           now.prescribed, summary, [], prov,
                           reason="no market-implied path available for this bank")

    meetings = list(fit.meetings)[:max_meetings]
    path = dict(macrorules.prescribed_path(x, meetings, rule=rule, assume=assume,
                                           start=asof))

    rows = []
    for i, m in enumerate(meetings):
        # seg_rates[0] is today; seg_rates[i+1] is the level AFTER meeting i.
        seg = float(fit.seg_rates[i + 1]) if i + 1 < len(fit.seg_rates) else None
        if seg is None:
            continue
        priced = policy_from_overnight(bank, seg)
        pres = path.get(m)
        rows.append(MeetingCompare(m, priced, float(fit.cum_bp[i]), pres,
                                   None if pres is None else (pres - priced) * 100.0))

    return RadarResult(bank, True, asof, x.policy_rate, now.key, now.name, now.prescribed,
                       summary, rows, prov, stirpaths.strip_store_asof())


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

# The bank's flagship strip ticker, for the Hot Sheet's sector filter.
_RADAR_TICKER = {"FED": "SFRA Comdty", "ECB": "ERA Comdty", "BOE": "SFIA Comdty"}


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
        for bk in stirpaths.BANKS:
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
            # Per-bank honesty (module convention): say so when the rule ran on
            # assumed inputs rather than measured ones — the BoE leg always does.
            caveat = ""
            if res.provenance is not None and res.provenance.assumed:
                caveat = f" ({' and '.join(res.provenance.assumed)} assumed for this bloc)"
            side = "above" if gap > 0 else "below"
            out.append(hotsheet.item(
                tag="MACRO", key=f"{bk}:rulegap", section="Policy rules",
                text=(f"The **{res.rule_name}** rule prescribes a policy rate "
                      f"**{abs(gap):.0f}bp {side}** what the market prices for the "
                      f"**{stirpaths.BANKS[bk].name}**{caveat}."),
                metric=f"{gap:+.0f} bp", sub=f"by the {last.meeting:%b %Y} meeting",
                heat=min(100.0, abs(gap) / 150.0 * 100.0),
                value=gap, ticker=_RADAR_TICKER.get(bk, ""),
                page="Macro Radar", book="ficc"))
    finally:
        macrodata._http = orig_http
    return out
