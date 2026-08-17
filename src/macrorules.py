"""Monetary policy rule engine (BASIS · Macro Rate Radar).

The formula fixed-income desks argue about — the one tying the output gap, inflation and
the policy rate together — is the Taylor rule and its descendants:

        i  =  r*  +  π  +  0.5·(π − π*)  +  b·gap

This module implements the five variants the Federal Reserve itself publishes in the
"Monetary Policy Rules" box of its semiannual Monetary Policy Report, generalised to the
ECB and BoE. Reproducing the Fed's own table matters: it gives us a published number to
validate against twice a year, instead of a plausible-looking figure nobody can check.

What this is and is not
-----------------------
Policy rules are PRESCRIPTIVE, not predictive. No central bank mechanically follows one,
and the level gap between prescription and actual policy is routinely 100bp+ and can
persist for years. Anyone reading a rule level as a forecast will lose money.

Where the tradeable signal actually lives, in rough order of usefulness:

  1. The CHANGE in the prescription — a rule that swings 60bp in a month because core
     came in hot is telling you something the strip may not have absorbed yet.
  2. The prescription versus WHAT IS ALREADY PRICED (src/stirpaths.py inverts the strip
     into a meeting path). The spread is the thesis; the level alone is not.
  3. The DISPERSION across the five rules. When Taylor and the shortfalls rule disagree
     by 150bp, the committee has genuine latitude and the distribution of outcomes is
     wide — which is an options view, not a directional one.

Sign convention: everything is in PERCENT, annualised, and every `gap` is positive when
the economy is running hot (output above potential / unemployment below its natural
rate), so a positive gap always pushes the prescribed rate UP.

Per-bank honesty
----------------
The three legs are NOT of equal quality, and the UI must say so:

  FED  Strongest. CBO publishes both potential GDP and the natural rate of unemployment,
       so the gap is a published gap. r* from HLW or the SEP longer-run median.
  ECB  Middling. No CBO equivalent; the output gap leans on HLW's own euro-area estimate,
       which is model-derived and ~2 quarters stale. Inflation is current (Eurostat).
  BOE  Weakest. HLW publishes no UK r* and the ONS publishes no potential output, so BOTH
       r* and the gap are assumptions here, not measurements. They are exposed as editable
       inputs and flagged on the page rather than dressed up as data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Okun's coefficient: how much output gap one point of unemployment gap is worth.
# The Fed's MPR writes its rules directly in unemployment-gap space using this = 2, which
# is why its Taylor rule shows a coefficient of 1 on (u* − u) rather than Taylor's
# original 0.5 on the output gap: 0.5 × 2 = 1. Keep the two forms consistent.
OKUN = 2.0

# The classic smoothing weight on the lagged policy rate in the inertial rule (Fed MPR).
INERTIA = 0.85


@dataclass
class RuleInputs:
    """Everything the rules need, already reduced to scalars in percent.

    Prefer `unemp`/`nairu` over `output_gap` where the data supports it: that is the form
    the Fed's own MPR uses, and unemployment is monthly, revised far less brutally than
    GDP, and available in real time. `output_gap` is the fallback for blocs with no
    published natural-rate series.
    """
    bank: str = "FED"
    infl: float = 2.0                 # core inflation, y/y %
    target: float = 2.0               # π*
    rstar: float = 0.75               # longer-run REAL neutral rate, %
    unemp: float | None = None        # u, %
    nairu: float | None = None        # u*, %
    output_gap: float | None = None   # % of potential, +ve = hot (used if unemp is None)
    policy_rate: float = 0.0          # current policy setting, % (the level rules land on)
    prev_policy_rate: float | None = None    # i_{t−1} for the inertial rule
    # Lag-4 values for the first-difference rule (a year ago):
    gap_lag4: float | None = None
    okun: float = OKUN
    infl_label: str = "core inflation"

    def gap(self) -> float | None:
        """Output gap in percent, positive when hot. Derived from the unemployment gap
        when unemployment is available, else taken as given."""
        if self.unemp is not None and self.nairu is not None:
            # u below u* = hot = positive gap.
            return self.okun * (self.nairu - self.unemp)
        return self.output_gap

    def u_gap(self) -> float | None:
        """(u* − u) in percentage points, positive when hot — the Fed MPR's native form."""
        if self.unemp is not None and self.nairu is not None:
            return self.nairu - self.unemp
        if self.output_gap is not None:
            return self.output_gap / self.okun
        return None

    def infl_gap(self) -> float:
        return self.infl - self.target


@dataclass
class RuleResult:
    key: str
    name: str
    prescribed: float | None          # % — the level the rule points at
    formula: str                      # human-readable, with the numbers substituted
    terms: dict = field(default_factory=dict)
    note: str = ""
    ok: bool = True
    reason: str = ""

    def vs_actual(self, actual: float) -> float | None:
        """Prescription minus the current policy setting, in BASIS POINTS.
        Positive = the rule says policy is too easy."""
        if self.prescribed is None:
            return None
        return (self.prescribed - actual) * 100.0


# ── the five rules ──────────────────────────────────────────────────────────────────
def _need(*vals) -> bool:
    return all(v is not None for v in vals)


def taylor93(x: RuleInputs) -> RuleResult:
    """Taylor (1993), the original: equal 0.5 weights on the inflation and output gaps.

    In the Fed MPR's unemployment-gap form the output-gap term becomes 1.0·(u* − u),
    because Okun's factor of 2 absorbs the 0.5."""
    ug = x.u_gap()
    if not _need(ug):
        return RuleResult("taylor93", "Taylor (1993)", None, "", ok=False,
                          reason="no output or unemployment gap available")
    p = x.rstar + x.infl + 0.5 * x.infl_gap() + 1.0 * ug
    return RuleResult(
        "taylor93", "Taylor (1993)", p,
        f"{x.rstar:.2f} + {x.infl:.2f} + 0.5×({x.infl:.2f} − {x.target:.2f}) "
        f"+ 1.0×({ug:+.2f}) = {p:.2f}%",
        {"r*": x.rstar, "π": x.infl, "infl gap": x.infl_gap(), "u gap": ug})


def balanced(x: RuleInputs) -> RuleResult:
    """Balanced-approach rule: double the weight on the real-economy gap. This is the one
    most often quoted as "the Taylor rule" in market commentary, and the closest single
    rule to how the FOMC has actually behaved over the past two decades."""
    ug = x.u_gap()
    if not _need(ug):
        return RuleResult("balanced", "Balanced approach", None, "", ok=False,
                          reason="no output or unemployment gap available")
    p = x.rstar + x.infl + 0.5 * x.infl_gap() + 2.0 * ug
    return RuleResult(
        "balanced", "Balanced approach", p,
        f"{x.rstar:.2f} + {x.infl:.2f} + 0.5×({x.infl:.2f} − {x.target:.2f}) "
        f"+ 2.0×({ug:+.2f}) = {p:.2f}%",
        {"r*": x.rstar, "π": x.infl, "infl gap": x.infl_gap(), "u gap": ug})


def shortfalls(x: RuleInputs) -> RuleResult:
    """Balanced approach (shortfalls) — the post-2020 framework rule.

    The real-economy term is floored at zero: the rule eases when unemployment runs ABOVE
    its natural rate, but does not call for tightening merely because the labour market is
    tight. That asymmetry is the whole point of the 2020 framework revision, and it is why
    this rule sat far below the others through 2022-23 while the others screamed hike."""
    ug = x.u_gap()
    if not _need(ug):
        return RuleResult("shortfalls", "Balanced approach (shortfalls)", None, "",
                          ok=False, reason="no output or unemployment gap available")
    term = min(ug, 0.0)
    p = x.rstar + x.infl + 0.5 * x.infl_gap() + 2.0 * term
    note = ("labour-market term inactive — unemployment is at or below its natural rate, "
            "and this rule does not tighten for tightness alone") if ug > 0 else ""
    return RuleResult(
        "shortfalls", "Balanced approach (shortfalls)", p,
        f"{x.rstar:.2f} + {x.infl:.2f} + 0.5×({x.infl:.2f} − {x.target:.2f}) "
        f"+ 2.0×min({ug:+.2f}, 0) = {p:.2f}%",
        {"r*": x.rstar, "π": x.infl, "infl gap": x.infl_gap(), "u gap": ug,
         "active": ug <= 0}, note)


def inertial(x: RuleInputs) -> RuleResult:
    """Inertial rule: 85% of last period's actual setting plus 15% of the balanced-approach
    prescription. Central banks move in 25bp steps and hate reversing, so this is usually
    the best-behaved of the five as a description of what actually happens next."""
    base = balanced(x)
    prev = x.prev_policy_rate if x.prev_policy_rate is not None else x.policy_rate
    if not base.ok or base.prescribed is None:
        return RuleResult("inertial", "Inertial", None, "", ok=False, reason=base.reason)
    p = INERTIA * prev + (1 - INERTIA) * base.prescribed
    return RuleResult(
        "inertial", "Inertial", p,
        f"{INERTIA:.2f}×{prev:.2f} + {1 - INERTIA:.2f}×{base.prescribed:.2f} = {p:.2f}%",
        {"prev": prev, "balanced": base.prescribed})


def first_difference(x: RuleInputs) -> RuleResult:
    """First-difference rule: prescribes a CHANGE from the current setting, not a level.

    Its great virtue is that r* cancels out entirely — no estimate of the neutral rate is
    needed. Given r* is both the most disputed input and the one carrying the widest error
    bars (HLW's own confidence intervals span more than a point), a rule immune to it is a
    genuinely useful cross-check on the other four.

    Needs the gap from a year ago; without it the rule cannot be evaluated, and saying so
    is better than silently substituting zero."""
    ug = x.u_gap()
    if not _need(ug, x.gap_lag4):
        return RuleResult("firstdiff", "First difference", None, "", ok=False,
                          reason="needs the unemployment gap from four quarters ago")
    prev = x.prev_policy_rate if x.prev_policy_rate is not None else x.policy_rate
    delta = 0.5 * x.infl_gap() + (ug - x.gap_lag4)
    p = prev + delta
    return RuleResult(
        "firstdiff", "First difference", p,
        f"{prev:.2f} + 0.5×({x.infl:.2f} − {x.target:.2f}) "
        f"+ ({ug:+.2f} − {x.gap_lag4:+.2f}) = {p:.2f}%",
        {"prev": prev, "Δ": delta, "u gap": ug, "u gap −4q": x.gap_lag4},
        "r* cancels out of this rule — it is the one prescription independent of the "
        "neutral-rate estimate")


ALL_RULES = [taylor93, balanced, shortfalls, inertial, first_difference]
RULE_ORDER = ["taylor93", "balanced", "shortfalls", "inertial", "firstdiff"]


def evaluate(x: RuleInputs, rules=None) -> list[RuleResult]:
    """Run every rule against one set of inputs."""
    return [r(x) for r in (rules or ALL_RULES)]


# ── summary across rules ────────────────────────────────────────────────────────────
@dataclass
class RuleSummary:
    results: list[RuleResult]
    actual: float
    median: float | None
    lo: float | None
    hi: float | None
    dispersion_bp: float | None       # hi − lo, in bp: how much latitude the committee has
    median_gap_bp: float | None       # median prescription − actual, in bp

    @property
    def verdict(self) -> str:
        """Plain, non-advisory language — this text reaches client-facing PDFs, so it
        observes rather than recommends (house compliance rule)."""
        if self.median_gap_bp is None:
            return "Not enough data to evaluate the rule set."
        g = self.median_gap_bp
        if abs(g) < 25:
            return ("The rule set sits broadly in line with the current policy setting "
                    f"({g:+.0f}bp median gap).")
        direction = "above" if g > 0 else "below"
        return (f"The median rule prescription sits {abs(g):.0f}bp {direction} the current "
                f"policy setting, which may be worth a closer look against what the strip "
                f"is pricing.")


def summarise(results: list[RuleResult], actual: float) -> RuleSummary:
    vals = sorted(r.prescribed for r in results if r.ok and r.prescribed is not None)
    if not vals:
        return RuleSummary(results, actual, None, None, None, None, None)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return RuleSummary(results, actual, med, vals[0], vals[-1],
                       (vals[-1] - vals[0]) * 100.0, (med - actual) * 100.0)


# ── forward path: what the rules imply over the next N meetings ─────────────────────
@dataclass
class PathAssumption:
    """How inflation and the gap evolve from here — the bridge from a static prescription
    to something comparable with a market-implied meeting path.

    Defaults are deliberately bland mean reversion rather than a house forecast: inflation
    decays toward target and the gap closes, both with a half-life the user can set. The
    point of the page is to compare a TRANSPARENT macro baseline against market pricing,
    not to smuggle in a forecast nobody can inspect."""
    infl_half_life_q: float = 6.0     # quarters for (π − π*) to halve
    gap_half_life_q: float = 8.0      # quarters for the gap to halve
    infl_shock: float = 0.0           # pp added to today's inflation (scenario slider)
    gap_shock: float = 0.0            # pp added to today's gap
    rstar_shift: float = 0.0          # pp added to r*


def prescribed_path(x: RuleInputs, meeting_dates: list, rule=balanced,
                    assume: PathAssumption | None = None) -> list[tuple]:
    """The rule's prescribed policy level at each future meeting date, under `assume`.

    Returns [(date, prescribed_%), ...] aligned to `meeting_dates`, so it can be plotted
    straight on top of the market-implied step path from src/stirpaths.py.

    Time is measured in quarters from the first meeting; the decay is applied to today's
    inflation and gap. Deliberately simple and legible — a fancier forecast here would be
    a forecast the user cannot audit."""
    a = assume or PathAssumption()
    if not meeting_dates:
        return []
    base_infl = x.infl + a.infl_shock
    base_ug = x.u_gap()
    if base_ug is None:
        return []
    base_ug += a.gap_shock / x.okun
    t0 = meeting_dates[0]
    out = []
    prev_rate = x.prev_policy_rate if x.prev_policy_rate is not None else x.policy_rate
    for d in meeting_dates:
        q = max(0.0, (d - t0).days / 91.3)
        infl_t = x.target + (base_infl - x.target) * 0.5 ** (q / a.infl_half_life_q)
        ug_t = base_ug * 0.5 ** (q / a.gap_half_life_q)
        xi = RuleInputs(bank=x.bank, infl=infl_t, target=x.target,
                        rstar=x.rstar + a.rstar_shift,
                        output_gap=ug_t * x.okun, policy_rate=x.policy_rate,
                        prev_policy_rate=prev_rate, gap_lag4=x.gap_lag4, okun=x.okun)
        r = rule(xi)
        if r.prescribed is None:
            continue
        out.append((d, r.prescribed))
        # the inertial rule chains off its own previous prescription along the path
        prev_rate = r.prescribed
    return out


# ── assembling inputs from the live data layer ──────────────────────────────────────
# Fallbacks for blocs where nobody publishes the number. These are ASSUMPTIONS and the UI
# must present them as editable inputs, never as data.
DEFAULT_NAIRU = {"FED": None,      # CBO publishes it (NROU) — never guess for the US
                 "ECB": 6.8,       # euro-area structural unemployment, broad consensus
                 "BOE": 4.25}      # BoE's own medium-term equilibrium U assumption
DEFAULT_RSTAR = {"FED": None,      # HLW publishes it
                 "ECB": None,      # HLW publishes it
                 "BOE": 0.75}      # no HLW UK estimate exists — pure assumption


@dataclass
class InputProvenance:
    """Where each number came from, so the page can mark assumptions honestly."""
    sources: dict = field(default_factory=dict)      # field -> human string
    assumed: list = field(default_factory=list)      # fields that are NOT measured
    stale: list = field(default_factory=list)        # fields whose source lags badly
    missing: list = field(default_factory=list)


def inputs_from_data(bank: str, *, nairu: float | None = None,
                     rstar: float | None = None, use_core: bool = True,
                     when=None) -> tuple[RuleInputs, InputProvenance]:
    """Build a RuleInputs for one bank from the free data layer.

    Centralises three things that are easy to get wrong at a call site:

      1. CBO series are PROJECTIONS — `latest_actual` is used throughout, never `latest`.
      2. The first-difference rule needs the unemployment gap from four quarters ago, so
         that is looked up from history here rather than left unavailable.
      3. Where a bloc publishes no r* or NAIRU, the fallback is recorded in `provenance
         .assumed` so the page can flag it instead of passing it off as measured.
    """
    from src import macrodata

    bank = bank.upper()
    prov = InputProvenance()
    data = macrodata.BLOC_INPUTS[bank]()
    when = when or date.today()

    def pick(key, label):
        s = data.get(key)
        if s is None or not getattr(s, "ok", False) or not s.obs:
            prov.missing.append(label)
            return None, None
        hit = s.asof(when)
        if hit is None:
            prov.missing.append(label)
            return None, None
        prov.sources[label] = f"{s.source} · {s.title} ({hit[0]})"
        if getattr(s, "stale_note", ""):
            prov.stale.append(label)
        return hit[1], s

    infl, _ = pick("core_infl" if use_core else "headline_infl", "inflation")
    if infl is None:                       # fall back to headline rather than give up
        infl, _ = pick("headline_infl", "inflation")
    policy, _ = pick("policy", "policy rate")
    u, u_series = pick("unemp", "unemployment")

    # r*
    rs, _ = (None, None)
    if rstar is None:
        rs, _ = pick("rstar", "r*")
    if rstar is not None:
        rs = rstar
        prov.sources["r*"] = "user input"
        prov.assumed.append("r*")
    elif rs is None:
        rs = DEFAULT_RSTAR.get(bank) or 0.75
        prov.sources["r*"] = f"assumed {rs:.2f}% — no published estimate for this bloc"
        prov.assumed.append("r*")
        if "r*" in prov.missing:
            prov.missing.remove("r*")

    # NAIRU
    nr = nairu
    if nr is not None:
        prov.sources["NAIRU"] = "user input"
        prov.assumed.append("NAIRU")
    else:
        nr, _ = pick("nairu", "NAIRU")
        if nr is None:
            nr = DEFAULT_NAIRU.get(bank)
            if nr is not None:
                prov.sources["NAIRU"] = f"assumed {nr:.2f}% — not published for this bloc"
                prov.assumed.append("NAIRU")
                # It is substituted, so it is an assumption, not a hole. Reporting it as
                # both would show the page two contradictory warnings for one field.
                if "NAIRU" in prov.missing:
                    prov.missing.remove("NAIRU")

    # Output gap: prefer the unemployment form; fall back to a published gap series.
    out_gap = None
    if u is None or nr is None:
        g, _ = pick("gap", "output gap")
        out_gap = g

    # Lagged unemployment gap for the first-difference rule.
    gap_lag4 = None
    lag_date = date(when.year - 1, when.month, 1)
    if u_series is not None and nr is not None:
        prior = u_series.asof(lag_date)
        if prior is not None:
            nairu_series = data.get("nairu")
            nr_then = nr
            if nairu_series is not None and getattr(nairu_series, "ok", False):
                hit = nairu_series.asof(lag_date)
                if hit:
                    nr_then = hit[1]
            gap_lag4 = nr_then - prior[1]

    x = RuleInputs(bank=bank, infl=infl if infl is not None else 2.0,
                   target=data.get("target", 2.0), rstar=rs if rs is not None else 0.75,
                   unemp=u, nairu=nr, output_gap=out_gap,
                   policy_rate=policy if policy is not None else 0.0,
                   prev_policy_rate=policy, gap_lag4=gap_lag4,
                   infl_label="core inflation" if use_core else "headline inflation")
    return x, prov


# ── validation against the Cleveland Fed's published calculation ────────────────────
# We check our arithmetic against a live third party rather than a hardcoded snapshot.
# The Cleveland Fed publishes its own seven-rule calculation as a spreadsheet — the
# parameters, the forecasts feeding it, and the resulting prescriptions. Feeding our
# engine their inputs and comparing to their outputs is a real test: it fails when a
# coefficient drifts, and it keeps testing against fresh numbers every month.
#
# Their rule definitions, recovered by reconciling the spreadsheet arithmetic:
#     Taylor (1993)            r* + π + 0.5(π − π*) + 0.5·gap      on HEADLINE PCE
#     Taylor (1999) w/ core    r* + π + 0.5(π − π*) + 1.0·gap      on CORE PCE
#                              — identical to our balanced-approach rule
#     Inertial                 ρ·i(t−1) + (1 − ρ)·[Taylor (1999) core],  ρ = 0.8
#
# We validate against the CBO block specifically, because CBO publishes an OUTPUT GAP
# directly. The SPF block requires converting an unemployment gap through Cleveland's own
# Okun coefficient, which reproduces to ~2bp rather than exactly — close enough to confirm
# but not clean enough to assert on.
CLEVELAND_SOURCE = "Congressional Budget Office"


def validate_against_cleveland(tol: float = 0.01) -> dict:
    """Reproduce the Cleveland Fed's own rule prescriptions from their own inputs.

    Returns {'ok', 'checks': [...], 'asof', 'reason'}. `ok` is False if the data could not
    be fetched OR if any check fails — this is meant to be surfaced on the Data-health
    board, not quietly swallowed."""
    from src import macrodata

    blob = macrodata.cleveland_policy_rules()
    if not blob.get("ok"):
        return {"ok": False, "checks": [], "asof": "",
                "reason": blob.get("reason", "Cleveland data unavailable")}

    p = blob["params"]
    fc = blob["forecasts"].get(CLEVELAND_SOURCE, {})
    rules = blob["rules"].get(CLEVELAND_SOURCE, {})
    rstar, target, rho = p.get("rstar"), p.get("target", 2.0), p.get("rho", 0.8)
    if rstar is None:
        return {"ok": False, "checks": [], "asof": blob.get("asof", ""),
                "reason": "Cleveland parameter sheet has no r*"}

    def series(name_start: str) -> dict:
        for k, v in fc.items():
            if k.lower().startswith(name_start.lower()):
                return v
        return {}

    headline = series("PCE inflation (Y/Y")
    core = series("Core PCE inflation")
    gaps = series("Output gap")
    their_taylor = rules.get("Taylor (1993) rule", {})
    their_t99 = rules.get("Core inflation in Taylor (1999) rule", {})
    their_inertial = rules.get("Inertial rule", {})

    # The first quarter of the table is a SEED, not a prescription: the spreadsheet puts
    # the actual funds rate there and every rule reports the identical value. Comparing
    # against it would fail every rule by ~100-190bp for no reason. Detect it by that
    # signature (all rules equal) rather than assuming it is always column one.
    seed_quarters = set()
    if rules:
        by_q: dict[float, set] = {}
        for vals in rules.values():
            for q, v in vals.items():
                by_q.setdefault(q, set()).add(round(v, 6))
        seed_quarters = {q for q, vs in by_q.items() if len(vs) == 1 and len(rules) > 1}

    checks = []

    def add(q, rule_key, ours, theirs):
        if ours is None or theirs is None or q in seed_quarters:
            return
        checks.append({"quarter": q, "rule": rule_key, "ours": round(ours, 4),
                       "theirs": round(theirs, 4), "diff_bp": round((ours - theirs) * 100, 2),
                       "ok": abs(ours - theirs) <= tol})

    for q in sorted(gaps):
        gap = gaps.get(q)
        if gap is None:
            continue
        if q in headline and q in their_taylor:
            x = RuleInputs(infl=headline[q], target=target, rstar=rstar, output_gap=gap)
            add(q, "taylor93", taylor93(x).prescribed, their_taylor[q])
        if q in core and q in their_t99:
            x = RuleInputs(infl=core[q], target=target, rstar=rstar, output_gap=gap)
            add(q, "balanced (=Taylor 1999 core)", balanced(x).prescribed, their_t99[q])

    # Inertial chains off the PREVIOUS quarter's inertial prescription, seeded by the
    # actual funds rate in the first quarter of the table.
    qs = sorted(set(core) & set(gaps) & set(their_inertial))
    for i, q in enumerate(qs):
        if i == 0:
            continue
        prev_q = qs[i - 1]
        prev = their_inertial.get(prev_q)
        if prev is None:
            continue
        x = RuleInputs(infl=core[q], target=target, rstar=rstar, output_gap=gaps[q])
        t99 = balanced(x).prescribed
        if t99 is None:
            continue
        add(q, f"inertial (rho={rho})", rho * prev + (1 - rho) * t99, their_inertial[q])

    return {"ok": bool(checks) and all(c["ok"] for c in checks), "checks": checks,
            "asof": blob.get("asof", ""),
            "reason": "" if checks else "no overlapping quarters to check"}


if __name__ == "__main__":
    res = validate_against_cleveland()
    print(f"Cleveland Fed cross-check — {res.get('asof', '')}")
    if not res["checks"]:
        print("  UNAVAILABLE:", res.get("reason"))
    for c in res["checks"]:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['quarter']:<8} "
              f"{c['rule']:<30} ours {c['ours']:>8.4f}  theirs {c['theirs']:>8.4f}  "
              f"{c['diff_bp']:+.2f}bp")
    print("OVERALL:", "PASS" if res["ok"] else "FAIL")
