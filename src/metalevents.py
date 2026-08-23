"""metalevents.py — the release study across the metals complex, and the cross-metal test.

goldevents is the engine and is metal-agnostic once parameterised: it takes two fix
series and a set of release dates. This module drives it across gold, platinum and
palladium, and adds the one test the per-metal study cannot do.

WHY A SEPARATE CROSS-METAL TEST

The per-metal study answers "does this metal move more than usual around this
release, against its own baseline". Run on all three it returns:

    release             GOLD           PLATINUM       PALLADIUM
    Employment report   1.67x  SIG     1.11x          1.00x

It is tempting to read that as "gold responds and the PGMs do not". That inference is
the significant-versus-not-significant fallacy: a result clearing a threshold and
another failing it does not establish that the two DIFFER. The difference has to be
tested directly, and `cross_metal_payroll_test()` does it — bootstrapping the gap
between the ratios so the answer comes with a confidence interval rather than a
comparison of asterisks.

    GOLD - PLATINUM    +0.456  95% CI [+0.140, +0.777]  p 0.0038
    GOLD - PALLADIUM   +0.539  95% CI [+0.225, +0.861]  p 0.0010

Both intervals exclude zero and both survive correction for the two comparisons, so
the claim is supported: gold's payroll premium is genuinely larger than the PGMs',
not merely differently labelled.

A NOTE ON WHY THE NUMBERS HERE DIFFER SLIGHTLY FROM THE PER-METAL STUDY

The per-metal ratios (1.67 / 1.11 / 1.00) use the year-and-weekday-matched,
release-free baseline that the client report requires. The bootstrap below uses a
simpler all-other-days baseline, because the matched baseline resamples strata and
does not bootstrap cleanly. That shifts the levels a little (1.60 / 1.14 / 1.06) and
does NOT affect the comparison: the same baseline definition is applied to all three
metals, so the difference is measured like for like. Quoting the two sets of numbers
interchangeably would be wrong, which is why both are stated.

WHAT THE RESULT MEANS

The ordering is monotone in how industrial the metal is — gold (monetary), platinum
(part investment, part autocatalyst), palladium (~80% autocatalyst) — and palladium's
1.00x is exact inertness. That is economically coherent rather than a curiosity: it
says the metals are different ASSETS, not one asset with different noise levels, and
a framework built for gold should not be pointed at the PGMs unmodified.

CLI:
    python src/metalevents.py --study      re-run the per-metal studies (slow, ~20 min)
    python src/metalevents.py --compare    the cross-metal bootstrap only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldstore as gs                                      # noqa: E402
import goldevents as ge                                     # noqa: E402

STORE = ROOT / "data" / "gold_store"

# Metals carrying an LBMA AM and PM fix, so the intraday release window exists.
# Silver is absent by construction — a single noon auction has no window.
STUDY_METALS = ("GOLD", "PLATINUM", "PALLADIUM")

SEED = 20260823
RECENT_FROM = "2016-01-01"


def study_path(metal: str) -> Path:
    return (STORE / "event_study.json" if metal == "GOLD"
            else STORE / f"event_study_{metal.lower()}.json")


def fix_ids(metal: str) -> tuple:
    return f"LBMA_{metal}_PM_USD", f"LBMA_{metal}_AM_USD"


def run_studies(metals=STUDY_METALS, draws: int = 20000) -> dict:
    """Re-run the per-metal release study. Slow — permutation tests dominate."""
    out = {}
    for m in metals:
        pm, am = fix_ids(m)
        out[m] = ge.compute(draws=draws, verbose=False, write=True,
                            pm=pm, am=am, out_file=str(study_path(m)))
        print(f"  {m:10s} {out[m]['sample']}  {out[m]['n_trading_days']} days")
    return out


def load_studies(metals=STUDY_METALS) -> dict:
    out = {}
    for m in metals:
        p = study_path(m)
        if p.exists():
            out[m] = json.loads(p.read_text(encoding="utf-8"))
    return out


def comparison_table(metals=STUDY_METALS) -> pd.DataFrame:
    """The per-metal ratios side by side, on the matched release-free baseline."""
    d = load_studies(metals)
    if not d:
        return pd.DataFrame()
    base = next(iter(d.values()))
    rows = []
    for key in base["results"]["am_to_pm_intraday"]:
        row = {"release": key.split(" (")[0]}
        for m, study in d.items():
            b = (study["results"]["am_to_pm_intraday"].get(key) or {}).get("recent") or {}
            row[m] = b.get("ratio_matched_clean") or b.get("ratio_matched")
            row[f"{m}_sig"] = bool(b.get("significant"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _abs_window(metal: str, start: str = RECENT_FROM) -> pd.Series:
    """|log(PM/AM)| — the intraday release window, as an absolute move."""
    pm_id, am_id = fix_ids(metal)
    pm, am = gs.get_series(pm_id), gs.get_series(am_id)
    both = pd.concat([am.rename("am"), pm.rename("pm")], axis=1, sort=False).dropna()
    both = both[(both > 0).all(axis=1)]
    s = np.log(both["pm"] / both["am"]).abs()
    return s[s.index >= pd.Timestamp(start)]


def cross_metal_payroll_test(metals=STUDY_METALS, draws: int = 20000,
                             reference: str = "GOLD",
                             start: str = RECENT_FROM) -> dict:
    """Bootstrap the DIFFERENCE in payroll-day response between metals.

    Comparing "significant" against "not significant" does not establish a
    difference. This resamples event days and baseline days independently for each
    metal, forms each metal's ratio on every draw, and reports the distribution of
    the gap against the reference metal.

    The baseline here is all non-payroll days rather than the year-and-weekday
    matched, release-free baseline the per-metal study uses, because strata do not
    resample cleanly. The same definition is applied to every metal, so the
    comparison is like for like even though the levels shift slightly.
    """
    rng = np.random.default_rng(SEED)
    scan = pd.bdate_range("1990-01-01", pd.Timestamp.today().normalize())
    pay, _audit = ge.release_dates("PAYROLLS", scan)

    per = {}
    for m in metals:
        a = _abs_window(m, start)
        if a.empty:
            continue
        mask = a.index.isin(pay)
        ev, base = a[mask].to_numpy(), a[~mask].to_numpy()
        if len(ev) < 30 or len(base) < 200:
            continue
        per[m] = {"ev": ev, "base": base,
                  "n_event": int(len(ev)), "n_base": int(len(base)),
                  "ratio": float(ev.mean() / base.mean())}
    if reference not in per:
        return {"error": f"no window data for {reference}"}

    def boot(d):
        e = rng.choice(d["ev"], (draws, len(d["ev"])), replace=True).mean(axis=1)
        b = rng.choice(d["base"], (draws, len(d["base"])), replace=True).mean(axis=1)
        return e / b

    ref_draw = boot(per[reference])
    comparisons = []
    others = [m for m in per if m != reference]
    for m in others:
        diff = ref_draw - boot(per[m])
        p = 2 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        comparisons.append({
            "vs": m,
            "diff": per[reference]["ratio"] - per[m]["ratio"],
            "ci_low": float(lo), "ci_high": float(hi), "p": p,
            # Bonferroni across the comparisons actually made.
            "survives_correction": bool(p < 0.05 / max(len(others), 1)),
        })
    return {
        "reference": reference, "start": start, "draws": draws,
        "n_payroll_releases": int(len(pay)),
        "ratios": {m: {k: v for k, v in d.items() if k not in ("ev", "base")}
                   for m, d in per.items()},
        "comparisons": comparisons,
        "baseline_note": ("all non-payroll days; the per-metal study uses a "
                          "year+weekday matched, release-free baseline, so levels "
                          "differ slightly from that table"),
    }


def verdict(res: dict) -> str:
    """Derived, never asserted."""
    cs = res.get("comparisons") or []
    if not cs:
        return "no comparison available"
    ref = res["reference"]
    beat = [c for c in cs if c["survives_correction"] and c["ci_low"] > 0]
    if len(beat) == len(cs):
        names = ", ".join(c["vs"].title() for c in beat)
        return (f"{ref.title()} responds to payrolls by more than {names}, and the "
                f"difference survives correction — the metals are different assets, "
                f"not one asset with different noise")
    if not beat:
        return (f"No metal's payroll response differs from {ref.title()} once the "
                f"difference is tested directly")
    names = ", ".join(c["vs"].title() for c in beat)
    return f"{ref.title()}'s payroll response exceeds {names} only"


def main() -> int:
    args = set(sys.argv[1:]) or {"--compare"}
    if "--study" in args:
        print("Per-metal release studies (slow):")
        run_studies()
    tab = comparison_table()
    if not tab.empty:
        print("\nRelease-window premium, matched release-free baseline, last 10y:\n")
        cols = ["release"] + [m for m in STUDY_METALS if m in tab.columns]
        show = tab[cols].copy()
        for m in STUDY_METALS:
            if m in show.columns:
                show[m] = [f"{v:.2f}x" + ("  SIG" if s else "     ")
                           for v, s in zip(tab[m], tab[f"{m}_sig"])]
        print(show.to_string(index=False))
    res = cross_metal_payroll_test()
    if "error" in res:
        print(f"\n{res['error']}")
        return 1
    print(f"\nCross-metal payroll test — bootstrap, {res['draws']:,} resamples, "
          f"from {res['start']}")
    print(f"  baseline: {res['baseline_note']}\n")
    for m, r in res["ratios"].items():
        print(f"  {m:11s} ratio {r['ratio']:.3f}   "
              f"({r['n_event']} payroll days vs {r['n_base']} others)")
    print()
    for c in res["comparisons"]:
        mark = "" if c["survives_correction"] else "   (fails correction)"
        print(f"  {res['reference']} - {c['vs']:11s} {c['diff']:+.3f}  "
              f"95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p={c['p']:.4f}{mark}")
    print(f"\n  {verdict(res)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
