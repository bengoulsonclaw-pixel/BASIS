"""metalrv.py — relative value across the precious-metals complex.

The question this answers is Ben's: when do the metals go "out of line" with each
other, and is that a fact you can do anything with?

WHY THIS AND NOT ANOTHER DIRECTIONAL MODEL

The gold engine's finding was that no macro driver forecasts gold's direction at 5,
60 or 250 days — every robust relationship is CONTEMPORANEOUS. That result points
here rather than away: if the drivers move with the metal but not before it, then
whatever is left to model lives in how the metals move relative to EACH OTHER, where
there are real economic anchors that a single price does not have.

  gold/silver   silver's mine supply is ~70% a by-product of copper/lead/zinc, so it
                barely responds to the silver price. Both metals share the same
                monetary drivers, so the ratio strips those out and leaves silver's
                industrial demand against gold's reserve demand.
  plat/pall     the two substitute in autocatalysts. That is an engineering decision
                with a cost threshold, so the ratio has an anchor a purely financial
                spread does not — automakers reformulated when palladium reached a
                large premium.
  plat/gold     no substitution channel and no shared supply. Included precisely
                because it is the one people quote most and the one with the weakest
                economic story.

WHAT A RATIO ASSUMES, AND WHY THAT IS NOT GOOD ENOUGH

A log ratio log(A) - log(B) imposes a cointegrating coefficient of exactly 1: it
asserts a 1% move in B should be met by a 1% move in A. Nothing guarantees that.
Engle-Granger estimates the coefficient from the data instead and tests whether the
RESIDUAL is stationary, which is the actual definition of "these two move together".

Because that coefficient is estimated rather than assumed, the residual's ADF
statistic no longer follows the standard Dickey-Fuller distribution — the regression
has already minimised its variance, which biases the test toward finding
stationarity. MacKinnon's critical values for the estimated-coefficient case are
used, and they are materially more demanding than the standard ones.

TESTING DISCIPLINE, INHERITED FROM THE GOLD WORK

  * ADF is AUGMENTED. An unaugmented test on gold/silver gives -3.95 and looks
    conclusive; 21 lags take it to -3.23. The difference is serial correlation in the
    differences being attributed to the level.
  * Every statistic is corrected across the pairs tested. Three spreads is three
    chances to find something.
  * Every result is re-run on non-overlapping sub-periods. A relationship that only
    exists in one decade is a story about that decade.
  * Statistical cointegration is NOT a tradeable edge, and the two are reported
    separately. `signal_value()` asks the only question that matters commercially:
    given the spread's z-score today, what happened NEXT? It is walk-forward, it
    averages over every sampling phase, and it is scored against doing nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import metals                                               # noqa: E402

# MacKinnon (2010) asymptotic critical values, Engle-Granger residual ADF, constant
# and no trend, TWO variables (one estimated cointegrating coefficient). These are
# the right thresholds when beta is fitted; the standard DF values (-3.43/-2.86)
# would over-reject because the regression has already minimised residual variance.
EG_CRITICAL = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}

# Plain ADF criticals, for the ratio (beta imposed, nothing estimated).
ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}

DEFAULT_LAGS = 21          # one trading month
PAIRS = [("GOLD", "SILVER"), ("PLATINUM", "PALLADIUM"), ("PLATINUM", "GOLD")]


# ---------------------------------------------------------------------------
# the statistics
# ---------------------------------------------------------------------------
def adf(series: pd.Series, lags: int = DEFAULT_LAGS) -> dict:
    """Augmented Dickey-Fuller with a constant. Returns t, half-life and n.

    The lag terms exist to absorb serial correlation in the differences. Without
    them that correlation is loaded onto the level coefficient and the test says
    "mean-reverting" about a series that merely has momentum in its increments.
    """
    s = pd.Series(series).dropna()
    d = s.diff().dropna()
    cols = {"lvl": s.shift(1)}
    for L in range(1, lags + 1):
        cols[f"d{L}"] = d.shift(L)
    frame = pd.concat([d.rename("y"), pd.DataFrame(cols)], axis=1,
                      sort=False).dropna()
    if len(frame) < lags + 30:
        return {"t": float("nan"), "half_life": float("nan"), "n": len(frame)}
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[c].to_numpy() for c in cols])
    y = frame["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt((resid @ resid / dof) * np.linalg.pinv(X.T @ X)[1, 1])
    gamma = beta[1]
    return {"t": float(gamma / se),
            "half_life": float(np.log(2) / -gamma) if gamma < 0 else float("inf"),
            "n": int(len(frame)), "gamma": float(gamma)}


def engle_granger(a: pd.Series, b: pd.Series, lags: int = DEFAULT_LAGS) -> dict:
    """Cointegrate log(a) on log(b) with a constant; ADF the residual.

    Engle-Granger is NOT symmetric — regressing a on b and b on a give different
    residuals and different statistics — so the caller gets both and the honest
    verdict is the weaker of the two.
    """
    la, lb = np.log(a.dropna()), np.log(b.dropna())
    idx = la.index.intersection(lb.index)
    la, lb = la.reindex(idx), lb.reindex(idx)
    X = np.column_stack([np.ones(len(lb)), lb.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, la.to_numpy(), rcond=None)
    resid = pd.Series(la.to_numpy() - X @ coef, index=idx)
    out = adf(resid, lags)
    out.update({"alpha": float(coef[0]), "beta": float(coef[1]),
                "resid": resid, "n_obs": int(len(idx))})
    return out


def _verdict(t: float, crit: dict, n_tests: int = 1) -> str:
    """Significance with a Bonferroni-style tightening for the number of pairs.

    Correcting a critical VALUE is not exact — the right correction is on the
    p-value — but the ADF p-surface is not available here, so the level is divided
    and the next tabulated threshold used. It errs strict, which is the direction to
    err when three spreads are being scanned for one story.
    """
    for level in (0.01, 0.05, 0.10):
        if t < crit[level] and level <= 0.10 / max(n_tests, 1):
            return f"stationary at {int(level * 100)}%"
    if t < crit[0.10]:
        return "borderline — fails the multiple-comparison correction"
    return "not stationary"


def pair_report(m1: str, m2: str, panel: pd.DataFrame | None = None,
                lags: int = DEFAULT_LAGS, n_tests: int = 1) -> dict:
    """Everything worth knowing about one pair: ratio, cointegration, stability."""
    p = panel if panel is not None else metals.panel()
    d = p[[m1, m2]].dropna()
    ratio = np.log(d[m1] / d[m2]).rename(f"{m1}/{m2}")

    r_adf = adf(ratio, lags)
    eg_ab = engle_granger(d[m1], d[m2], lags)
    eg_ba = engle_granger(d[m2], d[m1], lags)
    # Engle-Granger is direction-dependent; report the WEAKER side. Quoting the
    # better of two orderings is picking the winner of a two-horse race you ran
    # yourself.
    weaker = eg_ab if eg_ab["t"] > eg_ba["t"] else eg_ba

    eras = []
    for lo, hi in ((1990, 2002), (2002, 2014), (2014, 2027)):
        seg = ratio[(ratio.index.year >= lo) & (ratio.index.year < hi)]
        if len(seg) > 500:
            eras.append({"period": f"{lo}-{hi - 1}", "t": adf(seg, lags)["t"],
                         "n": len(seg)})

    lvl = float(np.exp(ratio.iloc[-1]))
    return {
        "pair": f"{m1}/{m2}", "m1": m1, "m2": m2,
        "n_obs": int(len(d)),
        "span": f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}",
        "level": lvl,
        "pctile": float((ratio <= ratio.iloc[-1]).mean() * 100),
        "median": float(np.exp(ratio.median())),
        # the ratio, beta imposed at 1
        "ratio_adf_t": r_adf["t"], "ratio_half_life": r_adf["half_life"],
        "ratio_verdict": _verdict(r_adf["t"], ADF_CRITICAL, n_tests),
        # the fitted cointegrating relationship
        "eg_beta": eg_ab["beta"], "eg_t_ab": eg_ab["t"], "eg_t_ba": eg_ba["t"],
        "eg_t": weaker["t"], "eg_half_life": weaker["half_life"],
        "eg_verdict": _verdict(weaker["t"], EG_CRITICAL, n_tests),
        # does it hold across eras?
        "eras": eras,
        "era_stable": bool(eras) and all(e["t"] < ADF_CRITICAL[0.10] for e in eras),
        "_ratio": ratio, "_resid": weaker["resid"],
    }


def scan(pairs=None, panel: pd.DataFrame | None = None) -> list:
    """Every pair, each corrected for how many were tested."""
    pairs = pairs or PAIRS
    p = panel if panel is not None else metals.panel()
    return [pair_report(a, b, p, n_tests=len(pairs)) for a, b in pairs]


# ---------------------------------------------------------------------------
# does knowing it is stretched actually help?
# ---------------------------------------------------------------------------
def signal_value(spread: pd.Series, horizon: int = 60, window: int = 750,
                 entry_z: float = 2.0) -> dict:
    """Walk-forward: does a stretched spread predict its own subsequent move?

    Cointegration is a statement about a sample. This is the commercial question, and
    it is asked the way the gold harness asks everything:

      * the z-score uses a TRAILING window only — no full-sample mean or sd, which
        would let today's spread know where it eventually ended up;
      * the forward move is measured over `horizon` days and sampled every `horizon`
        days so no two observations share a window, and the result is averaged over
        ALL `horizon` phases rather than the one that happens to start at row 0;
      * the comparison is against doing nothing, not against a coin flip.

    A positive `mean_reversion` means a stretched spread narrowed on average, which
    is what a mean-reverting spread is supposed to do.
    """
    s = pd.Series(spread).dropna()
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    fwd = s.shift(-horizon) - s
    d = pd.concat([z.rename("z"), fwd.rename("fwd")], axis=1).dropna()
    if len(d) < horizon * 3:
        return {"insufficient": True, "n": len(d)}

    per_phase = []
    for phase in range(horizon):
        seg = d.iloc[phase::horizon]
        if len(seg) < 20:
            continue
        hit = seg[seg["z"].abs() >= entry_z]
        # A stretched spread should move AGAINST its own stretch: positive z (wide)
        # should be followed by a fall. -sign(z)*fwd is positive when it does.
        rev = (-np.sign(hit["z"]) * hit["fwd"]) if len(hit) else pd.Series(dtype=float)
        per_phase.append({
            "corr": float(np.corrcoef(seg["z"], seg["fwd"])[0, 1])
                    if seg["z"].std() > 0 else np.nan,
            "n_signals": int(len(hit)),
            "mean_reversion": float(rev.mean()) if len(rev) else np.nan,
            "hit_rate": float((rev > 0).mean()) if len(rev) else np.nan,
        })
    if not per_phase:
        return {"insufficient": True, "n": len(d)}

    def _m(k):
        v = [p[k] for p in per_phase if p[k] == p[k]]
        return float(np.mean(v)) if v else float("nan")

    def _spread_of(k):
        v = [p[k] for p in per_phase if p[k] == p[k]]
        return (float(min(v)), float(max(v))) if v else (float("nan"),) * 2

    hits = [p["hit_rate"] for p in per_phase if p["hit_rate"] == p["hit_rate"]]
    return {
        "horizon": horizon, "entry_z": entry_z, "n": int(len(d)),
        "z_vs_forward_corr": _m("corr"),
        "signals_per_phase": _m("n_signals"),
        "mean_reversion": _m("mean_reversion"),
        "hit_rate": _m("hit_rate"),
        "hit_rate_phase_spread": _spread_of("hit_rate"),
        # A hit rate that swings 30 points on the arbitrary sampling offset is not a
        # hit rate, and the spread travels with the number so it cannot be quoted
        # alone.
        "phases_scored": len(hits),
    }


def signal_by_era(spread: pd.Series, horizon: int = 60,
                  entry_z: float = 2.0) -> list:
    """`signal_value` on non-overlapping 12-year blocks.

    This turned out to be the decisive test, not cointegration. Gold/silver looks
    mean-reverting on the full sample and is borderline-significant, but the effect
    is almost entirely post-2014: a stretched ratio narrowed 87% of the time in
    2014-2026 and only 42% in the 1990s, which is worse than a coin flip. A full
    sample that averages those two is describing neither.

    Platinum/palladium is the opposite and the more useful finding: it fails every
    stationarity test AND narrows less than half the time in all three eras, with a
    positive z-to-forward correlation throughout. That is mild momentum, stable over
    36 years, in the spread an RV desk would most naturally assume reverts. There is
    a mechanism for it — autocatalyst substitution needs a platform redesign, so the
    demand response to a stretched ratio arrives in years, not quarters, and the
    spread keeps going in the meantime.
    """
    out = []
    for lo, hi in ((1990, 2002), (2002, 2014), (2014, 2027)):
        seg = spread[(spread.index.year >= lo) & (spread.index.year < hi)]
        if len(seg) < 800:
            continue
        sv = signal_value(seg, horizon=horizon, window=500, entry_z=entry_z)
        if sv.get("insufficient"):
            continue
        out.append({"period": f"{lo}-{hi - 1}", "corr": sv["z_vs_forward_corr"],
                    "hit_rate": sv["hit_rate"],
                    "signals": sv["signals_per_phase"]})
    return out


def signal_verdict(eras: list) -> str:
    """One phrase, DERIVED — never a hardcoded claim about live numbers."""
    if len(eras) < 2:
        return "too short to judge"
    hits = [e["hit_rate"] for e in eras]
    corrs = [e["corr"] for e in eras]
    if all(h > 0.55 for h in hits):
        return "narrows from stretched in every era"
    if all(h < 0.50 for h in hits) and all(c > 0 for c in corrs):
        return "TRENDS from stretched in every era — fading it has not worked"
    if hits[-1] > 0.60 and hits[0] < 0.50:
        return "reverts recently, did not historically — one era, not a regularity"
    return "no consistent behaviour from stretched"


def main() -> int:
    rows = scan()
    print(f"Precious-metals relative value — {rows[0]['span']}\n")
    print(f"  {'pair':16s} {'level':>8s} {'pctile':>7s} {'ratio t':>8s} "
          f"{'EG beta':>8s} {'EG t':>7s} {'half-life':>10s}")
    for r in rows:
        hl = r["eg_half_life"]
        hls = f"{hl:8.0f} d" if np.isfinite(hl) and hl < 1e5 else "       -"
        print(f"  {r['pair']:16s} {r['level']:8.2f} {r['pctile']:6.1f}% "
              f"{r['ratio_adf_t']:8.2f} {r['eg_beta']:8.3f} {r['eg_t']:7.2f} {hls}")
    print(f"\n  Critical values — ratio (beta imposed): {ADF_CRITICAL[0.05]:.2f} at 5%;"
          f"  cointegration (beta fitted): {EG_CRITICAL[0.05]:.2f} at 5%")
    print(f"  Corrected for {len(rows)} pairs tested.\n")

    for r in rows:
        print(f"  {r['pair']}")
        print(f"    ratio          {r['ratio_verdict']}")
        print(f"    cointegration  {r['eg_verdict']}   "
              f"(a on b {r['eg_t_ab']:.2f}, b on a {r['eg_t_ba']:.2f} — weaker shown)")
        print("    by era         " + "   ".join(
            f"{e['period']} t={e['t']:.2f}" for e in r["eras"])
            + ("   STABLE" if r["era_stable"] else "   not stable across eras"))
        sv = signal_value(r["_ratio"])
        if sv.get("insufficient"):
            print("    tradeable?     insufficient data")
        else:
            lo, hi = sv["hit_rate_phase_spread"]
            print(f"    tradeable?     z-vs-forward corr {sv['z_vs_forward_corr']:+.3f}; "
                  f"at |z|>2, {sv['hit_rate']:.1%} narrowed "
                  f"(phases {lo:.1%}..{hi:.1%}), "
                  f"{sv['signals_per_phase']:.1f} signals per phase")
        eras = signal_by_era(r["_ratio"])
        if eras:
            print("    ...by era      " + "   ".join(
                f"{e['period']} {e['hit_rate']:.0%}" for e in eras)
                + f"   -> {signal_verdict(eras)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
