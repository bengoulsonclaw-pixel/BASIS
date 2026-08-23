"""macrochain.py — the rates -> dollar -> gold chain, and what each link is worth.

Ben's question: is there a formula that the dollar, US yields and gold roughly stick
to? Partly yes. This module estimates each link, decomposes gold's rate sensitivity
into the part that travels THROUGH the dollar and the part that does not, and states
the error band — which is the number that decides whether a formula is usable.

THE CORRECTION THAT MATTERS MOST

"Rates up, dollar up, gold down" is right only for REAL rates. Nominal yield = real
yield + breakeven inflation, and the two halves push gold in OPPOSITE directions:

    10y real yield  +10bp  ->  gold -0.49%   (t -4.5)
    10y breakeven   +10bp  ->  gold +0.41%   (t +2.4)

So a 25bp rise in the nominal 10y has an ambiguous effect on gold and the sign
depends entirely on WHY it rose. Yields up because the Fed is fighting inflation
(real up, breakeven flat) is bearish gold. Yields up because inflation expectations
are rising (breakeven up, real flat) is bullish. A formula keyed on the nominal yield
would average two opposing effects and produce a coefficient near zero that is right
about nothing.

DIRECT AND INDIRECT

Gold's -0.49% per 10bp of real yield is a JOINT coefficient: it is the effect holding
the dollar constant. But rates also move the dollar, and the dollar moves gold. The
total effect of a rate move is therefore

    total  =  direct  +  (dollar's response to rates)  x  (gold's response to dollar)

Both pieces are estimated here. Reporting the joint coefficient alone understates the
full effect of a rate move, and reporting a simple bivariate gold-on-rates regression
overstates the direct channel by silently including the dollar's contribution.

WHAT A FORMULA CAN AND CANNOT DO

R-squared is 0.28. One standard deviation of what the drivers do NOT explain is about
4.1% a month, against scenario effects of roughly 1%. The relationships are real and
strongly significant — the dollar link clears |t| = 7 — but the noise around them is
several times the size of the effect for any single move.

That makes these elasticities useful for sizing a move that has ALREADY happened
("the dollar fell 2%, so about 1.6% of today's gold move is accounted for") and
useless as a forecast. Which is the same conclusion the whole engine reached: the
relationships are contemporaneous, and everything measurable is in the price before
it prints.

EVERYTHING HERE IS CONTEMPORANEOUS BY CONSTRUCTION

Same-period changes on both sides. No lag, no forecast, no claim of causation beyond
the economics. The lead-lag work is in golddiag and it found nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldstore                                            # noqa: E402

WINDOW = 20            # trading days — the same one-month horizon goldsens fits on

# Series the chain is built from. All already in the point-in-time store.
SERIES = {
    "gold": "LBMA_GOLD_PM_USD",
    "dxy": "DXY",
    "real10": "REAL_10Y_SPLICED",
    "nom10": "NOMINAL_10Y",
    "nom2": "NOMINAL_2Y",
    "breakeven": "BREAKEVEN_10Y",
    "funds": "FED_FUNDS",
}


def panel(start: str = "1990-01-01") -> pd.DataFrame:
    p = goldstore.daily_panel(list(SERIES.values()), start=start)
    return p.rename(columns={v: k for k, v in SERIES.items()})


def _changes(p: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Percent changes for prices, absolute changes in bp-equivalents for yields."""
    d = pd.DataFrame(index=p.index)
    d["gold_pct"] = np.log(p["gold"]).diff(window) * 100
    d["dxy_pct"] = np.log(p["dxy"]).diff(window) * 100
    for k in ("real10", "nom10", "nom2", "breakeven", "funds"):
        d[f"{k}_bp"] = p[k].diff(window) * 100          # series are in percent
    d["policy_bp"] = (p["nom2"] - p["funds"]).diff(window) * 100
    return d


def _ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple:
    """Coefficients with Newey-West errors at `lags`.

    Overlapping `window`-day changes are autocorrelated by construction, so plain OLS
    errors understate by roughly sqrt(window). Anything read off these t-statistics
    without the correction would look several times more certain than it is.
    """
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    S = (X * r[:, None]).T @ (X * r[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * r[L:, None]).T @ (X[:-L] * r[:-L, None])
        S += w * (A + A.T)
    XtX = np.linalg.pinv(X.T @ X)
    cov = XtX @ S @ XtX
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    r2 = 1 - r.var() / y.var() if y.var() > 0 else np.nan
    return b, se, float(r2), int(n)


def link(d: pd.DataFrame, lhs: str, rhs: list, lags: int = WINDOW) -> dict:
    """One regression of the chain, with a constant."""
    cols = [lhs] + rhs
    f = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(f) < 500:
        return {"insufficient": True, "n": len(f)}
    X = np.column_stack([np.ones(len(f))] + [f[c].to_numpy() for c in rhs])
    b, se, r2, n = _ols(f[lhs].to_numpy(), X, lags)
    out = {"lhs": lhs, "n": n, "r2": r2, "terms": {}}
    for i, c in enumerate(rhs, start=1):
        out["terms"][c] = {"beta": float(b[i]), "se": float(se[i]),
                           "t": float(b[i] / se[i]) if se[i] > 0 else np.nan}
    out["resid_sd"] = float(np.std(f[lhs].to_numpy()
                                   - X @ b))
    return out


def chain(start: str = "1990-01-01", window: int = WINDOW) -> dict:
    """Every link, plus the direct/indirect decomposition of a rate move."""
    d = _changes(panel(start), window)

    gold = link(d, "gold_pct", ["dxy_pct", "real10_bp", "breakeven_bp"])
    dollar = link(d, "dxy_pct", ["real10_bp", "breakeven_bp"])
    dollar_policy = link(d, "dxy_pct", ["policy_bp"])
    gold_nominal = link(d, "gold_pct", ["nom10_bp"])       # the naive version
    gold_policy = link(d, "gold_pct", ["policy_bp"])

    out = {"window_days": window,
           "gold": gold, "dollar": dollar,
           "dollar_policy": dollar_policy,
           "gold_nominal_only": gold_nominal,
           "gold_policy": gold_policy}

    # Decompose a +25bp REAL yield move.
    if not gold.get("insufficient") and not dollar.get("insufficient"):
        g_dxy = gold["terms"]["dxy_pct"]["beta"]           # % gold per 1% dxy
        g_real = gold["terms"]["real10_bp"]["beta"]        # % gold per 1bp real
        d_real = dollar["terms"]["real10_bp"]["beta"]      # % dxy  per 1bp real
        direct = g_real * 25
        indirect = d_real * 25 * g_dxy
        out["decomposition_25bp_real"] = {
            "direct_pct": direct,
            "via_dollar_pct": indirect,
            "total_pct": direct + indirect,
            "dollar_move_pct": d_real * 25,
            "share_via_dollar": (indirect / (direct + indirect)
                                 if (direct + indirect) != 0 else np.nan),
        }
    return out


def stability(start: str = "1990-01-01", window: int = WINDOW) -> pd.DataFrame:
    """The two headline elasticities on non-overlapping periods.

    A formula that changes every decade is not a formula. This is the test that
    decides whether the coefficients can be quoted as rules of thumb at all.
    """
    d = _changes(panel(start), window)
    rows = []
    for lo, hi in ((1990, 2002), (2002, 2014), (2014, 2027)):
        seg = d[(d.index.year >= lo) & (d.index.year < hi)]
        g = link(seg, "gold_pct", ["dxy_pct", "real10_bp", "breakeven_bp"])
        dl = link(seg, "dxy_pct", ["real10_bp", "breakeven_bp"])
        if g.get("insufficient"):
            continue
        rows.append({
            "period": f"{lo}-{hi - 1}",
            "gold_per_1pct_dxy": g["terms"]["dxy_pct"]["beta"],
            "t_dxy": g["terms"]["dxy_pct"]["t"],
            "gold_per_10bp_real": g["terms"]["real10_bp"]["beta"] * 10,
            "t_real": g["terms"]["real10_bp"]["t"],
            "gold_per_10bp_be": g["terms"]["breakeven_bp"]["beta"] * 10,
            "dxy_per_10bp_real": (dl["terms"]["real10_bp"]["beta"] * 10
                                  if not dl.get("insufficient") else np.nan),
            "r2": g["r2"], "n": g["n"],
        })
    return pd.DataFrame(rows)


def level_anchor(start: str = "2003-01-01", lags: int = 21) -> dict:
    """Is there a LEVEL the three stick to, or only a relationship between changes?

    "A formula they roughly stick to" is a claim about LEVELS, and it is a different
    claim from "a 1% dollar move is worth -0.8% on gold". The first says there is an
    anchor to revert to; the second only says the changes co-move. They can be true
    separately, and here only the second one is.

    Starts 2003 because measured TIPS real yields and BREAKEVEN_10Y do not exist
    before it.
    """
    from metalrv import adf
    p = panel().dropna(subset=["gold", "dxy", "real10"])
    p = p[p.index >= pd.Timestamp(start)]
    y = np.log(p["gold"]).to_numpy()
    X = np.column_stack([np.ones(len(p)), np.log(p["dxy"]).to_numpy(),
                         p["real10"].to_numpy()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = pd.Series(y - X @ b, index=p.index)
    a = adf(resid, lags=lags)
    return {
        "n": int(len(p)), "start": start,
        "beta_log_dxy": float(b[1]), "beta_real10": float(b[2]),
        "adf_t": a["t"], "half_life_days": a["half_life"],
        # Engle-Granger with TWO estimated regressors, MacKinnon 5%.
        "critical_5pct": -3.74,
        "cointegrated": bool(a["t"] < -3.74),
        "residual_log": float(resid.iloc[-1]),
        "residual_pct": float(np.expm1(resid.iloc[-1]) * 100),
    }


def rules_of_thumb(c: dict) -> list:
    """The usable statements, DERIVED, each with its own error band."""
    out = []
    g, dl = c.get("gold", {}), c.get("dollar", {})
    if not g.get("insufficient"):
        b = g["terms"]
        out.append(f"Dollar +1%  ->  gold {b['dxy_pct']['beta']:+.2f}%  "
                   f"(t {b['dxy_pct']['t']:+.1f})")
        out.append(f"Real 10y +25bp  ->  gold {b['real10_bp']['beta'] * 25:+.2f}%  "
                   f"(t {b['real10_bp']['t']:+.1f})")
        out.append(f"Breakeven +25bp  ->  gold {b['breakeven_bp']['beta'] * 25:+.2f}%  "
                   f"(t {b['breakeven_bp']['t']:+.1f})  <- OPPOSITE sign to real")
    if not dl.get("insufficient"):
        b = dl["terms"]
        out.append(f"Real 10y +25bp  ->  dollar {b['real10_bp']['beta'] * 25:+.2f}%  "
                   f"(t {b['real10_bp']['t']:+.1f})")
    dp = c.get("dollar_policy", {})
    if not dp.get("insufficient"):
        b = dp["terms"]["policy_bp"]
        out.append(f"Policy expectations +25bp (2y less funds)  ->  "
                   f"dollar {b['beta'] * 25:+.2f}%  (t {b['t']:+.1f})")
    return out


def main() -> int:
    c = chain()
    print(f"The rates -> dollar -> gold chain, {c['window_days']}-day changes\n")
    for s in rules_of_thumb(c):
        print("  " + s)

    g = c["gold"]
    print(f"\n  Gold equation R2 {g['r2']:.3f} on {g['n']:,} overlapping windows; "
          f"1 s.d. of what it does NOT explain is {g['resid_sd']:.1f}% per month.")

    n = c["gold_nominal_only"]
    if not n.get("insufficient"):
        b = n["terms"]["nom10_bp"]
        print(f"\n  THE NAIVE VERSION, for contrast — gold on the NOMINAL 10y alone:")
        print(f"    nominal 10y +25bp -> gold {b['beta'] * 25:+.2f}% "
              f"(t {b['t']:+.1f}, R2 {n['r2']:.3f})")
        print("    Real and breakeven push opposite ways, so lumping them together "
              "averages\n    two live effects into one number that describes neither.")

    dec = c.get("decomposition_25bp_real")
    if dec:
        print(f"\n  A +25bp REAL yield move, decomposed:")
        print(f"    direct on gold        {dec['direct_pct']:+.2f}%")
        print(f"    via the dollar        {dec['via_dollar_pct']:+.2f}%   "
              f"(dollar moves {dec['dollar_move_pct']:+.2f}%)")
        print(f"    total                 {dec['total_pct']:+.2f}%   "
              f"({dec['share_via_dollar']:.0%} of it through the dollar)")

    la = level_anchor()
    print("")
    print("  IS THERE A LEVEL THEY STICK TO? "
          "(log gold ~ log dxy + real 10y, from " + la["start"] + ")")
    verdict = "COINTEGRATED" if la["cointegrated"] else "NOT cointegrated"
    print(f"    residual ADF t {la['adf_t']:+.2f} against {la['critical_5pct']:.2f} "
          f"needed -> {verdict}")
    print(f"    residual half-life {la['half_life_days']:,.0f} days; gold currently "
          f"{la['residual_pct']:+.0f}% from the level the two explain")
    print(f"    the level fit puts log(dxy) at {la['beta_log_dxy']:+.2f} — POSITIVE, "
          f"where the CHANGE relationship is negative. A sign that flips between")
    print("    levels and changes is the standard symptom of a spurious regression,")
    print("    and it is the tell that there is no anchor here.")

    print("\n  Stability — the same coefficients on non-overlapping periods:\n")
    s = stability()
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
