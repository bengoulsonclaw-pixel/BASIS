"""metalsupply.py — is platinum/palladium pricing carry a SUPPLY signature gold lacks?

Track B of the metals work. The thesis being tested, stated so it can fail:

    Gold holds roughly sixty years of above-ground stock against annual mine supply,
    so a mine disruption cannot move it — which is one reason every macro driver
    tested against gold turned out to be contemporaneous rather than leading.
    Platinum's buffer is one to two years and ~70% of mine supply is South African;
    palladium is ~40% Russian. If supply concentration matters, the PGMs should carry
    a country-specific risk loading that gold and silver do not.

WHAT THIS IS NOT

It is not an event study on dated disruptions. That would need a verifiable list of
strike dates, shaft incidents and load-shedding escalations, and no free feed
supplies one: Stats SA sits behind bot protection, FRED's South African mining
production series was retired by the OECD and stops at 2023-10, and hand-typing
event dates from memory would be fabrication with statistics attached. What follows
uses a DAILY, VERIFIABLE, EXOGENOUS proxy instead — the currency of the producing
country — and is explicit that a proxy is what it is.

THE IDENTIFICATION PROBLEM, AND WHAT IS DONE ABOUT IT

The rand is not a clean South-Africa signal. It moves with the dollar, with EM risk
appetite generally, and with the commodity complex — including platinum itself. Used
raw it would mostly measure the dollar, and every metal would "load on South Africa".

So the country factor is the RESIDUAL of the currency on a global block: the dollar
index and an EM FX basket that deliberately excludes the country in question. What
survives is the part of the rand that is not the dollar and not EM-wide, which is as
close to idiosyncratically South African as free daily data allows.

WHAT THE TEST RETURNED — THE THESIS FAILS

Run on 2004-2026, the country loadings do NOT line up with supply exposure:

    metal       ZA factor        SA share of mine supply
    silver      -0.175 (t -3.5)  none
    platinum    -0.096 (t -3.0)  ~70%
    palladium   -0.083 (t -2.4)  ~35%
    gold        -0.034 (t -1.7)  ~2%

Silver has no South African supply concentration whatsoever and carries the LARGEST
loading, raw and volatility-normalised. That breaks the identification on its own:
whatever the residualised rand measures, it cannot be South African platinum supply,
because the metal with no exposure to it responds most.

The SIGN is wrong too. A positive factor is idiosyncratic rand weakness — the
condition under which mines struggle — and every metal FALLS. A supply disruption
would push platinum up. What this looks like instead is residual EM/commodity risk
appetite, which all four metals share and the higher-beta ones feel more.

(The rank correlation between annualised volatility and |loading| is 0.4 on four
observations, which is far too little to call beta THE explanation, so it is not
claimed. What is claimed is only the negative: the ordering does not match supply
and the sign does not match supply.)

The Russia factor produces nothing anywhere — palladium's loading is -0.003 (t -1.3)
despite Russia being ~40% of world supply. And nothing leads: across six
metal-country pairs and ten lags, the largest forecasting correlation is 0.037, which
is what sixty tests produce by chance.

So the conclusion of the gold work extends here rather than breaking: the thin
inventory buffer is real, but it does not show up as a tradeable supply signature in
the price. If PGM supply shocks move these metals, they do it through dated events
this proxy cannot see, and testing that needs an event list no free feed supplies.

NO PREDICTED SIGN

The supply story does not give one. South African stress disrupts supply, which lifts
platinum; but a weaker rand also cuts producers' dollar costs, which makes marginal
output economic and pushes the other way; and rand weakness usually travels with
EM risk-off, which hurts industrial demand. Three channels, opposing signs. So the
coefficient is reported, never assumed — the same discipline as PRIOR_SIGN in the
gold diagnostics, which is recorded and never used to flip a result.

TIMING

Metal returns are struck fix-to-fix on the LBMA benchmark (15:00 London). Yahoo's FX
close is a New York close, about seven hours later. For a CONTEMPORANEOUS sensitivity
that mismatch is acceptable — both span the same calendar day's news — and it is
stated rather than hidden. It would NOT be acceptable for a forecasting test, which
is why `lead_lag()` shifts the factor by whole days and never uses the same-day value.
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
import metals                                               # noqa: E402

# Country currency -> the EM basket used to strip out "EM generally". The country's
# OWN currency is never in its own control basket, which would regress it on itself.
COUNTRY_FX = {
    "ZA": {"fx": "USDZAR", "basket": ["USDBRL", "USDMXN", "USDINR", "USDTRY"]},
    "RU": {"fx": "USDRUB", "basket": ["USDBRL", "USDMXN", "USDINR", "USDTRY"]},
}

YAHOO = {"USDZAR": "ZAR=X", "USDRUB": "RUB=X", "USDBRL": "BRL=X",
         "USDMXN": "MXN=X", "USDINR": "INR=X", "USDTRY": "TRY=X",
         "DXY": "DX-Y.NYB"}

# Which country should matter for which metal, if the supply thesis holds. Recorded
# to make the test falsifiable — a loading on the WRONG country is evidence against
# the story, not a second finding.
EXPECTED_EXPOSURE = {
    "PLATINUM": ["ZA"],
    "PALLADIUM": ["RU", "ZA"],      # ~40% Russian, ~35% South African
    "GOLD": [],                     # SA is ~2% of world gold supply today
    "SILVER": [],                   # by-product of base metals, not SA/RU concentrated
}

# Palladium's Russian exposure has a datable regime break. Tested separately rather
# than averaged across, because a loading that only exists after 2022 is a different
# claim from one that held for twenty years.
SANCTIONS_FROM = "2022-02-24"


# ---------------------------------------------------------------------------
def fetch_fx(force: bool = False) -> pd.DataFrame:
    """Daily FX and the dollar index from Yahoo, wide."""
    import yfinance as yf
    out = {}
    for name, tkr in YAHOO.items():
        try:
            h = yf.Ticker(tkr).history(period="max", auto_adjust=False)["Close"]
            h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
            s = h[h > 0].dropna()
            if len(s):
                out[name] = s
        except Exception as e:
            print(f"  {name:8s} fetch failed: {type(e).__name__}: {e}")
    return pd.DataFrame(out).sort_index()


def ingest_fx(force: bool = False) -> int:
    """Land the FX block in the point-in-time store.

    lag=1, not 0. These are New York closes and the metals' returns are struck at the
    15:00 London fix, so a day-t FX close did not exist when the day-t metal return
    was measured. Stamping it lag=0 would hand any forecasting test seven hours of
    hindsight on its most influential regressor — the same defect that moved the
    gold 5-day hit rate from 55.3% to 52.5% when it was corrected there.
    """
    df = fetch_fx(force=force)
    n = 0
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        goldstore.register(f"FX_{col}", description=f"{col} daily close",
                           unit="rate", native_freq="daily", typical_lag_days=1,
                           bucket="macro", source_url="https://finance.yahoo.com",
                           published_at_approximated=False)
        w = goldstore.put(f"FX_{col}", s, source="Yahoo Finance", typical_lag_days=1)
        n += w
        print(f"  FX_{col:10s} +{w:6d} rows   {s.index.min().date()} -> {s.index.max().date()}")
    return n


# ---------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> tuple:
    """Coefficients, residuals and Newey-West standard errors."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    lag = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        A = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (A + A.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return b, resid, np.sqrt(np.maximum(np.diag(cov), 0))


def country_factor(country: str, fx: pd.DataFrame) -> pd.Series:
    """The part of a producer currency that is neither the dollar nor EM-wide.

    Returns daily log changes of the residual. A positive value means the currency
    weakened by more than the dollar and the EM complex explain — i.e. country-
    specific stress.
    """
    spec = COUNTRY_FX[country]
    cols = [spec["fx"], "DXY"] + [c for c in spec["basket"] if c != spec["fx"]]
    have = [c for c in cols if c in fx.columns]
    d = np.log(fx[have]).diff().dropna()
    if spec["fx"] not in d.columns or len(d) < 250:
        return pd.Series(dtype=float)
    controls = [c for c in have if c != spec["fx"]]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in controls])
    y = d[spec["fx"]].to_numpy()
    b, resid, _ = _ols(y, X)
    return pd.Series(resid, index=d.index, name=f"{country}_factor")


# ---------------------------------------------------------------------------
def sensitivity(metal: str, fx: pd.DataFrame, factors: dict,
                start: str = "2004-01-01") -> dict:
    """Contemporaneous loading of a metal on the global block plus country factors.

    The global block is the dollar and the metal's own EM/risk context; the country
    factors are added on top. A PGM-specific supply signature shows up as a
    significant country loading that gold and silver do not share.
    """
    px = metals.benchmark_series(metal)
    r = np.log(px[px > 0]).diff().dropna()
    d = pd.DataFrame({"y": r})
    d["dxy"] = np.log(fx["DXY"]).diff()
    for c, f in factors.items():
        d[f"{c}_factor"] = f
    d = d[d.index >= pd.Timestamp(start)].dropna()
    if len(d) < 500:
        return {"metal": metal, "insufficient": True, "n": len(d)}

    cols = [c for c in d.columns if c != "y"]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    b, resid, se = _ols(d["y"].to_numpy(), X)
    out = {"metal": metal, "n": int(len(d)),
           "span": f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}",
           "r2": float(1 - resid.var() / d["y"].var()), "loadings": {}}
    for i, c in enumerate(cols, start=1):
        out["loadings"][c] = {"beta": float(b[i]), "se": float(se[i]),
                              "t": float(b[i] / se[i]) if se[i] > 0 else float("nan")}
    return out


def lead_lag(metal: str, factor: pd.Series, max_lag: int = 10,
             start: str = "2004-01-01") -> pd.DataFrame:
    """Does the country factor LEAD the metal, or only move with it?

    This is the question that decides whether any of it is tradeable, and it is the
    question the gold work answered negatively for every macro driver. Lag 0 is the
    contemporaneous case and is reported but flagged: with a New York FX close
    against a 15:00 London fix, lag 0 already contains hours the metal could not have
    seen. Only lags >= 1 are honest forecasting evidence.
    """
    px = metals.benchmark_series(metal)
    r = np.log(px[px > 0]).diff().dropna()
    rows = []
    for L in range(0, max_lag + 1):
        d = pd.concat([r.rename("y"), factor.shift(L).rename("f")],
                      axis=1, sort=False).dropna()
        d = d[d.index >= pd.Timestamp(start)]
        if len(d) < 250:
            continue
        c = float(np.corrcoef(d["f"], d["y"])[0, 1])
        # Overlapping daily returns are not the issue here (lag-1 daily is clean),
        # but the t is still computed on the effective sample.
        t = c * np.sqrt(max(len(d) - 2, 1)) / np.sqrt(max(1 - c * c, 1e-12))
        rows.append({"lag": L, "corr": c, "t": t, "n": len(d),
                     "usable_as_forecast": L >= 1})
    return pd.DataFrame(rows)


def scan(start: str = "2004-01-01") -> dict:
    """Every metal against every country factor, corrected for the number tested."""
    fx = fetch_fx()
    factors = {c: country_factor(c, fx) for c in COUNTRY_FX}
    factors = {c: f for c, f in factors.items() if len(f)}
    results = [sensitivity(m, fx, factors, start) for m in metals.METALS]
    n_tests = sum(len(factors) for r in results if not r.get("insufficient"))
    return {"results": results, "factors": list(factors), "n_tests": n_tests,
            "bonferroni_t": float(abs(_two_sided_z(0.05 / max(n_tests, 1))))}


def _two_sided_z(p: float) -> float:
    """Normal quantile for a two-sided p, without scipy."""
    from math import sqrt
    # Acklam's inverse-normal approximation, adequate at these tail sizes.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    q = p / 2.0
    if q < 0.02425:
        r = sqrt(-2 * np.log(q))
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
            ((((dd[0]*r+dd[1])*r+dd[2])*r+dd[3])*r+1)
    r = q - 0.5
    s = r * r
    return -(((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
        (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


def main() -> int:
    out = scan()
    crit = out["bonferroni_t"]
    print("Precious metals — country supply-risk loadings\n")
    print(f"  Country factor = producer currency residualised on the dollar and an EM")
    print(f"  basket that excludes it. {out['n_tests']} loadings tested; |t| > {crit:.2f} "
          f"required after Bonferroni.\n")
    print(f"  {'metal':11s} {'n':>6s} {'R2':>6s} {'dxy':>17s} " +
          " ".join(f"{c + ' factor':>17s}" for c in out["factors"]))
    for r in out["results"]:
        if r.get("insufficient"):
            print(f"  {r['metal']:11s} insufficient ({r['n']} obs)")
            continue
        cells = []
        for c in ["dxy"] + [f"{c}_factor" for c in out["factors"]]:
            L = r["loadings"].get(c)
            cells.append(f"{L['beta']:+8.3f}(t{L['t']:+5.1f})" if L else " " * 17)
        print(f"  {r['metal']:11s} {r['n']:6d} {r['r2']:6.3f} " + " ".join(cells))

    print("\n  Expected exposure if the supply story holds:")
    for m, cs in EXPECTED_EXPOSURE.items():
        print(f"    {m:11s} {', '.join(cs) if cs else '(none — control)'}")

    print("\n  Does the country factor LEAD the metal? (lag 0 is not forecasting evidence)")
    fx = fetch_fx()
    for c in out["factors"]:
        f = country_factor(c, fx)
        for m in ("PLATINUM", "PALLADIUM", "GOLD"):
            ll = lead_lag(m, f)
            if ll.empty:
                continue
            fwd = ll[ll["usable_as_forecast"]]
            best = fwd.iloc[fwd["corr"].abs().argmax()]
            print(f"    {c} -> {m:10s} lag0 corr {ll.iloc[0]['corr']:+.3f}  |  "
                  f"best forecasting lag {int(best['lag'])}: "
                  f"corr {best['corr']:+.3f} (t {best['t']:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
