"""goldregime.py — Stage 4 pre-test: does ANY driver lead gold *within* a regime?

Why this exists before the regime model, not after
--------------------------------------------------
Stage 1 scanned every feature against gold at lags 0-60 and found eight relationships
clearing a multiple-comparison threshold, all of them COINCIDENT. I used that to argue
the regime layer would fail. That argument has a hole in it, and it is worth stating
plainly: **a full-sample correlation cannot detect a regime-dependent lead.** If a
driver leads gold positively in one state and negatively in another, the pooled
correlation is roughly zero and the Stage 1 scan reports nothing — which is precisely
the situation a regime model is built for.

So this re-asks the Stage 1 question *inside* each regime. It fits no model. If no
driver leads in any state, Stage 4 is dead and we will have shown it rather than
assumed it. If one does, Stage 4 is live and the spec's dynamic-coefficient model is
worth building.

The statistical trap this is designed to avoid
----------------------------------------------
Splitting into regimes MULTIPLIES the tests. 28 features x 61 lags x 4 states is 6,832
simultaneous comparisons, and at a naive 5% bar roughly 340 of them clear by
construction. Regime-splitting is otherwise a superb machine for manufacturing
discoveries: slice until something looks significant, then tell a story about the
slice. The Bonferroni threshold here counts EVERY cell including the regime dimension,
so a within-regime result has to be far stronger than a pooled one to survive — as it
should, because it was found after more looking.

Regimes are computable point-in-time
------------------------------------
Both splits use only trailing data, so a regime label at date t is one a trader could
have assigned on date t:

    VOL    trailing 60d realised vol above/below its own EXPANDING median
    TREND  gold above/below its trailing 200d average

Expanding, not full-sample: using the whole sample's median to label 1995 would decide
that year's regime using data from 2020.

CLI:  python src/goldregime.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import goldfeatures, goldstore  # noqa: E402
from src.golddiag import PRIOR_SIGN, r_threshold  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
OUT_FILE = STORE_DIR / "regime_pretest.json"

PRICE = "LBMA_GOLD_PM_USD"
MAX_LAG = 60
MIN_OBS = 400            # a regime with fewer usable days is not scored

# Features that are FUNCTIONS OF THE GOLD PRICE ITSELF. They cannot be evidence that
# "a driver leads gold" inside a price-defined regime, because the regime label and
# the feature are built from the same series.
#
# The sharpest case: the trend regime is `price > MA200`, and `gold_dist_200d` is
# `price/MA200 - 1`. Inside the uptrend subset that feature is CONSTRAINED POSITIVE by
# construction. Conditioning on a variable and then correlating the same variable with
# returns inside the conditioned subset manufactures exactly the result this pre-test
# was built to look for. Any lead they show is momentum plus a selection artefact, not
# a driver.
PRICE_DERIVED = {
    "gold_dist_200d", "gold_dist_50d", "gold_mom_12m_1m", "gold_fx_breadth",
    "gold_silver_ratio_z_5y", "gold_cpi_ratio_z_10y", "gold_m2_ratio_z_10y",
}


def label_regimes(price: pd.Series) -> pd.DataFrame:
    """Point-in-time regime labels: (vol state, trend state).

    Both use expanding statistics. A full-sample median would label 1995 using 2020
    data — the same lookahead the winsoriser was fixed for."""
    ret = np.log(price / price.shift(1))
    vol = ret.rolling(60, min_periods=40).std()
    vol_med = vol.expanding(min_periods=250).median()
    ma200 = price.rolling(200, min_periods=120).mean()
    out = pd.DataFrame({
        "vol": np.where(vol > vol_med, "high_vol", "low_vol"),
        "trend": np.where(price > ma200, "uptrend", "downtrend"),
    }, index=price.index)
    out.loc[vol.isna() | vol_med.isna(), "vol"] = None
    out.loc[ma200.isna(), "trend"] = None
    out["state"] = out["vol"].astype(str) + "/" + out["trend"].astype(str)
    out.loc[out["vol"].isna() | out["trend"].isna(), "state"] = None
    return out


def scan_within(feats: pd.DataFrame, ret: pd.Series, mask: pd.Series,
                max_lag: int = MAX_LAG) -> pd.DataFrame:
    """corr(feature[t], daily gold return[t+L]) restricted to dates where `mask`.

    The LAG is applied on the full series and only then restricted, so a lag never
    steps across a regime boundary into a different state's returns."""
    rows = []
    for f in feats.columns:
        x = feats[f]
        for lag in range(0, max_lag + 1):
            d = pd.concat([x, ret.shift(-lag)], axis=1)
            d = d[mask.reindex(d.index).fillna(False)].dropna()
            if len(d) < MIN_OBS:
                continue
            r = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
            if np.isfinite(r):
                rows.append({"feature": f, "lag": lag, "r": r, "n": len(d)})
    return pd.DataFrame(rows)


def run() -> dict:
    feats, _ = goldfeatures.load()
    panel = goldstore.daily_panel([PRICE], start=str(feats.index.min().date()))
    price = panel[PRICE].reindex(feats.index).ffill()
    ret = np.log(price / price.shift(1))

    reg = label_regimes(price)
    states = [s for s in reg["state"].dropna().unique()]

    per_state, all_rows = {}, []
    for st in states:
        mask = reg["state"] == st
        sc = scan_within(feats, ret, mask)
        if sc.empty:
            continue
        sc["state"] = st
        all_rows.append(sc)
        per_state[st] = int(mask.sum())
    if not all_rows:
        return {"error": "no regime had enough observations"}
    scan = pd.concat(all_rows, ignore_index=True)

    # Bonferroni over EVERY cell, including the regime dimension. Splitting the
    # sample is what makes this necessary: it multiplies the looks, and a threshold
    # that ignores that turns regime analysis into a discovery machine.
    n_tests = int(len(scan))
    n_med = int(scan["n"].median())
    thr_raw = r_threshold(n_med, 0.05)
    thr_adj = r_threshold(n_med, 0.05 / max(n_tests, 1))

    # EACH TEST AT ITS OWN n. Judging every cell at the median n is too lenient for
    # the small regimes and too strict for the large ones — and the small regimes are
    # exactly where a spurious result appears. The only external driver that looked
    # significant here (cot_mm_net_chg_4w, n=626, |r|=0.115) fails its own threshold
    # of 0.176. Same defect I fixed in golddiag and reintroduced.
    alpha_adj = 0.05 / max(n_tests, 1)
    scan["own_threshold"] = [r_threshold(int(n), alpha_adj) for n in scan["n"]]
    scan["clears_raw"] = scan["r"].abs() > thr_raw
    scan["clears_adj"] = scan["r"].abs() > scan["own_threshold"]
    scan["price_derived"] = scan["feature"].isin(PRICE_DERIVED)
    leading_all = scan[(scan["lag"] > 0) & scan["clears_adj"]]
    leading = leading_all[~leading_all["price_derived"]]
    coincident = scan[(scan["lag"] == 0) & scan["clears_adj"]]

    # Also: does any feature's PEAK move to a positive lag inside a regime?
    peaks = (scan.loc[scan.groupby(["state", "feature"])["r"]
                      .apply(lambda s: s.abs().idxmax())]
             .sort_values("r", key=abs, ascending=False))

    out = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "sample": {"from": str(feats.index.min().date()),
                   "to": str(feats.index.max().date()), "rows": int(len(feats))},
        "states": per_state,
        "n_tests": n_tests,
        "median_n_per_test": n_med,
        "threshold_raw_5pct": thr_raw,
        "threshold_bonferroni": thr_adj,
        "n_clearing_adjusted": int(scan["clears_adj"].sum()),
        "n_coincident_adjusted": int(len(coincident)),
        "n_LEADING_external": int(len(leading)),
        "n_LEADING_price_derived_excluded": int(len(leading_all) - len(leading)),
        "leading_detail": json.loads(leading.sort_values("r", key=abs, ascending=False)
                                     .head(40).to_json(orient="records")),
        "leading_excluded_as_circular": json.loads(
            leading_all[leading_all["price_derived"]]
            .sort_values("r", key=abs, ascending=False).head(20)
            .to_json(orient="records")),
        "peaks": json.loads(peaks.head(30).to_json(orient="records")),
        "verdict": ("STAGE 4 LIVE — an EXTERNAL driver leads gold inside a regime"
                    if len(leading) else
                    "STAGE 4 DEAD — no external driver leads gold in any regime, "
                    "at any lag, once each test is judged at its own sample size and "
                    "price-derived features are excluded as circular"),
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    o = run()
    if "error" in o:
        print(o["error"])
        return 1
    print("Stage 4 pre-test — does any driver LEAD gold within a regime?\n")
    print(f"  sample {o['sample']['from']} -> {o['sample']['to']}  "
          f"({o['sample']['rows']:,} rows)")
    print("  regime sizes (trading days):")
    for st, n in sorted(o["states"].items(), key=lambda kv: -kv[1]):
        print(f"    {st:26s} {n:6,d}")
    print(f"\n  {o['n_tests']:,} simultaneous tests (features x lags x regimes)")
    print(f"  5% threshold |r| > {o['threshold_raw_5pct']:.3f}   "
          f"Bonferroni |r| > {o['threshold_bonferroni']:.3f}")
    print(f"  clearing adjusted: {o['n_clearing_adjusted']}  "
          f"({o['n_coincident_adjusted']} coincident, "
          f"{o['n_LEADING_external']} LEADING from external drivers)")
    if o.get("n_LEADING_price_derived_excluded"):
        print(f"  ({o['n_LEADING_price_derived_excluded']} apparent leads EXCLUDED as "
              f"circular — price-derived feature inside a price-defined regime)")
    if o["leading_detail"]:
        print("\n  LEADING relationships surviving the adjusted threshold:")
        for r in o["leading_detail"][:15]:
            print(f"    {r['state']:24s} {r['feature']:26s} lag {int(r['lag']):3d}  "
                  f"r {r['r']:+.3f}  n={int(r['n'])}")
    print(f"\n  >>> {o['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
