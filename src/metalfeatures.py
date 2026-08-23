"""metalfeatures.py — the gold feature layer, per metal, with what each one can support.

Track D, milestone 3. The gold engine's 28 features do not all exist for the other
metals, and pretending otherwise is the failure this module is built to avoid.

WHAT TRANSFERS AND WHAT DOES NOT

    block                    GOLD  SILVER  PLATINUM  PALLADIUM   why
    macro (18 features)       yes    yes      yes       yes      metal-independent
    COT positioning (3)       yes    yes      yes       yes      all four have CFTC codes
    ETF flows (3)             yes    no       no        no       tonnage needs a per-issuer
                                                                 scraper; ETF PRICE is just
                                                                 the metal price again and
                                                                 would be a fake feature
    Shanghai premium (2)      yes    no       no        no       no PGM equivalent exists
    Central-bank demand (2)   yes    n/a      n/a       n/a      nobody holds PGM reserves

So Ag/Pt/Pd get 21 features against gold's 28. That asymmetry is reported with every
result rather than buried, because "we found less" and "we looked with less" are
different statements and only one of them is about the market.

THE PRIOR, STATED BEFORE THE TEST

Gold had 28 features, 21 years and point-in-time discipline, and NOTHING led it. The
metals here get fewer features, and Track B established that platinum and palladium
respond even less to macro than gold does — palladium is exactly inert to the one
release that moves gold. The expectation is therefore another negative.

It is still worth running. Track B also showed the metals are different assets, so
assuming the gold result transfers is precisely the error that finding warns against.
The point of milestone 3 is that it is CHEAP: a lead-lag scan settles whether there is
anything for milestones 4-8 to model, and if nothing leads, the modelling stack has
nothing to work with — which is exactly how the gold work concluded.

TARGETS ARE STRUCK ON EACH METAL'S OWN BENCHMARK

Not on a common index. Silver's benchmark is a single noon auction and the PGMs' are
PM fixes, so the windows differ; using one metal's fix to time another's return would
introduce a timing error dressed as a feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldfeatures as gf                                   # noqa: E402
import goldstore                                            # noqa: E402
import metals                                               # noqa: E402

START = "1990-01-01"

# The macro block, metal-independent. These are the series goldfeatures' shared
# features are built from.
MACRO_SERIES = [
    "REAL_10Y", "REAL_10Y_SPLICED", "REAL_5Y", "BREAKEVEN_10Y", "NOMINAL_10Y",
    "NOMINAL_2Y", "FED_FUNDS", "DXY", "SPX", "VIX", "HY_SPREAD", "BAA_SPREAD",
    "CPI", "M2", "EURUSD", "USDJPY", "USDINR", "USDCNY",
]


def series_for(metal: str) -> dict:
    """Which store series back this metal's features."""
    prefix = "" if metal == "GOLD" else f"{metal}_"
    return {
        "price": metals.BENCHMARK[metal],
        "cot_net": f"{prefix}COT_MM_NET",
        "cot_oi": f"{prefix}COT_OI",
    }


def available_blocks(metal: str) -> dict:
    """Honest inventory of what this metal can and cannot support."""
    return {
        "macro": True,
        "cot": True,
        "etf_flows": metal == "GOLD",          # tonnage feed is gold-only for now
        "shanghai_premium": metal == "GOLD",
        "central_bank": metal == "GOLD",
    }


def build(metal: str, start: str = START, winsorise: bool = True):
    """(features, targets) for one metal, using only blocks it can actually support."""
    sids = series_for(metal)
    panel_series = sorted(set(MACRO_SERIES + list(sids.values())
                              + ["LBMA_GOLD_PM_USD", "LBMA_SILVER_USD"]))
    p = goldstore.daily_panel(panel_series, start=start)
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()

    # The shared macro features, built by the gold layer on this panel. Its
    # metal-specific features come out NaN here (their series are absent) and are
    # dropped below rather than carried as empty columns.
    raw = gf._build_all(p)
    f = pd.DataFrame(raw, index=p.index)
    f = f.dropna(axis=1, how="all")

    # This metal's own positioning, replacing gold's.
    g = lambda c: p[c] if c in p.columns else pd.Series(np.nan, index=p.index)  # noqa: E731
    net, oi = g(sids["cot_net"]), g(sids["cot_oi"])
    if net.notna().any():
        f["cot_mm_net_pct_3y"] = gf._pctile(net, 756)
        f["cot_mm_net_chg_4w"] = gf._chg(net, 20)
        f["cot_mm_net_pct_oi"] = (net / oi.replace(0, np.nan)) if oi.notna().any() else np.nan

    # Own-price context, on THIS metal's benchmark rather than gold's.
    px = p[sids["price"]] if sids["price"] in p.columns else pd.Series(np.nan, index=p.index)
    f["own_dist_200d"] = gf._dist(px, 200)
    f["own_dist_50d"] = gf._dist(px, 50)
    f["own_mom_12m_1m"] = px.shift(21) / px.shift(273) - 1.0

    if winsorise:
        for c in f.columns:
            f[c] = gf._winsorise(f[c])

    t = build_targets(metal, p, px)
    keep = f.notna().sum() > 250
    f = f.loc[:, keep[keep].index]
    return f, t


def build_targets(metal: str, p: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """Forward returns on this metal's own benchmark.

    The forward-filled tail is cut, exactly as in the gold layer: daily_panel carries
    the last known fix to today, so a stalled feed would otherwise produce forward
    returns of exactly 0.0 and train on them as observations.
    """
    last_real = goldstore.last_reference(metals.BENCHMARK[metal])
    if last_real is not None:
        px = px.where(px.index <= last_real)
    out = pd.DataFrame(index=p.index)
    rv = np.log(px / px.shift(1)).rolling(gf.VOL_WINDOW, min_periods=30).std()
    for name, h in gf.TARGET_HORIZONS.items():
        r = np.log(px.shift(-h) / px)
        out[name] = r
        out[name + "_scaled"] = r / (rv * np.sqrt(h)).replace(0, np.nan)
    out["realised_vol_60d"] = rv
    return out


def inventory() -> pd.DataFrame:
    """Feature counts per metal, so a thinner search is never read as a thinner market."""
    rows = []
    for m in metals.METALS:
        blocks = available_blocks(m)
        rows.append({"metal": m, **{k: ("yes" if v else "no") for k, v in blocks.items()}})
    return pd.DataFrame(rows)


def main() -> int:
    print("Feature blocks by metal:\n")
    print(inventory().to_string(index=False))
    print()
    for m in metals.METALS:
        f, t = build(m)
        if f.empty:
            print(f"  {m:11s} no panel")
            continue
        n_obs = int(t["fwd_ret_60d"].notna().sum())
        print(f"  {m:11s} {f.shape[1]:3d} features x {f.shape[0]:5d} dates "
              f"({f.index.min():%Y-%m} -> {f.index.max():%Y-%m}), "
              f"{n_obs:,} usable 60d targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
