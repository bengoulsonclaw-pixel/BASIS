"""goldfeatures.py — Milestone 3: the feature and target tables (spec §4, §5).

Every feature in the spec's §4.2 list, computed only from `goldstore.daily_panel`,
written to a `features` table keyed `(feature_id, date)` so the backtest and the live
path read byte-identical numbers. Nothing here touches the observations table; the
lint rule in tests/test_goldstore.py enforces that.

The four rules from §4.1, and why each one bites
------------------------------------------------
1. **No raw levels.** Gold went $1,300 -> $4,500 and M2 went up almost monotonically
   over the same window. Regress one trending series on another and you get a
   beautiful t-stat measuring nothing but the fact that time passed. Every feature
   here is a change, a z-score, a percentile or a ratio.
2. **Forward-fill from `published_at`, never `reference_date`.** Handled upstream by
   `goldstore.as_known_series`. CPI for January is *about* January but unknowable
   until mid-February; filling from the reference date hands the model three weeks
   of future, every month, in silence.
3. **Winsorise on an EXPANDING window, not the full sample.** Clipping at the
   full-sample 1st/99th percentile leaks: in 2015 you cannot know that March 2020
   will redefine what a tail looks like. `_winsorise` uses only data up to each row.
4. **Store the numbers.** A feature recomputed slightly differently at serving time
   than at fit time is the classic train/serve skew, and it is invisible — the model
   just quietly underperforms its backtest.

What is deliberately absent
---------------------------
* `risk_reversal_25d` — the gold options archive starts 2025-08 (252 rows). One year
  cannot support a fit that needs twenty.
* `india_imports_yoy`, `wgc_bar_coin_yoy` — sources identified during the gap review
  but not yet ingested (Comtrade is one API call per month; WGC is parsed but gated).
* `gold_aisc_ratio` — no free AISC feed; hand-seeding is a Milestone-2 leftover.
Each is registered in `MISSING_FEATURES` so the gap is visible in the output rather
than being a silent absence from a list nobody re-reads.

CLI:  python src/goldfeatures.py [--build] [--summary]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import goldstore  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
FEATURES_FILE = STORE_DIR / "features.parquet"
TARGETS_FILE = STORE_DIR / "targets.parquet"
FEATURE_META = STORE_DIR / "feature_meta.json"

# 1990. This was the last of three self-imposed caps (golddata START, the LBMA
# filter, and this one) that between them left a 6-year evaluable window.
START = "1990-01-01"

# Spec §5: three horizons, measured off the LBMA PM fix.
TARGET_HORIZONS = {"fwd_ret_5d": 5, "fwd_ret_60d": 60, "fwd_ret_250d": 250}
TARGET_PRICE = "LBMA_GOLD_PM_USD"
VOL_WINDOW = 60

# Inputs the panel must carry for the features below.
PANEL_SERIES = [
    "REAL_10Y", "REAL_10Y_SPLICED", "REAL_5Y", "BREAKEVEN_10Y", "NOMINAL_10Y", "NOMINAL_2Y", "FED_FUNDS",
    "DXY", "USDCNY", "USDJPY", "USDINR", "EURUSD",
    "VIX", "CREDIT_BAA", "SPX",
    "COT_MM_NET", "COT_OI", "GLD_TONNES", "GLD_PX", "GDX_PX",
    "SGE_AU9999", "CB_GOLD_WORLD",
    "LBMA_GOLD_PM_USD", "LBMA_GOLD_AM_USD", "LBMA_GOLD_PM_EUR", "LBMA_GOLD_PM_GBP",
    "LBMA_SILVER_USD", "COMEX_SILVER_FRONT", "CPI", "M2",
]

# Features computable back to 1990 — the DEEP set. Everything else needs a source
# that starts later: TIPS levels (2003), GLD (2004), COT (2006), the Shanghai premium
# (2016). The deep set trades breadth for roughly four times the evaluable sample,
# which is the binding constraint on every result in this project.
DEEP_FEATURES = [
    "real_yield_10y_chg_20d", "dxy_dist_50d", "dxy_chg_20d", "curve_2s10s",
    "fed_cut_odds_chg_20d",
    "vix_z_1y", "hy_spread_chg_20d", "spx_dist_200d", "real_rate_vol_20d",
    "gold_dist_200d", "gold_dist_50d", "gold_mom_12m_1m",
    "gold_silver_ratio_z_5y", "gold_cpi_ratio_z_10y", "gold_m2_ratio_z_10y",
]

MISSING_FEATURES = {
    "risk_reversal_25d": "gold options archive starts 2025-08 (252 rows) — too short",
    "india_imports_yoy": "UN Comtrade identified (1 call/month) but not yet ingested",
    "wgc_bar_coin_yoy": "WGC GDT parsed but ingestion pending",
    "gold_aisc_ratio": "no free AISC feed — hand-seed pending",
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _chg(s: pd.Series, n: int) -> pd.Series:
    """Absolute change over n business days — for series already in rate space."""
    return s - s.shift(n)


def _pct(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def _z(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 4)).std()
    return (s - m) / sd.replace(0, np.nan)


def _pctile(s: pd.Series, window: int) -> pd.Series:
    """Where the latest value sits in its own trailing window, 0-100. Spec asks for
    COT as a percentile of its 3-year range rather than a z-score because positioning
    is bounded and lumpy, and a z-score of a bounded series overstates the tails."""
    return s.rolling(window, min_periods=max(60, window // 4)).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def _dist(s: pd.Series, window: int) -> pd.Series:
    """Percent above/below a trailing moving average."""
    return s / s.rolling(window, min_periods=max(20, window // 4)).mean() - 1.0


def _winsorise(s: pd.Series, lo: float = 0.01, hi: float = 0.99,
               min_periods: int = 250) -> pd.Series:
    """Clip to EXPANDING-window percentiles (spec §4.1).

    Full-sample winsorising is a subtle leak: clipping a 2015 observation at a bound
    computed from a window that includes March 2020 uses knowledge of a crisis five
    years away. Expanding bounds only ever look backwards, so early rows are clipped
    loosely and later ones tightly — which is also the honest description of what a
    forecaster actually knew."""
    lo_b = s.expanding(min_periods=min_periods).quantile(lo)
    hi_b = s.expanding(min_periods=min_periods).quantile(hi)
    return s.clip(lower=lo_b, upper=hi_b)


# ---------------------------------------------------------------------------
# feature definitions — feature_id -> (bucket, description, builder)
# ---------------------------------------------------------------------------
def _build_all(p: pd.DataFrame) -> dict:
    """Every feature in spec §4.2 that our ingested series can support."""
    f: dict = {}
    g = lambda c: p[c] if c in p.columns else pd.Series(np.nan, index=p.index)  # noqa: E731

    # ── monetary ────────────────────────────────────────────────────────
    # CHANGES may use the spliced series; LEVELS may not.
    #
    # The pre-2003 segment is 10y nominal less trailing 12m CPI, level-adjusted by the
    # mean gap measured over the 2003-2008 overlap (-0.41pp). That removes most of the
    # join but not all of it: a backward-looking proxy and a forward-looking TIPS yield
    # differ by the gap between realised and expected inflation, which is itself
    # time-varying, so no constant offset can join them cleanly. The residual step is
    # about -0.5pp on three-month means.
    #
    # A 20-day CHANGE is contaminated only across the 20 days spanning the join, and a
    # 20-day volatility of changes not at all. A 252-day Z-SCORE would carry the level
    # error for a year and read the join as a violent repricing — so `real_yield_10y_z_1y`
    # below stays on measured TIPS only, and starts in 2003.
    f["real_yield_10y_chg_20d"] = _chg(g("REAL_10Y_SPLICED"), 20)
    f["real_yield_10y_z_1y"] = _z(g("REAL_10Y"), 252)
    f["breakeven_10y_chg_20d"] = _chg(g("BREAKEVEN_10Y"), 20)
    f["dxy_dist_50d"] = _dist(g("DXY"), 50)
    f["dxy_chg_20d"] = _pct(g("DXY"), 20)
    f["curve_2s10s"] = g("NOMINAL_10Y") - g("NOMINAL_2Y")
    # Cuts priced over the coming year. fedpath.py backs the same read out of the
    # live SR3 strip but keeps no history; 2y-less-funds runs to 1990 for free.
    f["fed_cut_odds_chg_20d"] = _chg(g("NOMINAL_2Y") - g("FED_FUNDS"), 20)

    # ── flows and positioning ───────────────────────────────────────────
    f["cot_mm_net_pct_3y"] = _pctile(g("COT_MM_NET"), 756)
    f["cot_mm_net_chg_4w"] = _chg(g("COT_MM_NET"), 20)
    # Net position scaled by open interest — a raw contract count is not comparable
    # across a decade in which gold OI roughly doubled.
    f["cot_mm_net_pct_oi"] = (g("COT_MM_NET") / g("COT_OI").replace(0, np.nan))
    f["etf_tonnage_chg_4w"] = _pct(g("GLD_TONNES"), 20)
    f["etf_tonnage_chg_12w"] = _pct(g("GLD_TONNES"), 60)
    f["gld_flow_z_1y"] = _z(g("GLD_TONNES").diff(), 252)

    # ── physical ────────────────────────────────────────────────────────
    # Shanghai premium in $/oz, computed here rather than stored, because it is a
    # cross-source derivation (SGE close vs the London AM fix vs USDCNY) and the
    # store holds observations, not derivations.
    # Reference leg is the AM fix, not PM. SGE's day session closes ~08:30 London,
    # the AM fix is 10:30, the PM fix 15:00. Measured against PM the series carries a
    # whole London session of drift that has nothing to do with Chinese demand — on a
    # 2% day that is ~$90 of pure timing on a signal whose real range is tens of
    # dollars. Even against AM the mismatch is real, hence the 5-day smooth.
    sh_usd_oz = g("SGE_AU9999") * 31.1034768 / g("USDCNY").replace(0, np.nan)
    f["shanghai_premium_usd"] = (sh_usd_oz - g("LBMA_GOLD_AM_USD")).rolling(
        5, min_periods=3).mean()
    f["shanghai_premium_z_1y"] = _z(f["shanghai_premium_usd"], 252)
    cb = g("CB_GOLD_WORLD")
    f["cb_net_purchases_12m"] = cb - cb.shift(252)
    f["cb_net_purchases_yoy_chg"] = f["cb_net_purchases_12m"].diff(252)

    # ── risk ────────────────────────────────────────────────────────────
    f["vix_z_1y"] = _z(g("VIX"), 252)
    f["hy_spread_chg_20d"] = _chg(g("CREDIT_BAA"), 20)
    f["spx_dist_200d"] = _dist(g("SPX"), 200)
    # Realised vol OF the real yield: gold cares about how violently the discount
    # rate is moving, not only where it sits.
    # Volatility of CHANGES, so the spliced series is safe here — a level offset
    # differences away entirely.
    f["real_rate_vol_20d"] = (g("REAL_10Y_SPLICED").diff()
                              .rolling(20, min_periods=10).std())

    # ── valuation and trend ─────────────────────────────────────────────
    px = g(TARGET_PRICE)
    f["gold_dist_200d"] = _dist(px, 200)
    f["gold_dist_50d"] = _dist(px, 50)
    # 12-month return EXCLUDING the most recent month — the standard momentum
    # construction, which drops the short-horizon reversal that pollutes a plain
    # 12m return.
    f["gold_mom_12m_1m"] = np.log(px.shift(20) / px.shift(252))
    # LBMA silver, not the COMEX contract: the deep store starts 2016, which left a
    # 5-year z-score with under two usable years.
    silver = g("LBMA_SILVER_USD").where(g("LBMA_SILVER_USD").notna(),
                                        g("COMEX_SILVER_FRONT"))
    f["gold_silver_ratio_z_5y"] = _z(px / silver.replace(0, np.nan), 1260)
    f["gold_cpi_ratio_z_10y"] = _z(px / g("CPI").replace(0, np.nan), 2520)
    f["gold_m2_ratio_z_10y"] = _z(px / g("M2").replace(0, np.nan), 2520)
    # "When gold makes highs in every currency the move is about gold; when it only
    # rises in dollars the move is about the dollar." Share of the non-USD legs above
    # their own 200d average.
    legs = {"EUR": g("LBMA_GOLD_PM_EUR"), "GBP": g("LBMA_GOLD_PM_GBP"),
            "JPY": px * g("USDJPY"), "INR": px * g("USDINR"),
            "CNY": px * g("USDCNY")}
    above = [(v > v.rolling(200, min_periods=120).mean()).astype(float)
             for v in legs.values() if v.notna().any()]
    if above:
        f["gold_fx_breadth"] = pd.concat(above, axis=1).mean(axis=1)

    return f


FEATURE_BUCKETS = {
    "real_yield_10y_chg_20d": "monetary", "real_yield_10y_z_1y": "monetary",
    "breakeven_10y_chg_20d": "monetary", "dxy_dist_50d": "monetary",
    "dxy_chg_20d": "monetary", "curve_2s10s": "monetary",
    "fed_cut_odds_chg_20d": "monetary",
    "cot_mm_net_pct_3y": "flows", "cot_mm_net_chg_4w": "flows",
    "cot_mm_net_pct_oi": "flows", "etf_tonnage_chg_4w": "flows",
    "etf_tonnage_chg_12w": "flows", "gld_flow_z_1y": "flows",
    "shanghai_premium_usd": "physical", "shanghai_premium_z_1y": "physical",
    "cb_net_purchases_12m": "physical", "cb_net_purchases_yoy_chg": "physical",
    "vix_z_1y": "risk", "hy_spread_chg_20d": "risk", "spx_dist_200d": "risk",
    "real_rate_vol_20d": "risk",
    "gold_dist_200d": "valuation", "gold_dist_50d": "valuation",
    "gold_mom_12m_1m": "valuation", "gold_silver_ratio_z_5y": "valuation",
    "gold_cpi_ratio_z_10y": "valuation", "gold_m2_ratio_z_10y": "valuation",
    "gold_fx_breadth": "valuation",
}

# Features left unwinsorised: bounded or already-normalised constructions where
# clipping would destroy the meaning of the bound rather than tame a tail.
NO_WINSORISE = {"cot_mm_net_pct_3y", "gold_fx_breadth"}


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def build_targets(p: pd.DataFrame) -> pd.DataFrame:
    """Spec §5: forward log returns off the LBMA PM fix, plus volatility-scaled
    versions.

    The scaled target is the one to model. A 3% move in a calm market and a 3% move
    in a panic are not the same event, and a model fitted on raw returns spends its
    capacity learning the volatility cycle instead of the direction. Scale first,
    convert to a probability afterwards."""
    px = p[TARGET_PRICE]
    out = pd.DataFrame(index=p.index)
    rv = np.log(px / px.shift(1)).rolling(VOL_WINDOW, min_periods=30).std()
    for name, h in TARGET_HORIZONS.items():
        r = np.log(px.shift(-h) / px)
        out[name] = r
        # scale by the vol KNOWN AT THE TIME (trailing), never by realised vol over
        # the forward window — that would be scaling by the answer.
        out[name + "_scaled"] = r / (rv * np.sqrt(h)).replace(0, np.nan)
    out["realised_vol_60d"] = rv
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build(start: str = START, winsorise: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = goldstore.daily_panel(PANEL_SERIES, start=start)
    raw = _build_all(panel)
    feats = pd.DataFrame(raw, index=panel.index)

    if winsorise:
        for c in feats.columns:
            if c not in NO_WINSORISE:
                feats[c] = _winsorise(feats[c])

    targets = build_targets(panel)

    # long format, keyed (feature_id, date) per spec §4.1
    long = (feats.stack(future_stack=True).rename("value").reset_index()
            .rename(columns={"level_0": "date", "level_1": "feature_id"}))
    long = long[["feature_id", "date", "value"]].dropna(subset=["value"])
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(FEATURES_FILE, index=False)
    targets.to_parquet(TARGETS_FILE)

    meta = {
        "built_from": len(PANEL_SERIES),
        "features": {c: {"bucket": FEATURE_BUCKETS.get(c, ""),
                         "n": int(feats[c].notna().sum()),
                         "first": str(feats[c].first_valid_index()),
                         "last": str(feats[c].last_valid_index())}
                     for c in feats.columns},
        "missing": MISSING_FEATURES,
        "winsorised": winsorise,
    }
    FEATURE_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return feats, targets


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide feature frame + targets, from the stored tables — the SAME numbers the
    backtest saw, which is the entire reason they are stored."""
    if not FEATURES_FILE.exists():
        return build()
    long = pd.read_parquet(FEATURES_FILE)
    feats = long.pivot(index="date", columns="feature_id", values="value")
    feats.index = pd.to_datetime(feats.index)
    targets = pd.read_parquet(TARGETS_FILE)
    targets.index = pd.to_datetime(targets.index)
    return feats.sort_index(), targets.sort_index()


def main() -> int:
    feats, targets = build()
    print(f"features.parquet  {feats.shape[1]} features x {feats.shape[0]} dates "
          f"({feats.index.min().date()} -> {feats.index.max().date()})")
    print(f"targets.parquet   {list(TARGET_HORIZONS)} + scaled\n")
    by = {}
    for c in feats.columns:
        by.setdefault(FEATURE_BUCKETS.get(c, "?"), []).append(c)
    for bucket in ("monetary", "flows", "physical", "risk", "valuation"):
        for c in by.get(bucket, []):
            s = feats[c]
            print(f"  {bucket:10s} {c:26s} n={s.notna().sum():5d}  "
                  f"from {str(s.first_valid_index())[:10]}  "
                  f"latest {s.dropna().iloc[-1]:+.4f}" if s.notna().any() else
                  f"  {bucket:10s} {c:26s} EMPTY")
    print("\n  not built (source gaps):")
    for k, v in MISSING_FEATURES.items():
        print(f"    {k:22s} {v}")
    if "--summary" in sys.argv:
        print("\n", targets.tail(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
