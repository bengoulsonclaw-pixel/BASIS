"""goldmodel.py — the 🥇 Gold Driver Model.

What this model is, and what it deliberately is not
---------------------------------------------------
Fitted on ten years of daily data (2016-08 → today), the driver set in the gold
framework — real yields, the dollar, Fed pricing, ETF flows, positioning, Asian
physical, official demand — **explains** gold's one-month move well: R² ≈ 0.31 from
the exogenous macro block alone, ≈ 0.47 once the co-moving flow variables are added
back. It does **not** forecast it: every driver's forward
information coefficient sits inside ±0.17 with an overlap-adjusted |t| < 2, and a
zero-parameter signed composite of all of them scores IC ≈ +0.08 (t ≈ 0.9). Ten
years of gold is about 112 independent months; that is not enough sample for
effects this size to clear noise, and pretending otherwise would be the whole
game lost.

So the model leads with what the data supports:

  1. SENSITIVITIES  gold % per unit move in each driver, fitted jointly so the
     collinear rates/dollar block does not double-count, and fitted on EXOGENOUS
     drivers only so ETF and positioning flows cannot absorb the effect being
     measured. The number a desk actually uses: "10bp off real yields is worth
     about half a percent on gold."
  2. SCENARIO       run a view on the drivers through those sensitivities and get
     the implied gold move, with the residual risk stated.
  3. FAIR VALUE     cumulative unexplained move — how far gold has run beyond what
     its drivers account for over the past year. The debasement / official-bid
     residual, made visible.
  4. ATTRIBUTION    decompose the move just realised into per-driver pieces.
  5. TILT           a directional score, reported WITH its measured skill, which is
     weak. It is a lean, not a forecast, and the code says so out loud.

Modelling decisions worth defending
-----------------------------------
* TARGET is the log return of the panama-adjusted continuous gold future over H
  business days (H=21 ≈ four weeks). Adjusted, not raw: raw returns book the roll
  gap as P&L.
* FEATURES are all CHANGES or Z-SCORES, never raw levels. Gold went from $1,300 to
  $4,500 over the sample; regressing on the level of anything trending buys a
  beautiful in-sample fit and no information.
* NO LOOKAHEAD. Standardisation stats, coefficients and gap-filling medians come
  from the training slice only, and that slice closes H days before the prediction
  date because y is not observable before then.
* OVERLAPPING RETURNS inflate every t-stat by roughly sqrt(H). Significance here
  divides the sample by H first, and results are also read on non-overlapping
  windows.
* RIDGE, not OLS. Real yields, the dollar and Fed pricing are three views of one
  thing. OLS hands that block wild offsetting coefficients; the penalty keeps the
  attribution readable, which matters more here than the last point of fit.
* SIGN PRIORS ARE NOT REFITTED. Two come out backwards in sample — the Shanghai
  premium and washed-out positioning (see SPEC). They are left as the framework
  states them and the disagreement is reported. Flipping a prior because the
  sample disagreed is how a backtest gets fitted to its own noise.

CLI:  python src/goldmodel.py [--horizon 21] [--rebuild] [--full]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import golddata  # noqa: E402

SIGNALS = _ROOT / "data" / "signals"
MODEL_FILE = SIGNALS / "gold_model.json"

HORIZONS = (5, 10, 21, 42)
DEFAULT_H = 21               # ~4 weeks — "the coming weeks"
Z_WINDOW = 750               # ~3y trailing window for level z-scores
MIN_TRAIN = 750              # no out-of-sample point until 3y of realised targets
RIDGE_LAMBDA = 3.0           # standardised units; tuned for readable attribution


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------
def _z(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Trailing z-score. Rolling, not expanding: a 2016 mean is not the right
    yardstick for a 2026 real yield, and these features mean 'extreme relative to
    the recent regime', not relative to all history."""
    m = s.rolling(window, min_periods=window // 3).mean()
    sd = s.rolling(window, min_periods=window // 3).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-4, 4)


def _chg(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)


def _ret(s: pd.Series, n: int) -> pd.Series:
    return np.log(s / s.shift(n))


# name -> (block, description, prior sign on gold)
SPEC = {
    # ── direction: the cost of holding it ────────────────────────────────
    "real10_chg21":   ("Real rates", "10y TIPS yield, 1m change", -1),
    "real10_z":       ("Real rates", "10y TIPS yield vs 3y range", -1),
    "be10_chg21":     ("Real rates", "10y breakeven, 1m change", +1),
    "cuts_chg21":     ("Fed path", "Cuts priced (2y less funds), 1m change", -1),
    "cuts_z":         ("Fed path", "Cuts priced vs 3y range", -1),
    # ── direction: the dollar ────────────────────────────────────────────
    "dxy_ret21":      ("Dollar", "DXY, 1m return", -1),
    "dxy_z":          ("Dollar", "DXY vs 3y range", -1),
    "cny_ret21":      ("Dollar", "USD/CNY, 1m return", -1),
    # ── fear of holding something else ───────────────────────────────────
    "vix_z":          ("Risk", "VIX vs 3y range", +1),
    "vix_chg21":      ("Risk", "VIX, 1m change", +1),
    "credit_chg21":   ("Risk", "Baa credit spread, 1m change", +1),
    # ── the slow floor ───────────────────────────────────────────────────
    "cb_z":           ("Official demand", "Central bank net buying, 12m sum", +1),
    # PRIOR DISAGREES WITH SAMPLE. The framework reads a wide Shanghai premium as
    # strong Chinese buying, i.e. bullish. In this sample it runs the other way on
    # both clocks: the premium collapses INTO rallies (coincident corr -0.29) and a
    # rich premium leads slightly weaker gold (IC -0.11). That is the price-
    # sensitive physical buyer stepping back — exactly the behaviour the framework
    # describes, with the sign reversed once gold is the thing doing the moving.
    # Prior kept as stated; the conflict is reported, not buried.
    "sge_prem_z":     ("Asian physical", "Shanghai premium over London", +1),
    "carry_chg21":    ("Physical stress", "COMEX 1st/2nd carry, 1m change", -1),
    "efp_z":          ("Physical stress", "COMEX over London fix, 5d avg", +1),
    # ── timing: flows and positioning ────────────────────────────────────
    "gld_chg21":      ("ETF flows", "SPDR GLD tonnes, 1m % change", +1),
    "gld_z":          ("ETF flows", "SPDR GLD tonnes vs 3y range", +1),
    # PRIOR DISAGREES WITH SAMPLE. Crowded longs are supposed to precede sharper
    # pullbacks. Over 2016-2026 a COT index above 80 was followed by +0.93% over the
    # next month and a washed-out sub-20 reading by +0.08% — mildly the wrong way
    # round. Kept as the framework states it.
    "cot_pos":        ("Positioning", "Managed-money COT index, centred", -1),
    "mm_chg21":       ("Positioning", "Managed-money net, 1m change", -1),
    # ── cross-checks ─────────────────────────────────────────────────────
    "gsr_z":          ("Cross-check", "Gold/silver ratio vs 3y range", -1),
    "miners_lead21":  ("Cross-check", "Miners minus bullion, 1m return gap", +1),
    "ccy_breadth":    ("Cross-check", "Share of USD/EUR/GBP/JPY/INR at 3m highs", +1),
    # ── controls ─────────────────────────────────────────────────────────
    "gold_mom63":     ("Momentum", "Gold, 3m return", +1),
    "gold_dist200":   ("Momentum", "Gold vs its 200d average", -1),
}

BLOCK_ORDER = ["Real rates", "Fed path", "Dollar", "Risk", "Official demand",
               "Asian physical", "Physical stress", "ETF flows", "Positioning",
               "Cross-check", "Momentum"]

# EXOGENOUS drivers — things that move gold rather than move with it. This is the
# set behind the sensitivities, the scenario engine and the fair-value gap, and it
# is the only set from which a causal-flavoured statement is defensible.
MACRO_DRIVERS = ["real10_chg21", "be10_chg21", "cuts_chg21", "dxy_ret21",
                 "cny_ret21", "credit_chg21", "vix_chg21"]

# CO-MOVING flow variables. ETF tonnage and managed-money net rise BECAUSE gold
# rallied at least as much as the reverse, so they belong in a description of a
# move and never in a scenario. Adding them lifts R² from 0.31 to 0.47 — and drags
# the 10y real-yield coefficient from -0.51% per 10bp (t=-2.8) to -0.12% (t=-0.7),
# because the flow variables absorb the very effect the model is trying to price.
# That is textbook endogeneity, and it is why the two blocks are fitted separately.
FLOW_DRIVERS = ["gld_chg21", "mm_chg21"]

# Deliberately EXCLUDES the momentum controls and the cross-checks: miners, the
# gold/silver ratio and currency breadth are gold by another name, and putting them
# on the right-hand side would lift R² to something impressive and meaningless.
COINCIDENT = MACRO_DRIVERS + FLOW_DRIVERS

# feature -> (label, unit step, unit text) for the sensitivity table
UNITS = {
    "real10_chg21": ("10y real yield", 0.10, "+10bp"),
    "be10_chg21":   ("10y breakeven", 0.10, "+10bp"),
    "cuts_chg21":   ("Cuts priced (2y less funds)", 0.10, "+10bp"),
    "dxy_ret21":    ("Dollar (DXY)", 0.01, "+1%"),
    "cny_ret21":    ("USD/CNY", 0.01, "+1%"),
    "credit_chg21": ("Baa credit spread", 0.10, "+10bp"),
    "vix_chg21":    ("VIX", 1.0, "+1pt"),
    "gld_chg21":    ("SPDR GLD tonnes", 0.01, "+1%"),
    "mm_chg21":     ("Managed-money net", 1.0, "+1 sd"),
}


def features(F: pd.DataFrame) -> pd.DataFrame:
    """The standardisable feature matrix. Every column is a change or a z-score;
    nothing carries a trending level."""
    X = pd.DataFrame(index=F.index)
    g = lambda c: F[c] if c in F.columns else pd.Series(np.nan, index=F.index)  # noqa: E731

    X["real10_chg21"] = _chg(g("real_10y"), 21)
    X["real10_z"] = _z(g("real_10y"))
    X["be10_chg21"] = _chg(g("breakeven_10y"), 21)
    X["cuts_chg21"] = _chg(g("cuts_priced"), 21)
    X["cuts_z"] = _z(g("cuts_priced"))

    X["dxy_ret21"] = _ret(g("dxy"), 21)
    X["dxy_z"] = _z(g("dxy"))
    X["cny_ret21"] = _ret(g("usdcny"), 21)

    X["vix_z"] = _z(g("vix"))
    # The CHANGE, not the level, is what belongs in a returns regression. Fitted as
    # a level z-score the VIX came back with the wrong sign (-0.33% per sd); as a
    # change it prints the right one. A level regressed on a return is a units
    # mismatch, and it quietly mis-signs the risk block.
    X["vix_chg21"] = _chg(g("vix"), 21)
    # Baa, not the ICE BofA high-yield OAS: FRED licence-limits the ICE family to a
    # rolling 3y, which would cut this feature — and every fit including it — to a
    # quarter of the sample. See golddata.FRED_IDS.
    X["credit_chg21"] = _chg(g("credit_baa"), 21)

    X["cb_z"] = _z(g("cb_net_12m"))
    X["sge_prem_z"] = _z(g("sge_premium"))
    X["carry_chg21"] = _chg(g("gc_carry"), 21)
    # EFP is smoothed before z-scoring: the COMEX settle lands 3.5h after the London
    # PM fix, so a single day's print is largely intraday drift — momentum wearing a
    # physical-tightness costume. The 5d average keeps the dislocation signal and
    # drops most of the timing noise.
    X["efp_z"] = _z(g("efp").rolling(5, min_periods=3).mean())

    X["gld_chg21"] = g("gld_tonnes").pct_change(21)
    X["gld_z"] = _z(g("gld_tonnes"))
    X["cot_pos"] = (g("cot_index") - 50.0) / 50.0
    X["mm_chg21"] = _z(_chg(g("mm_net"), 21))

    X["gsr_z"] = _z(g("gold_silver_ratio"))
    X["miners_lead21"] = _ret(g("miners_vs_bullion"), 21)
    # "When gold makes highs in every currency the move is about gold; when it only
    # rises in dollars the move is about the dollar." That test, made numeric: how
    # many of the five currency legs sit within 0.5% of a 3m high.
    legs = [c for c in ("lbma_usd", "gold_eur", "gold_gbp", "gold_jpy", "gold_inr")
            if c in F.columns]
    if legs:
        hi = pd.concat([(F[c] >= F[c].rolling(63, min_periods=40).max() * 0.995)
                        for c in legs], axis=1)
        X["ccy_breadth"] = hi.mean(axis=1)

    X["gold_mom63"] = _ret(g("gold"), 63)
    X["gold_dist200"] = g("gold") / g("gold").rolling(200, min_periods=120).mean() - 1.0

    return X.reindex(columns=list(SPEC))


def target(F: pd.DataFrame, h: int = DEFAULT_H) -> pd.Series:
    """Forward log return of adjusted gold over the next h business days."""
    return np.log(F["gold"].shift(-h) / F["gold"]).rename(f"fwd_{h}d")


def trailing(F: pd.DataFrame, h: int = DEFAULT_H) -> pd.Series:
    """The h-day return just realised — left-hand side of the explanatory fit."""
    return np.log(F["gold"] / F["gold"].shift(h)).rename(f"trail_{h}d")


# ---------------------------------------------------------------------------
# statistics (numpy only — this repo carries no scipy/sklearn and does not need
# to for a ridge and a rank correlation)
# ---------------------------------------------------------------------------
def _rankv(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 30:
        return np.nan
    return float(np.corrcoef(_rankv(d.iloc[:, 0].to_numpy()),
                             _rankv(d.iloc[:, 1].to_numpy()))[0, 1])


def overlap_t(ic: float, n: int, h: int) -> float:
    """t-stat on a statistic measured from OVERLAPPING h-day returns.

    Consecutive observations share h-1 days of the same future, so the honest
    sample size is n/h, not n. Skipping this is how a 0.15 correlation on 2,500
    daily rows becomes a 't of 7.5' that is really a t of 1.6."""
    n_eff = max(int(n / max(h, 1)), 3)
    if ic is None or not np.isfinite(ic) or abs(ic) >= 1:
        return np.nan
    return float(ic * np.sqrt(n_eff - 2) / np.sqrt(1 - ic ** 2))


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge on already-standardised X and demeaned y."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------------------------------------------------------------------------
# 1. the explanatory fit — sensitivities, scenarios, attribution
# ---------------------------------------------------------------------------
def fit_coincident(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H,
                   lam: float = RIDGE_LAMBDA, cols: list | None = None) -> dict:
    """Ridge of the h-day gold return on the SAME period's driver changes.

    A same-period relationship, not a forecast — its job is to price the drivers,
    and it does that well (R² ≈ 0.47 at h=21). Standard errors carry the overlap
    correction, so reported t-stats rest on n/h effective observations rather than
    the flattering daily count."""
    cols = cols or [c for c in MACRO_DRIVERS if c in X.columns]
    y = trailing(F, h)
    d = pd.concat([X[cols], y.rename("y")], axis=1).dropna()
    mu, sd = d[cols].mean(), d[cols].std().replace(0, np.nan)
    Z = ((d[cols] - mu) / sd).fillna(0.0)
    ymu = float(d["y"].mean())
    beta = pd.Series(ridge(Z.to_numpy(), (d["y"] - ymu).to_numpy(), lam), index=cols)

    fitted = Z.to_numpy() @ beta.to_numpy() + ymu
    resid = pd.Series(d["y"].to_numpy() - fitted, index=d.index)
    r2 = float(1 - resid.var() / d["y"].var())

    # Overlap-corrected coefficient standard errors: the effective sample is n/h,
    # so the residual variance is spread over n_eff - k degrees of freedom, not
    # n - k. Without this every coefficient here would look 4-5x more certain.
    n_eff = max(int(len(d) / h), len(cols) + 2)
    xtx_inv = np.linalg.inv(Z.to_numpy().T @ Z.to_numpy() + lam * np.eye(len(cols)))
    scale = float(resid.var()) * len(d) / max(n_eff - len(cols), 1)
    se = pd.Series(np.sqrt(np.diag(xtx_inv) * scale), index=cols)

    return {"cols": cols, "beta_std": beta, "se_std": se, "mu": mu, "sd": sd,
            "ymu": ymu, "r2": r2, "resid": resid, "n": int(len(d)),
            "n_eff": n_eff, "resid_sd": float(resid.std()), "h": h}


def sensitivities(fit: dict) -> pd.DataFrame:
    """Gold % per unit move in each driver — the desk-usable form of the fit.

    beta_std is 'log return per 1 sd of the standardised feature'; dividing by the
    feature's own sd puts it back into natural units, and each is then quoted per
    the step a trader actually thinks in (10bp, 1%, 1 sd)."""
    rows = []
    for c in fit["cols"]:
        label, step, unit = UNITS.get(c, (c, 1.0, "+1 unit"))
        per_nat = float(fit["beta_std"][c] / fit["sd"][c])
        se = float(fit["se_std"][c])
        rows.append({"feature": c, "driver": label, "move": unit,
                     "gold_pct": float(np.expm1(per_nat * step) * 100),
                     "t_stat": float(fit["beta_std"][c] / se) if se else np.nan,
                     "prior_sign": SPEC[c][2] if c in SPEC else 0})
    out = pd.DataFrame(rows)
    out["agrees_with_prior"] = np.sign(out["gold_pct"]) == np.sign(out["prior_sign"])
    return out.reindex(out["gold_pct"].abs().sort_values(ascending=False).index)


def scenario(fit: dict, moves: dict) -> dict:
    """Run a view on the drivers through the fitted sensitivities.

    `moves` is in NATURAL units: {'real10_chg21': -0.20} means real yields fall 20bp
    over the horizon; {'dxy_ret21': -0.02} means the dollar drops 2%. Unspecified
    drivers are held at ZERO change, not at their sample mean — a scenario should
    state what it assumes rather than smuggle in a drift."""
    total, parts = 0.0, {}
    for c in fit["cols"]:
        contrib = float(fit["beta_std"][c] / fit["sd"][c] * float(moves.get(c, 0.0)))
        parts[c] = float(np.expm1(contrib) * 100)
        total += contrib
    return {"gold_pct": float(np.expm1(total) * 100), "parts_pct": parts,
            "resid_sd_pct": float(fit["resid_sd"] * 100),
            "note": f"unexplained move is +/-{fit['resid_sd'] * 100:.1f}% at 1sd"}


def attribution(fit: dict, F: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Decompose the h-day move just realised into per-driver contributions."""
    h = fit["h"]
    row = X.iloc[-1][fit["cols"]]
    z = ((row - fit["mu"]) / fit["sd"]).fillna(0.0)
    contrib = z * fit["beta_std"]
    actual = float(trailing(F, h).iloc[-1])
    explained = float(contrib.sum() + fit["ymu"])
    return {"window_days": h,
            "actual_pct": float(np.expm1(actual) * 100),
            "explained_pct": float(np.expm1(explained) * 100),
            "unexplained_pct": float(np.expm1(actual - explained) * 100),
            "parts_pct": {c: float(np.expm1(v) * 100) for c, v in contrib.items()},
            "driver_moves": {c: float(row[c]) for c in fit["cols"]}}


def fair_value(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H,
               lam: float = RIDGE_LAMBDA, min_train: int = MIN_TRAIN,
               months: int = 12) -> pd.Series:
    """Cumulative unexplained move over the trailing `months` windows, walk-forward.

    Each point is the residual of one h-day window fitted only on data that closed
    before that window opened, and the points are sampled h days apart so no return
    is counted twice. The rolling sum answers the question the framework raises but
    cannot price: how much of gold's run is NOT explained by rates, the dollar,
    flows and positioning — i.e. how much is the official bid and the debasement
    trade."""
    cols = [c for c in MACRO_DRIVERS if c in X.columns]
    y = trailing(F, h)
    d = pd.concat([X[cols], y.rename("y")], axis=1).dropna()
    if len(d) <= min_train + h:
        return pd.Series(dtype=float)
    out, idx = {}, d.index
    for i in range(min_train, len(idx), h):
        tr = d.iloc[:max(i - h + 1, 1)]           # training window must be closed
        if len(tr) < min_train:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, np.nan)
        Z = ((tr[cols] - mu) / sd).fillna(0.0)
        ymu = float(tr["y"].mean())
        b = ridge(Z.to_numpy(), (tr["y"] - ymu).to_numpy(), lam)
        z = ((d[cols].iloc[i] - mu) / sd).fillna(0.0)
        out[idx[i]] = float(d["y"].iloc[i] - (float(z.to_numpy() @ b) + ymu))
    r = pd.Series(out).sort_index()
    return (r.rolling(months, min_periods=max(3, months // 2)).sum() * 100).rename("fv_gap_pct")


# ---------------------------------------------------------------------------
# 2. relationships and stability
# ---------------------------------------------------------------------------
def driver_table(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H) -> pd.DataFrame:
    """Per-driver: does it move WITH gold (coincident, explanatory) and does it move
    BEFORE gold (forward IC, predictive) — plus where it stands today."""
    y_fwd, y_now = target(F, h), trailing(F, 21)
    rows = []
    for name, (block, desc, sign) in SPEC.items():
        if name not in X.columns:
            continue
        s = X[name]
        ic = spearman(s, y_fwd)
        n = int(pd.concat([s, y_fwd], axis=1).dropna().shape[0])
        cur = s.dropna()
        rows.append({"feature": name, "block": block, "description": desc,
                     "prior_sign": sign,
                     "coincident_corr": spearman(s, y_now),
                     "forward_ic": ic, "t_stat": overlap_t(ic, n, h), "n": n,
                     "latest": float(cur.iloc[-1]) if len(cur) else np.nan,
                     "pctile": float((cur <= cur.iloc[-1]).mean() * 100) if len(cur) else np.nan})
    out = pd.DataFrame(rows)
    out["block"] = pd.Categorical(out["block"], BLOCK_ORDER, ordered=True)
    out["_abs"] = out["coincident_corr"].abs()
    return out.sort_values(["block", "_abs"], ascending=[True, False]).drop(columns="_abs")


def rolling_beta(F: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling 1y correlation of WEEKLY gold returns against the headline drivers.

    The stability check, and the most important table here: gold's correlation to
    the 10y real yield ran -0.5 to -0.7 for most of the sample, decayed to -0.19
    through 2024 and turned POSITIVE (+0.30) across 2025 before reverting to -0.45.
    A driver that flips sign cannot carry a static coefficient, and a full-sample
    average hides exactly that."""
    w = F.resample("W-FRI").last()
    gr = np.log(w["gold"] / w["gold"].shift(1))
    win = max(window // 5, 26)
    pairs = {"real_10y": ("Real 10y", "diff"), "dxy": ("DXY", "ret"),
             "vix": ("VIX", "diff"), "gld_tonnes": ("GLD tonnes", "ret"),
             "cuts_priced": ("Cuts priced", "diff"), "usdcny": ("USD/CNY", "ret"),
             "credit_baa": ("Baa spread", "diff")}
    out = {}
    for col, (label, how) in pairs.items():
        if col not in w.columns:
            continue
        d = (w[col] - w[col].shift(1)) if how == "diff" else np.log(w[col] / w[col].shift(1))
        out[label] = gr.rolling(win, min_periods=win // 2).corr(d)
    return pd.DataFrame(out).dropna(how="all")


# ---------------------------------------------------------------------------
# 3. the directional tilt — reported with its (weak) measured skill
# ---------------------------------------------------------------------------
def composite(X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Zero-parameter signed composite: z-score each feature, multiply by its PRIOR
    sign, average within block, average the blocks equally.

    No coefficients are fitted, so there is nothing to overfit — which is the point.
    With ~112 independent months in the sample a 23-parameter fit has no hope of
    separating signal from noise, and `walk_forward` confirms it (negative
    out-of-sample IC). Equal-weighting the theory is the honest estimator here."""
    z = pd.DataFrame({n: np.sign(sp) * _z(X[n], Z_WINDOW)
                      for n, (bk, ds, sp) in SPEC.items() if n in X.columns})
    blocks: dict = {}
    for n, (bk, ds, sp) in SPEC.items():
        if n in z.columns:
            blocks.setdefault(bk, []).append(n)
    bs = pd.DataFrame({bk: z[c].mean(axis=1) for bk, c in blocks.items()})
    return bs.mean(axis=1).rename("composite"), bs


def tilt_skill(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H) -> dict:
    """Measured skill of the composite, with the always-long benchmark alongside:
    gold rose through most of this sample, so 'up' is a strong null and beating
    direction alone proves nothing."""
    comp, bs = composite(X)
    y = target(F, h)
    d = pd.concat([comp, y], axis=1).dropna()
    if len(d) < 200:
        return {"n": int(len(d))}
    yc = d.columns[1]
    ic = spearman(d["composite"], d[yc])
    nonov = d.iloc[::h]
    blocks = {}
    for bk in bs.columns:
        dd = pd.concat([bs[bk], y], axis=1).dropna()
        i = spearman(dd[bk], dd[dd.columns[1]])
        blocks[bk] = {"ic": i, "t": overlap_t(i, len(dd), h)}
    t = overlap_t(ic, len(d), h)
    return {"n": int(len(d)), "n_independent": int(len(nonov)),
            "ic": ic, "ic_t": t,
            "ic_nonoverlap": spearman(nonov["composite"], nonov[yc]),
            "hit_rate": float((np.sign(d["composite"]) == np.sign(d[yc])).mean()),
            "always_long_hit": float((d[yc] > 0).mean()),
            "blocks": blocks,
            "verdict": ("no reliable directional edge — treat as a lean, not a forecast"
                        if not np.isfinite(t) or abs(t) < 2 else "statistically significant")}


def walk_forward(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H,
                 lam: float = RIDGE_LAMBDA, step: int = 5,
                 min_train: int = MIN_TRAIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window ridge on the FULL feature set, refit every `step` days.

    Kept as the control experiment, not as the product. It is what a naive
    kitchen-sink fit produces here — out-of-sample IC ≈ -0.03, hit rate below
    always-long — and having that number on the record is what justifies shipping
    the zero-parameter composite instead."""
    y = target(F, h)
    cols = [c for c in X.columns if X[c].notna().sum() > min_train]
    dates = X.index
    preds, actuals, when, beta_hist = [], [], [], {}
    beta = fit = None

    for i in range(min_train + h, len(dates)):
        t = dates[i]
        if beta is None or (i - min_train - h) % step == 0:
            cutoff = t - pd.Timedelta(days=int(h * 1.5))
            tr = X.loc[:cutoff, cols]
            ytr = y.loc[:cutoff].reindex(tr.index)
            d = tr[ytr.notna()]
            ytr = ytr[ytr.notna()]
            if len(d) < min_train:
                continue
            med = d.median()
            Xtr = d.fillna(med)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, np.nan)
            Ztr = ((Xtr - mu) / sd).fillna(0.0)
            ymu = float(ytr.mean())
            beta = ridge(Ztr.to_numpy(), ytr.to_numpy() - ymu, lam)
            beta_hist[t] = pd.Series(beta, index=cols)
            fit = (med, mu, sd, ymu)
        if beta is None:
            continue
        med, mu, sd, ymu = fit
        z = ((X.loc[t, cols].fillna(med) - mu) / sd).fillna(0.0)
        preds.append(float(z.to_numpy() @ beta + ymu))
        actuals.append(float(y.loc[t]) if pd.notna(y.loc[t]) else np.nan)
        when.append(t)

    res = pd.DataFrame({"pred": preds, "actual": actuals}, index=pd.DatetimeIndex(when))
    # betas come back alongside, never on res.attrs: pandas compares attrs on every
    # concat, and comparing a DataFrame-valued attr raises.
    return res, (pd.DataFrame(beta_hist).T if beta_hist else pd.DataFrame())


def score(res: pd.DataFrame, h: int = DEFAULT_H) -> dict:
    d = res.dropna()
    if len(d) < 60:
        return {"n": int(len(d))}
    ic = spearman(d["pred"], d["actual"])
    nonov = d.iloc[::h]
    return {"n": int(len(d)), "n_independent": int(len(nonov)),
            "ic": ic, "ic_t": overlap_t(ic, len(d), h),
            "hit_rate": float((np.sign(d["pred"]) == np.sign(d["actual"])).mean()),
            "always_long_hit": float((d["actual"] > 0).mean()),
            "strategy_sum_logret": float((np.sign(nonov["pred"]) * nonov["actual"]).sum()),
            "buyhold_sum_logret": float(nonov["actual"].sum())}


def tilt_now(F: pd.DataFrame, X: pd.DataFrame, h: int = DEFAULT_H) -> dict:
    """Today's directional lean, decomposed by block, with the historical return for
    this composite quintile attached so the size of the claim stays visible."""
    comp, bs = composite(X)
    y = target(F, h)
    d = pd.concat([comp, y], axis=1).dropna()
    yc = d.columns[1]
    cur = float(comp.dropna().iloc[-1])
    cuts = [d["composite"].quantile(x) for x in (0.2, 0.4, 0.6, 0.8)]
    which = int(np.searchsorted(cuts, cur))
    q = pd.Series(np.searchsorted(cuts, d["composite"].to_numpy()), index=d.index)
    hist = d.loc[q == which, yc]
    return {"score": cur,
            "pctile": float((comp.dropna() <= cur).mean() * 100),
            "quintile": which + 1,
            "blocks": {bk: float(bs[bk].dropna().iloc[-1]) for bk in bs.columns
                       if bs[bk].notna().any()},
            "quintile_mean_pct": float(np.expm1(hist.mean()) * 100) if len(hist) else np.nan,
            "quintile_hit_rate": float((hist > 0).mean()) if len(hist) else np.nan,
            "quintile_n": int(len(hist)),
            "sample_mean_pct": float(np.expm1(d[yc].mean()) * 100),
            "sample_hit_rate": float((d[yc] > 0).mean())}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def compute(h: int = DEFAULT_H, rebuild: bool = False) -> dict:
    F, status = golddata.build(force=True) if rebuild else golddata.load()
    X = features(F)
    fit = fit_coincident(F, X, h)                                  # exogenous only
    full = fit_coincident(F, X, h, cols=[c for c in COINCIDENT if c in X.columns])
    gap = fair_value(F, X, h)
    res, _betas = walk_forward(F, X, h)

    out = {
        "asof": F.index[-1].date().isoformat(),
        "gold_last": float(F["gold_raw"].iloc[-1]),
        "horizon_days": h,
        "data_status": {k: v for k, v in status.items() if not k.startswith("_")},
        "explanatory": {
            "r2_macro": fit["r2"], "n": fit["n"], "n_effective": fit["n_eff"],
            "resid_sd_pct": fit["resid_sd"] * 100,
            # The flow-inclusive fit is reported for completeness and never used
            # for scenarios: ETF tonnage and managed-money net are co-movers, so
            # the extra R² is partly gold explaining itself.
            "r2_with_flows": full["r2"], "resid_sd_with_flows_pct": full["resid_sd"] * 100,
        },
        "sensitivities": json.loads(sensitivities(fit).to_json(orient="records")),
        "sensitivities_with_flows": json.loads(
            sensitivities(full).to_json(orient="records")),
        "attribution": attribution(fit, F, X),
        "attribution_with_flows": attribution(full, F, X),
        "fair_value_gap_pct": float(gap.iloc[-1]) if len(gap) else None,
        "fair_value_gap_pctile": (float((gap <= gap.iloc[-1]).mean() * 100)
                                  if len(gap) else None),
        "tilt": tilt_now(F, X, h),
        "tilt_skill": tilt_skill(F, X, h),
        "kitchen_sink_skill": score(res, h),
        "drivers": json.loads(driver_table(F, X, h).to_json(orient="records")),
        "stability": json.loads(rolling_beta(F).resample("YE").last().round(3)
                                .to_json(orient="index", date_format="iso")),
        "built": datetime.now().isoformat(timespec="seconds"),
    }
    MODEL_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    h = DEFAULT_H
    if "--horizon" in sys.argv:
        h = int(sys.argv[sys.argv.index("--horizon") + 1])
    o = compute(h=h, rebuild="--rebuild" in sys.argv)
    e, t, k, a = o["explanatory"], o["tilt"], o["tilt_skill"], o["attribution"]
    print(f"Gold Driver Model — {o['asof']}, gold {o['gold_last']:,.0f}, horizon {h}d")
    print(f"  explains    R2 {e['r2_macro']:.2f} of the {h}d move from macro drivers "
          f"({e['n']} rows / {e['n_effective']} independent), unexplained sd "
          f"{e['resid_sd_pct']:.1f}%")
    print(f"              R2 {e['r2_with_flows']:.2f} adding ETF/positioning flows "
          f"— co-movers, descriptive only")
    print(f"  last {h}d    actual {a['actual_pct']:+.2f}%  = explained "
          f"{a['explained_pct']:+.2f}%  + unexplained {a['unexplained_pct']:+.2f}%")
    if o["fair_value_gap_pct"] is not None:
        print(f"  fair value  gold {o['fair_value_gap_pct']:+.1f}% vs drivers over 12m "
              f"({o['fair_value_gap_pctile']:.0f}th pctile of that gap)")
    print(f"  tilt        {t['score']:+.2f} ({t['pctile']:.0f}th pctile, quintile "
          f"{t['quintile']}/5) -> historically {t['quintile_mean_pct']:+.2f}% next {h}d "
          f"vs {t['sample_mean_pct']:+.2f}% base")
    print(f"  skill       IC {k['ic']:+.3f} (t={k['ic_t']:+.1f}) — {k['verdict']}")
    if "--full" in sys.argv:
        print("\n  Sensitivities — gold % per move in each driver:")
        for s in o["sensitivities"]:
            flag = "" if s["agrees_with_prior"] else "   <- against prior sign"
            print(f"    {s['driver']:28s} {s['move']:>7s} -> {s['gold_pct']:+6.2f}% "
                  f"(t={s['t_stat']:+.1f}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
