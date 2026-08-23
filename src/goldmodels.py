"""goldmodels.py — Milestone 6 / Stage 2: the regularised linear baseline (spec §6).

An elastic net per horizon, fitted through the Milestone 4 harness, with two things
the spec is specific about:

  * **Hyperparameters chosen by PURGED cross-validation inside the training window**
    (§7.3). Ordinary K-fold would put a row dated T in the training folds while a
    validation fold covers T+h — the label overlaps, and alpha gets tuned on the
    answer. Purging drops every training row whose label window touches the
    validation block, plus an embargo either side.
  * **Newey-West standard errors at lag = the forecast horizon** (§6 Stage 2). The
    residuals of an h-day overlapping regression are autocorrelated by construction;
    plain OLS errors understate the true ones by roughly sqrt(h), which turns a
    t of 1.2 into a t of 4 and a coefficient worth nothing into a finding.

Why elastic net and not OLS
---------------------------
Real yields, the dollar and Fed pricing are three views of one thing, and the feature
matrix carries several near-duplicates by design (`dxy_dist_50d` and `dxy_chg_20d`,
`gold_cpi_ratio_z_10y` and `gold_m2_ratio_z_10y`). OLS answers collinearity with huge
offsetting coefficients that are individually meaningless and jointly unstable.
The L1 term drops redundant columns and the L2 term shares weight across the ones
that survive, which is the only way the coefficient table is worth reading.

What Stage 1 predicts about this
--------------------------------
Every feature that cleared the multiple-comparison threshold in `golddiag` is
COINCIDENT — none leads gold. A linear model over coincident features has nothing to
forecast with, so this should NOT beat always-long out of sample. Writing that down
before running it is the point: a baseline that surprises you is usually a leak.

CLI:  python src/goldmodels.py [--horizon 60d] [--coefficients]
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

from src import goldbacktest as bt  # noqa: E402
from src import goldfeatures, golddiag  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
COEF_FILE = STORE_DIR / "stage2_coefficients.json"

# The grid must reach far enough that the CV optimum is INTERIOR. The first version
# stopped at 0.1 and the CV picked exactly 0.1 — a boundary solution, which means the
# grid, not the data, chose the answer. Extended until the optimum sits inside it; if
# CV wants the top of the range it is saying "shrink everything to zero", and that is
# a result worth reading rather than an artefact worth hiding.
ALPHAS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
L1_RATIOS = (0.1, 0.5, 0.9)
N_CV_SPLITS = 5


# ---------------------------------------------------------------------------
# purged cross-validation (spec §7.3)
# ---------------------------------------------------------------------------
def purged_cv_splits(n: int, n_splits: int, horizon: int, embargo: int | None = None):
    """Contiguous validation blocks with overlapping training labels purged.

    Yields (train_positions, val_positions). A training row at position p carries a
    label spanning [p, p+horizon]; it is admissible only if that whole span sits
    clear of the validation block by at least `embargo` on the relevant side.

    Contiguous blocks, not shuffled folds — spec §12 forbids random splits anywhere,
    and shuffling a time series destroys exactly the dependence the purge exists to
    respect."""
    embargo = horizon if embargo is None else embargo
    if n_splits < 2 or n < n_splits * (horizon + embargo + 10):
        return
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        v0, v1 = edges[k], edges[k + 1]
        val = np.arange(v0, v1)
        if len(val) < 5:
            continue
        pos = np.arange(n)
        before = pos + horizon < v0 - embargo       # label closes before the block
        after = pos > v1 + embargo                  # row starts after the block
        train = pos[before | after]
        if len(train) < 50:
            continue
        yield train, val


# ---------------------------------------------------------------------------
# Newey-West (spec §6 Stage 2)
# ---------------------------------------------------------------------------
def newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC standard errors with Bartlett weights at `lags`.

    With h-day overlapping returns the residuals are autocorrelated out to h by
    construction, so the OLS covariance is wrong by roughly a factor of sqrt(h).
    Reporting a coefficient's significance without this correction is the single
    easiest way to publish a finding that is not there."""
    n, k = X.shape
    if n <= k + 1:
        return np.full(k, np.nan)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)
    Xu = X * resid[:, None]
    S = Xu.T @ Xu
    for lag in range(1, int(lags) + 1):
        if lag >= n:
            break
        w = 1.0 - lag / (lags + 1.0)                # Bartlett kernel
        G = Xu[lag:].T @ Xu[:-lag]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class ElasticNetModel(bt._Base):
    """Elastic net with purged-CV hyperparameter selection, fit-time only.

    Everything that could leak — the median used to fill gaps, the standardisation
    mean and scale, the chosen alpha — is computed inside `fit`, on the training
    slice the harness handed over, and nothing else."""

    name = "elastic_net"
    _horizon = 60
    _alphas = ALPHAS
    _l1_ratios = L1_RATIOS
    _n_splits = N_CV_SPLITS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        super().fit(X, y)
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import ElasticNet

        self.cols_ = [c for c in X.columns if X[c].notna().sum() > len(X) * 0.5]
        Xd = X[self.cols_]
        self.median_ = Xd.median()
        Xf = Xd.fillna(self.median_).fillna(0.0)
        self.mu_x_ = Xf.mean()
        self.sd_x_ = Xf.std().replace(0, np.nan).fillna(1.0)
        Z = ((Xf - self.mu_x_) / self.sd_x_).to_numpy()
        yv = y.to_numpy()
        self.y_mu_ = float(np.mean(yv))
        yc = yv - self.y_mu_

        n_nonconverged = 0
        best = (np.inf, self._alphas[len(self._alphas) // 2], self._l1_ratios[1])
        splits = list(purged_cv_splits(len(Z), self._n_splits, self._horizon))
        if splits:
            for a in self._alphas:
                for l1 in self._l1_ratios:
                    errs = []
                    for tr, va in splits:
                        # max_iter must be generous. At small alpha, coordinate
                        # descent on a collinear matrix needs many passes, and a fit
                        # that stops early scores a WORSE validation MSE — so the
                        # grid search would reject small alphas for failing to
                        # converge rather than for overfitting, and the "no features"
                        # conclusion would be an artefact of the solver.
                        m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=100000,
                                       tol=1e-5, fit_intercept=False, random_state=0)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", ConvergenceWarning)
                            m.fit(Z[tr], yc[tr])
                            if any(issubclass(w.category, ConvergenceWarning)
                                   for w in caught):
                                n_nonconverged += 1
                        errs.append(float(np.mean((yc[va] - m.predict(Z[va])) ** 2)))
                    mse = float(np.mean(errs))
                    if mse < best[0]:
                        best = (mse, a, l1)
        self.cv_mse_, self.alpha_, self.l1_ratio_ = best
        # Flag a boundary solution rather than reporting it as a choice.
        self.alpha_at_boundary_ = bool(
            splits and self.alpha_ in (self._alphas[0], self._alphas[-1]))

        self.n_nonconverged_ = n_nonconverged
        self.model_ = ElasticNet(alpha=self.alpha_, l1_ratio=self.l1_ratio_,
                                 max_iter=100000, tol=1e-5, fit_intercept=False,
                                 random_state=0)
        self.model_.fit(Z, yc)
        self.coef_ = pd.Series(self.model_.coef_, index=self.cols_)

        resid = yc - self.model_.predict(Z)
        se = newey_west_se(Z, resid, lags=self._horizon)
        self.se_ = pd.Series(se, index=self.cols_)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.t_ = pd.Series(np.where(self.se_ > 0, self.coef_ / self.se_, np.nan),
                                index=self.cols_)
        self.n_selected_ = int((self.coef_.abs() > 1e-12).sum())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.cols_].fillna(self.median_).fillna(0.0)
        Z = ((Xf - self.mu_x_) / self.sd_x_).to_numpy()
        return self.model_.predict(Z) + self.y_mu_


def elastic_net_for(horizon: int, **kw):
    """Configured subclass — the harness instantiates models with no arguments, so
    the horizon (needed for both the purge and the Newey-West lag) is bound here."""
    return type("ElasticNet_%dd" % horizon, (ElasticNetModel,),
                {"_horizon": int(horizon), "name": "elastic_net", **kw})


# ---------------------------------------------------------------------------
def coefficient_report(target: str = "fwd_ret_60d") -> pd.DataFrame:
    """Fit once on the full development window and report the coefficient table with
    Newey-West t-stats — spec §6 Stage 2's explicit deliverable.

    This is an IN-SAMPLE description of what the model leans on, not evidence of
    skill. Skill is what the harness measures."""
    feats, targs = goldfeatures.load()
    cut = bt.holdout_start(feats.index)
    h = bt.HORIZON_DAYS[target]
    X = feats.loc[:cut]
    y = targs[target + "_scaled"].reindex(X.index)
    d = pd.concat([X, y.rename("_y")], axis=1).dropna(subset=["_y"])
    m = elastic_net_for(h)()
    m.fit(d.drop(columns="_y"), d["_y"])
    out = pd.DataFrame({"coef": m.coef_, "nw_se": m.se_, "nw_t": m.t_})
    out["selected"] = out["coef"].abs() > 1e-12
    out["prior"] = [golddiag.PRIOR_SIGN.get(c, 0) for c in out.index]
    out.attrs = {}
    return out.sort_values("coef", key=abs, ascending=False), m


def main() -> int:
    target = "fwd_ret_60d"
    if "--horizon" in sys.argv:
        target = "fwd_ret_" + sys.argv[sys.argv.index("--horizon") + 1].lstrip("fwd_ret_")
    if target not in bt.HORIZON_DAYS:
        print(f"unknown horizon {target}; use one of {list(bt.HORIZON_DAYS)}")
        return 2

    tbl, m = coefficient_report(target)
    print(f"Stage 2 elastic net — {target}, development window only")
    edge = "  <- AT GRID BOUNDARY" if getattr(m, "alpha_at_boundary_", False) else ""
    if getattr(m, "n_nonconverged_", 0):
        print(f"  WARNING: {m.n_nonconverged_} CV fits did not converge — the grid "
              f"search may be rejecting small alphas for solver reasons")
    print(f"  alpha={m.alpha_:.4f}{edge}  l1_ratio={m.l1_ratio_:.1f}  "
          f"purged-CV MSE={m.cv_mse_:.4f}  {m.n_selected_}/{len(tbl)} features selected")
    print(f"  Newey-West lag = {m._horizon} (the forecast horizon)\n")
    print(f"    {'feature':28s} {'coef':>9s} {'NW se':>9s} {'NW t':>7s}  prior")
    for name, r in tbl[tbl["selected"]].iterrows():
        flag = "" if r["prior"] == 0 else (
            "  ok" if np.sign(r["coef"]) == np.sign(r["prior"]) else "  <- vs prior")
        print(f"    {name:28s} {r['coef']:+9.4f} {r['nw_se']:9.4f} "
              f"{r['nw_t']:+7.2f}{flag}")
    kept = tbl[tbl["selected"]]
    strong = kept[kept["nw_t"].abs() > 2]
    print(f"\n  {len(strong)} of {len(kept)} selected coefficients clear |t| > 2 "
          f"with Newey-West errors")
    COEF_FILE.write_text(json.dumps({
        "target": target, "alpha": m.alpha_, "l1_ratio": m.l1_ratio_,
        "n_selected": m.n_selected_, "nw_lag": m._horizon,
        "coefficients": json.loads(tbl.to_json(orient="index")),
    }, indent=2, default=str), encoding="utf-8")
    print(f"  written to {COEF_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
