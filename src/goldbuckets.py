"""goldbuckets.py — Milestone 7 / Stage 3: bucket scores + the §8 output contract.

Stage 3 is a different bet from Stage 2, not a refinement of it. The elastic net was
handed 26 free parameters and, under purged CV, set all of them to zero — with ~112
independent months there is not enough sample to estimate 26 coefficients. Bucket
scores sidestep that: the equal-weight variant fits NOTHING inside a bucket, so a
penalty cannot shrink it away and it needs no sample to estimate. The theory supplies
the signs; the data only ever chooses how much to trust each bucket.

Both variants the spec asks for are implemented and compared:

    equal   z-score each feature on training statistics, multiply by its PRIOR sign,
            average within bucket, squash to [-1, +1]. Zero fitted parameters.
    net     a per-bucket elastic net fitted to the target; the bucket score is that
            sub-model's standardised prediction. Five buckets x a handful of features
            each is a far smaller estimation problem than one 26-feature fit.

...crossed with both weighting schemes:

    fixed   the spec's §6 Stage 3 table, per horizon.
    fitted  weights re-estimated on the training slice only, then compared against
            the fixed set — which is what the spec means by "treat these as starting
            values".

Four models, all scored through the Milestone 4 harness against the same four
benchmarks. Nothing here is exempt from §12.

CLI:  python src/goldbuckets.py [--contract] [--horizon 60d]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import goldbacktest as bt  # noqa: E402
from src import goldfeatures, goldmodels, goldstore  # noqa: E402
from src.golddiag import PRIOR_SIGN  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
CONTRACT_FILE = STORE_DIR / "signal.json"

BUCKETS = ("monetary", "flows", "physical", "risk", "valuation")

# Spec §6 Stage 3 starting weights, per horizon.
FIXED_WEIGHTS = {
    5:   {"monetary": 0.20, "flows": 0.30, "physical": 0.05, "risk": 0.20,
          "valuation": 0.25},
    60:  {"monetary": 0.40, "flows": 0.20, "physical": 0.15, "risk": 0.10,
          "valuation": 0.15},
    250: {"monetary": 0.25, "flows": 0.10, "physical": 0.30, "risk": 0.05,
          "valuation": 0.30},
}

# Feature -> bucket. goldfeatures.FEATURE_BUCKETS is the source of truth; this maps
# its labels onto the spec's five names.
_BUCKET_ALIAS = {"monetary": "monetary", "flows": "flows", "physical": "physical",
                 "risk": "risk", "valuation": "valuation"}


def feature_buckets() -> dict:
    return {f: _BUCKET_ALIAS.get(b, b)
            for f, b in goldfeatures.FEATURE_BUCKETS.items()}


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------
class BucketModel(bt._Base):
    """Five bucket scores in [-1, +1], combined by horizon weights.

    Every statistic that could leak — the per-feature median, mean and standard
    deviation, the per-bucket sub-models, the fitted weights — is computed inside
    `fit` on the training slice the harness supplied."""

    name = "bucket"
    _horizon = 60
    _mode = "equal"           # "equal" | "net"
    _refit_weights = False

    # ---- helpers ----------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.cols_].fillna(self.median_).fillna(0.0)
        return (Xf - self.mu_) / self.sd_

    def _scores_equal(self, Z: pd.DataFrame) -> pd.DataFrame:
        """Signed z-scores averaged within bucket, squashed to [-1, +1].

        tanh rather than a hard clip: a clip makes every extreme reading identical
        and throws away the difference between 'stretched' and 'unprecedented',
        while tanh is smooth, bounded, and near-linear in the middle where most
        observations live."""
        out = {}
        for b in BUCKETS:
            cols = [c for c in self.cols_ if self.buckets_.get(c) == b]
            if not cols:
                out[b] = pd.Series(0.0, index=Z.index)
                continue
            signed = pd.concat([Z[c] * float(np.sign(PRIOR_SIGN.get(c, 0)) or 0.0)
                                for c in cols], axis=1)
            out[b] = np.tanh(signed.mean(axis=1))
        return pd.DataFrame(out, index=Z.index)

    def _scores_net(self, Z: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for b in BUCKETS:
            sub = self.sub_.get(b)
            cols = [c for c in self.cols_ if self.buckets_.get(c) == b]
            if sub is None or not cols:
                out[b] = pd.Series(0.0, index=Z.index)
                continue
            raw = sub.predict(Z[cols].to_numpy())
            sd = self.sub_sd_.get(b) or 1.0
            out[b] = pd.Series(np.tanh(raw / sd), index=Z.index)
        return pd.DataFrame(out, index=Z.index)

    def bucket_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._prepare(X)
        return self._scores_equal(Z) if self._mode == "equal" else self._scores_net(Z)

    # ---- fit / predict ----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        super().fit(X, y)
        self.buckets_ = feature_buckets()
        self.cols_ = [c for c in X.columns
                      if X[c].notna().sum() > len(X) * 0.5 and c in self.buckets_]
        Xd = X[self.cols_]
        self.median_ = Xd.median()
        Xf = Xd.fillna(self.median_).fillna(0.0)
        self.mu_ = Xf.mean()
        self.sd_ = Xf.std().replace(0, np.nan).fillna(1.0)
        Z = (Xf - self.mu_) / self.sd_
        yv = y.to_numpy()
        self.y_mu_, self.y_sd_ = float(np.mean(yv)), float(np.std(yv)) or 1.0

        self.sub_, self.sub_sd_ = {}, {}
        if self._mode == "net":
            from sklearn.linear_model import ElasticNet
            for b in BUCKETS:
                cols = [c for c in self.cols_ if self.buckets_.get(c) == b]
                if len(cols) < 2:
                    continue
                m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=100000, tol=1e-5,
                               fit_intercept=False, random_state=0)
                m.fit(Z[cols].to_numpy(), yv - self.y_mu_)
                self.sub_[b] = m
                pred = m.predict(Z[cols].to_numpy())
                self.sub_sd_[b] = float(np.std(pred)) or 1.0

        # Which buckets actually have features in THIS training slice. The physical
        # bucket loses both Shanghai-premium features on any window starting before
        # 2017 (coverage is judged over the whole slice), and without renormalising,
        # 5-30% of the weight was being multiplied by a hard zero for roughly a third
        # of the sample — silently shrinking the combined score toward nothing.
        self.live_buckets_ = [b for b in BUCKETS
                              if any(self.buckets_.get(c) == b for c in self.cols_)]
        base = FIXED_WEIGHTS.get(self._horizon, FIXED_WEIGHTS[60])
        live_total = sum(base[b] for b in self.live_buckets_) or 1.0
        self.weights_ = {b: (base[b] / live_total if b in self.live_buckets_ else 0.0)
                         for b in BUCKETS}
        self.dead_buckets_ = [b for b in BUCKETS if b not in self.live_buckets_]
        self.weights_fixed_ = dict(base)
        scores = self.bucket_scores(X)
        if self._refit_weights:
            self.weights_ = self._fit_weights(scores, yv - self.y_mu_)

    def _fit_weights(self, scores: pd.DataFrame, yc: np.ndarray) -> dict:
        """Re-estimate the horizon weights on the training slice.

        Ridge, and then normalised so the absolute weights sum to 1 — which keeps the
        fitted set on the same scale as the spec's fixed set, so "compare against the
        fixed set" is a comparison of shape rather than of magnitude. Negative weights
        are allowed: if the data says a bucket works backwards, hiding that behind a
        non-negativity constraint would make the comparison meaningless."""
        S = scores[list(BUCKETS)].to_numpy()
        lam = 1.0
        try:
            beta = np.linalg.solve(S.T @ S + lam * np.eye(S.shape[1]), S.T @ yc)
        except np.linalg.LinAlgError:
            return dict(self.weights_fixed_)
        total = float(np.abs(beta).sum())
        if not np.isfinite(total) or total < 1e-12:
            return dict(self.weights_fixed_)
        return {b: float(w / total) for b, w in zip(BUCKETS, beta)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s = self.bucket_scores(X)
        combo = sum(s[b] * self.weights_.get(b, 0.0) for b in BUCKETS)
        # bucket scores are bounded [-1,1]; the target is in standard deviations, so
        # rescale by the training target's spread to keep predictions comparable
        return (combo * self.y_sd_ + self.y_mu_).to_numpy()


def bucket_model_for(horizon: int, mode: str = "equal", refit: bool = False):
    tag = f"bucket_{mode}_{'fitted' if refit else 'fixed'}"
    return type(f"Bucket_{mode}_{horizon}", (BucketModel,),
                {"_horizon": int(horizon), "_mode": mode, "_refit_weights": refit,
                 "name": tag})


def all_bucket_models(horizon: int) -> list:
    return [bucket_model_for(horizon, "equal", False),
            bucket_model_for(horizon, "equal", True),
            bucket_model_for(horizon, "net", False),
            bucket_model_for(horizon, "net", True)]


# ---------------------------------------------------------------------------
# §8 output contract
# ---------------------------------------------------------------------------
def _confidence(disagreement: float, n_flags: int, beats_benchmark: bool) -> str:
    """Low / medium / high.

    Deliberately conservative: a model that does not beat always-long can never be
    reported as high confidence, whatever the bucket scores agree on. Confidence is
    about the forecast being worth acting on, not about the drivers lining up."""
    if not beats_benchmark:
        return "low"
    if disagreement > 0.5 or n_flags > 2:
        return "low"
    return "medium" if disagreement > 0.25 else "high"


def _top_drivers(model: BucketModel, X_row: pd.DataFrame, bucket: str, k: int = 2):
    cols = [c for c in model.cols_ if model.buckets_.get(c) == bucket]
    if not cols:
        return []
    Z = model._prepare(X_row)
    signed = {c: float(Z[c].iloc[-1] * (np.sign(PRIOR_SIGN.get(c, 0)) or 0.0))
              for c in cols}
    return [c for c, _ in sorted(signed.items(), key=lambda kv: -abs(kv[1]))[:k]]


def build_contract(target: str = "fwd_ret_60d", mode: str = "equal",
                   refit: bool = False, beats_benchmark: bool = False) -> dict:
    """One row of the §8 contract for the latest available date."""
    feats, targs = goldfeatures.load()
    h = bt.HORIZON_DAYS[target]
    cut = bt.holdout_start(feats.index)

    y = targs[target + "_scaled"].reindex(feats.index)
    train = pd.concat([feats.loc[:cut], y.loc[:cut].rename("_y")], axis=1)
    train = train.dropna(subset=["_y"])
    m = bucket_model_for(h, mode, refit)()
    m.fit(train.drop(columns="_y"), train["_y"])

    row = feats.iloc[[-1]]
    scores = m.bucket_scores(row).iloc[-1]
    pred = float(m.predict(row)[0])
    prob = float(bt._norm_cdf(np.array([pred]))[0])

    vol = targs["realised_vol_60d"].reindex(feats.index).ffill().iloc[-1]
    expected_return = float(pred * vol * np.sqrt(h)) if np.isfinite(vol) else float("nan")

    disagreement = float(np.std([scores[b] for b in BUCKETS]))
    flags = list(goldstore.stale_flags())
    if "cot_mm_net_pct_3y" in feats.columns:
        pctl = feats["cot_mm_net_pct_3y"].dropna()
        if len(pctl) and pctl.iloc[-1] > 80:
            flags.append("positioning_crowded")
        elif len(pctl) and pctl.iloc[-1] < 20:
            flags.append("positioning_washed_out")
    for fid, reason in goldfeatures.MISSING_FEATURES.items():
        flags.append(f"feature_unavailable:{fid}")
    if not beats_benchmark:
        # The single most important flag in the payload. Every §12 comparison so far
        # says this engine does not beat always-long, and a consumer reading only
        # probability_up would never learn that.
        flags.insert(0, "model_does_not_beat_always_long_benchmark")

    return {
        "as_of": str(feats.index[-1].date()),
        "horizon": target.replace("fwd_ret_", ""),
        "probability_up": round(prob, 4),
        "expected_return": round(expected_return, 5),
        "confidence": _confidence(disagreement, len(flags), beats_benchmark),
        # Stage 4 is Milestone 8. Emitting a plausible label here would be inventing
        # a classifier that has not been built or validated.
        "regime": "unclassified (regime layer is Milestone 8)",
        "buckets": {
            b: {
                "score": round(float(scores[b]), 4),
                "weight": round(float(m.weights_.get(b, 0.0)), 4),
                "contribution": round(float(scores[b] * m.weights_.get(b, 0.0)), 4),
                "top_drivers": _top_drivers(m, row, b),
            } for b in BUCKETS
        },
        "disagreement": round(disagreement, 4),
        "flags": flags,
        "model": {"variant": m.name, "weights_fixed": m.weights_fixed_,
                  "weights_used": {k: round(v, 4) for k, v in m.weights_.items()},
                  "trained_to": str(cut.date())},
    }


def main() -> int:
    target = "fwd_ret_60d"
    if "--horizon" in sys.argv:
        target = "fwd_ret_" + sys.argv[sys.argv.index("--horizon") + 1].lstrip("fwd_ret_")
    if "--contract" in sys.argv:
        c = build_contract(target)
        CONTRACT_FILE.write_text(json.dumps(c, indent=2), encoding="utf-8")
        print(json.dumps(c, indent=2))
        return 0

    h = bt.HORIZON_DAYS[target]
    feats, targs = goldfeatures.load()
    cut = bt.holdout_start(feats.index)
    y = targs[target + "_scaled"].reindex(feats.index)
    d = pd.concat([feats.loc[:cut], y.loc[:cut].rename("_y")], axis=1).dropna(subset=["_y"])

    print(f"Stage 3 bucket weights — {target}, development window\n")
    print(f"    {'bucket':12s} {'spec fixed':>11s} {'refitted':>10s}")
    fixed = FIXED_WEIGHTS[h]
    for mode in ("equal", "net"):
        m = bucket_model_for(h, mode, refit=True)()
        m.fit(d.drop(columns="_y"), d["_y"])
        print(f"\n  {mode} scores:")
        for b in BUCKETS:
            print(f"    {b:12s} {fixed[b]:11.2f} {m.weights_[b]:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
