"""goldbacktest.py — Milestone 4: the walk-forward harness (spec §7).

Built BEFORE any model, deliberately. A harness written after the model it is meant
to judge tends to be written until the model passes; written first, it is just a
measuring instrument, and the four benchmarks are already in it waiting.

Nothing in this module knows what a model is beyond `fit(X, y)` / `predict(X)`. The
four benchmarks implement that interface and are the bar every later model has to
clear (spec §12: no model ships without beating all four on walk-forward).

The three things that make a backtest honest
--------------------------------------------
1. **Expanding window, refit on a cadence.** Train on everything up to T, predict
   forward, roll. Monthly refits for the 5-day model, quarterly for the longer ones.
2. **Purge and embargo.** A row dated T carries a label that is not resolved until
   T+h, so training on it while testing at t < T+h trains on the answer. Rows are
   purged until their label has fully resolved, and then an additional `embargo`
   (defaulting to the horizon) is dropped on top. Without this, overlapping forward
   returns leak across the train/test boundary and every metric flatters.
3. **A locked holdout.** The last three years are not touched during development.
   `run()` refuses to score them unless explicitly unlocked, and unlocking writes a
   dated line to the run log — so "we only looked once" is a fact on disk rather
   than a recollection.

Calibration expectations, from the spec, recorded so nobody misreads a result
----------------------------------------------------------------------------
    5-day    53-55% is a strong result. **Above 60% means a bug or a leak** — the
             harness raises it as a warning rather than a triumph.
    60-day   56-60% is achievable; this horizon holds most of the real edge.
    250-day  high hit rates appear here purely because gold drifts up. Always
             compare to always-long, never to 50%.

CLI:
    python src/goldbacktest.py --benchmarks         score the four, dev period only
    python src/goldbacktest.py --benchmarks --holdout   UNLOCKS the holdout (logged)
    python src/goldbacktest.py --log                show past runs
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import goldfeatures  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
RUN_LOG = STORE_DIR / "backtest_runs.jsonl"

# Spec §7.4 — ten years of training before the first prediction.
MIN_TRAIN_DAYS = 2520
# Spec §7.5 — the final three years are locked.
HOLDOUT_YEARS = 3
# Spec §7 metrics — 10bp per trade.
COST_BPS = 10.0
# Spec §7 — long when probability above 0.55, flat below.
LONG_THRESHOLD = 0.55

REFIT_DAYS = {"fwd_ret_5d": 21, "fwd_ret_60d": 63, "fwd_ret_250d": 63}
HORIZON_DAYS = {"fwd_ret_5d": 5, "fwd_ret_60d": 60, "fwd_ret_250d": 250}

# Spec §7 calibration bands: (floor worth noting, ceiling above which we suspect a bug)
CALIBRATION = {"fwd_ret_5d": (0.53, 0.60), "fwd_ret_60d": (0.56, 0.65),
               "fwd_ret_250d": (0.00, 1.00)}


# ---------------------------------------------------------------------------
# small stats, numpy only
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    v = np.asarray(x, dtype=float)
    return np.array([0.5 * (1.0 + erf(float(z) / sqrt(2.0))) for z in v.ravel()]
                    ).reshape(v.shape)


def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks with TIES AVERAGED.

    The first version assigned ties consecutive ranks in array order, which for a
    constant prediction meant ranking it in CALENDAR order — and a calendar-ordered
    "prediction" correlates with anything that trends. That is the only reason
    random_walk, which predicts all zeros, scored IC +0.020 / +0.035 / +0.191 across
    the three horizons, and it is why the claim "the only model with positive IC at
    all three horizons" was measured with a broken instrument."""
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average the ranks within each run of equal values
    srt = a[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b) -> float:
    d = pd.concat([pd.Series(a).reset_index(drop=True),
                   pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    if len(d) < 20:
        return float("nan")
    x, y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    # A constant series has no ranking. Returning a number here invents one.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a hit rate.

    Wilson rather than the textbook normal interval because n here is the number of
    INDEPENDENT observations — the sample divided by the horizon — and at a 250-day
    horizon that can be under twenty. The normal approximation is badly behaved at
    those counts and would quote a confidence the data cannot support."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(model_correct: np.ndarray, base_correct: np.ndarray) -> tuple:
    """Exact two-sided McNemar test on PAIRED directional calls.

    Two unpaired Wilson intervals and a bare difference — what this module reported
    before — cannot answer "is this model better than that one", because the two
    models are scored on the SAME days and their errors are heavily correlated. Only
    the discordant pairs carry information: b = model right / base wrong, c = the
    reverse. A +2.6pp "edge" over 461 observations is 12 net calls, and the paired
    test is the only thing that says whether 12 is a lot."""
    both = np.isfinite(model_correct) & np.isfinite(base_correct)
    m, base = model_correct[both].astype(bool), base_correct[both].astype(bool)
    b = int(np.sum(m & ~base))
    c = int(np.sum(~m & base))
    n = b + c
    if n == 0:
        return b, c, float("nan")
    k = min(b, c)
    if n <= 60:
        # exact binomial, two-sided
        from math import comb
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return b, c, float(min(1.0, 2.0 * tail))
    # Normal approximation with a continuity correction. The exact form computes
    # 2**n, which overflows a float above n≈1000 — and n IS the discordant-pair
    # count, which reached ~250 only once the sample was deepened from 6 to 21 years.
    # A bug the shallow sample could never have surfaced.
    # max(..., 0): with b == c the continuity correction would go NEGATIVE and hand
    # back p ≈ 0.91 for two models that agreed exactly.
    z = max(abs(b - c) - 1.0, 0.0) / sqrt(n)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return b, c, float(min(1.0, max(0.0, p)))


# ---------------------------------------------------------------------------
# benchmark models — the bar every real model must clear (spec §7)
# ---------------------------------------------------------------------------
class _Base:
    benchmark_only = False
    name = "base"
    needs = ()                      # feature columns required; () = none

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.mu_ = float(y.mean())
        self.sd_ = float(y.std()) or 1.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RandomWalk(_Base):
    """No change. The null that says gold is a martingale and nothing is knowable."""
    name = "random_walk"

    def predict(self, X):
        return np.zeros(len(X))


class AlwaysLong(_Base):
    """Gold drifted up for the whole sample. Plenty of models fail to beat this, and
    a 250-day hit rate that looks spectacular is usually just this in disguise.

    This is a POSITION benchmark, not a probability forecast, and it has to be held
    long unconditionally or it is not the thing its name claims. It used to emit the
    trailing mean of the vol-scaled target, which the harness then put through
    Phi(): at the 5-day horizon that is about 0.52, below the 0.55 entry threshold,
    so the "always long" benchmark sat in CASH for the whole sample and booked a
    time_in_market of 0.0 while the page displayed its hit rate as a strategy result.

    Emitting a large positive constant makes Phi() saturate and the position stick at
    1.0. Its Brier score is meaningless by construction (it forecasts P(up)=1 every
    day) and `benchmark_only` tells the reporting layer not to score it as a forecast.
    The hit-rate benchmark is `base_rate`, computed separately from the realised
    up-frequency, and buy-and-hold P&L is measured directly off `actual` — so neither
    the headline comparison nor `excess_vs_buyhold` ever depended on this class."""
    name = "always_long"
    benchmark_only = True

    def predict(self, X):
        # 8 sd: Phi(8) rounds to 1.0, so the position is long on every date.
        return np.full(len(X), 8.0)


class Momentum12m(_Base):
    """Pure 12-month momentum, skipping the last month."""
    name = "momentum_12m"
    needs = ("gold_mom_12m_1m",)

    def predict(self, X):
        return X["gold_mom_12m_1m"].fillna(0.0).to_numpy()


class RealYieldOnly(_Base):
    """The single variable the framework calls the most important. Sign is negative:
    rising real yields, falling gold. Fitted scale, fixed sign."""
    name = "real_yield_only"
    needs = ("real_yield_10y_chg_20d",)

    def fit(self, X, y):
        super().fit(X, y)
        x = X["real_yield_10y_chg_20d"].fillna(0.0).to_numpy()
        v = float(np.var(x))
        self.beta_ = float(np.cov(x, y.to_numpy())[0, 1] / v) if v > 0 else 0.0

    def predict(self, X):
        return self.beta_ * X["real_yield_10y_chg_20d"].fillna(0.0).to_numpy()


BENCHMARKS = (RandomWalk, AlwaysLong, Momentum12m, RealYieldOnly)


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------
def holdout_start(index: pd.DatetimeIndex) -> pd.Timestamp:
    """First date of the locked holdout — the last HOLDOUT_YEARS of the sample."""
    return pd.Timestamp(index.max()) - pd.DateOffset(years=HOLDOUT_YEARS)


def walk_forward(model_cls, X: pd.DataFrame, y: pd.Series, horizon: int,
                 refit_days: int, min_train: int = MIN_TRAIN_DAYS,
                 embargo: int | None = None, end: pd.Timestamp | None = None,
                 start: pd.Timestamp | None = None, on_fit=None) -> pd.DataFrame:
    """Expanding-window walk-forward with purge + embargo.

    At prediction date t the training set is every row whose label has fully
    resolved by `t - embargo`: rows dated T with `T + horizon <= t - embargo`. The
    purge is the `T + horizon` term (the label must be in the past) and the embargo
    is the extra buffer on top. Default embargo is the horizon itself, which is the
    strict reading of spec §7.2 and costs a little sample to buy the guarantee that
    no overlapping label straddles the boundary.

    `on_fit(t, last_train_date, n_rows)` fires at each refit. It exists so the purge
    boundary can be asserted against what the model was actually handed rather than
    against a reading of this function, and it is useful for diagnosing a suspicious
    result later."""
    embargo = horizon if embargo is None else embargo
    d = pd.concat([X, y.rename("_y")], axis=1)
    dates = X.index
    # Horizons are in TRADING days (the target is `shift(-h)` on a business-day
    # index), so the purge must be too. Subtracting calendar days delivered a 1-3
    # trading-day embargo at h=5 and 106-108 at h=250 — nothing leaked, but the
    # guarantee was not the one the docstring claimed, and the test asserting it was
    # tautological because it recomputed the same calendar quantity.
    # Positions are on the FULL index; `dates` below is filtered by start/end, so
    # looking a full-index position up in the filtered array runs off the end.
    full_dates = X.index
    pos_of = pd.Series(np.arange(len(full_dates)), index=full_dates)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates < pd.Timestamp(end)]

    preds, whens, model = [], [], None
    last_fit = None
    for t in dates:
        cut_pos = int(pos_of[t]) - int(horizon + embargo)
        if cut_pos < 1:
            continue
        cutoff = full_dates[cut_pos]
        if last_fit is None or (t - last_fit).days >= refit_days:
            tr = d.loc[:cutoff].dropna(subset=["_y"])
            need = [c for c in getattr(model_cls, "needs", ()) if c in tr.columns]
            tr = tr.dropna(subset=need) if need else tr
            if len(tr) < min_train:
                continue
            model = model_cls()
            model.fit(tr.drop(columns="_y"), tr["_y"])
            last_fit = t
            if on_fit is not None:
                on_fit(t, tr.index.max(), len(tr))
        if model is None:
            continue
        row = X.loc[[t]]
        try:
            p = float(np.asarray(model.predict(row)).ravel()[0])
        except Exception:
            continue
        preds.append(p)
        whens.append(t)

    out = pd.DataFrame({"pred": preds}, index=pd.DatetimeIndex(whens))
    out["actual"] = y.reindex(out.index)
    return out


def to_probability(res: pd.DataFrame) -> pd.Series:
    """Map a predicted **vol-scaled** return to P(up) as Phi(pred).

    Models are fitted on the volatility-scaled target, so a prediction is already
    expressed in standard deviations of the h-period return — which makes the normal
    CDF of it the implied probability directly, with no free parameters.

    The first version of this z-scored predictions by their OWN dispersion, and that
    was wrong in a way worth recording. AlwaysLong emits a near-constant, so its
    dispersion is nearly zero, and dividing by it amplified refit-to-refit noise into
    confident 0.05/0.95 probabilities. The result: a Brier score of 0.43 against a
    climatology of 0.25, and a "buy and hold" benchmark that churned in and out of the
    market and lost money. Scaling a forecast by how much it happens to vary is not a
    calibration, it is a magnifying glass pointed at noise."""
    return pd.Series(_norm_cdf(res["pred"].to_numpy()), index=res.index)


def metrics(res: pd.DataFrame, horizon: int, base: pd.DataFrame | None = None) -> dict:
    """Spec §7's metric list, averaged over EVERY sampling phase.

    Non-overlapping sampling is necessary — consecutive daily rows share almost all
    of their forward window — but `d.iloc[::horizon]` silently picks ONE of `horizon`
    equally valid offsets, always the one starting at row 0. On the real 5-day result
    the five phases gave edges over always-long of +2.60, +2.39, +0.65, +0.22 and
    +1.74 percentage points. Reporting phase 0 alone turned a +1.5pp average into a
    +2.6pp headline, and the choice of phase was an accident of where the index
    happened to start.

    So every statistic is computed per phase and averaged, and `*_phase_spread`
    records the range. A number that swings from +0.22 to +2.60 depending on an
    arbitrary offset should never be quoted as a point estimate, and now it cannot be
    quoted without its spread sitting next to it.

    `base` is the benchmark's result frame; when supplied, a PAIRED McNemar test is
    run against it on the same dates."""
    d = res.dropna(subset=["pred", "actual"])
    if len(d) < 30:
        return {"n": int(len(d)), "insufficient": True}
    prob_all = to_probability(d)

    per_phase = []
    for phase in range(horizon):
        nonov = d.iloc[phase::horizon]
        if len(nonov) < 10:
            continue
        prob = prob_all.reindex(nonov.index)
        up = (nonov["actual"] > 0).astype(int)
        directional = nonov["pred"] != 0
        nd = int(directional.sum())
        if nd:
            hits = int(((nonov.loc[directional, "pred"] > 0)
                        == (nonov.loc[directional, "actual"] > 0)).sum())
            hit_rate = hits / nd
        else:
            hits, hit_rate = 0, float("nan")
        pos = (prob > LONG_THRESHOLD).astype(float)
        turns = pos.diff().abs().fillna(pos.abs())
        pnl = pos * nonov["actual"] - turns * (COST_BPS / 1e4)
        curve = pnl.cumsum()
        per_phase.append({
            "n_ind": len(nonov), "n_dir": nd, "hits": hits, "hit_rate": hit_rate,
            "base": float(up.mean()),
            "brier": float(((prob - up) ** 2).mean()),
            "pnl": float(pnl.sum()), "buyhold": float(nonov["actual"].sum()),
            "dd": float((curve - curve.cummax()).min()) if len(curve) else float("nan"),
            "turnover": float(turns.sum()), "trades": int((turns > 0).sum()),
            "time_in_market": float(pos.mean()),
        })
    if not per_phase:
        return {"n": int(len(d)), "insufficient": True}

    def _m(k):
        vals = [p[k] for p in per_phase if np.isfinite(p[k])]
        return float(np.mean(vals)) if vals else float("nan")

    def _span(vals):
        """Min/max, NaN-safe. A model that makes no directional calls has an
        all-NaN hit rate, and np.nanmin on that warns and returns NaN — noise in the
        log for a case that is perfectly legitimate (random_walk does it by design)."""
        v = [x for x in vals if np.isfinite(x)]
        return [float(min(v)), float(max(v))] if v else [float("nan"), float("nan")]

    hit_rate = _m("hit_rate")
    base_rate = _m("base")
    hits_tot = int(sum(p["hits"] for p in per_phase))
    ndir_tot = int(sum(p["n_dir"] for p in per_phase))
    lo, hi = wilson_interval(int(round(hit_rate * per_phase[0]["n_dir"])),
                             per_phase[0]["n_dir"]) if ndir_tot else (np.nan, np.nan)

    # Pooled over every daily row — uses all the evidence, at the cost of
    # overlapping observations, so it is reported for comparison but never used for
    # significance.
    dd = d[d["pred"] != 0]
    pooled_hit = (float(((dd["pred"] > 0) == (dd["actual"] > 0)).mean())
                  if len(dd) else float("nan"))
    pooled_base = float((d["actual"] > 0).mean())

    out = {
        "n": int(len(d)),
        "n_independent": int(np.mean([p["n_ind"] for p in per_phase])),
        "n_directional": int(np.mean([p["n_dir"] for p in per_phase])),
        "phases_scored": len(per_phase),
        "hit_rate": hit_rate, "hit_ci": [lo, hi],
        "hit_phase_spread": _span([p["hit_rate"] for p in per_phase]),
        "always_long_hit": base_rate,
        "edge_vs_always_long": (hit_rate - base_rate if np.isfinite(hit_rate)
                                else float("nan")),
        "edge_phase_spread": _span([p["hit_rate"] - p["base"] for p in per_phase]),
        "pooled_hit_rate": pooled_hit,
        "pooled_edge": pooled_hit - pooled_base if np.isfinite(pooled_hit) else np.nan,
        "brier": _m("brier"),
        "ic": spearman(d["pred"], d["actual"]),
        "ic_n_effective": int(len(d) / max(horizon, 1)),
        "strategy_sum_logret": _m("pnl"),
        "buyhold_sum_logret": _m("buyhold"),
        "excess_vs_buyhold": _m("pnl") - _m("buyhold"),
        "max_drawdown": _m("dd"),
        "turnover": _m("turnover"),
        "trades": _m("trades"),
        "time_in_market": _m("time_in_market"),
    }

    if base is not None:
        b = base.dropna(subset=["pred", "actual"])
        common = d.index.intersection(b.index)
        if len(common) > 30:
            mc = ((d.loc[common, "pred"] > 0)
                  == (d.loc[common, "actual"] > 0)).to_numpy()
            bc = (b.loc[common, "actual"] > 0).to_numpy()   # always-long is right when up
            step = max(horizon, 1)
            bb, cc, pv = mcnemar_p(mc[::step], bc[::step])
            out["mcnemar"] = {"b_model_only": bb, "c_base_only": cc, "p_value": pv,
                              "n_discordant": bb + cc}
    return out


def check_calibration(target: str, m: dict, model: str = "") -> list:
    """Spec §12: a 5-day hit rate above 60% is treated as a bug until proven
    otherwise. This returns warnings rather than raising, because the harness's job
    is to report honestly — including reporting that a result is too good."""
    warn = []
    # Name the model. Three identical "no directional calls" lines with no
    # attribution is not a warning, it is a puzzle.
    who = f"{target}/{model}" if model else target
    if m.get("insufficient"):
        return [f"{who}: only {m.get('n', 0)} observations — not scored"]
    floor, ceiling = CALIBRATION.get(target, (0.0, 1.0))
    hr = m.get("hit_rate", float("nan"))
    if not np.isfinite(hr):
        return [f"{who}: makes no directional calls — hit rate undefined"]
    if hr > ceiling:
        warn.append(f"{who}: hit rate {hr:.1%} exceeds {ceiling:.0%} — TREAT AS A "
                    f"BUG OR LEAK until proven otherwise (spec §12)")
    if m["n_independent"] < 20:
        warn.append(f"{who}: only {m['n_independent']} independent observations — "
                    f"the confidence interval is wider than the effect")
    return warn


# ---------------------------------------------------------------------------
# run logging (spec §7.6 — no unlogged runs)
# ---------------------------------------------------------------------------
def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True, timeout=15
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _feature_version(X: pd.DataFrame) -> str:
    """Hash of the feature set, so a run record identifies exactly which columns the
    model saw. Renaming or adding a feature changes the version."""
    return hashlib.md5(",".join(sorted(X.columns)).encode()).hexdigest()[:10]


def log_run(record: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_log() -> list:
    if not RUN_LOG.exists():
        return []
    return [json.loads(line) for line in RUN_LOG.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
def run(models=BENCHMARKS, targets=None, use_holdout: bool = False,
        scaled: bool = True, note: str = "", feature_set=None,
        min_train: int = MIN_TRAIN_DAYS) -> dict:
    """Score models across the horizons and write a run record.

    `use_holdout=False` (the default) stops the evaluation window at the start of the
    locked three-year holdout. Passing True is the one-time final evaluation and is
    recorded as such in the log, with its date."""
    feats, targs = goldfeatures.load()
    if feature_set:
        # A named subset (e.g. goldfeatures.DEEP_FEATURES) trades breadth for depth.
        # Restricting columns also drops any BUCKET that has no surviving feature,
        # which BucketModel renormalises over — so a deep run tests a narrower model,
        # not the same one on more data.
        keep = [c for c in feature_set if c in feats.columns]
        feats = feats[keep].dropna(how="all")
    targets = targets or list(HORIZON_DAYS)
    cut = holdout_start(feats.index)

    results: dict = {}
    warnings: list = []
    for target in targets:
        h = HORIZON_DAYS[target]
        col = target + "_scaled" if scaled else target
        y = targs[col].reindex(feats.index)
        X = feats
        per_model = {}
        base_res = None
        # AlwaysLong first, so every later model can be tested PAIRED against it.
        ordered = ([e for e in models if isinstance(e, type) and e is AlwaysLong]
                   + [e for e in models if not (isinstance(e, type) and e is AlwaysLong)])
        for entry in ordered:
            # An entry is either a model CLASS (used as-is) or a factory f(horizon)
            # returning one — the elastic net needs the horizon for both its purge
            # and its Newey-West lag, and the harness instantiates with no arguments.
            mc = entry if isinstance(entry, type) else entry(h)
            res = walk_forward(mc, X, y, horizon=h, refit_days=REFIT_DAYS[target],
                               min_train=min_train,
                               start=cut if use_holdout else None,
                               end=None if use_holdout else cut)
            # score against RAW returns — P&L and hit rates must be in return space,
            # not in the scaled space the model was fitted on
            res["actual"] = targs[target].reindex(res.index)
            m = metrics(res, h, base=base_res)
            if mc is AlwaysLong:
                base_res = res
            per_model[mc.name] = m
            warnings += check_calibration(target, m, mc.name)
        results[target] = per_model

    record = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_hash(),
        "feature_version": _feature_version(feats),
        "n_features": int(feats.shape[1]),
        "window": "HOLDOUT (locked)" if use_holdout else "development",
        "holdout_start": str(cut.date()),
        "target_scaled": scaled,
        "models": [(m.name if isinstance(m, type) else m(21).name) for m in models],
        "min_train_days": min_train,
        "feature_set": "deep" if feature_set else "full",
        "cost_bps": COST_BPS,
        "note": note,
        "results": results,
        "warnings": warnings,
    }
    log_run(record)
    return record


def main() -> int:
    if "--log" in sys.argv:
        for r in read_log():
            print(f"{r['run_at']}  {r['window']:18s} git={r['git']} "
                  f"features={r['feature_version']} models={len(r['models'])}")
        return 0

    use_holdout = "--holdout" in sys.argv
    if use_holdout:
        print("\n  *** UNLOCKING THE THREE-YEAR HOLDOUT ***")
        print("  Spec §7.5: run against it ONCE, at the end. This is being logged.\n")
    # --full adds the candidate models to the four benchmarks; --deep runs them on
    # the long-history feature subset. Both configurations existed only as
    # hand-assembled one-off calls, so the run the BASIS page displays could not be
    # reproduced from the CLI — and a plain `python goldbacktest.py` silently replaced
    # a 7-model record with a 4-model one.
    models, feature_set, bits = list(BENCHMARKS), None, []
    if "--full" in sys.argv or "--deep" in sys.argv:
        try:
            from goldmodels import ElasticNetModel
            from goldbuckets import bucket_model_for
            models += [ElasticNetModel,
                       bucket_model_for(60, "equal", False),
                       bucket_model_for(60, "equal", True)]
            bits.append("candidate models")
        except Exception as e:
            print(f"  (candidate models unavailable: {e})")
    if "--deep" in sys.argv:
        import goldfeatures
        feature_set = goldfeatures.DEEP_FEATURES
        bits.append(f"DEEP feature set ({len(feature_set)} features)")
    note = ("holdout evaluation" if use_holdout else
            (" + ".join(bits) if bits else "benchmark baseline"))
    rec = run(models=tuple(models), use_holdout=use_holdout,
              feature_set=feature_set, note=note)

    print(f"Gold backtest — {rec['window']}, holdout starts {rec['holdout_start']}")
    print(f"  git {rec['git']}  features {rec['feature_version']} "
          f"({rec['n_features']})  cost {rec['cost_bps']:.0f}bp\n")
    for target, per in rec["results"].items():
        print(f"  {target}")
        print(f"    {'model':18s} {'indep':>6s} {'hit':>7s} {'vs long':>8s} "
              f"{'spread':>14s} {'McNemar':>8s} {'IC':>7s} {'P&L':>8s} {'vs B&H':>8s}")
        for name, m in per.items():
            if m.get("insufficient"):
                print(f"    {name:18s} {m['n']:5d}  insufficient data")
                continue
            hr = m["hit_rate"]
            hrs = f"{hr:6.1%}" if np.isfinite(hr) else "     -"
            ed = (f"{m['edge_vs_always_long']:+7.1%}"
                  if np.isfinite(m["edge_vs_always_long"]) else "      -")
            sp = m.get("edge_phase_spread", [np.nan, np.nan])
            sps = (f"{sp[0]:+.1%}..{sp[1]:+.1%}" if np.isfinite(sp[0]) else "        -")
            mc_ = m.get("mcnemar") or {}
            _pv = mc_.get("p_value")
            # `x == x` was the NaN test, but it is also True for None==None, so a
            # McNemar block that never ran (no common dates) passed the guard and
            # KeyError'd on the subscript. Test for a real number instead.
            ps = f"{_pv:8.3f}" if isinstance(_pv, float) and _pv == _pv else "       -"
            # P&L against BUY-AND-HOLD, not against a benchmark that never traded.
            print(f"    {name:18s} {m['n_independent']:6d} {hrs} {ed} {sps:>14s} "
                  f"{ps} {m['ic']:+7.3f} {m['strategy_sum_logret']:+8.3f} "
                  f"{m['excess_vs_buyhold']:+8.3f}")
            if m.get("time_in_market") == 0:
                print(f"    {'':18s}   ^ never entered the market — its P&L of 0.000 is "
                      f"cash, not a strategy")
        print()
    if rec["warnings"]:
        print("  WARNINGS:")
        for w in dict.fromkeys(rec["warnings"]):
            print(f"    - {w}")
    print(f"\n  logged to {RUN_LOG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
