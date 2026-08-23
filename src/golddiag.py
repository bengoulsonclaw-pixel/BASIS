"""golddiag.py — Milestone 5 / Stage 1: diagnostics, no model (spec §6 Stage 1).

"This tells you which features carry signal before you spend time fitting." So this
module fits nothing. It measures three things and writes them down:

  1. **Rolling 252-day correlations** of every feature against every target — and,
     more usefully than the average, how STABLE each relationship is. A driver whose
     correlation flips sign across the sample cannot carry a static coefficient, and
     a full-sample number hides exactly that.
  2. **Cross-correlation at lags 0-60**, to separate features that move *with* gold
     from features that move *before* it. Only the second kind can forecast.
  3. **A heatmap and a written summary**, so the answer survives contact with a human.

Two statistical decisions that change the conclusions
-----------------------------------------------------
**The lead-lag study runs on DAILY returns, not on the forward targets.** Correlating
a feature against a 60-day forward return at 61 different lags means every observation
overlaps its neighbours 59/60ths of the way, so the correlations are enormously
autocorrelated and their significance is fiction. Against the daily return each
observation is used exactly once per lag, so n is honest and a t-stat means something.
The cost is that the daily return is noisier; the benefit is that the answer is real.

**Significance is Bonferroni-adjusted.** 28 features x 61 lags is 1,708 simultaneous
tests. At n≈4,500 the naive 5% threshold is |r| > 0.029, which roughly 85 pure-noise
cells would clear by construction — and the largest of them would look like a
discovery. The adjusted threshold is reported alongside the raw one, and the summary
counts only cells that clear it.

Output: data/gold_store/diagnostics.json + diagnostics.html (heatmap, native CSS —
no matplotlib in this repo and none added).

CLI:  python src/golddiag.py [--open]
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

from src import goldbacktest, goldfeatures, goldstore  # noqa: E402

STORE_DIR = _ROOT / "data" / "gold_store"
DIAG_JSON = STORE_DIR / "diagnostics.json"
DIAG_HTML = STORE_DIR / "diagnostics.html"

MAX_LAG = 60
ROLL_WINDOW = 252
PRICE = "LBMA_GOLD_PM_USD"

# Prior sign per feature, from the framework. Kept here so the diagnostics can say
# "agrees" or "disagrees" rather than leaving a reader to work it out — and NOT used
# to flip anything, because refitting a prior to the sample is how a backtest gets
# fitted to its own noise.
PRIOR_SIGN = {
    "real_yield_10y_chg_20d": -1, "real_yield_10y_z_1y": -1,
    "breakeven_10y_chg_20d": +1, "dxy_dist_50d": -1, "dxy_chg_20d": -1,
    "curve_2s10s": 0, "fed_cut_odds_chg_20d": -1,
    "cot_mm_net_pct_3y": -1, "cot_mm_net_chg_4w": -1, "cot_mm_net_pct_oi": -1,
    "etf_tonnage_chg_4w": +1, "etf_tonnage_chg_12w": +1, "gld_flow_z_1y": +1,
    "shanghai_premium_usd": +1, "shanghai_premium_z_1y": +1,
    "cb_net_purchases_12m": +1, "cb_net_purchases_yoy_chg": +1,
    "vix_z_1y": +1, "hy_spread_chg_20d": +1, "spx_dist_200d": -1,
    "real_rate_vol_20d": 0,
    "gold_dist_200d": -1, "gold_dist_50d": -1, "gold_mom_12m_1m": +1,
    "gold_silver_ratio_z_5y": -1, "gold_cpi_ratio_z_10y": -1,
    "gold_m2_ratio_z_10y": -1, "gold_fx_breadth": +1,
}


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def _t_from_r(r: float, n: int) -> float:
    if not np.isfinite(r) or abs(r) >= 1 or n < 5:
        return float("nan")
    return float(r * sqrt(n - 2) / sqrt(1 - r * r))


def _p_two_sided(t: float, n: int) -> float:
    """Normal approximation to the two-sided p-value — fine at n in the thousands,
    and this repo carries no scipy."""
    if not np.isfinite(t):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))))


def r_threshold(n: int, alpha: float) -> float:
    """Smallest |r| that clears `alpha` at sample size n (normal approximation)."""
    # invert the normal CDF by bisection — no scipy, and this runs once
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if _p_two_sided(_t_from_r(mid, n), n) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# 1. rolling correlations and stability
# ---------------------------------------------------------------------------
def rolling_stability(feats: pd.DataFrame, targets: pd.DataFrame,
                      window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Per (feature, target): full-sample correlation plus how much it moves around.

    `sign_stability` is the fraction of rolling windows agreeing in sign with the
    full-sample correlation. Near 1.0 means a relationship that held throughout;
    near 0.5 means one that reversed as often as it persisted, which is a
    coefficient no static model can carry."""
    rows = []
    for tname in goldbacktest.HORIZON_DAYS:
        h = goldbacktest.HORIZON_DAYS[tname]
        y = targets[tname]
        for f in feats.columns:
            d = pd.concat([feats[f], y], axis=1).dropna()
            if len(d) < window * 2:
                continue
            full = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
            roll = d.iloc[:, 0].rolling(window).corr(d.iloc[:, 1]).dropna()
            if roll.empty:
                continue
            roll = roll.replace([np.inf, -np.inf], np.nan).dropna()
            if roll.empty:
                continue
            same = float((np.sign(roll) == np.sign(full)).mean())
            # One-sidedness of the rolling sign, which — unlike agreement with the
            # full-sample sign — stays meaningful when the full-sample correlation is
            # near zero. A tiny full-sample r has an essentially arbitrary sign, so
            # "6% agreement" reads as instability when it is really just noise about
            # zero. 0.5 = coin flip, 1.0 = never changed sign.
            share_pos = float((roll > 0).mean())
            one_sided = max(share_pos, 1.0 - share_pos)
            # overlapping forward returns: the honest sample is n/h
            n_eff = max(int(len(d) / h), 3)
            rows.append({
                "feature": f, "target": tname, "corr": full,
                "roll_mean": float(roll.mean()), "roll_min": float(roll.min()),
                "roll_max": float(roll.max()), "sign_stability": same,
                "share_positive": share_pos, "one_sided": one_sided,
                "n": int(len(d)), "n_effective": n_eff,
                "t_overlap_adj": _t_from_r(full, n_eff),
                "prior": PRIOR_SIGN.get(f, 0),
                "agrees_with_prior": bool(np.sign(full) == np.sign(PRIOR_SIGN.get(f, 0))
                                          and PRIOR_SIGN.get(f, 0) != 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. cross-correlation — the lead-lag question
# ---------------------------------------------------------------------------
def cross_correlations(feats: pd.DataFrame, price: pd.Series,
                       max_lag: int = MAX_LAG) -> tuple:
    """corr(feature[t], daily gold return[t + L]) for L in 0..max_lag.

    L = 0 is the same day's return: a feature that only scores there is COINCIDENT
    and explains rather than predicts. A feature carrying information the market has
    not yet priced should score at L > 0. Using the daily return keeps every
    observation independent within a lag, which the forward targets cannot do."""
    ret = np.log(price / price.shift(1))
    out, counts = {}, {}
    for f in feats.columns:
        x = feats[f]
        series = {}
        for lag in range(0, max_lag + 1):
            d = pd.concat([x, ret.shift(-lag)], axis=1).dropna()
            if len(d) < 250:
                continue
            series[lag] = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
        if series:
            out[f] = series
            counts[f] = len(pd.concat([x, ret], axis=1).dropna())
    xc = pd.DataFrame(out).T          # rows = feature, cols = lag
    # Per-feature n, not the global minimum. Taking the min let the shortest series
    # (the Shanghai premium, 2,461 rows) set the significance bar for features with
    # 5,900 — making every threshold needlessly conservative and understating what
    # the long series can actually support.
    return xc, pd.Series(counts)


def lead_lag_summary(xc: pd.DataFrame, counts: pd.Series, n_tests: int) -> pd.DataFrame:
    """Peak lag and peak correlation per feature, each judged at ITS OWN sample size."""
    n_med = int(counts.median())
    raw = r_threshold(n_med, 0.05)
    adj = r_threshold(n_med, 0.05 / max(n_tests, 1))
    rows = []
    for f in xc.index:
        n_f = int(counts.get(f, n_med))
        raw_f = r_threshold(n_f, 0.05)
        adj_f = r_threshold(n_f, 0.05 / max(n_tests, 1))
        s = xc.loc[f].dropna()
        if s.empty:
            continue
        peak_lag = int(s.abs().idxmax())
        peak = float(s.loc[peak_lag])
        at0 = float(s.get(0, np.nan))
        lead = s.drop(index=0, errors="ignore")
        rows.append({
            "feature": f,
            "corr_lag0": at0,
            "peak_lag": peak_lag,
            "peak_corr": peak,
            "best_lead_lag": int(lead.abs().idxmax()) if not lead.empty else np.nan,
            "best_lead_corr": float(lead.loc[lead.abs().idxmax()]) if not lead.empty
            else np.nan,
            "n": n_f,
            "threshold_raw": raw_f,
            "threshold_bonferroni": adj_f,
            "clears_raw": bool(abs(peak) > raw_f),
            "clears_bonferroni": bool(abs(peak) > adj_f),
            "coincident": bool(peak_lag == 0),
            "prior": PRIOR_SIGN.get(f, 0),
        })
    out = pd.DataFrame(rows)
    out.attrs = {}
    return out.sort_values("peak_corr", key=abs, ascending=False), raw, adj


# ---------------------------------------------------------------------------
# 3. heatmap — native HTML/CSS, matching heatmap_html.py's no-dependency approach
# ---------------------------------------------------------------------------
def _cell_colour(v: float, scale: float) -> str:
    if not np.isfinite(v):
        return "#1b1f27"
    t = max(-1.0, min(1.0, v / scale))
    if t >= 0:                                    # green for positive
        r, g, b = int(44 - 20 * t), int(55 + 60 * t), int(66 - 20 * t)
    else:                                         # red for negative
        r, g, b = int(44 + 90 * -t), int(55 - 25 * -t), int(66 - 30 * -t)
    return f"rgb({r},{g},{b})"


def heatmap_html(xc: pd.DataFrame, summary: pd.DataFrame, raw: float, adj: float,
                 stab: pd.DataFrame) -> str:
    """Lead-lag heatmap + stability table as one self-contained HTML block."""
    scale = float(np.nanpercentile(np.abs(xc.to_numpy()), 98)) or 0.05
    lags = [c for c in xc.columns if c % 5 == 0]
    order = summary.set_index("feature").index.tolist()
    order = [f for f in order if f in xc.index]

    head = "".join(f"<th>{lg}</th>" for lg in lags)
    body = []
    for f in order:
        cells = []
        for lg in lags:
            v = xc.loc[f, lg] if lg in xc.columns else np.nan
            mark = "&bull;" if np.isfinite(v) and abs(v) > adj else ""
            cells.append(f'<td style="background:{_cell_colour(v, scale)}" '
                         f'title="{f} lag {lg}: r={v:+.3f}">{mark}</td>')
        pk = summary.set_index("feature").loc[f]
        body.append(f'<tr><th class="rl">{f}</th>{"".join(cells)}'
                    f'<td class="num">{pk["peak_corr"]:+.3f}</td>'
                    f'<td class="num">{int(pk["peak_lag"])}</td></tr>')

    st_rows = []
    for _, r in stab[stab["target"] == "fwd_ret_60d"].sort_values(
            "one_sided").head(12).iterrows():
        st_rows.append(f'<tr><td class="rl">{r["feature"]}</td>'
                       f'<td class="num">{r["corr"]:+.3f}</td>'
                       f'<td class="num">{r["roll_min"]:+.2f}</td>'
                       f'<td class="num">{r["roll_max"]:+.2f}</td>'
                       f'<td class="num">{r["one_sided"]:.0%}</td></tr>')

    return f"""<!doctype html><meta charset="utf-8">
<title>Gold Signal Engine — Stage 1 diagnostics</title>
<style>
 body{{background:#12151b;color:#CDD3DB;font:13px 'IBM Plex Sans',Segoe UI,sans-serif;
       padding:24px;max-width:1400px}}
 h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 8px;color:#AEB7C2}}
 p.sub{{color:#9FA9B5;margin:0 0 18px}}
 table{{border-collapse:collapse;font-size:11px}}
 th,td{{padding:3px 6px;text-align:center}}
 th.rl{{text-align:left;color:#CDD3DB;font-weight:400;white-space:nowrap;padding-right:12px}}
 td.num{{font-family:'IBM Plex Mono',monospace;color:#CDD3DB}}
 thead th{{color:#9FA9B5;font-weight:400}}
 td{{color:#e8edf3;font-size:10px}}
 .note{{color:#9FA9B5;margin-top:10px;line-height:1.5}}
</style>
<h1>Gold Signal Engine — Stage 1 diagnostics</h1>
<p class="sub">No model fitted. Built {datetime.now():%Y-%m-%d %H:%M}.</p>

<h2>Lead–lag: corr(feature[t], daily gold return[t+lag])</h2>
<table><thead><tr><th class="rl">feature</th>{head}
<th>peak r</th><th>lag</th></tr></thead><tbody>{"".join(body)}</tbody></table>
<p class="note">&bull; marks a cell clearing the <b>Bonferroni-adjusted</b> threshold
|r| &gt; {adj:.3f} ({len(xc)}&times;{len(xc.columns)} = {len(xc) * len(xc.columns):,}
simultaneous tests). The unadjusted 5% threshold is |r| &gt; {raw:.3f} — at this
sample size roughly 5% of pure-noise cells clear that, so it is shown only to make
the gap visible. Lag 0 is the same day: a feature scoring only there is
<i>coincident</i>, and explains gold rather than forecasting it.</p>

<h2>Least stable relationships (60-day target, rolling {ROLL_WINDOW}d)</h2>
<table><thead><tr><th class="rl">feature</th><th>full-sample r</th><th>roll min</th>
<th>roll max</th><th>one-sided</th></tr></thead><tbody>{"".join(st_rows)}</tbody></table>
<p class="note">One-sidedness is how lopsided the rolling correlation's sign is:
100% never changed sign, 50% flipped as often as it held. It is reported instead of
agreement-with-the-full-sample because a near-zero full-sample correlation has an
essentially arbitrary sign, which makes that measure read as instability when it is
really just noise about zero. Values near 50% are the case for the regime layer.</p>
"""


# ---------------------------------------------------------------------------
def run() -> dict:
    feats, targets = goldfeatures.load()
    panel = goldstore.daily_panel([PRICE], start=str(feats.index.min().date()))
    price = panel[PRICE].reindex(feats.index).ffill()

    stab = rolling_stability(feats, targets)
    xc, counts = cross_correlations(feats, price)
    n_tests = int(xc.shape[0] * xc.shape[1]) if len(xc) else 1
    summary, raw, adj = lead_lag_summary(xc, counts, n_tests)
    n = int(counts.median()) if len(counts) else 0

    DIAG_HTML.write_text(heatmap_html(xc, summary, raw, adj, stab), encoding="utf-8")

    lead_only = summary[(~summary["coincident"]) & summary["clears_bonferroni"]]
    coincident = summary[summary["coincident"] & summary["clears_bonferroni"]]
    out = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "n_observations": n,
        "n_tests": n_tests,
        "threshold_raw_5pct": raw,
        "threshold_bonferroni": adj,
        "n_clearing_raw": int(summary["clears_raw"].sum()),
        "n_clearing_bonferroni": int(summary["clears_bonferroni"].sum()),
        "n_coincident": int(len(coincident)),
        "n_leading": int(len(lead_only)),
        "lead_lag": json.loads(summary.to_json(orient="records")),
        "stability": json.loads(stab.to_json(orient="records")),
        "heatmap": str(DIAG_HTML),
    }
    DIAG_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    o = run()
    s = pd.DataFrame(o["lead_lag"])
    stab = pd.DataFrame(o["stability"])
    print("Stage 1 diagnostics — no model fitted\n")
    print(f"  {o['n_observations']:,} daily observations, {o['n_tests']:,} simultaneous "
          f"tests")
    print(f"  5% threshold |r| > {o['threshold_raw_5pct']:.3f}   "
          f"Bonferroni-adjusted |r| > {o['threshold_bonferroni']:.3f}")
    print(f"  clearing raw: {o['n_clearing_raw']}/{len(s)}   "
          f"clearing adjusted: {o['n_clearing_bonferroni']}/{len(s)}")
    print(f"  of those adjusted-significant: {o['n_coincident']} COINCIDENT (lag 0), "
          f"{o['n_leading']} LEADING (lag > 0)\n")

    print("  strongest relationships (by |peak r| against the daily return):")
    print(f"    {'feature':28s} {'lag0':>7s} {'peak':>7s} {'lag':>4s}  {'sig':<5s} kind")
    for _, r in s.head(14).iterrows():
        sig = "adj" if r["clears_bonferroni"] else ("5%" if r["clears_raw"] else "-")
        kind = "coincident" if r["coincident"] else f"leads by {int(r['peak_lag'])}d"
        print(f"    {r['feature']:28s} {r['corr_lag0']:+7.3f} {r['peak_corr']:+7.3f} "
              f"{int(r['peak_lag']):4d}  {sig:<5s} {kind}")

    print("\n  least stable at the 60-day horizon — one-sidedness of the rolling sign")
    print("  (50% = the correlation flipped as often as it held):")
    sub = stab[stab["target"] == "fwd_ret_60d"].sort_values("one_sided")
    for _, r in sub.head(8).iterrows():
        print(f"    {r['feature']:28s} r={r['corr']:+.3f}  "
              f"range [{r['roll_min']:+.2f},{r['roll_max']:+.2f}]  "
              f"one-sided {r['one_sided']:.0%} (+ve {r['share_positive']:.0%})")
    print(f"\n  heatmap -> {o['heatmap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
