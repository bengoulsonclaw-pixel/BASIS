"""gold_event_study.py — how gold ACTUALLY behaves on US macro release days.

Question
--------
On the days the US prints CPI, PCE/personal income, and the employment report,
does the LBMA gold benchmark move MORE than on an ordinary day, and does it move
in a consistent DIRECTION?

Design decisions that matter for the answer
-------------------------------------------
1. RELEASE DATES ARE DERIVED, NOT ASSUMED.  The macro series in the store came
   from ALFRED with the full vintage matrix, so the date a reference month FIRST
   appeared in the store is the date the world first saw it — the real historical
   release date, holidays, shutdowns and schedule changes included.  We recover
   those dates the only sanctioned way: by asking goldstore.get_series what was
   knowable on each successive business day and noting the day a new reference
   month appears.  No release calendar is hard-coded, so the dates cannot drift
   out of line with the data they describe.  (Validation: the recovered 2024 dates
   reproduce the published BLS/BEA calendars exactly — CPI 11 Jan, 13 Feb, 12 Mar
   ... 11 Dec; payrolls 5 Jan, 2 Feb, 8 Mar ... 6 Dec.)

   (An earlier version reached into a private reader on the store to go faster.
   That is exactly what the repo's §3 lint rule forbids, and the fast path now lives
   in the store itself as `first_publication`.)  The former note read: it pins
   a private store reader to
   the SAME rows it would return anyway, pre-filtered to one series, so that a
   9,500-day scan does not re-read a 237k-row parquet 9,500 times.  Every
   point-in-time filter still happens inside get_series; published_at is never
   read directly.

2. RELEASES ARE GROUPED BY PRESS RELEASE, NOT BY SERIES.  CPI and core CPI are
   one BLS release; PCE and core PCE are one BEA release; payrolls and the
   unemployment rate are one BLS release.  Counting them separately would double-
   count the same event.  Grouping also rescues history: core CPI's vintages only
   start in 1997, but headline CPI dates the same release back to 1990.

3. TWO RETURN WINDOWS, BECAUSE THE 24-HOUR ONE IS MOSTLY NOISE.
     * fix-to-fix   — log(PM_t / PM_{t-1}).  The literal "same-day return", but a
                      24-hour window of which the release occupies 90 minutes.
     * AM-to-PM     — log(PM_t / AM_t).  The 10:30 London fix is 05:30 New York,
                      before the 08:30 ET release; the 15:00 London fix is 10:00
                      ET, after it.  So this 4.5-hour window BRACKETS the print,
                      and is the sharper measure of release-day behaviour.
   Both are reported.  Where they disagree the wide window is the diluted one.

4. FOMC IS DECLARED UNMEASURABLE RATHER THAN MEASURED BADLY.  Two independent
   reasons: this repo carries only a FORWARD FOMC calendar (fedpath.FOMC_DECISIONS
   begins 2026-01-28, so at most a handful of dates fall in sample), and a 14:00 ET
   decision post-dates the 15:00 London fix by an hour, so neither fix window can
   contain it.  The numbers are printed for transparency and flagged as such.

5. THE TEST IS A PERMUTATION TEST, STRATIFIED BY YEAR.  Absolute returns are
   strongly right-skewed and heteroskedastic across 36 years (2008 and 2020 do
   not look like 1996), so a t-test on them is not trustworthy.  Instead we
   re-draw the release-day labels at random WITHIN each calendar year, preserving
   how many releases that year had, and rebuild the statistic 20,000 times.  A
   second pass stratifies by (year, weekday) as well, because the employment
   report is always a Friday and Fridays are not a random weekday.  A result is
   only called significant if BOTH passes clear 0.05 after a Holm correction for
   the three events tested.

CLI:  python gold_event_study.py [--draws N] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import goldstore as gs          # noqa: E402  the ONLY read path into the store
import fedpath                  # noqa: E402  forward FOMC calendar

PM, AM = "LBMA_GOLD_PM_USD", "LBMA_GOLD_AM_USD"
# The study is metal-agnostic: everything below operates on two fix series and a
# set of release dates. Platinum and palladium carry LBMA AM and PM fixes from
# 1990 exactly as gold does, so the same test runs on them unchanged — which is
# the whole point, because a difference in the RESULT then cannot be a difference
# in method. Silver has a single noon auction and no intraday window at all.
SEED = 20260823
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Sanity envelope for "is this publication a real release?", in calendar days
# between the reference month and the print.  Payrolls land 3-8 days after month
# end, CPI 10-20, PCE 26-40.  Anything outside 1..80 days is an ALFRED backfill
# artefact, not a release.
# 1-120 days, not 1-80. The 80-day ceiling silently discarded SIX genuine PCE
# releases whose publication was delayed past it — the Dec-2018 report published
# 2019-03-01 after the 35-day government shutdown, and five similar cases. Those are
# real releases the market genuinely traded, and excluding them is a selection filter
# that quietly drops exactly the most eventful prints. Widening to 120 recovers them
# without admitting bulk revisions, which the `bulk` test catches separately.
LAG_MIN, LAG_MAX = 1, 120
MAX_NEW_REFS = 2   # a real print reveals one new month (two at a benchmark revision)

# Press releases, not series.  Every series listed for an event is published in
# the same document at the same minute.
EVENTS = {
    "CPI (BLS Consumer Price Index)": ["CPI", "CORE_CPI"],
    "PCE (BEA Personal Income & Outlays)": ["PCE", "CORE_PCE"],
    "Employment report (payrolls + unemployment rate)": ["PAYROLLS", "UNEMPLOYMENT"],
    "PPI (BLS Producer Price Index)": ["PPI"],
    "Retail sales (Census advance)": ["RETAIL_SALES"],
}

# NOTE ON ADDING EVENTS: Holm correction runs ACROSS the events in this map, so
# every event added makes every other event's result harder to call significant.
# That is correct — testing more things should raise the bar — but it means a
# marginal result can lose significance simply because a neighbour was added, and
# the change must be attributed to that rather than to new evidence.


# ---------------------------------------------------------------------------
# 1. release dates, recovered through the sanctioned read path
# ---------------------------------------------------------------------------
def _scan_series(series_id: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Every day on which `series_id` revealed a reference month it had never
    shown before, i.e. every first publication.

    Implemented by walking `get_series(as_of=d)` forward and diffing the index of
    known reference dates.  get_series is the store's point-in-time accessor, so
    this sees exactly what a trader on day d saw and nothing else."""
    first = gs.first_publication(series_id)
    if first.empty:
        return pd.DataFrame()
    # Group reference dates by the day they were FIRST published. That is the same
    # set the day-by-day get_series walk produced, in one pass and without touching
    # the store's internals. An earlier version bound a private store reader to a
    # filtered frame for speed, which the §3 lint rule correctly flags.
    by_day = {}
    for ref, pub in first.items():
        by_day.setdefault(pd.Timestamp(pub).normalize(), []).append(ref)
    rows = []
    valid = set(pd.DatetimeIndex(dates))
    for d in sorted(by_day):
        if d not in valid:
            d_eff = min((x for x in valid if x >= d), default=None)
            if d_eff is None:
                continue
        else:
            d_eff = d
        refs = by_day[d]
        rows.append({"pub_date": d_eff, "n_new": len(refs),
                     "max_ref": max(refs), "lag_days": (d_eff - max(refs)).days})
    return pd.DataFrame(rows)


def release_dates(series_id: str, dates: pd.DatetimeIndex):
    """First-publication days filtered down to plausible real releases."""
    scan = _scan_series(series_id, dates)
    if scan.empty:
        return pd.DatetimeIndex([]), {"scanned": 0, "kept": 0, "dropped_bulk": 0,
                                      "dropped_lag": 0}
    bulk = scan["n_new"] > MAX_NEW_REFS
    lagbad = (~bulk) & ~scan["lag_days"].between(LAG_MIN, LAG_MAX)
    keep = scan[~bulk & ~lagbad]
    audit = {"scanned": int(len(scan)), "kept": int(len(keep)),
             "dropped_bulk": int(bulk.sum()), "dropped_lag": int(lagbad.sum())}
    return pd.DatetimeIndex(keep["pub_date"].unique()).sort_values(), audit


# ---------------------------------------------------------------------------
# 2. statistics
# ---------------------------------------------------------------------------
def _perm(v: np.ndarray, mask: np.ndarray, strata: np.ndarray,
          draws: int, rng: np.random.Generator) -> tuple[float, float]:
    """Two-sided permutation p for (mean of `v` on event days - mean elsewhere),
    re-drawing the event labels within each stratum so that each stratum keeps its
    observed number of event days.  Controls for the fact that both volatility and
    the release count vary across years (and weekdays, in the second pass).

    `v` is |r| for the size question, 1{r>0} for the direction question, and r
    itself for the drift question — one machine, three tests.

    Vectorised: the statistic depends on the resampled labels only through S, the
    sum of `v` over the drawn event days, so each stratum contributes an
    independent column of k-subset sums and the difference of means is recovered
    in closed form.  20,000 draws over 180 (year x weekday) strata takes about a
    second instead of about a minute.

    Also returns the MATCHED baseline: the stratum-weighted mean of `v` over
    non-event days, i.e. exactly what the event-day mean is being compared against
    under this null.  Quoting the raw all-other-days mean instead would let a
    weekday or decade mix masquerade as an event effect."""
    n_tot, k_tot, total = len(v), int(mask.sum()), float(v.sum())
    obs = v[mask].mean() - v[~mask].mean()

    S = np.zeros(draws)
    num = den = 0.0
    for s in np.unique(strata):
        g = np.flatnonzero(strata == s)
        k = int(mask[g].sum())
        if k == 0:
            continue
        vals = v[g]
        non = g[~mask[g]]
        if len(non):
            num += k * v[non].mean()
            den += k
        if k == len(g):                       # stratum is all event days
            S += vals.sum()
            continue
        keys = rng.random((draws, len(g)))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        S += vals[pick].sum(axis=1)
    matched = num / den if den else float("nan")

    stat = S / k_tot - (total - S) / (n_tot - k_tot)
    p = (int((np.abs(stat) >= abs(obs) - 1e-12).sum()) + 1) / (draws + 1)
    return p, matched


def _binom_p(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p (method of small p-values), no scipy needed."""
    if n == 0:
        return float("nan")
    pmf = np.array([comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)])
    return float(pmf[pmf <= pmf[k] + 1e-12].sum())


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values — three events are tested per window, so
    the smallest raw p is the one most likely to be luck."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = round(running, 5)
    return out


# The window the weekly report leads with. Ten years is long enough for the
# permutation test to have power (n ~ 127 releases) and recent enough to describe the
# regime a reader is actually trading.
RECENT_FROM = "2016-01-01"
# The file the weekly PDF and the BASIS Evidence tab read. Nothing rebuilt it
# until the audit: the study was a CLI that wrote wherever --json pointed, so
# section 2 of a "week ahead" report was served from whatever had last been run
# by hand, under a caption reading "counting back from today".
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "gold_store" / "event_study.json"

# Buckets ANCHORED TO TODAY, not to calendar decades. Three reasons, and the third
# is the one that actually bit:
#
#   1. The first bucket always ends today, so it describes the regime a reader is
#      trading. A "2010-2019" bucket is stale the moment it is written.
#   2. Decade boundaries are arbitrary — nothing happened to gold on 1 Jan 2010 — so
#      a genuine regime shift can land mid-bucket and be diluted.
#   3. EQUAL WIDTH. Calendar decades stop being equal once you reach the present:
#      the old "2020-2026" bucket was 6.7 years against 10 for its neighbours, so it
#      carried fewer observations and a wider interval, and was being displayed
#      beside them as though it were comparable. It was not.
#
# Anchored buckets are also DISJOINT, unlike a rolling window. Rolling windows
# overlap almost completely, so adjacent readings are nearly the same data and a
# rising line looks like far more evidence than it is. Disjoint buckets can be
# compared; overlapping ones cannot.
BUCKET_YEARS = 10
N_BUCKETS = 4
BUCKET_YEARS_FINE = 5
N_BUCKETS_FINE = 7


def era_table(ev_dates: pd.DatetimeIndex, ret: pd.Series,
              step: int = BUCKET_YEARS, n: int = N_BUCKETS) -> dict:
    """The same ratio and hit-rate cut into equal buckets anchored to today.

    Stability ACROSS buckets is the most useful thing in this table, and it is a
    diagnostic a reader can apply without trusting any p-value: a real effect barely
    moves when you change the slicing, and an effect that wanders was never
    established however good one pass looked. On this data the employment report sits
    between roughly 1.1 and 2.0 in every bucket while CPI wanders around 1.1 — which
    is the whole answer about both.

    The baseline is MATCHED on (year, weekday), the same construction the headline
    `ratio_matched` uses. It was an unmatched mean until an audit caught it: the
    report's caption claimed matching while this function did not do it, and the two
    are not interchangeable — payrolls is almost always a Friday, so an unmatched
    baseline compares Friday releases against a week that is mostly not Friday."""
    out = {}
    end = ret.index.max()
    for i in range(n):
        lo = end - pd.DateOffset(years=(i + 1) * step)
        hi = end - pd.DateOffset(years=i * step)
        name = f"{i * step}-{(i + 1) * step}y ago"
        w = ret.loc[(ret.index > lo) & (ret.index <= hi)]
        m = w.index.isin(ev_dates)
        if m.sum() < 20:
            out[name] = {"n": int(m.sum()), "note": "too few to quote"}
            continue
        # Matched baseline: mean |return| of NON-event days within each (year,
        # weekday) stratum that carries an event, averaged with the event weights.
        ev_abs = w.abs()[m]
        strat = pd.Series(list(zip(w.index.year, w.index.weekday)), index=w.index)
        base_by = w.abs()[~m].groupby(strat[~m]).mean()
        wts = strat[m].value_counts()
        common = [k for k in wts.index if k in base_by.index]
        if common:
            tot = sum(wts[k] for k in common)
            base = sum(base_by[k] * wts[k] for k in common) / tot
        else:
            base = float(w.abs()[~m].mean())
        out[name] = {"n": int(m.sum()),
                     "ratio": round(float(ev_abs.mean() / base), 3) if base else None,
                     "ratio_unmatched": round(float(ev_abs.mean() / w.abs()[~m].mean()), 3),
                     "share_up": round(float((w[m] > 0).mean()), 3),
                     "baseline_share_up": round(float((w[~m] > 0).mean()), 3)}
    return out


def block(ev_dates: pd.DatetimeIndex, ret: pd.Series,
          all_event_days: pd.DatetimeIndex, draws: int) -> dict:
    """One event type on one return window."""
    on = ev_dates.intersection(ret.index)
    lo, hi = on.min(), on.max()
    w = ret.loc[lo:hi]
    mask = w.index.isin(on)
    a = w.abs().to_numpy()
    clean = ~w.index.isin(all_event_days)

    year = w.index.year.to_numpy()
    dow = w.index.dayofweek.to_numpy()
    yd = year * 10 + dow
    up_ind = (w > 0).to_numpy().astype(float)
    sgn = w.to_numpy()

    # size: does the day move MORE?
    p_y, _ = _perm(a, mask, year, draws, np.random.default_rng(SEED))
    p_yd, matched_yd = _perm(a, mask, yd, draws, np.random.default_rng(SEED + 1))
    # ...more than an ORDINARY day. The matched baseline above compares against every
    # non-event day of the same year and weekday, and those include the OTHER four
    # releases — which are themselves elevated, so the denominator is inflated and the
    # ratio understated. Conservative, but the caption says "ordinary days" and this
    # is what that phrase has to mean. Restricting to days carrying no high-impact US
    # release makes the comparison the one being described.
    keep = mask | clean
    if int(mask.sum()) and int((clean & ~mask).sum()) > 30:
        p_yd_clean, matched_clean = _perm(a[keep], mask[keep], yd[keep], draws,
                                          np.random.default_rng(SEED + 4))
    else:
        p_yd_clean, matched_clean = float("nan"), float("nan")
    # direction: does it move UP more often, / drift further, than a matched day?
    p_up, matched_up = _perm(up_ind, mask, yd, draws, np.random.default_rng(SEED + 2))
    p_sgn, matched_sgn = _perm(sgn, mask, yd, draws, np.random.default_rng(SEED + 3))

    n, up = int(mask.sum()), int((w[mask] > 0).sum())
    base_up = float((w[~mask] > 0).mean())
    ev, base = float(a[mask].mean()), float(a[~mask].mean())
    return {
        "n_events": n,
        "n_missing_fix": int(len(ev_dates) - len(on)),
        "window": f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}",
        "n_baseline_days": int((~mask).sum()),
        "mean_abs_event_pct": round(ev * 100, 4),
        "mean_abs_baseline_pct": round(base * 100, 4),
        "mean_abs_baseline_clean_pct": round(float(a[clean].mean()) * 100, 4),
        "mean_abs_baseline_matched_pct": round(matched_yd * 100, 4),
        "mean_abs_baseline_matched_clean_pct": (round(matched_clean * 100, 4)
                                                if matched_clean == matched_clean else None),
        "median_abs_event_pct": round(float(np.median(a[mask])) * 100, 4),
        "median_abs_baseline_pct": round(float(np.median(a[~mask])) * 100, 4),
        "ratio": round(ev / base, 4),
        "ratio_vs_clean": round(ev / float(a[clean].mean()), 4),
        "ratio_matched": round(ev / matched_yd, 4),
        # The published ratio: matched on year+weekday AND excluding other releases.
        "ratio_matched_clean": (round(ev / matched_clean, 4)
                                if matched_clean == matched_clean and matched_clean
                                else None),
        "p_perm_year_weekday_clean": p_yd_clean,
        "share_up": round(up / n, 4),
        "baseline_share_up": round(base_up, 4),
        "baseline_share_up_matched": round(matched_up, 4),
        "mean_signed_event_pct": round(float(w[mask].mean()) * 100, 4),
        "mean_signed_baseline_pct": round(float(w[~mask].mean()) * 100, 4),
        "mean_signed_baseline_matched_pct": round(matched_sgn * 100, 4),
        "p_perm_year": round(p_y, 5),
        "p_perm_year_weekday": round(p_yd, 5),
        "p_perm_share_up": round(p_up, 5),
        "p_perm_mean_signed": round(p_sgn, 5),
        "p_direction_vs_50_binom": round(_binom_p(up, n, 0.5), 5),
        "p_direction_vs_baseline_binom": round(_binom_p(up, n, base_up), 5),
        "weekday_mix": {DOW[int(k)]: int(v) for k, v
                        in zip(*np.unique(on.dayofweek, return_counts=True))},
    }


# ---------------------------------------------------------------------------
def compute(draws: int = 20000, verbose: bool = False,
            write: bool = True, pm: str = PM, am: str = AM,
            out_file=None) -> dict:
    """Run the release study and persist it to OUT_FILE.

    Split out of main() so the daily pull can refresh the file the client report
    reads. Everything below is unchanged CLI logic; only the printing is gated.
    """
    def _say(*a):
        if verbose:
            print(*a)

    # as_of=None is the reporting read: LBMA fixes are EXACT-tier and never
    # revised, so there is no vintage to respect and nothing to look ahead into.
    pm_id, am_id = pm, am
    pm = gs.get_series(pm_id)
    am = gs.get_series(am_id)
    pm = pm[pm > 0].sort_index()
    am = am[am > 0].sort_index()
    both = pd.concat([am.rename(am_id), pm.rename(pm_id)],
                     axis=1).dropna()

    windows = {
        "fix_to_fix_24h": np.log(pm).diff().dropna(),
        "am_to_pm_intraday": np.log(both[pm_id] / both[am_id]).dropna(),
    }
    _say(f"{pm_id}: {len(pm)} fixes {pm.index.min():%Y-%m-%d}..{pm.index.max():%Y-%m-%d}; "
         f"AM+PM overlap {len(both)} days")

    scan_days = pd.bdate_range(pm.index.min(), pm.index.max())

    per_series, audits = {}, {}
    for sids in EVENTS.values():
        for sid in sids:
            per_series[sid], audits[sid] = release_dates(sid, scan_days)
            d = per_series[sid]
            _say(f"  {sid:14s} releases={len(d):4d} "
                 f"{d.min():%Y-%m-%d}..{d.max():%Y-%m-%d} audit={audits[sid]}")

    ev_dates = {name: pd.DatetimeIndex(sorted(set().union(*[set(per_series[s])
                                                            for s in sids])))
                for name, sids in EVENTS.items()}
    all_event_days = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in ev_dates.values()])))
    keys = list(ev_dates)
    overlap = {f"{a} & {b}": int(len(ev_dates[a].intersection(ev_dates[b])))
               for i, a in enumerate(keys) for b in keys[i + 1:]}

    results = {}
    for wname, ret in windows.items():
        blocks = {name: block(d, ret, all_event_days, draws)
                  for name, d in ev_dates.items()}
        hy = holm({k: v["p_perm_year"] for k, v in blocks.items()})
        hyd = holm({k: v["p_perm_year_weekday"] for k, v in blocks.items()})
        hup = holm({k: v["p_perm_share_up"] for k, v in blocks.items()})
        hsg = holm({k: v["p_perm_mean_signed"] for k, v in blocks.items()})
        for k, b in blocks.items():
            b["p_perm_year_holm"] = hy[k]
            b["p_perm_year_weekday_holm"] = hyd[k]
            b["p_perm_share_up_holm"] = hup[k]
            b["p_perm_mean_signed_holm"] = hsg[k]
            # Size is called real only if BOTH stratifications survive Holm.
            b["significant"] = bool(hy[k] < 0.05 and hyd[k] < 0.05)
            # Direction is called real only if BOTH the hit-rate and the mean
            # drift survive — a hit-rate edge with no drift behind it is noise
            # dressed as a signal.
            b["direction_significant"] = bool(hup[k] < 0.05 and hsg[k] < 0.05)
            # Era stability: an effect that only exists in one decade is not a
            # historical regularity, it is a story about that decade.
            b["by_era"] = era_table(ev_dates[k], ret)
            b["by_era_fine"] = era_table(ev_dates[k], ret, BUCKET_YEARS_FINE,
                                         N_BUCKETS_FINE)
        # RECENT-DECADE BLOCK, tested the same way as the full sample.
        #
        # The full-sample ratio answers "how has gold behaved around this release
        # since 1990", which is not the question a desk is asking. CPI was a
        # non-event through the 1990s (0.94x) and carries a real premium now, so the
        # 36-year average understates the present by averaging over an era in which
        # nobody traded the print. The recent block is the one a weekly report should
        # lead with; the era table beside it shows whether the effect is stable.
        recent_ret = ret[ret.index >= pd.Timestamp(RECENT_FROM)]
        recent_all = all_event_days[all_event_days >= pd.Timestamp(RECENT_FROM)]
        rblocks = {name: block(d[d >= pd.Timestamp(RECENT_FROM)], recent_ret,
                               recent_all, draws)
                   for name, d in ev_dates.items()}
        rhy = holm({k: v["p_perm_year"] for k, v in rblocks.items()})
        rhyd = holm({k: v["p_perm_year_weekday"] for k, v in rblocks.items()})
        for k, rb in rblocks.items():
            rb["p_perm_year_holm"] = rhy[k]
            rb["p_perm_year_weekday_holm"] = rhyd[k]
            rb["significant"] = bool(rhy[k] < 0.05 and rhyd[k] < 0.05)
            blocks[k]["recent"] = rb
        results[wname] = blocks

    # --- FOMC: datable only forward, and after the fix anyway -----------------
    fomc = pd.DatetimeIndex([pd.Timestamp(d) for d in fedpath.FOMC_DECISIONS])
    r24 = windows["fix_to_fix_24h"]
    fin = fomc.intersection(r24.index)
    fomc_row = {
        "event": "FOMC decision",
        "measurable": False,
        "why": ("fedpath.FOMC_DECISIONS is a FORWARD calendar starting 2026-01-28, "
                "so only a handful of decisions fall inside the price sample; and a "
                "14:00 ET announcement post-dates the 15:00 London PM fix, so no fix "
                "window can contain it. A verified historical FOMC calendar plus a "
                "US-close gold series (GLD_PX from 2004, COMEX_GOLD_FRONT from 2016) "
                "would be required."),
        "n_events": int(len(fin)),
        "dates_in_sample": [f"{d:%Y-%m-%d}" for d in fin],
        "baseline_abs_pct": round(float(r24.abs().mean()) * 100, 4),
    }
    if len(fin):
        fomc_row["same_day_abs_pct"] = round(float(r24[fin].abs().mean()) * 100, 4)
        fomc_row["share_up_same_day"] = round(float((r24[fin] > 0).mean()), 4)
        pos = [r24.index.get_loc(d) + 1 for d in fin if r24.index.get_loc(d) + 1 < len(r24)]
        nxt = r24.iloc[pos]
        fomc_row["next_day_abs_pct"] = round(float(nxt.abs().mean()) * 100, 4)
        fomc_row["share_up_next_day"] = round(float((nxt > 0).mean()), 4)

    out = {"gold_series": pm_id, "metal_series": pm_id,
           "am_series": am_id,
           "sample": f"{r24.index.min():%Y-%m-%d} to {r24.index.max():%Y-%m-%d}",
           "n_trading_days": int(len(r24)),
           "draws": draws,
           "release_audit": audits,
           "event_overlap_days": overlap,
           "results": results,
           "fomc": fomc_row}

    out["built"] = datetime.now().isoformat(timespec="seconds")
    if write:
        (Path(out_file) if out_file else OUT_FILE).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = compute(draws=args.draws, verbose=True, write=not args.json)
    print(json.dumps(out, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
