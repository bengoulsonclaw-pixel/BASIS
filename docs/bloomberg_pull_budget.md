# Bloomberg Daily Pull Budget — audit & target spec

*Audited 2026-08-18, following Bloomberg's (Laura Parente) DAPI best-practice email.
Companion to the capacity playbook: pulls run once daily in the morning snapshot;
reviews trigger on **pattern changes**, not just volume — keep the shape stable.*

**STATUS 2026-08-18 (same day): the optimization package below is IMPLEMENTED** —
ATM triple-pull deduped (shared `atm=` frame through skew/term), COT price store
mirrors the deep store (zero pull), deepstore volume/yield tails read the snapshot
parquets, stircurve tail 12→5d with outage self-heal, on-demand OI chain pulls
removed (`get_oi_chain(live=True)` only from the weekly `--oi` job; there was NO
scheduled Monday task — the capture was always the manual page button), equities
yfinance failure now raises instead of silently re-arming Bloomberg, pullguard
ledger counts every leg (fixed legs analytic + runtime tally from instrumented
chain/ladder/store pull sites). Owncurve additionally reworked to Ben's smile spec:
bracketing-strike interpolation at 80/90/100/110/120% moneyness (was: single
nearest strike at 90/110 only), 5 futures walked (was 6), option-chain lists
cached 7 days. First live morning will verify the realized hit count.

**GUARD (added same day, Ben's ask): the src/bbg.py gateway.** Every xbbg request
now passes one chokepoint that (a) counts hits at the request site — the ledger is
metered, not estimated; (b) registers every (security, field) pair with its calling
leg and reports any pair pulled by TWO different legs in one session, in the
morning log, `data/pull_duplicates.json`, and a 🩺 Data health row; (c) is enforced
by `tests/test_bbg_gateway.py` (pre-push gated): direct xbbg imports outside the
gateway fail the suite. Same-site re-requests (serial fallbacks, forced refreshes)
are retries, not duplicates; deliberate cross-leg overlaps live in
`bbg.ACCEPTED_OVERLAPS` with reasons.

**OWNERSHIP RULE**: every security×field family has exactly ONE owning module —
prices / vol surface / put-call / OI chains = datafeed · deep history = deepstore ·
own-curve option marks = owncurve · STIR ladders = stircurve · strips/fixings =
stirpaths · precious metals = pm_bbg. Anything else that needs the data reads the
STORE. New features ask "who already owns this?" before adding any pull.

---

## 1. What we actually pull today (measured, not estimated)

**Morning snapshot fetch phase ≈ 7,000–9,000 security×field hits/day**
(drops to ≈ 4,600–6,000 once the own-skew backfill drip self-terminates).
**Weekly OI job (Mondays) ≈ 10,000–80,000 hits** — by far the largest single job.
The `pullguard` ledger logs only ~1,780/morning — it understates reality by 4–5×
because owncurve, stircurve, stirpaths, deepstore, cotdata and the skew drip are
outside its estimate.

| Leg | What | Hits/day |
|---|---|---|
| Prices (`get_history`) | 89 FICC generics, PX_SETTLE/PX_LAST, 400d | ~89 |
| Yields | 11 yield sources, PX_LAST, 400d | ~11 |
| Volume | 89 × FUT_AGGTE_VOL, 400d | ~89 |
| Implied vol (30d ATM) | 73 listed `_DF` + 15 FX V1M + Euribor composite | ~90 |
| Skew (90/110/100%) | 74 listed × 3 + 15 FX × 3 | ~267 |
| Term (1M/3M/6M/12M) | 74 listed × 4 + 15 FX × 4 | ~356 |
| Put/call OI+volume | 89 × 4 fields | ~356 |
| Live quote (bdp) | 89 × 3 fields | ~267 |
| COT price store tail | 47 × PX_SETTLE, 10d tail | ~47 |
| Deep store tail | 5 frames (prices/front2/contract/volume/yields) | ~351 |
| STIR curve ladder | 3 products, full strike ladders, 12d tail | ~400–700 |
| STIR strips + fixings | 102 contracts × 2 + 3 fixings | ~207 |
| **Own-curve marks** | 49 products: FUT_CHAIN + OPT_CHAIN ×6 futs + ATM/wing bdp | **~2,800–3,400** |
| Skew backfill drip | 2 products/morning, ~14 months (self-terminating) | ~2,000–3,000 → 0 |

Outside the snapshot: hourly `surface_topup` (re-runs the full ~713-hit vol
surface only when the settled row is incomplete), the "⚡ Live pull" STIR button
(tiny), and — the one genuinely unbounded path — the **on-demand OI chain** on
the Open Interest page (up to ~29 bds + ~1,900-security bdp per product, per
click, no cache).

## 2. Redundancy found (same data pulled more than once)

1. **The 30DAY 100%-moneyness ATM field is pulled THREE times every morning** —
   by the implied-vol leg, the skew leg (as its ATM anchor), and the term leg
   (as the 1M tenor). Same 74 securities, same field, same 400-day window,
   three separate bdh calls, three parquets. Same story for the FX V1M leg.
   ~178 wasted hits/day, re-wasted each time surface_topup fires.
2. **COT price store duplicates the deep store** — cotdata's 47 markets are a
   strict subset of deepstore's 89; both pull `'1'`-generic PX_SETTLE with 10y
   backfills and 10-day tails into separate parquets. ~47 hits/day + a whole
   duplicate backfill path.
3. **Yields pulled twice** — datafeed's 400-day leg and deepstore's tail hit the
   same 11 tickers/field. ~11/day.
4. **Volume pulled twice** — 'A' generics (datafeed) and '1' generics
   (deepstore) for FUT_AGGTE_VOL, which per deepstore's own docstring returns
   the same value on any generic. ~89/day.
5. **Vendor 90/110 wings** still pulled full-width even where
   `own_skew_history` already covers the product (35 wing-capable names).
   ~148/day reclaimable once the drip completes. (Term stays vendor — own term
   history isn't deep/uniform enough yet, per volbt.)
6. **Dead code**: `fedpath._bloomberg_strip` has no callers.
7. **Latent hazard**: all equity Bloomberg paths (quotes/history/constituents/
   `eqfunda` 30-field fundamentals — the July-2026 block) silently re-arm if
   yfinance ever fails to import, because `_use_yf()` degrades to Bloomberg.
8. **eqdisp page-load pulls** (`INDX_MWEIGHT` + full-membership mkt-cap bdp +
   membership IV bdh) only read their caches in *snapshot* mode — in bloomberg
   mode every Equity Dispersion render re-pulls, with no freshness guard.

## 3. Laura's four recommendations, applied to us

- **"Request only the securities and fields you need"** — our main violation.
  Fixes: dedupe the triple ATM pull; fold cotdata into deepstore; drop the
  duplicate volume/yield legs; trim `owncurve.fetch_marks` (it fetches 6
  futures' full option chains per product but `assemble_pillars` keeps only 5
  pillars with MIN_DTE=7 — fetch 5, skip chains that can't survive the gate).
- **"Reduce unnecessary refreshes"** — mostly already right (once-daily
  snapshot, settle-based). Remaining offenders: eqdisp bloomberg-mode renders,
  the uncapped on-demand OI page, and stircurve recomputing a 12-day tail daily
  when 3–5 days would catch settle revisions at a third of the cost.
- **"Batch requests"** — we already batch well (grouped bdh by field, batched
  bdp). No change needed.
- **"Use subscriptions for continuously updating data"** — not applicable: we
  are a settlement-data shop; one daily poll is the correct request type. The
  one live-ish path (live quote bdp) is a single batched call.

## 4. Target daily pull spec (proposed)

| Leg | Spec | Hits/day |
|---|---|---|
| Prices | 89 generics, settle field, 400d | 89 |
| Yields | 11 sources, PX_LAST, 400d — **single pull, shared with deepstore** | 11 |
| Volume | 89 × FUT_AGGTE_VOL — **one generic family only** | 89 |
| Vol surface (merged) | ONE field-deduped batch: ATM 30d (90) + wings (148, own-history products excluded once drip done) + term 3M/6M/12M (267; 1M ≡ the ATM pull) | ~360–505 |
| Put/call | 89 × 4 | 356 |
| Live quote | 89 × 3 bdp | 267 |
| Deep store tail | prices/front2/contract (~250); **absorbs COT store**; volume+yields legs dropped | ~250 |
| STIR curve | 3 products, **5-day tail** | ~150–250 |
| STIR strips + fixings | unchanged | ~207 |
| Own-curve marks | trimmed to 5 futs/product + chain-gating | ~2,300–2,800 |
| **Total** | | **≈ 4,100–4,800/day** (vs 7–9k now) |

Plus guards: cache + cap the on-demand OI page (reuse Monday's parquet, live
pull behind an explicit button with a per-day cap); make `_use_yf()` fail loud
instead of falling back to Bloomberg; make eqdisp read its caches in bloomberg
mode; fix `pullguard` to count owncurve/stircurve/stirpaths/deepstore so the
ledger matches Bloomberg's record if a review is ever re-litigated.

Weekly (Mondays): OI chain job unchanged in cadence but worth capping strikes
per expiry harder if reviews recur — it dwarfs everything else.
