# Gold Signal Engine — repo survey and gap list

**Status:** Milestones 1-7 complete; history deepened to 1990 — survey, gap list, schema, point-in-time
accessor and leak tests (M1); 57 series ingested point-in-time (M2); 28 features +
6 targets computed and stored (M3); walk-forward harness + four benchmarks (M4).
Stage 1 diagnostics (M5); Stage 2 elastic net (M6). Milestone 7 (bucket scores) not
started. **The three-year holdout has not been touched.**
**Written:** 2026-08-22 · **Updated:** 2026-08-22 after Ben's review.

---

## 1. What BASIS actually is

| Spec expects | BASIS reality |
|---|---|
| Database engine | **None.** No SQL anywhere — no sqlite, postgres, duckdb, or SQLAlchemy in the tree. |
| ORM / query layer | **None.** Storage is parquet + JSON files under `data/`. |
| Time-series store | `data/price_store/deep_*.parquet` (the deep panama-adjusted store, `src/deepstore.py`) and ~48 per-module parquets under `data/signals/`. |
| Closest thing to a table | `src/eqfunda.py` — an **append-only long-format parquet DB** keyed `as_of, ticker, field, value`, with a manifest and idempotent same-day replacement. |
| Ingestion jobs | `run_daily.py` (signal rebuild), `run_pull.py` (the self-healing Bloomberg pull behind the Pull button), per-report `*_scheduled_email.py` drivers. |
| Scheduler | Windows Task Scheduler, XML definitions committed in `scheduled_tasks/`. |
| Config / secrets | Flat files under `data/`: `fred_key.txt`, `eia_key.txt`, `email_recipients.json`, `automation.json`. Env-var override on each. |
| Test framework | pytest, golden-file locks under `tests/`, run via `run_tests.py`; a **pre-push git hook blocks a push on red** when the push contains code changes. |
| Dashboard | Streamlit, single `app.py` with per-page engines in `src/`. |
| Model/stat libs | **numpy + pandas only.** No scipy, no sklearn, no statsmodels, no LightGBM. |

### Consequences for the spec

**§3 (storage) cannot be implemented as written.** The spec gives DDL for a
Postgres `observations` table. Introducing one would violate the spec's own §0.2
("do not introduce a second database"), because BASIS has no first database to
extend. The defensible reading is to keep the **schema** and drop the **engine**:
implement `observations` as an append-only long parquet carrying exactly the
specified columns (`series_id, reference_date, published_at, value, revision,
source, is_synthetic`) plus a `series_meta.json`, following the `eqfunda` precedent
that already works in this repo. `get_series(series_id, as_of)` remains the single
accessor and the lint/test rule in §3 still applies verbatim.

**§6 stages 2 and 5 need new dependencies.** Elastic net needs sklearn; LightGBM
needs LightGBM. Neither is in `requirements.txt` and this repo has deliberately
stayed numpy-only. Ridge is closed-form in numpy (already done); elastic net is not
practical to hand-roll well. **This needs a decision before Milestone 6.**

**§9's nightly job conflicts with a standing instruction.** The morning pull was
explicitly made press-a-button and its every-15-minute scheduled task was deleted
the same day it was created. A nightly auto-run needs sign-off rather than
assumption. The model compute itself is already wired into `run_daily.py`, which
runs when the pull runs.

---

## 2. Gap list

Legend: **✅ present** · **⚠️ partial** · **❌ missing**

### 2.1 Price and market (daily)

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| LBMA gold PM fix, USD/oz | ✅ | daily | **1968-04-01** | `prices.lbma.org.uk/json/gold_pm.json`, free, no key. AM fix also held. |
| XAU/USD spot close | ❌ | — | — | Yahoo `XAUUSD=X` 404s. LBMA fix is the benchmark substitute; Bloomberg `XAU Curncy` exists if licensed. |
| COMEX front settlement | ✅ | daily | 2016-08-08 | Deep store `GCA Comdty`, panama-adjusted + raw. |
| COMEX volume | ✅ | daily | 2016-08-08 | `deep_volume.parquet`. |
| COMEX open interest | ⚠️ | weekly | 2006-06-13 | Not in the deep store. COT carries OI weekly; `datafeed` has OI-chain probes but no archive. |
| Gold in EUR | ✅ | daily | 1968 | LBMA native leg. |
| Gold in GBP | ✅ | daily | 1968 | LBMA native leg. |
| Gold in JPY / INR | ✅ | daily | 2010 | Derived: LBMA × Yahoo `JPY=X` / `INR=X`. |
| Gold in CNY | ✅ | daily | 2010 | Derived via Yahoo `CNY=X`. |
| Silver spot | ✅ | daily | **1968** | LBMA silver JSON (verified free). Also `SIA Comdty` from 2016. |
| Gold ETF price | ✅ | daily | 2004-11-18 | Yahoo `GLD`, `IAU`. |
| Gold ETF tonnage | ✅ | daily | **2004-11-18** | SPDR historical archive (xlsx endpoint), free, no key, full history each pull. |

### 2.2 Rates and monetary (daily)

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| US 10y TIPS yield | ✅ | daily | **2003-01-02** | FRED `DFII10`. **Note: not 1997.** The spec assumes TIPS launch; FRED's 10y constant-maturity real series begins 2003, so the synthetic splice in §2.7 must cover 1990–2002, not 1990–1996. |
| US 5y TIPS yield | ✅ | daily | 2003-01-02 | FRED `DFII5`. |
| US 10y nominal | ✅ | daily | 1962 | FRED `DGS10`. |
| 10y breakeven | ✅ | daily | 2003 | FRED `T10YIE`. |
| US 2y nominal | ✅ | daily | 1976 | FRED `DGS2`. |
| Fed funds futures, next 4 FOMC | ⚠️ | — | — | `src/fedpath.py` backs the implied path out of the **live** SR3 strip and **keeps no history**. Deep store has `FFA` (2016) and `SFRA` (2018) front contracts only — not the strip. Derived proxy `DGS2 − DFF` runs from 1976 and is what the current build uses. **A true cut-probability history needs the full SR3 strip archived going forward, or a Bloomberg backfill.** |
| DXY | ✅ | daily | 1971 | Yahoo `DX-Y.NYB`, same-day. FRED `DTWEXBGS` also held but publishes ~1 week late. |
| USD/CNY, USD/JPY, EUR/USD | ✅ | daily | 2010 | Yahoo. FRED `DEXCHUS`/`DEXJPUS` from 1971 with a ~1wk lag. |

### 2.3 Flows and positioning

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| COT managed money long/short/net | ✅ | weekly | **2006-06-13** | `data/signals/cot_history.parquet`, CFTC disaggregated, contract 088691. |
| COT producer/merchant net | ⚠️ | weekly | 2006 | `src/cotdata.py` pulls only the managed-money fields. The CFTC dataset carries producer/merchant; it is a field-list change, not a new source. |
| COT open interest | ⚠️ | weekly | 2006 | Same — available in the dataset, not currently extracted. |
| Reference date vs publication date | ❌ | — | — | Only the Tuesday reference date is stored. The Friday publication timestamp the spec requires is **not** recorded and must be added. |
| SPDR GLD tonnage | ✅ | daily | 2004-11-18 | As above. |
| Total known global ETF tonnage | ⚠️ | monthly | ~2023 | Bloomberg `ETFGTOTL Index` via `src/pm_bbg.py`, resampled monthly. No free daily/weekly equivalent found. |
| Gold 25d risk reversal + IV | ⚠️ | daily | **2025-08-29** | `data/signals/skew_history.parquet` and `volatility_history.parquet`, 252 rows. One year only — usable as context, useless for a 20-year fit. |

### 2.4 Physical demand

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| SGE benchmark → USD/oz, and premium vs London | ✅ | daily | **2016-12-19** | `sge.com.cn/graph/Dailyhq` Au99.99, free, no key. Premium computed here rather than taken from the Bloomberg CIX, so it now has history and no licence dependency. |
| SGE withdrawals | ⚠️ | monthly | — | Not held, and **no JSON endpoint** — `/graph/Weight` and `/graph/DeliveryWeight` both 302. The figures sit in a monthly Chinese PDF, one file per month (`/static/upload/file/*.pdf`). Obtainable, but it is a parser project, not a fetch. The OPEC MOMR pipeline is the precedent. |
| India gold import volumes | ✅ | monthly | **~2000** | **Solved 2026-08-22.** UN Comtrade public *preview* endpoint, no key: `comtradeapi.un.org/public/v1/preview/C/M/HS?reporterCode=699&cmdCode=7108&flowCode=M&partnerCode=0`. Verified Jan-2024 = **71.4t**, matching the published figure. Current to 2026-03 (~5 month lag). Caveat: **one period per request**, so ~240 calls to backfill 20 years, one per month thereafter. |
| India import duty step series | ⚠️ | — | — | Not a feed by nature — roughly ten policy changes in twenty years, announced in budgets. Hand-maintained JSON, following the `data/pm_pgm.json` seed pattern. |
| WGC quarterly demand (jewellery, bar/coin, ETF, CB, tech, supply, recycling) | ⚠️ | quarterly | — | **Gated; account created 2026-08-23** (basisreports@gmail.com, Google SSO). The files are real .xlsx but return **403** unsigned-in. `src/wgc_fetch.py` handles the signed-in fetch via the repo's real-Chrome persistent-profile pattern. **ENTITLED — confirmed 2026-08-22**, all four datasets download on the free account. **Licence read 2026-08-22** — see the licence section below. Internal analysis is fine; client-facing gated pending XP compliance. |
| Central bank monthly reserve changes, **world aggregate** | ✅ | monthly | **2010-01** | IMF SDMX `IMF.STA,IL / GX010`. ~6wk lag. |
| Central bank reserve changes, **by country** | ✅ | monthly | **2010-01** | **Solved 2026-08-22.** Same IMF endpoint — the first key dimension is COUNTRY, and `GX010` was simply the world-aggregate code. Leaving it blank (`.RGV_REVS..M`) returns **178 countries in one call**. Codes are **ISO3, not ISO2** (`POL`, not `PL` — ISO2 returns zero series silently). Validated against known holdings: US 8,133.5t, Germany 3,349.5t, China 2,346.4t, Poland 632.4t (**+117t/12m**), Kazakhstan +61t, Brazil +42.8t — exactly the leaders the framework names. **Gotcha: do not sum all reporters** — the response includes aggregate pseudo-countries (`GX010` world, `BIS`), so a naive total double-counts and prints ~1,600t/yr against WGC's 863t. |

### 2.5 Risk and macro

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| VIX | ✅ | daily | 1990-01-02 | FRED `VIXCLS`. |
| US IG credit spread | ⚠️ | daily | **2023-08-22** | FRED `BAMLC0A0CM` — **licence-limited to a rolling ~3 years, silently.** Asking for a 1990 start returns 2023 with no error. |
| US HY credit spread | ⚠️ | daily | **2023-08-22** | FRED `BAMLH0A0HYM2`, same rolling-3y licence limit. |
| Long-history credit substitute | ✅ | daily | **1990** (series to 1986) | FRED `BAA10Y` / `AAA10Y` (Moody's over the 10y). This is what the current build uses. |
| CPI y/y, core CPI | ✅ | monthly | 1947 | FRED `CPIAUCSL`, `CPILFESL`. |
| PCE, core PCE | ✅ | monthly | 1959 | FRED `PCEPI`, `PCEPILFE`. |
| Nonfarm payrolls, unemployment | ✅ | monthly | 1939 / 1948 | FRED `PAYEMS`, `UNRATE`. |
| M2 | ✅ | monthly | 1959 | FRED `M2SL`. |
| Federal debt outstanding | ✅ | quarterly | 1966 | FRED `GFDEBTN`. |
| Deficit as % GDP | ✅ | annual | 1929 | FRED `FYFSGDA188S`. Annual only — thin for a 60d model. |
| S&P 500 close | ⚠️ | daily | 2016-08-22 | FRED `SP500` is **also licence-limited to 10 years**. Substitutes: deep store `ESA Index` (2016) or Yahoo `^GSPC` (1927). |
| Geopolitical risk index | ❌ | — | — | Not in BASIS. **Skipping per spec §2.5.** |

### 2.6 Supply and cost

| Series | | Freq | History from | Source / note |
|---|---|---|---|---|
| Aggregate producer AISC | ❌ | quarterly | — | **No free feed exists.** It is a derived figure from eight to ten producers' quarterly results. Spec §2.6 already permits manual entry; `data/pm_pgm.json` is the precedent. |
| Global mine production | ⚠️ | **annual** | 1900s | Free at **annual** frequency: USGS Mineral Commodity Summaries PDF downloads cleanly (no key, verified). Quarterly only from WGC, which is gated. Annual is arguably sufficient — mine supply moves a few percent a year and the spec itself files it under long-horizon valuation. |

### 2.7 History requirement — the binding constraints

Spec asks for a 20-year minimum, 1990 target.

| Constraint | Reaches back to | Effect |
|---|---|---|
| **Deep store is capped at 10 years** (`deepstore.STORE_YEARS = 10`) | 2016-08-08 | Every COMEX-derived feature — carry, EFP, futures volume — is limited to 10 years. This is a config constant, but extending it means a large Bloomberg backfill against the daily-capacity limit. |
| LBMA fix (the spec's designated target series) | **1968** | Targets, valuation and FX-breadth features can meet the 1990 goal. |
| COT | 2006 | 20 years — meets the minimum. |
| SPDR tonnage | 2004 | 22 years — meets the minimum. |
| SGE premium | 2016-12 | 10 years — below the minimum, no deeper free source found. |
| TIPS real yield | 2003 | Needs the synthetic splice for 1990–2002. |
| Gold options IV / RR | 2025-08 | 1 year — far below. |

**The important consequence:** because the LBMA fix goes back to 1968, targets can
be built to the spec's history requirement even though the futures store cannot.
The current build uses the panama-adjusted COMEX contract as the target and is
therefore 10 years deep; switching the target to the LBMA fix per spec §5 both
complies and roughly triples the sample. That is a change I'd recommend and have
not made.

---

## 3. Point-in-time: what's achievable

The spec is right that this is the biggest failure mode, and BASIS currently has
**no publication timestamps anywhere**. What exists to build on:

* **ALFRED vintages already work.** `macrodata.fred(sid, vintage=...)` returns a
  series as it stood on a given date, and it is already used by the Macro Rate
  Radar's vintage backtest. Verified on payrolls: May-2024 read **158,543k** on
  10 Jun 2024 and reads **157,608k** today — a 935k revision that a naive backtest
  would have traded on with hindsight. Every FRED series in §2.2/§2.5 can be made
  genuinely point-in-time at no extra cost.
* **Market data needs no revision handling** — prices, LBMA fixes, SGE, GLD tonnage
  and Yahoo closes are final when printed. `published_at = reference_date + lag`
  is exact for these, not an approximation.
* **COT needs a real publication timestamp**: Tuesday reference, Friday 15:30 ET
  release. Currently only the reference date is stored, so any backtest reading COT
  before Friday is trading on unpublished data. This is a genuine live leak in the
  present build and the highest-priority fix.
* **IMF central bank data** carries a ~6 week lag and is revised. Approximate.

Series that would carry the spec's "backtests are optimistic" flag: COT (until the
publication date is added), IMF central banks, WGC quarterly, India imports.

---

## 4. What has already been built, and how it maps

Two modules were written before this spec arrived, in response to the original
brief. They are working and tested, and they overlap parts of Milestones 2–5.

* **`src/golddata.py`** — eight live sources into one aligned daily frame
  (`data/signals/gold_features.parquet`), each with provenance and staleness
  recorded. Covers most of §2.1–§2.5. No mocks: a dead source degrades to its last
  good cache or drops out, never to synthetic data.
* **`src/goldmodel.py`** — Stage 1 diagnostics, a ridge explanatory fit,
  sensitivities, a scenario engine, a walk-forward fair-value gap, a zero-parameter
  bucket composite (§6 stage 3's equal-weight option), and walk-forward scoring
  with the always-long benchmark.
* **`tests/test_goldmodel.py`** — 10 tests, all passing, including a no-lookahead
  lock on the feature matrix and on the fair-value gap.
* Wired into `run_daily.py` so it recomputes with the pull.

**Where it already agrees with the spec:** no raw levels; changes and z-scores
only; expanding-window walk-forward with the training slice closed H days before
prediction; overlap-corrected significance; always-long as the benchmark; no
random splits.

**Where it differs and would need changing:**

| Spec | Current build |
|---|---|
| Target = LBMA PM fix, 3 horizons (5d / 60d / 250d) | Panama-adjusted COMEX, single 21-business-day horizon |
| Volatility-scaled targets → probability | Raw log return, no probability output |
| Point-in-time `observations` table + sole accessor | Direct parquet reads, no `published_at` |
| Elastic net (Stage 2) | Closed-form ridge |
| Regime layer (Stage 4) | Diagnosed but not modelled — rolling correlations are reported, coefficients are static |
| Forecast log + self-scoring | Not built |
| 20-year minimum history | 10 years |

**Findings from it that bear on the spec's calibration targets (§7):**

* The macro driver block explains **R² = 0.31** of gold's 21-day move; adding ETF
  and positioning flows takes it to 0.47, but those are co-movers — including them
  drags the 10y real-yield coefficient from −0.51% per 10bp (t = −2.8) to −0.12%
  (t = −0.7). The two blocks must be fitted separately.
* Fitted sensitivities, all theory-correct in sign: DXY +1% → **−0.72%** (t = −3.4);
  10y real yield +10bp → **−0.51%** (t = −2.8); 10y breakeven +10bp → +0.41%.
* **Direction is not predictable at this horizon on this sample.** Every driver's
  forward IC sits inside ±0.17 with |t| < 2; the equal-weight composite scores
  IC +0.08 (t = 0.9); a walk-forward kitchen-sink ridge scores **negative** IC and
  a hit rate *below* always-long. This is consistent with the spec's own guidance
  that 53–55% is a strong 5-day result — and it is a warning that the 60-day
  56–60% target may not be reachable without the longer history and the regime
  layer.
* The **cumulative unexplained move** (gold beyond what macro drivers account for,
  12m rolling) tracks the known history closely: −5.7% in 2020, **+25.8% in 2022**
  — the year the real-yield link broke and central banks first bought 1,000+ tonnes
  — then +20.4% in 2024 and +26.7% in 2025. This independently confirms the regime
  break the spec describes in §6 Stage 4.
* Two of the framework's sign priors come out **backwards** in sample and have been
  left unrefitted, with the disagreement documented in code: a wide Shanghai premium
  leads *weaker* gold (IC −0.11), and a crowded COT (>80) was followed by +0.93%
  versus +0.08% after a washed-out reading (<20).

---

## 5. Decisions taken (Ben, 2026-08-22)

| Question | Decision |
|---|---|
| **Target series** | **Switch to the LBMA PM fix.** Roughly triples usable history (1968 vs 2016) and meets the spec's 20-year minimum. |
| **Dependencies** | **Add sklearn and LightGBM.** Elastic net at Stage 2, LightGBM at Stage 5 under the spec's hard constraints (depth ≤ 4, ≥ 200 samples/leaf, early stopping). |
| **Nightly job** | **Compute on pull.** The model recomputes inside `run_daily.py`; no new scheduled task, so the press-a-button rule stands. Consequence: the forecast log only gains a row on days the pull runs, so §9's self-scoring will have gaps and must not treat them as missing-at-random. |
| **Storage** | Long-parquet implementation of the §3 schema — no database exists to extend. |

Still open, and not blocking Milestone 2:

* **Deep store depth.** `STORE_YEARS = 10` stands for now. With the target moved to
  LBMA this only limits the COMEX-derived features (carry, EFP, futures volume).
* **Accepted gaps — revisited 2026-08-22.** The six were not one problem, and two of
  them were never really gaps at all. Sorted by what is actually wrong with each:

  1. *I had not looked* — the spec said not to hunt, so I did not.
     **Per-country central bank reserves** and **India import volumes** are both free,
     keyless and now verified. Take them.
  2. *Licensed* — **WGC quarterly demand** is a hard 403; the data is real and
     machine-readable but needs a Goldhub account and licence acceptance, which also
     bears on client redistribution. **Quarterly mine production** is inside the same
     gate; the annual USGS figure is free.
  3. *Published, but not as data* — **SGE withdrawals** exist only as a monthly
     Chinese PDF. Obtainable, but a parser build.
  4. *Not a feed by nature* — **India import duty** (≈10 events in 20 years) and
     **AISC** (a roll-up of producer results). Hand-maintained, as the spec allows.

  **Recommendation.** Take the two free ones now, in Milestone 2. Hand-seed duty and
  AISC — an hour each, and they only need touching a few times a year. Defer SGE
  withdrawals and WGC. The reason to defer is not difficulty: it is that the
  diagnostics already show even *daily, clean* drivers carry no forward predictive
  power at a weeks horizon, so a quarterly series arriving on a six-week lag will not
  move the 5d or 60d forecast. These belong to the 250-day bucket and to narrative
  attribution. Building a Chinese-PDF parser before the 250d model exists would be
  spending the effort in the wrong order.

---

## 6. Milestone 1 — delivered

* **`src/goldstore.py`** — the point-in-time store. The spec's DDL implemented as an
  append-only long parquet (`data/gold_store/observations.parquet`) with the exact
  specified columns and primary key `(series_id, reference_date, revision)`, plus
  `series_meta.json`. `get_series(series_id, as_of)` is the sole read path.
* **True revision history, one call per series.** ALFRED's `output_type=3` returns
  the full vintage matrix — every reference date against every vintage that changed
  it — so FRED series land with real publication dates and real revision numbers
  rather than an assumed lag. Verified on payrolls: May-2024 reads 158,543 →
  158,432 → 157,828 depending on the as-of date.
* **Three honesty tiers** recorded per series in `series_meta`: EXACT (market data,
  final when printed), VINTAGE (ALFRED, genuinely point-in-time), LAGGED
  (`reference_date + typical_lag`, flagged `published_at_approximated`, and any
  backtest resting on these must be described as optimistic).
* **`tests/test_goldstore.py`** — 13 tests. Full suite now **224 passing**. They
  cover the inclusive `published_at` boundary, the latest-*qualifying*-vintage rule
  (the subtle leak where `max(published_at)` is taken before filtering rather than
  after), revision numbering, idempotent re-writes, the synthetic flag, staleness
  flags, the ALFRED matrix parser, and **spec §3's lint rule** — which was confirmed
  to fail when a deliberate violation was introduced, then restored.

### Release cadence vs forecast horizon — built in, not written down

Ben's point, and it is now enforced by code rather than remembered: **a series
cannot drive a forecast over a window shorter than its own release cycle.**

`goldstore.horizon_role()` derives a verdict per horizon from the two facts every
series is registered with — `native_freq` and `typical_lag_days` — and returns
**drives / marginal / static**. Two tests, because cadence and lag fail differently:

* **Refresh** — `horizon / cadence`. Below ~0.5 the series is a constant across the
  window; above ~2 it refreshes often enough to carry signal.
* **Lag** — if `typical_lag > horizon`, the figure arrives after the window it would
  have informed has closed, and the verdict is downgraded a step.

Nothing is hand-typed per series, so this cannot drift out of date the way a comment
would. `horizon_matrix()` prints it; **`horizon_inputs(horizon)` is what the model
layer builds its feature set from**, which turns the rule into a guard — a quarterly
feed cannot quietly end up in the 5-day fit.

| Series | Bucket | Freq | Lag (d) | 5d | 60d | 250d |
|---|---|---|---|---|---|---|
| 10y TIPS yield | monetary | daily | 1 | drives | drives | drives |
| Dollar (DXY) | monetary | daily | 0 | drives | drives | drives |
| LBMA PM fix | valuation | daily | 0 | drives | drives | drives |
| SPDR GLD tonnage | flows | daily | 1 | drives | drives | drives |
| Shanghai premium | physical | daily | 0 | drives | drives | drives |
| COT managed money | flows | weekly | 3 | *marginal* | drives | drives |
| Central bank reserves | physical | monthly | 45 | **static** | drives | drives |
| SGE withdrawals | physical | monthly | 20 | **static** | drives | drives |
| India imports | physical | monthly | 150 | **static** | *marginal* ¹ | drives |
| WGC quarterly demand | physical | quarterly | 45 | **static** | *marginal* | drives |
| Aggregate AISC | valuation | quarterly | 60 | **static** | *marginal* | drives |
| Mine production | valuation | annual | 180 | **static** | **static** | *marginal* |
| India import duty | physical | annual | 0 | **static** | **static** | *marginal* |

¹ India's imports refresh twice inside a 60-day window — cadence alone says "drives"
— but they arrive ~150 days late, so the lag test demotes them. Central bank
reserves, monthly on a 45-day lag, pass the same test and survive.

**This is the deferral argument, made checkable.** Every feed still outstanding —
SGE withdrawals, WGC quarterly, AISC, mine production — is `static` at 5 days and no
better than `marginal` at 60. They cannot move the two horizons that carry the
tradeable signal. They belong to the 250-day bucket and to narrative attribution,
which is why building a Chinese-PDF parser before the 250-day model exists would be
effort spent in the wrong order.

### Goldhub — entitlement confirmed, and what it actually bought us

Probe run 2026-08-22 on the free basisreports@gmail.com account: **all four datasets
downloadable.** Two of the four pages serve the *same* workbook (gold-demand-by-country
and gold-supply-and-demand-statistics, identical md5), so it is three distinct
datasets, and the fetcher now dedupes on content hash.

| Dataset | What it contains | History | Verdict |
|---|---|---|---|
| **Gold ETF flows** | 145 funds worldwide — tonnage, ounces, AUM per fund; monthly flows in US$ split North America / Europe / Asia / Other | monthly | **Closes a gap I had marked Bloomberg-only.** Total known global ETF holdings (§2.3) no longer needs `ETFGTOTL Index`, and the regional split is more than Bloomberg was giving us. |
| **Central bank holdings** | per-country official gold, monthly and annual sheets | **2002 → 2026**, 294 month-columns | **Eight years deeper than the IMF feed** (2010). Wide format — years across columns — so it needs unpivoting, and the country column is interleaved with a comments column. |
| **Gold Demand Trends** | Gold Balance, Jewellery, Bar and Coin, Consumer per Capita, Exec Summary | quarterly | The §2.4 demand breakdown, in full. |

**What this does not change.** All three are monthly or quarterly, so under the cadence
rule they remain `static` at 5 days and no better than `marginal` at 60. They
substantially upgrade the **250-day bucket** and the narrative attribution; they do not
touch the two horizons carrying the tradeable signal. The deferral argument for SGE
withdrawals stands unchanged.

### WGC licence — what the documents actually say

Read from the workbooks' own Disclaimer sheets and gold.org's site terms, 2026-08-22.
This is a record of the wording, not legal advice; the call is XP compliance's.

**The general rule** (identical Disclaimer sheet in the GDT and central-bank
workbooks; the ETF one differs only by omitting the LBMA clause):

> "Reproduction or redistribution of any of this information is expressly prohibited
> without the prior written consent of World Gold Council ... **except as
> specifically provided below**."

**The carve-out that matters for client commentary:**

> "The use of the statistics in this information is **permitted for the purposes of
> review and commentary (including media commentary) in line with fair industry
> practice**, subject to the following two pre-conditions: (i) only **limited
> extracts** of data or analysis be used; and (ii) any and all use of these
> statistics is accompanied by a **citation to World Gold Council** and, where
> appropriate, to Metals Focus."

**The hard prohibition** — narrower, absolute, and easy to breach by accident:

> "LBMA Gold Price information provided by the World Gold Council may be used by you
> **internally** ... but **may not be used for any other purpose**. LBMA Gold Price
> information provided by the World Gold Council **may not be disclosed by you to
> anyone else**."

This restricts the WGC-supplied *copy* of the price, not the LBMA fix itself, which
`golddata.lbma()` pulls straight from prices.lbma.org.uk under its own terms. So the
rule is simply *never let a price column out of a WGC workbook*, and it costs nothing
— we hold the same number from the primary source. `strip_forbidden_columns()`
enforces it at ingest, with a test, because the ETF workbook puts "Gold Price (rhs)"
directly beside the flow columns we want.

**The unresolved tension, for a human.** gold.org's site terms say the site is for
*"personal and educational purposes only ... personal, non-commercial use"* and
forbid reproduction *"without the prior written authorisation of WGC"*. The workbook
disclaimer expressly permits *"review and commentary ... in line with fair industry
practice"*. The workbook disclaimer is the more specific instrument and travels with
the data, but a broker distributing client research is not obviously "personal,
non-commercial". `CLIENT_FACING_APPROVED = False` until that is settled.

**If it is approved**, two conditions bind: limited extracts only (derived
aggregates, not table dumps — which is what the model does anyway), and every
published figure carries `WGC_CITATION` = "Source: World Gold Council; Metals Focus".
Note the workbooks also state the data is *"for educational purposes only"* and
*"nothing contained herein is intended to constitute a recommendation, investment
advice"* — consistent with the existing client-commentary-not-advice rule.

### Goldhub access — operational notes

`src/wgc_fetch.py` (2026-08-23). Runs on **`.venv\Scripts\python.exe`**, not the
global interpreter — Playwright is only installed in the venv.

* `--login` opens real Chrome at Goldhub, waits for a human sign-in, and persists the
  session to `data/.wgc_chrome_profile`. Interactive once; unattended thereafter.
* `--probe` reports which datasets actually download on this account. Entitlement is
  judged on the payload's **magic bytes** (`PK` = a real xlsx), never on the status
  code — Goldhub serves its refusal page with HTTP 200.
* Download links carry an opaque per-edition file id (`/download/file/20975/…`) that
  changes each quarter, so links are discovered by scanning the dataset page rather
  than hardcoded. A hardcoded id would serve stale data silently for three months.

**Two things kept out of git** (`.gitignore` updated): the Chrome profile, because it
holds a live authenticated Google session for the same mailbox Morning Coffee sends
from and this repo pushes to GitHub and syncs to the VPS; and `data/wgc_inbox/`,
because it holds licensed data.

---

## 7. Milestone 2 — ingestion, delivered

`src/goldingest.py`. **55 series in the store**, each with an honest `published_at`.
Wired into `run_daily.py`, so market data and COT refresh with the pull.

### The COT leak is closed

`cot_history.parquet` stored the Tuesday reference date and nothing else, so any
backtest reading it before Friday was trading on a report the market had not seen —
three days of look-ahead on a positioning signal, every week, across the whole
sample. The CFTC API offers no publication field (194 columns, none of them a release
date), so `cot_published_at()` reconstructs it: Friday 15:30 ET, slipped one business
day per federal holiday in the reference week. Verified against eight known releases,
including July 4 2025 where the holiday lands *on* the Friday — the case where
counting the slip and separately skipping the holiday double-counts and lands a day
late. Now demonstrably true against the store:

| as of | latest COT reference visible |
|---|---|
| Tue 18 Aug | 11 Aug |
| Thu 20 Aug | 11 Aug |
| Fri 21 Aug 12:00 | 11 Aug |
| **Fri 21 Aug 16:00** | **18 Aug** |

The 15:30 stamp is load-bearing: at midnight a Friday-morning refit would read Friday
afternoon's report.

### Only revisable series go through ALFRED

The first ingest run asked ALFRED for the full vintage matrix of every FRED series
and the daily ones all failed — 400, 504, 500. Not a transient fault: a daily series
from 1990 is a grid of ~9,000 reference dates against ~9,000 vintages, and it is the
wrong question anyway, because TIPS yields, the 2y, the VIX and Baa are **never
restated**. Those are fetched normally with a one-business-day lag, which is exact
for them. Monthly and quarterly macro genuinely is revised and keeps the ALFRED path.

Payrolls proves the tier is real — 6,421 rows across 439 reference months:

| as of | May-2024 payrolls |
|---|---|
| 2024-06-06 | *not yet published* |
| 2024-06-07 | 158,543 |
| 2024-08-01 | 158,432 |
| 2025-03-01 | 157,828 |
| 2026-08-01 | 157,608 |

A model fitted on today's vintage would have been handed 935k of information that did
not exist.

### Horizon gating, live

26 series admissible at 5 days, 57 at 60 days, 58 at 250 days — the cadence rule
doing its job on the real registry rather than on an example.

### Notes from the run

* **Russia is flagged stale automatically** (`CB_GOLD_RUS`, last report 2025-11) — the
  staleness rule catching a real reporting halt, not a synthetic case.
* Per-country central bank reserves ingested for 18 major holders plus the world
  aggregate. `IMF_AGGREGATES` records the pseudo-countries that must be excluded from
  any sum.
* COT open interest was already in `cot_history.parquet` — the earlier gap list
  marked it ⚠️ partial, and it is in fact present and now ingested as `COT_OI`.

---

## 8. Milestone 3 — features and targets, delivered

`src/goldfeatures.py`. **28 features x 5,907 dates (2004 → 2026)**, 143,132 rows in
the long `features` table keyed `(feature_id, date)`, plus 6 targets. Wired into
`run_daily.py`. 13 new tests, every formula checked against a value worked out by
hand in the test rather than snapshotted from the implementation.

### Point-in-time filling, demonstrated

`goldstore.as_known_series` replays publications in order rather than querying per
date (O(n log n) instead of O(days x rows)), maintaining `{reference_date: value}`
and quoting the value at the latest reference date seen. A revision to an *older*
reference date therefore does not move the current level; a revision to the current
one does.

The result, on real payrolls:

| date | PAYROLLS in the panel |
|---|---|
| 2024-06-05 | 158,286 (April's print) |
| 2024-06-06 | 158,286 |
| **2024-06-07** | **158,543** ← May released |

It steps on the release date, not on the month end. Filling from `reference_date`
would have put May's figure on a 31 May row and handed the model a week of future,
every month.

### Two bugs the build surfaced

* **The gold/silver ratio was capped at 2016.** It used the deep store's COMEX
  silver contract, so a *five-year* z-score had under two usable years. LBMA silver
  is free back to 2005 in the same feed we already call — n went from 2,306 to
  **5,330**, and the feature now falls back to COMEX only where LBMA is absent.
* **The Shanghai premium was referencing the PM fix.** SGE's day session closes
  ~08:30 London, the AM fix is 10:30, the PM fix 15:00 — so against PM the series
  carried a whole London session of drift. Latest reading moved from -$68 to -$42.
  `golddata` already had this right; the feature layer had drifted from it.

### The lint rule earned its keep, and was too broad

`test_no_module_reads_the_observation_table_directly` failed on `goldfeatures.py`
— correctly firing, but on a false positive: the needle `gold_store` matched the
*directory*, which the feature layer legitimately writes its own tables into.
Narrowed to `observations.parquet` / `OBS_FILE` / `_read_obs`. Guarding the folder
name would have pushed feature outputs somewhere arbitrary to appease a test.

### Declared gaps

`MISSING_FEATURES` carries four, each with a reason, so they surface in the CLI
output rather than being absent from a list nobody re-reads: `risk_reversal_25d`
(options archive is 252 rows), `india_imports_yoy` (Comtrade identified, not
ingested), `wgc_bar_coin_yoy` (WGC parsed, gated), `gold_aisc_ratio` (no free feed).

---

## 9. Milestone 4 — the walk-forward harness, delivered

`src/goldbacktest.py`, built before any model exists. 13 tests. Nothing in it knows
what a model is beyond `fit(X, y)` / `predict(X)`, so the four benchmarks and every
later model are scored by identical code.

### Purge and embargo

At prediction date `t` the last training row `T` must satisfy `T + horizon + embargo
<= t`. The purge is the `T + horizon` term — a row's label is not resolved until
`T+h`, so training on it while testing at `t < T+h` trains on the answer. The embargo
is an extra buffer on top, defaulting to the horizon itself (the strict reading of
§7.2).

`walk_forward` takes an `on_fit(t, last_train, n)` callback so this boundary is
asserted against **what the model was actually handed**, not against a reading of the
code. The test iterates every refit and checks the gap from the date that fit was
*triggered* — an earlier version compared against the first prediction of the whole
run, which mistakes later refits legitimately training on newer data for a leak.

### The holdout is locked, and unlocking is a logged event

The final three years are excluded by default. `--holdout` unlocks them, prints a
warning, and writes a dated record to `backtest_runs.jsonl` — so "we only looked
once" is a fact on disk rather than a recollection. Every run logs the git hash, a
hash of the feature set, model config and results (§7.6: no unlogged runs).

### Two scoring bugs the first run exposed

Both were in my own metrics, and both flattered or maligned a benchmark rather than
producing an obviously wrong number:

* **`sign(0)` is not a direction.** RandomWalk predicts all zeros, and scoring
  `(pred > 0) == (actual > 0)` counted every zero as a call for *down*. That handed
  it a hit rate of exactly one minus the always-long rate — a number that reads like
  a result and is an artefact. Hit rate is now measured only over rows where the
  model made a directional call, with `n_directional` reported alongside.
* **Probability by self-normalisation was a magnifying glass on noise.** The first
  version z-scored predictions by their own dispersion. AlwaysLong emits a
  near-constant, so its dispersion is nearly zero, and dividing by it turned
  refit-to-refit jitter into confident 0.05/0.95 probabilities — Brier 0.43 against a
  climatology of 0.25, and a buy-and-hold benchmark that churned in and out of the
  market and lost money. Since models are fitted on the **volatility-scaled** target,
  a prediction is already in standard deviations of the h-period return, so
  `P(up) = Phi(pred)` is the implied probability directly, with no free parameters.

### The four benchmarks, development window (2015→2023-08)

Run 2026-08-22, git 729c80e, feature set be8e33a017 (28 features), 10bp per trade.
Hit rate is over directional calls only; P&L is the spec's rule (long when P(up) >
0.55, flat otherwise) on non-overlapping observations.

| horizon | model | indep | hit | vs always-long | IC | Brier | P&L | maxDD |
|---|---|---|---|---|---|---|---|---|
| **5d** | random_walk | 461 | – | – | +0.020 | 0.250 | 0.000 | 0.000 |
| | always_long | 461 | 52.7% | – | −0.094 | 0.250 | 0.000 | 0.000 |
| | momentum_12m | 416 | 49.8% | −4.1% | −0.051 | 0.253 | +0.034 | −0.165 |
| | real_yield_only | 461 | 50.3% | −2.4% | −0.011 | 0.250 | 0.000 | 0.000 |
| **60d** | random_walk | 38 | – | – | +0.035 | 0.250 | 0.000 | 0.000 |
| | **always_long** | 38 | **60.5%** | – | −0.286 | 0.242 | **+0.462** | −0.164 |
| | momentum_12m | 34 | 38.2% | −20.6% | −0.207 | 0.252 | +0.069 | −0.108 |
| | real_yield_only | 38 | 44.7% | −15.8% | −0.098 | 0.249 | 0.000 | 0.000 |
| **250d** | random_walk | 8 | – | – | +0.191 | 0.250 | 0.000 | 0.000 |
| | **always_long** | 8 | **62.5%** | – | −0.413 | 0.248 | **+0.468** | −0.056 |
| | momentum_12m | 7 | 42.9% | −14.3% | −0.073 | 0.260 | +0.184 | −0.068 |
| | real_yield_only | 8 | 37.5% | −25.0% | +0.041 | 0.254 | 0.000 | 0.000 |

**Read this before reading any later model result.**

1. **Always-long is a hard bar**, exactly as §7 warns — 60.5% at 60 days and 62.5% at
   250 days, with the best P&L at both. Any model quoting a 60% hit rate at 60 days
   has matched buy-and-hold, not beaten it.
2. **Neither single-driver benchmark beats it.** Momentum and real-yield-alone both
   trail by 4–25 percentage points and carry negative IC at the longer horizons.
   That is consistent with the earlier finding that these drivers explain gold
   without forecasting it.
3. **`real_yield_only` never trades at 60d or 250d** (P&L exactly 0.000). Its
   vol-scaled predictions never clear Phi(0.55) = 0.126 standard deviations, so the
   spec's rule never fires. A single variable is too weak a signal to reach the
   threshold — useful to know before concluding a richer model "added" trading.
4. **250 days has 8 independent observations.** It is not scoreable and the harness
   says so. Nothing at that horizon should be reported as evidence.

### Calibration guards, from §7 and §12

`check_calibration` returns warnings rather than raising, because the harness's job is
to report honestly — including reporting that a result is *too good*. A 5-day hit rate
above 60% is flagged **"TREAT AS A BUG OR LEAK until proven otherwise"**, and there is
a test asserting that a deliberately leaky feature trips it. Fewer than 20 independent
observations also raises a warning; at the 250-day horizon that fires, and it should.


---

## 10. Milestone 5 — Stage 1 diagnostics, delivered

`src/golddiag.py`, 8 tests. No model fitted, per §6 Stage 1. Outputs
`diagnostics.json` and `diagnostics.html` (native CSS heatmap — this repo has no
matplotlib or plotly and none was added, matching `heatmap_html.py`'s approach).

### The headline, and it is unambiguous

**Eight features clear the multiple-comparison threshold. All eight are COINCIDENT.
Zero lead.**

| feature | r at lag 0 | peak r | peak lag | significance |
|---|---|---|---|---|
| gold_dist_50d | +0.234 | +0.234 | 0 | adjusted |
| gld_flow_z_1y | +0.171 | +0.171 | 0 | adjusted |
| shanghai_premium_z_1y | −0.131 | −0.131 | 0 | adjusted |
| shanghai_premium_usd | −0.122 | −0.122 | 0 | adjusted |
| gold_dist_200d | +0.119 | +0.119 | 0 | adjusted |
| dxy_dist_50d | −0.105 | −0.105 | 0 | adjusted |
| dxy_chg_20d | −0.091 | −0.091 | 0 | adjusted |
| gold_fx_breadth | +0.072 | +0.072 | 0 | adjusted |
| *fed_cut_odds_chg_20d* | −0.023 | −0.037 | *49* | raw 5% only |
| *breakeven_10y_chg_20d* | +0.021 | −0.034 | *49* | raw 5% only |

The three features whose peak sits at a positive lag (49, 49, 40 days) all fail the
adjusted threshold — which is precisely what a scan over 1,708 cells produces from
noise. Note also that two of them peak with the **opposite sign** to their lag-0
correlation, another signature of a fitted artefact rather than a relationship.

This is the same conclusion the earlier ad-hoc model reached, now established
properly: **these drivers explain gold; they do not lead it.**

### Two statistical choices that drive that conclusion

**The lead-lag study runs on DAILY returns, not the forward targets.** Correlating a
feature against a 60-day forward return at 61 lags means neighbouring observations
overlap 59/60ths of the way; the correlations are massively autocorrelated and their
significance is fiction. Against the daily return each observation is used once per
lag, so n is honest. The daily return is noisier — that is the price of a real answer.

**Significance is Bonferroni-adjusted.** 28 features × 61 lags = 1,708 simultaneous
tests. At the median n the raw 5% bar is |r| > 0.026 and the adjusted bar is
|r| > 0.056. Fourteen features clear the raw bar; **eight** clear the adjusted one.
A test plants twelve pure-noise features and asserts that none of them clears —
it is the test that fails if the adjustment is ever dropped.

### Stability: the case for the regime layer, quantified

Rolling 252-day correlations against the 60-day target, ranked by how one-sided the
sign is (50% = flipped as often as it held):

| feature | full-sample r | rolling range | one-sided |
|---|---|---|---|
| cb_net_purchases_12m | +0.015 | [−0.72, +0.81] | **51%** |
| cb_net_purchases_yoy_chg | +0.029 | [−0.77, +0.81] | **51%** |
| fed_cut_odds_chg_20d | −0.047 | [−0.45, +0.49] | 52% |
| hy_spread_chg_20d | +0.027 | [−0.71, +0.55] | 54% |
| vix_z_1y | +0.043 | [−0.66, +0.66] | 54% |
| dxy_chg_20d | −0.047 | [−0.49, +0.54] | 55% |

Central bank buying swings between −0.72 and +0.81 and is positive in 51% of windows
— a coin flip. No static coefficient can represent that, which is the empirical case
for §6 Stage 4 rather than an assertion about it.

### Three defects found in my own diagnostics

* **The global-minimum sample size.** Thresholds were computed from the shortest
  series (the Shanghai premium, 2,461 rows), setting the bar for features with 5,900.
  Now each feature is judged at its own n — which moved the adjusted threshold from
  0.084 to 0.056 and promoted `gold_fx_breadth` from "raw only" to adjusted.
* **NaN leaking into the stability stats.** `gold_fx_breadth` is constant over some
  windows, giving an undefined correlation and a NaN min/max.
* **The stability metric itself was misleading.** Agreement-with-the-full-sample-sign
  reported 6% for a feature whose full-sample r was +0.045 — noise about zero, whose
  sign is arbitrary. Replaced with one-sidedness of the rolling sign, which stays
  meaningful when the full-sample number is near zero.

### Carried into Milestone 6

Stage 2, the elastic net, validated through the Milestone 4 harness. Requires
sklearn (approved 2026-08-22; `requirements.txt` still needs updating). Stage 1 says
plainly what to expect: a linear model over features that are all coincident should
**not** beat always-long out of sample, and the harness is built to say so.


---

## 11. Milestone 6 — Stage 2 elastic net, delivered

`src/goldmodels.py`, 12 tests. `scikit-learn==1.9.0` added to `requirements.txt`
(approved 2026-08-22; the repo was deliberately numpy-only before this).

### The verdict, against spec §12

**The elastic net does not beat all four benchmarks. It does not ship as a
forecasting model.**

| horizon | model | indep | hit | vs always-long | IC | Brier | P&L |
|---|---|---|---|---|---|---|---|
| **5d** | always_long | 461 | 52.7% | – | −0.094 | 0.250 | 0.000 |
| | **elastic_net** | 461 | **52.9%** | **+0.2%** | −0.021 | 0.249 | +0.085 |
| **60d** | always_long | 38 | 60.5% | – | −0.286 | 0.242 | +0.462 |
| | **elastic_net** | 38 | **60.5%** | **+0.0%** | −0.286 | 0.242 | +0.462 |
| **250d** | always_long | 8 | 62.5% | – | −0.413 | 0.248 | +0.468 |
| | **elastic_net** | 8 | **50.0%** | **−12.5%** | −0.372 | 0.332 | +0.467 |

At 60 days the elastic net is **numerically identical to always-long** — same hit
rate, same IC, same Brier, same P&L to three decimals. That is not a coincidence and
not a bug: the model selects zero features, so its prediction is the training mean,
which *is* the always-long model. At 5 days it adds 0.2 percentage points on 461
independent observations — roughly one extra correct call. At 250 days it is 12.5
points worse on 8 observations, which is noise in both directions.

### Purged CV shrinks everything to zero

Fitted on the development window at 60 days:

```
alpha=0.3000   l1_ratio=0.9   purged-CV MSE=1.1708   0/26 features selected
```

Given a free choice of penalty, purged cross-validation prefers the null model to any
combination of the 26 features. That is exactly what Stage 1 predicted — every
feature clearing the multiple-comparison threshold is coincident, and a linear model
over coincident features has nothing to forecast with.

### Three things that had to be checked before believing it

* **A boundary CV solution.** The first run reported `alpha=0.1` with 3 features and
  looked like a modest result — but 0.1 was the *largest value in the grid*, so the
  grid chose the answer, not the data. Extending to 3.0 moved the optimum to an
  interior 0.3, improved CV MSE from 1.276 to 1.171, and dropped selection to zero.
  `alpha_at_boundary_` now flags the condition, with a test.
* **Solver non-convergence.** The harness run emitted `ConvergenceWarning`s at small
  alphas. That mattered: a fit that stops early scores a *worse* validation MSE, so
  the grid search would have been rejecting small alphas for solver reasons rather
  than for overfitting — and "no features" would have been an artefact. Re-run at
  `max_iter=100000, tol=1e-5`: **zero non-converged fits, and the result is
  identical**. The count is now tracked and reported.
* **The Newey-West correction, tested on the right shape.** An earlier version of the
  test used i.i.d. regressors and found HAC errors *smaller* than OLS — correctly, as
  it happens. HAC corrects autocorrelation in the score `X_t·u_t`, not in the residual
  alone, and white-noise regressors decorrelate the products. The real feature matrix
  is all rolling z-scores and moving-average distances — strongly persistent, on
  overlapping returns. Both legs persistent is when the correction bites, and there it
  runs 2×+ the naive errors.

### What this means for Milestones 7-8

Stage 3's bucket scores are the next thing to try, and they are a *different* bet
rather than a refinement of this one: equal-weighted theory-signed buckets have no
fitted parameters, so they cannot be shrunk to zero by a penalty and do not need the
sample size a 26-parameter fit does. Stage 4's regime layer is the other live option,
and Stage 1 already quantified the case for it — central bank buying's rolling
correlation swings [−0.72, +0.81] and is positive in 51% of windows.

Stage 5 (LightGBM) should be approached with the spec's own scepticism. If a
regularised linear model over these features selects nothing, a tree model that finds
structure is more likely to have found the sample than the signal.


---

## 12. The deep re-run — the definitive result

After the adversarial audit invalidated the Stage 3 headline, three self-imposed
history caps were lifted (`golddata.START` 2010, the LBMA filter 2005,
`goldfeatures.START` 2004 — none of them a data limit) and spec §2.7's synthetic
pre-2003 real yield was implemented. The evaluable window went from **6.2 years to
21.1 years**; 60-day independent observations from **27 to ~101**; and the 250-day
horizon became scoreable for the first time.

Everything was then re-run on the DEEP feature set (15 features, 1992→2023).

### The verdict

**No model beats buying and holding gold. Several are significantly worse.**

| horizon | model | indep | hit | vs always-long | McNemar p | vs buy&hold |
|---|---|---|---|---|---|---|
| **5d** | always_long | 1243 | 54.4% | – | – | −1.860 |
| | momentum_12m | 1198 | 53.2% | −1.6% | 0.684 | −1.066 |
| | elastic_net | 1243 | 52.4% | −1.9% | 0.067 | −1.726 |
| | real_yield_only | 1187 | 49.8% | −5.2% | **0.002** | −1.970 |
| | **bucket_equal_fixed** | 1243 | **47.5%** | **−6.9%** | **0.003** | −1.646 |
| **60d** | always_long | 101 | 59.8% | – | – | −1.534 |
| | elastic_net | 101 | 55.7% | −4.1% | 0.648 | −1.059 |
| | bucket_equal_fixed | 101 | 47.3% | −12.5% | 0.775 | −1.385 |
| **250d** | always_long | 22 | 71.3% | – | – | −1.075 |
| | momentum_12m | 22 | 65.9% | −4.2% | 1.000 | −0.962 |
| | bucket_equal_fixed | 22 | 35.1% | −36.2% | 0.146 | −1.599 |

**`bucket_equal_fixed` — the model that looked like the first winner — is
significantly WORSE than always-long on the deep sample (p = 0.003).** Its +2.6pp on
six years is −6.9pp on twenty-one, with the p-value now pointing the wrong way. That
is as clean a demonstration as this project will produce that the original result was
a short-window artefact.

Cumulatively, over the 21-year development window gold returned **+542% to +618%**
(sum of non-overlapping log returns 1.86–1.97). The best model captured 1.443 against
buy-and-hold's 1.971 — a shortfall of 0.53 in log terms. Every model that trades in
and out underperforms simply holding the metal.

### Caveats stated rather than buried

* The deep set has **no flows and no physical bucket** — COT starts 2006, GLD 2004,
  Shanghai 2016. That is 35% of the spec's 60-day weight, renormalised away. The deep
  run therefore tests a **three-bucket** model, not the same model on more data.
* The 250-day row rests on 22 independent observations. It is scoreable at last, but
  a 36-point gap there is not a precise quantity.
* `always_long` as implemented is *not* buy-and-hold: it predicts the training mean
  and then applies the spec's 0.55 probability rule, which leaves it flat most of the
  time at 5 days. `buyhold_sum_logret` is the honest benchmark and is now reported.

### Stage 1, re-asked with four times the power

9,247 daily observations (was 2,461), Bonferroni threshold tightened from |r| > 0.056
to 0.043. **Eight features clear it; all eight are coincident; zero lead.** The
lead-lag conclusion is not a small-sample artefact.

Two movements worth recording: `gld_flow_z_1y` fell from +0.171 to +0.123 once its
publication stamp was corrected — visible evidence of the look-ahead being removed —
and the dollar's coincident correlation collapsed from −0.105 to −0.027 across 21
years, meaning "the dollar comes second" is period-dependent rather than structural.

### One more bug the deepening surfaced

The exact McNemar test computes `2**n` on the discordant-pair count. At six years that
was ~100 and fine; at twenty-one it is ~250 and overflows a float. Now exact below
n=60 and a continuity-corrected normal approximation above — clamped at zero, because
with b == c the correction goes negative and returned p ≈ 0.91 for two models that
agreed exactly. A defect the shallow sample could never have reached.
