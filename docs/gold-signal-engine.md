# Gold Signal Engine — repo survey and gap list

**Status:** Milestone 1 **complete** — survey, gap list, schema, point-in-time
accessor, and leak tests all landed. Milestone 2 (ingestion) not started.
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
| WGC quarterly demand (jewellery, bar/coin, ETF, CB, tech, supply, recycling) | ❌ | quarterly | — | **Genuinely gated.** The machine-readable file exists — `gold.org/download/file/20975/GDT_Tables_Q2'26_EN.xlsx` — but returns **HTTP 403** without a Goldhub account. Needs registration and acceptance of licence terms, which also bears on whether it may appear in client PDFs. |
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

### Carried into Milestone 2

The **COT publication-date leak** is the first thing to fix: `cot_history.parquet`
stores only the Tuesday reference date, so anything reading it before the Friday
15:30 ET release is trading on unpublished data. The store now has somewhere
correct to put that timestamp.
