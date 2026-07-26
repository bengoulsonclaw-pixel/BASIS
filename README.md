# Strategy Monitor

A local Streamlit dashboard that flags potential futures trading opportunities
per strategy and exports a client-style PDF. It runs on **synthetic data** today;
swap in **Bloomberg** by editing one file when your API is live.

## Run it
```powershell
# from this folder:
.venv\Scripts\python.exe run_daily.py                 # compute today's signals
.venv\Scripts\python.exe -m streamlit run app.py      # launch the dashboard
```
Then open http://localhost:8501 — click a strategy button, tick the rows you
like, and hit **Generate PDF report**.

## Going live with Bloomberg
The client libraries (`blpapi`, `xbbg`) are installed in the venv. To switch
from synthetic data to live Bloomberg:

1. **Open the Bloomberg Terminal on this PC and log in.** This starts `bbcomm`
   and opens the Desktop API on port 8194 — your login is the authentication.
2. **Verify the connection:**
   ```powershell
   .venv\Scripts\python.exe check_bloomberg.py
   ```
   "SUCCESS" means you're connected and entitled. If it connects but returns no
   data / an entitlement error, ask your Bloomberg rep to enable Desktop API /
   data access (`WAPI<GO>`).
3. **Run live** — no code edit, just set the switch and launch:
   ```powershell
   $env:DATAFEED_MODE = "bloomberg"
   .venv\Scripts\python.exe run_daily.py
   .venv\Scripts\python.exe -m streamlit run app.py
   ```

Every strategy then runs on real settlement data. (Back to demo: `Remove-Item Env:\DATAFEED_MODE`.)

## Layout
| Path | What it does |
|------|--------------|
| `app.py` | Dashboard: one button per strategy → table → Export PDF |
| `run_daily.py` | Scheduled job: pull data, run strategies, cache results |
| `src/datafeed.py` | **The data adapter** (mock ↔ Bloomberg) — the only file to change to go live |
| `src/universe.py` | Your instrument universe and the mean-reversion pairs |
| `src/strategies/` | One module per strategy (`mean_reversion`, `trend`, `ma_crossover`, `flag_breakout`, `support_resistance`, `fibonacci`, `breakout_retest`, `momentum`, `bollinger`, `carry`, `volatility`, `skew`, `termstructure`, `cot`, `ag_fundamentals`) |
| `src/report.py` + `templates/report.html` | Per-strategy table PDF (headless Chrome) |
| `src/reportkit.py` | Shared branding + chart helpers for the visual reports |
| `src/volreport.py` + `templates/volreport.html` | Visual **Volatility** Report PDF (implied vs realized) |
| `src/skewreport.py` + `templates/skewreport.html` | Visual **Skew Volatility** Report PDF (25Δ put−call skew) |
| `src/termreport.py` + `templates/termreport.html` | Visual **Vol Term Structure** Report PDF (1M/3M/6M/12M curve) |
| `src/oireport.py` + `templates/oireport.html` | Per-product **Open Interest** heatmap PDF (strike × expiry-month grid) — single product or whole-book chartbook |
| `src/flagreport.py` + `templates/flagreport.html` | Visual **Flag Breakout** Report PDF (readiness bar + per-product flag charts) |
| `backtest_flags.py` | Walk-forward flag-breakout follow-through study (calibrate the detector knobs) |
| `src/tascore.py` | Cross-strategy **conviction scoring** (per-signal strength + per-product score) shared by the TA hub & report |
| `src/convreport.py` + `templates/convreport.html` | **Technical Analysis** report PDF (conviction leaderboard + summary table, then per-pick multi-indicator charts & neutral write-ups) |
| `src/tareport.py` + `templates/tareport.html` | _Retired_ standalone TA-overview PDF — merged into the Technical Analysis report above; kept for reference |
| `data/signals/` | Cached daily opportunities (written by `run_daily.py`) |

## Schedule the daily run (Windows Task Scheduler)
Create a Basic Task that runs daily after settlements:
- **Program/script:** `C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard\.venv\Scripts\python.exe`
- **Arguments:** `run_daily.py`
- **Start in:** `C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard`

### Weekly option-OI pull (Mondays — Terminal must be logged in)
The fixed-income option chains are heavy, so they're a **separate weekly job**. Create a Basic
Task, trigger **Weekly · Monday**, that runs the OI capture with Bloomberg live:
- **Program/script:** `…\.venv\Scripts\python.exe`
- **Arguments:** `snapshot.py --oi`
- **Start in:** `…\strategy-dashboard`
- It needs the Terminal open/logged in at run time (it pulls the 11 chains live). Or just click
  **↻ Refresh OI** on the Open Interest page on a Monday.

## The Volatility Report
`src/strategies/volatility.py` compares **1-month ATM implied vol** against
**~1-month (21 trading day) realized vol** for every market. The signal is the
**z-score of the implied−realized spread over the trailing year** (|z| ≥ 1.5
flags): rich (implied dear → sell vol) or cheap (implied under-pricing → buy vol).
Realized is computed from `PX_SETTLE` (works in demo mode); implied comes from the
option surface (FX from the OTC pair vol). It produces two things:
- **Dashboard rows** — the full cross-section on the *Volatility* strategy button.
- **A visual client PDF** — `src/volreport.py` + `templates/volreport.html`: a
  scatter of implied vs realized (distance from the 45° line = the spread), a
  ranked diverging bar of the most stretched markets, and a flagged table.
  Generate it from the **Volatility** page → *Generate Volatility Report*.

**Coverage / things to confirm on the Terminal** (verified live 2026-06-03):
- **Implied field** (`src/datafeed.py` → `IMPLIED_VOL_FIELD` =
  `30DAY_IMPVOL_100.0%MNY_DF`): works for US indices, bonds, energy, metals, ags,
  softs. Non-US equity indices and FX return nothing on it →
- **Overrides** (`src/universe.py` → `IMPLIED_VOL_OVERRIDE`): FX → OTC pair vol
  (`EURUSDV1M Curncy`, … 15 mapped). HEA/CCA/PPA Curncy remain unmapped.
- **Non-US equity indices are excluded for now** (`_excluded()` in
  `volatility.py`) — only US index futures (ES/NQ/RTY/Dow) are included. The
  verified vol-index sources (V2X/V1X/V3X/VNKY) are kept commented in
  `IMPLIED_VOL_OVERRIDE` to re-enable later.
- **STIRs are excluded** (`EXCLUDE_ASSETS` in `volatility.py`): money-market
  futures are pinned near 100 so their *price* realized vol ≈ 0 and isn't
  comparable to option-implied. Rate vol needs a dedicated normal/bp treatment.
- **Realized leg** is 21d from front-contract settles; for a roll-clean number you
  can switch the live leg to Bloomberg's `VOLATILITY_30D`.

Tunable knobs at the top of `volatility.py` (`STAT_WINDOW`, `Z_FLAG`). Natural next
step: add a 3-month tenor (vs 63d realized) and a term-structure column.

## The Skew Volatility Report
Its **own** strategy + report (separate from Volatility). `src/strategies/skew.py`
ranks the normalized 25-delta skew **`(25Δ put − call) / ATM`**, z-scored over the
same 252-day window (positive = puts richer than calls; `|z| ≥ 1.5` flags). The
computation is `volatility.compute_skew_table()`; rows feed the *Skew Volatility*
dashboard page and the rich table is cached to `data/signals/skew.parquet`.
Generate the visual PDF from that page → *Generate Skew Volatility Report*
(`src/skewreport.py`: put-vs-call scatter, ranked bars, flagged table).

- **Wings**: listed markets use the **90/110% moneyness** vols of the surface; **FX**
  uses the native **OTC 25Δ risk reversal** (`…25R1M`/`…25B1M Curncy`).
- **Excluded**: **Bonds and STIRs** (`SKEW_EXCLUDE_EXTRA`) — at low price vol the
  moneyness wings sit far OTM where the surface is garbage (US 2Y came out at skew
  6.2); any `|skew| > SKEW_CAP (2.5)` is also dropped. Non-US indices excluded as above.
- Limitation: 90/110% ≈ 25Δ only at medium vol. For a true 25Δ listed skew,
  interpolate vol-scaled moneyness wings.

## The Vol Term Structure Report
Its own strategy + report. `src/strategies/termstructure.py` reads ATM implied vol at
**1M / 3M / 6M / 12M** and tracks the **3M − 1M slope**, z-scored over the same 252-day
window: high z = steep **contango** (front cheap → buy front), low z = **backwardation**
(front rich → sell front); `|z| ≥ 1.5` flags a calendar-spread idea (fade the slope). It
also carries each tenor's implied-vs-realized premium (1M↔21d, 3M↔63d, 6M↔126d, 12M↔252d).
Rows feed the *Vol Term Structure* page; the rich table is cached to
`data/signals/termstructure.parquet`. Generate the visual PDF from that page →
*Generate Vol Term Structure Report* (`src/termreport.py`): a 1M-vs-3M scatter, slope-z
bars, per-product curve small-multiples with **realized overlaid** (implied solid vs
matched-window realized dashed), a VRP-by-tenor chart, and a flagged table.

- **Data** (`datafeed.get_term_structure` / `TENORS`), **verified live 2026-06-04**: listed reads
  the surface ATM — the front in **days** (`30DAY`) but ≥3M in **months**
  (`3MTH/6MTH/12MTH_IMPVOL_100.0%MNY_DF`); the `90/180/360DAY` day-tokens publish nothing. FX
  reads the OTC tenor vols (`…V1M/V3M/V6M/V1Y`). Live = 62 markets, real curves (S&P contango,
  Corn / OJ backwardation).
- Bonds are kept (the ATM curve is fine); STIRs and non-US indices excluded as elsewhere.

## The Open Interest Report
A per-product view of **listed-option open interest** as a **strike × expiry-month
heatmap**: each cell is the *total* open interest (puts + calls) struck there, shaded
pale→deep-red by size, with a dashed line at spot. The biggest strikes show where
positioning — and dealer hedging — concentrate (frequent pin / magnet levels into
expiry). It lives on the **Open Interest** page (*Positioning & Flow*): pick a product
for a live heatmap, then export **this product** (one page) or the **whole book** (one
compact heatmap per market) as a branded PDF.

- **Data** (`datafeed.get_oi_chain`): a tidy `(expiry, strike, call_oi, put_oi)` grid for
  one underlying. **Mock** synthesises a realistic ATM-peaked surface (front-loaded,
  quarterly bumps, index put-skew) so it works offline. **Live** pulls the option chain
  (`OPT_CHAIN` → per-option `OPT_STRIKE_PX` / `OPT_EXPIRE_DT` / `OPT_PUT_CALL` /
  `OPEN_INT`) and pivots puts vs calls — fields are **PROVISIONAL**, confirm per asset
  class on the Terminal (OMON<GO> / FLDS<GO>); thin/absent chains (much of FX) come back
  empty. **Snapshot**: the option chains are a **separate weekly job** — `python snapshot.py --oi`
  captures the **11 fixed-income products** (`OI_SNAPSHOT_TICKERS` — Euribor · SONIA · SOFR, US
  2/5/10/30 and German 2/5/10/30) to `data/snapshot/oi_chain.parquet`. Run it **Mondays** (or the
  **↻ Refresh OI** button on the page); the *daily* snapshot no longer pulls OI, to keep the
  Bloomberg draw light. Every other product pulls its chain **live on demand** in bloomberg mode.
  The live pull walks the generic **and dated** contracts (`OPT_CHAIN` / `FUT_CHAIN`) to get the
  whole strip, bounds it by `OI_LIVE_*`, then a batched `OPEN_INT` bdp grouped by the option's
  `OPT_EXPIRE_DT`. `n_strikes` / `n_expiries` are display windows applied at read.
- The app builds the grid and hands `src/oireport.py` a parquet to render (the report is
  a pure renderer, like the others). Distinct from **Put/Call Ratios**, which carries
  each product's *aggregated* OI totals over time rather than the per-strike grid.
- **Fixed Income book** — a curated one-click PDF (the 🏛️ button): **one product per page**
  with its full strike chain, in tenor order — STIRs (SOFR · SONIA · Euribor), then **US vs
  German** at 2 / 5 / 10 / 30 years (the two of each tenor on consecutive pages), ~11 pages.
  The order and per-tenor strike grids live in `FI_OI_PAGES` in `app.py`; rendered via
  `oireport.py --scope grouped` (the input parquet carries `page` / `page_title`).

## The Flag Breakout Report
A continuation-pattern monitor: it scans the book for **flags and pennants** — a sharp
**flagpole** then a tight consolidation (parallel channel = flag, converging = pennant) —
and ranks each by **breakout readiness** (0–100; 100 = price sitting on the breakout trendline). Readiness is stored
**signed by pattern** (+ bull / − bear) so it slots into the standard ±trigger machinery:
`readiness ≥ trigger` flags a **bull breakout setup** (breaks up), `≤ −trigger` a **bear**
(breaks down). Default trigger 70 (tunable on the page, 50–100). Lives on the **Flag
Breakout** page (*Price & Trend*).

- **Detection** (`src/strategies/flag_breakout.py`, close-based): the flagpole is a fast,
  strong move (≥ both a 4% floor and ~2.5σ of the product's own daily moves over its
  length); the flag is the 5–15 days since (≤ ~3 weeks), requiring ≤ 50% retrace, a channel
  (trendline ±2σ) much tighter than the pole, and a gentle counter/flat slope. Every knob is
  a named constant at the top of the module.
- **Targets** (the measured move, per the desk literature): each flag carries a **target**
  (the flagpole height projected from the breakout), a **stop** beyond the far edge of the
  flag, and the **reward:risk** on a break-level entry — shown on the page metrics, drawn on
  the dashboard chart (green target / red stop) and listed in the report.
- **Volume confirmation**: with a volume series (`datafeed.get_volume_history` →
  `FUT_AGGTE_VOL` — aggregate across all contracts; verified live across every asset class
  2026-06-20, where the active-contract `PX_VOLUME` reads 0 right after a roll), each flag
  carries a **dry-up ratio** (flag vs pole average volume; < 1 = the textbook contraction)
  and a **breakout surge**. Volume is a shown *confirmation*, not a gate — a missing series
  just shows none, and any stray zero is treated as missing. Mock synthesises a
  move-size-driven series so the dry-up is visible offline. (`diag_volume.py` re-runs the
  field check on the Terminal.)
- **The page** draws price with its flagpole, shaded channel and dashed breakout line,
  plus a volume subpanel. The **visual PDF** (`src/flagreport.py` +
  `templates/flagreport.html`): a signed-readiness ranked bar and a price+channel panel
  per flagged market. Generate it from the page → *Generate Flag Breakout Report*.
- **Calibration**: `backtest_flags.py` walks history and reports the forward continuation
  after breakouts (hit rate + average move) by pattern, horizon, volume confirmation and
  trigger — so the knobs are tuned on real Bloomberg depth, not by eye
  (`.venv\Scripts\python.exe backtest_flags.py`). The rich per-flag table is cached to
  `data/signals/flag_breakout.parquet` (+ `_history`) for the report.

## Technical signal strategies (price action & oscillators)
Four classic-TA monitors (Murphy-style), all computed from the close (+volume) history —
each a button under *Price & Trend* or *Momentum & Bands*, with a per-market chart plus the
standard trigger / table / tick-rows-PDF machinery:

- **Support & Resistance** (`support_resistance.py`) — tested horizontal levels from swing
  pivots (strength = touches); buy near support / sell near resistance, scored 0–100 by
  proximity and signed + support / − resistance. Broken levels flip role.
- **Fibonacci Retracement** (`fibonacci.py`) — auto-Fib on each product's dominant swing
  (23.6 / 38.2 / 50 / 61.8 / 78.6 %); flags price reacting at a key level (golden zone) —
  long on an up-leg pullback, short on a down-leg bounce — with a target (the prior extreme),
  a stop beyond 78.6 % and R:R.
- **Breakout & Retest** (`breakout_retest.py`) — a level broken on a volume surge, then
  retested after it flips role (old resistance → support, and the mirror). Conditional —
  only active retests are listed; uses `get_volume_history` for the breakout-volume gate.
- **Momentum (RSI/MACD)** (`momentum.py`) — Wilder RSI(14) + MACD(12/26/9) with **RSI
  divergence** as the headline reversal warning; a signed momentum score, bullish vs bearish.
- **Bollinger Squeeze** (`bollinger.py`) — Bands(20, 2σ); a **squeeze** (bandwidth in a low
  percentile of its year) flags compressed volatility coiling for a breakout, and a close
  outside the band is the break. Signed squeeze intensity (0–100).

Tunable constants sit at the top of each module. All four are close-only (Breakout & Retest
also reads volume) and so run identically in mock, snapshot and live.

### 🔬 Technical Analysis overview
`render_ta_overview()` in `app.py` is the **hub** for the technical book — reached from the
**🔬 Technical Analysis** nav button; the nine strategies live under the matching sidebar group.
It re-flags every technical strategy at its trigger (`src/tascore.py → ta_flagged`) and ranks
everything by a **cross-strategy conviction score** (`tascore.score_products`): each signal is
normalised to a 0–100 **strength** on its own scale, and a product's **score** is the signed sum
of those strengths across the strategies flagging it (confluence × strength, longs netted against
shorts — agreement amplifies, conflicts cancel). The page shows headline counts, a
**stacked-signals** table (flagged by 2+, ranked by score), per-strategy counts with one-click
drill-down, a **chart gallery** of the top stacked setups (price + the visual strategies' levels
overlaid), and the full filterable leaderboard. Trend is held to a selective bar (|3-month| ≥ 10%)
so it doesn't flood the hub (`tascore.TA_HUB_TRIGGER`).

The **📈 Generate Technical Analysis Report** button renders a branded PDF (`src/convreport.py` +
`templates/convreport.html`): page 1 is the conviction-leaderboard bar and the stacked-signals
summary table; the per-pick multi-indicator charts and neutral write-ups follow from page 2. This
is the merged report (the old standalone "TA Report" has been retired). `tascore` is shared, so the
PDF scores identically to the page.

## Adding a strategy
Copy `src/strategies/trend.py`, rewrite `find_opportunities()` to return the
standard columns (see `src/strategies/base.py`), then add it to the list in
`run_daily.py` and to `STRATEGY_ORDER` in `app.py`. For a visual report like
Volatility/Skew, add an entry to `REPORTS` in `app.py` and a `*report.py` + template.
