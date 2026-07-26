# Precious Metals Fundamentals — verified source inventory

Verified live **16 Jul 2026** (web-checked). This is the wiring plan for the Monitor's
mock-flagged blocks (src/pmdata.py) and the trigger calendar for the release synopses.
Flags: **FREE** = free to access · **REG** = free but registration/login · **PAID** = licence
required · redistribution notes are for client-facing PDFs (we summarise with attribution,
never repost files/tables wholesale).

## A. Monthly Monitor — data blocks

| Block | Source | Freq / lag | Access | Automation | Status |
|---|---|---|---|---|---|
| COT positioning | CFTC Public Reporting API (Socrata), disaggregated combined `kh3c-gbw2`; GC `088691` (Micro is `088695` — don't match by name), SI `084691`, PL `076651`, PA `075651`; managed-money fields | Weekly Fri 15:30 ET (Tue positions) | FREE, public domain | Clean JSON API, no auth (already reused via src/cotdata.py) | **LIVE in pmdata** |
| Real yield / dollar | FRED `DFII10` (10y TIPS), `DTWEXBGS` (broad USD) via api.stlouisfed.org | Daily, 1bd lag | FREE (needs free API key; web host 403s plain clients) | Key in `data/fred_key.txt` or `$FRED_API_KEY` — **get a free key at fred.stlouisfed.org/docs/api/api_key.html** | Wired; falls back to mock until key present |
| ETF holdings | Bloomberg total-known-ETF-holdings fields (all four metals); WGC Goldhub monthly xlsx as gold cross-check (REG) | Daily (BBG) / monthly (WGC) | BBG entitlement / REG | Small field set — negligible vs daily hit limit | MOCK → wire via datafeed/snapshot |
| COMEX/NYMEX stocks | CME daily depository files: `delivery_reports/Gold_Stocks.xls`, `Silver_stocks.xls`, `PA-PL_Stck_Rprt.xls` (legacy .xls, xlrd) | Daily ~16:00 CT, prior-day | FREE; CME ToU restrict systematic redistribution — summarise only | **Akamai 403s plain clients** → src/pm_fetch.py real-Chrome; archive in `data/signals/pm_comex.parquet` (builds from 14 Jul 2026 — chart appears once ≥3 snapshots) | **LIVE** |
| Delivery notices | CME `MetalsIssuesAndStopsReport.pdf` (+MTD/YTD) | Daily in delivery periods | FREE, same CME caveats | Same Akamai caveat; PDF table extraction | Optional later |
| LBMA vaults/clearing | `cdn.lbma.org.uk/downloads/LBMA-London-Vault-Holdings-Data-<Month>-<Year>.xlsx` (5th b-day, prior month); clearing xlsx similar | Monthly | FREE, published for transparency | Plain HTTP works; month-stamped filename → scrape landing page | MOCK (not yet in report) → candidate addition |
| **LBMA prices** | ICE Benchmark Administration (took over **PL/PA on 1 Jul 2026** from LME; runs all four fixes) | — | **PAID licence for any use/redistribution** | Do NOT scrape/republish — quote COMEX settlements instead (no restriction) | Deliberately excluded |
| Swiss gold trade | BAZG Swiss-Impex (HS 7108 gold / 7106 silver / 7110 PGMs; bar code 7108.1200 with origin keys 911–914); machine-readable via I14Y / opendata.swiss | Monthly, ~3 wks after month-end | FREE; "open use, attribute; commercial use → permission" — email stat@bazg.admin.ch once | Direct open CSV (no browser!): `ocean.nivel.bazg.admin.ch/open-data-reports/TN8_EXP_en/TN8_EXP_en.zip` (~684 MB monthly, whole revised history) → filter `Tariffnumber8 == 7108.1200`; archive `data/signals/pm_swiss.parquet` | **LIVE** (latest May 2026) |
| Central banks | IMF Data portal (data.imf.org, SDMX REST — old IFS restructured into it) + WGC Goldhub monthly commentary/xlsx (REG) | Monthly, ~2-month lag; misses unreported buying (China) — cite WGC GDT quarterly estimate alongside | FREE (IMF) / REG (WGC) | **Tested**: `api.imf.org/external/sdmx/2.1/data/IMF.STA,IL/GX010.RGV_REVS..M` (world aggregate, fine troy oz ×31.1035/1e6 = tonnes); wired in pmdata.fetch_central_banks, cache `pm_cb.parquet`. Diffs are noisy (TUR swaps, revisions; RUS stopped reporting Nov-25) — always "reported basis" | **LIVE** (latest May 2026) |
| Premiums (SGE/India) | Compute ourselves: SGE benchmark (BBG tickers; or en.sge.com.cn) in CNY/g → $/oz minus London; India = IBJA rates (ibjarates.com) vs landed cost (spot×USDINR +6% duty +3% GST) | Daily fixes | FREE | Bloomberg fields cleanest (avoids scraping); SGE monthly Data Highlights PDF for withdrawals (~2–5 bd after month-end) | MOCK → wire |
| US Mint | usmint.gov bullion-sales page (monthly) + tidy CSVs on /data (quarterly refresh, e.g. `bullion-american-eagle-silver.csv`) | Monthly, days after month-end | FREE, public domain | src/pm_fetch.py: in-page `fetch()` of the tidy CSVs (`bullion-american-eagle-gold/silver.csv`) rides the browser's Cloudflare pass; CSVs verified current (June 2026). Gold CSV = 4 coin-weight groups, oz = units × weight. Archive `pm_mint.parquet` | **LIVE** (2024-01 → 2026-06) |
| Perth Mint | perthmint.com investor blog monthly sales update (now incl. platinum) | Monthly, first half of following month | FREE | Scrape listing page (slugs inconsistent); Kitco coverage as fallback | Optional later |
| PGM balances | WPIC Platinum Quarterly + JM PGM Market Report (see B) | Quarterly / annual | FREE | `data/pm_pgm.json` seeded with REAL published figures (Pt: WPIC Q1-26 — 2025 −1,191 koz, 2026f −297; Pd: JM May-26 — 2025 −416, 2026f +214). Re-seed each release; next WPIC 9 Sep 2026 | **LIVE** (seeded 16 Jul 2026) |
| Auto production | OICA quarterly/annual xlsx (2025: 96.4m units); S&P Global Mobility free monthly press headline between quarters | Quarterly+ | FREE (cite with attribution) | Scrape statistics page for current file link | MOCK → wire |
| Solar PV (silver) | IEA-PVPS Snapshot (April, stable URL `/snapshot-reports/snapshot-YYYY/`, no form) — preferred over SolarPower Europe GMO (REG form); methodologies differ (698 vs 664 GW for 2025) — pick one, footnote it | Annual | FREE | Trivial annual fetch | Context only |

## B. Release synopses — trigger calendar

> **Engine BUILT 16 Jul 2026**: `src/pmrel.py` (detect/fetch/parse → hand-editable JSON in
> `data/pm_releases/`) + `src/pmrelreport.py` (one-page branded PDF with a context chart) +
> `pm_release_scheduled_email.py` (daily 08:30 check, task "PM Release Synopses (daily)",
> default OFF, Ben-only recipients). Covers **wgc** + **wpic**; validated on the Q1 2026
> editions. Silver Institute / JM annuals: add to PUBS when wanted.

| Publication | Cadence | 2026 observed / next | Access | Auto-detection |
|---|---|---|---|---|
| WGC Gold Demand Trends | Quarterly, ~4–5 wks after qtr | Q1-26 published 29 Apr; **Q2 due ~end Jul 2026** | FREE (data xlsx behind free Goldhub login) | Stable URL pattern `gold-demand-trends-q{N}-{YYYY}`; newsletter email |
| WPIC Platinum Quarterly | Quarterly, ~7 wks after qtr | Q1-26 on 18 May; **next pre-announced 9 Sep 2026** | FREE incl. Excel tables (data by Metals Focus) | Pre-announced dates; scrape archive page (file IDs change); email list |
| Silver Institute World Silver Survey | Annual, mid-April | 2026 edition out 15 Apr | FREE PDF | WordPress: predictable `/wp-content/uploads/{YYYY}/04/` + RSS `/feed/` |
| Silver Institute Interim Review | Annual, mid-Nov | Last 13 Nov 2025; next ~mid-Nov 2026 | FREE PDF | Same |
| Johnson Matthey PGM Market Report | **Annual only since 2022** (May, LPPM week) | 2026 edition out May 2026 | FREE (click-through) | Poll matthey.com media page (PDF URLs tokenised — don't template) |
| Heraeus Precious Appraisal | Weekly (Mon) | Live (Ed. 25, 13 Jul 2026); written by SFA (Oxford) | FREE + email list | Email ingestion (Gmail API pattern) — desk colour, not a synopsis trigger |
| Metals Focus Precious Metals Weekly | Weekly | Live | FREE (Mailchimp signup); flagship annuals PAID | Email ingestion |

## C. Licensing summary for client PDFs

- **No meaningful restriction:** CFTC, FRED, IMF, US Mint (public domain); LBMA vault/clearing data.
- **Summarise with attribution, don't repost files/tables:** WGC, WPIC, Silver Institute/Metals Focus, JM, Heraeus, OICA, CME files. WGC & WPIC are the most explicit ("no commercial reproduction without written consent") — press-release headline figures are intended for citation; a one-time permission email to WGC + WPIC + Metals Focus is cheap insurance.
- **Hard no:** LBMA/IBA benchmark **prices** in any redistributed form without an IBA licence (incl. the WGC-carried LBMA price data, marked internal-use-only). Use COMEX settlements.
- Swiss BAZG: attribute + one-time commercial-use permission email (stat@bazg.admin.ch).

## D. Remaining wiring (as of 16 Jul 2026 evening)

DONE: COMEX / Mint / Swiss / IMF / PGM seed / gold+silver ETF (`ETFGTOTL`, `ETSITOTL` via
src/pm_bbg.py) / real yield + dollar (`USGGT10Y`, `BBDXY` via pm_bbg — no FRED key needed
while a terminal refreshes the cache; the FRED-key route stays as backup).

1. ~~Pt/Pd ETF + Shanghai leg~~ **DONE via Ben's tickers**: `ETFHPLAT`/`ETFHPALL Index`
   (= ETF Securities Ltd physical holdings — an issuer family, not an all-ETF total; labelled
   honestly in the footnote) and `.SHGOLDOZ G Index` (Ben's CIX, Shanghai gold in USD/oz;
   premium = minus `XAU Curncy`, daily closes → `pm_prem.parquet`; CIX resolves only under
   Ben's terminal login). **The Monitor is now 100% live — mock_blocks: none.**
2. India premium: no BBG series — compute vs IBJA landed cost or cite Reuters (parked).
3. Auto production: PARKED — OICA's 2025 site redesign removed the downloadable stats
   (JS/Elementor only); chart dropped from the report, WPIC/JM synopsis bullets carry the
   auto-demand story. Rewire if OICA restores files or another free source appears.
4. Optional additions: LBMA vault xlsx, CME delivery notices, Perth Mint.
5. NYMEX Pt/Pd stocks chart self-unlocks once `pm_comex.parquet` has ≥3 daily snapshots.

The monthly emailer runs `src/pm_fetch.py` first (CME + Mint via real Chrome, Swiss 684 MB pull) — best-effort, falls back to the existing archives on failure. `--no-fetch` skips it.
