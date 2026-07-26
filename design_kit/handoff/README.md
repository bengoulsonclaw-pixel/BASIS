# Handoff: BASIS terminal redesign (institutional research dashboard)

## Overview
Redesign of the BASIS futures-desk research dashboard (Streamlit app at basisterminal.com).
The goal was a sharper execution of the existing gold-on-dark identity — institutional
research-terminal feel (Bloomberg / Interactive Brokers / thinkorswim level of density and
typographic discipline) while keeping the XP-derived gold #F5C518 that also appears in client
PDFs. The design also resolves the app's structural split: FICC and Equities are separate
desks with **different module sets**, so navigation is desk-scoped rather than one flat list.

## About the Design Files
The file in this bundle is a **design reference created in HTML** — a prototype showing the
intended look, density and behavior. It is **not production code to copy directly**.
The task is to recreate this design in the target codebase's existing environment
(here: a **Streamlit** app — Python + custom CSS/theme config, `st.columns`, `st.sidebar`,
`st.dataframe`/`st.markdown` tables) using that environment's established patterns.
Where the HTML uses a construct Streamlit cannot express, keep the *visual* result and
reimplement the mechanism.

Streamlit constraints the design deliberately respects:
- Everything is a vertical flow of blocks and columns — no free-form canvas, no overlapping layers.
- Panels are simple bordered blocks; grids are `st.columns` or a CSS grid inside one `st.markdown` block.
- Both themes ship (dark = desk default, light = "brochure" look) via CSS custom properties.

## Fidelity
**High-fidelity.** Final colors, typography, spacing, density and states. Recreate pixel-close
using the app's own styling layer. All values below are exact.

## Design Tokens

CSS custom properties are defined once on `:root` and overridden on `[data-theme="light"]`.
Every component reads tokens — no literal palette colors outside these blocks (exception:
heatmap cell fills, listed separately).

### Dark (default)
| Token | Hex | Use |
|---|---|---|
| --canvas | #0F1216 | page background |
| --surface | #161A20 | panel background, masthead |
| --surface2 | #1C212A | table header row, active nav row, chart plot area |
| --sidebar | #0A0C10 | sidebar background |
| --line | #232935 | panel borders, table header rule, grid gaps |
| --line2 | #333B49 | secondary button border, dividers, empty meter fill |
| --hair | rgba(255,255,255,.05) | table row rules, inner cell dividers |
| --text | #E7EAEE | primary text |
| --dim | #98A1AD | secondary text, table body |
| --faint | #626C7A | micro labels, column headers |
| --gold | #F5C518 | accent: active state, conviction bars, primary action |
| --goldDeep | #D9971C | lower-conviction bars, link hover |
| --goldWash | rgba(245,197,24,.10) | gold chip / active segment background |
| --goldLine | rgba(245,197,24,.34) | gold border |
| --up | #46C58A | positive |
| --down | #EC6A57 | negative |
| --blue | #5B9BF0 | reserved (charts) |

### Light
--canvas #EDF0F3 · --surface #FFFFFF · --surface2 #F4F6F9 · --sidebar #E4E8EC ·
--line #D2D7DE · --line2 #BEC5CE · --hair rgba(0,0,0,.07) · --text #12161B · --dim #57606B ·
--faint #7D8792 · --gold #A87A0C · --goldDeep #8A6208 · --goldWash rgba(245,197,24,.16) ·
--goldLine rgba(168,122,12,.38) · --up #0F7A45 · --down #C0392B · --blue #1F5FA8

> Note: on light, gold is **darkened to #A87A0C for text and borders** (raw #F5C518 fails
> contrast on white). Fills/washes keep the raw gold.

### Density tokens
| Token | comfortable | compact |
|---|---|---|
| --rowpad | 7px | 4.5px |
| --rowfs | 12.5px | 12px |
| --hpad | 22px | 16px |

Table cell padding is `var(--rowpad) 10px` (14px on the first/last column), so density is a
single switch. Expose it as a user preference.

### Typography
- Families: **IBM Plex Sans** (400/450/500/600/700) for prose and labels; **IBM Plex Mono**
  (400/500/600) for every number, code, ticker symbol and micro-label. Both are OFL and
  self-hostable — do not fall back to system UI fonts.
- Base body: 13px / 1.45.
- Wordmark: 700, 20px (masthead) / 18px (sidebar), letter-spacing .10–.12em, silver→gold
  gradient text: `linear-gradient(96deg,#EEF0F3,#C0C5CC 46%,#CBA53C 70%,#F4CC3A)` with
  `background-clip:text`.
- Panel title (h3): 11px, 600, letter-spacing .18em, uppercase, --text.
- Page-bar title (h2): 15px, 600, letter-spacing .02em.
- Panel meta / breadcrumb: Mono 9.5px, letter-spacing .12–.30em, uppercase, --faint or --dim.
- Table column header: Mono 9.5px, 500, letter-spacing .16em, uppercase, --faint, on --surface2.
- Table cell: 12.5px (--rowfs); numeric cells Mono with `font-variant-numeric: tabular-nums`,
  right-aligned. Numbers must never be proportional.
- Sidebar nav item: 12.5px; index number Mono 10px; right-hand tag Mono 9px.
- KPI value: Mono 16px; conviction score Mono 26px 600.
- Body prose: 12.5px / 1.6, --dim, `text-wrap: pretty`.

### Geometry
- **Border radius: 0 everywhere.** No rounded cards, no pills. Sharpness is the whole point.
- Border width 1px. Panels: `background: var(--surface); border: 1px solid var(--line)`.
- Spacing scale: 1, 4, 6, 8, 9, 10, 11, 14, 16, 22px. Gap between stacked panels: 14px.
  Panel header padding: 9px 14px. Page padding: var(--hpad).
- **No shadows at all.** Separation comes from borders and surface steps only.
- Grid gaps in heatmaps are 1px of --line showing through a 1px-padded container (hairline grid).
- Active nav marker: `box-shadow: inset 2px 0 0 var(--gold)` (a left rule, not a fill).
  Active desk segment: `box-shadow: inset 0 -2px 0 var(--gold)` + --goldWash background.

## Layout

```
┌──────────┬─────────────────────────────────────────────────┐
│ sidebar  │ masthead (58px)                                 │
│ 236px    ├─────────────────────────────────────────────────┤
│ fixed    │ ticker rail (auto-fit, min 126px per cell)      │
│          ├─────────────────────────────────────────────────┤
│          │ page bar (title · kicker · counts)              │
│          │ panels, stacked, 14px gap                       │
│          │ footer disclaimer line                          │
└──────────┴─────────────────────────────────────────────────┘
```

### Sidebar (236px, --sidebar, 1px right border) — top to bottom
1. **Logo block** (padding 16px 14px 14px, bottom border): 18px chevron mark
   (gold `M20 20 L58 50 L20 80` + grey `M92 20 L54 50 L92 80`, stroke-width 11, square caps),
   gradient "BASIS" wordmark, then "RESEARCH TERMINAL" Mono 8.5px .30em indented 26px.
2. **Demo-mode badge**: gold-wash box, 5px square dot + "DEMO MODE — SYNTHETIC", Mono 9.5px.
3. **TERMINAL** section: single item "Overview" (index 00, tag ⌘0) — the cross-asset home,
   which sits *above* the desk split.
4. **DESK** section: two-up segmented control **FICC | EQUITIES**, Mono 10px 600 .16em,
   1px --line grid; active segment = gold wash + gold underline. Under it a Mono 9px row
   showing the selected desk's scope and live-signal count ("36 markets" / "11 signals").
5. **<DESK> MODULES**: the module list **swaps with the selected desk**.
   - FICC: Desk Overview, Curve & Spreads, Rates Signals (4), Credit (2), FX Majors (3),
     Carry & Roll, CFTC Positioning, Auction Calendar, Macro Fundamentals.
   - Equities: Desk Overview, Index Board, Sector Breadth, Volatility & Skew (3),
     Dispersion (2), Gamma & Flow, Earnings Calendar, Factor Rotation (3), Single-name Screen.
   - Row = 22px index column (Mono 10px, turns gold when active) / label / right tag
     (live-signal count, Mono 9px). Active row: --surface2 + gold left rule + 500 weight.
     Hover: --surface background, --text label.
6. **CROSS-ASSET**: Correlations, Strategy Builder, Trade Testing, Data Health (tag "OK").
   Indexed C1–C4. These are the only modules that persist across desks.
7. **Footer status** (margin-top auto, top border): three Mono 9.5px rows —
   SIGNALS 21:47:02 ET · FEED ● BBG live 42ms (dot in --up) · BUILD v4.2.1.

### Masthead (58px min-height, --surface, bottom border, padding 0 var(--hpad))
Left: gradient wordmark, 1px×22px divider (--line2), **breadcrumb** Mono 9.5px .24em uppercase,
`white-space:nowrap; overflow:hidden; text-overflow:ellipsis` —
"BASIS · FICC DESK · CURVE & SPREADS", or "ANALYSIS · STRATEGY · INDICATORS" on Overview.
Right: live clock (Mono 10.5px --faint), then three controls, 8px gap, all 30px tall, radius 0:
- **PULL SNAPSHOT** — transparent bg, 1px --goldLine border, --gold label, 11px 600 .10em
  uppercase, nowrap, Mono ⌘P hint at 60% opacity; hover fills --goldWash.
- **RE-RUN SIGNALS** — same but --line2 border, --dim label; hover --text / --dim border.
- **Theme toggle** — 30×30 square, 14px sun icon (stroke 1.8, currentColor); hover gold.

### Ticker rail (--surface2, bottom border)
`grid-template-columns: repeat(auto-fit, minmax(126px, 1fr))`; each cell padding 8px 12px,
`min-width:0`, right + bottom 1px --hair (so it reads correctly when it wraps).
Cell = Mono 9.5px .14em --faint symbol line ("GC · GOLD"), then a wrapping baseline row of
Mono 13px 500 price + Mono 10.5px change in --up/--down.
Contents: GC 2412.60 +1.24% · SI 31.482 +1.58% · CL 79.14 −1.72% · NQ 20418.25 −0.84% ·
ES 5612.50 −0.31% · TY 4.235 +3.2bp · DX 104.62 +0.18% · VX 14.28 +2.90%.

### Page bar
Bottom-bordered row: h2 title + Mono kicker (ellipsised) on the left, Mono count on the right.
- Overview → "Terminal overview" / "Cross-asset · both desks" / "88 MARKETS · 27 LIVE SIGNALS"
- Desk → "FICC desk" / "<active module> · 36 markets" / "36 MARKETS · 11 SIGNALS"

## Screens / Views

### 1. Overview (cross-asset home)
- **Universe coverage** panel: header ("UNIVERSE COVERAGE" + "88 / 88 ACTIVE" in gold), then a
  4-column grid — Fixed income 18/18, Indices 20/20, Commodities 32/32, FX 18/18 — each a
  label/value row over a 3px meter (--line track, gold fill). Replaces the old pill chips.
- **Overnight moves** table (6 rows): Market · Last · Chg % · σ 1M · Z-range. The Z-range cell
  is a 4px centered bar: a 1px --faint center tick with a green bar growing right or a red bar
  growing left, width ∝ |z|/2σ.
- **Cross-asset heatmap**: 4-column, 1px-gap grid of 12 tiles (min-height 56px), each Mono 10px
  symbol over Mono 14px 600 z-score, white text. Footer legend: "−2σ" · 4px gradient bar · "+2σ".
- **Top technical opportunities** (9 columns): Market · Strategy · Signal · Level · Objective ·
  Invalidation · Risk · R:R · Conviction. The market cell carries a 2px inset left rule in
  --up/--down for direction. Signal cell = square 10px Mono badge (LONG green wash / SHORT red
  wash) plus optional gold "NEW" badge. Conviction = flex row of a 4px gold meter + Mono 12px
  600 score, gold ≥70, --goldDeep below.
- **Signal detail** panel: 1.35fr/1fr split. Left = title "Gold — Long", confluence subtitle,
  right-aligned CONVICTION label + Mono 26px gold score, then a 150px bordered chart
  (gold price line 1.75px, dashed grey MA 1.25px, dashed green objective line, dashed red
  invalidation line, three 5% grid lines, corner Mono labels OBJ / INV / "GC1 · DAILY · 6M").
  Right = 2-column KPI grid separated by --hair (Level, Objective, Invalidation, Risk to stop,
  Reward:risk, Horizon; Mono 14px values, up/down colored where directional), then the analyst
  note: Mono 9px "ANALYST NOTE" label + 12.5px/1.6 prose with a 2px --goldLine left rule.

### 2. FICC desk — Overview module
- Two-up: **US Treasury curve** chart (today gold vs 1w dashed grey, 7 tenor dots, Mono axis
  labels 3M→30Y) | **Rates board** table (Tenor · Yield · 1D bp · 1W bp · σ 3M · Pctl 1Y, 6 rows).
- **Spreads & credit** table: Spread · Level · 1D · 1M · σ 1M · Regime · 1Y range.
  Regime is a square badge (STEEPENING green / RANGE neutral --line2 / TIGHT · 8TH PCTL gold);
  1Y range is a 4px percentile meter.
- Two-up: **FX majors** (Pair · Spot · Chg % · 1M IV · σ 1M, 5 rows) | **Sovereign 10Y** 4-column
  z-score heatmap (US, DE, FR, IT, GB, JP, CA, AU) with the same legend footer.

### 3. FICC desk — Curve & Spreads module (fully designed)
- **Control bar** panel: four labelled chip groups on one row — Curve (UST/BUND/GILT/JGB),
  Basis (PAR YIELD/ZERO/FORWARD), Compare (1D/**1W**/1M/3M), Fit (SPLINE/NSS). Chips are
  Mono 10px 600 .14em, 4px 10px padding, 1px border; selected = gold wash + gold border + gold text.
  This is the reusable **module control bar** pattern — every module page should open with one.
- **UST par curve** panel: legend in the header (14px gold rule = TODAY, dashed --dim = 1W AGO,
  timestamp), 232px chart with 4 grid lines, dashed 1w line, gold today line with 3px dots,
  Mono y-axis labels (5.40/4.90/4.40/3.90) and x labels 3M…30Y.
- Two-up: **Spread & fly monitor** (Structure · Level · 1D · 1M · σ 3M · 1Y pctl meter;
  2s10s, 5s30s, 3M10Y, 2s5s10s fly, 5s10s30s fly, 10Y swap spread) |
  **Curve read** panel (4 KPI cells: Slope regime "Steepening" in --up, Level shift 1W,
  Inversion, Richest point in gold; then the gold-ruled analyst note).

### 4. Equities desk — Overview module
- Two-up: **Index board** (Contract · Last · Chg % · ATR % · Vs 50D · 20-day sparkline SVG,
  1.5px stroke colored --up/--down/--dim; ES, NQ, RTY, FESX, NKD) |
  **Volatility structure** (4 KPI cells: VIX spot, Term slope, 25Δ skew, IV−RV 20D in gold;
  then M1–M4 futures curve rows and a green "Contango · carry favours short vol" line).
- **Sector breadth**: `auto-fit minmax(118px,1fr)` heatmap of all 11 GICS sectors, footer line
  "Advancers 4 / 11 · rotation into cyclicals ex-tech" + inline gradient legend.

### 5. Equities desk — Volatility & Skew module (fully designed)
- **Control bar**: Underlying (ES/NQ/RTY/FESX), Tenor (1W/**1M**/3M/6M), Measure (IV/IV−RV/SKEW),
  Lookback (1M/**1Y**/5Y).
- Two-up: **Term structure** chart (spot→M4, gold today vs dashed prior) |
  **Skew · 1M** chart (IV by delta 10ΔP→10ΔC, vertical ATM guide, ring marker at ATM).
- **Implied vol surface**: `78px + repeat(5, 1fr)` grid, 1px --line gaps. Header row and row
  labels sit on --surface2; cells are Mono 12.5px on heat fills, tenors 1W/1M/3M/6M ×
  10Δ put / 25Δ put / ATM / 25Δ call / 10Δ call. Footer: read-out line + 10→24 gradient legend.
- **Volatility signals** table: Structure · Signal (SELL SKEW / LONG ROLL / LONG DISPERSION
  badges) · Entry · Carry/day · Max loss · Conviction meter.

### Heatmap fill scale (shared by every heatmap and the vol surface)
+2σ #0F5C36 · +1.5σ #17734A · +0.8σ #245C43 · ~0 #2C3742 · −0.4σ #4A3A38 ·
−1.0σ #6B3630 · −1.5σ #7A2D26 · −1.8σ+ #8E2F26. Text white (≥.82 opacity for the symbol line).
These are intentionally *desaturated* versus the old palette so a wall of tiles stays readable.
Legend gradient: `linear-gradient(90deg,#8E2F26,#4A3A38,#2C3742,#245C43,#0F5C36)`.

## Interactions & Behavior
- **Desk switch** (segmented control): sets `desk`, sets `page = desk`, resets `mod` to
  "overview". The module list, page bar, breadcrumb and body all follow.
- **Module click**: sets `mod` and forces `page` to that desk. Active row gets the gold left
  rule; breadcrumb and page-bar kicker update. Cross-asset modules set `mod` without changing desk.
- **Overview** returns to the cross-asset home but *remembers* the last desk, so the module list
  below the switcher stays populated.
- **Theme toggle** flips `data-theme` on the root; every color is a token so nothing else changes.
- **Density** ("comfortable" | "compact") flips `data-density` and only changes --rowpad/--rowfs/--hpad.
- **Hover**: nav rows → --surface bg + --text; table rows → --surface2; buttons as described.
  No transitions are specified — instant state change reads as more "terminal"; if you add any,
  cap at 80ms.
- **Clock** in the masthead ticks every second (HH:MM:SS ET).
- **Overflow discipline** (important, these were real bugs): the ticker rail auto-fits and its
  cells have `min-width:0` with a wrapping value row; every wide table sits in a panel with
  `overflow-x:auto` so it scrolls *inside* its border instead of pushing the page wide;
  button labels and the breadcrumb are `nowrap` (breadcrumb ellipsises).

## State Management
Four keys drive everything (in Streamlit: `st.session_state`):
- `page`: "home" | "ficc" | "equities" — which body renders.
- `desk`: "ficc" | "equities" — which module list the sidebar shows (survives going Home).
- `mod`: module id within the desk, default "overview" — drives the active nav row, the
  breadcrumb, the page-bar kicker, and which module body renders.
- `theme`: "dark" | "light"; plus `density`: "comfortable" | "compact".

Module→body mapping currently implemented: FICC `overview` and `curve`; Equities `overview`
and `vol`. Every other module id falls back to its desk overview — those bodies are still to be
designed; wire the routing so adding one is a single branch.

Data: all figures in the prototype are synthetic. Real wiring is a Bloomberg snapshot pull
(the "PULL SNAPSHOT" action) plus the signal engine re-run ("RE-RUN SIGNALS"), both of which
should show a progress state and update the "SIGNALS <time>" line in the sidebar footer.

## Assets
- **Fonts**: IBM Plex Sans + IBM Plex Mono (SIL OFL). Loaded from Google Fonts in the prototype;
  self-host in production so the PDF reports and the app match.
- **Logo**: inline SVG double-chevron (gold + grey), no raster asset. Wordmark is live text with
  a gradient clip — reuse the same gradient in the PDF templates.
- No images, no icon font. The only icon is the inline sun/moon SVG on the theme toggle.

## Files
- `BASIS Terminal.dc.html` — the full prototype: sidebar, masthead, ticker, Overview,
  FICC overview + Curve & Spreads, Equities overview + Volatility & Skew, both themes,
  both densities. Open it directly in a browser; click the desk segments and module rows.

## Notes for implementation
- Keep the gold strictly as an accent: active state, meters, primary action, alerts. If gold is
  filling large areas, something is wrong.
- Every number must be Mono + tabular-nums + right-aligned. This single rule does most of the
  "institutional" work.
- Radius 0 and no shadows are load-bearing — do not soften them back toward the old card look.
