"""Per-strategy trigger specs — the single source of truth for each model's
trigger threshold, its plain-English trigger statement, the maths explainer, and
how to re-flag rows at a chosen threshold.

The dashboard (app.py) uses this to render each page's trigger control, the
"how this is calculated" paragraph, and the live re-flagging of the table, and
to pass the trigger statement into the report generators. The threshold only
affects which rows are FLAGGED — the underlying z-score / return is unchanged —
so re-flagging is instant and needs no data re-pull (mirrors reportkit.reflag).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Each spec:
#   default/min/max/step/label : the per-page number input (None default = no trigger)
#   metric_label               : what the headline number is
#   hi / lo                    : (signal, direction) where metric >= +threshold / <= -threshold
#   trigger(t)                 : one-line trigger statement (shown on the page + report)
#   math                       : markdown explaining the formula
SPECS = {
    "Mean Reversion": {
        "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1, "label": "Trigger — flag when |z| ≥",
        "metric_label": "z-score",
        "hi": ("Short spread", -1), "lo": ("Long spread", 1),
        "trigger": lambda t: f"|z-score| ≥ {t:g} — the spread is stretched from its 90-day mean.",
        "math": (
            "**Mean reversion — z-score of the spread.**\n\n"
            "For each pair the spread is `A − B` for same-unit legs (e.g. WTI − Brent) or the "
            "ratio `A ÷ B` (e.g. Gold / Silver). Over a rolling **90-business-day** window we take "
            "its mean (μ) and standard deviation (σ):\n\n"
            "`z = (today's spread − μ) ÷ σ`\n\n"
            "Flag when **|z| ≥ trigger** — `z ≥ +t` → spread **rich** (sell A / buy B); "
            "`z ≤ −t` → spread **cheap** (buy A / sell B). The half-life (≈ days to revert) is an "
            "Ornstein–Uhlenbeck fit on `Δspread = β · spread(−1)`, half-life `= −ln 2 ÷ β`.\n\n"
            "**Difference (`A − B`) vs ratio (`A ÷ B`).** Chosen per pair by convention. Use the "
            "**difference** when the legs share units, sit at a similar scale and trade ~1:1, so the "
            "market already quotes them as a differential (WTI − Brent in $/bbl, Bund − OAT, "
            "Chicago − KC wheat). Use the **ratio** when the legs are on different scales or move "
            "proportionally, where a raw difference would just track the bigger leg (Gold / Silver "
            "≈ 84×, gold / copper, soybean oil / meal). The underlying test is which form is "
            "*stationary* (mean-reverting around a stable level): a roughly constant **absolute** gap "
            "→ difference; a roughly constant **multiple** → ratio."
        ),
    },
    "Trend": {
        "default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5, "label": "Trigger — flag when |3m return| ≥ (%)",
        "metric_label": "3m return %",
        "hi": ("Long", 1), "lo": ("Short", -1),
        "trigger": lambda t: (f"3-month return ≥ +{t:g}% → Long, ≤ −{t:g}% → Short "
                              "(direction = sign of the 3-month return; set 0 to show every market)."),
        "math": (
            "**Trend — time-series momentum.**\n\n"
            "Direction is the **sign of the trailing 3-month (63-trading-day) return**, "
            "`r = today's price ÷ price 63 days ago − 1`. It is sanity-checked against a 20-day vs "
            "100-day moving-average crossover; if the MA crossover disagrees with the return sign "
            "the row is tagged **mixed**. Flag when **|r| ≥ trigger %** — set the trigger to 0 to "
            "rank the whole book by trend sign, or raise it to keep only strong trends."
        ),
    },
    "MA Crossover": {
        # The signal is a binary crossover AND a 3-month-return confirmation — not a
        # tunable z-score — so there's no generic ±threshold input here, only the
        # explainer. (default=None mirrors COT / AG Fundamentals.)
        "default": None,
        "metric_label": "MA gap %",
        "math": (
            "**Moving-average crossover — golden / death cross, EMA- and momentum-confirmed.**\n\n"
            "A four-MA ribbon, of which three drive the signal:\n\n"
            "- **50-SMA vs 200-SMA** — the *golden cross* (50 above 200, bullish) or *death "
            "cross* (50 below 200, bearish) sets the primary direction.\n"
            "- **15-day EMA** — must sit on the **trend side of the 50**: above it to confirm a "
            "golden cross, below it to confirm a death cross (short-term momentum still leading).\n"
            "- **3-month (63-day) return** — `r = price ÷ price 63 days ago − 1`, must agree in sign.\n\n"
            "A golden cross is taken **Long only if the 15-EMA > 50-SMA and `r > 0`**; a death "
            "cross is taken **Short only if the 15-EMA < 50-SMA and `r < 0`**. If either "
            "confirmation disagrees the trade is skipped (**“—” unconfirmed**). The **100-day "
            "SMA** is drawn on the chart for context (the intermediate trend) but does not gate "
            "the signal, and the **50-day is coloured by its slope** (green rising / red falling).\n\n"
            "Headline **MA gap %** `= (50-SMA ÷ 200-SMA − 1) × 100` is how far the 50 sits above "
            "(＋) or below (−) the 200 — the conviction behind the cross; markets are ranked "
            "confirmed-first, then by |MA gap|. Runs on the same liquid trend universe as the "
            "Trend page (which instead leads on the 3-month return with a 20/100 MA as context). "
            "**Bond / rate futures are excluded** — MA crossovers lost money on every pair tested "
            "in a ~10-year backtest, so the page doesn't signal on them."
        ),
    },
    "MA Swing": {
        "default": None,
        "metric_label": "MA gap %",
        "math": (
            "**MA crossover — swing variant (20/50).**\n\n"
            "Same rule as the position-trade MA Crossover, on the **shorter periods the "
            "literature pairs with swing trading**:\n\n"
            "- **20-SMA vs 50-SMA** — the golden cross (20 above 50) / death cross (20 below 50) "
            "sets the direction.\n"
            "- **9-day EMA** — must sit on the trend side of the 20 (above to confirm a golden "
            "cross, below for a death cross).\n"
            "- **1-month (21-day) return** — must agree in sign.\n\n"
            "Long needs `20 > 50`, `9-EMA > 20` and a positive 1-month return; Short the mirror; "
            "otherwise **“—” unconfirmed**. The **200-day SMA** is drawn on the chart as the "
            "big-picture trend (handy for seeing whether a swing is with or against the major "
            "trend) but does not gate the signal. Headline **MA gap %** `= (20 ÷ 50 − 1) × 100`.\n\n"
            "Faster than the 50/200 page — more signals, but more prone to whipsaws in choppy "
            "markets (the classic crossover trade-off). Same liquid trend universe, with "
            "**bond / rate futures excluded** (crossovers were a backtested money-loser on rates)."
        ),
    },
    "Flag Breakout": {
        "default": 70.0, "min": 50.0, "max": 100.0, "step": 5.0,
        "label": "Trigger — flag when breakout readiness ≥",
        "metric_label": "breakout readiness",
        "hi": ("Bull breakout setup", 1), "lo": ("Bear breakout setup", -1),
        "trigger": lambda t: (f"breakout readiness ≥ {t:g} (bull flag, breaks up) or ≤ −{t:g} "
                              "(bear flag, breaks down) — price is coiled within reach of the "
                              "flag's trendline. Lower the trigger to catch them earlier."),
        "math": (
            "**Flag breakout — a pole, a consolidation, and the trendline break.**\n\n"
            "A *flag* is a continuation pattern: a sharp **flagpole**, then a brief low-slope "
            "**consolidation** that drifts against the pole between two roughly parallel "
            "trendlines, before price **breaks back out** in the pole's direction.\n\n"
            "Per product, on the close series:\n\n"
            "- **Flagpole** — a fast, strong move: a run (4–20 days) whose size clears both an "
            "absolute floor (4%) and ~2.5σ of the product's own daily moves over its length.\n"
            "- **Flag** — the 5–15 days since the pole's tip (≤ ~3 weeks; longer starts to read as "
            "a reversal), requiring (a) ≤ 50% of the pole retraced, (b) a channel (the trendline ±2σ "
            "of its residuals) much tighter than the pole, and (c) a gentle slope, leaning against or "
            "flat versus the pole.\n"
            "- **Flag vs pennant** — a consolidation whose range narrows into the apex is labelled a "
            "**pennant** (converging); a roughly parallel one stays a **flag**. Both are continuation "
            "patterns traded identically (same pole, breakout and measured move).\n"
            "- **Breakout readiness** — where today's close sits across that channel toward the "
            "breakout edge, 0–100: `100 × (close − far wall) ÷ channel width`. **100 = on the "
            "breakout line**, above 100 = already through.\n"
            "- **Target / stop / R:R** — the measured move: the flagpole height projected from the "
            "breakout (`target = breakout ± pole height`), a **stop** beyond the far edge of the flag, "
            "and the reward:risk on a break-level entry.\n\n"
            "It is stored **signed by pattern** — `+readiness` for a bull flag, `−readiness` for a "
            "bear flag — so the trigger reads symmetrically: `metric ≥ +t` → **bull breakout "
            "setup**, `metric ≤ −t` → **bear breakout setup**; markets are ranked nearest-breakout "
            "first.\n\n"
            "_Close-based_ (pole, channel, breakout, target and stop are all from settlement closes). "
            "**Volume confirmation** — the flag should dry up on volume and surge on the break — is "
            "shown where a volume series is available (`FUT_AGGTE_VOL`), as a flag-vs-pole dry-up "
            "ratio; it annotates rather than gates the signal."
        ),
    },
    "Volatility": {
        "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1, "label": "Trigger — flag when |z| ≥",
        "metric_label": "spread z (1y)",
        "hi": ("Rich — sell vol", -1), "lo": ("Cheap — buy vol", 1),
        "trigger": lambda t: f"|z-score of (implied − realized)| ≥ {t:g} vs the trailing year.",
        "math": (
            "**Volatility — implied vs realized (the vol-risk premium).**\n\n"
            "Per market: **realized** vol = annualised 21-day close-to-close, "
            "`std(daily log returns) × √252`; **implied** = the constant-maturity 1-month ATM "
            "option vol. The signal is the **z-score of the spread (implied − realized)** over the "
            "trailing **252 days**:\n\n"
            "`z = ((IV − RV) today − 1-year mean) ÷ 1-year std`\n\n"
            "`z ≥ +t` → implied **rich** vs delivered (sell vol); `z ≤ −t` → **cheap** (buy vol). "
            "STIRs are excluded (their option vol is normal/bp, not price vol)."
        ),
    },
    "Skew Volatility": {
        "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1, "label": "Trigger — flag when |z| ≥",
        "metric_label": "skew z (1y)",
        "hi": ("Rich — sell skew", -1), "lo": ("Cheap — buy skew", 1),
        "trigger": lambda t: f"|z-score of the 25Δ skew| ≥ {t:g} vs the trailing year.",
        "math": (
            "**Skew — the 25-delta risk reversal.**\n\n"
            "Skew = **(25Δ put − 25Δ call) ÷ ATM** (listed markets interpolate true 25Δ wings off "
            "the moneyness surface; FX uses the native OTC 25Δ risk reversal), z-scored over the "
            "trailing **252 days**:\n\n"
            "`z = (today's skew − 1-year mean) ÷ 1-year std`\n\n"
            "`z ≥ +t` → puts **rich** vs calls (sell skew); `z ≤ −t` → puts **cheap** (buy skew). "
            "STIRs are excluded."
        ),
    },
    "Vol Term Structure": {
        "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1, "label": "Trigger — flag when |z| ≥",
        "metric_label": "slope z (1y)",
        "hi": ("Steep — front cheap", 1), "lo": ("Inverted — front rich", -1),
        "trigger": lambda t: (f"slope z-score ≥ +{t:g} (steep / front cheap) or ≤ −{t:g} "
                              "(inverted / front rich) vs the trailing year."),
        "math": (
            "**Vol term structure — the calendar slope.**\n\n"
            "Read ATM implied vol at **1M / 3M / 6M / 12M**; the signal is the **3M − 1M slope**, "
            "z-scored over the trailing **252 days**:\n\n"
            "`z = (today's slope − 1-year mean) ÷ 1-year std`\n\n"
            "`z ≥ +t` → steep contango, front **cheap** (buy 1M / sell 3M); `z ≤ −t` → inverted, "
            "front **rich** (sell 1M / buy 3M). Each tenor also shows its implied-vs-realized premium."
        ),
    },
    "COT Reports": {
        # COT has its own controls on the page (a 0–100 cutoff, not a symmetric ±z),
        # so the generic trigger input is disabled here — only the maths explainer shows.
        "default": None,
        "metric_label": "COT Index (0–100)",
        "math": (
            "**COT positioning — managed money, via the CFTC Commitments of Traders.**\n\n"
            "Weekly speculative net positioning from the free CFTC report (Tuesday as-of, "
            "published Friday; Futures-and-Options combined). **Commodities** use the "
            "Disaggregated report's **Managed Money** bucket; **financials** (equity indices, "
            "rates, FX) use the Traders-in-Financial-Futures **Leveraged Funds** bucket "
            "(managed money has no analogue there). `Net = long − short` contracts.\n\n"
            "The headline is the **COT Index** — a 0–100 stochastic of the net position over "
            "the trailing **3 years (156 weeks)**:\n\n"
            "`COT Index = 100 × (net − 3y-min) ÷ (3y-max − 3y-min)`\n\n"
            "**0** = most net-short in 3 years, **100** = most net-long. Flagged **crowded long** "
            "at ≥ cutoff (default 80) and **crowded short** at ≤ 100 − cutoff (20). Positioning "
            "extremes are frequently read as contrarian, but this page is descriptive — not a "
            "recommendation. ICE-Europe (Brent, Gasoil, Robusta), Eurex and other non-US markets "
            "have no CFTC COT and are omitted."
        ),
    },
    "AG Fundamentals": {
        # Like COT, AG has its own page controls (a USDA calendar + stocks), so the
        # generic symmetric ±z trigger is disabled — only the maths explainer shows.
        "default": None,
        "metric_label": "days to USDA report",
        "math": (
            "**AG fundamentals — USDA supply/demand + the report calendar.**\n\n"
            "Two free-data reads (speculative positioning lives on the COT page):\n\n"
            "**1. Report event risk.** The verified 2026 USDA release calendar (WASDE, Crop "
            "Production, Grain Stocks, Prospective Plantings, Acreage, Cattle on Feed, Hogs & Pigs) "
            "is mapped onto each ag contract; a market is flagged when one of its reports lands "
            "within the next **14 days** — `metric` = days to that report.\n\n"
            "**2. Stocks tightness.** With the free NASS QuickStats key set (`NASS_API_KEY`), "
            "national grain stocks are pulled per crop and ranked against the crop's own history: a "
            "percentile **≤ 25** flags **tight** (bullish), **≥ 75** flags **ample** (bearish)."
        ),
    },
    "Support & Resistance": {
        "default": 50.0, "min": 0.0, "max": 100.0, "step": 5.0,
        "label": "Trigger — flag when level proximity ≥",
        "metric_label": "level proximity",
        "hi": ("Long (buy the dip)", 1), "lo": ("Short (sell the rally)", -1),
        "trigger": lambda t: (f"price within {t:g}/100 proximity of a tested level — at/near support "
                              "→ buy-dip (long), at/near resistance → sell-rally (short)."),
        "math": (
            "**Support & resistance — tested horizontal levels (price action).**\n\n"
            "Levels are built from **swing pivots** on the close series — a pivot high/low is a bar that "
            "is the strict max/min within ±4 bars. Pivots at a similar price (within ~1.5% / 1 ATR) are "
            "**clustered** into a level; its strength = **touches** (pivots in the cluster, ≥ 2 to count). "
            "Broken levels flip role (old resistance → support and vice-versa).\n\n"
            "The headline **level proximity** 0–100 = `100 × (1 − distance ÷ near-band)` (near-band ≈ 2% "
            "of price), signed by side: **+** when price sits on **support** (buy the dip), **−** on "
            "**resistance** (sell the rally). Flag when **|proximity| ≥ trigger**. Close-only; ATR is a "
            "close-to-close proxy used only to size the clustering tolerance."
        ),
    },
    "Fibonacci Retracement": {
        "default": 60.0, "min": 0.0, "max": 100.0, "step": 5.0,
        "label": "Trigger — flag when Fib proximity ≥",
        "metric_label": "Fib proximity",
        "hi": ("Long (buy the dip)", 1), "lo": ("Short (sell the rally)", -1),
        "trigger": lambda t: (f"price within {t:g}/100 of a key Fibonacci retracement of the dominant "
                              "swing — on an up-leg → buy-dip (long), on a down-leg → sell-rally (short)."),
        "math": (
            "**Fibonacci retracement — reaction at a key level of the dominant swing.**\n\n"
            "The dominant swing is the **highest high and lowest low** over the lookback (~180 sessions); "
            "whichever is the more recent sets the leg (high-after-low = up-leg, else down-leg). The "
            "retracement grid is drawn across it — **23.6 / 38.2 / 50 / 61.8 / 78.6 %** — and the **golden "
            "zone** (38.2–61.8 %, the 61.8 % especially) is where pullbacks most often hold.\n\n"
            "Headline **Fib proximity** 0–100 = how close today's close sits to the nearest *key* level "
            "(38.2 / 50 / 61.8 %), signed **+** on an up-leg pullback (buy the dip = support) and **−** on a "
            "down-leg bounce (sell the rally = resistance). Flag when **|proximity| ≥ trigger**. Each setup "
            "carries a **target** (the prior swing extreme — retrace and resume), a **stop** beyond the "
            "**78.6 %** level (a deeper close breaks it) and the reward:risk. Close-only; the 50 % level is "
            "included by convention (not a true Fibonacci ratio)."
        ),
    },
    "Breakout & Retest": {
        "default": 60.0, "min": 0.0, "max": 100.0, "step": 5.0,
        "label": "Trigger — flag when retest proximity ≥",
        "metric_label": "retest proximity",
        "hi": ("Long — retest", 1), "lo": ("Short — retest", -1),
        "trigger": lambda t: (f"price within {t:g}/100 of a level it broke and is now retesting — "
                              "upside break retested from above → long, downside from below → short."),
        "math": (
            "**Breakout & retest — the role-reversal entry.**\n\n"
            "A prior swing level (pivot high = resistance, pivot low = support) gives way on a **decisive "
            "close** beyond it (> 0.3 ATR) within the last ~20 days, ideally on a **volume surge** "
            "(breakout-bar volume ≥ 1.3× its 20-day average — a gate when volume is available). Price then "
            "**pulls back to retest** the broken level, which has flipped role (broken resistance → new "
            "support; broken support → new resistance).\n\n"
            "**Retest proximity** 0–100 = how close today's close sits to that flipped level (100 = on it), "
            "signed **+** for a long (upside-break) retest, **−** for a short (downside) one. Flag when "
            "**|proximity| ≥ trigger**. A conditional pattern — only active retests are listed. "
            "Close-based, with a close-to-close ATR proxy."
        ),
    },
    "Momentum (RSI/MACD)": {
        "default": 40.0, "min": 0.0, "max": 100.0, "step": 5.0,
        "label": "Trigger — flag when |momentum score| ≥",
        "metric_label": "momentum score",
        "hi": ("Bullish momentum", 1), "lo": ("Bearish momentum", -1),
        "trigger": lambda t: (f"|momentum score| ≥ {t:g} — a bullish (long) or bearish (short) setup from "
                              "a fresh MACD cross or (the headline) a live RSI divergence."),
        "math": (
            "**Momentum — RSI + MACD, divergence-led.**\n\n"
            "- **RSI** — Wilder's 14-period; > 70 overbought, < 30 oversold.\n"
            "- **MACD** — 12/26 EMA difference, 9-EMA signal, histogram; the line crossing **above** the "
            "signal is bullish, **below** bearish (a cross within ~3 bars is *fresh*).\n"
            "- **RSI divergence** (the headline) — price **higher-high while RSI lower-high** is bearish; "
            "price **lower-low while RSI higher-low** is bullish, read off the two most recent swings. It "
            "counts only while **live**: the confirming swing must be within **10 bars** and the RSI gap "
            "at least **3 points**, so a divergence that has already resolved into the rally it called — "
            "or one resting on a 1–2 point gap that flips with the choice of swing bars — is not "
            "re-reported as a fresh setup.\n\n"
            "Direction is set by an **event** — a live divergence or a fresh MACD cross — never by an RSI "
            "extreme alone (*“RSI > 70, therefore sell”* is the classic false signal in a trend). "
            "Extension instead acts as a **veto**, blocking a signal that points further into the move "
            "already made; a bullish divergence arriving into overbought RSI therefore reads **flat**, "
            "not bullish. The **momentum score** = `2 · max(0, room)` — where *room* is how far RSI sits "
            "the **favourable** side of 50 for that direction, so an extended market scores nothing for "
            "being extended — plus a divergence bonus and a fresh-cross bonus where each agrees with the "
            "direction, clipped to ±100. Flag when **|score| ≥ trigger**. Close-only."
        ),
    },
    "Bollinger Squeeze": {
        # Rich labels (a break vs a squeeze-watch, both signed) are set in the strategy, so the
        # generic symmetric ±relabel is disabled here — only the maths explainer shows.
        "default": None,
        "metric_label": "squeeze (0-100)",
        "math": (
            "**Bollinger squeeze — volatility compression before a move.**\n\n"
            "Bollinger Bands wrap the **20-day SMA** of the close in a **±2σ** envelope (sample std). "
            "**Bandwidth** = (upper − lower) ÷ mid measures how wide the bands are. A **squeeze** is "
            "bandwidth in a **low percentile** of its own trailing year (≤ 20th by default): volatility "
            "has coiled tight and tends to release in a sharp directional move.\n\n"
            "The headline **squeeze intensity** = `100 − bandwidth percentile` (100 = tightest of the "
            "year), signed by the lean — **+** for an upside break / upper-half coil, **−** for a "
            "downside one. A close **outside** the band (%B > 1 or < 0), especially fresh out of a "
            "squeeze, is the **breakout**. Close-only."
        ),
    },
    "Elliott Wave": {
        # Rich phase labels ("Wave 3 underway ▲", "Five waves complete ▼") are set in the
        # strategy — a generic ±relabel would lose the phase — so no symmetric trigger input;
        # the flag bar (wave fit ≥ 55) lives in the strategy module.
        "default": None,
        "metric_label": "wave fit (0-100)",
        "math": (
            "**Elliott Wave — a rules-based impulse count (the deterministic version).**\n\n"
            "Classic Elliott counting is subjective; this page removes the judgement calls. Swing "
            "pivots come from an **adaptive ZigZag** (the reversal threshold scales with each "
            "market's own volatility — ≈ 2.2× its weekly σ, floored at 2.5% and capped at 12%), and "
            "the most recent pivots are tested against the **three hard impulse rules**:\n\n"
            "1. **Wave 2** never retraces past the start of wave 1;\n"
            "2. **Wave 3** pushes beyond wave 1's top and is never the shortest of waves 1/3/5;\n"
            "3. **Wave 4** never overlaps wave 1's price territory.\n\n"
            "A market is flagged at one of three junctures: a completed **wave-2 pullback → wave 3 "
            "underway** (with the impulse — the strongest setup), a completed **wave-4 pullback → "
            "wave 5 underway** (with the impulse, maturing), or **five waves complete → corrective "
            "phase likely** (against the impulse).\n\n"
            "The headline **wave fit** 0–100 scores how *textbook* the count is: wave-2 retrace "
            "peaking at **61.8%** (accepted 23.6–88.6%), wave-4 at **38.2%**, wave-3 extension "
            "toward **1.618× wave 1**, plus an **alternation** bonus when waves 2 and 4 differ in "
            "depth. Signed by direction (+ constructive / − cautious); flagged when **fit ≥ 55**. "
            "Each juncture carries the classic projection as an objective — wave 3 ≈ **1.618× wave 1** "
            "off the wave-2 low, wave 5 ≈ **wave 1** off the wave-4 low, a completed impulse "
            "correcting toward the **prior wave-4** extreme — with the invalidation at the last "
            "counted pivot. Close-only; fixed income is counted on **yields**."
        ),
    },
    "Ichimoku Cloud": {
        # Rich event labels ("Cloud breakout up ▲", "Bullish TK cross ▲") are set in the
        # strategy; flag gate (a fresh event, score ≥ 58) lives in the module.
        "default": None,
        "metric_label": "Ichimoku score (0-100)",
        "math": (
            "**Ichimoku Kinko Hyo — the cloud, and confluence.**\n\n"
            "Ichimoku frames a market with five lines (standard **9 / 26 / 52 / 26**); this suite "
            "is settlement-based, so the n-period highs/lows use the **rolling max/min of closes** "
            "(a standard close-based Ichimoku):\n\n"
            "- **Tenkan-sen** (conversion) = (9-high + 9-low) ÷ 2\n"
            "- **Kijun-sen** (base) = (26-high + 26-low) ÷ 2\n"
            "- **Senkou Span A** = (Tenkan + Kijun) ÷ 2, plotted **26 ahead**  ┐ the **cloud** "
            "(Kumo)\n"
            "- **Senkou Span B** = (52-high + 52-low) ÷ 2, plotted **26 ahead**  ┘\n"
            "- **Chikou** (lagging) = the close, plotted **26 behind**.\n\n"
            "Ichimoku's first rule is **no trade inside the cloud**, so the signal keys off the "
            "price's position relative to it — **above → constructive, below → cautious** — scored "
            "by how many of the three other elements confirm: the **Tenkan/Kijun** cross, the "
            "**future cloud** colour (Span A vs B, 26 ahead), and the **lagging span** vs price 26 "
            "back. `score = 40 + 16 × (# confirming, 0–3) + a clear-of-cloud bonus`.\n\n"
            "Only a **fresh event** is flagged (the actionable moment — a persistent trend above "
            "the cloud is already caught by Trend/MA): a **cloud breakout** (price leaving the "
            "cloud) or a **Tenkan/Kijun cross** in the last ~6 sessions, in the agreeing direction. "
            "Signed by direction; flagged at score ≥ 58. The chart draws the cloud (with its "
            "26-session forward projection), Tenkan, Kijun and the lagging span. Fixed income is "
            "read on **yields** (above the cloud = yields higher = the future lower). Close-only."
        ),
    },
    "On-Balance Volume": {
        # Rich condition labels ("Breakout confirmed by volume ▲", "Hidden distribution ▼")
        # are set in the strategy; a generic ±relabel would lose them. Flag bar (score ≥ 55)
        # lives in the module.
        "default": None,
        "metric_label": "OBV score (0-100)",
        "math": (
            "**On-Balance Volume — is volume behind the move?**\n\n"
            "OBV (Granville) runs a cumulative total of volume **signed by the day's direction** "
            "(up close +volume, down close −volume; volume = `FUT_AGGTE_VOL`, aggregated across all "
            "listed contracts). Price says *where* the market went; OBV says whether **participation "
            "went with it**. Three reads, in priority order:\n\n"
            "1. **OBV divergence** (the trend-failure warning) — price makes a **higher high while "
            "OBV makes a lower high** (bearish: the rally is thinning) or a **lower low while OBV "
            "makes a higher low** (bullish), read off the two most recent ±4-bar swing pivots.\n"
            "2. **Breakout (un)confirmation** — price at/through a fresh **55-day** extreme with OBV "
            "at its own extreme → **volume-backed breakout**; with OBV well short of its range top "
            "→ **breakout lacks volume** (suspect).\n"
            "3. **Hidden accumulation / distribution** — price flat over 20 days (within ¾ of its "
            "own 20-day σ) while OBV climbs or slides hard (|z| ≥ 1.2 of its 20-day changes) — "
            "someone is quietly working orders.\n\n"
            "Headline **OBV score** 0–100 (signed by direction), flagged at **≥ 55**.\n\n"
            "**FX futures are excluded**: most FX turnover is OTC (swaps/forwards), so CME FX "
            "futures volume is a small venue-specific slice — OBV there would read roll cycles and "
            "listed-market quirks, not the market's true flow. Rates stay in: Treasury/STIR futures "
            "ARE the volume centre of their markets. Fixed income runs on **yields** (constructive "
            "= yields higher on volume = sell the bond). Close-only; markets with no volume series "
            "are skipped."
        ),
    },
    "Money Flow Index": {
        # Rich condition labels ("Overbought on real flow ▼", "Early distribution ▼") are set
        # in the strategy; flag bar (score ≥ 55) lives in the module.
        "default": None,
        "metric_label": "MFI score (0-100)",
        "math": (
            "**Money Flow Index — RSI, but volume-weighted.**\n\n"
            "MFI rebuilds RSI on **money flow** (price × volume) instead of price alone, over "
            "**14 sessions**: `MFI = 100 × up-flow ÷ (up-flow + down-flow)`. This suite is "
            "settlement-based, so the classic typical-price (H+L+C)/3 is replaced by the **close** "
            "— flow = close × volume, signed by the day's direction. It answers: *is the momentum "
            "carried by real volume?* Four reads, in priority order:\n\n"
            "1. **Volume-confirmed extreme** — MFI **≥ 80** = overbought on real flow "
            "(distribution risk, cautious) / **≤ 20** = oversold (accumulation zone, "
            "constructive). When RSI agrees (> 70 / < 30) the read **validates the RSI signal** "
            "(score bonus) — momentum extremes carried by volume are the ones that matter.\n"
            "2. **RSI unconfirmed by flow** — RSI stretched but MFI nowhere near its extreme: "
            "price momentum **without volume behind it**, an early warning the move may not carry.\n"
            "3. **Early distribution / accumulation** — price still trending (|20d| ≥ 2%) while "
            "MFI sits on the wrong side of 50 **and falling/rising** — flow leaking against the "
            "trend before price shows it.\n"
            "4. **Flow shift** — MFI crossing **50** on a steep, agreeing 10-day slope (≥ 15): "
            "momentum changing hands with volume support.\n\n"
            "Headline **MFI score** 0–100 (signed), flagged at **≥ 55**. **FX futures are "
            "excluded** (FX volume is OTC — listed futures flow misleads; same reasoning as OBV); "
            "rates stay in (futures are the real volume venue). Fixed income runs on **yields**. "
            "Markets with no volume series are skipped."
        ),
    },
    "Carry": {
        "default": None,
        "math": (
            "**Carry / roll yield — coming soon.**\n\n"
            "Carry needs at least two points on each futures curve (front vs deferred) to measure "
            "annualised roll yield. Wire in the Bloomberg strip (e.g. CL1/CL2…, or the generic "
            "2nd/3rd contract) and this page will rank markets by roll."
        ),
    },
}


# ── Saved per-strategy trigger defaults (data/trigger_defaults.json) ─────────
# Each strategy page seeds its trigger control from here (so the user's chosen
# trigger loads on every launch); set via the page's "Set default" button.
TRIGGER_DEFAULTS = Path(__file__).resolve().parents[1] / "data" / "trigger_defaults.json"


def load_trigger_defaults() -> dict:
    """{strategy: trigger value} saved by the page "Set default" buttons; {} if none."""
    try:
        return json.loads(TRIGGER_DEFAULTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def trigger_default(strategy: str, fallback):
    """The saved default trigger for a strategy, else the spec/page fallback."""
    try:
        v = load_trigger_defaults().get(strategy)
        return float(fallback) if v is None else float(v)
    except Exception:
        return float(fallback)


def save_trigger_default(strategy: str, value) -> None:
    """Persist `value` as the startup default trigger for `strategy`."""
    d = load_trigger_defaults()
    d[strategy] = value
    TRIGGER_DEFAULTS.parent.mkdir(parents=True, exist_ok=True)
    TRIGGER_DEFAULTS.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


# ── Technical Analysis report defaults (data/ta_report_defaults.json) ────────
# How the report is BUILT, saved once by the desk: which selection rule, how many picks,
# whether to AI-polish, and (for the quality-bar rule) the floors. The app seeds its
# controls from here on every launch, and the WEEKLY emailed report runs on it — so
# "Set as default" in the app is what the scheduled send obeys. (trigger_defaults.json
# can't hold these: it coerces every value to float, and `mode` is a string.)
# One defaults file PER BOOK — FICC futures and equities each carry their own report build
# settings (the FICC file also drives the weekly emailed report). `scope` selects the file.
_TA_REPORT_DIR = Path(__file__).resolve().parents[1] / "data"
TA_REPORT_FILES = {"ficc": _TA_REPORT_DIR / "ta_report_defaults.json",
                   "equities": _TA_REPORT_DIR / "eq_ta_report_defaults.json"}
TA_REPORT_FILE = TA_REPORT_FILES["ficc"]        # back-compat alias (the FICC/weekly-email file)
TA_REPORT_DEFAULTS = {
    "mode": "overall",          # per_side | overall | threshold
    "top_n": 10,                # picks (per side in balanced mode; a CAP in quality-bar mode)
    "ai_polish": True,          # conversational rewrite of the write-ups
    "min_conviction": 60.0,     # quality-bar floors
    "min_score": 150.0,
}


def ta_report_defaults(scope: str = "ficc") -> dict:
    """The saved report build settings for `scope` ('ficc' | 'equities'), falling back to
    TA_REPORT_DEFAULTS. Unknown keys in the file are ignored, so an old/partial file can't break
    the report."""
    out = dict(TA_REPORT_DEFAULTS)
    try:
        f = TA_REPORT_FILES.get(scope, TA_REPORT_FILES["ficc"])
        saved = json.loads(f.read_text(encoding="utf-8"))
        out.update({k: v for k, v in saved.items() if k in TA_REPORT_DEFAULTS})
    except Exception:
        pass
    return out


def save_ta_report_defaults(scope: str = "ficc", **kw) -> None:
    """Persist any subset of the report build settings for `scope` (merged over what's saved)."""
    d = ta_report_defaults(scope)
    d.update({k: v for k, v in kw.items() if k in TA_REPORT_DEFAULTS})
    f = TA_REPORT_FILES.get(scope, TA_REPORT_FILES["ficc"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


# ── TA Backtester defaults (data/tabt_defaults.json / eq_tabt_defaults.json) ─────────────────
# One file PER BOOK, same convention as the TA report defaults above — FICC and Equities keep
# independent thresholds/exit rules since the two books trade very differently.
TABT_FILES = {"ficc": _TA_REPORT_DIR / "tabt_defaults.json",
             "equities": _TA_REPORT_DIR / "eq_tabt_defaults.json"}
TABT_DEFAULTS = {
    "strategies": [],          # [] = "not set" — the page seeds from the book's confluence set
    "min_conviction": 30.0,
    "min_score": 60.0,
    "direction": "both",       # both | long | short
    "exit_rule": "reversal",   # reversal | score_drop | hold_days
    "hold_days": 10.0,
    "stop_pct": 0.0,
    "take_pct": 0.0,
    "size": 0.0,               # 0.0 = "not set" — the page falls back to 1 lot / 100 shares
}


def tabt_defaults(scope: str = "ficc") -> dict:
    """The saved TA Backtester settings for `scope` ('ficc' | 'equities'), falling back to
    TABT_DEFAULTS. Unknown keys in the file are ignored, so an old/partial file can't break the page."""
    out = dict(TABT_DEFAULTS)
    try:
        f = TABT_FILES.get(scope, TABT_FILES["ficc"])
        saved = json.loads(f.read_text(encoding="utf-8"))
        out.update({k: v for k, v in saved.items() if k in TABT_DEFAULTS})
    except Exception:
        pass
    return out


def save_tabt_defaults(scope: str = "ficc", **kw) -> None:
    """Persist any subset of the TA Backtester settings for `scope` (merged over what's saved)."""
    d = tabt_defaults(scope)
    d.update({k: v for k, v in kw.items() if k in TABT_DEFAULTS})
    f = TABT_FILES.get(scope, TABT_FILES["ficc"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


def fi_action(signal, direction, ticker) -> str:
    """Append the FUTURES action to a fixed-income technical signal so the row says BOTH the
    yield read and what to do to profit. FI runs on yields, so a rising-yield signal
    (direction > 0) means the bond price falls → **sell** the bond/future; falling yields →
    **buy** it. Non-FI, or unflagged rows (direction 0 / NaN), are returned unchanged. Only the
    technical strategies opt in (reflag_rows fi_yield=True / run_daily)."""
    from .universe import is_fixed_income, is_bond
    try:
        d = int(direction)
    except (TypeError, ValueError):
        d = 0
    if d == 0 or not is_fixed_income(str(ticker)):
        return signal
    inst = "bond" if is_bond(str(ticker)) else "future"
    return f"{signal} · {'sell' if d > 0 else 'buy'} the {inst}"


def reflag_rows(view: pd.DataFrame, threshold: float, hi, lo, fi_yield: bool = False) -> pd.DataFrame:
    """Re-derive (signal, direction) on the dashboard rows from the cached `metric`
    (the z-score, or the 3-month return for Trend) at a chosen threshold — instant,
    no recompute. `hi`/`lo` are (signal, direction) for metric >= +t / <= -t.

    fi_yield=True (the technical strategies, which chart fixed income as yields) appends the
    futures action to FI rows via fi_action, e.g. 'Long · sell the bond'."""
    out = view.copy()
    m = pd.to_numeric(out["metric"], errors="coerce")
    out["signal"] = (pd.Series("—", index=out.index, dtype=object)
                     .mask(m >= threshold, hi[0]).mask(m <= -threshold, lo[0]))
    out["direction"] = (pd.Series(0, index=out.index, dtype="int64")
                        .mask(m >= threshold, hi[1]).mask(m <= -threshold, lo[1]))
    if fi_yield and "instruments" in out.columns:
        out["signal"] = [fi_action(s, d, tk) for s, d, tk
                         in zip(out["signal"], out["direction"], out["instruments"])]
    return out
