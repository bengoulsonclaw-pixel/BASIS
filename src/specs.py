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
                              "RSI extremes, a fresh MACD cross, and (the headline) RSI divergence."),
        "math": (
            "**Momentum — RSI + MACD, divergence-led.**\n\n"
            "- **RSI** — Wilder's 14-period; > 70 overbought, < 30 oversold.\n"
            "- **MACD** — 12/26 EMA difference, 9-EMA signal, histogram; the line crossing **above** the "
            "signal is bullish, **below** bearish (a cross within ~3 bars is *fresh*).\n"
            "- **RSI divergence** (the headline) — price **higher-high while RSI lower-high** is bearish; "
            "price **lower-low while RSI higher-low** is bullish, read off the two most recent swings.\n\n"
            "Direction = **bullish** if a bullish divergence OR oversold RSI OR a fresh bullish cross "
            "(bearish the mirror). The **momentum score** = `2·|RSI − 50|` + a divergence bonus + a "
            "fresh-cross bonus, signed by direction and clipped to ±100. Flag when **|score| ≥ trigger**. "
            "Close-only."
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


def reflag_rows(view: pd.DataFrame, threshold: float, hi, lo) -> pd.DataFrame:
    """Re-derive (signal, direction) on the dashboard rows from the cached `metric`
    (the z-score, or the 3-month return for Trend) at a chosen threshold — instant,
    no recompute. `hi`/`lo` are (signal, direction) for metric >= +t / <= -t."""
    out = view.copy()
    m = pd.to_numeric(out["metric"], errors="coerce")
    out["signal"] = (pd.Series("—", index=out.index, dtype=object)
                     .mask(m >= threshold, hi[0]).mask(m <= -threshold, lo[0]))
    out["direction"] = (pd.Series(0, index=out.index, dtype="int64")
                        .mask(m >= threshold, hi[1]).mask(m <= -threshold, lo[1]))
    return out
