"""Plain-English trigger documentation — how each technical strategy decides to buy / sell.

Written for the desk (and for Ben's clients' "why did it buy there?" questions), one entry per
strategy in tascore.TA_STRATEGIES. Every sentence was distilled from the strategy MODULES' actual
code during the 2026-08-08 signal audit (tests/test_signals_*.py fixture-tests every rule below:
a clean textbook long fixture must flag +1, the short mirror −1 — 80-test suite locks it), so this
file describes what the code DOES, not what a textbook says it should do. If a strategy's logic
changes, update the entry AND the fixture test together.

Rendered by the TA Backtester page's "How each strategy decides" expander; reusable anywhere.

Convention notes the page shows alongside these entries:
- FICC fixed income signals compute on YIELDS — a "buy" read there means rising yields, which the
  backtester and reports translate to SELLING the bond/STIR future (and the mirror).
- The hub/backtester additionally applies the conviction/score bar on top of each trigger; where a
  hub bar differs from the module's own page default it is called out in the entry.
"""

TRIGGER_DOCS = {
    "Mean Reversion": {
        "buys": "Flags 'Long spread' — buy the first leg, sell the second — when the pair's spread "
                "(a price difference for same-unit pairs like WTI–Brent, a ratio otherwise; "
                "fixed-income legs run on yields) sits 1.5 standard deviations or more below its "
                "rolling 90-day mean. Page default flags at |z| ≥ 1.5 and the hub/backtester bar "
                "is the same 1.5. Each flag carries the estimated mean-reversion half-life in days.",
        "sells": "Flags 'Short spread' — sell the first leg, buy the second — when the spread sits "
                 "1.5 standard deviations or more above its 90-day mean; anything inside the "
                 "±1.5z band stays silent.",
    },
    "Trend": {
        "buys": "Buys when the trailing 3-month (63-session) return is positive — direction is "
                "purely the sign of that return, sanity-checked against a 20 vs 100-day "
                "moving-average cross (a disagreement is marked 'mixed' but does not block). "
                "Page default flags every market (trigger 0); the hub/backtester bar only flags "
                "moves of 10% or more.",
        "sells": "Sells when the 3-month return is negative, same MA-cross annotation; hub bar "
                 "is a 3-month loss of 10% or more.",
    },
    "MA Crossover": {
        "buys": "Buys on a 50/200-day golden cross, but only when all three confirmations line "
                "up: the 15-day EMA sitting above the 50-day average, the trailing 3-month "
                "return positive, and the 50 at least 2% above the 200 (shallow, fresh crosses "
                "do not trade). No separate hub threshold — this gate is the gate everywhere.",
        "sells": "Sells on a 50/200 death cross with the 15-day EMA below the 50, the 3-month "
                 "return negative, and the gap at least 2% the other way; anything unconfirmed "
                 "shows as no trade.",
    },
    "MA Swing": {
        "buys": "Buys on a 20/50-day golden cross confirmed by the 9-day EMA above the 20-day "
                "average, a positive trailing 1-month (21-session) return, and the 20 at least "
                "0.5% above the 50 — the position crossover's rule set at swing speed.",
        "sells": "Sells on a 20/50 death cross with the 9-day EMA below the 20, the 1-month "
                 "return negative, and at least a 0.5% gap; unconfirmed set-ups show as no trade.",
    },
    "Flag Breakout": {
        "buys": "A bull flag at the right edge: a 4-20 bar pole up at least 4% (and at least "
                "2.5σ of the product's own 60-day daily volatility, scaled by pole length), then "
                "a 5-15 bar consolidation that gives back no more than half the pole, holds "
                "inside a channel no wider than 0.75× the pole, and drifts no steeper than half "
                "the pole's pace with it. Breakout readiness is price's position across that "
                "channel (100 = on the breakout line; beyond 100 = already through, capped 140); "
                "flags at 70 — page and hub alike. Volume drying up through the flag (≤90% of "
                "pole volume) earns a confirmation tick, never a veto.",
        "sells": "The bear-flag mirror — a fast pole down, a gentle upward drift, price pressing "
                 "the lower channel line; flags at readiness −70.",
    },
    "Support & Resistance": {
        "buys": "Price closes within 2% at-or-above a horizontal support built from at least 2 "
                "swing-low touches (4-bar-each-side pivots over the past 252 sessions, touches "
                "clustered within ~1.5% of price or one 14-day average daily move). Flags when "
                "level proximity (100 = sitting exactly on the level) reaches 50 — page default "
                "and hub bar are the same 50. A level that flipped role after a break trades the "
                "same way and is called out as role-reversed.",
        "sells": "Mirror: price closes within 2% at-or-below a tested resistance (2+ swing-high "
                 "touches), flagged at proximity 50.",
    },
    "Fibonacci Retracement": {
        "buys": "The dominant swing of the last 180 sessions (the leg must span at least 5% of "
                "price) ended with the high more recent — an up-leg — and price has pulled back "
                "onto a key retracement (38.2, 50 or 61.8%). Proximity fades from 100 on the "
                "level to 0 at 1% away; flags at 60 (within ~0.4% of the level), page and hub "
                "alike. Stop reference sits beyond the 78.6% level; target is the prior swing "
                "high.",
        "sells": "Same with the low more recent — a down-leg — and price bouncing up into a key "
                 "retracement acting as resistance; flags at −60.",
    },
    "Breakout & Retest": {
        "buys": "Within the last 20 sessions a close cleared a prior swing high (3-bar-each-side "
                "pivot) by more than 0.3× the 14-day average daily move — and if volume data "
                "exists, the breakout bar traded at least 1.3× its trailing 20-day average, "
                "otherwise the break is discarded. Price has since pulled back to within 1.5% of "
                "the broken level from above (broken resistance now support); flags when retest "
                "proximity reaches 60, page and hub alike. No qualifying break-plus-retest means "
                "no row at all.",
        "sells": "Mirror: a decisive close below a prior swing low, now being retested from "
                 "below (broken support now resistance); flags at −60.",
    },
    "Momentum (RSI/MACD)": {
        "buys": "Flags long only on an event: a live bullish RSI divergence (price lower low "
                "while 14-day RSI makes a higher low, read off the two latest swings in a 40-bar "
                "window, confirming swing within the last 10 bars and RSI gap ≥3 points) or a "
                "12/26/9 MACD line crossing above its signal within the last 3 bars — and only "
                "if RSI is not already above 70. Needs the signed momentum score (2× the RSI "
                "room below 50, +25 for the divergence, +15 for the fresh cross) to reach +40; "
                "hub bar the same.",
        "sells": "Mirror: a bearish divergence (higher price high on a lower RSI high) or a MACD "
                 "cross below signal within 3 bars, blocked if RSI is already under 30; flags at "
                 "−40. An RSI extreme alone never fires — it only vetoes signals pointing further "
                 "into the move.",
    },
    "Bollinger Squeeze": {
        "buys": "A close above the upper band (20-day average +2σ) flags an upside break; short "
                "of a break, a squeeze (bandwidth in the tightest 20% of its trailing year) with "
                "price in the upper 40% of the band flags an upside watch. The break/watch "
                "labels are the trigger — the hub carries them through unchanged.",
        "sells": "Mirror: a close below the lower band is a downside break; a squeeze with "
                 "price in the lower 40% of the band is a downside watch. Price near the middle "
                 "of a squeeze stays unsigned.",
    },
    "Elliott Wave": {
        "buys": "A deterministic impulse count (ZigZag pivots at a reversal threshold of 2.2× "
                "each market's weekly volatility, floored at 2.5% / capped at 12%, over the last "
                "240 sessions) passes the three hard Elliott rules, and price is advancing off a "
                "completed wave-2 or wave-4 pullback in an UP impulse — or a five-wave DOWN move "
                "has just completed (corrective bounce due). Flags when the wave-fit score "
                "(textbook-ness of the retraces and the wave-3 extension) reaches 55 and the "
                "count's last pivot is no older than 45 sessions.",
        "sells": "The mirror: wave 3 or 5 underway in a DOWN impulse, or five waves UP just "
                 "completed (a correction likely).",
    },
    "Ichimoku Cloud": {
        "buys": "Buys only on a fresh event within the last 6 sessions — a close breaking up "
                "through the cloud, or a bullish Tenkan/Kijun cross with price already above the "
                "cloud — and only when the confluence score reaches 58 (40 base + 16 per "
                "confirming element among the TK cross, future cloud colour and lagging span, "
                "plus clearance and event bonuses). Standard 9/26/52 settings on closes. A "
                "persistent trend above the cloud without a fresh event is shown but not flagged "
                "— stale trends are the Trend/MA reads' job.",
        "sells": "Mirror: a fresh downside cloud break or bearish TK cross with price below the "
                 "cloud, score 58+; anything inside the cloud is an automatic no-trade.",
    },
    "On-Balance Volume": {
        "buys": "Flags long at a score of 55+ from the strongest of: a bullish OBV divergence "
                "(price lower low while OBV holds a higher low, swings confirmed within the last "
                "15 bars), a 55-day price high with OBV also at 90%+ of its own range "
                "(volume-confirmed breakout), a 55-day price low OBV refuses to confirm, or "
                "hidden accumulation (price flat over 20 days while OBV climbs, z ≥ 1.2). FX "
                "futures are excluded — listed FX volume is not the real market.",
        "sells": "Mirror: bearish OBV divergence, a 55-day low confirmed by OBV at its own low, "
                 "a new price high with OBV stuck below 70% of its range (breakout lacks "
                 "volume), or hidden distribution (price flat, OBV sliding, z ≤ −1.2).",
    },
    "Money Flow Index": {
        "buys": "Flags long at a score of 55+ when 14-day money flow (price × volume) pins MFI "
                "at 20 or below — oversold on real flow, stronger when RSI ≤30 agrees; also on "
                "selling that is not volume-backed (RSI ≤30 but MFI still above 40), early "
                "accumulation (price down 2%+ over 20 days while MFI sits above 55 and rising), "
                "or an upward flow shift (MFI crossing up through 50 on a 15+ point 10-day "
                "gain). FX futures are excluded.",
        "sells": "Mirror: MFI 80+ is overbought on real flow (RSI 70+ validating), RSI 70+ with "
                 "MFI under 60 warns the rally lacks flow, early distribution (price up 2%+ with "
                 "MFI under 45 and falling), or a downward flow shift through 50.",
    },
    "Donchian Channel": {
        "buys": "Buys when today's close sits at or beyond 90% of the prior 20-session high-low "
                "channel — a signed channel position of +80 or more on the −100/+100 scale, with "
                "a genuine new 20-day high reading beyond +100 (capped 140). Objective is one "
                "channel-height above the broken band; invalidation is the opposite band. Hub "
                "bar is the same +80.",
        "sells": "Sells at a channel position of −80 or worse — at or through the 20-day low; "
                 "same objective/invalidation logic mirrored.",
    },
    "Aroon": {
        "buys": "Buys when the 25-session Aroon oscillator (Up minus Down) reaches +50 — fresh "
                "highs keep printing while the last low ages out; +100 means a new 25-day high "
                "today. Hub bar is the same +50.",
        "sells": "Sells at an oscillator of −50 or below — fresh lows dominating; readings near "
                 "zero are chop and never flag.",
    },
}
