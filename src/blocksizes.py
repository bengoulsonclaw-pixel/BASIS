"""Minimum block-trade sizes (lots) for each product's futures and listed options,
plus how the minimum applies to STRATEGY blocks (sum of legs vs each leg).

Seeded from the exchanges' published rules, all pulled 2026-07-02:
  • CME/CBOT/NYMEX/COMEX — the official block-minimum-thresholds Excel linked from
    cmegroup.com/clearing/trading-practices/block-trades.html + MRAN RA2606-5
    (Rule 526, effective 29-Jun-2026). Rates/FX/equity minimums split by session:
    RTH 07:00–16:00 CT Mon–Fri, ETH 00:00–07:00 CT, ATH 16:00–24:00 CT + weekends.
    Strategy mechanics: CME/CBOT intra-commodity FUTURES spreads = sum of legs,
    but their OPTIONS combos = each leg (FX options are the exception: sum of
    legs); NYMEX/COMEX = sum of legs for everything (≥ the larger min for
    inter-commodity); Treasury calendar-spread blocks are prohibited outright.
    Covered (options vs futures) blocks on all four exchanges: the options leg
    meets the options minimum, the futures leg just has to match the delta.
  • ICE Futures Europe / Endex / US — IFEU Block Trades Policy (Dec 2025) +
    Appendices A/C/D (Feb–Jun 2026), Endex Block & LIS sheet, IFUS Block Trade
    FAQ (Feb 2026). IFEU STIRs tier by colour (White = front quarterlies) and
    publish dedicated STRATEGY minimums (≈ 2× outright) that the sum of the
    legs must meet; IFEU oil and IFUS softs strategies sum the legs against
    the outright minimum (the larger one for inter-commodity).
  • Eurex — Contract Specifications (Eurex14e, 25-May-2026) + T7 Functional
    Reference: the TES minimum applies to EACH LEG of a strategy (the strategy's
    minimum equals the outright's; vola-strategy futures legs are delta-sized).
  • Euronext — LIS thresholds applicable 03-Jun-2026; liquid 2-leg packages are
    per-leg, illiquid/multi-leg packages need only one leg over the threshold.
  • JPX/OSE J-NET, KRX negotiated block rules, ASX BTF (Procedure 4820), SGX NLT
    (Reg Notice 4.1.11 — one leg meeting the minimum qualifies the strategy).

These are INDICATIVE — exchanges revise thresholds; verify against the rulebook
before quoting a client. The table is editable from the app's Minimum Block Sizes
page and persists to data/blocksizes.json (stored fields win over the seed
per-field, so new seed columns still reach an existing store).
"""
from __future__ import annotations

import json
from pathlib import Path

# ticker -> (futures min, options min, strategy rule, note) — strings, since many
# minimums are tiered ("2,000 RTH / 1,000 ETH / 500 ATH") or not a number
# ("Not eligible"). The strategy rule says what the minimum is measured against
# for spread/combination blocks: the SUM of the legs or EACH leg individually.
_SEED = {
    # ── INDICES ────────────────────────────────────────────────────────────
    "ESA Index":  ("Not eligible outright", "100",
                   "Each leg (options combos & BTIC spreads)",
                   "Outright futures blocks not permitted — BTIC (EST) only: 500 RTH / 100 ETH / 100 ATH; "
                   "options 250 if executed as a derived-price block."),
    "NQA Index":  ("Not eligible outright", "20",
                   "Each leg (options combos & BTIC spreads)",
                   "Outright futures blocks not permitted — BTIC (NQT) only: 500 RTH / 100 ETH / 100 ATH."),
    "RTYA Index": ("40", "40",
                   "Fut spreads: sum of legs · options combos: each leg",
                   "Outright blocks permitted, all hours; BTIC (RLT) also 40, BTIC spreads each leg."),
    "DMA Index":  ("Not eligible outright", "Not eligible",
                   "Each leg (BTIC spreads)",
                   "Outright futures blocks not permitted — BTIC (YMT) only: 500 RTH / 100 ETH / 100 ATH; "
                   "no block-eligible options listing."),
    "VGA Index":  ("2,000", "1,000",
                   "Each leg (TES strategy min = outright min)",
                   "Eurex TES; options incl. weeklies under OESX; vola-strategy futures legs delta-sized."),
    "CAA Index":  ("600", "200",
                   "Each leg (liquid 2-leg packages); one leg suffices for illiquid/multi-leg",
                   "Euronext LIS (pre-trade) thresholds from 03-Jun-2026; post-trade deferral tiers 3,400 fut / 720 opt."),
    "Z A Index":  ("500", "500",
                   "Sum of legs ≥ strategy min (1,000)",
                   "IFEU strategy minimum 1,000 for futures AND standard options; TIC blocks 50; "
                   "flex/block-only option series min 1."),
    "GXA Index":  ("250", "350",
                   "Each leg (TES strategy min = outright min)",
                   "Eurex TES; ODAX incl. weekly expiries."),
    "SMA Index":  ("500", "300",
                   "Each leg (TES strategy min = outright min)", "Eurex TES."),
    "NKA Index":  ("1 (J-NET)", "1 (J-NET)",
                   "N/A — each issue trades as its own J-NET trade (min 1)",
                   "OSE has no block-minimum concept — J-NET off-auction accepts 1 lot or more; "
                   "Nikkei 225 mini options min 10; micro futures not J-NET eligible."),
    "KMA Index":  ("20", "20",
                   "Per instrument — listed futures spreads are themselves block-eligible",
                   "KRX negotiated block: front-month/settlement-week series min 20 (max 5,000 per application; "
                   "options max 10,000); other months min 1."),
    "XPA Index":  ("200", "200",
                   "N/A — nearest-quarterly only, no roll blocks (7+ leg option strategies: 1 lot/leg)",
                   "ASX block (BTF): nearest quarterly only, not within 5 business days of expiry; "
                   "using the BTF for roll business breaches the ASIC MIRs; options all months."),

    # ── STIRs ──────────────────────────────────────────────────────────────
    "SERA Comdty": ("500 RTH / 250 ETH / 125 ATH", "1,000 RTH / 500 ETH / 250 ATH",
                    "Fut spreads: sum of legs · options combos: each leg",
                    "Inter-commodity STIR spreads: each leg meets the SMALLER product's min."),
    "SFRA Comdty": ("2,000 RTH / 1,000 ETH / 500 ATH", "5,000 RTH / 2,500 ETH / 1,250 ATH",
                    "Fut spreads: sum of legs · options combos: each leg",
                    "Mid-curve options same thresholds as standards; inter-commodity STIR spreads: "
                    "each leg meets the SMALLER product's min."),
    "FFA Comdty":  ("2,000 RTH / 1,000 ETH / 500 ATH", "Standard options not eligible",
                    "Fut spreads: sum of legs · options combos: each leg",
                    "6M & 12M mid-curve options: 1,500 RTH / 750 ETH / 375 ATH."),
    "ERA Comdty":  ("3,000 white / 1,500 red / 500 back", "3,000 white / 1,000 red / 1,000 mid-curve",
                    "Sum of legs ≥ strategy min (2× outright: 6,000 white / 3,000 red / 1,000 back)",
                    "IFEU tiers by colour (White = front 4 quarterlies; serials 500); legs spanning "
                    "colours take the larger strategy min; options strategies use the outright min."),
    "TKYA Comdty": ("750 white / 375 red / 125 back", "250 white / 100 red / 100 mid-curve",
                    "Sum of legs ≥ strategy min (2× outright: 1,500 white / 750 red / 250 back)",
                    "IFEU tiers by colour; options strategies use the outright min."),
    "SFIA Comdty": ("250", "500 white / 250 red / 250 mid-curve",
                    "Sum of legs ≥ strategy min (500)",
                    "Futures 250 all months (White = front 5 expiries for the options tiers); "
                    "options strategies use the outright min."),

    # ── BONDS ──────────────────────────────────────────────────────────────
    "USA Comdty":  ("2,000 RTH / 1,000 ETH / 500 ATH", "5,000 RTH / 2,500 ETH / 1,250 ATH",
                    "Cal spreads PROHIBITED · inter (NOB etc.) & options combos: each leg",
                    "Treasury futures calendar-spread blocks prohibited; inter-commodity legs each "
                    "meet their own product's min."),
    "WNA Comdty":  ("1,500 RTH / 750 ETH / 375 ATH", "800 RTH / 600 ETH / 300 ATH",
                    "Cal spreads PROHIBITED · inter & options combos: each leg", ""),
    "UXYA Comdty": ("3,000 RTH / 1,500 ETH / 750 ATH", "1,400 RTH / 700 ETH / 350 ATH",
                    "Cal spreads PROHIBITED · inter & options combos: each leg", ""),
    "TYA Comdty":  ("5,000 RTH / 2,500 ETH / 1,250 ATH", "7,500 RTH / 3,750 ETH / 1,875 ATH",
                    "Cal spreads PROHIBITED · inter & options combos: each leg", ""),
    "FVA Comdty":  ("5,000 RTH / 2,500 ETH / 1,250 ATH", "7,500 RTH / 3,750 ETH / 1,875 ATH",
                    "Cal spreads PROHIBITED · inter & options combos: each leg", ""),
    "TUA Comdty":  ("5,000 RTH / 2,500 ETH / 1,250 ATH", "2,000 RTH / 1,000 ETH / 500 ATH",
                    "Cal spreads PROHIBITED · inter & options combos: each leg", ""),
    "UBA Comdty":  ("100", "50", "Each leg (TES strategy min = outright min)", "Eurex TES."),
    "RXA Comdty":  ("2,000", "100", "Each leg (TES strategy min = outright min)",
                    "Eurex TES; OGBL incl. weekly expiries; vola-strategy futures legs delta-sized."),
    "OEA Comdty":  ("3,000", "250", "Each leg (TES strategy min = outright min)", "Eurex TES."),
    "DUA Comdty":  ("4,000", "500", "Each leg (TES strategy min = outright min)", "Eurex TES."),
    "OATA Comdty": ("250", "100", "Each leg (TES strategy min = outright min)", "Eurex TES."),
    "G A Comdty":  ("500", "100",
                    "Sum of legs ≥ strategy min (front 500 / other months 1,000)",
                    "IFEU: legs spanning buckets take the larger strategy min; deferred-publication tier 1,500."),

    # ── FX (CME; single threshold all hours, majors split monthly/quarterly) ─
    "ECA Curncy":  ("20 monthly / 150 quarterly", "250", "Sum of legs (futures AND options combos)",
                    "FX options are CME's sum-of-legs exception; mixed-min legs sum to the larger min."),
    "BPA Curncy":  ("20 monthly / 100 quarterly", "250", "Sum of legs (futures AND options combos)", ""),
    "JYA Curncy":  ("20 monthly / 150 quarterly", "250", "Sum of legs (futures AND options combos)", ""),
    "SFA Curncy":  ("100", "250", "Sum of legs (futures AND options combos)", ""),
    "CDA Curncy":  ("20 monthly / 100 quarterly", "250", "Sum of legs (futures AND options combos)", ""),
    "ADA Curncy":  ("20 monthly / 100 quarterly", "250", "Sum of legs (futures AND options combos)", ""),
    "NVA Curncy":  ("50", "50", "Sum of legs (futures AND options combos)", ""),
    "BRA Curncy":  ("50", "50", "Sum of legs (futures AND options combos)", ""),
    "PEA Curncy":  ("100", "50", "Sum of legs (futures AND options combos)",
                    "Options minimum sits below the futures minimum."),
    "RAA Curncy":  ("50", "50", "Sum of legs (futures AND options combos)", ""),
    "SEA Curncy":  ("10", "No options listed", "Sum of legs (futures spreads)", ""),
    "NOA Curncy":  ("10", "No options listed", "Sum of legs (futures spreads)", ""),
    "KOA Curncy":  ("10", "20", "Sum of legs (futures AND options combos)", ""),
    "SIRA Curncy": ("10", "50", "Sum of legs (futures AND options combos)", ""),
    "ISA Curncy":  ("10", "20", "Sum of legs (futures AND options combos)", ""),
    "HEA Curncy":  ("10", "20", "Sum of legs (futures AND options combos)", ""),
    "CCA Curncy":  ("10", "20", "Sum of legs (futures AND options combos)", ""),
    "PPA Curncy":  ("10", "20", "Sum of legs (futures AND options combos)", ""),

    # ── ENERGY ─────────────────────────────────────────────────────────────
    "COA Comdty":  ("100", "25",
                    "Sum of legs (inter-commodity: ≥ the larger min)",
                    "IFEU; vol trades: only the options leg (25) has a minimum. Options min is the "
                    "'all ICE oil options' tier."),
    "CLA Comdty":  ("50", "100",
                    "Sum of legs (inter-commodity: ≥ the larger min)",
                    "NYMEX sums legs for futures AND options strategies; options (LO) min sits above the futures min."),
    "XBA Comdty":  ("25", "25", "Sum of legs (inter-commodity: ≥ the larger min)", ""),
    "NGA Comdty":  ("50", "100", "Sum of legs (inter-commodity: ≥ the larger min)",
                    "Options min is ON (American); European financial options (LN) 15."),
    "FJSA Comdty": ("1", "1", "N/A — minimum is 1 lot",
                    "TFM/TFO on ICE Endex; USD/MMBtu first-line (TFU) 5."),
    "HOA Comdty":  ("25", "25", "Sum of legs (inter-commodity: ≥ the larger min)", ""),
    "QSA Comdty":  ("100", "25",
                    "Sum of legs (inter-commodity: ≥ the larger min)",
                    "IFEU; HOGO legs sum to the larger min (100). Options min is the 'all ICE oil options' tier."),
    "MOA Comdty":  ("50", "25",
                    "Not specified in the Endex policy — confirm before sizing at the min",
                    "EUA on ICE Endex; EUA Mini 1; UKA (IFEU) 10 fut / 5 opt."),
    "CUAA Comdty": ("5", "5", "Sum of legs (inter-commodity: ≥ the larger min)",
                    "Swap-style monthly futures; option is the CVR average-price option."),

    # ── METALS ─────────────────────────────────────────────────────────────
    "GCA Comdty":  ("25", "25", "Sum of legs (inter-commodity: ≥ the larger min)",
                    "COMEX sums legs for futures AND options strategies; weekly gold options 10."),
    "SIA Comdty":  ("25", "15", "Sum of legs (inter-commodity: ≥ the larger min)", "Weekly silver options 10."),
    "PLA Comdty":  ("10", "10", "Sum of legs (inter-commodity: ≥ the larger min)", ""),
    "PAA Comdty":  ("10", "10", "Sum of legs (inter-commodity: ≥ the larger min)", ""),
    "HGA Comdty":  ("20 active / 5 other months", "10",
                    "Sum of legs (inter-commodity: ≥ the larger min)",
                    "Futures min is month-dependent: 20 in the nearby + second active month, 5 elsewhere."),
    "ALEA Comdty": ("5", "5", "Sum of legs (inter-commodity: ≥ the larger min)", ""),
    "SCOA Comdty": ("5", "5", "One leg meeting the min qualifies the whole strategy",
                    "SGX Negotiated Large Trade (NLT), same-underlying multi-leg only — no combining "
                    "different customers' orders into inter-commodity strategies."),

    # ── AGRICULTURE ────────────────────────────────────────────────────────
    "C A Comdty":  ("140", "200", "Fut spreads: sum of legs · options combos: each leg",
                    "Inter-commodity futures: sum of legs ≥ the largest leg's min (select pairs have "
                    "fixed per-leg ratios instead)."),
    "S A Comdty":  ("100", "75", "Fut spreads: sum of legs · options combos: each leg", ""),
    "W A Comdty":  ("100", "75", "Fut spreads: sum of legs · options combos: each leg", ""),
    "KWA Comdty":  ("50", "75", "Fut spreads: sum of legs · options combos: each leg", ""),
    "SMA Comdty":  ("50", "75", "Fut spreads: sum of legs · options combos: each leg", ""),
    "BOA Comdty":  ("50", "75", "Fut spreads: sum of legs · options combos: each leg", ""),
    "RRA Comdty":  ("10", "20", "Fut spreads: sum of legs · options combos: each leg", ""),

    # ── SOFTS ──────────────────────────────────────────────────────────────
    "KCA Comdty":  ("250", "250", "Sum of legs (inter-commodity: ≥ the largest min)",
                    "IFUS sums legs for all strategy blocks; 100 as the Arabica leg of the "
                    "Arabica/Robusta arb; calendar-spread options 100."),
    "DFA Comdty":  ("Not eligible outright", "Not eligible", "N/A — not block-eligible",
                    "IFEU softs have no outright block facility — 170 lots only as the Robusta leg of the IFUS arb."),
    "CCA Comdty":  ("Not eligible", "350", "Sum of legs (options strategies)",
                    "IFUS blocks are options-only for cocoa; calendar-spread options 100."),
    "SBA Comdty":  ("250", "250", "Sum of legs (inter-commodity: ≥ the largest min)",
                    "100 as the No.11 leg of the No.11/White arb; calendar-spread options 100."),
    "CTA Comdty":  ("500", "250", "Sum of legs (inter-commodity: ≥ the largest min)",
                    "Calendar-spread options 100."),
    "JOA Comdty":  ("Not eligible", "100", "Sum of legs (options strategies)",
                    "IFUS blocks are options-only for FCOJ."),
    "LHA Comdty":  ("50", "100", "Fut spreads: sum of legs · options combos: each leg",
                    "Inter-commodity futures: sum of legs ≥ the largest leg's min."),
    "LCA Comdty":  ("50", "100", "Fut spreads: sum of legs · options combos: each leg", ""),
    "FCA Comdty":  ("50", "100", "Fut spreads: sum of legs · options combos: each leg",
                    "Inter-commodity futures: sum of legs ≥ the largest leg's min."),
}

STORE = Path(__file__).resolve().parents[1] / "data" / "blocksizes.json"

_BLANK = {"fut": "", "opt": "", "strat": "", "note": ""}


def _seed_map() -> dict:
    return {tk: {"fut": f, "opt": o, "strat": s, "note": n} for tk, (f, o, s, n) in _SEED.items()}


def load_map() -> dict:
    """ticker -> {fut, opt, strat, note}. Stored fields win over the seed PER FIELD,
    so a store written before a column existed still picks up that column's seed."""
    seed = _seed_map()
    if not STORE.exists():
        save_map(seed)
        return seed
    try:
        stored = json.loads(STORE.read_text(encoding="utf-8")).get("blocks", {})
    except Exception:
        stored = {}
    return {tk: {**_BLANK, **seed.get(tk, {}), **stored.get(tk, {})}
            for tk in seed.keys() | stored.keys()}


def save_map(m: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps({"blocks": m}, indent=2), encoding="utf-8")
