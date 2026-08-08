"""Indicative trading hours per product, for the Home "Market Hours" timeline.

Each product maps to an exchange SESSION PROFILE: the full electronic session (when the
contract is tradeable) and the liquid / primary window (where the volume is), expressed in
the exchange's LOCAL time. `day_segments()` converts a profile to a chosen reference
timezone for a given date (DST-correct via zoneinfo) and returns bars on a 0–24h axis,
splitting any session that wraps past midnight.

These are INDICATIVE regular-session hours (ex-holidays, ex-special-sessions) — good enough
to see what's open and when across the book, not an exchange-official calendar. Times are
easy to correct here if a desk wants them exact.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# Sector colour code (user spec) — full session = the colour faded, liquid window = solid.
ASSET_COLORS = {
    "Indices": "#7E57C2",       # purple
    "STIRs": "#5BB8E8",         # light blue
    "Bonds": "#1E5FA8",         # darker blue
    "FX": "#8A8F99",            # grey
    "Energy": "#33A95E",        # green
    "Metals": "#D23B3B",        # red
    "Agriculture": "#EBC23A",   # yellow
    "Softs": "#EBC23A",         # yellow
}
ASSET_LIQUID = {"FX": "#1F1F1F"}    # FX reads grey (session) + near-black (liquid) = "grey and black"

# Daily settlement time per profile, in LOCAL exchange time (indicative — easy to correct).
SETTLE = {
    "cme_eq": "15:00", "cme_fx": "14:00", "cme_stir": "14:00", "cbot_rates": "14:00",
    "cbot_grain": "13:20", "cme_live": "13:05", "nymex_en": "14:30", "comex_me": "13:30",
    "eurex_bond": "17:15", "eurex_eq": "17:30", "euronext_eq": "17:30", "ice_ftse": "16:30",
    "ice_gilt": "16:15", "ice_stir_eu": "17:00", "ice_brent": "19:30", "ice_gasoil": "16:30", "ice_ttf": "17:00",
    "ice_eua": "17:00", "ice_robusta": "16:30", "ice_sugar": "13:00", "ice_coffee": "13:30",
    "ice_cocoa": "11:50", "ice_cotton": "14:20", "ice_oj": "14:00", "cash_eu_de": "17:30",
    "cash_uk": "16:30", "jpx_nikkei": "15:15", "krx_kospi": "15:45", "asx_spi": "16:30",
    "sgx_iron": "18:30",
}

# profile id -> (exchange label, IANA tz, full blocks, liquid blocks)
# blocks: list of ("HH:MM","HH:MM"); if end <= start the block crosses midnight.
PROFILES = {
    "cme_eq":      ("CME Globex", "America/Chicago",   [("17:00", "16:00")], [("08:30", "15:15")]),
    "cme_fx":      ("CME Globex", "America/Chicago",   [("17:00", "16:00")], [("02:00", "14:00")]),
    "cme_stir":    ("CME Globex", "America/Chicago",   [("17:00", "16:00")], [("07:00", "14:00")]),
    "cbot_rates":  ("CBOT (CME)", "America/Chicago",   [("17:00", "16:00")], [("07:00", "14:00")]),
    "cbot_grain":  ("CBOT (CME)", "America/Chicago",   [("19:00", "07:45"), ("08:30", "13:20")], [("08:30", "13:20")]),
    "cme_live":    ("CME",        "America/Chicago",   [("08:30", "13:05")], [("08:30", "13:05")]),
    "nymex_en":    ("NYMEX (CME)", "America/New_York", [("18:00", "17:00")], [("09:00", "14:30")]),
    "comex_me":    ("COMEX (CME)", "America/New_York", [("18:00", "17:00")], [("08:20", "13:30")]),
    "eurex_bond":  ("Eurex",      "Europe/Berlin",     [("01:10", "22:00")], [("08:00", "17:15")]),
    "eurex_eq":    ("Eurex",      "Europe/Berlin",     [("01:10", "22:00")], [("09:00", "17:30")]),
    "euronext_eq": ("Euronext",   "Europe/Paris",      [("08:00", "22:00")], [("09:00", "17:30")]),
    "ice_ftse":    ("ICE Europe", "Europe/London",     [("01:00", "21:00")], [("08:00", "16:30")]),
    "ice_gilt":    ("ICE Europe", "Europe/London",     [("08:00", "18:00")], [("08:00", "18:00")]),
    "ice_stir_eu": ("ICE Europe", "Europe/London",     [("01:00", "21:00")], [("07:00", "18:00")]),
    # Brent stays liquid through the London day and the NY overlap, right into the ICE
    # settlement (19:30 London = 14:30 ET in summer — the same instant WTI settles).
    "ice_brent":   ("ICE Europe", "Europe/London",     [("01:00", "23:00")], [("08:00", "19:30")]),
    "ice_gasoil":  ("ICE Europe", "Europe/London",     [("01:00", "23:00")], [("08:00", "16:30")]),
    "ice_ttf":     ("ICE Endex",  "Europe/London",     [("08:00", "18:00")], [("08:00", "18:00")]),
    "ice_eua":     ("ICE Europe", "Europe/London",     [("07:00", "17:00")], [("07:00", "17:00")]),
    "ice_robusta": ("ICE Europe", "Europe/London",     [("09:00", "17:30")], [("09:00", "17:30")]),
    "ice_sugar":   ("ICE US",     "America/New_York",  [("02:30", "13:00")], [("02:30", "13:00")]),
    "ice_coffee":  ("ICE US",     "America/New_York",  [("04:15", "13:30")], [("04:15", "13:30")]),
    "ice_cocoa":   ("ICE US",     "America/New_York",  [("04:45", "13:30")], [("04:45", "13:30")]),
    "ice_cotton":  ("ICE US",     "America/New_York",  [("21:00", "14:20")], [("08:00", "14:20")]),
    "ice_oj":      ("ICE US",     "America/New_York",  [("08:00", "14:00")], [("08:00", "14:00")]),
    "cash_eu_de":  ("Cash equity", "Europe/Berlin",    [("09:00", "17:30")], [("09:00", "17:30")]),
    "cash_uk":     ("Cash equity (LSE)", "Europe/London", [("08:00", "16:30")], [("08:00", "16:30")]),
    "jpx_nikkei":  ("Osaka (JPX)", "Asia/Tokyo",       [("08:45", "15:15"), ("16:30", "06:00")], [("09:00", "15:00")]),
    "krx_kospi":   ("KRX",        "Asia/Seoul",        [("09:00", "15:45"), ("18:00", "05:00")], [("09:00", "15:45")]),
    "asx_spi":     ("ASX",        "Australia/Sydney",  [("09:50", "16:30"), ("17:10", "07:00")], [("09:50", "16:30")]),
    # Iron ore: the contract lists on SGX, but price discovery / peak liquidity is the
    # Dalian (DCE) day sessions. SGT and Beijing time are both UTC+8 year-round (no DST),
    # so the DCE clock times below are authored directly in the SGX profile; the liquid
    # bar is RELABELLED "Dalian (DCE)" via LIQUID_LABEL. DCE day sessions (Beijing):
    # 09:00–10:15, 10:30–11:30, 13:30–15:00.
    "sgx_iron":    ("SGX",        "Asia/Singapore",    [("07:25", "19:00"), ("19:15", "05:15")],
                    [("09:00", "10:15"), ("10:30", "11:30"), ("13:30", "15:00")]),
}

# Products whose LIQUID window belongs to a different venue than the listing exchange.
# liquid label shown in the tooltip, and the holiday calendar that governs that liquid
# window (so the liquid bar disappears when THAT market is shut, even if the listing
# exchange is open). The clock times live in the profile above (see sgx_iron note).
LIQUID_LABEL = {"sgx_iron": "Dalian (DCE)"}
LIQUID_CAL = {"sgx_iron": "CN"}

# full Bloomberg ticker -> profile id
PROFILE_OF = {
    # Indices — index futures vs cash index
    "ESA Index": "cme_eq", "NQA Index": "cme_eq", "RTYA Index": "cme_eq", "DMA Index": "cme_eq",
    "VGA Index": "eurex_eq", "GXA Index": "eurex_eq", "SMA Index": "eurex_eq",
    "CAA Index": "eurex_eq", "CFA Index": "euronext_eq", "Z A Index": "ice_ftse",
    "SX5E Index": "cash_eu_de", "SX7E Index": "cash_eu_de", "DAX Index": "cash_eu_de", "UKX Index": "cash_uk",
    "NKA Index": "jpx_nikkei", "KMA Index": "krx_kospi", "XPA Index": "asx_spi",
    # STIRs
    "SERA Comdty": "cme_stir", "SFRA Comdty": "cme_stir", "FFA Comdty": "cme_stir",
    "ERA Comdty": "ice_stir_eu", "TKYA Comdty": "ice_stir_eu", "SFIA Comdty": "ice_stir_eu",
    # Bonds
    "USA Comdty": "cbot_rates", "WNA Comdty": "cbot_rates", "UXYA Comdty": "cbot_rates",
    "TYA Comdty": "cbot_rates", "FVA Comdty": "cbot_rates", "TUA Comdty": "cbot_rates",
    "UBA Comdty": "eurex_bond", "RXA Comdty": "eurex_bond", "OEA Comdty": "eurex_bond",
    "DUA Comdty": "eurex_bond", "OATA Comdty": "eurex_bond", "G A Comdty": "ice_gilt",
    # Energy
    "COA Comdty": "ice_brent", "QSA Comdty": "ice_gasoil",
    "CLA Comdty": "nymex_en", "XBA Comdty": "nymex_en", "NGA Comdty": "nymex_en", "HOA Comdty": "nymex_en",
    "FJSA Comdty": "ice_ttf", "MOA Comdty": "ice_eua", "CUAA Comdty": "cbot_grain",
    # Metals
    "GCA Comdty": "comex_me", "SIA Comdty": "comex_me", "PLA Comdty": "comex_me",
    "PAA Comdty": "comex_me", "HGA Comdty": "comex_me", "ALEA Comdty": "comex_me",
    "SCOA Comdty": "sgx_iron",
    # Agriculture
    "C A Comdty": "cbot_grain", "S A Comdty": "cbot_grain", "W A Comdty": "cbot_grain",
    "KWA Comdty": "cbot_grain", "SMA Comdty": "cbot_grain", "BOA Comdty": "cbot_grain", "RRA Comdty": "cbot_grain",
    # Softs
    "KCA Comdty": "ice_coffee", "DFA Comdty": "ice_robusta", "CCA Comdty": "ice_cocoa",
    "SBA Comdty": "ice_sugar", "CTA Comdty": "ice_cotton", "JOA Comdty": "ice_oj",
    "LHA Comdty": "cme_live", "LCA Comdty": "cme_live", "FCA Comdty": "cme_live",
    # FX — all CME Globex
    "ECA Curncy": "cme_fx", "BPA Curncy": "cme_fx", "BRA Curncy": "cme_fx", "CDA Curncy": "cme_fx",
    "SFA Curncy": "cme_fx", "JYA Curncy": "cme_fx", "ADA Curncy": "cme_fx", "NVA Curncy": "cme_fx",
    "PEA Curncy": "cme_fx", "RAA Curncy": "cme_fx", "SIRA Curncy": "cme_fx", "SEA Curncy": "cme_fx",
    "HEA Curncy": "cme_fx", "KOA Curncy": "cme_fx", "NOA Curncy": "cme_fx", "CCA Curncy": "cme_fx",
    "ISA Curncy": "cme_fx", "PPA Curncy": "cme_fx",
}

# FX (all CME Globex) + sensible fallbacks by asset class for anything unmapped.
_ASSET_FALLBACK = {"FX": "cme_fx", "Indices": "cme_eq", "Bonds": "cbot_rates", "STIRs": "cme_stir",
                   "Energy": "nymex_en", "Metals": "comex_me", "Agriculture": "cbot_grain", "Softs": "ice_coffee"}

# ── Holiday / half-day awareness ──────────────────────────────────────────────────────
# Each profile maps to a holiday CALENDAR (a national/exchange group). On a full-closure
# date the product shows no bars ("Closed — <name>"); on a half-day the session is
# truncated to the early-close time and the day's settlement moves to that time.
# These are INDICATIVE regular-market calendars — easy to extend each year below.
CALENDAR_OF = {
    # CME-family futures trade a shortened Globex session on MLK/Presidents'/Juneteenth;
    # ICE US softs follow the NYSE and are FULLY closed those days — hence two US calendars.
    "cme_eq": "US_CME", "cme_fx": "US_CME", "cme_stir": "US_CME", "cbot_rates": "US_CME",
    "cbot_grain": "US_CME", "cme_live": "US_CME", "nymex_en": "US_CME", "comex_me": "US_CME",
    "ice_sugar": "US_ICE", "ice_coffee": "US_ICE", "ice_cocoa": "US_ICE", "ice_cotton": "US_ICE",
    "ice_oj": "US_ICE",
    "eurex_bond": "DE", "eurex_eq": "DE", "cash_eu_de": "DE",
    "euronext_eq": "FR",
    "ice_ftse": "UK", "ice_gilt": "UK", "ice_stir_eu": "UK", "ice_brent": "UK", "ice_gasoil": "UK",
    "ice_ttf": "UK", "ice_eua": "UK", "ice_robusta": "UK", "cash_uk": "UK",
    "jpx_nikkei": "JP", "krx_kospi": "KR", "asx_spi": "AU", "sgx_iron": "SG",
}


def _d(name, *isos):                            # {iso: name} for a list of dates sharing a name
    return {iso: name for iso in isos}


def _span(name, start, end):                    # {iso: name} for an inclusive date range
    s, e, out = date.fromisoformat(start), date.fromisoformat(end), {}
    while s <= e:
        out[s.isoformat()], s = name, s + timedelta(days=1)
    return out


# Indicative regular-market holiday calendars, verified for 2026; 2027 included where
# published (China 2027 is the State Council ESTIMATE — refresh from gov.cn ~Nov 2026).
# calendar -> {ISO date: holiday name}  — market FULLY closed.
CLOSURES: dict[str, dict[str, str]] = {
    "US_CME": {
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Memorial Day", "2026-05-25", "2027-05-31"),
        **_d("Independence Day", "2026-07-03", "2027-07-02"),
        **_d("Labor Day", "2026-09-07", "2027-09-06"),
        **_d("Thanksgiving", "2026-11-26", "2027-11-25"),
        **_d("Christmas Day", "2026-12-25"), **_d("Christmas (observed)", "2027-12-24"),
    },
    "US_ICE": {                                 # NYSE-aligned: also fully shut on the 3 federal Mondays
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Martin Luther King Jr. Day", "2026-01-19", "2027-01-18"),
        **_d("Presidents' Day", "2026-02-16", "2027-02-15"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Memorial Day", "2026-05-25", "2027-05-31"),
        **_d("Juneteenth", "2026-06-19"), **_d("Juneteenth (observed)", "2027-06-18"),
        **_d("Independence Day", "2026-07-03", "2027-07-02"),
        **_d("Labor Day", "2026-09-07", "2027-09-06"),
        **_d("Thanksgiving", "2026-11-26", "2027-11-25"),
        **_d("Christmas Day", "2026-12-25"), **_d("Christmas (observed)", "2027-12-24"),
    },
    "UK": {
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Easter Monday", "2026-04-06", "2027-03-29"),
        **_d("Early May Bank Holiday", "2026-05-04", "2027-05-03"),
        **_d("Spring Bank Holiday", "2026-05-25", "2027-05-31"),
        **_d("Summer Bank Holiday", "2026-08-31", "2027-08-30"),
        **_d("Christmas Day", "2026-12-25"), **_d("Christmas (observed)", "2027-12-27"),
        **_d("Boxing Day (observed)", "2026-12-28", "2027-12-28"),
    },
    "DE": {                                     # Eurex/Xetra — no weekday substitutes; Dec 24 & 31 fully shut
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Easter Monday", "2026-04-06", "2027-03-29"),
        **_d("Labour Day", "2026-05-01"),
        **_d("Whit Monday", "2026-05-25", "2027-05-17"),
        **_d("Christmas Eve", "2026-12-24", "2027-12-24"),
        **_d("Christmas Day", "2026-12-25"),
        **_d("New Year's Eve", "2026-12-31", "2027-12-31"),
    },
    "FR": {                                     # Euronext Paris — no weekday substitutes
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Easter Monday", "2026-04-06", "2027-03-29"),
        **_d("Labour Day", "2026-05-01"),
        **_d("Christmas Day", "2026-12-25"),
    },
    "JP": {
        **_d("New Year", "2026-01-01", "2026-01-02"), **_d("New Year", "2027-01-01"),
        **_d("Coming of Age Day", "2026-01-12", "2027-01-11"),
        **_d("Foundation Day", "2026-02-11", "2027-02-11"),
        **_d("Emperor's Birthday", "2026-02-23", "2027-02-23"),
        **_d("Vernal Equinox", "2026-03-20"), **_d("Vernal Equinox (observed)", "2027-03-22"),
        **_d("Shōwa Day", "2026-04-29", "2027-04-29"),
        **_d("Constitution Day", "2027-05-03"),
        **_d("Greenery Day", "2026-05-04", "2027-05-04"),
        **_d("Children's Day", "2026-05-05", "2027-05-05"),
        **_d("Constitution Day (observed)", "2026-05-06"),
        **_d("Marine Day", "2026-07-20", "2027-07-19"),
        **_d("Mountain Day", "2026-08-11", "2027-08-11"),
        **_d("Respect for the Aged Day", "2026-09-21", "2027-09-20"),
        **_d("Citizens' Holiday", "2026-09-22"),
        **_d("Autumnal Equinox", "2026-09-23", "2027-09-23"),
        **_d("Sports Day", "2026-10-12", "2027-10-11"),
        **_d("Culture Day", "2026-11-03", "2027-11-03"),
        **_d("Labour Thanksgiving", "2026-11-23", "2027-11-23"),
        **_d("Year-end", "2026-12-31", "2027-12-31"),
    },
    "KR": {
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_span("Seollal (Lunar New Year)", "2026-02-16", "2026-02-18"),
        **_d("Seollal (Lunar New Year)", "2027-02-08", "2027-02-09"),
        **_d("Independence Movement Day (observed)", "2026-03-02"),
        **_d("Independence Movement Day", "2027-03-01"),
        **_d("Children's Day", "2026-05-05", "2027-05-05"),
        **_d("Buddha's Birthday (observed)", "2026-05-25"),
        **_d("Buddha's Birthday", "2027-05-13"),
        **_d("Local Election Day", "2026-06-03"),
        **_d("Memorial Day (observed)", "2027-06-07"),
        **_d("Liberation Day (observed)", "2026-08-17", "2027-08-16"),
        **_span("Chuseok", "2026-09-24", "2026-09-25"), **_d("Chuseok (substitute)", "2026-09-28"),
        **_span("Chuseok", "2027-09-14", "2027-09-16"),
        **_d("National Foundation Day (observed)", "2026-10-05", "2027-10-04"),
        **_d("Hangeul Day", "2026-10-09"), **_d("Hangeul Day (observed)", "2027-10-11"),
        **_d("Christmas Day", "2026-12-25"), **_d("Christmas (observed)", "2027-12-27"),
        **_d("Year-end (KRX)", "2026-12-31", "2027-12-31"),
    },
    "AU": {                                     # ASX (Sydney/NSW); no ANZAC substitute taken
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_d("Australia Day", "2026-01-26", "2027-01-26"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Easter Monday", "2026-04-06", "2027-03-29"),
        **_d("King's Birthday", "2026-06-08", "2027-06-14"),
        **_d("Christmas Day", "2026-12-25"), **_d("Christmas (observed)", "2027-12-27"),
        **_d("Boxing Day (observed)", "2026-12-28", "2027-12-28"),
    },
    "SG": {                                     # in-lieu Monday only for SUNDAY holidays (not Saturday)
        **_d("New Year's Day", "2026-01-01", "2027-01-01"),
        **_span("Chinese New Year", "2026-02-17", "2026-02-18"),
        **_d("Chinese New Year (in lieu)", "2027-02-08"),
        **_d("Hari Raya Puasa", "2027-03-10"),
        **_d("Good Friday", "2026-04-03", "2027-03-26"),
        **_d("Labour Day", "2026-05-01"),
        **_d("Hari Raya Haji", "2026-05-27", "2027-05-17"),
        **_d("Vesak Day (in lieu)", "2026-06-01"), **_d("Vesak Day", "2027-05-20"),
        **_d("National Day (in lieu)", "2026-08-10"), **_d("National Day", "2027-08-09"),
        **_d("Deepavali (in lieu)", "2026-11-09"), **_d("Deepavali", "2027-10-28"),
        **_d("Christmas Day", "2026-12-25"),
    },
    "CN": {                                     # DCE / mainland — 2026 OFFICIAL; 2027 ESTIMATE (refresh ~Nov 2026)
        **_span("New Year", "2026-01-01", "2026-01-03"),
        **_span("Spring Festival", "2026-02-15", "2026-02-23"),
        **_span("Qingming", "2026-04-04", "2026-04-06"),
        **_span("Labour Day", "2026-05-01", "2026-05-05"),
        **_span("Dragon Boat Festival", "2026-06-19", "2026-06-21"),
        **_span("Mid-Autumn Festival", "2026-09-25", "2026-09-27"),
        **_span("National Day", "2026-10-01", "2026-10-07"),
        **_span("New Year (est.)", "2027-01-01", "2027-01-03"),
        **_span("Spring Festival (est.)", "2027-02-05", "2027-02-13"),
        **_span("Qingming (est.)", "2027-04-03", "2027-04-05"),
        **_span("Labour Day (est.)", "2027-05-01", "2027-05-05"),
        **_span("Dragon Boat Festival (est.)", "2027-06-09", "2027-06-11"),
        **_d("Mid-Autumn Festival (est.)", "2027-09-15"),
        **_span("National Day (est.)", "2027-10-01", "2027-10-07"),
    },
}
# calendar -> {ISO date: ("HH:MM" early-close in exchange-LOCAL time, name)} — half-day.
# Times are indicative: CME early closes vary by product (~12:00–12:30 CT); ICE softs 13:00 ET.
HALF_DAYS: dict[str, dict[str, tuple[str, str]]] = {
    "US_CME": {
        "2026-01-19": ("12:00", "Martin Luther King Jr. Day"),
        "2026-02-16": ("12:00", "Presidents' Day"),
        "2026-06-19": ("12:00", "Juneteenth"),
        "2026-11-27": ("12:15", "Day after Thanksgiving"),
        "2026-12-24": ("12:15", "Christmas Eve"),
        "2027-01-18": ("12:00", "Martin Luther King Jr. Day"),
        "2027-02-15": ("12:00", "Presidents' Day"),
        "2027-06-18": ("12:00", "Juneteenth (observed)"),
        "2027-11-26": ("12:15", "Day after Thanksgiving"),
        "2027-12-23": ("12:15", "day before Christmas (observed)"),
    },
    "US_ICE": {                                 # 1:00 PM ET standard softs early close
        "2026-11-27": ("13:00", "Day after Thanksgiving"),
        "2026-12-24": ("13:00", "Christmas Eve"),
        "2027-11-26": ("13:00", "Day after Thanksgiving"),
        "2027-12-23": ("13:00", "day before Christmas (observed)"),
    },
    "UK": {
        "2026-12-24": ("12:30", "Christmas Eve"), "2026-12-31": ("12:30", "New Year's Eve"),
        "2027-12-24": ("12:30", "Christmas Eve"), "2027-12-31": ("12:30", "New Year's Eve"),
    },
    "DE": {},
    "FR": {
        "2026-12-24": ("14:05", "Christmas Eve"), "2026-12-31": ("14:05", "New Year's Eve"),
        "2027-12-24": ("14:05", "Christmas Eve"), "2027-12-31": ("14:05", "New Year's Eve"),
    },
    "JP": {}, "KR": {},
    "AU": {
        "2026-12-24": ("14:10", "Christmas Eve"), "2026-12-31": ("14:10", "New Year's Eve"),
        "2027-12-24": ("14:10", "Christmas Eve"), "2027-12-31": ("14:10", "New Year's Eve"),
    },
    "SG": {
        "2026-02-16": ("12:00", "Chinese New Year's Eve"),
        "2026-12-24": ("12:00", "Christmas Eve"), "2026-12-31": ("12:00", "New Year's Eve"),
        "2027-02-05": ("12:00", "Chinese New Year's Eve"),
        "2027-12-24": ("12:00", "Christmas Eve"), "2027-12-31": ("12:00", "New Year's Eve"),
    },
    "CN": {},
}

TZ_CHOICES = {                              # label -> IANA tz, for the page selector
    "New York (ET)": "America/New_York", "London (GMT/BST)": "Europe/London",
    "Chicago (CT)": "America/Chicago", "São Paulo (BRT)": "America/Sao_Paulo",
    "UTC": "UTC", "Dubai (GST)": "Asia/Dubai", "Singapore (SGT)": "Asia/Singapore",
    "Tokyo (JST)": "Asia/Tokyo", "Sydney (AEDT)": "Australia/Sydney",
}


def profile_id(ticker: str, asset: str = "") -> str:
    return PROFILE_OF.get(ticker) or _ASSET_FALLBACK.get(asset, "cme_eq")


def exchange_of(ticker: str, asset: str = "") -> str:
    """The listing-exchange label for a product (e.g. 'NYMEX (CME)', 'ICE Europe')."""
    return PROFILES[profile_id(ticker, asset)][0]


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _blocks_minutes(blocks):
    out = []
    for a, b in blocks:
        s, e = _hm(a), _hm(b)
        length = (e - s) % 1440 or 1440
        out.append((s, length))
    return out


def _offset_min(tzname: str, ref_date: date) -> int:
    dt = datetime(ref_date.year, ref_date.month, ref_date.day, 12, tzinfo=ZoneInfo(tzname))
    return int(dt.utcoffset().total_seconds() // 60)


def _to_ref(blocks, exch_tz: str, ref_tz: str, ref_date: date):
    """Convert local-time blocks to (start_hour, end_hour) segments on the ref-tz 0–24h axis."""
    shift = _offset_min(ref_tz, ref_date) - _offset_min(exch_tz, ref_date)
    segs = []
    for s, length in _blocks_minutes(blocks):
        start = (s + shift) % 1440
        end = start + length
        if end <= 1440:
            segs.append((start / 60, end / 60))
        else:                               # wraps past midnight on the ref axis → two pieces
            segs.append((start / 60, 24.0))
            segs.append((0.0, (end - 1440) / 60))
    return segs


def _point_min(hhmm: str, exch_tz: str, target_tz: str, ref_date: date) -> int:
    shift = _offset_min(target_tz, ref_date) - _offset_min(exch_tz, ref_date)
    return (_hm(hhmm) + shift) % 1440


def _to_ref_point(hhmm: str, exch_tz: str, ref_tz: str, ref_date: date) -> float:
    return _point_min(hhmm, exch_tz, ref_tz, ref_date) / 60


def _mmhh(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def _fmt(blocks) -> str:
    return " + ".join(f"{a}–{b}" for a, b in blocks)


def _fmt_in_tz(blocks, exch_tz: str, target_tz: str, ref_date: date) -> str:
    """Blocks formatted as 'HH:MM–HH:MM' in `target_tz` (for the tooltip's ET line)."""
    shift = _offset_min(target_tz, ref_date) - _offset_min(exch_tz, ref_date)
    out = []
    for s, length in _blocks_minutes(blocks):
        start = (s + shift) % 1440
        out.append(f"{_mmhh(start)}–{_mmhh((start + length) % 1440)}")
    return " + ".join(out)


def _truncate(blocks, ec: str):
    """Cap a session at an early-close time `ec` (HH:MM, exchange-local). Truncates the
    block the close falls inside; keeps earlier blocks; drops blocks after the close.
    Used for half-days — close times only move EARLIER, so this is conservative."""
    ec_m = _hm(ec)
    out = []
    for a, b in blocks:
        s = _hm(a)
        length = (_hm(b) - s) % 1440 or 1440
        off = (ec_m - s) % 1440          # minutes from this block's open to the early close
        if 0 < off < length:
            out.append((a, ec))          # early close lands inside this block → truncate it
        elif off >= length:
            out.append((a, b))           # block already shut before the early close → keep
        # off == 0 → block opens exactly at the close → nothing left, drop it
    return out


def holiday_status(ticker: str, ref_date: date, asset: str = "") -> dict:
    """Is this product's exchange shut (or early-closing) on `ref_date`?"""
    cal = CALENDAR_OF.get(profile_id(ticker, asset))
    iso = ref_date.isoformat()
    closed = CLOSURES.get(cal, {}).get(iso) if cal else None
    half = HALF_DAYS.get(cal, {}).get(iso) if cal else None
    return {"closed": closed, "half_day": half}      # closed: name|None ; half_day: (ec,name)|None


def day_segments(ticker: str, ref_tz: str, ref_date: date, asset: str = "") -> dict:
    """Everything the timeline needs for one product, in the chosen reference timezone."""
    pid = profile_id(ticker, asset)
    label, exch_tz, full, liquid = PROFILES[pid]
    settle_local = SETTLE.get(pid)
    liquid_exch = LIQUID_LABEL.get(pid, label)      # liquid window's venue label (may differ)
    ET = "America/New_York"

    cal = CALENDAR_OF.get(pid)
    iso = ref_date.isoformat()
    closed = CLOSURES.get(cal, {}).get(iso) if cal else None
    half = HALF_DAYS.get(cal, {}).get(iso) if cal else None

    out = {"exchange": label, "exch_tz": exch_tz, "liquid_exch": liquid_exch,
           "closed": closed, "half_day": half[1] if half else None,
           "early_close": half[0] if half else None}

    if closed:                                       # market shut → no bars, no settlement
        out.update(full=[], liquid=[], full_local="—", liquid_local="—",
                   full_et="—", liquid_et="—", settle=None, settle_local=None, settle_et=None)
        return out

    if half:                                         # half-day → truncate to the early close
        ec = half[0]
        full = _truncate(full, ec)
        liquid = _truncate(liquid, ec)
        if settle_local:
            settle_local = ec                        # settlement taken at the early close (indicative)

    # The liquid window may be governed by a different market's calendar (e.g. iron ore's
    # liquidity is the Dalian session) — if THAT market is shut today, drop the liquid bar.
    lcal = LIQUID_CAL.get(pid)
    liquid_closed = CLOSURES.get(lcal, {}).get(iso) if lcal else None
    if liquid_closed:
        liquid = []

    out.update(
        full=_to_ref(full, exch_tz, ref_tz, ref_date),
        liquid=_to_ref(liquid, exch_tz, ref_tz, ref_date),
        full_local=_fmt(full), liquid_local=(_fmt(liquid) if liquid else f"{liquid_exch} closed"),
        full_et=_fmt_in_tz(full, exch_tz, ET, ref_date),
        liquid_et=(_fmt_in_tz(liquid, exch_tz, ET, ref_date) if liquid else "—"),
        settle=_to_ref_point(settle_local, exch_tz, ref_tz, ref_date) if settle_local else None,
        settle_local=settle_local,
        settle_et=_mmhh(_point_min(settle_local, exch_tz, ET, ref_date)) if settle_local else None,
    )
    return out


def is_open(segments_full, now_hour: float) -> bool:
    return any(a <= now_hour < b for a, b in segments_full)
