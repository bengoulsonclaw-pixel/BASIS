"""Seasonal weather-premium windows for agricultural commodities.

In a crop's growing / weather-sensitive months, options carry a structural premium
(implied vol >> trailing realized) that compensates for genuine weather risk — not
pure mispricing. So a "rich vol → sell" flag on an ag *in season* should be read as
"premium for real risk", not a free sale. These windows are heuristic
(northern-hemisphere row crops + the obvious softs); tune the months/tickers as needed.

Imported both as `from seasonal import ...` (the report subprocesses, src on sys.path)
and `from src import seasonal` (the app).
"""
from __future__ import annotations

# ticker -> (months with a structurally elevated weather premium, short label)
_WEATHER = {
    "C A Comdty":  ({5, 6, 7, 8},  "Jun–Aug US weather/pollination"),
    "S A Comdty":  ({6, 7, 8, 9},  "Jun–Sep US weather"),
    "BOA Comdty":  ({6, 7, 8, 9},  "Jun–Sep soy weather"),
    "SMA Comdty":  ({6, 7, 8, 9},  "Jun–Sep soy weather"),
    "W A Comdty":  ({4, 5, 6, 7},  "Apr–Jul wheat weather"),
    "KWA Comdty":  ({4, 5, 6, 7},  "Apr–Jul wheat weather"),
    "RRA Comdty":  ({5, 6, 7, 8},  "May–Aug growing season"),
    "KCA Comdty":  ({6, 7, 8},     "Jun–Aug Brazil frost risk"),
    "CTA Comdty":  ({6, 7, 8},     "Jun–Aug US cotton weather"),
    "JOA Comdty":  ({8, 9, 10},    "Aug–Oct hurricane season"),
}


def weather_note(ticker: str, month: int) -> str:
    """Short label if `ticker` is an ag in its seasonal weather-premium window for
    `month` (1–12); '' otherwise."""
    info = _WEATHER.get(ticker)
    return info[1] if info and month in info[0] else ""


def is_seasonal(ticker: str, month: int) -> bool:
    return bool(weather_note(ticker, month))
