"""World-clock cities, current weather, and (optional) per-city background photos.

Time is rendered live client-side (JS, in app.py) from each city's IANA timezone.
Weather + day/night come from Open-Meteo (free, no key, one call). City photos come
from Unsplash (needs a free Access Key in data/unsplash_key.txt or $UNSPLASH_KEY) —
a daytime and a night shot per city, picked by the live is_day flag. Stdlib only.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

_KEY_FILE = Path(__file__).resolve().parents[1] / "data" / "unsplash_key.txt"

# name · IANA timezone (JS Intl uses this) · lat · lon
CITIES = [
    {"name": "Chicago",   "tz": "America/Chicago",   "lat": 41.85,  "lon": -87.65},
    {"name": "New York",  "tz": "America/New_York",  "lat": 40.71,  "lon": -74.01},
    {"name": "São Paulo", "tz": "America/Sao_Paulo", "lat": -23.55, "lon": -46.63},
    {"name": "London",    "tz": "Europe/London",     "lat": 51.51,  "lon": -0.13},
    {"name": "Dubai",     "tz": "Asia/Dubai",        "lat": 25.20,  "lon": 55.27},
    {"name": "Singapore", "tz": "Asia/Singapore",    "lat": 1.35,   "lon": 103.82},
]

# ---- clean inline-SVG weather glyphs (replace the muddy emoji) --------------
_IC = "width='20' height='20' viewBox='0 0 24 24' fill='none'"
_CLOUD_HI = ("<path d='M6.8 14a4 4 0 0 1-.3-8 5.2 5.2 0 0 1 9.9 1.2A3.5 3.5 0 0 1 16.2 14z' "
             "fill='#d4dae3'/>")                       # cloud high, leaving room for drops below
_CLOUD_MID = ("<path d='M6.5 18a4.2 4.2 0 0 1-.3-8.38 5.4 5.4 0 0 1 10.3 1.3A3.7 3.7 0 0 1 16.5 18z' "
              "fill='#d4dae3'/>")
_SVG = {
    "sun": (f"<svg {_IC} stroke='#ffce3a' stroke-width='1.8' stroke-linecap='round'>"
            "<circle cx='12' cy='12' r='3.8' fill='#ffce3a' stroke='none'/>"
            "<path d='M12 2.8v2.4M12 18.8v2.4M2.8 12h2.4M18.8 12h2.4M5.5 5.5l1.7 1.7"
            "M16.8 16.8l1.7 1.7M18.5 5.5l-1.7 1.7M7.2 16.8l-1.7 1.7'/></svg>"),
    "moon": (f"<svg {_IC}><path d='M20.5 14.8A8.2 8.2 0 0 1 9.2 3.5 7.3 7.3 0 1 0 20.5 14.8z' "
             "fill='#e7ecf6'/></svg>"),
    "partly": (f"<svg {_IC}><circle cx='16.5' cy='7.3' r='3' fill='#ffce3a'/>"
               "<g stroke='#ffce3a' stroke-width='1.3' stroke-linecap='round'>"
               "<path d='M16.5 1.8v1.5M21.7 7.3H20.2M20.4 3.4l-1 1M20.4 11.2l-1-1'/></g>"
               f"{_CLOUD_HI}</svg>"),
    "cloud": f"<svg {_IC}>{_CLOUD_MID}</svg>",
    "fog": (f"<svg {_IC}>{_CLOUD_HI}<g stroke='#aeb6c2' stroke-width='1.5' stroke-linecap='round'>"
            "<path d='M6 18.5h12M8 21h8'/></g></svg>"),
    "rain": (f"<svg {_IC}>{_CLOUD_HI}<g stroke='#74b3ff' stroke-width='1.7' stroke-linecap='round'>"
             "<path d='M8.5 16.6l-1 2.6M12 16.6l-1 2.6M15.5 16.6l-1 2.6'/></g></svg>"),
    "storm": (f"<svg {_IC}>{_CLOUD_HI}<polygon points='13,12.6 9.6,17.9 11.9,17.9 10.9,21.6 "
              "15.1,15.4 12.5,15.4 13.7,12.6' fill='#ffce3a'/></svg>"),
    "snow": (f"<svg {_IC}>{_CLOUD_HI}<g fill='#fff'><circle cx='9' cy='17.6' r='1.1'/>"
             "<circle cx='12.5' cy='19.6' r='1.1'/><circle cx='15.6' cy='17.3' r='1.1'/></g></svg>"),
}
_STORM = {95, 96, 99}
_RAIN = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
_SNOW = {71, 73, 75, 77, 85, 86}


def _icon(code, is_day=True) -> str:
    """A clean inline-SVG weather glyph for the card (day/night aware for clear skies)."""
    if code in _STORM:
        return _SVG["storm"]
    if code in _RAIN:
        return _SVG["rain"]
    if code in _SNOW:
        return _SVG["snow"]
    if code in (45, 48):
        return _SVG["fog"]
    if code == 0:
        return _SVG["sun"] if is_day else _SVG["moon"]
    if code in (1, 2):
        return _SVG["partly"] if is_day else _SVG["cloud"]
    return _SVG["cloud"]


def _rain_uri() -> str:
    """A tiling SVG of scattered short droplets at varied positions — reads as real rain
    rather than the regular diagonal lines a CSS gradient makes. Animated via
    background-position in app.py's clock CSS."""
    streaks = [(10, 6, 9, 13), (26, 14, 25, 21), (42, 8, 41, 15), (54, 18, 53, 25),
               (16, 30, 15, 37), (34, 38, 33, 45), (48, 34, 47, 41), (8, 48, 7, 55),
               (28, 54, 27, 61), (44, 52, 43, 59), (20, 64, 19, 71), (38, 70, 37, 77),
               (56, 62, 55, 69)]
    lines = "".join(f"<line x1='{a}' y1='{b}' x2='{c}' y2='{d}'/>" for a, b, c, d in streaks)
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='60' height='80' viewBox='0 0 60 80'>"
           "<g stroke='rgb(205,224,255)' stroke-width='1.1' stroke-linecap='round' opacity='0.55'>"
           + lines + "</g></svg>")
    return "data:image/svg+xml," + urllib.parse.quote(svg)


RAIN_URI = _rain_uri()


def _effect(code) -> str:
    """A weather overlay class for the card: 'rain', 'snow', or '' (none)."""
    if code in _RAIN:
        return "rain"
    if code in _SNOW:
        return "snow"
    return ""


def fetch_weather() -> list:
    """[{temp:int|None, icon, is_day:bool, effect:str}, ...] aligned with CITIES —
    current °C, condition emoji, day/night flag and a precip-effect class from
    Open-Meteo. One HTTP call; safe placeholders on any error."""
    lats = ",".join(str(c["lat"]) for c in CITIES)
    lons = ",".join(str(c["lon"]) for c in CITIES)
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}"
           "&current=temperature_2m,weather_code,is_day")
    out = []
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        if isinstance(data, dict):            # single-location returns an object
            data = [data]
        for d in data:
            cur = d.get("current") or {}
            t = cur.get("temperature_2m")
            code = cur.get("weather_code")
            out.append({"temp": round(t) if isinstance(t, (int, float)) else None,
                        "icon": _icon(code, bool(cur.get("is_day", 1))),
                        "is_day": bool(cur.get("is_day", 1)),
                        "effect": _effect(code),
                        "clouds": code not in (0, 1)})
    except Exception:
        out = []
    while len(out) < len(CITIES):             # pad on partial/failed response
        out.append({"temp": None, "icon": _SVG["cloud"], "is_day": True, "effect": "", "clouds": False})
    return out[:len(CITIES)]


# --- optional city background photos (Unsplash) ----------------------------
def load_key() -> str:
    """Unsplash Access Key from $UNSPLASH_KEY or data/unsplash_key.txt; '' if none."""
    k = (os.environ.get("UNSPLASH_KEY") or "").strip()
    if k:
        return k
    try:
        return _KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _unsplash(query: str, key: str) -> str | None:
    url = ("https://api.unsplash.com/search/photos?per_page=1&orientation=landscape"
           "&content_filter=high&query=" + urllib.parse.quote(query)
           + "&client_id=" + key)
    try:
        req = urllib.request.Request(url, headers={"Accept-Version": "v1"})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        res = data.get("results") or []
        if res:
            urls = res[0].get("urls") or {}
            return urls.get("regular") or urls.get("small")
    except Exception:
        pass
    return None


def fetch_city_photos(key: str) -> dict:
    """{city_name: {'day': url|None, 'night': url|None}} via Unsplash, or {} with no key."""
    if not key:
        return {}
    out = {}
    for c in CITIES:
        out[c["name"]] = {
            "day":   _unsplash(c["name"] + " city skyline daytime", key),
            "night": _unsplash(c["name"] + " city skyline night", key),
        }
    return out
