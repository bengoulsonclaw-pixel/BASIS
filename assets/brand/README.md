# BASIS brand assets

**BASIS — Analysis · Strategy · Indicators.** The brand for the Strategy Monitor app.

The mark is a *convergence* play-button: a grey bracket on the left, a silver edge
(top) and a gold edge (bottom) converging on a gold dot — analysis and strategy
resolving into a single indicator/decision.

## Files
| File | What it is | Used by |
|------|------------|---------|
| `basis-icon.svg` | Rounded dark app-icon tile — mark only (scalable) | source of truth for the app icon |
| `basis-icon-full.svg` | App-icon tile WITH the "BASIS" wordmark | the large sizes (256/128) of `basis.ico` |
| `basis-social.svg` | Mark + wordmark on a full-bleed dark square | source of the social avatars |
| `basis-icon-512.png`, `-256.png` | Raster app icon (mark only) | Streamlit `page_icon` (browser tab) |
| `basis-social-1024/512/400.png` | Social avatar (mark + wordmark), circle-crop safe | X / Instagram / LinkedIn profile pictures |
| `basis.ico` | Multi-res desktop icon: full logo @256/128, mark @64→16 | the **BASIS** desktop shortcut |
| `basis-header-dark.svg` | Mark + wordmark, silver→gold (for dark bg) | reference |
| `basis-header-light.svg` | Mark + wordmark, slate→gold (for light bg) | reference |
| `build_icons.py` | Regenerates every PNG + the `.ico` from the SVG sources | run in the repo venv |

The live app does **not** load the header SVGs from disk — `src/brand.py` renders
the mark + wordmark inline so it re-colours instantly with the theme toggle.

**Regenerating rasters:** edit an SVG, then `python assets/brand/build_icons.py` (uses the
bundled Playwright Chromium). The wordmark is only legible above ~64 px, so the app icon /
favicon and the small `.ico` frames stay mark-only; the full `❯ BASIS` logo appears at
256/128 px (desktop shortcut) and in the social avatars.

## Palette
| Token | Dark | Light |
|-------|------|-------|
| canvas | `#161616` | `#FFFFFF` |
| surface | `#1E1F22` | `#F6F7F9` |
| sidebar | `#1B1C1F` | `#F4F5F7` |
| text | `#ECEEF1` | `#2A2E34` |
| muted | `#9AA0A8` | `#5A616A` |
| gold (primary) | `#F5C518` | `#C8901A` |
| wordmark | silver→gold `#EEF0F3 · #C0C5CC · #CBA53C · #F4CC3A` | slate→gold `#3D434B · #C39A33 · #E0A81C` |

Shared mark gradients: gold `#F6D24A → #D9971C`, silver `#F2F4F6 → #9CA2A9`.

## Wiring
- `src/brand.py` — palettes, inline logo, theme CSS, masthead + sun/moon toggle, `themed_dataframe`.
- `app.py` — `brand.apply()` (after `set_page_config`), `brand.sidebar_logo()`, `brand.masthead()`.
- `.streamlit/config.toml` — the dark theme as the first-paint default.

## Note
The legacy "B.A.S.I.S" logo (gold arrow over candlesticks, dotted lettering) was
**retired** in this rebrand and is intentionally not part of this suite.
