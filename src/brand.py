"""BASIS brand kit for the Strategy Monitor dashboard.

Single source of truth for the rebrand: the colour palettes, the inline logo
(convergence mark + BASIS wordmark), the dark/light theme CSS, and the masthead
with its sun/moon toggle.

app.py wires it in three places:
    brand.apply()          # once, right after st.set_page_config — injects theme CSS
    brand.sidebar_logo()   # top of the sidebar
    brand.masthead(...)    # first element in the main column (carries the toggle)

The toggle flips st.session_state["basis_theme"] between "dark" and "light" and
reruns; apply() then re-skins everything from the active palette, so a single CSS
template covers both modes (and fully overrides the config.toml dark default).
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

import streamlit as st

try:                                   # charts are re-themed only when altair is present
    import altair as alt
except Exception:                      # pragma: no cover
    alt = None

_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "brand"
ICON_PNG = _ASSETS / "basis-icon-512.png"          # st.set_page_config(page_icon=...)
_PREF_FILE = Path(__file__).resolve().parent.parent / "data" / "ui_prefs.json"  # remembers theme

_FONT = ("'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
         "'Helvetica Neue', Arial, sans-serif")
_MONO = "'IBM Plex Mono', Consolas, 'SF Mono', Menlo, monospace"

# --- palettes --------------------------------------------------------------
# `word` = the BASIS wordmark gradient stops (left→right). Dark theme runs
# silver→gold on a dark canvas; light theme runs slate→gold on white.
# Tuned to XP Investimentos' own brand palette (sampled from the XP/Eurex
# brochure): yellow #F5C518 · rich black #0A0A0A · dark grey #2A2A2A ·
# off-white #F6F6F6 · white #FFFFFF. Dark = XP rich black + yellow; light = the
# white "brochure" look (white / off-white / near-black + the same XP yellow).
# Both palettes are LAYERED rather than flat: the canvas sits a clear step away from the
# card surfaces and the sidebar, so pages read as zones instead of one solid black/white
# sheet. Dark = slate greys lifted off pure black; light = a grey canvas with white cards.
# 2026-07 "terminal" redesign (Claude Design handoff, design_kit/handoff/README.md):
# darker canvas, hairline separations, radius 0, no shadows. Key mapping vs the
# handoff tokens: border=--line, border_soft=--hair, btn_border/faint=--line2/--faint,
# gold_soft=--goldWash, label_ring=--goldLine. On LIGHT, gold text/borders darken to
# #A87A0C (raw #F5C518 fails contrast on white); washes keep the raw gold.
DARK = dict(
    name="dark",
    canvas="#0F1216", glass="rgba(22,26,32,.85)", surface="#161A20", surface2="#1C212A",
    border="#232935", border_soft="rgba(255,255,255,.05)",
    btn="transparent", btn_border="#333B49", btn_hover="#1C212A",
    btn_gold="rgba(245,197,24,.10)", btn_gold_hover="rgba(245,197,24,.16)",
    label_ring="rgba(245,197,24,.34)",
    # Secondary-text floors RAISED 2026-08-15 (Ben: "barely readable... across the
    # entire app") — was text_dim/caption #C3CAD3, faint #9AA4B0, tagline #8A94A1.
    # Never re-dim below these.
    text="#E7EAEE", text_dim="#CDD3DB", caption="#CDD3DB", faint="#AEB7C2",
    sidebar="#0A0C10",
    gold="#F5C518", gold_deep="#D9971C", gold_soft="rgba(245,197,24,.10)",
    bracket="#9FA9B5", tagline="#9FA9B5",
    cal_ink="#B6BFC9", cal_ink_strong="#E7EAEE",
    word=(("0", "#EEF0F3"), ("0.46", "#C0C5CC"), ("0.70", "#CBA53C"), ("1", "#F4CC3A")),
)
LIGHT = dict(
    name="light",
    canvas="#EDF0F3", glass="rgba(255,255,255,.85)", surface="#FFFFFF", surface2="#F4F6F9",
    border="#D2D7DE", border_soft="rgba(0,0,0,.07)",
    btn="transparent", btn_border="#BEC5CE", btn_hover="#F4F6F9",
    btn_gold="rgba(245,197,24,.16)", btn_gold_hover="rgba(245,197,24,.24)",
    label_ring="rgba(168,122,12,.38)",
    text="#12161B", text_dim="#3F4854", caption="#3F4854", faint="#5B6570",
    sidebar="#E4E8EC",
    gold="#A87A0C", gold_deep="#8A6208", gold_soft="rgba(245,197,24,.16)",
    bracket="#6A737E", tagline="#6A737E",
    cal_ink="#4A545F", cal_ink_strong="#12161B",
    word=(("0", "#12161B"), ("0.58", "#12161B"), ("0.73", "#8A6208"), ("1", "#A87A0C")),
)
PALETTES = {"dark": DARK, "light": LIGHT}


def theme() -> str:
    return st.session_state.get("basis_theme", "dark")


def palette() -> dict:
    return PALETTES[theme()]


def _load_pref() -> str | None:
    try:
        val = json.loads(_PREF_FILE.read_text()).get("theme")
        return val if val in PALETTES else None
    except Exception:
        return None


def _save_pref(theme_name: str) -> None:
    try:
        _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PREF_FILE.write_text(json.dumps({"theme": theme_name}))
    except Exception:
        pass


def chart_colors(pal: dict | None = None) -> dict:
    """Palette-driven Altair mark colours, so lines/marks read on either canvas."""
    pal = pal or palette()
    dark = pal["name"] == "dark"
    return {
        "ink":    pal["text"],                       # primary lines: price, net, oscillator, slow MA
        "muted":  pal["text_dim"],                   # bands, medians, parity, mean
        "series": "#5B9BF0" if dark else "#1F5FA8",  # blue
        "long":   "#46C58A" if dark else "#1F7A44",  # green
        "short":  "#EC6A57" if dark else "#C62828",  # red
        "accent": pal["gold"],                       # gold headline line
        "halo":   pal["canvas"],                     # outline behind the gold line (matches canvas)
    }


# --- logo ------------------------------------------------------------------
def _stops(stops) -> str:
    return "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in stops)


# The masthead lockup stacks the BASIS wordmark over the tagline, both forced to
# the SAME width (shared textLength) so their left/right edges align. That width is
# BASIS' natural advance at font-size 76, so the wordmark keeps normal spacing — it
# is ENLARGED, not letter-spread — and the tagline is sized to fill the same width
# beneath it. The lockup's on-screen size is set by the height passed to header_svg.
_WORD_W = 224           # BASIS natural advance width (viewBox units) = shared width
# Advance width of "ANALYSIS · STRATEGIES · INDICATORS" per 1u of font-size in this
# font stack (measured 16.4 for the old STRATEGY wording, scaled for the two extra
# glyphs); lets us size the tagline so its natural width ≈ _WORD_W, leaving
# textLength only a hair of spacing to nudge.
_TAG_ADVANCE = 17.4


def mark_svg(pal: dict, height: int = 86) -> str:
    """Just the convergence ❯ mark (no wordmark) — the landing hero anchors it at
    the page's left while the enlarged BASIS text centres independently."""
    u = pal["name"] + "mk"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-6 -8 172 124" height="{height}" '
        f'role="img" aria-label="BASIS mark" style="display:block">'
        f'<defs>'
        f'<linearGradient id="mgo_{u}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#F6D24A"/><stop offset="1" stop-color="#D9971C"/></linearGradient>'
        f'<linearGradient id="msi_{u}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#F2F4F6"/><stop offset="1" stop-color="#9CA2A9"/></linearGradient>'
        f'</defs>'
        f'<path d="M8 2 L150 52" fill="none" stroke="url(#msi_{u})" stroke-width="12" stroke-linecap="round"/>'
        f'<path d="M8 102 L150 52" fill="none" stroke="url(#mgo_{u})" stroke-width="12" stroke-linecap="round"/>'
        f'<circle cx="150" cy="52" r="11" fill="url(#mgo_{u})"/></svg>'
    )


def word_svg(pal: dict, height: int = 64) -> str:
    """Just the BASIS wordmark — the exact gradient/geometry of the sidebar lockup
    (a CSS text-gradient approximation on the landing hero read as the wrong
    colours; the SVG is the single source of the brand look)."""
    u = pal["name"] + "wd"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 96" height="{height}" '
        f'role="img" aria-label="BASIS" style="display:block;font-family:{_FONT}">'
        f'<defs><linearGradient id="bww_{u}" x1="0" y1="0" x2="1" y2="0">'
        f'{_stops(pal["word"])}</linearGradient></defs>'
        f'<text x="4" y="76" font-family="{_FONT}" font-size="76" font-weight="700" '
        f'letter-spacing="2.5" fill="url(#bww_{u})">BASIS</text>'
        f'</svg>'
    )


def header_svg(pal: dict, height: int = 34, tagline: bool = False) -> str:
    """Convergence mark + BASIS wordmark, themed for `pal`.

    tagline=False -> mark + wordmark only (used in the sidebar).
    tagline=True  -> stacked lockup: the wordmark with
    "Analysis · Strategies · Indicators" directly beneath it, both pinned to the
    SAME width via a shared `textLength`, so their left and right edges align and
    they scale together as one unit.
    """
    u = pal["name"]
    defs = (
        f'<defs>'
        f'<linearGradient id="bw_{u}" x1="0" y1="0" x2="1" y2="0">{_stops(pal["word"])}</linearGradient>'
        f'<linearGradient id="go_{u}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#F6D24A"/><stop offset="1" stop-color="#D9971C"/></linearGradient>'
        f'<linearGradient id="si_{u}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#F2F4F6"/><stop offset="1" stop-color="#9CA2A9"/></linearGradient>'
        f'</defs>'
    )
    # Clean bold ❯ mark, matching the app icon: silver edge (top) + gold edge (bottom) converging
    # on a gold node. The old measurement bracket was dropped for the simpler, bolder icon look,
    # so the sidebar logo now reads the same as the icon everywhere.
    mark = (
        f'<g transform="translate(14,26) scale(0.62)">'
        f'<path d="M8 2 L150 52" fill="none" stroke="url(#si_{u})" stroke-width="12" stroke-linecap="round"/>'
        f'<path d="M8 102 L150 52" fill="none" stroke="url(#go_{u})" stroke-width="12" stroke-linecap="round"/>'
        f'<circle cx="150" cy="52" r="11" fill="url(#go_{u})"/></g>'
    )
    if not tagline:
        # viewBox trimmed to the drawn content (mark ends ~x107, wordmark ~x368) so no dead
        # slack on the right; width:100% + height:auto lets it fill a narrow container
        # (the sidebar) at full size and shrink cleanly rather than letterbox.
        nat_w = round(height * 368 / 120)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 368 120" '
            f'height="{height}" role="img" aria-label="BASIS" '
            f'style="display:block;width:100%;max-width:{nat_w}px;height:auto;'
            f'font-family:{_FONT}">'
            f'{defs}{mark}'
            f'<text x="132" y="82" font-family="{_FONT}" font-size="76" font-weight="700" '
            f'letter-spacing="2.5" fill="url(#bw_{u})">BASIS</text>'
            f'</svg>'
        )

    w = _WORD_W
    f_sub = round(w / _TAG_ADVANCE, 1)          # tagline size so its natural width ≈ w
    vbw = 132 + w + 16
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 18 {vbw:.0f} 86" '
        f'height="{height}" role="img" '
        f'aria-label="BASIS — Analysis · Strategies · Indicators" '
        f'style="display:block;max-width:100%;font-family:{_FONT}">'
        f'{defs}{mark}'
        f'<text x="132" y="82" font-family="{_FONT}" font-size="76" font-weight="700" '
        f'textLength="{w}" lengthAdjust="spacing" fill="url(#bw_{u})">BASIS</text>'
        f'<text x="132" y="100" font-family="{_FONT}" font-size="{f_sub}" font-weight="600" '
        f'textLength="{w}" lengthAdjust="spacing" fill="{pal["tagline"]}">'
        f'ANALYSIS · STRATEGIES · INDICATORS</text>'
        f'</svg>'
    )


# --- theme css -------------------------------------------------------------
_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;450;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root { --basis-gold:$gold; --basis-gold-deep:$gold_deep;
        --basis-cal-ink:$cal_ink; --basis-cal-ink-strong:$cal_ink_strong;
        --basis-mono:'IBM Plex Mono', Consolas, 'SF Mono', Menlo, monospace; }

/* terminal geometry: sharp corners everywhere — separation comes from borders and
   surface steps, never radius or shadows (design_kit/handoff/README.md). */
*, *::before, *::after { border-radius:0 !important; }

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stBottom"] {
    background:$canvas !important; color:$text;
}
html { font-size:14px; }               /* rem base: the whole app steps down to terminal density */
.stApp, .stMarkdown, p, li, label, span,
div[data-testid="stMarkdownContainer"] {
    color:$text;
    font-family:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}
body { font-variant-numeric:tabular-nums; }
/* heading ladder (handoff): page title 15px .02em; section = 11px 600 .18em uppercase
   micro-title — the single strongest "institutional" move after mono numerals.
   !important + the container selector outranks Streamlit's own heading classes. */
[data-testid="stMarkdownContainer"] h1, .stApp h1 {
    color:$text; font-size:1.30rem !important; font-weight:600; letter-spacing:.02em;
}
[data-testid="stMarkdownContainer"] h2, .stApp h2 {
    color:$text; font-size:1.07rem !important; font-weight:600; letter-spacing:.02em;
}
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5, .stApp h3, .stApp h4, .stApp h5 {
    color:$text; font-size:.86rem !important; font-weight:600;
    letter-spacing:.16em; text-transform:uppercase;
}
/* frosted translucent header: content scrolling beneath shows through a blur instead of
   being hard-amputated by an opaque bar (the "cut-off button" effect). */
/* the FIXED top bar (clocks + masthead) now covers the viewport top, so the Streamlit
   header goes fully transparent — only its floating buttons (menu) remain visible.
   pointer-events: the invisible header still sits ABOVE the bar and was swallowing real
   mouse clicks on the theme toggle — clicks now pass through except on its buttons. */
[data-testid="stHeader"] {
    background:transparent;
    backdrop-filter:none; -webkit-backdrop-filter:none;
    pointer-events:none;
}
/* Streamlit's own CSS re-enables pointer-events on the toolbar (the ⋮ strip), and its
   container DIV stretches LEFT over the theme toggle — eating the click even though the
   header itself is click-transparent. Kill events on the whole toolbar subtree and
   re-enable only on actual clickables (the ⋮ menu button). */
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stHeader"] [data-testid="stToolbar"] * { pointer-events:none; }
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="stToolbar"] button,
[data-testid="stHeader"] [data-testid="stToolbar"] a { pointer-events:auto; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { background:transparent; color:$text_dim; }
[data-testid="stAppDeployButton"] { display:none; }   /* hide Streamlit "Deploy" */
/* hide the run-status / "Stop" toolbar pill (long jobs already show st.spinner). */
[data-testid="stStatusWidget"] { display:none; }
button[data-testid="stBaseButton-header"] { background:transparent !important; color:$text_dim !important; }
[data-testid="stCaptionContainer"], small, .stCaption,
[data-testid="stCaptionContainer"] * { color:$caption !important; }
/* main-content captions: larger + roomier so the dense help lines read easily */
.block-container [data-testid="stCaptionContainer"],
.block-container [data-testid="stCaptionContainer"] * {
    font-size:.95rem !important; line-height:1.55 !important;
}
a, a:visited { color:$gold; }
hr { border-color:$border; }
/* the scroll-to-top script block: zero visual footprint — its 0-height iframe would still
   eat a 16px flex gap between the masthead and the page. display:none iframes still run
   their scripts. (Hide both the keyed block and, on newer Streamlit, its layout wrapper.) */
div.st-key-basis_scroll_top,
[data-testid="stLayoutWrapper"]:has(> .st-key-basis_scroll_top) { display:none; }
.block-container { padding-top:127px; }   /* clears the FIXED top bar (clocks + masthead, ~97px) */
/* full-bleed terminal width: content uses the whole viewport minus the page padding
   (the old centered ~46rem column read as a website, not a terminal). */
.block-container { max-width:100% !important; padding-left:1.6rem; padding-right:1.6rem; }

/* sidebar */
[data-testid="stSidebar"] { background:$sidebar; border-right:1px solid $border; }
[data-testid="stSidebar"] * { color:$text; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:$caption !important; }
/* pull the logo block up: shrink the near-empty header strip (it only holds the collapse
   arrow) and drop its 16px bottom margin... */
[data-testid="stSidebarHeader"] {
    padding:.35rem .75rem !important; height:auto !important; margin-bottom:0 !important;
}
/* ...and remove the invisible style-carrier markdown above the logo — at zero height it
   still eats a full 16px flex gap. */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display:none;
}
/* the logo + FICC/Equities switch stay pinned to the top while the nav list scrolls;
   the opaque sidebar-colour background hides the entries sliding beneath. Sticky must sit
   on the stLayoutWrapper AROUND the keyed block — the keyed block itself fills its wrapper
   exactly, leaving sticky no room to pin. */
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(> .st-key-basis_sidebar_sticky) {
    position:sticky; top:0; z-index:20;
    background:$sidebar; padding-bottom:.3rem;
}
/* sidebar nav rows (handoff): flat text rows, not buttons. Normal case, left-aligned;
   hover = surface step; ACTIVE (Streamlit type=primary) = surface2 fill + inset 2px
   gold left rule. Overrides the global uppercase/bordered button treatment. */
[data-testid="stSidebar"] .stButton>button {
    background:transparent !important; border:none !important; color:$text_dim !important;
    text-transform:none !important; letter-spacing:0 !important;
    font-weight:400; justify-content:flex-start; box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton>button p {
    font-size:.9rem !important; text-transform:none !important; letter-spacing:0 !important;
    text-align:left; width:100%;
}
[data-testid="stSidebar"] .stButton>button > div { width:100%; justify-content:flex-start; }
[data-testid="stSidebar"] .stButton>button:hover {
    background:$surface !important; color:$text !important;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background:$surface2 !important; box-shadow:inset 2px 0 0 $gold !important;
    color:$text !important; font-weight:500;
}
[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] * { color:$text !important; }

/* buttons — terminal treatment (handoff): transparent fills, 1px borders, uppercase
   labels, no shadows, no transitions. Gold-bordered = the important action;
   grey-bordered = everything else. */
.stButton>button, .stDownloadButton>button {
    font-weight:600; font-size:.74rem !important;
    letter-spacing:.04em; text-transform:uppercase;
    box-shadow:none !important; transition:none !important;
}
.stButton>button p, .stDownloadButton>button p {
    font-size:.74rem !important; line-height:1.3 !important;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"], .stDownloadButton>button {
    background:transparent !important; color:$text_dim !important;
    border:1px solid $btn_border !important;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton>button:hover {
    background:$btn_hover !important; border-color:$text_dim !important; color:$text !important;
}
/* buttons — primary: the gold-bordered terminal action (transparent fill, gold label;
   hover fills with the gold wash). */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background:transparent !important; border:1px solid $label_ring !important;
    color:$gold !important; font-weight:600;
}
button[kind="primary"]:hover, button[kind="primary"]:active, button[kind="primary"]:focus,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:active,
button[data-testid="stBaseButton-primary"]:focus {
    background:$btn_gold !important; border-color:$gold !important; color:$gold !important;
}
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color:$gold !important; }

/* widget labels: micro-labels — mono, uppercase, faint (the ring treatment is retired;
   labels identify controls typographically now). */
[data-testid="stWidgetLabel"] { margin-bottom:.2rem; background:transparent; border:none; }
[data-testid="stWidgetLabel"] p {
    color:$faint !important; font-family:var(--basis-mono);
    font-size:.70rem !important; letter-spacing:.16em; text-transform:uppercase;
}
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
[data-testid="stToggle"] [data-testid="stWidgetLabel"] {
    border:none; padding:0; margin-bottom:0;
}

/* inputs / selects / popovers */
input, textarea,
[data-baseweb="input"]>div, [data-baseweb="base-input"],
[data-baseweb="select"]>div, [data-baseweb="textarea"] {
    background:$surface !important; color:$text !important; border-color:$border !important;
}
/* SELECT dropdowns (selectbox / multiselect) get the same gold left-accent as the expander
   dropdowns — so every "open me" control reads the same way. */
[data-baseweb="select"]>div { border-left:2px solid $label_ring !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [role="option"] {
    background:$surface !important; color:$text !important;
}
/* multiselect chips (Send-to recipients, sector filters): terminal chip — gold wash,
   gold border, gold text (the handoff's selected-chip pattern; no solid fills).
   An inset box-shadow, NOT `border`: Chrome rasterizes a border as 4 INDEPENDENT edge
   rectangles, each individually floored to whole device pixels — at some DPI/zoom
   combinations one edge (usually the left) floors to 0 while its neighbours floor to 1,
   so the border reads as missing on that one side only. A box-shadow is painted as ONE
   continuous stroke, immune to that per-edge rounding split, at any scale/zoom. Widening
   border-width (1px -> 2px) papered over this at 125% but not at every zoom level. */
[data-baseweb="tag"] { background:$btn_gold !important; border:none !important;
    box-shadow:inset 0 0 0 1.5px $label_ring !important;
    max-width:none !important; flex-shrink:0 !important; padding-left:12px !important; overflow:visible !important; }
[data-baseweb="tag"], [data-baseweb="tag"] span, [data-baseweb="tag"] * { color:$gold !important; }
/* show the WHOLE recipient — no width cap and NEVER shrink, so the text can't clip; chips wrap instead */
[data-baseweb="tag"] span, [data-baseweb="tag"] div {
    max-width:none !important; flex-shrink:0 !important; overflow:visible !important; text-overflow:clip !important; }
[data-baseweb="tag"] svg { fill:$gold !important; color:$gold !important; }
/* Higher-specificity twin of the above — BaseWeb's component CSS loads AFTER this theme and
   was re-clipping the chip's first character; this selector ([select] [tag] = 0,2,x) outranks
   BaseWeb's class rules so the full recipient shows. Border/box-shadow redeclared here too —
   BaseWeb's own same-or-higher-specificity rule for THIS nested case can otherwise win. */
[data-baseweb="select"] [data-baseweb="tag"] {
    overflow:visible !important; max-width:none !important; flex-shrink:0 !important;
    padding-left:14px !important; background:$btn_gold !important;
    border:none !important; box-shadow:inset 0 0 0 1.5px $label_ring !important;
    /* real margin off the field's left edge: when BaseWeb's runtime CSS drags the chip box
       left (the first-character clip, above), padding only saves the TEXT — the box's left
       corner still gets cut. A margin keeps the whole box inside the field either way. */
    margin-left:8px !important; }
[data-baseweb="select"] [data-baseweb="tag"] span, [data-baseweb="select"] [data-baseweb="tag"] div {
    overflow:visible !important; text-overflow:clip !important; max-width:none !important;
    flex-shrink:0 !important; background:transparent !important; color:$gold !important; }

/* expander / containers / metric */
/* expanders as terminal panels: surface + 1px line; the summary is a
   surface2 header strip with an uppercase micro-title and a gold chevron.
   The line is an OUTSET box-shadow ring, not `border` — same Chrome per-edge
   pixel-rounding dodge as the chips above (a real border's left edge floors
   to 0 device px at some zooms and vanishes); outset (not inset) because the
   opaque summary strip would paint over an inset ring's top edge. */
[data-testid="stExpander"] { background:$surface; border:none; box-shadow:0 0 0 1px $border; }
/* LABEL RECIPE E (Ben, 2026-08-16): labels are sentence case in the APP FONT —
   readable word-shapes — sized ~.8-.85rem at $caption weight 600 with near-zero
   tracking. The old mono-caps micro-label (tiny + UPPERCASE + wide tracking)
   was hard to read everywhere. Mono stays for VALUES, tickers and clocks. */
[data-testid="stExpander"] summary {
    background:$surface2 !important;
    font-weight:600; padding:.5rem .9rem !important;
    letter-spacing:.01em; text-transform:none; font-size:.85rem;
}
[data-testid="stExpander"] summary p { font-size:.85rem !important; letter-spacing:.01em; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:$text; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover * { color:$gold !important; }
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    fill:$gold !important; color:$gold !important;
}
/* metric cells: flat panel, mono value, faint mono micro-label — the handoff KPI cell.
   Ring not border, per the expander note. */
[data-testid="stMetric"] { background:$surface; border:none; box-shadow:0 0 0 1px $border; padding:.6rem .9rem; }
[data-testid="stMetricValue"] { color:$text; font-family:var(--basis-mono); font-size:1.15rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color:$caption !important;
    font-size:.85rem !important; letter-spacing:.01em; text-transform:none;
    font-weight:600;
}
[data-testid="stMetricDelta"] { font-family:var(--basis-mono); }

/* dataframe / editor / table — ring not border, per the expander note */
[data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] {
    background:$surface; border:none; box-shadow:0 0 0 1px $border;
}
/* markdown/html tables (report previews, small boards) follow the handoff table spec */
[data-testid="stMarkdownContainer"] table { border-collapse:collapse; }
[data-testid="stMarkdownContainer"] th {
    background:$surface2; color:$caption;
    font-size:.8rem; font-weight:600; letter-spacing:.02em; text-transform:none;
    border-bottom:1px solid $border; text-align:left;
}
[data-testid="stMarkdownContainer"] td { border-bottom:1px solid $border_soft; }

/* tabs */
[data-baseweb="tab-list"] { border-bottom:1px solid $border; gap:.25rem; }
button[data-baseweb="tab"] { color:$text_dim; }
button[data-baseweb="tab"][aria-selected="true"] { color:$gold; }
[data-baseweb="tab-highlight"] { background:$gold !important; }

/* accents */
[data-testid="stCheckbox"] svg, [data-baseweb="checkbox"] svg { color:$gold; }
[data-baseweb="slider"] [role="slider"] { background:$gold !important; }
code, pre, kbd { background:$surface2; color:$text; }
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:$border; border-radius:8px; }
::-webkit-scrollbar-track { background:transparent; }

/* masthead */
.basis-masthead { display:flex; align-items:center; gap:1rem; }
.basis-tag { color:$tagline; font-size:.66rem; letter-spacing:.34em; font-weight:600;
             text-transform:uppercase; white-space:nowrap; }
/* the stacked BASIS+tagline lockup is rendered inline in header_svg(tagline=True) */
.basis-rule { height:1px; border:0; margin:.5rem 0 1.05rem; background:$border; }

/* theme toggle: 30px square terminal control */
.st-key-basis_theme_toggle button {
    background:transparent; color:$text_dim; border:1px solid $btn_border;
    padding:.3rem .7rem; min-height:0; line-height:1.1; font-weight:600;
}
.st-key-basis_theme_toggle button:hover { border-color:$gold; color:$gold; }
.st-key-basis_theme_toggle button p { font-size:1.15rem; line-height:1; margin:0; }

/* hover tooltips (portal-rendered) — match the palette in both themes */
div:has(> .stTooltipContent), .stTooltipContent { background:$surface2 !important; }
.stTooltipContent, .stTooltipContent * { color:$text !important; }

/* ── terminal shell (handoff) ─────────────────────────────────────────── */
/* fixed top bar: the world-clock rail over the masthead row, pinned to the viewport
   top on EVERY page (stays put while scrolling). The wrapper is the fixed element;
   .block-container and the sidebar are padded down to clear it. */
/* The bar starts AFTER the sidebar (which stays on top, full height, logo at the very
   top): --basis-topbar-left tracks the sidebar's live width, synced by the clock
   script every 500ms (the sidebar is drag-resizable, so CSS alone can't know it). */
[data-testid="stLayoutWrapper"]:has(> .st-key-basis_topbar) {
    position:fixed; top:0; left:var(--basis-topbar-left, 0px); right:0; z-index:999989;
    width:auto !important; max-width:none !important;   /* honour left+right, not the stamped px width */
    background:$canvas; border-bottom:1px solid $border;
}
div.st-key-basis_topbar { gap:0 !important; }
div.st-key-basis_topbar [data-testid="stElementContainer"] { margin-bottom:0; }
/* Streamlit stamps explicit PIXEL widths on blocks (computed for the bar's original
   in-flow position) — once the bar is fixed and shifted right of the sidebar, those
   stale widths overflow the viewport and push the ET clock + theme toggle off screen.
   Everything inside the bar must be fluid. */
div.st-key-basis_topbar,
div.st-key-basis_topbar [data-testid="stVerticalBlock"],
div.st-key-basis_topbar [data-testid="stHorizontalBlock"],
div.st-key-basis_topbar [data-testid="stLayoutWrapper"],
div.st-key-basis_topbar [data-testid="stElementContainer"] {
    width:100% !important; max-width:100% !important;
}

/* masthead bar */
[data-testid="stLayoutWrapper"]:has(> .st-key-basis_masthead),
.st-key-basis_masthead {
    background:$surface; border-bottom:1px solid $border;
}
.st-key-basis_masthead { padding:.65rem 1.2rem .6rem; }
.bt-mast { display:flex; align-items:center; gap:14px; min-height:40px; }
.bt-word {
    font-weight:700; font-size:2.2rem; letter-spacing:.11em; line-height:1.05;
    background:linear-gradient(96deg,#EEF0F3,#C0C5CC 46%,#CBA53C 70%,#F4CC3A);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.bt-div { width:1px; height:30px; background:$btn_border; }
.bt-crumb {
    font-size:.78rem; letter-spacing:.05em;
    text-transform:uppercase; color:$caption; font-weight:600;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;
}
.bt-clock { font-family:var(--basis-mono); font-size:.75rem; color:$faint; white-space:nowrap; }
.st-key-basis_masthead .stButton>button { min-height:30px; padding:.15rem .5rem; }
/* the theme toggle (on the top bar's clocks row) — right padding keeps it clear of
   Streamlit's floating ⋮ menu button in the corner */
.st-key-basis_theme_toggle { display:flex; align-items:center; justify-content:center; padding-right:44px; }
.st-key-basis_theme_toggle button {
    width:34px !important; min-width:34px; height:34px; min-height:34px; padding:0;
    display:flex; align-items:center; justify-content:center;   /* sharp corners: matches the app's other buttons */
}

/* ticker rail */
.bt-rail {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(126px,1fr));
    background:$surface2; border-bottom:1px solid $border; margin:0 -1.6rem;
}
.bt-cell { padding:8px 12px; min-width:0; border-right:1px solid $border_soft;
           border-bottom:1px solid $border_soft; }
.bt-sym { font-family:var(--basis-mono); font-size:.72rem; letter-spacing:.05em;
          color:$caption; text-transform:uppercase; white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis; }
.bt-px { font-family:var(--basis-mono); font-size:.93rem; font-weight:500; color:$text;
         display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }
.bt-px span { font-size:.75rem; }

/* page bar */
.bt-pagebar { display:flex; align-items:baseline; gap:14px;
              border-bottom:1px solid $border; padding:.35rem 0 .5rem; margin-bottom:.9rem; }
.bt-pb-title { font-size:1.07rem; font-weight:600; letter-spacing:.02em; color:$text; }
.bt-pb-kicker { font-size:.78rem; letter-spacing:.02em; font-weight:600;
                color:$caption; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bt-pb-right { font-size:.78rem; letter-spacing:.02em; color:$text_dim; }

/* panel header strip — ring not border (expander note); the flush tablewrap
   below paints its background over this ring's bottom edge, so the junction
   stays a single 1px line exactly as the old border-bottom:none gave. */
.bt-panelhead { display:flex; justify-content:space-between; align-items:center;
                background:$surface; border:none; box-shadow:0 0 0 1px $border;
                padding:9px 14px; margin-top:.35rem; }
.bt-panelhead > span:first-child { font-size:.85rem; font-weight:600; letter-spacing:.02em; color:$text; }
.bt-ph-right { font-size:.75rem; letter-spacing:.02em; font-weight:600; color:$gold; }

/* terminal table */
.bt-tablewrap { border:none; box-shadow:0 0 0 1px $border; background:$surface; overflow-x:auto; margin-bottom:.9rem; }
.bt-table { width:100%; border-collapse:collapse; font-size:.89rem; }
.bt-table th { background:$surface2; color:$caption;
               font-size:.8rem; font-weight:600; letter-spacing:.02em; text-transform:none;
               text-align:left; padding:6px 10px; border-bottom:1px solid $border; white-space:nowrap; }
.bt-table th.num { text-align:right; }
.bt-table td { padding:7px 10px; border-bottom:1px solid $border_soft; color:$text_dim; }
.bt-table td:first-child { color:$text; font-weight:500; padding-left:14px; }
.bt-table td.num { font-family:var(--basis-mono); text-align:right; white-space:nowrap; }
.bt-table tr:hover td { background:$surface2; }
.bt-table td.zcell { min-width:86px; }
.zbar { position:relative; height:4px; background:transparent; }
.ztick { position:absolute; left:50%; top:-2px; width:1px; height:8px; background:$faint; }
.zfill { position:absolute; top:0; height:4px; }

/* sidebar: desk segmented control (the FICC / Equities columns) */
[data-testid="stSidebar"] .st-key-side_ficc button,
[data-testid="stSidebar"] .st-key-side_equities button {
    border:1px solid $border !important; background:transparent !important;
    font-family:var(--basis-mono) !important; font-weight:600;
    letter-spacing:.16em !important; text-transform:uppercase !important;
    justify-content:center; box-shadow:none !important; color:$text_dim !important;
}
[data-testid="stSidebar"] .st-key-side_ficc button p,
[data-testid="stSidebar"] .st-key-side_equities button p {
    font-size:.78rem !important; letter-spacing:.05em !important; text-transform:uppercase !important;
}
[data-testid="stSidebar"] .st-key-side_ficc button[kind="primary"],
[data-testid="stSidebar"] .st-key-side_equities button[kind="primary"] {
    background:$gold_soft !important; box-shadow:inset 0 -2px 0 $gold !important;
    color:$gold !important;
}
[data-testid="stSidebar"] .st-key-side_ficc button[kind="primary"] *,
[data-testid="stSidebar"] .st-key-side_equities button[kind="primary"] * { color:$gold !important; }

/* sidebar micro section labels + footer */
.bt-sect { font-size:.74rem; letter-spacing:.04em; font-weight:600;
           text-transform:uppercase; color:$caption; margin:.9rem 0 .25rem .35rem; }
.bt-sbfoot { border-top:1px solid $border; margin-top:1rem; padding:.6rem .35rem 0; }
.bt-sbfoot div { display:flex; justify-content:space-between; align-items:center;
                 font-size:.74rem; color:$caption;
                 letter-spacing:.02em; padding:.14rem 0; }
.bt-sbfoot div span:last-child { color:$text_dim; }
.bt-sbfoot .dot { display:inline-block; width:6px; height:6px; margin-right:6px; }
</style>
""")


def apply() -> None:
    """Initialise the theme and inject its CSS. Call once, right after
    st.set_page_config(). The PREF FILE is the single source of truth — every
    session (laptop tab, VPS view, a second browser) re-reads it each run, so
    all tabs follow the same theme. This matters because the NATIVE Streamlit
    theme (_sync_native_theme) is process-global: two sessions on different
    themes would fight over it, leaving e.g. a dark tab with light table chrome
    (white borders) after a light tab reran. One shared pref removes the fight."""
    st.session_state["basis_theme"] = _load_pref() or st.session_state.get(
        "basis_theme", "dark")
    _sync_native_theme()
    st.markdown(_CSS.safe_substitute(palette()), unsafe_allow_html=True)
    _apply_altair_theme()
    _inject_manifest()


def _inject_manifest() -> None:
    """Add <link rel="manifest"> (+ theme-color) to the OUTER document's <head> — the
    PWA manifest that lets Chrome offer a real "Install BASIS", which gets its own
    cached icon everywhere (window, taskbar, Start Menu) instead of the generic Chrome
    icon a plain --app=URL shortcut gets. Streamlit has no official API for <head>
    injection, so this reaches out from a components.html iframe via window.parent —
    the same technique the world-clock rail already uses to sync CSS vars onto the
    outer page (see _world_clocks in app.py). Guarded so a rerun never adds a duplicate
    <link>. Needs [server] enableStaticServing = true (.streamlit/config.toml) so
    /app/static/manifest.json + its icons actually resolve."""
    import streamlit.components.v1 as components
    components.html(
        "<script>try{var d=window.parent.document;"
        "if(!d.querySelector('link[rel=\"manifest\"]')){"
        "var l=d.createElement('link');l.rel='manifest';l.href='/app/static/manifest.json';"
        "d.head.appendChild(l);"
        "var m=d.createElement('meta');m.name='theme-color';m.content='#F5C518';"
        "d.head.appendChild(m);"
        "}}catch(e){}</script>", height=0)


def _sync_native_theme() -> None:
    """Drive Streamlit's NATIVE theme from the active palette. The data grid
    (st.dataframe) is canvas-rendered and ignores the app's CSS entirely — its
    header fill and grid borders come from the Streamlit theme, so on the light
    palette they stayed config.toml-dark (black borders on white tables). The
    theme is re-sent to the frontend on each rerun, so the toggle's st.rerun()
    re-skins the grids too. Process-global — safe only because apply() pins
    every session to the shared pref file."""
    from streamlit import config as _st_config
    pal = palette()
    opts = {
        "theme.base": pal["name"],
        "theme.primaryColor": pal["gold"],
        "theme.backgroundColor": pal["canvas"],
        "theme.secondaryBackgroundColor": pal["surface2"],
        "theme.textColor": pal["text"],
        "theme.borderColor": pal["border"],
        "theme.dataframeBorderColor": pal["border"],
        "theme.dataframeHeaderBackgroundColor": pal["surface2"],
    }
    for _k, _v in opts.items():
        try:
            if _st_config.get_option(_k) != _v:
                _st_config.set_option(_k, _v)
        except Exception:
            pass                      # unknown option on an older Streamlit — skip


# --- rendered pieces -------------------------------------------------------
def sidebar_logo() -> None:
    """BASIS lockup at the top of the sidebar (replaces the old title)."""
    pal = palette()
    st.markdown(
        f'<div style="padding:.15rem 0 .35rem">{header_svg(pal, height=64)}'
        # Negative top margin pulls the tag up into the svg box's empty bottom band; the
        # bottom margin pushes the FICC/Equities row down the same amount — leaving the tag
        # at the OPTICAL midpoint between the drawn wordmark and the buttons (~12px each).
        # letter-spacing tightened vs .basis-tag's .34em — this string is twice
        # "Research Terminal"'s length and must still fit the sidebar width
        f'<div class="basis-tag" style="margin-top:-.5rem;margin-bottom:.4rem;'
        f'letter-spacing:.18em;text-align:center">Analysis · Strategies · Indicators'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def theme_toggle() -> None:
    """The dark/light toggle (☀/☾) — its own element so the top bar can place it
    wherever fits (currently on the clocks row, filling the right-hand space)."""
    dark = theme() == "dark"
    label = "☀" if dark else "☾"
    nxt = "light" if dark else "dark"
    if st.button(label, key="basis_theme_toggle", use_container_width=True,
                 help=f"Switch to {nxt} mode"):
        st.session_state["basis_theme"] = nxt
        _save_pref(nxt)
        st.rerun()


def masthead(breadcrumb: str | None = None, toggle: bool = True) -> None:
    """Terminal masthead bar (handoff): gradient wordmark · divider · mono breadcrumb,
    on a surface strip with a bottom border. Same bar on every page; the breadcrumb
    carries the location. toggle=True keeps the theme toggle in-row (legacy layout);
    the fixed top bar renders the toggle on the clocks row instead and passes False."""
    crumb = (breadcrumb or "ANALYSIS · STRATEGIES · INDICATORS").upper()
    with st.container(key="basis_masthead"):
        mast = (f'<div class="bt-mast">'
                f'<span class="bt-word">BASIS</span>'
                f'<span class="bt-div"></span>'
                f'<span class="bt-crumb">{crumb}</span>'
                f'</div>')
        if toggle:
            left, right = st.columns([0.86, 0.14], vertical_alignment="center")
            with left:
                st.markdown(mast, unsafe_allow_html=True)
            with right:
                theme_toggle()
        else:
            st.markdown(mast, unsafe_allow_html=True)


def ticker_rail(cells: list[dict]) -> None:
    """Quote rail under the masthead (handoff): auto-fit mono cells — symbol line,
    price + coloured change. cells = [{sym, name, last, chg, up}] (chg preformatted)."""
    if not cells:
        return
    pal = palette()
    up, down = ("#46C58A", "#EC6A57") if pal["name"] == "dark" else ("#0F7A45", "#C0392B")
    html = ['<div class="bt-rail">']
    for c in cells:
        col = up if c.get("up") else down
        html.append(
            f'<div class="bt-cell"><div class="bt-sym">{c["sym"]}</div>'
            f'<div class="bt-px">{c["last"]}'
            f'<span style="color:{col}">{c["chg"]}</span></div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def page_bar(title: str, kicker: str = "", right: str = "") -> None:
    """Bottom-bordered page-title row: h2-weight title + mono kicker left, mono count right."""
    st.markdown(
        f'<div class="bt-pagebar"><span class="bt-pb-title">{title}</span>'
        f'<span class="bt-pb-kicker">{kicker.upper()}</span>'
        f'<span class="bt-pb-right">{right.upper()}</span></div>', unsafe_allow_html=True)


def panel_header(title: str, right: str = "") -> None:
    """Terminal panel header strip (uppercase micro-title + mono right meta)."""
    st.markdown(
        f'<div class="bt-panelhead"><span>{title.upper()}</span>'
        f'<span class="bt-ph-right">{right.upper()}</span></div>', unsafe_allow_html=True)


def terminal_table(rows: list[dict], columns: list[dict]) -> None:
    """Handoff-spec HTML table. columns = [{key, label, align?, color?, zbar?, pbar?,
    help?, help_key?, keep_case?}]: color=True paints +green/−red from the raw value; zbar=True
    renders the centered z-range bar (value in σ, clamped at ±2); pbar=True a 0–100
    percentile bar (left-anchored fill, centre tick = the median, gold fill inside the
    top/bottom decile); help adds a hover tooltip to the header, help_key names a row field
    whose text becomes a per-CELL tooltip (e.g. what a metric means); keep_case skips the
    upper-casing for symbol labels (±2σ would otherwise render as ±2Σ)."""
    pal = palette()
    up, down = ("#46C58A", "#EC6A57") if pal["name"] == "dark" else ("#0F7A45", "#C0392B")
    out = ['<div class="bt-tablewrap"><table class="bt-table"><thead><tr>']
    for c in columns:
        cls = ' class="num"' if c.get("align") == "right" or c.get("color") else ""
        lbl = c["label"] if c.get("keep_case") else c["label"].upper()
        ttl = ""
        if c.get("help"):
            ttl = f' title="{str(c["help"]).replace(chr(34), "&quot;")}" style="cursor:help"'
        out.append(f'<th{cls}{ttl}>{lbl}</th>')
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>')
        for c in columns:
            v = r.get(c["key"])
            if c.get("zbar"):
                z = 0.0 if v is None or v != v else max(-2.0, min(2.0, float(v)))
                w = abs(z) / 2 * 50                      # half-width %, centre tick at 50%
                side = ("left:50%;background:" + up) if z >= 0 else \
                       (f"left:{50 - w:.0f}%;background:" + down)
                out.append(f'<td class="zcell"><div class="zbar">'
                           f'<div class="ztick"></div>'
                           f'<div class="zfill" style="{side};width:{w:.0f}%"></div></div></td>')
                continue
            if c.get("pbar"):
                p = 0.0 if v is None or v != v else max(0.0, min(100.0, float(v)))
                colr = pal["gold"] if (p <= 10.0 or p >= 90.0) else pal["faint"]
                out.append(f'<td class="zcell"><div class="zbar">'
                           f'<div class="ztick"></div>'
                           f'<div class="zfill" style="left:0;width:{p:.0f}%;'
                           f'background:{colr}"></div></div></td>')
                continue
            txt = "—" if v is None or (isinstance(v, float) and v != v) else str(v)
            tip = ""
            if c.get("help_key"):
                _h = r.get(c["help_key"])
                if _h:
                    tip = (f' title="{str(_h).replace(chr(34), "&quot;")}"'
                           ' style="cursor:help;text-decoration:underline dotted 1px;'
                           'text-underline-offset:3px"')
            style = ""
            if c.get("color") and isinstance(v, (int, float)) and v == v:
                style = f' style="color:{up if v > 0 else down};font-weight:600"'
                txt = c.get("fmt", "{:+.2f}").format(v)
            elif c.get("fmt") and isinstance(v, (int, float)) and v == v:
                txt = c["fmt"].format(v)
            cls = ' class="num"' if c.get("align") == "right" or c.get("color") else ""
            if cls and style:
                out.append(f'<td class="num"{style}>{txt}</td>')
            elif tip and not style:                 # per-cell tooltip (dotted underline hint)
                out.append(f'<td{cls}><span{tip}>{txt}</span></td>')
            else:
                out.append(f'<td{cls}{style}>{txt}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def sidebar_footer(rows: list[tuple[str, str, str]]) -> None:
    """Mono status rows for the sidebar foot: (label, value, dot_color_or_empty)."""
    out = ['<div class="bt-sbfoot">']
    for label, value, dot in rows:
        d = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
        out.append(f'<div><span>{label.upper()}</span><span>{d}{value}</span></div>')
    out.append('</div>')
    st.markdown("".join(out), unsafe_allow_html=True)


# --- data tables -----------------------------------------------------------
def themed_dataframe(df, fmt, colorers=None, *, na_rep=None, height=None,
                     pal: dict | None = None, column_config: dict | None = None,
                     key: str | None = None, selection_mode=None, on_select=None):
    """st.dataframe whose cells follow the active palette.

    Streamlit's data grid is canvas-rendered and ignores the app's CSS, so it
    would otherwise stay on the config (dark) theme. It *does* honour a Styler's
    per-cell `background-color`/`color`, so we paint a palette base first, then
    let any `colorers` (e.g. red/green move tints) win on their own columns.

    colorers: list of (subset_columns, styler_func). Applied after the base so
    their `color:` overrides the base text colour on those columns.

    Pass `key` + `selection_mode` + `on_select` to make the grid clickable; the
    widget's return value (rows the user picked) is then handed back to the caller.
    Returns None for the plain, non-interactive case.
    """
    pal = pal or palette()
    kw = {"use_container_width": True, "hide_index": True}
    if height is not None:                  # Streamlit rejects height=None
        kw["height"] = height
    if column_config is not None:           # per-column width/label control (keeps narrow cols narrow)
        kw["column_config"] = column_config
    if key is not None:
        kw["key"] = key
    if selection_mode is not None:
        kw["selection_mode"] = selection_mode
    if on_select is not None:
        kw["on_select"] = on_select
    try:
        sty = df.style.format(fmt, na_rep=na_rep) if na_rep is not None else df.style.format(fmt)
        sty = sty.set_properties(**{"background-color": pal["surface"], "color": pal["text"]})
        for subset, fn in (colorers or []):
            sty = sty.apply(fn, subset=subset)
        return st.dataframe(sty, **kw)
    except Exception:                       # Styler needs jinja2 — plain fallback
        plain = df.copy()

        def _one(v, fn):
            try:
                return fn(v) if callable(fn) else fn.format(v)
            except Exception:
                return na_rep if na_rep is not None else v

        for col, fn in (fmt or {}).items():
            if col in plain:
                plain[col] = plain[col].map(lambda v, fn=fn: _one(v, fn))
        return st.dataframe(plain, **kw)


# --- altair charts ---------------------------------------------------------
_ALT_THEME_DONE = False


def _alt_theme_config() -> dict:
    """Vega-Lite config from the *active* palette — read live so the theme
    follows the toggle. Marks stay per-chart (see chart_colors)."""
    pal = palette()
    return {"config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "line": {"strokeWidth": 2.4},      # default for lines that don't set a width
        "trail": {"size": 2.4},
        "axis": {"labelColor": pal["text_dim"], "titleColor": pal["text_dim"],
                 "domainColor": pal["border"], "tickColor": pal["border"],
                 "gridColor": pal["border_soft"]},
        "legend": {"labelColor": pal["text"], "titleColor": pal["text_dim"]},
        "header": {"labelColor": pal["text"], "titleColor": pal["text_dim"]},
        "title": {"color": pal["text"], "subtitleColor": pal["text_dim"]},
    }}


def _apply_altair_theme() -> None:
    """Register + enable a BASIS Altair theme (chart background/axes/legend follow
    the palette). Registered once; the config function re-reads the palette each
    render, so the toggle re-skins every chart."""
    global _ALT_THEME_DONE
    if alt is None:
        return
    try:
        if not _ALT_THEME_DONE:
            try:                       # Altair >= 5.5 (new API)
                alt.theme.register("basis", enable=True)(_alt_theme_config)
            except Exception:          # Altair < 5.5
                alt.themes.register("basis", _alt_theme_config)
            _ALT_THEME_DONE = True
        try:
            alt.theme.enable("basis")
        except Exception:
            alt.themes.enable("basis")
    except Exception:
        pass


def show_chart(chart, **kwargs) -> None:
    """st.altair_chart with the BASIS theme. theme=None lets our registered Altair
    theme win instead of Streamlit's built-in (config-locked) one."""
    st.altair_chart(chart, use_container_width=True, theme=None, **kwargs)
