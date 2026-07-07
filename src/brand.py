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

_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
         "'Helvetica Neue', Arial, sans-serif")

# --- palettes --------------------------------------------------------------
# `word` = the BASIS wordmark gradient stops (left→right). Dark theme runs
# silver→gold on a dark canvas; light theme runs slate→gold on white.
# Tuned to XP Investimentos' own brand palette (sampled from the XP/Eurex
# brochure): yellow #F5C518 · rich black #0A0A0A · dark grey #2A2A2A ·
# off-white #F6F6F6 · white #FFFFFF. Dark = XP rich black + yellow; light = the
# white "brochure" look (white / off-white / near-black + the same XP yellow).
DARK = dict(
    name="dark",
    canvas="#0A0A0A", glass="rgba(10,10,10,.78)", surface="#161616", surface2="#1F1F1F",
    border="#2A2A2A", border_soft="rgba(255,255,255,.08)",
    btn="#35353F", btn_border="#45454E", btn_hover="#42424E",
    btn_gold="#514727", btn_gold_hover="#63562E", label_ring="rgba(245,197,24,.55)",
    text="#ECEEF1", text_dim="#CBD0D7", caption="#D2D7DD",
    sidebar="#101010",
    gold="#F5C518", gold_deep="#D9971C", gold_soft="rgba(245,197,24,.14)",
    bracket="#8A8F96", tagline="#8A8F96",
    word=(("0", "#EEF0F3"), ("0.5", "#C0C5CC"), ("0.72", "#CBA53C"), ("1", "#F4CC3A")),
)
LIGHT = dict(
    name="light",
    canvas="#FFFFFF", glass="rgba(255,255,255,.8)", surface="#F6F6F6", surface2="#ECECEC",
    border="#E3E3E3", border_soft="#ECECEC",
    btn="#FFFFFF", btn_border="#C4C4C4", btn_hover="#EEEEEE",
    btn_gold="#FBF3D0", btn_gold_hover="#F6E9A8", label_ring="rgba(200,144,26,.6)",
    text="#1A1A1A", text_dim="#3A3D42", caption="#42454A",
    sidebar="#F6F6F6",
    gold="#F5C518", gold_deep="#C8901A", gold_soft="rgba(245,197,24,.16)",
    bracket="#9A9A9A", tagline="#6A6A6A",
    word=(("0", "#1A1A1A"), ("0.58", "#1A1A1A"), ("0.73", "#C8901A"), ("1", "#E0A81C")),
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
# Advance width of "ANALYSIS · STRATEGY · INDICATORS" per 1u of font-size in this
# font stack (measured); lets us size the tagline so its natural width ≈ _WORD_W,
# leaving textLength only a hair of spacing to nudge.
_TAG_ADVANCE = 16.4


def header_svg(pal: dict, height: int = 34, tagline: bool = False) -> str:
    """Convergence mark + BASIS wordmark, themed for `pal`.

    tagline=False -> mark + wordmark only (used in the sidebar).
    tagline=True  -> stacked lockup: the wordmark with
    "Analysis · Strategy · Indicators" directly beneath it, both pinned to the
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
    mark = (
        f'<g transform="translate(14,26) scale(0.62)">'
        f'<line x1="6" y1="2" x2="6" y2="102" stroke="{pal["bracket"]}" stroke-width="2"/>'
        f'<line x1="2.5" y1="2" x2="9.5" y2="2" stroke="{pal["bracket"]}" stroke-width="2"/>'
        f'<line x1="2.5" y1="102" x2="9.5" y2="102" stroke="{pal["bracket"]}" stroke-width="2"/>'
        f'<path d="M8 2 L150 52" fill="none" stroke="url(#si_{u})" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M8 102 L150 52" fill="none" stroke="url(#go_{u})" stroke-width="8" stroke-linecap="round"/>'
        f'<circle cx="150" cy="52" r="8.5" fill="url(#go_{u})"/></g>'
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
        f'aria-label="BASIS — Analysis · Strategy · Indicators" '
        f'style="display:block;max-width:100%;font-family:{_FONT}">'
        f'{defs}{mark}'
        f'<text x="132" y="82" font-family="{_FONT}" font-size="76" font-weight="700" '
        f'textLength="{w}" lengthAdjust="spacing" fill="url(#bw_{u})">BASIS</text>'
        f'<text x="132" y="100" font-family="{_FONT}" font-size="{f_sub}" font-weight="600" '
        f'textLength="{w}" lengthAdjust="spacing" fill="{pal["tagline"]}">'
        f'ANALYSIS · STRATEGY · INDICATORS</text>'
        f'</svg>'
    )


# --- theme css -------------------------------------------------------------
_CSS = Template("""
<style>
:root { --basis-gold:$gold; --basis-gold-deep:$gold_deep; }

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stBottom"] {
    background:$canvas !important; color:$text;
}
.stApp, .stMarkdown, p, li, label, span,
div[data-testid="stMarkdownContainer"] { color:$text; }
h1, h2, h3, h4, h5, h6 { color:$text; font-weight:700; letter-spacing:.2px; }
/* frosted translucent header: content scrolling beneath shows through a blur instead of
   being hard-amputated by an opaque bar (the "cut-off button" effect). */
[data-testid="stHeader"] {
    background:$glass;
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
}
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
.block-container { padding-top:2.9rem; }   /* clears the fixed header with margin to spare */

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

/* buttons — secondary: gold ring + gold-TINTED FILL, so a button is the most obviously
   clickable thing on the page. The visual ladder: label = thin gold ring only (no fill),
   button = gold ring + gold-tinted fill, primary/active = solid gold tile. */
.stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:600; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"], .stDownloadButton>button {
    background:$btn_gold !important; color:$text !important; border:1px solid $gold !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 1px 2px rgba(0,0,0,.28);
    transition: background .12s ease, border-color .12s ease, color .12s ease, box-shadow .12s ease;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton>button:hover {
    background:$btn_gold_hover !important; border-color:$gold !important; color:$gold !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 2px 6px rgba(0,0,0,.35);
}
/* sidebar nav: keep the clear fill+border (visible/clickable) but FLAT — no raised shadow — so the
   stacked nav list stays clean; the raised look is reserved for main-content action buttons. */
[data-testid="stSidebar"] .stButton>button { box-shadow:none !important; }
/* buttons — primary (gold tile, dark label in both themes) */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background:$gold !important; border:1px solid $gold !important;
    color:#0A0A0A !important; font-weight:700;
}
button[kind="primary"]:hover, button[kind="primary"]:active, button[kind="primary"]:focus,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:active,
button[data-testid="stBaseButton-primary"]:focus {
    background:$gold_deep !important; border-color:$gold_deep !important; color:#000 !important;
}
/* the label is an inner <p>/<span> Streamlit colours light — force it dark on the gold tile */
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color:#0A0A0A !important; }

/* widget labels: a thin gold RING (no fill) so labels stand out — one rung below buttons
   on the ladder (buttons carry ring + gold fill; a ring alone means "this names a control,
   it isn't the control"). width:fit-content keeps the ring hugging the text. */
[data-testid="stWidgetLabel"] {
    border:1px solid $label_ring; border-radius:7px;
    padding:.12rem .55rem; width:fit-content; max-width:100%;
    margin-bottom:.3rem; background:transparent;
}
[data-testid="stWidgetLabel"] p { color:$text_dim; }
/* checkbox/toggle labels sit BESIDE the control, not above it — a ring there boxes the
   whole row awkwardly, so leave those unringed. */
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
[data-baseweb="select"]>div { border-left:4px solid $gold !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [role="option"] {
    background:$surface !important; color:$text !important;
}
/* multiselect chips (Send-to recipients, sector filters): force DARK text on the gold
   tag so it stays legible — same treatment as the gold primary button. */
[data-baseweb="tag"] { background:$gold !important; border-color:$gold !important; max-width:none !important; flex-shrink:0 !important; padding-left:12px !important; overflow:visible !important; }
[data-baseweb="tag"], [data-baseweb="tag"] span, [data-baseweb="tag"] * { color:#0A0A0A !important; }
/* show the WHOLE recipient — no width cap and NEVER shrink, so the text can't clip; chips wrap instead */
[data-baseweb="tag"] span, [data-baseweb="tag"] div {
    max-width:none !important; flex-shrink:0 !important; overflow:visible !important; text-overflow:clip !important; }
[data-baseweb="tag"] svg { fill:#0A0A0A !important; color:#0A0A0A !important; }
/* Higher-specificity twin of the above — BaseWeb's component CSS loads AFTER this theme and
   was re-clipping the chip's first character; this selector ([select] [tag] = 0,2,x) outranks
   BaseWeb's class rules so the full recipient shows. */
[data-baseweb="select"] [data-baseweb="tag"] {
    overflow:visible !important; max-width:none !important; flex-shrink:0 !important;
    padding-left:14px !important; background:$gold !important;
    /* real margin off the field's left edge: when BaseWeb's runtime CSS drags the chip box
       left (the first-character clip, above), padding only saves the TEXT — the box's left
       corner still gets cut. A margin keeps the whole box inside the field either way. */
    margin-left:8px !important; }
[data-baseweb="select"] [data-baseweb="tag"] span, [data-baseweb="select"] [data-baseweb="tag"] div {
    overflow:visible !important; text-overflow:clip !important; max-width:none !important;
    flex-shrink:0 !important; background:transparent !important; color:#0A0A0A !important; }

/* expander / containers / metric */
/* dropdowns (expanders): a gold-tinted header, gold left-accent and gold chevron so they clearly
   read as openable dropdowns — visibly different from the grey action buttons. */
[data-testid="stExpander"] { background:$surface; border:1px solid $btn_border; border-radius:10px; }
[data-testid="stExpander"] summary {
    background:linear-gradient(90deg, $gold_soft, $btn 62%) !important;
    border-left:4px solid $gold !important; border-radius:8px;
    font-weight:700; padding:.5rem .8rem !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:$text; }
[data-testid="stExpander"] summary:hover { background:$btn_hover !important; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover * { color:$gold !important; }
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    fill:$gold !important; color:$gold !important;
}
/* metric / label cards: a clear NEUTRAL (grey) ring so they stand out as info boxes — distinct from
   the GOLD interactive elements (buttons get a gold ring + fill; dropdowns a gold accent). */
[data-testid="stMetric"] { background:$surface; border:1.5px solid $btn_border; border-radius:11px; padding:.6rem .9rem; }
[data-testid="stMetricValue"] { color:$text; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:$text_dim; }

/* dataframe / editor / table */
[data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] {
    background:$surface; border:1px solid $border; border-radius:10px;
}

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
.basis-rule { height:1px; border:0; margin:.5rem 0 1.15rem;
              background:linear-gradient(90deg, $gold 0%, $border 34%, transparent 100%); }

/* sun/moon toggle pill (Streamlit tags the container .st-key-<key>) */
.st-key-basis_theme_toggle button {
    background:$surface; color:$text; border:1px solid $border; border-radius:999px;
    padding:.3rem .85rem; min-height:0; line-height:1.1; font-weight:600;
}
.st-key-basis_theme_toggle button:hover { border-color:$gold; color:$gold; }
.st-key-basis_theme_toggle button p { font-size:.82rem; }

/* hover tooltips (portal-rendered) — match the palette in both themes */
div:has(> .stTooltipContent), .stTooltipContent { background:$surface2 !important; }
.stTooltipContent, .stTooltipContent * { color:$text !important; }
</style>
""")


def apply() -> None:
    """Initialise the theme and inject its CSS. Call once, right after
    st.set_page_config(). The theme defaults to the user's last saved choice."""
    if "basis_theme" not in st.session_state:
        st.session_state["basis_theme"] = _load_pref() or "dark"
    st.markdown(_CSS.safe_substitute(palette()), unsafe_allow_html=True)
    _apply_altair_theme()


# --- rendered pieces -------------------------------------------------------
def sidebar_logo() -> None:
    """BASIS lockup at the top of the sidebar (replaces the old title)."""
    pal = palette()
    st.markdown(
        f'<div style="padding:.15rem 0 .35rem">{header_svg(pal, height=84)}'
        f'<div class="basis-tag" style="margin-top:.5rem">Research Monitor</div></div>',
        unsafe_allow_html=True,
    )


def masthead() -> None:
    """Main-column masthead: the full BASIS lockup (wordmark + tagline) on the left,
    sun/moon toggle on the right, closed with a gold hairline rule. The same size on
    every page — Home and inner pages alike."""
    pal = palette()
    left, right = st.columns([0.82, 0.18], vertical_alignment="center")
    with left:
        logo = (f'<div style="padding:.15rem 0 .3rem">'
                f'{header_svg(pal, height=84, tagline=True)}</div>')
        st.markdown(logo, unsafe_allow_html=True)
    with right:
        dark = theme() == "dark"
        label = "☀️  Light" if dark else "🌙  Dark"
        nxt = "light" if dark else "dark"
        clicked = st.button(
            label, key="basis_theme_toggle", use_container_width=True,
            help=f"Switch to {nxt} mode",
        )
        if clicked:
            st.session_state["basis_theme"] = nxt
            _save_pref(nxt)
            st.rerun()
    st.markdown('<hr class="basis-rule">', unsafe_allow_html=True)


# --- data tables -----------------------------------------------------------
def themed_dataframe(df, fmt, colorers=None, *, na_rep=None, height=None,
                     pal: dict | None = None) -> None:
    """st.dataframe whose cells follow the active palette.

    Streamlit's data grid is canvas-rendered and ignores the app's CSS, so it
    would otherwise stay on the config (dark) theme. It *does* honour a Styler's
    per-cell `background-color`/`color`, so we paint a palette base first, then
    let any `colorers` (e.g. red/green move tints) win on their own columns.

    colorers: list of (subset_columns, styler_func). Applied after the base so
    their `color:` overrides the base text colour on those columns.
    """
    pal = pal or palette()
    kw = {"use_container_width": True, "hide_index": True}
    if height is not None:                  # Streamlit rejects height=None
        kw["height"] = height
    try:
        sty = df.style.format(fmt, na_rep=na_rep) if na_rep is not None else df.style.format(fmt)
        sty = sty.set_properties(**{"background-color": pal["surface"], "color": pal["text"]})
        for subset, fn in (colorers or []):
            sty = sty.apply(fn, subset=subset)
        st.dataframe(sty, **kw)
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
        st.dataframe(plain, **kw)


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
