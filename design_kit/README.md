# BASIS — design kit

BASIS is a futures-desk research dashboard (Streamlit app + branded PDF reports)
run by a multi-asset futures broker. Live at basisterminal.com. This kit holds
the current visual language as standalone HTML so it can be critiqued and
redesigned in Claude Design.

## The ask

Make it look **more professional** — sharper execution of the existing identity,
not a new identity. The gold-on-dark brand (tuned to the XP yellow #F5C518) also
appears in client PDFs, so the palette family should survive the redesign.
Think: institutional research portal / terminal — tighter typography, better
information density, cleaner tables, consistent components.

## Hard constraints (it must ship in Streamlit)

- Component styling, tokens, typography, spacing, color: fully translatable.
- Radical layout inventions (free-form canvases, complex grids): NOT translatable.
  Streamlit is a vertical flow of blocks with columns — design within that.
- Both themes matter: dark is the default desk look, light is the "brochure" look.
- The sidebar carries the logo + page nav; the masthead carries the wordmark,
  tagline and a sun/moon theme toggle.

## Files

- `tokens/colors.html`, `tokens/type.html` — current tokens, both palettes
- `components/*.html` — faithful replicas of the core UI pieces
- `screens/home-dark.html` — composite of the Home page for overall feel
