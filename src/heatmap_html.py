"""On-screen Market heatmap as native HTML/CSS — used by both home pages (FICC + Equities).

Same nested-treemap geometry (`_slice` / `_grid_slices`) and σ→colour as the PIL `eqheatmap`
renderer, but emitted as absolutely-positioned <div> tiles (percentage coordinates) instead of a
raster. Rendered through ``st.components.v1.html`` it stays crisp at any zoom, uses the app font,
scales with the column width, and shows a name / % / σ tooltip on hover — so the heatmap reads as
"properly run on screen" rather than a pasted report picture. PIL-free; no new dependency.

    render_html(sections_in, height_px, sub_headers=True) -> str      # one <style>+<div> block

``sections_in`` mirrors ``eqheatmap.render_image``::

    [(section, [(sub, [(tile_name, pct, sigma), ...]), ...]), ...]

For a one-level map (FICC: sector → product) pass a single sub per section and
``sub_headers=False`` — the sub header is then omitted and the tiles fill the section body.
"""
from __future__ import annotations

import html as _html

from .eqheatmap import _grid_slices, _heat_color, _heat_color_sigma, _slice

_CAP, _FLOOR, _FALLBACK = 4.0, 0.5, 1.0     # tile size ∝ |σ|, clamped — identical to eqheatmap
_V = 1000.0                                  # virtual canvas; rects are converted to % of this
_W_NOM = 1120.0                              # assumed on-screen width (px) — for text-fit gating only

_STYLE = """
<style>
  html,body{margin:0;padding:0;background:transparent;}
  .bx-tm{position:relative;width:100%;height:__H__px;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
  .bx-t{position:absolute;box-sizing:border-box;border:1px solid rgba(255,255,255,.13);
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        overflow:hidden;color:#fff;text-align:center;line-height:1.06;cursor:default;
        transition:filter .09s ease,outline-color .09s ease;}
  .bx-t:hover{filter:brightness(1.16) saturate(1.04);outline:2px solid #F5C518;
              outline-offset:-2px;z-index:7;}
  .bx-nm{font-weight:700;max-width:97%;overflow:hidden;text-overflow:ellipsis;
         display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-clamp:2;
         text-shadow:0 1px 2px rgba(0,0,0,.55);}
  .bx-v{margin-top:.14em;opacity:.96;white-space:nowrap;font-variant-numeric:tabular-nums;
        text-shadow:0 1px 2px rgba(0,0,0,.55);}
  .bx-h{position:absolute;box-sizing:border-box;display:flex;align-items:center;
        padding:0 .45em;font-weight:800;letter-spacing:.03em;color:#F5C518;
        overflow:hidden;white-space:nowrap;text-transform:uppercase;}
  .bx-h-sec{background:#0A0A0A;}
  .bx-h-sub{background:#2B2B2B;}
</style>
"""


def _psize(sig) -> float:
    if sig is None or sig != sig:
        return _FALLBACK
    return max(_FLOOR, min(_CAP, abs(float(sig))))


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


def _rgb(t) -> str:
    return f"rgb({int(t[0])},{int(t[1])},{int(t[2])})"


def _clamp(lo, v, hi):
    return max(lo, min(hi, v))


def _pos(x, y, w, h) -> str:
    """An absolute virtual rect -> a left/top/width/height style string in %."""
    return (f"left:{x / _V * 100:.3f}%;top:{y / _V * 100:.3f}%;"
            f"width:{w / _V * 100:.3f}%;height:{h / _V * 100:.3f}%")


def render_html(sections_in, height_px, sub_headers: bool = True) -> str:
    """Build the treemap as a self-contained <style>+<div> block for st.components.v1.html.
    Tile AREA ∝ |σ| (clamped 0.5–4), colour green(up)/red(down) deepening with |σ|; sections and
    tiles are ordered biggest-move-first. Returns "" when there's nothing to draw."""
    H = float(max(140, int(height_px)))

    def vpx(px):                              # real px -> virtual units (legible header heights)
        return px / H * _V

    # 1) section → sub → tiles, with sizes; biggest-move-first at each level (mirrors eqheatmap).
    sections = []
    for sec, subs in sections_in:
        sub_list = []
        for subname, items in subs:
            data = [(nm, float(pct), sig) for (nm, pct, sig) in items]
            if not data:
                continue
            data.sort(key=lambda t: -_psize(t[2]))
            sizes = [_psize(t[2]) for t in data]
            sub_list.append((subname, data, sizes, sum(sizes)))
        if sub_list:
            sub_list.sort(key=lambda s: -s[3])
            sections.append((sec, sub_list, sum(s[3] for s in sub_list)))
    if not sections:
        return ""
    sections.sort(key=lambda s: -s[2])

    out = []

    def header(x, y, w, h_v, text, kind):
        hpx = h_v / _V * H
        if hpx < 8 or w / _V * _W_NOM < 16:
            return
        fpx = _clamp(9, hpx * 0.62, 15)
        out.append(f'<div class="bx-h bx-h-{kind}" style="{_pos(x, y, w, h_v)};'
                   f'font-size:{fpx:.0f}px">{_esc(text)}</div>')

    def tile(x, y, w, h, nm, pct, sig):
        rh, rw = h / _V * H, w / _V * _W_NOM
        col = _heat_color_sigma(sig) if (sig is not None and sig == sig) else _heat_color(pct)
        has = sig is not None and sig == sig
        vlab = f"{pct:+.2f}%  ·  {sig:+.1f}σ" if has else f"{pct:+.2f}%"
        body = ""
        if rh >= 12 and rw >= 20:                                  # name (up to 2 lines, ellipsis)
            npx = _clamp(8, min(rh * 0.30, rw * 0.24), 15)
            body = f'<div class="bx-nm" style="font-size:{npx:.0f}px">{_esc(nm)}</div>'
            if rh >= 30 and rw >= 46:                              # + the %/σ value line
                vp = _clamp(8, round(npx * 0.82), 12)
                body += f'<div class="bx-v" style="font-size:{vp:.0f}px">{_esc(vlab)}</div>'
        out.append(f'<div class="bx-t" style="{_pos(x, y, w, h)};background:{_rgb(col)}" '
                   f'title="{_esc(nm)} — {_esc(vlab)}">{body}</div>')

    SEC_GAP, SUB_GAP = _V * 0.006, _V * 0.004
    for (sec, sub_list, _ss), (sx, sy, sw, sh) in zip(
            sections, _slice([s[2] for s in sections], 0, 0, _V, _V, "v")):
        x, y = sx, sy + SEC_GAP
        w, h = sw, sh - 2 * SEC_GAP
        if w < 6 or h < 6:
            continue
        sec_head = min(h * 0.18, vpx(24))
        header(x, y, w, sec_head, sec.upper(), "sec")
        for (subname, data, sizes, _su), (ux, uy, uw, uh) in zip(
                sub_list, _slice([s[3] for s in sub_list], x, y + sec_head, w, h - sec_head, "h")):
            ax, ay = ux + SUB_GAP, uy + SUB_GAP
            aw, ah = uw - 2 * SUB_GAP, uh - 2 * SUB_GAP
            if aw < 4 or ah < 4:
                continue
            sub_head = 0.0
            if sub_headers and (ah / _V * H) > 46:
                sub_head = min(ah * 0.18, vpx(16))
                header(ax, ay, aw, sub_head, subname.upper(), "sub")
            for (nm, pct, sig), (bx, by, bw, bh) in zip(
                    data, _grid_slices(sizes, ax, ay + sub_head, aw, ah - sub_head)):
                tile(bx, by, bw, bh, nm, pct, sig)

    return _STYLE.replace("__H__", str(int(H))) + '<div class="bx-tm">' + "".join(out) + "</div>"
