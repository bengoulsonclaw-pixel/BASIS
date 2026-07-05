"""Finviz-style nested treemap (PIL) for the Equities heatmap — tile AREA ∝ the overnight move in σ,
colour = direction (green up / red down) deepening with |σ|. Ported from the Morning Coffee futures
heatmap so the Equities map matches the FICC one; self-contained (PIL only — no new dependency).

Nesting for equities: INDEX band (black header) → GICS SECTOR column (dark header) → stock tiles.
render_image(sections) takes  [(index_name, [(sector_name, [(name, pct, sigma), ...]), ...]), ...]
and returns a PIL.Image; sections/sub-sections/tiles are ordered biggest-move-first internally.
"""
from __future__ import annotations

import os
from functools import lru_cache

BLACK, DARK, YELLOW = "000000", "2B2B2B", "F5C518"


def _hex(h):
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=128)
def _load_font(bold=False, px=24):
    from PIL import ImageFont
    win = os.environ.get("WINDIR", r"C:\Windows")
    cands = ([os.path.join(win, "Fonts", "arialbd.ttf"), "arialbd.ttf"] if bold
             else [os.path.join(win, "Fonts", "arial.ttf"), "arial.ttf"])
    for c in cands:
        try:
            return ImageFont.truetype(c, px)
        except Exception:
            continue
    return ImageFont.load_default()


# ── layout: slice-and-dice (tile area ∝ size) ─────────────────────────────────
def _slice(sizes, x, y, w, h, axis):
    """1-D proportional layout: tiles fill (x,y,w,h) in order, sized ∝ `sizes`."""
    tot = float(sum(max(0.0, float(s)) for s in sizes)) or 1.0
    rects, off = [], 0.0
    for s in sizes:
        frac = max(0.0, float(s)) / tot
        if axis == "v":
            rects.append((x, y + off, w, h * frac)); off += h * frac
        else:
            rects.append((x + off, y, w * frac, h)); off += w * frac
    return rects


def _grid_slices(sizes, x, y, w, h):
    """A few aligned COLUMNS (widths ∝ column size-sum) with items stacked vertically inside
    (heights ∝ size). Column edges line up; tile area still ∝ size. Input order preserved."""
    n = len(sizes)
    if n == 0:
        return []
    if n == 1:
        return [(x, y, w, h)]
    aspect = w / max(1.0, h)
    ncol = max(1, min(n, int((n * aspect) ** 0.5 + 0.5)))
    per = n / ncol
    groups, idx = [], 0
    for c in range(ncol):
        end = n if c == ncol - 1 else int(round((c + 1) * per))
        if end > idx:
            groups.append(list(range(idx, end))); idx = end
    col_rects = _slice([sum(sizes[i] for i in g) for g in groups], x, y, w, h, "h")
    out = [(x, y, 0, 0)] * n
    for g, (cx, cy, cw, ch) in zip(groups, col_rects):
        for i, r in zip(g, _slice([sizes[i] for i in g], cx, cy, cw, ch, "v")):
            out[i] = r
    return out


# ── colour ────────────────────────────────────────────────────────────────────
def _heat_color(pct, cap=3.0):
    """±3% fallback scale (used when a stock has no σ history)."""
    t = max(-1.0, min(1.0, pct / cap)); a = abs(t) ** 0.8
    nz = (44, 46, 50)
    tg = (30, 165, 72) if t >= 0 else (208, 52, 49)
    return tuple(int(nz[i] + (tg[i] - nz[i]) * a) for i in range(3))


def _heat_color_sigma(sigma, cap=3.0):
    """Green up / red down, deepening with |σ|, saturating at `cap` σ."""
    if sigma is None or sigma != sigma:
        sigma = 0.0
    t = max(-1.0, min(1.0, sigma / cap)); a = abs(t) ** 0.7
    nz = (46, 48, 52)
    tg = (28, 165, 72) if t >= 0 else (208, 52, 49)
    return tuple(int(nz[i] + (tg[i] - nz[i]) * a) for i in range(3))


def _wrap_lines(draw, text, font, max_w):
    words = text.split()
    if not words:
        return [text]
    lines, cur = [], words[0]
    for wd in words[1:]:
        if draw.textlength(cur + " " + wd, font=font) <= max_w:
            cur += " " + wd
        else:
            lines.append(cur); cur = wd
    lines.append(cur)
    return lines


def _draw_tile(d, tmp, nm, pct, sigma, box, cap_px):
    """One stock tile: fill by σ, white outline, centred name + '%  ·  σ' label with font-shrink so
    the words fit; degrades name+value → name → truncated → % → blank so nothing overflows."""
    x0, y0, x1, y1 = box
    fill = (_heat_color_sigma(sigma) if (sigma is not None and sigma == sigma) else _heat_color(pct))
    d.rectangle([x0, y0, x1, y1], fill=fill, outline=(255, 255, 255), width=2)
    tw, hh = x1 - x0, y1 - y0
    if tw < 20 or hh < 13:
        return
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    maxw = tw - max(4, int(tw * 0.07))
    avail_h = hh - 5
    has_sig = sigma is not None and sigma == sigma
    vlabel = f"{pct:+.2f}%  ·  {sigma:+.1f}σ" if has_sig else f"{pct:+.2f}%"
    MINF = 8
    cap = max(MINF, min(cap_px, int(tw * 0.6), int(hh * 0.5)))
    words = nm.split()

    def wrapped(px):
        f = _load_font(True, px)
        ok = all(tmp.textlength(wd, font=f) <= maxw for wd in words)
        return f, _wrap_lines(tmp, nm, f, maxw), ok

    def draw_name(nf, lines, lh, ytop):
        for li, ln in enumerate(lines):
            d.text((cx, ytop + lh * (li + 0.5)), ln, fill=(255, 255, 255), font=nf, anchor="mm")

    for px in range(cap, MINF - 1, -1):                # 1) name (words fit) + value line
        nf, lines, ok = wrapped(px)
        if not ok:
            continue
        vpx, lh = max(MINF, int(px * 0.78)), px * 1.14
        need = len(lines) * lh + vpx * 1.3
        if need <= avail_h:
            ytop = cy - need / 2
            draw_name(nf, lines, lh, ytop)
            vf, vt = _load_font(True, vpx), vlabel
            if tmp.textlength(vt, font=vf) > maxw:
                vt = f"{pct:+.2f}%"
            while vt and tmp.textlength(vt, font=vf) > maxw:
                vt = vt[:-1]
            vt = vt.rstrip(" ·")
            if vt:
                d.text((cx, ytop + len(lines) * lh + vpx * 0.62), vt,
                       fill=(255, 255, 255), font=vf, anchor="mm")
            return
    for px in range(cap, MINF - 1, -1):                # 2) name only
        nf, lines, ok = wrapped(px)
        if not ok:
            continue
        lh = px * 1.14
        if len(lines) * lh <= avail_h:
            draw_name(nf, lines, lh, cy - (len(lines) * lh) / 2); return
    f, out = _load_font(True, MINF), []                # 3) truncated name at MINF
    for ln in _wrap_lines(tmp, nm, f, maxw):
        if tmp.textlength(ln, font=f) > maxw:
            s = ln
            while s and tmp.textlength(s + "…", font=f) > maxw:
                s = s[:-1]
            ln = (s + "…") if s else ""
        out.append(ln)
    lh = MINF * 1.14
    if out and len(out) * lh <= avail_h:
        draw_name(f, out, lh, cy - (len(out) * lh) / 2); return
    small = f"{pct:+.1f}%"                             # 4) just the %
    if tmp.textlength(small, font=f) <= maxw and MINF * 1.2 <= avail_h:
        d.text((cx, cy), small, fill=(255, 255, 255), font=f, anchor="mm")


def render_image(sections_in, width_in, height_in, dpi=140):
    """Nested treemap PNG: index band (biggest-move-first, full width) → GICS sector column (width ∝
    σ-sum) → stock tiles (area ∝ |σ|). `sections_in` = [(index, [(sector, [(name,pct,sigma)…])…])…].
    Returns a PIL.Image (RGB)."""
    from PIL import Image, ImageDraw
    W = max(8, int(width_in * dpi)); H = max(8, int(height_in * dpi))
    img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img); tmp = d
    blk, drk, ylw = _hex(BLACK), _hex(DARK), _hex(YELLOW)

    CAP, FLOOR, FALLBACK = 4.0, 0.5, 1.0
    def psize(sig):
        if sig is None or sig != sig:
            return FALLBACK
        return max(FLOOR, min(CAP, abs(float(sig))))

    def hdr(x, y, w, h, text, fill, txt):
        h = max(1, int(h)); w = int(w)
        d.rectangle([x, y, x + w, y + h], fill=fill)
        if h < 9 or w < 18:
            return
        px = max(7, int(h * 0.64)); f = _load_font(True, px)
        while px > 7 and d.textlength(text, font=f) > w - 8:
            px -= 1; f = _load_font(True, px)
        t = text
        while t and d.textlength(t + "…", font=f) > w - 6:
            t = t[:-1]
        if t and t != text:
            t += "…"
        if t:
            d.text((x + 5, y + h / 2), t, fill=txt, font=f, anchor="lm")

    # Build the index → sector → stock tree with sizes, biggest-move-first at each level.
    sections = []
    for sec, subs in sections_in:
        sub_list = []
        for subname, items in subs:
            data = [(nm, float(pct), sig) for (nm, pct, sig) in items]
            if not data:
                continue
            data.sort(key=lambda t: -psize(t[2]))
            sizes = [psize(t[2]) for t in data]
            sub_list.append((subname, data, sizes, sum(sizes)))
        if sub_list:
            sub_list.sort(key=lambda s: -s[3])
            sections.append((sec, sub_list, sum(s[3] for s in sub_list)))
    if not sections:
        return img
    sections.sort(key=lambda s: -s[2])

    SEC_GAP = max(2, int(dpi * 0.022)); SUB_GAP = max(1, int(dpi * 0.012))
    SEC_H = int(dpi * 0.185); SUB_H = int(dpi * 0.130)

    sec_rects = _slice([s[2] for s in sections], 0, 0, W, H, "v")
    for (sec, sub_list, _ss), (sx, sy, sw, sh) in zip(sections, sec_rects):
        x = int(sx); y = int(sy + SEC_GAP); w = int(sw); h = int(sh - 2 * SEC_GAP)
        if w < 12 or h < 12:
            continue
        sec_head = min(SEC_H, int(h * 0.16))
        hdr(x, y, w, sec_head, sec.upper(), blk, ylw)
        sub_rects = _slice([s[3] for s in sub_list], x, y + sec_head, w, h - sec_head, "h")
        for (subname, data, sizes, _su), (ux, uy, uw, uh) in zip(sub_list, sub_rects):
            ax = int(ux + SUB_GAP); ay = int(uy + SUB_GAP)
            aw = int(uw - 2 * SUB_GAP); ah = int(uh - 2 * SUB_GAP)
            if aw < 8 or ah < 8:
                continue
            sub_head = min(SUB_H, int(ah * 0.18)) if ah > 46 else 0
            cap_px = int(dpi * 0.145)
            if sub_head:
                hdr(ax, ay, aw, sub_head, subname.upper(), drk, ylw)
            prod_rects = _grid_slices(sizes, ax, ay + sub_head, aw, ah - sub_head)
            for (nm, pct, sig), (bx, by, bw, bh) in zip(data, prod_rects):
                _draw_tile(d, tmp, nm, pct, sig,
                           (int(bx), int(by), int(bx + bw), int(by + bh)), cap_px)
    return img
