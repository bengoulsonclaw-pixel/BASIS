"""Regenerate the BASIS raster icons from the SVG sources in this folder.

    (repo venv)  python assets/brand/build_icons.py

Renders each SVG crisply at native pixel sizes via the repo's bundled Playwright Chromium
(no new dependency), then writes:

  basis-icon-512.png, basis-icon-256.png     mark-only app icon -> Streamlit page_icon (browser tab)
  basis-social-1024/512/400.png              mark + wordmark    -> social avatars (X / Instagram / …)
  basis.ico                                  multi-resolution desktop-shortcut icon:
                                               256 / 128 px = full "> BASIS" logo (basis-icon-full.svg)
                                                64 / 48 / 32 / 16 px = mark only (basis-icon.svg),
                                               because the wordmark is unreadable below ~64 px.
"""
import io
import os
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_local = ROOT / "playwright-browsers"
if _local.exists() and "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_local)

from PIL import Image                                   # noqa: E402
from playwright.sync_api import sync_playwright         # noqa: E402


def _render(browser, svg: Path, size: int) -> Image.Image:
    """Render one SVG to a transparent-cornered RGBA image of exactly size x size px."""
    pg = browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
    pg.goto(svg.resolve().as_uri())
    pg.wait_for_timeout(150)
    png = pg.screenshot(omit_background=True)           # keep the rounded-tile corners transparent
    pg.close()
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _build_ico(images, path: Path) -> None:
    """Pack several PIL images into one multi-resolution .ico (PNG-compressed entries)."""
    imgs = sorted(images, key=lambda im: im.size[0])
    blobs = []
    for im in imgs:
        buf = io.BytesIO(); im.save(buf, format="PNG"); blobs.append(buf.getvalue())
    out = io.BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(imgs)))     # ICONDIR: reserved, type=1, count
    offset = 6 + 16 * len(imgs)
    for im, blob in zip(imgs, blobs):
        w, h = im.size
        out.write(struct.pack("<BBBBHHII",              # ICONDIRENTRY
                              w if w < 256 else 0, h if h < 256 else 0,
                              0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    for blob in blobs:
        out.write(blob)
    path.write_bytes(out.getvalue())


def main() -> None:
    mark, full, social = HERE / "basis-icon.svg", HERE / "basis-icon-full.svg", HERE / "basis-social.svg"
    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            _render(b, mark, 512).save(HERE / "basis-icon-512.png")
            _render(b, mark, 256).save(HERE / "basis-icon-256.png")
            for s in (1024, 512, 400):
                _render(b, social, s).save(HERE / f"basis-social-{s}.png")
            _build_ico([_render(b, full, 256), _render(b, full, 128),
                        _render(b, mark, 64), _render(b, mark, 48),
                        _render(b, mark, 32), _render(b, mark, 16)],
                       HERE / "basis.ico")
        finally:
            b.close()
    print("rebuilt:", ", ".join(sorted(f.name for f in HERE.glob("basis*.png"))), "+ basis.ico")


if __name__ == "__main__":
    main()
