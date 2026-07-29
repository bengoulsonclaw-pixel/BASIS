"""Regenerate the BASIS raster icons from the SVG sources in this folder.

    (repo venv)  python assets/brand/build_icons.py

Renders each SVG crisply at native pixel sizes via the repo's bundled Playwright Chromium
(no new dependency), then writes:

  basis-icon-512.png, basis-icon-256.png     mark-only app icon -> Streamlit page_icon (browser tab)
  basis-social-1024/512/400.png              mark + wordmark    -> social avatars (X / Instagram / …)
  basis.ico                                  desktop-shortcut icon: the clean bold mark at every
                                             size (16-256).

NB: the .ico MUST use BMP frames (bitmap_format="bmp"). PNG-compressed .ico frames open fine in
Pillow but render BLANK in Windows Explorer / on desktop shortcuts — that bit us once.
"""
import io
import os
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

ICO_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]


def _render(browser, svg: Path, size: int) -> Image.Image:
    """Render one SVG to a transparent-cornered RGBA image of exactly size x size px."""
    pg = browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
    pg.goto(svg.resolve().as_uri())
    pg.wait_for_timeout(150)
    png = pg.screenshot(omit_background=True)           # keep the rounded-tile corners transparent
    pg.close()
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _save_ico(img: Image.Image, path: Path) -> None:
    """Write a Windows-safe multi-size .ico from one image. BMP frames only (see module note)."""
    try:
        img.save(path, format="ICO", sizes=ICO_SIZES, bitmap_format="bmp")
    except TypeError:                                   # Pillow < 9.3 has no bitmap_format kwarg
        img.save(path, format="ICO", sizes=ICO_SIZES)


def main() -> None:
    mark, social = HERE / "basis-icon.svg", HERE / "basis-social.svg"
    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            m512 = _render(b, mark, 512)
            m512.save(HERE / "basis-icon-512.png")
            _render(b, mark, 256).save(HERE / "basis-icon-256.png")
            for s in (1024, 512, 400):
                _render(b, social, s).save(HERE / f"basis-social-{s}.png")
            _save_ico(m512, HERE / "basis.ico")         # desktop shortcut — clean bold mark, all sizes
        finally:
            b.close()
    print("rebuilt:", ", ".join(sorted(f.name for f in HERE.glob("basis*.png"))), "+ basis.ico")


if __name__ == "__main__":
    main()
