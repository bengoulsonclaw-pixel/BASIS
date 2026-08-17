"""Layout check: screenshot the running BASIS server at phone AND desktop width,
and print the geometry the responsive CSS depends on (fixed top-bar height vs the
.block-container clearance, plus any element overflowing the viewport).

Usage:  .venv\\Scripts\\python.exe shot_mobile.py [tag] [url]
Kept out of the app — a dev tool for verifying the responsive CSS.
"""
import sys
from playwright.sync_api import sync_playwright

tag = sys.argv[1] if len(sys.argv) > 1 else "shot"
url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8501"

PROBE = """() => {
  const k = document.querySelector('.st-key-basis_topbar');
  let n = k, fx = null;
  while (n) { if (getComputedStyle(n).position === 'fixed') { fx = n; break; } n = n.parentElement; }
  const bc = document.querySelector('.block-container');
  const over = [];
  document.querySelectorAll('.block-container *').forEach(e => {
    const r = e.getBoundingClientRect();
    if (r.width && r.right > window.innerWidth + 2)
      over.push((e.className || e.tagName).toString().slice(0, 40) + ' ->' + Math.round(r.right));
  });
  return {
    bar: fx ? Math.round(fx.getBoundingClientRect().height) : null,
    varH: getComputedStyle(document.documentElement).getPropertyValue('--basis-topbar-h'),
    padTop: bc ? getComputedStyle(bc).paddingTop : null,
    contentTop: bc ? Math.round(bc.getBoundingClientRect().top + parseFloat(getComputedStyle(bc).paddingTop)) : null,
    overflow: over.slice(0, 6),
    scrollW: document.documentElement.scrollWidth, innerW: window.innerWidth,
  };
}"""

with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp, mob in (("mob", {"width": 412, "height": 915}, True),
                          ("desk", {"width": 1512, "height": 950}, False)):
        ctx = b.new_context(viewport=vp, device_scale_factor=2, is_mobile=mob, has_touch=mob)
        pg = ctx.new_page()
        pg.goto(url, wait_until="load", timeout=180000)
        for _ in range(60):
            pg.wait_for_timeout(2000)
            if pg.evaluate("()=>!!document.querySelector('.daycal')"):
                break
        pg.wait_for_timeout(5000)
        pg.screenshot(path=f"logs/{tag}_{name}.png")
        pg.screenshot(path=f"logs/{tag}_{name}_full.png", full_page=True)
        print(name, pg.evaluate(PROBE))
        ctx.close()
    b.close()
