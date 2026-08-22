"""Dev tool: drive the running BASIS server to the Brazil Production page and
screenshot it, so a layout/render change can be checked in the real app rather
than reasoned about. Not imported by anything.

Usage:  .venv\\Scripts\\python.exe shot_brazil.py [tag] [commodity-label]
"""
import re
import sys

from playwright.sync_api import sync_playwright

tag = sys.argv[1] if len(sys.argv) > 1 else "brazil"
pick = sys.argv[2] if len(sys.argv) > 2 else None
URL = "http://localhost:8501"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1100})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL, wait_until="networkidle", timeout=90000)
        pg.wait_for_timeout(8000)

        pg.get_by_role("button", name="06 · Fundamentals").click(timeout=60000)
        pg.wait_for_timeout(18000)
        # the tab row is CSS-uppercased, so match case-insensitively
        pg.get_by_role("button", name=re.compile("Brazil Production", re.I)).click(timeout=90000)
        pg.wait_for_timeout(14000)

        if pick:
            pg.get_by_role("combobox").first.click()
            pg.wait_for_timeout(800)
            pg.get_by_text(pick, exact=False).last.click()
            pg.wait_for_timeout(7000)

        body = pg.inner_text("body")
        for marker in ("Traceback", "Error", "error"):
            if marker in body:
                idx = body.find(marker)
                print(f"!! page text contains '{marker}': ...{body[max(0, idx - 200):idx + 700]}...")
                break
        pg.screenshot(path=f"logs/{tag}.png", full_page=True)
        print(f"saved logs/{tag}.png  ({len(body)} chars of text)")
        if errs:
            print("JS errors:", errs[:3])
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
