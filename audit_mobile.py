"""Phone-width audit: walk the modules and report what is GENUINELY too wide.

Usage:  .venv\\Scripts\\python.exe audit_mobile.py [url] [width] [ficc|equities]

Companion to shot_mobile.py — that one shows a page, this one finds what still
overflows on it. Note the classification: st.dataframe (glide-data-grid) and the
house .bt-tablewrap tables scroll inside their own frame, so their inner content
is *supposed* to be wider than the screen. Only "spills" (nothing clips or
scrolls it — it runs off the page) and "clipped" (cut off with no way to see the
rest) are real problems.
"""
import sys, json
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 412
DESK = (sys.argv[3] if len(sys.argv) > 3 else "ficc").lower()

FICC_PAGES = ["01 · Market Information", "02 · Confluence", "03 · Technical Analysis",
              "04 · Volatility", "05 · Positioning & Flow", "07 · Correlations",
              "08 · Curve / RV", "09 · Seasonality", "10 · STIR Paths"]
EQ_PAGES = ["01 · Technical Analysis", "02 · Company Fundamentals", "03 · Earnings Calendar",
            "04 · Single Stock Correlations", "05 · Index Dispersion", "06 · Client ETFs"]
PAGES = EQ_PAGES if DESK.startswith("eq") else FICC_PAGES

PROBE = """() => {
  const VW = window.innerWidth, out = [];
  const name = e => (e.dataset && e.dataset.testid) ||
    (typeof e.className === 'string' ? e.className.split(' ')[0] : '') || e.tagName;
  document.querySelectorAll('.block-container *').forEach(e => {
    const r = e.getBoundingClientRect();
    if (!r.width || r.right <= VW + 2) return;
    // classify: does any ancestor scroll or clip this?
    let n = e.parentElement, verdict = 'spills', host = null;
    while (n && n !== document.body) {
      const cs = getComputedStyle(n), nr = n.getBoundingClientRect();
      const clips = cs.overflowX === 'auto' || cs.overflowX === 'scroll' || cs.overflowX === 'hidden';
      if (clips && nr.right <= VW + 2) {
        verdict = (cs.overflowX === 'hidden') ? 'clipped' : 'scrollable';
        host = name(n) + ':' + Math.round(nr.width);
        break;
      }
      n = n.parentElement;
    }
    if (verdict === 'scrollable') return;
    if (e.parentElement && e.parentElement.getBoundingClientRect().right > VW + 2) return;  // outermost only
    let label = '';
    const svg = e.tagName === 'svg' ? e : e.querySelector && e.querySelector('svg');
    if (svg) {
      const t = [...svg.querySelectorAll('text')].map(x => ({ s: x.textContent || '', w: x.getBBox().width }))
        .sort((a, b) => b.w - a.w)[0];
      if (t) label = t.s.slice(0, 60) + ' [' + Math.round(t.w) + 'px]';
    }
    out.push({ el: name(e).slice(0, 34), w: Math.round(r.width), over: Math.round(r.right - VW),
               verdict, host, label });
  });
  return out.slice(0, 10);
}"""


def wait_ready(pg, sel=".daycal"):
    """Wait for the front door, tolerating the one-time viewport-cookie reload
    (a fresh browser profile has no basis_vw cookie, so the first load reloads)."""
    for _ in range(60):
        try:
            pg.wait_for_timeout(2000)
            if pg.evaluate(f"()=>!!document.querySelector('{sel}')"):
                return
        except Exception:
            pg.wait_for_load_state("load", timeout=60000)   # navigated mid-poll


def open_page(pg, label):
    pg.eval_on_selector_all('[data-testid="stExpandSidebarButton"] button, [data-testid="stExpandSidebarButton"]',
                            "els=>els[0]&&els[0].click()")
    pg.wait_for_timeout(1200)
    pg.get_by_text(label, exact=True).first.click(timeout=25000)
    for _ in range(50):
        pg.wait_for_timeout(2000)
        if pg.evaluate("()=>!document.querySelector('[data-testid=\"stStatusWidget\"]')"):
            break
    pg.wait_for_timeout(4000)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": W, "height": 915}, device_scale_factor=1,
                        is_mobile=W < 820, has_touch=W < 820)
    pg = ctx.new_page()
    pg.goto(url, wait_until="load", timeout=180000)
    wait_ready(pg)
    if DESK.startswith("eq"):                     # the desk switch at the top of the nav
        pg.eval_on_selector_all('[data-testid="stExpandSidebarButton"] button, [data-testid="stExpandSidebarButton"]',
                                "els=>els[0]&&els[0].click()")
        pg.wait_for_timeout(1200)
        pg.click('.st-key-side_equities button', timeout=25000)   # label is CSS-uppercased
        pg.wait_for_timeout(6000)
    for label in PAGES:
        try:
            open_page(pg, label)
            bad = pg.evaluate(PROBE)
            print(f"{label:<28} {'CLEAN' if not bad else json.dumps(bad, ensure_ascii=False)}")
            if bad:
                pg.screenshot(path="logs/audit_" + label.split('·')[0].strip() + ".png", full_page=True)
        except Exception as e:
            print(f"{label:<28} SKIP ({str(e).splitlines()[0][:70]})")
    b.close()
