"""Shared branding + chart helpers for the report PDFs (volreport, skewreport).

These scripts run standalone (the app calls them as subprocesses), so this module
is imported top-level — it sets the matplotlib Agg backend before pyplot loads.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLACK, YELLOW = "#0A0A0A", "#F5C518"
RICH, CHEAP, NEUTRAL = "#C62828", "#2E7D32", "#9E9E9E"   # red / green / grey

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 9.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def data_uri(path: Path) -> str:
    """Embed an image as a base64 data URI (self-contained HTML)."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    if data[:4] == b"\x89PNG":
        mime = "image/png"
    elif data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    else:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def png(fig, dpi: int = 160) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def color(direction: int) -> str:
    return RICH if direction < 0 else CHEAP if direction > 0 else NEUTRAL


def launch_chromium(p):
    """Launch a headless Chromium for rendering, in priority order:
      1. a forced channel (env BASIS_PDF_CHANNEL = 'msedge' / 'chrome'),
      2. the bundled Playwright Chromium (dev PC / full deploy bundle),
      3. the machine's installed Microsoft Edge, then Google Chrome,
      4. last resort — a one-off `playwright install` (needs internet).
    Step 3 is what lets the SLIM work-PC bundle (no bundled browser) still produce PDFs,
    using the PC's own Edge/Chrome — both are Chromium, so page.pdf() works the same."""
    import os
    chan = os.getenv("BASIS_PDF_CHANNEL", "").strip()
    if chan:
        return p.chromium.launch(channel=chan)
    try:
        return p.chromium.launch()
    except Exception:
        pass
    for c in ("msedge", "chrome"):
        try:
            return p.chromium.launch(channel=c)
        except Exception:
            pass
    import subprocess as _sp, sys as _sys
    _sp.run([_sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
            check=False)
    return p.chromium.launch()


def render_pdf(html: str, out_path, landscape: bool = False) -> str:
    """Render HTML to an A4 PDF via headless Chromium (portrait by default; landscape=True for
    the wide fixed-income strip). Uses the bundled Chromium, or the machine's Edge/Chrome if
    none is bundled — see launch_chromium."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = launch_chromium(p)
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(path=str(out_path), format="A4", print_background=True, landscape=landscape,
                     margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        finally:
            browser.close()
    return out_path


def render_png(html: str, out_path, width: int = 600, scale: int = 2) -> str:
    """Render HTML to a SINGLE full-height PNG via headless Chromium — one continuous page,
    no A4 pagination (used for the email-overview image). `width` is the CSS layout width;
    `scale` is the device pixel ratio (2 = retina-crisp). Same browser selection as render_pdf
    (bundled Chromium, else the machine's Edge/Chrome)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": width, "height": 1200},
                                    device_scale_factor=scale)
            page.set_content(html, wait_until="load")
            page.screenshot(path=str(out_path), full_page=True)
        finally:
            browser.close()
    return out_path


def legend(ax, rich_lbl: str, cheap_lbl: str):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=RICH, mec="white", ms=7, label=rich_lbl),
               Line2D([0], [0], marker="o", ls="", mfc=CHEAP, mec="white", ms=7, label=cheap_lbl),
               Line2D([0], [0], marker="o", ls="", mfc=NEUTRAL, mec="white", ms=7, label="Neutral")]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True,
              framealpha=0.9, edgecolor="#ccc")


def flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged rows: rich (descending z) then cheap (ascending z)."""
    f = df[df["direction"] != 0]
    return pd.concat([f[f["direction"] < 0].sort_values("z", ascending=False),
                      f[f["direction"] > 0].sort_values("z")])


def reflag(df: pd.DataFrame, threshold: float, hi, lo) -> pd.DataFrame:
    """Re-derive (signal, direction) from the stored z-score and a flag threshold,
    so the report's trigger can be tuned at generation time without re-running the
    daily job (z doesn't depend on the threshold; only the flag does). `hi` / `lo`
    are (label, direction) applied where z >= +threshold / z <= -threshold."""
    out = df.copy()
    z = pd.to_numeric(out["z"], errors="coerce")
    out["signal"] = (pd.Series("—", index=out.index, dtype=object)
                     .mask(z >= threshold, hi[0]).mask(z <= -threshold, lo[0]))
    out["direction"] = (pd.Series(0, index=out.index, dtype="int64")
                        .mask(z >= threshold, hi[1]).mask(z <= -threshold, lo[1]))
    return out


def bar_png(df: pd.DataFrame, value_col: str = "spread", value_fmt: str = "{:+.1f}",
            xlabel: str = "", n: int = 8) -> str:
    """Diverging horizontal bar of the z-score, rich (red) vs cheap (green), grey when
    NOT flagged. `n` = top-n each side; pass n=None to show the WHOLE book (every market
    ranked by z, flagged in colour and the rest greyed, labelling only the flagged)."""
    d = df.dropna(subset=["z"])
    whole = n is None
    if whole:
        sub = d.sort_values("z")          # barh draws bottom-up: most cheap at bottom
    else:
        sub = pd.concat([d.sort_values("z", ascending=False).head(n),
                         d.sort_values("z").head(n)]).drop_duplicates("ticker").sort_values("z")
    if sub.empty:
        sub = d.sort_values("z")
    big = len(sub) > 28
    h = min(9.6, max(2.8, 0.16 * len(sub))) if whole else max(2.8, 0.30 * len(sub))
    fig, ax = plt.subplots(figsize=(6.1, h))
    ypos = range(len(sub))
    ax.barh(list(ypos), sub["z"], color=[color(x) for x in sub["direction"]],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(sub["market"], fontsize=5.2 if big else 6.8)
    span = float(sub["z"].abs().max()) or 1.0
    for i, (z, v, dirn) in enumerate(zip(sub["z"], sub[value_col], sub["direction"])):
        if whole and dirn == 0:           # whole-book view: only label the flagged bars
            continue
        ax.text(z + (span * 0.02 if z >= 0 else -span * 0.02), i, value_fmt.format(v),
                va="center", ha="left" if z >= 0 else "right", fontsize=5 if big else 6, color="#333")
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.set_xlabel(xlabel)
    ax.set_title("All markets ranked by spread z-score — flagged in colour" if whole else "Most stretched markets")
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.005)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return png(fig)
