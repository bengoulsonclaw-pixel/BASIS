"""Shared branding + chart helpers for the report PDFs (volreport, skewreport).

These scripts run standalone (the app calls them as subprocesses), so this module
is imported top-level — it sets the matplotlib Agg backend before pyplot loads.
"""
from __future__ import annotations

import base64
import re
from io import BytesIO
from pathlib import Path

import logging as _logging
import matplotlib
matplotlib.use("Agg")
# Silence the benign 'findfont: Font family Arial not found' fallback noise. Arial is a
# standard Windows font, but while matplotlib rebuilds its font cache (e.g. after a version
# bump) every lookup transiently logs that warning — which floods a report's stderr and can
# mask the real error if a run does fail. This only hushes the fallback logging, not errors.
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
import matplotlib.pyplot as plt
import pandas as pd

BLACK, YELLOW = "#0A0A0A", "#F5C518"
RICH, CHEAP, NEUTRAL = "#C62828", "#2E7D32", "#9E9E9E"   # red / green / grey
WARN = "#F5C518"          # house yellow (--yellow) marks — z at trigger, sign unconfirmed
WARN_TX = "#C8901A"       # the report's established amber TEXT shade (drift, seasonal ☼)
WARN_SIGNALS = {"Premium compressing", "Discount narrowing"}

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 9.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def pretty_date(value) -> str:
    """The house long-form as-of stamp: `Mon 20th July 2026`.

    Takes whatever the callers happen to be holding — a date/datetime/Timestamp, or the
    `"%Y-%m-%d %H:%M"` string run_daily writes into meta.json — and returns the input
    UNCHANGED if it can't be parsed, so a bad stamp can never raise inside a title band.

    Two deliberate choices: the clock time is dropped (these are daily reports; the minute
    a PDF happened to render is noise to the reader), and this formats at DISPLAY time only
    — meta.json's `as_of` stays sortable ISO because it doubles as a cache key and is parsed
    back with `strptime` elsewhere.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        dt = pd.to_datetime(value)
        if pd.isna(dt):
            return text
    except Exception:
        return text
    d = int(dt.day)
    suffix = "th" if 11 <= d % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"{dt:%a} {d}{suffix} {dt:%B} {dt.year}"


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


def row_color(direction: int, signal: str = "") -> str:
    """Directional colour, with the house-amber warning tier for sign-mismatched
    triggers (|z| past the flag but implied still on the usual side of realized)."""
    if direction < 0:
        return RICH
    if direction > 0:
        return CHEAP
    return WARN if signal in WARN_SIGNALS else NEUTRAL


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


def legend(ax, rich_lbl: str, cheap_lbl: str, warn_lbl: str | None = None):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=RICH, mec="white", ms=7, label=rich_lbl),
               Line2D([0], [0], marker="o", ls="", mfc=CHEAP, mec="white", ms=7, label=cheap_lbl)]
    if warn_lbl:
        handles.append(Line2D([0], [0], marker="o", ls="", mfc=WARN, mec="white", ms=7, label=warn_lbl))
    handles.append(Line2D([0], [0], marker="o", ls="", mfc=NEUTRAL, mec="white", ms=7, label="Neutral"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True,
              framealpha=0.9, edgecolor="#ccc")


def flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged rows: rich (descending z) then cheap (ascending z)."""
    f = df[df["direction"] != 0]
    return pd.concat([f[f["direction"] < 0].sort_values("z", ascending=False),
                      f[f["direction"] > 0].sort_values("z")])


def reflag(df: pd.DataFrame, threshold: float, hi, lo, gate_col: str | None = None) -> pd.DataFrame:
    """Re-derive (signal, direction) from the stored z-score and a flag threshold,
    so the report's trigger can be tuned at generation time without re-running the
    daily job (z doesn't depend on the threshold; only the flag does). `hi` / `lo`
    are (label, direction) applied where z >= +threshold / z <= -threshold.

    `gate_col` (e.g. "spread") demands the SIGN of that column agree with the flag's
    direction before a rich/cheap label is issued: a stretched z on a still-negative
    spread means the discount is narrowing — not rich vol — and labelling it a sale
    would advise selling implied below delivered movement. Sign-mismatched extremes
    keep direction 0 under neutral 'Discount narrowing' / 'Premium compressing'."""
    out = df.copy()
    z = pd.to_numeric(out["z"], errors="coerce")
    out["signal"] = (pd.Series("—", index=out.index, dtype=object)
                     .mask(z >= threshold, hi[0]).mask(z <= -threshold, lo[0]))
    out["direction"] = (pd.Series(0, index=out.index, dtype="int64")
                        .mask(z >= threshold, hi[1]).mask(z <= -threshold, lo[1]))
    if gate_col and gate_col in out.columns:
        lvl = pd.to_numeric(out[gate_col], errors="coerce")
        hi_mis = (z >= threshold) & ~(lvl > 0)
        lo_mis = (z <= -threshold) & ~(lvl < 0)
        out.loc[hi_mis, "signal"] = "Discount narrowing"
        out.loc[lo_mis, "signal"] = "Premium compressing"
        out.loc[hi_mis | lo_mis, "direction"] = 0
    return out


_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten", "eleven", "twelve"]


def num_word(n: int) -> str:
    """Spell out small counts (one … twelve); fall back to digits beyond that."""
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else str(n)


def join_names(names) -> str:
    """['A', 'B', 'C'] -> 'A, B and C' (names HTML-escaped for safe inlining)."""
    import html as _html
    names = [_html.escape(str(x)) for x in names]
    if len(names) <= 1:
        return names[0] if names else ""
    if len(names) == 2:
        return names[0] + " and " + names[1]
    return ", ".join(names[:-1]) + " and " + names[-1]


def names_by_z(g: pd.DataFrame) -> list:
    """Market names, most-stretched (|z|) first."""
    return list(g.reindex(g["z"].abs().sort_values(ascending=False).index)["market"])


# ---------------------------------------------------------------------------
# Shared caption machinery — the Fable rewrite pipe (ai_polish.py run by the
# Morning Coffee interpreter) plus the small text helpers the per-chart caption
# builders share. Every report's captions fall back to their deterministic
# template notes on ANY failure, so generation never depends on the model.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent


def mc_python() -> str:
    """The Morning Coffee interpreter (has anthropic + the API key); '' if not found."""
    base = Path(r"C:\Users\Ben\AppData\Local\Python")
    for exe in sorted(base.glob("pythoncore-*-64/python.exe"), reverse=True):
        if exe.exists():
            return str(exe)
    import shutil
    return shutil.which("python") or ""


def ai_rewrite(texts: list, system: str) -> list:
    """`texts` rewritten in a natural desk voice via ai_polish.py (Fable 5 chain, custom
    system prompt) in ONE batched call; the originals on ANY failure — offline, no key,
    timeout — so generation never blocks on the model."""
    if not texts:
        return texts
    import json as _json
    import subprocess as _sp
    import tempfile as _tmp
    try:
        py = mc_python()
        if not py:
            return texts
        with _tmp.TemporaryDirectory() as td:
            inp, outp, sysf = Path(td) / "in.json", Path(td) / "out.json", Path(td) / "system.txt"
            inp.write_text(_json.dumps(texts, ensure_ascii=False), encoding="utf-8")
            sysf.write_text(system, encoding="utf-8")
            r = _sp.run([py, str(_REPO_ROOT / "ai_polish.py"), str(inp), str(outp), str(sysf)],
                        capture_output=True, text=True, timeout=240)
            if r.returncode == 0 and outp.exists():
                got = _json.loads(outp.read_text(encoding="utf-8"))
                if isinstance(got, list) and len(got) == len(texts):
                    return [g if isinstance(g, str) and g.strip() else t for g, t in zip(got, texts)]
    except Exception:
        pass
    return texts


def md_bold(s: str) -> str:
    """Escape, then **…** -> <b>…</b>, for safe HTML injection of a caption."""
    import html as _html
    import re as _re
    return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _html.escape(s))


def ordinal(n) -> str:
    """English ordinal: 1->1st, 2->2nd, 3->3rd, 11->11th, 92->92nd."""
    n = int(round(float(n)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Signal and context strings inside the daily stores are written for the DESK, so they carry
# the instruction the desk would act on — "Cheap — buy skew", "Short (sell the rally)",
# "· sell the bond" (the yield->price translation on fixed income). Client-facing copy is
# neutral observation, never advice, so anything that renders a store string verbatim on a
# client page has to strip that layer first. Every report before the FICC tearsheet fed the
# AI writer numbers rather than store text, so this is the first place it was needed.
#
# The phrase set is closed: these strings are produced by our own strategies (src/specs.py),
# not by a vendor, so a map is exact where a general-purpose rewrite would not be.
_ADVICE_SUBS = [
    (r"\s*·\s*(?:buy|sell) the (?:bond|future)\b", ""),      # FI yield->price instruction
    (r"\s*\((?:buy the dip|sell the rally)\)", ""),          # Long (buy the dip)
    (r"\s*[—-]\s*(buy-dip|sell-rally) zone\b", " zone"),     # ...0.5% above — sell-rally zone
    (r"\s*[—-]\s*(?:buy-dip|sell-rally)\b", ""),             # ...9.7% above — buy-dip
    (r"\b(Cheap)\s*[—-]\s*buy (?:skew|vol)\b", r"\1"),       # Cheap — buy vol
    (r"\b(Rich)\s*[—-]\s*sell (?:skew|vol)\b", r"\1"),       # Rich — sell skew
    (r"\s*[—-]\s*(?:Buy|Sell)\b[^;·|]*", ""),                # — Buy Soy Oil / Sell Soy Meal
]
_ADVICE_LEFT = re.compile(r"\b(buy|sell|bought|sold|recommend\w*)\b", re.I)

# "Raymond James moved to **Strong Buy** from Outperform" — here Buy is the NAME of a
# broker's rating, a reportable fact, not us telling a client to do anything. Ratings are
# recognised by the phrase that introduces one (moved to / initiated at / cut to), which is
# what separates them from "Buy Soybean Oil / Sell Soybean Meal" — that stays blocked.
_RATING = re.compile(
    r"\b(?:to|at|from|as)\s+\*{0,2}(?:Strong\s+|Conviction\s+)?"
    r"(?:Buy|Sell|Hold|Overweight|Underweight|Outperform|Underperform|Accumulate|Reduce)\*{0,2}",
)


def client_safe(text) -> str:
    """Strip desk instructions out of a store string, leaving the observation.

    'Short (sell the rally) · buy the future' -> 'Short'. Raises if a buy/sell word
    survives: a client page silently carrying advice is the failure worth being loud
    about, and the caller can then extend the map above rather than ship it.
    """
    out = str(text or "")
    for pat, repl in _ADVICE_SUBS:
        out = re.sub(pat, repl, out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip(" ·—-")
    if _ADVICE_LEFT.search(_RATING.sub(" ", out)):
        raise ValueError(f"advice language survived client_safe(): {out!r}")
    return out


def when_phrase(dt: pd.Timestamp, asof: pd.Timestamp) -> str:
    """'early March' / 'mid October 2025' — a human date for a caption."""
    part = "early" if dt.day <= 10 else "mid" if dt.day <= 20 else "late"
    out = f"{part} {dt.strftime('%B')}"
    if dt.year != asof.year:
        out += f" {dt.year}"
    return out


def snapshot_stamp() -> dict:
    """{'settle': 'Mon 20th July 2026', 'pulled': '21 Jul 2026, 14:42'} read from the
    snapshot manifest — the data-provenance stamp for each report's Sources & timing
    line. Blank fields when the manifest is missing or the data isn't a live Bloomberg
    pull (demo renders then simply omit the timing sentence)."""
    import json as _json
    try:
        m = _json.loads((_REPO_ROOT / "data" / "snapshot" / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {"settle": "", "pulled": ""}
    if str(m.get("source", "")) != "bloomberg":
        return {"settle": "", "pulled": ""}
    settle = pretty_date(m.get("as_of", ""))
    pulled = ""
    try:
        dt = pd.to_datetime(str(m.get("created", "")))
        if pd.notna(dt):
            pulled = f"{dt:%d %b %Y}, {dt:%H:%M}"
    except Exception:
        pass
    return {"settle": settle, "pulled": pulled}


def history_fill_height(n: int) -> float:
    """Panel height for n flagged-history sections, chosen so pages always LOOK right:
    NEVER a single orphaned panel on a page, and light pages fill up with taller
    panels instead of white space. Mechanics: pick the per-page count c — everything
    on one page when n <= 5, else the largest of 5/4/3 whose remainder isn't 1 — then
    stretch the height to fill the page (budget ~10.15in incl. the section banner,
    ~0.76in per section of caption+margins, clamped to a readable 1.25-2.0in)."""
    if n <= 0:
        return 1.45
    c = n if n <= 5 else next((k for k in (5, 4, 3) if n % k != 1), 5)
    return min(2.0, max(1.25, 10.15 / c - 0.76))


def history_panel(market: str, direction: int, h: pd.DataFrame, value_col: str,
                  height: float = 1.45) -> str:
    """One market's full-width dual-axis 1-year panel: `value_col` (shaded, coloured by
    the signal direction, left axis) against the underlying price (grey, right axis).
    The what-is-what legend is printed once by each template above its sections."""
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(6.1, height))
    dates = pd.to_datetime(h["date"])
    c = color(direction)
    ax.axhline(0, color="#BBB", lw=0.6, zorder=1)
    ax.fill_between(dates, h[value_col], 0, color=c, alpha=0.18, zorder=2)
    ax.plot(dates, h[value_col], color=c, lw=1.1, zorder=3)
    ax.set_title(market, fontsize=6.6)
    ax.tick_params(axis="y", labelsize=5, colors=c, length=2)
    ax.tick_params(axis="x", labelsize=5, length=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2 = ax.twinx()
    ax2.plot(dates, h["price"], color="#444", lw=1.0, zorder=4)
    ax2.tick_params(axis="y", labelsize=5, colors="#444", length=2)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout(pad=0.5)
    return png(fig)


def last_episode(series: pd.Series, high_side: bool, exclude_days: int = 30):
    """When `series` (date-indexed) last sat at/beyond today's level, EXCLUDING the
    current `exclude_days` episode — (Timestamp, value) or None. `high_side` picks the
    direction: True = last time at/above today's value, False = at/below."""
    s = series.dropna()
    if len(s) < 5:
        return None
    today = float(s.iloc[-1])
    prior = s[s.index < s.index[-1] - pd.Timedelta(days=exclude_days)]
    hit = prior[prior >= today] if high_side else prior[prior <= today]
    if not len(hit):
        return None
    return hit.index[-1], float(hit.iloc[-1])


def bar_png(df: pd.DataFrame, value_col: str = "spread", value_fmt: str = "{:+.1f}",
            xlabel: str = "", n: int = 8, max_h: float = 9.6, label_fs: float | None = None) -> str:
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
    h = min(max_h, max(2.8, 0.16 * len(sub))) if whole else max(2.8, 0.30 * len(sub))
    fig, ax = plt.subplots(figsize=(6.1, h))
    ypos = range(len(sub))
    sigs = sub["signal"] if "signal" in sub.columns else pd.Series("", index=sub.index)
    ax.barh(list(ypos), sub["z"], color=[row_color(x, s) for x, s in zip(sub["direction"], sigs)],
            edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=BLACK, lw=0.9, zorder=4)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(sub["market"], fontsize=label_fs if label_fs is not None else (5.2 if big else 6.8))
    span = float(sub["z"].abs().max()) or 1.0
    # 5-day realized warning glyphs at the bar tips — ▾ = the event that inflated the
    # 1M realized has likely already happened; ▴ = realizing is accelerating past it.
    rvs = sub["rv_state"] if "rv_state" in sub.columns else pd.Series("", index=sub.index)
    rvs = rvs.fillna("")
    for i, (z, st) in enumerate(zip(sub["z"], rvs)):
        if st in ("decay", "heat"):
            x = z + (span * 0.028 if z >= 0 else -span * 0.028)
            ax.scatter([x], [i], marker="v" if st == "decay" else "^", s=16, color=RICH,
                       edgecolor="white", linewidth=0.5, zorder=5)
    for i, (z, v, dirn, sg, st) in enumerate(zip(sub["z"], sub[value_col], sub["direction"], sigs, rvs)):
        if whole and dirn == 0 and sg not in WARN_SIGNALS:   # label flagged + amber-watch bars
            continue
        off = span * (0.065 if st in ("decay", "heat") else 0.02)   # clear the warning glyph
        ax.text(z + (off if z >= 0 else -off), i, value_fmt.format(v),
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
