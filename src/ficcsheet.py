"""FICC product tearsheet — every book's current read on one product, on one branded page.

A client rings about coffee. Answering used to mean visiting eight module pages in turn —
Technical Analysis, Volatility, Skew, Term Structure, Seasonality, COT, Put/Call, Hot Sheet —
each with its own product picker, then retyping the numbers. The Equities side has had a
one-click company tearsheet since 2026-07; the FICC book, which is the one actually brokered,
had none (Ben, 2026-08-26).

Nothing here is computed and nothing is pulled: every number is read from the ticker-keyed
stores the morning pull already writes into data/signals/. A book with nothing to say about
the product is left out rather than drawn empty, so a quiet product gets a short page.

Client-facing, so every store string goes through reportkit.client_safe() first — the stores
are written for the desk and say "buy skew" / "sell the rally" / "· sell the bond", which must
never reach a client page ([[client-commentary-not-advice]]).

    python src/ficcsheet.py "KCA Comdty" out.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the `src` package importable (deepstore / universe use relative imports), and src/
# itself, so `reportkit` resolves the same way it does for the other report modules.
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reportkit import (client_safe, data_uri, ordinal, png,        # noqa: E402
                       pretty_date, render_pdf, NEUTRAL, RICH)
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from jinja2 import Environment, FileSystemLoader                  # noqa: E402

from src import universe as u                                     # noqa: E402

SIG = ROOT / "data" / "signals"
TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"
GOLD = "#C8901A"

HIST_DAYS = 252          # a year of context on every chart
RECENT_SHEET_DAYS = 30   # how far back to look for Hot Sheet mentions
MAX_MENTIONS = 6


# ── store access ─────────────────────────────────────────────────────────────
def _read(name: str):
    try:
        d = pd.read_parquet(SIG / f"{name}.parquet")
    except Exception:
        return None
    return None if d is None or d.empty else d


def _row(name: str, ticker: str):
    """The single row a cross-section store holds for this product, as a plain dict.

    A product missing from a store is normal, not an error: putcall drops products whose
    option legs are too thin to quote ([[project-putcall-ratios]]), skew covers 58 of 70.
    """
    d = _read(name)
    if d is None or "ticker" not in d.columns:
        return None
    hit = d[d["ticker"] == ticker]
    return None if hit.empty else hit.iloc[0].to_dict()


def _f(v):
    """float or None — the stores mix NaN, None and numpy scalars."""
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, dec: int = 2) -> str:
    v = _f(v)
    return "—" if v is None else f"{v:,.{dec}f}"


def _pctl_txt(v) -> str:
    v = _f(v)
    return "—" if v is None else f"{ordinal(v)} pctl"


def _dirword(direction) -> str:
    """Long / Short / Neutral — the same words the client already sees on the Technical
    Analysis Report. On fixed income these are yield-space signals (the TA engine runs on
    yields), which the page footnotes rather than silently translating."""
    d = int(_f(direction) or 0)
    return "Long" if d > 0 else "Short" if d < 0 else "Neutral"


# ── the books ────────────────────────────────────────────────────────────────
def _tech_block(ticker: str):
    """Which technical methods currently flag the product, and which way — straight from the
    daily opportunities store, the same rows the Technical Analysis hub scores."""
    d = _read("opportunities")
    if d is None or "instruments" not in d.columns:
        return None
    hit = d[d["instruments"] == ticker]
    if hit.empty:
        return None
    rows = []
    for r in hit.itertuples(index=False):
        dirn = int(_f(getattr(r, "direction", 0)) or 0)
        rows.append({
            "strategy": str(r.strategy),
            "dir": dirn,
            "read": _dirword(dirn),
            "metric": _fmt(getattr(r, "metric", None), 2),
            "metric_label": str(getattr(r, "metric_label", "") or ""),
            "context": client_safe(getattr(r, "context", "")),
        })
    rows.sort(key=lambda x: (-abs(x["dir"]), x["strategy"]))
    up = sum(1 for x in rows if x["dir"] > 0)
    dn = sum(1 for x in rows if x["dir"] < 0)
    net = "Long" if up > dn else "Short" if dn > up else "Mixed"
    return {"rows": rows, "up": up, "down": dn, "flat": len(rows) - up - dn, "net": net}


def _vol_block(ticker: str):
    """The vol book's three legs side by side — level (implied vs realized), term slope and
    skew. Each is z-scored by the module that owns it against that product's own trailing
    year, so they are comparable here even though they measure different things."""
    rows = []
    v = _row("volatility", ticker)
    if v:
        iv, rv = _f(v.get("iv")), _f(v.get("rv"))
        rows.append({"label": "Implied vs realized (1M)",
                     "value": f"{_fmt(iv, 1)} vs {_fmt(rv, 1)} vols" if iv is not None else "—",
                     "z": _f(v.get("z")), "pctl": _pctl_txt(v.get("pctl")),
                     "note": client_safe(v.get("signal", ""))})
    t = _row("termstructure", ticker)
    if t:
        a, b = _f(t.get("iv_1m")), _f(t.get("iv_3m"))
        rows.append({"label": "Term structure (1M → 3M)",
                     "value": f"{_fmt(a, 1)} → {_fmt(b, 1)} vols" if a is not None else "—",
                     "z": _f(t.get("z")), "pctl": _pctl_txt(t.get("pctl")),
                     "note": client_safe(t.get("signal", ""))})
    s = _row("skew", ticker)
    if s:
        p, atm, c = _f(s.get("put")), _f(s.get("atm")), _f(s.get("call"))
        rows.append({"label": "Skew (put / ATM / call)",
                     "value": f"{_fmt(p, 1)} / {_fmt(atm, 1)} / {_fmt(c, 1)}" if p is not None else "—",
                     "z": _f(s.get("z")), "pctl": _pctl_txt(s.get("pctl")),
                     "note": client_safe(s.get("signal", ""))})
    return {"rows": rows} if rows else None


def _pos_block(ticker: str):
    """Standing positioning — the CFTC net and the option open-interest balance, each quoted
    against the product's own history because raw levels do not compare across products
    (index options sit structurally put-heavy from hedging)."""
    rows = []
    c = _row("cot", ticker)
    if c:
        net_pct, idx = _f(c.get("net_pct_oi")), _f(c.get("cot_index"))
        cat = str(c.get("category", "") or "").strip()
        rows.append({
            "label": f"CFTC net — {cat}" if cat else "CFTC net",
            "value": f"{net_pct:+.0f}% of open interest" if net_pct is not None else "—",
            "hist": f"{ordinal(idx)} of its 3y range" if idx is not None else "—",
            "note": client_safe(c.get("signal", "")),
            "asof": str(c.get("date", ""))[:10],
        })
    p = _row("putcall", ticker)
    if p:
        rows.append({
            "label": "Option open interest — puts ÷ calls",
            "value": _fmt(p.get("pc_oi"), 2),
            "hist": _pctl_txt(p.get("oi_pctl")),
            "note": client_safe(p.get("signal", "")),
            "asof": "",
        })
    return {"rows": rows} if rows else None


def _seas_block(ticker: str):
    r = _row("seas_spread_screen", ticker)
    if not r or _f(r.get("z")) is None:
        return None
    return {"legs": client_safe(r.get("legs", "")), "unit": str(r.get("unit", "") or ""),
            "now": _f(r.get("now")), "norm": _f(r.get("norm")), "dev": _f(r.get("dev")),
            "z": _f(r.get("z")), "years": _f(r.get("years")),
            "note": client_safe(r.get("signal", ""))}


def _mentions(ticker: str) -> list:
    """What the Hot Sheet has said about this product lately — the desk's own recent read,
    which is where a client conversation usually starts. Rows the Hot Sheet marks
    internal_only never leave the building, so they are dropped here."""
    d = _read("hotsheet_history")
    if d is None or not {"ticker", "date"} <= set(d.columns):
        return []
    d = d[d["ticker"] == ticker].copy()
    if "internal_only" in d.columns:
        d = d[~d["internal_only"].fillna(False).astype(bool)]
    if d.empty:
        return []
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] >= d["date"].max() - pd.Timedelta(days=RECENT_SHEET_DAYS)]
    d = d.sort_values("date", ascending=False)
    out, seen = [], set()
    for r in d.itertuples(index=False):
        key = str(getattr(r, "key", "") or getattr(r, "text", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append({"date": f"{r.date:%d %b}", "tag": str(getattr(r, "tag", "") or ""),
                    "text": client_safe(str(getattr(r, "text", "")).replace("**", ""))})
        if len(out) >= MAX_MENTIONS:
            break
    return out


# ── charts ───────────────────────────────────────────────────────────────────
def _tidy(ax) -> None:
    ax.tick_params(labelsize=6.5)
    ax.grid(True, color="#EEE", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _price_png(ticker: str, is_fi: bool) -> str:
    """A year of the product's own signal-space series — yields for fixed income, price
    elsewhere — so the chart matches the dimension every technical read on this page is
    computed in ([[project-fi-yield-ta]])."""
    try:
        from src import deepstore
        s = deepstore.get_ta([ticker])[ticker].dropna().iloc[-HIST_DAYS:]
    except Exception:
        return ""
    if len(s) < 20:
        return ""
    fig, ax = plt.subplots(figsize=(6.9, 1.75))
    ax.plot(s.index, s.values, color="#0A0A0A", lw=1.1)
    for win, col in ((50, GOLD), (200, NEUTRAL)):
        if len(s) >= win:
            ax.plot(s.index, s.rolling(win).mean(), color=col, lw=0.9, label=f"{win}d")
    ax.set_ylabel("Yield (%)" if is_fi else "Price", fontsize=7)
    _tidy(ax)
    if len(s) >= 50:
        ax.legend(fontsize=6, frameon=False, loc="best")
    return png(fig)


def _vol_png(ticker: str) -> str:
    """Implied against realized over the year — the picture behind the vol table's first row."""
    d = _read("volatility_history")
    if d is None or "ticker" not in d.columns:
        return ""
    h = d[d["ticker"] == ticker].copy()
    if len(h) < 20:
        return ""
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date").iloc[-HIST_DAYS:]
    fig, ax = plt.subplots(figsize=(6.9, 1.5))
    ax.plot(h["date"], h["iv"], color=RICH, lw=1.0, label="implied 1M")
    ax.plot(h["date"], h["rv"], color=NEUTRAL, lw=1.0, label="realized")
    ax.set_ylabel("vols", fontsize=7)
    _tidy(ax)
    ax.legend(fontsize=6, frameon=False, loc="best")
    return png(fig)


# ── assembly ─────────────────────────────────────────────────────────────────
def _asof() -> str:
    try:
        return json.loads((SIG / "meta.json").read_text(encoding="utf-8")).get("as_of", "")
    except Exception:
        return ""


def _level(ticker: str):
    """Last traded level and the session move, off the RAW store — this is a published number
    a client can check against a screen, so it must not be panama-adjusted
    ([[project-deep-store-switch]])."""
    try:
        from src import deepstore
        s = deepstore.get_raw([ticker])[ticker].dropna()
    except Exception:
        return None, None, None
    if s.empty:
        return None, None, None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else None
    chg = None if prev is None else last - prev
    pct = None if not prev else chg / abs(prev) * 100.0
    return last, chg, pct


def gather(ticker: str) -> dict:
    """Everything the books currently hold on one product. Reads caches only."""
    if ticker not in u.INSTRUMENTS:
        raise ValueError(f"{ticker} is not in the universe")
    is_fi = bool(u.is_fixed_income(ticker))
    v = _row("volatility", ticker) or {}
    dec = int(min(_f(v.get("px_dec")) or 2, 2))
    last, chg, pct = _level(ticker)
    sd = _f(v.get("iv_sd"))
    return {
        "ticker": ticker,
        "market": u.name(ticker),
        "asset": u.asset(ticker),
        "region": u.region(ticker) or "",
        "is_fi": is_fi,
        "dec": dec,
        "last": last, "chg": chg, "pct": pct,
        "last_txt": _fmt(last, dec),
        "chg_txt": ("—" if chg is None else f"{chg:+,.{dec}f}"
                    + ("" if pct is None else f"  ({pct:+.2f}%)")),
        # the daily 1σ move the option market is pricing, in the product's own points —
        # the number that says whether the last session's move was ordinary.
        "sd_txt": ("" if sd is None else f"1σ daily ≈ {_fmt(sd, dec)} pts"),
        "sd_mult": (None if (not sd or chg is None) else abs(chg) / sd),
        "asof": _asof(),
        "tech": _tech_block(ticker),
        "vol": _vol_block(ticker),
        "pos": _pos_block(ticker),
        "seas": _seas_block(ticker),
        "mentions": _mentions(ticker),
        "price_png": _price_png(ticker, is_fi),
        "vol_png": _vol_png(ticker),
    }


def render_html(d: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("ficcsheet.html").render(
        d=d, asof=pretty_date(d.get("asof") or ""),
        mention_days=RECENT_SHEET_DAYS,
        logo=data_uri(ASSETS / "logo.png"),
    )


def build_pdf(ticker: str, out_path) -> str:
    return render_pdf(render_html(gather(ticker)), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-page FICC product tearsheet")
    ap.add_argument("ticker")
    ap.add_argument("out_pdf")
    args = ap.parse_args()
    build_pdf(args.ticker, args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
