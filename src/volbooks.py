"""Hot Sheet seam for the vol book's OTHER two legs — the skew z-score book and the vol
term-structure book.

Both are computed every morning by src/strategies/{volatility,termstructure}.py and cached to
data/signals/{skew,termstructure}.parquet, and until now neither reached the Hot Sheet: its
`discover()` scans only the TOP level of src/, so a provider living in src/strategies/ is
invisible to it (see [[project-hot-sheet]]). The level leg already speaks, through
src/volmove.py; this is the seam for the two that didn't.

On 2026-08-26 that silence covered 8 flagged skew markets and 4 flagged term-structure
markets — including two vol books agreeing on one product (Brazilian Real: skew at its 100th
percentile AND term structure at its 93rd), which is exactly the kind of corroboration the
sheet exists to surface.

Cache-only, like every provider: reads the parquet the morning compute wrote and never
triggers a fetch. Prose is neutral observation — no buy/sell/recommend — because these rows
reach the client-facing Morning Coffee page ([[client-commentary-not-advice]]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "data" / "signals"

SKEW_MAX = 3          # rows carried per book — the sheet is a screen, not an inventory
TERM_MAX = 2
MIN_Z = 1.5           # the books' own flag bar (strategies/volatility.py Z_FLAG)


def _read(name: str) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(SIG / f"{name}.parquet")
    except Exception:
        return None
    return None if d is None or d.empty else d


def _flagged(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """The book's own flagged rows, strongest |z| first."""
    if d is None or "direction" not in d.columns or "z" not in d.columns:
        return pd.DataFrame()
    f = d[(d["direction"] != 0) & d["z"].notna()].copy()
    if f.empty:
        return f
    f["_a"] = f["z"].abs()
    return f[f["_a"] >= MIN_Z].sort_values("_a", ascending=False).head(n)


def _skew_items(hotsheet, ordinal) -> list:
    """Where a product's put wing is priced against its call wing, versus its own year.

    Distinct from src/skewreal.py's SKEW rows, which compare ONE wing against a realized
    spot-vol path. This is the put-versus-call balance — the nearest thing in the book to a
    sentiment read — and it is the one that had no provider at all."""
    d = _read("skew")
    out = []
    for r in _flagged(d, SKEW_MAX).itertuples(index=False):
        rich = r.direction < 0                      # book's convention: -1 = rich
        side = "rich" if rich else "cheap"
        pct = int(round(r.pctl)) if np.isfinite(r.pctl) else None
        out.append(hotsheet.item(
            tag="SKEW-Z", key=f"{r.ticker}:{side}", section="Volatility",
            text=(f"**{r.market}** skew screens **{side}** — its 90% put against its 110% "
                  f"call sits at the {ordinal(pct)} percentile of the year"
                  f" (put {r.put:.1f} / ATM {r.atm:.1f} / call {r.call:.1f} vols)."),
            heat=hotsheet.heat_from_z(r.z), metric=f"z {r.z:+.1f}",
            sub="(90% put − 110% call) ÷ ATM, 1y", value=float(r.z),
            ticker=r.ticker, page="Skew Volatility", book="ficc"))
    return out


def _term_items(hotsheet, ordinal) -> list:
    """Where the vol curve's 3M-over-1M slope sits against its own year — the calendar-spread
    read the term book computes and nothing surfaced."""
    d = _read("termstructure")
    out = []
    for r in _flagged(d, TERM_MAX).itertuples(index=False):
        steep = r.direction > 0
        side = "steep" if steep else "inverted"
        tail = ("front vol cheap against the back" if steep
                else "front vol bid against the back")
        pct = int(round(r.pctl)) if np.isfinite(r.pctl) else None
        out.append(hotsheet.item(
            tag="TERM", key=f"{r.ticker}:{side}", section="Volatility",
            text=(f"**{r.market}** vol term structure screens **{side}** — {tail}, at the "
                  f"{ordinal(pct)} percentile of the year "
                  f"(1M {r.iv_1m:.1f} vs 3M {r.iv_3m:.1f} vols)."),
            heat=hotsheet.heat_from_z(r.z), metric=f"z {r.z:+.1f}",
            sub="3M − 1M slope, 1y", value=float(r.z),
            ticker=r.ticker, page="Term Structure", book="ficc"))
    return out


def radar_items() -> list:
    """Hot Sheet provider — the skew z book and the vol term-structure book. Returns []
    rather than raising on any missing or malformed store."""
    try:
        from src import hotsheet
        from src.reportkit import ordinal
        return _skew_items(hotsheet, ordinal) + _term_items(hotsheet, ordinal)
    except Exception:
        return []
