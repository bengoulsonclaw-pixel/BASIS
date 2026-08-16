"""BASIS Weekly Review — the cross-module weekly wrap, exception-based by design.

One branded PDF, Monday morning, that collects what each module is ALREADY shouting
about — the review never invents analysis. Every bullet earns its place by clearing
its home module's existing bar: the vol book's own z-flags, the curve monitor's ±2σ,
COT's crowded-positioning cutoffs, the TA report's saved threshold mode, the daily
correlation-break alert, the seasonality screen's min-sample windows and the precious-
metals monitor's flow lines. Quiet modules simply don't appear.

Front page: an AI-polished lead paragraph (desk voice, same chain as the scorecard),
then the "Worth a look this week" list — one line per exception, tagged by module,
delta-annotated "new this week" vs "3rd week running" against the previous edition
(data/signals/weekreview_last.json, rolled only on the scheduled --baseline run — the
convreport/sigscore convention, so ad-hoc previews never eat the week's deltas).
Then the releases digest: what dropped this week (one reused synopsis line each) and
the coming week's calendar.

Everything reads existing parquet/json stores — no Bloomberg at run time, builds
offline in seconds. Every collector is isolated: a module whose store is missing or
broken drops out of the report instead of killing it.

Run standalone (the app and the weekly emailer call it as a subprocess):
    python src/weekreview.py data/Weekly_Review.pdf --asof "2026-08-10" --baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault("DATAFEED_MODE", "snapshot")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from reportkit import pretty_date, data_uri, render_pdf, ordinal

TEMPLATES = ROOT / "templates"
ASSETS = TEMPLATES / "assets"
SIG_DIR = ROOT / "data" / "signals"
LAST_FILE = SIG_DIR / "weekreview_last.json"
BREAK_LOG = SIG_DIR / "sectorcorr_break_log.json"

# per-module caps — the front page is a screen, not an inventory
VOL_MAX = 3
CURVE_MAX = 3
COT_EXTREME_MAX = 2
COT_SHIFT_MAX = 2
TA_MAX = 3
CORR_MAX = 2
SEAS_MAX = 3
SEAS_HORIZON_WEEKS = 2      # "opening in the next fortnight"
SEAS_MIN_HIT = 0.70         # seasmon.HIT_STRONG — only strong windows make the wrap
PM_MAX = 2
PM_STALE_DAYS = 45          # PM monitor is monthly; drop the section if it's gone stale


# ---------------------------------------------------------------------------
# bullet plumbing
# ---------------------------------------------------------------------------
def _bullet(tag: str, key: str, text: str, metric: str = "", sub: str = "",
            bar: float | None = None) -> dict:
    """`key` identifies the same story across editions (for the new/repeat badge);
    `text` may carry **bold** markers. `metric`/`sub` is the flag's headline number,
    pulled out of the prose and set in its own right-hand column; `bar` (0-1) draws
    the small gauge under it."""
    return {"tag": tag, "key": f"{tag}:{key}", "text": text, "metric": metric, "sub": sub,
            "bar": (max(0.05, min(1.0, bar)) if bar is not None else None)}


def _md2html(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s or "")


def _fmt(v, dec=1) -> str:
    return f"{v:,.{dec}f}"


# ---------------------------------------------------------------------------
# collectors — one per module; each returns a list of bullets (possibly empty)
# and reads only what's already on disk
# ---------------------------------------------------------------------------
def collect_vol() -> list:
    """Names beyond the vol book's own z-flags (direction != 0 in the cached
    cross-section), with the 1σ daily move and the tightest realized peer."""
    det = pd.read_parquet(SIG_DIR / "volatility.parquet")
    fl = det[det["direction"] != 0].copy()
    fl = fl.reindex(fl["z"].abs().sort_values(ascending=False).index).head(VOL_MAX)
    peers_cm = None
    try:
        import volcorr
        peers_cm = volcorr.load("realized", "1y")
    except Exception:
        pass
    by_ticker = det.set_index("ticker")
    out = []
    for r in fl.itertuples(index=False):
        side = "rich" if r.direction < 0 else "cheap"
        txt = (f"**{r.market}** implied vol screens **{side}** — "
               f"{ordinal(int(round(r.pctl)))} percentile of the year; "
               f"1σ daily move ≈ {_fmt(r.iv_sd, int(min(r.px_dec, 2)))} points.")
        if peers_cm is not None:
            try:
                import volcorr
                ps = volcorr.peers(r.ticker, peers_cm, 0.6)
                pick = None
                for p, c in ps:                      # prefer a peer flagged the other way
                    if p in by_ticker.index and by_ticker.loc[p, "direction"] == -r.direction:
                        pick = (p, c, by_ticker.loc[p, "market"], True)
                        break
                if pick is None and ps:
                    p, c = ps[0]
                    if p in by_ticker.index:
                        pick = (p, c, by_ticker.loc[p, "market"], False)
                if pick:
                    _, c, mkt, opp = pick
                    txt += (f" Tightest realized peer: {mkt} (ρ {c:.2f}"
                            + (", screening the other way)." if opp else ")."))
            except Exception:
                pass
        out.append(_bullet("VOL", f"{r.ticker}:{side}", txt,
                           metric=f"z {r.z:+.1f}", sub="IV−RV spread, 1y",
                           bar=abs(r.z) / 4.0))
    # STIR book, same bar, bp units
    try:
        stir = pd.read_parquet(SIG_DIR / "stirvol.parquet")
        sfl = stir[stir["direction"] != 0]
        for r in sfl.reindex(sfl["z"].abs().sort_values(ascending=False).index).itertuples(index=False):
            if len(out) >= VOL_MAX + 1:
                break
            side = "rich" if r.direction < 0 else "cheap"
            out.append(_bullet("VOL", f"{r.ticker}:{side}",
                               f"**{r.market}** rate vol screens **{side}** — "
                               f"1σ daily move ≈ {_fmt(r.iv_bp)} bp.",
                               metric=f"z {r.z:+.1f}", sub="IV−RV spread, 1y",
                               bar=abs(r.z) / 4.0))
    except Exception:
        pass
    return out


def collect_curve() -> list:
    """The 19-spread curve/RV book's own flags (|z| ≥ its 2σ bar), with the
    full-history percentile."""
    from src import curvemon
    m = curvemon.monitor()
    fl = m[(m["direction"] != 0) & m["z"].notna()].copy()
    fl = fl.reindex(fl["z"].abs().sort_values(ascending=False).index).head(CURVE_MAX)
    out = []
    for r in fl.itertuples(index=False):
        pct = (f" — {ordinal(int(round(r.pctl)))} percentile of the stored history"
               if r.pctl == r.pctl else "")
        out.append(_bullet("CURVE", f"{r.key}:{r.signal}",
                           f"**{r.name}** screens **{r.signal.lower()}** at {_fmt(r.level, r.dp)} "
                           f"{r.unit}{pct}.",
                           metric=f"z {r.z:+.1f}", sub="vs its 1y band",
                           bar=abs(r.z) / 4.0))
    return out


def collect_cot() -> list:
    """Friday's positioning extremes (the COT report's own 80/20 cutoff) plus the
    week's largest net shifts (its _weekly_shifts selection)."""
    from src import cotdata
    detail, hist = cotdata.compute(max_age_hours=1e9)   # cache only — never a fetch here
    out = []
    ext = detail[detail["direction"] != 0].head(COT_EXTREME_MAX)   # already |idx−50| sorted
    for r in ext.itertuples(index=False):
        side = "long" if r.direction > 0 else "short"
        out.append(_bullet("COT", f"{r.ticker}:{side}",
                           f"**{r.market}** {r.category.lower()} positioning is **crowded "
                           f"{side}** — net {r.net_pct_oi:+.0f}% of open interest "
                           f"({r.date:%d %b}).",
                           metric=f"idx {r.cot_index:.0f}", sub="COT index, 0–100",
                           bar=abs(r.cot_index - 50.0) / 50.0))
    try:
        from src import cotreport
        for s in cotreport._weekly_shifts(detail, hist)[:COT_SHIFT_MAX]:
            out.append(_bullet("COT", f"shift:{s['market']}",
                               f"**{s['market']}**: the week's net positioning move was "
                               f"**{s['chg']:+,.0f}** contracts ({s['pct']:+.1f}% of open "
                               f"interest).",
                               metric=f"{s['z']:+.1f}σ", sub="vs 3y of weekly moves",
                               bar=abs(s["z"]) / 5.0))
    except Exception:
        pass
    return out


def collect_technical() -> list:
    """The TA report's own saved threshold mode — top conviction picks only — plus
    one line from the scorecard: how the composite's judged cohort resolved."""
    out = []
    try:
        from src.specs import ta_report_defaults
        from src import convreport
        rd = ta_report_defaults("ficc")
        sig = pd.read_parquet(SIG_DIR / "opportunities.parquet")
        sel = convreport.select(sig, top_n=TA_MAX, mode="threshold",
                                min_conviction=float(rd["min_conviction"]),
                                min_score=float(rd["min_score"]))
        for p in sel["constructive"] + sel["cautious"]:
            side = "constructive" if p in sel["constructive"] else "cautious"
            out.append(_bullet("TECH", f"{p['market']}:{side}",
                               f"**{p['market']}** clears the desk's technical quality bar on the "
                               f"**{side}** side — **{p['n']}** strategies align, "
                               f"score {p['score']:+.0f}.",
                               metric=f"conv {p['conviction']:.0f}", sub="of 100",
                               bar=p["conviction"] / 100.0))
    except Exception:
        traceback.print_exc()
    return out


SCORECARD_CALLS = 3          # best/worst resolved calls shown in the folded-in scorecard


def scorecard_section() -> dict | None:
    """The Weekly Signal Scorecard folded in as this report's technical section —
    everything computed by sigscore's own functions, nothing re-derived. Returns None
    (section absent) if the ledger is missing."""
    from src import sigledger, sigscore
    led = sigledger.load()
    if led is None or led.empty:
        return None
    wk = sigscore.week_slice(led)
    core_wk = wk[wk["strategy"] != sigledger.CONFLUENCE]
    n_resolved = sum(len(sigscore.resolved_cohort(led, h)) for h in sigledger.HORIZONS)
    r21 = sigscore.resolved_cohort(led, 21)
    r21 = r21[r21["strategy"] != sigledger.CONFLUENCE]
    hit21_wk = r21["hit21"].mean() * 100.0 if len(r21) else None
    hi = led["date"].max()
    y1 = led[(led["date"] >= hi - pd.DateOffset(years=1))
             & (led["strategy"] != sigledger.CONFLUENCE)]
    hit21_1y = y1["hit21"].mean() * 100.0 if y1["hit21"].notna().any() else None
    best, worst = sigscore.call_rows(led)
    prev = sigscore.load_last()          # the standalone scorecard's own edition store —
    lg_rows, lg_labels = sigscore.league_rows(led, prev.get("league_1y", {}))
    return {
        "league": lg_rows, "league_labels": lg_labels, "league_has_prev": bool(prev),
        "n_week": f"{len(core_wk):,}",
        "longs": f"{int((core_wk['direction'] > 0).sum()):,}",
        "shorts": f"{int((core_wk['direction'] < 0).sum()):,}",
        "n_resolved": f"{n_resolved:,}",
        "hit21_wk": f"{hit21_wk:.1f}%" if hit21_wk is not None else "—",
        "hit21_wk_bg": sigscore._cell_bg(hit21_wk, 50.0, 8.0) if hit21_wk is not None else "",
        "hit21_1y": f"{hit21_1y:.1f}%" if hit21_1y is not None else "—",
        "verdicts": sigscore.verdict_rows(led),
        "best_calls": best[:SCORECARD_CALLS], "worst_calls": worst[:SCORECARD_CALLS],
    }


def collect_corr_breaks() -> list:
    """What the daily correlation-break alert flagged this week — read from the
    append-only log the alert now keeps; fall back to recomputing the latest
    session if the log is empty."""
    week_ago = date.today() - timedelta(days=7)
    rows = []
    try:
        log = json.loads(BREAK_LOG.read_text(encoding="utf-8"))
        rows = [r for r in log if date.fromisoformat(r["date"]) >= week_ago]
    except Exception:
        pass
    if not rows:
        try:
            from src import sectorcorr
            ex = sectorcorr.percentile_extremes("realized", date.today())
            rows = [{"date": str(date.today()), **r._asdict()} if hasattr(r, "_asdict")
                    else {"date": str(date.today()), **dict(r)}
                    for r in ex.head(CORR_MAX).to_dict("records")]
        except Exception:
            return []
    best = {}                                    # dedupe a pair across the week, keep the widest
    for r in rows:
        k = tuple(sorted((r["a"], r["b"])))
        if k not in best or abs(r["diff"]) > abs(best[k]["diff"]):
            best[k] = r
    rows = sorted(best.values(), key=lambda r: -abs(r["diff"]))[:CORR_MAX]
    try:
        from src.universe import name as _uname
    except Exception:
        _uname = lambda t: t
    out = []
    for r in rows:
        kind = "breaking down" if r.get("kind") == "breakdown" else "moving in lockstep"
        a, b = (_uname(r["a"]) or r["a"]), (_uname(r["b"]) or r["b"])
        out.append(_bullet("CORR", f"{r['a']}|{r['b']}:{r.get('kind', '')}",
                           f"**{a} × {b}** are {kind} — 1-month correlation "
                           f"**{r['corr_1m']:+.2f}** against {r['corr_1y']:+.2f} over the year, "
                           f"an extreme of the pair's own range.",
                           metric=f"Δρ {r['diff']:+.2f}", sub="1M vs 1Y",
                           bar=abs(r["diff"]) / 1.2))
    return out


def collect_seasonality() -> list:
    """Strong seasonal windows (the screen's own ≥5-year sample gate) opening in
    the next fortnight."""
    from src import seasmon, universe
    frames = seasmon.load_frames(universe.TREND_UNIVERSE)
    weekly = seasmon.weekly_changes(frames)
    this_wk = date.today().isocalendar()[1]
    cands = []
    for t in universe.TREND_UNIVERSE:
        try:
            win = seasmon.best_windows(weekly, t)
        except Exception:
            continue
        if win is None or win.empty:
            continue
        for r in win.itertuples(index=False):
            ahead = (int(r.start) - this_wk) % 52
            if ahead <= SEAS_HORIZON_WEEKS and r.hit >= SEAS_MIN_HIT:
                cands.append((t, ahead, r))
    cands.sort(key=lambda x: (-x[2].hit, -x[2].n))
    out, seen = [], set()
    for t, ahead, r in cands:
        if t in seen:
            continue
        seen.add(t)
        opens = "is open now" if ahead == 0 else ("opens next week" if ahead == 1
                                                 else f"opens within {ahead} weeks")
        out.append(_bullet("SEAS", f"{t}:{r.label}",
                           f"**{universe.name(t)}** enters a seasonal window ({r.label}) that "
                           f"{opens} — **{r.dir.lower()}**, median move {r.med:+.1f}%.",
                           metric=f"{r.wins}/{r.n} yrs", sub="observed sample",
                           bar=float(r.hit)))
        if len(out) >= SEAS_MAX:
            break
    return out


def collect_pm() -> list:
    """The precious-metals monitor's flow lines — only metals whose ETF/vault
    numbers actually moved (its own rounds-to-zero suppression)."""
    d = json.loads((ROOT / "data" / "pm_monitor.json").read_text(encoding="utf-8"))
    try:
        asof = datetime.strptime(d.get("asof", ""), "%d %b %Y").date()
        if (date.today() - asof).days > PM_STALE_DAYS:
            return []
        stamp = f" (as of {d['asof']})" if (date.today() - asof).days > 10 else ""
    except Exception:
        stamp = ""
    out = []
    for m in d.get("metals", []):
        parts = []
        if m.get("etf_disp"):
            word = "outflows" if (m.get("etf_chg_1m") or 0) < 0 else "inflows"
            parts.append(f"ETF holdings saw {word} of **{m['etf_disp'].lstrip('+-')}** on the month")
        if m.get("comex_disp"):
            word = "drew" if str(m["comex_disp"]).startswith("-") else "built"
            parts.append(f"exchange registered stocks {word} {m['comex_disp'].lstrip('+-')}")
        if parts:
            out.append(_bullet("METALS", f"{m['key']}:flows",
                               f"**{m['name']}** flows moved — " + " and ".join(parts) + f"{stamp}.",
                               metric=(m.get("etf_disp") or m.get("comex_disp") or ""),
                               sub="on the month"))
        if len(out) >= PM_MAX:
            break
    return out


STIR_LEDGER = ROOT / "data" / "stir_meeting_ledger.json"
STIR_WOW_MIN_BP = 5.0        # a meeting must reprice this much on the week to make the wrap
_stir_now: dict = {}         # this run's implied map — saved by save_stir_ledger on baseline runs


def _stir_implied() -> dict:
    """bank key -> {decision iso: implied per-meeting bp} — stirpaths'
    default_bank_fit, so the Monday wrap quotes EXACTLY the numbers the Home
    cards / cockpits / ledger show (a stale quarterlies-only inline copy here
    once diverged from the cockpit by 28bp on an ECB meeting)."""
    from src import stirpaths
    today = date.today()
    out = {}
    for bk in stirpaths.BANKS:
        ip = stirpaths.default_bank_fit(bk, today)
        if ip is not None:
            out[bk] = {m.isoformat(): float(bp)
                       for m, bp in zip(ip.meetings, ip.per_meeting_bp)}
    return out


def collect_stir() -> list:
    """Meeting-week alerts: a central bank decides within 7 days (what the strip
    prices, which STIR options die around it), plus the week's biggest per-meeting
    repricings vs the ledger the Monday baseline run rolls."""
    from src import stirpaths
    global _stir_now
    today = date.today()
    cur = _stir_implied()
    _stir_now = {"asof": today.isoformat(), "per": cur}
    try:
        prev = json.loads(STIR_LEDGER.read_text(encoding="utf-8")).get("per", {})
    except Exception:
        prev = {}
    out = []
    for bk, bank in stirpaths.BANKS.items():
        for m in [x for x in bank.meetings if today <= x <= today + timedelta(days=7)]:
            bp = cur.get(bk, {}).get(m.isoformat(), 0.0)
            d, p = stirpaths.implied_odds(bp, bank.step_bp)
            odds_str = "a hold" if d == "hold" else f"{p * 100:.0f}% of a {d}"
            d0 = prev.get(bk, {}).get(m.isoformat())
            dtx = f" ({bp - d0:+.0f}bp vs last week)" if d0 is not None else ""
            optl = []
            for pr in stirpaths.bank_products(bk):
                if not pr.has_options:
                    continue
                for r in stirpaths.expiry_rows(pr, today, 2):
                    if r.kind == "Option" and today <= r.expiry <= today + timedelta(days=7):
                        optl.append(f"{pr.short} {r.month} options die {r.expiry:%a %d %b} — "
                                    f"{'before' if r.expiry < m else 'on/after'} the decision")
            opt_txt = (" " + "; ".join(optl[:2]) + ".") if optl else ""
            out.append(_bullet("STIR", f"{bk}:{m.isoformat()}",
                               f"**{bank.name}** decides **{m:%A %d %b}** — the strip prices "
                               f"**{bp:+.0f}bp**, {odds_str}{dtx}.{opt_txt}",
                               metric=f"{bp:+.0f} bp", sub=f"{bank.meeting_name} this week",
                               bar=abs(bp) / bank.step_bp))
    moves = []
    for bk in cur:
        for iso, bp in cur[bk].items():
            d0 = prev.get(bk, {}).get(iso)
            if d0 is not None and abs(bp - d0) >= STIR_WOW_MIN_BP:
                moves.append((abs(bp - d0), bk, iso, bp - d0, bp))
    for _, bk, iso, dd, bp in sorted(moves, reverse=True)[:3]:
        bank = stirpaths.BANKS[bk]
        m = date.fromisoformat(iso)
        word = "added easing" if dd < 0 else "took easing out"
        out.append(_bullet("STIR", f"{bk}:wow:{iso}",
                           f"**{bank.name} repricing** — the market {word} at the {m:%b %Y} "
                           f"{bank.meeting_name}: **{dd:+.0f}bp** on the week "
                           f"(now {bp:+.0f}bp priced).",
                           metric=f"{dd:+.0f} bp", sub="on the week", bar=abs(dd) / 25.0))
    return out


def save_stir_ledger() -> None:
    """Roll the per-meeting implied ledger — called on baseline (scheduled) runs
    only, so ad-hoc previews never eat the week-on-week comparison."""
    if _stir_now:
        STIR_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        STIR_LEDGER.write_text(json.dumps(_stir_now, indent=1), encoding="utf-8")


SECTIONS = [
    ("Volatility", collect_vol),
    ("Curve / RV", collect_curve),
    ("STIR meetings", collect_stir),
    ("Positioning", collect_cot),
    ("Technical", collect_technical),
    ("Correlations", collect_corr_breaks),
    ("Seasonality", collect_seasonality),
    ("Precious metals", collect_pm),
]


def collect_all() -> tuple[list, dict, list]:
    """Run every collector; a module that fails drops out rather than killing the
    report. Returns (flat bullets, per-module counts, template section groups)."""
    bullets, counts, groups = [], {}, []
    for label, fn in SECTIONS:
        try:
            got = fn() or []
        except Exception:
            print(f"[weekreview] {label} collector failed:", file=sys.stderr)
            traceback.print_exc()
            got = []
        if got:
            counts[label] = len(got)
            bullets.extend(got)
            groups.append({"title": label, "rows": got})
    return bullets, counts, groups


# ---------------------------------------------------------------------------
# releases digest — what dropped, what's coming
# ---------------------------------------------------------------------------
def _synopsis_lines(dropped: list) -> dict:
    """label -> one reused synopsis headline, where an engine has one on disk."""
    lines = {}
    labels = {e["label"] for e in dropped}
    if any("OPEC" in l for l in labels):
        try:
            newest = max((ROOT.parent / "opec").glob("*_synopsis.json"), key=lambda p: p.stat().st_mtime)
            lines["OPEC MOMR"] = json.loads(newest.read_text(encoding="utf-8")).get("headline", "")
        except Exception:
            pass
    try:                                        # WGC / WPIC publish unscheduled — check by mtime
        week_ago = datetime.now().timestamp() - 8 * 86400
        for p in (ROOT / "data" / "pm_releases").glob("*_synopsis.json"):
            if p.stat().st_mtime >= week_ago:
                d = json.loads(p.read_text(encoding="utf-8"))
                lines[f"{d.get('publisher', 'PM')} {d.get('edition', '')}".strip()] = d.get("headline", "")
    except Exception:
        pass
    return lines


def releases_digest() -> tuple[list, list]:
    from src import repcal
    today = date.today()
    ev = repcal.calendar_events()
    dropped = [e for e in ev if today - timedelta(days=7) <= e["date"] < today
               and "COT" not in e["label"]]     # COT already has its own section
    coming = [e for e in ev if today <= e["date"] <= today + timedelta(days=7)]
    syn = _synopsis_lines(dropped)
    drop_rows = []
    for e in sorted(dropped, key=lambda x: x["date"]):
        head = next((h for k, h in syn.items() if k and e["label"] in k or k in e["label"]), "")
        drop_rows.append({"date": e["date"].strftime("%a %d %b"), "label": e["label"],
                          "head": head, "tip": e.get("tip", "")})
    for k, h in syn.items():                    # unscheduled releases (WGC/WPIC) — mtime-detected
        if h and not any(k in r["label"] or r["label"] in k for r in drop_rows):
            drop_rows.append({"date": "this week", "label": k, "head": h, "tip": ""})
    come_rows = [{"date": e["date"].strftime("%a %d %b"), "label": e["label"],
                  "tip": e.get("tip", "")} for e in sorted(coming, key=lambda x: x["date"])]
    return drop_rows, come_rows


def calendar_rows(drop_rows: list, come_rows: list) -> list:
    """One merged release-calendar table: last week's drops (with their synopsis line)
    then the week ahead, in date order."""
    rows = [{"date": r["date"], "label": r["label"], "status": "dropped",
             "note": r["head"] or "full synopsis available on request"} for r in drop_rows]
    rows += [{"date": r["date"], "label": r["label"], "status": "due", "note": ""}
             for r in come_rows]
    return rows


# ---------------------------------------------------------------------------
# previous-edition store — "new this week" vs "still stretched (3rd week)"
# ---------------------------------------------------------------------------
def load_last() -> dict:
    try:
        return json.loads(LAST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_last(asof, streaks: dict) -> None:
    LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_FILE.write_text(json.dumps({"asof": str(asof), "streaks": streaks},
                                    ensure_ascii=False, indent=1), encoding="utf-8")


def apply_streaks(bullets: list, prev: dict) -> dict:
    """Annotate each bullet in place; return the new streak map for save_last."""
    old = prev.get("streaks", {})
    streaks = {}
    for b in bullets:
        n = int(old.get(b["key"], 0)) + 1
        streaks[b["key"]] = n
        # Quiet by default: flag what's NEW, and only call out persistence once it's a
        # story (3rd week on). A badge on every line is noise, not information.
        b["badge"] = ("new this week" if n == 1 and prev else
                      f"{ordinal(n)} week running" if n >= 3 else "")
    return streaks


# ---------------------------------------------------------------------------
# intro — deterministic template, AI-polished into the desk voice by default
# ---------------------------------------------------------------------------
INTRO_SYSTEM = (
    "You are a senior futures strategist writing the opening paragraph of the desk's "
    "cross-asset Weekly Review for professional clients. Rewrite the terse note you "
    "are given so it reads like a real person framing the week ahead — flowing, plain-"
    "English, confident but relaxed, three to five sentences.\n"
    "HARD RULES: keep EVERY number and percentage EXACTLY as given, wrapped in the "
    "same **bold** markers; keep every product and module name; stay neutral and "
    "observational — these are screens the desk's models flagged, never advice: no "
    "buy/sell/recommend language and nothing that implies the reader should act.\n"
    "Return ONLY a JSON array with the single rewritten string.")


def intro_text(bullets: list, counts: dict, n_new: int, drop_rows: list,
               come_rows: list) -> str:
    n = len(bullets)
    mods = len(counts)
    parts = [f"The desk's screens go into the week with **{n}** flags across "
             f"**{mods}** modules"
             + (f" — **{n_new}** of them new this week." if n_new else ".")]
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        parts.append("The heaviest reads sit in "
                     + ", ".join(f"**{k}** ({v})" for k, v in top) + ".")
    if drop_rows:
        parts.append(f"**{len(drop_rows)}** release{'s' if len(drop_rows) != 1 else ''} "
                     f"dropped this week ({', '.join(r['label'] for r in drop_rows[:3])}).")
    if come_rows:
        parts.append(f"The coming week brings {', '.join(r['label'] for r in come_rows[:4])}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def render_html(asof: str = "", ai_polish: bool = True, update_baseline: bool = False) -> str:
    bullets, counts, groups = collect_all()
    drop_rows, come_rows = releases_digest()
    try:
        scorecard = scorecard_section()
    except Exception:
        print("[weekreview] scorecard section failed:", file=sys.stderr)
        traceback.print_exc()
        scorecard = None
    prev = load_last()
    streaks = apply_streaks(bullets, prev)
    n_new = sum(1 for b in bullets if b.get("badge") == "new this week")

    intro = intro_text(bullets, counts, n_new, drop_rows, come_rows)
    if scorecard and scorecard["hit21_wk"] != "—":
        intro += (f" On the track record, the signal book hit **{scorecard['hit21_wk']}** on the "
                  f"calls judged at 21 sessions this week — the full scorecard is inside.")
    if ai_polish:
        from reportkit import ai_rewrite
        intro = ai_rewrite([intro], INTRO_SYSTEM)[0]

    if update_baseline or not LAST_FILE.exists():
        save_last(asof or str(date.today()), streaks)
    if update_baseline or not STIR_LEDGER.exists():
        save_stir_ledger()

    for b in bullets:
        b["html"] = _md2html(b["text"])

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("weekreview_report.html").render(
        asof=pretty_date(asof or str(date.today())),
        intro=_md2html(intro),
        bullets=bullets, groups=groups, n_bullets=len(bullets), n_new=n_new,
        n_modules=len(counts), has_prev=bool(prev),
        scorecard=scorecard,
        cal_rows=calendar_rows(drop_rows, come_rows),
        drop_rows=drop_rows, come_rows=come_rows,
        logo=data_uri(ASSETS / "logo.png"), watermark=data_uri(ASSETS / "building.jpg"),
    )


def build_pdf(out_path, asof: str = "", ai_polish: bool = True,
              update_baseline: bool = False) -> str:
    return render_pdf(render_html(asof, ai_polish, update_baseline), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--asof", default="")
    ap.add_argument("--baseline", action="store_true",
                    help="roll the previous-edition store (the scheduled weekly run passes "
                         "this; ad-hoc previews don't, so kicking the tyres never eats the deltas)")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the AI polish of the intro (deterministic template only)")
    args = ap.parse_args()
    out = build_pdf(args.out, asof=args.asof, ai_polish=not args.no_ai,
                    update_baseline=args.baseline)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
