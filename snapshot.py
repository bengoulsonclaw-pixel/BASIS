"""Morning snapshot — pull every Bloomberg input the reports need, ONCE, and cache
it to data/snapshot/ as parquet. Run it in the morning with the Terminal open:

    (PowerShell)  $env:DATAFEED_MODE="bloomberg"; python snapshot.py

Then run the app with DATAFEED_MODE=snapshot and everything (signals, reports,
threshold tweaks, report-code edits) works all day from these files — Terminal
closed. Run with DATAFEED_MODE=mock to snapshot synthetic data (an offline test
of the whole pipeline).

    python snapshot.py --excel out.xlsx   # dump the current snapshot to one .xlsx
    python snapshot.py --oi               # WEEKLY (Mon): capture ONLY the fixed-income OI chains
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.datafeed import (MODE, get_history, get_yield_history, get_volume_history,
                          get_implied_vol_history, get_live_quote, get_skew_components,
                          get_term_structure, get_putcall, get_oi_chain, TENOR_LABELS,
                          _PUTCALL_KEYS, _OI_COLS, OI_SNAPSHOT_TICKERS)
from src.universe import INSTRUMENTS

SNAP = Path(__file__).parent / "data" / "snapshot"

# How many forward expiries of each option chain to cache (the report shows ≤10; this gives
# headroom while bounding the file). All strikes are kept — the report windows them at read.
OI_CHAIN_CAPTURE_EXPIRIES = 24

# Snapshot frames, in workbook order (one parquet + one Excel sheet each). The option chain
# (oi_chain) is the odd one out — a tidy LONG grid with a `ticker` column, not date-indexed —
# so it's saved/read directly rather than through _save.
FRAMES = (["prices", "yields", "volume", "implied_vol", "skew_put", "skew_call", "skew_atm"]
          + [f"term_{lab.lower()}" for lab in TENOR_LABELS]
          + [f"putcall_{k}" for k in _PUTCALL_KEYS] + ["oi_chain", "live"])


def _save(df: pd.DataFrame, name: str) -> int:
    SNAP.mkdir(parents=True, exist_ok=True)
    df.rename_axis("date").reset_index().to_parquet(SNAP / f"{name}.parquet", index=False)
    return int(df.shape[0])


def _oi_chain_frame(tickers) -> pd.DataFrame:
    """The 11 fixed-income products' listed-option chains (full strike grid, the strip of
    expiries) as one tidy LONG frame with a `ticker` column — the shape _read_oi_snapshot
    expects. Only OI_SNAPSHOT_TICKERS are captured (the Fixed Income book): this is a rates
    tool and pulling more / more often would burn data limits — other products pull their chain
    LIVE on demand. This seam lets the Open Interest report show REAL numbers in snapshot mode."""
    fi = [t for t in tickers if t in OI_SNAPSHOT_TICKERS]
    frames = []
    for t in fi:
        try:
            c = get_oi_chain(t, n_expiries=OI_CHAIN_CAPTURE_EXPIRIES, n_strikes=None)
        except Exception:
            c = None
        if c is not None and not c.empty:
            frames.append(c.assign(ticker=t))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["ticker", *_OI_COLS]))


def _existing_manifest() -> dict:
    p = SNAP / "manifest.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _oi_markets_cached() -> int:
    """How many products the LAST weekly OI capture wrote (read from oi_chain.parquet)."""
    p = SNAP / "oi_chain.parquet"
    if not p.exists():
        return 0
    try:
        d = pd.read_parquet(p)
        return int(d["ticker"].nunique()) if ("ticker" in d.columns and len(d)) else 0
    except Exception:
        return 0


def run_oi() -> int:
    """WEEKLY (Monday) job — capture ONLY the 11 fixed-income option chains and write
    oi_chain.parquet, leaving every other cached input untouched. Run with the Terminal up:
        (PowerShell)  $env:DATAFEED_MODE="bloomberg"; python snapshot.py --oi
    """
    SNAP.mkdir(parents=True, exist_ok=True)
    oi = _oi_chain_frame(list(INSTRUMENTS))
    n = int(oi["ticker"].nunique()) if not oi.empty else 0
    if n == 0:                                          # Bloomberg down / not connected — never WIPE
        print("OI capture returned nothing (is the Terminal/API up?) — kept the existing oi_chain.parquet.")
        return 0
    oi.to_parquet(SNAP / "oi_chain.parquet", index=False)
    m = _existing_manifest()
    m["oi_markets"] = n
    m["oi_as_of"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    (SNAP / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"OI chains captured (weekly): {n} fixed-income products -> {SNAP / 'oi_chain.parquet'}")
    return n


# ── Equities pull switches ──────────────────────────────────────────────────────────────────
# These were turned OFF (2026-07-24) when per-stock prices + the ~70k-hit weekly fundamentals
# re-pull kept re-tripping Bloomberg's -4002 workflow-review block. Re-enabled 2026-07-26: the
# Equities side pulls quotes / history / fundamentals FREE from Yahoo Finance (src/yfin.py)
# and, since 2026-07-27, index membership FREE from the tracker ETFs' published holdings
# files (src/etfmembers.py — IVV/IWM/QQQ/DIA/ISF/EUE/EXS1) — so a normal equities pull
# costs ZERO Bloomberg hits. Bloomberg INDX_MEMBERS remains only a last-resort membership
# fallback. Set BASIS_EQ_SOURCE=bloomberg to restore the old Terminal-first behaviour.
PULL_EQUITY_CONSTITUENTS = True    # membership (Bloomberg/cache) + quotes/history (Yahoo)
PULL_FUNDAMENTALS        = True    # Company Fundamentals DB refresh (Yahoo)


def run_equities() -> dict:
    """The EQUITIES pull — index membership + overnight quotes/short history, plus the
    weekly-guarded Company Fundamentals refresh. Quotes/history/fundamentals ride Yahoo
    Finance (free) by default; Bloomberg only refreshes membership when the Terminal is up.
    Gated behind PULL_EQUITY_CONSTITUENTS / PULL_FUNDAMENTALS.
    Its own job (CLI --equities, and the 'Pull equities data' button on the Equities home).
    Guarded like the futures pull: a dead pull never wipes the caches."""
    if not PULL_EQUITY_CONSTITUENTS and not PULL_FUNDAMENTALS:
        print("  Equities pull is OFF — individual-stock (constituent) prices and Company "
              "Fundamentals are disabled to keep the Bloomberg pull light.")
        print("  Equity INDEX numbers come from the FICC pull (asset class 'Indices'); the "
              "existing equities caches are left as-is.")
        return {"ok": False, "disabled": True,
                "reason": "equities constituent + fundamentals pull disabled"}

    eq, fr = {}, {}
    if PULL_EQUITY_CONSTITUENTS:
        try:
            from src import equities
            eq = equities.build_snapshot()
            print(f"  Equities: {eq.get('n_memberships', 0)} constituents / {eq.get('n_unique', 0)} "
                  f"unique across {len(eq.get('indices', {}))} indices"
                  if eq.get("ok") else f"  Equities snapshot skipped: {eq.get('reason')}")
        except Exception as e:
            print(f"  (Equities snapshot skipped: {e})")

    if PULL_FUNDAMENTALS:
        # Company fundamentals — WEEKLY-guarded append to data/equities/fundamentals.parquet
        # (fundamentals move slowly; a fresh-enough last pull is left alone). Guarded the same
        # way: a failure never blocks the pull, a dead pull never wipes the DB.
        try:
            from src import eqfunda
            fr = eqfunda.maybe_refresh(max_age_days=7)
            if fr.get("skipped"):
                print(f"  Fundamentals: last pull {fr.get('last_pull')} is {fr.get('age_days')}d old — kept.")
            elif fr.get("ok"):
                print(f"  Fundamentals: {fr.get('n_tickers', 0)} names appended ({fr.get('last_pull')}).")
            else:
                print(f"  Fundamentals pull skipped: {fr.get('reason')}")
        except Exception as e:
            print(f"  (Fundamentals pull skipped: {e})")

    # Rough hit budget (a "hit" = security x field, Bloomberg's daily-capacity unit).
    # On the Yahoo source the quotes/history/fundamentals legs cost ZERO Bloomberg hits —
    # only the membership meta pull (name/sector per constituent) touches the Terminal.
    from src import equities as _eq
    if _eq._use_yf():
        print("  Bloomberg hits this pull: ZERO — membership from the free ETF holdings "
              "files, quotes/history/fundamentals from Yahoo Finance.")
    else:
        est = int(eq.get("n_unique", 0) or 0) * 3 + int(fr.get("n_tickers", 0) or 0) * 30
        if est:
            print(f"  Estimated Bloomberg hits this pull: ~{est:,} (security x field, rough)")

    if eq.get("ok"):                       # record the pull in the shared manifest
        SNAP.mkdir(parents=True, exist_ok=True)
        m = _existing_manifest()
        m["equities"] = eq.get("indices", {})
        m["equities_pulled"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        (SNAP / "manifest.json").write_text(json.dumps(m, indent=2))
    return eq


def run(include_equities: bool = False) -> dict:
    """Pull the DAILY FICC inputs (LIVE if DATAFEED_MODE=bloomberg) and cache to data/snapshot/.
    Option OI chains are NOT pulled here — they're a separate weekly job (run_oi / --oi).
    The Equities side has its OWN pull (run_equities / --equities); pass include_equities=True
    (CLI --with-equities) to chain both in one run.
    Concurrency-locked: a second run started while one is in flight exits immediately
    (two pulls racing on the same files double-spends Bloomberg and risks bad writes)."""
    lock = SNAP / ".pull.lock"
    try:
        if lock.exists():
            age_min = (pd.Timestamp.now() - pd.Timestamp(lock.stat().st_mtime, unit="s")).total_seconds() / 60
            if age_min < 45:                      # a live pull; stale locks self-expire
                print(f"Another snapshot pull is already running (lock is {age_min:.0f} min old) "
                      "— this one is exiting. Wait for it to finish.")
                return _existing_manifest()
        SNAP.mkdir(parents=True, exist_ok=True)
        lock.write_text(pd.Timestamp.now().isoformat())
    except Exception:
        pass
    try:
        return _run_locked(include_equities)
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def _run_locked(include_equities: bool = False) -> dict:
    tickers = list(INSTRUMENTS)
    # Rough hit budget (a "hit" = security x field, Bloomberg's daily-capacity unit): the FICC
    # pull touches ~20 fields per contract across prices/yields/vols/skew/term/put-call/live.
    print(f"  Estimated Bloomberg hits this pull: ~{len(tickers) * 20:,} (security x field, rough)")
    prices = get_history(tickers)
    # GUARD — never overwrite a good snapshot with nothing. When the Bloomberg Terminal is
    # closed / logged out the pull returns an EMPTY price frame; because prices.parquet is
    # every report's offline fallback, saving that empties the whole book (exactly how the
    # Morning Coffee report ended up with 0/80 products). Keep what's on disk and bail —
    # the same protection the weekly OI job (run_oi) already applies.
    if prices is None or getattr(prices, "empty", True) or len(prices) == 0:
        SNAP.mkdir(parents=True, exist_ok=True)
        prev = _existing_manifest()
        prev["last_skipped_pull"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        (SNAP / "manifest.json").write_text(json.dumps(prev, indent=2))
        print("Snapshot pull returned NO price data (is the Bloomberg Terminal open + logged "
              "in?) — KEPT the existing snapshot; nothing was overwritten.")
        return prev
    ylds = get_yield_history(tickers)                 # benchmark yields for bond futures (FI TA)
    volume = get_volume_history(tickers)              # daily contract volume (flag confirmation)
    iv = get_implied_vol_history(tickers)
    skew = get_skew_components(tickers)
    ts = get_term_structure(tickers)
    pc = get_putcall(tickers)                         # options OI & volume, puts vs calls
    live = get_live_quote(tickers)                    # current px vs prior settle (CHG_*_1D)
    live_as_of = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    _save(prices, "prices")
    _save(ylds, "yields")
    _save(volume, "volume")
    _save(iv, "implied_vol")
    _save(skew["put"], "skew_put")
    _save(skew["call"], "skew_call")
    _save(skew["atm"], "skew_atm")
    for lab in TENOR_LABELS:
        _save(ts[lab], f"term_{lab.lower()}")
    for k in _PUTCALL_KEYS:
        _save(pc[k], f"putcall_{k}")
    SNAP.mkdir(parents=True, exist_ok=True)
    # Option OI chains are a SEPARATE weekly job (run_oi / --oi); the daily snapshot leaves the
    # last capture in place and carries its count + date forward into the manifest below.
    prev = _existing_manifest()
    live.rename_axis("ticker").reset_index().to_parquet(SNAP / "live.parquet", index=False)  # ticker-indexed

    # Extend the persistent ~10y COT price DB (incremental: only the new dates). This is
    # the Bloomberg-connected moment, so it's where the DB grows; it's a no-op off-Bloomberg.
    try:
        from src import cotdata
        store = cotdata.update_price_store(list(cotdata.COT_MAP))
        print(f"  COT price DB: {store.shape[0]} dates x {store.shape[1]} markets")
        cotdata.compute(force=True)     # rebuild COT history/signals so they ingest the fresh deep prices
        print("  COT signals rebuilt from the price DB")
    except Exception as e:
        print(f"  (COT price DB update skipped: {e})")

    # Our own constant-90d STIR ATM curve (settlement-inverted, src/stircurve.py) — top up
    # the last ~2 weeks so settles finalise and expiries roll. Backfilled 13 months 2026-07-22.
    try:
        from src import stircurve
        stircurve.update_history(log=lambda *a: None)
        print("  STIR 90d curve history topped up (our own construction)")
    except Exception as e:
        print(f"  (STIR curve update skipped: {e})")

    # Our own 30d/90d ATM curve for the whole listed-futures book (src/owncurve.py) —
    # THE VOL BOOK'S IMPLIED SOURCE since 2026-07-22 (backfilled 13 months): the vol
    # cross-section overlays this history onto the vendor frame, so this append must
    # stay AHEAD of any signals rebuild. bbg30 rides along as the standing cross-check.
    try:
        from src import owncurve
        owncurve.append_today(log=lambda *a: None)
        print("  Own-curve 30d/90d book rebuilt + history appended")
    except Exception as e:
        print(f"  (own-curve book update skipped: {e})")

    # Equities are a SEPARATE pull (run_equities / --equities, wired to the Equities home
    # page) so the FICC morning pull stays fast — chain both with include_equities=True.
    eq = {}
    if include_equities:
        eq = run_equities()

    manifest = {
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": str(prices.index.max().date()) if len(prices) else "",
        "source": MODE,                       # "bloomberg" for a real morning pull
        "n_tickers": len(tickers),
        "price_rows": int(len(prices)),
        "iv_markets": int(iv.notna().any().sum()),
        "oi_markets": _oi_markets_cached(),    # from the last weekly OI capture (run_oi / --oi)
        "oi_as_of": prev.get("oi_as_of", ""),  # when that weekly OI capture last ran
        "live_as_of": live_as_of,             # when the prior-settle->now quote was pulled
        "live_n": int(live["pct"].notna().sum()) if "pct" in live.columns else 0,
        # futures-only pulls carry the LAST equities pull's record forward, not blank it
        "equities": eq.get("indices", {}) if eq.get("ok") else prev.get("equities", {}),
    }
    SNAP.mkdir(parents=True, exist_ok=True)
    (SNAP / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_excel(out_path: str) -> str:
    """Dump the cached snapshot parquets to one multi-sheet .xlsx for eyeballing."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        wrote = False
        for name in FRAMES:
            p = SNAP / f"{name}.parquet"
            if p.exists():
                pd.read_parquet(p).to_excel(xl, sheet_name=name[:31], index=False)
                wrote = True
        if not wrote:                         # ExcelWriter needs at least one sheet
            pd.DataFrame({"note": ["no snapshot yet — run snapshot.py first"]}).to_excel(
                xl, sheet_name="empty", index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="", help="dump the cached snapshot to this .xlsx and exit")
    ap.add_argument("--oi", action="store_true",
                    help="WEEKLY job: capture ONLY the fixed-income option chains (run Mondays)")
    ap.add_argument("--equities", action="store_true",
                    help="run ONLY the Equities pull (membership + quotes + fundamentals)")
    ap.add_argument("--with-equities", action="store_true",
                    help="chain the Equities pull onto the FICC snapshot (old combined behaviour)")
    args = ap.parse_args()
    if args.excel:
        export_excel(args.excel)
        print("Wrote", args.excel)
        return
    if args.oi:
        run_oi()
        return
    if args.equities:
        eq = run_equities()
        print(json.dumps(eq, indent=2, default=str))
        # nonzero on a dead pull so the app's button shows the failure instead of "refreshed";
        # a DELIBERATELY-disabled pull ("disabled") is not a failure, so it exits 0.
        raise SystemExit(0 if (eq.get("ok") or eq.get("disabled")) else 1)
    m = run(include_equities=args.with_equities)
    print("Snapshot written to", SNAP)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
