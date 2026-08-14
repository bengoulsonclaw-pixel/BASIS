"""One-off backfill of the signal cache (src/sigcache.py).

    python backfill_signals.py                       # FICC: every missing (day, strategy) pair
    python backfill_signals.py --rebuild             # FICC: wipe and recompute from scratch
    python backfill_signals.py --equities            # equities book (~2,600 names), last 3y
    python backfill_signals.py --equities --years 5  # deeper, if the price cache allows
    python backfill_signals.py --equities --pull     # deepen the Yahoo price cache first

FICC is pure local compute off the deep price store — no Bloomberg, no Terminal; ~10y ×
16 strategies is a few hours of CPU. Equities scores the whole ~2,600-name Yahoo book, so
each (day, strategy) pair costs seconds — the default 3y window is ~10h of CPU. Both runs
are resume-safe (flush every few days), so an interrupted run keeps its progress and
re-running just continues. The daily pulls keep each cache topped up once this has run.

The equities window needs price history ~400 sessions deeper than the earliest signal day
(the hub's trailing window); --pull tops the eqta parquet cache up from Yahoo first
(depth-preserving merge — routine refreshes can't truncate it back).
"""
import argparse
import os

os.environ.setdefault("DATAFEED_MODE", "snapshot")

import pandas as pd  # noqa: E402

from src import sigcache  # noqa: E402  (env var must be set before datafeed imports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the existing cache for the chosen book first")
    ap.add_argument("--equities", action="store_true",
                    help="backfill the equities book instead of FICC")
    ap.add_argument("--years", type=float, default=3.0,
                    help="equities only: how many years of signal days to backfill (default 3)")
    ap.add_argument("--pull", action="store_true",
                    help="equities only: deepen the Yahoo price cache first (signal years "
                         "+ the 400-session warm-up)")
    ap.add_argument("--flush-every", type=int, default=None,
                    help="persist every N days (default: 25 FICC / 5 equities)")
    args = ap.parse_args()
    scope = "equities" if args.equities else "ficc"
    prefix = "eq" if scope == "equities" else "ficc"

    if args.rebuild and sigcache.STORE_DIR.exists():
        for p in sigcache.STORE_DIR.glob(f"{prefix}_*.parquet"):
            p.unlink()
        print(f"Existing {scope} cache wiped — full recompute.")

    days = None
    if scope == "equities":
        from src import eqta
        if args.pull:
            sessions = int(args.years * 252) + 420
            tks = eqta.universe_tickers()
            print(f"Deepening the Yahoo price cache: {len(tks)} names × ~{sessions} sessions …")
            close, _ = eqta.backfill(tks, sessions=sessions)
            print(f"  price cache now {close.shape[0]} sessions × {close.shape[1]} names "
                  f"({close.index.min():%Y-%m-%d} -> {close.index.max():%Y-%m-%d})")
            sigcache.book_frames(refresh=True, scope=scope)
        sig, _ = sigcache.book_frames(scope=scope)
        all_days = sig.dropna(how="all").index
        cutoff = all_days.max() - pd.DateOffset(years=args.years)
        days = list(all_days[all_days >= cutoff])
        print(f"Equities backfill window: {len(days)} sessions "
              f"({days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d})")

    flush = args.flush_every or (5 if scope == "equities" else 25)
    n = sigcache.extend(days=days, log=print, scope=scope, flush_every=flush)
    cov = sigcache.coverage(scope)
    n_days = cov["date"].nunique() if not cov.empty else 0
    print(f"\nDone: {n} (day x strategy) pairs computed this run; {scope} cache now covers "
          f"{n_days} sessions x {cov['strategy'].nunique() if not cov.empty else 0} strategies.")
    if n and scope == "equities":
        from src import sigledger
        print("Rebuilding the equities ledger …")
        sigledger.rebuild(log=print, scope=scope)


if __name__ == "__main__":
    main()
