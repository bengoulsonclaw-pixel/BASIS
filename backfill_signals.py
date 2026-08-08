"""One-off backfill of the FICC signal cache (src/sigcache.py).

    python backfill_signals.py             # compute every missing (day, strategy) pair
    python backfill_signals.py --rebuild   # wipe and recompute from scratch

Pure local compute off the deep price store — no Bloomberg, no Terminal. ~10y × 16
strategies is a few hours of CPU; the run is resume-safe (flushes every ~25 days), so
an interrupted run keeps its progress and re-running just continues. The morning
snapshot's compute phase keeps the cache topped up daily once this has run.
"""
import argparse
import os

os.environ.setdefault("DATAFEED_MODE", "snapshot")

from src import sigcache  # noqa: E402  (env var must be set before datafeed imports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the existing cache first and recompute everything")
    args = ap.parse_args()
    if args.rebuild and sigcache.STORE_DIR.exists():
        for p in sigcache.STORE_DIR.glob("ficc_*.parquet"):
            p.unlink()
        print("Existing cache wiped — full recompute.")
    n = sigcache.extend(log=print)
    cov = sigcache.coverage()
    days = cov["date"].nunique() if not cov.empty else 0
    print(f"\nDone: {n} (day x strategy) pairs computed this run; "
          f"cache now covers {days} sessions x {cov['strategy'].nunique() if not cov.empty else 0} strategies.")


if __name__ == "__main__":
    main()
