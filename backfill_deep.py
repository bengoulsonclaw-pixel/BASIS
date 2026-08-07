"""One-off backfill of the deep '1'-generic price store (src/deepstore.py).

Run with the Bloomberg Terminal open + logged in:

    python backfill_deep.py

~350 hits total (hits = security x field, independent of date depth — a fifth of one
normal morning snapshot), so no need to stage it. Safe to re-run: already-deep tickers
only get a ~10-day tail; anything missing or truncated is re-backfilled. The morning
snapshot also does this automatically (snapshot.py fetch phase), so if the store is
empty tonight it will simply build itself on the next Terminal morning.
"""
import os

os.environ.setdefault("DATAFEED_MODE", "bloomberg")

from src import deepstore  # noqa: E402  (env var must be set before datafeed imports)


def main():
    print("Backfilling deep price store (~10y of '1'/'2' generics + contract strings, "
          "volume, bond benchmark yields)…")
    store = deepstore.update(log=print)
    if store.empty:
        print("\nNo data came back — is the Bloomberg Terminal open and logged in?")
        return
    cov = deepstore.coverage()
    print(f"\nStore: {store.shape[0]} dates x {store.shape[1]} markets")
    if not cov.empty:
        cov = cov.sort_values("first")
        print(f"Depth: {cov['first'].min()} -> {cov['last'].max()}; "
              f"median days/ticker {int(cov['days'].median())}; "
              f"rolls detected {int(cov['rolls'].sum())}")
        shallow = cov[cov["days"] < deepstore.HEAL_MIN_DAYS]
        if not shallow.empty:
            print("Still shallow (< ~3y — will self-heal on later runs, or the market "
                  "is simply young):")
            for _, r in shallow.iterrows():
                print(f"  {r['ticker']}: {r['first']} -> {r['last']} ({r['days']}d)")


if __name__ == "__main__":
    main()
