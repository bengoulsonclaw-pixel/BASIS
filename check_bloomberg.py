"""Check whether THIS machine can pull live Bloomberg data.

Run:  .venv\\Scripts\\python.exe check_bloomberg.py

Requires the Bloomberg Terminal running + logged in on this PC, and the client
libraries installed (blpapi from Bloomberg's index, then xbbg).
"""
import sys

try:  # ensure non-ASCII output doesn't crash on a captured Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    try:
        import pandas as pd
        from xbbg import blp
    except Exception as e:
        print("FAIL - client libraries not installed:", e)
        print("Install:")
        print("  pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi")
        print("  pip install xbbg")
        return 1

    end = pd.Timestamp.today().normalize()
    start = (end - pd.tseries.offsets.BDay(10)).strftime("%Y-%m-%d")
    try:
        df = blp.bdh("CLA Comdty", "PX_SETTLE", start, end.strftime("%Y-%m-%d"))
    except Exception as e:
        print("FAIL - could not query Bloomberg:", repr(e))
        print("Checklist:  Terminal running?  Logged in on THIS PC?  API enabled (WAPI<GO>)?")
        return 3

    if df is None or len(df) == 0:
        print("CONNECTED, but no data returned - check the ticker / your entitlements.")
        return 2

    print("SUCCESS - live Bloomberg data pulled.")
    print("shape:", df.shape)
    print(df.tail())                       # xbbg returns a narwhals frame (no .to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
