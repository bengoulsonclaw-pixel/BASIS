"""BASIS front-door (2026-10-05) — the thing the browser actually talks to on :8501.

WHY: after a reboot the Streamlit server takes a few minutes to cold-boot, and anything
that opens BASIS in that window (the gold PWA clicked by hand, or a window Windows restores
on login) hit a raw Chrome "ERR_CONNECTION_REFUSED". This tiny always-up server removes that:

  - Streamlit now runs on an INTERNAL port (BASIS_APP_PORT, default 8502).
  - This process listens on the PUBLIC port (BASIS_FRONT_PORT, default 8501) and, per
    connection, simply TRIES to reach Streamlit:
      * if Streamlit answers  -> pipe the bytes straight through (transparent for HTTP *and*
        the websocket; no protocol parsing, so Streamlit behaves exactly as if direct), and
      * if it does not (still booting / restarting / down) -> return a branded
        "BASIS is starting…" page that reloads itself every 2s, so the instant the app is
        ready the next reload lands on the real thing.

It is deliberately dumb and stateless — there is no "ready" flag to get wrong, it just
forwards when it can and shows the splash when it can't. If this process ever dies the
keeper restarts it; and the real app stays directly reachable on :8502 as a bypass.
"""
from __future__ import annotations

import asyncio
import os

APP_HOST = "127.0.0.1"
APP_PORT = int(os.environ.get("BASIS_APP_PORT", "8502"))      # Streamlit, internal
FRONT_HOST = "0.0.0.0"
FRONT_PORT = int(os.environ.get("BASIS_FRONT_PORT", "8501"))  # what the browser hits
CONNECT_TIMEOUT = 1.5                                          # how long to wait for Streamlit
BUF = 65536

_SPLASH_HTML = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="2">
<title>BASIS — starting…</title>
<style>
  :root{color-scheme:dark}
  html,body{height:100%;margin:0}
  body{background:#0E1117;color:#ECEEF1;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
       display:flex;align-items:center;justify-content:center}
  .box{text-align:center;transform:translateY(-6%)}
  .mark{font:800 46px/1 Segoe UI,Arial,sans-serif;letter-spacing:.14em;color:#F5C518}
  .mark span{color:#ECEEF1}
  .tag{margin-top:10px;font-size:12px;letter-spacing:.34em;text-transform:uppercase;color:#9BA1A9}
  .msg{margin-top:34px;font-size:15px;color:#CBD0D7}
  .sub{margin-top:8px;font-size:12.5px;color:#9BA1A9}
  .bar{margin:26px auto 0;width:220px;height:3px;background:rgba(245,197,24,.18);overflow:hidden;border-radius:3px}
  .bar>i{display:block;height:100%;width:40%;background:#F5C518;border-radius:3px;
         animation:slide 1.1s ease-in-out infinite}
  @keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}
</style></head><body>
  <div class="box">
    <div class="mark">BA<span>S</span>IS</div>
    <div class="tag">Analysis · Strategies · Indicators</div>
    <div class="msg">Starting up…</div>
    <div class="sub">The server is warming up — this page will load it automatically.</div>
    <div class="bar"><i></i></div>
  </div>
</body></html>"""

_SPLASH = ("HTTP/1.1 200 OK\r\n"
           "Content-Type: text/html; charset=utf-8\r\n"
           f"Content-Length: {len(_SPLASH_HTML.encode('utf-8'))}\r\n"
           "Cache-Control: no-store, must-revalidate\r\n"
           "Connection: close\r\n"
           "\r\n" + _SPLASH_HTML).encode("utf-8")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(BUF)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    # Try to reach Streamlit. Connect FIRST (before touching the client's bytes) so the
    # request is forwarded verbatim when the app is up.
    try:
        srv_reader, srv_writer = await asyncio.wait_for(
            asyncio.open_connection(APP_HOST, APP_PORT), timeout=CONNECT_TIMEOUT)
    except Exception:
        srv_reader = srv_writer = None

    if srv_writer is None:
        # Streamlit not reachable -> serve the splash and close.
        try:
            client_writer.write(_SPLASH)
            await client_writer.drain()
        except Exception:
            pass
        finally:
            try:
                client_writer.close()
            except Exception:
                pass
        return

    # Streamlit is up -> transparent two-way pipe (handles HTTP and the websocket alike).
    await asyncio.gather(
        _pipe(client_reader, srv_writer),
        _pipe(srv_reader, client_writer),
    )


async def _main() -> None:
    # reuse_address=False so exactly ONE front-door can hold the port. On Windows the default
    # (SO_REUSEADDR) lets two processes "bind" the same port with undefined routing, so if a
    # keeper race ever starts a second front-door it must lose the bind and exit, not share it.
    try:
        server = await asyncio.start_server(_handle, FRONT_HOST, FRONT_PORT, reuse_address=False)
    except OSError as e:
        print(f"front-door: :{FRONT_PORT} already held by another front-door — exiting cleanly ({e})",
              flush=True)
        return
    addr = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"BASIS front-door listening on {addr} -> {APP_HOST}:{APP_PORT}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
