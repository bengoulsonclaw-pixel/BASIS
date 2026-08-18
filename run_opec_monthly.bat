@echo off
REM ===========================================================================
REM  OPEC MOMR synopsis - monthly job (Windows Task Scheduler), inbox-watch.
REM  OPEC gates the PDF behind a form + Cloudflare bot-check (Aug 2026), so the
REM  download is manual: drop this month's MOMR PDF into opec\inbox (upload it on
REM  the OPEC Report page). This job then builds the branded synopsis + chart deck
REM  and emails the "opec" recipient list, sending ONCE per edition (marker-gated).
REM  No Chrome, no fetch: if no new PDF is in the inbox it exits quietly.
REM ===========================================================================
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" opec_scheduled_email.py >> "data\signals\opec_run.log" 2>&1
