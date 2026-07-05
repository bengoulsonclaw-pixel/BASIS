@echo off
REM ===========================================================================
REM  OPEC MOMR synopsis - unattended monthly job (Windows Task Scheduler).
REM  Runs on release-window days; self-gates via a marker so it sends ONCE per
REM  edition. Drives a persistent real-Chrome profile to fetch past Cloudflare
REM  (a brief Chrome window appears), builds the branded synopsis + chart deck,
REM  and emails the desk list (the "opec" recipients on the OPEC Report page).
REM  Must run in the logged-on interactive session (Chrome needs a desktop).
REM ===========================================================================
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" opec_scheduled_email.py >> "data\signals\opec_run.log" 2>&1
