@echo off
rem Monthly Precious Metals Fundamentals monitor — build + email (gated by the
rem dashboard's auto-email toggle). Registered as Windows task
rem "Precious Metals Monitor (monthly)".
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" precious_metals_scheduled_email.py >> "data\signals\pm_run.log" 2>&1
