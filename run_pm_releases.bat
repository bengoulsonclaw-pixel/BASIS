@echo off
rem Daily check for new WGC Gold Demand Trends / WPIC Platinum Quarterly editions;
rem builds + emails a one-page synopsis on a new release (gated by the dashboard's
rem auto-email toggle). Registered as Windows task "PM Release Synopses (daily)".
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" pm_release_scheduled_email.py >> "data\signals\pm_releases_run.log" 2>&1
