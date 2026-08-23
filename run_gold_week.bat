@echo off
rem Weekly Gold - Week Ahead report: builds the PDF and emails it (gated by the
rem dashboard's auto-email toggle). Registered as Windows task "Gold Week Ahead (Mondays)".
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" gold_week_scheduled_email.py >> "data\signals\gold_week_run.log" 2>&1
