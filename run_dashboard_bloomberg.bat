@echo off
REM ===========================================================================
REM  BASIS (Strategy Monitor) - LIVE BLOOMBERG launcher (for the work PC).
REM  Same as run_dashboard.bat but connects to the Bloomberg Desktop API instead
REM  of reading the cached snapshot.
REM
REM  BEFORE running: the Bloomberg Terminal must be OPEN and LOGGED IN on this PC.
REM  (If Bloomberg isn't available, use run_dashboard.bat for snapshot mode.)
REM
REM  Leave this window open while you use the app; close it to stop the server.
REM ===========================================================================
cd /d "%~dp0"
set DATAFEED_MODE=bloomberg
set PYTHONUTF8=1
REM Use the bundled Playwright browser if present (so the PDF reports work offline).
if exist "%~dp0playwright-browsers" set "PLAYWRIGHT_BROWSERS_PATH=%~dp0playwright-browsers"

echo Clearing any previous BASIS instance on port 8501...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting BASIS (LIVE Bloomberg) on http://localhost:8501  (close this window to stop)
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause
