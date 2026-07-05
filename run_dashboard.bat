@echo off
REM ===========================================================================
REM  BASIS (Strategy Monitor) - double-click launcher.
REM  Clears any previous instance still holding port 8501 (fixes the recurring
REM  "localhost refused to connect" / stale-code problem after a relaunch), then
REM  starts fresh in SNAPSHOT mode (reads data/snapshot/, no Bloomberg). Use the
REM  Home page's "Pull Bloomberg Snapshot" button to refresh from Bloomberg.
REM  Leave this window open while you use the app; close it to stop the server.
REM ===========================================================================
cd /d "%~dp0"
set DATAFEED_MODE=snapshot
set PYTHONUTF8=1
REM Use the bundled Playwright browser if this is a deployed copy (no-op on the dev PC).
if exist "%~dp0playwright-browsers" set "PLAYWRIGHT_BROWSERS_PATH=%~dp0playwright-browsers"

echo Clearing any previous BASIS instance on port 8501...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting BASIS on http://localhost:8501  (close this window to stop)
REM Open the browser a few seconds later, once the server has had time to bind
REM (avoids the "can't connect" flash from opening the URL too early).
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause
