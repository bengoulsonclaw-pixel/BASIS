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
REM Chromeless Chrome APP WINDOW (--app=) instead of a browser tab — see
REM run_dashboard.bat for the reasoning. Falls back to a normal tab if Chrome
REM isn't found at either usual install path.
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; $c='C:\Program Files\Google\Chrome\Application\chrome.exe'; if (-not (Test-Path $c)) { $c='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe' }; if (Test-Path $c) { Start-Process -FilePath $c -ArgumentList '--app=http://localhost:8501','--window-size=1440,900' } else { Start-Process 'http://localhost:8501' }"
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause
