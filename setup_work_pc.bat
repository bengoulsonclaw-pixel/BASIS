@echo off
REM ===========================================================================
REM  BASIS - one-time setup on the work PC (fully OFFLINE; no internet needed).
REM  Double-click this once, from inside the copied BASIS folder.
REM  It builds a private Python environment and installs every package from the
REM  bundled wheels\ folder. Re-running it is safe.
REM ===========================================================================
setlocal
cd /d "%~dp0"

echo.
echo ===============  BASIS setup  ===============
echo.

REM --- locate Python ----------------------------------------------------------
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
echo Using Python:
%PY% --version
if errorlevel 1 (
  echo.
  echo  [X] Python was not found. Install Python first ^(see SETUP_WORK_PC.md^), then re-run.
  pause & exit /b 1
)

REM --- create the virtual environment -----------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo  [X] Could not create the venv. & pause & exit /b 1 )
)

REM --- install ALL packages offline from the bundled wheels -------------------
echo Installing packages from wheels\ ^(offline^) ...
".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt
if errorlevel 1 (
  echo.
  echo  [X] Offline install failed. Common causes ^(see SETUP_WORK_PC.md^):
  echo      - the work PC's Python version differs from the one the wheels were built for
  echo      - blpapi/xbbg not in wheels  ^(install them online from Bloomberg's index^)
  pause & exit /b 1
)

echo.
echo ===============  Setup complete  ===============
echo.
echo  To run BASIS:
echo    - LIVE Bloomberg : make sure the Bloomberg Terminal is open + logged in,
echo                       then double-click  run_dashboard_bloomberg.bat
echo    - Snapshot only  : double-click  run_dashboard.bat
echo.
pause
