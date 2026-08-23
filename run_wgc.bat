@echo off
rem World Gold Council (Goldhub) fetcher — DOUBLE-CLICK THIS FILE.
rem
rem   run_wgc.bat            first-run flow: sign in, then report what's downloadable
rem   run_wgc.bat --probe    just re-check entitlement (session already saved)
rem   run_wgc.bat --fetch    download + archive the datasets into data\wgc_inbox
rem
rem Uses .venv\Scripts\python.exe on purpose: Playwright is installed there, not in
rem the global Python, so the global interpreter fails with ModuleNotFoundError.
cd /d "%~dp0"
set PYTHONUTF8=1

if "%~1"=="--probe" goto probe
if "%~1"=="--fetch" goto fetch

echo.
echo  ================================================================
echo   STEP 1 of 2 - sign in to Goldhub
echo  ================================================================
echo.
echo   A Chrome window will open. Sign in as basisreports@gmail.com,
echo   then come back to THIS window and press Enter.
echo.
".venv\Scripts\python.exe" src\wgc_fetch.py --login
if errorlevel 1 goto done

echo.
echo  ================================================================
echo   STEP 2 of 2 - checking what this account can download
echo  ================================================================
echo.
".venv\Scripts\python.exe" src\wgc_fetch.py --probe
echo.
echo   ^>^> Copy the ENTITLED / BLOCKED lines above and send them to Claude.
goto done

:probe
".venv\Scripts\python.exe" src\wgc_fetch.py --probe
goto done

:fetch
".venv\Scripts\python.exe" src\wgc_fetch.py
goto done

:done
echo.
pause
