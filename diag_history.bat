@echo off
REM ===========================================================================
REM  Double-click to check how far back Bloomberg history goes for each market.
REM  The Bloomberg Terminal must be OPEN and LOGGED IN (same as a snapshot).
REM  Results are saved to history_check.txt and opened in Notepad.
REM ===========================================================================
cd /d "%~dp0"
set DATAFEED_MODE=bloomberg
set PYTHONUTF8=1
echo Checking Bloomberg history depth for each market...
echo (this takes a minute or two - the Terminal must be open and logged in)
echo.
".venv\Scripts\python.exe" diag_history.py > "history_check.txt" 2>&1
echo DONE - results saved to history_check.txt
start "" notepad "history_check.txt"
