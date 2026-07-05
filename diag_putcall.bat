@echo off
REM ===========================================================================
REM  Double-click to check how much PUT/CALL history Bloomberg serves per
REM  product, by ticker form (active 'A' generic vs 1st-generic vs source).
REM  The Bloomberg Terminal must be OPEN and LOGGED IN (same as a snapshot).
REM  Results are saved to putcall_history_check.txt and opened in Notepad.
REM ===========================================================================
cd /d "%~dp0"
set DATAFEED_MODE=bloomberg
set PYTHONUTF8=1
echo Checking Bloomberg put/call history depth for each product...
echo (this takes a minute or two - the Terminal must be open and logged in)
echo.
".venv\Scripts\python.exe" diag_putcall.py > "putcall_history_check.txt" 2>&1
echo DONE - results saved to putcall_history_check.txt
start "" notepad "putcall_history_check.txt"
