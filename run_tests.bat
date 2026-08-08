@echo off
rem BASIS regression suite — double-click to run the golden-file engine tests.
rem   run_tests.bat            run the suite (green/red recorded for the Data health page)
rem   run_tests.bat --regen    re-baseline the goldens after a DELIBERATE behaviour change
cd /d "%~dp0"
".venv\Scripts\python.exe" run_tests.py %*
echo.
pause
