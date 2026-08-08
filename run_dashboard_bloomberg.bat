@echo off
REM ===========================================================================
REM  BASIS (Strategy Monitor) - LIVE BLOOMBERG launcher (for the work PC).
REM  Same as run_dashboard.bat but connects to the Bloomberg Desktop API instead
REM  of reading the cached snapshot.
REM
REM  BEFORE running: the Bloomberg Terminal must be OPEN and LOGGED IN on this PC.
REM  (If Bloomberg isn't available, use run_dashboard.bat for snapshot mode.)
REM
REM  No console stays open any more: launch_basis.ps1 runs the server invisibly
REM  (output -> logs\basis_server.log) and CLOSING EVERY BASIS WINDOW SHUTS THE
REM  SERVER DOWN TOO (closing just one of several no longer kills it).
REM ===========================================================================
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launch_basis.ps1" -Mode bloomberg
