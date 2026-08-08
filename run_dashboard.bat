@echo off
REM ===========================================================================
REM  BASIS (Strategy Monitor) - double-click launcher (SNAPSHOT mode).
REM
REM  No console stays open any more: this hands off to launch_basis.ps1, which
REM  runs the server INVISIBLY (output -> logs\basis_server.log) and watches for
REM  open BASIS windows - CLOSING EVERY BASIS WINDOW SHUTS THE SERVER DOWN TOO
REM  (closing just one of several no longer kills it).
REM  It also clears any previous instance still holding port 8501 first, so a
REM  relaunch always picks up the latest code.
REM
REM  Snapshot mode reads data/snapshot/ (no Bloomberg needed); use the Home
REM  page's "Pull Bloomberg Snapshot" button to refresh from Bloomberg.
REM ===========================================================================
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launch_basis.ps1" -Mode snapshot
