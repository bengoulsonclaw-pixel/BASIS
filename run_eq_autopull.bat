@echo off
rem BASIS Equities Auto-Pull - the scheduled twin of the Equities home
rem "Pull equities data" button. Created/removed by the Auto-pull control
rem on the Equities home (Windows Task Scheduler task "BASIS Equities Auto Pull").
rem Yahoo quotes/history (+ weekly fundamentals when due); Bloomberg only
rem refreshes index membership when the Terminal happens to be up.
cd /d "C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard"
set DATAFEED_MODE=bloomberg
set PYTHONUTF8=1
.venv\Scripts\python.exe snapshot.py --equities >> "%LOCALAPPDATA%\basis_eq_autopull.log" 2>&1
if %errorlevel%==0 (
  rem same as the button: fresh data -> GitHub -> VPS site within ~15 min
  .venv\Scripts\python.exe -c "from src import gitbackup; gitbackup._push()" >> "%LOCALAPPDATA%\basis_eq_autopull.log" 2>&1
)
