@echo off
rem BASIS nightly backup push — commits and pushes each repo if anything changed.
rem Scheduled task: "BASIS Nightly Backup Push" (22:00 daily). The daily data pulls
rem accumulate non-regenerable history (own vol curve, fundamentals DB), so this
rem keeps GitHub current even when no dev work happens for weeks.
rem Log: %LOCALAPPDATA%\basis_backup_push.log (outside the repos on purpose).

set "LOG=%LOCALAPPDATA%\basis_backup_push.log"
echo ===== %date% %time% ===== >> "%LOG%"

call :backup "C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard"
call :backup "C:\Users\Ben\OneDrive\Desktop\AI"
call :backup "C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements"
exit /b 0

:backup
cd /d "%~1"
echo --- %~1 >> "%LOG%"
git add -A >> "%LOG%" 2>&1
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Nightly auto-backup" >> "%LOG%" 2>&1
) else (
    echo no changes >> "%LOG%"
)
rem Push even with no new commit — retries anything a previous night failed to push.
git push >> "%LOG%" 2>&1
exit /b 0
