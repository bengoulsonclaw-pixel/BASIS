# ===========================================================================
#  BASIS always-on server keeper (runs hidden via the "BASIS Server" logon task).
#
#  WHY: the BASIS icon (installed Chrome app / any shortcut) only opens a WINDOW —
#  it cannot start the server, so every morning began with ERR_CONNECTION_REFUSED
#  (recurred all week, Aug 2026). This loop keeps a server on port 8501 at all
#  times: it starts one if the port is free and restarts it if it ever dies
#  (crash, sleep/wake casualty, stray kill). Any BASIS window then works at any
#  moment, with nothing to remember.
#
#  Mode: snapshot (reads data/snapshot; Bloomberg pulls are still on-demand from
#  the Home page). Logs -> logs\basis_server.log / basis_server_err.log.
#  To stop it for good: disable/delete the "BASIS Server" scheduled task, then
#  kill the python on port 8501.
# ===========================================================================
Set-Location $PSScriptRoot
# ONE keeper only (2026-08-22): every `schtasks /Run` from the launcher (or a hand
# start) spawned ANOTHER copy of this loop — two or three keepers then raced each
# other on port 8501 after every restart (competing Start-Process calls, crash
# loops, 2-minute revivals instead of 10 seconds). A second instance now exits.
# (match the -File invocation only — a diagnostic shell whose command text merely
#  MENTIONS the script must never count as a running keeper)
$others = @(Get-CimInstance Win32_Process -Filter 'Name = "powershell.exe" OR Name = "pwsh.exe"' |
            Where-Object { $_.ProcessId -ne $PID -and
                           $_.CommandLine -match '-File\s+\S*run_basis_server\.ps1' })
if ($others.Count -gt 0) { exit 0 }
$env:DATAFEED_MODE = "snapshot"
$env:PYTHONUTF8 = "1"
if (Test-Path "$PSScriptRoot\playwright-browsers") {
    $env:PLAYWRIGHT_BROWSERS_PATH = "$PSScriptRoot\playwright-browsers"
}
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null

while ($true) {
    $listening = @(Get-NetTCPConnection -LocalPort 8501 -State Listen `
                       -ErrorAction SilentlyContinue).Count
    if ($listening -eq 0) {
        # fileWatcherType none: the production server must NOT hot-reload — OneDrive
        # touching repo files raced Streamlit's watcher into KeyError crashes
        # (run_daily / src.universe / src.strategies) at import and mid-session.
        # Code changes reach it by restarting the server, not by saving files.
        $server = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" `
            -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", "8501", `
                          "--server.headless", "true", "--browser.gatherUsageStats", "false", `
                          "--server.fileWatcherType", "none" `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$PSScriptRoot\logs\basis_server.log" `
            -RedirectStandardError "$PSScriptRoot\logs\basis_server_err.log"
        # wait while THIS server lives; when it exits (or loses the port) loop around
        Wait-Process -Id $server.Id -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5          # breathe before restarting (crash loops)
    } else {
        Start-Sleep -Seconds 15
    }
}
