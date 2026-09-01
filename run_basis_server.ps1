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
        # pythonw.exe (2026-09-01), NOT python.exe: on Windows 11 the default console host is
        # Windows Terminal, which IGNORES -WindowStyle Hidden and shows the server's console anyway
        # (Ben was left with stray "…python.exe" terminal windows cluttering the taskbar). pythonw is
        # the GUI-subsystem Python — it allocates NO console at all, so nothing can surface. stdout/
        # stderr are still redirected to the log files below, so we lose no logging.
        $server = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\pythonw.exe" `
            -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", "8501", `
                          "--server.headless", "true", "--browser.gatherUsageStats", "false", `
                          "--server.fileWatcherType", "none" `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$PSScriptRoot\logs\basis_server.log" `
            -RedirectStandardError "$PSScriptRoot\logs\basis_server_err.log"
        # HEALTH-MONITOR (revised 2026-08-29): the FIRST version killed the server after just ~30s
        # of failed health checks — which over-killed a perfectly HEALTHY server whenever the machine
        # was briefly busy (a 96s test-suite run starved the health handler and the server got
        # needlessly restarted). This version is deliberately conservative: it restarts ONLY on
        # (a) a process that has EXITED, (b) a boot that never comes up in ~7 min, or (c) a server
        # that stays unresponsive CONTINUOUSLY for 5 MINUTES — a genuine multi-minute hang. Any
        # transient slowness (test runs, heavy renders, GC) resets the timer and is ignored, so the
        # keeper never fights a server that is merely busy.
        $everHealthy = $false
        $bootDeadline = (Get-Date).AddSeconds(420)   # ~7 min cold-boot allowance
        $unhealthySince = $null
        $hung = $false
        while (-not $server.HasExited) {
            Start-Sleep -Seconds 20
            $healthy = $false
            try {
                $healthy = (Invoke-WebRequest "http://localhost:8501/_stcore/health" `
                             -UseBasicParsing -TimeoutSec 20).StatusCode -eq 200
            } catch { $healthy = $false }
            if ($healthy) { $everHealthy = $true; $unhealthySince = $null; continue }
            if (-not $everHealthy) {
                if ((Get-Date) -ge $bootDeadline) { $hung = $true; break }   # never came up in ~7 min
                continue
            }
            # was serving, now not — only a CONTINUOUS 5-min outage counts as a real hang
            if ($null -eq $unhealthySince) { $unhealthySince = Get-Date }
            if (((Get-Date) - $unhealthySince).TotalSeconds -ge 300) { $hung = $true; break }
        }
        if ($hung) {
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
            Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -match 'streamlit' -and $_.CommandLine -match '8501' } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        }
        Start-Sleep -Seconds 5          # breathe before restarting (crash loops)
    } else {
        Start-Sleep -Seconds 15
    }
}
