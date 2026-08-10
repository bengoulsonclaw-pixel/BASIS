# BASIS always-on server (installed 2026-08-10, Ben's ask after one ERR_CONNECTION_REFUSED
# too many): the Windows scheduled task "BASIS Server" runs this at logon, so the app window
# ALWAYS has a server behind it no matter how it's opened — launcher, taskbar PWA icon,
# restored window after reboot. Keeps the server alive forever: restarts streamlit if it
# ever exits; stands down quietly (cheap poll) while another server owns :8501 (e.g. one
# spawned by launch_basis.ps1), and takes over the moment that one goes away.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "data\server_task.log"
while ($true) {
    $busy = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
    if (-not $busy) {
        Add-Content $log "$(Get-Date -Format s) starting BASIS server on :8501"
        $env:DATAFEED_MODE = "snapshot"
        & "$root\.venv\Scripts\python.exe" -m streamlit run "$root\app.py" `
            --server.port 8501 --server.headless true --browser.gatherUsageStats false *>> $log
        Add-Content $log "$(Get-Date -Format s) server exited - restart check in 5s"
        Start-Sleep -Seconds 5
    } else {
        Start-Sleep -Seconds 15
    }
}
