# ===========================================================================
#  BASIS launcher / lifetime supervisor (run via run_dashboard*.bat, hidden).
#
#  Replaces the old always-open black console: the server now runs INVISIBLY
#  (its output goes to logs\basis_server.log instead of a console window), and
#  this script stays alive only to watch for open BASIS windows — once NO window
#  has been connected to the server for ~30s, it is shut down. Nothing is left
#  running, but closing one of several open BASIS windows no longer kills it.
#
#  -Mode snapshot   (default) read data/snapshot/, no Bloomberg needed
#  -Mode bloomberg  live Bloomberg Desktop API (Terminal must be open + logged in)
# ===========================================================================
param([string]$Mode = "snapshot")

Set-Location $PSScriptRoot
$env:DATAFEED_MODE = $Mode
$env:PYTHONUTF8 = "1"
if (Test-Path "$PSScriptRoot\playwright-browsers") {
    $env:PLAYWRIGHT_BROWSERS_PATH = "$PSScriptRoot\playwright-browsers"
}

# --- clear any previous instance still holding port 8501 (stale-code guard) ---
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

# --- start the server hidden, logging to logs\ --------------------------------
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null
$server = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", "8501", `
                  "--server.headless", "true", "--browser.gatherUsageStats", "false" `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$PSScriptRoot\logs\basis_server.log" `
    -RedirectStandardError "$PSScriptRoot\logs\basis_server_err.log"

# --- wait until it answers (app.py imports a lot — allow a generous warm-up) --
$up = $false
foreach ($i in 1..60) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
    if ($server.HasExited) { break }
}
if (-not $up) {
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    [System.Windows.Forms.MessageBox]::Show(
        "BASIS failed to start - see logs\basis_server_err.log", "BASIS",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    try { & taskkill /PID $server.Id /T /F 2>$null | Out-Null } catch { }
    exit 1
}

# --- open BASIS as its own app window, tied to this supervisor ----------------
# Preferred: the INSTALLED "BASIS" web app in the user's main Chrome profile —
# that window carries the real BASIS taskbar icon AND BASIS's own gold title bar
# (manifest theme_color); a plain --app window ignores both and paints the bar
# with Chrome's default profile theme instead. The app id is derived by Chrome
# from start_url http://localhost:8501 and stays stable across reinstalls. The
# spawned process usually hands off to the already-running Chrome and exits at
# once — harmless: the connection watcher below owns the server's lifetime.
$browser = @("C:\Program Files\Google\Chrome\Application\chrome.exe",
             "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
             "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             "C:\Program Files\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$appId  = "fkkhajlpfoflidlepdpofgkmlcgcobng"
$pwaDir = Join-Path $env:LOCALAPPDATA `
    "Google\Chrome\User Data\Default\Web Applications\Manifest Resources\$appId"
if ($browser -and $browser -like "*chrome.exe" -and (Test-Path $pwaDir)) {
    $win = Start-Process -FilePath $browser -PassThru -ArgumentList `
        "--profile-directory=Default", "--app-id=$appId"
} elseif ($browser) {
    # BASIS app not installed (or Chrome missing): chromeless --app window on a
    # dedicated profile. NB its title bar shows the browser's theme colour, not
    # BASIS gold — install the app (⋮ > Cast, save and share > Install page as
    # app) to get the branded window back.
    $profileDir = Join-Path $env:LOCALAPPDATA "BASIS\app-window-profile"
    New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    $win = Start-Process -FilePath $browser -PassThru -ArgumentList `
        "--app=http://localhost:8501", "--window-size=1440,900", `
        "--user-data-dir=$profileDir", "--no-first-run", "--no-default-browser-check"
} else {
    # No Chrome/Edge at the usual paths: plain browser tab in the default browser.
    Start-Process "http://localhost:8501"
    $win = $null
}

# --- lifetime: serve while ANY BASIS window is open, not just the one above -----
# Waiting on $win alone was wrong twice over: (a) if the browser hands the launch
# off to an already-running instance the spawned process exits at once, and (b) the
# user mostly lives in the INSTALLED "BASIS" app window (part of the main Chrome
# process), which this script never spawned. Both looked like "window closed" and
# shut a healthy server down mid-use. So also watch real connections to port 8501,
# and only stop once there has been no window AND no connection for ~30s.
$idle = 0
while ($true) {
    Start-Sleep -Seconds 3
    if ($server.HasExited) { break }
    $winAlive = ($win -and -not $win.HasExited)
    $conns = @(Get-NetTCPConnection -LocalPort 8501 -State Established `
                   -ErrorAction SilentlyContinue).Count
    if ($winAlive -or $conns -gt 0) { $idle = 0; continue }
    $idle += 3
    if ($idle -ge 30) { break }
}

# --- every BASIS window closed -> stop the server (process tree + port sweep) --
try { & taskkill /PID $server.Id /T /F 2>$null | Out-Null } catch { }
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
