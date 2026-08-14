# ===========================================================================
#  BASIS launcher (run via run_dashboard*.bat, hidden).
#
#  Since 2026-08-14 the server itself is owned by the "BASIS Server" logon task
#  (run_basis_server.ps1 keep-alive loop) — it is ALWAYS on and self-healing, so
#  clicking any BASIS icon works at any time. This script only has to:
#    1. make sure that keeper/server is actually up (kick the task if not),
#    2. open BASIS as its own app window,
#  and exit. It must NOT port-clear or kill servers — that fights the keeper.
#
#  -Mode snapshot   (default) just ensure + open. The keeper's server runs
#                   snapshot mode; Bloomberg pulls are on-demand from Home.
#  -Mode bloomberg  live Desktop-API mode: replaces the running server with a
#                   DATAFEED_MODE=bloomberg one (the keeper tolerates it while
#                   it holds port 8501, and revives snapshot mode after it dies).
# ===========================================================================
param([string]$Mode = "snapshot")

Set-Location $PSScriptRoot

function Test-Basis {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

if ($Mode -eq "bloomberg") {
    # live mode wants ITS server on the port: clear it, start bloomberg-mode.
    # The keeper sees the port occupied and leaves it alone; when this server
    # eventually dies, the keeper brings snapshot mode back automatically.
    Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    $env:DATAFEED_MODE = "bloomberg"
    $env:PYTHONUTF8 = "1"
    New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null
    Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" `
        -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", "8501", `
                      "--server.headless", "true", "--browser.gatherUsageStats", "false", `
                      "--server.fileWatcherType", "none" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput "$PSScriptRoot\logs\basis_server.log" `
        -RedirectStandardError "$PSScriptRoot\logs\basis_server_err.log" | Out-Null
} elseif (-not (Test-Basis)) {
    # keeper task should be running the server — kick it (no-op if already going)
    schtasks /Run /TN "BASIS Server" 2>$null | Out-Null
}

# --- wait until the server answers (app.py imports a lot — generous warm-up) --
$up = $false
foreach ($i in 1..60) {
    if (Test-Basis) { $up = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $up) {
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    [System.Windows.Forms.MessageBox]::Show(
        "BASIS server did not answer - see logs\basis_server_err.log " +
        "(is the 'BASIS Server' scheduled task enabled?)", "BASIS",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    exit 1
}

# --- open BASIS as its own app window ------------------------------------------
# Preferred: the INSTALLED "BASIS" web app in the user's main Chrome profile —
# real BASIS taskbar icon + BASIS's own gold title bar (manifest theme_color);
# a plain --app window ignores both and paints Chrome's default profile theme.
# The app id is derived by Chrome from start_url http://localhost:8501 and stays
# stable across reinstalls. Server lifetime is the keeper's job — this script
# exits as soon as the window is launched.
$browser = @("C:\Program Files\Google\Chrome\Application\chrome.exe",
             "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
             "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             "C:\Program Files\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$appId  = "fkkhajlpfoflidlepdpofgkmlcgcobng"
$pwaDir = Join-Path $env:LOCALAPPDATA `
    "Google\Chrome\User Data\Default\Web Applications\Manifest Resources\$appId"
if ($browser -and $browser -like "*chrome.exe" -and (Test-Path $pwaDir)) {
    Start-Process -FilePath $browser -ArgumentList `
        "--profile-directory=Default", "--app-id=$appId" | Out-Null
} elseif ($browser) {
    # BASIS app not installed (or Chrome missing): chromeless --app window on a
    # dedicated profile. NB its title bar shows the browser's theme colour, not
    # BASIS gold — install the app (⋮ > Cast, save and share > Install page as
    # app) to get the branded window back.
    $profileDir = Join-Path $env:LOCALAPPDATA "BASIS\app-window-profile"
    New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    Start-Process -FilePath $browser -ArgumentList `
        "--app=http://localhost:8501", "--window-size=1440,900", `
        "--user-data-dir=$profileDir", "--no-first-run", "--no-default-browser-check" | Out-Null
} else {
    Start-Process "http://localhost:8501"
}
