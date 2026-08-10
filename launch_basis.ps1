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

# Win32 window-title probe: TRUE while any visible top-level window is a BASIS
# window (installed app, --app window, or a fronted browser tab — all carry
# "Strategy Monitor" in the title, even on an error page). Get-Process can't do
# this: app windows live inside the main Chrome process, whose MainWindowTitle
# only reports one window.
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class BasisWin {
  delegate bool EnumProc(IntPtr h, IntPtr lp);
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr lp);
  [DllImport("user32.dll")] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  public static bool Any() {
    bool found = false;
    EnumWindows((h, lp) => {
      if (!IsWindowVisible(h)) return true;
      var sb = new StringBuilder(256); GetWindowText(h, sb, 256);
      if (sb.ToString().Contains("Strategy Monitor")) { found = true; return false; }
      return true;
    }, IntPtr.Zero);
    return found;
  }
}
'@
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

# --- lifetime: serve while ANY BASIS window is on screen ------------------------
# Three signals, because each alone has a hole: the spawned $win exits at once
# when Chrome hands the launch off to a running instance; TCP connections vanish
# when an OPEN window is sitting on an error/frozen page (killing the server then
# strands the user — hitting Reload must always work); and the title probe alone
# would miss the moment between server-up and the window appearing. Shut down
# only after ~30s with none of the three.
$idle = 0
while ($true) {
    Start-Sleep -Seconds 3
    if ($server.HasExited) { break }
    $winAlive = ($win -and -not $win.HasExited)
    $winOpen  = [BasisWin]::Any()
    $conns = @(Get-NetTCPConnection -LocalPort 8501 -State Established `
                   -ErrorAction SilentlyContinue).Count
    if ($winAlive -or $winOpen -or $conns -gt 0) { $idle = 0; continue }
    $idle += 3
    if ($idle -ge 30) { break }
}

# --- every BASIS window closed -> stop the server (own process tree only) ------
# No blanket port-8501 sweep here: it raced a concurrent relaunch and killed the
# NEW instance's healthy server. Orphans are handled by the next launch's
# port-clear at the top instead.
try { & taskkill /PID $server.Id /T /F 2>$null | Out-Null } catch { }
